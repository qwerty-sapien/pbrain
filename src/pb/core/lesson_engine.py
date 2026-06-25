# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Unified lesson runtime for teach, study, and practise."""

from __future__ import annotations

import json
import math
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timedelta
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from pb.core.graph_writer import make_slug
from pb.core.learning_dossier import (
    LearningDossierUpdater,
    LessonDossierSignals,
    question_pattern_summary,
    resolve_subtopic_dossier_key,
)
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.learning_prompting import language_instruction, learning_intent_style_guidance
from pb.core.naming import stored_display_title
from pb.core.question_transform import QuestionTransformService
from pb.core.renderables import renderable_cli_text
from pb.llm.drafts import (
    LessonEvaluationDraft,
    LessonNoteDraft,
    LessonPageDraft,
    LessonPlanDraft,
    LessonQuestionDraft,
    LearningPartnerTurnDraft,
)
from pb.llm.runtime import DraftGenerationError, LLMRuntime
from pb.storage.config import get_config


QUESTION_TIME_THRESHOLDS: dict[str, tuple[int, int]] = {
    "mcq": (12, 30),
    "multi_select": (18, 40),
    "cloze": (15, 35),
    "short_text": (25, 50),
    "error_correction": (35, 70),
    "reorder": (28, 60),
    "free_production": (45, 90),
}

POINTS_PER_HINT = 0.5

RECOGNITION_TYPES = {"mcq", "multi_select", "cloze"}
PRODUCTION_TYPES = {"short_text", "free_production", "error_correction", "reorder"}


def _iso_now() -> str:
    return datetime.utcnow().isoformat()


def _normalize_text(value: object) -> str:
    text = renderable_cli_text(str(value or "")).strip().lower()
    text = "".join(
        char for char in unicodedata.normalize("NFKD", text) if not unicodedata.combining(char)
    )
    text = re.sub(r"\s+", " ", text)
    return text


def _split_answers(raw: str) -> list[str]:
    text = raw.strip()
    if not text:
        return []
    if "|" in text:
        return [item.strip() for item in re.split(r"\s*\|\s*", text) if item.strip()]
    if "\n" in text or ";" in text:
        return [item.strip() for item in re.split(r"\s*(?:;|\n)\s*", text) if item.strip()]
    if re.fullmatch(r"\d+(?:[\s,]+\d+)*", text):
        return [item.strip() for item in re.split(r"[\s,]+", text) if item.strip()]
    return [text]


def _bool_int(value: bool) -> int:
    return 1 if value else 0


def _parse_json_object(raw: str, default: dict[str, Any] | None = None) -> dict[str, Any]:
    payload = (raw or "").strip()
    if not payload:
        return dict(default or {})
    try:
        loaded = json.loads(payload)
    except Exception:
        return dict(default or {})
    return loaded if isinstance(loaded, dict) else dict(default or {})


def _parse_json_list(raw: str, default: list[Any] | None = None) -> list[Any]:
    payload = (raw or "").strip()
    if not payload:
        return list(default or [])
    try:
        loaded = json.loads(payload)
    except Exception:
        return list(default or [])
    return loaded if isinstance(loaded, list) else list(default or [])


def _short_lesson_slug(text: str, *, fallback: str, existing: set[str] | None = None, max_len: int = 27) -> str:
    """Create a short snake-case slug with at most one underscore."""

    existing = existing or set()
    parts = [part for part in re.split(r"[^a-z0-9]+", text.lower().strip()) if part]
    if not parts:
        base = fallback
    elif len(parts) == 1:
        base = parts[0]
    else:
        base = f"{parts[0]}_{''.join(parts[1:])}"
    base = re.sub(r"[^a-z0-9_]+", "", base).strip("_") or fallback
    if base.count("_") > 1:
        head, tail = base.split("_", 1)
        tail = tail.replace("_", "")
        base = f"{head}_{tail}"
    base = base[:max_len].strip("_") or fallback[:max_len]
    slug = base
    counter = 2
    while slug in existing:
        suffix = str(counter)
        slug = f"{base[: max_len - len(suffix)]}{suffix}".strip("_") or f"{fallback[: max_len - len(suffix)]}{suffix}"
        counter += 1
    return slug


def _answer_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(a=_normalize_text(left), b=_normalize_text(right)).ratio()


def _dedupe_preserving_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        clean = str(value).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        deduped.append(clean)
    return deduped


def _normalize_lesson_hints(
    hints: list[str],
    *,
    prompt: str = "",
    skill_label: str = "",
) -> list[str]:
    clean = _dedupe_preserving_order([str(item).strip() for item in hints if str(item).strip()])
    focus = skill_label.strip() or (prompt.strip().splitlines()[0] if prompt.strip() else "the current concept")
    focus = re.sub(r"\s+", " ", focus).strip()[:80] or "the current concept"
    fallback = [
        f"Start from the main contrast in {focus}.",
        "Separate what must be true from what merely sounds related.",
        "Check each part of your answer against the page explanation, then answer only the target asked.",
    ]
    for item in fallback:
        if len(clean) >= 3:
            break
        if item not in clean:
            clean.append(item)
    return clean[:3]


def _coerce_multi_select_shape(question_draft: LessonQuestionDraft) -> None:
    if question_draft.question_type != "multi_select":
        return
    correct = _dedupe_preserving_order(list(question_draft.correct_choices))
    if len(correct) >= 2:
        question_draft.correct_choices = correct
        return
    if correct:
        question_draft.question_type = "mcq"
        question_draft.correct_choices = [correct[0]]
        question_draft.accepted_answers = [correct[0]]
        question_draft.reveal_answer = correct[0]


def _limit_lesson_question_choices(question_draft: LessonQuestionDraft) -> LessonQuestionDraft:
    question_draft.hints = _normalize_lesson_hints(
        list(question_draft.hints),
        prompt=question_draft.prompt,
        skill_label=question_draft.skill_label or question_draft.skill_slug,
    )
    _coerce_multi_select_shape(question_draft)
    if question_draft.question_type not in {"mcq", "multi_select", "cloze"}:
        return question_draft

    choices = _dedupe_preserving_order(list(question_draft.choices))
    priority = _dedupe_preserving_order(
        list(question_draft.correct_choices)
        + ([question_draft.reveal_answer] if str(question_draft.reveal_answer or "").strip() else [])
    )
    capped: list[str] = []
    for item in priority + choices:
        if item not in capped:
            capped.append(item)
        if len(capped) >= 5:
            break

    question_draft.choices = capped
    question_draft.correct_choices = [item for item in _dedupe_preserving_order(list(question_draft.correct_choices)) if item in capped]
    accepted_answers = _dedupe_preserving_order(list(question_draft.accepted_answers))
    if question_draft.correct_choices:
        accepted_answers = question_draft.correct_choices + [
            item for item in accepted_answers if item not in question_draft.correct_choices
        ]
    question_draft.accepted_answers = accepted_answers
    if question_draft.question_type in {"mcq", "cloze"} and question_draft.correct_choices:
        question_draft.reveal_answer = question_draft.correct_choices[0]
    _coerce_multi_select_shape(question_draft)
    return question_draft


@dataclass(frozen=True)
class MultiSelectStats:
    total_options: int
    total_correct: int
    selected_correct: int
    wrong_selections: int
    omissions: int
    selected_values: tuple[str, ...]
    correct_selected_values: tuple[str, ...]


def _resolve_choice_tokens(question: "LessonQuestionRecord", raw_answer: str) -> list[str]:
    tokens = _split_answers(raw_answer)
    choices = [str(item).strip() for item in question.prompt_json.get("choices", []) if str(item).strip()]
    normalized_choice_map = {_normalize_text(choice): choice for choice in choices}
    resolved: list[str] = []
    for token in tokens:
        clean = str(token).strip()
        if not clean:
            continue
        if re.fullmatch(r"\d+", clean):
            index = int(clean) - 1
            if 0 <= index < len(choices):
                resolved.append(choices[index])
                continue
        without_number = re.sub(r"^\s*\d+[\).\s-]+", "", clean).strip()
        mapped = normalized_choice_map.get(_normalize_text(without_number))
        resolved.append(mapped or clean)
    return resolved


def _multi_select_stats(question: "LessonQuestionRecord", raw_answer: str) -> MultiSelectStats:
    choices = [str(item).strip() for item in question.prompt_json.get("choices", []) if str(item).strip()]
    selected_values = tuple(_resolve_choice_tokens(question, raw_answer))
    correct_values = [
        str(item).strip()
        for item in (question.answer_json.get("correct_choices", []) or question.answer_json.get("accepted_answers", []))
        if str(item).strip()
    ]
    correct_norm = {_normalize_text(item) for item in correct_values}
    selected_norm = {_normalize_text(item) for item in selected_values}
    selected_correct_norm = selected_norm & correct_norm
    correct_selected_values = tuple(
        item for item in selected_values if _normalize_text(item) in selected_correct_norm
    )
    total_options = max(len(choices), len(correct_norm), len(selected_norm), 1)
    return MultiSelectStats(
        total_options=total_options,
        total_correct=len(correct_norm),
        selected_correct=len(selected_correct_norm),
        wrong_selections=len(selected_norm - correct_norm),
        omissions=len(correct_norm - selected_norm),
        selected_values=selected_values,
        correct_selected_values=correct_selected_values,
    )


def _resolve_learning_model_binding(runtime: LLMRuntime, policy_name: str) -> str:
    policy = getattr(getattr(runtime.config, "learning", None), "model_policy", None)
    configured = str(getattr(policy, policy_name, "") or "").strip()
    if not configured:
        return runtime.config.model_roles.default
    try:
        bindings = runtime.role_bindings()
    except Exception:
        bindings = {}
    return bindings.get(configured, configured) or runtime.config.model_roles.default


# Bad distractors that indicate the LLM echoed internal categories instead of content
_BAD_DISTRACTOR_SET = frozenset({
    "random detail",
    "surface wording",
    "unrelated exception",
    "none of the above",
    "all of the above",
})

_EMBEDDING_MODEL = "gemini-embedding-2"
_EMBEDDING_RELEVANCE_THRESHOLD = 0.12


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    mag_a = sum(x * x for x in a) ** 0.5
    mag_b = sum(x * x for x in b) ** 0.5
    if not mag_a or not mag_b:
        return 0.0
    return dot / (mag_a * mag_b)


def _get_embedding(text: str) -> list[float] | None:
    """Return an embedding vector for text using gemini-embedding-2. Silent None on any failure."""
    try:
        from pb.llm.gemini import get_client
        client = get_client()
        if not client.is_available():
            return None
        from google.genai import types
        response = client._client.models.embed_content(
            model=_EMBEDDING_MODEL,
            contents=text[:2048],
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        return list(response.embeddings[0].values)
    except Exception:
        return None


def _validate_lesson_draft(
    draft,
    *,
    source_scope: str,
    topic: str,
    has_context: bool,
    runtime: LLMRuntime,
) -> None:
    """Validate generated lesson content before it is persisted.

    Raises DraftGenerationError(empty_category) if structural checks fail.
    Embedding relevance check (when context is present) is best-effort: any
    embedding failure silently skips that check.
    """
    from pb.llm.runtime import DraftAttempt, ProviderErrorDetails

    def _fail(reason: str) -> None:
        raise DraftGenerationError(
            source_scope=source_scope,
            prompt_template_version="",
            attempts=[],
            error=ProviderErrorDetails(
                category="empty",
                provider="validation",
                model="deterministic",
                raw_message=reason,
                http_status=None,
                retryable=False,
            ),
        )

    # --- Deterministic structural validation ---
    for page in draft.pages:
        page_title_norm = (page.title or "").strip().lower()
        for question in page.questions:
            # 1. Internal label leak: skill_label identical to page title
            skill_norm = (question.skill_label or "").strip().lower()
            if skill_norm and skill_norm == page_title_norm:
                _fail(
                    f"Question skill_label '{question.skill_label}' matches page title "
                    f"'{page.title}' — internal label leaked into content."
                )

            # 2. Generic bad distractors (MCQ only)
            if question.question_type in {"mcq", "multi_select"}:
                correct_norm = {c.strip().lower() for c in question.correct_choices if c.strip()}
                distractors = [
                    c.strip().lower()
                    for c in question.choices
                    if c.strip().lower() not in correct_norm and c.strip()
                ]
                if distractors and all(d in _BAD_DISTRACTOR_SET for d in distractors):
                    _fail(
                        f"MCQ question '{question.prompt[:60]}' has only generic bad distractors "
                        f"({distractors}) — no diagnostic value."
                    )

            # 3. Missing answer
            has_accepted = any(str(a).strip() for a in question.accepted_answers)
            has_reveal = bool(str(question.reveal_answer or "").strip())
            has_correct = any(str(c).strip() for c in question.correct_choices)
            if not has_accepted and not has_reveal and not has_correct:
                _fail(
                    f"Question '{question.prompt[:60]}' has no accepted_answers, "
                    f"correct_choices, or reveal_answer."
                )

    # --- Embedding relevance check (only when context documents are present) ---
    if not has_context:
        return

    topic_vec = _get_embedding(topic)
    if topic_vec is None:
        return  # embedding unavailable — skip silently

    for page in draft.pages:
        for question in page.questions:
            prompt_text = str(question.prompt or "").strip()
            if not prompt_text:
                continue
            q_vec = _get_embedding(prompt_text)
            if q_vec is None:
                continue
            sim = _cosine_similarity(topic_vec, q_vec)
            if sim < _EMBEDDING_RELEVANCE_THRESHOLD:
                _fail(
                    f"Question '{prompt_text[:60]}' appears off-topic "
                    f"(embedding similarity to topic = {sim:.3f} < {_EMBEDDING_RELEVANCE_THRESHOLD})."
                )


class LessonRunRecord(BaseModel):
    """Top-level persisted lesson state for one active session."""

    id: str
    session_id: str
    task_id: str
    branch: str = "study"
    lesson_mode: str = "study"
    title: str = ""
    lesson_status: str = "active"
    active_page_slug: str = ""
    active_question_slug: str = ""
    active_page_index: int = 0
    active_question_index: int = 0
    total_points: float = 0.0
    ready_to_finish: bool = False
    note_path: str = ""
    retry_queue: list[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_iso_now)
    updated_at: str = Field(default_factory=_iso_now)


class LessonPageRecord(BaseModel):
    """Persisted page state inside a lesson run."""

    id: str
    lesson_run_id: str
    session_id: str
    page_slug: str
    title: str
    intro_text: str = ""
    sequence_index: int = 0
    status: str = "pending"
    question_count: int = 0
    created_at: str = Field(default_factory=_iso_now)
    updated_at: str = Field(default_factory=_iso_now)


class LessonQuestionRecord(BaseModel):
    """Persisted question state inside a lesson page."""

    id: str
    lesson_run_id: str
    session_id: str
    page_slug: str
    question_slug: str
    skill_slug: str = ""
    question_type: str
    prompt_json: dict[str, Any] = Field(default_factory=dict)
    answer_json: dict[str, Any] = Field(default_factory=dict)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    sequence_index: int = 0
    status: str = "pending"
    hint_level: int = 0
    revealed: bool = False
    mastered: bool = False
    queued_retry: bool = False
    retry_of_question_slug: str = ""
    retry_generation: int = 0
    next_review_at: str = ""
    created_at: str = Field(default_factory=_iso_now)
    updated_at: str = Field(default_factory=_iso_now)


class LessonAttemptRecord(BaseModel):
    """Persisted attempt-level evidence for one lesson answer."""

    id: str
    lesson_run_id: str
    session_id: str
    page_slug: str
    question_slug: str
    skill_slug: str = ""
    answer_text: str = ""
    result: str
    response_ms: int = 0
    hint_level: int = 0
    points_delta: float = 0.0
    error_tags: list[str] = Field(default_factory=list)
    evaluator_confidence: float | None = None
    model_used: str = ""
    created_at: str = Field(default_factory=_iso_now)


class LessonSkillStateRecord(BaseModel):
    """Aggregated lesson diagnostics for one skill slug."""

    id: str
    lesson_run_id: str
    session_id: str
    skill_slug: str
    recognition_status: str = "fragile"
    production_status: str = "fragile"
    overall_status: str = "fragile"
    error_tags: list[str] = Field(default_factory=list)
    attempt_count: int = 0
    next_review_at: str = ""
    updated_at: str = Field(default_factory=_iso_now)


@dataclass
class LessonSnapshot:
    """Current renderable lesson state."""

    run: LessonRunRecord
    page: LessonPageRecord | None
    question: LessonQuestionRecord | None
    page_questions: list[LessonQuestionRecord]
    pages: list[LessonPageRecord]
    footer_commands: list[str]
    feedback_lines: list[str]
    header_note: str = ""


class LessonNoteWriter:
    """Upsert a canonical knowledge dossier from the persisted lesson state."""

    def __init__(self, runtime_ctx, runtime: LLMRuntime):
        self.runtime_ctx = runtime_ctx
        self.runtime = runtime

    def write_note(
        self,
        *,
        repo,
        run: LessonRunRecord,
        task,
        session,
        topic: str,
        domain: str,
    ) -> Path | None:
        if run.lesson_mode not in {"teach", "study"}:
            return None

        pages = repo.list_lesson_pages(run.id)
        questions = repo.list_lesson_questions(run.id)
        attempts = repo.list_lesson_attempts(run.id)
        skill_states = repo.list_lesson_skill_states(run.id)

        draft = self._draft_note(
            run=run,
            task=task,
            topic=topic,
            domain=domain,
            pages=pages,
            questions=questions,
            attempts=attempts,
            skill_states=skill_states,
        )
        if draft is None:
            return None

        updater = LearningDossierUpdater(Path(self.runtime_ctx.vault_path))
        key = resolve_subtopic_dossier_key(
            session=session,
            task=task,
            domain=domain,
            subtopic=topic,
        )
        return updater.upsert(
            key=key,
            session=session,
            task=task,
            lesson=LessonDossierSignals(
                summary=draft.summary or f"Worked through {topic}.",
                explanation=draft.intuitive_explanation,
                key_points=tuple(draft.key_points),
                misconceptions=tuple(draft.misconceptions),
                next_moves=tuple(draft.next_moves),
                question_patterns=tuple(question_pattern_summary(attempts, questions)),
                fragile_concepts=tuple(
                    state.skill_slug.replace("_", " ")
                    for state in skill_states
                    if state.overall_status != "strong"
                ),
            ),
        )

    def _draft_note(
        self,
        *,
        run: LessonRunRecord,
        task,
        topic: str,
        domain: str,
        pages: list[LessonPageRecord],
        questions: list[LessonQuestionRecord],
        attempts: list[LessonAttemptRecord],
        skill_states: list[LessonSkillStateRecord],
    ) -> LessonNoteDraft | None:
        if not self.runtime.health().available:
            return self._fallback_note(run=run, topic=topic, pages=pages, skill_states=skill_states)

        prompt = (
            "Write a rich but compact study note for a completed lesson.\n"
            "Use intuitive, low-jargon language and preserve conceptual fidelity.\n"
            "Do not include markdown fences.\n"
            f"Topic: {topic}\n"
            f"Domain: {domain}\n"
            f"Lesson mode: {run.lesson_mode}\n"
            f"Task title: {getattr(task, 'title', '')}\n"
            f"Pages: {[{'title': page.title, 'intro': page.intro_text} for page in pages]}\n"
            f"Question prompts: {[question.prompt_json.get('prompt', '') for question in questions[:12]]}\n"
            f"Attempts: {[attempt.model_dump(mode='json') for attempt in attempts[:18]]}\n"
            f"Skill states: {[state.model_dump(mode='json') for state in skill_states]}\n"
        )
        try:
            return self.runtime.generate_draft(
                LessonNoteDraft,
                prompt,
                source_scope=f"lesson_note:{run.id}",
                model=_resolve_learning_model_binding(self.runtime, "session_explain"),
                max_output_tokens=4000,
            ).payload
        except DraftGenerationError:
            return self._fallback_note(run=run, topic=topic, pages=pages, skill_states=skill_states)

    @staticmethod
    def _fallback_note(
        *,
        run: LessonRunRecord,
        topic: str,
        pages: list[LessonPageRecord],
        skill_states: list[LessonSkillStateRecord],
    ) -> LessonNoteDraft:
        fragile = [state.skill_slug.replace("_", " ") for state in skill_states if state.overall_status != "strong"]
        return LessonNoteDraft(
            title=f"{topic} lesson",
            summary=f"Worked through {len(pages)} page(s) on {topic} in {run.lesson_mode} mode.",
            intuitive_explanation=f"The lesson kept returning to the central idea behind {topic} and forced active use instead of passive reading.",
            key_points=[page.title for page in pages[:5]],
            misconceptions=fragile[:5],
            next_moves=[f"Revisit {fragile[0]} with one fresh example."] if fragile else [f"Review {topic} with one transfer example."],
        )


class LessonEngine:
    """Persistent lesson state machine used by teach, study, and practise."""

    def __init__(
        self,
        *,
        runtime: LLMRuntime,
        runtime_ctx,
        repo,
        task,
        session,
        branch: str,
        objective: str,
        topic: str,
        domain: str,
        mode: str = "",
        clarifier_answers: dict[str, str] | None = None,
        confidence_level: float = 0.0,
    ):
        self.runtime = runtime
        self.runtime_ctx = runtime_ctx
        self.repo = repo
        self.task = task
        self.session = session
        self.branch = branch
        self.objective = objective
        self.topic = topic
        self.domain = domain
        self.mode = mode
        self.clarifier_answers = clarifier_answers or {}
        self.confidence_level = max(0.0, min(1.0, float(confidence_level)))
        self.lesson_mode = self._resolve_lesson_mode()
        self.last_feedback: list[str] = []
        self._question_opened_at: dict[str, datetime] = {}
        self.ensure_initialized()

    def _resolve_lesson_mode(self) -> str:
        lowered_branch = (self.branch or "").strip().lower()
        lowered_mode = (self.mode or "").strip().lower()
        if lowered_branch == "teach":
            return "teach"
        if lowered_branch == "study" and "teach" in lowered_mode:
            return "teach"
        if lowered_branch in {"practise", "practice"}:
            return "practise"
        return "study"

    def ensure_initialized(self) -> None:
        run = self.repo.get_lesson_run(self.session.id)
        if run is not None:
            self._touch_question_clock(run)
            self._sync_session_pointer(run)
            return

        plan = self._plan_lesson()
        pages = list(plan.pages or [])
        lesson_title = plan.lesson_title.strip() or stored_display_title(self.task) or self.topic or "Lesson"
        run = LessonRunRecord(
            id=self.session.id,
            session_id=self.session.id,
            task_id=self.task.id,
            branch=self.branch,
            lesson_mode=self.lesson_mode,
            title=lesson_title,
            lesson_status="active",
            created_at=_iso_now(),
            updated_at=_iso_now(),
        )
        self.repo.create_lesson_run(run)

        page_slug_seen: set[str] = set()
        question_slug_seen: set[str] = set()
        first_page_slug = ""
        first_question_slug = ""
        for page_index, page_draft in enumerate(pages):
            page_slug = _short_lesson_slug(page_draft.title or f"page {page_index + 1}", fallback=f"page{page_index + 1}", existing=page_slug_seen)
            page_slug_seen.add(page_slug)
            page = LessonPageRecord(
                id=f"{run.id}:{page_slug}",
                lesson_run_id=run.id,
                session_id=self.session.id,
                page_slug=page_slug,
                title=page_draft.title,
                intro_text=page_draft.intro or page_draft.focus,
                sequence_index=page_index,
                status="pending",
                question_count=len(page_draft.questions),
            )
            self.repo.create_lesson_page(page)
            if not first_page_slug:
                first_page_slug = page_slug
            for question_index, question_draft in enumerate(page_draft.questions):
                question_slug = _short_lesson_slug(
                    question_draft.skill_slug or question_draft.title or question_draft.prompt,
                    fallback=f"q{page_index + 1}{question_index + 1}",
                    existing=question_slug_seen,
                )
                question_slug_seen.add(question_slug)
                display_items = list(question_draft.ordered_items)
                if question_draft.question_type == "reorder" and display_items:
                    if len(display_items) > 1:
                        display_items = display_items[1:] + display_items[:1]
                prompt_json = {
                    "title": question_draft.title,
                    "prompt": question_draft.prompt,
                    "choices": list(question_draft.choices),
                    "display_items": display_items,
                }
                answer_json = {
                    "accepted_answers": list(question_draft.accepted_answers),
                    "correct_choices": list(question_draft.correct_choices),
                    "ordered_items": list(question_draft.ordered_items),
                    "hints": list(question_draft.hints),
                    "reveal_answer": question_draft.reveal_answer,
                    "error_tags": list(question_draft.error_tags),
                    "skill_label": question_draft.skill_label,
                    "page_intro": question_draft.page_intro,
                    "evaluator_notes": question_draft.evaluator_notes,
                }
                question = LessonQuestionRecord(
                    id=f"{run.id}:{question_slug}",
                    lesson_run_id=run.id,
                    session_id=self.session.id,
                    page_slug=page_slug,
                    question_slug=question_slug,
                    skill_slug=_short_lesson_slug(
                        question_draft.skill_slug or question_draft.skill_label or page_draft.title,
                        fallback=f"skill{page_index + 1}",
                    ),
                    question_type=question_draft.question_type,
                    prompt_json=prompt_json,
                    answer_json=answer_json,
                    metadata_json=QuestionTransformService.initial_metadata(
                        prompt_json=prompt_json,
                        answer_json=answer_json,
                        active_context_ids=self._active_context_identifiers(),
                    ),
                    sequence_index=question_index,
                    status="pending",
                )
                self.repo.create_lesson_question(question)
                if not first_question_slug:
                    first_question_slug = question_slug

        run.active_page_slug = first_page_slug
        run.active_question_slug = first_question_slug
        run.active_page_index = 0
        run.active_question_index = 0
        run.updated_at = _iso_now()
        self.repo.update_lesson_run(run)
        self._touch_question_clock(run)
        self._sync_session_pointer(run)

    def _sync_session_pointer(self, run: LessonRunRecord) -> None:
        generated = dict(getattr(self.session, "generated_names", {}) or {})
        generated["lesson_run_id"] = run.id
        generated["lesson_status"] = run.lesson_status
        generated["lesson_progress"] = {
            "page_slug": run.active_page_slug,
            "question_slug": run.active_question_slug,
            "points": run.total_points,
            "ready_to_finish": run.ready_to_finish,
        }
        self.session.generated_names = generated
        self.repo.update_session(self.session)

    def _touch_question_clock(self, run: LessonRunRecord) -> None:
        if run.active_question_slug and run.active_question_slug not in self._question_opened_at:
            self._question_opened_at[run.active_question_slug] = datetime.utcnow()

    def _active_context_identifiers(self) -> list[str]:
        generated = dict(getattr(self.session, "generated_names", {}) or {})
        active_context = generated.get("active_context_scope")
        if not isinstance(active_context, dict):
            return []
        identifiers: list[str] = []
        for key in ("source_bundle_id", "domain_id"):
            value = str(active_context.get(key, "") or "").strip()
            if value:
                identifiers.append(value)
        for source_ref in list(active_context.get("source_refs", []) or []):
            clean = str(source_ref).strip()
            if clean:
                identifiers.append(clean)
        seen: set[str] = set()
        ordered: list[str] = []
        for item in identifiers:
            if item in seen:
                continue
            seen.add(item)
            ordered.append(item)
        return ordered

    def _plan_lesson(self) -> LessonPlanDraft:
        health = self.runtime.health()
        if not health.available:
            return self._fallback_lesson_plan()

        meta = parse_learning_task_metadata(self.task)
        feynman_persona = ""
        if "feynman" in (self.mode or "").lower():
            feynman_persona = (
                "\n\nTutor persona: Communicate with the precision of Richard Feynman, "
                "the accessibility of Matt Parker, and the wonder of Carl Sagan. "
                "Use layman terms wherever they are equally accurate. "
                "Use academic terms only where no accurate layman equivalent exists — "
                "and when you do, unpack them briefly. "
                "Never perform jargon for its own sake. "
                "Ask one targeted follow-up question per identified gap. "
                "Questions must be answerable in 2 sentences or fewer (35-40 words max). "
                "First move: invite the learner to explain the concept in their own words. "
                "Identify missing elements and gaps in their explanation. "
                "Then address one gap at a time with a direct follow-up question.\n"
            )
        prompt = (
            language_instruction(self.topic, configured=get_config().ui.language)
            + "Create a page-based lesson plan for ProductiveBrain.\n"
            "Return a compact lesson with 3-4 pages. Each page should have 2-4 questions and no filler.\n"
            "Modes:\n"
            "- teach: explanation -> guided checks -> application\n"
            "- study: recall -> concept repair -> transfer\n"
            "- practise: drill -> feedback -> repeated production\n"
            "Question types must vary across the lesson and may include mcq, multi_select, cloze, short_text, free_production, error_correction, reorder.\n"
            "Strongly prefer mcq, multi_select, and cloze. Use short_text, free_production, or error_correction only when recognition clearly cannot test the target.\n"
            "Include exactly 3 distinct hints for each question, ordered from light nudge to stronger scaffold, plus a reveal answer.\n"
            "Keep choices keyboard-friendly. For cloze, include [____] in the prompt.\n"
            "For mcq, multi_select, and cloze, include at most 5 model-generated choices because the UI reserves a sixth slot for inline typing.\n"
            "For multi_select, use 4-5 options whenever possible and include at least 2 correct choices; if only one answer is correct, use mcq instead.\n"
            "If a typed answer would normally require accents, diacritics, or other hard-to-type characters, add easy ASCII equivalents to accepted_answers.\n"
            + learning_intent_style_guidance()
            + f"Topic: {self.topic}\n"
            + f"Domain: {self.domain}\n"
            + f"Objective: {self.objective}\n"
            + f"Lesson mode: {self.lesson_mode}\n"
            + f"Learner confidence level: {self.confidence_level:.2f} (0.0 = new topic, 1.0 = mastered)\n"
            + (
                "COLD START — confidence is below 0.3: the first page MUST be an orientation "
                "or concept-explanation page. Do NOT open with recall or quiz questions before "
                "the learner has been given material to retrieve.\n"
                if self.confidence_level < 0.3 else ""
            )
            + f"Clarifier answers: {self.clarifier_answers}\n"
            + f"Task metadata steps: {meta.steps or []}\n"
            + feynman_persona
        )
        result = self.runtime.generate_draft(
            LessonPlanDraft,
            prompt,
            source_scope=f"lesson_plan:{self.session.id}",
            model=_resolve_learning_model_binding(self.runtime, "lesson_planning"),
            max_output_tokens=15000,
        )
        draft = result.payload
        if not draft.pages:
            from pb.llm.runtime import DraftAttempt, ProviderErrorDetails
            raise DraftGenerationError(
                source_scope=f"lesson_plan:{self.session.id}",
                prompt_template_version=self.runtime.config.llm.prompt_template_version,
                attempts=list(result.attempts),
                error=ProviderErrorDetails(
                    category="empty",
                    provider=result.model.split(":")[0] if ":" in result.model else result.model,
                    model=result.model.split(":", 1)[1] if ":" in result.model else result.model,
                    raw_message="LLM returned a lesson plan with no pages.",
                    http_status=200,
                    retryable=False,
                ),
            )
        for page in draft.pages:
            for question in page.questions:
                _limit_lesson_question_choices(question)
        _validate_lesson_draft(
            draft,
            source_scope=f"lesson_plan:{self.session.id}",
            topic=self.topic,
            has_context=bool(self._active_context_identifiers()),
            runtime=self.runtime,
        )
        return draft

    def _fallback_lesson_plan(self) -> LessonPlanDraft:
        topic = self.topic or stored_display_title(self.task) or "the concept"
        domain = self.domain or "learning"
        skill_base = _short_lesson_slug(topic, fallback="concept")
        hints = [
            f"Anchor the answer in {topic}, not in a nearby-sounding detail.",
            "Name the rule, then test whether the example actually follows it.",
            "Use the page explanation to separate the correct mechanism from distractors.",
        ]
        pages = [
            LessonPageDraft(
                title="Orient the Concept",
                focus=f"Build a stable first pass on {topic}.",
                intro=f"{topic} is being studied inside {domain}. Start by distinguishing the core mechanism from adjacent facts.",
                questions=[
                    LessonQuestionDraft(
                        title="Core discrimination",
                        prompt=f"Which statement best captures the learning target for {topic}?",
                        question_type="mcq",
                        skill_slug=f"{skill_base}_core",
                        skill_label=f"{topic} core",
                        choices=[
                            f"Explain {topic} by naming the mechanism and when it applies.",
                            f"Memorize an isolated phrase about {topic} without using it.",
                            f"Treat every example in {domain} as equivalent.",
                            "Skip the concept and only track completion.",
                        ],
                        accepted_answers=[f"Explain {topic} by naming the mechanism and when it applies."],
                        correct_choices=[f"Explain {topic} by naming the mechanism and when it applies."],
                        hints=hints,
                        reveal_answer=f"Explain {topic} by naming the mechanism and when it applies.",
                    ),
                    LessonQuestionDraft(
                        title="Explain back",
                        prompt=f"Explain {topic} in one or two sentences without just naming the term.",
                        question_type="free_production",
                        skill_slug=f"{skill_base}_core",
                        skill_label=f"{topic} core",
                        accepted_answers=[
                            f"{topic} is understood when I can explain the mechanism and apply it to a fresh case.",
                        ],
                        hints=hints,
                        reveal_answer=f"{topic} is understood when I can explain the mechanism and apply it to a fresh case.",
                    ),
                ],
            ),
            LessonPageDraft(
                title="Check the Mechanism",
                focus="Separate valid signals from distractors.",
                intro=f"A useful understanding of {topic} can pick out what matters and ignore plausible but wrong cues.",
                questions=[
                    LessonQuestionDraft(
                        title="Signal set",
                        prompt=f"Select the signals that would show real understanding of {topic}.",
                        question_type="multi_select",
                        skill_slug=f"{skill_base}_signals",
                        skill_label=f"{topic} signals",
                        choices=[
                            "States the mechanism in plain language.",
                            "Applies it to a fresh example.",
                            "Repeats only a memorized label.",
                            "Confuses it with an unrelated neighboring idea.",
                            "Explains when the idea would not apply.",
                        ],
                        accepted_answers=[
                            "States the mechanism in plain language.",
                            "Applies it to a fresh example.",
                            "Explains when the idea would not apply.",
                        ],
                        correct_choices=[
                            "States the mechanism in plain language.",
                            "Applies it to a fresh example.",
                            "Explains when the idea would not apply.",
                        ],
                        hints=hints,
                        reveal_answer="States the mechanism in plain language | Applies it to a fresh example | Explains when the idea would not apply.",
                    ),
                    LessonQuestionDraft(
                        title="Short explanation",
                        prompt=f"In one sentence, what makes {topic} worth studying for your current goal?",
                        question_type="short_text",
                        skill_slug=f"{skill_base}_why",
                        skill_label=f"{topic} purpose",
                        accepted_answers=[
                            f"It helps me use {topic} toward the stated learning goal.",
                            f"{topic} supports the current goal by making the concept usable, not just familiar.",
                        ],
                        hints=hints,
                        reveal_answer=f"It helps me use {topic} toward the stated learning goal.",
                    ),
                ],
            ),
            LessonPageDraft(
                title="Repair and Transfer",
                focus="Produce, correct, and order the idea.",
                intro=f"Finish by using {topic} actively: correct an error, order the steps, and produce a transfer answer.",
                questions=[
                    LessonQuestionDraft(
                        title="Error correction",
                        prompt=f"Correct this claim: '{topic} is learned once I recognize the term.'",
                        question_type="error_correction",
                        skill_slug=f"{skill_base}_repair",
                        skill_label=f"{topic} repair",
                        accepted_answers=[
                            f"{topic} is learned when I can explain and apply it, not just recognize the term.",
                        ],
                        hints=hints,
                        reveal_answer=f"{topic} is learned when I can explain and apply it, not just recognize the term.",
                    ),
                    LessonQuestionDraft(
                        title="Order the loop",
                        prompt=f"Put the {topic} learning loop in order.",
                        question_type="reorder",
                        skill_slug=f"{skill_base}_order",
                        skill_label=f"{topic} sequence",
                        ordered_items=[
                            "State the goal.",
                            "Explain the mechanism.",
                            "Apply it to a fresh case.",
                        ],
                        hints=hints,
                        reveal_answer="State the goal -> Explain the mechanism -> Apply it to a fresh case.",
                    ),
                    LessonQuestionDraft(
                        title="Transfer answer",
                        prompt=f"A compact label for the current lesson target is [____].",
                        question_type="cloze",
                        skill_slug=f"{skill_base}_transfer",
                        skill_label=f"{topic} transfer",
                        choices=[topic, domain, "completion", "metadata"],
                        accepted_answers=[topic],
                        correct_choices=[topic],
                        hints=hints,
                        reveal_answer=topic,
                    ),
                ],
            ),
        ]
        for page in pages:
            for question in page.questions:
                _limit_lesson_question_choices(question)
        return LessonPlanDraft(
            lesson_title=f"{topic} lesson",
            summary=f"Deterministic lesson plan for {topic}.",
            mode=self.lesson_mode if self.lesson_mode in {"teach", "study", "practise"} else "study",
            pages=pages,
        )

    def snapshot_for(self, *, page_slug: str = "", question_slug: str = "") -> LessonSnapshot:
        run = self.repo.get_lesson_run(self.session.id)
        if run is None:
            self.ensure_initialized()
            run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        pages = self.repo.list_lesson_pages(run.id)
        resolved_question_slug = question_slug or run.active_question_slug
        question = self.repo.get_lesson_question(run.id, resolved_question_slug) if resolved_question_slug else None
        resolved_page_slug = page_slug or (question.page_slug if question is not None else run.active_page_slug)
        page = self.repo.get_lesson_page(run.id, resolved_page_slug) if resolved_page_slug else None
        page_questions = self.repo.list_lesson_questions(run.id, page.page_slug if page is not None else None) if page is not None else []
        header_note = ""
        if run.ready_to_finish:
            header_note = "Lesson cleared. Use /finish when you want to close the session."
        footer_commands = ["/hint", "/answer", "/harder", "/easier", "/intuitive"]
        if question is not None and question.revealed:
            footer_commands.append("/skip")
        return LessonSnapshot(
            run=run,
            page=page,
            question=question,
            page_questions=page_questions,
            pages=pages,
            footer_commands=footer_commands,
            feedback_lines=list(self.last_feedback),
            header_note=header_note,
        )

    def current_snapshot(self) -> LessonSnapshot:
        return self.snapshot_for()

    def turn_for(self, *, page_slug: str = "", question_slug: str = "") -> LearningPartnerTurnDraft:
        snapshot = self.snapshot_for(page_slug=page_slug, question_slug=question_slug)
        question = snapshot.question
        page = snapshot.page
        if snapshot.run.ready_to_finish or question is None:
            return LearningPartnerTurnDraft(
                reply=snapshot.header_note or "Lesson ready. Use /finish to close the session.",
                current_step_index=(page.sequence_index + 1) if page is not None else 0,
                total_steps=len([item for item in snapshot.pages if item.page_slug != "mistakes"]),
                current_step_title=page.title if page is not None else "Lesson complete",
                current_objective=page.intro_text if page is not None else "",
                step_status=snapshot.run.lesson_status,
                corrections=list(snapshot.feedback_lines),
                question_type="free_text",
            )

        prompt = str(question.prompt_json.get("prompt", "") or "")
        choices = [str(item).strip() for item in question.prompt_json.get("choices", []) if str(item).strip()][:5]
        cloze_choices = choices if question.question_type == "cloze" else []
        if question.question_type == "reorder":
            display_items = [str(item).strip() for item in question.prompt_json.get("display_items", []) if str(item).strip()]
            if display_items:
                joined = "\n".join(f"{index}. {item}" for index, item in enumerate(display_items, start=1))
                prompt = f"{prompt}\n{joined}\nType the order as numbers, e.g. `2 1 3`."
        return LearningPartnerTurnDraft(
            reply=prompt,
            corrections=list(snapshot.feedback_lines),
            current_step_index=(page.sequence_index + 1) if page is not None else 0,
            total_steps=len([item for item in snapshot.pages if item.page_slug != "mistakes"]),
            current_step_title=page.title if page is not None else "",
            current_objective=page.intro_text if page is not None else "",
            success_check=self._page_progress_label(page.page_slug if page is not None else ""),
            step_status="retry" if question.retry_of_question_slug else "active",
            question_type=question.question_type,
            mcq_options=choices if question.question_type in {"mcq", "multi_select"} else [],
            cloze_blank_options=cloze_choices,
            support_cards=[page.intro_text] if page is not None and page.intro_text else [],
            next_action=snapshot.header_note,
        )

    def current_turn(self) -> LearningPartnerTurnDraft:
        return self.turn_for()

    def answer_current(self, raw_answer: str) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        if question is None:
            self.last_feedback = ["Lesson state could not find the active question."]
            return self.current_turn()

        opened_at = self._question_opened_at.get(question.question_slug, datetime.utcnow())
        response_ms = max(0, int((datetime.utcnow() - opened_at).total_seconds() * 1000))
        attempts = self.repo.list_lesson_attempts(run.id, question.question_slug)
        evaluation = self._evaluate_answer(question=question, raw_answer=raw_answer)
        points_delta = self._points_delta(
            question=question,
            result=evaluation.result,
            response_ms=response_ms,
            attempts=attempts,
            raw_answer=raw_answer,
        )
        attempt = LessonAttemptRecord(
            id=_short_lesson_slug(f"attempt {len(attempts) + 1} {question.question_slug}", fallback="attempt"),
            lesson_run_id=run.id,
            session_id=self.session.id,
            page_slug=question.page_slug,
            question_slug=question.question_slug,
            skill_slug=question.skill_slug,
            answer_text=raw_answer,
            result=evaluation.result,
            response_ms=response_ms,
            hint_level=question.hint_level,
            points_delta=points_delta,
            error_tags=list(evaluation.error_tags),
            evaluator_confidence=evaluation.confidence,
            model_used=_resolve_learning_model_binding(
                self.runtime,
                "complex_free_response_eval" if question.question_type in PRODUCTION_TYPES else "answer_check",
            ) if self.runtime.health().available else "",
        )
        self.repo.create_lesson_attempt(attempt)
        run.total_points += points_delta

        if evaluation.result in {"wrong", "close"}:
            if not question.queued_retry:
                self._enqueue_retry_question(run=run, question=question)
                question.queued_retry = True
            recent_attempts = [*attempts, attempt]
            if self._consecutive_misses(recent_attempts) >= 3:
                self._retune_pending_retry_for_easier_verification(run=run, question=question)
                question.revealed = True
                question.mastered = False
                question.status = "revealed"
                question.next_review_at = self._review_signal(question=question, result="revealed")
                question.updated_at = _iso_now()
                self.repo.update_lesson_question(question)
                self.last_feedback = self._forced_reveal_feedback(question)
                run.updated_at = _iso_now()
                self.repo.update_lesson_run(run)
                self._sync_session_pointer(run)
                self._update_skill_state(run.id, question.skill_slug)
                return self.current_turn()
            hint_text = self._hint_for(question)
            question.hint_level = min(question.hint_level + 1, 3)
            question.updated_at = _iso_now()
            self.repo.update_lesson_question(question)
            self.last_feedback = [line for line in [evaluation.feedback, hint_text] if line.strip()]
            run.updated_at = _iso_now()
            self.repo.update_lesson_run(run)
            self._update_skill_state(run.id, question.skill_slug)
            return self.current_turn()

        if evaluation.result == "revealed":
            if not question.queued_retry:
                self._enqueue_retry_question(run=run, question=question)
                question.queued_retry = True
            question.revealed = True
            question.mastered = False
            question.status = "revealed"
            question.next_review_at = self._review_signal(question=question, result="revealed")
            question.updated_at = _iso_now()
            self.repo.update_lesson_question(question)
            self.last_feedback = [evaluation.feedback or self._reveal_answer(question)]
            self._advance_after_resolution(run=run, question=question)
            self._update_skill_state(run.id, question.skill_slug)
            return self.current_turn()

        if evaluation.result == "skipped":
            if not question.queued_retry:
                self._enqueue_retry_question(run=run, question=question)
                question.queued_retry = True
            question.mastered = False
            question.status = "skipped"
            question.next_review_at = self._review_signal(question=question, result="skipped")
            question.updated_at = _iso_now()
            self.repo.update_lesson_question(question)
            self.last_feedback = [evaluation.feedback or "Skipped. This item will come back in the mistake loop."]
            self._advance_after_resolution(run=run, question=question)
            self._update_skill_state(run.id, question.skill_slug)
            return self.current_turn()

        question.mastered = not question.retry_of_question_slug and not attempts and question.hint_level == 0
        if question.retry_of_question_slug and not question.revealed:
            question.mastered = True
        question.status = "correct"
        question.next_review_at = self._review_signal(question=question, result="correct")
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        if question.retry_of_question_slug and question.metadata_json.get("retry_strategy") == "easier_then_verify":
            self._enqueue_verification_question(run=run, retry_question=question)
        self.last_feedback = [evaluation.feedback or "Correct."]
        self._advance_after_resolution(run=run, question=question)
        self._update_skill_state(run.id, question.skill_slug)
        return self.current_turn()

    def reveal_current_answer(self) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        if question is None:
            self.last_feedback = ["No active question to reveal."]
            return self.current_turn()
        if not question.queued_retry:
            self._enqueue_retry_question(run=run, question=question)
            question.queued_retry = True
        question.revealed = True
        question.status = "revealed"
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        self.last_feedback = [self._reveal_answer(question)]
        self._update_skill_state(run.id, question.skill_slug)
        return self.current_turn()

    def skip_current_question(self) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        if question is None:
            self.last_feedback = ["No active question to skip."]
            return self.current_turn()
        return self.answer_with_forced_result(question=question, result="skipped", answer_text="/skip")

    def answer_with_forced_result(self, *, question: LessonQuestionRecord, result: str, answer_text: str) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        opened_at = self._question_opened_at.get(question.question_slug, datetime.utcnow())
        response_ms = max(0, int((datetime.utcnow() - opened_at).total_seconds() * 1000))
        attempts = self.repo.list_lesson_attempts(run.id, question.question_slug)
        points_delta = self._points_delta(question=question, result=result, response_ms=response_ms, attempts=attempts)
        self.repo.create_lesson_attempt(
            LessonAttemptRecord(
                id=_short_lesson_slug(f"attempt {len(attempts) + 1} {question.question_slug}", fallback="attempt"),
                lesson_run_id=run.id,
                session_id=self.session.id,
                page_slug=question.page_slug,
                question_slug=question.question_slug,
                skill_slug=question.skill_slug,
                answer_text=answer_text,
                result=result,
                response_ms=response_ms,
                hint_level=question.hint_level,
                points_delta=points_delta,
                error_tags=list(question.answer_json.get("error_tags", [])),
            )
        )
        run.total_points += points_delta
        if not question.queued_retry:
            self._enqueue_retry_question(run=run, question=question)
            question.queued_retry = True
        question.mastered = False
        question.revealed = question.revealed or result == "revealed"
        question.status = result
        question.next_review_at = self._review_signal(question=question, result=result)
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        run.updated_at = _iso_now()
        self.repo.update_lesson_run(run)
        if result == "revealed":
            self.last_feedback = [self._reveal_answer(question)]
        elif result == "skipped":
            self.last_feedback = ["Skipped. This question will return in the mistake loop."]
        else:
            self.last_feedback = [f"Recorded as {result}."]
        self._advance_after_resolution(run=run, question=question)
        self._update_skill_state(run.id, question.skill_slug)
        return self.current_turn()

    def use_hint(self) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        if question is None:
            self.last_feedback = ["No active question."]
            return self.current_turn()
        if question.revealed or question.hint_level >= 3:
            return self._reveal_after_exhausted_hints(run=run, question=question)
        hint_text = self._hint_for(question)
        question.hint_level = min(question.hint_level + 1, 3)
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        self.last_feedback = [hint_text]
        return self.current_turn()

    def explain_current(self, *, intuitive: bool = False) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        page = self.repo.get_lesson_page(run.id, run.active_page_slug)
        if question is None:
            self.last_feedback = ["No active question to explain."]
            return self.current_turn()
        explanation = self._generate_explanation(question=question, page=page, intuitive=intuitive)
        self.last_feedback = [explanation]
        return self.current_turn()

    def drill_current(self) -> LearningPartnerTurnDraft:
        return self._replace_current_question(transform="drill")

    def change_difficulty(self, direction: str) -> LearningPartnerTurnDraft:
        return self._replace_current_question(transform=direction)

    def _replace_current_question(self, *, transform: str) -> LearningPartnerTurnDraft:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        question = self.repo.get_lesson_question(run.id, run.active_question_slug)
        if question is None:
            self.last_feedback = ["No active question to transform."]
            return self.current_turn()
        replacement = self._generate_transformed_question(question=question, transform=transform, retry_generation=question.retry_generation)
        metadata_json = QuestionTransformService.transformed_metadata(question, transform=transform)
        question.prompt_json = replacement.prompt_json
        question.answer_json = replacement.answer_json
        question.question_type = replacement.question_type
        question.metadata_json = metadata_json
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        self._question_opened_at[question.question_slug] = datetime.utcnow()
        if transform == "drill":
            self.last_feedback = ["Fresh drill generated on the same concept."]
        elif transform == "harder":
            self.last_feedback = ["Difficulty raised one notch."]
        else:
            self.last_feedback = ["Difficulty lowered one notch."]
        return self.current_turn()

    def skill_diagnostics(self) -> list[LessonSkillStateRecord]:
        run = self.repo.get_lesson_run(self.session.id)
        assert run is not None
        states = self.repo.list_lesson_skill_states(run.id)
        if states:
            return states
        question_slugs = {question.skill_slug for question in self.repo.list_lesson_questions(run.id)}
        for skill_slug in sorted(question_slugs):
            self._update_skill_state(run.id, skill_slug)
        return self.repo.list_lesson_skill_states(run.id)

    def recall_candidates(self) -> list[str]:
        states = self.skill_diagnostics()
        return [f"Explain {state.skill_slug.replace('_', ' ')} from memory." for state in states[:5]]

    def next_drill(self) -> str:
        states = self.skill_diagnostics()
        fragile = next((state for state in states if state.overall_status != "strong"), None)
        if fragile is None:
            return f"Use {self.topic} on one fresh transfer example."
        return f"Run one new rep focused on {fragile.skill_slug.replace('_', ' ')}."

    def _evaluate_answer(self, *, question: LessonQuestionRecord, raw_answer: str) -> LessonEvaluationDraft:
        stripped = raw_answer.strip()
        if not stripped:
            return LessonEvaluationDraft(result="wrong", feedback="No answer yet.", hint=self._hint_for(question), confidence=0.0)

        local = self._evaluate_locally(question=question, raw_answer=stripped)
        if local is not None:
            return local

        if not self.runtime.health().available:
            return LessonEvaluationDraft(result="wrong", feedback="That answer does not match the expected target yet.", hint=self._hint_for(question), confidence=0.0)

        prompt = (
            "Evaluate this learner answer for a ProductiveBrain lesson question.\n"
            "Return only correct, close, or wrong.\n"
            "Do not reveal the exact answer in feedback or hints.\n"
            "For cloze, short_text, free_production, and error_correction, grade the idea rather than exact wording: "
            "accept spelling errors, missing keywords, or alternate phrasing when the inference is technically sound. "
            "Use close for substantially correct but incomplete answers, and wrong only when the core idea is absent.\n"
            f"Question prompt: {question.prompt_json.get('prompt', '')}\n"
            f"Question type: {question.question_type}\n"
            f"Choices: {question.prompt_json.get('choices', [])}\n"
            f"Expected data: {question.answer_json}\n"
            f"Learner answer: {stripped}\n"
        )
        policy_name = "complex_free_response_eval" if question.question_type in PRODUCTION_TYPES else "answer_check"
        try:
            evaluation = self.runtime.generate_draft(
                LessonEvaluationDraft,
                prompt,
                source_scope=f"lesson_eval:{question.question_slug}",
                model=_resolve_learning_model_binding(self.runtime, policy_name),
                max_output_tokens=4000,
            ).payload
            if not evaluation.hint.strip():
                evaluation.hint = self._hint_for(question)
            return evaluation
        except DraftGenerationError:
            return LessonEvaluationDraft(result="wrong", feedback="That answer is still off target.", hint=self._hint_for(question), confidence=0.0)

    def _evaluate_locally(self, *, question: LessonQuestionRecord, raw_answer: str) -> LessonEvaluationDraft | None:
        normalized = _normalize_text(raw_answer)
        accepted = [_normalize_text(item) for item in question.answer_json.get("accepted_answers", []) if str(item).strip()]
        correct_choices = [_normalize_text(item) for item in question.answer_json.get("correct_choices", []) if str(item).strip()]

        if question.question_type == "mcq":
            resolved = _resolve_choice_tokens(question, raw_answer)
            if len(resolved) == 1:
                normalized = _normalize_text(resolved[0])
            if normalized in correct_choices or normalized in accepted:
                return LessonEvaluationDraft(result="correct", feedback="Correct.", hint="", confidence=1.0)
            best = max((_answer_similarity(raw_answer, item) for item in question.answer_json.get("correct_choices", []) or []), default=0.0)
            if best >= 0.85:
                return LessonEvaluationDraft(result="close", feedback="Close, but not quite the intended option.", hint=self._hint_for(question), confidence=0.65)
            return LessonEvaluationDraft(result="wrong", feedback="That option misses the main discrimination.", hint=self._hint_for(question), confidence=0.9)

        if question.question_type == "multi_select":
            stats = _multi_select_stats(question, raw_answer)
            answer_set = {_normalize_text(item) for item in stats.selected_values}
            correct_set = set(correct_choices or accepted)
            if answer_set and answer_set == correct_set:
                return LessonEvaluationDraft(result="correct", feedback="Correct.", hint="", confidence=1.0)
            if stats.selected_correct:
                feedback = "Partly right, but the set is incomplete or includes a distractor."
                if stats.total_options >= 4 and stats.correct_selected_values:
                    feedback += "\nCorrect selections: " + "; ".join(stats.correct_selected_values)
                return LessonEvaluationDraft(result="close", feedback=feedback, hint=self._hint_for(question), confidence=0.7)
            return LessonEvaluationDraft(result="wrong", feedback="Those selections do not target the correct feedback set.", hint=self._hint_for(question), confidence=0.9)

        if question.question_type in {"cloze", "short_text", "error_correction"}:
            resolved = _resolve_choice_tokens(question, raw_answer)
            if len(resolved) == 1:
                normalized = _normalize_text(resolved[0])
            if normalized in accepted or normalized in correct_choices:
                return LessonEvaluationDraft(result="correct", feedback="Correct.", hint="", confidence=1.0)
            best = max((_answer_similarity(raw_answer, item) for item in question.answer_json.get("accepted_answers", []) or []), default=0.0)
            if best >= 0.86:
                return LessonEvaluationDraft(result="correct", feedback="Correct.", hint="", confidence=0.82)
            if best >= 0.62:
                return LessonEvaluationDraft(result="close", feedback="Close. Tighten the wording to match the target idea.", hint=self._hint_for(question), confidence=0.7)
            if self.runtime.health().available:
                return None
            return LessonEvaluationDraft(result="wrong", feedback="That answer still misses the target idea.", hint=self._hint_for(question), confidence=0.55)

        if question.question_type == "reorder":
            expected = list(question.answer_json.get("ordered_items", []) or [])
            display_items = list(question.prompt_json.get("display_items", []) or expected)
            if not expected:
                return None
            picked_indexes = [int(item) for item in re.findall(r"\d+", raw_answer)]
            if picked_indexes and len(picked_indexes) == len(display_items):
                ordered = [display_items[index - 1] for index in picked_indexes if 0 < index <= len(display_items)]
                if ordered == expected:
                    return LessonEvaluationDraft(result="correct", feedback="Correct order.", hint="", confidence=1.0)
                if sum(1 for left, right in zip(ordered, expected) if left == right) >= max(1, len(expected) - 1):
                    return LessonEvaluationDraft(result="close", feedback="The order is nearly right. One move is misplaced.", hint=self._hint_for(question), confidence=0.72)
                return LessonEvaluationDraft(result="wrong", feedback="That order changes the intended progression.", hint=self._hint_for(question), confidence=0.9)
            return LessonEvaluationDraft(result="wrong", feedback="Enter the order as space-separated numbers.", hint=self._hint_for(question), confidence=0.6)

        if question.question_type == "free_production":
            if accepted:
                if any(item in normalized for item in accepted):
                    return LessonEvaluationDraft(result="correct", feedback="Correct.", hint="", confidence=0.82)
                best = max((_answer_similarity(raw_answer, item) for item in question.answer_json.get("accepted_answers", []) or []), default=0.0)
                if best >= 0.62:
                    return LessonEvaluationDraft(result="close", feedback="Close. Make the underlying concept more explicit.", hint=self._hint_for(question), confidence=0.65)
            return None

        return None

    def _points_delta(
        self,
        *,
        question: LessonQuestionRecord,
        result: str,
        response_ms: int,
        attempts: list[LessonAttemptRecord],
        raw_answer: str = "",
    ) -> float:
        if question.question_type == "multi_select":
            return self._cap_question_negative_points(
                question,
                attempts,
                self._multi_select_points_delta(
                    question=question,
                    result=result,
                    response_ms=response_ms,
                    attempts=attempts,
                    raw_answer=raw_answer,
                ),
            )

        if result == "close":
            delta = 0 if question.question_type in {"short_text", "free_production", "error_correction"} else -1
            return self._cap_question_negative_points(question, attempts, delta)
        if result == "wrong":
            delta = -1 if question.question_type in {"short_text", "free_production", "error_correction"} else -2
            return self._cap_question_negative_points(question, attempts, delta)
        if result == "revealed":
            return self._cap_question_negative_points(question, attempts, -3)
        if result == "skipped":
            return 0
        if result != "correct":
            return 0

        if attempts or question.revealed:
            return self._apply_hint_point_ceiling(
                question,
                1 if not question.revealed else 0,
                max_points=3,
            )

        fast_seconds, normal_seconds = QUESTION_TIME_THRESHOLDS.get(question.question_type, (30, 60))
        seconds = response_ms / 1000.0
        if seconds <= fast_seconds:
            return self._apply_hint_point_ceiling(question, 3, max_points=3)
        if seconds <= normal_seconds:
            return self._apply_hint_point_ceiling(question, 2, max_points=3)
        return self._apply_hint_point_ceiling(question, 1, max_points=3)

    def _multi_select_points_delta(
        self,
        *,
        question: LessonQuestionRecord,
        result: str,
        response_ms: int,
        attempts: list[LessonAttemptRecord],
        raw_answer: str,
    ) -> float:
        stats = _multi_select_stats(question, raw_answer)
        if result == "revealed":
            return -3
        if result == "skipped":
            return 0
        if result == "correct":
            max_points = self._multi_select_correct_max_points(stats)
            if attempts or question.revealed:
                return self._apply_hint_point_ceiling(
                    question,
                    1 if not question.revealed else 0,
                    max_points=max_points,
                )
            return self._apply_hint_point_ceiling(question, max_points, max_points=max_points)
        if stats.selected_correct <= 0:
            return -3
        raw_score = (
            stats.selected_correct
            - (2 * stats.wrong_selections)
            - (1.5 * stats.omissions)
        ) / max(1, stats.total_options)
        if raw_score > 0:
            return math.ceil(raw_score)
        if raw_score < 0:
            return math.floor(raw_score)
        return 0

    @staticmethod
    def _multi_select_correct_max_points(stats: MultiSelectStats) -> int:
        total_correct = max(0, stats.total_correct)
        if total_correct >= stats.total_options / 2:
            param = max(0, stats.total_options - total_correct)
        else:
            param = total_correct
        return 1 + math.ceil(param / 2)

    @staticmethod
    def _apply_hint_point_ceiling(
        question: LessonQuestionRecord,
        proposed_delta: float,
        *,
        max_points: float,
    ) -> float:
        if proposed_delta <= 0:
            return proposed_delta
        hint_count = max(0, int(question.hint_level or 0))
        ceiling = max(0.0, float(max_points) - (POINTS_PER_HINT * hint_count))
        return min(float(proposed_delta), ceiling)

    @staticmethod
    def _negative_floor_for(question: LessonQuestionRecord) -> int:
        if question.question_type == "multi_select":
            return -5
        if question.question_type in {"short_text", "free_production", "error_correction"}:
            return -3
        return -4

    def _cap_question_negative_points(
        self,
        question: LessonQuestionRecord,
        attempts: list[LessonAttemptRecord],
        proposed_delta: float,
    ) -> float:
        if proposed_delta >= 0:
            return proposed_delta
        floor = self._negative_floor_for(question)
        previous_total = sum(float(attempt.points_delta or 0) for attempt in attempts)
        if previous_total <= floor:
            return 0
        return max(proposed_delta, floor - previous_total)

    @staticmethod
    def _consecutive_misses(attempts: list[LessonAttemptRecord]) -> int:
        count = 0
        for attempt in reversed(attempts):
            if attempt.result not in {"wrong", "close"}:
                break
            count += 1
        return count

    def _forced_reveal_feedback(self, question: LessonQuestionRecord) -> list[str]:
        return [
            f"Answer: {self._reveal_answer(question)}",
            "Use /skip when you're ready to move on. This will return as an easier retry.",
        ]

    def _reveal_after_exhausted_hints(
        self,
        *,
        run: LessonRunRecord,
        question: LessonQuestionRecord,
    ) -> LearningPartnerTurnDraft:
        if question.revealed:
            self.last_feedback = self._forced_reveal_feedback(question)
            return self.current_turn()
        attempts = self.repo.list_lesson_attempts(run.id, question.question_slug)
        opened_at = self._question_opened_at.get(question.question_slug, datetime.utcnow())
        response_ms = max(0, int((datetime.utcnow() - opened_at).total_seconds() * 1000))
        points_delta = self._points_delta(
            question=question,
            result="wrong",
            response_ms=response_ms,
            attempts=attempts,
            raw_answer="/hint",
        )
        self.repo.create_lesson_attempt(
            LessonAttemptRecord(
                id=_short_lesson_slug(f"attempt {len(attempts) + 1} {question.question_slug}", fallback="attempt"),
                lesson_run_id=run.id,
                session_id=self.session.id,
                page_slug=question.page_slug,
                question_slug=question.question_slug,
                skill_slug=question.skill_slug,
                answer_text="/hint",
                result="revealed",
                response_ms=response_ms,
                hint_level=question.hint_level,
                points_delta=points_delta,
                error_tags=list(question.answer_json.get("error_tags", [])),
            )
        )
        run.total_points += points_delta
        if not question.queued_retry:
            self._enqueue_retry_question(run=run, question=question)
            question.queued_retry = True
        self._retune_pending_retry_for_easier_verification(run=run, question=question)
        question.revealed = True
        question.mastered = False
        question.status = "revealed"
        question.next_review_at = self._review_signal(question=question, result="revealed")
        question.updated_at = _iso_now()
        self.repo.update_lesson_question(question)
        run.updated_at = _iso_now()
        self.repo.update_lesson_run(run)
        self._sync_session_pointer(run)
        self.last_feedback = self._forced_reveal_feedback(question)
        self._update_skill_state(run.id, question.skill_slug)
        return self.current_turn()

    def _hint_for(self, question: LessonQuestionRecord) -> str:
        hints = _normalize_lesson_hints(
            [str(item).strip() for item in question.answer_json.get("hints", []) if str(item).strip()],
            prompt=str(question.prompt_json.get("prompt", "") or ""),
            skill_label=str(question.answer_json.get("skill_label", "") or question.skill_slug),
        )
        hint_index = min(max(question.hint_level, 0), len(hints) - 1)
        return hints[hint_index]

    def _reveal_answer(self, question: LessonQuestionRecord) -> str:
        reveal = str(question.answer_json.get("reveal_answer", "") or "").strip()
        if reveal:
            return reveal
        accepted = [str(item).strip() for item in question.answer_json.get("accepted_answers", []) if str(item).strip()]
        if accepted:
            return " | ".join(accepted) if question.question_type == "multi_select" else accepted[0]
        choices = [str(item).strip() for item in question.answer_json.get("correct_choices", []) if str(item).strip()]
        if choices:
            return " | ".join(choices) if question.question_type == "multi_select" else choices[0]
        return "No stored answer."

    def _advance_after_resolution(self, *, run: LessonRunRecord, question: LessonQuestionRecord) -> None:
        self._question_opened_at.pop(question.question_slug, None)
        self._update_page_status(run.id, question.page_slug)
        next_question = self._next_active_question(run.id)
        if next_question is None:
            run.lesson_status = "ready_to_finish"
            run.ready_to_finish = True
            run.active_page_slug = question.page_slug
            run.active_question_slug = ""
            run.updated_at = _iso_now()
            self.repo.update_lesson_run(run)
            self._sync_session_pointer(run)
            return

        run.active_page_slug = next_question.page_slug
        run.active_question_slug = next_question.question_slug
        page = self.repo.get_lesson_page(run.id, next_question.page_slug)
        run.active_page_index = page.sequence_index if page is not None else run.active_page_index
        run.active_question_index = next_question.sequence_index
        run.lesson_status = "retry" if next_question.retry_of_question_slug else "active"
        run.ready_to_finish = False
        run.updated_at = _iso_now()
        self.repo.update_lesson_run(run)
        self._touch_question_clock(run)
        self._sync_session_pointer(run)

    def _next_active_question(self, lesson_run_id: str) -> LessonQuestionRecord | None:
        questions = self.repo.list_lesson_questions(lesson_run_id)
        normal = [
            question
            for question in questions
            if not question.retry_of_question_slug and question.status not in {"correct", "revealed", "skipped"}
        ]
        if normal:
            normal.sort(key=lambda item: (self._page_index(lesson_run_id, item.page_slug), item.sequence_index))
            return normal[0]
        retries = [
            question
            for question in questions
            if question.retry_of_question_slug and question.status not in {"correct", "revealed", "skipped"}
        ]
        if retries:
            retries.sort(key=lambda item: (self._page_index(lesson_run_id, item.page_slug), item.sequence_index))
            return retries[0]
        return None

    def _page_index(self, lesson_run_id: str, page_slug: str) -> int:
        page = self.repo.get_lesson_page(lesson_run_id, page_slug)
        return page.sequence_index if page is not None else 0

    def _update_page_status(self, lesson_run_id: str, page_slug: str) -> None:
        page = self.repo.get_lesson_page(lesson_run_id, page_slug)
        if page is None:
            return
        questions = self.repo.list_lesson_questions(lesson_run_id, page_slug)
        unresolved = [question for question in questions if question.status not in {"correct", "revealed", "skipped"}]
        page.status = "complete" if not unresolved else "active"
        page.updated_at = _iso_now()
        self.repo.update_lesson_page(page)

    def _page_progress_label(self, page_slug: str) -> str:
        run = self.repo.get_lesson_run(self.session.id)
        if run is None or not page_slug:
            return ""
        questions = self.repo.list_lesson_questions(run.id, page_slug)
        if not questions:
            return ""
        cleared = sum(1 for question in questions if question.status in {"correct", "revealed", "skipped"})
        return f"{cleared}/{len(questions)} cleared"

    def _enqueue_retry_question(
        self,
        *,
        run: LessonRunRecord,
        question: LessonQuestionRecord,
        transform: str = "retry",
        retry_strategy: str = "",
        verification_for_retry: str = "",
    ) -> LessonQuestionRecord:
        retry_page = self.repo.get_lesson_page(run.id, "mistakes")
        if retry_page is None:
            retry_page = LessonPageRecord(
                id=f"{run.id}:mistakes",
                lesson_run_id=run.id,
                session_id=self.session.id,
                page_slug="mistakes",
                title="Mistake Loop",
                intro_text="Transformed retries for questions that were wrong, close, revealed, or skipped.",
                sequence_index=len(self.repo.list_lesson_pages(run.id)),
                status="pending",
            )
            self.repo.create_lesson_page(retry_page)
        retry_source = self._generate_transformed_question(question=question, transform=transform, retry_generation=question.retry_generation + 1)
        existing_retry_questions = self.repo.list_lesson_questions(run.id, "mistakes")
        existing_question_slugs = {item.question_slug for item in self.repo.list_lesson_questions(run.id)}
        retry_slug = _short_lesson_slug(f"{question.question_slug} retry", fallback="retry", existing=existing_question_slugs)
        metadata = QuestionTransformService.retry_metadata(question)
        metadata["last_transform"] = transform
        metadata["difficulty"] = transform if transform in {"retry", "easier", "harder", "drill"} else "retry"
        if retry_strategy:
            metadata["retry_strategy"] = retry_strategy
        if verification_for_retry:
            metadata["verification_for_retry"] = verification_for_retry
        retry_question = LessonQuestionRecord(
            id=f"{run.id}:{retry_slug}",
            lesson_run_id=run.id,
            session_id=self.session.id,
            page_slug="mistakes",
            question_slug=retry_slug,
            skill_slug=question.skill_slug,
            question_type=retry_source.question_type,
            prompt_json=retry_source.prompt_json,
            answer_json=retry_source.answer_json,
            metadata_json=metadata,
            sequence_index=len(existing_retry_questions),
            status="pending",
            retry_of_question_slug=question.question_slug,
            retry_generation=question.retry_generation + 1,
        )
        self.repo.create_lesson_question(retry_question)
        retry_page.question_count += 1
        retry_page.updated_at = _iso_now()
        self.repo.update_lesson_page(retry_page)
        run.retry_queue = list(run.retry_queue) + [retry_slug]
        run.updated_at = _iso_now()
        self.repo.update_lesson_run(run)
        return retry_question

    def _retune_pending_retry_for_easier_verification(
        self,
        *,
        run: LessonRunRecord,
        question: LessonQuestionRecord,
    ) -> None:
        retries = [
            item
            for item in self.repo.list_lesson_questions(run.id, "mistakes")
            if item.retry_of_question_slug == question.question_slug and item.status == "pending"
        ]
        if not retries:
            self._enqueue_retry_question(
                run=run,
                question=question,
                transform="easier",
                retry_strategy="easier_then_verify",
            )
            question.queued_retry = True
            return
        retry = retries[0]
        replacement = self._generate_transformed_question(
            question=question,
            transform="easier",
            retry_generation=max(retry.retry_generation, question.retry_generation + 1),
        )
        retry.question_type = replacement.question_type
        retry.prompt_json = replacement.prompt_json
        retry.answer_json = replacement.answer_json
        metadata = QuestionTransformService.hydrated_metadata(retry)
        metadata["last_transform"] = "easier"
        metadata["difficulty"] = "easier"
        metadata["retry_strategy"] = "easier_then_verify"
        retry.metadata_json = metadata
        retry.updated_at = _iso_now()
        self.repo.update_lesson_question(retry)

    def _enqueue_verification_question(
        self,
        *,
        run: LessonRunRecord,
        retry_question: LessonQuestionRecord,
    ) -> None:
        original_slug = retry_question.retry_of_question_slug
        if not original_slug:
            return
        existing = [
            item
            for item in self.repo.list_lesson_questions(run.id, "mistakes")
            if item.metadata_json.get("verification_for_retry") == retry_question.question_slug
        ]
        if existing:
            return
        original = self.repo.get_lesson_question(run.id, original_slug) or retry_question
        self._enqueue_retry_question(
            run=run,
            question=original,
            transform="retry",
            retry_strategy="verify_understanding",
            verification_for_retry=retry_question.question_slug,
        )

    def _generate_transformed_question(
        self,
        *,
        question: LessonQuestionRecord,
        transform: str,
        retry_generation: int,
    ) -> LessonQuestionRecord:
        if self.runtime.health().available:
            prompt = (
                "Create a transformed ProductiveBrain lesson question on the same underlying concept.\n"
                "Keep it close to the original, but not identical.\n"
                "Do not merely swap names or numbers.\n"
                "For retry transforms, preserve high fidelity to the original skill while changing the discrimination path.\n"
                f"Transform kind: {transform}\n"
                f"Original question type: {question.question_type}\n"
                f"Original prompt data: {question.prompt_json}\n"
                f"Original answer data: {question.answer_json}\n"
            )
            policy_name = "drill_generation" if transform == "drill" else "small_retry"
            try:
                generated = self.runtime.generate_draft(
                    LessonQuestionDraft,
                    prompt,
                    source_scope=f"lesson_transform:{question.question_slug}:{transform}",
                    model=_resolve_learning_model_binding(self.runtime, policy_name),
                    max_output_tokens=4000,
                ).payload
                _limit_lesson_question_choices(generated)
                return LessonQuestionRecord(
                    id=question.id,
                    lesson_run_id=question.lesson_run_id,
                    session_id=question.session_id,
                    page_slug=question.page_slug,
                    question_slug=question.question_slug,
                    skill_slug=question.skill_slug,
                    question_type=generated.question_type,
                    prompt_json={
                        "title": generated.title,
                        "prompt": generated.prompt,
                        "choices": list(generated.choices),
                        "display_items": list(generated.ordered_items),
                    },
                    answer_json={
                        "accepted_answers": list(generated.accepted_answers),
                        "correct_choices": list(generated.correct_choices),
                        "ordered_items": list(generated.ordered_items),
                        "hints": list(generated.hints),
                        "reveal_answer": generated.reveal_answer,
                        "error_tags": list(generated.error_tags),
                        "skill_label": generated.skill_label,
                        "page_intro": generated.page_intro,
                        "evaluator_notes": generated.evaluator_notes,
                    },
                    metadata_json=QuestionTransformService.hydrated_metadata(question),
                    sequence_index=question.sequence_index,
                    status="pending",
                    retry_of_question_slug=question.retry_of_question_slug,
                    retry_generation=retry_generation,
                )
            except DraftGenerationError:
                pass

        prompt_json = dict(question.prompt_json)
        answer_json = dict(question.answer_json)
        prompt = str(prompt_json.get("prompt", "") or "")
        if transform == "harder":
            prompt_json["prompt"] = f"Harder variant: {prompt}"
        elif transform == "easier":
            prompt_json["prompt"] = f"Easier variant: {prompt}"
        elif transform == "drill":
            prompt_json["prompt"] = f"Drill variant: {prompt}"
        else:
            prompt_json["prompt"] = f"Retry variant: {prompt}"

        if question.question_type in {"mcq", "multi_select", "cloze"}:
            base_choices = [str(item).strip() for item in prompt_json.get("choices", []) if str(item).strip()]
            transformed_choices: list[str] = []
            for index, choice in enumerate(base_choices):
                if transform == "retry":
                    if index == 0:
                        transformed_choices.append(choice)
                    else:
                        transformed_choices.append(f"{choice} via {question.skill_slug.replace('_', ' ')}")
                elif transform == "harder":
                    transformed_choices.append(f"{choice} under constraint")
                elif transform == "easier":
                    transformed_choices.append(choice.replace(" via ", " ").replace(" under constraint", ""))
                else:
                    transformed_choices.append(f"drill: {choice}")
            if transformed_choices and transformed_choices != base_choices:
                prompt_json["choices"] = transformed_choices
                if question.question_type == "cloze":
                    answer_json["accepted_answers"] = [transformed_choices[0]]
                    answer_json["correct_choices"] = [transformed_choices[0]]
                    answer_json["reveal_answer"] = transformed_choices[0]
                elif question.question_type == "mcq":
                    answer_json["correct_choices"] = [transformed_choices[0]]
                    answer_json["accepted_answers"] = [transformed_choices[0]]
                elif question.question_type == "multi_select":
                    correct_choices = list(answer_json.get("correct_choices", []) or [])
                    if correct_choices:
                        answer_json["correct_choices"] = [transformed_choices[base_choices.index(item)] if item in base_choices else item for item in correct_choices]
                        answer_json["accepted_answers"] = list(answer_json["correct_choices"])

        if question.question_type == "reorder":
            ordered_items = list(answer_json.get("ordered_items", []) or [])
            if ordered_items and len(ordered_items) > 1:
                prompt_json["display_items"] = ordered_items[1:] + ordered_items[:1]

        return LessonQuestionRecord(
            id=question.id,
            lesson_run_id=question.lesson_run_id,
            session_id=question.session_id,
            page_slug=question.page_slug,
            question_slug=question.question_slug,
            skill_slug=question.skill_slug,
            question_type=question.question_type,
            prompt_json=prompt_json,
            answer_json=answer_json,
            metadata_json=QuestionTransformService.hydrated_metadata(question),
            sequence_index=question.sequence_index,
            status="pending",
            retry_of_question_slug=question.retry_of_question_slug,
            retry_generation=retry_generation,
        )

    def _generate_explanation(
        self,
        *,
        question: LessonQuestionRecord,
        page: LessonPageRecord | None,
        intuitive: bool,
    ) -> str:
        if not self.runtime.health().available:
            style = "intuitive" if intuitive else "plain"
            return f"{style.title()} explanation: focus on {question.skill_slug.replace('_', ' ')} and connect it back to the current page before answering."

        prompt = (
            "Explain the current lesson concept without revealing the exact answer.\n"
            f"Style: {'intuitive, low-jargon' if intuitive else 'clear and direct'}\n"
            f"Page title: {page.title if page is not None else ''}\n"
            f"Page intro: {page.intro_text if page is not None else ''}\n"
            f"Question prompt: {question.prompt_json.get('prompt', '')}\n"
            f"Question data: {question.answer_json}\n"
        )
        policy_name = "lesson_hint_intuitive" if intuitive else "session_explain"
        try:
            evaluation = self.runtime.generate_draft(
                LessonEvaluationDraft,
                prompt,
                source_scope=f"lesson_explain:{question.question_slug}:{policy_name}",
                model=_resolve_learning_model_binding(self.runtime, policy_name),
                max_output_tokens=4000,
            ).payload
            text = evaluation.feedback.strip() or evaluation.hint.strip()
            return text or f"Focus on {question.skill_slug.replace('_', ' ')} before answering."
        except DraftGenerationError:
            return f"Focus on {question.skill_slug.replace('_', ' ')} before answering."

    def _review_signal(self, *, question: LessonQuestionRecord, result: str) -> str:
        now = datetime.utcnow()
        if result == "revealed":
            return (now + timedelta(minutes=30)).isoformat()
        if result == "skipped":
            return (now + timedelta(hours=2)).isoformat()
        if result == "correct":
            if question.hint_level > 0:
                return (now + timedelta(hours=18)).isoformat()
            if question.question_type in PRODUCTION_TYPES:
                return (now + timedelta(days=4)).isoformat()
            return (now + timedelta(days=2)).isoformat()
        if result in {"close", "wrong"}:
            return (now + timedelta(hours=12)).isoformat()
        return (now + timedelta(days=1)).isoformat()

    def _update_skill_state(self, lesson_run_id: str, skill_slug: str) -> None:
        questions = [question for question in self.repo.list_lesson_questions(lesson_run_id) if question.skill_slug == skill_slug]
        attempts = [attempt for attempt in self.repo.list_lesson_attempts(lesson_run_id) if attempt.skill_slug == skill_slug]
        if not questions:
            return

        recognition_attempts = []
        production_attempts = []
        for attempt in attempts:
            question = self.repo.get_lesson_question(lesson_run_id, attempt.question_slug)
            if question is None:
                continue
            if question.question_type in RECOGNITION_TYPES:
                recognition_attempts.append(attempt)
            if question.question_type in PRODUCTION_TYPES:
                production_attempts.append(attempt)
        recognition_status = self._classify_attempts(recognition_attempts)
        production_status = self._classify_attempts(production_attempts)
        recognition_questions = [question for question in questions if question.question_type in RECOGNITION_TYPES]
        production_questions = [question for question in questions if question.question_type in PRODUCTION_TYPES]
        if recognition_status == "fragile" and not recognition_attempts:
            if any(question.revealed or question.status in {"revealed", "skipped"} for question in recognition_questions):
                recognition_status = "needs_repair"
        if production_status == "fragile" and not production_attempts:
            if any(question.revealed or question.status in {"revealed", "skipped"} for question in production_questions):
                production_status = "needs_repair"
        overall_status = "strong"
        if "needs_repair" in {recognition_status, production_status}:
            overall_status = "needs_repair"
        elif "fragile" in {recognition_status, production_status}:
            overall_status = "fragile"
        error_tags: list[str] = []
        next_review = ""
        for question in questions:
            for tag in question.answer_json.get("error_tags", []) or []:
                clean = str(tag).strip()
                if clean and clean not in error_tags:
                    error_tags.append(clean)
            if question.next_review_at and (not next_review or question.next_review_at < next_review):
                next_review = question.next_review_at

        state = LessonSkillStateRecord(
            id=f"{lesson_run_id}:{skill_slug}",
            lesson_run_id=lesson_run_id,
            session_id=self.session.id,
            skill_slug=skill_slug,
            recognition_status=recognition_status,
            production_status=production_status,
            overall_status=overall_status,
            error_tags=error_tags,
            attempt_count=len(attempts),
            next_review_at=next_review,
            updated_at=_iso_now(),
        )
        existing = self.repo.get_lesson_skill_state(lesson_run_id, skill_slug)
        if existing is None:
            self.repo.create_lesson_skill_state(state)
        else:
            self.repo.update_lesson_skill_state(state)

    @staticmethod
    def _classify_attempts(attempts: list[LessonAttemptRecord]) -> str:
        if not attempts:
            return "fragile"
        latest = attempts[-1]
        if latest.result == "correct" and all(attempt.result == "correct" for attempt in attempts[-2:]):
            return "strong"
        if latest.result == "correct":
            return "fragile"
        return "needs_repair"
