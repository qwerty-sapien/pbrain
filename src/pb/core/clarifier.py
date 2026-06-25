# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Contextual clarifying-question generation for learning flows."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from rich.markup import escape

from pb.cli.console import get_console
from pb.cli.helpers import prompt_text
from pb.cli.pickers import pick_many_choices, pick_single_choice
from pb.core.feedback_profile import load_feedback_guidance
from pb.core.learner_memory import build_global_learner_profile
from pb.core.learning_curriculum import needs_curriculum_clarification
from pb.core.learning_prompting import language_instruction, learning_intent_style_guidance
from pb.core.product_control import ControlState
from pb.core.prerequisites import build_prerequisite_chain
from pb.core.scope_resolution import matching_goals
from pb.core.staging import build_learning_context
from pb.llm.drafts import ClarifierQuestionDraft, ClarifierQuestionSetDraft
from pb.llm.runtime import DraftGenerationError, LLMRuntime
from pb.storage.config import get_config

MAX_CONTEXT_OPTIONS = 6
RESERVED_CONTEXT_OPTIONS = 2
DISCUSS_SENTINEL = "__discuss__"
_EXPLICIT_NEGATIONS = {
    "n/a",
    "na",
    "neither",
    "none",
    "nothing",
}

def _normalized_answer_text(text: str) -> str:
    return " ".join((text or "").strip().lower().split())


def is_explicit_negation(text: str) -> bool:
    """Return True when the text clearly means none of the presented options."""

    normalized = _normalized_answer_text(text)
    return normalized in _EXPLICIT_NEGATIONS


def trim_context_option_candidates(
    option_candidates: list[str],
    *,
    max_total: int = MAX_CONTEXT_OPTIONS,
    reserve_slots: int = RESERVED_CONTEXT_OPTIONS,
) -> list[str]:
    """Trim visible context-shaping options so discuss + inline always fit."""

    max_options = max(0, max_total - reserve_slots)
    trimmed: list[str] = []
    for option in option_candidates:
        clean = option.strip()
        if clean and clean not in trimmed:
            trimmed.append(clean)
        if len(trimmed) >= max_options:
            break
    return trimmed


def _recent_sessions(repo, *, limit: int = 4) -> list[dict[str, str]]:
    rows: list[tuple[Any, dict[str, str]]] = []
    for task in repo.list_tasks():
        for session in repo.list_sessions_for_task(task.id):
            rows.append(
                (
                    session.start_at,
                    {
                        "branch": getattr(session, "branch", "study"),
                        "scope": getattr(session, "subject_scope", "") or task.title,
                        "outcome": getattr(session, "actual_outcome", "") or "",
                    },
                )
            )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in rows[:limit]]


def _vault_snippets(vault_path: Path, raw_request: str, *, limit: int = 3) -> list[str]:
    if not vault_path.exists():
        return []
    tokens = [token.lower() for token in raw_request.split() if len(token) > 2][:4]
    if not tokens:
        return []
    matches: list[str] = []
    for note_path in sorted((vault_path / "knowledge").rglob("*.md")) if (vault_path / "knowledge").exists() else []:
        try:
            body = note_path.read_text(encoding="utf-8")
        except Exception:
            continue
        lowered = body.lower()
        if not any(token in lowered for token in tokens):
            continue
        excerpt = " ".join(body.split())[:180]
        matches.append(f"{note_path.name}: {excerpt}")
        if len(matches) >= limit:
            break
    return matches


def _recent_clarifier_answers(repo, *, limit: int = 6) -> list[dict[str, Any]]:
    rows: list[tuple[Any, dict[str, Any]]] = []
    for task in repo.list_tasks():
        for session in repo.list_sessions_for_task(task.id):
            generated = getattr(session, "generated_names", {}) or {}
            answers = generated.get("clarifier_answers")
            records = generated.get("clarifier_answer_records")
            if not isinstance(answers, dict) and not isinstance(records, list):
                continue
            rows.append(
                (
                    session.start_at,
                    {
                        "branch": getattr(session, "branch", "study"),
                        "scope": getattr(session, "subject_scope", "") or task.title,
                        "answers": answers if isinstance(answers, dict) else {},
                        "records": records if isinstance(records, list) else [],
                    },
                )
            )
    rows.sort(key=lambda item: item[0], reverse=True)
    return [payload for _, payload in rows[:limit]]


@dataclass
class ClarifierAnswerRecord:
    question: str
    answer: str
    answer_kind: str = "short_text"
    selected_option: str = ""
    options_presented: list[str] = field(default_factory=list)
    excluded_options: list[str] = field(default_factory=list)
    inferred_signal_type: str = ""


@dataclass
class ClarifierAnswerBundle:
    answers: dict[str, str] = field(default_factory=dict)
    records: list[ClarifierAnswerRecord] = field(default_factory=list)


def persist_clarifier_answers(entity, bundle: ClarifierAnswerBundle) -> None:
    """Persist answered clarifiers onto a task/session-like entity."""

    if entity is None or (not bundle.answers and not bundle.records):
        return
    generated = dict(getattr(entity, "generated_names", {}) or {})
    generated["clarifier_answers"] = dict(bundle.answers)
    generated["clarifier_answer_records"] = [asdict(record) for record in bundle.records]
    entity.generated_names = generated


def _echo_clarifier_answer(record: ClarifierAnswerRecord) -> None:
    if not record.answer.strip():
        return
    color = "blue" if record.answer_kind in {"selected_option", "multi_select", "explicit_exclusion"} else "green"
    get_console().print(f"[dim]?[/] {escape(record.question)} [{color}]{escape(record.answer)}[/]")


def clarifier_prompt_block(bundle: ClarifierAnswerBundle | None) -> str:
    """Serialize clarifier answers with enough structure to bind later drafting."""

    if bundle is None or not bundle.records:
        return ""
    payload = [
        {
            "question": record.question,
            "answer": record.answer,
            "answer_kind": record.answer_kind,
            "selected_option": record.selected_option,
            "options_presented": list(record.options_presented),
            "excluded_options": list(record.excluded_options),
            "inferred_signal_type": record.inferred_signal_type,
        }
        for record in bundle.records
        if record.answer.strip() or record.excluded_options
    ]
    if not payload:
        return ""
    return (
        "Structured clarifier answers JSON:\n"
        + json.dumps(payload, ensure_ascii=True)
        + "\nTreat any `excluded_options` as explicitly out of scope unless the user later reintroduces them.\n"
    )


def _prerequisite_option_candidates(
    focus: str,
    context: dict[str, Any],
    control_state: ControlState | None,
    *,
    runtime: LLMRuntime | None,
) -> list[str]:
    domain = str(context.get("domain") or focus).strip() or focus
    try:
        chain = build_prerequisite_chain(
            domain,
            focus,
            getattr(control_state, "knowns", []) or [],
            getattr(control_state, "unknowns", []) or [],
            runtime=runtime,
        )
    except Exception:
        return []
    if not chain:
        return []
    picks: list[str] = []
    for index in (0, 1, 2, 4, len(chain) - 1):
        if 0 <= index < len(chain):
            candidate = chain[index].strip()
            if candidate and candidate not in picks:
                picks.append(candidate)
    return picks[:5]


def build_clarifier_context(
    repo,
    runtime_ctx,
    *,
    raw_request: str,
    scope: str,
    mode: str = "",
    domain: str = "",
    learner_level: str = "",
    control_state: ControlState | None = None,
) -> dict[str, Any]:
    """Build the compact context packet used by the clarifier."""
    vault_path = getattr(runtime_ctx, "vault_path", Path("."))
    return {
        "raw_request": raw_request,
        "scope": scope,
        "mode": mode,
        "domain": domain,
        "learner_level": learner_level,
        "local_context": build_learning_context(repo, runtime_ctx),
        "matching_goals": matching_goals(repo, raw_request),
        "recent_sessions": _recent_sessions(repo),
        "feedback_guidance": load_feedback_guidance(vault_path, scope),
        "vault_snippets": _vault_snippets(vault_path, raw_request),
        "learner_profile": build_global_learner_profile(repo, runtime_ctx),
        "recent_clarifier_answers": _recent_clarifier_answers(repo),
        "prior_control_state": control_state.model_dump(mode="json") if control_state is not None else {},
    }


class ClarifierService:
    """Generate a tiny, contextual clarifier question set."""

    def __init__(self, runtime: LLMRuntime):
        self.runtime = runtime

    def _role_binding(self) -> str:
        roles = self.runtime.config.model_roles
        return (
            roles.fast_inference
            or roles.fast
            or roles.default
        )

    def generate_questions(
        self,
        intent: str,
        context: dict[str, Any],
        *,
        max_questions: int = 3,
        scope: str = "learn",
        control_state: ControlState | None = None,
        prior_question_signatures: list[str] | None = None,
    ) -> list[ClarifierQuestionDraft]:
        """Return a small contextual question batch, or one fallback question."""
        used_signatures = [item.strip().lower() for item in prior_question_signatures or [] if item.strip()]
        _lang_cfg = get_config().ui.language
        prompt = (
            language_instruction(intent, configured=_lang_cfg)
            + "You are generating a tiny clarifier batch for a learning-first CLI.\n"
            "Ask only the questions that unlock the next executable study or practice step.\n"
            "Rules:\n"
            "- Ask 0 to 3 questions.\n"
            "- Questions must be domain-specific and grounded in the provided context.\n"
            "- Do not ask generic coaching filler like 'what feels hardest right now'.\n"
            "- Include option_candidates when a tight menu would reduce ambiguity.\n"
            "- When you include option_candidates, keep them concrete, low-jargon, and cap them at 4 because the UI reserves slots for discuss + inline input.\n"
            "- For speech, accent, pronunciation, fluency, or naturalness requests, default to performance coaching or imitation framing, not academic analysis.\n"
            "- Do not suggest sociolinguistics, unrelated dialect comparisons, or academic background unless the user explicitly asks for them.\n"
            "- Fill why_this_matters, inferred_signal_type, downstream_effect, and confidence for each question.\n"
            "- Avoid repeating the same question family if prior control state or prior questions already covered it.\n"
            "- If the intent is already specific enough, return an empty list.\n"
            + learning_intent_style_guidance()
            + f"Workflow scope: {scope}\n"
            + f"Intent: {intent}\n"
            + f"Prior question signatures: {used_signatures}\n"
            + f"Prior control state: {(control_state.model_dump(mode='json') if control_state is not None else {})}\n"
            + f"Context JSON: {context}\n"
        )
        try:
            draft = self.runtime.generate_draft(
                ClarifierQuestionSetDraft,
                prompt,
                source_scope=f"clarifier:{scope}:{intent[:80]}",
                model=self._role_binding(),
                max_output_tokens=4000,
            ).payload
        except Exception:
            return self._fallback_questions(
                intent,
                context,
                control_state=control_state,
                prior_question_signatures=used_signatures,
            )
        filtered: list[ClarifierQuestionDraft] = []
        for question in draft.questions:
            signature = self._question_signature(question)
            if signature in used_signatures:
                continue
            filtered.append(question)
            used_signatures.append(signature)
            if len(filtered) >= max_questions:
                break
        if filtered:
            return filtered
        return self._fallback_questions(
            intent,
            context,
            control_state=control_state,
            prior_question_signatures=used_signatures,
        )

    def _question_signature(self, question: ClarifierQuestionDraft) -> str:
        signal = (question.inferred_signal_type or "").strip().lower()
        stem = " ".join(question.question.lower().split())
        return f"{signal}:{stem}"

    def _fallback_questions(
        self,
        intent: str,
        context: dict[str, Any],
        *,
        control_state: ControlState | None = None,
        prior_question_signatures: list[str] | None = None,
    ) -> list[ClarifierQuestionDraft]:
        focus = " ".join((intent or "").split()) or "this topic"
        mode = context.get("mode") or ""
        broad = needs_curriculum_clarification(focus, preferred_branch=mode)
        signal_counts = dict(getattr(control_state, "signal_counts", {}) or {})
        previous = set(prior_question_signatures or [])
        prerequisite_candidates = _prerequisite_option_candidates(
            focus,
            context,
            control_state,
            runtime=self.runtime,
        )

        questions: list[ClarifierQuestionDraft] = []
        if mode == "practise" and "scope:" not in " ".join(previous):
            questions.append(
                ClarifierQuestionDraft(
                    question=f"What exact drill or performance test should the next block on {focus} target?",
                    reason="The next block should attack one concrete rep, not a vague practice intention.",
                    answer_type="short_text",
                    optional=False,
                    option_candidates=["One isolated drill", "Timed repetition set", "Cold performance test"],
                    why_this_matters="Practice planning needs a concrete rep target before duration or coaching cues are useful.",
                    inferred_signal_type="wrong_scope",
                    downstream_effect="Sets the drill_type and practice_stage for the next practice block.",
                    confidence=0.62,
                )
            )
        elif "scope:" not in " ".join(previous):
            questions.append(
                ClarifierQuestionDraft(
                    question=f"What expected output should the next block on {focus} produce?",
                    reason="One concrete target is enough to keep the next block executable.",
                    answer_type="short_text",
                    optional=False,
                    option_candidates=["Explain one concept clearly", "Work one guided example", "Complete one diagnostic check"],
                    why_this_matters="The next block should optimize for one clear output rather than a vague intention.",
                    inferred_signal_type="wrong_scope",
                    downstream_effect="Anchors the next study or teaching block around one concrete outcome.",
                    confidence=0.64,
                )
            )

        if broad and "foundational" not in signal_counts and "needs_prerequisite" not in " ".join(previous):
            questions.append(
                ClarifierQuestionDraft(
                    question=f"Which starting layer should come first for {focus}?",
                    reason="Broad or ambitious requests need an explicit starting layer before they become a usable plan.",
                    answer_type="single_choice" if prerequisite_candidates else "short_text",
                    optional=False,
                    option_candidates=prerequisite_candidates or [
                        "Start from prerequisites",
                        "Start from one worked example",
                        "Start from the final target and diagnose gaps",
                    ],
                    why_this_matters=(
                        "A broad request needs a concrete entry layer before pb can produce a dependency-aware plan. "
                        "For advanced topics, that usually means proving the prerequisite floor first."
                    ),
                    inferred_signal_type="needs_prerequisite",
                    downstream_effect="Determines the earliest layer the next plan or block should target.",
                    confidence=0.58,
                )
            )
        elif signal_counts.get("foundational", 0) > 0 and "prerequisite" not in " ".join(previous):
            questions.append(
                ClarifierQuestionDraft(
                    question=f"What shaky prerequisite in {focus} should we rebuild first?",
                    reason="Prior feedback suggests the learner may need a lower restart point.",
                    answer_type="short_text",
                    optional=False,
                    option_candidates=["Vocabulary and notation", "Worked examples", "Formal definitions", "Application problems"],
                    why_this_matters="Repeated foundational signals should move the path downward instead of slightly simplifying the same target.",
                    inferred_signal_type="needs_prerequisite",
                    downstream_effect="Helps choose a restart point for a rebase or rebuild.",
                    confidence=0.74,
                )
            )

        return questions


def ask_clarifier_questions(questions: list[ClarifierQuestionDraft]) -> ClarifierAnswerBundle:
    """Prompt the user for clarifier answers and keep typed/selected intent distinct."""

    bundle = ClarifierAnswerBundle()
    for question in questions:
        options_presented = trim_context_option_candidates(question.option_candidates)
        lowered_answer_type = (question.answer_type or "").strip().lower()

        if options_presented and lowered_answer_type in {"multi_select", "multiselect", "many"}:
            selected = pick_many_choices(
                [(option, option) for option in options_presented] + [(DISCUSS_SENTINEL, "Let's discuss this")],
                title=question.question,
                text=question.why_this_matters or question.reason,
                allow_inline_edit=True,
                inline_prompt="Type your answer",
                return_result=True,
            )
            if getattr(selected, "kind", "") == "cancel" or selected is None:
                if question.optional:
                    continue
                return bundle
            selected_values: list[str] = []
            inline_answer = ""
            if getattr(selected, "kind", "") == "inline_text":
                raw_value = getattr(selected, "value", []) or []
                if isinstance(raw_value, list):
                    selected_values = [str(item).strip() for item in raw_value if str(item).strip()]
                    inline_answer = selected_values.pop() if selected_values else ""
                else:
                    inline_answer = str(raw_value or "").strip()
            else:
                raw_value = getattr(selected, "value", []) if hasattr(selected, "value") else selected
                if isinstance(raw_value, list):
                    selected_values = [str(item).strip() for item in raw_value if str(item).strip()]
            discussion_text = ""
            if DISCUSS_SENTINEL in selected_values:
                selected_values = [value for value in selected_values if value != DISCUSS_SENTINEL]
                discussion_text = prompt_text("Discuss", default="").strip()
            excluded_options = options_presented if inline_answer and is_explicit_negation(inline_answer) and not selected_values else []
            answer_parts = [value for value in selected_values if value.strip()]
            if discussion_text:
                answer_parts.append(discussion_text)
            if inline_answer:
                answer_parts.append(inline_answer)
            if excluded_options and not answer_parts:
                answer_parts.append(inline_answer)
            answer = "; ".join(answer_parts)
            if not answer and question.optional:
                continue
            if not answer and excluded_options:
                answer = inline_answer
            if not answer and discussion_text:
                answer = discussion_text
            answer_kind = "multi_select"
            if excluded_options:
                answer_kind = "explicit_exclusion"
            elif discussion_text and not selected_values and not inline_answer:
                answer_kind = "discussion_text"
            elif inline_answer and not selected_values:
                answer_kind = "custom_text"
            record = ClarifierAnswerRecord(
                question=question.question,
                answer=answer,
                answer_kind=answer_kind,
                selected_option="; ".join(selected_values),
                options_presented=options_presented,
                excluded_options=excluded_options,
                inferred_signal_type=question.inferred_signal_type,
            )
            bundle.answers[question.question] = answer
            bundle.records.append(record)
            _echo_clarifier_answer(record)
            continue

        if options_presented:
            selected = pick_single_choice(
                [(option, option) for option in options_presented] + [(DISCUSS_SENTINEL, "Let's discuss this")],
                title=question.question,
                text=question.why_this_matters or question.reason,
                allow_inline_edit=True,
                inline_prompt="Type your answer",
                return_result=True,
            )
            if getattr(selected, "kind", "") == "cancel" or selected is None:
                if question.optional:
                    continue
                return bundle
            if getattr(selected, "kind", "") == "inline_text":
                answer = str(getattr(selected, "value", "") or "").strip()
                excluded_options = options_presented if answer and is_explicit_negation(answer) else []
                answer_kind = "explicit_exclusion" if excluded_options else "custom_text"
                selected_option = ""
            else:
                selected_value = str(getattr(selected, "value", selected) or "").strip()
                excluded_options = []
                if selected_value == DISCUSS_SENTINEL:
                    answer = prompt_text("Discuss", default="").strip()
                    answer_kind = "discussion_text"
                    selected_option = ""
                else:
                    answer = selected_value
                    answer_kind = "selected_option"
                    selected_option = answer
            if not answer and question.optional:
                continue
            record = ClarifierAnswerRecord(
                question=question.question,
                answer=answer,
                answer_kind=answer_kind,
                selected_option=selected_option,
                options_presented=options_presented,
                excluded_options=excluded_options,
                inferred_signal_type=question.inferred_signal_type,
            )
            bundle.answers[question.question] = answer
            bundle.records.append(record)
            _echo_clarifier_answer(record)
            continue

        answer = prompt_text(question.question, default="").strip()
        if not answer and question.optional:
            continue
        record = ClarifierAnswerRecord(
            question=question.question,
            answer=answer,
            answer_kind="short_text",
            inferred_signal_type=question.inferred_signal_type,
        )
        bundle.answers[question.question] = answer
        bundle.records.append(record)
        _echo_clarifier_answer(record)
    return bundle
