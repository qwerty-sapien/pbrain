# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Socratic debrief engine for Phase 18 Socratic Capture.

Provides:
- Domain session detection (D-02)
- SocraticDebriefEngine: Flash Lite turn-by-turn interview (D-06)
- Adaptive diagnostic mode for downstream Anki gap capture
- Note builder with brief/deep templates (D-08, D-16)
- Frontmatter builder with enforced source:socratic (D-11)
- Socratic card extraction — verbatim Q&A, no LLM (D-22, ANKI-03)
- Wikilink inference via Flash Lite (D-09)
- Tier-2 note preview and confirm (D-13)
- Interactive debrief loop helper shared by all CLI entry points
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
import select
import sys
import time
from typing import Optional

import structlog

from pb.storage.yaml_io import dump_compact_yaml, extract_structured_yaml

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Domain session detection (D-02)
# ---------------------------------------------------------------------------


def is_domain_session(task, vault_path: Path) -> bool:
    """True if task's project maps to a knowledge/ subfolder with _state.md.

    A session is a domain session when:
    1. The active task has a project
    2. That project name maps to a folder under knowledge/
    3. That folder has a _state.md file (meaning it is an active learning domain)

    Deterministic — no LLM call.
    """
    if not task or not getattr(task, "project", None):
        return False
    project_obj = task.project
    project_name = getattr(project_obj, "name", "") or ""
    if not project_name:
        return False
    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return False
    domain_dir = knowledge_dir / project_name
    if domain_dir.is_dir() and (domain_dir / "_state.md").exists():
        return True
    return False


# ---------------------------------------------------------------------------
# SocraticDebriefEngine (D-06 coaching partner tone)
# ---------------------------------------------------------------------------


class SocraticDebriefEngine:
    """Turn-by-turn Flash Lite interview engine for Socratic debrief.

    Generates personalized questions using _state.md context and conversation
    history. Supports both short debriefs and longer adaptive diagnostics.
    """

    def __init__(
        self,
        domain: str,
        state_md_content: str,
        max_rounds: int,
        topic: str = "",
        *,
        strict: bool = False,
        adaptive: bool = False,
        teaching: bool = False,
        difficulty_start: str = "",
        difficulty_limit: str = "",
        soft_cap_rounds: Optional[int] = None,
        model: Optional[str] = None,
        bloom_level: str = "",
        prompt_guidance: str = "",
        time_limit_minutes: Optional[int] = None,
    ):
        self._domain = domain
        self._state_context = state_md_content
        self._topic = topic
        self._history: list[dict[str, str]] = []
        self._round = 0
        self._max = max_rounds
        self._strict = strict
        self._adaptive = adaptive
        self._teaching = teaching
        self._difficulty_start = difficulty_start.strip()
        self._difficulty_limit = difficulty_limit.strip()
        self._soft_cap = soft_cap_rounds or max_rounds
        self._model = model
        self._bloom_level = bloom_level.strip().lower()
        self._prompt_guidance = prompt_guidance.strip()
        self._time_limit_minutes = (
            max(3, min(15, int(time_limit_minutes)))
            if time_limit_minutes is not None
            else None
        )
        self._started_at = time.monotonic()
        self._stopped = False
        self._answered_rounds = 0
        self._exit_reason = ""
        self._minimum_answered_rounds = 8 if adaptive else (4 if teaching else 0)
        self._client = None  # lazy init

    def _get_client(self):
        if self._client is None:
            from pb.llm.gemini import get_client
            self._client = get_client()
        return self._client

    def get_question(self, prior_answer: str = "") -> Optional[str]:
        """Generate next question. First call: no prior answer.

        Returns None when max rounds reached or LLM unavailable.
        """
        if self._stopped or self._round >= self._max:
            if not self._exit_reason and self._round >= self._max:
                self._exit_reason = "completed"
            return None
        client = self._get_client()
        if not client.is_available():
            if not self._exit_reason:
                self._exit_reason = "llm_unavailable"
            return None
        if prior_answer:
            self._history.append({"role": "user", "text": prior_answer})
            self._answered_rounds += 1
        if self.time_remaining_seconds is not None and self.time_remaining_seconds <= 0:
            if not self._exit_reason:
                self._exit_reason = "time_limit"
            self._stopped = True
            return None
        prompt = self._build_prompt(prior_answer)
        from pb.llm.gemini import FLASH_LITE_MODEL, resolve_model

        model = resolve_model(self._model, fallback=FLASH_LITE_MODEL)
        # Round 0 uses a longer timeout: on Vertex AI, the ADC token refresh
        # on the very first call can easily exceed 15-20s, silently returning
        # None and causing run_debrief_loop to skip the whole session.
        if self._round == 0:
            timeout = 30
        elif self._adaptive:
            timeout = 20
        else:
            timeout = 15
        kwargs = {"timeout": timeout}
        if self._adaptive:
            kwargs["max_output_tokens"] = 4000
        question = client.generate_with_model(prompt, model, **kwargs)
        if question:
            question = question.strip()
            if question.upper() == "DONE":
                if self._answered_rounds < self._minimum_answered_rounds:
                    question = self._fallback_continuation_question()
                else:
                    self._stopped = True
                    self._exit_reason = "completed"
                    return None
            if question:
                self._history.append({"role": "mentor", "text": question})
                self._round += 1
                return question
        return None

    def _fallback_continuation_question(self) -> str:
        """Return a deterministic continuation prompt when the model stops too early."""
        if self._teaching:
            if self._answered_rounds <= 1:
                return "State the core idea in one clear sentence without leaning on the original wording."
            if self._answered_rounds == 2:
                return "Apply the idea to one fresh example and say what you would look for first."
            return "Before we wrap, what mistake would you now avoid, and what would you try next?"
        if self._adaptive:
            return "That is not enough to finish yet. What concrete missing step or distinction would you verify next?"
        return "What part still feels least stable, and how would you test it from memory?"

    def record_exit(self, reason: str) -> None:
        """Record an external exit reason without losing collected answers."""
        if not self._exit_reason:
            self._exit_reason = reason
        self._stopped = True

    @property
    def exit_reason(self) -> str:
        return self._exit_reason

    @property
    def answered_rounds(self) -> int:
        return self._answered_rounds

    @property
    def completed_normally(self) -> bool:
        return self._exit_reason in {"completed", "time_limit"}

    @property
    def time_remaining_seconds(self) -> Optional[int]:
        if self._time_limit_minutes is None:
            return None
        elapsed = int(time.monotonic() - self._started_at)
        return max(0, self._time_limit_minutes * 60 - elapsed)

    def build_completion_summary(self) -> str:
        """Return a short end-of-loop summary instead of stopping silently."""
        pairs = self.collect_answers()
        if not pairs:
            return ""
        topic = self._topic or self._domain
        if self._teaching:
            key_answer = pairs[0][1].strip()
            next_focus = pairs[-1][1].strip()
            return (
                f"You now have a working foothold in {topic}. "
                f"Key idea: {key_answer[:180]}. "
                f"Next, push it forward by testing {next_focus[:140]} on a fresh prompt."
            )
        if self._adaptive:
            prefix = (
                f"Diagnostic time box ended after {len(pairs)} exchanges on {topic}. "
                if self._exit_reason == "time_limit"
                else f"Diagnostic captured {len(pairs)} exchanges on {topic}. "
            )
            return prefix + "The transcript is ready to be turned into follow-up gaps, cards, and next questions."
        return (
            f"Captured {len(pairs)} exchanges on {topic}. "
            "Carry the sharpest answer into recall, a follow-up note, or the next deliberate attempt."
        )

    def should_continue(self) -> bool:
        """True if there are rounds remaining."""
        return not self._stopped and self._round < self._max

    @property
    def round_number(self) -> int:
        return self._round

    def collect_answers(self) -> list[tuple[str, str]]:
        """Return list of (question, answer) pairs from conversation history.

        Pairs are extracted from mentor/user turn sequence. Unanswered final
        questions (no following user turn) are omitted.
        """
        pairs = []
        i = 0
        while i < len(self._history) - 1:
            if self._history[i]["role"] == "mentor" and self._history[i + 1]["role"] == "user":
                pairs.append((self._history[i]["text"], self._history[i + 1]["text"]))
                i += 2
            else:
                i += 1
        return pairs

    def _build_prompt(self, prior_answer: str) -> str:
        """Build Flash Lite prompt with coaching partner tone (D-06).

        Includes _state.md content (capped at 2000 chars) for personalization.
        Includes conversation history for follow-up context.
        """
        if self._adaptive:
            return self._build_adaptive_prompt(prior_answer)
        if self._teaching:
            return self._build_teaching_prompt(prior_answer)

        lines = [
            "You are a coaching partner helping a learner reflect on their study session.",
            f"Domain: {self._domain}",
            "",
            "Context from domain state:",
            self._state_context[:2000],  # cap context (T-18-06 mitigation)
            "",
        ]
        if self._history:
            lines.append("Conversation so far:")
            for entry in self._history:
                role_label = "Mentor" if entry["role"] == "mentor" else "Learner"
                lines.append(f"  {role_label}: {entry['text']}")
            lines.append("")
        if self._topic:
            lines.append(f"Focus topic: {self._topic}")
            lines.append("")
        if self._prompt_guidance:
            lines.append("User feedback guidance:")
            lines.append(self._prompt_guidance[:1500])
            lines.append("")
        if prior_answer:
            lines.append(f"The learner just said: {prior_answer}")
            lines.append("Ask a follow-up question that probes deeper into what they learned.")
        else:
            if self._topic:
                lines.append(f"Ask ONE opening question specifically about: {self._topic}")
            else:
                lines.append("Ask ONE opening question about what the learner discovered or struggled with in this session.")
                lines.append("Use the domain context above to make it specific (e.g. reference a topic from session summaries).")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Ask exactly ONE question")
        lines.append("- Coaching partner tone: warm, specific, curious")
        lines.append("- Reference specific topics from the domain context when possible")
        lines.append("- Keep under 2 sentences")
        return "\n".join(lines)

    def _build_teaching_prompt(self, prior_answer: str) -> str:
        """Build a supportive Socratic teaching prompt."""
        bloom_level = self._bloom_level or "apply"
        lines = [
            "You are a supportive Socratic teacher helping a learner internalise a new concept.",
            f"Domain: {self._domain}",
            f"Target Bloom's Taxonomy Level: {bloom_level.capitalize()}",
            "",
            "Context from domain state:",
            self._state_context[:2000],
            "",
        ]
        if self._history:
            lines.append("Conversation so far:")
            for entry in self._history:
                role_label = "Teacher" if entry["role"] == "mentor" else "Learner"
                lines.append(f"  {role_label}: {entry['text']}")
            lines.append("")
        if self._topic:
            lines.append(f"Focus concept: {self._topic}")
            lines.append("")
        if self._prompt_guidance:
            lines.append("User feedback guidance:")
            lines.append(self._prompt_guidance[:1500])
            lines.append("")
        if prior_answer:
            lines.append(f"The learner just answered: {prior_answer}")
            lines.append(
                "Ask ONE short next question or micro-prompt that checks understanding, "
                "corrects gently if needed, and moves one step toward application."
            )
        else:
            lines.append("Ask ONE short prior-knowledge check or anchoring question to begin the lesson.")
        lines.append("")
        lines.append("Rules:")
        lines.append("- Ask exactly ONE question, or output ONLY DONE if the teaching loop is genuinely complete")
        lines.append("- This is teaching, not a diagnostic or interrogation")
        lines.append("- Progress from orienting -> explaining -> applying -> checking")
        lines.append("- Keep the tone warm, specific, and confidence-building")
        lines.append("- Keep under 2 sentences")
        return "\n".join(lines)

    def _build_adaptive_prompt(self, prior_answer: str) -> str:
        """Build a stricter adaptive diagnostic prompt for Anki gap finding."""
        import os
        bloom_level = os.environ.get("PB_BLOOMS_LEVEL", "apply").lower()
        
        difficulty_start = self._difficulty_start or "foundational basics"
        difficulty_limit = self._difficulty_limit or "unbounded"
        
        lines = [
            "You are a Socratic diagnostic interviewer preparing downstream Anki cards.",
            f"Domain: {self._domain}",
            f"Target Bloom's Taxonomy Level: {bloom_level.capitalize()}",
            f"Starting difficulty: {difficulty_start}",
            f"Difficulty ceiling: {difficulty_limit}",
            f"Current round: {self._round + 1} of {self._max}",
            f"Soft cap: around {self._soft_cap} exchanges",
        ]
        if self._time_limit_minutes is not None:
            remaining_seconds = self.time_remaining_seconds or 0
            remaining_minutes, remaining_secs = divmod(remaining_seconds, 60)
            lines.extend(
                [
                    f"Time limit: {self._time_limit_minutes} minutes",
                    f"Time remaining: {remaining_minutes}m {remaining_secs:02d}s",
                ]
            )
        lines.extend(
            [
            "",
            "CRITICAL ANTI-PEDANTRY RULE:",
            f"Match the interrogation depth strictly to the target Bloom's level ({bloom_level}).",
            ]
        )
        
        if bloom_level in ["remember", "understand", "apply"]:
            lines.extend([
                "Since the focus is foundational or practical application:",
                "DO NOT ask pedantic theoretical questions, linguistic theory, or deep origin rules.",
                "Focus entirely on active recall, practical usage examples, and correcting immediate mistakes.",
                "Keep questions highly pragmatic.",
            ])
            
        lines.extend([
            "",
            "Context from domain state:",
            self._state_context[:2200],
            "",
        ])
        if self._prompt_guidance:
            lines.extend([
                "User feedback guidance:",
                self._prompt_guidance[:1500],
                "",
            ])

        if self._history:
            lines.append("Conversation so far:")
            for entry in self._history:
                role_label = "Interviewer" if entry["role"] == "mentor" else "Learner"
                lines.append(f"  {role_label}: {entry['text']}")
            lines.append("")
        if self._topic:
            lines.append(f"Focus topic: {self._topic}")
            lines.append("")
        if prior_answer:
            lines.append(f"The learner just answered: {prior_answer}")
            lines.append(
                "Decide whether they showed mastery, partial understanding, or confusion, "
                "then adjust the next question accordingly."
            )
        else:
            opening = (
                f"Start with the most basic testable question about {self._topic}."
                if self._topic
                else "Start with the most basic testable question in this domain."
            )
            lines.append(opening)
        lines.append("")
        lines.append("Rules:")
        lines.append("- Ask exactly ONE question, or output ONLY DONE if the diagnostic is genuinely complete")
        lines.append("- Start from basics and steadily ramp up difficulty")
        lines.append("- If the learner falters, lower difficulty slightly and verify the missing step before ramping again")
        lines.append("- Be strict: reject vague answers and probe assumptions")
        lines.append("- Prefer short, precise questions; do not answer your own question")
        lines.append("- Avoid lists, labels, commentary, praise, or preambles")
        lines.append("- Keep each question under 3 sentences")
        lines.append("- Do not stop before at least 8 answered rounds unless the user clearly exits")
        remaining_seconds = self.time_remaining_seconds
        if remaining_seconds is not None:
            if remaining_seconds >= 180:
                lines.append("- There is still useful time left. A deeper question is fine when it exposes a structural gap, and you can allow a short pause for thinking if the payoff is high.")
            elif remaining_seconds >= 90:
                lines.append("- You are in the middle stretch. Prefer direct, efficient gap-finding questions and avoid opening brand-new branches.")
            else:
                lines.append("- Time is short. Ask a quick, high-signal verification question or close the single biggest unresolved gap, then wrap.")
        if self._round + 1 >= self._soft_cap:
            lines.append(
                "- You are near the soft cap. Only output DONE if there are no major unresolved gaps; "
                "otherwise close the biggest gap next."
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Frontmatter builder (D-11 — enforced programmatically, not by LLM)
# ---------------------------------------------------------------------------


def build_socratic_frontmatter(domain: str, slug: str, wikilinks: list[str]) -> dict:
    """Build frontmatter dict. source: socratic is enforced in code, not by LLM.

    T-18-07 mitigation: programmatic enforcement prevents LLM from omitting the tag.
    """
    return {
        "source": "socratic",
        "learning_stage": "#new",
        "domain": domain,
        "links": wikilinks,
        "created": date.today().isoformat(),
    }


# ---------------------------------------------------------------------------
# Note builder (D-08, D-16 — two templates, verbatim user words)
# ---------------------------------------------------------------------------


def build_socratic_note(
    qa_pairs: list[tuple[str, str]],
    domain: str,
    slug: str,
    wikilinks: list[str],
    template: str = "brief",
) -> str:
    """Build a complete Socratic note with frontmatter and body.

    template="brief"  — from pb finish: ## Insight, ## Links (D-16)
    template="deep"   — from pb learn: ## Key Insight, ## Connections,
                        ## Open Questions, ## Cross-Domain (D-16)

    User's exact words are preserved. Flash Lite only adds structural
    scaffolding (D-08). No LLM call is made here.
    """
    from pb.vault.lifecycle import write_frontmatter

    fm = build_socratic_frontmatter(domain, slug, wikilinks)
    if template == "brief":
        body = _build_brief_body(qa_pairs, wikilinks)
    else:
        body = _build_deep_body(qa_pairs, wikilinks)
    return write_frontmatter(fm, body)


def _build_brief_body(qa_pairs: list[tuple[str, str]], wikilinks: list[str]) -> str:
    """Brief template: ## Insight, ## Links (D-16).

    User's verbatim answers are placed directly under each question.
    """
    lines = ["## Insight", ""]
    for q, a in qa_pairs:
        lines.append(f"> {q}")
        lines.append("")
        lines.append(a)  # user's exact words
        lines.append("")
    lines.append("## Links")
    lines.append("")
    for link in wikilinks:
        lines.append(f"- [[{link}]]")
    return "\n".join(lines)


def _build_deep_body(qa_pairs: list[tuple[str, str]], wikilinks: list[str]) -> str:
    """Deep template: ## Key Insight, ## Connections, ## Open Questions, ## Cross-Domain (D-16).

    Maps first Q&A to Key Insight, Q2-Q3 to Connections, Q4 to Open Questions,
    Q5 to Cross-Domain. User's exact words are preserved throughout.
    """
    lines = ["## Key Insight", ""]
    if qa_pairs:
        # First Q&A is the key insight
        q, a = qa_pairs[0]
        lines.append(f"> {q}")
        lines.append("")
        lines.append(a)  # user's exact words
        lines.append("")
    lines.append("## Connections")
    lines.append("")
    for q, a in qa_pairs[1:3]:  # Q2-Q3 as connections
        lines.append(f"> {q}")
        lines.append("")
        lines.append(a)  # user's exact words
        lines.append("")
    for link in wikilinks:
        lines.append(f"- [[{link}]]")
    lines.append("")
    lines.append("## Open Questions")
    lines.append("")
    if len(qa_pairs) > 3:
        q, a = qa_pairs[3]
        lines.append(f"> {q}")
        lines.append("")
        lines.append(a)  # user's exact words
        lines.append("")
    lines.append("## Cross-Domain")
    lines.append("")
    if len(qa_pairs) > 4:
        q, a = qa_pairs[4]
        lines.append(f"> {q}")
        lines.append("")
        lines.append(a)  # user's exact words
        lines.append("")
    return "\n".join(lines)


def build_teach_lesson_note(
    *,
    domain: str,
    slug: str,
    topic: str,
    qa_pairs: list[tuple[str, str]],
    wikilinks: list[str],
    summary: str,
    key_insight: str,
    downstream_concepts: list[str],
    next_attempts: list[str],
    completed: bool,
) -> str:
    """Build a durable lesson note from a teach session transcript."""
    from pb.vault.lifecycle import write_frontmatter

    fm = build_socratic_frontmatter(domain, slug, wikilinks)
    fm["conversation_kind"] = "teach"
    fm["topic"] = topic or slug
    fm["status"] = "complete" if completed else "partial"

    lines = [
        "## Lesson Summary",
        "",
        summary.strip() or f"Guided teaching session on {topic or domain}.",
        "",
        "## Key Insight",
        "",
        key_insight.strip() or (qa_pairs[0][1] if qa_pairs else "No key insight captured."),
        "",
        "## Dialogue Seeds",
        "",
    ]
    for question, answer in qa_pairs:
        lines.append(f"> {question}")
        lines.append("")
        lines.append(answer)
        lines.append("")

    lines.extend(["## Linked Concepts", ""])
    if wikilinks:
        lines.extend(f"- [[{link}]]" for link in wikilinks)
    else:
        lines.append("_No linked concepts inferred yet._")
    lines.append("")

    lines.extend(["## Future Concepts", ""])
    if downstream_concepts:
        lines.extend(f"- {item}" for item in downstream_concepts)
    else:
        lines.append(f"- Explore how {topic or domain} connects to a nearby prerequisite or follow-on concept.")
    lines.append("")

    lines.extend(["## Next Attempts", ""])
    if next_attempts:
        lines.extend(f"- {item}" for item in next_attempts)
    else:
        lines.extend(
            [
                f"- Explain {topic or domain} again from memory in one tight paragraph.",
                f"- Try one fresh application or worked example involving {topic or domain}.",
                f"- Generate recall prompts for {topic or domain} once the lesson feels stable.",
            ]
        )
    lines.append("")

    return write_frontmatter(fm, "\n".join(lines).rstrip() + "\n")


# ---------------------------------------------------------------------------
# Socratic card extraction (D-22 — verbatim Q&A, no LLM, card_type=socratic)
# ---------------------------------------------------------------------------


def extract_socratic_cards(
    qa_pairs: list[tuple[str, str]],
    note_slug: str,
    deck: str,
    domain: str,
) -> list[dict]:
    """Extract Anki cards from debrief Q&A pairs.

    No LLM — user's words are verbatim (D-22, ANKI-03).
    card_type is always "socratic".
    """
    import datetime

    cards = []
    for i, (question, answer) in enumerate(qa_pairs):
        card_id = f"{note_slug}-socratic-{i}"
        now = datetime.datetime.now().isoformat()
        cards.append({
            "id": card_id,
            "note_slug": note_slug,
            "front": question,
            "back": answer,
            "card_type": "socratic",
            "status": "pending",
            "deck": deck,
            "tags": dump_compact_yaml([domain, "socratic"]),
            "anki_model": "Basic",
            "created_at": now,
            "updated_at": now,
        })
    return cards


def build_diagnostic_report(
    qa_pairs: list[tuple[str, str]],
    domain: str,
    state_md_content: str,
    *,
    topic: str = "",
    difficulty_start: str = "",
    difficulty_limit: str = "",
    note_types: Optional[list[str]] = None,
    model: Optional[str] = None,
) -> dict:
    """Summarize a diagnostic transcript into YAML-ready downstream signals."""
    note_types = list(note_types or [])
    transcript = [
        {"question": question, "answer": answer}
        for question, answer in qa_pairs
    ]
    report = {
        "domain": domain,
        "topic": topic,
        "difficulty_start": difficulty_start or "foundational basics",
        "difficulty_limit": difficulty_limit or "unbounded",
        "rounds": len(qa_pairs),
        "note_types": note_types,
        "transcript": transcript,
        "summary": "",
        "strengths": [],
        "knowledge_gaps": [],
        "candidate_cards": [],
        "readiness": {
            "overall": "unknown",
            "next_focus": "",
        },
    }
    if not qa_pairs:
        return report

    try:
        from pb.llm.gemini import FLASH_MODEL, get_client, resolve_model

        client = get_client()
        if client.is_available():
            transcript_text = "\n\n".join(
                f"Q{i}: {question}\nA{i}: {answer}"
                for i, (question, answer) in enumerate(qa_pairs, start=1)
            )
            note_type_list = ", ".join(note_types) if note_types else "Basic"
            prompt = (
                "You are turning a strict Socratic diagnostic transcript into downstream Anki generation guidance.\n"
                f"Domain: {domain}\n"
                f"Topic: {topic or 'general'}\n"
                f"Starting difficulty: {difficulty_start or 'foundational basics'}\n"
                f"Difficulty ceiling: {difficulty_limit or 'unbounded'}\n"
                f"Allowed note types: {note_type_list}\n\n"
                "Domain state context:\n"
                f"{state_md_content[:1800]}\n\n"
                "Transcript:\n"
                f"{transcript_text[:7000]}\n\n"
                "Return YAML only with keys:\n"
                "summary: string\n"
                "strengths:\n"
                "  - string\n"
                "knowledge_gaps:\n"
                "  - concept: string\n"
                "    observed_issue: string\n"
                "    evidence: string\n"
                "    recommended_note_types:\n"
                "      - string\n"
                "    target_difficulty: string\n"
                "candidate_cards:\n"
                "  - note_type: string\n"
                "    concept: string\n"
                "    why_now: string\n"
                "readiness:\n"
                "  overall: needs_foundation | developing | solid | advanced\n"
                "  next_focus: string\n"
                "Be concrete. Prefer gaps that should become cards."
            )
            result = client.generate_with_model(
                prompt,
                resolve_model(model, fallback=FLASH_MODEL),
                timeout=40,
                max_output_tokens=4000,
            )
            structured = extract_structured_yaml(result or "", {})
            if isinstance(structured, dict):
                report.update({k: v for k, v in structured.items() if v is not None})
    except Exception as exc:
        logger.debug("socratic.diagnostic_report_failed", error=str(exc))

    if not report.get("knowledge_gaps"):
        gaps = []
        for question, answer in qa_pairs:
            if len(answer.split()) >= 5:
                continue
            gaps.append({
                "concept": question[:80],
                "observed_issue": "Answer was too thin to verify understanding.",
                "evidence": answer,
                "recommended_note_types": note_types or ["Basic"],
                "target_difficulty": difficulty_start or "foundational basics",
            })
        report["knowledge_gaps"] = gaps

    if not report.get("summary"):
        report["summary"] = (
            f"Diagnostic completed for {domain} across {len(qa_pairs)} exchanges."
        )

    return report


# ---------------------------------------------------------------------------
# Wikilink inference helper (D-09 — Flash Lite infers from context)
# ---------------------------------------------------------------------------


def infer_wikilinks(
    user_text: str,
    domain: str,
    state_md_content: str,
    vault_path: Path,
) -> list[str]:
    """Use Flash Lite to infer wikilinks from user's text + domain context (D-09).

    Returns list of note stem names to wikify.
    Returns empty list if LLM unavailable.
    T-18-05 mitigation: returns candidate names only; caller validates against vault.db.
    """
    try:
        from pb.llm.gemini import get_client, FLASH_LITE_MODEL
        client = get_client()
        if not client.is_available():
            return []
        # Get existing note slugs from vault.db for candidate matching
        existing_slugs = _get_existing_slugs(vault_path)
        prompt = (
            "Given the following user insight text and context, suggest existing notes to link via [[wikilink]].\n"
            f"Domain: {domain}\n"
            f"Domain state context:\n{state_md_content[:1000]}\n\n"
            f"User's insight text:\n{user_text}\n\n"
            f"Existing notes in vault (candidates for wikilinks):\n{', '.join(existing_slugs[:100])}\n\n"
            "Return ONLY a YAML list of note names to link.\n"
            "If no candidates found, return [].\n"
            "Only suggest notes that are genuinely related to the user's text."
        )
        result = client.generate_with_model(prompt, FLASH_LITE_MODEL, timeout=10)
        if result:
            links = extract_structured_yaml(result.strip(), [])
            if isinstance(links, list):
                return [str(lnk) for lnk in links if isinstance(lnk, str)]
        return []
    except Exception:
        return []


def _get_existing_slugs(vault_path: Path) -> list[str]:
    """Get note slugs from vault.db nodes table. Returns empty list on error."""
    try:
        from pb.vault.graph_store import open_vault_db
        conn = open_vault_db(vault_path)
        try:
            rows = conn.execute(
                "SELECT slug FROM nodes ORDER BY updated_at DESC LIMIT 200"
            ).fetchall()
            return [r[0] for r in rows]
        finally:
            conn.close()
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Tier-2 note preview and confirm (D-13, 18-UI-SPEC Pattern B)
# ---------------------------------------------------------------------------


def show_note_preview_and_confirm(
    console,
    note_content: str,
    slug: str,
    domain: str,
    trust: bool = False,
    vault_path: Optional[Path] = None,
    state_md_content: str = "",
) -> Optional[str]:
    """Display Rich-rendered draft, offer save/edit/cancel.

    Returns final note content string (saved), or None if cancelled.

    D-13: User sees draft before any file is written.
    D-14: Post-edit wikilink inference runs unless --trust flag set.
    """
    console.rule("[header]Draft Note[/]")
    console.print(f"[subheader]{slug}.md[/]")
    console.print(note_content)
    console.print()
    console.print("[dim]s[/] save  [dim]e[/] edit in $EDITOR  [dim]c[/] cancel")

    try:
        choice = input("> ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return None

    if choice == "s":
        return note_content
    elif choice == "e":
        import os
        import subprocess
        import tempfile

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            f.write(note_content)
            tmp_path = f.name
        editor = os.environ.get("EDITOR", "vim")
        subprocess.run([editor, tmp_path])
        edited = Path(tmp_path).read_text()
        os.unlink(tmp_path)
        if not trust and vault_path:
            console.print("[warn]Post-edit: re-running wikilink inference...[/]")
            from pb.vault.lifecycle import read_frontmatter, write_frontmatter
            fm, body = read_frontmatter(edited)
            new_links = infer_wikilinks(body, domain, state_md_content, vault_path)
            if new_links:
                fm["links"] = list(set(fm.get("links", []) + new_links))
                edited = write_frontmatter(fm, body)
        # Re-show preview recursively (D-14)
        return show_note_preview_and_confirm(
            console, edited, slug, domain, trust, vault_path, state_md_content
        )
    return None


# ---------------------------------------------------------------------------
# Interactive debrief turn loop (shared by pb finish, pb learn, pb note)
# ---------------------------------------------------------------------------


def run_debrief_loop(
    engine: SocraticDebriefEngine,
    console,
    *,
    note_slug: str = "",
    deck: str = "",
    domain: str = "",
    generate_anki: bool = False,
) -> list[tuple[str, str]]:
    """Run the interactive debrief turn loop. Returns collected Q&A pairs.

    Shared helper called by all three CLI entry points (pb finish, pb learn,
    pb note) so the turn loop is implemented once and tested once.

    When generate_anki=True and engine._max >= 5 (deep debrief), fires ANKI-03 hook
    to extract Socratic cards after the Q&A exchange (D-01, D-02, D-03).
    """
    console.print("[dim]Type 'finish', 'done', 'q', 'skip', or Ctrl+C to exit[/]")
    console.print()

    question = engine.get_question()
    if not question:
        console.print("[dim]LLM unavailable — debrief skipped[/]")
        return []

    while engine.should_continue() and question:
        console.print(f"[dim]Q{engine.round_number}[/] {question}")
        if engine.time_remaining_seconds is not None:
            remaining_seconds = engine.time_remaining_seconds
            remaining_minutes, remaining_secs = divmod(remaining_seconds, 60)
            console.print(f"[dim]Time left: {remaining_minutes}m {remaining_secs:02d}s[/]")
        try:
            answer, read_status = _read_debrief_input("> ", timeout_seconds=engine.time_remaining_seconds)
        except (EOFError, KeyboardInterrupt):
            console.print()
            engine.record_exit("interrupted")
            break
        if read_status == "timeout":
            console.print()
            console.print("[dim]Time limit reached — wrapping the diagnostic here.[/]")
            engine.record_exit("time_limit")
            break
        if answer is None:
            console.print()
            engine.record_exit("interrupted")
            break
        lowered = answer.lower()
        if lowered in ("skip", "/exit", "/quit", "", "q"):
            engine.record_exit("user_exit")
            break
        if lowered in ("finish", "done", "/finish", "/done"):
            engine.record_exit("completed" if engine.collect_answers() else "user_exit")
            break
        question = engine.get_question(answer)

    pairs = engine.collect_answers()
    summary = engine.build_completion_summary() if engine.completed_normally else ""
    if summary:
        console.print()
        console.print("[subheader]Session Summary[/]")
        console.print(summary)
    console.print(f"\n[dim]Debrief complete -- {engine.round_number} rounds[/]")

    # ANKI-03 hook: only on deep debriefs (engine._max >= 5) when caller opts in (D-01, D-02)
    if generate_anki and engine._max >= 5 and pairs and note_slug:
        try:
            import uuid as _uuid
            from pb.vault.anki_client import (
                insert_cards_to_db,
                _insert_run_log_entry,
            )
            _run_id = str(_uuid.uuid4())[:8]
            _cards = extract_socratic_cards(pairs, note_slug, deck, domain)
            for _c in _cards:
                _c["run_id"] = _run_id
                _c["domain"] = domain or ""
            _inserted = insert_cards_to_db(_cards)
            if _inserted:
                console.print(
                    f"[dim]Generated {_inserted} Anki card(s) from debrief (run {_run_id})[/]"
                )
            _insert_run_log_entry(_run_id, note_slug, None, _inserted, "socratic")
        except Exception as _exc:
            # Non-fatal: hook failure must never crash the debrief (D-01)
            try:
                logger.debug("socratic.anki_hook_failed", error=str(_exc))
            except Exception:
                pass

    return pairs


def _read_debrief_input(prompt: str, *, timeout_seconds: Optional[int] = None) -> tuple[Optional[str], str]:
    """Read one debrief answer, optionally enforcing a hard timeout."""
    if timeout_seconds is not None and timeout_seconds <= 0:
        return None, "timeout"
    if timeout_seconds is None or not sys.stdin.isatty():
        return input(prompt).strip(), "ok"

    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        ready, _, _ = select.select([sys.stdin], [], [], timeout_seconds)
    except (OSError, ValueError):
        return input("").strip(), "ok"
    if not ready:
        return None, "timeout"
    line = sys.stdin.readline()
    if line == "":
        return None, "eof"
    return line.strip(), "ok"
