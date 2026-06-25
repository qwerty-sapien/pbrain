# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""FinishAssessmentAgent -- AI-conducted post-session assessment.

Uses captured session evidence and LLM structured output to produce sub-skill
scores and retry queue items without adding finish-time questions.
"""

from __future__ import annotations

import re
import sys
from typing import Any, Optional, TYPE_CHECKING

import structlog
from pydantic import BaseModel

from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.session_blueprints import blueprint_from_payload

if TYPE_CHECKING:
    from pb.domain.models import Session, Task

logger = structlog.get_logger()

# LOCKED constants (CLAUDE.md -- do not change)
FLASH_LITE_MODEL = "gemini-3.1-flash-lite-preview"
FLASH_MODEL = "gemini-3-flash-preview"

# Weak sub-skill threshold: score <= this triggers auto-retry queue (D-10)
WEAK_THRESHOLD = 2

_ESCALATE_RE = re.compile(r"^ESCALATE:\s*(flash|pro)\s*$", re.IGNORECASE)


class SubSkillScore(BaseModel):
    """Score for a single sub-skill assessed during the session."""

    name: str
    score: int  # 1-5 scale
    is_weak: bool  # True if score <= WEAK_THRESHOLD
    notes: str = ""
    cluster_parent: str = ""  # per D-12: novel sub-skills must cluster under existing taxonomy


class AssessmentResult(BaseModel):
    """Structured output from the finish assessment agent."""

    sub_skill_scores: list[SubSkillScore]
    critique: str
    retry_items: list[str]  # items to enqueue in retry queue
    model_used: str = ""


class FinishAssessmentAgent:
    """AI-conducted post-session assessment agent.

    Per D-05: Runs an AI-conducted assessment, not a static form.
    Per D-07: Agent is strict and frank -- harsh rather than lenient.
    Per D-08: Biased toward expeditious completion -- terminates early if proficiency is clear.
    Per D-09: Escalating LLM tier -- Flash Lite -> Flash on ambiguity.
    Per D-10: Weak sub-skills (score <= WEAK_THRESHOLD) auto-feed retry queue.
    """

    def __init__(self) -> None:
        self._model_used = FLASH_LITE_MODEL

    def is_available(self) -> bool:
        """Check if LLM runtime is available for assessment."""
        try:
            from pb.llm.runtime import LLMRuntime

            return LLMRuntime().health().available
        except Exception:
            return False

    def run(
        self,
        session: "Session",
        task: "Task",
        domain: str,
    ) -> Optional[AssessmentResult]:
        """Run the evidence-derived assessment. Returns AssessmentResult or None.

        Returns None if:
        - Non-TTY context (tests, MCP, piped input)
        - LLM not available
        - Session duration < 5 minutes (too short for meaningful assessment)
        """
        # Non-TTY guard (per Pitfall 2 in RESEARCH.md)
        if not sys.stdin.isatty():
            logger.debug("finish_assessment.skipped", reason="non_tty")
            return None

        if not self.is_available():
            logger.debug("finish_assessment.skipped", reason="llm_unavailable")
            return None

        # Short session guard (Open Question 2 in RESEARCH.md)
        duration_min = 0
        if session.end_at and session.start_at:
            duration_min = max(0, int((session.end_at - session.start_at).total_seconds() / 60))
        if duration_min < 5:
            logger.debug("finish_assessment.skipped", reason="short_session", duration_min=duration_min)
            return None

        try:
            from pb.core.domain_templates import get_template
            from pb.llm.runtime import LLMRuntime

            meta = parse_learning_task_metadata(task)
            generated = dict(getattr(session, "generated_names", {}) or {})
            blueprint = blueprint_from_payload(
                generated.get("session_blueprint") if isinstance(generated.get("session_blueprint"), dict) else meta.session_blueprint
            )
            template = get_template(
                domain,
                branch=getattr(session, "branch", "") or meta.branch or "study",
                session=session,
                task=task,
            )
            runtime = LLMRuntime()

            assessment_targets = self._assessment_targets(
                session=session,
                task=task,
                domain=domain,
                template=template,
                blueprint=blueprint,
            )
            if not assessment_targets:
                logger.debug("finish_assessment.skipped", reason="no_assessment_targets")
                return None

            evidence_block = self._evidence_block(session=session, task=task)
            result = self._run_assessment(
                runtime=runtime,
                session=session,
                task=task,
                domain=domain,
                selected_skills=assessment_targets,
                template=template,
                blueprint=blueprint,
                duration_min=duration_min,
                evidence_block=evidence_block,
            )

            if result is not None:
                # Mark weak sub-skills (D-10)
                for ss in result.sub_skill_scores:
                    ss.is_weak = ss.score <= WEAK_THRESHOLD

                # Per D-12: Enforce cluster_parent for novel sub-skills.
                # Any sub-skill not in the base taxonomy must have a cluster_parent
                # that IS in the taxonomy. If it doesn't, mark it as session-only
                # by clearing cluster_parent (it won't persist to taxonomy).
                base_taxonomy = set(template.sub_skill_taxonomy)
                for ss in result.sub_skill_scores:
                    if ss.name not in base_taxonomy:
                        if ss.cluster_parent and ss.cluster_parent in base_taxonomy:
                            # Valid clustering -- novel sub-skill under existing category
                            logger.debug(
                                "finish_assessment.novel_skill_clustered",
                                name=ss.name,
                                cluster_parent=ss.cluster_parent,
                            )
                        else:
                            # No valid cluster_parent -- mark as session-only
                            ss.cluster_parent = ""
                            ss.notes = (ss.notes + " [session-only: not in base taxonomy]").strip()
                            logger.debug(
                                "finish_assessment.novel_skill_session_only",
                                name=ss.name,
                            )

                # Auto-generate retry items from weak sub-skills
                auto_retry = []
                for ss in result.sub_skill_scores:
                    if ss.is_weak:
                        auto_retry.append(
                            f"{domain}: {ss.name} needs targeted practice (scored {ss.score}/5)"
                        )
                if auto_retry and not result.retry_items:
                    result.retry_items = auto_retry
                elif auto_retry:
                    # Merge auto-generated with agent-suggested, deduplicate
                    existing = set(result.retry_items)
                    for item in auto_retry:
                        if item not in existing:
                            result.retry_items.append(item)

                result.model_used = self._model_used
                logger.info(
                    "finish_assessment.completed",
                    domain=domain,
                    skills_assessed=len(result.sub_skill_scores),
                    retry_items=len(result.retry_items),
                    model=self._model_used,
                )

            return result

        except KeyboardInterrupt:
            logger.debug("finish_assessment.cancelled")
            return None
        except Exception as e:
            logger.warning("finish_assessment.failed", error=str(e))
            return None

    @staticmethod
    def _append_unique(items: list[str], value: object) -> None:
        text = str(value or "").strip()
        if not text:
            return
        key = text.lower()
        if key not in {item.lower() for item in items}:
            items.append(text)

    def _assessment_targets(
        self,
        *,
        session,
        task,
        domain: str,
        template,
        blueprint,
    ) -> list[str]:
        """Infer assessment targets from captured evidence, not user prompts."""

        targets: list[str] = []
        if blueprint is not None:
            for skill in blueprint.subskills:
                self._append_unique(targets, skill)
        for skill in getattr(template, "sub_skill_taxonomy", []) or []:
            self._append_unique(targets, skill)

        generated = dict(getattr(session, "generated_names", {}) or {})
        evidence_items = generated.get("learning_partner_evidence")
        if isinstance(evidence_items, list):
            for item in evidence_items:
                if isinstance(item, dict):
                    self._append_unique(targets, item.get("subskill"))

        progress = generated.get("learning_partner_progress")
        if isinstance(progress, dict):
            self._append_unique(targets, progress.get("page_slug"))
            self._append_unique(targets, progress.get("question_slug"))

        compact = generated.get("learning_partner_compact")
        if isinstance(compact, dict):
            for key in ("detected_gaps", "unknowns", "corrections"):
                values = compact.get(key)
                if isinstance(values, list):
                    for value in values:
                        self._append_unique(targets, value)

        self_reports = generated.get("learner_self_reports")
        if isinstance(self_reports, list):
            for item in self_reports:
                if isinstance(item, dict):
                    self._append_unique(targets, item.get("topic"))

        if not targets:
            self._append_unique(targets, getattr(session, "subject_scope", "") or getattr(task, "title", "") or domain)
        return targets

    @staticmethod
    def _format_mapping_items(items: list[Any], *, keys: tuple[str, ...]) -> list[str]:
        lines: list[str] = []
        for item in items:
            if not isinstance(item, dict):
                text = str(item).strip()
                if text:
                    lines.append(f"- {text}")
                continue
            parts = []
            for key in keys:
                value = str(item.get(key, "") or "").strip()
                if value:
                    parts.append(f"{key}={value}")
            if parts:
                lines.append(f"- {'; '.join(parts)}")
        return lines

    def _evidence_block(self, *, session, task) -> str:
        """Render compact evidence signals for the assessment prompt."""

        generated = dict(getattr(session, "generated_names", {}) or {})
        lines: list[str] = []

        outcome = str(getattr(session, "actual_outcome", "") or "").strip()
        errors = str(getattr(session, "observed_errors", "") or "").strip()
        next_adjustment = str(getattr(session, "next_adjustment", "") or "").strip()
        if outcome:
            lines.append(f"- finish_note: {outcome}")
        if errors:
            lines.append(f"- observed_errors: {errors}")
        if next_adjustment:
            lines.append(f"- next_adjustment: {next_adjustment}")

        evidence_items = generated.get("learning_partner_evidence")
        if isinstance(evidence_items, list) and evidence_items:
            lines.append("Learning partner evidence:")
            lines.extend(
                self._format_mapping_items(
                    evidence_items[-12:],
                    keys=("source", "subskill", "note", "evidence"),
                )
            )

        compact = generated.get("learning_partner_compact")
        if isinstance(compact, dict) and compact:
            lines.append("Learning partner compact memory:")
            for key in ("summary", "knowns", "unknowns", "detected_gaps", "corrections", "next_drill", "next_action"):
                value = compact.get(key)
                if isinstance(value, list):
                    cleaned = [str(item).strip() for item in value if str(item).strip()]
                    if cleaned:
                        lines.append(f"- {key}: {', '.join(cleaned[:8])}")
                elif str(value or "").strip():
                    lines.append(f"- {key}: {str(value).strip()}")

        self_reports = generated.get("learner_self_reports")
        if isinstance(self_reports, list) and self_reports:
            lines.append("Learner self-reports:")
            lines.extend(
                self._format_mapping_items(
                    self_reports[-6:],
                    keys=("topic", "level", "confidence", "evidence", "note", "created_at"),
                )
            )

        if not lines:
            title = str(getattr(task, "title", "") or getattr(session, "subject_scope", "") or "session").strip()
            return f"- Only basic session metadata was available for {title}."
        return "\n".join(lines)

    def _run_assessment(
        self,
        runtime,
        session,
        task,
        domain: str,
        selected_skills: list[str],
        template,
        blueprint,
        duration_min: int,
        evidence_block: str,
    ) -> Optional[AssessmentResult]:
        """Run the LLM assessment with escalation protocol.

        Per D-09: Start with Flash Lite, escalate to Flash if ambiguity detected.
        Per D-07: Agent is strict and frank.
        Per D-08: Biased toward expeditious completion.
        Per D-12: Instruct LLM to cluster novel sub-skills under existing taxonomy.
        """
        skills_list = ", ".join(selected_skills)
        taxonomy_list = ", ".join(template.sub_skill_taxonomy)
        blueprint_block = ""
        if blueprint is not None:
            blueprint_block = (
                f"\nSKILL KIND: {blueprint.skill_kind.value}\n"
                f"PRIMARY FRAME: {blueprint.primary_frame.value}\n"
                f"SECONDARY FRAMES: {', '.join(item.value for item in blueprint.secondary_frames) or 'none'}\n"
                f"EVIDENCE CONTRACTS: {', '.join(item.value for item in blueprint.evidence_contract) or 'none'}\n"
                f"FEEDBACK SOURCES: {', '.join(item.value for item in blueprint.feedback_sources) or 'none'}\n"
                f"STOP CONDITION: {blueprint.stop_condition}\n"
                f"COACH RULES: {' | '.join(blueprint.coach_rules) or 'none'}\n"
            )

        assessment_prompt = f"""You are a strict, honest learning assessment agent. You must be HARSH rather than lenient -- better to underestimate than overestimate the learner's level. If there is any doubt, score LOW.

DOMAIN: {domain}
TASK: {task.title}
DURATION: {duration_min} minutes
ASSESSMENT TARGETS INFERRED FROM EVIDENCE: {skills_list}
BASE TAXONOMY: {taxonomy_list}
{blueprint_block}
EVIDENCE SIGNALS:
{evidence_block}

For each sub-skill listed above, assign a score from 1-5:
  1 = No meaningful progress, fundamental gaps remain
  2 = Attempted but significant weaknesses evident
  3 = Adequate understanding but not reliable under pressure
  4 = Solid grasp, minor gaps only
  5 = Strong proficiency demonstrated clearly

SCORING BIAS: Default to scores 2-3 unless strong evidence of proficiency. A score of 4+ should be rare and reserved for clearly demonstrated mastery.

SUB-SKILL CLUSTERING (D-12): You MUST use sub-skill names from the BASE TAXONOMY listed above whenever possible. If you identify a genuinely novel sub-skill not covered by ANY existing taxonomy category, you may add it but you MUST set its `cluster_parent` field to the most relevant base taxonomy category. Only leave `cluster_parent` empty if no existing category is even remotely applicable. Prefer using existing taxonomy names over inventing new ones.

Also provide:
- A brief, frank critique of the session (2-3 sentences, be direct and honest)
- A list of specific retry items for any weak areas (score <= {WEAK_THRESHOLD})

If you detect strong proficiency across all skills, you may assign high scores and keep the critique brief (per expeditious completion bias).

If this assessment requires deeper domain reasoning than you can handle, respond with exactly: ESCALATE: flash

Return your assessment as valid JSON matching the schema provided."""

        try:
            from pb.llm.runtime import DraftGenerationError

            roles = getattr(runtime.config, "model_roles", None)
            fast_binding = str(getattr(roles, "fast_inference", "") or "").strip()
            default_binding = str(getattr(roles, "default", "") or "").strip()
            primary_model = fast_binding or f"gemini:{FLASH_LITE_MODEL}"
            escalation_model = default_binding or f"gemini:{FLASH_MODEL}"

            # Attempt with Flash Lite first
            try:
                draft = runtime.generate_draft(
                    schema_cls=AssessmentResult,
                    prompt=assessment_prompt,
                    source_scope="finish_assessment",
                    model=primary_model,
                    timeout=30,
                )
            except DraftGenerationError as e:
                logger.warning("finish_assessment.llm_failed", error=str(e))
                return None

            if draft and draft.payload:
                # Check for escalation signal in raw response
                raw = getattr(draft, "raw_response", "") or ""
                escalate_to = self._parse_escalation(raw)
                if escalate_to:
                    logger.debug("finish_assessment.escalating", to_model=escalate_to)
                    self._model_used = escalation_model
                    try:
                        draft = runtime.generate_draft(
                            schema_cls=AssessmentResult,
                            prompt=assessment_prompt,
                            source_scope="finish_assessment",
                            model=escalation_model,
                            timeout=45,
                        )
                    except DraftGenerationError as e:
                        logger.warning("finish_assessment.escalation_failed", error=str(e))
                        return None

                if draft and draft.payload:
                    # Extract model name from "provider:model" format
                    raw_model = getattr(draft, "model", "") or ""
                    self._model_used = raw_model.split(":", 1)[-1] if ":" in raw_model else raw_model or self._model_used
                    return draft.payload  # type: ignore[return-value]

            return None

        except Exception as e:
            logger.warning("finish_assessment.llm_failed", error=str(e))
            return None

    @staticmethod
    def _parse_escalation(text: str) -> Optional[str]:
        """Parse ESCALATE: flash/pro from LLM response.

        Same protocol as BrainEngine._parse_escalation in brain.py.
        Only flash escalation is supported for assessment (not pro).
        """
        m = _ESCALATE_RE.match(text.strip())
        if not m:
            return None
        target = m.group(1).lower()
        return FLASH_MODEL if target == "flash" else None  # only flash escalation for assessment
