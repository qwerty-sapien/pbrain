# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review agent — LLM-augmented synthesis of sessions, goals, and commitments (Phase 10).

Synthesizes daily or weekly review output that references specific goal progress,
surfaces commitments, and identifies drift/neglect patterns (ACCT-02, ALIGN-02).

Avoidance detection surfaces HERE (D-04) — not as unsolicited nudges elsewhere.
"""

from __future__ import annotations

from typing import Optional

import structlog
from pydantic import BaseModel, Field

from pb.agents import register_agent
from pb.agents.base import AgentHandler
from pb.core.agent_instruction_judge import agent_instruction_suffix
from pb.core.dispatch_models import AgentOutput, DispatchSession
from pb.llm.structured import structured_output_call

logger = structlog.get_logger()


class ReviewSynthesis(BaseModel):
    """LLM-produced review synthesis payload."""

    summary: str = Field(description="A concise summary of the review period")
    goal_progress: str = Field(
        description="Reference to specific goal progress (name goals by name)"
    )
    commitments_status: str = Field(
        description="Status of commitments made; flag any that are overdue or neglected"
    )
    avoidance_flags: str = Field(
        default="",
        description=(
            "Goals or commitments not touched during this period. "
            "Empty string if nothing flagged."
        ),
    )
    next_actions: str = Field(
        description="1-3 actionable next steps produced by the review"
    )


_DAILY_SYSTEM_PROMPT = """\
You are a focused daily review assistant. Your job is to synthesize the user's day:
1. Reference specific goals by name when describing progress.
2. Surface any commitments made today and their status (done / in progress / not started).
3. Flag any active goals that were completely neglected today (per ACCT-02) — name them specifically.
4. Capture thoughts or notes that imply drift from stated goals (ALIGN-02).
5. Produce 1-3 concrete next actions — not a vague list.
6. Tone: direct, honest, not encouraging-for-the-sake-of-it.
Avoidance detection is LEGITIMATE here (D-04). Surface it clearly but without escalating pressure.
"""

_WEEKLY_SYSTEM_PROMPT = """\
You are a focused weekly review assistant. Your job is to synthesize the user's week:
1. Reference specific goals by name when describing progress or lack of progress.
2. Surface all commitments made this week and their status.
3. Flag any active goals neglected for the full week (ACCT-02) — name them.
4. Identify recurring patterns (skipped domain, repeated avoidance, drift from priorities).
5. Produce 1-3 actionable next steps for the coming week.
6. Tone: direct, strategic, not reassuring-by-default.
Avoidance detection is LEGITIMATE here (D-04). Surface it clearly.
"""


class ReviewAgent(AgentHandler):
    """Review specialised agent.

    Synthesizes sessions, goals, and commitments into actionable review output.
    LLM augmentation via mid-tier Flash; deterministic fallback via build_next_candidates.
    """

    agent_id: str = "review"
    display_name: str = "Review"
    model_tier: str = "mid"

    async def handle(
        self,
        intent: str,
        session: DispatchSession,
        *,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """Synthesize review output from loaded context.

        Args:
            intent: Raw user intent string (used to detect scope: day/week).
            session: Current dispatch session.
            context: Dict with optional keys:
                - "goals": list of GoalArc objects
                - "commitments": list of commitment dicts or objects
                - "sessions": list of session dicts from the review window
                - "repo": Repository (used for deterministic fallback)

        Returns:
            AgentOutput with synthesized review text, status=complete.
        """
        ctx = context or {}
        scope = self._detect_scope(intent)
        base_prompt = _WEEKLY_SYSTEM_PROMPT if scope == "week" else _DAILY_SYSTEM_PROMPT
        system_prompt = base_prompt + agent_instruction_suffix(self.agent_id)

        goals = ctx.get("goals", [])
        commitments = ctx.get("commitments", [])
        sessions = ctx.get("sessions", [])
        repo = ctx.get("repo")

        synthesis_prompt = self._build_prompt(scope, goals, commitments, sessions)

        result = await structured_output_call(
            synthesis_prompt,
            ReviewSynthesis,
            system_prompt=system_prompt,
            tier="mid",
        )

        if result is None:
            logger.warning("review.llm_failed", scope=scope, session_id=session.id)
            return self._deterministic_fallback(repo, scope)

        response = self._format_response(result, scope)
        logger.info("review.synthesis_complete", scope=scope, session_id=session.id)

        return AgentOutput(
            in_scope=True,
            response=response,
            status="complete",
        )

    def _detect_scope(self, intent: str) -> str:
        """Detect review scope from intent text: 'week' or 'day'."""
        lower = intent.lower()
        if any(w in lower for w in ("week", "weekly", "7 day", "seven day")):
            return "week"
        return "day"

    def _build_prompt(
        self,
        scope: str,
        goals: list,
        commitments: list,
        sessions: list,
    ) -> str:
        """Build the synthesis prompt from loaded context."""
        period = "this week" if scope == "week" else "today"

        goal_lines = "\n".join(
            f"- {getattr(g, 'title', str(g))} [domain: {getattr(g, 'domain', '')}]"
            for g in goals[:10]
        ) or "None"

        commitment_lines = "\n".join(
            (
                f"- {c.get('description', '')} (status: {c.get('status', 'active')}, "
                f"due: {c.get('due_date', 'none')})"
                if isinstance(c, dict)
                else f"- {c}"
            )
            for c in commitments[:10]
        ) or "None"

        session_lines = "\n".join(
            (
                f"- {s.get('branch', 'focus')} session on "
                f"{s.get('subject_scope', s.get('task_id', 'unknown'))} "
                f"[goal: {s.get('goal_id', 'none')}]"
                if isinstance(s, dict)
                else f"- {s}"
            )
            for s in sessions[:20]
        ) or "None recorded"

        return (
            f"Review scope: {scope} ({period})\n\n"
            f"Active goals:\n{goal_lines}\n\n"
            f"Commitments:\n{commitment_lines}\n\n"
            f"Sessions {period}:\n{session_lines}\n\n"
            f"Synthesize an actionable {scope}ly review."
        )

    def _format_response(self, result: ReviewSynthesis, scope: str) -> str:
        """Format the LLM synthesis into readable review output."""
        period = "Weekly" if scope == "week" else "Daily"
        parts = [f"## {period} Review\n"]

        if result.summary:
            parts.append(result.summary)

        if result.goal_progress:
            parts.append(f"\n**Goal progress:** {result.goal_progress}")

        if result.commitments_status:
            parts.append(f"\n**Commitments:** {result.commitments_status}")

        # D-04: avoidance detection is legitimate in review context
        if result.avoidance_flags:
            parts.append(f"\n**Flagged:** {result.avoidance_flags}")

        if result.next_actions:
            parts.append(f"\n**Next actions:**\n{result.next_actions}")

        return "\n".join(parts)

    def _deterministic_fallback(self, repo, scope: str) -> AgentOutput:
        """Fallback when LLM fails: use build_next_candidates for basic output."""
        period = "Weekly" if scope == "week" else "Daily"
        if repo is None:
            return AgentOutput(
                in_scope=True,
                response=f"## {period} Review\n\nCould not synthesize review — no context available.",
                status="complete",
            )
        try:
            from pb.core.action_routing import build_next_candidates

            candidates = build_next_candidates(repo, limit=5)
            lines = [f"## {period} Review (basic)\n"]
            lines.append("Top next actions:")
            for c in candidates[:5]:
                lines.append(f"- {c.human_label or c.command}: {c.short_reason}")
            return AgentOutput(
                in_scope=True,
                response="\n".join(lines),
                status="complete",
            )
        except Exception as exc:
            logger.warning("review.fallback_error", error=str(exc))
            return AgentOutput(
                in_scope=True,
                response=f"## {period} Review\n\nCould not generate review at this time.",
                status="complete",
            )


# Register at module level so importing this module side-effects the registry
register_agent("review", ReviewAgent())
