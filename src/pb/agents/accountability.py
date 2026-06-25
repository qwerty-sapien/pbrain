# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Accountability agent — creates durable commitment records (Phase 10).

Persists commitments to SQLite, enforces spam guard (>5/session),
optionally surfaces avoidance when goals/commitments context is provided.

Follows D-03: commitments are passive + contextual; no unsolicited nudging.
Follows D-04: avoidance detection only surfaces in pb next / pb review context.
"""

from __future__ import annotations

from typing import Optional

import structlog
from pydantic import BaseModel, Field

from pb.agents import register_agent
from pb.agents.base import AgentHandler
from pb.core.agent_instruction_judge import agent_instruction_suffix
from pb.core.dispatch_models import AgentOutput, DispatchSession
from pb.core.models import generate_internal_id, utc_now
from pb.llm.structured import structured_output_call
from pb.storage.database import get_connection

logger = structlog.get_logger()

_COMMITMENT_SPAM_THRESHOLD = 5


class CommitmentExtraction(BaseModel):
    """LLM extraction of commitment details from user intent."""

    description: str = Field(description="What the user is committing to")
    due_date: Optional[str] = Field(
        default=None,
        description="ISO date YYYY-MM-DD if mentioned, otherwise null",
    )
    is_commitment: bool = Field(
        description="True if intent is genuinely an accountability request"
    )
    avoidance_note: str = Field(
        default="",
        description=(
            "If context shows the user is avoiding an existing commitment, "
            "describe which one and how many days since last touched. Empty string if none."
        ),
    )


_EXTRACTION_SYSTEM_PROMPT = """\
You are a commitment extraction assistant. Analyse the user's intent and determine:
1. Whether it is a genuine accountability/commitment request (not just a todo or general chat).
2. What specific commitment they are making, cleaned of filler words.
3. If they mention a deadline, extract it as YYYY-MM-DD.
4. If the provided context contains existing commitments or goals that the user appears to be
   avoiding or neglecting (e.g. not mentioned, repeatedly deferred), note the most relevant one.
   Pay special attention to items with approaching deadlines — a commitment due within 3 days
   that the user is ignoring is more urgent than one due in a month.
   Only flag avoidance when context data is provided — do NOT fabricate neglect signals.

Accountability requests include: "keep me accountable for X", "I commit to Y", "hold me to Z",
"remind me to do A by date", "make sure I finish B". Pure todos ("buy milk") are NOT commitments.
"""


class AccountabilityAgent(AgentHandler):
    """Accountability specialised agent.

    Creates durable commitment records in SQLite (ACCT-01).
    Detects and notes avoidance patterns when context is available (ACCT-02).
    Kicks back non-accountability requests to the dispatcher (in_scope=False).
    """

    agent_id: str = "accountability"
    display_name: str = "Accountability"
    model_tier: str = "mid"

    async def handle(
        self,
        intent: str,
        session: DispatchSession,
        *,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """Extract commitment from intent, persist to SQLite, surface avoidance if relevant.

        Args:
            intent: Raw user intent string.
            session: Current dispatch session record.
            context: Optional dict. May include "goals" and "commitments" for
                     avoidance detection; keys match ReviewAgent convention.

        Returns:
            AgentOutput. in_scope=False if not an accountability request;
            in_scope=True with persisted commitment and optional avoidance note otherwise.
        """
        ctx = context or {}
        extraction = await self._extract_commitment(intent, ctx)

        if extraction is None:
            # LLM failed — try to proceed with a minimal commitment
            logger.warning("accountability.extraction_failed", session_id=session.id)
            return await self._persist_and_respond(
                session_id=session.id,
                description=intent.strip(),
                due_date=None,
                avoidance_note="",
            )

        if not extraction.is_commitment:
            logger.info(
                "accountability.not_a_commitment",
                session_id=session.id,
                intent=intent[:80],
            )
            return AgentOutput(
                in_scope=False,
                scope_reason="Not an accountability request",
                response="",
            )

        return await self._persist_and_respond(
            session_id=session.id,
            description=extraction.description,
            due_date=extraction.due_date,
            avoidance_note=extraction.avoidance_note,
        )

    async def _extract_commitment(
        self, intent: str, ctx: dict
    ) -> Optional[CommitmentExtraction]:
        """Call mid-tier LLM to extract structured commitment data."""
        context_block = ""
        goals = ctx.get("goals", [])
        commitments = ctx.get("commitments", [])
        deadline_todos = ctx.get("deadline_todos", [])
        if goals or commitments or deadline_todos:
            goals_summary = ", ".join(
                getattr(g, "title", str(g)) for g in goals[:5]
            )
            commitments_summary = "; ".join(
                (c.get("description", "") if isinstance(c, dict) else str(c))
                for c in commitments[:5]
            )
            deadline_lines = []
            for t in deadline_todos[:5]:
                due = getattr(t, "due_date", None)
                due_str = due.strftime("%Y-%m-%d") if due else "no deadline"
                deadline_lines.append(f"{getattr(t, 'title', '?')} (due {due_str})")
            deadline_summary = "; ".join(deadline_lines) if deadline_lines else "none"

            context_block = (
                f"\n\nContext (use only for avoidance detection):\n"
                f"Active goals: {goals_summary or 'none'}\n"
                f"Existing commitments: {commitments_summary or 'none'}\n"
                f"Deadline-approaching tasks: {deadline_summary}"
            )

        prompt = (
            f"User intent: {intent!r}{context_block}\n\n"
            f"Extract commitment details."
        )
        try:
            return await structured_output_call(
                prompt,
                CommitmentExtraction,
                system_prompt=_EXTRACTION_SYSTEM_PROMPT
                + agent_instruction_suffix(self.agent_id),
                tier="mid",
            )
        except Exception as exc:
            logger.warning("accountability.llm_error", error=str(exc))
            return None

    async def _persist_and_respond(
        self,
        *,
        session_id: str,
        description: str,
        due_date: Optional[str],
        avoidance_note: str,
    ) -> AgentOutput:
        """Insert commitment into SQLite and build response."""
        commitment_id = generate_internal_id()
        created_at = utc_now().isoformat()

        try:
            with get_connection() as conn:
                # T-10-09: fully parameterised INSERT, no string interpolation
                conn.execute(
                    "INSERT INTO commitments "
                    "(id, description, created_at, due_date, status, session_id) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (commitment_id, description, created_at, due_date, "active", session_id),
                )
                conn.commit()

                # Commitment spam guard (T-10-09)
                row = conn.execute(
                    "SELECT COUNT(*) FROM commitments WHERE session_id = ?",
                    (session_id,),
                ).fetchone()
                session_count: int = row[0] if row else 0

        except Exception as exc:
            logger.warning(
                "accountability.persist_error",
                error=str(exc),
                session_id=session_id,
            )
            return AgentOutput(
                in_scope=True,
                response=f"I tried to record your commitment ({description}) but hit a storage error. Please try again.",
                status="complete",
            )

        logger.info(
            "accountability.commitment_created",
            commitment_id=commitment_id,
            session_id=session_id,
            due_date=due_date,
        )

        due_suffix = f" by {due_date}" if due_date else ""
        response = (
            f"Committed: {description}{due_suffix}. "
            f"I'll surface this in your next review and recommendations."
        )

        if session_count > _COMMITMENT_SPAM_THRESHOLD:
            response += (
                f"\n\nYou've created {_COMMITMENT_SPAM_THRESHOLD}+ commitments this session. "
                f"Are these all real commitments?"
            )

        # D-04: avoidance note only surfaces when caller explicitly passes context
        # (i.e. called from pb next / pb review context, not from a cold do "...")
        if avoidance_note:
            response += f"\n\nNote: {avoidance_note}"

        return AgentOutput(
            in_scope=True,
            response=response,
            status="complete",
        )


# Register at module level so importing this module side-effects the registry
register_agent("accountability", AccountabilityAgent())
