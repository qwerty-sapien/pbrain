# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Agent spawner — creates persistent domain agents (Phase 10).

Implements D-07 (goal + 2-interaction gate), D-08 (goal-setting route),
D-09 (batched questions capped at 3), and the agent sprawl guard (T-10-10).
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
from pb.core.scope_resolution import match_goal, match_track
from pb.llm.structured import structured_output_call
from pb.storage.database import get_connection

logger = structlog.get_logger()

_INTERACTION_GATE = 2
_SPRAWL_THRESHOLD = 10


class DomainExtraction(BaseModel):
    """LLM extraction of domain keyword from a user intent."""

    domain: str = Field(description="The primary domain or subject area being discussed")


class SpawnerClarifications(BaseModel):
    """Batched clarification questions (max 3) before agent creation."""

    questions: list[str] = Field(
        description="Up to 3 clarifying questions needed to configure the domain agent well"
    )


_DOMAIN_EXTRACTION_PROMPT = (
    "Extract the primary domain or subject area from the user's intent. "
    "For 'practise German Akkusativ' → 'german'. "
    "For 'help me with calculus integrals' → 'calculus'. "
    "For 'startup sales strategy' → 'sales'. "
    "Return a single lowercase domain keyword."
)


class SpawnerAgent(AgentHandler):
    """Agent spawner — mints persistent domain agents on first substantive engagement.

    Gate sequence (D-07/D-08):
      1. No goal for domain → route to goal-setting (D-08).
      2. Goal exists but interaction_count < 2 → accumulate, encourage (D-07).
      3. Goal + 2 interactions → create agent.

    Batched questions capped at 3 (D-09).
    Warns at 10+ existing agents (T-10-10 agent sprawl guard).
    """

    agent_id: str = "spawner"
    display_name: str = "Agent Spawner"
    model_tier: str = "mid"

    async def handle(
        self,
        intent: str,
        session: DispatchSession,
        *,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """Gate and create persistent domain agents.

        Args:
            intent: Raw user intent string.
            session: Current dispatch session.
            context: Optional dict with "repo" (Repository).

        Returns:
            AgentOutput describing progress toward agent creation.
        """
        ctx = context or {}
        repo = ctx.get("repo")

        # --- Step 1: resolve domain ---
        domain = await self._resolve_domain(intent, repo)
        if not domain:
            return AgentOutput(
                in_scope=True,
                response="I couldn't identify a clear domain for your request. Could you be more specific?",
                status="active",
            )

        # --- Step 2: check goal existence (D-08) ---
        matched_goal = match_goal(repo, domain) if repo else None
        if matched_goal is None and repo is not None:
            # Try matching by intent directly
            matched_goal = match_goal(repo, intent)

        if matched_goal is None:
            logger.info("spawner.no_goal_found", domain=domain)
            return AgentOutput(
                in_scope=True,
                status="active",
                response=(
                    f"I don't see a goal for {domain} yet. Let's set one up first."
                ),
                options=[
                    f"Create a goal for {domain}",
                    f"Skip goal and just capture this",
                ],
            )

        # --- Step 3: D-07 interaction gating ---
        row = self._get_agent_row(domain)
        if row is None:
            # First time seeing this domain — insert a tracker row
            self._upsert_agent_row(domain, matched_goal.id, interaction_count=1)
            logger.info(
                "spawner.first_interaction",
                domain=domain,
                goal_id=matched_goal.id,
            )
            return AgentOutput(
                in_scope=True,
                response=(
                    f"Got it. I'm learning about your {domain} needs. "
                    f"(1/{_INTERACTION_GATE} interactions before I create a dedicated agent.)"
                ),
                status="complete",
            )

        current_count: int = row["interaction_count"]
        new_count = current_count + 1
        self._update_interaction_count(domain, new_count)

        if new_count < _INTERACTION_GATE:
            logger.info(
                "spawner.accumulating_interactions",
                domain=domain,
                count=new_count,
            )
            return AgentOutput(
                in_scope=True,
                response=(
                    f"Got it. I'm learning about your {domain} needs. "
                    f"({new_count}/{_INTERACTION_GATE} interactions before I create a dedicated agent.)"
                ),
                status="complete",
            )

        # --- Step 4: create the persistent agent ---
        return await self._create_agent(domain, matched_goal, session, ctx)

    async def _resolve_domain(self, intent: str, repo) -> str:
        """Try scope_resolution first; fall back to lite LLM extraction."""
        # Try match_goal / match_track for existing-goal domains
        if repo is not None:
            matched_goal = match_goal(repo, intent)
            if matched_goal and getattr(matched_goal, "domain", ""):
                return matched_goal.domain.lower().strip()
            matched_track = match_track(repo, intent)
            if matched_track and getattr(matched_track, "name", ""):
                return matched_track.name.lower().strip()

        # LLM fallback
        try:
            result = await structured_output_call(
                f"{_DOMAIN_EXTRACTION_PROMPT}"
                f"{agent_instruction_suffix(self.agent_id)}\n\nIntent: {intent!r}",
                DomainExtraction,
                tier="lite",
            )
            return result.domain.lower().strip() if result else ""
        except Exception as exc:
            logger.warning("spawner.domain_extraction_error", error=str(exc))
            return ""

    def _get_agent_row(self, domain: str):
        """Query dispatch_agents for an existing domain row."""
        try:
            with get_connection() as conn:
                return conn.execute(
                    "SELECT id, interaction_count, goal_id FROM dispatch_agents WHERE domain = ?",
                    (domain,),
                ).fetchone()
        except Exception as exc:
            logger.warning("spawner.db_read_error", error=str(exc))
            return None

    def _upsert_agent_row(self, domain: str, goal_id: str, interaction_count: int) -> None:
        """Insert a new tracker row for the domain (pre-creation phase)."""
        row_id = generate_internal_id()
        created_at = utc_now().isoformat()
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT OR IGNORE INTO dispatch_agents "
                    "(id, domain, goal_id, config_json, created_at, interaction_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (row_id, domain, goal_id, "{}", created_at, interaction_count),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("spawner.db_upsert_error", error=str(exc))

    def _update_interaction_count(self, domain: str, new_count: int) -> None:
        """Increment interaction_count for the domain."""
        try:
            with get_connection() as conn:
                conn.execute(
                    "UPDATE dispatch_agents SET interaction_count = ? WHERE domain = ?",
                    (new_count, domain),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("spawner.db_update_error", error=str(exc))

    async def _create_agent(self, domain: str, matched_goal, session: DispatchSession, ctx: dict) -> AgentOutput:
        """Create a persistent domain agent with optional D-09 clarification questions."""
        # T-10-10: agent sprawl guard
        sprawl_warning = ""
        try:
            with get_connection() as conn:
                count_row = conn.execute(
                    "SELECT COUNT(*) FROM dispatch_agents WHERE domain NOT LIKE 'builtin:%'",
                ).fetchone()
                total_agents: int = count_row[0] if count_row else 0
        except Exception:
            total_agents = 0

        if total_agents >= _SPRAWL_THRESHOLD:
            sprawl_warning = (
                f"\n\nYou have {_SPRAWL_THRESHOLD}+ agents. "
                f"Consider reviewing via `pb config agents`."
            )

        # D-09: batched clarifying questions (up to 3)
        questions = await self._get_clarifying_questions(domain, matched_goal, ctx)
        if questions:
            return AgentOutput(
                in_scope=True,
                status="active",
                response=(
                    f"I'm ready to create your {domain} agent linked to '{matched_goal.title}'. "
                    f"A few quick questions to configure it well:{sprawl_warning}"
                ),
                options=questions[:3],  # D-09: cap at 3
            )

        # Insert full agent record
        agent_id = generate_internal_id()
        created_at = utc_now().isoformat()
        try:
            with get_connection() as conn:
                conn.execute(
                    "INSERT INTO dispatch_agents "
                    "(id, domain, goal_id, config_json, created_at, interaction_count) "
                    "VALUES (?, ?, ?, ?, ?, ?)"
                    " ON CONFLICT(id) DO NOTHING",
                    (agent_id, domain, matched_goal.id, "{}", created_at, _INTERACTION_GATE),
                )
                # Update the pre-existing tracker row if it exists
                conn.execute(
                    "UPDATE dispatch_agents SET goal_id = ?, interaction_count = ? WHERE domain = ?",
                    (matched_goal.id, _INTERACTION_GATE, domain),
                )
                conn.commit()
        except Exception as exc:
            logger.warning("spawner.agent_create_error", error=str(exc))
            return AgentOutput(
                in_scope=True,
                response=f"Tried to create your {domain} agent but hit a storage error. Please try again.",
                status="complete",
            )

        logger.info(
            "spawner.agent_created",
            domain=domain,
            goal_id=matched_goal.id,
            agent_id=agent_id,
        )

        return AgentOutput(
            in_scope=True,
            response=(
                f"Created your {domain} agent, linked to goal: {matched_goal.title}.{sprawl_warning}"
            ),
            status="complete",
        )

    async def _get_clarifying_questions(self, domain: str, matched_goal, ctx: dict) -> list[str]:
        """Use mid-tier LLM to generate up to 3 clarifying questions (D-09)."""
        goals_summary = getattr(matched_goal, "title", "")
        desc = getattr(matched_goal, "description", "") or ""
        prompt = (
            f"You are configuring a new persistent agent for the domain '{domain}' "
            f"linked to the goal '{goals_summary}'. "
            f"Goal description: {desc!r}\n\n"
            f"Generate up to 3 targeted clarifying questions that would help configure "
            f"this agent well. Focus on gaps in context, not open-ended generalities. "
            f"Return an empty list if the existing context is sufficient."
        )
        try:
            result = await structured_output_call(
                prompt,
                SpawnerClarifications,
                system_prompt=agent_instruction_suffix(self.agent_id).strip(),
                tier="mid",
            )
            return result.questions[:3] if result else []
        except Exception as exc:
            logger.warning("spawner.clarifications_error", error=str(exc))
            return []


# Register at module level so importing this module side-effects the registry
register_agent("spawner", SpawnerAgent())
