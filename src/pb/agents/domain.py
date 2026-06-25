# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain agent — auto-minted template for persistent domain specialists (Phase 10).

Each DomainAgent instance is dynamically created and registered by the spawner
or dispatcher for a specific domain + goal combination.

Implements:
- D-01: Goal link shown once at dispatch (Linked to: <goal_title>).
- ALIGN-01: Domain requests connected to higher-level goals.
- ALIGN-03: Vague goal refinement through clarifying questions.
- T-10-11: scope_reason is internal only, never in InteractionEnvelope.
- T-10-12: 429 on scope check → UNDETERMINED (in_scope=True), never False.
"""

from __future__ import annotations

from typing import Optional

import structlog
from pydantic import BaseModel, Field

from pb.agents import register_agent
from pb.agents.base import AgentHandler
from pb.core.dispatch_models import AgentOutput, DispatchSession
from pb.core.agent_instruction_judge import agent_instruction_suffix
from pb.llm.structured import structured_output_call

logger = structlog.get_logger()


class ScopeCheck(BaseModel):
    """LLM scope classification for this domain agent."""

    in_scope: bool = Field(
        description="True if the user's intent is related to this domain agent's domain"
    )
    reason: str = Field(
        default="",
        description="Internal reason for scope decision (never shown to user)",
    )
    is_vague_goal: bool = Field(
        default=False,
        description=(
            "True if the intent expresses a vague goal (e.g. 'get better at X') "
            "rather than a concrete request that can be acted on immediately"
        ),
    )


class DomainResponse(BaseModel):
    """LLM-generated domain-specific response."""

    response: str = Field(description="The agent's natural-language response to the user")
    options: list[str] = Field(
        default_factory=list,
        description="Follow-up options or clarification questions if needed",
    )
    status: str = Field(
        default="complete",
        description="active | complete — active if session should continue",
    )


_VAGUE_GOAL_SYSTEM_PROMPT = """\
The user has expressed a vague goal rather than a concrete request.
Your job is to ask targeted questions that sharpen the goal into something measurable and actionable.
Ask at most 3 questions. Focus on: current level, specific target, timeline, measurable outcome.
Do NOT accept a vague goal as-is. Do NOT generate a task list from a vague goal.
Return a DomainResponse with status=active and questions as options.
"""


class DomainAgent(AgentHandler):
    """Auto-minted domain specialist agent.

    Instantiated per domain+goal combination by create_domain_agent().
    Not registered at module load — only registered when create_domain_agent() is called.
    """

    def __init__(
        self,
        agent_id: str,
        display_name: str,
        domain: str,
        goal_id: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.display_name = display_name
        self.domain = domain
        self.goal_id = goal_id
        self.model_tier = "mid"
        # Track whether goal link has been shown for this session (D-01)
        self._goal_shown: set[str] = set()

    async def handle(
        self,
        intent: str,
        session: DispatchSession,
        *,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """Process user intent within this domain context.

        Args:
            intent: Raw user intent string.
            session: Current dispatch session.
            context: Optional dict with "repo", "goals", "sessions", "notes".

        Returns:
            AgentOutput. in_scope=False if intent doesn't relate to this domain.
            in_scope=True (UNDETERMINED) on 429 — never False on rate-limit (T-10-12).
        """
        ctx = context or {}

        # --- Scope check (T-10-12: 429 → UNDETERMINED, never False) ---
        try:
            scope_result = await self._check_scope(intent)
        except Exception as exc:
            # Any error including 429 → UNDETERMINED (in_scope=True fallback)
            logger.warning(
                "domain.scope_check_error",
                agent_id=self.agent_id,
                error=str(exc),
            )
            scope_result = ScopeCheck(in_scope=True, reason="UNDETERMINED")

        if scope_result is None:
            # structured_output_call returned None (429 or failure) → UNDETERMINED
            logger.warning(
                "domain.scope_check_undetermined",
                agent_id=self.agent_id,
                intent=intent[:80],
            )
            scope_result = ScopeCheck(in_scope=True, reason="UNDETERMINED")

        if not scope_result.in_scope:
            # T-10-11: scope_reason is internal, never shown to user
            logger.info(
                "domain.out_of_scope",
                agent_id=self.agent_id,
                reason=scope_result.reason,
            )
            return AgentOutput(
                in_scope=False,
                scope_reason=scope_result.reason,
                response="",
            )

        # --- ALIGN-03: vague goal refinement ---
        if scope_result.is_vague_goal:
            return await self._refine_vague_goal(intent, session, ctx)

        # --- Goal link (D-01 / ALIGN-01): shown once per session ---
        goal_prefix = self._goal_prefix(session.id, ctx)

        # --- Domain-specific response ---
        response_output = await self._domain_response(intent, ctx)

        if response_output is None:
            return AgentOutput(
                in_scope=True,
                response=f"{goal_prefix}I'm ready to help with {self.domain}, but couldn't generate a response. Please try again.",
                status="complete",
            )

        final_response = (
            f"{goal_prefix}{response_output.response}" if goal_prefix else response_output.response
        )

        return AgentOutput(
            in_scope=True,
            response=final_response,
            options=response_output.options,
            status=response_output.status,
        )

    async def _check_scope(self, intent: str) -> Optional[ScopeCheck]:
        """Lite-tier LLM scope check. Returns None on 429 (UNDETERMINED)."""
        prompt = (
            f"Domain agent domain: {self.domain!r}\n"
            f"User intent: {intent!r}\n\n"
            f"Is this intent related to the domain '{self.domain}'? "
            f"Also check if it is a vague goal (e.g. 'get better at X', 'improve my Y')."
        )
        return await structured_output_call(prompt, ScopeCheck, tier="lite")

    def _goal_prefix(self, session_id: str, ctx: dict) -> str:
        """Return goal link prefix if not yet shown this session (D-01)."""
        if session_id in self._goal_shown:
            return ""
        if not self.goal_id:
            return ""
        # Try to look up goal title from context
        goals = ctx.get("goals", [])
        goal_title = ""
        for g in goals:
            if getattr(g, "id", None) == self.goal_id:
                goal_title = getattr(g, "title", "")
                break
        if not goal_title:
            # Try repo
            repo = ctx.get("repo")
            if repo is not None:
                try:
                    for g in repo.list_goal_arcs(status=None):
                        if getattr(g, "id", "") == self.goal_id:
                            goal_title = getattr(g, "title", "")
                            break
                except Exception:
                    pass

        if goal_title:
            self._goal_shown.add(session_id)
            return f"Linked to: {goal_title}\n\n"
        return ""

    async def _refine_vague_goal(
        self, intent: str, session: DispatchSession, ctx: dict
    ) -> AgentOutput:
        """ALIGN-03: ask sharpening questions instead of accepting vague goal."""
        prompt = (
            f"The user expressed a vague goal about {self.domain!r}: {intent!r}\n\n"
            f"Generate up to 3 targeted questions that will sharpen this into "
            f"a measurable, actionable goal. Do not accept it as-is."
        )
        result = await structured_output_call(
            prompt,
            DomainResponse,
            system_prompt=_VAGUE_GOAL_SYSTEM_PROMPT,
            tier="mid",
        )
        if result is None:
            return AgentOutput(
                in_scope=True,
                response=(
                    f"I'd like to help you with {self.domain}, but I need a bit more clarity. "
                    f"What specifically are you trying to achieve?"
                ),
                options=[
                    "What's your current level?",
                    "What does success look like?",
                    "What's your timeline?",
                ],
                status="active",
            )
        return AgentOutput(
            in_scope=True,
            response=result.response,
            options=result.options[:3],  # cap refinement options at 3
            status="active",
        )

    async def _domain_response(self, intent: str, ctx: dict) -> Optional[DomainResponse]:
        """Mid-tier LLM domain-specific response."""
        goals = ctx.get("goals", [])
        sessions = ctx.get("sessions", [])
        notes = ctx.get("notes", [])

        goal_context = ""
        if goals:
            goal_context = "Relevant goals: " + ", ".join(
                getattr(g, "title", str(g)) for g in goals[:3]
            )

        session_context = ""
        if sessions:
            recent = sessions[-3:]
            session_context = "Recent sessions: " + "; ".join(
                (s.get("subject_scope", "") or s.get("task_id", "") if isinstance(s, dict) else str(s))
                for s in recent
            )

        system_prompt = (
            f"You are a specialised assistant for the domain '{self.domain}'. "
            f"Respond helpfully and concretely to the user's request. "
            f"Reference the user's goals and prior sessions when relevant. "
            f"If the request is clear, execute it — don't ask unnecessary questions."
            f"{agent_instruction_suffix(self.agent_id)}"
        )

        prompt_parts = [f"User request: {intent!r}"]
        if goal_context:
            prompt_parts.append(goal_context)
        if session_context:
            prompt_parts.append(session_context)

        prompt = "\n".join(prompt_parts)
        try:
            return await structured_output_call(
                prompt,
                DomainResponse,
                system_prompt=system_prompt,
                tier="mid",
            )
        except Exception as exc:
            logger.warning("domain.response_error", agent_id=self.agent_id, error=str(exc))
            return None


def create_domain_agent(
    agent_id: str,
    display_name: str,
    domain: str,
    goal_id: Optional[str] = None,
) -> DomainAgent:
    """Factory: create and register a DomainAgent instance.

    Args:
        agent_id: Unique agent ID (e.g. "domain_german").
        display_name: Human-readable name (e.g. "German B1").
        domain: Domain keyword (e.g. "german").
        goal_id: Optional goal ID to link this agent to a GoalArc.

    Returns:
        The created and registered DomainAgent instance.
    """
    agent = DomainAgent(agent_id, display_name, domain, goal_id)
    register_agent(agent_id, agent)
    logger.info(
        "domain.agent_registered",
        agent_id=agent_id,
        domain=domain,
        goal_id=goal_id,
    )
    return agent


# DomainAgent is NOT registered at module bottom — it's dynamically instantiated.
# Use create_domain_agent() to mint and register domain agents.
