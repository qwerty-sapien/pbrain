# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Stateful LLM dispatcher — the new brain for ``pb do`` (Phase 10).

Replaces ``suggest_commands_for_intent`` (keyword-heuristic) with a
structured-output LLM call that classifies intent into one of 12 dispatch
modes, opens or continues sessions with stickiness, enforces adjacency rules,
gates domain agents (D-07/D-08), and applies Phase 12 agent-weight reranking.

Exports:
    dispatch            — main async entry point (fresh or continuation)
    continue_session    — convenience wrapper for known session_id
"""

from __future__ import annotations

import json
import logging
from typing import Optional

import structlog

from pb.core.agent_weights import (
    normalized_effective_weights,
    record_agent_weight_event,
    sort_sessions_by_weight,
)
from pb.core.adjacency import check_adjacency
from pb.core.agent_instruction_judge import (
    format_patch_announcement,
    judge_agent_instruction_fit,
)
from pb.core.dispatch_models import (
    DispatchDecision,
    DispatchSession,
    InteractionEnvelope,
)
from pb.llm.structured import structured_output_call
from pb.mcp.protocol import (
    create_session,
    get_session,
    list_active_sessions,
    update_session,
)
from pb.storage.database import get_connection

logger = structlog.get_logger()


# ---------------------------------------------------------------------------
# Dispatch system prompt (module constant — D-13: context-guided, not patched)
# ---------------------------------------------------------------------------

_DISPATCH_SYSTEM_PROMPT = """\
You are the intent classifier for ProductiveBrain (pb), a terminal-native personal operating layer.

Given a user's free-text input and their current context (active goals, todos, commitments,
open sessions), you route the intent to exactly ONE of the following 12 modes:

MODE TAXONOMY
=============
- plan               : creating a plan, weekly/daily planning, shaping a schedule
- todo               : capturing, reviewing, completing, or listing tasks/todos
- goal_refinement    : clarifying, updating, or reflecting on higher-level goals
- review             : end-of-day or weekly reflection, progress review, retrospective
- accountability_intervention : commitment tracking, habit accountability, keeping user on track
- study_session      : conceptual study, reading, Anki review, understanding theory
- practice_drill     : deliberate practice, reps, drills, skills like music/sport/language
- teaching_explanation : Socratic/interactive teaching where pb plays the tutor
- note_organisation  : organising vault notes, filing, linking, archiving
- thought_capture    : capturing a fleeting thought, idea, or brain-dump (no processing needed)
- next_action_selection : "what should I do now?" — surfaces the best next action given context
- domain_delegation  : routing to a specialised domain agent (e.g. domain_german, domain_coding)

OUTPUT SCHEMA (JSON)
====================
{
  "agent_id": "<string — one of the mode keys above, or 'capture' for thought_capture, or domain_<name>>",
  "candidate_agent_ids": ["<ordered plausible agent ids — include agent_id as the first item>"],
  "confidence": <float 0.0–1.0>,
  "in_scope": <boolean — true unless the request is completely out of scope for pb>,
  "scope_reason": "<internal reason string — NOT shown to user>"
}

AGENT ID MAPPING
================
- plan               → "plan"
- todo               → "todo"
- goal_refinement    → "goal_refinement"
- review             → "review"
- accountability_intervention → "accountability"
- study_session      → "study"
- practice_drill     → "practice"
- teaching_explanation → "teach"
- note_organisation  → "notes"
- thought_capture    → "capture"
- next_action_selection → "next_action"
- domain_delegation  → "domain_<name>" (e.g. "domain_german", "domain_coding")

FEW-SHOT EXAMPLES
=================
Input: "practise German"
Context: goal "Reach B1 German" active
Output: {"agent_id": "domain_german", "candidate_agent_ids": ["domain_german", "practice"], "confidence": 0.95, "in_scope": true, "scope_reason": "domain practice for active German goal"}

Input: "buy milk"
Context: no active goal matches
Output: {"agent_id": "capture", "candidate_agent_ids": ["capture", "todo"], "confidence": 0.99, "in_scope": true, "scope_reason": "personal errand, capture as thought"}

Input: "keep me accountable for finishing the thesis"
Context: goal "PhD thesis" active
Output: {"agent_id": "accountability", "candidate_agent_ids": ["accountability", "review"], "confidence": 0.92, "in_scope": true, "scope_reason": "explicit accountability request"}

Input: "what should I do now?"
Context: 3 active todos, 1 open goal
Output: {"agent_id": "next_action", "candidate_agent_ids": ["next_action", "accountability"], "confidence": 0.90, "in_scope": true, "scope_reason": "explicit next-action query"}

Input: "review my day"
Context: session ended 2h ago
Output: {"agent_id": "review", "candidate_agent_ids": ["review", "accountability"], "confidence": 0.95, "in_scope": true, "scope_reason": "daily review intent"}

RULES
=====
- User intent is always passed as a user-role message; NEVER injected into the system prompt.
- Prefer the most specific agent_id possible given the context.
- Always return candidate_agent_ids in descending plausibility order, with agent_id first.
- If no mode fits clearly, use "capture" (confidence < 0.5) rather than guessing.
- "in_scope" is False only if the request is completely outside what a personal productivity tool handles.
- scope_reason is an internal routing note — short, diagnostic.
"""


# ---------------------------------------------------------------------------
# Phase 12 / later-phase hooks
# ---------------------------------------------------------------------------


def _dispatch_ranking_hook(
    decision: DispatchDecision,
    *,
    excluded_agent_ids: Optional[set[str]] = None,
) -> DispatchDecision:
    """Phase 12 plugs in frecency re-ranking for plausible fresh-dispatch agents.

    Only reranks within the LLM-provided candidate list.  Intent-match still
    dominates via a strong dispatch-rank prior, and high-confidence decisions
    remain untouched unless an explicit agent exclusion removes the top choice.
    """
    excluded = {agent_id for agent_id in (excluded_agent_ids or set()) if agent_id}
    ordered_candidates: list[str] = []
    for agent_id in [decision.agent_id, *decision.candidate_agent_ids]:
        normalized = (agent_id or "").strip()
        if not normalized or normalized in ordered_candidates or normalized in excluded:
            continue
        ordered_candidates.append(normalized)

    if not ordered_candidates:
        fallback_agent = _CAPTURE_AGENT_ID if _CAPTURE_AGENT_ID not in excluded else decision.agent_id
        decision.agent_id = fallback_agent
        decision.candidate_agent_ids = [fallback_agent]
        return decision

    decision.candidate_agent_ids = ordered_candidates
    if decision.confidence >= 0.90 or len(ordered_candidates) == 1:
        decision.agent_id = ordered_candidates[0]
        return decision

    effective_weights = normalized_effective_weights(ordered_candidates)
    dispatch_priors = {
        agent_id: max(0.50, 1.00 - (index * 0.05))
        for index, agent_id in enumerate(ordered_candidates)
    }
    reranked = sorted(
        ordered_candidates,
        key=lambda agent_id: (
            (0.80 * dispatch_priors.get(agent_id, 0.50))
            + (0.20 * effective_weights.get(agent_id, 0.50)),
            dispatch_priors.get(agent_id, 0.50),
            effective_weights.get(agent_id, 0.50),
        ),
        reverse=True,
    )
    decision.agent_id = reranked[0]
    decision.candidate_agent_ids = reranked
    return decision


def _resume_ordering_hook(sessions: list[DispatchSession]) -> list[DispatchSession]:
    """Phase 12 plugs in weight-based ordering.

    Active dispatch sessions are ordered by effective agent weight first, then
    by most-recent update time.
    """
    return sort_sessions_by_weight(sessions)


def _adjacency_suggestion_hook(
    blocked_envelope: InteractionEnvelope,
) -> InteractionEnvelope:
    """Annotate blocked adjacency envelopes with weight-aware session suggestions."""
    try:
        ranked_sessions = sort_sessions_by_weight(list_active_sessions())
    except Exception as exc:
        logger.warning("dispatcher.adjacency_suggestion_failed", error=str(exc))
        return blocked_envelope
    if not ranked_sessions:
        return blocked_envelope

    suggested = ranked_sessions[-1]
    fields = dict(blocked_envelope.fields or {})
    fields.setdefault("suggested_session_id", suggested.id)
    fields.setdefault("suggested_agent_id", suggested.agent_id)
    blocked_envelope.fields = fields
    suggestion = f"Pause {suggested.agent_id} and continue"
    if suggestion not in blocked_envelope.options:
        blocked_envelope.options = [suggestion, *list(blocked_envelope.options)]
    return blocked_envelope


def _finish_pause_approval_hook(session: DispatchSession) -> None:
    """Record non-completion lifecycle signals for the frecency scorer."""
    try:
        refreshed = get_session(session.id) or session
        if refreshed.status == "paused":
            _record_weight_event_safely(
                refreshed.agent_id,
                "session_paused",
                session_id=refreshed.id,
                metadata={"source": "finish_pause_hook"},
            )
        elif refreshed.status in {"blocked", "error"}:
            _record_weight_event_safely(
                refreshed.agent_id,
                "kickback",
                session_id=refreshed.id,
                metadata={"source": "finish_pause_hook", "status": refreshed.status},
            )
    except Exception as exc:
        logger.warning(
            "dispatcher.finish_pause_hook_failed",
            session_id=getattr(session, "id", ""),
            error=str(exc),
        )


def _record_weight_event_safely(
    agent_id: str,
    event_kind: str,
    *,
    session_id: str = "",
    metadata: Optional[dict] = None,
) -> None:
    """Persist a scorer event without letting failures break routing."""
    try:
        record_agent_weight_event(
            agent_id,
            event_kind,
            source_kind="human",
            session_id=session_id,
            metadata=metadata or {},
        )
    except Exception as exc:
        logger.warning(
            "dispatcher.agent_weight_event_failed",
            agent_id=agent_id,
            event_kind=event_kind,
            error=str(exc),
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _load_active_commitments() -> list[dict]:
    """Query commitments WHERE status='active', ordered by due_date ASC."""
    try:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT id, description, due_date
                FROM commitments
                WHERE status = 'active'
                ORDER BY due_date ASC NULLS LAST
                LIMIT 5
                """,
            ).fetchall()
        return [
            {
                "id": row["id"],
                "description": row["description"],
                "due_date": row["due_date"],
            }
            for row in rows
        ]
    except Exception as exc:
        logger.warning("dispatcher.load_commitments_failed", error=str(exc))
        return []


def _build_dispatch_user_prompt(
    intent: str,
    goals: list,
    todos: list,
    commitments: list[dict],
    active_sessions: list[DispatchSession],
) -> str:
    """Build the user-role prompt for the dispatch LLM call.

    Per D-13: user intent is in the user-role message, never injected into the
    system prompt.  Context is appended below the intent.
    """
    lines = [f"User intent: {intent}", "", "Context:"]

    if goals:
        goal_titles = [getattr(g, "title", str(g)) for g in goals[:5]]
        lines.append(f"  Active goals: {', '.join(goal_titles)}")
    else:
        lines.append("  Active goals: none")

    if todos:
        todo_titles = [getattr(t, "title", str(t)) for t in todos[:3]]
        lines.append(f"  Top todos: {', '.join(todo_titles)}")
    else:
        lines.append("  Top todos: none")

    if commitments:
        comm_desc = [c.get("description", "") for c in commitments[:3]]
        lines.append(f"  Active commitments: {', '.join(comm_desc)}")
    else:
        lines.append("  Active commitments: none")

    if active_sessions:
        sess = active_sessions[0]
        lines.append(f"  Open session: agent={sess.agent_id}, id={sess.id[:8]}...")
    else:
        lines.append("  Open session: none")

    return "\n".join(lines)


def _get_or_create_domain_agent_record(
    conn, domain: str
) -> Optional[dict]:
    """Return the dispatch_agent row for a domain agent, or None if not found."""
    row = conn.execute(
        "SELECT id, domain, goal_id, interaction_count FROM dispatch_agents WHERE domain = ?",
        (domain,),
    ).fetchone()
    return dict(row) if row else None



async def _handle_domain_gating(
    agent_id: str,
    intent: str,
    goals: list,
    initial_context: dict,
) -> Optional[InteractionEnvelope]:
    """Apply D-07/D-08 domain agent gating.

    D-07: domain agents require goal existence + interaction_count >= 2.
    D-08: if no goal exists for the domain, route to goal-setting.

    Returns:
        None if the domain agent is allowed to proceed.
        An InteractionEnvelope if gating should block/redirect the dispatch.
    """
    # Extract domain name from agent_id (e.g. "domain_german" -> "german")
    domain = agent_id[len("domain_"):] if agent_id.startswith("domain_") else agent_id

    # Find matching goal
    goal_for_domain = None
    for goal in goals:
        goal_domain = (getattr(goal, "domain", "") or "").lower()
        goal_title = (getattr(goal, "title", "") or "").lower()
        if domain.lower() in goal_domain or domain.lower() in goal_title:
            goal_for_domain = goal
            break

    try:
        with get_connection() as conn:
            agent_row = _get_or_create_domain_agent_record(conn, domain)

            if agent_row is None:
                # No agent row yet — bootstrap it if a goal exists
                if goal_for_domain is None:
                    # D-08: No goal — route to goal-setting
                    logger.info(
                        "dispatcher.domain_no_goal",
                        domain=domain,
                        agent_id=agent_id,
                    )
                    return InteractionEnvelope(
                        session_id="",
                        status="active",
                        prompt=f"Let's set a goal for {domain} first.",
                        options=[
                            f"Create a goal for {domain}",
                            "Skip goal for now",
                            "Cancel",
                        ],
                        fields={"domain": domain},
                    )
                # Goal exists but no agent row yet — create it and allow
                return None

            # Agent row exists — check interaction_count and goal
            interaction_count = agent_row.get("interaction_count", 0)
            has_goal = goal_for_domain is not None

            if not has_goal:
                # D-08: no goal for this domain
                logger.info(
                    "dispatcher.domain_no_goal",
                    domain=domain,
                    agent_id=agent_id,
                )
                return InteractionEnvelope(
                    session_id="",
                    status="active",
                    prompt=f"Let's set a goal for {domain} first.",
                    options=[
                        f"Create a goal for {domain}",
                        "Skip goal for now",
                        "Cancel",
                    ],
                    fields={"domain": domain},
                )

            # D-07: goal exists but < 2 interactions — allow (spawner owns count mutation)
            if interaction_count < 2:
                logger.info(
                    "dispatcher.domain_gating_warmup",
                    domain=domain,
                    interaction_count=interaction_count,
                )
                return None  # Allow — still warming up

            # Fully gated: goal exists and >= 2 interactions
            return None

    except Exception as exc:
        logger.warning(
            "dispatcher.domain_gating_error",
            domain=domain,
            error=str(exc),
        )
        return None  # Fail open — let dispatch proceed


# ---------------------------------------------------------------------------
# Fallback capture envelope
# ---------------------------------------------------------------------------

_CAPTURE_AGENT_ID = "capture"


async def _fallback_to_capture(
    intent: str,
    reason: str,
    initial_context: dict,
) -> InteractionEnvelope:
    """Create a session with the capture agent as fallback."""
    from pb.agents import resolve_agent  # local import avoids circular dep

    logger.info("dispatcher.fallback_capture", reason=reason, intent=intent)

    session = create_session(
        _CAPTURE_AGENT_ID,
        context={**initial_context, "kickback_count": 0, "fallback_reason": reason},
    )

    agent = resolve_agent(_CAPTURE_AGENT_ID)
    if agent is not None:
        try:
            output = await agent.handle(intent, session, context=None)
            update_session(session.id, status=output.status)
            return agent.to_envelope(session, output)
        except Exception as exc:
            logger.warning("dispatcher.capture_agent_error", error=str(exc))

    # Bare envelope if capture agent not registered yet
    update_session(session.id, status="complete")
    return InteractionEnvelope(
        session_id=session.id,
        status="complete",
        prompt="I've captured that. What would you like to do with it?",
        options=["Save as thought", "Save as todo", "Discard"],
        fields={"fallback_reason": reason},
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def dispatch(
    repo,
    intent: str,
    *,
    session_id: Optional[str] = None,
    excluded_agent_ids: Optional[set[str]] = None,
) -> InteractionEnvelope:
    """Classify intent and dispatch to the appropriate agent.

    Entry point for ``pb do``.  Two paths:

    **Continuation path** (``session_id`` provided):
        Looks up the existing session, resolves its agent, and calls
        ``agent.handle(intent, session)``.  If the agent kicks back
        (``output.in_scope == False``), increments ``kickback_count`` in
        session context.  After 2 kickbacks, falls through to capture.

    **Fresh dispatch path** (no ``session_id``):
        Loads user context (goals, todos, commitments, open sessions), builds a
        context-aware prompt, calls the LLM via ``structured_output_call``,
        applies Phase 12 ranking hook, checks adjacency, applies D-07/D-08
        domain gating, creates a new session, and dispatches to the resolved
        agent.

    Args:
        repo: Repository instance (provides list_goal_arcs, list_tasks).
        intent: The raw user intent string.
        session_id: Optional existing session ID for continuation.

    Returns:
        InteractionEnvelope with session_id, status, prompt, options, fields.
    """
    from pb.agents import resolve_agent  # local import avoids circular dep
    from pb.core.action_routing import _best_todo_tasks  # local import

    # ------------------------------------------------------------------
    # CONTINUATION PATH
    # ------------------------------------------------------------------
    if session_id is not None:
        return await _continue_session(repo, intent, session_id)

    # ------------------------------------------------------------------
    # FRESH DISPATCH PATH
    # ------------------------------------------------------------------

    # Step a: Load context
    try:
        goals = repo.list_goal_arcs(status=None) or []
    except Exception as exc:
        logger.warning("dispatcher.load_goals_failed", error=str(exc))
        goals = []

    try:
        todos = _best_todo_tasks(repo)[:3]
    except Exception as exc:
        logger.warning("dispatcher.load_todos_failed", error=str(exc))
        todos = []

    commitments = _load_active_commitments()

    active_sessions = list_active_sessions()
    # Apply Phase 12 resume-ordering hook
    active_sessions = _resume_ordering_hook(active_sessions)

    initial_context = {
        "intent": intent,
        "goals": [getattr(g, "title", str(g)) for g in goals[:5]],
        "kickback_count": 0,
    }

    # Step b: Build dispatch prompt
    user_prompt = _build_dispatch_user_prompt(
        intent, goals, todos, commitments, active_sessions
    )

    # Step c: LLM call for dispatch decision
    decision: Optional[DispatchDecision] = await structured_output_call(
        user_prompt,
        DispatchDecision,
        system_prompt=_DISPATCH_SYSTEM_PROMPT,
        tier="lite",
    )

    if decision is None:
        logger.warning("dispatcher.llm_returned_none", intent=intent)
        return await _fallback_to_capture(intent, "llm_failure", initial_context)

    # T-10-05 mitigation: log scope_reason internally, never propagate to envelope
    logger.info(
        "dispatcher.decision",
        agent_id=decision.agent_id,
        confidence=decision.confidence,
        in_scope=decision.in_scope,
        scope_reason=decision.scope_reason,
    )

    # Step d: Phase 12 dispatch-ranking hook
    decision = _dispatch_ranking_hook(
        decision,
        excluded_agent_ids=excluded_agent_ids,
    )

    # Step e: Adjacency check
    current_agent_id = active_sessions[0].agent_id if active_sessions else None
    blocked = check_adjacency(current_agent_id, decision.agent_id, intent)
    if blocked is not None:
        blocked.fields["original_intent"] = intent
        blocked = _adjacency_suggestion_hook(blocked)
        return blocked

    # Step f: D-07/D-08 domain agent gating
    if decision.agent_id.startswith("domain_"):
        gate_result = await _handle_domain_gating(
            decision.agent_id, intent, goals, initial_context
        )
        if gate_result is not None:
            return gate_result

    # Step g: Create session
    session = create_session(decision.agent_id, context=initial_context)

    # Step h: Resolve and execute agent
    agent = resolve_agent(decision.agent_id)
    if agent is None:
        logger.warning(
            "dispatcher.agent_not_registered",
            agent_id=decision.agent_id,
        )
        # Close the ghost session before falling back — prevents a status='active'
        # row that would block all subsequent dispatches (T-10-20 mitigation).
        update_session(session.id, status="error")
        # Fall to capture on unregistered agent
        return await _fallback_to_capture(
            intent, f"agent_not_registered:{decision.agent_id}", initial_context
        )

    try:
        output = await agent.handle(
            intent,
            session,
            context={
                "goals": goals,
                "todos": todos,
                "commitments": commitments,
                "repo": repo,
                "deadline_todos": [
                    t for t in todos
                    if getattr(t, "due_date", None) is not None
                ],
            },
        )
    except Exception as exc:
        logger.warning(
            "dispatcher.agent_handle_error",
            agent_id=decision.agent_id,
            error=str(exc),
        )
        update_session(session.id, status="error")
        return await _fallback_to_capture(intent, f"agent_error:{exc}", initial_context)

    # Persist updated session status
    update_session(session.id, status=output.status)
    _record_weight_event_safely(
        decision.agent_id,
        "dispatch_selected",
        session_id=session.id,
        metadata={"intent": intent},
    )
    if output.status == "complete":
        _record_weight_event_safely(
            decision.agent_id,
            "session_completed",
            session_id=session.id,
            metadata={"intent": intent},
        )
    if output.misalignment_signal:
        patch_record = await judge_agent_instruction_fit(
            agent_id=decision.agent_id,
            session_id=session.id,
            feedback_text=output.misalignment_signal,
            evidence=[f"intent: {intent}", f"response_status: {output.status}"],
            trigger_kind="agent_misalignment",
            auto_apply=True,
        )
        if patch_record is not None:
            output.fields["agent_instruction_patch_id"] = patch_record.id
            output.fields["agent_instruction_patch_status"] = patch_record.status
            output.response = (
                f"{output.response}\n\n{format_patch_announcement(patch_record)}"
            ).strip()

    # Later-phase hook stub — no-op for Phase 12
    _finish_pause_approval_hook(session)

    return agent.to_envelope(session, output)


async def continue_session(
    repo,
    intent: str,
    session_id: str,
) -> InteractionEnvelope:
    """Convenience wrapper: continue an existing session with new intent.

    Equivalent to ``dispatch(repo, intent, session_id=session_id)``.
    """
    return await dispatch(repo, intent, session_id=session_id)


# ---------------------------------------------------------------------------
# Continuation path (private)
# ---------------------------------------------------------------------------


async def _continue_session(
    repo,
    intent: str,
    session_id: str,
) -> InteractionEnvelope:
    """Handle the continuation path for an existing session."""
    from pb.agents import resolve_agent  # local import avoids circular dep

    session = get_session(session_id)
    if session is None or session.status != "active":
        logger.warning(
            "dispatcher.session_not_found_or_inactive",
            session_id=session_id,
        )
        # Fall through to fresh dispatch
        return await dispatch(repo, intent)

    agent = resolve_agent(session.agent_id)
    if agent is None:
        logger.warning(
            "dispatcher.continuation_agent_not_registered",
            agent_id=session.agent_id,
            session_id=session_id,
        )
        return await _fallback_to_capture(intent, "agent_not_registered", {})

    # Rebuild context from repo so multi-turn agents retain goals/todos/repo
    from pb.core.action_routing import _best_todo_tasks

    try:
        goals = repo.list_goal_arcs(status=None) or []
    except Exception:
        goals = []
    try:
        todos = _best_todo_tasks(repo)[:3]
    except Exception:
        todos = []

    continuation_context = {"goals": goals, "todos": todos, "repo": repo}

    try:
        output = await agent.handle(intent, session, context=continuation_context)
    except Exception as exc:
        logger.warning(
            "dispatcher.continuation_handle_error",
            session_id=session_id,
            error=str(exc),
        )
        update_session(session_id, status="error")
        return InteractionEnvelope(
            session_id=session_id,
            status="error",
            prompt="Something went wrong. Please try again.",
        )

    # Handle kickback (out-of-scope response from agent)
    if not output.in_scope:
        try:
            context_data = json.loads(session.context_json)
        except (json.JSONDecodeError, TypeError):
            context_data = {}

        kickback_count = context_data.get("kickback_count", 0) + 1
        context_data["kickback_count"] = kickback_count

        logger.info(
            "dispatcher.kickback",
            session_id=session_id,
            agent_id=session.agent_id,
            kickback_count=kickback_count,
            scope_reason=output.scope_reason,
        )
        _record_weight_event_safely(
            session.agent_id,
            "kickback",
            session_id=session_id,
            metadata={"intent": intent, "scope_reason": output.scope_reason},
        )

        if kickback_count >= 2:
            # T-10-07: After 2 kickbacks, force to capture (prevents infinite loop)
            update_session(session_id, status="blocked", context_json=json.dumps(context_data))
            return await _fallback_to_capture(
                intent,
                f"kickback_overflow:{session.agent_id}",
                {"intent": intent, "kickback_count": kickback_count},
            )

        # Re-dispatch with incremented kickback count (fresh dispatch path)
        update_session(session_id, context_json=json.dumps(context_data))
        return await dispatch(repo, intent)

    # Normal continuation — persist updated status
    update_session(session_id, status=output.status)
    _record_weight_event_safely(
        session.agent_id,
        "session_continued",
        session_id=session_id,
        metadata={"intent": intent},
    )
    if output.status == "complete":
        _record_weight_event_safely(
            session.agent_id,
            "session_completed",
            session_id=session_id,
            metadata={"intent": intent},
        )
    if output.misalignment_signal:
        patch_record = await judge_agent_instruction_fit(
            agent_id=session.agent_id,
            session_id=session.id,
            feedback_text=output.misalignment_signal,
            evidence=[f"intent: {intent}", f"response_status: {output.status}"],
            trigger_kind="agent_misalignment",
            auto_apply=True,
        )
        if patch_record is not None:
            output.fields["agent_instruction_patch_id"] = patch_record.id
            output.fields["agent_instruction_patch_status"] = patch_record.status
            output.response = (
                f"{output.response}\n\n{format_patch_announcement(patch_record)}"
            ).strip()

    # Later-phase hook stub
    _finish_pause_approval_hook(session)

    return agent.to_envelope(session, output)
