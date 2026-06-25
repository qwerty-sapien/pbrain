# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Adjacency rule enforcement for the dispatch subsystem.

Implements the shipped auto-pause adjacency policy plus Phase 12 weight-aware
pause targeting. Rules determine whether a new agent can be started alongside
an existing active session.

Exports:
    MAX_PARALLEL_ADJACENT_SESSIONS
    agents_are_adjacent
    check_adjacency
"""

from __future__ import annotations

from typing import Optional

import structlog

from pb.core.agent_weights import choose_lowest_weight_session, record_agent_weight_event
from pb.core.dispatch_models import InteractionEnvelope
from pb.storage.database import get_connection

logger = structlog.get_logger()

# Maximum number of concurrent adjacent sessions the user can stack.
MAX_PARALLEL_ADJACENT_SESSIONS: int = 3


def _extract_domain(agent_id: str) -> str:
    """Extract domain name from agent_id (e.g. 'domain_german' -> 'german')."""
    return agent_id[len("domain_"):] if agent_id.startswith("domain_") else agent_id


def _lookup_agent_by_domain(conn, agent_id: str):
    """Look up dispatch_agents row by domain, handling the domain_ prefix."""
    domain = _extract_domain(agent_id)
    return conn.execute(
        "SELECT domain, goal_id FROM dispatch_agents WHERE domain = ?",
        (domain,),
    ).fetchone()


def agents_are_adjacent(agent_a_id: str, agent_b_id: str) -> bool:
    """Return True if the two agents share a domain or goal_id.

    Looks up both agent records from the ``dispatch_agents`` table by domain.
    If either agent is not found (e.g. built-in agents like capture/accountability
    that may not have a row), returns False — treat them as non-adjacent to
    everything except themselves.

    Two agents are adjacent when:
    1. Both have the same non-empty ``domain`` value, OR
    2. Both have the same non-None ``goal_id`` value.
    """
    if agent_a_id == agent_b_id:
        # Same agent is always "adjacent" to itself.
        return True

    try:
        with get_connection() as conn:
            row_a = _lookup_agent_by_domain(conn, agent_a_id)
            row_b = _lookup_agent_by_domain(conn, agent_b_id)
    except Exception as exc:
        logger.warning(
            "adjacency.db_error",
            agent_a=agent_a_id,
            agent_b=agent_b_id,
            error=str(exc),
        )
        return False

    if row_a is None or row_b is None:
        # At least one agent has no DB record — treat as non-adjacent.
        logger.debug(
            "adjacency.agent_not_in_db",
            agent_a_found=row_a is not None,
            agent_b_found=row_b is not None,
        )
        return False

    domain_a = (row_a["domain"] or "").strip()
    domain_b = (row_b["domain"] or "").strip()
    goal_a = row_a["goal_id"]
    goal_b = row_b["goal_id"]

    # Condition 1: same non-empty domain
    if domain_a and domain_b and domain_a == domain_b:
        return True

    # Condition 2: same non-None goal_id
    if goal_a is not None and goal_b is not None and goal_a == goal_b:
        return True

    return False


def _record_pause_event_safely(agent_id: str, session_id: str, *, reason: str) -> None:
    """Persist a pause event without breaking dispatch flow."""
    try:
        record_agent_weight_event(
            agent_id,
            "session_paused",
            source_kind="human",
            session_id=session_id,
            metadata={"reason": reason},
        )
    except Exception as exc:
        logger.warning(
            "adjacency.pause_event_failed",
            agent_id=agent_id,
            session_id=session_id,
            error=str(exc),
        )


def check_adjacency(
    current_agent_id: Optional[str],
    new_agent_id: str,
    new_intent: str,
) -> Optional[InteractionEnvelope]:
    """Enforce adjacency rules before a new agent session is opened.

    Implements the full decision tree:

    1. capture agent     → ALWAYS ALLOWED (D-06).  Returns None.
    2. No active session → ALLOWED.  Returns None.
    3. Same agent        → CONTINUATION.  Returns None.
    4. Adjacent agents   → STACK if under cap, auto-pause the lowest-weight
                           active session if at cap.
    5. Non-adjacent      → auto-pause the current active session and allow.

    Returns:
        None if the new session is allowed.
        A blocked InteractionEnvelope if it should be rejected.
    """
    # Rule 1 — D-06: capture bypasses adjacency unconditionally (both directions).
    if new_agent_id == "capture" or current_agent_id == "capture":
        logger.debug("adjacency.capture_bypass", current=current_agent_id, new=new_agent_id)
        return None

    # Rule 2 — No active session means no adjacency constraint.
    if current_agent_id is None:
        return None

    # Rule 3 — Continuing the same agent.
    if current_agent_id == new_agent_id:
        return None

    # Rule 4 — Adjacent agents: stack up to cap, auto-pause lowest-weight if at cap.
    if agents_are_adjacent(current_agent_id, new_agent_id):
        from pb.mcp.protocol import list_active_sessions, update_session

        active_sessions = list_active_sessions()
        count = len(active_sessions)

        if count < MAX_PARALLEL_ADJACENT_SESSIONS:
            logger.info(
                "adjacency.adjacent_stack_allowed",
                current_agent=current_agent_id,
                new_agent=new_agent_id,
                active_count=count,
            )
            return None

        pause_target = choose_lowest_weight_session(active_sessions)
        if pause_target is None:
            return None
        update_session(pause_target.id, status="paused")
        _record_pause_event_safely(
            pause_target.agent_id,
            pause_target.id,
            reason="adjacent_cap",
        )
        logger.info(
            "adjacency.auto_paused_at_cap",
            paused_session=pause_target.id,
            paused_agent=pause_target.agent_id,
            new_agent=new_agent_id,
        )
        return None

    # Rule 5 — D-05: auto-pause non-adjacent session and allow switch.
    from pb.mcp.protocol import list_active_sessions, update_session

    for sess in list_active_sessions():
        if sess.agent_id == current_agent_id:
            update_session(sess.id, status="paused")
            _record_pause_event_safely(
                sess.agent_id,
                sess.id,
                reason="non_adjacent_switch",
            )
            logger.info(
                "adjacency.auto_paused_non_adjacent",
                paused_session=sess.id,
                paused_agent=current_agent_id,
                new_agent=new_agent_id,
            )
            break
    return None


def _resolve_display_name(agent_id: str) -> str:
    """Return the display name for an agent, falling back to agent_id."""
    try:
        from pb.agents import resolve_agent  # local import avoids circular dep

        handler = resolve_agent(agent_id)
        if handler is not None:
            return getattr(handler, "display_name", agent_id)
    except Exception:
        pass
    return agent_id
