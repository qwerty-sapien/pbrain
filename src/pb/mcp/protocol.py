# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interaction session lifecycle for the dispatch subsystem (Phase 10).

Each pb invocation is a fresh process, so session state is persisted to
SQLite (dispatch_sessions table) via the existing get_connection() helper.
Sessions are NOT held in-memory dictionaries.

Exports:
    create_session, get_session, update_session, advance_session,
    list_active_sessions, InteractionSession (alias for DispatchSession)
"""

from __future__ import annotations

import asyncio
import json
import secrets
from datetime import datetime
from typing import Optional

import structlog

from pb.core.dispatch_models import DispatchSession, InteractionEnvelope
from pb.storage.database import get_connection

logger = structlog.get_logger()

# Public alias for callers that want the session type
InteractionSession = DispatchSession


def create_session(
    agent_id: str,
    context: Optional[dict] = None,
) -> DispatchSession:
    """Create a new dispatch session and persist it to SQLite.

    Args:
        agent_id: ID of the agent that will handle this session.
        context: Optional initial context dict (serialised to JSON).

    Returns:
        Newly created DispatchSession model.
    """
    session_id = secrets.token_hex(16)
    now = datetime.utcnow().isoformat()
    context_json = json.dumps(context or {})

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dispatch_sessions
                (id, agent_id, status, context_json, created_at, updated_at,
                 judged, judged_at)
            VALUES (?, ?, 'active', ?, ?, ?, 0, NULL)
            """,
            (session_id, agent_id, context_json, now, now),
        )
        conn.commit()

    return DispatchSession(
        id=session_id,
        agent_id=agent_id,
        status="active",
        context_json=context_json,
        created_at=datetime.fromisoformat(now),
        updated_at=datetime.fromisoformat(now),
    )


def get_session(session_id: str) -> Optional[DispatchSession]:
    """Look up a dispatch session by ID.

    Returns:
        DispatchSession model, or None if not found.
    """
    with get_connection() as conn:
        row = conn.execute(
            "SELECT * FROM dispatch_sessions WHERE id = ?",
            (session_id,),
        ).fetchone()

    if row is None:
        return None

    return _row_to_session(row)


def update_session(
    session_id: str,
    *,
    status: Optional[str] = None,
    context_json: Optional[str] = None,
) -> None:
    """Partially update a dispatch session row.

    Args:
        session_id: Target session ID.
        status: New status string (e.g. "active", "complete", "error").
        context_json: Replacement context JSON string.
    """
    now = datetime.utcnow().isoformat()
    updates: list[str] = ["updated_at = ?"]
    params: list[object] = [now]

    if status is not None:
        updates.append("status = ?")
        params.append(status)
    if context_json is not None:
        updates.append("context_json = ?")
        params.append(context_json)

    params.append(session_id)

    with get_connection() as conn:
        conn.execute(
            f"UPDATE dispatch_sessions SET {', '.join(updates)} WHERE id = ?",
            params,
        )
        conn.commit()


def advance_session(
    session_id: str,
    *,
    select: Optional[int] = None,
    fill: Optional[dict] = None,
) -> dict:
    """Advance a session by invoking the registered agent's handle() method.

    Builds an intent string from ``select`` (option index) or ``fill`` (free text
    or key-value pairs), calls the agent, persists the new session state, and
    returns the InteractionEnvelope as a plain dict.

    Args:
        session_id: Target session to advance.
        select: 1-based option index chosen by the user (or caller).
        fill: Dict with free-text intent or field values (key "text" is treated
              as the raw intent string).

    Returns:
        InteractionEnvelope.model_dump() dict, or an error dict on failure.
    """
    _error = {
        "session_id": session_id,
        "status": "error",
        "prompt": "Something went wrong. Please try again.",
        "options": [],
        "fields": {},
    }

    try:
        session = get_session(session_id)
        if session is None:
            logger.warning("advance_session.not_found", session_id=session_id)
            return _error

        from pb.agents import resolve_agent  # local import avoids circular dep at module load

        agent = resolve_agent(session.agent_id)
        if agent is None:
            logger.warning(
                "advance_session.agent_not_found",
                session_id=session_id,
                agent_id=session.agent_id,
            )
            return _error

        # Build intent string
        if fill:
            intent = fill.get("text") or json.dumps(fill)
        elif select is not None:
            intent = f"select:{select}"
        else:
            intent = ""

        # Call agent — asyncio.run() if no running loop; else await directly
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop is not None and loop.is_running():
            # We're already inside an async context; schedule as a coroutine.
            # This path is taken by async callers (e.g. tests, MCP server).
            import concurrent.futures
            future = asyncio.ensure_future(
                agent.handle(intent, session, context=None)
            )
            output = loop.run_until_complete(future)
        else:
            output = asyncio.run(agent.handle(intent, session, context=None))

        envelope = agent.to_envelope(session, output)

        # Persist updated session status
        new_status = output.status if output.status else "active"
        update_session(session_id, status=new_status)

        return envelope.model_dump()

    except Exception as exc:
        logger.warning(
            "advance_session.error",
            session_id=session_id,
            error=str(exc),
            exc_info=True,
        )
        return _error


def list_active_sessions() -> list[DispatchSession]:
    """Return all sessions with status='active'."""
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM dispatch_sessions WHERE status = 'active' ORDER BY created_at DESC"
        ).fetchall()

    return [_row_to_session(row) for row in rows]


def deactivate_all_sessions() -> int:
    """Mark all active dispatch_sessions as 'deactivated'. Returns count closed."""
    now = datetime.utcnow().isoformat()
    with get_connection() as conn:
        cursor = conn.execute(
            "UPDATE dispatch_sessions SET status = 'deactivated', updated_at = ? WHERE status = 'active'",
            (now,),
        )
        conn.commit()
        return cursor.rowcount


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _row_to_session(row: object) -> DispatchSession:
    """Convert a sqlite3.Row to a DispatchSession model."""
    r = dict(row)
    judged_at_raw = r.get("judged_at")
    return DispatchSession(
        id=r["id"],
        agent_id=r["agent_id"],
        status=r["status"],
        context_json=r.get("context_json", "{}"),
        created_at=datetime.fromisoformat(r["created_at"]),
        updated_at=datetime.fromisoformat(r["updated_at"]),
        judged=bool(r.get("judged", 0)),
        judged_at=datetime.fromisoformat(judged_at_raw) if judged_at_raw else None,
    )
