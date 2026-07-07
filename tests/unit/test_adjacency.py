# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for pb.core.adjacency (Phase 10).

Tests cover:
  - MAX_PARALLEL_ADJACENT_SESSIONS constant value
  - agents_are_adjacent: same domain -> True
  - agents_are_adjacent: same goal_id -> True
  - agents_are_adjacent: different domain and goal -> False
  - check_adjacency: capture agent -> None (always allowed, D-06)
  - check_adjacency: no active session -> None (no constraint)
  - check_adjacency: non-adjacent active session -> blocked envelope (D-05)

All tests use a temporary SQLite database.
"""

from __future__ import annotations

import secrets
from datetime import datetime

import pytest

from pb.core.dispatch_models import InteractionEnvelope
from pb.storage.database import DB_FILENAME, init_db, set_db_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path):
    """Isolated SQLite DB for each test."""
    db_path = tmp_path / DB_FILENAME
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    from pb.storage import database as db_module
    db_module._db_path = None


def _insert_dispatch_agent(conn, agent_id: str, domain: str, goal_id=None):
    """Insert a row into dispatch_agents for testing adjacency."""
    now = datetime.utcnow().isoformat()
    conn.execute(
        "INSERT INTO dispatch_agents (id, domain, goal_id, config_json, created_at, interaction_count) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (agent_id, domain, goal_id, "{}", now, 0),
    )
    conn.commit()


# ---------------------------------------------------------------------------
# MAX_PARALLEL_ADJACENT_SESSIONS
# ---------------------------------------------------------------------------


def test_max_parallel_adjacent_sessions_is_three():
    """MAX_PARALLEL_ADJACENT_SESSIONS must equal 3 (cap defined in D-05)."""
    from pb.core.adjacency import MAX_PARALLEL_ADJACENT_SESSIONS

    assert MAX_PARALLEL_ADJACENT_SESSIONS == 3


# ---------------------------------------------------------------------------
# agents_are_adjacent
# ---------------------------------------------------------------------------


def test_agents_are_adjacent_same_domain_returns_true(temp_db):
    """Two agents sharing the same domain are adjacent (looked up by domain, not id)."""
    from pb.core.adjacency import agents_are_adjacent
    from pb.storage.database import get_connection

    with get_connection() as conn:
        _insert_dispatch_agent(conn, "uuid-alpha", "german")
        _insert_dispatch_agent(conn, "uuid-beta", "german")

    # Dispatcher passes logical agent_ids like "domain_german"; adjacency strips the prefix
    assert agents_are_adjacent("domain_german", "domain_german") is True


def test_agents_are_adjacent_same_goal_id_returns_true(temp_db):
    """Two agents sharing the same goal_id are adjacent."""
    from pb.core.adjacency import agents_are_adjacent
    from pb.storage.database import get_connection

    with get_connection() as conn:
        _insert_dispatch_agent(conn, "uuid-x", "study", goal_id="goal-123")
        _insert_dispatch_agent(conn, "uuid-y", "practice", goal_id="goal-123")

    assert agents_are_adjacent("domain_study", "domain_practice") is True


def test_agents_are_adjacent_different_domain_and_goal_returns_false(temp_db):
    """Two agents with different domain and no shared goal_id are not adjacent."""
    from pb.core.adjacency import agents_are_adjacent
    from pb.storage.database import get_connection

    with get_connection() as conn:
        _insert_dispatch_agent(conn, "uuid-p", "cooking", goal_id="goal-aaa")
        _insert_dispatch_agent(conn, "uuid-q", "coding", goal_id="goal-bbb")

    assert agents_are_adjacent("domain_cooking", "domain_coding") is False


def test_agents_are_adjacent_same_agent_id_returns_true(temp_db):
    """Same agent is always adjacent to itself."""
    from pb.core.adjacency import agents_are_adjacent

    assert agents_are_adjacent("capture", "capture") is True


def test_agents_are_adjacent_missing_db_record_returns_false(temp_db):
    """Agents with no DB record are treated as non-adjacent."""
    from pb.core.adjacency import agents_are_adjacent

    # Neither agent has a row in the DB
    assert agents_are_adjacent("ghost-a", "ghost-b") is False


# ---------------------------------------------------------------------------
# check_adjacency
# ---------------------------------------------------------------------------


def test_check_adjacency_capture_agent_always_allowed():
    """check_adjacency always returns None for the capture agent (D-06)."""
    from pb.core.adjacency import check_adjacency

    # Regardless of active session
    result = check_adjacency("review", "capture", "buy milk")
    assert result is None


def test_check_adjacency_no_active_session_returns_none():
    """check_adjacency returns None when current_agent_id is None (no constraint)."""
    from pb.core.adjacency import check_adjacency

    result = check_adjacency(None, "review", "review my day")
    assert result is None


def test_check_adjacency_same_agent_returns_none():
    """check_adjacency returns None when new agent is the same as current (continuation)."""
    from pb.core.adjacency import check_adjacency

    result = check_adjacency("review", "review", "continue the review")
    assert result is None


def test_check_adjacency_non_adjacent_auto_pauses(temp_db):
    """Non-adjacent switch auto-pauses the current session and returns None."""
    import secrets
    from pb.core.adjacency import check_adjacency
    from pb.mcp.protocol import create_session, get_session
    from pb.storage.database import get_connection

    with get_connection() as conn:
        _insert_dispatch_agent(conn, "uuid-cook", "cooking", goal_id="goal-cook")
        _insert_dispatch_agent(conn, "uuid-code", "coding", goal_id="goal-code")

    sess = create_session("domain_cooking")
    result = check_adjacency("domain_cooking", "domain_coding", "help me with Python")

    assert result is None
    updated = get_session(sess.id)
    assert updated.status == "paused"


def test_check_adjacency_adjacent_at_cap_auto_pauses_lowest_weight_session(temp_db):
    """Adjacent-at-cap auto-pauses the lowest-weight eligible session."""
    import secrets
    from datetime import datetime
    from pb.core.adjacency import MAX_PARALLEL_ADJACENT_SESSIONS, check_adjacency
    from pb.core.agent_weights import record_agent_weight_event, set_weight_override
    from pb.mcp.protocol import get_session
    from pb.storage.database import get_connection

    with get_connection() as conn:
        _insert_dispatch_agent(conn, "uuid-ger-study", "study", goal_id="goal-ger")
        _insert_dispatch_agent(conn, "uuid-ger-practice", "practice", goal_id="goal-ger")
        _insert_dispatch_agent(conn, "uuid-ger-vocab", "vocab", goal_id="goal-ger")
        _insert_dispatch_agent(conn, "uuid-ger-listening", "listening", goal_id="goal-ger")

        now = datetime.utcnow().isoformat()
        session_ids = {}
        for agent_id in ("domain_study", "domain_practice", "domain_vocab"):
            sid = secrets.token_hex(16)
            session_ids[agent_id] = sid
            conn.execute(
                "INSERT INTO dispatch_sessions "
                "(id, agent_id, status, context_json, created_at, updated_at) "
                "VALUES (?, ?, 'active', '{}', ?, ?)",
                (sid, agent_id, now, now),
            )
        conn.commit()

    record_agent_weight_event("domain_study", "session_completed", source_kind="human", created_at=now)
    record_agent_weight_event("domain_practice", "dispatch_selected", source_kind="human", created_at=now)
    set_weight_override("domain_study", "suppress")
    set_weight_override("domain_practice", "pin")

    result = check_adjacency("domain_study", "domain_listening", "practise German")

    assert result is None
    paused = get_session(session_ids["domain_study"])
    assert paused.status == "paused"
    assert get_session(session_ids["domain_practice"]).status == "active"
    assert get_session(session_ids["domain_vocab"]).status == "active"
