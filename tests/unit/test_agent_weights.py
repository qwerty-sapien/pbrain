# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for Phase 12 agent weight and frecency scoring."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pb.core.action_routing import build_next_candidates
from pb.core.agent_weights import (
    get_agent_weight_snapshot,
    record_agent_weight_event,
)
from pb.core.dispatch_models import InteractionEnvelope
from pb.mcp.protocol import create_session
from pb.storage.database import DB_FILENAME, get_connection, init_db, set_db_path


@pytest.fixture
def temp_db(tmp_path):
    """Isolated SQLite DB for each test."""
    db_path = tmp_path / DB_FILENAME
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    from pb.storage import database as db_module

    db_module._db_path = None


def _minimal_repo() -> MagicMock:
    repo = MagicMock()
    repo.get_active_session.return_value = None
    repo.list_due_action_reminders.return_value = []
    repo.list_tasks.return_value = []
    repo.list_time_blocks_for_date.return_value = []
    repo.list_sessions_in_range.return_value = []
    repo.list_goal_arcs.return_value = []
    return repo


def test_human_events_outrank_identical_agent_events(temp_db):
    """Human-source engagement should rank above equivalent agent chatter."""
    created_at = datetime.utcnow().isoformat()
    record_agent_weight_event("review", "session_completed", source_kind="human", created_at=created_at)
    record_agent_weight_event("accountability", "session_completed", source_kind="agent", created_at=created_at)

    review = get_agent_weight_snapshot("review")
    accountability = get_agent_weight_snapshot("accountability")

    assert review.frecency_score > accountability.frecency_score


def test_agent_frecency_scores_table_is_maintained(temp_db):
    """Phase 12 score rows are persisted in the requested frecency table."""
    created_at = datetime.utcnow().isoformat()
    snapshot = record_agent_weight_event("review", "session_completed", source_kind="human", created_at=created_at)

    with get_connection() as conn:
        row = conn.execute(
            "SELECT agent_id, frecency_score FROM agent_frecency_scores WHERE agent_id = ?",
            ("review",),
        ).fetchone()
        legacy = conn.execute(
            "SELECT agent_id, frecency_score FROM agent_weight_cache WHERE agent_id = ?",
            ("review",),
        ).fetchone()

    assert row is not None
    assert row["frecency_score"] == pytest.approx(snapshot.frecency_score)
    assert legacy is not None
    assert legacy["frecency_score"] == pytest.approx(snapshot.frecency_score)


def test_agent_source_uses_half_weight_multiplier(temp_db):
    """Agent-source event contributions count at exactly 0.5x the human equivalent."""
    created_at = datetime.utcnow().isoformat()
    record_agent_weight_event("review", "session_completed", source_kind="human", created_at=created_at)
    record_agent_weight_event("accountability", "session_completed", source_kind="agent", created_at=created_at)

    review = get_agent_weight_snapshot("review")
    accountability = get_agent_weight_snapshot("accountability")

    assert accountability.short_frecency == pytest.approx(review.short_frecency * 0.5, rel=1e-6)
    assert accountability.medium_frecency == pytest.approx(review.medium_frecency * 0.5, rel=1e-6)
    assert accountability.long_frecency == pytest.approx(review.long_frecency * 0.5, rel=1e-6)


def test_wrong_event_lowers_cached_score(temp_db):
    """Explicit `wrong` feedback should reduce the cached frecency score."""
    created_at = datetime.utcnow().isoformat()
    record_agent_weight_event("review", "session_completed", source_kind="human", created_at=created_at)
    before = get_agent_weight_snapshot("review")

    record_agent_weight_event("review", "wrong", source_kind="human", created_at=created_at)
    after = get_agent_weight_snapshot("review")

    assert after.frecency_score < before.frecency_score


def test_negative_events_lower_dispatch_prior(temp_db):
    """Pause/kickback-heavy histories should have a lower dispatch prior."""
    created_at = datetime.utcnow().isoformat()
    record_agent_weight_event("review", "session_completed", source_kind="human", created_at=created_at)

    record_agent_weight_event("accountability", "session_completed", source_kind="human", created_at=created_at)
    record_agent_weight_event("accountability", "kickback", source_kind="human", created_at=created_at)
    record_agent_weight_event("accountability", "session_paused", source_kind="human", created_at=created_at)

    review = get_agent_weight_snapshot("review")
    accountability = get_agent_weight_snapshot("accountability")

    assert accountability.dispatch_prior < review.dispatch_prior



def test_build_next_candidates_orders_commitments_by_linked_agent_weight(temp_db):
    """Commitment candidates should read and apply linked agent weight ordering."""
    now = datetime.utcnow().isoformat()
    record_agent_weight_event("accountability", "session_completed", source_kind="human", created_at=now)

    with get_connection() as conn:
        conn.execute(
            """
            INSERT INTO dispatch_sessions (id, agent_id, status, context_json, created_at, updated_at)
            VALUES (?, ?, 'active', '{}', ?, ?)
            """,
            ("sess-review", "review", now, now),
        )
        conn.execute(
            """
            INSERT INTO dispatch_sessions (id, agent_id, status, context_json, created_at, updated_at)
            VALUES (?, ?, 'active', '{}', ?, ?)
            """,
            ("sess-acct", "accountability", now, now),
        )
        conn.execute(
            """
            INSERT INTO commitments (id, description, created_at, due_date, status, session_id)
            VALUES (?, ?, ?, ?, 'active', ?)
            """,
            ("cmt-review", "Check the review thread", now, now, "sess-review"),
        )
        conn.execute(
            """
            INSERT INTO commitments (id, description, created_at, due_date, status, session_id)
            VALUES (?, ?, ?, ?, 'active', ?)
            """,
            ("cmt-acct", "Check the accountability thread", now, now, "sess-acct"),
        )
        conn.commit()

    candidates = build_next_candidates(_minimal_repo(), limit=5)
    commitment_candidates = [candidate for candidate in candidates if candidate.source == "commitment"]

    assert len(commitment_candidates) == 2
    assert commitment_candidates[0].agent_id == "accountability"
