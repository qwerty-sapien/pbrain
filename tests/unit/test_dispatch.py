# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for pb.core.dispatch_models and pb.core.dispatcher.

Tests cover:
  - DispatchDecision validation (confidence bounds)
  - InteractionEnvelope status values
  - DispatchSession default ID generation
  - Commitment default status
  - DispatchAgent default interaction_count
  - Phase 12 dispatch reranking and session ordering
  - _load_active_commitments (empty on fresh DB, returns dicts when populated)
  - dispatch is a coroutine function
  - create_session / get_session round-trip via protocol module

All tests use a temporary SQLite database (set via set_db_path + init_db).
LLM calls are mocked so no API keys are required.
"""

from __future__ import annotations

import asyncio
import inspect
import secrets
from datetime import datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from pb.core.dispatch_models import (
    Commitment,
    DispatchAgent,
    DispatchDecision,
    DispatchSession,
    InteractionEnvelope,
)
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
    # Reset global db path so it does not leak into other tests
    from pb.storage import database as db_module
    db_module._db_path = None


# ---------------------------------------------------------------------------
# DispatchDecision
# ---------------------------------------------------------------------------


def test_dispatch_decision_valid():
    """DispatchDecision accepts a valid decision with confidence in [0, 1]."""
    d = DispatchDecision(
        agent_id="capture",
        candidate_agent_ids=["capture", "todo"],
        confidence=0.85,
        in_scope=True,
        scope_reason="thought capture",
    )
    assert d.agent_id == "capture"
    assert d.candidate_agent_ids == ["capture", "todo"]
    assert d.confidence == 0.85
    assert d.in_scope is True


def test_dispatch_decision_confidence_lower_bound():
    """Confidence of exactly 0.0 is accepted."""
    d = DispatchDecision(agent_id="review", confidence=0.0, in_scope=True)
    assert d.confidence == 0.0


def test_dispatch_decision_confidence_upper_bound():
    """Confidence of exactly 1.0 is accepted."""
    d = DispatchDecision(agent_id="review", confidence=1.0, in_scope=True)
    assert d.confidence == 1.0


def test_dispatch_decision_confidence_too_high():
    """Confidence > 1.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        DispatchDecision(agent_id="review", confidence=1.1, in_scope=True)


def test_dispatch_decision_confidence_negative():
    """Confidence < 0.0 raises ValidationError."""
    with pytest.raises(ValidationError):
        DispatchDecision(agent_id="review", confidence=-0.1, in_scope=True)


def test_dispatch_decision_scope_reason_defaults_empty():
    """scope_reason defaults to empty string."""
    d = DispatchDecision(agent_id="capture", confidence=0.5, in_scope=True)
    assert d.scope_reason == ""


# ---------------------------------------------------------------------------
# InteractionEnvelope
# ---------------------------------------------------------------------------


def test_interaction_envelope_status_active():
    """InteractionEnvelope accepts status='active'."""
    e = InteractionEnvelope(session_id="abc", status="active")
    assert e.status == "active"


def test_interaction_envelope_status_complete():
    """InteractionEnvelope accepts status='complete'."""
    e = InteractionEnvelope(session_id="abc", status="complete")
    assert e.status == "complete"


def test_interaction_envelope_status_blocked():
    """InteractionEnvelope accepts status='blocked'."""
    e = InteractionEnvelope(session_id="abc", status="blocked")
    assert e.status == "blocked"


def test_interaction_envelope_status_error():
    """InteractionEnvelope accepts status='error'."""
    e = InteractionEnvelope(session_id="abc", status="error")
    assert e.status == "error"


def test_interaction_envelope_defaults():
    """InteractionEnvelope defaults: empty prompt, empty options, empty fields."""
    e = InteractionEnvelope(session_id="abc", status="active")
    assert e.prompt == ""
    assert e.options == []
    assert e.fields == {}


# ---------------------------------------------------------------------------
# DispatchSession
# ---------------------------------------------------------------------------


def test_dispatch_session_default_id_is_32_char_hex():
    """DispatchSession default ID is a 32-char hex string (token_hex(16))."""
    s = DispatchSession(agent_id="capture")
    assert len(s.id) == 32
    # Must be valid hex
    int(s.id, 16)


def test_dispatch_session_id_unique():
    """Two DispatchSession instances have different default IDs."""
    s1 = DispatchSession(agent_id="capture")
    s2 = DispatchSession(agent_id="capture")
    assert s1.id != s2.id


def test_dispatch_session_default_status_active():
    """DispatchSession defaults to status='active'."""
    s = DispatchSession(agent_id="review")
    assert s.status == "active"


def test_dispatch_session_default_context_json():
    """DispatchSession default context_json is '{}'."""
    s = DispatchSession(agent_id="review")
    assert s.context_json == "{}"


# ---------------------------------------------------------------------------
# Commitment
# ---------------------------------------------------------------------------


def test_commitment_default_status_active():
    """Commitment defaults to status='active'."""
    c = Commitment(description="Finish the report")
    assert c.status == "active"


def test_commitment_id_generated():
    """Commitment generates a non-empty ID by default."""
    c = Commitment(description="Finish the report")
    assert c.id
    assert isinstance(c.id, str)


def test_commitment_due_date_optional():
    """Commitment due_date defaults to None."""
    c = Commitment(description="Exercise daily")
    assert c.due_date is None


# ---------------------------------------------------------------------------
# DispatchAgent
# ---------------------------------------------------------------------------


def test_dispatch_agent_default_interaction_count_zero():
    """DispatchAgent interaction_count defaults to 0."""
    a = DispatchAgent(domain="german")
    assert a.interaction_count == 0


def test_dispatch_agent_id_generated():
    """DispatchAgent generates a non-empty ID by default."""
    a = DispatchAgent(domain="calculus")
    assert a.id
    assert isinstance(a.id, str)


def test_dispatch_agent_goal_id_optional():
    """DispatchAgent goal_id defaults to None."""
    a = DispatchAgent(domain="german")
    assert a.goal_id is None


# ---------------------------------------------------------------------------
# Dispatcher hooks
# ---------------------------------------------------------------------------


def test_dispatch_ranking_hook_prefers_weighted_agent_for_ambiguous_candidates(temp_db):
    """Low-confidence routing may flip to a higher-weight plausible candidate."""
    from pb.core.agent_weights import record_agent_weight_event
    from pb.core.dispatcher import _dispatch_ranking_hook

    record_agent_weight_event("accountability", "session_completed", source_kind="human")

    d = DispatchDecision(
        agent_id="review",
        candidate_agent_ids=["review", "accountability"],
        confidence=0.55,
        in_scope=True,
    )
    result = _dispatch_ranking_hook(d)
    assert result is d
    assert result.agent_id == "accountability"
    assert result.candidate_agent_ids[0] == "accountability"


def test_dispatch_ranking_hook_preserves_high_confidence_choice(temp_db):
    """High-confidence routing keeps the original LLM winner."""
    from pb.core.agent_weights import record_agent_weight_event
    from pb.core.dispatcher import _dispatch_ranking_hook

    record_agent_weight_event("accountability", "session_completed", source_kind="human")

    d = DispatchDecision(
        agent_id="review",
        candidate_agent_ids=["review", "accountability"],
        confidence=0.95,
        in_scope=True,
    )
    result = _dispatch_ranking_hook(d)
    assert result.agent_id == "review"
    assert result.candidate_agent_ids == ["review", "accountability"]


def test_resume_ordering_hook_sorts_by_weight_then_recency(temp_db):
    """_resume_ordering_hook orders sessions by weight before recency tie-breaks."""
    from pb.core.agent_weights import record_agent_weight_event
    from pb.core.dispatcher import _resume_ordering_hook

    record_agent_weight_event("review", "session_completed", source_kind="human")

    sessions = [
        DispatchSession(
            agent_id="accountability",
            updated_at=datetime.utcnow(),
        ),
        DispatchSession(
            agent_id="review",
            updated_at=datetime.utcnow() - timedelta(days=1),
        ),
    ]
    result = _resume_ordering_hook(sessions)
    assert result is not sessions
    assert len(result) == 2
    assert result[0].agent_id == "review"


def test_adjacency_suggestion_hook_adds_lowest_weight_pause_target(temp_db):
    """Blocked adjacency envelopes get a weight-aware session suggestion."""
    from pb.core.agent_weights import record_agent_weight_event
    from pb.core.dispatcher import _adjacency_suggestion_hook
    from pb.mcp.protocol import create_session

    review_session = create_session("review")
    accountability_session = create_session("accountability")
    record_agent_weight_event("accountability", "session_completed", source_kind="human")

    envelope = InteractionEnvelope(
        session_id="",
        status="blocked",
        prompt="Switch blocked.",
        options=["Finish current session"],
        fields={},
    )
    result = _adjacency_suggestion_hook(envelope)

    assert result.fields["suggested_session_id"] == review_session.id
    assert result.fields["suggested_agent_id"] == "review"
    assert result.options[0] == "Pause review and continue"
    assert accountability_session.id != review_session.id


def test_finish_pause_approval_hook_records_paused_signal(temp_db):
    """The finish/pause hook records lifecycle evidence for paused sessions."""
    from pb.core.dispatcher import _finish_pause_approval_hook
    from pb.mcp.protocol import create_session, update_session
    from pb.storage.database import get_connection

    session = create_session("review")
    update_session(session.id, status="paused")

    _finish_pause_approval_hook(session)

    with get_connection() as conn:
        row = conn.execute(
            """
            SELECT event_kind, source_kind
            FROM agent_weight_events
            WHERE agent_id = ? AND session_id = ?
            """,
            ("review", session.id),
        ).fetchone()

    assert row is not None
    assert row["event_kind"] == "session_paused"
    assert row["source_kind"] == "human"


def test_continuation_dispatch_does_not_call_fresh_ranking_hook(temp_db, monkeypatch):
    """Continuation path bypasses the fresh-dispatch reranking hook."""
    from pb.agents.base import AgentHandler
    from pb.core.dispatch_models import AgentOutput
    from pb.core.dispatcher import dispatch
    from pb.mcp.protocol import create_session

    class DummyAgent(AgentHandler):
        agent_id = "dummy"
        display_name = "Dummy"

        async def handle(self, intent, session, *, context=None):
            return AgentOutput(in_scope=True, response="ok", status="complete")

    session = create_session("dummy", context={"intent": "continue"})
    repo = MagicMock()
    repo.list_goal_arcs.return_value = []
    repo.list_tasks.return_value = []
    monkeypatch.setattr(
        "pb.agents.resolve_agent",
        lambda agent_id: DummyAgent() if agent_id == "dummy" else None,
    )
    monkeypatch.setattr(
        "pb.core.dispatcher._dispatch_ranking_hook",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("fresh ranking should not run")),
    )

    envelope = asyncio.run(dispatch(repo, "continue", session_id=session.id))
    assert envelope.status == "complete"


def test_fresh_dispatch_uses_routing_model_operation(temp_db, monkeypatch):
    """Fresh dispatch asks structured output for the routing/fast role."""
    from pb.core import dispatcher

    captured: dict[str, object] = {}

    async def fake_structured_output_call(*args, **kwargs):
        captured.update(kwargs)
        return DispatchDecision(
            agent_id="capture",
            candidate_agent_ids=["capture"],
            confidence=0.95,
            in_scope=True,
        )

    repo = MagicMock()
    repo.list_goal_arcs.return_value = []
    repo.list_tasks.return_value = []
    monkeypatch.setattr(dispatcher, "structured_output_call", fake_structured_output_call)
    monkeypatch.setattr("pb.agents.resolve_agent", lambda agent_id: None)

    asyncio.run(dispatcher.dispatch(repo, "what should I do now?"))

    assert captured["tier"] == "lite"
    assert captured["operation"] == "routing"


# ---------------------------------------------------------------------------
# _load_active_commitments
# ---------------------------------------------------------------------------


def test_load_active_commitments_empty_on_fresh_db(temp_db):
    """_load_active_commitments returns [] on a freshly initialised DB."""
    from pb.core.dispatcher import _load_active_commitments

    result = _load_active_commitments()
    assert result == []


def test_load_active_commitments_returns_dicts(temp_db):
    """_load_active_commitments returns list[dict] when commitments exist."""
    from pb.core.dispatcher import _load_active_commitments
    from pb.storage.database import get_connection

    now = datetime.utcnow().isoformat()
    session_id = secrets.token_hex(16)

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO dispatch_sessions "
            "(id, agent_id, status, context_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "accountability", "active", "{}", now, now),
        )
        conn.execute(
            "INSERT INTO commitments "
            "(id, description, created_at, due_date, status, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("cmt-001", "Finish the thesis", now, None, "active", session_id),
        )
        conn.commit()

    result = _load_active_commitments()
    assert len(result) == 1
    assert isinstance(result[0], dict)
    assert result[0]["id"] == "cmt-001"
    assert result[0]["description"] == "Finish the thesis"


def test_load_active_commitments_ignores_inactive(temp_db):
    """_load_active_commitments only returns status='active' rows."""
    from pb.core.dispatcher import _load_active_commitments
    from pb.storage.database import get_connection

    now = datetime.utcnow().isoformat()
    session_id = secrets.token_hex(16)

    with get_connection() as conn:
        conn.execute(
            "INSERT INTO dispatch_sessions "
            "(id, agent_id, status, context_json, created_at, updated_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (session_id, "accountability", "active", "{}", now, now),
        )
        conn.execute(
            "INSERT INTO commitments "
            "(id, description, created_at, due_date, status, session_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            ("cmt-002", "Completed already", now, None, "complete", session_id),
        )
        conn.commit()

    result = _load_active_commitments()
    assert result == []


# ---------------------------------------------------------------------------
# dispatch is a coroutine function
# ---------------------------------------------------------------------------


def test_dispatch_is_coroutine():
    """dispatch() must be declared async (inspect.iscoroutinefunction)."""
    from pb.core.dispatcher import dispatch

    assert inspect.iscoroutinefunction(dispatch)


# ---------------------------------------------------------------------------
# create_session / get_session round-trip
# ---------------------------------------------------------------------------


def test_create_and_get_session_roundtrip(temp_db):
    """create_session persists to SQLite; get_session retrieves same session."""
    from pb.mcp.protocol import create_session, get_session

    session = create_session("review", context={"intent": "review my day"})
    assert session.id
    assert session.agent_id == "review"
    assert session.status == "active"

    retrieved = get_session(session.id)
    assert retrieved is not None
    assert retrieved.id == session.id
    assert retrieved.agent_id == "review"
    assert retrieved.status == "active"


def test_get_session_nonexistent_returns_none(temp_db):
    """get_session returns None for a session ID that does not exist."""
    from pb.mcp.protocol import get_session

    result = get_session("0000000000000000000000000000ffff")
    assert result is None


def test_create_session_context_persisted(temp_db):
    """create_session serialises context dict to JSON in the DB."""
    import json

    from pb.mcp.protocol import create_session, get_session

    ctx = {"intent": "practise German", "kickback_count": 0}
    session = create_session("domain_german", context=ctx)

    retrieved = get_session(session.id)
    stored = json.loads(retrieved.context_json)
    assert stored["intent"] == "practise German"
    assert stored["kickback_count"] == 0


# ---------------------------------------------------------------------------
# deactivate_all_sessions
# ---------------------------------------------------------------------------


def test_deactivate_all_sessions_clears_active(temp_db):
    """deactivate_all_sessions() marks all active sessions as 'deactivated'."""
    from pb.mcp.protocol import create_session, deactivate_all_sessions, get_session

    s1 = create_session("review", context={"intent": "review my day"})
    s2 = create_session("capture", context={"intent": "buy milk"})

    count = deactivate_all_sessions()

    assert count == 2
    assert get_session(s1.id).status == "deactivated"
    assert get_session(s2.id).status == "deactivated"


def test_deactivate_all_sessions_ignores_complete(temp_db):
    """deactivate_all_sessions() does not touch sessions that are already complete."""
    from pb.mcp.protocol import create_session, deactivate_all_sessions, get_session, update_session

    active = create_session("review")
    done = create_session("capture")
    update_session(done.id, status="complete")

    count = deactivate_all_sessions()

    assert count == 1
    assert get_session(done.id).status == "complete"  # unchanged


def test_deactivate_all_sessions_returns_zero_when_none_active(temp_db):
    """deactivate_all_sessions() returns 0 on a fresh DB with no active sessions."""
    from pb.mcp.protocol import deactivate_all_sessions

    count = deactivate_all_sessions()
    assert count == 0


# ---------------------------------------------------------------------------
# _is_natural_language_input (NL detection gate)
# ---------------------------------------------------------------------------


def test_is_natural_language_input_multiword():
    """_is_natural_language_input returns True for multi-word input (>=3 tokens)."""
    from pb.cli.shell import _is_natural_language_input

    # Multi-word natural language — 3+ tokens
    assert _is_natural_language_input(["I", "am", "overwhelmed"], "I am overwhelmed") is True
    assert _is_natural_language_input(["what", "should", "I", "do"], "what should I do") is True


def test_is_natural_language_input_single_word_is_false(temp_db):
    """_is_natural_language_input returns False for single unknown words with no active session."""
    from pb.cli.shell import _is_natural_language_input

    # Single-word commands that are NOT NL prefixes
    assert _is_natural_language_input(["vim"], "vim") is False
    assert _is_natural_language_input(["test"], "test") is False


def test_is_natural_language_input_nl_prefix():
    """_is_natural_language_input returns True for NL trigger prefixes."""
    from pb.cli.shell import _is_natural_language_input

    # NL trigger prefix — 2 words but starts with "what "
    assert _is_natural_language_input(["what", "next"], "what next") is True
    # "how" prefix
    assert _is_natural_language_input(["how", "do", "I"], "how do I") is True
