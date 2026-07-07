# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for pb/agents/ — agent registration, attributes, and envelope conversion (Phase 10).

Tests cover:
  - All 4 surviving agents registered in the registry after import
    (accountability, spawner, review, domain — capture moved to memo/)
  - AccountabilityAgent attributes (agent_id, model_tier)
  - SpawnerAgent attributes (agent_id)
  - ReviewAgent attributes (agent_id)
  - DomainAgent factory (create_domain_agent registers under given ID)
  - AgentHandler.to_envelope produces correct InteractionEnvelope from AgentOutput

LLM calls are mocked so no API keys are required.
"""

from __future__ import annotations

import pytest

from pb.core.dispatch_models import AgentOutput, DispatchSession, InteractionEnvelope
from pb.storage.database import DB_FILENAME, init_db, set_db_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def _clean_registry():
    """Snapshot/restore agent registry for tests that register extra agents."""
    from pb.agents import _AGENT_REGISTRY
    snapshot = dict(_AGENT_REGISTRY)
    yield
    _AGENT_REGISTRY.clear()
    _AGENT_REGISTRY.update(snapshot)


@pytest.fixture
def temp_db(tmp_path):
    """Isolated SQLite DB for each test."""
    db_path = tmp_path / DB_FILENAME
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    from pb.storage import database as db_module
    db_module._db_path = None


# ---------------------------------------------------------------------------
# Ensure surviving agents are imported (which triggers registry side-effects)
# ---------------------------------------------------------------------------


def _import_surviving_agents():
    """Import surviving agent modules to trigger their register_agent() side-effects.

    Note: pb.agents.capture was moved to memo/ in phase 14 and is no longer
    imported here. Its tests now live in memo/tests/.
    """
    import pb.agents.accountability  # noqa: F401
    import pb.agents.domain  # noqa: F401
    import pb.agents.review  # noqa: F401
    import pb.agents.spawner  # noqa: F401


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


def test_accountability_agent_registered():
    """AccountabilityAgent is registered under 'accountability' after module import."""
    from pb.agents import resolve_agent
    import pb.agents.accountability  # noqa: F401

    agent = resolve_agent("accountability")
    assert agent is not None
    assert agent.agent_id == "accountability"


def test_spawner_agent_registered():
    """SpawnerAgent is registered under 'spawner' after module import."""
    from pb.agents import resolve_agent
    import pb.agents.spawner  # noqa: F401

    agent = resolve_agent("spawner")
    assert agent is not None
    assert agent.agent_id == "spawner"


def test_review_agent_registered():
    """ReviewAgent is registered under 'review' after module import."""
    from pb.agents import resolve_agent
    import pb.agents.review  # noqa: F401

    agent = resolve_agent("review")
    assert agent is not None
    assert agent.agent_id == "review"


def test_all_surviving_builtin_agents_registered():
    """All three auto-registering built-in agents appear in the registry.

    capture was moved to memo/ in phase 14 and is no longer a pb built-in.
    """
    from pb.agents import list_agents
    _import_surviving_agents()

    registered = list_agents()
    for expected_id in ("accountability", "spawner", "review"):
        assert expected_id in registered, f"Agent '{expected_id}' not in registry"

    # capture must NOT be in pb's registry — it lives in memo/
    assert "capture" not in registered, "capture agent must not be in pb registry (moved to memo/)"


# ---------------------------------------------------------------------------
# AccountabilityAgent attributes
# ---------------------------------------------------------------------------


def test_accountability_agent_model_tier_is_mid():
    """AccountabilityAgent.model_tier should be 'mid'."""
    from pb.agents import resolve_agent
    import pb.agents.accountability  # noqa: F401

    agent = resolve_agent("accountability")
    assert agent.model_tier == "mid"


def test_accountability_agent_display_name():
    """AccountabilityAgent.display_name is 'Accountability'."""
    from pb.agents import resolve_agent
    import pb.agents.accountability  # noqa: F401

    agent = resolve_agent("accountability")
    assert agent.display_name == "Accountability"


# ---------------------------------------------------------------------------
# SpawnerAgent attributes
# ---------------------------------------------------------------------------


def test_spawner_agent_id():
    """SpawnerAgent.agent_id is 'spawner'."""
    from pb.agents import resolve_agent
    import pb.agents.spawner  # noqa: F401

    agent = resolve_agent("spawner")
    assert agent.agent_id == "spawner"


def test_spawner_agent_display_name():
    """SpawnerAgent.display_name is 'Agent Spawner'."""
    from pb.agents import resolve_agent
    import pb.agents.spawner  # noqa: F401

    agent = resolve_agent("spawner")
    assert agent.display_name == "Agent Spawner"


# ---------------------------------------------------------------------------
# ReviewAgent attributes
# ---------------------------------------------------------------------------


def test_review_agent_id():
    """ReviewAgent.agent_id is 'review'."""
    from pb.agents import resolve_agent
    import pb.agents.review  # noqa: F401

    agent = resolve_agent("review")
    assert agent.agent_id == "review"


def test_review_agent_model_tier():
    """ReviewAgent.model_tier is 'mid'."""
    from pb.agents import resolve_agent
    import pb.agents.review  # noqa: F401

    agent = resolve_agent("review")
    assert agent.model_tier == "mid"


# ---------------------------------------------------------------------------
# DomainAgent factory
# ---------------------------------------------------------------------------


def test_create_domain_agent_registers_under_given_id(_clean_registry):
    """create_domain_agent() creates a DomainAgent and registers it in the registry."""
    from pb.agents import resolve_agent
    from pb.agents.domain import create_domain_agent

    agent = create_domain_agent(
        agent_id="domain_german",
        display_name="German B1",
        domain="german",
        goal_id="goal-001",
    )

    assert agent is not None
    assert agent.agent_id == "domain_german"
    assert agent.domain == "german"
    assert agent.goal_id == "goal-001"

    resolved = resolve_agent("domain_german")
    assert resolved is agent


def test_create_domain_agent_without_goal_id(_clean_registry):
    """create_domain_agent() works without a goal_id."""
    from pb.agents.domain import create_domain_agent

    agent = create_domain_agent(
        agent_id="domain_calculus",
        display_name="Calculus",
        domain="calculus",
    )
    assert agent.goal_id is None


# ---------------------------------------------------------------------------
# AgentHandler.to_envelope (tested via AccountabilityAgent — generic base behaviour)
# ---------------------------------------------------------------------------


def test_to_envelope_maps_output_to_envelope():
    """AgentHandler.to_envelope produces a correct InteractionEnvelope.

    Ported from the original CaptureAgent version — uses AccountabilityAgent
    since capture was moved to memo/. The to_envelope logic lives in AgentHandler
    base and is agent-agnostic.
    """
    from pb.agents import resolve_agent
    import pb.agents.accountability  # noqa: F401

    agent = resolve_agent("accountability")
    session = DispatchSession(agent_id="accountability")
    output = AgentOutput(
        in_scope=True,
        response="Commitment recorded: finish report",
        options=["View commitments", "Cancel"],
        fields={"commitment_type": "work"},
        status="complete",
    )

    envelope = agent.to_envelope(session, output)

    assert isinstance(envelope, InteractionEnvelope)
    assert envelope.session_id == session.id
    assert envelope.status == "complete"
    assert envelope.prompt == "Commitment recorded: finish report"
    assert envelope.options == ["View commitments", "Cancel"]
    assert envelope.fields == {"commitment_type": "work"}


def test_to_envelope_scope_reason_not_in_envelope():
    """scope_reason from AgentOutput must NOT appear in InteractionEnvelope (T-10-03).

    Ported from the original CaptureAgent version — uses AccountabilityAgent
    since capture was moved to memo/. The scope_reason guard lives in AgentHandler
    base and applies to all agents.
    """
    from pb.agents import resolve_agent
    import pb.agents.accountability  # noqa: F401

    agent = resolve_agent("accountability")
    session = DispatchSession(agent_id="accountability")
    output = AgentOutput(
        in_scope=False,
        scope_reason="Internal routing note — never show to user",
        response="",
        status="complete",
    )

    envelope = agent.to_envelope(session, output)

    # scope_reason must not appear anywhere in the envelope
    assert not hasattr(envelope, "scope_reason")
    envelope_dict = envelope.model_dump()
    assert "scope_reason" not in envelope_dict
    # Also must not appear in the prompt text
    assert "Internal routing note" not in (envelope.prompt or "")
