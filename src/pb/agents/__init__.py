# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Agent registry for the dispatch subsystem (Phase 10).

Provides register/resolve/list helpers that mirror the pb/mcp/pending.py
pattern but for AgentHandler instances rather than MCP tool implementations.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pb.agents.base import AgentHandler

_AGENT_REGISTRY: dict[str, "AgentHandler"] = {}
_AGENTS_LOADED = False


def register_agent(agent_id: str, handler: "AgentHandler") -> None:
    """Register a specialised agent handler under its agent_id."""
    _AGENT_REGISTRY[agent_id] = handler


def _ensure_agents_loaded() -> None:
    """Import all built-in agent modules so they self-register."""
    global _AGENTS_LOADED
    if _AGENTS_LOADED:
        return
    _AGENTS_LOADED = True
    import pb.agents.accountability  # noqa: F401
    import pb.agents.spawner  # noqa: F401
    import pb.agents.review  # noqa: F401
    import pb.agents.domain  # noqa: F401


def resolve_agent(agent_id: str) -> Optional["AgentHandler"]:
    """Look up a registered agent by ID.  Returns None if not found."""
    _ensure_agents_loaded()
    return _AGENT_REGISTRY.get(agent_id)


def list_agents() -> list[str]:
    """Return sorted list of registered agent IDs."""
    return sorted(_AGENT_REGISTRY.keys())
