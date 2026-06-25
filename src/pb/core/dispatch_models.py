# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Pydantic models for the agentic dispatch subsystem (Phase 10).

These types form the wire contract between the dispatcher, specialised agents,
the MCP interaction protocol, and the CLI pickers.  All LLM JSON is validated
through these models — never trusted raw.
"""

from __future__ import annotations

import secrets
from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field

from pb.core.models import generate_internal_id, utc_now


class DispatchDecision(BaseModel):
    """Routing decision produced by the LLM dispatcher."""

    agent_id: str = Field(description="ID of the agent to dispatch to")
    candidate_agent_ids: list[str] = Field(
        default_factory=list,
        description="Ordered plausible agent choices used for Phase 12 reranking",
    )
    confidence: float = Field(ge=0.0, le=1.0, description="Routing confidence")
    in_scope: bool = Field(description="Whether input is in scope for selected agent")
    scope_reason: str = Field(default="", description="Internal only - never shown to user")


class AgentOutput(BaseModel):
    """Structured output returned by every specialised agent."""

    in_scope: bool = Field(description="False if intent is outside this agent's domain")
    scope_reason: str = Field(default="", description="Internal - never shown to user")
    misalignment_signal: str = Field(
        default="",
        description="Phase 12 hook; recorded, not acted on",
    )
    response: str = Field(description="The agent's natural-language response to the user")
    options: list[str] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)
    status: str = Field(default="complete", description="active | complete | blocked | error")


class InteractionEnvelope(BaseModel):
    """Uniform wire envelope returned across all surfaces (CLI, MCP, swarm)."""

    session_id: str
    status: str = Field(description="active | complete | blocked | error")
    prompt: str = Field(default="")
    options: list[str] = Field(default_factory=list)
    fields: dict[str, str] = Field(default_factory=dict)


class DispatchSession(BaseModel):
    """Persisted session record for a single dispatch interaction thread."""

    id: str = Field(default_factory=lambda: secrets.token_hex(16))
    agent_id: str
    status: str = "active"
    context_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)
    updated_at: datetime = Field(default_factory=utc_now)
    judged: bool = False
    judged_at: Optional[datetime] = None


class Commitment(BaseModel):
    """Durable accountability commitment record."""

    id: str = Field(default_factory=generate_internal_id)
    description: str
    created_at: datetime = Field(default_factory=utc_now)
    due_date: Optional[str] = None
    status: str = "active"
    session_id: Optional[str] = None


class DispatchAgent(BaseModel):
    """Registered specialised agent descriptor."""

    id: str = Field(default_factory=generate_internal_id)
    domain: str
    goal_id: Optional[str] = None
    config_json: str = "{}"
    created_at: datetime = Field(default_factory=utc_now)
    interaction_count: int = 0
