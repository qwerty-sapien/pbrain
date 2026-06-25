# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AgentHandler ABC — contract all specialised agents implement (Phase 10)."""

from __future__ import annotations

import abc
from typing import Optional

import structlog

from pb.core.dispatch_models import AgentOutput, DispatchSession, InteractionEnvelope

logger = structlog.get_logger()


class AgentHandler(abc.ABC):
    """Abstract base class for all specialised dispatch agents.

    Every agent must declare its ``agent_id``, ``display_name``, and implement
    ``handle()`` to process a user intent within a session context.
    """

    agent_id: str
    display_name: str
    model_tier: str = "mid"  # lite | mid | heavy

    @abc.abstractmethod
    async def handle(
        self,
        intent: str,
        session: DispatchSession,
        *,
        context: Optional[dict] = None,
    ) -> AgentOutput:
        """Process user intent and return structured output.

        Args:
            intent: The raw user intent string.
            session: The current dispatch session record.
            context: Optional extra context dict (e.g. goal data, history).

        Returns:
            AgentOutput with in_scope flag, response, options, and status.
        """
        ...

    def to_envelope(
        self, session: DispatchSession, output: AgentOutput
    ) -> InteractionEnvelope:
        """Convert AgentOutput to the uniform InteractionEnvelope wire format.

        scope_reason stays in AgentOutput (internal).  It is intentionally NOT
        propagated to the envelope (T-10-03 mitigation).
        """
        return InteractionEnvelope(
            session_id=session.id,
            status=output.status,
            prompt=output.response,
            options=output.options,
            fields=output.fields,
        )
