# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Provider-agnostic structured output helper (Phase 10).

Provides a single async entry point that routes structured JSON calls through
the configured model roles. All LLM JSON is validated through a Pydantic model;
raw JSON is never returned.
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar

import structlog
from pydantic import BaseModel

from pb.core.model_policy import resolve_model_binding
from pb.llm.runtime import DraftGenerationError, LLMRuntime
from pb.storage.config import get_config

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

_TIER_TO_OPERATION: dict[str, str] = {
    "lite": "routing",
    "mid": "session_explain",
    "heavy": "lesson_planning",
}


async def structured_output_call(
    prompt: str,
    response_model: Type[T],
    *,
    system_prompt: str = "",
    tier: str = "lite",
    operation: str | None = None,
    model: str | None = None,
    timeout: int = 45,
    max_output_tokens: int = 4000,
) -> Optional[T]:
    """Call an LLM and return a validated Pydantic model instance.

    Uses the shared LLM runtime so provider selection follows configured
    model roles such as ``fast_inference`` for routing.

    Args:
        prompt: User-facing prompt text.
        response_model: Pydantic model class to validate the JSON against.
        system_prompt: Optional system/context prefix.
        tier: Compatibility tier — "lite" | "mid" | "heavy".
        operation: Optional model-policy operation; overrides tier mapping.
        model: Optional explicit provider:model binding.
        timeout: Request timeout in seconds.
        max_output_tokens: Maximum generated output tokens.

    Returns:
        Validated model instance, or None on persistent failure.
    """
    full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt
    operation_name = operation or _TIER_TO_OPERATION.get(tier, "routing")

    try:
        config = get_config()
        binding = model or resolve_model_binding(config, operation_name)
        draft = LLMRuntime(config).generate_draft(
            response_model,
            full_prompt,
            source_scope=f"structured_output:{operation_name}:{response_model.__name__}",
            model=binding,
            timeout=timeout,
            max_output_tokens=max_output_tokens,
        )
        payload = draft.payload
        if isinstance(payload, response_model):
            return payload
        return response_model.model_validate(payload)
    except DraftGenerationError as exc:
        logger.warning(
            "structured_output.generation_failed",
            operation=operation_name,
            model=model,
            category=exc.error.category,
            error=exc.error.raw_message,
        )
        return None
    except Exception as exc:
        logger.warning(
            "structured_output.error",
            operation=operation_name,
            model=model,
            error=str(exc),
        )
        return None
