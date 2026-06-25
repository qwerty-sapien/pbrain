# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Provider-agnostic structured output helper (Phase 10).

Provides a single async entry point that hides the differences between Gemini's
native ``response_schema`` and the OpenAI-compatible ``response_format`` path.
All LLM JSON is validated through a Pydantic model — raw JSON is never returned.
"""

from __future__ import annotations

from typing import Optional, Type, TypeVar

import structlog
from pydantic import BaseModel, ValidationError

from pb.llm.gemini import (
    FLASH_LITE_MODEL,
    FLASH_MODEL,
    PRO_MODEL,
    _should_retry_rate_limit,
    _stable_fallback_for,
    get_client,
)

logger = structlog.get_logger()

T = TypeVar("T", bound=BaseModel)

_TIER_TO_MODEL: dict[str, str] = {
    "lite": FLASH_LITE_MODEL,
    "mid": FLASH_MODEL,
    "heavy": PRO_MODEL,
}


async def structured_output_call(
    prompt: str,
    response_model: Type[T],
    *,
    system_prompt: str = "",
    tier: str = "lite",
) -> Optional[T]:
    """Call an LLM and return a validated Pydantic model instance.

    Uses Gemini's native response_schema when available; falls back to
    JSON-mode on OpenAI-compatible providers.

    Args:
        prompt: User-facing prompt text.
        response_model: Pydantic model class to validate the JSON against.
        system_prompt: Optional system/context prefix.
        tier: Model tier — "lite" | "mid" | "heavy".

    Returns:
        Validated model instance, or None on persistent failure.
    """
    model_id = _TIER_TO_MODEL.get(tier, FLASH_LITE_MODEL)
    full_prompt = f"{system_prompt}\n\n{prompt}".strip() if system_prompt else prompt

    gemini_client = get_client()
    if gemini_client.is_available():
        return await _gemini_structured_call(
            full_prompt, response_model, model_id=model_id
        )

    # Fallback: OpenAI-compatible provider
    return await _openai_compat_structured_call(
        full_prompt, response_model, model_id=model_id
    )


async def _gemini_structured_call(
    prompt: str,
    response_model: Type[T],
    *,
    model_id: str,
) -> Optional[T]:
    """Gemini structured output using native response_schema."""
    try:
        from google.genai import types  # type: ignore[import]
    except ImportError:
        logger.warning("structured_output.gemini_sdk_missing")
        return None

    client = get_client()
    if not client.is_available() or client._client is None:
        return None

    config = types.GenerateContentConfig(
        response_schema=response_model,
        response_mime_type="application/json",
    )

    last_exc: Optional[Exception] = None
    current_model = model_id

    for attempt in range(2):
        try:
            response = client._client.models.generate_content(
                model=current_model,
                contents=prompt,
                config=config,
            )
            text = response.text
            try:
                return response_model.model_validate_json(text)
            except ValidationError as ve:
                if attempt == 0:
                    # Retry once with validation error appended
                    prompt = (
                        f"{prompt}\n\nPrevious attempt failed validation: {ve}. "
                        "Please fix the JSON to match the required schema exactly."
                    )
                    continue
                logger.warning(
                    "structured_output.validation_failure",
                    model=current_model,
                    error=str(ve),
                )
                return None
        except Exception as exc:
            if _should_retry_rate_limit(exc):
                if attempt == 0:
                    fallback = _stable_fallback_for(current_model)
                    if fallback:
                        current_model = fallback
                        last_exc = exc
                        continue
                logger.warning(
                    "structured_output.rate_limit_exhausted",
                    model=current_model,
                    error=str(exc),
                )
                return None
            logger.warning(
                "structured_output.gemini_error",
                model=current_model,
                error=str(exc),
            )
            return None

    if last_exc is not None:
        logger.warning(
            "structured_output.rate_limit_exhausted",
            model=current_model,
            error=str(last_exc),
        )
    return None


async def _openai_compat_structured_call(
    prompt: str,
    response_model: Type[T],
    *,
    model_id: str,
) -> Optional[T]:
    """OpenAI-compatible structured output using json_schema response_format."""
    try:
        from pb.llm.runtime import OpenAICompatibleProviderClient  # type: ignore[attr-defined]
        from pb.storage.config import get_config
    except ImportError:
        logger.warning("structured_output.openai_compat_unavailable")
        return None

    try:
        cfg = get_config()
        provider_cfg = getattr(cfg, "provider", None)
        if provider_cfg is None:
            return None
        oc = OpenAICompatibleProviderClient(provider_cfg)
    except Exception:
        return None

    json_schema = response_model.model_json_schema()
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": response_model.__name__,
            "schema": json_schema,
            "strict": True,
        },
    }

    for attempt in range(2):
        try:
            raw = oc.generate_with_model(
                model_id,
                prompt,
                response_format=response_format,
            )
            if raw is None:
                return None
            try:
                return response_model.model_validate_json(raw)
            except ValidationError as ve:
                if attempt == 0:
                    prompt = (
                        f"{prompt}\n\nPrevious attempt failed validation: {ve}. "
                        "Please fix the JSON to match the required schema exactly."
                    )
                    continue
                logger.warning(
                    "structured_output.validation_failure",
                    model=model_id,
                    error=str(ve),
                )
                return None
        except Exception as exc:
            logger.warning(
                "structured_output.openai_compat_error",
                model=model_id,
                error=str(exc),
            )
            return None

    return None
