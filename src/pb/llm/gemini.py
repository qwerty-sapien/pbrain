# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Gemini Flash Lite integration for review scoring.

Uses google-genai SDK (not deprecated google-generativeai).
Supports both AI Studio (GEMINI_API_KEY) and Vertex AI (GOOGLE_CLOUD_PROJECT).
Per D-10: Follow up on ALL custom text responses.
Per D-11: Maximum 1 follow-up round.
Per D-12: Assign 1-10 score with brief rationale.
Per D-13: If API unavailable, caller falls back to numeric input.
"""

from __future__ import annotations

import os
import random
import re
import sys
import time
from dataclasses import dataclass
from typing import Optional

import structlog

from pb.core.error_logging import log_error
from pb.events import emit as _emit_event
from pb.llm.policy import resolve_timeout, slow_thinking_notice

logger = structlog.get_logger()

# Model to use for scoring - fast and cost-effective
MODEL_ID = "gemini-3.1-flash-lite-preview"

# Model tier constants for multi-tier generation
FLASH_LITE_MODEL = "gemini-3.1-flash-lite-preview"
FLASH_MODEL = "gemini-3-flash-preview"
PRO_MODEL = "gemini-3.1-pro-preview"

# Env var hint shown when no credentials are configured
_CREDS_HINT = "set GEMINI_API_KEY (AI Studio) or GOOGLE_CLOUD_PROJECT (Vertex AI)"

MODEL_ALIASES = {
    "flash-lite": FLASH_LITE_MODEL,
    "flash_lite": FLASH_LITE_MODEL,
    "flash": FLASH_MODEL,
    "pro": PRO_MODEL,
}

# Preview -> stable GA fallback, used ONLY on a gemini rate-limit (429 /
# RESOURCE_EXHAUSTED). The free preview tiers have very tight quota; the stable
# GA endpoints carry real/separate quota, so on a 429 we retry on stable rather
# than blacking out. This is gemini-specific and applies to BOTH AI Studio and
# Vertex (same model IDs) — non-gemini providers never reach this client.
#
# IDs verified against https://ai.google.dev/gemini-api/docs/models on
# 2026-06-03: gemini-3.5-flash (GA 2026-05-19) and gemini-3.1-flash-lite are GA;
# no stable 3.x Pro endpoint exists, so the latest stable Pro is gemini-2.5-pro
# (Pro-tier fallback user-confirmed 2026-06-03). The preview constants above stay
# the locked primaries — this map is a 429 safety net, not a default change.
_RATE_LIMIT_STABLE_FALLBACK = {
    FLASH_MODEL: "gemini-3.5-flash",
    FLASH_LITE_MODEL: "gemini-3.1-flash-lite",
    PRO_MODEL: "gemini-2.5-pro",
}


def _stable_fallback_for(model: str) -> Optional[str]:
    """Return the stable GA model to retry on after a gemini rate-limit, else None.

    Only the gemini preview constants map to a stable fallback. An already-stable
    model, or any non-gemini model, returns None (no further upgrade).
    """
    return _RATE_LIMIT_STABLE_FALLBACK.get((model or "").strip())


@dataclass
class GeminiErrorDetails:
    """Normalized Gemini provider failure details."""

    category: str
    raw_message: str
    http_status: int | None = None
    retryable: bool = False


@dataclass
class TokenBumpInfo:
    """Records that the MAX_TOKENS auto-retry was triggered."""

    succeeded: bool
    final_tokens: int
    original_tokens: int
    model: str


@dataclass
class GeminiGenerationResult:
    """Detailed Gemini generation outcome."""

    text: str | None = None
    error: GeminiErrorDetails | None = None
    token_bump: TokenBumpInfo | None = None


def resolve_model(model: Optional[str], fallback: str = FLASH_MODEL) -> str:
    """Resolve a user-provided model name or shorthand alias."""
    raw = (model or "").strip()
    if not raw:
        return fallback
    return MODEL_ALIASES.get(raw.lower(), raw)


def _is_vertex_configured() -> bool:
    """Return True when Vertex credentials should be used."""
    return bool(os.environ.get("GOOGLE_CLOUD_PROJECT"))


def _vertex_location() -> str:
    """Return the configured Vertex region, defaulting to global."""
    return os.environ.get("GOOGLE_CLOUD_LOCATION", "global")


def _create_genai_client():
    """Create a google-genai Client using Vertex AI or AI Studio credentials.

    Priority: GOOGLE_CLOUD_PROJECT (Vertex AI) > GEMINI_API_KEY (AI Studio).
    """
    from google import genai

    project = os.environ.get("GOOGLE_CLOUD_PROJECT")
    if project:
        client = genai.Client(vertexai=True, project=project, location=_vertex_location())
        logger.debug("gemini.client", backend="vertex", project=project, location=_vertex_location())
        return client

    api_key = os.environ.get("GEMINI_API_KEY")
    if api_key:
        client = genai.Client(api_key=api_key)
        logger.debug("gemini.client", backend="aistudio")
        return client

    return None


_BACKOFF_BASE_SECONDS = 1.5
_BACKOFF_CAP_SECONDS = 30.0
_BACKOFF_RNG = random.Random()


def _retry_backoff_seconds(
    attempt: int,
    *,
    base: float = _BACKOFF_BASE_SECONDS,
    cap: float = _BACKOFF_CAP_SECONDS,
    rng: "random.Random | None" = None,
) -> float:
    """Exponential backoff with equal jitter for 429/RESOURCE_EXHAUSTED retries.

    The delay band doubles each attempt (``base * 2**(attempt-1)``), capped at
    ``cap``. Equal jitter keeps a positive floor (half the band) so we still back
    off meaningfully, while randomising the upper half so parallel clients that
    share one quota do not retry in lockstep (thundering herd). ``attempt`` is
    1-based.
    """
    r = rng if rng is not None else _BACKOFF_RNG
    exp = base * (2 ** max(0, attempt - 1))
    temp = min(cap, exp)
    half = temp / 2.0
    return half + r.uniform(0.0, half)


def _should_retry_rate_limit(exc: Exception) -> bool:
    """Return True for transient rate-limit or capacity errors."""
    message = str(exc).lower()
    markers = (
        "429",
        "resource_exhausted",
        "rate limit",
        "quota",
        "too many requests",
    )
    return any(marker in message for marker in markers)


def _extract_http_status(exc: Exception) -> int | None:
    message = str(exc)
    match = re.search(r"\b([1-5]\d\d)\b", message)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _classify_error(exc: Exception) -> GeminiErrorDetails:
    message = str(exc).strip() or exc.__class__.__name__
    lower = message.lower()
    status = _extract_http_status(exc)

    if "timed out" in lower or "timeout" in lower:
        return GeminiErrorDetails("timeout", message, http_status=status, retryable=True)
    if status in {401, 403} or any(marker in lower for marker in ("permission", "unauthorized", "forbidden", "api key", "authentication", "credentials")):
        return GeminiErrorDetails("auth", message, http_status=status, retryable=False)
    if status in {400, 404} or any(marker in lower for marker in ("not found", "unknown model", "invalid argument", "bad request", "unsupported model")):
        return GeminiErrorDetails("config", message, http_status=status, retryable=False)
    if status == 429 or any(marker in lower for marker in ("resource_exhausted", "rate limit", "quota", "credits", "too many requests")):
        return GeminiErrorDetails("quota", message, http_status=status or 429, retryable=True)
    if status is not None and status >= 500:
        return GeminiErrorDetails("upstream", message, http_status=status, retryable=True)
    if any(
        marker in lower
        for marker in (
            "name or service not known",
            "nodename nor servname provided",
            "temporary failure in name resolution",
            "dns",
            "connection reset",
            "connection refused",
            "network is unreachable",
            "failed to establish a new connection",
        )
    ):
        return GeminiErrorDetails("network", message, http_status=status, retryable=True)
    return GeminiErrorDetails("unknown", message, http_status=status, retryable=False)


def _response_snapshot(response: object | None) -> object:
    if response is None:
        return None
    for method_name in ("model_dump", "to_json_dict"):
        method = getattr(response, method_name, None)
        if callable(method):
            try:
                return method()
            except Exception:
                pass
    json_method = getattr(response, "model_dump_json", None)
    if callable(json_method):
        try:
            return json_method()
        except Exception:
            pass
    text = getattr(response, "text", None)
    if text is not None:
        return {"text": text}
    return repr(response)


def _enum_label(value: object | None) -> str:
    if value is None:
        return ""
    raw = getattr(value, "name", None) or getattr(value, "value", None) or value
    text = str(raw).strip()
    for prefix in (
        "FINISH_REASON_",
        "BLOCKED_REASON_",
        "HARM_CATEGORY_",
        "HARM_PROBABILITY_",
        "HARM_SEVERITY_",
    ):
        if text.startswith(prefix):
            text = text[len(prefix):]
    return text.lower().replace("_", " ")


def _collect_text_parts(parts: object | None) -> str:
    if not isinstance(parts, list):
        return ""
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if text is None:
            continue
        rendered = str(text).strip()
        if rendered:
            texts.append(rendered)
    return "\n".join(texts).strip()


def _extract_text_from_response(response: object | None) -> str:
    if response is None:
        return ""
    top_level_parts = getattr(response, "parts", None)
    recovered = _collect_text_parts(top_level_parts)
    if recovered:
        return recovered
    candidates = getattr(response, "candidates", None) or []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        parts = getattr(content, "parts", None)
        recovered = _collect_text_parts(parts)
        if recovered:
            return recovered
    return ""


def _summarize_usage_metadata(usage: object | None) -> str:
    if usage is None:
        return ""
    items: list[str] = []
    for attr in ("prompt_token_count", "candidates_token_count", "total_token_count"):
        value = getattr(usage, attr, None)
        if value is not None:
            label = attr.replace("_token_count", "")
            items.append(f"{label}={value}")
    return f"usage({', '.join(items)})" if items else ""


def _summarize_safety_ratings(ratings: object | None) -> str:
    if not isinstance(ratings, list):
        return ""
    items: list[str] = []
    for rating in ratings[:3]:
        category = _enum_label(getattr(rating, "category", None))
        if not category:
            continue
        blocked = getattr(rating, "blocked", None)
        probability = _enum_label(getattr(rating, "probability", None))
        entry = category
        if blocked:
            entry += " blocked"
        elif probability and probability != "unspecified":
            entry += f" {probability}"
        items.append(entry)
    return ", ".join(items)


def _candidate_part_kinds(candidate: object | None) -> list[str]:
    content = getattr(candidate, "content", None)
    parts = getattr(content, "parts", None)
    if not isinstance(parts, list):
        return []
    kinds: list[str] = []
    for part in parts:
        for attr in (
            "function_call",
            "function_response",
            "code_execution_result",
            "executable_code",
            "inline_data",
            "file_data",
            "tool_call",
            "tool_response",
            "part_metadata",
        ):
            value = getattr(part, attr, None)
            if value is not None and attr not in kinds:
                kinds.append(attr)
    return kinds


def _describe_empty_response(response: object | None) -> str:
    if response is None:
        return (
            "Gemini returned no usable text in an HTTP 200 response. "
            "This is usually not a credits issue; quota failures typically surface as 429/RESOURCE_EXHAUSTED instead."
        )

    details: list[str] = []
    prompt_feedback = getattr(response, "prompt_feedback", None)
    block_reason = _enum_label(getattr(prompt_feedback, "block_reason", None))
    block_reason_message = str(getattr(prompt_feedback, "block_reason_message", "") or "").strip()
    if block_reason and block_reason != "unspecified":
        detail = f"prompt blocked: {block_reason}"
        if block_reason_message:
            detail += f" ({block_reason_message})"
        details.append(detail)
    elif block_reason_message:
        details.append(block_reason_message)

    prompt_safety = _summarize_safety_ratings(getattr(prompt_feedback, "safety_ratings", None))
    if prompt_safety:
        details.append(f"prompt safety={prompt_safety}")

    candidates = getattr(response, "candidates", None) or []
    if candidates:
        candidate = candidates[0]
        finish_reason = _enum_label(getattr(candidate, "finish_reason", None))
        finish_message = str(getattr(candidate, "finish_message", "") or "").strip()
        if finish_reason and finish_reason != "unspecified":
            details.append(f"finish reason={finish_reason}")
        if finish_message:
            details.append(finish_message)
        part_kinds = _candidate_part_kinds(candidate)
        if part_kinds:
            details.append(f"candidate parts={', '.join(part_kinds)}")
        candidate_safety = _summarize_safety_ratings(getattr(candidate, "safety_ratings", None))
        if candidate_safety:
            details.append(f"candidate safety={candidate_safety}")

    usage = _summarize_usage_metadata(getattr(response, "usage_metadata", None))
    if usage:
        details.append(usage)

    suffix = " ".join(detail for detail in details if detail).strip()
    if suffix:
        suffix = f" {suffix}"
    return (
        "Gemini returned no usable text in an HTTP 200 response."
        f"{suffix} "
        "This is usually not a credits issue; quota failures typically surface as 429/RESOURCE_EXHAUSTED instead."
    ).strip()


def _is_max_tokens_finish(response: object | None) -> bool:
    """Return True if the response was truncated by hitting the MAX_TOKENS output limit."""
    if response is None:
        return False
    candidates = getattr(response, "candidates", None) or []
    if not candidates:
        return False
    finish_reason = _enum_label(getattr(candidates[0], "finish_reason", None))
    return finish_reason == "max tokens"


def _log_gemini_failure(
    *,
    event: str,
    model: str,
    prompt: str,
    exc: Exception | None = None,
    response: object | None = None,
    status: int | None = None,
    extra: dict[str, object] | None = None,
) -> None:
    try:
        log_error(
            event=event,
            message=str(exc) if exc is not None else event,
            exc=exc,
            command="gemini",
            status=status,
            request_body={
                "provider": "gemini",
                "model": model,
                "prompt": prompt,
            },
            response_status=status,
            response_body=_response_snapshot(response),
            extra=extra,
        )
    except Exception:
        pass


class GeminiClient:
    """
    Wrapper for Gemini API client.

    Gracefully degrades when credentials not configured (per D-13).
    """

    def __init__(self):
        self._client: Optional[object] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Check if Gemini API is available."""
        if self._available is not None:
            return self._available

        try:
            client = _create_genai_client()
            if client is None:
                logger.debug("gemini.check", available=False, reason="no_credentials")
                self._available = False
                return False
            self._client = client
            self._available = True
            logger.debug("gemini.check", available=True)
            return True
        except ImportError:
            logger.debug("gemini.check", available=False, reason="sdk_not_installed")
            self._available = False
            return False
        except Exception as e:
            logger.debug("gemini.check", available=False, reason=str(e))
            self._available = False
            return False

    def generate(self, prompt: str) -> Optional[str]:
        """
        Generate text using Gemini.

        Returns None if client not available or API call fails.
        """
        if not self.is_available() or self._client is None:
            return None

        try:
            response = self._client.models.generate_content(
                model=MODEL_ID,
                contents=prompt,
            )
            return response.text
        except Exception as e:
            _log_gemini_failure(
                event="gemini.generate_exception",
                model=MODEL_ID,
                prompt=prompt,
                exc=e,
                status=_extract_http_status(e),
            )
            logger.debug("gemini.generate", error=str(e))
            return None

    def _build_generation_config(
        self,
        model: str,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """Build an SDK config object with Vertex-aware thinking settings."""
        try:
            from google.genai import types
        except ImportError:
            return None

        kwargs: dict[str, object] = {}
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["maxOutputTokens"] = max_output_tokens

        resolved = resolve_model(model, fallback=FLASH_MODEL)
        lower = resolved.lower()

        # Gemini 3 on Vertex uses thinking levels; keep Flash and Pro at HIGH by default.
        # Skip thinking for tiny output budgets (e.g. probes) — thinking tokens
        # consume the budget and leave no room for visible text.
        tiny_budget = max_output_tokens is not None and max_output_tokens < 64
        if _is_vertex_configured() and "flash-lite" not in lower and not tiny_budget:
            if lower.startswith("gemini-3") and ("flash" in lower or "pro" in lower):
                kwargs["thinkingConfig"] = types.ThinkingConfig(
                    thinkingLevel=types.ThinkingLevel.HIGH
                )
            elif lower.startswith("gemini-2.5-") and ("flash" in lower or "pro" in lower):
                budget = 32768 if "pro" in lower else 24576
                kwargs["thinkingConfig"] = types.ThinkingConfig(
                    thinkingBudget=budget
                )

        return types.GenerateContentConfig(**kwargs) if kwargs else None

    def _build_structured_generation_config(
        self,
        schema_cls: type,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ):
        """Build an SDK config object for native structured output."""
        try:
            from google.genai import types
        except ImportError:
            return None

        kwargs: dict[str, object] = {
            "response_schema": schema_cls,
            "response_mime_type": "application/json",
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        if max_output_tokens is not None:
            kwargs["maxOutputTokens"] = max_output_tokens
        return types.GenerateContentConfig(**kwargs)

    def generate_with_model(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> Optional[str]:
        result = self.generate_with_model_result(
            prompt,
            model,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        return result.text

    def _generate_with_config_builder(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        config_builder=None,
    ) -> GeminiGenerationResult:
        req_start = time.monotonic()
        if not self.is_available() or self._client is None:
            return GeminiGenerationResult(
                error=GeminiErrorDetails(
                    category="auth",
                    raw_message=f"Gemini is not available; {_CREDS_HINT}.",
                    http_status=None,
                    retryable=False,
                )
            )
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

            resolved_model = resolve_model(model, fallback=FLASH_MODEL)
            requested_timeout = timeout
            current_model = resolved_model
            timeout = resolve_timeout("gemini", current_model, requested_timeout)
            config = config_builder(current_model) if callable(config_builder) else None
            _emit_event(
                "llm.request.started",
                model=current_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
                timeout=timeout,
            )

            def _call(call_model, call_config):
                kwargs = {
                    "model": call_model,
                    "contents": prompt,
                }
                if call_config is not None:
                    kwargs["config"] = call_config
                return self._client.models.generate_content(**kwargs)

            attempts = 6
            fallback_used = False
            _token_retries = 0
            _original_max_tokens = max_output_tokens or 8192
            _current_max_tokens = _original_max_tokens
            for attempt in range(1, attempts + 1):
                try:
                    with ThreadPoolExecutor(max_workers=1) as pool:
                        future = pool.submit(_call, current_model, config)
                        with slow_thinking_notice("gemini", current_model):
                            response = future.result(timeout=timeout)
                    text = getattr(response, "text", None)
                    if text is None or not str(text).strip():
                        recovered = _extract_text_from_response(response)
                        if recovered:
                            logger.debug(
                                "gemini.generate_with_model.recovered_text_parts",
                                model=current_model,
                                attempt=attempt,
                            )
                            _emit_event("llm.request.finished", model=current_model, status="ok", duration=time.monotonic() - req_start)
                            bump = TokenBumpInfo(True, _current_max_tokens, _original_max_tokens, current_model) if _token_retries > 0 else None
                            return GeminiGenerationResult(text=recovered, token_bump=bump)
                        if _is_max_tokens_finish(response) and _token_retries < 2:
                            _token_retries += 1
                            new_max = int(_original_max_tokens * 1.5) if _token_retries == 1 else int(_original_max_tokens * 3.0)
                            _current_max_tokens = new_max
                            logger.debug(
                                "gemini.generate_with_model.max_tokens_retry",
                                model=current_model,
                                attempt=attempt,
                                token_retry=_token_retries,
                                new_max_output_tokens=new_max,
                            )
                            config = self._build_generation_config(
                                current_model,
                                temperature=temperature,
                                max_output_tokens=new_max,
                            )
                            continue
                        _log_gemini_failure(
                            event="gemini.empty_response",
                            model=current_model,
                            prompt=prompt,
                            response=response,
                            status=200,
                            extra={"attempt": attempt, "timeout_seconds": timeout, "token_retries": _token_retries},
                        )
                        _emit_event("llm.request.finished", model=current_model, status="empty", duration=time.monotonic() - req_start)
                        bump = TokenBumpInfo(False, _current_max_tokens, _original_max_tokens, current_model) if _token_retries > 0 else None
                        return GeminiGenerationResult(
                            error=GeminiErrorDetails(
                                category="empty",
                                raw_message=_describe_empty_response(response),
                                http_status=200,
                                retryable=False,
                            ),
                            token_bump=bump,
                        )
                    _emit_event("llm.request.finished", model=current_model, status="ok", duration=time.monotonic() - req_start)
                    bump = TokenBumpInfo(True, _current_max_tokens, _original_max_tokens, current_model) if _token_retries > 0 else None
                    return GeminiGenerationResult(text=str(text), token_bump=bump)
                except FuturesTimeout:
                    _log_gemini_failure(
                        event="gemini.timeout",
                        model=current_model,
                        prompt=prompt,
                        exc=TimeoutError(f"Timed out after {timeout}s waiting for Gemini."),
                        extra={"attempt": attempt, "timeout_seconds": timeout},
                    )
                    logger.debug(
                        "gemini.generate_with_model.timeout",
                        model=current_model,
                        timeout=timeout,
                        attempt=attempt,
                    )
                    _emit_event("llm.request.finished", model=current_model, status="timeout", duration=time.monotonic() - req_start, error_class="timeout")
                    return GeminiGenerationResult(
                        error=GeminiErrorDetails(
                            category="timeout",
                            raw_message=f"Timed out after {timeout}s waiting for Gemini.",
                            http_status=None,
                            retryable=True,
                        )
                    )
                except Exception as exc:
                    if _should_retry_rate_limit(exc):
                        fallback_model = _stable_fallback_for(current_model)
                        if fallback_model and not fallback_used and not os.environ.get("PB_NO_FALLBACK"):
                            logger.debug(
                                "gemini.generate_with_model.rate_limit_fallback",
                                from_model=current_model,
                                to_model=fallback_model,
                                attempt=attempt,
                            )
                            _emit_event(
                                "llm.request.model_fallback",
                                from_model=current_model,
                                to_model=fallback_model,
                                reason="rate_limit",
                                attempt=attempt,
                            )
                            current_model = fallback_model
                            config = config_builder(current_model) if callable(config_builder) else None
                            timeout = resolve_timeout("gemini", current_model, requested_timeout)
                            fallback_used = True
                            if attempt < attempts:
                                time.sleep(_retry_backoff_seconds(attempt))
                                continue
                        elif attempt < attempts:
                            sleep_seconds = _retry_backoff_seconds(attempt)
                            logger.debug(
                                "gemini.generate_with_model.retry",
                                model=current_model,
                                attempt=attempt,
                                sleep_seconds=sleep_seconds,
                                error=str(exc),
                            )
                            time.sleep(sleep_seconds)
                            continue
                    classified = _classify_error(exc)
                    _log_gemini_failure(
                        event="gemini.generate_exception",
                        model=current_model,
                        prompt=prompt,
                        exc=exc,
                        status=classified.http_status,
                        extra={"attempt": attempt, "category": classified.category},
                    )
                    logger.debug(
                        "gemini.generate_with_model",
                        model=current_model,
                        error=classified.raw_message,
                        attempt=attempt,
                    )
                    _emit_event("llm.request.finished", model=current_model, status="error", duration=time.monotonic() - req_start, error_class=classified.category)
                    return GeminiGenerationResult(error=classified)
        except Exception as e:
            _log_gemini_failure(
                event="gemini.generate_outer_exception",
                model=model,
                prompt=prompt,
                exc=e,
                status=_extract_http_status(e),
            )
            logger.debug("gemini.generate_with_model", model=model, error=str(e))
            _emit_event("llm.request.finished", model=model, status="error", duration=time.monotonic() - req_start, error_class=type(e).__name__)
            return GeminiGenerationResult(error=_classify_error(e))

    def generate_with_model_result(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> GeminiGenerationResult:
        """Generate text using a specific model tier.

        Args:
            prompt: The prompt text
            model: Model ID string (use FLASH_LITE_MODEL, FLASH_MODEL, or PRO_MODEL constants)
            timeout: Max seconds to wait for response (default 30)
            temperature: Optional sampling temperature
            max_output_tokens: Optional max output token limit

        Returns:
            Generated text, or None if unavailable or error
        """
        return self._generate_with_config_builder(
            prompt,
            model,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            config_builder=lambda current_model: self._build_generation_config(
                current_model,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

    def generate_structured_with_model_result(
        self,
        prompt: str,
        schema_cls: type,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = 0.0,
        max_output_tokens: Optional[int] = None,
    ) -> GeminiGenerationResult:
        """Generate structured JSON using Gemini's native response_schema path."""
        return self._generate_with_config_builder(
            prompt,
            model,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            config_builder=lambda _current_model: self._build_structured_generation_config(
                schema_cls,
                temperature=temperature,
                max_output_tokens=max_output_tokens,
            ),
        )

    def generate_with_tools(self, prompt: str, model: str, tools: list) -> Optional[str]:
        """Generate text with function calling tools (AFC).

        The SDK automatically handles the call loop: model requests a
        function call, SDK executes the Python callable, sends the result
        back, and repeats until the model emits a final text response.

        Args:
            prompt: The prompt text
            model: Model ID string
            tools: List of Python callables the model may invoke

        Returns:
            Final text response after all tool calls complete, or None on error
        """
        if not self.is_available() or self._client is None:
            return None
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            from google.genai import types
            resolved_model = resolve_model(model, fallback=FLASH_MODEL)
            timeout = resolve_timeout("gemini", resolved_model, 30)

            def _call():
                return self._client.models.generate_content(
                    model=resolved_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(tools=tools),
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                with slow_thinking_notice("gemini", resolved_model):
                    response = future.result(timeout=timeout)
            return response.text
        except FuturesTimeout:
            logger.debug("gemini.generate_with_tools", model=model, error=f"Timed out after {timeout}s")
            return None
        except Exception as e:
            _log_gemini_failure(
                event="gemini.generate_with_tools_exception",
                model=model,
                prompt=prompt,
                exc=e,
                status=_extract_http_status(e),
                extra={"tools": [getattr(tool, "__name__", repr(tool)) for tool in tools]},
            )
            logger.debug("gemini.generate_with_tools", model=model, error=str(e))
            return None

    def generate_with_grounding(self, prompt: str, model: str) -> Optional[str]:
        """Generate text with Google Search grounding enabled.

        Uses Gemini's built-in Google Search tool for factual grounding.
        Only use for concept/knowledge queries where web context helps.

        Args:
            prompt: The prompt text
            model: Model ID string (FLASH_MODEL or PRO_MODEL -- grounding not supported on Lite)

        Returns:
            Generated text with grounded information, or None if unavailable
        """
        if not self.is_available() or self._client is None:
            return None
        try:
            from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
            from google.genai import types
            resolved_model = resolve_model(model, fallback=FLASH_MODEL)
            timeout = resolve_timeout("gemini", resolved_model, 30)

            def _call():
                return self._client.models.generate_content(
                    model=resolved_model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        tools=[types.Tool(google_search=types.GoogleSearch())]
                    ),
                )

            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(_call)
                with slow_thinking_notice("gemini", resolved_model):
                    response = future.result(timeout=timeout)
            return response.text
        except FuturesTimeout:
            logger.debug("gemini.generate_with_grounding", model=model, error=f"Timed out after {timeout}s")
            return None
        except Exception as e:
            _log_gemini_failure(
                event="gemini.generate_with_grounding_exception",
                model=model,
                prompt=prompt,
                exc=e,
                status=_extract_http_status(e),
            )
            logger.debug("gemini.generate_with_grounding", model=model, error=str(e))
            return None

    async def generate_streaming_async(
        self,
        async_chat,
        message: str,
    ) -> str:
        """Stream response tokens to stdout via AsyncChat; return full text.

        The caller creates the AsyncChat via raw_client.aio.chats.create() and passes
        the per-turn message. This method handles the streaming loop only.

        Follows the same guard pattern as all other generate methods (CLUX-05).
        Writes chunks inline with sys.stdout; adds final newline.

        Args:
            async_chat: AsyncChat instance from client.aio.chats.create()
            message: Full message including per-turn context

        Returns:
            Complete response text (all chunks joined), or "" on error/unavailable.
        """
        if not self.is_available() or self._client is None:
            return ""
        try:
            parts = []
            async for chunk in await async_chat.send_message_stream(message):
                if chunk.text:
                    sys.stdout.write(chunk.text)
                    sys.stdout.flush()
                    parts.append(chunk.text)
            sys.stdout.write("\n")
            return "".join(parts)
        except Exception as e:
            _log_gemini_failure(
                event="gemini.streaming_exception",
                model="streaming",
                prompt=message,
                exc=e,
                status=_extract_http_status(e),
            )
            logger.debug("gemini.generate_streaming_async", error=str(e))
            return ""


# Module-level singleton for convenience
_client: Optional[GeminiClient] = None


def get_client() -> GeminiClient:
    """Get the singleton Gemini client."""
    global _client
    if _client is None:
        _client = GeminiClient()
    return _client


def score_text_response(question: str, response: str) -> Optional[tuple[int, str]]:
    """
    Score a text response using Gemini Flash Lite (per D-12).

    Args:
        question: The review question that was asked
        response: User's text response

    Returns:
        (score, rationale) tuple, or None if API unavailable
    """
    client = get_client()
    if not client.is_available():
        return None

    prompt = f"""You are evaluating a self-reflection response from a daily productivity review.

Question asked: {question}
User's response: {response}

Rate this response on a scale of 1-10 based on:
- Clarity and specificity (not vague)
- Actionable insight (can inform behavior change)
- Honesty and self-awareness

Respond in this exact format (nothing else):
SCORE: [number 1-10]
RATIONALE: [one sentence explanation]"""

    result = client.generate(prompt)
    if result is None:
        return None

    try:
        lines = result.strip().split('\n')
        score_line = next((l for l in lines if l.startswith('SCORE:')), None)
        rationale_line = next((l for l in lines if l.startswith('RATIONALE:')), None)

        if score_line is None or rationale_line is None:
            logger.debug("gemini.score_parse", error="missing_fields", result=result[:100])
            return None

        score = int(score_line.split(':')[1].strip())
        score = max(1, min(10, score))  # Clamp to 1-10
        rationale = rationale_line.split(':', 1)[1].strip()

        logger.debug("gemini.score", score=score)
        return (score, rationale)
    except (ValueError, IndexError) as e:
        logger.debug("gemini.score_parse", error=str(e), result=result[:100])
        return None


def generate_followup(question: str, response: str) -> Optional[str]:
    """
    Generate a follow-up question for vague responses (per D-10, D-11).

    Args:
        question: The original review question
        response: User's text response that may be vague

    Returns:
        Follow-up question string, or None if API unavailable
    """
    client = get_client()
    if not client.is_available():
        return None

    prompt = f"""You are conducting a daily productivity review. The user gave a vague or incomplete response.

Original question: {question}
User's response: {response}

Generate ONE short follow-up question (max 15 words) to help them be more specific.
Just output the question, nothing else."""

    result = client.generate(prompt)
    if result is None:
        return None

    # Clean up the response
    followup = result.strip().strip('"').strip()
    logger.debug("gemini.followup", length=len(followup))
    return followup
