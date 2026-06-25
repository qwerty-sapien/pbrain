# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared LLM runtime contract for learning workflows."""

from __future__ import annotations

import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, TypeVar

from pydantic import BaseModel, ValidationError as PydanticValidationError

from pb.core.error_logging import log_error
from pb.core.models import GenerationProvenance
from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL, TokenBumpInfo, get_client, resolve_model
from pb.llm.json_utils import extract_json_block
from pb.llm.policy import resolve_timeout, slow_thinking_notice
from pb.storage.config import Config, ProviderConfig, get_config


T = TypeVar("T", bound=BaseModel)

_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1"

_LLM_RUNTIME_HINT = (
    "This workflow only works when you are online and your configured LLM API is valid "
    "(key present, not expired, enough credits/quota, and model access available)."
)

_STRICT_JSON_SUFFIX = (
    "You must return exactly one JSON object and nothing else. "
    "No Markdown, no code fences, no prose, no comments, no extra keys, "
    "no omitted required keys, no trailing text. Never return an empty response."
)


@dataclass
class LLMHealth:
    """Resolved runtime state for the configured LLM backend."""

    configured: bool
    available: bool
    provider: str
    backend: str
    default_model: str
    structured_output: bool
    credential_source: str
    message: str


@dataclass
class LLMProbeResult:
    """Outcome of a cheap live LLM request check."""

    available: bool
    provider: str
    backend: str
    model: str
    credential_source: str
    category: str
    message: str
    debug_message: str = ""
    http_status: int | None = None
    retryable: bool = False


@dataclass
class GeneratedDraft:
    """Validated draft plus provenance metadata."""

    payload: BaseModel
    model: str
    source_scope: str
    prompt_template_version: str
    raw_response: str
    attempts: tuple["DraftAttempt", ...] = ()


@dataclass
class DraftAttempt:
    """One provider/model attempt while generating a structured draft."""

    provider: str
    model: str
    prompt_kind: str
    status: str
    raw_message: str = ""
    http_status: int | None = None
    retryable: bool = False


@dataclass
class ProviderErrorDetails:
    """Normalized provider or downstream error details."""

    category: str
    provider: str
    model: str
    raw_message: str
    http_status: int | None = None
    retryable: bool = False


@dataclass
class ProviderGenerationResult:
    """Detailed provider generation outcome."""

    text: str | None = None
    error: ProviderErrorDetails | None = None
    token_bump: TokenBumpInfo | None = None


class DraftGenerationError(RuntimeError):
    """Raised when the runtime cannot produce a usable structured draft."""

    def __init__(
        self,
        *,
        source_scope: str,
        prompt_template_version: str,
        attempts: list[DraftAttempt],
        error: ProviderErrorDetails,
    ) -> None:
        super().__init__(error.raw_message)
        self.source_scope = source_scope
        self.prompt_template_version = prompt_template_version
        self.attempts = tuple(attempts)
        self.error = error

    @property
    def user_fixable(self) -> bool:
        return self.error.category in {"auth", "config", "quota"}

    def to_user_message(self, *, debug: bool = False) -> str:
        message = _provider_error_user_message(self.error)
        if debug and self.error.raw_message:
            return message + f"\nDebug details: {self.debug_details()}"
        return message

    def debug_details(self) -> str:
        details = [
            f"provider={self.error.provider}",
            f"model={self.error.model}",
            f"category={self.error.category}",
        ]
        if self.error.http_status is not None:
            details.append(f"http_status={self.error.http_status}")
        if self.error.raw_message:
            details.append(f"raw={self.error.raw_message}")
        return " | ".join(details)


class ProviderClient(Protocol):
    """Minimal provider client contract."""

    def is_available(self) -> bool:
        ...

    def generate_with_model(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ProviderGenerationResult:
        ...

    @property
    def credential_source(self) -> str:
        ...

    @property
    def backend(self) -> str:
        ...


def _parse_role_binding(binding: str) -> tuple[str, str]:
    raw = (binding or "").strip()
    if ":" not in raw:
        return "gemini", raw or FLASH_MODEL
    provider, model = raw.split(":", 1)
    return provider.strip().lower(), model.strip()


def _extract_http_status(exc: Exception) -> int | None:
    message = str(exc)
    match = re.search(r"\b([1-5]\d\d)\b", message)
    if match:
        try:
            return int(match.group(1))
        except ValueError:  # pragma: no cover - defensive
            return None
    return None


def _classify_http_error(provider: str, model: str, exc: Exception) -> ProviderErrorDetails:
    status = _extract_http_status(exc)
    message = str(exc).strip() or exc.__class__.__name__
    lowered = message.lower()
    if status in {401, 403}:
        category = "auth"
    elif status in {400, 404}:
        category = "config"
    elif status == 429:
        category = "quota"
    elif status is not None and status >= 500:
        category = "upstream"
    elif isinstance(exc, TimeoutError) or "timeout" in lowered:
        category = "timeout"
    elif any(
        marker in lowered
        for marker in (
            "nodename nor servname provided",
            "name or service not known",
            "temporary failure in name resolution",
            "connection refused",
            "network is unreachable",
            "failed to establish a new connection",
        )
    ):
        category = "network"
    else:
        category = "unknown"
    return ProviderErrorDetails(
        category=category,
        provider=provider,
        model=model,
        raw_message=message,
        http_status=status,
        retryable=category in {"quota", "upstream", "network", "timeout"},
    )


def _downstream_error(provider: str, model: str, message: str) -> ProviderErrorDetails:
    return ProviderErrorDetails(
        category="downstream",
        provider=provider,
        model=model,
        raw_message=message,
        http_status=200,
        retryable=False,
    )


def _provider_error_user_message(error: ProviderErrorDetails) -> str:
    provider = error.provider or "LLM provider"
    if error.category == "auth":
        return (
            f"{provider} credentials or access are not working for this workflow.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Fix the configured credentials or model access, or continue manually."
        )
    if error.category == "config":
        return (
            f"{provider} rejected the configured request or model settings.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Check the configured model binding or request settings, or continue manually."
        )
    if error.category == "quota":
        return (
            f"{provider} is refusing requests because of quota, credits, or rate limits.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Wait, add credits, or retry later. You can also continue manually now."
        )
    if error.category == "upstream":
        return (
            f"{provider} is currently returning an upstream service failure.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Retry later, or continue manually now."
        )
    if error.category == "network":
        return (
            f"pb could not reach {provider} over the network.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Check connectivity and retry, or continue manually now."
        )
    if error.category == "downstream":
        return (
            f"{provider} responded, but pb could not turn that response into a usable draft.\n"
            "This looks like a pb-side downstream issue. Please report it, and continue manually for now."
        )
    if error.category == "timeout":
        return (
            f"{provider} took too long to respond.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "Retry later, or continue manually now."
        )
    if error.category == "empty":
        return (
            f"{provider} returned no usable draft for this workflow.\n"
            f"{_LLM_RUNTIME_HINT}\n"
            "You can continue manually now."
        )
    return (
        f"{provider} could not generate a usable draft for this workflow.\n"
        f"{_LLM_RUNTIME_HINT}\n"
        "You can continue manually now."
    )


def _expanded_empty_retry_budget(max_output_tokens: int) -> int:
    return min(max(max_output_tokens * 2, 4000), 8000)


def _tier_for_model(model: str) -> str:
    """Map a model ID to its user-facing tier label."""
    lower = (model or "").lower()
    if "lite" in lower:
        return "fast"
    if "pro" in lower:
        return "pro"
    return "balanced"


def _notify_token_bump(bump, config) -> None:  # type: ignore[no-untyped-def]
    """Prompt the user to raise their default max-tokens when a 3x retry was needed."""
    from pb.llm.gemini import TokenBumpInfo  # avoid circular at module level
    if not isinstance(bump, TokenBumpInfo):
        return
    # Only prompt on the 3x tier (final_tokens >= 2.5× original) to avoid noise on 1.5× bumps.
    if bump.final_tokens < bump.original_tokens * 2.5:
        return
    if not sys.stdin.isatty():
        return
    tier = _tier_for_model(bump.model)
    if bump.succeeded:
        prompt_msg = (
            f"Only when max_tokens = {bump.final_tokens} did the request succeed "
            f"(your previous limit: {bump.original_tokens}). "
            f"Revise your max token limit for [bold]{tier}[/] models? (y/n) "
        )
    else:
        estimate = int(bump.final_tokens * 1.3)
        prompt_msg = (
            f"Even when max_tokens = {bump.final_tokens} the request still failed "
            f"(needing about {estimate} tokens). "
            f"Revise your max token limit for [bold]{tier}[/] models? (y/n) "
        )
    try:
        from pb.cli.console import get_err_console
        console = get_err_console()
        console.print(f"\n[warn]Token limit auto-adjusted:[/] {prompt_msg}", end="")
        ans = input()
    except Exception:
        return
    if ans.strip().lower() == "y":
        new_tokens = bump.final_tokens
        try:
            from pb.storage.config import set_model_max_tokens
            set_model_max_tokens(tier, new_tokens)
            from pb.cli.console import get_err_console
            get_err_console().print(f"[success]Saved:[/] {tier} max_tokens → {new_tokens}")
        except Exception:
            pass


def _http_error_body(exc: urllib.error.HTTPError) -> str:
    try:
        body = exc.read()
    except Exception:
        return ""
    try:
        return body.decode("utf-8", errors="replace")
    except Exception:
        return repr(body)


class GeminiProviderClient:
    """Adapter around the existing Gemini runtime."""

    def __init__(self) -> None:
        self._client = get_client()

    @property
    def credential_source(self) -> str:
        if os.environ.get("GOOGLE_CLOUD_PROJECT"):
            return "vertex"
        if os.environ.get("GEMINI_API_KEY"):
            return "aistudio"
        return "none"

    @property
    def backend(self) -> str:
        return self.credential_source if self.credential_source != "none" else "auto"

    def is_available(self) -> bool:
        return self._client.is_available()

    def generate_with_model(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ProviderGenerationResult:
        result = self._client.generate_with_model_result(
            prompt,
            model,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if result.error is not None:
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category=result.error.category,
                    provider="gemini",
                    model=model,
                    raw_message=result.error.raw_message,
                    http_status=result.error.http_status,
                    retryable=result.error.retryable,
                ),
                token_bump=result.token_bump,
            )
        return ProviderGenerationResult(text=result.text, token_bump=result.token_bump)

    def generate_structured_with_model(
        self,
        prompt: str,
        schema_cls: type[T],
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = 0.0,
        max_output_tokens: Optional[int] = None,
    ) -> ProviderGenerationResult:
        result = self._client.generate_structured_with_model_result(
            prompt,
            schema_cls,
            model,
            timeout=timeout,
            temperature=temperature,
            max_output_tokens=max_output_tokens,
        )
        if result.error is not None:
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category=result.error.category,
                    provider="gemini",
                    model=model,
                    raw_message=result.error.raw_message,
                    http_status=result.error.http_status,
                    retryable=result.error.retryable,
                ),
                token_bump=result.token_bump,
            )
        return ProviderGenerationResult(text=result.text, token_bump=result.token_bump)


class OpenAICompatibleProviderClient:
    """HTTP client for OpenAI-compatible chat-completions endpoints."""

    def __init__(self, provider_name: str, config: ProviderConfig) -> None:
        self.provider_name = provider_name
        self.config = config

    @property
    def credential_source(self) -> str:
        env_name = (self.config.api_key_env or "").strip()
        if env_name and os.environ.get(env_name):
            return env_name
        return "none"

    @property
    def backend(self) -> str:
        return (self.config.base_url or "").strip() or self.provider_name

    def is_available(self) -> bool:
        return self.credential_source != "none"

    def generate_with_model(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ProviderGenerationResult:
        api_key = os.environ.get(self.config.api_key_env or "")
        base_url = (self.config.base_url or "").rstrip("/")
        if not api_key or not base_url:
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="auth",
                    provider=self.provider_name,
                    model=model,
                    raw_message=f"Missing credentials or base URL for {self.provider_name}.",
                    retryable=False,
                )
            )

        timeout = resolve_timeout(self.provider_name, model, timeout)
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        request = urllib.request.Request(
            f"{base_url}/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )

        try:
            with slow_thinking_notice(self.provider_name, model):
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            choices = body.get("choices") or []
            if not choices:
                log_error(
                    event="llm.openai_compatible.empty_choices",
                    message=f"{self.provider_name} returned no choices.",
                    command=self.provider_name,
                    status=200,
                    request_body=payload,
                    response_status=200,
                    response_body=body,
                    extra={"provider": self.provider_name, "model": model},
                )
                return ProviderGenerationResult(
                    error=ProviderErrorDetails(
                        category="empty",
                        provider=self.provider_name,
                        model=model,
                        raw_message="Provider returned no choices.",
                        http_status=200,
                    )
                )
            message = choices[0].get("message") or {}
            content = message.get("content")
            if isinstance(content, str):
                return ProviderGenerationResult(text=content)
            if isinstance(content, list):
                rendered = "\n".join(
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ).strip()
                if rendered:
                    return ProviderGenerationResult(text=rendered)
            log_error(
                event="llm.openai_compatible.empty_content",
                message=f"{self.provider_name} returned an empty content payload.",
                command=self.provider_name,
                status=200,
                request_body=payload,
                response_status=200,
                response_body=body,
                extra={"provider": self.provider_name, "model": model},
            )
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="empty",
                    provider=self.provider_name,
                    model=model,
                    raw_message="Provider returned an empty content payload.",
                    http_status=200,
                )
            )
        except urllib.error.HTTPError as exc:
            classified = _classify_http_error(self.provider_name, model, exc)
            log_error(
                event="llm.openai_compatible.http_error",
                message=classified.raw_message,
                exc=exc,
                command=self.provider_name,
                status=classified.http_status,
                request_body=payload,
                response_status=classified.http_status,
                response_body=_http_error_body(exc),
                extra={"provider": self.provider_name, "model": model, "category": classified.category},
            )
            return ProviderGenerationResult(error=classified)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            classified = _classify_http_error(self.provider_name, model, exc)
            log_error(
                event="llm.openai_compatible.transport_error",
                message=classified.raw_message,
                exc=exc,
                command=self.provider_name,
                status=classified.http_status,
                request_body=payload,
                extra={"provider": self.provider_name, "model": model, "category": classified.category},
            )
            return ProviderGenerationResult(error=classified)


class AnthropicProviderClient:
    """HTTP client for Anthropic Claude Messages API."""

    def __init__(self, config: ProviderConfig) -> None:
        self.config = config

    @property
    def credential_source(self) -> str:
        env_name = (self.config.api_key_env or "").strip()
        if env_name and os.environ.get(env_name):
            return env_name
        return "none"

    @property
    def backend(self) -> str:
        return (self.config.base_url or "").strip() or _ANTHROPIC_BASE_URL

    def is_available(self) -> bool:
        return self.credential_source != "none"

    def generate_with_model(
        self,
        prompt: str,
        model: str,
        timeout: int = 30,
        *,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
    ) -> ProviderGenerationResult:
        api_key = os.environ.get(self.config.api_key_env or "")
        base_url = ((self.config.base_url or "").strip() or _ANTHROPIC_BASE_URL).rstrip("/")
        if not api_key:
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="auth",
                    provider="anthropic",
                    model=model,
                    raw_message="Missing Anthropic credentials.",
                    retryable=False,
                )
            )

        timeout = resolve_timeout("anthropic", model, timeout)
        payload: dict[str, object] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_output_tokens or 8192,
        }
        if temperature is not None:
            payload["temperature"] = temperature

        request = urllib.request.Request(
            f"{base_url}/messages",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "x-api-key": api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            },
            method="POST",
        )

        try:
            with slow_thinking_notice("anthropic", model):
                with urllib.request.urlopen(request, timeout=timeout) as response:
                    body = json.loads(response.read().decode("utf-8"))
            parts = body.get("content") or []
            rendered = "\n".join(
                item.get("text", "")
                for item in parts
                if isinstance(item, dict) and item.get("type") == "text"
            ).strip()
            if rendered:
                return ProviderGenerationResult(text=rendered)
            log_error(
                event="llm.anthropic.empty_content",
                message="Anthropic returned an empty content payload.",
                command="anthropic",
                status=200,
                request_body=payload,
                response_status=200,
                response_body=body,
                extra={"provider": "anthropic", "model": model},
            )
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="empty",
                    provider="anthropic",
                    model=model,
                    raw_message="Anthropic returned an empty content payload.",
                    http_status=200,
                )
            )
        except urllib.error.HTTPError as exc:
            classified = _classify_http_error("anthropic", model, exc)
            log_error(
                event="llm.anthropic.http_error",
                message=classified.raw_message,
                exc=exc,
                command="anthropic",
                status=classified.http_status,
                request_body=payload,
                response_status=classified.http_status,
                response_body=_http_error_body(exc),
                extra={"provider": "anthropic", "model": model, "category": classified.category},
            )
            return ProviderGenerationResult(error=classified)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            classified = _classify_http_error("anthropic", model, exc)
            log_error(
                event="llm.anthropic.transport_error",
                message=classified.raw_message,
                exc=exc,
                command="anthropic",
                status=classified.http_status,
                request_body=payload,
                extra={"provider": "anthropic", "model": model, "category": classified.category},
            )
            return ProviderGenerationResult(error=classified)


class LLMRuntime:
    """Validated draft generation over the configured provider registry."""

    def __init__(self, config: Optional[Config] = None):
        self.config = config or get_config()
        provider_name, _ = self.default_binding()
        self.client = self._client_for_provider(provider_name)

    def default_binding(self) -> tuple[str, str]:
        """Return the configured default provider/model pair."""
        binding = self.config.model_roles.default or f"{self.config.llm.provider}:{self.config.llm.default_model}"
        return _parse_role_binding(binding)

    def role_bindings(self) -> dict[str, str]:
        """Return configured model role bindings."""
        return {
            "default": self.config.model_roles.default,
            "planner": self.config.model_roles.planner,
            "reviewer": self.config.model_roles.reviewer,
            "recall": self.config.model_roles.recall,
            "fast": self.config.model_roles.fast,
            "fast_inference": self.config.model_roles.fast_inference,
            "namer": self.config.model_roles.namer,
        }

    def configured_providers(self) -> dict[str, ProviderConfig]:
        """Return configured providers."""
        return self.config.providers

    def _client_for_provider(self, provider_name: str) -> ProviderClient:
        normalized = (provider_name or "gemini").strip().lower()
        provider_cfg = self.config.providers.get(normalized, ProviderConfig())
        if normalized == "gemini":
            return GeminiProviderClient()
        if normalized in {"openai", "openrouter"}:
            return OpenAICompatibleProviderClient(normalized, provider_cfg)
        if normalized == "anthropic":
            return AnthropicProviderClient(provider_cfg)
        # Fall back to OpenAI-compatible for custom providers with a base URL.
        return OpenAICompatibleProviderClient(normalized, provider_cfg)

    def _resolve_provider_and_model(self, model: Optional[str] = None) -> tuple[str, str]:
        selected = (model or self.config.model_roles.default or "").strip()
        if not selected:
            return self.default_binding()
        provider, resolved_model = _parse_role_binding(selected)
        if provider not in self.config.providers and ":" not in selected:
            default_provider, _ = self.default_binding()
            provider = default_provider
        return provider, resolved_model

    def health(self) -> LLMHealth:
        provider_name, model_name = self.default_binding()
        client = self._client_for_provider(provider_name)
        provider_cfg = self.config.providers.get(provider_name, ProviderConfig())
        available = client.is_available()
        configured = client.credential_source != "none"

        if provider_name == "gemini":
            message = (
                "Gemini credentials are configured. Live requests are not probed here and can still fail if you are offline, quota-limited, or Gemini returns empty content."
                if available
                else "Gemini credentials are missing or invalid. Set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
            )
        elif provider_name == "openai":
            message = (
                "OpenAI credentials are configured. Live requests are not probed here and can still fail if you are offline, quota-limited, or the API rejects the request."
                if available
                else f"Set {provider_cfg.api_key_env or 'OPENAI_API_KEY'} to enable OpenAI."
            )
        elif provider_name == "anthropic":
            message = (
                "Anthropic credentials are configured. Live requests are not probed here and can still fail if you are offline, quota-limited, or the API rejects the request."
                if available
                else f"Set {provider_cfg.api_key_env or 'ANTHROPIC_API_KEY'} to enable Claude."
            )
        elif provider_name == "openrouter":
            message = (
                "OpenRouter credentials are configured. Live requests are not probed here and can still fail if you are offline, quota-limited, or the API rejects the request."
                if available
                else f"Set {provider_cfg.api_key_env or 'OPENROUTER_API_KEY'} to enable OpenRouter."
            )
        else:
            message = (
                f"{provider_name} credentials are configured. Live requests are not probed here and can still fail if you are offline, quota-limited, or the provider rejects the request."
                if available
                else f"Set {provider_cfg.api_key_env or 'the configured API key env var'} to enable {provider_name}."
            )

        return LLMHealth(
            configured=configured,
            available=available,
            provider=provider_name,
            backend=client.backend,
            default_model=model_name,
            structured_output=True,
            credential_source=client.credential_source,
            message=message,
        )

    def live_probe(self, *, model: Optional[str] = None, timeout: int = 12) -> LLMProbeResult:
        """Issue a tiny live request to verify end-to-end provider access."""
        health = self.health()
        provider_name, selected_model = self._resolve_provider_and_model(model)
        client = self._client_for_provider(provider_name)
        if not health.available:
            bootstrap_error = ProviderErrorDetails(
                category="auth",
                provider=provider_name,
                model=selected_model,
                raw_message=health.message,
                http_status=None,
                retryable=False,
            )
            return LLMProbeResult(
                available=False,
                provider=provider_name,
                backend=client.backend,
                model=selected_model,
                credential_source=client.credential_source,
                category=bootstrap_error.category,
                message=_provider_error_user_message(bootstrap_error),
                debug_message=bootstrap_error.raw_message,
                http_status=bootstrap_error.http_status,
                retryable=bootstrap_error.retryable,
            )

        result = client.generate_with_model(
            "Reply with exactly OK.",
            selected_model,
            timeout=timeout,
            temperature=0.0,
            max_output_tokens=1000,
        )
        if result.error is not None:
            return LLMProbeResult(
                available=False,
                provider=provider_name,
                backend=client.backend,
                model=selected_model,
                credential_source=client.credential_source,
                category=result.error.category,
                message=_provider_error_user_message(result.error),
                debug_message=result.error.raw_message,
                http_status=result.error.http_status,
                retryable=result.error.retryable,
            )

        rendered = (result.text or "").strip()
        if not rendered:
            empty_error = ProviderErrorDetails(
                category="empty",
                provider=provider_name,
                model=selected_model,
                raw_message="Live probe returned an empty body.",
                http_status=200,
                retryable=False,
            )
            return LLMProbeResult(
                available=False,
                provider=provider_name,
                backend=client.backend,
                model=selected_model,
                credential_source=client.credential_source,
                category=empty_error.category,
                message=_provider_error_user_message(empty_error),
                debug_message=empty_error.raw_message,
                http_status=empty_error.http_status,
                retryable=empty_error.retryable,
            )

        return LLMProbeResult(
            available=True,
            provider=provider_name,
            backend=client.backend,
            model=selected_model,
            credential_source=client.credential_source,
            category="ok",
            message=f"Live {provider_name} probe succeeded.",
            debug_message=rendered,
            http_status=200,
            retryable=False,
        )

    def require(self, purpose: str) -> LLMHealth:
        health = self.health()
        if not health.available:
            raise RuntimeError(
                "ProductiveBrain requires an LLM for this workflow.\n\n"
                f"Requested workflow: {purpose}\n\n"
                "This only works when:\n"
                "1. You have an internet connection, and\n"
                "2. Your configured LLM API is working (valid key, enough credits/quota, supported model access).\n\n"
                "Run:\n"
                "  pb init\n\n"
                "Then configure a provider, default model, and credentials."
            )
        return health

    def make_stage_recorder(self, workflow: str, intent: str, *, route_hint: str = ""):
        from pb.core.staging import StageRecorder

        try:
            from pb.runtime import runtime_from_config

            runtime = runtime_from_config(self.config)
            data_dir = runtime.data_dir
        except Exception:
            try:
                from pb.storage.config import get_data_dir

                data_dir = get_data_dir(self.config)
            except Exception:
                data_dir = Path(tempfile.gettempdir()) / "productivebrain"
        return StageRecorder(data_dir=data_dir, workflow=workflow, intent=intent, route_hint=route_hint)

    def candidate_models(self, provider_name: str, selected_model: str) -> list[str]:
        """Return the provider-specific model fallback ladder."""
        if provider_name != "gemini":
            return [selected_model]
        resolved = resolve_model(selected_model, fallback=FLASH_MODEL)
        models = [resolved]
        if resolved == FLASH_LITE_MODEL:
            models.append(FLASH_MODEL)
        if self.config.llm.auto_pro_fallback and resolved in {FLASH_LITE_MODEL, FLASH_MODEL}:
            models.append(PRO_MODEL)
        ordered: list[str] = []
        for item in models:
            if item and item not in ordered:
                ordered.append(item)
        return ordered

    def generate_draft(
        self,
        schema_cls: type[T],
        prompt: str,
        *,
        source_scope: str,
        prompt_template_version: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 45,
        max_output_tokens: int = 4000,
    ) -> GeneratedDraft:
        health = self.health()
        if not health.available:
            raise DraftGenerationError(
                source_scope=source_scope,
                prompt_template_version=prompt_template_version or self.config.llm.prompt_template_version,
                attempts=[],
                error=ProviderErrorDetails(
                    category="auth",
                    provider=health.provider,
                    model=health.default_model,
                    raw_message=health.message,
                    http_status=None,
                    retryable=False,
                ),
            )
        provider_name, selected_model = self._resolve_provider_and_model(model)
        client = self._client_for_provider(provider_name)
        schema_json = json.dumps(schema_cls.model_json_schema(), indent=2, sort_keys=True)
        base_prompt = (
            "Return ONLY valid JSON. Do not include markdown fences or commentary.\n"
            f"Schema:\n{schema_json}\n\n"
            f"Task:\n{prompt.strip()}\n"
        )
        strict_prompt = base_prompt + "\n\n" + _STRICT_JSON_SUFFIX
        attempts: list[DraftAttempt] = []
        resolved_prompt_version = prompt_template_version or self.config.llm.prompt_template_version
        last_error = ProviderErrorDetails(
            category="empty",
            provider=provider_name,
            model=selected_model,
            raw_message="The LLM returned no draft. Nothing was persisted.",
            http_status=200 if health.available else None,
        )

        def _build_generated(raw_response: str, candidate_model: str) -> GeneratedDraft:
            payload = schema_cls.model_validate_json(extract_json_block(raw_response))
            return GeneratedDraft(
                payload=payload,
                model=f"{provider_name}:{candidate_model}",
                source_scope=source_scope,
                prompt_template_version=resolved_prompt_version,
                raw_response=raw_response,
                attempts=tuple(attempts),
            )

        def _retry_after_empty(candidate_model: str) -> tuple[str | None, ProviderErrorDetails | None]:
            retry_budget = _expanded_empty_retry_budget(max_output_tokens)
            retry_result = client.generate_with_model(
                strict_prompt,
                candidate_model,
                timeout=timeout,
                max_output_tokens=retry_budget,
            )
            if retry_result.error is not None:
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="draft-empty-retry",
                        status="error",
                        raw_message=retry_result.error.raw_message,
                        http_status=retry_result.error.http_status,
                        retryable=retry_result.error.retryable,
                    )
                )
                return None, retry_result.error
            retry_raw = (retry_result.text or "").strip()
            if not retry_raw:
                retry_error = ProviderErrorDetails(
                    category="empty",
                    provider=provider_name,
                    model=candidate_model,
                    raw_message="Retry after empty returned an empty body.",
                    http_status=200,
                    retryable=False,
                )
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="draft-empty-retry",
                        status="error",
                        raw_message=retry_error.raw_message,
                        http_status=retry_error.http_status,
                    )
                )
                return None, retry_error
            attempts.append(
                DraftAttempt(
                    provider=provider_name,
                    model=candidate_model,
                    prompt_kind="draft-empty-retry",
                    status="ok",
                )
            )
            return retry_raw, None

        for candidate_model in self.candidate_models(provider_name, selected_model):
            native_generator = getattr(type(client), "generate_structured_with_model", None)
            if provider_name == "gemini" and callable(native_generator):
                native_result = client.generate_structured_with_model(
                    prompt,
                    schema_cls,
                    candidate_model,
                    timeout=timeout,
                    temperature=0.0,
                    max_output_tokens=max_output_tokens,
                )
                if native_result.error is not None:
                    attempts.append(
                        DraftAttempt(
                            provider=provider_name,
                            model=candidate_model,
                            prompt_kind="draft-native",
                            status="error",
                            raw_message=native_result.error.raw_message,
                            http_status=native_result.error.http_status,
                            retryable=native_result.error.retryable,
                        )
                    )
                    last_error = native_result.error
                    if native_result.error.category in {"auth", "quota"}:
                        break
                else:
                    native_raw = (native_result.text or "").strip()
                    if native_raw:
                        attempts.append(
                            DraftAttempt(
                                provider=provider_name,
                                model=candidate_model,
                                prompt_kind="draft-native",
                                status="ok",
                            )
                        )
                        try:
                            draft = _build_generated(native_raw, candidate_model)
                            _notify_token_bump(native_result.token_bump, self.config)
                            return draft
                        except (ValueError, PydanticValidationError, json.JSONDecodeError) as exc:
                            parse_error = f"{exc.__class__.__name__}: {exc}"
                            log_error(
                                event="llm.runtime.native_parse_error",
                                message=parse_error,
                                exc=exc,
                                config=self.config,
                                command=source_scope,
                                status=200,
                                request_body={
                                    "provider": provider_name,
                                    "model": candidate_model,
                                    "prompt_kind": "draft-native",
                                    "source_scope": source_scope,
                                    "prompt": prompt,
                                },
                                response_status=200,
                                response_body=native_raw,
                                extra={"schema": schema_cls.__name__},
                            )
                            last_error = _downstream_error(
                                provider_name,
                                candidate_model,
                                parse_error,
                            )
                            attempts.append(
                                DraftAttempt(
                                    provider=provider_name,
                                    model=candidate_model,
                                    prompt_kind="draft-native-parse",
                                    status="error",
                                    raw_message=parse_error,
                                    http_status=200,
                                )
                            )
                    else:
                        last_error = ProviderErrorDetails(
                            category="empty",
                            provider=provider_name,
                            model=candidate_model,
                            raw_message="Gemini native structured output returned an empty body.",
                            http_status=200,
                            retryable=False,
                        )
                        attempts.append(
                            DraftAttempt(
                                provider=provider_name,
                                model=candidate_model,
                                prompt_kind="draft-native",
                                status="error",
                                raw_message=last_error.raw_message,
                                http_status=last_error.http_status,
                            )
                        )

            result = client.generate_with_model(
                base_prompt,
                candidate_model,
                timeout=timeout,
                max_output_tokens=max_output_tokens,
            )
            if result.error is not None:
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="draft",
                        status="error",
                        raw_message=result.error.raw_message,
                        http_status=result.error.http_status,
                        retryable=result.error.retryable,
                    )
                )
                last_error = result.error
                if result.error.category == "empty":
                    raw, retry_error = _retry_after_empty(candidate_model)
                    if raw is None:
                        last_error = retry_error or last_error
                        continue
                    try:
                        draft = _build_generated(raw, candidate_model)
                        _notify_token_bump(result.token_bump, self.config)
                        return draft
                    except (ValueError, PydanticValidationError, json.JSONDecodeError) as exc:
                        parse_error = f"{exc.__class__.__name__}: {exc}"
                        log_error(
                            event="llm.runtime.parse_error",
                            message=parse_error,
                            exc=exc,
                            config=self.config,
                            command=source_scope,
                            status=200,
                            request_body={
                                "provider": provider_name,
                                "model": candidate_model,
                                "prompt_kind": "draft-empty-retry",
                                "source_scope": source_scope,
                                "prompt": strict_prompt,
                            },
                            response_status=200,
                            response_body=raw,
                            extra={"schema": schema_cls.__name__},
                        )
                        last_error = _downstream_error(
                            provider_name,
                            candidate_model,
                            parse_error,
                        )
                        attempts.append(
                            DraftAttempt(
                                provider=provider_name,
                                model=candidate_model,
                                prompt_kind="draft-empty-retry-parse",
                                status="error",
                                raw_message=parse_error,
                                http_status=200,
                            )
                        )
                        continue
                if result.error.category in {"auth", "quota"}:
                    break
                continue

            raw = (result.text or "").strip()
            if not raw:
                # Log empty output with structured error (D-07 step 3)
                log_error(
                    event="llm.runtime.empty_output",
                    message="Model returned empty response - retrying with stronger settings.",
                    config=self.config,
                    command=source_scope,
                    status=200,
                    request_body={
                        "provider": provider_name,
                        "model": candidate_model,
                        "finish_reason": getattr(result, "finish_reason", None),
                    },
                    response_status=200,
                    response_body="",
                )
                last_error = ProviderErrorDetails(
                    category="empty",
                    provider=provider_name,
                    model=candidate_model,
                    raw_message="The model returned an empty response.",
                    http_status=200,
                    retryable=False,
                )
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="draft",
                        status="error",
                        raw_message=last_error.raw_message,
                        http_status=last_error.http_status,
                    )
                )
                raw, retry_error = _retry_after_empty(candidate_model)
                if raw is None:
                    last_error = retry_error or last_error
                    continue
            attempts.append(
                DraftAttempt(
                    provider=provider_name,
                    model=candidate_model,
                    prompt_kind="draft",
                    status="ok",
                )
            )
            try:
                draft = _build_generated(raw, candidate_model)
                _notify_token_bump(result.token_bump, self.config)
                return draft
            except (ValueError, PydanticValidationError, json.JSONDecodeError) as exc:
                parse_error = f"{exc.__class__.__name__}: {exc}"
                log_error(
                    event="llm.runtime.parse_error",
                    message=parse_error,
                    exc=exc,
                    config=self.config,
                    command=source_scope,
                    status=200,
                    request_body={
                        "provider": provider_name,
                        "model": candidate_model,
                        "prompt_kind": "draft",
                        "source_scope": source_scope,
                        "prompt": base_prompt,
                    },
                    response_status=200,
                    response_body=raw,
                    extra={"schema": schema_cls.__name__},
                )
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="draft-parse",
                        status="error",
                        raw_message=parse_error,
                        http_status=200,
                    )
                )
                repair_prompt = (
                    "Fix the invalid draft below so it becomes valid JSON for the schema. "
                    "Return ONLY valid JSON.\n\n"
                    f"Schema:\n{schema_json}\n\n"
                    f"Invalid draft:\n{raw}"
                )
                repaired = client.generate_with_model(
                    repair_prompt,
                    candidate_model,
                    timeout=timeout,
                    max_output_tokens=max_output_tokens,
                )
                if repaired.error is not None:
                    attempts.append(
                        DraftAttempt(
                            provider=provider_name,
                            model=candidate_model,
                            prompt_kind="repair",
                            status="error",
                            raw_message=repaired.error.raw_message,
                            http_status=repaired.error.http_status,
                            retryable=repaired.error.retryable,
                        )
                    )
                    last_error = repaired.error
                    continue
                repaired_raw = (repaired.text or "").strip()
                if not repaired_raw:
                    last_error = _downstream_error(
                        provider_name,
                        candidate_model,
                        "Repair pass returned an empty body after invalid structured output.",
                    )
                    attempts.append(
                        DraftAttempt(
                            provider=provider_name,
                            model=candidate_model,
                            prompt_kind="repair",
                            status="error",
                            raw_message=last_error.raw_message,
                            http_status=200,
                        )
                    )
                    continue
                attempts.append(
                    DraftAttempt(
                        provider=provider_name,
                        model=candidate_model,
                        prompt_kind="repair",
                        status="ok",
                    )
                )
                try:
                    payload = schema_cls.model_validate_json(extract_json_block(repaired_raw))
                    return GeneratedDraft(
                        payload=payload,
                        model=f"{provider_name}:{candidate_model}",
                        source_scope=source_scope,
                        prompt_template_version=resolved_prompt_version,
                        raw_response=repaired_raw,
                        attempts=tuple(attempts),
                    )
                except (ValueError, PydanticValidationError, json.JSONDecodeError) as repair_exc:
                    log_error(
                        event="llm.runtime.repair_parse_error",
                        message=f"{repair_exc.__class__.__name__}: {repair_exc}",
                        exc=repair_exc,
                        config=self.config,
                        command=source_scope,
                        status=200,
                        request_body={
                            "provider": provider_name,
                            "model": candidate_model,
                            "prompt_kind": "repair",
                            "source_scope": source_scope,
                            "prompt": repair_prompt,
                        },
                        response_status=200,
                        response_body=repaired_raw,
                        extra={"schema": schema_cls.__name__},
                    )
                    last_error = _downstream_error(
                        provider_name,
                        candidate_model,
                        f"{repair_exc.__class__.__name__}: {repair_exc}",
                    )
                    attempts.append(
                        DraftAttempt(
                            provider=provider_name,
                            model=candidate_model,
                            prompt_kind="repair-parse",
                            status="error",
                            raw_message=last_error.raw_message,
                            http_status=200,
                        )
                    )

        raise DraftGenerationError(
            source_scope=source_scope,
            prompt_template_version=resolved_prompt_version,
            attempts=attempts,
            error=last_error,
        )

    def build_provenance(
        self,
        *,
        artifact_kind: str,
        artifact_id: str,
        generated_draft: GeneratedDraft,
        accepted_by_user: bool,
    ) -> GenerationProvenance:
        return GenerationProvenance(
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            generated_by_model=generated_draft.model,
            prompt_template_version=generated_draft.prompt_template_version,
            source_scope=generated_draft.source_scope,
            accepted_by_user=accepted_by_user,
        )
