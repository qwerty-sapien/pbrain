"""Provider-registry tests for the ProductiveBrain LLM runtime."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

from pydantic import BaseModel

from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL
from pb.llm.runtime import (
    DraftGenerationError,
    LLMHealth,
    LLMRuntime,
    ProviderErrorDetails,
    ProviderGenerationResult,
)
from pb.storage.config import (
    Config,
    GeneralConfig,
    ModelRolesConfig,
    OPENAI_FAST_MODEL,
    ProviderConfig,
    VaultProfileConfig,
)


def _config_for_provider(provider: str, model: str) -> Config:
    return Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp"),
        vaults={
            "main": VaultProfileConfig(path="/tmp", data_dir="/tmp/pb-data"),
        },
        providers={
            "gemini": ProviderConfig(api_key_env="GEMINI_API_KEY", default_model=FLASH_MODEL),
            "openai": ProviderConfig(api_key_env="OPENAI_API_KEY", default_model="gpt-5-mini"),
            "anthropic": ProviderConfig(api_key_env="ANTHROPIC_API_KEY", default_model="claude-sonnet-4-5"),
            "openrouter": ProviderConfig(api_key_env="OPENROUTER_API_KEY", default_model="openai/gpt-5-mini"),
        },
        model_roles=ModelRolesConfig(
            default=f"{provider}:{model}",
            planner=f"{provider}:{model}",
            reviewer="anthropic:claude-sonnet-4-5",
            recall="openai:gpt-5-mini",
            fast="gemini:gemini-3-flash-preview",
        ),
    )


def test_gemini_locked_model_ids_remain_unchanged() -> None:
    assert FLASH_LITE_MODEL == "gemini-3.1-flash-lite-preview"
    assert FLASH_MODEL == "gemini-3-flash-preview"
    assert PRO_MODEL == "gemini-3.1-pro-preview"


def test_model_roles_are_exposed() -> None:
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))
    roles = runtime.role_bindings()

    assert roles["default"] == "gemini:gemini-3-flash-preview"
    assert roles["reviewer"] == "anthropic:claude-sonnet-4-5"
    assert roles["recall"] == "openai:gpt-5-mini"


def test_structured_output_lite_uses_fast_inference_role(monkeypatch) -> None:
    from pb.llm import structured

    class AnswerDraft(BaseModel):
        answer: str

    config = Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp"),
        vaults={"main": VaultProfileConfig(path="/tmp", data_dir="/tmp/pb-data")},
        providers={
            "openai": ProviderConfig(
                api_key_env="OPENAI_API_KEY",
                default_model="gpt-5.4",
            )
        },
        model_roles=ModelRolesConfig(default="openai:gpt-5.4"),
    )
    captured: dict[str, str] = {}

    class FakeRuntime:
        def __init__(self, runtime_config):
            self.config = runtime_config

        def generate_draft(self, schema_cls, prompt, **kwargs):
            captured["model"] = kwargs["model"]
            captured["source_scope"] = kwargs["source_scope"]
            return SimpleNamespace(payload=schema_cls(answer="ok"))

    monkeypatch.setattr(structured, "get_config", lambda: config)
    monkeypatch.setattr(structured, "LLMRuntime", FakeRuntime)

    result = asyncio.run(
        structured.structured_output_call(
            "choose",
            AnswerDraft,
            tier="lite",
        )
    )

    assert result == AnswerDraft(answer="ok")
    assert captured["model"] == f"openai:{OPENAI_FAST_MODEL}"
    assert captured["source_scope"] == "structured_output:routing:AnswerDraft"


def test_openai_provider_health_uses_env(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    runtime = LLMRuntime(_config_for_provider("openai", "gpt-5-mini"))
    health = runtime.health()

    assert health.provider == "openai"
    assert health.default_model == "gpt-5-mini"
    assert health.credential_source == "OPENAI_API_KEY"
    assert health.available is True


def test_missing_credentials_fail_helpfully(monkeypatch) -> None:
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    runtime = LLMRuntime(_config_for_provider("openrouter", "openai/gpt-5-mini"))
    health = runtime.health()

    assert health.provider == "openrouter"
    assert health.available is False
    assert "OPENROUTER_API_KEY" in health.message


def test_gemini_candidate_models_do_not_degrade_by_default() -> None:
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))

    assert runtime.candidate_models("gemini", FLASH_MODEL) == [FLASH_MODEL]


def test_gemini_candidate_models_include_pro_only_when_enabled() -> None:
    config = _config_for_provider("gemini", FLASH_MODEL)
    config.llm.auto_pro_fallback = True
    runtime = LLMRuntime(config)

    assert runtime.candidate_models("gemini", FLASH_MODEL) == [FLASH_MODEL, PRO_MODEL]


def test_gemini_flash_lite_escalates_to_flash_before_pro() -> None:
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_LITE_MODEL))

    assert runtime.candidate_models("gemini", FLASH_LITE_MODEL) == [FLASH_LITE_MODEL, FLASH_MODEL]


def test_gemini_flash_lite_can_continue_to_pro_when_enabled() -> None:
    config = _config_for_provider("gemini", FLASH_LITE_MODEL)
    config.llm.auto_pro_fallback = True
    runtime = LLMRuntime(config)

    assert runtime.candidate_models("gemini", FLASH_LITE_MODEL) == [FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL]


def test_draft_generation_error_hides_raw_provider_details_by_default() -> None:
    error = DraftGenerationError(
        source_scope="clarify:test",
        prompt_template_version="test",
        attempts=[],
        error=ProviderErrorDetails(
            category="config",
            provider="gemini",
            model="gemini-3-flash",
            raw_message=(
                "404 NOT_FOUND projects/my-gcp-project-12345/locations/global/publishers/google/models/gemini-3-flash"
            ),
            http_status=404,
            retryable=False,
        ),
    )

    message = error.to_user_message()

    assert "projects/my-gcp-project-12345" not in message
    assert "gemini-3-flash" not in message
    assert "rejected the configured request or model settings" in message

    debug_message = error.to_user_message(debug=True)
    assert "projects/my-gcp-project-12345" in debug_message


def test_live_probe_returns_sanitized_provider_category(monkeypatch) -> None:
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))

    class _FakeClient:
        credential_source = "vertex"
        backend = "vertex"

        def generate_with_model(self, prompt, model, timeout=30, *, temperature=None, max_output_tokens=None):
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="config",
                    provider="gemini",
                    model=model,
                    raw_message="404 NOT_FOUND projects/my-gcp-project-12345 ...",
                    http_status=404,
                    retryable=False,
                )
            )

    monkeypatch.setattr(
        runtime,
        "health",
        lambda: LLMHealth(
            configured=True,
            available=True,
            provider="gemini",
            backend="vertex",
            default_model=FLASH_MODEL,
            structured_output=True,
            credential_source="vertex",
            message="configured",
        ),
    )
    monkeypatch.setattr(runtime, "_client_for_provider", lambda provider_name: _FakeClient())

    probe = runtime.live_probe()

    assert probe.available is False
    assert probe.category == "config"
    assert "projects/my-gcp-project-12345" not in probe.message
    assert "configured request or model settings" in probe.message
    assert "projects/my-gcp-project-12345" in probe.debug_message
