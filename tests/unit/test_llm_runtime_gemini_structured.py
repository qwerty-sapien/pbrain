# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Gemini structured-draft runtime tests."""

from __future__ import annotations

import pytest

from pb.llm.drafts import GoalDraft, GoalRoadmapDraft, GoalRoadmapNodeDraft
from pb.llm.gemini import FLASH_MODEL
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
    ProviderConfig,
    VaultProfileConfig,
)


def _config_for_provider(provider: str, model: str) -> Config:
    return Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp"),
        vaults={"main": VaultProfileConfig(path="/tmp", data_dir="/tmp/pb-data")},
        providers={
            "gemini": ProviderConfig(api_key_env="GEMINI_API_KEY", default_model=model),
        },
        model_roles=ModelRolesConfig(
            default=f"{provider}:{model}",
            planner=f"{provider}:{model}",
            reviewer=f"{provider}:{model}",
            recall=f"{provider}:{model}",
            fast=f"{provider}:{model}",
        ),
    )


def _healthy_gemini_runtime() -> LLMRuntime:
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))
    runtime.health = lambda: LLMHealth(
        configured=True,
        available=True,
        provider="gemini",
        backend="auto",
        default_model=FLASH_MODEL,
        structured_output=True,
        credential_source="test",
        message="ready",
    )
    return runtime


def test_goal_roadmap_uses_native_gemini_structured_path() -> None:
    """Roadmap drafts should use Gemini's response_schema path before legacy prompts."""

    roadmap = GoalRoadmapDraft(
        summary="Foundations first, then split into sibling branches.",
        nodes=[
            GoalRoadmapNodeDraft(node_id="phase-1", title="Foundations", scope="core ideas"),
            GoalRoadmapNodeDraft(
                node_id="phase-2",
                title="Theory branch",
                scope="proofs",
                prerequisites=["phase-1"],
            ),
            GoalRoadmapNodeDraft(
                node_id="phase-3",
                title="Practice branch",
                branch="practise",
                scope="drills",
                prerequisites=["phase-1"],
            ),
        ],
    )

    class _FakeGeminiClient:
        def __init__(self) -> None:
            self.structured_calls: list[dict[str, object]] = []
            self.legacy_calls = 0

        def generate_structured_with_model(
            self,
            prompt,
            schema_cls,
            model,
            timeout=30,
            *,
            temperature=0.0,
            max_output_tokens=None,
        ):
            self.structured_calls.append(
                {
                    "prompt": prompt,
                    "schema_cls": schema_cls,
                    "model": model,
                    "timeout": timeout,
                    "temperature": temperature,
                    "max_output_tokens": max_output_tokens,
                }
            )
            return ProviderGenerationResult(text=roadmap.model_dump_json())

        def generate_with_model(self, *args, **kwargs):
            self.legacy_calls += 1
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="empty",
                    provider="gemini",
                    model=FLASH_MODEL,
                    raw_message="Gemini returned no usable text in an HTTP 200 response. finish reason=max tokens",
                    http_status=200,
                    retryable=False,
                )
            )

    runtime = _healthy_gemini_runtime()
    client = _FakeGeminiClient()
    runtime._client_for_provider = lambda provider_name: client

    draft = runtime.generate_draft(
        GoalRoadmapDraft,
        "build a branching roadmap",
        source_scope="goal_roadmap:test",
    )

    assert draft.payload.summary == roadmap.summary
    assert [node.node_id for node in draft.payload.nodes] == ["phase-1", "phase-2", "phase-3"]
    assert len(client.structured_calls) == 1
    assert client.legacy_calls == 0
    assert client.structured_calls[0]["schema_cls"] is GoalRoadmapDraft
    assert client.structured_calls[0]["temperature"] == 0.0
    assert client.structured_calls[0]["max_output_tokens"] == 4000


def test_gemini_native_structured_path_falls_back_to_legacy_json_prompt() -> None:
    """A native structured failure should fall back once to the legacy JSON prompt path."""

    payload = GoalDraft(
        title="Calculus",
        description="Learn calculus.",
        domain="calculus",
        execution_mode="study",
        horizon="quarter",
        framework="Bloom",
        study_framework="bloom_retrieval",
        target_bloom_stage="apply",
        success_definition="Solve guided calculus problems.",
        feedback_source="artifact",
        evidence_type="artifact",
    )

    class _FakeGeminiClient:
        def __init__(self) -> None:
            self.structured_calls = 0
            self.legacy_calls = 0

        def generate_structured_with_model(
            self,
            prompt,
            schema_cls,
            model,
            timeout=30,
            *,
            temperature=0.0,
            max_output_tokens=None,
        ):
            self.structured_calls += 1
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="upstream",
                    provider="gemini",
                    model=model,
                    raw_message="transient native structured failure",
                    http_status=503,
                    retryable=True,
                )
            )

        def generate_with_model(self, prompt, model, timeout=30, *, temperature=None, max_output_tokens=None):
            self.legacy_calls += 1
            return ProviderGenerationResult(text=payload.model_dump_json())

    runtime = _healthy_gemini_runtime()
    client = _FakeGeminiClient()
    runtime._client_for_provider = lambda provider_name: client

    draft = runtime.generate_draft(
        GoalDraft,
        "build a goal",
        source_scope="goal:test",
    )

    assert draft.payload.title == "Calculus"
    assert client.structured_calls == 1
    assert client.legacy_calls == 1


def test_legacy_empty_max_tokens_retries_with_expanded_budget_before_failure() -> None:
    """An empty-body max-token failure should retry once with a larger token budget."""

    class _FakeLegacyGeminiClient:
        def __init__(self) -> None:
            self.calls: list[int] = []

        def generate_with_model(self, prompt, model, timeout=30, *, temperature=None, max_output_tokens=None):
            self.calls.append(max_output_tokens or 0)
            return ProviderGenerationResult(
                error=ProviderErrorDetails(
                    category="empty",
                    provider="gemini",
                    model=model,
                    raw_message="Gemini returned no usable text in an HTTP 200 response. finish reason=max tokens",
                    http_status=200,
                    retryable=False,
                )
            )

    runtime = _healthy_gemini_runtime()
    client = _FakeLegacyGeminiClient()
    runtime._client_for_provider = lambda provider_name: client

    with pytest.raises(DraftGenerationError) as exc_info:
        runtime.generate_draft(
            GoalDraft,
            "build a goal",
            source_scope="goal:empty-retry",
            max_output_tokens=2000,
        )

    assert exc_info.value.error.category == "empty"
    assert client.calls == [2000, 4000]
