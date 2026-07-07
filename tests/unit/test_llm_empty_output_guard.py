"""Unit tests for the LLM empty-output retry guard (D-10).

Tests cover:
- Empty first response triggers retry with stronger settings; valid retry succeeds
- Two consecutive empties raise DraftGenerationError with category='empty'
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pb.llm.gemini import FLASH_MODEL
from pb.llm.runtime import (
    DraftGenerationError,
    LLMRuntime,
    ProviderGenerationResult,
)
from pb.llm.drafts import GoalDraft
from pb.storage.config import (
    Config,
    GeneralConfig,
    ProviderConfig,
    ModelRolesConfig,
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


_VALID_GOAL_DRAFT = GoalDraft(
    title="Calculus",
    description="Learn calculus.",
    domain="calculus",
    execution_mode="study",
    horizon="quarter",
    framework="Bloom",
    study_framework="bloom_retrieval",
    target_bloom_stage="apply",
    success_definition="Apply calculus.",
    feedback_source="artifact",
    evidence_type="artifact",
)


def test_empty_output_retries_with_stronger_settings(monkeypatch):
    """Empty first response triggers one retry; second non-empty response succeeds."""
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    call_count = {"n": 0}

    def fake_generate(prompt, model, *, timeout, max_output_tokens):
        call_count["n"] += 1
        if call_count["n"] == 1:
            return ProviderGenerationResult(text="", error=None)
        return ProviderGenerationResult(
            text=_VALID_GOAL_DRAFT.model_dump_json(), error=None,
        )

    client_mock = MagicMock()
    client_mock.generate_with_model.side_effect = fake_generate

    with patch.object(runtime, "_client_for_provider", return_value=client_mock):
        draft = runtime.generate_draft(
            GoalDraft, "build a goal", source_scope="test:calculus",
        )

    assert call_count["n"] == 2, f"Expected 2 calls (empty + retry), got {call_count['n']}"
    assert draft.payload.domain == "calculus"


def test_empty_output_raises_draft_generation_error_after_retry(monkeypatch):
    """Two consecutive empty responses raise DraftGenerationError, not crash."""
    runtime = LLMRuntime(_config_for_provider("gemini", FLASH_MODEL))
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")

    def always_empty(prompt, model, *, timeout, max_output_tokens):
        return ProviderGenerationResult(text="", error=None)

    client_mock = MagicMock()
    client_mock.generate_with_model.side_effect = always_empty

    with patch.object(runtime, "_client_for_provider", return_value=client_mock):
        with pytest.raises(DraftGenerationError) as exc_info:
            runtime.generate_draft(
                GoalDraft, "build a goal", source_scope="test:fail",
            )

    assert exc_info.value.error.category == "empty"
