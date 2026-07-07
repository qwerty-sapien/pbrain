# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for shared LLM request policy helpers."""

from types import SimpleNamespace

from pb.llm.gemini import FLASH_MODEL, PRO_MODEL
from pb.llm.policy import model_display_label, resolve_timeout


def test_resolve_timeout_extends_gemini_pro_to_configured_window() -> None:
    config = SimpleNamespace(llm=SimpleNamespace(long_model_timeout_seconds=120))

    assert resolve_timeout("gemini", PRO_MODEL, 30, config=config) == 120


def test_resolve_timeout_keeps_flash_budget_when_not_premium() -> None:
    config = SimpleNamespace(llm=SimpleNamespace(long_model_timeout_seconds=120))

    assert resolve_timeout("gemini", FLASH_MODEL, 30, config=config) == 30


def test_model_display_label_uses_human_names() -> None:
    assert model_display_label("gemini", PRO_MODEL) == "Gemini Pro"
    assert model_display_label("anthropic", "claude-opus-4-1") == "Claude Opus"
    assert model_display_label("openai", "gpt-5.5") == "GPT-5.5"
