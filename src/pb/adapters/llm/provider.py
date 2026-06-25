# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""LLM provider interface backed by the configured Gemini runtime."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

import structlog

from pb.llm.gemini import FLASH_MODEL, get_client


logger = structlog.get_logger()


class LLMProvider(ABC):
    """Small adapter interface for legacy call sites."""

    @abstractmethod
    def summarize_session(self, session_notes: str) -> str:
        """Summarize session notes into a concise recap."""

    @abstractmethod
    def compress_clip(self, raw_text: str) -> str:
        """Compress captured text into a concise packet."""

    @abstractmethod
    def extract_actions(self, notes: str) -> list[str]:
        """Extract explicit next actions from notes."""

    @abstractmethod
    def draft_session_summary(self, context: str) -> str:
        """Draft a concise session summary from context."""


class GeminiProvider(LLMProvider):
    """Gemini-backed provider. Raises when credentials are unavailable."""

    def __init__(self, model: str = FLASH_MODEL):
        self.model = model
        self._client = get_client()
        if not self._client.is_available():
            raise RuntimeError(
                "pb requires an LLM for this workflow.\n\n"
                "Run:\n"
                "  pb init llm\n\n"
                "Then configure Gemini credentials via GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT."
            )

    def _generate(self, prompt: str, *, max_output_tokens: int = 800) -> str:
        result = self._client.generate_with_model(
            prompt,
            self.model,
            timeout=30,
            max_output_tokens=max_output_tokens,
        )
        if not result or not result.strip():
            raise RuntimeError("Gemini returned an empty response.")
        return result.strip()

    def summarize_session(self, session_notes: str) -> str:
        return self._generate(
            "Summarize this learning session in 4-6 tight sentences. "
            "Focus on what was learned, what degraded, and the next adjustment.\n\n"
            f"{session_notes}",
        )

    def compress_clip(self, raw_text: str) -> str:
        return self._generate(
            "Compress this learning artifact into a concise study note. Preserve specific facts and action items.\n\n"
            f"{raw_text}",
        )

    def extract_actions(self, notes: str) -> list[str]:
        raw = self._generate(
            "Extract up to five explicit next actions from these learning notes. "
            "Return one action per line with no bullets or numbering.\n\n"
            f"{notes}",
            max_output_tokens=4000,
        )
        return [line.strip() for line in raw.splitlines() if line.strip()]

    def draft_session_summary(self, context: str) -> str:
        return self._generate(
            "Draft a short markdown summary for this learning session. "
            "Include summary, evidence, friction, and next action headings.\n\n"
            f"{context}",
        )


def get_provider(provider_name: Optional[str] = None) -> LLMProvider:
    """Return the configured Gemini-backed provider."""
    logger.debug("llm.get_provider", requested=provider_name or "gemini", actual="gemini")
    return GeminiProvider()
