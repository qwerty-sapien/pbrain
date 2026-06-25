# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared question-presentation transforms for live lesson sessions."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


class QuestionTransformService:
    """Preserve durable question identity while mutating presentation state."""

    _DEFAULT_DIFFICULTY = "baseline"

    @classmethod
    def initial_metadata(
        cls,
        *,
        prompt_json: dict[str, Any],
        answer_json: dict[str, Any],
        source_refs: list[str] | None = None,
        active_context_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        """Return the default transform metadata payload for a new lesson item."""

        return {
            "difficulty": cls._DEFAULT_DIFFICULTY,
            "last_transform": "",
            "original_prompt_json": deepcopy(prompt_json),
            "original_answer_json": deepcopy(answer_json),
            "source_refs": [str(item).strip() for item in (source_refs or []) if str(item).strip()],
            "active_context_ids": [
                str(item).strip() for item in (active_context_ids or []) if str(item).strip()
            ],
        }

    @classmethod
    def hydrated_metadata(cls, question: Any) -> dict[str, Any]:
        """Return a complete metadata payload for the current question row."""

        existing = deepcopy(getattr(question, "metadata_json", {}) or {})
        metadata = existing if isinstance(existing, dict) else {}
        if not metadata.get("original_prompt_json"):
            metadata["original_prompt_json"] = deepcopy(getattr(question, "prompt_json", {}) or {})
        if not metadata.get("original_answer_json"):
            metadata["original_answer_json"] = deepcopy(getattr(question, "answer_json", {}) or {})
        metadata["difficulty"] = str(
            metadata.get("difficulty", cls._DEFAULT_DIFFICULTY) or cls._DEFAULT_DIFFICULTY
        )
        metadata["last_transform"] = str(metadata.get("last_transform", "") or "")
        metadata["source_refs"] = [
            str(item).strip()
            for item in list(metadata.get("source_refs", []) or [])
            if str(item).strip()
        ]
        metadata["active_context_ids"] = [
            str(item).strip()
            for item in list(metadata.get("active_context_ids", []) or [])
            if str(item).strip()
        ]
        return metadata

    @classmethod
    def transformed_metadata(cls, question: Any, *, transform: str) -> dict[str, Any]:
        """Return updated metadata after a presentation-only transform."""

        metadata = cls.hydrated_metadata(question)
        metadata["last_transform"] = transform
        metadata["difficulty"] = cls._difficulty_after(
            str(metadata.get("difficulty", cls._DEFAULT_DIFFICULTY) or cls._DEFAULT_DIFFICULTY),
            transform=transform,
        )
        return metadata

    @classmethod
    def retry_metadata(cls, question: Any) -> dict[str, Any]:
        """Return metadata for a retry question cloned from an existing item."""

        metadata = cls.hydrated_metadata(question)
        metadata["last_transform"] = "retry"
        metadata["difficulty"] = "retry"
        return metadata

    @classmethod
    def _difficulty_after(cls, current: str, *, transform: str) -> str:
        normalized = (transform or "").strip().lower()
        if normalized in {"harder", "easier", "retry", "drill"}:
            return normalized
        return current or cls._DEFAULT_DIFFICULTY
