# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Provider-neutral LLM helpers for legacy review prompts."""

from __future__ import annotations

from typing import Optional

import structlog
from pydantic import BaseModel, Field

from pb.core.model_policy import resolve_model_binding
from pb.llm.runtime import DraftGenerationError, LLMRuntime

logger = structlog.get_logger()


class LLMClient:
    """Small availability facade for legacy interactive review code."""

    def __init__(self, runtime: LLMRuntime | None = None) -> None:
        self._runtime = runtime

    def _get_runtime(self) -> LLMRuntime:
        if self._runtime is None:
            self._runtime = LLMRuntime()
        return self._runtime

    def is_available(self) -> bool:
        try:
            return self._get_runtime().health().available
        except Exception:
            return False


class _ReviewScoreDraft(BaseModel):
    score: int = Field(ge=1, le=10)
    rationale: str


class _FollowupDraft(BaseModel):
    question: str


def score_text_response(question: str, response: str) -> Optional[tuple[int, str]]:
    """Score a free-text review response with the configured LLM provider."""
    try:
        runtime = LLMRuntime()
        binding = resolve_model_binding(runtime.config, "answer_check")
        draft = runtime.generate_draft(
            _ReviewScoreDraft,
            (
                "Evaluate this self-reflection response from a daily learning review.\n\n"
                f"Question asked: {question}\n"
                f"User response: {response}\n\n"
                "Score clarity, specificity, actionable insight, and honest self-awareness. "
                "Return a score from 1 to 10 and a one-sentence rationale."
            ),
            source_scope="review:score_text_response",
            model=binding,
            timeout=30,
            max_output_tokens=1200,
        )
    except DraftGenerationError as exc:
        logger.debug("review.score_text_response_failed", category=exc.error.category)
        return None
    except Exception as exc:
        logger.debug("review.score_text_response_error", error=str(exc))
        return None

    payload = draft.payload
    if not isinstance(payload, _ReviewScoreDraft):
        payload = _ReviewScoreDraft.model_validate(payload)
    return payload.score, payload.rationale.strip()


def generate_followup(question: str, response: str) -> Optional[str]:
    """Generate one concise follow-up question with the configured LLM provider."""
    try:
        runtime = LLMRuntime()
        binding = resolve_model_binding(runtime.config, "small_retry")
        draft = runtime.generate_draft(
            _FollowupDraft,
            (
                "The user gave a vague or incomplete response during a daily learning review.\n\n"
                f"Original question: {question}\n"
                f"User response: {response}\n\n"
                "Generate one short follow-up question, at most 15 words, that helps them be more specific."
            ),
            source_scope="review:generate_followup",
            model=binding,
            timeout=30,
            max_output_tokens=800,
        )
    except DraftGenerationError as exc:
        logger.debug("review.generate_followup_failed", category=exc.error.category)
        return None
    except Exception as exc:
        logger.debug("review.generate_followup_error", error=str(exc))
        return None

    payload = draft.payload
    if not isinstance(payload, _FollowupDraft):
        payload = _FollowupDraft.model_validate(payload)
    return payload.question.strip().strip('"') or None
