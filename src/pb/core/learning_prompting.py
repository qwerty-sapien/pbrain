# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared prompt guidance for learning-first drafting flows."""

from __future__ import annotations

_ENGLISH_CODES = {"en", "english", "eng"}


def language_instruction(user_input: str = "", *, configured: str = "auto") -> str:
    """Return a language directive for LLM prompts.

    configured='auto': instruct the model to match the user's input language.
    configured='en' (or variants): no directive (English is the model default).
    configured=<code>: instruct the model to use that language exclusively.
    """
    lang = (configured or "auto").strip().lower()
    if lang in _ENGLISH_CODES:
        return ""
    if lang == "auto":
        return (
            "IMPORTANT: Respond in the same language as the user's input. "
            "If the user wrote in Chinese, respond in Chinese. "
            "If in French, respond in French. "
            "Match the user's language exactly — do not translate to English unless explicitly asked.\n"
        )
    return f"IMPORTANT: Respond in {configured}. Do not use English unless the user explicitly requests it.\n"


def learning_intent_style_guidance() -> str:
    """Return prompt guidance for intent-first, low-jargon learning UX."""

    return (
        "Match the user's likely intent and keep the language low-jargon by default.\n"
        "For accent, pronunciation, naturalness, fluency, or speaking-style requests, default to performance coaching, imitation, or production practice.\n"
        "Do not introduce sociolinguistics, dialect-comparison analysis, language history, or academic framing unless the user explicitly asks for it.\n"
        "Do not widen a region-specific speech request into unrelated dialect or regional comparisons unless the user explicitly asks for them.\n"
        "If the user already signals fluency, do not inject beginner prerequisites or standard-language detours unless they explicitly ask for them.\n"
    )
