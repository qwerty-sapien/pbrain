# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared prompt guidance for learning-first drafting flows."""

from __future__ import annotations

import unicodedata

_ENGLISH_CODES = {"en", "english", "eng"}

# Unicode script ranges we can name deterministically. The value is the human
# language name we pin the model to. Anything outside these ranges is treated as
# Latin-script English so the default path always resolves to a concrete pin.
_SCRIPT_LANGUAGES: tuple[tuple[str, str], ...] = (
    ("CJK", "Chinese"),
    ("HIRAGANA", "Japanese"),
    ("KATAKANA", "Japanese"),
    ("HANGUL", "Korean"),
    ("CYRILLIC", "Russian"),
    ("ARABIC", "Arabic"),
    ("HEBREW", "Hebrew"),
    ("DEVANAGARI", "Hindi"),
    ("GREEK", "Greek"),
    ("THAI", "Thai"),
)


def detect_language_name(text: str) -> str:
    """Best-effort, dependency-free language name from the dominant script.

    Delegating language *detection* to the LLM is what let the Gemini preview
    models drift into Chinese on English input. We instead resolve a concrete
    language name here and pin it explicitly. Latin-script text (the common
    case) resolves to English.
    """

    counts: dict[str, int] = {}
    for char in text or "":
        if not char.isalpha():
            continue
        try:
            name = unicodedata.name(char)
        except ValueError:
            continue
        for prefix, language in _SCRIPT_LANGUAGES:
            if prefix in name:
                counts[language] = counts.get(language, 0) + 1
                break
    if not counts:
        return "English"
    return max(counts, key=counts.get)


def _pin(language: str) -> str:
    return (
        f"IMPORTANT: Write your entire response in {language}. "
        "Do not switch to or mix in any other language at any point, "
        "unless the user explicitly asks you to.\n"
    )


def language_instruction(user_input: str = "", *, configured: str = "auto") -> str:
    """Return a language directive for LLM prompts.

    Always emits a positive, single-language pin. The Gemini preview models do
    not reliably default to English, so an empty directive (or an 'auto'
    directive that lists other languages as examples) let them respond in
    Chinese at random. We resolve one concrete language and pin it.

    configured='auto': detect the user's language from ``user_input`` and pin it
        (English when the input is Latin-script or empty).
    configured='en' (or variants): pin English explicitly.
    configured=<code/name>: pin that language exclusively.
    """
    lang = (configured or "auto").strip().lower()
    if lang in _ENGLISH_CODES:
        return _pin("English")
    if lang == "auto":
        return _pin(detect_language_name(user_input))
    return _pin(configured.strip())


def learning_intent_style_guidance() -> str:
    """Return prompt guidance for intent-first, low-jargon learning UX."""

    return (
        "Match the user's likely intent and keep the language low-jargon by default.\n"
        "For accent, pronunciation, naturalness, fluency, or speaking-style requests, default to performance coaching, imitation, or production practice.\n"
        "Do not introduce sociolinguistics, dialect-comparison analysis, language history, or academic framing unless the user explicitly asks for it.\n"
        "Do not widen a region-specific speech request into unrelated dialect or regional comparisons unless the user explicitly asks for them.\n"
        "If the user already signals fluency, do not inject beginner prerequisites or standard-language detours unless they explicitly ask for them.\n"
        "All learner-facing mathematical, statistical, logical, symbolic, or formula-like text must use explicit LaTeX delimiters.\n"
        "Use inline math as `$...$` and display math as `$$...$$`; every LaTeX command must keep its leading backslash, e.g. `$\\mathbb{R}^n$`.\n"
        "Never emit pseudo-math such as S'_{i}, f_i(...), Pa(S'_i), or plain ASCII equations when LaTeX is appropriate.\n"
        "When a schema field supports a `RenderableText` object, put math-bearing content in `{text: ..., is_latex: true}` or keep the math delimiters inside the text field.\n"
    )
