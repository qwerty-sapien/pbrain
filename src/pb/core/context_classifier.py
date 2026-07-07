# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""LLM classification of uploaded context sources.

The deterministic intake in :mod:`pb.core.context_file_intake` can extract text
but cannot tell what a source is *about* — it falls back to filename heuristics
and a small keyword table, which collapses most real papers to "general
learning". This module takes an extracted text excerpt and asks a cheap Flash
Lite call for the real broad domain, a one-line topic summary, and a scope
boundary sentence that names the source. It is deliberately fail-safe: any error
(offline, no key, quota, malformed output) returns ``None`` so intake never
breaks.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from pb.core.learning_prompting import language_instruction
from pb.llm.gemini import FLASH_LITE_MODEL
from pb.llm.runtime import LLMRuntime


class SourceClassification(BaseModel):
    """Real, content-derived identity for one uploaded source."""

    domain_name: str = Field(
        description="Broad knowledge domain in 2-5 words, e.g. 'causal inference' or 'inorganic chemistry'."
    )
    topic_summary: str = Field(
        description="One sentence (<=200 chars) stating what this specific source is about."
    )
    scope_boundary: str = Field(
        description="One sentence telling a tutor to stay within this source, naming it."
    )


def _classification_prompt(filename: str, excerpt: str) -> str:
    return (
        language_instruction(configured="en")
        + "You are classifying one uploaded learning source for a study tool.\n"
        "Use ONLY the filename and text excerpt below. Do not invent facts that "
        "are not supported by the excerpt.\n\n"
        f"Filename: {filename}\n"
        f"Text excerpt:\n\"\"\"\n{excerpt}\n\"\"\"\n\n"
        "Return the broad knowledge domain this belongs to, a one-sentence "
        "summary of what THIS source is about, and a one-sentence scope boundary "
        "that names the source and tells a tutor to stay within it unless the "
        "learner explicitly asks for outside material."
    )


def classify_context_source(
    runtime: LLMRuntime,
    *,
    filename: str,
    excerpt: str,
) -> SourceClassification | None:
    """Classify one source excerpt, or return ``None`` if it cannot be done."""

    text = (excerpt or "").strip()
    if len(text) < 40:
        return None
    if not runtime.health().available:
        return None
    try:
        draft = runtime.generate_draft(
            SourceClassification,
            _classification_prompt(filename, text[:1600]),
            source_scope="context_source_classification",
            model=FLASH_LITE_MODEL,
            timeout=30,
            max_output_tokens=400,
        )
    except Exception:
        return None
    payload = draft.payload
    if not isinstance(payload, SourceClassification):
        return None
    if not payload.domain_name.strip():
        return None
    return payload
