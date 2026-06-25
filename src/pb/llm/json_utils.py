# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared JSON extraction helpers for LLM responses."""

from __future__ import annotations


def extract_json_block(raw: str) -> str:
    """Extract the first JSON object/array from an LLM response."""
    text = (raw or "").strip()
    if not text:
        raise ValueError("Empty model response.")
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    if text.startswith("{") or text.startswith("["):
        return text
    start = min([index for index in (text.find("{"), text.find("[")) if index != -1], default=-1)
    if start == -1:
        raise ValueError("No JSON object found in model response.")
    end = max(text.rfind("}"), text.rfind("]"))
    if end == -1 or end <= start:
        raise ValueError("Incomplete JSON payload in model response.")
    return text[start : end + 1]
