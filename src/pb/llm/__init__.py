# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Provider-neutral LLM helpers for pbrain."""

__all__ = ["LLMClient", "LLMRuntime", "score_text_response", "generate_followup"]


def __getattr__(name: str):
    if name in {"LLMClient", "score_text_response", "generate_followup"}:
        from pb.llm import review

        return getattr(review, name)
    if name == "LLMRuntime":
        from pb.llm.runtime import LLMRuntime

        return LLMRuntime
    raise AttributeError(name)
