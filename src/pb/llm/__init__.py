# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Gemini Flash Lite client for review scoring.

Per LLM_PROVIDER_SPEC.md: LLM cannot mutate state, only score/summarize.
"""

from pb.llm.gemini import GeminiClient, score_text_response, generate_followup

__all__ = ["GeminiClient", "score_text_response", "generate_followup"]
