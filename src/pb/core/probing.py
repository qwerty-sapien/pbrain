# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Socratic concept probing engine (D-47 to D-49).

When a concept definition exceeds 30 words, the probing engine activates
with a frank/impatient mentor persona that asks pointed questions to
force the user to crystallize vague understanding.

The probing is NOT a test -- it's adaptive Socratic questioning that
assumes technical foundations but struggles to articulate precisely.
"""
from __future__ import annotations

from typing import Optional

import structlog

from pb.llm.gemini import get_client

logger = structlog.get_logger()

WORD_THRESHOLD = 30
MAX_ROUNDS = 5

SYSTEM_PERSONA = (
    "You are a frank, impatient mentor who does NOT hand-hold. "
    "You force the student to find the answer themselves with hints, not explanations. "
    "You tolerate rambling but redirect to precision. "
    "You ask thought-provoking questions with varying difficulty. "
    "You assume the user has technical foundations but struggles to articulate precisely. "
    "You never explain the answer -- you ask a question that makes the gap obvious. "
    "Keep questions short (1-2 sentences max). One question at a time."
)


def should_probe(definition: str) -> bool:
    """Check if definition is long enough to warrant probing (D-47: >30 words)."""
    if not definition or not definition.strip():
        return False
    word_count = len(definition.strip().split())
    return word_count > WORD_THRESHOLD


class ProbingEngine:
    """Drives a Socratic probing session for a concept definition."""

    def __init__(self, concept_name: str, definition: str, domain: str = ""):
        self.concept_name = concept_name
        self.definition = definition
        self.domain = domain
        self._round = 0
        self._done = False
        self._history: list[dict[str, str]] = []  # {"role": "mentor"/"user", "text": "..."}
        self._client = get_client()

    @property
    def round_number(self) -> int:
        return self._round

    def should_continue(self) -> bool:
        """Whether the probing session should ask another question."""
        if self._done:
            return False
        if self._round >= MAX_ROUNDS:
            return False
        return True

    def mark_done(self) -> None:
        """User wants to exit probing."""
        self._done = True

    def build_initial_prompt(self) -> str:
        """Build the first probing question prompt."""
        domain_ctx = f" in the domain of {self.domain}" if self.domain else ""
        return (
            f"{SYSTEM_PERSONA}\n\n"
            f"The student defined '{self.concept_name}'{domain_ctx} as:\n"
            f'"{self.definition}"\n\n'
            f"This definition is wordy. Ask ONE pointed question that exposes "
            f"the gap between what they wrote and what they actually understand. "
            f"Force them to be more precise."
        )

    def build_followup_prompt(self, user_answer: str) -> str:
        """Build a follow-up probing prompt based on user's answer."""
        history_text = ""
        for entry in self._history:
            prefix = "Mentor" if entry["role"] == "mentor" else "Student"
            history_text += f"{prefix}: {entry['text']}\n"

        return (
            f"{SYSTEM_PERSONA}\n\n"
            f"Concept: '{self.concept_name}'\n"
            f"Original definition: \"{self.definition}\"\n\n"
            f"Conversation so far:\n{history_text}\n"
            f"Student's latest answer: \"{user_answer}\"\n\n"
            f"Based on their answer, ask ONE follow-up question that pushes "
            f"them deeper. If they're getting more precise, increase difficulty. "
            f"If they're still vague, redirect with a hint."
        )

    def get_question(self, user_answer: Optional[str] = None) -> Optional[str]:
        """Get the next probing question from the LLM.

        Args:
            user_answer: User's response to the previous question (None for first question).

        Returns:
            Question string, or None if LLM unavailable.
        """
        if not self._client.is_available():
            return None

        if user_answer is not None:
            self._history.append({"role": "user", "text": user_answer})

        if self._round == 0:
            prompt = self.build_initial_prompt()
        else:
            prompt = self.build_followup_prompt(user_answer or "")

        response = self._client.generate(prompt)
        if response:
            self._history.append({"role": "mentor", "text": response})
            self._round += 1
            return response

        return None

    def get_summary(self) -> str:
        """Summarize what was discussed during probing."""
        lines = [f"Probing session for '{self.concept_name}': {self._round} rounds"]
        for entry in self._history:
            prefix = "Q" if entry["role"] == "mentor" else "A"
            lines.append(f"  {prefix}: {entry['text'][:80]}...")
        return "\n".join(lines)
