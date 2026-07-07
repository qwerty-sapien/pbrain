"""Unit tests for Socratic probing engine (D-47 to D-49).

Tests cover:
- Word count trigger (should_probe)
- Prompt construction (build_initial_prompt, build_followup_prompt)
- Round counting and exit conditions (should_continue, mark_done)
- LLM interaction (get_question mocked)
- Persona content validation
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pb.core.probing import (
    MAX_ROUNDS,
    SYSTEM_PERSONA,
    WORD_THRESHOLD,
    ProbingEngine,
    should_probe,
)


# ---------------------------------------------------------------------------
# should_probe tests
# ---------------------------------------------------------------------------


def test_should_probe_empty_string():
    """Empty definition should not trigger probing."""
    assert should_probe("") is False


def test_should_probe_whitespace_only():
    """Whitespace-only definition should not trigger probing."""
    assert should_probe("   ") is False


def test_should_probe_short_definition():
    """A definition under 30 words should return False."""
    assert should_probe("A short definition with fewer than thirty words here.") is False


def test_should_probe_exactly_30_words():
    """Exactly 30 words should NOT trigger probing (threshold is >30)."""
    definition = " ".join(["word"] * 30)
    assert should_probe(definition) is False


def test_should_probe_31_words_triggers():
    """31 words should trigger probing (threshold >30)."""
    definition = " ".join(["word"] * 31)
    assert should_probe(definition) is True


def test_should_probe_long_definition():
    """A long definition should return True."""
    long_def = "a " * 50
    assert should_probe(long_def) is True


def test_should_probe_word_threshold_value():
    """WORD_THRESHOLD constant is 30."""
    assert WORD_THRESHOLD == 30


# ---------------------------------------------------------------------------
# SYSTEM_PERSONA content tests
# ---------------------------------------------------------------------------


def test_persona_contains_frank():
    """Persona must include 'frank'."""
    assert "frank" in SYSTEM_PERSONA.lower()


def test_persona_contains_impatient():
    """Persona must include 'impatient'."""
    assert "impatient" in SYSTEM_PERSONA.lower()


def test_persona_does_not_hand_hold():
    """Persona must emphasize NOT hand-holding."""
    assert "not" in SYSTEM_PERSONA.lower() or "NOT" in SYSTEM_PERSONA
    assert "hand-hold" in SYSTEM_PERSONA.lower() or "hand_hold" in SYSTEM_PERSONA.lower()


# ---------------------------------------------------------------------------
# ProbingEngine construction tests
# ---------------------------------------------------------------------------


def test_probing_engine_initial_round_is_zero():
    """Engine starts at round 0."""
    engine = ProbingEngine("VAE", "A variational autoencoder is a thing.")
    assert engine.round_number == 0


def test_probing_engine_should_continue_initially():
    """Engine should continue when fresh (under max rounds)."""
    engine = ProbingEngine("VAE", "def")
    assert engine.should_continue() is True


def test_probing_engine_should_continue_false_after_max_rounds():
    """should_continue returns False when _round >= MAX_ROUNDS."""
    engine = ProbingEngine("VAE", "def")
    engine._round = MAX_ROUNDS
    assert engine.should_continue() is False


def test_probing_engine_should_continue_false_after_mark_done():
    """should_continue returns False after mark_done is called."""
    engine = ProbingEngine("VAE", "def")
    engine.mark_done()
    assert engine.should_continue() is False


def test_max_rounds_constant():
    """MAX_ROUNDS is 5."""
    assert MAX_ROUNDS == 5


# ---------------------------------------------------------------------------
# Prompt construction tests
# ---------------------------------------------------------------------------


def test_build_initial_prompt_contains_concept_name():
    """Initial prompt must include the concept name."""
    engine = ProbingEngine("Variational Autoencoder", "some definition")
    prompt = engine.build_initial_prompt()
    assert "Variational Autoencoder" in prompt


def test_build_initial_prompt_contains_definition():
    """Initial prompt must include the definition text."""
    definition = "A model that learns latent representations."
    engine = ProbingEngine("VAE", definition)
    prompt = engine.build_initial_prompt()
    assert definition in prompt


def test_build_initial_prompt_contains_precise():
    """Initial prompt should push for precision."""
    engine = ProbingEngine("VAE", "definition text")
    prompt = engine.build_initial_prompt()
    assert "precise" in prompt.lower()


def test_build_initial_prompt_contains_persona():
    """Initial prompt should embed the SYSTEM_PERSONA."""
    engine = ProbingEngine("VAE", "definition text")
    prompt = engine.build_initial_prompt()
    assert "frank" in prompt.lower() or SYSTEM_PERSONA[:30] in prompt


def test_build_initial_prompt_includes_domain_when_provided():
    """Domain should appear in initial prompt when supplied."""
    engine = ProbingEngine("Attention", "a mechanism", domain="NLP")
    prompt = engine.build_initial_prompt()
    assert "NLP" in prompt


def test_build_followup_prompt_includes_user_answer():
    """Follow-up prompt must include the user's previous answer."""
    engine = ProbingEngine("VAE", "long definition")
    engine._history = [{"role": "mentor", "text": "What does latent mean?"}]
    prompt = engine.build_followup_prompt("I think latent means hidden")
    assert "I think latent means hidden" in prompt


def test_build_followup_prompt_includes_history():
    """Follow-up prompt should include conversation history."""
    engine = ProbingEngine("VAE", "some definition")
    engine._history = [{"role": "mentor", "text": "First question?"}]
    prompt = engine.build_followup_prompt("My answer here")
    assert "First question?" in prompt


def test_build_followup_prompt_includes_concept():
    """Follow-up prompt must include concept name."""
    engine = ProbingEngine("Backpropagation", "some definition")
    prompt = engine.build_followup_prompt("some answer")
    assert "Backpropagation" in prompt


# ---------------------------------------------------------------------------
# get_question LLM interaction tests
# ---------------------------------------------------------------------------


def test_get_question_returns_none_when_llm_unavailable():
    """get_question returns None when LLM is not available."""
    engine = ProbingEngine("VAE", "definition")
    engine._client = MagicMock()
    engine._client.is_available.return_value = False
    result = engine.get_question()
    assert result is None


def test_get_question_returns_string_when_llm_available():
    """get_question returns the LLM response string."""
    engine = ProbingEngine("VAE", "some definition text")
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "What is the key property of the latent space?"
    result = engine.get_question()
    assert result == "What is the key property of the latent space?"


def test_get_question_increments_round():
    """Each call to get_question increments the round counter."""
    engine = ProbingEngine("VAE", "some definition text")
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "A question?"
    assert engine.round_number == 0
    engine.get_question()
    assert engine.round_number == 1


def test_get_question_adds_to_history():
    """get_question appends mentor response to history."""
    engine = ProbingEngine("VAE", "definition")
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "What does regularization do here?"
    engine.get_question()
    assert len(engine._history) == 1
    assert engine._history[0]["role"] == "mentor"
    assert "regularization" in engine._history[0]["text"]


def test_get_question_with_user_answer_adds_to_history():
    """When user_answer provided, it is added to history before mentor response."""
    engine = ProbingEngine("VAE", "definition")
    engine._round = 1  # Simulate already past first question
    engine._history = [{"role": "mentor", "text": "Initial question?"}]
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "Follow-up question?"
    engine.get_question(user_answer="My answer")
    # History: mentor Q1, user A1, mentor Q2
    roles = [h["role"] for h in engine._history]
    assert "user" in roles
    assert roles.count("mentor") == 2


def test_get_question_returns_none_when_generate_returns_none():
    """get_question returns None if LLM generate returns None."""
    engine = ProbingEngine("VAE", "definition")
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = None
    result = engine.get_question()
    assert result is None


# ---------------------------------------------------------------------------
# get_summary test
# ---------------------------------------------------------------------------


def test_get_summary_contains_concept_name():
    """Summary includes the concept name and round count."""
    engine = ProbingEngine("Entropy", "definition here")
    engine._round = 2
    engine._history = [
        {"role": "mentor", "text": "What type of entropy?"},
        {"role": "user", "text": "Shannon entropy"},
    ]
    summary = engine.get_summary()
    assert "Entropy" in summary
    assert "2" in summary
