"""Integration tests for Socratic probing in the concept note creation flow (D-47 to D-49).

Tests verify:
- should_probe threshold for various definition lengths
- ProbingEngine drives a mocked session correctly
- Max-rounds exit condition
- mark_done exit condition
- pb note concept command is registered in CLI
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pb.core.probing import ProbingEngine, should_probe


# ---------------------------------------------------------------------------
# 1. should_probe threshold
# ---------------------------------------------------------------------------


def test_should_probe_threshold():
    """should_probe returns correct bool for various lengths."""
    assert should_probe("") is False
    assert should_probe("short") is False
    assert should_probe(" ".join(["word"] * 30)) is False   # exactly 30 words: False
    assert should_probe(" ".join(["word"] * 31)) is True    # 31 words: True
    assert should_probe(" ".join(["word"] * 100)) is True   # way over: True


# ---------------------------------------------------------------------------
# 2. Mock probing session
# ---------------------------------------------------------------------------


def test_probing_engine_mock_session():
    """ProbingEngine drives a question and increments round counter."""
    engine = ProbingEngine("VAE", "some definition here")
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "What does the latent space represent?"

    question = engine.get_question()

    assert question == "What does the latent space represent?"
    assert engine.round_number == 1


# ---------------------------------------------------------------------------
# 3. Max-rounds exit condition
# ---------------------------------------------------------------------------


def test_probing_exits_on_max_rounds():
    """After 5 rounds, should_continue() returns False."""
    engine = ProbingEngine("Entropy", "a " * 50)
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "A question?"

    for _ in range(5):
        engine.get_question(user_answer="an answer" if engine.round_number > 0 else None)

    assert engine.round_number == 5
    assert engine.should_continue() is False


# ---------------------------------------------------------------------------
# 4. mark_done exit condition
# ---------------------------------------------------------------------------


def test_probing_exits_on_mark_done():
    """After mark_done(), should_continue() returns False regardless of rounds."""
    engine = ProbingEngine("Gradient", "long definition text " * 5)
    assert engine.should_continue() is True
    engine.mark_done()
    assert engine.should_continue() is False


# ---------------------------------------------------------------------------
# 5. CLI command registration
# ---------------------------------------------------------------------------


def test_concept_note_command_registered():
    """pb note concept command appears in CLI help output."""
    from pb.cli.main import app

    runner = CliRunner()
    result = runner.invoke(app, ["note", "--help"])
    assert result.exit_code == 0
    assert "concept" in result.output


# ---------------------------------------------------------------------------
# 6. _run_probing skips when LLM unavailable
# ---------------------------------------------------------------------------


def test_run_probing_skips_when_llm_unavailable(capsys):
    """_run_probing prints skip message when LLM is not available."""
    from pb.cli.commands.note import _run_probing

    with patch("pb.core.probing.get_client") as mock_get:
        mock_client = MagicMock()
        mock_client.is_available.return_value = False
        mock_get.return_value = mock_client

        _run_probing("VAE", "a " * 40, "ML")
        captured = capsys.readouterr()
        assert "Probing skipped" in captured.out


# ---------------------------------------------------------------------------
# 7. ProbingEngine history tracking across rounds
# ---------------------------------------------------------------------------


def test_probing_engine_history_grows_across_rounds():
    """History entries accumulate across multiple get_question calls."""
    engine = ProbingEngine("Attention", "some definition " * 5)
    engine._client = MagicMock()
    engine._client.is_available.return_value = True
    engine._client.generate.return_value = "A follow-up question?"

    engine.get_question()                          # round 1
    engine.get_question(user_answer="My answer")   # round 2

    # After round 1: 1 mentor entry
    # After round 2: +1 user + 1 mentor = 3 total
    assert len(engine._history) == 3
    roles = [h["role"] for h in engine._history]
    assert roles == ["mentor", "user", "mentor"]
