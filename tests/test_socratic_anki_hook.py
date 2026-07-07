"""Tests for ANKI-03 hook wired into run_debrief_loop() (26-03).

RED phase: these tests are written before implementation.
They verify:
- Backward compatibility (no new required args)
- Hook fires on deep debrief (engine._max >= 5) with generate_anki=True
- Hook skipped on mini debrief (engine._max < 5)
- Hook failure is non-fatal (pairs still returned on exception)
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch, call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine(max_rounds: int, pairs=None):
    """Create a mock SocraticDebriefEngine.

    get_question() returns a question on first call, then None to end the loop.
    should_continue() returns False so the while loop exits immediately after
    the first iteration.
    """
    engine = MagicMock()
    engine._max = max_rounds
    engine.round_number = max_rounds
    # First call returns a real question; second call returns None (loop ends)
    engine.get_question.side_effect = ["What did you learn?", None]
    engine.should_continue.return_value = False  # loop exits after first question
    if pairs is None:
        pairs = []
    engine.collect_answers.return_value = pairs
    return engine


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_run_debrief_loop_backward_compat():
    """Calling with only (engine, console) — no new kwargs — must work identically."""
    from pb.vault.socratic import run_debrief_loop

    pairs = [("Q1", "A1"), ("Q2", "A2")]
    engine = _make_engine(max_rounds=5, pairs=pairs)
    console = MagicMock()

    result = run_debrief_loop(engine, console)

    assert isinstance(result, list)
    assert result == pairs


def test_hook_fires_on_deep_debrief():
    """When generate_anki=True and engine._max=5, extract_socratic_cards must be called."""
    from pb.vault.socratic import run_debrief_loop

    pairs = [("Q1", "A1"), ("Q2", "A2")]
    engine = _make_engine(max_rounds=5, pairs=pairs)
    console = MagicMock()

    fake_cards = [{"id": "x-socratic-0", "front": "Q1", "back": "A1"}]

    with patch("pb.vault.socratic.extract_socratic_cards", return_value=fake_cards) as mock_extract, \
         patch("pb.vault.anki_client.insert_cards_to_db", return_value=1) as mock_insert, \
         patch("pb.vault.anki_client._insert_run_log_entry") as mock_log:
        result = run_debrief_loop(
            engine,
            console,
            generate_anki=True,
            note_slug="my-note",
            deck="German",
            domain="deutsch",
        )

    mock_extract.assert_called_once_with(pairs, "my-note", "German", "deutsch")
    assert result == pairs


def test_hook_skipped_on_mini_debrief():
    """When engine._max < 5 (mini debrief), extract_socratic_cards must NOT be called."""
    from pb.vault.socratic import run_debrief_loop

    pairs = [("Q1", "A1")]
    engine = _make_engine(max_rounds=2, pairs=pairs)
    console = MagicMock()

    with patch("pb.vault.socratic.extract_socratic_cards") as mock_extract:
        result = run_debrief_loop(
            engine,
            console,
            generate_anki=True,
            note_slug="my-note",
            deck="German",
            domain="deutsch",
        )

    mock_extract.assert_not_called()
    assert result == pairs


def test_hook_failure_non_fatal():
    """If extract_socratic_cards raises, run_debrief_loop must still return pairs."""
    from pb.vault.socratic import run_debrief_loop

    pairs = [("Q1", "A1"), ("Q2", "A2")]
    engine = _make_engine(max_rounds=5, pairs=pairs)
    console = MagicMock()

    with patch("pb.vault.socratic.extract_socratic_cards", side_effect=RuntimeError("DB gone")):
        # Must not raise
        result = run_debrief_loop(
            engine,
            console,
            generate_anki=True,
            note_slug="my-note",
            deck="German",
            domain="deutsch",
        )

    assert result == pairs
