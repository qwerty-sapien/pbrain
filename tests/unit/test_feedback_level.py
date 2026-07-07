from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import typer

from pb.cli.commands.feedback import feedback_command, feedback_level


def test_feedback_level_writes_vault_assertion_and_active_session_signal(tmp_path):
    session = SimpleNamespace(generated_names={})
    repo = MagicMock()
    repo.get_active_session.return_value = session
    runtime = SimpleNamespace(vault_path=tmp_path)
    ctx = SimpleNamespace(obj={"runtime": runtime, "repo": repo})

    with patch("pb.cli.commands.feedback.get_console") as console_factory:
        console_factory.return_value = MagicMock()
        feedback_level(
            ctx,
            topic_words=["analytic", "continuation"],
            level=3,
            confidence=2,
            evidence="I completed one derivation with one hint.",
            note="Still shaky around the pole.",
        )

    path = tmp_path / "direction" / "preferences" / "learner-levels.md"
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    assert "analytic continuation" in content
    assert "level=3" in content
    assert session.generated_names["learner_self_reports"][0]["topic"] == "analytic continuation"
    repo.update_session.assert_called_once_with(session)


def test_feedback_wrong_uses_wrong_agent_path_not_generic_proposal(tmp_path):
    ctx = SimpleNamespace(
        invoked_subcommand=None,
        obj={"runtime": SimpleNamespace(vault_path=tmp_path), "repo": MagicMock()},
    )

    with patch("pb.cli.commands.feedback.list_active_sessions", return_value=[]), \
         patch("pb.cli.commands.feedback._capture_feedback_proposal", side_effect=AssertionError("must not save generic feedback")), \
         patch("pb.cli.commands.feedback.get_err_console") as err_console_factory:
        err_console = MagicMock()
        err_console_factory.return_value = err_console
        with pytest.raises(typer.Exit) as exc_info:
            feedback_command(ctx, surface="wrong", note_words=["route", "this", "to", "practise"])

    assert exc_info.value.exit_code == 40
    message = err_console.print.call_args.args[0]
    assert "pb feedback general" in message
    assert "pb do <intent>" in message
