"""Integration tests for the quick capture CLI surfaces."""

from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from pb.cli.main import app
from pb.storage.repository import Repository


runner = CliRunner()


def test_home_command_renders_dashboard(temp_db, temp_config):
    result = runner.invoke(app, ["home"])

    assert result.exit_code == 0
    assert "ProductiveBrain" in result.output
    assert "Suggested commands:" in result.output


def test_thought_command_writes_markdown_note(temp_db, temp_config):
    result = runner.invoke(app, ["thought", "Abort", "propagation", "still", "feels", "fuzzy"])

    assert result.exit_code == 0
    assert "Captured thought:" in result.output

    thought_dir = Path(temp_config.general.vault_path) / "Learning" / "Inbox" / "pb" / "thoughts"
    notes = list(thought_dir.glob("*.md"))
    assert len(notes) == 1
    content = notes[0].read_text(encoding="utf-8")
    assert "type: thought" in content
    assert "Abort propagation still feels fuzzy" in content


def test_todo_command_creates_next_action_task(temp_db, temp_config):
    result = runner.invoke(app, ["todo", "Email", "advisor", "/due", "2026-05-21"])

    assert result.exit_code == 0
    assert "Captured todo: Email advisor" in result.output
    assert "Due: 2026-05-21" in result.output

    tasks = Repository().list_tasks()
    assert len(tasks) == 1
    assert tasks[0].title == "Email advisor"
    assert tasks[0].work_type == "todo"
