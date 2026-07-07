"""Integration tests for CLI timer features.

Tests pb start task picker, duration prompt, and pb now enhancements.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pb.cli.helpers import parse_duration
from pb.cli.main import app
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task, TimeBlock
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path


runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    config_path = tmp_path / "config.toml"
    config_path.write_text(f'''
[general]
vault_path = "{vault_path}"

[storage]
data_dir = "{tmp_path}"
''')

    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)

    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    config_module._config = config

    yield tmp_path

    config_module._config = None
    db_module._db_path = None


def create_ready_task(title: str) -> str:
    """Helper: add a task (created ACTIVE). Returns task ID."""
    result = runner.invoke(app, ["add", title])
    task_id = result.output.split()[1].rstrip("...")
    return task_id


class TestParseDuration:
    """Tests for parse_duration helper function."""

    def test_parse_plain_integer(self):
        """Plain integer is parsed as minutes."""
        assert parse_duration("30") == 30
        assert parse_duration("60") == 60
        assert parse_duration("5") == 5
        assert parse_duration("120") == 120

    def test_parse_minutes_with_suffix(self):
        """Minutes with suffix parsed correctly."""
        assert parse_duration("30m") == 30
        assert parse_duration("30 min") == 30
        assert parse_duration("30min") == 30
        assert parse_duration("30 minutes") == 30
        assert parse_duration("30minutes") == 30
        assert parse_duration("30mins") == 30

    def test_parse_hours(self):
        """Hours converted to minutes."""
        assert parse_duration("1h") == 60
        assert parse_duration("1.5h") == 90
        assert parse_duration("0.5h") == 30
        assert parse_duration("2 hr") == 120
        assert parse_duration("2hrs") == 120
        assert parse_duration("1 hour") == 60
        assert parse_duration("2hours") == 120

    def test_parse_invalid_returns_none(self):
        """Invalid formats return None."""
        assert parse_duration("invalid") is None
        assert parse_duration("abc") is None
        assert parse_duration("") is None
        assert parse_duration("30x") is None
        assert parse_duration("h") is None
        assert parse_duration("m") is None

    def test_parse_with_whitespace(self):
        """Whitespace is stripped."""
        assert parse_duration("  30  ") == 30
        assert parse_duration("  30m  ") == 30
        assert parse_duration("  1h  ") == 60

    def test_parse_case_insensitive(self):
        """Parsing is case insensitive."""
        assert parse_duration("30M") == 30
        assert parse_duration("1H") == 60
        assert parse_duration("30MIN") == 30
        assert parse_duration("1HR") == 60
        assert parse_duration("30MINUTES") == 30

    def test_parse_decimal_hours(self):
        """Decimal hours are handled correctly."""
        assert parse_duration("0.25h") == 15
        assert parse_duration("0.75h") == 45
        assert parse_duration("2.5h") == 150


class TestPtStartTaskPicker:
    """Tests for pb start with no args (task picker mode).

    In non-TTY contexts (CliRunner), prompt_toolkit picker is skipped and
    pick_or_prompt falls back to a numbered list.
    """

    def test_start_no_args_shows_task_picker(self, cli_env):
        """pb start with no args shows numbered fallback picker in non-TTY (D-01)."""
        create_ready_task("Task One")
        create_ready_task("Task Two")

        # Non-TTY: picker skipped, numbered list shown; pick task 1
        result = runner.invoke(app, ["start"], input="1\n\n")

        assert result.exit_code == 0, result.output
        assert "Started:" in result.output
        assert "Task One" in result.output

    def test_start_no_tasks_shows_error(self, cli_env):
        """pb start with no available tasks shows error."""
        result = runner.invoke(app, ["start"])

        assert (
            "No tasks to start" in result.output
            or "No tasks available" in result.output
            or "No items available" in result.output
        )

    def test_start_selects_task_from_picker(self, cli_env):
        """pb start allows selecting second task via numbered fallback in non-TTY."""
        create_ready_task("First Task")
        create_ready_task("Second Task")

        # Non-TTY: pick task 2 from numbered list
        result = runner.invoke(app, ["start"], input="2\n\n")

        assert "Started:" in result.output
        assert "Second Task" in result.output


class TestPtStartDuration:
    """Tests for pb start duration handling."""

    def test_start_with_duration_flag(self, cli_env):
        """pb start --duration accepts duration flag (Phase 23: duration shown in confirmation)."""
        task_id = create_ready_task("Test Task")

        result = runner.invoke(app, ["start", task_id, "--duration", "30m"])

        assert "Started:" in result.output
        # Phase 23: duration embedded in confirmation "Started: Test Task (30m)"
        assert "30m" in result.output

    def test_start_with_hours_duration(self, cli_env):
        """pb start --duration accepts hours format (Phase 23: duration shown in confirmation)."""
        task_id = create_ready_task("Test Task")

        result = runner.invoke(app, ["start", task_id, "--duration", "1.5h"])

        assert "Started:" in result.output
        # Phase 23: duration embedded in confirmation "Started: Test Task (90m)"
        assert "90m" in result.output

    def test_start_invalid_duration_shows_error(self, cli_env):
        """pb start with invalid duration shows error."""
        task_id = create_ready_task("Test Task")

        result = runner.invoke(app, ["start", task_id, "--duration", "invalid"])

        assert "Invalid duration" in result.output


class TestPtNowEnhanced:
    """Tests for pb now with elapsed/remaining time."""

    def test_now_no_active_task(self, cli_env):
        """pb now with no active task shows message."""
        result = runner.invoke(app, ["now"])

        # Phase 23: format_now_output returns "No active session."
        assert "No active" in result.output

    def test_now_shows_active_task(self, cli_env):
        """pb now shows currently active task."""
        task_id = create_ready_task("Active Task")
        runner.invoke(app, ["start", task_id, "--duration", "30m"])

        result = runner.invoke(app, ["now"])

        assert "Active: Active Task" in result.output

    def test_now_shows_task_with_duration(self, cli_env):
        """pb now shows active task when started with duration.

        Note: Timer state is in-memory only, so elapsed/remaining time
        is not available across separate CLI invocations. The timer
        functionality works within a single session (tested in unit tests).
        """
        task_id = create_ready_task("Test Task")
        runner.invoke(app, ["start", task_id, "--duration", "30m"])

        result = runner.invoke(app, ["now"])

        assert "Active: Test Task" in result.output


class TestPtPauseTimerIntegration:
    """Tests for pb pause timer behavior."""

    def test_pause_stops_session(self, cli_env):
        """pb pause stops the active session."""
        task_id = create_ready_task("Test Task")
        runner.invoke(app, ["start", task_id])

        result = runner.invoke(app, ["pause"])

        assert "Paused:" in result.output

    def test_pause_no_active_session(self, cli_env):
        """pb pause with no session shows error."""
        result = runner.invoke(app, ["pause"])

        assert "No active session" in result.output


# TestPtInterrupt removed in Phase 8 (D-13): pb interrupt command removed from CLI
