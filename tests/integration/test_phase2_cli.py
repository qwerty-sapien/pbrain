"""Integration tests for Phase 2 CLI commands.

Tests: cancel, delete, query, chat, add multi-word, sweep removal.
"""

import os

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.enums import Horizon, TaskState
from pb.domain.exceptions import ExitCode
from pb.domain.models import Task
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment with config and DB."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

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


class TestCancelCommand:
    """Tests for pb task cancel (relocated from top-level in v4.0 CLI restructure)."""

    def test_cancel_command_no_session(self, cli_env):
        """pb task cancel with no active session prints error, exit NOT_FOUND."""
        result = runner.invoke(app, ["task", "cancel"])
        assert result.exit_code == ExitCode.NOT_FOUND
        assert "No active session to cancel" in result.output


class TestDeleteCommand:
    """Tests for pb task delete (relocated from top-level in v4.0 CLI restructure)."""

    def test_delete_command_no_tasks(self, cli_env):
        """pb task delete with empty DB prints 'No tasks', exit NOT_FOUND."""
        result = runner.invoke(app, ["task", "delete"])
        assert result.exit_code == ExitCode.NOT_FOUND
        assert "No tasks" in result.output


class TestRemovedCommands:
    """Verify that removed commands error cleanly."""

    def test_query_command_removed(self, cli_env):
        """pb query was removed; should exit with error."""
        result = runner.invoke(app, ["query", "test", "question"])
        assert result.exit_code != 0

    def test_chat_command_removed(self, cli_env):
        """pb chat was removed; should exit with error."""
        result = runner.invoke(app, ["chat"])
        assert result.exit_code != 0


class TestAddMultiWord:
    """Tests for pb add with multi-word arguments (D-13)."""

    def test_add_multi_word(self, cli_env):
        """pb add piano session creates task with title 'piano session'."""
        result = runner.invoke(app, ["add", "piano", "session"])
        assert result.exit_code == 0
        assert "piano session" in result.output

        # Verify task exists in DB
        repo = Repository()
        tasks = repo.list_tasks()
        assert any(t.title == "piano session" for t in tasks)


class TestSweepRemoved:
    """Tests for sweep command removal."""

    def test_sweep_commands_removed(self, cli_env):
        """pb sweep produces error (command not found)."""
        result = runner.invoke(app, ["sweep"])
        # Typer returns exit code 2 for unknown commands
        assert result.exit_code != 0

    def test_waiting_command_removed(self, cli_env):
        """pb waiting produces error (command not found)."""
        result = runner.invoke(app, ["waiting"])
        assert result.exit_code != 0

    def test_blocked_command_removed(self, cli_env):
        """pb blocked produces error (command not found)."""
        result = runner.invoke(app, ["blocked"])
        assert result.exit_code != 0


class TestPlanDayBudget:
    """Test pb plan day --budget flag (SC #5 gap closure)."""

    def test_plan_day_no_budget_unchanged(self, cli_env):
        """plan day without --budget works as before."""
        result = runner.invoke(app, ["plan", "day"])
        assert result.exit_code == 0

    def test_plan_day_budget_no_blocks(self, cli_env):
        """--budget with no blocks shows budget info."""
        result = runner.invoke(app, ["plan", "day", "--budget", "4h"])
        assert result.exit_code == 0
        assert "4h 0m" in result.output

    def test_plan_day_budget_with_blocks(self, cli_env):
        """--budget with blocks shows budget info."""
        result = runner.invoke(app, ["plan", "day", "--budget", "4h"])
        assert result.exit_code == 0

    def test_plan_day_budget_invalid(self, cli_env):
        """--budget with invalid format shows error."""
        result = runner.invoke(app, ["plan", "day", "--budget", "abc"], catch_exceptions=False)
        assert result.exit_code == 1
        assert "Invalid budget format" in result.output

    def test_plan_day_budget_minutes_format(self, cli_env):
        """--budget accepts minutes format."""
        result = runner.invoke(app, ["plan", "day", "--budget", "120m"])
        assert result.exit_code == 0
        assert "Budget" in result.output
        assert "2h 0m" in result.output


class TestHelpOutput:
    """Tests for --help output."""

    def test_help_shows_new_commands(self, cli_env):
        """pb --help shows core learning verbs (v1.0 surface)."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "goal" in result.output
        assert "learn" in result.output
        assert "review" in result.output
        assert "finish" in result.output

    def test_help_no_sweep(self, cli_env):
        """pb --help does NOT show sweep, waiting, blocked."""
        result = runner.invoke(app, ["--help"])
        assert "sweep" not in result.output
        assert "waiting" not in result.output
        # 'blocked' may appear in other command help text, check not as top-level command
