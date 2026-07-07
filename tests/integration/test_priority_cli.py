"""Integration tests for priority CLI commands (Plan 02-06).

Tests: pb task score, pb task list (with --inbox, --today, --state filters).
"""

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.enums import Horizon, TaskState
from pb.domain.models import Task
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture
def cli_env(temp_db, temp_config):
    """Set up CLI test environment using shared temp fixtures."""
    yield


class TestTaskScore:
    """Tests for pb task score."""

    def test_score_command_no_tasks(self, cli_env):
        """pb task score with no tasks prints error, exit code 1."""
        result = runner.invoke(app, ["task", "score"])
        assert result.exit_code == 1
        assert "No tasks to score" in result.output

    def test_list_ranked_no_tasks(self, cli_env):
        """pb task list with no tasks prints 'No tasks found', exit code 0."""
        result = runner.invoke(app, ["task", "list"])
        assert result.exit_code == 0
        assert "No tasks found" in result.output

    def test_list_ranked_with_scored_task(self, cli_env):
        """pb task list shows scored task with score and eisenhower class."""
        repo = Repository()
        task = Task(
            title="High priority task",
            impact=5,
            urgency_score=4,
            strategic_value=3,
            effort=2,
            important=True,
            urgent=True,
            energy_required=4,
            work_type="deep",
        )
        repo.create_task(task)

        result = runner.invoke(app, ["task", "list"])
        assert result.exit_code == 0
        assert "6.0" in result.output
        assert "do_today" in result.output
        assert "schedule_first" in result.output

    def test_list_ranked_shows_unscored_tasks(self, cli_env):
        """pb task list shows unscored tasks with dash placeholders."""
        repo = Repository()
        task = Task(title="Unscored task")
        repo.create_task(task)

        result = runner.invoke(app, ["task", "list"])
        assert result.exit_code == 0
        assert "Unscored task" in result.output
        assert "unscored" in result.output

    def test_list_ranked_header_columns(self, cli_env):
        """pb task list output has SCORE, EISENHOWER, ACTION, ENERGY columns."""
        repo = Repository()
        task = Task(title="Some task")
        repo.create_task(task)

        result = runner.invoke(app, ["task", "list"])
        assert result.exit_code == 0
        assert "SCORE" in result.output
        assert "EISENHOWER" in result.output
        assert "ACTION" in result.output
        assert "ENERGY" in result.output


class TestTaskListFilters:
    """Tests for task list filters."""

    def test_list_today_only(self, cli_env):
        """--today shows only today horizon tasks, not week horizon tasks."""
        repo = Repository()
        today_task = Task(title="Today task", horizon=Horizon.TODAY, state=TaskState.ACTIVE)
        week_task = Task(title="Week task", horizon=Horizon.WEEK, state=TaskState.ACTIVE)
        repo.create_task(today_task)
        repo.create_task(week_task)

        result = runner.invoke(app, ["task", "list", "--today"])
        assert result.exit_code == 0
        assert "Today task" in result.output
        assert "Week task" not in result.output
