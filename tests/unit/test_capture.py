"""Unit tests for capture.py create_task (Phase 9).

Verifies that capture.py operates without Taskwarrior adapter:
- Tasks are always created in the SQLite repository
- No TaskwarriorAdapter or get_config references in create_task
- Horizon validation works correctly
- Tasks default to ACTIVE state
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pb.domain.enums import TaskState


class TestCreateTask:
    """create_task creates tasks in SQLite only — no adapter calls."""

    @patch("pb.cli.commands.capture.Repository")
    def test_create_task_creates_in_repo(self, MockRepo):
        """create_task always calls repo.create_task."""
        from pb.cli.commands.capture import create_task

        mock_repo_instance = MockRepo.return_value

        create_task("Write documentation", horizon="today")

        mock_repo_instance.create_task.assert_called_once()

    @patch("pb.cli.commands.capture.Repository")
    def test_create_task_active_state_by_default(self, MockRepo):
        """create_task creates task in ACTIVE state."""
        from pb.cli.commands.capture import create_task

        mock_repo_instance = MockRepo.return_value
        captured_tasks = []
        mock_repo_instance.create_task.side_effect = lambda t: captured_tasks.append(t)

        create_task("Review pull requests", horizon="today")

        assert len(captured_tasks) == 1
        assert captured_tasks[0].state == TaskState.ACTIVE

    @patch("pb.cli.commands.capture.Repository")
    def test_create_task_sets_title(self, MockRepo):
        """create_task preserves the task title."""
        from pb.cli.commands.capture import create_task

        mock_repo_instance = MockRepo.return_value
        captured_tasks = []
        mock_repo_instance.create_task.side_effect = lambda t: captured_tasks.append(t)

        create_task("My important task", horizon="today")

        assert captured_tasks[0].title == "My important task"

    @patch("pb.cli.commands.capture.Repository")
    def test_create_task_invalid_horizon_exits(self, MockRepo):
        """create_task raises typer.Exit(code=1) for invalid horizon."""
        import click
        from pb.cli.commands.capture import create_task

        with pytest.raises(click.exceptions.Exit) as exc_info:
            create_task("Task text", horizon="invalid")

        assert exc_info.value.exit_code == 1

    def test_create_task_has_no_taskwarrior_import(self):
        """capture module does not import TaskwarriorAdapter."""
        import pb.cli.commands.capture as capture_mod
        import inspect

        source = inspect.getsource(capture_mod)
        assert "TaskwarriorAdapter" not in source

    def test_create_task_has_no_get_config_import(self):
        """capture module does not import get_config for adapter use."""
        import pb.cli.commands.capture as capture_mod
        import inspect

        source = inspect.getsource(capture_mod)
        assert "get_config" not in source
