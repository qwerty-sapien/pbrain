"""Tests for TUI picker functions (D-13 to D-17)."""

from unittest.mock import patch

import pytest

from pb.domain.models import Task
from pb.domain.enums import TaskState


@pytest.fixture
def sample_tasks():
    return [
        Task(title="Fix login bug", state=TaskState.ACTIVE),
        Task(title="Write tests", state=TaskState.PAUSED),
        Task(title="Deploy app", state=TaskState.ACTIVE),
    ]


class TestPickTaskDialog:
    def test_returns_none_for_empty_list(self):
        from pb.cli.pickers import pick_task_dialog
        assert pick_task_dialog([]) is None

    def test_returns_none_in_non_tty(self, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_task_dialog
        monkeypatch.setattr("sys.stdin.isatty", lambda: False)
        assert pick_task_dialog(sample_tasks) is None

    def test_returns_selected_task(self, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_task_dialog
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("pb.cli.helpers._interactive_pick", lambda *a, **k: [1])
        result = pick_task_dialog(sample_tasks)
        assert result == sample_tasks[1]

    def test_returns_none_on_cancel(self, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_task_dialog
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("pb.cli.helpers._interactive_pick", lambda *a, **k: None)
        result = pick_task_dialog(sample_tasks)
        assert result is None

    def test_returns_sentinel_when_picker_requests_manual_input(self, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_task_dialog, _MANUAL_INPUT_SENTINEL
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        monkeypatch.setattr("pb.cli.helpers._interactive_pick", lambda *a, **k: _MANUAL_INPUT_SENTINEL)
        result = pick_task_dialog(sample_tasks)
        assert result == _MANUAL_INPUT_SENTINEL


class TestPickOrPrompt:
    @patch("pb.cli.pickers.pick_task_dialog")
    def test_returns_picker_result_when_selected(self, mock_pick, sample_tasks):
        from pb.cli.pickers import pick_or_prompt
        mock_pick.return_value = sample_tasks[0]
        result = pick_or_prompt(sample_tasks, find_fn=lambda x: None)
        assert result == sample_tasks[0]

    @patch("pb.cli.pickers.pick_task_dialog")
    def test_falls_back_to_numbered_on_zero_press(self, mock_pick, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_or_prompt, _MANUAL_INPUT_SENTINEL
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        mock_pick.return_value = _MANUAL_INPUT_SENTINEL
        monkeypatch.setattr("builtins.input", lambda _: "1")
        result = pick_or_prompt(sample_tasks, find_fn=lambda x: None)
        assert result == sample_tasks[0]

    @patch("pb.cli.pickers.pick_task_dialog")
    def test_falls_back_to_numbered_on_none(self, mock_pick, sample_tasks, monkeypatch):
        from pb.cli.pickers import pick_or_prompt
        monkeypatch.setattr("sys.stdin.isatty", lambda: True)
        mock_pick.return_value = None
        monkeypatch.setattr("builtins.input", lambda _: "2")
        result = pick_or_prompt(sample_tasks, find_fn=lambda x: None)
        assert result == sample_tasks[1]


class TestTaskLabel:
    def test_label_format(self, sample_tasks):
        from pb.cli.pickers import _task_label
        label = _task_label(sample_tasks[0])
        assert "Fix login bug" in label
        assert "%" not in label  # 0% completion omits suffix

    def test_label_shows_completion_percent(self, sample_tasks):
        from pb.cli.pickers import _task_label
        task = sample_tasks[0]
        task.completion = 42
        label = _task_label(task)
        assert "42%" in label

    def test_label_shows_working_when_active_task(self, sample_tasks):
        from pb.cli.pickers import _task_label
        task = sample_tasks[0]
        label = _task_label(task, active_task_id=task.id)
        assert "[working]" in label

    def test_label_shows_percent_when_not_active_task(self, sample_tasks):
        from pb.cli.pickers import _task_label
        task = sample_tasks[0]
        task.completion = 50
        label = _task_label(task, active_task_id="some-other-id")
        assert "[working]" not in label
        assert "50%" in label
