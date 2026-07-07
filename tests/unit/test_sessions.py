"""Unit tests for SessionManager core lifecycle (Phase 9).

Verifies that SessionManager operates without adapter side-effects:
- SQLite is the sole backend
- No TaskwarriorAdapter or TimewarriorAdapter references
- start_session, pause_session, finish_session, discard_session behave correctly
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock

import pytest

from pb.core.sessions import SessionManager
from pb.core.timer import TimerManager
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task


@pytest.fixture
def mock_repo():
    """Mock repository with sane defaults."""
    repo = MagicMock()
    repo.get_active_session.return_value = None
    repo.list_time_blocks_for_date.return_value = []
    repo.list_sessions_for_task.return_value = []
    return repo


@pytest.fixture
def mock_timer():
    """Mock TimerManager."""
    return MagicMock(spec=TimerManager)


@pytest.fixture
def ready_task():
    """Task in READY state."""
    return Task(id="task-ready-00001", title="Write tests", state=TaskState.ACTIVE)


@pytest.fixture
def paused_task():
    """Task in PAUSED state."""
    return Task(id="task-pause-00001", title="Paused work", state=TaskState.PAUSED)


class TestSessionManagerInit:
    """SessionManager __init__ is adapter-free."""

    def test_init_accepts_repo_and_timer(self, mock_repo, mock_timer):
        """SessionManager accepts repo and timer_manager; no adapter params."""
        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        assert manager.repo is mock_repo
        assert manager.timer_manager is mock_timer

    def test_init_creates_default_timer(self, mock_repo):
        """SessionManager creates a TimerManager by default."""
        manager = SessionManager(repo=mock_repo)
        assert isinstance(manager.timer_manager, TimerManager)

    def test_init_has_no_adapter_attributes(self, mock_repo):
        """SessionManager has no taskwarrior or timewarrior attributes."""
        manager = SessionManager(repo=mock_repo)
        assert not hasattr(manager, "taskwarrior")
        assert not hasattr(manager, "timewarrior")
        assert not hasattr(manager, "_tw_uuids")


class TestStartSession:
    """start_session creates session and starts timers; no adapter calls."""

    def test_start_session_creates_session(self, mock_repo, mock_timer, ready_task):
        """start_session creates a session in the repository."""
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        result = manager.start_session(ready_task, SessionMode.FOCUS)

        mock_repo.create_session.assert_called_once()
        assert result is not None

    def test_start_session_starts_timers(self, mock_repo, mock_timer, ready_task):
        """start_session calls timer_manager.start_session_timers()."""
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.start_session(ready_task, SessionMode.FOCUS)

        mock_timer.start_session_timers.assert_called_once()

    def test_start_session_raises_when_session_active(self, mock_repo, mock_timer, ready_task):
        """start_session raises RuleViolation when another session is already active."""
        from pb.domain.rules import RuleViolation

        active = Session(task_id="other-task-00001")
        mock_repo.get_active_session.return_value = active

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        with pytest.raises(RuleViolation):
            manager.start_session(ready_task, SessionMode.FOCUS)

    def test_start_session_raises_for_completed_task(self, mock_repo, mock_timer):
        """start_session raises RuleViolation for tasks with completion >= 100."""
        from pb.domain.rules import RuleViolation

        done_task = Task(id="task-done-00001", title="Done", state=TaskState.DONE, completion=100)
        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        with pytest.raises(RuleViolation):
            manager.start_session(done_task, SessionMode.FOCUS)

    def test_start_session_raises_for_paused_task(self, mock_repo, mock_timer):
        """start_session raises RuleViolation for paused tasks."""
        from pb.domain.rules import RuleViolation

        paused = Task(id="task-paused-00001", title="Paused", state=TaskState.PAUSED)
        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        with pytest.raises(RuleViolation):
            manager.start_session(paused, SessionMode.FOCUS)


class TestPauseSession:
    """pause_session stops timers; no timewarrior.stop() call."""

    def test_pause_session_returns_none_when_no_active(self, mock_repo, mock_timer):
        """pause_session returns None when no active session."""
        mock_repo.get_active_session.return_value = None

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        result = manager.pause_session()

        assert result is None
        mock_timer.stop_session_timers.assert_not_called()

    def test_pause_session_does_not_change_task_state(self, mock_repo, mock_timer, ready_task):
        """pause_session does not modify task state (task pause is separate)."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.pause_session("Got interrupted")

        mock_repo.update_task.assert_not_called()

    def test_pause_session_stops_timers(self, mock_repo, mock_timer, ready_task):
        """pause_session stops timers."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.pause_session()

        mock_timer.stop_session_timers.assert_called_once()


class TestFinishSession:
    """finish_session marks task done and stops timers; no adapter calls."""

    def test_finish_session_returns_none_when_no_active(self, mock_repo, mock_timer):
        """finish_session returns None when no active session."""
        mock_repo.get_active_session.return_value = None

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        result = manager.finish_session("done")

        assert result is None

    def test_finish_session_done_marks_task_done(self, mock_repo, mock_timer, ready_task):
        """finish_session('done') transitions task to DONE state."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.finish_session("done")

        calls = [c[0][0] for c in mock_repo.update_task.call_args_list]
        assert any(t.state == TaskState.DONE for t in calls)

    def test_finish_session_abandoned_leaves_task_active(self, mock_repo, mock_timer, ready_task):
        """finish_session('abandoned') does not change task state or completion."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.finish_session("abandoned")

        calls = [c[0][0] for c in mock_repo.update_task.call_args_list]
        assert all(t.state == TaskState.ACTIVE for t in calls)

    def test_finish_session_stops_timers(self, mock_repo, mock_timer, ready_task):
        """finish_session stops timers."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.finish_session("done")

        mock_timer.stop_session_timers.assert_called_once()


class TestDiscardSession:
    """discard_session reverts task state; no timewarrior.stop() call."""

    def test_discard_session_returns_none_when_no_active(self, mock_repo, mock_timer):
        """discard_session returns None when no active session."""
        mock_repo.get_active_session.return_value = None

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        result = manager.discard_session()

        assert result is None

    def test_discard_session_deletes_session(self, mock_repo, mock_timer, ready_task):
        """discard_session hard-deletes the session record."""
        active_session = Session(task_id=ready_task.id, id="sess-discard-00001")
        mock_repo.get_active_session.return_value = active_session

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.discard_session()

        mock_repo.delete_session.assert_called_once_with(active_session.id)

    def test_discard_session_stops_timers(self, mock_repo, mock_timer, ready_task):
        """discard_session stops timers."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer)
        manager.discard_session()

        mock_timer.stop_session_timers.assert_called_once()
