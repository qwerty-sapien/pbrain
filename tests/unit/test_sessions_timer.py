"""Unit tests for SessionManager timer integration.

Tests that SessionManager properly starts and stops timers during session lifecycle.
"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest

from pb.core.sessions import SessionManager
from pb.core.timer import TimerManager
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task, TimeBlock


@pytest.fixture
def mock_repo():
    """Create a mock repository."""
    repo = MagicMock()
    repo.get_active_session.return_value = None
    repo.list_time_blocks_for_date.return_value = []
    return repo


@pytest.fixture
def mock_timer_manager():
    """Create a mock timer manager."""
    return MagicMock(spec=TimerManager)


@pytest.fixture
def ready_task():
    """Create a task in ready state."""
    return Task(
        id="task-123",
        title="Test Task",
        state=TaskState.ACTIVE,
    )


class TestSessionManagerTimerIntegration:
    """Tests for SessionManager timer integration."""

    def test_start_session_starts_timers(self, mock_repo, mock_timer_manager, ready_task):
        """start_session() calls timer_manager.start_session_timers()."""
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        result = manager.start_session(ready_task, SessionMode.FOCUS)

        mock_timer_manager.start_session_timers.assert_called_once()
        call_kwargs = mock_timer_manager.start_session_timers.call_args[1]
        assert call_kwargs["session_id"]  # Session ID is generated, just verify it exists
        assert call_kwargs["task_title"] == "Test Task"

    def test_start_session_with_time_block_duration(self, mock_repo, mock_timer_manager, ready_task):
        """start_session() passes TimeBlock duration to timer manager."""
        block = TimeBlock(task_id=ready_task.id, duration_minutes=45)
        mock_repo.list_time_blocks_for_date.return_value = [block]
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        manager.start_session(ready_task, SessionMode.FOCUS)

        call_kwargs = mock_timer_manager.start_session_timers.call_args[1]
        assert call_kwargs["duration_minutes"] == 45

    def test_start_session_without_time_block(self, mock_repo, mock_timer_manager, ready_task):
        """start_session() passes None duration when no TimeBlock exists."""
        mock_repo.list_time_blocks_for_date.return_value = []
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        manager.start_session(ready_task, SessionMode.FOCUS)

        call_kwargs = mock_timer_manager.start_session_timers.call_args[1]
        assert call_kwargs["duration_minutes"] is None

    def test_start_session_with_mismatched_block(self, mock_repo, mock_timer_manager, ready_task):
        """start_session() ignores TimeBlock for different task."""
        other_block = TimeBlock(task_id="other-task", duration_minutes=60)
        mock_repo.list_time_blocks_for_date.return_value = [other_block]
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        manager.start_session(ready_task, SessionMode.FOCUS)

        call_kwargs = mock_timer_manager.start_session_timers.call_args[1]
        assert call_kwargs["duration_minutes"] is None

    def test_pause_session_stops_timers(self, mock_repo, mock_timer_manager, ready_task):
        """pause_session() calls timer_manager.stop_session_timers()."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task
        mock_repo.update_session.return_value = active_session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        manager.pause_session()

        mock_timer_manager.stop_session_timers.assert_called_once()

    def test_pause_session_no_active_session(self, mock_repo, mock_timer_manager):
        """pause_session() does not call stop_session_timers when no session."""
        mock_repo.get_active_session.return_value = None

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        result = manager.pause_session()

        assert result is None
        mock_timer_manager.stop_session_timers.assert_not_called()

    def test_finish_session_stops_timers(self, mock_repo, mock_timer_manager, ready_task):
        """finish_session() calls timer_manager.stop_session_timers()."""
        active_session = Session(task_id=ready_task.id)
        mock_repo.get_active_session.return_value = active_session
        mock_repo.get_task.return_value = ready_task
        mock_repo.update_session.return_value = active_session
        mock_repo.update_task.return_value = ready_task

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        manager.finish_session("done")

        mock_timer_manager.stop_session_timers.assert_called_once()

    def test_finish_session_no_active_session(self, mock_repo, mock_timer_manager):
        """finish_session() does not call stop_session_timers when no session."""
        mock_repo.get_active_session.return_value = None

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        result = manager.finish_session("done")

        assert result is None
        mock_timer_manager.stop_session_timers.assert_not_called()

    def test_get_elapsed_minutes_delegates_to_timer(self, mock_repo, mock_timer_manager):
        """get_elapsed_minutes() delegates to timer_manager."""
        mock_timer_manager.get_elapsed_minutes.return_value = 15

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        elapsed = manager.get_elapsed_minutes()

        assert elapsed == 15
        mock_timer_manager.get_elapsed_minutes.assert_called_once()

    def test_get_remaining_minutes_delegates_to_timer(self, mock_repo, mock_timer_manager):
        """get_remaining_minutes() delegates to timer_manager."""
        mock_timer_manager.get_remaining_minutes.return_value = 25

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        remaining = manager.get_remaining_minutes()

        assert remaining == 25
        mock_timer_manager.get_remaining_minutes.assert_called_once()

    def test_default_timer_manager_created(self, mock_repo):
        """SessionManager creates default TimerManager if not provided."""
        manager = SessionManager(repo=mock_repo)

        assert manager.timer_manager is not None
        assert isinstance(manager.timer_manager, TimerManager)

    def test_get_elapsed_minutes_db_fallback(self, mock_repo, mock_timer_manager):
        """get_elapsed_minutes() falls back to DB when timer returns None."""
        from datetime import timedelta
        mock_timer_manager.get_elapsed_minutes.return_value = None
        session = Session(task_id="task-123")
        session.start_at = datetime.utcnow() - timedelta(minutes=15)
        mock_repo.get_active_session.return_value = session

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        elapsed = manager.get_elapsed_minutes()

        assert elapsed is not None
        assert 14 <= elapsed <= 16

    def test_get_remaining_minutes_db_fallback(self, mock_repo, mock_timer_manager):
        """get_remaining_minutes() falls back to DB when timer returns None."""
        from datetime import timedelta
        mock_timer_manager.get_elapsed_minutes.return_value = None
        mock_timer_manager.get_remaining_minutes.return_value = None
        session = Session(task_id="task-123")
        session.start_at = datetime.utcnow() - timedelta(minutes=10)
        mock_repo.get_active_session.return_value = session
        block = TimeBlock(task_id="task-123", duration_minutes=30)
        mock_repo.list_time_blocks_for_date.return_value = [block]

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        remaining = manager.get_remaining_minutes()

        assert remaining is not None
        assert 19 <= remaining <= 21

    def test_get_remaining_minutes_db_fallback_no_block(self, mock_repo, mock_timer_manager):
        """get_remaining_minutes() returns None when no time block in DB."""
        mock_timer_manager.get_elapsed_minutes.return_value = None
        mock_timer_manager.get_remaining_minutes.return_value = None
        session = Session(task_id="task-123")
        mock_repo.get_active_session.return_value = session
        mock_repo.list_time_blocks_for_date.return_value = []

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
        remaining = manager.get_remaining_minutes()

        assert remaining is None


class TestSessionManagerTimerLifecycle:
    """Tests for timer lifecycle across session states."""

    def test_start_pause_start_cycle(self, mock_repo, mock_timer_manager, ready_task):
        """Timer starts and stops correctly across pause/resume cycle."""
        session = Session(task_id=ready_task.id)
        mock_repo.create_session.return_value = session
        mock_repo.get_active_session.side_effect = [None, session, None]
        mock_repo.update_task.return_value = ready_task
        mock_repo.update_session.return_value = session
        mock_repo.get_task.return_value = ready_task
        mock_repo.list_time_blocks_for_date.return_value = []

        manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)

        manager.start_session(ready_task, SessionMode.FOCUS)
        assert mock_timer_manager.start_session_timers.call_count == 1

        manager.pause_session()
        assert mock_timer_manager.stop_session_timers.call_count == 1

    def test_finish_different_outcomes(self, mock_repo, mock_timer_manager, ready_task):
        """Timer stops regardless of finish outcome."""
        outcomes = ["done", "partial", "blocked", "abandoned"]

        for outcome in outcomes:
            mock_timer_manager.reset_mock()
            active_session = Session(task_id=ready_task.id)
            mock_repo.get_active_session.return_value = active_session
            mock_repo.get_task.return_value = ready_task
            mock_repo.update_session.return_value = active_session
            mock_repo.update_task.return_value = ready_task

            manager = SessionManager(repo=mock_repo, timer_manager=mock_timer_manager)
            manager.finish_session(outcome)

            mock_timer_manager.stop_session_timers.assert_called_once(), f"Failed for outcome: {outcome}"
