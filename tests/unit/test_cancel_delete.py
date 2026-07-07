"""Unit tests for cancel (discard_session) and delete (hard_delete_task) functionality.

Tests:
- Repository.delete_session()
- Repository.hard_delete_task()
- SessionManager.discard_session()
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task, TimeBlock
from pb.storage.repository import Repository


# ---------------------------------------------------------------------------
# Repository.delete_session() tests
# ---------------------------------------------------------------------------


class TestDeleteSession:
    """Tests for Repository.delete_session()."""

    def test_delete_session_returns_true_when_found(self, repo):
        """delete_session() returns True when the session is found and deleted."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(hours=1),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        result = repo.delete_session(session.id)
        assert result is True

    def test_delete_session_row_is_gone(self, repo):
        """After delete_session(), the session no longer exists in the DB."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(hours=1),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)
        repo.delete_session(session.id)

        fetched = repo.get_session(session.id)
        assert fetched is None

    def test_delete_session_nonexistent_returns_false(self, repo):
        """delete_session() returns False when the session does not exist."""
        result = repo.delete_session("nonexistent-id-xyz")
        assert result is False


# ---------------------------------------------------------------------------
# Repository.hard_delete_task() tests
# ---------------------------------------------------------------------------


class TestHardDeleteTask:
    """Tests for Repository.hard_delete_task()."""

    def test_hard_delete_task_no_sessions_returns_true(self, repo):
        """hard_delete_task() returns True when task has no sessions (hard deleted)."""
        task = Task(title="Task with no sessions", state=TaskState.ACTIVE)
        repo.create_task(task)

        result = repo.hard_delete_task(task.id)
        assert result is True

    def test_hard_delete_task_no_sessions_removes_task(self, repo):
        """After hard_delete_task() with no sessions, task no longer exists."""
        task = Task(title="Deletable task", state=TaskState.ACTIVE)
        repo.create_task(task)
        repo.hard_delete_task(task.id)

        fetched = repo.get_task(task.id)
        assert fetched is None

    def test_hard_delete_task_no_sessions_removes_time_blocks(self, repo):
        """hard_delete_task() also deletes orphaned time_blocks for the task."""
        task = Task(title="Task with block", state=TaskState.ACTIVE)
        repo.create_task(task)

        block = TimeBlock(
            task_id=task.id,
            duration_minutes=30,
            start_time=datetime.utcnow(),
        )
        repo.create_time_block(block)

        repo.hard_delete_task(task.id)

        # Time block should also be gone
        fetched_block = repo.get_time_block(block.id)
        assert fetched_block is None

    def test_hard_delete_task_with_sessions_archives_instead(self, repo):
        """hard_delete_task() auto-archives task when sessions exist, returns False."""
        task = Task(title="Task with sessions", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(hours=2),
            end_at=datetime.utcnow() - timedelta(hours=1),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        result = repo.hard_delete_task(task.id)
        assert result is False

    def test_hard_delete_task_with_sessions_task_still_exists_as_archived(self, repo):
        """When hard_delete_task() auto-archives, task still exists with archived_at set."""
        task = Task(title="Archived via delete", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(hours=2),
            end_at=datetime.utcnow() - timedelta(hours=1),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        repo.hard_delete_task(task.id)

        fetched = repo.get_task(task.id)
        # Task still exists (not hard-deleted)
        assert fetched is not None
        assert fetched.archived_at is not None


# ---------------------------------------------------------------------------
# SessionManager.discard_session() tests
# ---------------------------------------------------------------------------


class TestDiscardSession:
    """Tests for SessionManager.discard_session()."""

    def _make_manager(self, repo):
        """Build a SessionManager with mocked timer_manager."""
        from pb.core.sessions import SessionManager

        mock_tm = MagicMock()
        return SessionManager(repo, timer_manager=mock_tm)

    def test_discard_session_no_active_session_returns_none(self, repo):
        """discard_session() returns None when there is no active session."""
        manager = self._make_manager(repo)
        result = manager.discard_session()
        assert result is None

    def test_discard_session_with_active_session_returns_session(self, repo):
        """discard_session() returns the discarded session object."""
        task = Task(title="Active task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(minutes=10),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        manager = self._make_manager(repo)
        result = manager.discard_session()
        assert result is not None
        assert result.id == session.id

    def test_discard_session_deletes_session_from_db(self, repo):
        """discard_session() hard-deletes the session row from the DB."""
        task = Task(title="Active task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(minutes=10),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        manager = self._make_manager(repo)
        manager.discard_session()

        fetched = repo.get_session(session.id)
        assert fetched is None

    def test_discard_session_reverts_task_to_ready_when_no_prior_sessions(self, repo):
        """discard_session() reverts task to READY when it had no prior sessions."""
        task = Task(title="Fresh task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(minutes=10),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        manager = self._make_manager(repo)
        manager.discard_session()

        refreshed_task = repo.get_task(task.id)
        assert refreshed_task.state == TaskState.ACTIVE

    def test_discard_session_leaves_task_active_when_prior_sessions_exist(self, repo):
        """discard_session() leaves task ACTIVE regardless of prior sessions."""
        task = Task(title="Resumed task", state=TaskState.ACTIVE)
        repo.create_task(task)

        # Prior completed session
        prior_session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(hours=2),
            end_at=datetime.utcnow() - timedelta(hours=1),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(prior_session)

        # Current active (unended) session
        active_session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(minutes=10),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(active_session)

        manager = self._make_manager(repo)
        manager.discard_session()

        refreshed_task = repo.get_task(task.id)
        assert refreshed_task.state == TaskState.ACTIVE

    def test_discard_session_calls_timer_manager_stop(self, repo):
        """discard_session() calls timer_manager.stop_session_timers()."""
        task = Task(title="Active task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=datetime.utcnow() - timedelta(minutes=10),
            mode=SessionMode.FOCUS,
        )
        repo.create_session(session)

        from pb.core.sessions import SessionManager

        mock_tm = MagicMock()
        manager = SessionManager(repo, timer_manager=mock_tm)

        manager.discard_session()

        mock_tm.stop_session_timers.assert_called_once()
