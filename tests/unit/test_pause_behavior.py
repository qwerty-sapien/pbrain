"""Unit tests for pause interval tracking and stale auto-cancel (D-12, D-13, D-14).

Tests:
- pause_intervals table CRUD via repository
- Stale pause detection (sessions paused > 3 hours)
- Auto-cancel behavior for stale paused sessions
"""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from pb.domain.enums import Horizon, SessionMode, TaskState
from pb.domain.models import Session, Task
from pb.storage.repository import Repository


class TestPauseIntervalCRUD:
    """Tests for pause interval repository methods (D-12)."""

    def test_create_pause_interval_inserts_row(self, repo):
        """create_pause_interval(session_id, pause_start) inserts row into pause_intervals table."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        pause_start = datetime.utcnow()
        interval_id = repo.create_pause_interval(session.id, pause_start)

        assert interval_id is not None
        assert len(interval_id) > 0

    def test_resume_pause_interval_sets_resume_at(self, repo):
        """resume_pause_interval(session_id) sets resume_at on the open interval."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        pause_start = datetime.utcnow()
        repo.create_pause_interval(session.id, pause_start)
        repo.resume_pause_interval(session.id)

        intervals = repo.list_pause_intervals(session.id)
        assert len(intervals) == 1
        assert intervals[0]["resume_at"] is not None

    def test_list_pause_intervals_returns_all_for_session(self, repo):
        """list_pause_intervals(session_id) returns all intervals for a session."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        # Create two pause intervals
        t1 = datetime.utcnow() - timedelta(hours=2)
        t2 = datetime.utcnow() - timedelta(hours=1)
        repo.create_pause_interval(session.id, t1)
        repo.resume_pause_interval(session.id)
        repo.create_pause_interval(session.id, t2)

        intervals = repo.list_pause_intervals(session.id)
        assert len(intervals) == 2
        # Ordered by pause_start
        assert intervals[0]["pause_start"] < intervals[1]["pause_start"]

    def test_pause_intervals_table_columns(self, repo):
        """pause_intervals table has columns: id, session_id, pause_start, resume_at."""
        task = Task(title="Test task", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        pause_start = datetime.utcnow()
        repo.create_pause_interval(session.id, pause_start)

        intervals = repo.list_pause_intervals(session.id)
        assert len(intervals) == 1
        interval = intervals[0]
        assert "id" in interval
        assert "session_id" in interval
        assert "pause_start" in interval
        assert "resume_at" in interval


class TestStalePauseDetection:
    """Tests for stale pause detection (D-13)."""

    def test_get_stale_pauses_returns_old_paused_sessions(self, repo):
        """get_stale_pauses(max_hours=3) returns paused sessions older than 3 hours."""
        task = Task(title="Stale task", state=TaskState.PAUSED)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        # Create pause interval 4 hours ago (stale)
        old_pause = datetime.utcnow() - timedelta(hours=4)
        repo.create_pause_interval(session.id, old_pause)

        stale = repo.get_stale_pauses(max_hours=3)
        assert len(stale) == 1
        assert stale[0]["session_id"] == session.id
        assert stale[0]["task_id"] == task.id

    def test_get_stale_pauses_returns_empty_when_none_stale(self, repo):
        """get_stale_pauses returns empty when no sessions are stale."""
        task = Task(title="Fresh task", state=TaskState.PAUSED)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        # Create pause interval 1 hour ago (not stale)
        recent_pause = datetime.utcnow() - timedelta(hours=1)
        repo.create_pause_interval(session.id, recent_pause)

        stale = repo.get_stale_pauses(max_hours=3)
        assert len(stale) == 0

    def test_get_stale_pauses_ignores_resumed_intervals(self, repo):
        """Resumed (closed) intervals should not appear as stale."""
        task = Task(title="Resumed task", state=TaskState.PAUSED)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        # Create stale pause interval but resume it
        old_pause = datetime.utcnow() - timedelta(hours=4)
        repo.create_pause_interval(session.id, old_pause)
        repo.resume_pause_interval(session.id)

        stale = repo.get_stale_pauses(max_hours=3)
        assert len(stale) == 0

    def test_get_stale_pauses_returns_stale_regardless_of_task_state(self, repo):
        """Stale session pauses are returned regardless of task state."""
        task = Task(title="Active task", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id, mode=SessionMode.FOCUS)
        repo.create_session(session)

        old_pause = datetime.utcnow() - timedelta(hours=4)
        repo.create_pause_interval(session.id, old_pause)

        stale = repo.get_stale_pauses(max_hours=3)
        assert len(stale) == 1
