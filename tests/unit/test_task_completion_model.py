"""Tests for Phase 12.1 task completion model.

Covers:
- 3-state TaskState enum (active, paused, done)
- completion score field (0-100)
- task postponement via paused_until
- auto-activate when paused_until has passed
- session pause does NOT change task state
- stale session close does NOT change task state
- finish_session sets completion=100 + state=done for 'done' outcome
- finish_session sets completion from pct for 'partial' outcome
- finish_session leaves task active for 'blocked' / 'abandoned' outcomes
- pb start rejects tasks with completion >= 100
- pb start rejects paused tasks
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock

import pytest

from pb.core.sessions import SessionManager
from pb.core.timer import TimerManager
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task
from pb.domain.rules import RuleViolation
from pb.storage.repository import Repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_timer():
    return MagicMock(spec=TimerManager)


@pytest.fixture
def active_task():
    return Task(id="task-active-00001", title="Active task", state=TaskState.ACTIVE, completion=0)


@pytest.fixture
def partial_task():
    """Task already 40% complete."""
    return Task(id="task-partial-00001", title="Partial task", state=TaskState.ACTIVE, completion=40)


# ---------------------------------------------------------------------------
# TaskState enum has exactly 3 values
# ---------------------------------------------------------------------------


class TestTaskStateEnum:
    def test_exactly_three_states(self):
        assert len(TaskState) == 3

    def test_active_value(self):
        assert TaskState.ACTIVE.value == "active"

    def test_paused_value(self):
        assert TaskState.PAUSED.value == "paused"

    def test_done_value(self):
        assert TaskState.DONE.value == "done"

    def test_no_old_states(self):
        state_values = {s.value for s in TaskState}
        old_states = {"inbox", "ready", "waiting", "blocked", "cancelled"}
        assert state_values.isdisjoint(old_states), f"Old states found: {state_values & old_states}"


# ---------------------------------------------------------------------------
# Task model — new fields
# ---------------------------------------------------------------------------


class TestTaskModelNewFields:
    def test_task_has_completion_field(self):
        task = Task(title="Test")
        assert hasattr(task, "completion")
        assert task.completion == 0

    def test_task_has_paused_until_field(self):
        task = Task(title="Test")
        assert hasattr(task, "paused_until")
        assert task.paused_until is None

    def test_task_has_pause_reason_field(self):
        task = Task(title="Test")
        assert hasattr(task, "pause_reason")
        assert task.pause_reason is None

    def test_default_state_is_active(self):
        task = Task(title="Test")
        assert task.state == TaskState.ACTIVE

    def test_completion_range_stored(self):
        task = Task(title="Test", completion=50)
        assert task.completion == 50

    def test_paused_until_stored(self):
        future = datetime.utcnow() + timedelta(days=7)
        task = Task(title="Test", state=TaskState.PAUSED, paused_until=future, pause_reason="waiting")
        assert task.paused_until == future
        assert task.pause_reason == "waiting"


# ---------------------------------------------------------------------------
# Repository — task postponement
# ---------------------------------------------------------------------------


class TestTaskPostponement:
    def test_postpone_task_sets_paused_state(self, repo):
        """Postponing a task sets state=paused and paused_until."""
        task = Task(title="To postpone", state=TaskState.ACTIVE)
        repo.create_task(task)

        future = datetime.utcnow() + timedelta(days=7)
        task.state = TaskState.PAUSED
        task.paused_until = future
        task.pause_reason = "waiting for review"
        repo.update_task(task)

        retrieved = repo.get_task(task.id)
        assert retrieved.state == TaskState.PAUSED
        assert retrieved.paused_until is not None
        assert retrieved.pause_reason == "waiting for review"

    def test_postponed_task_not_in_active_list(self, repo):
        """Paused tasks are not returned when filtering for active tasks."""
        active = Task(title="Active", state=TaskState.ACTIVE)
        paused = Task(title="Paused", state=TaskState.PAUSED,
                      paused_until=datetime.utcnow() + timedelta(days=7))
        repo.create_task(active)
        repo.create_task(paused)

        active_tasks = repo.list_tasks(state=TaskState.ACTIVE)
        assert any(t.id == active.id for t in active_tasks)
        assert not any(t.id == paused.id for t in active_tasks)


# ---------------------------------------------------------------------------
# Repository — auto-activate paused tasks
# ---------------------------------------------------------------------------


class TestAutoActivatePausedTasks:
    def test_auto_activate_sets_state_to_active(self, repo):
        """Tasks whose paused_until has passed are automatically activated."""
        past = datetime.utcnow() - timedelta(hours=1)
        task = Task(
            title="Expired pause",
            state=TaskState.PAUSED,
            paused_until=past,
            pause_reason="was waiting",
        )
        repo.create_task(task)

        count = repo.auto_activate_paused_tasks()
        assert count == 1

        retrieved = repo.get_task(task.id)
        assert retrieved.state == TaskState.ACTIVE
        assert retrieved.paused_until is None
        assert retrieved.pause_reason is None

    def test_auto_activate_ignores_future_pauses(self, repo):
        """Tasks with paused_until in the future are NOT activated."""
        future = datetime.utcnow() + timedelta(days=7)
        task = Task(
            title="Future pause",
            state=TaskState.PAUSED,
            paused_until=future,
        )
        repo.create_task(task)

        count = repo.auto_activate_paused_tasks()
        assert count == 0

        retrieved = repo.get_task(task.id)
        assert retrieved.state == TaskState.PAUSED

    def test_auto_activate_returns_count(self, repo):
        """auto_activate_paused_tasks returns the number of tasks activated."""
        past = datetime.utcnow() - timedelta(hours=1)
        t1 = Task(title="T1", state=TaskState.PAUSED, paused_until=past)
        t2 = Task(title="T2", state=TaskState.PAUSED, paused_until=past)
        t3 = Task(title="T3", state=TaskState.PAUSED,
                  paused_until=datetime.utcnow() + timedelta(days=1))
        repo.create_task(t1)
        repo.create_task(t2)
        repo.create_task(t3)

        count = repo.auto_activate_paused_tasks()
        assert count == 2

    def test_auto_activate_ignores_active_tasks(self, repo):
        """Active tasks are unaffected by auto_activate_paused_tasks."""
        task = Task(title="Already active", state=TaskState.ACTIVE)
        repo.create_task(task)

        count = repo.auto_activate_paused_tasks()
        assert count == 0
        retrieved = repo.get_task(task.id)
        assert retrieved.state == TaskState.ACTIVE

    def test_auto_activate_clears_pause_reason(self, repo):
        """Auto-activation clears both paused_until and pause_reason."""
        past = datetime.utcnow() - timedelta(hours=2)
        task = Task(
            title="Waiting task",
            state=TaskState.PAUSED,
            paused_until=past,
            pause_reason="waiting for design approval",
        )
        repo.create_task(task)
        repo.auto_activate_paused_tasks()

        retrieved = repo.get_task(task.id)
        assert retrieved.pause_reason is None
        assert retrieved.paused_until is None


# ---------------------------------------------------------------------------
# Session start guards
# ---------------------------------------------------------------------------


class TestSessionStartGuards:
    def test_cannot_start_session_on_done_task(self, repo, mock_timer):
        """start_session raises RuleViolation when task.completion >= 100."""
        task = Task(title="Done", state=TaskState.DONE, completion=100)
        repo.create_task(task)
        manager = SessionManager(repo, timer_manager=mock_timer)

        with pytest.raises(RuleViolation, match="already complete"):
            manager.start_session(task)

    def test_cannot_start_session_on_paused_task(self, repo, mock_timer):
        """start_session raises RuleViolation when task.state == PAUSED."""
        future = datetime.utcnow() + timedelta(days=7)
        task = Task(title="Paused", state=TaskState.PAUSED, paused_until=future)
        repo.create_task(task)
        manager = SessionManager(repo, timer_manager=mock_timer)

        with pytest.raises(RuleViolation, match="paused until"):
            manager.start_session(task)

    def test_can_start_session_on_partially_complete_task(self, repo, mock_timer):
        """start_session succeeds for task with completion < 100."""
        task = Task(title="Partial", state=TaskState.ACTIVE, completion=40)
        repo.create_task(task)
        manager = SessionManager(repo, timer_manager=mock_timer)

        session = manager.start_session(task)
        assert session is not None


# ---------------------------------------------------------------------------
# Session pause — does NOT change task state
# ---------------------------------------------------------------------------


class TestSessionPauseDoesNotChangeTaskState:
    def test_pause_session_leaves_task_active(self, repo, mock_timer):
        """pause_session (mid-work pause) does NOT change task state."""
        task = Task(title="Active", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.pause_session("interrupted by meeting")

        refreshed = repo.get_task(task.id)
        assert refreshed.state == TaskState.ACTIVE

    def test_pause_session_does_not_update_completion(self, repo, mock_timer):
        """pause_session does NOT change task.completion."""
        task = Task(title="Active", state=TaskState.ACTIVE, completion=30)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.pause_session()

        refreshed = repo.get_task(task.id)
        assert refreshed.completion == 30


# ---------------------------------------------------------------------------
# finish_session — outcome-driven completion
# ---------------------------------------------------------------------------


class TestFinishSessionCompletion:
    def test_done_sets_completion_100(self, repo, mock_timer):
        """finish_session('done') sets completion=100 and state=done."""
        task = Task(title="Task", state=TaskState.ACTIVE, completion=50)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("done")

        refreshed = repo.get_task(task.id)
        assert refreshed.completion == 100
        assert refreshed.state == TaskState.DONE

    def test_done_sets_completed_at(self, repo, mock_timer):
        """finish_session('done') records completed_at timestamp."""
        task = Task(title="Task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("done")

        refreshed = repo.get_task(task.id)
        assert refreshed.completed_at is not None

    def test_partial_sets_completion_from_pct(self, repo, mock_timer):
        """finish_session('partial', completion_pct=60) sets completion=60."""
        task = Task(title="Task", state=TaskState.ACTIVE, completion=0)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("partial", completion_pct=60)

        refreshed = repo.get_task(task.id)
        assert refreshed.completion == 60

    def test_partial_caps_completion_at_99(self, repo, mock_timer):
        """finish_session('partial', pct=100) caps at 99 to prevent premature done."""
        task = Task(title="Task", state=TaskState.ACTIVE, completion=0)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("partial", completion_pct=100)

        refreshed = repo.get_task(task.id)
        assert refreshed.completion == 99
        assert refreshed.state == TaskState.ACTIVE  # Not done until explicit 'done' outcome

    def test_blocked_leaves_task_active(self, repo, mock_timer):
        """finish_session('blocked') does not change task state or completion."""
        task = Task(title="Task", state=TaskState.ACTIVE, completion=25)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("blocked")

        refreshed = repo.get_task(task.id)
        assert refreshed.state == TaskState.ACTIVE
        assert refreshed.completion == 25

    def test_abandoned_leaves_task_active(self, repo, mock_timer):
        """finish_session('abandoned') does not change task state or completion."""
        task = Task(title="Task", state=TaskState.ACTIVE, completion=10)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        manager = SessionManager(repo, timer_manager=mock_timer)
        manager.finish_session("abandoned")

        refreshed = repo.get_task(task.id)
        assert refreshed.state == TaskState.ACTIVE
        assert refreshed.completion == 10


# ---------------------------------------------------------------------------
# cancel_stale_pauses — does NOT change task state
# ---------------------------------------------------------------------------


class TestStaleSessionCancelDoesNotAffectTask:
    def test_cancel_stale_pauses_leaves_task_active(self, repo, mock_timer):
        """Stale session closure does not modify task state or completion."""
        task = Task(title="Stale", state=TaskState.ACTIVE, completion=30)
        repo.create_task(task)

        session = Session(task_id=task.id)
        repo.create_session(session)

        # Create a stale pause interval (4 hours ago)
        old_pause = datetime.utcnow() - timedelta(hours=4)
        repo.create_pause_interval(session.id, old_pause)

        manager = SessionManager(repo, timer_manager=mock_timer)
        closed = manager.cancel_stale_pauses(max_hours=3)

        assert task.id in closed

        refreshed = repo.get_task(task.id)
        assert refreshed.state == TaskState.ACTIVE
        assert refreshed.completion == 30


# ---------------------------------------------------------------------------
# Database migration — _migrate_task_completion
# ---------------------------------------------------------------------------


class TestDatabaseMigration:
    def test_migration_adds_completion_column(self, repo):
        """After init_db, tasks table has completion column."""
        task = Task(title="Test", completion=50)
        repo.create_task(task)

        retrieved = repo.get_task(task.id)
        assert retrieved.completion == 50

    def test_migration_adds_paused_until_column(self, repo):
        """After init_db, tasks table has paused_until column."""
        future = datetime.utcnow() + timedelta(days=3)
        task = Task(title="Test", state=TaskState.PAUSED, paused_until=future)
        repo.create_task(task)

        retrieved = repo.get_task(task.id)
        assert retrieved.paused_until is not None

    def test_migration_adds_pause_reason_column(self, repo):
        """After init_db, tasks table has pause_reason column."""
        task = Task(title="Test", state=TaskState.PAUSED, pause_reason="blocked")
        repo.create_task(task)

        retrieved = repo.get_task(task.id)
        assert retrieved.pause_reason == "blocked"


# ---------------------------------------------------------------------------
# No old state machine in codebase
# ---------------------------------------------------------------------------


class TestNoOldStateMachine:
    def test_no_validate_state_transition(self):
        """validate_state_transition should not exist in rules module."""
        from pb.domain import rules
        assert not hasattr(rules, "validate_state_transition")

    def test_no_allowed_transitions(self):
        """ALLOWED_TRANSITIONS should not exist in rules module."""
        from pb.domain import rules
        assert not hasattr(rules, "ALLOWED_TRANSITIONS")

    def test_no_can_transition(self):
        """can_transition should not exist in rules module."""
        from pb.domain import rules
        assert not hasattr(rules, "can_transition")
