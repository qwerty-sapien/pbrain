"""Integration tests for full workflow.

Tests the complete cycle: capture → plan → start → finish → review
"""

from datetime import datetime, timedelta

import pytest

from pb.core.packet_engine import PacketEngine
from pb.core.planner import Planner
from pb.core.review_engine import ReviewEngine
from pb.core.sessions import SessionManager
from pb.domain.enums import SessionMode, TaskState
from pb.domain.models import Session, Task
from pb.domain.rules import RuleViolation
from pb.storage.repository import Repository


class TestCaptureToReviewWorkflow:
    """Test full workflow from capture to review."""

    def test_full_workflow(self, repo, temp_config, temp_dir):
        """
        Complete workflow test:
        1. Capture a task
        2. Move to ready
        3. Plan the day
        4. Start a session
        5. Finish the task
        6. Generate review
        """
        planner = Planner(repo)
        session_manager = SessionManager(repo)
        review_engine = ReviewEngine(repo)
        packet_engine = PacketEngine(vault_path=temp_dir)

        task = Task(title="Write tests", state=TaskState.ACTIVE)
        repo.create_task(task)

        active = planner.get_active_tasks()
        assert len(active) == 1

        daily_plan = planner.generate_daily_plan_summary()
        assert "Write tests" in daily_plan

        session = session_manager.start_session(task, intended_outcome="All tests pass")
        assert session is not None
        assert repo.get_task(task.id).state == TaskState.ACTIVE

        finished = session_manager.finish_session("done", "Tests written and passing")
        assert finished is not None
        assert repo.get_task(task.id).state == TaskState.DONE

        review = review_engine.generate_daily_review()
        assert "Write tests" in review
        assert "Daily Review" in review


class TestSingleActiveTaskInvariant:
    """INV-1: Integration tests for single active task."""

    def test_cannot_start_second_task(self, repo, temp_config):
        """Starting a second task while one is active must fail."""
        session_manager = SessionManager(repo)

        task1 = Task(title="Task 1", state=TaskState.ACTIVE)
        task2 = Task(title="Task 2", state=TaskState.ACTIVE)
        repo.create_task(task1)
        repo.create_task(task2)

        session_manager.start_session(task1)

        with pytest.raises(RuleViolation):
            session_manager.start_session(task2)

    def test_can_start_after_pause(self, repo, temp_config):
        """Can start a new task after pausing the current one."""
        session_manager = SessionManager(repo)

        task1 = Task(title="Task 1", state=TaskState.ACTIVE)
        task2 = Task(title="Task 2", state=TaskState.ACTIVE)
        repo.create_task(task1)
        repo.create_task(task2)

        session_manager.start_session(task1)
        session_manager.pause_session()

        session = session_manager.start_session(task2)
        assert session is not None


class TestSessionLifecycle:
    """Test session start/pause/finish lifecycle."""

    def test_start_creates_session(self, repo, temp_config):
        session_manager = SessionManager(repo)
        task = Task(title="Test", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = session_manager.start_session(task)

        assert session.task_id == task.id
        assert session.end_at is None

    def test_pause_ends_session(self, repo, temp_config):
        session_manager = SessionManager(repo)
        task = Task(title="Test", state=TaskState.ACTIVE)
        repo.create_task(task)

        session_manager.start_session(task)
        paused = session_manager.pause_session("Paused for lunch")

        assert paused.end_at is not None
        assert paused.actual_outcome == "Paused for lunch"
        assert repo.get_task(task.id).state == TaskState.ACTIVE

    def test_finish_completes_task(self, repo, temp_config):
        session_manager = SessionManager(repo)
        task = Task(title="Test", state=TaskState.ACTIVE)
        repo.create_task(task)

        session_manager.start_session(task)
        finished = session_manager.finish_session("done", "All done")

        assert finished.end_at is not None
        assert repo.get_task(task.id).state == TaskState.DONE

    def test_interrupt_increments_count(self, repo, temp_config):
        session_manager = SessionManager(repo)
        task = Task(title="Test", state=TaskState.ACTIVE)
        repo.create_task(task)

        session_manager.start_session(task)

        session_manager.log_interruption()
        session_manager.log_interruption()

        session = session_manager.get_current_session()
        assert session.interruption_count == 2


class TestReviewGeneration:
    """Test review report generation."""

    def test_daily_review_shows_sessions(self, repo, temp_config):
        session_manager = SessionManager(repo)
        review_engine = ReviewEngine(repo)

        task = Task(title="Review test", state=TaskState.ACTIVE)
        repo.create_task(task)

        session_manager.start_session(task)
        session_manager.finish_session("done")

        review = review_engine.generate_daily_review()

        assert "Review test" in review
        # New table format per D-01, D-02: Sessions count is in summary table
        assert "| Sessions" in review

    def test_weekly_review_aggregates(self, repo, temp_config):
        review_engine = ReviewEngine(repo)

        review = review_engine.generate_weekly_review()

        assert "Weekly Review" in review
        # Updated per D-19, D-20: weekly review uses summary table format
        assert "| METRIC" in review
        assert "| Tasks Completed" in review


class TestPlannerOperations:
    """Test planner functionality."""

    def test_active_tasks_listing(self, repo):
        planner = Planner(repo)

        task1 = Task(title="Active 1", state=TaskState.ACTIVE)
        task2 = Task(title="Active 2", state=TaskState.ACTIVE)
        task3 = Task(title="Paused", state=TaskState.PAUSED)
        repo.create_task(task1)
        repo.create_task(task2)
        repo.create_task(task3)

        active = planner.get_active_tasks()

        assert len(active) == 2

    def test_schedule_block(self, repo):
        planner = Planner(repo)
        task = Task(title="Blocked", state=TaskState.ACTIVE)
        repo.create_task(task)

        start = datetime.utcnow().replace(hour=9, minute=0)
        block, overlap = planner.schedule_block(task, start, 60)

        assert block.task_id == task.id
        assert block.duration_minutes == 60
        assert overlap is None

    def test_daily_plan_summary(self, repo):
        planner = Planner(repo)

        task = Task(title="Plan test", state=TaskState.ACTIVE)
        repo.create_task(task)

        summary = planner.generate_daily_plan_summary()

        assert "Daily Plan" in summary
        assert "Plan test" in summary
