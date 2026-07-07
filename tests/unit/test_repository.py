"""Tests for repository CRUD operations."""

import pytest

from pb.domain.enums import Horizon, ProjectStatus, TaskState
from pb.domain.models import GoalArc, Project, Session, Task, Track
from pb.domain.rules import RuleViolation
from pb.storage.repository import Repository


class TestTaskRepository:
    """Test Task CRUD operations."""

    def test_create_task(self, repo, sample_task):
        created = repo.create_task(sample_task)

        assert created.id == sample_task.id
        assert created.title == sample_task.title

    def test_get_task(self, repo, sample_task):
        repo.create_task(sample_task)

        retrieved = repo.get_task(sample_task.id)

        assert retrieved is not None
        assert retrieved.id == sample_task.id

    def test_get_nonexistent_task(self, repo):
        retrieved = repo.get_task("nonexistent")
        assert retrieved is None

    def test_list_tasks(self, repo):
        task1 = Task(title="Task 1")
        task2 = Task(title="Task 2")
        repo.create_task(task1)
        repo.create_task(task2)

        tasks = repo.list_tasks()

        assert len(tasks) == 2

    def test_list_tasks_by_state(self, repo):
        active_task = Task(title="Active", state=TaskState.ACTIVE)
        paused_task = Task(title="Paused", state=TaskState.PAUSED)
        repo.create_task(active_task)
        repo.create_task(paused_task)

        active_tasks = repo.list_tasks(state=TaskState.ACTIVE)

        assert len(active_tasks) == 1
        assert active_tasks[0].title == "Active"

    def test_update_task(self, repo, sample_task):
        sample_task.state = TaskState.ACTIVE
        repo.create_task(sample_task)

        sample_task.title = "Updated"
        sample_task.state = TaskState.ACTIVE
        repo.update_task(sample_task)

        retrieved = repo.get_task(sample_task.id)
        assert retrieved.title == "Updated"
        assert retrieved.state == TaskState.ACTIVE

    def test_multiple_active_tasks_allowed(self, repo):
        """Multiple tasks can be ACTIVE — focus is enforced at session level."""
        task1 = Task(title="Task 1", state=TaskState.ACTIVE)
        task2 = Task(title="Task 2", state=TaskState.ACTIVE)
        repo.create_task(task1)
        repo.create_task(task2)

        tasks = repo.list_tasks(state=TaskState.ACTIVE)
        assert len(tasks) == 2

    def test_get_active_task_via_session(self, repo):
        """get_active_task returns the task for the currently active session."""
        from pb.domain.models import Session

        task = Task(title="Active", state=TaskState.ACTIVE)
        repo.create_task(task)
        session = Session(task_id=task.id)
        repo.create_session(session)

        active = repo.get_active_task()

        assert active is not None
        assert active.id == task.id

    def test_get_active_task_none_without_session(self, repo):
        """get_active_task returns None when no session is active."""
        task = Task(title="Active", state=TaskState.ACTIVE)
        repo.create_task(task)

        active = repo.get_active_task()
        assert active is None


class TestProjectRepository:
    """Test Project CRUD operations."""

    def test_create_project(self, repo, sample_project):
        created = repo.create_project(sample_project)
        assert created.id == sample_project.id

    def test_project_packet_required(self, repo):
        """INV-2: Project requires packet_path."""
        project = Project(name="No packet", packet_path="")

        with pytest.raises(RuleViolation):
            repo.create_project(project)

    def test_list_projects(self, repo, sample_project):
        repo.create_project(sample_project)

        projects = repo.list_projects()

        assert len(projects) == 1


class TestSessionRepository:
    """Test Session CRUD operations."""

    def test_create_session(self, repo, sample_task):
        repo.create_task(sample_task)
        session = Session(task_id=sample_task.id)

        created = repo.create_session(session)

        assert created.id == session.id

    def test_get_active_session(self, repo, sample_task):
        repo.create_task(sample_task)
        session = Session(task_id=sample_task.id)
        repo.create_session(session)

        active = repo.get_active_session()

        assert active is not None
        assert active.end_at is None

    def test_session_phase8_fields_round_trip(self, repo, sample_task):
        """Phase 8 fields (expectation, completion_pct, distraction) survive DB round-trip."""
        repo.create_task(sample_task)
        session = Session(
            task_id=sample_task.id,
            expectation="finish graph writer",
            completion_pct=80,
            distraction=2,
        )
        repo.create_session(session)

        retrieved = repo.get_session(session.id)

        assert retrieved is not None
        assert retrieved.expectation == "finish graph writer"
        assert retrieved.completion_pct == 80
        assert retrieved.distraction == 2

    def test_session_phase8_fields_none_by_default(self, repo, sample_task):
        """Session created without Phase 8 fields reads back as None."""
        repo.create_task(sample_task)
        session = Session(task_id=sample_task.id)
        repo.create_session(session)

        retrieved = repo.get_session(session.id)

        assert retrieved is not None
        assert retrieved.expectation is None
        assert retrieved.completion_pct is None
        assert retrieved.distraction is None

    def test_update_session_persists_phase8_fields(self, repo, sample_task):
        """update_session persists Phase 8 fields."""
        repo.create_task(sample_task)
        session = Session(task_id=sample_task.id)
        repo.create_session(session)

        session.expectation = "write tests"
        session.completion_pct = 75
        session.distraction = 3
        repo.update_session(session)

        retrieved = repo.get_session(session.id)
        assert retrieved.expectation == "write tests"
        assert retrieved.completion_pct == 75
        assert retrieved.distraction == 3

    def test_init_db_idempotent(self, temp_dir):
        """Calling init_db twice on the same DB must not raise OperationalError."""
        from pb.storage.database import init_db, set_db_path
        db_path = temp_dir / "idempotent_test.db"
        set_db_path(db_path)
        init_db(db_path)
        # Second call must be idempotent — no OperationalError
        init_db(db_path)


class TestTrackRepository:
    """Test Track CRUD operations."""

    def test_create_track(self, repo, sample_track):
        created = repo.create_track(sample_track)
        assert created.name == sample_track.name

    def test_list_tracks(self, repo, sample_track):
        repo.create_track(sample_track)

        tracks = repo.list_tracks()

        assert len(tracks) == 1


class TestGoalArcRepository:
    """Test GoalArc CRUD operations."""

    def test_create_goal_arc(self, repo, sample_goal):
        created = repo.create_goal_arc(sample_goal)
        assert created.title == sample_goal.title

    def test_list_goal_arcs(self, repo, sample_goal):
        repo.create_goal_arc(sample_goal)

        goals = repo.list_goal_arcs()

        assert len(goals) == 1


class TestTimeBlockRepository:
    """Test TimeBlock CRUD operations."""

    def test_create_time_block(self, repo, sample_time_block):
        """Test creating a time block."""
        created = repo.create_time_block(sample_time_block)
        assert created.id == sample_time_block.id
        assert created.task_id == sample_time_block.task_id
        assert created.duration_minutes == 60

    def test_get_time_block(self, repo, sample_time_block):
        """Test retrieving a time block by exact ID."""
        repo.create_time_block(sample_time_block)
        retrieved = repo.get_time_block(sample_time_block.id)
        assert retrieved is not None
        assert retrieved.id == sample_time_block.id

    def test_get_time_block_nonexistent(self, repo):
        """Test that get_time_block returns None for unknown ID."""
        retrieved = repo.get_time_block("nonexistent")
        assert retrieved is None

    def test_get_time_block_any_date(self, repo, sample_task):
        """Test that get_time_block works for blocks on any date (not just today).

        Validates fix for Pitfall 5 in RESEARCH.md.
        """
        from datetime import datetime, timedelta
        from pb.domain.models import TimeBlock

        repo.create_task(sample_task)

        yesterday = datetime.utcnow() - timedelta(days=1)
        yesterday_start = yesterday.replace(hour=10, minute=0, second=0, microsecond=0)
        block = TimeBlock(
            task_id=sample_task.id,
            start_time=yesterday_start,
            duration_minutes=30,
        )
        repo.create_time_block(block)

        retrieved = repo.get_time_block(block.id)
        assert retrieved is not None
        assert retrieved.id == block.id

    def test_delete_time_block(self, repo, sample_time_block):
        """Test deleting a time block."""
        repo.create_time_block(sample_time_block)

        deleted = repo.delete_time_block(sample_time_block.id)
        assert deleted is True

        retrieved = repo.get_time_block(sample_time_block.id)
        assert retrieved is None

    def test_delete_nonexistent(self, repo):
        """Test that delete_time_block returns False for unknown ID."""
        deleted = repo.delete_time_block("nonexistent")
        assert deleted is False

    def test_update_time_block(self, repo, sample_time_block):
        """Test updating a time block's start_time and duration."""
        from datetime import timedelta

        repo.create_time_block(sample_time_block)

        new_start = sample_time_block.start_time + timedelta(hours=1)
        sample_time_block.start_time = new_start
        sample_time_block.duration_minutes = 90

        updated = repo.update_time_block(sample_time_block)
        assert updated.start_time == new_start
        assert updated.duration_minutes == 90

        retrieved = repo.get_time_block(sample_time_block.id)
        assert retrieved.start_time == new_start
        assert retrieved.duration_minutes == 90
