"""Unit tests for priority scoring, Eisenhower classification, and task ranking (Plan 02-06).

Covers:
- compute_priority_score() formula and validation
- classify_eisenhower() quadrant mapping
- get_priority_action() threshold mapping
- task_priority_score() and task_eisenhower() with Task model
- rank_tasks() descending sort
- WorkType enum values
- Task model new priority fields persist to DB and read back correctly
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from pb.domain.enums import EisenhowerClass, PriorityAction, WorkType
from pb.domain.models import Task
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db_dir(tmp_path):
    """Set up a temporary database for each test."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)

    vault_path = tmp_path / "vault"
    vault_path.mkdir()
    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    config_module._config = config

    yield db_path

    db_module._db_path = None
    config_module._config = None


@pytest.fixture
def repo(temp_db_dir):
    return Repository()


# ---------------------------------------------------------------------------
# WorkType enum
# ---------------------------------------------------------------------------


class TestWorkTypeEnum:
    """Tests for WorkType enum values."""

    def test_work_type_deep(self):
        assert WorkType.DEEP == "deep"

    def test_work_type_shallow(self):
        assert WorkType.SHALLOW == "shallow"

    def test_work_type_admin(self):
        assert WorkType.ADMIN == "admin"

    def test_work_type_meeting(self):
        assert WorkType.MEETING == "meeting"

    def test_work_type_recovery(self):
        assert WorkType.RECOVERY == "recovery"

    def test_work_type_planning(self):
        assert WorkType.PLANNING == "planning"

    def test_work_type_has_six_values(self):
        assert len(WorkType) == 6


# ---------------------------------------------------------------------------
# EisenhowerClass and PriorityAction enums
# ---------------------------------------------------------------------------


class TestEisenhowerClassEnum:
    def test_do_today(self):
        assert EisenhowerClass.DO_TODAY == "do_today"

    def test_schedule_deep_work(self):
        assert EisenhowerClass.SCHEDULE_DEEP_WORK == "schedule_deep_work"

    def test_batch_delegate_or_automate(self):
        assert EisenhowerClass.BATCH_DELEGATE_OR_AUTOMATE == "batch_delegate_or_automate"

    def test_delete_or_defer(self):
        assert EisenhowerClass.DELETE_OR_DEFER == "delete_or_defer"


class TestPriorityActionEnum:
    def test_schedule_first(self):
        assert PriorityAction.SCHEDULE_FIRST == "schedule_first"

    def test_schedule_if_capacity(self):
        assert PriorityAction.SCHEDULE_IF_CAPACITY == "schedule_if_capacity"

    def test_batch_delegate_simplify(self):
        assert PriorityAction.BATCH_DELEGATE_SIMPLIFY == "batch_delegate_simplify"

    def test_drop_or_defer(self):
        assert PriorityAction.DROP_OR_DEFER == "drop_or_defer"


# ---------------------------------------------------------------------------
# compute_priority_score
# ---------------------------------------------------------------------------


class TestComputePriorityScore:
    """Tests for compute_priority_score() formula."""

    def test_high_scores_low_effort(self):
        from pb.core.priority import compute_priority_score

        result = compute_priority_score(impact=5, urgency=4, strategic_value=3, effort=2)
        assert result == 6.0

    def test_low_scores_high_effort(self):
        from pb.core.priority import compute_priority_score

        result = compute_priority_score(impact=1, urgency=1, strategic_value=1, effort=5)
        assert abs(result - 0.6) < 0.001

    def test_effort_zero_raises_value_error(self):
        from pb.core.priority import compute_priority_score

        with pytest.raises(ValueError, match="Effort must be positive"):
            compute_priority_score(impact=5, urgency=5, strategic_value=5, effort=0)

    def test_effort_negative_raises_value_error(self):
        from pb.core.priority import compute_priority_score

        with pytest.raises(ValueError):
            compute_priority_score(impact=5, urgency=5, strategic_value=5, effort=-1)

    def test_equal_scores_equal_effort(self):
        from pb.core.priority import compute_priority_score

        result = compute_priority_score(impact=3, urgency=3, strategic_value=3, effort=3)
        assert result == 3.0

    def test_returns_float(self):
        from pb.core.priority import compute_priority_score

        result = compute_priority_score(impact=5, urgency=5, strategic_value=5, effort=3)
        assert isinstance(result, float)


# ---------------------------------------------------------------------------
# classify_eisenhower
# ---------------------------------------------------------------------------


class TestClassifyEisenhower:
    """Tests for classify_eisenhower() quadrant mapping."""

    def test_important_and_urgent_is_do_today(self):
        from pb.core.priority import classify_eisenhower

        result = classify_eisenhower(important=True, urgent=True)
        assert result == EisenhowerClass.DO_TODAY

    def test_important_not_urgent_is_schedule_deep_work(self):
        from pb.core.priority import classify_eisenhower

        result = classify_eisenhower(important=True, urgent=False)
        assert result == EisenhowerClass.SCHEDULE_DEEP_WORK

    def test_not_important_urgent_is_batch_delegate(self):
        from pb.core.priority import classify_eisenhower

        result = classify_eisenhower(important=False, urgent=True)
        assert result == EisenhowerClass.BATCH_DELEGATE_OR_AUTOMATE

    def test_not_important_not_urgent_is_delete_or_defer(self):
        from pb.core.priority import classify_eisenhower

        result = classify_eisenhower(important=False, urgent=False)
        assert result == EisenhowerClass.DELETE_OR_DEFER


# ---------------------------------------------------------------------------
# get_priority_action
# ---------------------------------------------------------------------------


class TestGetPriorityAction:
    """Tests for get_priority_action() threshold mapping."""

    def test_score_4_5_returns_schedule_first(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=4.5)
        assert result == PriorityAction.SCHEDULE_FIRST

    def test_score_exactly_4_returns_schedule_first(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=4.0)
        assert result == PriorityAction.SCHEDULE_FIRST

    def test_score_3_0_returns_schedule_if_capacity(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=3.0)
        assert result == PriorityAction.SCHEDULE_IF_CAPACITY

    def test_score_exactly_2_5_returns_schedule_if_capacity(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=2.5)
        assert result == PriorityAction.SCHEDULE_IF_CAPACITY

    def test_score_2_0_returns_batch_delegate_simplify(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=2.0)
        assert result == PriorityAction.BATCH_DELEGATE_SIMPLIFY

    def test_score_exactly_1_5_returns_batch_delegate_simplify(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=1.5)
        assert result == PriorityAction.BATCH_DELEGATE_SIMPLIFY

    def test_score_1_0_returns_drop_or_defer(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=1.0)
        assert result == PriorityAction.DROP_OR_DEFER

    def test_score_below_1_5_returns_drop_or_defer(self):
        from pb.core.priority import get_priority_action

        result = get_priority_action(score=1.4)
        assert result == PriorityAction.DROP_OR_DEFER


# ---------------------------------------------------------------------------
# rank_tasks
# ---------------------------------------------------------------------------


class TestRankTasks:
    """Tests for rank_tasks() descending sort."""

    def test_rank_tasks_sorts_descending(self):
        from pb.core.priority import rank_tasks

        task_a = Task(title="Low priority", impact=1, urgency_score=1, strategic_value=1, effort=5)
        task_b = Task(title="High priority", impact=5, urgency_score=4, strategic_value=3, effort=2)
        result = rank_tasks([task_a, task_b])
        assert result[0].title == "High priority"
        assert result[1].title == "Low priority"

    def test_rank_tasks_unscored_goes_last(self):
        from pb.core.priority import rank_tasks

        task_scored = Task(title="Scored", impact=2, urgency_score=2, strategic_value=2, effort=2)
        task_unscored = Task(title="Unscored")  # No priority fields
        result = rank_tasks([task_unscored, task_scored])
        assert result[0].title == "Scored"
        assert result[1].title == "Unscored"

    def test_rank_tasks_empty_list(self):
        from pb.core.priority import rank_tasks

        result = rank_tasks([])
        assert result == []

    def test_rank_tasks_single_task(self):
        from pb.core.priority import rank_tasks

        task = Task(title="Only task", impact=3, urgency_score=3, strategic_value=3, effort=3)
        result = rank_tasks([task])
        assert len(result) == 1


# ---------------------------------------------------------------------------
# Task model priority fields and DB persistence
# ---------------------------------------------------------------------------


class TestTaskModelPriorityFields:
    """Tests for Task model new priority fields."""

    def test_task_has_impact_field(self):
        task = Task(title="test")
        assert task.impact is None

    def test_task_has_urgency_score_field(self):
        task = Task(title="test")
        assert task.urgency_score is None

    def test_task_has_strategic_value_field(self):
        task = Task(title="test")
        assert task.strategic_value is None

    def test_task_has_effort_field(self):
        task = Task(title="test")
        assert task.effort is None

    def test_task_has_important_field(self):
        task = Task(title="test")
        assert task.important is None

    def test_task_has_urgent_field(self):
        task = Task(title="test")
        assert task.urgent is None

    def test_task_has_energy_required_field(self):
        task = Task(title="test")
        assert task.energy_required is None

    def test_task_has_work_type_field(self):
        task = Task(title="test")
        assert task.work_type is None

    def test_task_has_due_date_field(self):
        task = Task(title="test")
        assert task.due_date is None

    def test_task_has_scheduled_date_field(self):
        task = Task(title="test")
        assert task.scheduled_date is None

    def test_task_has_estimated_minutes_field(self):
        task = Task(title="test")
        assert task.estimated_minutes is None

    def test_task_has_actual_minutes_field(self):
        task = Task(title="test")
        assert task.actual_minutes is None


class TestTaskPriorityDBPersistence:
    """Tests for Task priority fields persisting to DB and reading back correctly."""

    def test_task_with_priority_fields_persists(self, repo):
        """Create a task with priority fields; read back; values match."""
        task = Task(
            title="Priority task",
            impact=5,
            urgency_score=4,
            strategic_value=3,
            effort=2,
            important=True,
            urgent=False,
            energy_required=3,
            work_type="deep",
        )
        created = repo.create_task(task)
        read_back = repo.get_task(created.id)

        assert read_back is not None
        assert read_back.impact == 5
        assert read_back.urgency_score == 4
        assert read_back.strategic_value == 3
        assert read_back.effort == 2
        assert read_back.important is True
        assert read_back.urgent is False
        assert read_back.energy_required == 3
        assert read_back.work_type == "deep"

    def test_task_with_null_priority_fields_persists(self, repo):
        """Create a task without priority fields; read back; values are None."""
        task = Task(title="Unscored task")
        created = repo.create_task(task)
        read_back = repo.get_task(created.id)

        assert read_back is not None
        assert read_back.impact is None
        assert read_back.urgency_score is None
        assert read_back.strategic_value is None
        assert read_back.effort is None

    def test_update_task_priority_fields_persist(self, repo):
        """Update task priority fields; read back; values updated."""
        task = Task(title="Updateable task")
        created = repo.create_task(task)

        created.impact = 3
        created.urgency_score = 2
        created.strategic_value = 4
        created.effort = 1
        created.important = True
        created.urgent = True
        repo.update_task(created)

        read_back = repo.get_task(created.id)
        assert read_back.impact == 3
        assert read_back.urgency_score == 2
        assert read_back.strategic_value == 4
        assert read_back.effort == 1
        assert read_back.important is True
        assert read_back.urgent is True
