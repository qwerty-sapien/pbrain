"""Tests for --budget flag on pb plan day (gap closure SC #5)."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from unittest.mock import patch

import pytest

from pb.cli.commands.plan import _apply_day_plan_refinement, parse_budget
from pb.core.planner import Planner
from pb.domain.models import TimeBlock, Task, generate_internal_id
from pb.domain.enums import TaskState, Horizon
from pb.llm.drafts import LearningPlanBlockDraft, MixedPlanDraft
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository
from pb.storage import config as config_module
from pb.storage.config import Config, GeneralConfig, StorageConfig


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

    from pb.storage import database as db_module
    db_module._db_path = None
    config_module._config = None


@pytest.fixture
def repo(temp_db_dir):
    return Repository()


@pytest.fixture
def planner(repo):
    return Planner(repo)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_today_block(repo: Repository, task_id: str, duration_minutes: int) -> TimeBlock:
    """Create a time block for today with the given duration and persist it."""
    block = TimeBlock(task_id=task_id, duration_minutes=duration_minutes)
    return repo.create_time_block(block)


def _make_task(repo: Repository, title: str = "Budget test task") -> Task:
    task = Task(title=title, state=TaskState.ACTIVE, horizon=Horizon.TODAY)
    return repo.create_task(task)


# ===========================================================================
# TestParseBudget — unit tests for parse_budget
# ===========================================================================

class TestParseBudget:
    """Unit tests for parse_budget duration string parsing."""

    def test_parse_hours(self):
        """4h -> 240 minutes."""
        assert parse_budget("4h") == 240

    def test_parse_minutes(self):
        """240m -> 240 minutes."""
        assert parse_budget("240m") == 240

    def test_parse_hours_and_minutes(self):
        """2h30m -> 150 minutes."""
        assert parse_budget("2h30m") == 150

    def test_parse_spaced_hours_and_minutes(self):
        """1h 10min -> 70 minutes."""
        assert parse_budget("1h 10min") == 70

    def test_parse_wordy_minutes(self):
        """2 minutes -> 2 minutes."""
        assert parse_budget("2 minutes") == 2

    def test_parse_fractional_hours(self):
        """1.5h -> 90 minutes."""
        assert parse_budget("1.5h") == 90

    def test_parse_bare_number_as_minutes(self):
        """90 -> 90 minutes (bare number treated as minutes)."""
        assert parse_budget("90") == 90

    def test_parse_zero_raises(self):
        """0h must raise ValueError (budget must be positive)."""
        with pytest.raises(ValueError):
            parse_budget("0h")

    def test_parse_invalid_raises(self):
        """'abc' must raise ValueError."""
        with pytest.raises(ValueError):
            parse_budget("abc")

    def test_parse_empty_raises(self):
        """Empty string must raise ValueError."""
        with pytest.raises(ValueError):
            parse_budget("")

    def test_parse_zero_minutes_raises(self):
        """0m must raise ValueError."""
        with pytest.raises(ValueError):
            parse_budget("0m")

    def test_parse_one_hour(self):
        """1h -> 60 minutes."""
        assert parse_budget("1h") == 60

    def test_parse_30m(self):
        """30m -> 30 minutes."""
        assert parse_budget("30m") == 30

    def test_parse_bare_1(self):
        """Bare '1' -> 1 minute."""
        assert parse_budget("1") == 1

    def test_parse_with_leading_whitespace(self):
        """Leading/trailing whitespace is tolerated."""
        assert parse_budget("  4h  ") == 240


class TestDayPlanRefinementHeuristics:
    def test_foundational_request_does_not_trigger_time_scaling(self):
        draft = MixedPlanDraft(
            summary="",
            blocks=[
                LearningPlanBlockDraft(
                    branch="study",
                    subject_scope="curvature tensors",
                    duration_minutes=25,
                    success_check="Explain the basics.",
                    reason="Start here.",
                ),
                LearningPlanBlockDraft(
                    branch="practise",
                    subject_scope="ricci flow application",
                    duration_minutes=35,
                    success_check="Apply it once.",
                    reason="Use it.",
                ),
            ],
        )

        refined = _apply_day_plan_refinement(
            draft,
            "make it much more foundational, i don't even know what curvature tensors are",
            budget_minutes=60,
        )

        assert refined is None


# ===========================================================================
# TestBudgetAllocation — planner budget section
# ===========================================================================

class TestBudgetAllocation:
    """Tests for budget section in generate_daily_plan_summary."""

    def test_no_budget_no_section(self, planner):
        """Without budget_minutes, output must NOT contain 'Budget'."""
        result = planner.generate_daily_plan_summary()
        assert "## Budget" not in result

    def test_budget_with_blocks(self, planner, repo):
        """Two blocks (60m + 90m = 150m committed) against 240m budget -> 90m remaining."""
        task = _make_task(repo)
        _make_today_block(repo, task.id, 60)
        _make_today_block(repo, task.id, 90)

        result = planner.generate_daily_plan_summary(budget_minutes=240)

        assert "## Budget" in result
        assert "4h 0m" in result        # budget: 240m
        assert "2h 30m" in result       # committed: 150m
        assert "1h 30m" in result       # remaining: 90m

    def test_budget_overcommitted(self, planner, repo):
        """Blocks totalling 300m against 240m budget -> OVER by 60m."""
        task = _make_task(repo)
        _make_today_block(repo, task.id, 180)
        _make_today_block(repo, task.id, 120)

        result = planner.generate_daily_plan_summary(budget_minutes=240)

        assert "## Budget" in result
        assert "OVER by" in result
        assert "1h 0m" in result        # over by 60m

    def test_budget_no_blocks(self, planner):
        """With no blocks, committed is 0m and remaining equals budget."""
        result = planner.generate_daily_plan_summary(budget_minutes=240)

        assert "## Budget" in result
        assert "4h 0m" in result        # budget
        assert "Committed: 0h 0m" in result
        assert "Remaining: 4h 0m" in result

    def test_budget_section_format(self, planner):
        """Budget section contains Budget, Committed, Remaining labels."""
        result = planner.generate_daily_plan_summary(budget_minutes=60)

        assert "Budget:" in result
        assert "Committed:" in result
        assert "Remaining:" in result

    def test_no_budget_backwards_compatible(self, planner, repo):
        """generate_daily_plan_summary() without args works as before."""
        task = _make_task(repo)
        _make_today_block(repo, task.id, 60)

        result = planner.generate_daily_plan_summary()

        assert "Daily Plan" in result
        assert "## Budget" not in result
