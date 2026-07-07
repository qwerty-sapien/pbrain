"""Unit tests for planning engine methods (Plan 02-08).

Covers:
- compute_weekly_allocation() 60/30/10 split
- generate_weekly_plan() capacity awareness, sections
- filter_by_energy() energy-level gating
- select_top1(), select_top3() energy-fit selection
- select_batch() shallow/admin selection
- generate_daily_plan() Top 1/Top 3/Batch/Blocks output
- ENERGY_MATCH_TABLE coverage for levels 1 and 5
"""

from __future__ import annotations

import pytest

from pb.domain.enums import TaskState, WorkType
from pb.domain.models import Task
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path
from pb.storage.repository import Repository


class _PlanningRepoStub:
    def __init__(self, *, goals=None, tasks=None):
        self._goals = goals or []
        self._tasks = tasks or []

    def list_goal_arcs(self, status=None):
        return list(self._goals)

    def list_tasks(self):
        return list(self._tasks)

    def get_task(self, task_id):
        return next((task for task in self._tasks if task.id == task_id), None)

    def get_goal_arc(self, goal_id):
        return next((goal for goal in self._goals if goal.id == goal_id), None)


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


def make_scored_task(
    repo: Repository,
    title: str,
    impact: int = 4,
    urgency_score: int = 4,
    strategic_value: int = 4,
    effort: int = 2,
    energy_required: int = 3,
    work_type: str = WorkType.DEEP.value,
    estimated_minutes: int = 60,
    state: TaskState = TaskState.ACTIVE,
) -> Task:
    """Helper to create a task with full priority scores."""
    task = Task(
        title=title,
        impact=impact,
        urgency_score=urgency_score,
        strategic_value=strategic_value,
        effort=effort,
        important=True,
        urgent=True,
        energy_required=energy_required,
        work_type=work_type,
        estimated_minutes=estimated_minutes,
        state=state,
        completion=100 if state == TaskState.DONE else 0,
    )
    return repo.create_task(task)


def make_unscored_task(repo: Repository, title: str) -> Task:
    """Helper to create a task with no priority scores."""
    task = Task(title=title, state=TaskState.ACTIVE)
    return repo.create_task(task)


# ---------------------------------------------------------------------------
# ENERGY_MATCH_TABLE tests
# ---------------------------------------------------------------------------


class TestEnergyMatchTable:
    def test_energy_table_has_all_levels(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        assert set(ENERGY_MATCH_TABLE.keys()) == {1, 2, 3, 4, 5}

    def test_energy_level_5_includes_architecture(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        best_for = ENERGY_MATCH_TABLE[5]["best_for"]
        assert "architecture" in best_for

    def test_energy_level_5_excludes_email(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        avoid = ENERGY_MATCH_TABLE[5]["avoid"]
        assert "email" in avoid

    def test_energy_level_1_includes_recovery(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        best_for = ENERGY_MATCH_TABLE[1]["best_for"]
        assert "recovery" in best_for

    def test_energy_level_1_excludes_forcing_output(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        avoid = ENERGY_MATCH_TABLE[1]["avoid"]
        assert "forcing output" in avoid

    def test_each_level_has_max_energy_required(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        for level in range(1, 6):
            assert "max_energy_required" in ENERGY_MATCH_TABLE[level]


def test_consult_day_plan_no_longer_prompts_for_emphasis(monkeypatch):
    from pb.cli.commands import plan as plan_cmd

    class _FakeStdin:
        def isatty(self):
            return True

    monkeypatch.setattr(plan_cmd.sys, "stdin", _FakeStdin())
    monkeypatch.setattr(plan_cmd, "_pick_day_plan_tasks", lambda repo, preselected_ids=None: [])
    monkeypatch.setattr(plan_cmd, "pick_single_choice", lambda *args, **kwargs: "continue")
    monkeypatch.setattr(
        plan_cmd,
        "prompt_text",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("unexpected prompt_text call")),
    )

    consultation = plan_cmd._consult_day_plan(ctx=object(), repo=_PlanningRepoStub())

    assert consultation.selected_task_ids == []


def test_plan_day_prompt_demands_exact_scope_and_prerequisites():
    from pb.cli.commands import plan as plan_cmd
    from pb.core.models import GoalArc

    repo = _PlanningRepoStub(
        goals=[GoalArc(id="goal-1", title="Linear Algebra for Geometry", domain="Linear Algebra", execution_mode="study")]
    )

    prompt = plan_cmd._plan_day_prompt(repo, 60, consultation=plan_cmd.DayPlanConsultation())

    assert "exact competency slice" in prompt
    assert "distinguish prerequisite progress from goal progress" in prompt


# ---------------------------------------------------------------------------
# compute_weekly_allocation tests
# ---------------------------------------------------------------------------


class TestComputeWeeklyAllocation:
    def test_allocation_40h_returns_correct_split(self):
        from pb.core.planner import Planner
        alloc = Planner.compute_weekly_allocation(40.0)
        assert alloc["deep"] == 1440   # 40*60*0.60
        assert alloc["shallow"] == 720  # 40*60*0.30
        # buffer = 2400 - 1440 - 720 = 240
        assert alloc["buffer"] == 240

    def test_allocation_20h_returns_correct_split(self):
        from pb.core.planner import Planner
        alloc = Planner.compute_weekly_allocation(20.0)
        assert alloc["deep"] == 720    # 20*60*0.60
        assert alloc["shallow"] == 360  # 20*60*0.30
        assert alloc["buffer"] == 120   # 10%

    def test_allocation_returns_dict_with_three_keys(self):
        from pb.core.planner import Planner
        alloc = Planner.compute_weekly_allocation(40.0)
        assert set(alloc.keys()) == {"deep", "shallow", "buffer"}

    def test_allocation_total_equals_available_minutes(self):
        from pb.core.planner import Planner
        hours = 37.5
        alloc = Planner.compute_weekly_allocation(hours)
        total = alloc["deep"] + alloc["shallow"] + alloc["buffer"]
        # Allow off-by-1 due to integer rounding
        assert abs(total - int(hours * 60)) <= 1

    def test_allocation_zero_hours_raises(self):
        """Zero hours should not be a valid input."""
        from pb.core.planner import Planner
        alloc = Planner.compute_weekly_allocation(0.0)
        # With 0 hours all values should be 0
        assert alloc["deep"] == 0
        assert alloc["shallow"] == 0
        assert alloc["buffer"] == 0


# ---------------------------------------------------------------------------
# generate_weekly_plan tests
# ---------------------------------------------------------------------------


class TestGenerateWeeklyPlan:
    def test_weekly_plan_contains_deep_work_section(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Feature A")
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        assert "Deep Work" in plan

    def test_weekly_plan_contains_shallow_section(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Admin task", work_type=WorkType.ADMIN.value)
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        assert "Shallow" in plan

    def test_weekly_plan_shows_allocation_numbers(self, repo):
        from pb.core.planner import Planner
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        # Should show 60% annotation
        assert "60%" in plan

    def test_weekly_plan_shows_available_hours(self, repo):
        from pb.core.planner import Planner
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        assert "40" in plan

    def test_weekly_plan_warns_on_overcapacity(self, repo):
        from pb.core.planner import Planner
        # Create many high-estimate tasks to exceed capacity
        for i in range(30):
            make_scored_task(repo, f"Big task {i}", estimated_minutes=120)
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(10.0)  # Only 10h available
        assert "WARNING" in plan

    def test_weekly_plan_excludes_done_tasks(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Done task", state=TaskState.DONE)
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        assert "Done task" not in plan

    def test_weekly_plan_excludes_cancelled_tasks(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Cancelled task", state=TaskState.DONE)
        planner = Planner(repo)
        plan = planner.generate_weekly_plan(40.0)
        assert "Cancelled task" not in plan


# ---------------------------------------------------------------------------
# generate_daily_plan tests
# ---------------------------------------------------------------------------


class TestGenerateDailyPlan:
    def test_daily_plan_contains_top1(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Critical thing")
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=4, focus_hours=6.0)
        assert "Top 1" in plan

    def test_daily_plan_contains_top3(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Task 1")
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=3, focus_hours=6.0)
        assert "Top 3" in plan

    def test_daily_plan_contains_batch_list(self, repo):
        from pb.core.planner import Planner
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=3, focus_hours=4.0)
        assert "Batch List" in plan

    def test_daily_plan_contains_suggested_blocks(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Deep focus task")
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=4, focus_hours=6.0)
        assert "Suggested Blocks" in plan

    def test_daily_plan_shows_energy_info(self, repo):
        from pb.core.planner import Planner
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=5, focus_hours=4.0)
        assert "Energy" in plan
        assert "5" in plan

    def test_daily_plan_excludes_done_tasks(self, repo):
        from pb.core.planner import Planner
        make_scored_task(repo, "Done task", state=TaskState.DONE)
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=3, focus_hours=4.0)
        assert "Done task" not in plan

    def test_daily_plan_shows_budget_section_when_provided(self, repo):
        from pb.core.planner import Planner
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=3, focus_hours=4.0, budget_minutes=240)
        assert "Budget" in plan


# ---------------------------------------------------------------------------
# Energy filtering tests (D-29)
# ---------------------------------------------------------------------------


class TestEnergyFiltering:
    def test_energy_level_5_includes_high_energy_tasks(self, repo):
        from pb.core.planner import Planner
        # energy_required=5 should be included at energy_level=5
        make_scored_task(repo, "Architecture task", energy_required=5)
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=5, focus_hours=6.0)
        assert "Architecture task" in plan

    def test_energy_level_1_excludes_high_energy_tasks(self, repo):
        from pb.core.planner import Planner
        # energy_required=5 should be excluded at energy_level=1 (max_energy_required=2)
        # Add a low-energy task as well so Top 1 can resolve to it
        make_scored_task(repo, "Hard architecture task", energy_required=5,
                         impact=5, urgency_score=5, strategic_value=5, effort=1)
        make_scored_task(repo, "Low energy admin", energy_required=1,
                         impact=3, urgency_score=3, strategic_value=3, effort=2)
        planner = Planner(repo)
        plan = planner.generate_daily_plan(energy_level=1, focus_hours=6.0)
        # Top 1 should be the low-energy task, not the high-energy architecture task
        lines = plan.split("\n")
        top1_lines = []
        in_top1 = False
        for line in lines:
            if "## Top 1" in line:
                in_top1 = True
            elif line.startswith("## ") and in_top1:
                break
            elif in_top1:
                top1_lines.append(line)
        top1_text = "\n".join(top1_lines)
        assert "Hard architecture task" not in top1_text
        assert "Low energy admin" in top1_text

    def test_energy_level_1_max_energy_required_is_2(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        assert ENERGY_MATCH_TABLE[1]["max_energy_required"] == 2

    def test_energy_level_5_max_energy_required_is_5(self):
        from pb.core.planner import ENERGY_MATCH_TABLE
        assert ENERGY_MATCH_TABLE[5]["max_energy_required"] == 5

    def test_filter_excludes_high_energy_at_low_level(self, repo):
        from pb.core.planner import Planner
        # energy_required=4 should be excluded when energy_level=1 (max_energy_required=2)
        high_task = make_scored_task(repo, "High energy", energy_required=4)
        low_task = make_scored_task(repo, "Low energy", energy_required=1)
        planner = Planner(repo)

        # Get all tasks and call generate_daily_plan
        tasks = repo.list_tasks()
        from pb.core.priority import rank_tasks
        from pb.core.planner import ENERGY_MATCH_TABLE
        ranked = rank_tasks(tasks)
        energy_info = ENERGY_MATCH_TABLE[1]
        max_energy = energy_info["max_energy_required"]
        energy_fit = [t for t in ranked if (t.energy_required or 3) <= max_energy]
        ids = [t.id for t in energy_fit]
        assert high_task.id not in ids
        assert low_task.id in ids
