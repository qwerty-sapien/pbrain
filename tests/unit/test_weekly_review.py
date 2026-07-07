"""Unit tests for weekly review metrics computation (Plan 11).

Tests compute_weekly_metrics and generate_weekly_reflection in ReviewEngine.
"""

from datetime import datetime, timedelta

import pytest

from pb.core.review_engine import ReviewEngine
from pb.domain.enums import Horizon, SessionMode, TaskState
from pb.domain.models import DailyDebrief, Session, Task, TimeBlock


def _monday_of(dt: datetime) -> datetime:
    """Return the Monday at 00:00:00 UTC for the week containing dt."""
    days = dt.weekday()
    return (dt - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)


class TestComputeWeeklyMetrics:
    """Tests for ReviewEngine.compute_weekly_metrics."""

    def test_empty_db_returns_zero_hours(self, repo):
        """With no sessions, deep/shallow/buffer hours are all 0."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["deep_hours"] == 0
        assert metrics["shallow_hours"] == 0
        assert metrics["buffer_hours"] == 0

    def test_empty_db_returns_zero_rates(self, repo):
        """With no debriefs, top1_completion_rate and avg_top3_count are 0."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["top1_completion_rate"] == 0
        assert metrics["avg_top3_count"] == 0

    def test_empty_db_most_common_blocker_is_none(self, repo):
        """With no debriefs, most_common_blocker is 'none'."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["most_common_blocker"] == "none"

    def test_empty_db_avg_energy_zeros(self, repo):
        """With no debriefs, avg_energy dict has zero values."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["avg_energy"]["morning"] == 0
        assert metrics["avg_energy"]["midday"] == 0
        assert metrics["avg_energy"]["evening"] == 0

    def test_empty_db_deferred_tasks_empty(self, repo):
        """With no tasks, deferred_tasks is empty list."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["deferred_tasks"] == []

    def test_empty_db_planned_and_actual_minutes_zero(self, repo):
        """With no time blocks and sessions, planned and actual minutes are 0."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["planned_minutes"] == 0
        assert metrics["actual_minutes"] == 0

    def test_metrics_keys_present(self, repo):
        """compute_weekly_metrics returns all required keys."""
        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        required_keys = [
            "deep_hours", "shallow_hours", "buffer_hours",
            "top1_completion_rate", "avg_top3_count",
            "most_common_blocker", "avg_energy",
            "deferred_tasks", "planned_minutes", "actual_minutes",
        ]
        for key in required_keys:
            assert key in metrics, f"Missing key: {key}"

    def test_session_with_deep_task_counted_in_deep_hours(self, repo):
        """A session linked to a deep-work task increments deep_hours."""
        week_start = _monday_of(datetime.utcnow())
        task = Task(
            title="Deep task",
            state=TaskState.ACTIVE,
            work_type="deep",
        )
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=week_start + timedelta(hours=1),
            end_at=week_start + timedelta(hours=3),  # 120 min = 2h
            mode=SessionMode.FOCUS,
            intended_outcome="focus",
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["deep_hours"] == pytest.approx(2.0)
        assert metrics["shallow_hours"] == 0
        assert metrics["buffer_hours"] == 0

    def test_session_with_shallow_task_counted_in_shallow_hours(self, repo):
        """A session linked to a shallow-work task increments shallow_hours."""
        week_start = _monday_of(datetime.utcnow())
        task = Task(
            title="Admin task",
            state=TaskState.ACTIVE,
            work_type="admin",
        )
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=week_start + timedelta(hours=1),
            end_at=week_start + timedelta(hours=1, minutes=30),  # 30 min
            mode=SessionMode.FOCUS,
            intended_outcome="admin",
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["shallow_hours"] == pytest.approx(0.5)

    def test_session_with_recovery_task_counted_in_buffer_hours(self, repo):
        """A session linked to a recovery task increments buffer_hours."""
        week_start = _monday_of(datetime.utcnow())
        task = Task(
            title="Recovery",
            state=TaskState.ACTIVE,
            work_type="recovery",
        )
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=week_start + timedelta(hours=2),
            end_at=week_start + timedelta(hours=2, minutes=30),
            mode=SessionMode.FOCUS,
            intended_outcome="rest",
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["buffer_hours"] == pytest.approx(0.5)

    def test_top1_completion_rate_all_yes(self, repo):
        """When all debriefs have top1_completed='yes', rate is 100%."""
        week_start = _monday_of(datetime.utcnow())
        for i in range(3):
            day = week_start + timedelta(days=i)
            debrief = DailyDebrief(
                review_date=day.strftime("%Y-%m-%d"),
                top1_completed="yes",
                biggest_blocker="other",
                energy_morning=3,
                energy_midday=3,
                energy_evening=3,
            )
            repo.create_daily_debrief(debrief)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["top1_completion_rate"] == pytest.approx(100.0)

    def test_top1_completion_rate_partial(self, repo):
        """top1_completion_rate is correct for mixed yes/no debriefs."""
        week_start = _monday_of(datetime.utcnow())
        completions = ["yes", "no", "yes", "no"]
        for i, comp in enumerate(completions):
            day = week_start + timedelta(days=i)
            debrief = DailyDebrief(
                review_date=day.strftime("%Y-%m-%d"),
                top1_completed=comp,
                biggest_blocker="other",
            )
            repo.create_daily_debrief(debrief)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["top1_completion_rate"] == pytest.approx(50.0)

    def test_most_common_blocker_counted(self, repo):
        """most_common_blocker returns the most frequent blocker string."""
        week_start = _monday_of(datetime.utcnow())
        blockers = ["low_energy", "low_energy", "interruption"]
        for i, blocker in enumerate(blockers):
            day = week_start + timedelta(days=i)
            debrief = DailyDebrief(
                review_date=day.strftime("%Y-%m-%d"),
                biggest_blocker=blocker,
            )
            repo.create_daily_debrief(debrief)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["most_common_blocker"] == "low_energy"

    def test_avg_energy_computed_correctly(self, repo):
        """avg_energy morning/midday/evening values are averaged from debriefs."""
        week_start = _monday_of(datetime.utcnow())
        energy_data = [
            (2, 3, 4),
            (4, 3, 2),
        ]
        for i, (m, mid, e) in enumerate(energy_data):
            day = week_start + timedelta(days=i)
            debrief = DailyDebrief(
                review_date=day.strftime("%Y-%m-%d"),
                energy_morning=m,
                energy_midday=mid,
                energy_evening=e,
            )
            repo.create_daily_debrief(debrief)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["avg_energy"]["morning"] == pytest.approx(3.0)
        assert metrics["avg_energy"]["midday"] == pytest.approx(3.0)
        assert metrics["avg_energy"]["evening"] == pytest.approx(3.0)

    def test_deferred_tasks_includes_incomplete_tasks(self, repo):
        """deferred_tasks includes tasks not completed during the week."""
        week_start = _monday_of(datetime.utcnow())
        # Task created before week end, not done
        task = Task(
            title="Stuck task",
            state=TaskState.ACTIVE,
            created_at=week_start + timedelta(days=1),
        )
        repo.create_task(task)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        titles = [t.title for t in metrics["deferred_tasks"]]
        assert "Stuck task" in titles

    def test_planned_minutes_sums_time_blocks(self, repo):
        """planned_minutes sums duration_minutes across week's time blocks."""
        week_start = _monday_of(datetime.utcnow())
        task = Task(title="Planned task", state=TaskState.ACTIVE)
        repo.create_task(task)

        block1 = TimeBlock(
            task_id=task.id,
            start_time=week_start + timedelta(hours=9),
            duration_minutes=60,
        )
        block2 = TimeBlock(
            task_id=task.id,
            start_time=week_start + timedelta(days=1, hours=9),
            duration_minutes=90,
        )
        repo.create_time_block(block1)
        repo.create_time_block(block2)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["planned_minutes"] == 150

    def test_actual_minutes_sums_session_durations(self, repo):
        """actual_minutes sums minutes of completed sessions in the week."""
        week_start = _monday_of(datetime.utcnow())
        task = Task(title="Session task", state=TaskState.ACTIVE)
        repo.create_task(task)

        session = Session(
            task_id=task.id,
            start_at=week_start + timedelta(hours=10),
            end_at=week_start + timedelta(hours=11),  # 60 min
            mode=SessionMode.FOCUS,
            intended_outcome="work",
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics(week_start=week_start)
        assert metrics["actual_minutes"] == 60


class TestGenerateWeeklyReflection:
    """Tests for ReviewEngine.generate_weekly_reflection."""

    def _base_metrics(self):
        return {
            "deep_hours": 10.0,
            "shallow_hours": 5.0,
            "buffer_hours": 2.0,
            "top1_completion_rate": 80.0,
            "avg_top3_count": 2.5,
            "most_common_blocker": "low_energy",
            "avg_energy": {"morning": 4.0, "midday": 3.0, "evening": 2.0},
            "deferred_tasks": [],
            "planned_minutes": 900,
            "actual_minutes": 1020,
        }

    def test_reflection_contains_keep_section(self, repo):
        """generate_weekly_reflection output contains 'Keep' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Keep" in output

    def test_reflection_contains_cut_or_batch_section(self, repo):
        """generate_weekly_reflection output contains 'Cut or Batch' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Cut or Batch" in output

    def test_reflection_contains_rescope_section(self, repo):
        """generate_weekly_reflection output contains 'Rescope' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Rescope" in output

    def test_reflection_contains_schedule_next_week_section(self, repo):
        """generate_weekly_reflection output contains 'Schedule Next Week' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Schedule Next Week" in output

    def test_reflection_contains_drop_candidates_section(self, repo):
        """generate_weekly_reflection output contains 'Drop Candidates' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Drop Candidates" in output

    def test_reflection_contains_energy_pattern_section(self, repo):
        """generate_weekly_reflection output contains 'Energy Pattern' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Energy Pattern" in output

    def test_reflection_contains_capacity_recommendation_section(self, repo):
        """generate_weekly_reflection output contains 'Capacity Recommendation' section."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "## Capacity Recommendation" in output

    def test_reflection_shows_hours_summary(self, repo):
        """generate_weekly_reflection shows deep/shallow/buffer hours."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "Deep work" in output
        assert "10.0h" in output
        assert "Shallow work" in output
        assert "5.0h" in output

    def test_reflection_shows_top1_completion_rate(self, repo):
        """generate_weekly_reflection shows Top 1 completion rate."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_reflection(self._base_metrics())
        assert "80%" in output

    def test_reflection_energy_front_loaded_pattern(self, repo):
        """When morning energy > evening + 1, recommends front-loaded deep work."""
        engine = ReviewEngine(repo)
        metrics = self._base_metrics()
        metrics["avg_energy"] = {"morning": 4.5, "midday": 3.0, "evening": 2.0}
        output = engine.generate_weekly_reflection(metrics)
        assert "Front-loaded" in output

    def test_reflection_deferred_tasks_listed(self, repo):
        """When deferred_tasks present, they appear in Schedule and Drop sections."""
        task = Task(title="Overdue task", state=TaskState.ACTIVE)
        engine = ReviewEngine(repo)
        metrics = self._base_metrics()
        metrics["deferred_tasks"] = [task]
        output = engine.generate_weekly_reflection(metrics)
        assert "Overdue task" in output

    def test_reflection_under_delivering_capacity(self, repo):
        """When actual < 70% of planned, shows under-delivering message."""
        engine = ReviewEngine(repo)
        metrics = self._base_metrics()
        metrics["planned_minutes"] = 1000
        metrics["actual_minutes"] = 600  # 60%
        output = engine.generate_weekly_reflection(metrics)
        assert "Under-delivering" in output or "under-delivering" in output

    def test_reflection_on_track_capacity(self, repo):
        """When actual is 70-110% of planned, shows on-track message."""
        engine = ReviewEngine(repo)
        metrics = self._base_metrics()
        metrics["planned_minutes"] = 1000
        metrics["actual_minutes"] = 900  # 90%
        output = engine.generate_weekly_reflection(metrics)
        assert "On track" in output or "on track" in output

    def test_reflection_returns_string(self, repo):
        """generate_weekly_reflection always returns a string."""
        engine = ReviewEngine(repo)
        result = engine.generate_weekly_reflection(self._base_metrics())
        assert isinstance(result, str)
        assert len(result) > 0
