"""Unit tests for AlignmentEngine."""

from datetime import datetime, timedelta

import pytest

from pb.core.alignment import AlignmentEngine, GoalBreakdown, _render_bar
from pb.domain.models import GoalArc, Session, Task, Track
from pb.storage.repository import Repository


class TestAlignmentRenderBar:
    """Tests for alignment module's _render_bar."""

    def test_render_bar_zero(self):
        result = _render_bar(0, 10)
        assert result == "          "

    def test_render_bar_full(self):
        result = _render_bar(100, 10)
        assert result == "----------"


class TestGoalBreakdown:
    """Tests for GoalBreakdown dataclass."""

    def test_goal_breakdown_creation(self):
        """GoalBreakdown dataclass works correctly."""
        gb = GoalBreakdown(
            goal_id="goal-1",
            title="Learn German",
            minutes=120,
            percent=40.0,
        )
        assert gb.goal_id == "goal-1"
        assert gb.title == "Learn German"
        assert gb.minutes == 120
        assert gb.percent == 40.0


class TestGetAlignment:
    """Tests for AlignmentEngine.get_alignment method."""

    def test_get_alignment_empty(self, repo):
        """Empty sessions returns empty breakdown."""
        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)
        assert breakdown == []

    def test_get_alignment_rolls_up_to_goal(self, repo):
        """Session time rolls up track -> goal arc."""
        # Create goal arc
        goal = GoalArc(id="goal-1", title="Become Fluent in German")
        repo.create_goal_arc(goal)

        # Create track linked to goal
        track = Track(
            id="track-1",
            name="German",
            linked_goal_arc_ids=["goal-1"],
        )
        repo.create_track(track)

        # Create task linked to track
        task = Task(
            id="task-1",
            title="Study German Grammar",
            linked_track_ids=["track-1"],
        )
        repo.create_task(task)

        # Create session
        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=60),
            end_at=now,
        )
        repo.create_session(session)

        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)

        assert len(breakdown) == 1
        assert breakdown[0].goal_id == "goal-1"
        assert breakdown[0].minutes == 60
        assert breakdown[0].percent == 100.0

    def test_get_alignment_unlinked_goes_to_other(self, repo):
        """Tasks without goal arc link go to Other."""
        # Create track NOT linked to any goal
        track = Track(
            id="track-1",
            name="Random Learning",
            linked_goal_arc_ids=[],  # No goals
        )
        repo.create_track(track)

        # Create task linked to track
        task = Task(
            id="task-1",
            title="Random task",
            linked_track_ids=["track-1"],
        )
        repo.create_task(task)

        # Create session
        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=now,
        )
        repo.create_session(session)

        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)

        assert len(breakdown) == 1
        assert breakdown[0].goal_id == "Other"
        assert breakdown[0].title == "Other"
        assert breakdown[0].minutes == 30

    def test_get_alignment_multiple_goals(self, repo):
        """Multiple goals are tracked separately."""
        # Create two goals
        goal1 = GoalArc(id="goal-1", title="German")
        goal2 = GoalArc(id="goal-2", title="Rust")
        repo.create_goal_arc(goal1)
        repo.create_goal_arc(goal2)

        # Create tracks
        track1 = Track(id="track-1", name="German", linked_goal_arc_ids=["goal-1"])
        track2 = Track(id="track-2", name="Rust", linked_goal_arc_ids=["goal-2"])
        repo.create_track(track1)
        repo.create_track(track2)

        # Create tasks
        task1 = Task(id="task-1", title="German", linked_track_ids=["track-1"])
        task2 = Task(id="task-2", title="Rust", linked_track_ids=["track-2"])
        repo.create_task(task1)
        repo.create_task(task2)

        # Create sessions
        now = datetime.utcnow()
        session1 = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=60),
            end_at=now - timedelta(minutes=30),
        )
        session2 = Session(
            id="session-2",
            task_id="task-2",
            start_at=now - timedelta(minutes=30),
            end_at=now,
        )
        repo.create_session(session1)
        repo.create_session(session2)

        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)

        assert len(breakdown) == 2
        # Should be sorted by minutes descending (both have 30 min)
        goal_ids = {b.goal_id for b in breakdown}
        assert goal_ids == {"goal-1", "goal-2"}

    def test_get_alignment_untracked_tasks_go_to_other(self, repo):
        """Tasks without any linked_track_ids go to Other."""
        # Create task with no tracks
        task = Task(
            id="task-1",
            title="Untracked task",
            linked_track_ids=[],
        )
        repo.create_task(task)

        # Create session
        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=now,
        )
        repo.create_session(session)

        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)

        assert len(breakdown) == 1
        assert breakdown[0].goal_id == "Other"
        assert breakdown[0].minutes == 30

    def test_get_alignment_ongoing_sessions_skipped(self, repo):
        """Ongoing sessions (no end_at) are not counted."""
        # Create goal and track
        goal = GoalArc(id="goal-1", title="German")
        repo.create_goal_arc(goal)

        track = Track(id="track-1", name="German", linked_goal_arc_ids=["goal-1"])
        repo.create_track(track)

        task = Task(id="task-1", title="German", linked_track_ids=["track-1"])
        repo.create_task(task)

        # Create ongoing session (no end_at)
        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=None,  # Ongoing
        )
        repo.create_session(session)

        engine = AlignmentEngine(repo)
        breakdown = engine.get_alignment(days=7)

        # Should be empty since ongoing sessions are skipped
        assert breakdown == []


class TestFormatAlignmentReport:
    """Tests for AlignmentEngine.format_alignment_report method."""

    def test_format_report_empty(self, repo):
        """Empty breakdown shows no sessions message."""
        engine = AlignmentEngine(repo)
        report = engine.format_alignment_report([], days=7)

        assert "# Alignment Report (Last 7 Days)" in report
        assert "No sessions recorded" in report

    def test_format_report_table_columns(self, repo):
        """Report contains correct table columns."""
        breakdown = [
            GoalBreakdown(goal_id="goal-1", title="German", minutes=60, percent=100.0),
        ]

        engine = AlignmentEngine(repo)
        report = engine.format_alignment_report(breakdown, days=7)

        assert "| GOAL" in report
        assert "| MINUTES" in report
        assert "| BAR" in report
        assert "German" in report
        assert "60" in report

    def test_format_report_total(self, repo):
        """Report shows total time at bottom."""
        breakdown = [
            GoalBreakdown(goal_id="goal-1", title="German", minutes=60, percent=50.0),
            GoalBreakdown(goal_id="goal-2", title="Rust", minutes=60, percent=50.0),
        ]

        engine = AlignmentEngine(repo)
        report = engine.format_alignment_report(breakdown, days=7)

        assert "**Total:** 120 minutes (2h 0m)" in report

    def test_format_report_custom_days(self, repo):
        """Report header reflects custom days parameter."""
        engine = AlignmentEngine(repo)
        report = engine.format_alignment_report([], days=30)

        assert "# Alignment Report (Last 30 Days)" in report

    def test_format_report_goal_arc_distribution_header(self, repo):
        """Report contains Goal Arc Distribution section."""
        breakdown = [
            GoalBreakdown(goal_id="goal-1", title="German", minutes=60, percent=100.0),
        ]

        engine = AlignmentEngine(repo)
        report = engine.format_alignment_report(breakdown, days=7)

        assert "## Goal Arc Distribution" in report
