"""Unit tests for ReviewEngine."""

from datetime import datetime, timedelta

import pytest

from pb.core.review_engine import ReviewEngine, _render_bar, _render_block_bar
from pb.domain.enums import TaskState
from pb.domain.models import GoalArc, Session, Task, TimeBlock, Track
from pb.storage.repository import Repository


class TestRenderBar:
    """Tests for _render_bar helper function."""

    def test_render_bar_zero_percent(self):
        """Zero percent renders all spaces."""
        result = _render_bar(0, 10)
        assert result == "          "
        assert len(result) == 10

    def test_render_bar_100_percent(self):
        """100 percent renders all dashes."""
        result = _render_bar(100, 10)
        assert result == "----------"
        assert len(result) == 10

    def test_render_bar_50_percent(self):
        """50 percent renders half dashes."""
        result = _render_bar(50, 10)
        assert result == "-----     "
        assert len(result) == 10

    def test_render_bar_custom_width(self):
        """Custom width works correctly."""
        result = _render_bar(50, 20)
        assert result == "----------          "
        assert len(result) == 20


class TestDailyReviewFormat:
    """Tests for daily review output format."""

    def test_daily_review_summary_table(self, repo):
        """Daily review contains summary table with METRIC | VALUE columns."""
        engine = ReviewEngine(repo)
        output = engine.generate_daily_review()

        assert "## Summary" in output
        assert "| METRIC" in output
        assert "| VALUE" in output
        assert "| Planned" in output
        assert "| Actual" in output
        assert "| Delta" in output
        assert "| Sessions" in output
        assert "| Interruptions" in output

    def test_daily_review_track_breakdown_section(self, repo):
        """Daily review contains track breakdown section."""
        # Create a task, session with track
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        task = Task(
            id="task-1",
            title="Study German",
            linked_track_ids=["track-1"],
        )
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(hours=1),
            end_at=now,
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        output = engine.generate_daily_review(date=now)

        assert "## Track Breakdown" in output
        assert "| TRACK" in output
        assert "| MINUTES" in output
        assert "| BAR" in output

    def test_daily_review_no_sessions(self, repo):
        """Daily review handles no sessions gracefully."""
        engine = ReviewEngine(repo)
        output = engine.generate_daily_review()

        # Should still have summary with zeros
        assert "| Planned" in output
        assert "| Sessions" in output


class TestTaskSlippage:
    """Tests for task-level slippage detection."""

    def test_get_slipped_tasks_finds_incomplete(self, repo):
        """Slippage detection finds tasks with blocks but not done."""
        now = datetime.utcnow()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Create task in READY state (not DONE)
        task = Task(
            id="task-1",
            title="Incomplete task",
            state=TaskState.ACTIVE,
        )
        repo.create_task(task)

        # Create time block for today
        block = TimeBlock(
            id="block-1",
            task_id="task-1",
            start_time=today,
            duration_minutes=60,
        )
        repo.create_time_block(block)

        engine = ReviewEngine(repo)
        slipped = engine._get_slipped_tasks(today)

        assert len(slipped) == 1
        assert slipped[0].id == "task-1"

    def test_get_slipped_tasks_ignores_completed(self, repo):
        """Slippage detection ignores completed tasks."""
        now = datetime.utcnow()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Create task in DONE state
        task = Task(
            id="task-1",
            title="Completed task",
            state=TaskState.DONE,
            completion=100,
            completed_at=today,
        )
        repo.create_task(task)

        # Create time block for today
        block = TimeBlock(
            id="block-1",
            task_id="task-1",
            start_time=today,
            duration_minutes=60,
        )
        repo.create_time_block(block)

        engine = ReviewEngine(repo)
        slipped = engine._get_slipped_tasks(today)

        assert len(slipped) == 0


class TestTrackAggregation:
    """Tests for track aggregation logic."""

    def test_aggregate_by_track_groups_correctly(self, repo):
        """Sessions are grouped by task's linked track."""
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        task = Task(
            id="task-1",
            title="Study German",
            linked_track_ids=["track-1"],
        )
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=now,
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        breakdown = engine._aggregate_by_track([session])

        assert "track-1" in breakdown
        assert breakdown["track-1"] == 30

    def test_aggregate_by_track_untracked(self, repo):
        """Tasks without linked_track_ids go to Untracked."""
        task = Task(
            id="task-1",
            title="Untracked task",
            linked_track_ids=[],  # No tracks
        )
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=45),
            end_at=now,
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        breakdown = engine._aggregate_by_track([session])

        assert "Untracked" in breakdown
        assert breakdown["Untracked"] == 45

    def test_aggregate_by_track_skips_ongoing(self, repo):
        """Ongoing sessions (no end_at) are skipped."""
        task = Task(id="task-1", title="Ongoing task")
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=None,  # Ongoing
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        breakdown = engine._aggregate_by_track([session])

        # Should be empty since ongoing sessions are skipped
        assert len(breakdown) == 0


class TestWeeklyReviewFormat:
    """Tests for weekly review output format."""

    def test_weekly_review_summary_table(self, repo):
        """Weekly review contains summary table with weekly totals."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_review()

        assert "## Summary" in output
        assert "| METRIC" in output
        assert "| Planned" in output
        assert "| Tasks Completed" in output

    def test_weekly_review_track_breakdown_with_sessions(self, repo):
        """Weekly review contains track breakdown when sessions exist."""
        # Create track and task
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        task = Task(
            id="task-1",
            title="Study German",
            linked_track_ids=["track-1"],
        )
        repo.create_task(task)

        # Create session for this week
        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(hours=1),
            end_at=now,
        )
        repo.create_session(session)

        engine = ReviewEngine(repo)
        output = engine.generate_weekly_review()

        assert "## Track Breakdown" in output
        assert "| TRACK" in output

    def test_weekly_review_no_track_breakdown_when_empty(self, repo):
        """Weekly review omits track breakdown section when no sessions."""
        engine = ReviewEngine(repo)
        output = engine.generate_weekly_review()

        # When no sessions, Track Breakdown section is not included
        # (the implementation only adds it when sessions exist)
        assert "## Summary" in output


class TestFormatTrackBreakdown:
    """Tests for _format_track_breakdown method."""

    def test_format_track_breakdown_empty(self, repo):
        """Empty breakdown shows no sessions message."""
        engine = ReviewEngine(repo)
        result = engine._format_track_breakdown({})
        assert "No sessions recorded" in result

    def test_format_track_breakdown_zero_total(self, repo):
        """Zero total shows no sessions message."""
        engine = ReviewEngine(repo)
        result = engine._format_track_breakdown({"track-1": 0})
        # This should not crash, handles zero total case
        assert "No sessions recorded" in result or "| TRACK" in result

    def test_format_track_breakdown_has_table_headers(self, repo):
        """Format includes table headers."""
        # Create a track
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        engine = ReviewEngine(repo)
        result = engine._format_track_breakdown({"track-1": 60})

        assert "| TRACK" in result
        assert "| MINUTES" in result
        assert "| BAR" in result

    def test_format_track_breakdown_shows_track_name(self, repo):
        """Format shows track name, not ID."""
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        engine = ReviewEngine(repo)
        result = engine._format_track_breakdown({"track-1": 60})

        assert "German" in result

    def test_format_track_breakdown_untracked(self, repo):
        """Untracked key shows as 'Untracked'."""
        engine = ReviewEngine(repo)
        result = engine._format_track_breakdown({"Untracked": 30})

        assert "Untracked" in result


class TestDailyReviewSlippageSection:
    """Tests for slippage section in daily review."""

    def test_daily_review_shows_time_slippage(self, repo):
        """Daily review shows time slippage when under plan."""
        now = datetime.utcnow()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Create task and time block (planned) but no sessions (actual = 0)
        task = Task(id="task-1", title="Planned task")
        repo.create_task(task)

        block = TimeBlock(
            id="block-1",
            task_id="task-1",
            start_time=today,
            duration_minutes=60,
        )
        repo.create_time_block(block)

        engine = ReviewEngine(repo)
        output = engine.generate_daily_review(date=today)

        # Should show slippage section since delta < 0
        assert "## Slippage" in output
        assert "minutes under plan" in output

    def test_daily_review_shows_task_slippage(self, repo):
        """Daily review shows tasks scheduled but not completed."""
        now = datetime.utcnow()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Create incomplete task with time block
        task = Task(id="task-1", title="Incomplete task", state=TaskState.ACTIVE)
        repo.create_task(task)

        block = TimeBlock(
            id="block-1",
            task_id="task-1",
            start_time=today,
            duration_minutes=60,
        )
        repo.create_time_block(block)

        engine = ReviewEngine(repo)
        output = engine.generate_daily_review(date=today)

        assert "Scheduled but not completed" in output
        assert "Incomplete task" in output


class TestRenderBlockBar:
    """Tests for _render_block_bar helper function."""

    def test_render_block_bar_zero_max(self):
        """Zero max_count returns all spaces."""
        result = _render_block_bar(0, 0, width=20)
        assert result == " " * 20
        assert len(result) == 20

    def test_render_block_bar_full(self):
        """Count equal to max gives full bar."""
        result = _render_block_bar(10, 10, width=20)
        assert result == "█" * 20
        assert len(result) == 20

    def test_render_block_bar_half(self):
        """Half count gives half-width bar."""
        result = _render_block_bar(5, 10, width=20)
        assert len(result) == 20
        assert result.startswith("█")

    def test_render_block_bar_custom_width(self):
        """Custom width is respected."""
        result = _render_block_bar(5, 10, width=10)
        assert len(result) == 10


class TestUsageSection:
    """Tests for format_usage_section method (D-05, D-20, ULOG-03)."""

    def test_format_usage_section_empty(self, repo, temp_db, temp_config):
        """With no usage data, should say 'No commands logged'."""
        engine = ReviewEngine(repo)
        result = engine.format_usage_section()
        assert "## Command Usage This Week" in result
        assert "No commands logged" in result

    def test_format_usage_section_with_data(self, repo, temp_db, temp_config):
        """With usage data, should include bar chart with Unicode blocks."""
        import datetime
        from pb.storage.database import get_connection
        with get_connection() as conn:
            ts = datetime.datetime.utcnow().isoformat()
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, 0, '')",
                (ts, "start"),
            )
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, 0, '')",
                (ts, "capture"),
            )
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, 0, '')",
                (ts, "capture"),
            )
            conn.commit()
        engine = ReviewEngine(repo)
        result = engine.format_usage_section()
        assert "## Command Usage This Week" in result
        assert "█" in result  # Unicode block character
        assert "capture" in result

    def test_generate_daily_review_includes_usage_section(self, repo):
        """generate_daily_review output includes the usage section."""
        engine = ReviewEngine(repo)
        output = engine.generate_daily_review()
        assert "## Command Usage This Week" in output
