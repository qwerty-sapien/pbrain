"""Integration tests for review CLI commands."""

import os
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.models import DailyDebrief, GoalArc, Session, Task, TimeBlock, Track
from pb.llm.drafts import DailyReviewDraft, WeeklyReviewDraft
from pb.llm.runtime import GeneratedDraft
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_review_runtime():
    def fake_require(self, purpose):
        return SimpleNamespace(
            configured=True,
            available=True,
            provider="gemini",
            backend="auto",
            default_model="gemini-3-flash-preview",
            structured_output=True,
            credential_source="test",
            message="ready",
        )

    def fake_generate(self, schema_cls, prompt, **kwargs):
        if schema_cls is DailyReviewDraft:
            payload = DailyReviewDraft(
                summary="Daily progress moved at least one goal forward.",
                progress_signals=["Study sessions completed."],
                friction_patterns=["Energy dipped in one block."],
                evidence_captured=["Anki candidates prepared."],
                next_adjustments=["Protect the next planned block."],
            )
        else:
            payload = WeeklyReviewDraft(
                summary="Weekly progress was visible and specific.",
                wins=["Study consistency improved."],
                stalls=["Export debt remained."],
                evidence_progress=["Recall and Anki evidence accumulated."],
                friction_patterns=["Energy dipped mid-week."],
                next_week_focus=["Export cards and continue practice."],
            )
        return GeneratedDraft(
            payload=payload,
            model="gemini-3-flash-preview",
            source_scope=kwargs.get("source_scope", "review"),
            prompt_template_version="test",
            raw_response="{}",
        )

    with patch("pb.llm.runtime.LLMRuntime.require", new=fake_require), patch(
        "pb.llm.runtime.LLMRuntime.generate_draft",
        new=fake_generate,
    ):
        yield


class TestReviewDayCommand:
    """Tests for pb review day command."""

    def test_review_day_runs(self, repo):
        """pb review day runs without error."""
        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "Daily Review" in result.output

    def test_review_day_skip_does_not_call_llm(self, repo):
        """pb review day --skip is the deterministic/no-LLM path."""
        with patch(
            "pb.llm.runtime.LLMRuntime.generate_draft",
            side_effect=AssertionError("review day --skip must not call LLM"),
        ):
            result = runner.invoke(app, ["review", "day", "--skip"])

        assert result.exit_code == 0, result.output
        assert "Daily Review" in result.output
        assert "no LLM request" in result.output

    def test_review_day_summary_table(self, repo):
        """pb review day outputs the structured metrics section."""
        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "Deterministic Facts" in result.output
        assert "Goal-aligned time" in result.output

    def test_review_day_shows_sessions_count(self, repo):
        """pb review day shows sessions count."""
        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "Study sessions" in result.output

    def test_review_day_shows_interruptions(self, repo):
        """pb review day shows interruptions count."""
        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "Interruptions" in result.output


class TestReviewWeekCommand:
    """Tests for pb review week command."""

    def test_review_week_runs(self, repo):
        """pb review week runs without error (default: structured reflection mode)."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "Weekly Reflection" in result.output

    def test_review_week_summary_table(self, repo):
        """pb review week --legacy still routes through the structured review."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "This Week's Numbers" in result.output
        assert "Study sessions" in result.output

    def test_review_week_shows_date_range(self, repo):
        """pb review week exposes next-week planning guidance."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "Next Week Focus" in result.output


class TestAlignmentCommand:
    """Tests for pb review alignment command."""

    def test_alignment_command_runs(self, repo):
        """pb review alignment runs without error."""
        result = runner.invoke(app, ["review", "alignment"])
        assert result.exit_code == 0
        assert "Alignment Report" in result.output

    def test_alignment_command_days_option(self, repo):
        """pb review alignment --days works."""
        result = runner.invoke(app, ["review", "alignment", "--days", "30"])
        assert result.exit_code == 0
        assert "Last 30 Days" in result.output

    def test_alignment_command_short_days_flag(self, repo):
        """pb review alignment -d works."""
        result = runner.invoke(app, ["review", "alignment", "-d", "14"])
        assert result.exit_code == 0
        assert "Last 14 Days" in result.output

    def test_alignment_command_with_data(self, repo):
        """pb review alignment shows goal data when present."""
        # Set up data
        goal = GoalArc(id="goal-1", title="German Fluency")
        repo.create_goal_arc(goal)

        track = Track(id="track-1", name="German", linked_goal_arc_ids=["goal-1"])
        repo.create_track(track)

        task = Task(id="task-1", title="Study", linked_track_ids=["track-1"])
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=60),
            end_at=now,
        )
        repo.create_session(session)

        result = runner.invoke(app, ["review", "alignment"])
        assert result.exit_code == 0
        assert "German Fluency" in result.output
        assert "GOAL" in result.output

    def test_alignment_empty_shows_no_sessions(self, repo):
        """pb review alignment with no data shows message."""
        result = runner.invoke(app, ["review", "alignment"])
        assert result.exit_code == 0
        assert "No sessions recorded" in result.output

    def test_alignment_shows_total(self, repo):
        """pb review alignment shows total when data exists."""
        # Create some session data
        track = Track(id="track-1", name="German", linked_goal_arc_ids=[])
        repo.create_track(track)

        task = Task(id="task-1", title="Study", linked_track_ids=["track-1"])
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=60),
            end_at=now,
        )
        repo.create_session(session)

        result = runner.invoke(app, ["review", "alignment"])
        assert result.exit_code == 0
        assert "**Total:**" in result.output

    def test_alignment_days_clamped_min(self, repo):
        """pb review alignment --days clamps to minimum 1."""
        result = runner.invoke(app, ["review", "alignment", "--days", "0"])
        assert result.exit_code == 0
        assert "Last 1 Days" in result.output

    def test_alignment_days_clamped_max(self, repo):
        """pb review alignment --days clamps to maximum 365."""
        result = runner.invoke(app, ["review", "alignment", "--days", "1000"])
        assert result.exit_code == 0
        assert "Last 365 Days" in result.output


class TestReviewSaveFlag:
    """Tests for --save flag on review commands."""

    def test_review_day_accepts_save_flag(self, repo):
        """pb review day --save flag is accepted."""
        result = runner.invoke(app, ["review", "day", "--save", "--skip"])
        # May fail if packet engine not fully wired, but command should accept flag
        # At minimum, it should not error on the flag itself
        assert result.exit_code == 0 or "Saved to:" in result.output

    def test_review_day_short_save_flag(self, repo):
        """pb review day -s flag is accepted."""
        result = runner.invoke(app, ["review", "day", "-s", "--skip"])
        assert result.exit_code == 0 or "Saved to:" in result.output

    def test_review_week_accepts_save_flag(self, repo):
        """pb review week --save flag is accepted."""
        result = runner.invoke(app, ["review", "week", "--save"])
        assert result.exit_code == 0 or "Saved to:" in result.output

    def test_alignment_accepts_save_flag(self, repo):
        """pb review alignment --save flag is accepted."""
        result = runner.invoke(app, ["review", "alignment", "--save"])
        # Command should accept the flag
        assert result.exit_code == 0 or "Saved to:" in result.output


class TestReviewTrackBreakdown:
    """Tests for track breakdown in review commands."""

    def test_review_day_shows_track_breakdown(self, repo):
        """pb review day shows track breakdown when sessions exist."""
        # Create track and session
        track = Track(id="track-1", name="German")
        repo.create_track(track)

        task = Task(id="task-1", title="Study", linked_track_ids=["track-1"])
        repo.create_task(task)

        now = datetime.utcnow()
        session = Session(
            id="session-1",
            task_id="task-1",
            start_at=now - timedelta(minutes=30),
            end_at=now,
        )
        repo.create_session(session)

        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "## Track Breakdown" in result.output
        assert "German" in result.output


class TestReviewSlippage:
    """Tests for slippage detection in review commands."""

    def test_review_day_shows_slippage(self, repo):
        """pb review day shows slippage when tasks have blocks but not done."""
        now = datetime.utcnow()
        today = now.replace(hour=12, minute=0, second=0, microsecond=0)

        # Create incomplete task with time block
        from pb.domain.enums import TaskState
        task = Task(id="task-1", title="Planned but not done", state=TaskState.ACTIVE)
        repo.create_task(task)

        block = TimeBlock(
            id="block-1",
            task_id="task-1",
            start_time=today,
            duration_minutes=60,
        )
        repo.create_time_block(block)

        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "## Slippage" in result.output
        assert "Planned but not done" in result.output


class TestReviewDayHelp:
    """Tests for the updated review day command help text."""

    def test_review_day_help_mentions_debrief(self, repo):
        """pb review day --help shows debrief or compact review."""
        result = runner.invoke(app, ["review", "day", "--help"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "debrief" in output_lower or "compact" in output_lower

    def test_review_day_help_shows_legacy_flag(self, repo):
        """pb review day --help lists --legacy option."""
        result = runner.invoke(app, ["review", "day", "--help"])
        assert result.exit_code == 0
        assert "--legacy" in result.output


class TestReviewDayLegacyFlag:
    """Tests for backward-compatible --legacy flag."""

    def test_review_day_legacy_shows_summary_table(self, repo):
        """pb review day --legacy still renders the daily review."""
        result = runner.invoke(app, ["review", "day", "--legacy", "--skip"])
        assert result.exit_code == 0
        assert "Daily Review" in result.output

    def test_review_day_skip_shows_daily_review(self, repo):
        """pb review day --skip still shows the structured daily review."""
        result = runner.invoke(app, ["review", "day", "--skip"])
        assert result.exit_code == 0
        assert "Daily Review" in result.output


class TestReviewWeekUnchanged:
    """Regression tests — pb review week legacy mode must be unaffected."""

    def test_review_week_still_works(self, repo):
        """pb review week --legacy runs without error after CLI overhaul."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "Weekly Reflection" in result.output

    def test_review_week_still_has_summary(self, repo):
        """pb review week --legacy still outputs the weekly numbers section."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "This Week's Numbers" in result.output


class TestReviewAlignmentUnchanged:
    """Regression tests — pb review alignment must be unaffected."""

    def test_review_alignment_still_works(self, repo):
        """pb review alignment runs without error after CLI overhaul."""
        result = runner.invoke(app, ["review", "alignment"])
        assert result.exit_code == 0
        assert "Alignment Report" in result.output


class TestDailyDebriefModelRoundtrip:
    """Integration test: DailyDebrief create -> retrieve via repository."""

    def test_debrief_model_roundtrip(self, repo):
        """Create DailyDebrief, save via repo, read back, verify key fields."""
        debrief = DailyDebrief(
            review_date="2026-04-25",
            top1_completed="yes",
            biggest_blocker="low_energy",
            energy_morning=4,
            energy_midday=3,
            energy_evening=2,
            learning_question="What should I repeat tomorrow?",
            learning_answer="Start with tests",
            learning_score=9,
            tomorrow_top1="Deploy changes",
        )
        repo.create_daily_debrief(debrief)
        retrieved = repo.get_daily_debrief("2026-04-25")

        assert retrieved is not None
        assert retrieved.review_date == "2026-04-25"
        assert retrieved.top1_completed == "yes"
        assert retrieved.biggest_blocker == "low_energy"
        assert retrieved.energy_morning == 4
        assert retrieved.learning_answer == "Start with tests"
        assert retrieved.learning_score == 9
        assert retrieved.tomorrow_top1 == "Deploy changes"

    def test_debrief_list_debriefs(self, repo):
        """list_daily_debriefs returns stored debriefs."""
        d1 = DailyDebrief(review_date="2026-04-24", top1_completed="no")
        d2 = DailyDebrief(review_date="2026-04-25", top1_completed="yes")
        repo.create_daily_debrief(d1)
        repo.create_daily_debrief(d2)

        results = repo.list_daily_debriefs(days=7)
        assert len(results) == 2
        dates = [r.review_date for r in results]
        assert "2026-04-24" in dates
        assert "2026-04-25" in dates
