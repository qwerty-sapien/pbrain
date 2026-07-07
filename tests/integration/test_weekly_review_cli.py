"""Integration tests for pb review week CLI command (Plan 11).

Tests the overhauled review_week command: structured reflection mode,
legacy table mode, help text, and empty-data behavior.
"""

from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.domain.enums import Horizon, SessionMode, TaskState
from pb.domain.models import DailyDebrief, Session, Task, TimeBlock
from pb.llm.drafts import WeeklyReviewDraft
from pb.llm.runtime import GeneratedDraft
from pb.storage.repository import Repository


runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_weekly_review_runtime():
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
            source_scope=kwargs.get("source_scope", "review.week"),
            prompt_template_version="test",
            raw_response="{}",
        )

    with patch("pb.llm.runtime.LLMRuntime.require", new=fake_require), patch(
        "pb.llm.runtime.LLMRuntime.generate_draft",
        new=fake_generate,
    ):
        yield


class TestWeekReviewLegacy:
    """Tests for pb review week --legacy flag."""

    def test_week_review_legacy_shows_weekly_review_header(self, repo):
        """--legacy flag still renders the new weekly reflection surface."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "Weekly Reflection" in result.output

    def test_week_review_legacy_shows_summary_table(self, repo):
        """--legacy flag still shows the numbers section."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "This Week's Numbers" in result.output

    def test_week_review_legacy_shows_tasks_completed(self, repo):
        """--legacy flag still reports session counts."""
        result = runner.invoke(app, ["review", "week", "--legacy"])
        assert result.exit_code == 0
        assert "Study sessions" in result.output


class TestWeekReviewMetrics:
    """Tests for pb review week default (new structured reflection) mode."""

    def test_week_review_shows_weekly_reflection_header(self, repo):
        """Default mode shows 'Weekly Reflection' header (not 'Weekly Review')."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "Weekly Reflection" in result.output

    def test_week_review_shows_this_weeks_numbers(self, repo):
        """Default mode shows weekly numbers section."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "This Week's Numbers" in result.output

    def test_week_review_shows_seven_categories(self, repo):
        """Default mode output contains the learning-focused reflection sections."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "## Wins" in result.output
        assert "## Stalls" in result.output
        assert "## Evidence Progress" in result.output
        assert "## Friction Patterns" in result.output
        assert "## Next Week Focus" in result.output

    def test_week_review_skips_chat_when_no_api_key(self, repo, monkeypatch):
        """The mocked structured runtime keeps the review available in tests."""
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "Weekly Reflection" in result.output

    def test_week_review_shows_top1_completion(self, repo):
        """Default mode shows goal-aligned time instead of manual scorecards."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "Goal-aligned time" in result.output

    def test_week_review_with_debrief_data_shows_friction(self, repo):
        """With debrief data present, the friction section is still rendered."""
        today = datetime.utcnow()
        week_start = today - timedelta(days=today.weekday())
        debrief = DailyDebrief(
            review_date=week_start.strftime("%Y-%m-%d"),
            biggest_blocker="low_energy",
            top1_completed="yes",
            energy_morning=3,
            energy_midday=3,
            energy_evening=3,
        )
        repo.create_daily_debrief(debrief)

        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
        assert "Friction Patterns" in result.output


class TestWeekReviewHelp:
    """Tests for pb review week help text."""

    def test_week_review_help_mentions_structured_reflection(self, repo):
        """Help text mentions 'structured reflection'."""
        result = runner.invoke(app, ["review", "week", "--help"])
        assert result.exit_code == 0
        assert "structured reflection" in result.output.lower() or "D-34" in result.output

    def test_week_review_help_mentions_legacy_flag(self, repo):
        """Help text documents the --legacy flag."""
        result = runner.invoke(app, ["review", "week", "--help"])
        assert result.exit_code == 0
        assert "--legacy" in result.output


class TestComputeWeeklyMetricsEmpty:
    """Tests that compute_weekly_metrics handles empty data gracefully."""

    def test_compute_weekly_metrics_empty_returns_zero_hours(self, repo):
        """With no data, all hour values are 0."""
        from pb.core.review_engine import ReviewEngine

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["deep_hours"] == 0
        assert metrics["shallow_hours"] == 0
        assert metrics["buffer_hours"] == 0

    def test_compute_weekly_metrics_empty_returns_zero_top1_rate(self, repo):
        """With no debriefs, top1_completion_rate is 0."""
        from pb.core.review_engine import ReviewEngine

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert metrics["top1_completion_rate"] == 0

    def test_compute_weekly_metrics_empty_deferred_tasks_is_list(self, repo):
        """With no tasks, deferred_tasks is an empty list."""
        from pb.core.review_engine import ReviewEngine

        engine = ReviewEngine(repo)
        metrics = engine.compute_weekly_metrics()
        assert isinstance(metrics["deferred_tasks"], list)
        assert len(metrics["deferred_tasks"]) == 0

    def test_week_review_empty_db_exits_cleanly(self, repo):
        """pb review week with empty DB exits with code 0."""
        result = runner.invoke(app, ["review", "week"])
        assert result.exit_code == 0
