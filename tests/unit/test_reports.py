"""Tests for reports.py: sparkline utility and ReportEngine.

TDD RED phase - tests written before implementation.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from pb.core.reports import sparkline, ReportEngine, SPARK_CHARS


# ---------------------------------------------------------------------------
# Sparkline tests
# ---------------------------------------------------------------------------

def test_sparkline_empty():
    """sparkline([]) returns empty string."""
    assert sparkline([]) == ""


def test_sparkline_all_same_max():
    """sparkline([5, 5, 5]) returns max chars."""
    result = sparkline([5, 5, 5])
    assert result == SPARK_CHARS[-1] * 3


def test_sparkline_length_matches_input():
    """Output length equals input length."""
    values = [1, 3, 5, 7, 3, 1]
    result = sparkline(values)
    assert len(result) == len(values)


def test_sparkline_uses_unicode_block_chars():
    """All chars in output are from SPARK_CHARS."""
    values = [1, 3, 5, 7, 3, 1]
    result = sparkline(values)
    for ch in result:
        assert ch in SPARK_CHARS


def test_sparkline_increasing_values():
    """Increasing values produce increasing char indices."""
    values = [1.0, 2.0, 3.0, 4.0]
    result = sparkline(values)
    # Each char index should be >= previous
    indices = [SPARK_CHARS.index(ch) for ch in result]
    for i in range(1, len(indices)):
        assert indices[i] >= indices[i - 1]


def test_sparkline_first_char_min_last_char_max():
    """With strictly increasing values, first is min char, last is max char."""
    values = [1.0, 5.0]
    result = sparkline(values)
    assert result[0] == SPARK_CHARS[0]
    assert result[-1] == SPARK_CHARS[-1]


def test_sparkline_single_value_same_as_all_same():
    """Single value treated as all-same."""
    result = sparkline([42.0])
    assert result == SPARK_CHARS[-1]


def test_sparkline_float_inputs():
    """Float inputs work correctly."""
    result = sparkline([0.5, 1.5, 2.5])
    assert len(result) == 3


# ---------------------------------------------------------------------------
# ReportEngine tests - day report
# ---------------------------------------------------------------------------

def _make_repo():
    """Create a mock repository."""
    repo = MagicMock()
    repo.list_sessions_in_range.return_value = []
    repo.list_time_blocks_for_date.return_value = []
    repo.list_tasks.return_value = []
    repo.list_daily_debriefs.return_value = []
    repo.list_goal_arcs.return_value = []
    repo.list_tracks.return_value = []
    repo.get_task.return_value = None
    return repo


def test_generate_day_report_header():
    """Day report starts with '# Day Report'."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_day_report(datetime(2026, 4, 25))
    assert "# Day Report" in result
    assert "2026-04-25" in result


def test_generate_day_report_shows_sessions_and_completion():
    """Day report includes Planned, Actual, Sessions, Completed."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_day_report(datetime(2026, 4, 25))
    assert "Planned" in result
    assert "Actual" in result
    assert "Sessions" in result
    assert "Completed" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - week report
# ---------------------------------------------------------------------------

def test_generate_week_report_header():
    """Week report starts with '# Week Report'."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_week_report()
    assert "# Week Report" in result


def test_generate_week_report_has_sparkline():
    """Week report includes daily sparkline."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_week_report()
    assert "Daily" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - energy report
# ---------------------------------------------------------------------------

def test_generate_energy_report_no_data():
    """Energy report with no debriefs shows helpful message."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_energy_report()
    assert "No daily debriefs found" in result


def test_generate_energy_report_with_data():
    """Energy report with debriefs shows sparklines."""
    repo = _make_repo()
    debrief = MagicMock()
    debrief.energy_morning = 3
    debrief.energy_midday = 4
    debrief.energy_evening = 2
    debrief.energy_task_match = "yes"
    repo.list_daily_debriefs.return_value = [debrief]

    engine = ReportEngine(repo)
    result = engine.generate_energy_report()
    assert "# Energy Report" in result
    assert "Morning" in result
    assert "Midday" in result
    assert "Evening" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - friction report
# ---------------------------------------------------------------------------

def test_generate_friction_report_no_data():
    """Friction report with no debriefs shows helpful message."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_friction_report()
    assert "No daily debriefs found" in result


def test_generate_friction_report_with_data():
    """Friction report shows frequency counts."""
    repo = _make_repo()
    d1 = MagicMock()
    d1.biggest_blocker = "low_energy"
    d2 = MagicMock()
    d2.biggest_blocker = "interruption"
    d3 = MagicMock()
    d3.biggest_blocker = "low_energy"
    repo.list_daily_debriefs.return_value = [d1, d2, d3]

    engine = ReportEngine(repo)
    result = engine.generate_friction_report()
    assert "# Friction Report" in result
    assert "Frequency" in result
    assert "low energy" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - priority report
# ---------------------------------------------------------------------------

def test_generate_priority_report_no_tasks():
    """Priority report with no tasks still shows structure."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_priority_report()
    assert "# Priority Report" in result
    assert "Eisenhower Distribution" in result
    assert "Priority Action Distribution" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - month report
# ---------------------------------------------------------------------------

def test_generate_month_report_header():
    """Month report has correct sections."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_month_report(datetime(2026, 4, 25))
    assert "# Month Report" in result
    assert "Month-over-Month" in result
    assert "Weekly Breakdown" in result
    assert "Daily Trend" in result


def test_generate_month_report_mom_comparison():
    """Month report includes this month vs last month."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_month_report(datetime(2026, 4, 25))
    assert "This month" in result
    assert "Last month" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - track report
# ---------------------------------------------------------------------------

def test_generate_track_report_not_found():
    """Track report returns helpful message when track doesn't exist."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_track_report("nonexistent")
    assert "Track not found" in result


def test_generate_track_report_found():
    """Track report shows stats for existing track."""
    repo = _make_repo()
    track = MagicMock()
    track.name = "MyTrack"
    track.id = "track-id-1"
    repo.list_tracks.return_value = [track]
    repo.list_sessions_in_range.return_value = []
    repo.list_tasks.return_value = []

    engine = ReportEngine(repo)
    result = engine.generate_track_report("mytrack")
    assert "# Track Report: MyTrack" in result
    assert "Total hours" in result
    assert "Sessions" in result
    assert "Tasks" in result
    assert "Weekly trend" in result


# ---------------------------------------------------------------------------
# ReportEngine tests - goals report
# ---------------------------------------------------------------------------

def test_generate_goals_report_no_goals():
    """Goals report with no goals shows helpful message."""
    repo = _make_repo()
    engine = ReportEngine(repo)
    result = engine.generate_goals_report()
    assert "No goals defined" in result


def test_generate_goals_report_with_goal():
    """Goals report includes goal title, status, horizon."""
    repo = _make_repo()
    goal = MagicMock()
    goal.title = "Ship MVP"
    goal.status = "active"
    goal.horizon = MagicMock()
    goal.horizon.value = "six_month"
    goal.target_value = None
    goal.metric_type = None
    goal.target_date = None
    goal.id = "goal-1"
    repo.list_goal_arcs.return_value = [goal]

    engine = ReportEngine(repo)
    result = engine.generate_goals_report()
    assert "Ship MVP" in result
    assert "active" in result
    assert "six_month" in result
