"""Tests for display helpers."""

from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from pb.cli.display import (
    USER_TZ,
    _build_clock_panel,
    format_date_local,
    format_datetime_local,
    live_session_clock,
)


class TestFormatDatetimeLocal:
    """Tests for format_datetime_local function."""

    def test_utc_to_local_time_only(self):
        """UTC 06:30 converts to 14:30 in UTC+8."""
        dt = datetime(2026, 4, 23, 6, 30, tzinfo=timezone.utc)
        result = format_datetime_local(dt)
        assert result == "14:30"

    def test_utc_to_local_with_date(self):
        """UTC 06:30 converts to 2026-04-23 14:30 in UTC+8."""
        dt = datetime(2026, 4, 23, 6, 30, tzinfo=timezone.utc)
        result = format_datetime_local(dt, include_date=True)
        assert result == "2026-04-23 14:30"

    def test_none_time_only(self):
        """None returns placeholder for time-only format."""
        result = format_datetime_local(None)
        assert result == "--:--"

    def test_none_with_date(self):
        """None returns placeholder for date+time format."""
        result = format_datetime_local(None, include_date=True)
        assert result == "---- -- --:--"

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetime (no tzinfo) is treated as UTC."""
        dt = datetime(2026, 4, 23, 6, 30)  # No tzinfo
        result = format_datetime_local(dt)
        assert result == "14:30"

    def test_midnight_crossing(self):
        """Time that crosses midnight when converted to local."""
        # 20:00 UTC = 04:00 next day in UTC+8
        dt = datetime(2026, 4, 23, 20, 0, tzinfo=timezone.utc)
        result = format_datetime_local(dt, include_date=True)
        assert result == "2026-04-24 04:00"


class TestFormatDateLocal:
    """Tests for format_date_local function."""

    def test_date_conversion(self):
        """Date converts correctly to local timezone."""
        dt = datetime(2026, 4, 23, 10, 0, tzinfo=timezone.utc)
        result = format_date_local(dt)
        assert result == "2026-04-23"

    def test_date_crosses_midnight(self):
        """Date that crosses midnight shows next day."""
        # 20:00 UTC on Apr 23 = 04:00 Apr 24 in UTC+8
        dt = datetime(2026, 4, 23, 20, 0, tzinfo=timezone.utc)
        result = format_date_local(dt)
        assert result == "2026-04-24"

    def test_none_returns_placeholder(self):
        """None returns placeholder."""
        result = format_date_local(None)
        assert result == "----"

    def test_naive_datetime_treated_as_utc(self):
        """Naive datetime is treated as UTC."""
        dt = datetime(2026, 4, 23, 10, 0)  # No tzinfo
        result = format_date_local(dt)
        assert result == "2026-04-23"


def test_clock_hint_is_styled_text_not_literal_markup():
    panel = _build_clock_panel("Focus block", elapsed_secs=120)

    assert "[dim]" not in panel.renderable.plain
    assert "Ctrl+C or q to hide" in panel.renderable.plain


def test_live_session_clock_exits_on_q(monkeypatch):
    session = SimpleNamespace(start_at=datetime.utcnow())
    task = SimpleNamespace(title="Focus block")
    calls = {"count": 0}

    def fake_read_clock_action(timeout: float):
        calls["count"] += 1
        return "q"

    monkeypatch.setattr("pb.cli.display._read_clock_action", fake_read_clock_action)

    live_session_clock(session, task, duration_minutes=None)

    assert calls["count"] == 1
