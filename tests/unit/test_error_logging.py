"""Tests for centralized daily error logging."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from pb.core.error_logging import format_logged_exception, log_error


def test_log_error_uses_daily_filename_and_line_numbers(temp_config):
    data_dir = Path(temp_config.storage.data_dir)
    today_name = f"{datetime.now(timezone.utc):%Y-%m-%d}.txt"

    first = log_error(
        event="unit.error",
        message="first failure",
        data_dir=data_dir,
        command="test",
        status=50,
    )
    second = log_error(
        event="unit.error",
        message="second failure",
        data_dir=data_dir,
        command="test",
        status=50,
    )

    assert first.path.name == today_name
    assert second.path == first.path
    assert first.line_number == 1
    assert second.line_number > first.line_number

    content = first.path.read_text(encoding="utf-8")
    assert "first failure" in content
    assert "second failure" in content
    assert "stdin:" in content
    assert "stdout:" in content


def test_format_logged_exception_shows_inline_only_for_short_messages(tmp_path):
    ref = log_error(
        event="unit.exception",
        message="x" * 400,
        data_dir=tmp_path,
        command="test",
    )

    assert format_logged_exception(RuntimeError("short message"), ref) == "short message"
    assert format_logged_exception(RuntimeError("x" * 400), ref) == (
        f"Error logged in Line {ref.line_number}, of {ref.filename}"
    )
