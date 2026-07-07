"""Unit tests for screen_time module."""

import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from datetime import datetime, timezone

from pb.core.screen_time import (
    check_screen_time_access,
    get_today_screen_time,
    format_app_name,
    ScreenTimeResult,
    AppUsage,
    MAC_EPOCH_OFFSET,
    KNOWLEDGE_DB_PATH,
    _datetime_to_mac_epoch,
    _get_today_bounds,
)


class TestCheckScreenTimeAccess:
    """Tests for check_screen_time_access function."""

    def test_returns_false_when_db_not_exists(self):
        """Should return False with message when knowledgeC.db doesn't exist."""
        with patch.object(Path, "exists", return_value=False):
            accessible, message = check_screen_time_access()
            assert accessible is False
            assert "not found" in message.lower()

    def test_returns_false_when_permission_denied(self):
        """Should return False with FDA message when permission denied."""
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", side_effect=PermissionError("access denied")):
                accessible, message = check_screen_time_access()
                assert accessible is False
                assert "Full Disk Access" in message

    def test_returns_true_when_accessible(self):
        """Should return True when file is readable."""
        mock_file = MagicMock()
        mock_file.__enter__ = MagicMock(return_value=mock_file)
        mock_file.__exit__ = MagicMock(return_value=False)
        mock_file.read = MagicMock(return_value=b"x")

        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", return_value=mock_file):
                accessible, message = check_screen_time_access()
                assert accessible is True
                assert message == ""

    def test_returns_false_on_generic_exception(self):
        """Should return False with error message on generic exception."""
        with patch.object(Path, "exists", return_value=True):
            with patch("builtins.open", side_effect=OSError("disk error")):
                accessible, message = check_screen_time_access()
                assert accessible is False
                assert "Cannot access Screen Time" in message
                assert "disk error" in message


class TestGetTodayScreenTime:
    """Tests for get_today_screen_time function."""

    def test_returns_unavailable_when_access_denied(self):
        """Should return unavailable result when FDA not granted."""
        with patch(
            "pb.core.screen_time.check_screen_time_access",
            return_value=(False, "Permission denied"),
        ):
            result = get_today_screen_time()
            assert result.available is False
            assert "Permission denied" in result.message
            assert result.apps == []

    def test_returns_empty_when_no_data(self):
        """Should return empty apps when query returns no results."""
        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = []
        mock_conn.execute.return_value = mock_cursor

        with patch(
            "pb.core.screen_time.check_screen_time_access",
            return_value=(True, ""),
        ):
            with patch("sqlite3.connect", return_value=mock_conn):
                result = get_today_screen_time()
                assert result.available is True
                assert result.apps == []

    def test_returns_top_apps_sorted_by_usage(self):
        """Should return top 5 apps sorted by usage minutes."""
        # Mock Row objects with dict-like access
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, key: {
            "app_id": "com.apple.Safari",
            "usage_minutes": 60.0,
        }[key]
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = lambda self, key: {
            "app_id": "com.apple.mail",
            "usage_minutes": 30.0,
        }[key]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [mock_row1, mock_row2]
        mock_conn.execute.return_value = mock_cursor

        with patch(
            "pb.core.screen_time.check_screen_time_access",
            return_value=(True, ""),
        ):
            with patch("sqlite3.connect", return_value=mock_conn):
                result = get_today_screen_time()
                assert result.available is True
                assert len(result.apps) == 2
                assert result.apps[0].app_id == "com.apple.Safari"
                assert result.apps[0].usage_minutes == 60.0
                assert result.apps[1].app_id == "com.apple.mail"
                assert result.apps[1].usage_minutes == 30.0

    def test_handles_sqlite_error(self):
        """Should return unavailable when SQLite error occurs."""
        import sqlite3

        with patch(
            "pb.core.screen_time.check_screen_time_access",
            return_value=(True, ""),
        ):
            with patch("sqlite3.connect", side_effect=sqlite3.Error("db locked")):
                result = get_today_screen_time()
                assert result.available is False
                assert "Error reading Screen Time" in result.message

    def test_filters_zero_usage(self):
        """Should filter out apps with zero or None usage."""
        mock_row1 = MagicMock()
        mock_row1.__getitem__ = lambda self, key: {
            "app_id": "com.apple.Safari",
            "usage_minutes": 60.0,
        }[key]
        mock_row2 = MagicMock()
        mock_row2.__getitem__ = lambda self, key: {
            "app_id": "com.empty.App",
            "usage_minutes": 0.0,
        }[key]
        mock_row3 = MagicMock()
        mock_row3.__getitem__ = lambda self, key: {
            "app_id": "com.null.App",
            "usage_minutes": None,
        }[key]

        mock_conn = MagicMock()
        mock_cursor = MagicMock()
        mock_cursor.fetchall.return_value = [mock_row1, mock_row2, mock_row3]
        mock_conn.execute.return_value = mock_cursor

        with patch(
            "pb.core.screen_time.check_screen_time_access",
            return_value=(True, ""),
        ):
            with patch("sqlite3.connect", return_value=mock_conn):
                result = get_today_screen_time()
                assert result.available is True
                # Should only have Safari (non-zero usage)
                assert len(result.apps) == 1
                assert result.apps[0].app_id == "com.apple.Safari"


class TestFormatAppName:
    """Tests for format_app_name function."""

    def test_extracts_last_component(self):
        """Should extract last component of bundle ID."""
        assert format_app_name("com.apple.Safari") == "Safari"
        assert format_app_name("com.google.Chrome") == "Chrome"

    def test_handles_simple_names(self):
        """Should handle names without dots."""
        assert format_app_name("Terminal") == "Terminal"

    def test_handles_complex_bundle_ids(self):
        """Should handle complex bundle IDs."""
        assert format_app_name("com.microsoft.VSCode") == "VSCode"
        assert format_app_name("com.apple.dt.Xcode") == "Xcode"

    def test_handles_empty_string(self):
        """Should handle empty string gracefully."""
        assert format_app_name("") == ""


class TestMacEpochConversion:
    """Tests for Mac epoch timestamp conversion."""

    def test_mac_epoch_offset_correct(self):
        """MAC_EPOCH_OFFSET should be seconds between Unix and Mac epoch."""
        # Mac epoch is 2001-01-01 00:00:00 UTC
        # Unix epoch is 1970-01-01 00:00:00 UTC
        # Difference is 31 years (with leap years)
        assert MAC_EPOCH_OFFSET == 978307200

    def test_datetime_to_mac_epoch(self):
        """Should convert datetime to Mac Absolute Time."""
        # Mac epoch itself should convert to ~0
        mac_epoch = datetime(2001, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        result = _datetime_to_mac_epoch(mac_epoch)
        assert abs(result) < 1  # Should be ~0

    def test_get_today_bounds_returns_tuple(self):
        """Should return two floats for midnight and now."""
        midnight, now = _get_today_bounds()
        assert isinstance(midnight, float)
        assert isinstance(now, float)
        assert midnight < now  # Midnight should be before now


class TestDataClasses:
    """Tests for data classes."""

    def test_app_usage_creation(self):
        """Should create AppUsage with required fields."""
        app = AppUsage(app_id="com.test.App", usage_minutes=42.5)
        assert app.app_id == "com.test.App"
        assert app.usage_minutes == 42.5

    def test_screen_time_result_creation(self):
        """Should create ScreenTimeResult with all fields."""
        result = ScreenTimeResult(
            available=True,
            message="",
            apps=[AppUsage(app_id="test", usage_minutes=10.0)],
        )
        assert result.available is True
        assert len(result.apps) == 1

    def test_screen_time_result_unavailable(self):
        """Should create unavailable ScreenTimeResult."""
        result = ScreenTimeResult(
            available=False,
            message="Full Disk Access required",
            apps=[],
        )
        assert result.available is False
        assert "Full Disk Access" in result.message
        assert result.apps == []


class TestKnowledgeDbPath:
    """Tests for KNOWLEDGE_DB_PATH constant."""

    def test_path_is_in_library(self):
        """Path should be in user's Library folder."""
        assert "Library" in str(KNOWLEDGE_DB_PATH)
        assert "Knowledge" in str(KNOWLEDGE_DB_PATH)
        assert "knowledgeC.db" in str(KNOWLEDGE_DB_PATH)
