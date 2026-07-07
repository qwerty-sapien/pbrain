"""Tests for usage_log table migration, log_usage(), and get_command_counts() (Phase 9, ULOG-01, ULOG-02)."""

import sqlite3
import datetime
from pathlib import Path

import pytest

from pb.storage.database import (
    init_db,
    get_connection,
    set_db_path,
    log_usage,
    get_command_counts,
    _migrate_usage_log,
)


@pytest.fixture
def tmp_db(tmp_path):
    """Create a temporary database for testing."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    set_db_path(None)  # Reset


class TestUsageLogMigration:
    """Test usage_log table creation (D-03)."""

    def test_migrate_creates_table(self, tmp_db):
        with get_connection() as conn:
            tables = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()}
        assert "usage_log" in tables

    def test_migrate_creates_indexes(self, tmp_db):
        with get_connection() as conn:
            indexes = {row[0] for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='index'"
            ).fetchall()}
        assert "idx_usage_log_timestamp" in indexes
        assert "idx_usage_log_command" in indexes

    def test_migrate_idempotent(self, tmp_db):
        with get_connection() as conn:
            _migrate_usage_log(conn)  # Second call should not error
            _migrate_usage_log(conn)  # Third call should not error


class TestLogUsage:
    """Test log_usage write + prune (D-02, D-04, ULOG-01, ULOG-02)."""

    def test_log_usage_writes_row(self, tmp_db):
        log_usage("start", 0, "")
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM usage_log").fetchall()
        assert len(rows) == 1
        assert rows[0]["command"] == "start"
        assert rows[0]["exit_code"] == 0

    def test_log_usage_writes_error(self, tmp_db):
        log_usage("review", 1, "Config not found")
        with get_connection() as conn:
            row = conn.execute("SELECT * FROM usage_log").fetchone()
        assert row["error"] == "Config not found"
        assert row["exit_code"] == 1

    def test_log_usage_prunes_old_entries(self, tmp_db):
        # Insert an old entry manually
        old_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=31)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, ?, ?)",
                (old_ts, "old_command", 0, ""),
            )
            conn.commit()
        # Log a new entry -- should prune the old one
        log_usage("new_command", 0, "")
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM usage_log").fetchall()
        commands = [r["command"] for r in rows]
        assert "old_command" not in commands
        assert "new_command" in commands

    def test_log_usage_keeps_recent_entries(self, tmp_db):
        # Insert a recent entry
        recent_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=5)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, ?, ?)",
                (recent_ts, "recent_command", 0, ""),
            )
            conn.commit()
        log_usage("new_command", 0, "")
        with get_connection() as conn:
            rows = conn.execute("SELECT * FROM usage_log").fetchall()
        assert len(rows) == 2  # Both kept

    def test_log_usage_never_raises(self, tmp_db, monkeypatch):
        # Force get_connection to raise
        def broken_conn():
            raise sqlite3.Error("boom")
        monkeypatch.setattr("pb.storage.database.get_connection", broken_conn)
        # Should not raise
        log_usage("test", 0, "")


class TestGetCommandCounts:
    """Test command count query for review section (D-05)."""

    def test_returns_counts(self, tmp_db):
        log_usage("start", 0, "")
        log_usage("start", 0, "")
        log_usage("review", 0, "")
        with get_connection() as conn:
            counts = get_command_counts(conn, days=7)
        assert counts["start"] == 2
        assert counts["review"] == 1

    def test_excludes_old_entries(self, tmp_db):
        old_ts = (datetime.datetime.utcnow() - datetime.timedelta(days=10)).isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, ?, ?)",
                (old_ts, "old", 0, ""),
            )
            conn.commit()
        log_usage("new", 0, "")
        with get_connection() as conn:
            counts = get_command_counts(conn, days=7)
        assert "old" not in counts
        assert counts["new"] == 1

    def test_returns_empty_dict_when_no_rows(self, tmp_db):
        with get_connection() as conn:
            counts = get_command_counts(conn, days=7)
        assert counts == {}
