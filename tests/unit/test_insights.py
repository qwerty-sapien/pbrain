"""Tests for InsightEngine (D-06 to D-08, ULOG-05)."""
import datetime
import sqlite3

import pytest

from pb.storage.database import init_db, set_db_path, get_connection, _migrate_usage_log


@pytest.fixture
def insight_db(tmp_path):
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    set_db_path(None)


def _insert_usage(conn, command, days_ago=0):
    ts = (datetime.datetime.utcnow() - datetime.timedelta(days=days_ago)).isoformat()
    conn.execute(
        "INSERT INTO usage_log (timestamp, command, exit_code, error) VALUES (?, ?, 0, '')",
        (ts, command),
    )
    conn.commit()


class TestReviewStaleness:
    def test_stale_review(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=1)
            # No "review" command at all
            engine = InsightEngine(conn)
            result = engine._review_staleness()
        assert result is not None
        assert "reviewed" in result.lower() or "review" in result.lower()

    def test_recent_review(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "review", days_ago=1)
            engine = InsightEngine(conn)
            result = engine._review_staleness()
        assert result is None


class TestIdleDetection:
    def test_idle_detected(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=5)
            engine = InsightEngine(conn)
            result = engine._idle_detection()
        assert result is not None
        assert "welcome back" in result.lower() or "last session" in result.lower()

    def test_not_idle(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=0)
            engine = InsightEngine(conn)
            result = engine._idle_detection()
        assert result is None


class TestStreakTracking:
    def test_streak_exists(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            for i in range(5):
                _insert_usage(conn, "start", days_ago=i)
            engine = InsightEngine(conn)
            result = engine._streak_tracking()
        assert result is not None
        assert "streak" in result.lower()

    def test_no_streak(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=0)
            # Gap at day 1
            _insert_usage(conn, "start", days_ago=3)
            engine = InsightEngine(conn)
            result = engine._streak_tracking()
        assert result is None  # Only 1-day streak

    def test_streak_milestone_icons(self):
        from pb.core.insights import streak_banner_text

        assert streak_banner_text(6) == "6-day streak!"
        assert streak_banner_text(7).endswith("⭐")
        assert streak_banner_text(14).endswith("🌟")
        assert streak_banner_text(21).endswith("✨")

    def test_streak_banner_style_progresses_to_gold_step(self):
        from pb.core.insights import streak_banner_style, streak_banner_style_for_message

        assert streak_banner_style(2) != streak_banner_style(3)
        assert streak_banner_style(14) != streak_banner_style(20)
        assert streak_banner_style(21) == "bold #ffd700"
        assert streak_banner_style_for_message("21-day streak! ✨") == "bold #ffd700"


class TestCommandPatternShift:
    def test_imbalanced_capture(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            for _ in range(10):
                _insert_usage(conn, "capture", days_ago=0)
            _insert_usage(conn, "start", days_ago=0)
            engine = InsightEngine(conn)
            result = engine._command_pattern_shift()
        assert result is not None

    def test_balanced_usage(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            for _ in range(3):
                _insert_usage(conn, "capture", days_ago=0)
                _insert_usage(conn, "start", days_ago=0)
            engine = InsightEngine(conn)
            result = engine._command_pattern_shift()
        assert result is None


class TestGetInsights:
    def test_max_count_respected(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            # No review (triggers staleness), old data (triggers idle)
            _insert_usage(conn, "start", days_ago=5)
            engine = InsightEngine(conn)
            insights = engine.get_insights(max_count=2)
        assert len(insights) <= 2

    def test_priority_order(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            # Trigger review staleness (highest priority)
            _insert_usage(conn, "start", days_ago=4)
            engine = InsightEngine(conn)
            insights = engine.get_insights(max_count=1)
        if insights:
            # First insight should be review staleness (highest priority)
            assert "review" in insights[0].lower()

    def test_empty_log_returns_empty(self, insight_db):
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            engine = InsightEngine(conn)
            insights = engine.get_insights(max_count=2)
        assert insights == []


class TestIdleDetectionSessionsAware:
    def test_dispatch_session_counts_as_activity(self, insight_db):
        """A recent dispatch session should suppress idle detection even if usage_log is old."""
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=5)
            conn.execute(
                "INSERT INTO dispatch_sessions (id, agent_id, status, context_json, created_at, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                ("ds1", "accountability", "complete", "{}",
                 datetime.datetime.utcnow().isoformat(),
                 datetime.datetime.utcnow().isoformat()),
            )
            conn.commit()
            engine = InsightEngine(conn)
            result = engine._idle_detection()
        assert result is None

    def test_learning_session_counts_as_activity(self, insight_db):
        """A recent learning session should suppress idle detection."""
        from pb.core.insights import InsightEngine
        with get_connection() as conn:
            _insert_usage(conn, "start", days_ago=5)
            conn.execute(
                "INSERT INTO sessions (id, task_id, start_at, mode, branch) "
                "VALUES (?, ?, ?, ?, ?)",
                ("s1", "fake_task_id", datetime.datetime.utcnow().isoformat(), "focus", "study"),
            )
            conn.commit()
            engine = InsightEngine(conn)
            result = engine._idle_detection()
        assert result is None
