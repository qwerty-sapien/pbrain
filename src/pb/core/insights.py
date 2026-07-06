# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Habit insight engine for pb shell launch (D-06 to D-08)."""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime, timedelta
from typing import Optional


_STREAK_COLOR_STOPS: tuple[tuple[int, tuple[int, int, int]], ...] = (
    (1, (217, 221, 227)),  # grey-white
    (3, (154, 226, 127)),  # light green
    (7, (16, 185, 129)),   # emerald green
    (10, (190, 242, 64)),  # yellow-green
    (14, (255, 127, 64)),  # yellow-vermillion
    (20, (255, 192, 77)),  # lighter yellow-vermillion before gold step
)
_STREAK_GOLD = "#ffd700"
_STREAK_RE = re.compile(r"(?P<days>\d+)-day streak")


def streak_banner_text(streak: int) -> str:
    """Return the user-facing streak banner text."""
    icon = ""
    if streak >= 21:
        icon = " ✨"
    elif streak >= 14:
        icon = " 🌟"
    elif streak >= 7:
        icon = " ⭐"
    return f"{streak}-day streak!{icon}"


def streak_banner_style(streak: int) -> str:
    """Return a Rich style for the streak banner color progression."""
    if streak >= 21:
        return f"bold {_STREAK_GOLD}"
    if streak <= _STREAK_COLOR_STOPS[0][0]:
        return f"bold {_rgb_to_hex(_STREAK_COLOR_STOPS[0][1])}"
    for (left_day, left_rgb), (right_day, right_rgb) in zip(
        _STREAK_COLOR_STOPS,
        _STREAK_COLOR_STOPS[1:],
    ):
        if left_day <= streak <= right_day:
            span = max(1, right_day - left_day)
            ratio = (streak - left_day) / span
            return f"bold {_rgb_to_hex(_interpolate_rgb(left_rgb, right_rgb, ratio))}"
    return f"bold {_rgb_to_hex(_STREAK_COLOR_STOPS[-1][1])}"


def streak_banner_style_for_message(message: str) -> str | None:
    """Return the streak style for an insight message, if it is a streak."""
    match = _STREAK_RE.search(message or "")
    if not match:
        return None
    return streak_banner_style(int(match.group("days")))


def _interpolate_rgb(
    left: tuple[int, int, int],
    right: tuple[int, int, int],
    ratio: float,
) -> tuple[int, int, int]:
    ratio = max(0.0, min(1.0, ratio))
    return tuple(round(l + (r - l) * ratio) for l, r in zip(left, right))


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return "#{:02x}{:02x}{:02x}".format(*rgb)


class InsightEngine:
    """Derives at most N habit insights from the usage_log table (D-07)."""

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn

    def get_insights(self, max_count: int = 2) -> list[str]:
        """Return up to max_count insight strings.

        Priority order per D-07:
        1. Review staleness
        2. Idle detection
        3. Command pattern shifts
        4. Streak tracking
        """
        insights: list[str] = []
        for check in [
            self._review_staleness,
            self._idle_detection,
            self._command_pattern_shift,
            self._streak_tracking,
        ]:
            if len(insights) >= max_count:
                break
            result = check()
            if result:
                insights.append(result)
        return insights

    def _review_staleness(self) -> Optional[str]:
        """'You haven't reviewed in N days' if no review command in 3+ days."""
        try:
            row = self.conn.execute(
                "SELECT timestamp FROM usage_log WHERE command IN ('review', 'review day') "
                "ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row is None:
                # Check if there's any data at all
                any_row = self.conn.execute("SELECT 1 FROM usage_log LIMIT 1").fetchone()
                if any_row is None:
                    return None  # No data yet
                return "You haven't run a review yet"
            last = datetime.fromisoformat(row["timestamp"])
            days = (datetime.utcnow() - last).days
            if days >= 3:
                return f"You haven't reviewed in {days} days"
        except Exception:
            pass
        return None

    def _idle_detection(self) -> Optional[str]:
        """'Welcome back! Last session was N days ago' if no activity for 2+ days.

        Checks usage_log, sessions, and dispatch_sessions for the most recent
        timestamp across all three tables.
        """
        try:
            candidates: list[datetime] = []
            row = self.conn.execute(
                "SELECT timestamp FROM usage_log ORDER BY timestamp DESC LIMIT 1"
            ).fetchone()
            if row:
                candidates.append(datetime.fromisoformat(row["timestamp"]))

            try:
                row2 = self.conn.execute(
                    "SELECT start_at FROM sessions ORDER BY start_at DESC LIMIT 1"
                ).fetchone()
                if row2:
                    candidates.append(datetime.fromisoformat(row2["start_at"]))
            except Exception:
                pass

            try:
                row3 = self.conn.execute(
                    "SELECT updated_at FROM dispatch_sessions ORDER BY updated_at DESC LIMIT 1"
                ).fetchone()
                if row3:
                    candidates.append(datetime.fromisoformat(row3["updated_at"]))
            except Exception:
                pass

            if not candidates:
                return None
            last = max(candidates)
            days = (datetime.utcnow() - last).days
            if days >= 2:
                return f"Welcome back! Last session was {days} days ago"
        except Exception:
            pass
        return None

    def _command_pattern_shift(self) -> Optional[str]:
        """Detect capture-heavy but execute-light patterns in last 7 days."""
        try:
            cutoff = (datetime.utcnow() - timedelta(days=7)).isoformat()
            rows = self.conn.execute(
                "SELECT command, COUNT(*) as cnt FROM usage_log "
                "WHERE timestamp >= ? GROUP BY command",
                (cutoff,),
            ).fetchall()
            counts = {r["command"]: r["cnt"] for r in rows}
            captures = counts.get("capture", 0) + counts.get("add", 0)
            executes = counts.get("start", 0) + counts.get("finish", 0)
            if captures > 0 and executes > 0 and captures / max(executes, 1) > 3:
                return "Lots of capturing, not much executing this week"
            if captures > 5 and executes == 0:
                return "Lots of capturing, not much executing this week"
        except Exception:
            pass
        return None

    def _streak_tracking(self) -> Optional[str]:
        """'N-day streak!' if consecutive days with at least one command."""
        try:
            cutoff = (datetime.utcnow() - timedelta(days=30)).isoformat()
            rows = self.conn.execute(
                "SELECT DISTINCT date(timestamp) as day FROM usage_log "
                "WHERE timestamp >= ? ORDER BY day DESC",
                (cutoff,),
            ).fetchall()
            if not rows:
                return None
            # Count consecutive days from today backward
            today = datetime.utcnow().date()
            streak = 0
            for i, row in enumerate(rows):
                day = datetime.strptime(row["day"], "%Y-%m-%d").date()
                expected = today - timedelta(days=i)
                if day == expected:
                    streak += 1
                else:
                    break
            if streak >= 2:
                return streak_banner_text(streak)
        except Exception:
            pass
        return None
