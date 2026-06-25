# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Screen time data from macOS knowledgeC.db.

Reads app usage data for daily review (per RINT-01).
Gracefully degrades when Full Disk Access unavailable (per D-04).
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import structlog


logger = structlog.get_logger()

# Mac Absolute Time epoch offset (2001-01-01 vs 1970-01-01)
MAC_EPOCH_OFFSET = 978307200

# Path to knowledgeC.db on macOS
KNOWLEDGE_DB_PATH = Path.home() / "Library/Application Support/Knowledge/knowledgeC.db"


@dataclass
class AppUsage:
    """App usage record."""
    app_id: str  # Bundle ID (e.g., com.apple.Safari)
    usage_minutes: float


@dataclass
class ScreenTimeResult:
    """Result of screen time query."""
    available: bool
    message: str  # Empty if available, error message if not
    apps: list[AppUsage]  # Top apps by usage (empty if not available)


def check_screen_time_access() -> tuple[bool, str]:
    """
    Check if knowledgeC.db is accessible (per D-04).

    Returns:
        (accessible, message) tuple. Message empty if accessible.
    """
    if not KNOWLEDGE_DB_PATH.exists():
        logger.debug("screen_time.check", available=False, reason="db_not_found")
        return False, "Screen Time database not found"

    try:
        # Attempt to open the file for reading
        with open(KNOWLEDGE_DB_PATH, "rb") as f:
            f.read(1)
        logger.debug("screen_time.check", available=True)
        return True, ""
    except PermissionError:
        logger.debug("screen_time.check", available=False, reason="permission_denied")
        return False, "Full Disk Access required for Terminal (System Settings > Privacy & Security > Full Disk Access)"
    except Exception as e:
        logger.debug("screen_time.check", available=False, reason=str(e))
        return False, f"Cannot access Screen Time: {e}"


def _datetime_to_mac_epoch(dt: datetime) -> float:
    """Convert Python datetime to Mac Absolute Time."""
    return dt.timestamp() - MAC_EPOCH_OFFSET


def _get_today_bounds() -> tuple[float, float]:
    """
    Get today's midnight and now in Mac epoch (per D-01: today only).

    Returns:
        (mac_midnight, mac_now) tuple
    """
    # Use local timezone for "today" boundaries
    now = datetime.now()
    midnight = now.replace(hour=0, minute=0, second=0, microsecond=0)

    # Convert to UTC for Mac epoch calculation
    now_utc = now.astimezone(timezone.utc)
    midnight_utc = midnight.astimezone(timezone.utc)

    return _datetime_to_mac_epoch(midnight_utc), _datetime_to_mac_epoch(now_utc)


def get_today_screen_time() -> ScreenTimeResult:
    """
    Get app usage for today from knowledgeC.db (per D-01, D-02).

    Returns top 5 apps by usage time per D-02.
    Gracefully degrades per D-04 if database inaccessible.

    Returns:
        ScreenTimeResult with availability status and app list
    """
    accessible, message = check_screen_time_access()
    if not accessible:
        return ScreenTimeResult(available=False, message=message, apps=[])

    mac_midnight, mac_now = _get_today_bounds()

    query = """
    SELECT
        ZOBJECT.ZVALUESTRING AS app_id,
        SUM(ZOBJECT.ZENDDATE - ZOBJECT.ZSTARTDATE) / 60.0 AS usage_minutes
    FROM ZOBJECT
    WHERE ZSTREAMNAME = '/app/usage'
      AND ZOBJECT.ZSTARTDATE >= ?
      AND ZOBJECT.ZENDDATE <= ?
      AND ZOBJECT.ZVALUESTRING IS NOT NULL
    GROUP BY ZOBJECT.ZVALUESTRING
    ORDER BY usage_minutes DESC
    LIMIT 5
    """

    try:
        conn = sqlite3.connect(str(KNOWLEDGE_DB_PATH))
        conn.row_factory = sqlite3.Row
        try:
            cursor = conn.execute(query, (mac_midnight, mac_now))
            rows = cursor.fetchall()

            apps = [
                AppUsage(app_id=row["app_id"], usage_minutes=row["usage_minutes"])
                for row in rows
                if row["usage_minutes"] and row["usage_minutes"] > 0
            ]

            logger.debug("screen_time.query", app_count=len(apps))
            return ScreenTimeResult(available=True, message="", apps=apps)
        finally:
            conn.close()
    except sqlite3.Error as e:
        logger.debug("screen_time.query", error=str(e))
        return ScreenTimeResult(
            available=False,
            message=f"Error reading Screen Time: {e}",
            apps=[]
        )


def format_app_name(bundle_id: str) -> str:
    """
    Format bundle ID for display.

    Extracts the last component of the bundle ID as a readable name.
    e.g., "com.apple.Safari" -> "Safari"
    """
    parts = bundle_id.split(".")
    if len(parts) >= 1:
        return parts[-1]
    return bundle_id
