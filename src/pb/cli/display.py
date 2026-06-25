# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Display helpers for CLI output.

Provides timezone-aware formatting for user-facing datetime display.
Storage remains UTC; conversion happens only at display time.
"""

import json
import select
import sys
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from rich.live import Live
from rich.markup import escape as _esc
from rich.panel import Panel
from rich.text import Text

from pb.cli.helpers import _read_key
from pb.core.entity_refs import display_ref

# User's timezone per PROJECT.md (UTC+8)
USER_TZ = ZoneInfo("Asia/Shanghai")


def format_datetime_local(dt: Optional[datetime], include_date: bool = False) -> str:
    """
    Format datetime in user's local timezone.

    Per D-01: 24-hour format (14:30)
    Per D-02: No timezone indicator

    Args:
        dt: UTC datetime to format (naive assumed UTC)
        include_date: Include YYYY-MM-DD prefix

    Returns:
        Formatted string like "14:30" or "2026-04-23 14:30"
    """
    if dt is None:
        return "---- -- --:--" if include_date else "--:--"

    # Ensure input has UTC tzinfo (handle naive datetimes)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(USER_TZ)

    if include_date:
        return local_dt.strftime("%Y-%m-%d %H:%M")
    return local_dt.strftime("%H:%M")


def format_date_local(dt: Optional[datetime]) -> str:
    """
    Format date only in user's local timezone.

    Args:
        dt: UTC datetime to format (naive assumed UTC)

    Returns:
        Formatted string like "2026-04-23" or "----"
    """
    if dt is None:
        return "----"

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    local_dt = dt.astimezone(USER_TZ)
    return local_dt.strftime("%Y-%m-%d")


# ---------------------------------------------------------------------------
# Live session clock helpers
# ---------------------------------------------------------------------------

def _format_hms(seconds: int) -> str:
    """Format seconds as HH:MM:SS."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def _build_clock_panel(task_title: str, elapsed_secs: int,
                        duration_secs: Optional[int] = None,
                        break_interval_min: int = 25) -> Panel:
    """Build a Rich Panel showing elapsed/remaining/overtime for the live clock."""
    lines = Text()
    lines.append(f"{task_title}\n", style="bold")
    elapsed_str = _format_hms(elapsed_secs)
    lines.append(f"Elapsed: {elapsed_str}\n", style="bold")
    if duration_secs is not None:
        remaining_secs = duration_secs - elapsed_secs
        if remaining_secs > 0:
            lines.append(f"Remaining: {_format_hms(remaining_secs)}\n", style="bold green")
        else:
            overtime_secs = abs(remaining_secs)
            lines.append(f"Overtime: +{_format_hms(overtime_secs)}\n", style="bold yellow")
    else:
        # Stopwatch mode: show break hint
        break_secs = break_interval_min * 60
        secs_to_break = break_secs - (elapsed_secs % break_secs)
        if secs_to_break < break_secs:
            lines.append(f"Break in: {_format_hms(secs_to_break)}\n", style="dim")
    lines.append("\n")
    lines.append("Ctrl+C or q to hide (session continues)", style="dim")
    return Panel(lines, title="[dim]pb session[/dim]", expand=False)


def live_session_clock(session, task, duration_minutes: Optional[int] = None,
                       break_interval_min: int = 25) -> None:
    """Block the terminal with a live updating clock until Ctrl+C.

    Session continues in SQLite after this function returns.
    Pressing `q` hides the widget while the session continues in SQLite.
    Ctrl+C exits the display cleanly as a fallback.
    """
    duration_secs = duration_minutes * 60 if duration_minutes is not None else None
    task_title = task.title if task else "Unknown task"
    try:
        with Live(refresh_per_second=1, screen=False) as live:
            while True:
                elapsed_secs = int(
                    (datetime.utcnow() - session.start_at).total_seconds()
                )
                panel = _build_clock_panel(
                    task_title, elapsed_secs, duration_secs, break_interval_min
                )
                live.update(panel)
                action = _read_clock_action(timeout=1.0)
                if action == "q":
                    break
    except KeyboardInterrupt:
        pass  # Session continues in DB; display exits cleanly


def _read_clock_action(timeout: float) -> Optional[str]:
    """Read a single live-clock action without blocking redraws."""
    if not sys.stdin.isatty():
        time.sleep(timeout)
        return None
    ready, _, _ = select.select([sys.stdin], [], [], timeout)
    if not ready:
        return None
    action = _read_key()
    if action == "q":
        return "q"
    return None


# ---------------------------------------------------------------------------
# pb now output formatter
# ---------------------------------------------------------------------------

def _compute_elapsed_min(session) -> Optional[int]:
    """Compute elapsed minutes from session.start_at. Cross-process safe."""
    if session is None or session.start_at is None:
        return None
    elapsed = datetime.utcnow() - session.start_at
    return int(elapsed.total_seconds() / 60)


def _compute_remaining_min(session) -> Optional[int]:
    """Compute remaining minutes from session.duration_minutes (new column). Returns None for stopwatch."""
    duration = getattr(session, "duration_minutes", None)
    if duration is None:
        return None
    elapsed = _compute_elapsed_min(session)
    if elapsed is None:
        return None
    remaining = duration - elapsed
    return remaining  # can be negative (overtime)


def format_now_output(session, task, mode: str = "rich") -> str:
    """Format pb now output in rich / plain / json mode.

    mode: "rich" | "plain" | "json"
    Guards against None session and None task.
    """
    if session is None:
        if mode == "json":
            return json.dumps({"active": False})
        return "No active session."

    task_name = task.title if task else f"task:{display_ref(session, 'session')}"
    elapsed = _compute_elapsed_min(session)
    remaining = _compute_remaining_min(session)
    timer_mode = getattr(session, "timer_mode", "stopwatch")
    interruptions = getattr(session, "interruption_count", 0)
    completion = getattr(session, "completion_pct", None)
    note = getattr(session, "actual_outcome", None)

    if mode == "json":
        return json.dumps({
            "task_id": session.task_id,
            "task_name": task_name,
            "mode": timer_mode,
            "started_at": session.start_at.isoformat() if session.start_at else None,
            "elapsed_min": elapsed,
            "remaining_min": remaining,
            "completion": completion,
            "interruption_count": interruptions,
            "note": note,
        })

    if mode == "plain":
        # Per UI-SPEC D-17b: "task_name elapsed_min" or "task_name -{remaining}" or "task_name +{overtime}"
        if remaining is not None:
            if remaining >= 0:
                return f"{task_name} -{remaining}"
            else:
                overtime = abs(remaining)
                return f"{task_name} +{overtime}"
        return f"{task_name} {elapsed or 0}"

    # rich mode
    if remaining is not None:
        if remaining >= 0:
            time_part = f"{remaining}m remaining"
        else:
            time_part = f"+{abs(remaining)}m overtime"
    else:
        time_part = f"{elapsed or 0}m elapsed"
    return f"[success]Active:[/] {_esc(task_name)}  [dim]{time_part}[/]"
