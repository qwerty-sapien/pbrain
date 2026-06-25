# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review engine for daily and weekly reviews.

Generates planned vs actual reports from local session data.
Implements hybrid table + prose format per D-01 through D-09.
Weekly review implements D-34 to D-36 chat-based structured reflection.
"""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

from pb.core.screen_time import get_today_screen_time, format_app_name
from pb.domain.enums import TaskState
from pb.domain.models import DailyDebrief, Session, Task, TimeBlock, Track
from pb.storage.repository import Repository


# Rotating learning questions per D-30 section D (exactly 7, one per weekday cycle)
LEARNING_QUESTIONS = [
    "What should I repeat tomorrow?",
    "What should I avoid tomorrow?",
    "What did I learn about my capacity?",
    "What task was more valuable than expected?",
    "What task was less valuable than expected?",
    "Where did I lose focus?",
    "What decision would have made today easier?",
]

# Predefined blocker options per D-30 section B (9 options)
DEBRIEF_BLOCKERS = [
    "unclear_next_action",
    "underestimated_effort",
    "interruption",
    "low_energy",
    "dependency",
    "emotional_resistance",
    "overcommitment",
    "tool_process_friction",
    "other",
]


def _render_bar(percent: float, width: int = 10) -> str:
    """Render visual bar using dashes and spaces per D-11."""
    filled = int(round(percent / 100 * width))
    empty = width - filled
    return "-" * filled + " " * empty


def _render_block_bar(count: int, max_count: int, width: int = 20) -> str:
    """Render Unicode block bar for usage chart (D-05, D-20)."""
    if max_count == 0:
        return " " * width
    filled = int(round(count / max_count * width))
    return "█" * filled + " " * (width - filled)


class ReviewEngine:
    """Generates review reports."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def _format_screen_time_section(self) -> str:
        """
        Format screen time data as table section (per D-02, D-03).

        Shows top 5 apps by usage time with visual bars.
        Returns empty string if screen time unavailable.
        """
        result = get_today_screen_time()

        if not result.available:
            return f"## Screen Time\n\n*{result.message}*\n\n"

        if not result.apps:
            return "## Screen Time\n\n*No app usage recorded today.*\n\n"

        # Calculate total for percentages
        total_minutes = sum(app.usage_minutes for app in result.apps)
        if total_minutes == 0:
            return "## Screen Time\n\n*No app usage recorded today.*\n\n"

        lines = [
            "## Screen Time",
            "",
            "| APP             | MINUTES |   % | BAR        |",
            "|-----------------|---------|-----|------------|",
        ]

        for app in result.apps:
            name = format_app_name(app.app_id)[:15]
            minutes = int(app.usage_minutes)
            pct = int(round(app.usage_minutes / total_minutes * 100))
            bar = _render_bar(pct, 10)
            lines.append(f"| {name:<15} | {minutes:>7} | {pct:>3} | {bar} |")

        lines.append("")
        return "\n".join(lines)

    def calculate_blocker_score(
        self,
        severity: int,
        frequency: int,
        session_hours: float
    ) -> float:
        """
        Calculate normalized blocker score (per D-14, RINT-04).

        Formula: severity x frequency / session_hours

        Args:
            severity: Blocker severity 1-10 from user input (per D-15)
            frequency: Number of blocker occurrences (default 1 for single question)
            session_hours: Total session hours for the day

        Returns:
            Normalized blocker score (distraction_rate)
        """
        if session_hours <= 0:
            # Avoid division by zero; use raw score if no sessions
            return float(severity * frequency)

        return (severity * frequency) / session_hours

    def _get_trend_arrow(self, today_score: float, yesterday_score: Optional[float]) -> str:
        """
        Get trend arrow comparing today vs yesterday (per D-16).

        Uses threshold of 0.5 to avoid noise (per RESEARCH.md Pitfall 5).

        Args:
            today_score: Today's blocker score
            yesterday_score: Yesterday's blocker score (None if no data)

        Returns:
            Arrow string: "^" (up/worse), "v" (down/better), "-" (stable), "" (no comparison)
        """
        if yesterday_score is None:
            return ""

        delta = today_score - yesterday_score
        threshold = 0.5  # Per RESEARCH.md Pitfall 5

        if delta > threshold:
            return " ^"  # Up arrow (higher = more blockers = worse)
        elif delta < -threshold:
            return " v"  # Down arrow (lower = fewer blockers = better)
        else:
            return " -"  # Stable

    def get_blocker_trend(self, date: datetime) -> tuple[Optional[float], str]:
        """
        Get today's blocker score and trend vs yesterday.

        Returns:
            (score, trend_arrow) tuple. Score is None if no blocker response today.
        """
        today_str = date.strftime("%Y-%m-%d")

        # Get today's blocker response
        responses = self.repo.get_review_responses_for_date(today_str)
        blocker_response = next(
            (r for r in responses if r.question_id == "blockers"),
            None
        )

        if blocker_response is None:
            return None, ""

        # Get session hours for normalization
        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        sessions = self.repo.list_sessions_in_range(start_of_day, end_of_day)

        total_minutes = 0
        for session in sessions:
            if session.end_at:
                delta = session.end_at - session.start_at
                total_minutes += int(delta.total_seconds() / 60)

        session_hours = total_minutes / 60.0

        # Calculate today's score (frequency=1 for single daily question)
        today_score = self.calculate_blocker_score(
            severity=blocker_response.numeric_score,
            frequency=1,
            session_hours=session_hours
        )

        # Get yesterday's response for trend
        yesterday_response = self.repo.get_yesterday_response("blockers", today_str)
        yesterday_score = None
        if yesterday_response:
            # Need yesterday's session hours too
            yesterday_dt = date - timedelta(days=1)
            yesterday_start = yesterday_dt.replace(hour=0, minute=0, second=0, microsecond=0)
            yesterday_end = yesterday_start + timedelta(days=1)
            yesterday_sessions = self.repo.list_sessions_in_range(yesterday_start, yesterday_end)

            yesterday_minutes = 0
            for session in yesterday_sessions:
                if session.end_at:
                    delta = session.end_at - session.start_at
                    yesterday_minutes += int(delta.total_seconds() / 60)

            yesterday_hours = yesterday_minutes / 60.0
            yesterday_score = self.calculate_blocker_score(
                severity=yesterday_response.numeric_score,
                frequency=1,
                session_hours=yesterday_hours
            )

        trend = self._get_trend_arrow(today_score, yesterday_score)
        return today_score, trend

    def _get_slipped_tasks(self, date: datetime) -> list[Task]:
        """
        Find tasks with time blocks scheduled for date but not completed.

        Per D-08, D-09: Query time_blocks, join with tasks, filter state != DONE.

        Args:
            date: Date to check for slipped tasks

        Returns:
            List of tasks that were scheduled but not completed
        """
        blocks = self.repo.list_time_blocks_for_date(date)
        slipped = []
        seen_task_ids: set[str] = set()

        for block in blocks:
            if block.task_id in seen_task_ids:
                continue
            seen_task_ids.add(block.task_id)

            task = self.repo.get_task(block.task_id)
            if task and task.completion < 100:
                slipped.append(task)

        return slipped

    def _aggregate_by_track(self, sessions: list[Session]) -> dict[str, int]:
        """
        Aggregate session minutes by track.

        Per D-13: Group by task.linked_track_ids[0].
        Per D-12: Tasks without track_id go to "Untracked".

        Args:
            sessions: List of sessions to aggregate

        Returns:
            Dictionary mapping track_id (or "Untracked") to total minutes
        """
        breakdown: dict[str, int] = {}
        for session in sessions:
            if session.end_at is None:
                continue
            task = self.repo.get_task(session.task_id)
            if task is None:
                continue

            minutes = int((session.end_at - session.start_at).total_seconds() / 60)

            # Use first linked track, or "Untracked" per D-12
            track_id = task.linked_track_ids[0] if task.linked_track_ids else "Untracked"
            breakdown[track_id] = breakdown.get(track_id, 0) + minutes

        return breakdown

    def _format_track_breakdown(self, breakdown: dict[str, int]) -> str:
        """
        Format track breakdown as table with visual bars per D-10.

        Columns: TRACK | MINUTES | % | BAR

        Args:
            breakdown: Dictionary from _aggregate_by_track

        Returns:
            Markdown-formatted table string
        """
        if not breakdown:
            return "No sessions recorded.\n"

        total = sum(breakdown.values())
        if total == 0:
            return "No sessions recorded.\n"

        lines = [
            "| TRACK           | MINUTES |   % | BAR        |",
            "|-----------------|---------|-----|------------|",
        ]

        # Sort by minutes descending
        for track_id, minutes in sorted(breakdown.items(), key=lambda x: -x[1]):
            if track_id == "Untracked":
                name = "Untracked"
            else:
                track = self.repo.get_track(track_id)
                name = track.name[:15] if track else track_id[:15]

            pct = int(round(minutes / total * 100))
            bar = _render_bar(pct, 10)

            lines.append(f"| {name:<15} | {minutes:>7} | {pct:>3} | {bar} |")

        return "\n".join(lines) + "\n"

    def get_rotating_question(self, date: Optional[datetime] = None) -> str:
        """Get today's rotating learning question (cycles through 7 per D-30 section D)."""
        if date is None:
            date = datetime.utcnow()
        day_of_year = date.timetuple().tm_yday
        index = day_of_year % len(LEARNING_QUESTIONS)
        return LEARNING_QUESTIONS[index]

    def generate_daily_debrief(self, debrief: DailyDebrief, date: Optional[datetime] = None) -> str:
        """Generate the 5-section debrief output per D-30.

        Sections: A. Completion, B. Friction, C. Energy, D. Learning, E. Tomorrow.
        Includes session stats (completion rate) and screen time.
        """
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)
        sessions = self.repo.list_sessions_in_range(start_of_day, end_of_day)

        # Compute session stats
        actual_minutes = 0
        for s in sessions:
            if s.end_at:
                actual_minutes += int((s.end_at - s.start_at).total_seconds() / 60)

        blocks = self.repo.list_time_blocks_for_date(date)
        planned_minutes = sum(b.duration_minutes for b in blocks)
        completion_rate = (actual_minutes / planned_minutes * 100) if planned_minutes > 0 else 0

        lines = [f"# Daily Debrief: {date.strftime('%Y-%m-%d')}", ""]

        # Summary stats line
        lines.append(
            f"Sessions: {len(sessions)} | Planned: {planned_minutes}m | "
            f"Actual: {actual_minutes}m | Completion: {completion_rate:.0f}%"
        )
        lines.append("")

        # A. Completion
        lines.append("## A. Completion")
        lines.append(f"  Top 1: {debrief.top1_completed or '-'}")
        if debrief.top3_completed:
            lines.append(f"  Top 3 completed: {', '.join(debrief.top3_completed)}")
        if debrief.what_shipped:
            lines.append(f"  Shipped: {debrief.what_shipped}")
        lines.append("")

        # B. Friction
        lines.append("## B. Friction")
        blocker_label = (debrief.biggest_blocker or "none").replace("_", " ")
        lines.append(f"  Primary friction: {blocker_label}")
        if debrief.blocker_note:
            lines.append(f"  Note: {debrief.blocker_note}")
        lines.append("")

        # C. Energy
        lines.append("## C. Energy")
        m = f"{debrief.energy_morning}/5" if debrief.energy_morning is not None else "-"
        mid = f"{debrief.energy_midday}/5" if debrief.energy_midday is not None else "-"
        e = f"{debrief.energy_evening}/5" if debrief.energy_evening is not None else "-"
        lines.append(f"  Morning: {m}  |  Midday: {mid}  |  Evening: {e}")
        lines.append(f"  Task-energy match: {debrief.energy_task_match or '-'}")
        lines.append("")

        # D. Learning
        lines.append("## D. Learning")
        if debrief.learning_question:
            lines.append(f"  Q: {debrief.learning_question}")
        if debrief.learning_answer:
            lines.append(f"  A: {debrief.learning_answer}")
        if debrief.learning_score is not None:
            lines.append(f"  Score: {debrief.learning_score}/10")
            if debrief.learning_rationale:
                lines.append(f"  ({debrief.learning_rationale})")
        lines.append("")

        # E. Tomorrow
        lines.append("## E. Tomorrow")
        if debrief.tomorrow_top1:
            lines.append(f"  Top 1: {debrief.tomorrow_top1}")
        if debrief.tomorrow_next_action:
            lines.append(f"  Next action: {debrief.tomorrow_next_action}")
        lines.append("")

        # Screen time (D-33)
        screen_time_section = self._format_screen_time_section()
        if screen_time_section:
            lines.append(screen_time_section)

        return "\n".join(lines)

    def generate_daily_review(self, date: Optional[datetime] = None) -> str:
        """
        Generate a daily review report.

        Uses hybrid format per D-01: summary table at top, prose sections below.
        Shows planned vs actual per REVW-01, slippage per REVW-02,
        interruptions per REVW-03.

        Args:
            date: Date to review (defaults to today)

        Returns:
            Markdown-formatted review report
        """
        if date is None:
            date = datetime.utcnow()

        start_of_day = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end_of_day = start_of_day + timedelta(days=1)

        sessions = self.repo.list_sessions_in_range(start_of_day, end_of_day)
        blocks = self.repo.list_time_blocks_for_date(date)

        planned_minutes = sum(b.duration_minutes for b in blocks)

        actual_minutes = 0
        for session in sessions:
            if session.end_at:
                delta = session.end_at - session.start_at
                actual_minutes += int(delta.total_seconds() / 60)

        # Calculate delta: positive = over plan, negative = under plan
        delta = actual_minutes - planned_minutes

        completed_tasks = []
        for session in sessions:
            task = self.repo.get_task(session.task_id)
            if task and task.completion >= 100:
                if task not in completed_tasks:
                    completed_tasks.append(task)

        total_interruptions = sum(s.interruption_count for s in sessions)

        # Get task-level slippage per D-08, D-09
        slipped_tasks = self._get_slipped_tasks(date)

        # Build summary table per D-01, D-02
        lines = [
            f"# Daily Review: {date.strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            "",
            "| METRIC            | VALUE              |",
            "|-------------------|-------------------|",
            f"| Planned           | {planned_minutes} min |",
            f"| Actual            | {actual_minutes} min  |",
            f"| Delta             | {delta:+d} min        |",
            f"| Sessions          | {len(sessions)}       |",
            f"| Interruptions     | {total_interruptions} |",
            "",
        ]

        # Screen time section per D-03 (after summary table, before track breakdown)
        screen_time_section = self._format_screen_time_section()
        if screen_time_section:
            lines.append(screen_time_section)

        # Track breakdown section per D-10, REVW-04
        if sessions:
            lines.append("## Track Breakdown")
            lines.append("")
            track_breakdown = self._aggregate_by_track(sessions)
            lines.append(self._format_track_breakdown(track_breakdown))

        # Completed tasks section per D-03
        if completed_tasks:
            lines.append("## Completed Tasks")
            lines.append("")
            for task in completed_tasks:
                lines.append(f"- [x] {task.title}")
            lines.append("")

        # Session log section per D-03
        if sessions:
            lines.append("## Session Log")
            lines.append("")
            for session in sessions:
                task = self.repo.get_task(session.task_id)
                task_title = task.title if task else "Unknown"
                start = session.start_at.strftime("%H:%M")
                end = session.end_at.strftime("%H:%M") if session.end_at else "ongoing"
                outcome = session.actual_outcome or "no outcome"
                lines.append(f"- {start}-{end}: {task_title} ({outcome})")
            lines.append("")

        # Slippage section per D-06, D-07, D-08
        # Show both time-level and task-level slippage
        if delta < 0 or slipped_tasks:
            lines.append("## Slippage")
            lines.append("")

            # Time-level slippage per D-07
            if delta < 0:
                lines.append(f"**Time:** {-delta} minutes under plan")
                lines.append("")

            # Task-level slippage per D-08
            if slipped_tasks:
                lines.append("**Scheduled but not completed:**")
                lines.append("")
                for task in slipped_tasks:
                    lines.append(f"- [ ] {task.title}")
                lines.append("")

        # Tomorrow section
        lines.append("## Tomorrow")
        lines.append("")
        lines.append("- First task: [Enter first task]")
        lines.append("- Key focus: [Enter key focus]")
        lines.append("")

        # Command usage this week (D-05, ULOG-03)
        usage_section = self.format_usage_section()
        if usage_section:
            lines.append(usage_section)

        return "\n".join(lines)

    def format_usage_section(self) -> str:
        """Format 'most used commands this week' section (D-05, D-20, ULOG-03)."""
        from pb.storage.database import get_connection, get_command_counts
        try:
            with get_connection() as conn:
                counts = get_command_counts(conn, days=7)
        except Exception:
            counts = {}
        if not counts:
            return "## Command Usage This Week\n\n  No commands logged this week.\n"
        max_count = max(counts.values())
        lines = ["## Command Usage This Week", ""]
        for cmd, cnt in counts.items():
            bar = _render_block_bar(cnt, max_count, width=20)
            lines.append(f"  pb {cmd:<15} {bar} {cnt}")
        lines.append("")
        return "\n".join(lines)

    def generate_weekly_review(self, week_start: Optional[datetime] = None) -> str:
        """
        Generate a weekly review report.

        Per D-19: Same structure as daily but aggregated over 7 days.
        Per D-20: Summary table shows weekly totals.
        Per D-21: Include track breakdown for the week.

        Args:
            week_start: Start of week (defaults to most recent Monday)

        Returns:
            Markdown-formatted weekly review
        """
        if week_start is None:
            today = datetime.utcnow()
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)

        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)
        week_end_display = week_end - timedelta(days=1)

        sessions = self.repo.list_sessions_in_range(week_start, week_end)

        # Calculate planned minutes from all blocks in the week
        planned_minutes = 0
        for day_offset in range(7):
            day = week_start + timedelta(days=day_offset)
            blocks = self.repo.list_time_blocks_for_date(day)
            planned_minutes += sum(b.duration_minutes for b in blocks)

        actual_minutes = 0
        for session in sessions:
            if session.end_at:
                delta = session.end_at - session.start_at
                actual_minutes += int(delta.total_seconds() / 60)

        total_interruptions = sum(s.interruption_count for s in sessions)

        tasks = self.repo.list_tasks()
        completed_this_week = [
            t for t in tasks
            if t.completed_at and week_start <= t.completed_at < week_end
        ]

        # Build summary table per D-20
        lines = [
            f"# Weekly Review: {week_start.strftime('%Y-%m-%d')} to {week_end_display.strftime('%Y-%m-%d')}",
            "",
            "## Summary",
            "",
            "| METRIC            | VALUE              |",
            "|-------------------|-------------------|",
            f"| Planned           | {planned_minutes} min |",
            f"| Actual            | {actual_minutes} min  |",
            f"| Delta             | {actual_minutes - planned_minutes:+d} min |",
            f"| Sessions          | {len(sessions)}       |",
            f"| Interruptions     | {total_interruptions} |",
            f"| Tasks Completed   | {len(completed_this_week)} |",
            "",
        ]

        # Track breakdown per D-21
        if sessions:
            lines.append("## Track Breakdown")
            lines.append("")
            track_breakdown = self._aggregate_by_track(sessions)
            lines.append(self._format_track_breakdown(track_breakdown))

        if completed_this_week:
            lines.append("## Completed Tasks")
            for task in completed_this_week:
                lines.append(f"- [x] {task.title}")
            lines.append("")

        lines.extend([
            "## Wins",
            "- [Enter wins]",
            "",
            "## What Slipped",
            "- [Enter what slipped]",
            "",
            "## Root Causes",
            "- [Enter root causes]",
            "",
            "## Changes for Next Week",
            "- [Enter changes]",
            "",
        ])

        return "\n".join(lines)

    def compute_weekly_metrics(self, week_start: Optional[datetime] = None) -> dict:
        """Compute weekly review metrics per D-35.

        Returns dict with keys:
        - deep_hours, shallow_hours, buffer_hours (float)
        - top1_completion_rate (float, 0-100)
        - avg_top3_count (float)
        - most_common_blocker (str)
        - avg_energy (dict with morning, midday, evening averages)
        - deferred_tasks (list of Task)
        - planned_minutes, actual_minutes (int)
        """
        if week_start is None:
            today = datetime.utcnow()
            days_since_monday = today.weekday()
            week_start = today - timedelta(days=days_since_monday)

        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)

        # Get sessions and classify by work type
        sessions = self.repo.list_sessions_in_range(week_start, week_end)
        deep_minutes = 0
        shallow_minutes = 0
        buffer_minutes = 0

        _shallow_types = {"shallow", "admin", "meeting", "planning"}
        _buffer_types = {"recovery"}

        for s in sessions:
            if not s.end_at:
                continue
            mins = int((s.end_at - s.start_at).total_seconds() / 60)
            task = self.repo.get_task(s.task_id)
            work_type = (task.work_type or "") if task else ""
            if work_type in _buffer_types:
                buffer_minutes += mins
            elif work_type in _shallow_types:
                shallow_minutes += mins
            else:
                # "deep" explicitly or unclassified
                deep_minutes += mins

        # Get daily debriefs for the week
        debriefs = self.repo.list_daily_debriefs(days=7)
        week_start_str = week_start.strftime("%Y-%m-%d")
        week_end_str = week_end.strftime("%Y-%m-%d")
        week_debriefs = [
            d for d in debriefs
            if week_start_str <= d.review_date < week_end_str
        ]

        # Top 1 completion rate
        top1_total = len([d for d in week_debriefs if d.top1_completed is not None])
        top1_yes = len([d for d in week_debriefs if d.top1_completed == "yes"])
        top1_rate = (top1_yes / top1_total * 100) if top1_total > 0 else 0

        # Avg Top 3 count
        top3_counts = [len(d.top3_completed) for d in week_debriefs if d.top3_completed]
        avg_top3 = sum(top3_counts) / len(top3_counts) if top3_counts else 0

        # Most common blocker
        blocker_values = [d.biggest_blocker for d in week_debriefs if d.biggest_blocker]
        blocker_counts = Counter(blocker_values)
        most_common = blocker_counts.most_common(1)[0][0] if blocker_counts else "none"

        # Average energy curve
        morning_vals = [d.energy_morning for d in week_debriefs if d.energy_morning is not None]
        midday_vals = [d.energy_midday for d in week_debriefs if d.energy_midday is not None]
        evening_vals = [d.energy_evening for d in week_debriefs if d.energy_evening is not None]
        avg_energy = {
            "morning": sum(morning_vals) / len(morning_vals) if morning_vals else 0,
            "midday": sum(midday_vals) / len(midday_vals) if midday_vals else 0,
            "evening": sum(evening_vals) / len(evening_vals) if evening_vals else 0,
        }

        # Deferred tasks (not done, not cancelled, not archived, created before week end)
        deferred = self.repo.list_tasks_deferred_this_week(week_start)

        # Planned vs actual
        planned_minutes = 0
        actual_minutes = deep_minutes + shallow_minutes + buffer_minutes
        for day_offset in range(7):
            day = week_start + timedelta(days=day_offset)
            blocks = self.repo.list_time_blocks_for_date(day)
            planned_minutes += sum(b.duration_minutes for b in blocks)

        return {
            "deep_hours": deep_minutes / 60,
            "shallow_hours": shallow_minutes / 60,
            "buffer_hours": buffer_minutes / 60,
            "top1_completion_rate": top1_rate,
            "avg_top3_count": avg_top3,
            "most_common_blocker": most_common,
            "avg_energy": avg_energy,
            "deferred_tasks": deferred,
            "planned_minutes": planned_minutes,
            "actual_minutes": actual_minutes,
        }

    def generate_weekly_reflection(self, metrics: dict) -> str:
        """Generate structured weekly reflection output per D-36.

        7 categories: Keep, Cut or Batch, Rescope, Schedule Next Week,
        Drop Candidates, Energy Pattern, Capacity Recommendation.
        """
        lines = ["# Weekly Reflection", ""]

        # Metrics summary
        lines.append("## This Week's Numbers")
        lines.append(f"  Deep work:       {metrics['deep_hours']:.1f}h")
        lines.append(f"  Shallow work:    {metrics['shallow_hours']:.1f}h")
        lines.append(f"  Buffer/recovery: {metrics['buffer_hours']:.1f}h")
        total = metrics["deep_hours"] + metrics["shallow_hours"] + metrics["buffer_hours"]
        lines.append(f"  Total:           {total:.1f}h")
        lines.append(
            f"  Planned: {metrics['planned_minutes']}m  |  Actual: {metrics['actual_minutes']}m"
        )
        lines.append(f"  Top 1 completion: {metrics['top1_completion_rate']:.0f}%")
        lines.append(f"  Avg Top 3 completed: {metrics['avg_top3_count']:.1f}")
        blocker_label = metrics["most_common_blocker"].replace("_", " ")
        lines.append(f"  Most common friction: {blocker_label}")
        lines.append("")

        # Energy pattern section
        avg_e = metrics["avg_energy"]
        lines.append("## Energy Pattern")
        lines.append(
            f"  Morning: {avg_e['morning']:.1f}/5  |  "
            f"Midday: {avg_e['midday']:.1f}/5  |  "
            f"Evening: {avg_e['evening']:.1f}/5"
        )
        if avg_e["morning"] > avg_e["evening"] + 1:
            lines.append("  Pattern: Front-loaded. Schedule deep work in morning blocks.")
        elif avg_e["evening"] > avg_e["morning"] + 1:
            lines.append("  Pattern: Evening surge. Consider shifting deep work later.")
        else:
            lines.append("  Pattern: Relatively flat. Energy sustains through the day.")
        lines.append("")

        # 7 output categories
        lines.append("## Keep (what drove meaningful outcomes)")
        lines.append("  [Fill during reflection chat]")
        lines.append("")

        lines.append("## Cut or Batch (consumed time without payoff)")
        lines.append("  [Fill during reflection chat]")
        lines.append("")

        lines.append("## Rescope (needs redesign)")
        lines.append("  [Fill during reflection chat]")
        lines.append("")

        lines.append("## Schedule Next Week")
        if metrics["deferred_tasks"]:
            lines.append("  Repeatedly deferred:")
            for t in metrics["deferred_tasks"][:5]:
                lines.append(f"    - {t.title}")
        lines.append("  [Add calendar blocks for next week]")
        lines.append("")

        lines.append("## Drop Candidates")
        if metrics["deferred_tasks"]:
            lines.append("  Consider dropping if deferred 3+ days:")
            for t in metrics["deferred_tasks"][:3]:
                lines.append(f"    - {t.title}")
        else:
            lines.append("  [Fill during reflection chat]")
        lines.append("")

        lines.append("## Capacity Recommendation")
        if metrics["planned_minutes"] > 0 and metrics["actual_minutes"] > 0:
            ratio = metrics["actual_minutes"] / metrics["planned_minutes"]
            if ratio < 0.7:
                lines.append(
                    f"  Under-delivering ({ratio:.0%} of plan). "
                    "Reduce weekly targets or reduce recurring friction."
                )
            elif ratio > 1.1:
                lines.append(
                    f"  Over-delivering ({ratio:.0%} of plan). "
                    "Increase targets or add buffer."
                )
            else:
                lines.append(
                    f"  On track ({ratio:.0%} of plan). Maintain current capacity."
                )
        else:
            lines.append("  Insufficient data — plan and track more consistently.")
        lines.append("")

        return "\n".join(lines)

    def get_session_stats(
        self, start: datetime, end: datetime
    ) -> dict:
        """
        Get session statistics for a date range.

        Args:
            start: Range start
            end: Range end

        Returns:
            Dictionary of statistics
        """
        sessions = self.repo.list_sessions_in_range(start, end)

        total_minutes = 0
        for session in sessions:
            if session.end_at:
                delta = session.end_at - session.start_at
                total_minutes += int(delta.total_seconds() / 60)

        return {
            "session_count": len(sessions),
            "total_minutes": total_minutes,
            "total_interruptions": sum(s.interruption_count for s in sessions),
        }
