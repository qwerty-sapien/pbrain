# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Report generation for all review and reporting commands.

Implements D-37 (sparklines), D-38 (month/track review),
D-39 (5 report commands), D-40 (concise actionable output).
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timedelta
from typing import Optional

import structlog

from pb.storage.repository import Repository

logger = structlog.get_logger()

# Sparkline characters per D-37
SPARK_CHARS = "▁▂▃▄▅▆▇█"

# Clamp for days parameter (T-02-12-02: DoS mitigation)
_MIN_DAYS = 1
_MAX_DAYS = 365


def _clamp_days(days: int) -> int:
    """Clamp days to a reasonable range to prevent DoS."""
    return max(_MIN_DAYS, min(_MAX_DAYS, days))


def sparkline(values: list[float]) -> str:
    """Generate compact inline sparkline from values per D-37."""
    if not values:
        return ""
    min_val = min(values)
    max_val = max(values)
    range_val = max_val - min_val
    if range_val == 0:
        # All same value — show as max
        return SPARK_CHARS[-1] * len(values)
    result = []
    for v in values:
        normalized = (v - min_val) / range_val
        index = int(normalized * (len(SPARK_CHARS) - 1))
        index = max(0, min(len(SPARK_CHARS) - 1, index))
        result.append(SPARK_CHARS[index])
    return "".join(result)


class ReportEngine:
    """Generates all report types."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def generate_day_report(self, date: Optional[datetime] = None) -> str:
        """Concise daily stats report per D-39."""
        if date is None:
            date = datetime.utcnow()
        start = date.replace(hour=0, minute=0, second=0, microsecond=0)
        end = start + timedelta(days=1)

        sessions = self.repo.list_sessions_in_range(start, end)
        blocks = self.repo.list_time_blocks_for_date(date)

        actual_mins = sum(
            int((s.end_at - s.start_at).total_seconds() / 60)
            for s in sessions if s.end_at
        )
        planned_mins = sum(b.duration_minutes for b in blocks)

        completed = []
        for s in sessions:
            t = self.repo.get_task(s.task_id)
            if t and t.completion >= 100 and t not in completed:
                completed.append(t)

        lines = [f"# Day Report: {date.strftime('%Y-%m-%d')}", ""]
        lines.append(
            f"Planned: {planned_mins}m  |  Actual: {actual_mins}m  |  "
            f"Sessions: {len(sessions)}  |  Completed: {len(completed)}"
        )
        lines.append("")

        if completed:
            lines.append("Completed:")
            for t in completed:
                lines.append(f"  [x] {t.title}")
            lines.append("")

        return "\n".join(lines)

    def generate_week_report(self, week_start: Optional[datetime] = None) -> str:
        """Concise weekly stats report per D-39."""
        if week_start is None:
            today = datetime.utcnow()
            week_start = today - timedelta(days=today.weekday())
        week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        week_end = week_start + timedelta(days=7)

        sessions = self.repo.list_sessions_in_range(week_start, week_end)
        actual_mins = sum(
            int((s.end_at - s.start_at).total_seconds() / 60)
            for s in sessions if s.end_at
        )

        # Daily breakdown
        daily_mins = []
        for d in range(7):
            day = week_start + timedelta(days=d)
            day_end = day + timedelta(days=1)
            day_sessions = [s for s in sessions if day <= s.start_at < day_end]
            dm = sum(
                int((s.end_at - s.start_at).total_seconds() / 60)
                for s in day_sessions if s.end_at
            )
            daily_mins.append(dm)

        lines = [
            f"# Week Report: {week_start.strftime('%Y-%m-%d')} to "
            f"{(week_end - timedelta(days=1)).strftime('%Y-%m-%d')}",
            "",
        ]
        lines.append(f"Total: {actual_mins}m ({actual_mins / 60:.1f}h)  |  Sessions: {len(sessions)}")
        spark = sparkline([float(m) for m in daily_mins])
        day_labels = "Mon Tue Wed Thu Fri Sat Sun"
        lines.append(f"Daily: {spark}  ({day_labels})")
        lines.append("")

        return "\n".join(lines)

    def generate_energy_report(self, days: int = 7) -> str:
        """Energy trends over time per D-39."""
        days = _clamp_days(days)
        debriefs = self.repo.list_daily_debriefs(days=days)

        if not debriefs:
            return (
                "# Energy Report\n\n"
                f"No daily debriefs found ({days} days). Run 'pb review day' to generate data.\n"
            )

        morning = [d.energy_morning for d in debriefs if d.energy_morning is not None]
        midday = [d.energy_midday for d in debriefs if d.energy_midday is not None]
        evening = [d.energy_evening for d in debriefs if d.energy_evening is not None]

        lines = [f"# Energy Report ({days} days)", ""]
        if morning:
            lines.append(
                f"Morning: {sparkline([float(v) for v in morning])}"
                f"  avg {sum(morning)/len(morning):.1f}/5"
            )
        else:
            lines.append("Morning: no data")
        if midday:
            lines.append(
                f"Midday:  {sparkline([float(v) for v in midday])}"
                f"  avg {sum(midday)/len(midday):.1f}/5"
            )
        else:
            lines.append("Midday:  no data")
        if evening:
            lines.append(
                f"Evening: {sparkline([float(v) for v in evening])}"
                f"  avg {sum(evening)/len(evening):.1f}/5"
            )
        else:
            lines.append("Evening: no data")
        lines.append("")

        # Energy-task match
        matches = [d.energy_task_match for d in debriefs if d.energy_task_match]
        if matches:
            yes_count = matches.count("yes")
            total = len(matches)
            lines.append(
                f"Energy-task match: {yes_count}/{total} days "
                f"({yes_count/total*100:.0f}% matched)"
            )
        lines.append("")

        return "\n".join(lines)

    def generate_friction_report(self, days: int = 7) -> str:
        """Recurring friction patterns per D-39."""
        days = _clamp_days(days)
        debriefs = self.repo.list_daily_debriefs(days=days)

        if not debriefs:
            return "# Friction Report\n\nNo daily debriefs found.\n"

        blockers = [d.biggest_blocker for d in debriefs if d.biggest_blocker]
        counts = Counter(blockers)

        lines = [f"# Friction Report ({days} days)", ""]
        if counts:
            lines.append("Frequency:")
            for blocker, count in counts.most_common():
                label = blocker.replace("_", " ")
                bar = "=" * count
                lines.append(f"  {label:<25} {bar} ({count})")
        else:
            lines.append("No friction recorded.")
        lines.append("")

        return "\n".join(lines)

    def generate_blocker_report(self, days: int = 7) -> str:
        """Backward-compatible alias for generate_friction_report."""
        return self.generate_friction_report(days=days)

    def generate_priority_report(self) -> str:
        """Priority distribution per D-39."""
        from pb.core.priority import task_priority_score, task_eisenhower, get_priority_action

        tasks = [
            t for t in self.repo.list_tasks()
            if t.completion < 100
            and t.archived_at is None
        ]

        scored = [t for t in tasks if task_priority_score(t) is not None]
        unscored = [t for t in tasks if task_priority_score(t) is None]

        # Eisenhower distribution
        eisen_counts: dict[str, int] = {}
        for t in scored:
            e = task_eisenhower(t)
            if e:
                eisen_counts[e.value] = eisen_counts.get(e.value, 0) + 1

        # Priority action distribution
        action_counts: dict[str, int] = {}
        for t in scored:
            score = task_priority_score(t)
            if score is not None:
                action = get_priority_action(score)
                action_counts[action.value] = action_counts.get(action.value, 0) + 1

        lines = ["# Priority Report", ""]
        lines.append(
            f"Total tasks: {len(tasks)}  |  Scored: {len(scored)}  |  Unscored: {len(unscored)}"
        )
        lines.append("")

        lines.append("Eisenhower Distribution:")
        for quad in [
            "do_today",
            "schedule_deep_work",
            "batch_delegate_or_automate",
            "delete_or_defer",
        ]:
            count = eisen_counts.get(quad, 0)
            label = quad.replace("_", " ")
            bar = "=" * count
            lines.append(f"  {label:<35} {bar} ({count})")
        lines.append("")

        lines.append("Priority Action Distribution:")
        for action in [
            "schedule_first",
            "schedule_if_capacity",
            "batch_delegate_simplify",
            "drop_or_defer",
        ]:
            count = action_counts.get(action, 0)
            label = action.replace("_", " ")
            bar = "=" * count
            lines.append(f"  {label:<25} {bar} ({count})")
        lines.append("")

        return "\n".join(lines)

    def generate_month_report(self, date: Optional[datetime] = None) -> str:
        """30-day aggregate with sparklines and MoM comparison per D-38."""
        if date is None:
            date = datetime.utcnow()

        end = date.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(days=1)
        start = end - timedelta(days=30)
        prev_start = start - timedelta(days=30)

        # Current 30 days
        sessions = self.repo.list_sessions_in_range(start, end)
        curr_mins = sum(
            int((s.end_at - s.start_at).total_seconds() / 60)
            for s in sessions if s.end_at
        )

        # Previous 30 days
        prev_sessions = self.repo.list_sessions_in_range(prev_start, start)
        prev_mins = sum(
            int((s.end_at - s.start_at).total_seconds() / 60)
            for s in prev_sessions if s.end_at
        )

        # Weekly breakdown (4 weeks within current month)
        weekly_mins = []
        for w in range(4):
            w_start = start + timedelta(weeks=w)
            w_end = w_start + timedelta(weeks=1)
            w_sessions = [s for s in sessions if w_start <= s.start_at < w_end]
            wm = sum(
                int((s.end_at - s.start_at).total_seconds() / 60)
                for s in w_sessions if s.end_at
            )
            weekly_mins.append(wm)

        # Daily sparkline
        daily_mins = []
        for d in range(30):
            day = start + timedelta(days=d)
            day_end = day + timedelta(days=1)
            day_sessions = [s for s in sessions if day <= s.start_at < day_end]
            dm = sum(
                int((s.end_at - s.start_at).total_seconds() / 60)
                for s in day_sessions if s.end_at
            )
            daily_mins.append(dm)

        lines = [
            f"# Month Report: {start.strftime('%Y-%m-%d')} to "
            f"{(end - timedelta(days=1)).strftime('%Y-%m-%d')}",
            "",
        ]

        # MoM comparison
        lines.append("## Month-over-Month")
        lines.append(f"  This month:  {curr_mins}m ({curr_mins / 60:.1f}h)")
        lines.append(f"  Last month:  {prev_mins}m ({prev_mins / 60:.1f}h)")
        if prev_mins > 0:
            change = ((curr_mins - prev_mins) / prev_mins) * 100
            direction = "+" if change >= 0 else ""
            lines.append(f"  Change:      {direction}{change:.0f}%")
        lines.append("")

        # Weekly breakdown
        lines.append("## Weekly Breakdown")
        for i, wm in enumerate(weekly_mins, 1):
            lines.append(f"  Week {i}: {wm}m ({wm / 60:.1f}h)")
        if weekly_mins:
            avg = sum(weekly_mins) / len(weekly_mins)
            lines.append(f"  Average: {avg:.0f}m ({avg / 60:.1f}h)")
        lines.append("")

        # Daily trend sparkline
        spark = sparkline([float(m) for m in daily_mins])
        lines.append("## Daily Trend")
        lines.append(f"  {spark}")
        lines.append("")

        return "\n".join(lines)

    def generate_track_report(self, track_name: str) -> str:
        """Per-track report per D-38.

        T-02-12-01: track_name is looked up against DB; no match returns friendly error.
        """
        tracks = self.repo.list_tracks()
        track = next(
            (t for t in tracks if t.name.lower() == track_name.lower()), None
        )

        if not track:
            available = ", ".join(t.name for t in tracks) if tracks else "none"
            return f"Track not found: {track_name}\nAvailable: {available}\n"

        # Last 30 days of sessions for this track
        end = datetime.utcnow()
        start = end - timedelta(days=30)
        sessions = self.repo.list_sessions_in_range(start, end)

        track_sessions = []
        for s in sessions:
            task = self.repo.get_task(s.task_id)
            if task and track.id in task.linked_track_ids:
                track_sessions.append(s)

        total_mins = sum(
            int((s.end_at - s.start_at).total_seconds() / 60)
            for s in track_sessions if s.end_at
        )

        # Tasks in this track
        all_tasks = self.repo.list_tasks()
        track_tasks = [t for t in all_tasks if track.id in t.linked_track_ids]
        completed = [t for t in track_tasks if t.completion >= 100]
        completion_rate = (len(completed) / len(track_tasks) * 100) if track_tasks else 0

        # Weekly trend
        weekly_mins = []
        for w in range(4):
            w_start = start + timedelta(weeks=w)
            w_end = w_start + timedelta(weeks=1)
            wm = sum(
                int((s.end_at - s.start_at).total_seconds() / 60)
                for s in track_sessions
                if s.end_at and w_start <= s.start_at < w_end
            )
            weekly_mins.append(wm)

        spark = sparkline([float(m) for m in weekly_mins])

        lines = [f"# Track Report: {track.name}", ""]
        lines.append(f"Total hours (30 days): {total_mins / 60:.1f}h")
        lines.append(f"Sessions: {len(track_sessions)}")
        lines.append(
            f"Tasks: {len(track_tasks)} "
            f"({len(completed)} completed, {completion_rate:.0f}% rate)"
        )
        lines.append(f"Weekly trend: {spark}")
        lines.append("")

        return "\n".join(lines)

    def generate_goals_report(self) -> str:
        """Goal milestone tracking per D-39."""
        goals = self.repo.list_goal_arcs()

        if not goals:
            return "# Goals Report\n\nNo goals defined. Use 'pb goal add' to create one.\n"

        lines = ["# Goals Report", ""]
        for goal in goals:
            lines.append(f"## {goal.title}")
            lines.append(f"  Status: {goal.status}  |  Horizon: {goal.horizon.value}")

            if goal.target_value is not None and goal.metric_type:
                lines.append(f"  Target: {goal.target_value} ({goal.metric_type})")

            if goal.target_date:
                days_left = (goal.target_date - datetime.utcnow()).days
                lines.append(
                    f"  Deadline: {goal.target_date.strftime('%Y-%m-%d')} ({days_left} days)"
                )

            # Count tasks linked to this goal
            tasks = self.repo.list_tasks()
            linked = [t for t in tasks if goal.id in t.linked_goal_arc_ids]
            completed = [t for t in linked if t.completion >= 100]
            if linked:
                rate = len(completed) / len(linked) * 100
                lines.append(f"  Tasks: {len(linked)} ({len(completed)} done, {rate:.0f}%)")
            lines.append("")

        return "\n".join(lines)
