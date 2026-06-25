# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Alignment engine for goal arc time distribution.

Shows how effort distributes across goal arcs over a rolling period.
Per REVW-06: Users can run `pb review alignment` to see effort vs goals.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta

from pb.core.naming import stored_short_title
from pb.storage.repository import Repository


def _render_bar(percent: float, width: int = 10) -> str:
    """Render visual bar using dashes and spaces per D-11."""
    filled = int(round(percent / 100 * width))
    empty = width - filled
    return "-" * filled + " " * empty


@dataclass
class GoalBreakdown:
    """Display data for one goal arc in alignment view."""

    goal_id: str
    title: str
    minutes: int
    percent: float


class AlignmentEngine:
    """Core alignment logic for goal arc time distribution."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def get_alignment(self, days: int = 7) -> list[GoalBreakdown]:
        """
        Get goal arc breakdown for the last N days.

        Per D-16: Default time period is last 7 days (rolling week).
        Per D-17: Aggregate track time up to linked goal arcs.
        Per D-18: Tracks not linked to any goal arc grouped under "Other".

        Args:
            days: Number of days to analyze (default 7)

        Returns:
            List of GoalBreakdown sorted by minutes descending
        """
        # Calculate date range
        end = datetime.utcnow()
        start = (end - timedelta(days=days)).replace(
            hour=0, minute=0, second=0, microsecond=0
        )

        # Get sessions in range
        sessions = self.repo.list_sessions_in_range(start, end)

        # Aggregate by track first
        track_minutes: dict[str, int] = {}
        for session in sessions:
            if session.end_at is None:
                continue
            task = self.repo.get_task(session.task_id)
            if task is None:
                continue

            minutes = int((session.end_at - session.start_at).total_seconds() / 60)

            track_id = task.linked_track_ids[0] if task.linked_track_ids else "Untracked"
            track_minutes[track_id] = track_minutes.get(track_id, 0) + minutes

        # Roll up tracks to goal arcs
        goal_minutes: dict[str, int] = {}
        tracks = self.repo.list_tracks(active_only=False)
        track_map = {t.id: t for t in tracks}

        for track_id, minutes in track_minutes.items():
            if track_id == "Untracked":
                goal_minutes["Other"] = goal_minutes.get("Other", 0) + minutes
                continue

            track = track_map.get(track_id)
            if track is None or not track.linked_goal_arc_ids:
                # Per D-18: Tracks not linked to any goal arc -> "Other"
                goal_minutes["Other"] = goal_minutes.get("Other", 0) + minutes
            else:
                # Attribute to first linked goal arc
                goal_id = track.linked_goal_arc_ids[0]
                goal_minutes[goal_id] = goal_minutes.get(goal_id, 0) + minutes

        # Build result with goal titles
        total = sum(goal_minutes.values())
        results: list[GoalBreakdown] = []

        for goal_id, minutes in sorted(goal_minutes.items(), key=lambda x: -x[1]):
            if goal_id == "Other":
                title = "Other"
            else:
                goal = self.repo.get_goal_arc(goal_id)
                title = stored_short_title(goal) if goal else goal_id

            pct = (minutes / total * 100) if total > 0 else 0.0
            results.append(GoalBreakdown(
                goal_id=goal_id,
                title=title,
                minutes=minutes,
                percent=pct,
            ))

        return results

    def format_alignment_report(self, breakdown: list[GoalBreakdown], days: int = 7) -> str:
        """
        Format alignment breakdown as markdown report.

        Per D-14, D-15: Table columns GOAL | MINUTES | % | BAR

        Args:
            breakdown: List of GoalBreakdown from get_alignment
            days: Number of days (for header)

        Returns:
            Markdown-formatted alignment report
        """
        if not breakdown:
            return f"# Alignment Report (Last {days} Days)\n\nNo sessions recorded.\n"

        lines = [
            f"# Alignment Report (Last {days} Days)",
            "",
            "## Goal Arc Distribution",
            "",
            "| GOAL                 | MINUTES |   % | BAR        |",
            "|----------------------|---------|-----|------------|",
        ]

        for g in breakdown:
            name = g.title if len(g.title) >= 20 else g.title.ljust(20)
            bar = _render_bar(g.percent, 10)
            lines.append(f"| {name} | {g.minutes:>7} | {g.percent:>3.0f} | {bar} |")

        lines.append("")

        total = sum(g.minutes for g in breakdown)
        hours = total // 60
        mins = total % 60
        lines.append(f"**Total:** {total} minutes ({hours}h {mins}m)")
        lines.append("")

        return "\n".join(lines)


__all__ = ["AlignmentEngine", "GoalBreakdown"]
