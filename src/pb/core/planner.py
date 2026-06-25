# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Planning engine for daily and weekly planning.

Handles plan creation, time block scheduling, and plan snapshots.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from typing import Optional

from pb.core.priority import get_priority_action, rank_tasks, task_priority_score
from pb.domain.enums import Horizon, PriorityAction, WorkType
from pb.domain.models import Task, TimeBlock
from pb.storage.repository import Repository


# Energy matching table per D-29
ENERGY_MATCH_TABLE = {
    5: {
        "best_for": "architecture, complex debugging, writing specs, hard coding",
        "avoid": "email, meetings",
        "max_energy_required": 5,
    },
    4: {
        "best_for": "feature work, algorithmic thinking, important decisions",
        "avoid": "context switching",
        "max_energy_required": 5,
    },
    3: {
        "best_for": "code review, refactoring, planning, medium-complexity",
        "avoid": "major strategic calls",
        "max_energy_required": 4,
    },
    2: {
        "best_for": "admin, docs cleanup, inbox, simple tickets",
        "avoid": "deep technical work",
        "max_energy_required": 3,
    },
    1: {
        "best_for": "recovery, walk, break, shutdown routine",
        "avoid": "forcing output",
        "max_energy_required": 2,
    },
}


class Planner:
    """Manages planning workflows."""

    def __init__(self, repo: Repository):
        self.repo = repo

    def get_active_tasks(self) -> list[Task]:
        """Get all active (non-done, non-paused) tasks."""
        return [t for t in self.repo.list_tasks() if t.completion < 100 and t.state.value == "active"]

    def get_today_tasks(self) -> list[Task]:
        """Get tasks scheduled for today."""
        tasks = self.repo.list_tasks()
        return [t for t in tasks if t.horizon == Horizon.TODAY and t.completion < 100]

    def schedule_block(
        self,
        task: Task,
        start_time: Optional[datetime],
        duration_minutes: int,
    ) -> tuple[TimeBlock, Optional[TimeBlock]]:
        """
        Schedule a time block for a task.

        Args:
            task: Task to schedule
            start_time: Block start time (None for duration-only blocks)
            duration_minutes: Block duration

        Returns:
            Tuple of (created block, overlapping block or None)
        """
        existing = self.get_today_blocks()
        overlap = self._check_overlap(existing, start_time, duration_minutes)

        block = TimeBlock(
            task_id=task.id,
            start_time=start_time,
            duration_minutes=duration_minutes,
        )
        created = self.repo.create_time_block(block)
        return (created, overlap)

    def get_today_blocks(self) -> list[TimeBlock]:
        """Get all time blocks scheduled for today."""
        return self.repo.list_time_blocks_for_date(datetime.utcnow())

    def _check_overlap(
        self,
        blocks: list[TimeBlock],
        start: Optional[datetime],
        duration: int,
    ) -> Optional[TimeBlock]:
        """Check if a proposed block overlaps with existing blocks.

        Returns None if start is None (duration-only blocks can't overlap).
        """
        if start is None:
            return None

        end = start + timedelta(minutes=duration)
        for b in blocks:
            if b.start_time is None:
                continue
            b_end = b.start_time + timedelta(minutes=b.duration_minutes)
            if start < b_end and end > b.start_time:
                return b
        return None

    def update_block(self, block: TimeBlock) -> tuple[TimeBlock, Optional[TimeBlock]]:
        """
        Update a time block's start_time and/or duration.

        Returns:
            Tuple of (updated block, overlapping block or None)
        """
        existing = self.get_today_blocks()
        others = [b for b in existing if b.id != block.id]
        overlap = None
        if block.start_time is not None:
            overlap = self._check_overlap(others, block.start_time, block.duration_minutes)
        updated = self.repo.update_time_block(block)
        return (updated, overlap)

    @staticmethod
    def _format_duration(minutes: int) -> str:
        """Format minutes as 'Xh Ym'."""
        h, m = divmod(abs(minutes), 60)
        return f"{h}h {m}m"

    def generate_daily_plan_summary(self, budget_minutes: Optional[int] = None) -> str:
        """
        Generate a text summary of the daily plan.

        Args:
            budget_minutes: If provided, append a Budget section showing committed vs remaining.

        Returns:
            Markdown-formatted daily plan
        """
        today = datetime.utcnow().strftime("%Y-%m-%d")
        lines = [f"# Daily Plan: {today}", ""]

        blocks = self.get_today_blocks()
        scheduled_blocks = [b for b in blocks if b.start_time is not None]
        unscheduled_blocks = [b for b in blocks if b.start_time is None]

        if scheduled_blocks:
            lines.append("## Scheduled Blocks")
            for block in scheduled_blocks:
                task = self.repo.get_task(block.task_id)
                task_title = task.title if task else "Unknown"
                start = block.start_time.strftime("%H:%M")
                end = (block.start_time + timedelta(minutes=block.duration_minutes)).strftime("%H:%M")
                lines.append(f"- {start}-{end}: {task_title} ({block.duration_minutes}m)")
            lines.append("")

        if unscheduled_blocks:
            lines.append("## Committed Time (unscheduled)")
            for block in unscheduled_blocks:
                task = self.repo.get_task(block.task_id)
                task_title = task.title if task else "Unknown"
                lines.append(f"- {task_title} ({block.duration_minutes}m)")
            lines.append("")

        active = self.get_active_tasks()
        if active:
            lines.append("## Active Tasks")
            for task in active:
                est = f" ({task.estimate_minutes}m)" if task.estimate_minutes else ""
                lines.append(f"- [ ] {task.title}{est}")
            lines.append("")

        if not (blocks or active):
            lines.append("No tasks planned for today.")
            lines.append("")

        if budget_minutes is not None:
            all_blocks = self.repo.list_time_blocks_created_for_date(datetime.utcnow())
            committed = sum(b.duration_minutes for b in all_blocks)
            remaining = budget_minutes - committed

            lines.append("## Budget")
            lines.append(f"  Budget:    {self._format_duration(budget_minutes)}")
            lines.append(f"  Committed: {self._format_duration(committed)}")
            if remaining >= 0:
                lines.append(f"  Remaining: {self._format_duration(remaining)}")
            else:
                lines.append(f"  Remaining: OVER by {self._format_duration(abs(remaining))}")
            lines.append("")

        return "\n".join(lines)

    def get_carryover_tasks(self) -> list[Task]:
        """Get tasks that were scheduled but not completed (carryover)."""
        tasks = self.repo.list_tasks()
        return [
            t for t in tasks
            if t.completion < 100
            and t.horizon == Horizon.TODAY
        ]

    def set_first_task(self, task: Task) -> Task:
        """Mark a task as the first focus task for today."""
        task.scheduled_start = datetime.utcnow()
        return self.repo.update_task(task)

    def generate_recurrence_instances(self, parent_block: TimeBlock, days_ahead: int = 7) -> list[TimeBlock]:
        """Generate recurrence instances from today forward only (D-10).

        Args:
            parent_block: The template block with recurrence_rule and series_id set.
            days_ahead: Number of future days to generate instances for.

        Returns:
            List of created TimeBlock instances (persisted to repo).
        """
        if parent_block.recurrence_rule is None:
            return []
        today = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
        instances = []
        for i in range(1, days_ahead + 1):
            target_date = today + timedelta(days=i)
            if parent_block.recurrence_rule == "weekly" and i % 7 != 0:
                continue
            new_start = None
            if parent_block.start_time:
                new_start = target_date.replace(
                    hour=parent_block.start_time.hour,
                    minute=parent_block.start_time.minute,
                    second=0,
                    microsecond=0,
                )
            block = TimeBlock(
                task_id=parent_block.task_id,
                start_time=new_start,
                duration_minutes=parent_block.duration_minutes,
                series_id=parent_block.series_id,
            )
            self.repo.create_time_block(block)
            instances.append(block)
        return instances

    def fork_series(self, block: TimeBlock) -> str:
        """Fork a recurrence series from the given block forward (D-10 'edit this and future').

        Returns the new series_id.
        """
        from pb.domain.models import generate_internal_id
        new_series_id = generate_internal_id()
        series_blocks = self.repo.list_time_blocks_by_series(block.series_id)
        for b in series_blocks:
            if b.start_time and block.start_time and b.start_time >= block.start_time:
                b.series_id = new_series_id
                self.repo.update_time_block(b)
        return new_series_id

    # ------------------------------------------------------------------
    # Weekly planning (D-26, D-27)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_weekly_allocation(available_hours: float = 40.0) -> dict:
        """Compute 60/30/10 weekly allocation in minutes per D-26."""
        total_minutes = int(available_hours * 60)
        deep = int(total_minutes * 0.60)
        shallow = int(total_minutes * 0.30)
        buffer = total_minutes - deep - shallow  # remainder goes to buffer
        return {"deep": deep, "shallow": shallow, "buffer": buffer}

    def generate_weekly_plan(self, available_hours: float = 40.0) -> str:
        """Generate capacity-aware weekly plan per D-27.

        Pulls ranked tasks, groups by work_type, fills deep/shallow capacity,
        warns on overcapacity, preserves buffer.
        """
        allocation = self.compute_weekly_allocation(available_hours)
        tasks = [
            t for t in self.repo.list_tasks()
            if t.completion < 100
            and t.archived_at is None
        ]
        ranked = rank_tasks(tasks)

        lines = ["# Weekly Plan", ""]
        lines.append(f"Available: {available_hours}h")
        lines.append(
            f"  Deep work:    {allocation['deep'] // 60}h {allocation['deep'] % 60}m (60%)"
        )
        lines.append(
            f"  Shallow work: {allocation['shallow'] // 60}h {allocation['shallow'] % 60}m (30%)"
        )
        lines.append(
            f"  Buffer:       {allocation['buffer'] // 60}h {allocation['buffer'] % 60}m (10%)"
        )
        lines.append("")

        # Group tasks by work type
        deep_tasks = [
            t for t in ranked
            if t.work_type in (WorkType.DEEP.value, "study", "practice", None)
            and task_priority_score(t) is not None
        ]
        shallow_tasks = [
            t for t in ranked
            if t.work_type in (
                WorkType.SHALLOW.value,
                WorkType.ADMIN.value,
                WorkType.MEETING.value,
                WorkType.PLANNING.value,
            )
        ]
        unscored = [t for t in ranked if task_priority_score(t) is None]

        # Fill deep capacity
        deep_remaining = allocation["deep"]
        lines.append("## Deep Work")
        if deep_tasks:
            for t in deep_tasks:
                est = t.estimated_minutes or t.estimate_minutes or 60
                if deep_remaining <= 0:
                    break
                score = task_priority_score(t)
                score_str = f"{score:.1f}" if score is not None else "-"
                lines.append(f"  [{score_str}] {t.title} ({est}m)")
                deep_remaining -= est
        else:
            lines.append("  No scored deep work tasks.")
        lines.append("")

        # Fill shallow capacity
        shallow_remaining = allocation["shallow"]
        lines.append("## Shallow / Admin")
        if shallow_tasks:
            for t in shallow_tasks:
                est = t.estimated_minutes or t.estimate_minutes or 30
                if shallow_remaining <= 0:
                    break
                score = task_priority_score(t)
                score_str = f"{score:.1f}" if score is not None else "-"
                lines.append(f"  [{score_str}] {t.title} ({est}m)")
                shallow_remaining -= est
        else:
            lines.append("  No shallow/admin tasks.")
        lines.append("")

        # Capacity warning — uses 90% threshold to preserve buffer
        total_estimated = sum(
            (t.estimated_minutes or t.estimate_minutes or 60)
            for t in ranked
            if task_priority_score(t) is not None
        )
        total_available = int(available_hours * 60)
        if total_estimated > int(total_available * 0.9):
            over = total_estimated - int(total_available * 0.9)
            lines.append(
                f"WARNING: Estimated work exceeds 90% capacity by {over}m. "
                "Consider deferring lower-priority tasks."
            )
            lines.append("")

        # Unscored tasks
        if unscored:
            lines.append(f"## Unscored Tasks ({len(unscored)})")
            for t in unscored[:10]:
                lines.append(f"  - {t.title}")
            if len(unscored) > 10:
                lines.append(
                    f"  ... and {len(unscored) - 10} more. "
                    "Score them before generating the weekly plan."
                )
            lines.append("")

        # Unscheduled high priority
        high_pri = [
            t for t in ranked
            if task_priority_score(t) is not None
            and get_priority_action(task_priority_score(t)) == PriorityAction.SCHEDULE_FIRST
            and t.scheduled_date is None
        ]
        if high_pri:
            lines.append("## Unscheduled High-Priority")
            for t in high_pri:
                score = task_priority_score(t)
                lines.append(f"  [{score:.1f}] {t.title} — needs a scheduled date")
            lines.append("")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Daily planning (D-28, D-29)
    # ------------------------------------------------------------------

    def generate_daily_plan(
        self,
        energy_level: int = 3,
        focus_hours: float = 4.0,
        emphasis: str = "mixed",
        fixed_minutes: int = 0,
        budget_minutes: Optional[int] = None,
    ) -> str:
        """Generate energy-aware daily plan per D-28.

        Args:
            energy_level: Current energy 1-5
            focus_hours: Available deep focus hours
            emphasis: "deep", "shallow", "recovery", "mixed"
            fixed_minutes: Minutes already committed to meetings/fixed
            budget_minutes: Optional budget cap from --budget flag
        """
        tasks = [
            t for t in self.repo.list_tasks()
            if t.completion < 100
            and t.archived_at is None
        ]
        ranked = rank_tasks(tasks)

        focus_minutes = int(focus_hours * 60)
        available_minutes = focus_minutes - fixed_minutes

        # Filter by energy fit per D-29
        energy_info = ENERGY_MATCH_TABLE.get(energy_level, ENERGY_MATCH_TABLE[3])
        max_energy = energy_info["max_energy_required"]

        energy_fit = [
            t for t in ranked
            if (t.energy_required or 3) <= max_energy
            and task_priority_score(t) is not None
        ]
        all_scored = [t for t in ranked if task_priority_score(t) is not None]

        # Top 1: highest priority, energy-fit
        top1 = energy_fit[0] if energy_fit else (all_scored[0] if all_scored else None)

        # Top 3: top 3 energy-fit tasks
        top3 = energy_fit[:3] if energy_fit else all_scored[:3]

        # Batch list: shallow/admin/low-energy tasks
        batch = [
            t for t in ranked
            if t.work_type in (WorkType.SHALLOW.value, WorkType.ADMIN.value)
            or (t.energy_required is not None and t.energy_required <= 2)
        ]

        lines = [
            f"# Daily Plan: {datetime.utcnow().strftime('%Y-%m-%d')}",
            "",
        ]

        # Phase 3 D-10: Goal context banner before energy/Top1
        try:
            from pb.core.goal_reader import GoalReader, generate_goal_banner
            reader = GoalReader()
            active_goals = reader.read_active_goals()
            banner = generate_goal_banner(active_goals)
            if banner:
                lines.append(banner)
        except Exception:
            pass  # Non-fatal: missing goals do not break daily plan

        lines.append(
            f"Energy: {energy_level}/5  |  Focus: {focus_hours}h  |  Emphasis: {emphasis}"
        )
        lines.append(f"Best for: {energy_info['best_for']}")
        lines.append(f"Avoid: {energy_info['avoid']}")
        lines.append("")

        # Top 1
        lines.append("## Top 1 (makes the day successful)")
        if top1:
            score = task_priority_score(top1)
            score_str = f"[{score:.1f}]" if score is not None else ""
            lines.append(f"  {score_str} {top1.title}")
        else:
            lines.append("  No scored tasks available.")
        lines.append("")

        # Top 3
        lines.append("## Top 3 (meaningful outcomes)")
        if top3:
            for t in top3:
                score = task_priority_score(t)
                score_str = f"[{score:.1f}]" if score is not None else ""
                est = t.estimated_minutes or t.estimate_minutes or "?"
                lines.append(f"  {score_str} {t.title} ({est}m)")
        else:
            lines.append("  No tasks to recommend.")
        lines.append("")

        # Batch list
        lines.append("## Batch List (low-cognition / admin windows)")
        if batch:
            for t in batch[:5]:
                est = t.estimated_minutes or t.estimate_minutes or "?"
                lines.append(f"  - {t.title} ({est}m)")
        else:
            lines.append("  No batch tasks available.")
        lines.append("")

        # Suggested blocks per D-28
        lines.append("## Suggested Blocks")
        block_minutes = 90 if emphasis == "deep" else (75 if emphasis == "mixed" else 45)
        break_minutes = 15
        remaining = available_minutes
        block_num = 0

        for t in (top3 or []):
            if remaining <= 0:
                break
            est = t.estimated_minutes or t.estimate_minutes or block_minutes
            actual_block = min(est, remaining, block_minutes)
            block_num += 1
            lines.append(
                f"  Block {block_num}: {t.title} ({actual_block}m) + {break_minutes}m break"
            )
            remaining -= (actual_block + break_minutes)

        if remaining > 0 and batch:
            block_num += 1
            lines.append(f"  Block {block_num}: Batch window ({min(remaining, 45)}m)")
        lines.append("")

        # Capacity check
        total_top3_est = sum(
            t.estimated_minutes or t.estimate_minutes or 60
            for t in (top3 or [])
        )
        if total_top3_est > available_minutes:
            over = total_top3_est - available_minutes
            lines.append(
                f"WARNING: Top 3 tasks exceed available focus by {over}m. "
                "Consider dropping one."
            )
            lines.append("")

        # Budget section
        if budget_minutes is not None:
            all_blocks = self.repo.list_time_blocks_created_for_date(datetime.utcnow())
            committed = sum(b.duration_minutes for b in all_blocks)
            remaining_budget = budget_minutes - committed
            lines.append("## Budget")
            lines.append(f"  Budget:    {self._format_duration(budget_minutes)}")
            lines.append(f"  Committed: {self._format_duration(committed)}")
            if remaining_budget >= 0:
                lines.append(f"  Remaining: {self._format_duration(remaining_budget)}")
            else:
                lines.append(
                    f"  Remaining: OVER by {self._format_duration(abs(remaining_budget))}"
                )
            lines.append("")

        active = self.get_active_tasks()
        if active:
            lines.append(f"## Active Tasks ({len(active)})")
            for task in active[:5]:
                lines.append(f"  - {task.title} [{task.completion}%]")
            if len(active) > 5:
                lines.append(f"  ... and {len(active) - 5} more")
            lines.append("")

        return "\n".join(lines)
