# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Planning and scheduling service.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.planner, pb.core.priority in later phases.
"""
from __future__ import annotations
from datetime import date
from typing import Optional, Protocol
from pb.core.models import Task, TimeBlock
from pb.core.base import BaseService


class PlanRepo(Protocol):
    """Protocol for plan/time-block persistence."""
    def get_time_blocks(self, plan_date: date) -> list[TimeBlock]: ...
    def create_time_block(self, block: TimeBlock) -> TimeBlock: ...
    def delete_time_block(self, block_id: str) -> None: ...


class PlanService(BaseService):
    """Manages daily planning, time blocks, and task prioritization.

    Constructor takes explicit deps per D-05. Composes with TaskService
    and GoalsService for priority ranking and alignment.
    """
    def __init__(self, repo: PlanRepo,
                 task_service: Optional[object] = None,
                 goals_service: Optional[object] = None):
        super().__init__()
        self.repo = repo
        self.task_service = task_service
        self.goals_service = goals_service

    def plan_day(self, energy_level: int = 3) -> list[TimeBlock]:
        raise NotImplementedError

    def get_time_blocks(self, plan_date: date | None = None) -> list[TimeBlock]:
        raise NotImplementedError

    def add_time_block(self, task_id: str, start_time: str,
                       duration_minutes: int) -> TimeBlock:
        raise NotImplementedError

    def remove_time_block(self, block_id: str) -> None:
        raise NotImplementedError

    def rank_tasks(self, tasks: list[Task],
                   energy_level: int = 3) -> list[Task]:
        raise NotImplementedError

    def suggest_schedule(self, tasks: list[Task]) -> list[TimeBlock]:
        raise NotImplementedError
