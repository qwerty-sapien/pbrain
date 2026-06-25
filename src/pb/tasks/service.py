# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Task and skill management service.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.sessions/skills in later phases.
"""
from __future__ import annotations
from typing import Optional, Protocol
from pb.core.models import Task
from pb.core.base import BaseService, LoggableMixin


class TaskRepo(Protocol):
    """Protocol for task persistence. Existing Repository satisfies this structurally."""

    def get_task(self, task_id: str) -> Task | None: ...
    def list_tasks(self, active_only: bool = True) -> list[Task]: ...
    def create_task(self, task: Task) -> Task: ...
    def update_task(self, task: Task) -> Task: ...
    def delete_task(self, task_id: str) -> None: ...


class TaskService(BaseService, LoggableMixin):
    """Manages task lifecycle and skill links.

    Constructor takes explicit deps per D-05 -- no singletons, no global state.
    """

    def __init__(self, repo: TaskRepo, ai: Optional[object] = None):
        super().__init__()
        self.repo = repo
        self.ai = ai

    def list_tasks(self, active_only: bool = True) -> list[Task]:
        raise NotImplementedError

    def create_task(self, title: str, **kwargs) -> Task:
        raise NotImplementedError

    def get_task(self, task_id: str) -> Task | None:
        raise NotImplementedError

    def update_task(self, task_id: str, **kwargs) -> Task:
        raise NotImplementedError

    def pause_task(self, task_id: str, days: int = 1) -> Task:
        raise NotImplementedError

    def resume_task(self, task_id: str) -> Task:
        raise NotImplementedError

    def complete_task(self, task_id: str, outcome: str = "done") -> Task:
        raise NotImplementedError

    def cancel_task(self, task_id: str) -> Task:
        raise NotImplementedError

    def archive_task(self, task_id: str) -> Task:
        raise NotImplementedError

    def restore_task(self, task_id: str) -> Task:
        raise NotImplementedError

    def delete_task(self, task_id: str) -> None:
        raise NotImplementedError

    def list_skills(self, task_id: str | None = None) -> list:
        raise NotImplementedError

    def link_skill(self, task_id: str, skill_name: str) -> None:
        raise NotImplementedError

    def score_task(self, task_id: str) -> float:
        raise NotImplementedError
