# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Thin repo adapter -- satisfies TaskRepo Protocol via monolithic Repository. ARCH-07."""
from __future__ import annotations

from pb.core.models import Task
from pb.storage.repository import Repository


class TaskRepoAdapter:
    """Wraps Repository to satisfy TaskRepo Protocol.

    No SQL here. All persistence delegated to Repository.
    Real SQL extraction happens in a later phase.
    """

    def __init__(self, repo: Repository):
        self._repo = repo

    def get_task(self, task_id: str) -> Task | None:
        return self._repo.get_task(task_id)

    def list_tasks(self, active_only: bool = True) -> list[Task]:
        # Repository.list_tasks uses include_archived param (inverted sense)
        return self._repo.list_tasks(include_archived=not active_only)

    def create_task(self, task: Task) -> Task:
        return self._repo.create_task(task)

    def update_task(self, task: Task) -> Task:
        return self._repo.update_task(task)

    def delete_task(self, task_id: str) -> None:
        # Repository uses hard_delete_task (returns bool); Protocol expects None
        self._repo.hard_delete_task(task_id)
