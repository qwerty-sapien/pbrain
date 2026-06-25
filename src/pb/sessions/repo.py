# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Thin repo adapter -- satisfies SessionRepo Protocol via monolithic Repository. ARCH-07."""
from __future__ import annotations

from pb.core.models import Session, Task
from pb.storage.repository import Repository


class SessionRepoAdapter:
    """Wraps Repository to satisfy SessionRepo Protocol.

    No SQL here. All persistence delegated to Repository.
    Real SQL extraction happens in a later phase.
    """

    def __init__(self, repo: Repository):
        self._repo = repo

    def get_active_session(self) -> Session | None:
        return self._repo.get_active_session()

    def create_session(self, session: Session) -> Session:
        return self._repo.create_session(session)

    def update_session(self, session: Session) -> Session:
        return self._repo.update_session(session)

    def delete_session(self, session_id: str) -> bool:
        return self._repo.delete_session(session_id)

    def delete_sessions_for_task(self, task_id: str) -> int:
        return self._repo.delete_sessions_for_task(task_id)

    def list_sessions(self, task_id: str | None = None) -> list[Session]:
        if task_id:
            return self._repo.list_sessions_for_task(task_id)
        # Repository has no bare list_sessions(); use list_sessions_in_range with wide window
        from datetime import datetime, timedelta
        start = datetime.utcnow() - timedelta(days=365)
        end = datetime.utcnow()
        return self._repo.list_sessions_in_range(start, end)

    def get_task(self, task_id: str) -> Task | None:
        return self._repo.get_task(task_id)

    def update_task(self, task: Task) -> None:
        self._repo.update_task(task)

    def force_delete_task(self, task_id: str) -> bool:
        return self._repo.force_delete_task(task_id)

    def list_sessions_for_task(self, task_id: str) -> list[Session]:
        return self._repo.list_sessions_for_task(task_id)

    def delete_time_blocks_for_task(self, task_id: str) -> int:
        return self._repo.delete_time_blocks_for_task(task_id)

    def delete_generation_provenance(
        self,
        *,
        artifact_kind: str | None = None,
        artifact_id: str | None = None,
    ) -> int:
        return self._repo.delete_generation_provenance(
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
        )

    def resume_pause_interval(self, session_id: str) -> None:
        self._repo.resume_pause_interval(session_id)

    def list_time_blocks_for_date(self, dt) -> list:
        return self._repo.list_time_blocks_for_date(dt)
