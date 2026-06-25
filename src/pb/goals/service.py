# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Goal alignment and tracking service.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.goal_reader,
pb.core.alignment in later phases.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional, Protocol
from pb.core.models import GoalArc, Track
from pb.core.base import BaseService


class GoalRepo(Protocol):
    """Protocol for goal persistence."""
    def get_goal(self, goal_id: str) -> GoalArc | None: ...
    def list_goals(self, active_only: bool = True) -> list[GoalArc]: ...
    def create_goal(self, goal: GoalArc) -> GoalArc: ...
    def update_goal(self, goal: GoalArc) -> GoalArc: ...


class GoalsService(BaseService):
    """Manages goal arcs, tracks, and alignment checking.

    Takes GoalRepo for persistence and optional vault_path for
    _state.md / _index.md goal stubs in the vault.
    """
    def __init__(self, repo: GoalRepo,
                 vault_path: Path | None = None):
        super().__init__()
        self.repo = repo
        self.vault_path = vault_path

    def list_goals(self, active_only: bool = True) -> list[GoalArc]:
        raise NotImplementedError

    def create_goal(self, title: str, horizon: str, **kwargs) -> GoalArc:
        raise NotImplementedError

    def get_goal_banner(self) -> str:
        raise NotImplementedError

    def check_alignment(self, task_id: str) -> float:
        raise NotImplementedError

    def list_tracks(self) -> list[Track]:
        raise NotImplementedError

    def link_task_to_goal(self, task_id: str, goal_id: str) -> None:
        raise NotImplementedError

    def goal_report(self, goal_id: str | None = None) -> dict:
        raise NotImplementedError
