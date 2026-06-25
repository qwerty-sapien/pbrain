# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Thin repo adapter -- satisfies GoalRepo Protocol via monolithic Repository. ARCH-07."""
from __future__ import annotations

from pb.core.models import GoalArc
from pb.storage.repository import Repository


class GoalRepoAdapter:
    """Wraps Repository to satisfy GoalRepo Protocol.

    No SQL here. All persistence delegated to Repository.
    Real SQL extraction happens in a later phase.
    """

    def __init__(self, repo: Repository):
        self._repo = repo

    def get_goal(self, goal_id: str) -> GoalArc | None:
        # Repository method is get_goal_arc, not get_goal
        return self._repo.get_goal_arc(goal_id)

    def list_goals(self, active_only: bool = True) -> list[GoalArc]:
        # Repository.list_goal_arcs uses status param, not active_only
        status = "active" if active_only else None
        return self._repo.list_goal_arcs(status=status)

    def create_goal(self, goal: GoalArc) -> GoalArc:
        return self._repo.create_goal_arc(goal)

    def update_goal(self, goal: GoalArc) -> GoalArc:
        # Repository may not have update_goal_arc; implement as delete+create or raise
        # For Phase 22 this is a thin wrapper; if update_goal_arc doesn't exist,
        # implement via fields update on the stored record
        if hasattr(self._repo, 'update_goal_arc'):
            return self._repo.update_goal_arc(goal)
        # Fallback: update via raw SQL is NOT allowed per ARCH-07.
        # Return as-is with a NotImplementedError marker for later phases.
        raise NotImplementedError("Repository.update_goal_arc not yet available")
