# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Priority scoring, Eisenhower classification, and task ranking.

Implements D-22 through D-25 from SPEC-DRAFT.
"""

from __future__ import annotations

from typing import Optional

from pb.domain.enums import EisenhowerClass, PriorityAction
from pb.domain.models import Task


def compute_priority_score(
    impact: int, urgency: int, strategic_value: int, effort: int
) -> float:
    """Compute priority score per D-23: (impact + urgency + strategic_value) / effort."""
    if effort <= 0:
        raise ValueError("Effort must be positive")
    return (impact + urgency + strategic_value) / effort


def classify_eisenhower(important: bool, urgent: bool) -> EisenhowerClass:
    """Classify task into Eisenhower quadrant per D-24."""
    if important and urgent:
        return EisenhowerClass.DO_TODAY
    elif important and not urgent:
        return EisenhowerClass.SCHEDULE_DEEP_WORK
    elif not important and urgent:
        return EisenhowerClass.BATCH_DELEGATE_OR_AUTOMATE
    else:
        return EisenhowerClass.DELETE_OR_DEFER


def get_priority_action(score: float) -> PriorityAction:
    """Determine priority action from score per D-25."""
    if score >= 4.0:
        return PriorityAction.SCHEDULE_FIRST
    elif score >= 2.5:
        return PriorityAction.SCHEDULE_IF_CAPACITY
    elif score >= 1.5:
        return PriorityAction.BATCH_DELEGATE_SIMPLIFY
    else:
        return PriorityAction.DROP_OR_DEFER


def task_priority_score(task: Task) -> Optional[float]:
    """Compute priority score for a task, or None if unscored."""
    if any(v is None for v in [task.impact, task.urgency_score, task.strategic_value, task.effort]):
        return None
    return compute_priority_score(task.impact, task.urgency_score, task.strategic_value, task.effort)


def task_eisenhower(task: Task) -> Optional[EisenhowerClass]:
    """Classify task into Eisenhower quadrant, or None if unscored."""
    if task.important is None or task.urgent is None:
        return None
    return classify_eisenhower(task.important, task.urgent)


def rank_tasks(tasks: list[Task]) -> list[Task]:
    """Rank tasks by priority score descending. Unscored tasks go last."""
    def sort_key(t: Task) -> float:
        score = task_priority_score(t)
        return score if score is not None else -1.0
    return sorted(tasks, key=sort_key, reverse=True)
