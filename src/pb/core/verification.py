# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Pre-task verification quiz engine for priority learning tasks."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class VerificationResult:
    understanding_pct: float = 0.0
    retention_pct: float = 0.0
    passed: bool = False
    spillover_weight: float = 0.0
    quiz_duration_minutes: int = 0


@dataclass
class VerificationContext:
    task_id: str
    prerequisite_task_ids: list[str] = field(default_factory=list)
    domain: str = ""
    time_limit_minutes: int = 10


def needs_verification(repo, task) -> bool:
    """Check if a task has unverified priority prerequisites."""
    meta_mod = __import__("pb.core.learning_metadata", fromlist=["parse_learning_task_metadata"])
    meta = meta_mod.parse_learning_task_metadata(task)
    if not getattr(meta, "depends_on", None):
        return False
    for dep_id in meta.depends_on:
        dep_task = repo.get_task(dep_id)
        if dep_task is None:
            continue
        verification = _get_verification_status(repo, dep_task)
        if verification is None or not verification.passed:
            return True
    return False


def _get_verification_status(repo, task) -> Optional[VerificationResult]:
    """Retrieve stored verification result for a task."""
    generated = getattr(task, "generated_names", {}) or {}
    verification_data = generated.get("verification")
    if verification_data is None:
        return None
    return VerificationResult(
        understanding_pct=verification_data.get("understanding_pct", 0),
        retention_pct=verification_data.get("retention_pct", 0),
        passed=verification_data.get("passed", False),
        spillover_weight=verification_data.get("spillover_weight", 0),
    )


def compute_spillover_weight(understanding: float, retention: float) -> float:
    """Determine how much prerequisite content leaks into downstream quizzes.

    Both >95% → 0 spillover. Lower values → proportionally more spillover.
    """
    if understanding >= 95 and retention >= 95:
        return 0.0
    avg = (understanding + retention) / 2
    return max(0.0, min(1.0, (95 - avg) / 95))


def estimate_quiz_duration(repo, prerequisite_task_ids: list[str]) -> int:
    """Estimate quiz duration in minutes (3-15) based on prerequisite complexity."""
    count = len(prerequisite_task_ids)
    if count <= 1:
        return 3
    if count <= 3:
        return 7
    return min(15, 5 + count * 2)


def store_verification(repo, task, result: VerificationResult) -> None:
    """Persist verification result on the task."""
    generated = dict(getattr(task, "generated_names", {}) or {})
    generated["verification"] = {
        "understanding_pct": result.understanding_pct,
        "retention_pct": result.retention_pct,
        "passed": result.passed,
        "spillover_weight": result.spillover_weight,
    }
    task.generated_names = generated
    repo.update_task(task)
