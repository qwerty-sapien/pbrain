# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared interactive task scoring helpers."""

from __future__ import annotations

from typing import Optional

import typer
from rich.markup import escape

from pb.cli.console import get_console, get_err_console
from pb.cli.helpers import prompt_text
from pb.core.learning_metadata import parse_learning_task_metadata


PLANNING_SCORE_FIELDS = (
    "impact",
    "urgency_score",
    "effort",
)


def task_missing_planning_scores(task) -> list[str]:
    """Return the planning-critical score fields still missing on a task."""
    missing: list[str] = []
    for field_name in PLANNING_SCORE_FIELDS:
        value = getattr(task, field_name, None)
        if value is None or value == "":
            missing.append(field_name)
    return missing


def task_is_planning_scored(task) -> bool:
    """Return True when all planning-critical fields are present."""
    return not task_missing_planning_scores(task)


def score_task_interactively(repo, task) -> bool:
    """Score a task with 3 compact questions; derive the rest."""
    console = get_console()
    meta = parse_learning_task_metadata(task)

    console.print(f"\n[header]Scoring[/] {escape(task.title)}")
    if getattr(task, "description", "").strip():
        console.print(f"[dim]{escape(task.description.splitlines()[0][:140])}[/]")

    try:
        task.impact = _prompt_1_to_5(
            "Impact", getattr(task, "impact", None),
            "How much does completing this move the needle?",
        )
        task.urgency_score = _prompt_1_to_5(
            "Urgency", getattr(task, "urgency_score", None),
            "How time-sensitive is this?",
        )
        task.effort = _prompt_1_to_5(
            "Effort", getattr(task, "effort", None),
            "How much work is this? (1=quick, 5=large)",
        )
    except (typer.Abort, EOFError, KeyboardInterrupt):
        get_err_console().print("[warn]Scoring cancelled.[/]")
        return False

    task.strategic_value = task.impact
    task.important = task.impact >= 3
    task.urgent = task.urgency_score >= 3
    task.energy_required = task.effort
    task.work_type = _infer_work_type(meta)

    repo.update_task(task)
    console.print(f"[success]Scored:[/] {escape(task.title)}")
    return True


def _infer_work_type(meta) -> str:
    branch = (getattr(meta, "branch", "") or "").lower()
    if branch == "study":
        return "study"
    if branch in {"practise", "practice"}:
        return "practice"
    return "deep"


def _prompt_1_to_5(label: str, default: Optional[int], help_text: str) -> int:
    default_text = str(default) if default is not None else "3"
    while True:
        raw = prompt_text(f"{label} (1-5): {help_text}", default=default_text)
        value = raw.strip() or default_text
        if value.isdigit() and 1 <= int(value) <= 5:
            return int(value)
        get_err_console().print("[warn]Enter a number from 1 to 5.[/]")
