# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Capture command - add tasks."""

from typing import Optional

import typer

from pb.core.naming import stored_display_title
from pb.domain.enums import Horizon, TaskState
from pb.domain.models import Task, generate_slug
from pb.storage.repository import Repository


app = typer.Typer(no_args_is_help=False)


def create_task(text: str, horizon: str = "today", skill: str = "", track: str = "", goal: str = "") -> None:
    """Core task creation logic used by both capture and add commands."""
    try:
        horizon_enum = Horizon(horizon)
    except ValueError:
        typer.echo(f"Invalid horizon: {horizon}. Use: today, week, month", err=True)
        raise typer.Exit(code=1)

    repo = Repository()
    task = Task(
        title=text,
        id=generate_slug(text),
        state=TaskState.ACTIVE,
        horizon=horizon_enum,
    )
    repo.create_task(task)

    if skill:
        from pb.core.skill_links import insert_skill_link
        from pb.core.skills import SkillManager
        skill_names = [s.strip() for s in skill.split(",") if s.strip()]
        _skill_mgr = SkillManager()
        for s in skill_names:
            insert_skill_link(task.id, s, source="pre-tag")
            _skill_mgr.create_skill(s)

    typer.echo(f"Added: {stored_display_title(task) or task.title}")


@app.callback(invoke_without_command=True)
def capture_task(
    ctx: typer.Context,
    words: Optional[list[str]] = typer.Argument(None, help="Task text to capture (no quotes needed)"),
    horizon: str = typer.Option("today", "--horizon", "-h", help="Horizon: today|week|month"),
    skill: str = typer.Option("", "--skill", help="Pre-tag skill(s), comma-separated"),
):
    """Add a new task. No quotes needed for multi-word text.

    Tasks are immediately available for `pb start`.
    """
    if ctx.invoked_subcommand is not None:
        return

    if not words:
        typer.echo("Usage: pb capture <text>", err=True)
        raise typer.Exit(code=1)

    text = " ".join(words)
    create_task(text, horizon, skill=skill)
