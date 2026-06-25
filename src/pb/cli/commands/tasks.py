# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Task management commands - archive/restore/cancel/delete/list."""

from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table

from pb.core.entity_refs import display_ref
from pb.cli.console import get_console, get_err_console
from pb.cli.helpers import pick_task
from pb.cli.task_scoring import score_task_interactively
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.matching import MatchCandidate, resolve_strict_match
from pb.core.sessions import SessionManager
from pb.domain.enums import Horizon
from pb.domain.exceptions import ExitCode
from pb.storage.repository import Repository

app = typer.Typer(invoke_without_command=True, no_args_is_help=False)


@app.callback(invoke_without_command=True)
def tasks_callback(ctx: typer.Context):
    """List all active tasks."""
    if ctx.invoked_subcommand is not None:
        return
    list_ranked_command(today_only=False)


def _find_task_by_prefix(repo: Repository, task_id: str, include_archived: bool = False):
    """Find a task by ID or prefix."""
    return repo.resolve_task_ref(task_id, include_archived=include_archived)


@app.command("archive")
def archive_command(
    task_id: str = typer.Argument(..., help="Task ID or prefix to archive"),
):
    """Archive a task (soft delete). Hidden from default views."""
    repo = Repository()

    task = _find_task_by_prefix(repo, task_id)
    if task is None:
        err_console = get_err_console()
        err_console.print(f"[error]Task not found: {escape(task_id)}[/]")
        raise typer.Exit(code=1)

    if task.archived_at is not None:
        err_console = get_err_console()
        err_console.print(f"[error]Task already archived: {escape(task.title)}[/]")
        raise typer.Exit(code=1)

    archived = repo.archive_task(task.id)
    if archived:
        console = get_console()
        console.print(f"[success]Archived: {escape(task.title)}[/]")
    else:
        err_console = get_err_console()
        err_console.print("[error]Failed to archive task[/]")
        raise typer.Exit(code=1)


@app.command("restore")
def restore_command(
    task_id: str = typer.Argument(..., help="Task ID or prefix to restore"),
):
    """Restore an archived task. Per D-04."""
    repo = Repository()

    # Must search including archived tasks
    task = _find_task_by_prefix(repo, task_id, include_archived=True)
    if task is None:
        err_console = get_err_console()
        err_console.print(f"[error]Task not found: {escape(task_id)}[/]")
        raise typer.Exit(code=1)

    if task.archived_at is None:
        err_console = get_err_console()
        err_console.print(f"[error]Task is not archived: {escape(task.title)}[/]")
        raise typer.Exit(code=1)

    restored = repo.restore_task(task.id)
    if restored:
        console = get_console()
        console.print(f"[success]Restored: {escape(task.title)}[/]")
    else:
        err_console = get_err_console()
        err_console.print("[error]Failed to restore task[/]")
        raise typer.Exit(code=1)


@app.command("cancel")
def cancel_command():
    """Cancel the active session. Discards it entirely -- no graph note, no time tracked."""
    repo = Repository()
    manager = SessionManager(repo)
    session = manager.discard_session()
    if session is None:
        err_console = get_err_console()
        err_console.print("[error]No active session to cancel.[/]")
        raise typer.Exit(code=ExitCode.NOT_FOUND)
    task = repo.get_task(session.task_id)
    task_title = task.title if task else "Unknown"
    console = get_console()
    console.print(f"[success]Cancelled: {escape(task_title)}[/]")


@app.command("delete")
def delete_command():
    """Delete tasks. Hard-deletes tasks without sessions; auto-archives tasks with sessions."""
    repo = Repository()
    tasks = repo.list_tasks()  # All non-archived active tasks
    if not tasks:
        err_console = get_err_console()
        err_console.print("[error]No tasks to delete.[/]")
        raise typer.Exit(code=ExitCode.NOT_FOUND)

    # D-18: Use numbered picker with multi-select and NLP search
    selected = pick_task(tasks, prompt_text="Select tasks to delete", multi_select=True)
    if not selected:
        raise typer.Exit(code=ExitCode.SUCCESS)

    deleted = []
    archived = []
    for task in selected:
        # T-02-01: parameterized SQL in hard_delete_task; task_id validated by repo lookup
        was_hard_deleted = repo.hard_delete_task(task.id)
        if was_hard_deleted:
            deleted.append(task.title)
        else:
            archived.append(task.title)

    console = get_console()
    parts = []
    if deleted:
        parts.append(f"Deleted: {', '.join(escape(t) for t in deleted)}")
    if archived:
        parts.append(f"Archived (has sessions): {', '.join(escape(t) for t in archived)}")
    console.print("[success]" + ". ".join(parts) + ".[/]")


def _strict_find_task(repo: Repository, query: str):
    """Find a task using exact/prefix match first, then strict fuzzy match."""
    task = _find_task_by_prefix(repo, query)
    if task is not None:
        return task
    tasks = [t for t in repo.list_tasks(include_archived=True) if t.archived_at is None]
    candidates: list[MatchCandidate] = []
    for task in tasks:
        meta = parse_learning_task_metadata(task)
        goal_titles = []
        for goal_id in getattr(task, "linked_goal_arc_ids", []) or []:
            goal = repo.get_goal_arc(goal_id)
            if goal is not None:
                goal_titles.append(goal.title)
        candidates.append(
            MatchCandidate(
                key=task.id,
                label=task.title,
                text=" | ".join(
                    part
                    for part in [
                        task.title,
                        task.description,
                        meta.scope,
                        meta.domain,
                        meta.goal_project_title,
                        ", ".join(goal_titles),
                    ]
                    if part
                ),
            )
        )
    result = resolve_strict_match(query, candidates)
    if result.accepted and result.matched_index is not None:
        return tasks[result.matched_index]
    if result.suggestions:
        err_console = get_err_console()
        err_console.print("[error]I don't know which task you mean.[/]")
        for suggestion in result.suggestions[:3]:
            err_console.print(f"  - {tasks[suggestion].title}")
        raise typer.Exit(code=1)
    return None


@app.command("score", hidden=True)
def score_command(
    task_id: Optional[str] = typer.Argument(None, help="Task ID (omit for picker)"),
):
    """Score a task's priority dimensions interactively (D-22)."""
    repo = Repository()

    if task_id:
        task = _strict_find_task(repo, task_id)
        if task is None:
            err_console = get_err_console()
            err_console.print(f"[error]Task not found: {escape(task_id)}[/]")
            raise typer.Exit(code=1)
    else:
        tasks = [t for t in repo.list_tasks() if t.completion < 100 and t.archived_at is None]
        if not tasks:
            err_console = get_err_console()
            err_console.print("[error]No tasks to score.[/]")
            raise typer.Exit(code=1)
        selected = pick_task(tasks, "Select task to score")
        if not selected:
            raise typer.Exit(code=0)
        task = selected[0]

    if not score_task_interactively(repo, task):
        raise typer.Exit(code=0)


@app.command("list")
def list_ranked_command(
    today_only: bool = typer.Option(False, "--today", help="Show only today's tasks"),
):
    """List tasks in priority order (D-22 ranked view)."""
    repo = Repository()

    if today_only:
        tasks = [
            t for t in repo.list_tasks()
            if t.horizon == Horizon.TODAY and t.completion < 100
        ]
    else:
        tasks = [
            t for t in repo.list_tasks()
            if t.completion < 100 and t.archived_at is None
        ]

    if not tasks:
        console = get_console()
        console.print("[dim]No tasks found.[/]")
        return

    from pb.core.priority import rank_tasks, task_priority_score, task_eisenhower, get_priority_action

    ranked = rank_tasks(tasks)

    console = get_console()
    t = Table(show_header=True, header_style="table.header",
              show_edge=False, show_lines=False, pad_edge=False, box=None)
    t.add_column("SCORE", justify="right", no_wrap=True)
    t.add_column("EISENHOWER", no_wrap=True)
    t.add_column("ACTION", no_wrap=True)
    t.add_column("ENERGY", justify="right", no_wrap=True)
    t.add_column("TASK")

    for task in ranked:
        score = task_priority_score(task)
        eisen = task_eisenhower(task)
        score_str = f"{score:.1f}" if score is not None else "-"
        eisen_str = eisen.value if eisen is not None else "unscored"
        action_str = get_priority_action(score).value if score is not None else "-"
        energy_str = str(task.energy_required) if task.energy_required else "-"
        t.add_row(
            f"[value.med]{score_str}[/]",
            escape(eisen_str),
            escape(action_str),
            energy_str,
            escape(task.title),
        )
    console.print(t)


@app.command("pause")
def list_paused_command():
    """List paused/postponed tasks."""
    repo = Repository()
    from pb.domain.enums import TaskState
    tasks = [t for t in repo.list_tasks() if t.state == TaskState.PAUSED]
    if not tasks:
        get_console().print("[dim]No paused tasks.[/]")
        return
    table = Table(show_header=True, show_edge=False, show_lines=False, pad_edge=False, box=None)
    table.add_column("ID", style="dim")
    table.add_column("Title")
    table.add_column("Until", style="dim")
    for t in tasks:
        until = t.paused_until.strftime("%Y-%m-%d") if t.paused_until else "indefinite"
        table.add_row(display_ref(t, "task"), t.title, until)
    get_console().print(table)
