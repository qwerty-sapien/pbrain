# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Session history commands -- pb session group."""
from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table

from pb.core.entity_refs import display_ref

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def session_callback(ctx: typer.Context):
    """Session history and management."""
    if ctx.invoked_subcommand is not None:
        return
    # Default: show recent sessions
    list_sessions_command()


@app.command("list")
def list_sessions_command(
    limit: int = typer.Option(10, "--limit", "-n", help="Number of sessions to show"),
):
    """List recent sessions."""
    from pb.cli.console import get_console
    from pb.storage.repository import Repository
    from datetime import datetime, timedelta

    repo = Repository()
    start = datetime.utcnow() - timedelta(days=30)
    end = datetime.utcnow()
    sessions = repo.list_sessions_in_range(start, end)
    sessions = sessions[:limit]

    console = get_console()
    if not sessions:
        console.print("[dim]No recent sessions.[/]")
        return

    t = Table(show_header=True, show_edge=False, show_lines=False, pad_edge=False, box=None)
    t.add_column("Date", style="dim")
    t.add_column("Task")
    t.add_column("Duration", justify="right")
    t.add_column("Outcome", style="dim")

    for s in sessions:
        task = repo.get_task(s.task_id)
        task_title = escape(task.title) if task else display_ref(s, "session")
        date_str = s.start_at.strftime("%Y-%m-%d %H:%M") if s.start_at else "?"
        dur = ""
        if s.start_at and s.end_at:
            mins = int((s.end_at - s.start_at).total_seconds() / 60)
            dur = f"{mins}m"
        outcome = s.actual_outcome or ""
        t.add_row(date_str, task_title, dur, outcome)

    console.print(t)


@app.command("show")
def show_session_command(
    session_id: str = typer.Argument(..., help="Session ID or prefix"),
):
    """Show details of a specific session."""
    from pb.cli.console import get_console, get_err_console
    from pb.storage.repository import Repository
    repo = Repository()
    console = get_console()
    match = repo.resolve_session_ref(session_id)
    if not match:
        get_err_console().print(f"[error]Session not found: {escape(session_id)}[/]")
        raise typer.Exit(code=1)

    task = repo.get_task(match.task_id)
    console.print(f"[header]Session:[/] {display_ref(match, 'session', parent_ref=display_ref(task, 'task') if task else '')}")
    console.print(f"  Task: {escape(task.title) if task else display_ref(match, 'session')}")
    if match.start_at:
        console.print(f"  Started: {match.start_at.strftime('%Y-%m-%d %H:%M')}")
    if match.end_at:
        console.print(f"  Ended: {match.end_at.strftime('%Y-%m-%d %H:%M')}")
    if match.start_at and match.end_at:
        mins = int((match.end_at - match.start_at).total_seconds() / 60)
        console.print(f"  Duration: {mins}m")
    if match.actual_outcome:
        console.print(f"  Outcome: {match.actual_outcome}")
    if match.intended_outcome:
        console.print(f"  Intended: {escape(match.intended_outcome)}")


@app.command("redo")
def redo_session_command(
    task_id: Optional[str] = typer.Argument(None, help="Task ID to reopen"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration"),
):
    """Reopen a finished task and start a new session (relocated from top-level)."""
    from pb.cli.commands.execute import redo_task
    redo_task(task_id, duration)


@app.command("resume")
def resume_session_command(
    task_id: Optional[str] = typer.Argument(None, help="Task ID to resume"),
):
    """Resume a paused task's session (relocated from top-level)."""
    from pb.cli.commands.execute import resume_task
    resume_task(task_id)
