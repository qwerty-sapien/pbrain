# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Immediate next-action recommendations and reminder queue handling."""

from __future__ import annotations

import sys
from datetime import timedelta
from typing import Optional

import typer

from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console, get_err_console
from pb.cli.pickers import pick_single_choice
from pb.core.agent_weights import record_agent_weight_event
from pb.core.action_routing import build_next_candidates
from pb.core.goal_roadmaps import ensure_goal_seed_tasks
from pb.core.models import ActionReminder, utc_now
from pb.core.timer import schedule_actionable_notification

app = typer.Typer(no_args_is_help=False)


def _verbose_mode(ctx: typer.Context) -> bool:
    find_root = getattr(ctx, "find_root", None)
    root = find_root() if callable(find_root) else ctx
    return bool(getattr(root, "obj", {}) and root.obj.get("verbose"))


def _picker_options(candidates) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    options = [(item.backing_command, item.human_label) for item in candidates]
    details = [item.short_reason for item in candidates]
    verbose_labels = [f"{item.human_label}  ({item.backing_command})" for item in candidates]
    return options, details, verbose_labels


def _schedule_reminder(repo, candidate, minutes: int) -> ActionReminder:
    reminder = ActionReminder(
        title=candidate.human_label,
        message=candidate.short_reason,
        target_command=candidate.backing_command,
        remind_at=utc_now() + timedelta(minutes=minutes),
        source_kind="next",
    )
    repo.create_action_reminder(reminder)
    schedule_actionable_notification(
        title=reminder.title,
        message=reminder.message,
        execute=f"pb next --reminder {reminder.id}",
        delay_minutes=minutes,
    )
    return reminder


def _record_candidate_selection(candidate) -> None:
    """Persist scorer events for weighted `pb next` candidates."""
    if getattr(candidate, "source", "") != "commitment":
        return
    agent_id = getattr(candidate, "agent_id", "") or ""
    if not agent_id:
        return
    try:
        record_agent_weight_event(
            agent_id,
            "commitment_followup_selected",
            source_kind="human",
            metadata={
                "command": candidate.backing_command,
                "human_label": candidate.human_label,
            },
        )
    except Exception:
        pass


def _run_reminder_action(ctx: typer.Context, reminder_id: str) -> None:
    repo = ctx.obj["repo"]
    runtime = (ctx.obj or {}).get("runtime")
    ensure_goal_seed_tasks(repo, repo.list_goal_arcs(status=None), vault_path=getattr(runtime, "vault_path", None))
    console = get_console()
    err_console = get_err_console()
    reminder = repo.get_action_reminder(reminder_id)
    if reminder is None:
        err_console.print(f"[error]Reminder not found: {reminder_id}[/]")
        raise typer.Exit(code=1)

    if not sys.stdin.isatty():
        console.print(reminder.title or reminder.target_command)
        return

    choice = pick_single_choice(
        [
            ("start", "Start"),
            ("15", "Remind in 15 min"),
            ("30", "Remind in 30 min"),
            ("45", "Remind in 45 min"),
            ("skip", "Skip"),
        ],
        title=reminder.title,
        text=reminder.message,
    )
    if choice == "start":
        reminder.status = "completed"
        repo.update_action_reminder(reminder)
        run_internal_command(ctx, reminder.target_command)
        return
    if choice in {"15", "30", "45"}:
        minutes = int(choice)
        reminder.status = "pending"
        reminder.remind_at = utc_now() + timedelta(minutes=minutes)
        repo.update_action_reminder(reminder)
        schedule_actionable_notification(
            title=reminder.title,
            message=reminder.message,
            execute=f"pb next --reminder {reminder.id}",
            delay_minutes=minutes,
        )
        console.print(f"[success]Reminder rescheduled for {minutes} minutes.[/]")
        return

    reminder.status = "skipped"
    repo.update_action_reminder(reminder)
    console.print("[dim]Reminder skipped.[/]")


@app.callback(invoke_without_command=True)
def next_action(
    ctx: typer.Context,
    run: bool = typer.Option(False, "--run", help="Run the selected recommendation"),
    schedule: Optional[int] = typer.Option(None, "--schedule", "-s", help="Remind me about the top recommendation in N minutes"),
    reminder: Optional[str] = typer.Option(None, "--reminder", help="Open the action chooser for a queued reminder"),
):
    """Show the most relevant next actions from local context."""
    if reminder:
        _run_reminder_action(ctx, reminder)
        return

    repo = ctx.obj["repo"]
    console = get_console()
    auto_yes = bool((ctx.obj or {}).get("yes"))
    # build_next_candidates includes active commitments and neglected goals (Phase 10, ACCT-01)
    candidates = build_next_candidates(repo, limit=5)
    if not candidates:
        # Phase 10: dispatcher fallback when no local candidates (ACCT-01, D-03)
        try:
            import asyncio
            from pb.core.dispatcher import dispatch
            envelope = asyncio.run(dispatch(repo, "what should I do now?"))
            if envelope.prompt:
                console.print(envelope.prompt)
            if envelope.options:
                for i, opt in enumerate(envelope.options, 1):
                    console.print(f"  {i}. {opt}")
            if envelope.prompt or envelope.options:
                return
        except Exception:
            pass  # Fall through to existing "no actions" message
        console.print("No next actions available.")
        return
    verbose_mode = _verbose_mode(ctx)

    if schedule is not None:
        reminder_row = _schedule_reminder(repo, candidates[0], schedule)
        console.print(f"[success]Reminder scheduled:[/] in {schedule} min for {candidates[0].human_label}")
        if verbose_mode:
            console.print(f"[dim]Command: pb {candidates[0].backing_command}[/]")
        return

    if sys.stdin.isatty() and not auto_yes:
        options, details, verbose_labels = _picker_options(candidates)
        selected = pick_single_choice(
            options,
            title="Choose next action",
            text="Pick what to do now, or capture a thought or todo.",
            details=details,
            verbose_labels=verbose_labels if verbose_mode else None,
        )
        if selected:
            selected_candidate = next(
                (candidate for candidate in candidates if candidate.backing_command == selected),
                None,
            )
            if selected_candidate is not None:
                _record_candidate_selection(selected_candidate)
            run_internal_command(ctx, selected)
        return

    if run:
        _record_candidate_selection(candidates[0])
        run_internal_command(ctx, candidates[0].backing_command)
        return

    console.print("[header]Today's best next action[/]")
    console.print()
    for index, candidate in enumerate(candidates, start=1):
        console.print(f"  [dim]{index}.[/] {candidate.human_label}")
        console.print(f"     [dim]Why:[/] {candidate.short_reason}")
        if verbose_mode:
            console.print(f"     [dim]pb {candidate.backing_command}[/]")
