# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Immediate next-action recommendations and reminder queue handling."""

from __future__ import annotations

import sys
from datetime import timedelta
from types import SimpleNamespace
from typing import Optional

import typer

from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console, get_err_console
from pb.cli.pickers import pick_single_choice
from pb.core.agent_weights import record_agent_weight_event
from pb.core.action_routing import build_next_candidates
from pb.core.context_scope import ContextScopeFilter
from pb.core.goal_roadmaps import ensure_goal_seed_tasks
from pb.core.interest_hierarchy import InterestHierarchyService, InterestNode
from pb.core.models import ActionReminder, utc_now
from pb.core.naming import stored_display_title
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


def _immediate_candidates(candidates) -> list:
    """Keep at most one operational row for `pb next`."""

    for source in ("active_session", "reminder"):
        for candidate in candidates:
            if getattr(candidate, "source", "") == source:
                return [candidate]
    return []


def _node_options(nodes: list[InterestNode]) -> tuple[list[tuple[str, str]], list[str], list[str]]:
    options: list[tuple[str, str]] = []
    details: list[str] = []
    verbose: list[str] = []
    for node in nodes:
        label = node.label
        if node.level != "leaf" and node.children:
            label = f"{node.label} ->"
        if node.archived:
            label = f"{label} [archived]"
        options.append((node.command, label))
        details.append(node.reason)
        verbose.append(f"{label}  ({node.command})")
    return options, details, verbose


def _render_next_directions(console, *, immediate, nodes: list[InterestNode], verbose_mode: bool, archive: bool, within: str) -> None:
    title = "Archived directions" if archive else "Next directions"
    if within:
        title = f"{title}: {within}"
    console.print(f"[header]{title}[/]")
    console.print()
    index = 1
    for candidate in immediate:
        console.print(f"  [dim]{index}.[/] Now: {candidate.human_label}")
        console.print(f"     [dim]because[/] {candidate.short_reason}")
        if verbose_mode:
            console.print(f"     [dim]pb {candidate.backing_command}[/]")
        index += 1
    for node in nodes:
        label = node.label if node.level == "leaf" else f"{node.label} ->"
        console.print(f"  [dim]{index}.[/] {label}")
        console.print(f"     [dim]because[/] {node.reason}")
        if verbose_mode:
            console.print(f"     [dim]pb {node.command}[/]")
        index += 1


def _lock_hint(context_filter: ContextScopeFilter, excluded_count: int) -> str:
    if not context_filter.locked or excluded_count <= 0:
        return ""
    label = context_filter.label or next(iter(context_filter.source_refs), "") or "locked context"
    return f"Context lock active: {label}. {excluded_count} outside direction(s) hidden; use `pb context unlock` to widen."


def _run_node_selection(ctx: typer.Context, node: InterestNode) -> None:
    """Run a classifier node without forcing a second prompt during an active session."""

    repo = ctx.obj["repo"]
    if node.level != "leaf":
        run_internal_command(ctx, node.command)
        return

    active_session = repo.get_active_session()
    if active_session is None:
        run_internal_command(ctx, node.command)
        return

    active_task = repo.get_task(active_session.task_id)
    active_title = (
        stored_display_title(active_task)
        or getattr(active_session, "subject_scope", "")
        or "current session"
    )
    console = get_console()
    console.print(f"[dim]Session active: {active_title}.[/]")
    console.print(f"Next scoped command: pb {node.command}")
    console.print("[dim]Use `pb finish`, `pb pause`, or `pb session list` before starting another session.[/]")


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
    archive: bool = typer.Option(False, "--archive", "-a", help="Show older inactive directions only"),
    within: Optional[str] = typer.Option(None, "--within", help="Drill into one next-direction node"),
    schedule: Optional[int] = typer.Option(None, "--schedule", "-s", help="Remind me about the top recommendation in N minutes"),
    reminder: Optional[str] = typer.Option(None, "--reminder", help="Open the action chooser for a queued reminder"),
):
    """Show compact next directions from local learning context."""
    if reminder:
        _run_reminder_action(ctx, reminder)
        return

    repo = ctx.obj["repo"]
    runtime = (ctx.obj or {}).get("runtime")
    console = get_console()
    auto_yes = bool((ctx.obj or {}).get("yes"))
    verbose_mode = _verbose_mode(ctx)
    archive_mode = archive if isinstance(archive, bool) else False
    within_ref = within if isinstance(within, str) else ""
    all_candidates = build_next_candidates(repo, limit=6)
    immediate = _immediate_candidates(all_candidates)
    context_filter = ContextScopeFilter.from_repo(repo)
    service = InterestHierarchyService(repo, vault_path=getattr(runtime, "vault_path", None))
    directions = service.build(
        archive=archive_mode,
        within=within_ref,
        limit=max(1, 6 - len(immediate)),
        immediate=immediate,
        context_filter=context_filter,
    )
    nodes = list(directions.nodes)
    lock_hint = _lock_hint(context_filter, len(directions.excluded_by_lock))
    if schedule is not None:
        schedule_target = immediate[0] if immediate else all_candidates[0] if all_candidates else None
        if schedule_target is None and nodes:
            first_node = nodes[0]
            schedule_target = SimpleNamespace(
                human_label=first_node.label,
                short_reason=first_node.reason,
                backing_command=first_node.command,
            )
        if schedule_target is None:
            schedule_target = SimpleNamespace(
                human_label="Choose the next learning direction",
                short_reason="No concrete local action is ready yet.",
                backing_command="next",
            )
        _schedule_reminder(repo, schedule_target, schedule)
        console.print(f"[success]Reminder scheduled:[/] in {schedule} min for {schedule_target.human_label}")
        if verbose_mode:
            console.print(f"[dim]Command: pb {schedule_target.backing_command}[/]")
        return

    if not immediate and not nodes:
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
        console.print("No next directions available.")
        return
    if sys.stdin.isatty() and not auto_yes:
        candidate_options, candidate_details, candidate_verbose = _picker_options(immediate)
        node_options, node_details, node_verbose = _node_options(nodes)
        selected = pick_single_choice(
            candidate_options + node_options,
            title="Choose next direction",
            text=lock_hint or "Pick an immediate action or open a learning direction.",
            details=candidate_details + node_details,
            verbose_labels=(candidate_verbose + node_verbose) if verbose_mode else None,
        )
        if selected:
            selected_candidate = next((candidate for candidate in immediate if candidate.backing_command == selected), None)
            if selected_candidate is not None:
                _record_candidate_selection(selected_candidate)
                run_internal_command(ctx, selected)
                return
            selected_node = next((node for node in nodes if node.command == selected), None)
            if selected_node is not None:
                _run_node_selection(ctx, selected_node)
        return

    if run:
        if immediate:
            _record_candidate_selection(immediate[0])
            run_internal_command(ctx, immediate[0].backing_command)
        elif nodes:
            _run_node_selection(ctx, nodes[0])
        return

    _render_next_directions(
        console,
        immediate=immediate,
        nodes=nodes,
        verbose_mode=verbose_mode,
        archive=archive_mode,
        within=within_ref,
    )
    if lock_hint:
        console.print(f"[dim]{lock_hint}[/]")
