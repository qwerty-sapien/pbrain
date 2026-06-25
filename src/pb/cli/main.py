# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Main CLI entrypoint.

Exit codes per D-07, D-08:
- 0: Success (2xx equivalent)
- 4x: User error (bad input, not found, conflict)
- 5x: System error (database, I/O, config)
"""

import os
import sqlite3
import sys
from pathlib import Path
from typing import Optional

import structlog
import typer
from typer.core import TyperGroup

from pb.cli.commands import (
    anki as anki_mod,
    brain as brain_mod,
    context as context_mod,
    config as config_cmd_mod,
    doctor as doctor_mod,
    do as do_mod,
    execute,
    feedback as feedback_mod,
    goals,
    init as init_mod,
    learn as learn_mod,
    mcp as mcp_mod,
    metric as metric_mod,
    model as model_mod,
    note as note_mod,
    notes as notes_mod,
    next as next_mod,
    plan,
    practise as practise_mod,
    review,
    set as set_mod,
    sync as sync_mod,
    system_ops,
    tasks as tasks_mod,
    study as study_mod,
    session as session_mod,
    teach as teach_mod,
    vault as vault_mod,
    vocab as vocab_mod,
    plugin as plugin_mod,
)
from pb.core.entity_refs import display_ref
from pb.core.error_logging import format_logged_exception, log_error
from pb.core.naming import stored_display_title
from pb.cli.helpers import confirm_choice
from pb.domain.exceptions import ExitCode, UserError, PbSystemError
from pb.domain.rules import RuleViolation
from pb.runtime import build_runtime_context, get_session_auto_yes, runtime_from_config
from pb.storage import config as config_module
from pb.storage.database import init_db, set_db_path


def _min_level_filter(logger, method_name, event_dict):
    """Suppress structured stderr logs in normal mode."""
    if method_name != "critical":
        raise structlog.DropEvent
    return event_dict


def _configure_structlog(*, verbose: bool) -> None:
    """Configure structlog for human TTY output or machine-readable pipes."""
    interactive = sys.stdout.isatty()
    processors = [structlog.processors.TimeStamper(fmt="iso")]
    if not verbose:
        processors.insert(0, _min_level_filter)
    if interactive:
        processors.append(structlog.dev.ConsoleRenderer())
    else:
        processors.append(structlog.processors.JSONRenderer())
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.BoundLogger,
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=False,
    )


_configure_structlog(verbose=False)

_active_command: Optional[str] = None
_active_data_dir: Optional[Path] = None

_GLOBAL_VALUE_OPTIONS = {"--config", "--vault"}
_GLOBAL_FLAG_OPTIONS = {"--yes", "--verbose", "-v", "--dryrun"}
_TOPIC_COMMAND_VALUE_OPTIONS = {
    "learn": {"--context"},
    "study": {"--duration", "-d", "--stage", "--level", "-l", "--context"},
    "practise": {"--duration", "-d", "--drill", "--cues", "--context"},
    "practice": {"--duration", "-d", "--drill", "--cues", "--context"},
    "teach": {"--duration", "-d", "--stage", "--level", "-l", "--context"},
}
_TOPIC_COMMAND_FLAG_OPTIONS = {
    "learn": {"--study", "-s", "--practise", "--practice", "-p", "--steps", "--yes"},
    "study": {
        "--apply",
        "-a",
        "--understand",
        "-u",
        "--evaluate",
        "-e",
        "--create",
        "-c",
        "--steps",
        "--yes",
    },
    "practise": {"--steps", "--yes"},
    "practice": {"--steps", "--yes"},
    "teach": {
        "--apply",
        "-a",
        "--understand",
        "-u",
        "--evaluate",
        "-e",
        "--create",
        "-c",
        "--steps",
        "--no-steps",
        "--yes",
    },
}
_TOPIC_COMMAND_SUBCOMMANDS = {
    "study": {"day", "skip", "delete", "plan", "start", "debrief", "resume", "recall", "vocab"},
    "practise": {"start", "resume", "drill", "session", "log"},
    "practice": {"start", "resume", "drill", "session", "log"},
}


def _normalize_global_option_order(argv: list[str]) -> list[str]:
    """Allow root options such as --yes to appear after subcommands.

    Typer/Click only accepts root options before the command path. In practice,
    humans and agent clients naturally type `pb goal add "..." --yes`. This
    keeps that low-ceremony form working by moving known root options to the
    front before Click parses the invocation.
    """
    if len(argv) <= 2:
        return argv

    program = argv[0]
    args = argv[1:]
    global_options: list[str] = []
    remaining: list[str] = []
    index = 0
    passthrough = False
    while index < len(args):
        token = args[index]
        if passthrough:
            remaining.append(token)
            index += 1
            continue
        if token == "--":
            passthrough = True
            remaining.append(token)
            index += 1
            continue
        if token in _GLOBAL_FLAG_OPTIONS:
            global_options.append(token)
            index += 1
            continue
        if token in _GLOBAL_VALUE_OPTIONS and index + 1 < len(args):
            global_options.extend([token, args[index + 1]])
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in _GLOBAL_VALUE_OPTIONS):
            global_options.append(token)
            index += 1
            continue
        remaining.append(token)
        index += 1
    return [program, *global_options, *remaining]


def _find_command_index(args: list[str]) -> int | None:
    """Return the first non-global token index after argv[0]."""
    index = 1
    while index < len(args):
        token = args[index]
        if token in _GLOBAL_FLAG_OPTIONS:
            index += 1
            continue
        if token in _GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in _GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        return index
    return None


def _option_takes_value(command: str, token: str) -> bool:
    value_options = _TOPIC_COMMAND_VALUE_OPTIONS.get(command, set())
    if token in value_options:
        return True
    if any(token.startswith(f"{option}=") for option in value_options if option.startswith("--")):
        return False
    return False


def _is_topic_command_flag(command: str, token: str) -> bool:
    if token in _TOPIC_COMMAND_FLAG_OPTIONS.get(command, set()):
        return True
    if token in _TOPIC_COMMAND_VALUE_OPTIONS.get(command, set()):
        return True
    return any(
        token.startswith(f"{option}=")
        for option in _TOPIC_COMMAND_VALUE_OPTIONS.get(command, set())
        if option.startswith("--")
    )


def _normalize_topic_command_option_order(argv: list[str]) -> list[str]:
    """Accept `pb study TOPIC --duration 10m` without polluting TOPIC.

    Topic fallback commands intentionally accept arbitrary words, which means
    Click treats unknown leading words as would-be subcommands and stops parsing
    later command options. Move known callback options directly after the
    command name before Click parses them.
    """
    if len(argv) <= 3:
        return argv
    command_index = _find_command_index(argv)
    if command_index is None:
        return argv
    command = argv[command_index]
    if command not in _TOPIC_COMMAND_VALUE_OPTIONS:
        return argv

    tail = argv[command_index + 1 :]
    if not tail:
        return argv
    first_word = next((token for token in tail if token and not token.startswith("-")), "")
    if first_word in _TOPIC_COMMAND_SUBCOMMANDS.get(command, set()):
        return argv

    options: list[str] = []
    topic: list[str] = []
    index = 0
    while index < len(tail):
        token = tail[index]
        if token == "--":
            topic.extend(tail[index:])
            break
        if _is_topic_command_flag(command, token):
            options.append(token)
            if _option_takes_value(command, token) and index + 1 < len(tail):
                options.append(tail[index + 1])
                index += 2
            else:
                index += 1
            continue
        topic.append(token)
        index += 1

    if not options:
        return argv
    return [*argv[: command_index + 1], *options, *topic]


class PbGroup(TyperGroup):
    """Root group that preserves pure help rendering before runtime bootstrap."""

    def parse_args(self, ctx, args):
        ctx.meta["help_requested"] = any(arg in {"--help", "-h"} for arg in args)
        return super().parse_args(ctx, args)

app = typer.Typer(
    cls=PbGroup,
    name="pb",
    help=(
        "ProductiveBrain\n"
        "Turn a learning goal or topic into the next concrete session, stick with it, "
        "and capture recall without leaving the terminal.\n\n"
        "Start here:\n"
        "  pb learn \"rust async cancellation\"\n"
        "  pb teach \"bayes rule\"\n"
        "  pb next\n"
        "  pb do \"drill piano scales for 20 minutes\"\n"
        "  pb finish \"learned the cancellation rules\"\n"
        "  pb review week\n\n"
        "Teach, study, and practise all run through the same page-based lesson engine, "
        "while manual, setup, and library commands remain available behind the canonical loop."
    ),
    no_args_is_help=False,
    invoke_without_command=True,
    add_completion=False,
)

# Canonical ProductiveBrain command surface
# Commands are grouped into named help panels for discoverability.

# Start group — primary entry points for learning and doing
app.add_typer(do_mod.app, name="do", rich_help_panel="Start",
              help="Route a free-text request to the best next action")
app.add_typer(next_mod.app, name="next", rich_help_panel="Start",
              help="Show the next concrete step from local context")
app.add_typer(learn_mod.app, name="learn", rich_help_panel="Start",
              help="Turn a topic or skill into a study or practise session")
app.add_typer(study_mod.app, name="study", rich_help_panel="Start",
              help="Start a focused lesson in study mode")
app.add_typer(practise_mod.app, name="practise", rich_help_panel="Start",
              help="Start a deliberate-practice lesson")
app.add_typer(practise_mod.app, name="practice", rich_help_panel="Start",
              help="Alias for practise")
app.add_typer(teach_mod.app, name="teach", rich_help_panel="Start",
              help="Start a guided lesson in teach mode")

# Direction group — goal setting, planning, and reviewing progress
app.add_typer(goals.app, name="goal", rich_help_panel="Direction",
              help="Set, track, and review goals")
app.add_typer(plan.app, name="plan", rich_help_panel="Direction",
              help="Create learning and work plans")
app.add_typer(review.app, name="review", rich_help_panel="Direction",
              help="Review progress and reflect")

# Capture group — quick capture of notes and feedback
app.add_typer(notes_mod.app, name="notes", rich_help_panel="Capture",
              help="Review quarantined notes and merge them when ready")
app.add_typer(feedback_mod.app, name="feedback", rich_help_panel="Capture",
              help="Capture scoped guidance for how each workflow should behave")


@app.command("thought", rich_help_panel="Capture")
def capture_thought(
    ctx: typer.Context,
    words: Optional[list[str]] = typer.Argument(None, help="Thought text to capture"),
) -> None:
    """Capture a quick thought as a Markdown inbox note."""
    text = " ".join(words or []).strip()
    if not text:
        typer.echo("Usage: pb thought <text>", err=True)
        raise typer.Exit(code=ExitCode.BAD_INPUT)

    from pb.core.clock import utc_now
    from pb.core.graph_writer import make_slug
    from pb.vault.lifecycle import write_frontmatter

    runtime = ctx.obj["runtime"]
    now = utc_now()
    slug = make_slug(text) or "thought"
    thought_dir = Path(runtime.vault_path) / "Learning" / "Inbox" / "pb" / "thoughts"
    thought_dir.mkdir(parents=True, exist_ok=True)
    note_path = thought_dir / f"{now.strftime('%Y%m%d-%H%M%S-%f')}-{slug}.md"
    frontmatter = {
        "type": "thought",
        "source": "pb",
        "status": "captured",
        "created": now.isoformat(timespec="seconds"),
    }
    body = f"# Thought\n\n{text}\n"
    note_path.write_text(write_frontmatter(frontmatter, body), encoding="utf-8")

    rel_path = note_path.relative_to(Path(runtime.vault_path))
    typer.echo(f"Captured thought: {rel_path}")


@app.command("todo", rich_help_panel="Capture")
def capture_todo(
    ctx: typer.Context,
    words: Optional[list[str]] = typer.Argument(None, help="Todo text; supports /due YYYY-MM-DD or @[YYYY-MM-DD]"),
) -> None:
    """Capture a todo so it appears in the next-action loop."""
    text = " ".join(words or []).strip()
    if not text:
        typer.echo("Usage: pb todo <text>", err=True)
        raise typer.Exit(code=ExitCode.BAD_INPUT)

    from pb.core.exceptions import ValidationError
    from pb.core.intake import create_task as create_todo_task

    try:
        task = create_todo_task(ctx.obj["repo"], text)
    except ValidationError as exc:
        typer.echo(f"Todo not captured: {exc}", err=True)
        raise typer.Exit(code=exc.exit_code)

    typer.echo(f"Captured todo: {stored_display_title(task) or task.title}")
    due_date = getattr(task, "due_date", None)
    if due_date is not None:
        typer.echo(f"Due: {due_date.strftime('%Y-%m-%d')}")


# Recall group — Anki flashcard generation and export
app.add_typer(anki_mod.app, name="anki", rich_help_panel="Recall",
              help="Generate or export Anki flashcards")

# Context group — durable source files, bundles, and locked study scope
app.add_typer(context_mod.app, name="context", rich_help_panel="Knowledge",
              help="Inspect, persist, bundle, and lock learning context sources")

# Setup group — initialisation, diagnostics, and configuration
app.add_typer(set_mod.app, name="set", rich_help_panel="Setup",
              help="Set model tiers, language, and other preferences")
app.add_typer(init_mod.app, name="init", rich_help_panel="Setup",
              help="Set up ProductiveBrain for the first time")
app.add_typer(doctor_mod.app, name="doctor", rich_help_panel="Setup",
              help="Check system health and configuration")
app.add_typer(model_mod.app, name="model", rich_help_panel="Setup", hidden=True,
              help="Configure LLM model and provider (use pb set model instead)")
app.add_typer(mcp_mod.app, name="mcp", rich_help_panel="Setup",
              help="MCP setup and diagnostics for agent clients")


@app.command("update", rich_help_panel="Setup")
def update_pb(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check", help="Only inspect update state"),
    dryrun: bool = typer.Option(False, "--dryrun", help="Fetch and report, but do not pull"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Update this ProductiveBrain checkout with a fast-forward pull."""
    state = system_ops.inspect_update_state()
    if not state.get("supported"):
        typer.echo(state["message"])
        raise typer.Exit(code=1)

    typer.echo(f"Repo: {state['repo_root']}")
    typer.echo(f"Current commit: {state['current_commit']}")
    if state["dirty"]:
        typer.echo("Working tree: dirty")
    else:
        typer.echo("Working tree: clean")

    if not (check or dryrun):
        if state["dirty"]:
            typer.echo("Refusing to update with a dirty working tree.")
            raise typer.Exit(code=1)
        if not (yes or confirm_choice("Pull the latest fast-forward changes?")):
            raise typer.Exit(code=0)

    result = system_ops.run_update(check=check, dryrun=dryrun)
    if result.get("target_commit"):
        typer.echo(f"Target commit: {result['target_commit']}")
    if result.get("lockfiles_changed"):
        typer.echo(f"Lockfiles changed: {', '.join(result['lockfiles_changed'])}")
    typer.echo(result.get("message", "Update complete."))
    if not result.get("ok", False):
        raise typer.Exit(code=1)


@app.command("reset", rich_help_panel="Setup")
def reset_pb(
    ctx: typer.Context,
    dryrun: bool = typer.Option(False, "--dryrun", help="Preview what would be deleted"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
    backup: bool = typer.Option(False, "--backup", help="Write a zip backup before clearing the vault"),
) -> None:
    """Delete vault contents and reset ProductiveBrain SQLite state."""
    runtime = ctx.obj["runtime"]
    vault_path = Path(runtime.vault_path)
    db_path = system_ops.default_db_path_from_runtime(runtime)
    state = system_ops.inspect_reset_state(vault_path=vault_path, db_path=db_path)
    typer.echo(f"Vault path: {state['vault_path']}")
    typer.echo(f"SQLite path: {state['db_path']}")
    typer.echo(f"Files inside vault: {state['file_count']}")
    if state["suspicious"]:
        typer.echo("Refusing to reset a suspicious vault path.")
        raise typer.Exit(code=1)
    if not dryrun and not (yes or confirm_choice("This will delete all files inside the configured vault and clear PB SQLite. Continue?")):
        raise typer.Exit(code=0)
    result = system_ops.run_reset(vault_path=vault_path, db_path=db_path, dryrun=dryrun, backup=backup)
    if result.get("backup_path"):
        typer.echo(f"Backup: {result['backup_path']}")
    typer.echo(result.get("message", "Reset complete."))
    if not result.get("ok", False):
        raise typer.Exit(code=1)

# Vault / knowledge surfaces
app.add_typer(vault_mod.app, name="vault", rich_help_panel="Knowledge", help="Inspect vault profiles, graph traversal, and linked-note health")

# Hidden compatibility/internal surfaces
app.add_typer(metric_mod.app, name="metric", hidden=True, help="Manage per-vault learning evidence metrics")
app.add_typer(sync_mod.app, name="sync", hidden=True, help="Mirror Markdown knowledge into per-vault SQLite state")
app.add_typer(config_cmd_mod.app, name="config", hidden=True, help="Inspect config and session-scoped settings")

# Hidden compatibility/internal surfaces retained for working flows
app.add_typer(tasks_mod.app, name="task", hidden=True, help="Legacy task management surface")
app.add_typer(tasks_mod.app, name="tasks", hidden=True, help="Legacy plural task management surface")
app.add_typer(session_mod.app, name="session", hidden=True, help="Legacy session history surface")
app.add_typer(brain_mod.brain_app, name="brain", hidden=True, help="Legacy brain command group")
app.add_typer(note_mod.app, name="note", hidden=True, help="Legacy singular note alias")
app.add_typer(plugin_mod.app, name="plugin", hidden=True, help="Legacy plugin surface")
app.command("vocab", hidden=True)(vocab_mod.add_vocab)


@app.command("suggest", hidden=True)
def suggest_alias(
    ctx: typer.Context,
    intent_words: list[str] = typer.Argument(None, help="Natural-language request"),
):
    """Compatibility alias for `pb do`."""
    do_mod.do_command(ctx, intent_words)


@app.command("start", hidden=True)
def start_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID (omit for task picker)"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g., 30, 30m, 1h)"),
    suggest: bool = typer.Option(False, "--suggest", help="Show duration suggestion from history"),
):
    """Start a focus session on a task."""
    execute.start_task(ctx, task_id, duration, suggest)


@app.command("pause", rich_help_panel="Session")
def pause_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to postpone (omit to pause active session)"),
    note: Optional[str] = typer.Option(None, "--note", "-n", help="Outcome note / pause reason"),
    days: Optional[int] = typer.Option(None, "--days", "-d", help="Postpone task for N days"),
):
    """Pause active session, or postpone a task for N days with --days."""
    if days is None:
        execute.pause_task(ctx, note)
        return

    from pb.domain.enums import TaskState
    from datetime import datetime, timedelta
    from pb.core.exceptions import NotFoundError

    repo = ctx.obj['repo']
    task = None
    if task_id:
        task = repo.get_task(task_id)
        if not task:
            for t in repo.list_tasks():
                if t.id.startswith(task_id):
                    task = t
                    break
    else:
        session_service = ctx.obj['factory']['session_service']()
        task_obj = session_service.get_current_task()
        if task_obj:
            session_service.pause_session(outcome=note)
            task = repo.get_task(task_obj.id)
        else:
            raise NotFoundError("No active session and no task ID given.")
    if not task:
        raise NotFoundError("Task not found.")

    task.state = TaskState.PAUSED
    task.paused_until = datetime.utcnow() + timedelta(days=days)
    task.pause_reason = note
    repo.update_task(task)
    typer.echo(f"Paused: {task.title} for {days} days (until {task.paused_until.strftime('%Y-%m-%d')})")
    if note:
        typer.echo(f"Reason: {note}")


@app.command("later", hidden=True)
def later_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to reset and postpone (defaults to the active session task)"),
    note: Optional[str] = typer.Option(None, "--note", "-n", help="Optional postponement note"),
):
    """Forget the current runtime history for a task, reset it, and postpone it indefinitely."""
    from pb.core.learning_curriculum import pause_curriculum_descendants
    from pb.core.exceptions import ConflictError, NotFoundError

    repo = ctx.obj["repo"]
    session_service = ctx.obj["factory"]["session_service"]()
    active_session = session_service.get_current_session()

    if task_id:
        task = execute._find_task(repo, task_id)
        if active_session is not None and active_session.task_id != task.id:
            raise ConflictError("Another task is currently active. Finish, pause, or later that one first.")
    else:
        if active_session is None:
            raise NotFoundError("No active session.")
        task = repo.get_task(active_session.task_id)
        if task is None:
            raise NotFoundError("Active task not found.")

    reset_task = session_service.reset_task_for_later(task.id)
    if reset_task is None:
        raise NotFoundError("Task not found.")
    if note:
        reset_task.pause_reason = note
        repo.update_task(reset_task)

    pause_curriculum_descendants(repo, reset_task.id)
    typer.echo(f"Moved to later: {reset_task.title}")
    typer.echo(f"Resume with: pb resume {display_ref(reset_task, 'task')}")


@app.command("delete", hidden=True)
def delete_task_forever(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to delete (defaults to the active session task)"),
):
    """Delete a task plus its session history as if it never happened."""
    from pb.core.learning_curriculum import curriculum_descendants
    from pb.core.exceptions import ConflictError, NotFoundError

    repo = ctx.obj["repo"]
    session_service = ctx.obj["factory"]["session_service"]()
    active_session = session_service.get_current_session()

    if task_id:
        task = execute._find_task(repo, task_id)
        if active_session is not None and active_session.task_id != task.id:
            raise ConflictError("Another task is currently active. Finish, pause, or delete that one first.")
    else:
        if active_session is None:
            raise NotFoundError("No active session.")
        task = repo.get_task(active_session.task_id)
        if task is None:
            raise NotFoundError("Active task not found.")

    cascade = curriculum_descendants(repo, task.id)
    for child in reversed(cascade):
        session_service.delete_task_permanently(child.id)
    deleted_task = session_service.delete_task_permanently(task.id)
    if deleted_task is None:
        raise NotFoundError("Task not found.")
    typer.echo(f"Deleted: {deleted_task.title}")


@app.command("resume", rich_help_panel="Session")
def resume_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to resume (omit for picker)"),
):
    """Resume a paused task."""
    execute.resume_task(ctx, task_id)


@app.command("finish", rich_help_panel="Session")
def finish_task(
    ctx: typer.Context,
    note_words: Optional[list[str]] = typer.Argument(None, help="Optional inline note"),
    completion: Optional[int] = typer.Option(None, "--completion", "-c",
                                              help="Completion % (default 100)"),
    yes: bool = typer.Option(False, "--yes", "-y",
                              help="Skip confirmation prompts (non-interactive mode)"),
    debrief: bool = typer.Option(False, "--debrief",
                                  help="Trigger Socratic debrief (opt-in only)"),
    skip: bool = typer.Option(False, "--skip", "-q",
                               help="Skip AI assessment -- writes bare evidence note only"),
):
    """Stop the active session. Optional inline note: `pb finish this was hard`."""
    execute.finish_task(ctx, note_words, completion, yes, debrief, skip)


@app.command("f", hidden=True)
def finish_alias(
    ctx: typer.Context,
    note_words: Optional[list[str]] = typer.Argument(None),
    completion: Optional[int] = typer.Option(None, "--completion", "-c"),
    yes: bool = typer.Option(False, "--yes", "-y"),
    debrief: bool = typer.Option(False, "--debrief"),
    skip: bool = typer.Option(False, "--skip", "-q"),
):
    """Alias for finish."""
    execute.finish_task(ctx, note_words, completion, yes, debrief, skip)


@app.command("now", hidden=True)
def show_now(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output full session JSON"),
    plain_out: bool = typer.Option(False, "--plain", help="Pipe-friendly: task_name elapsed_min"),
):
    """Show current session status."""
    execute.show_now(ctx, json_out, plain_out)


@app.command("add", hidden=True)
def add_task(
    words: list[str] = typer.Argument(..., help="Task text (no quotes needed)"),
    skill: str = typer.Option("", "--skill", help="Pre-tag skill(s), comma-separated"),
    track: str = typer.Option("", "--track", "-t", help="Link to track by name"),
    goal: str = typer.Option("", "--goal", "-g", help="Link to goal by title (partial match)"),
):
    """Quick add a task. No quotes needed for multi-word text."""
    from pb.cli.commands.capture import create_task
    text = " ".join(words)
    create_task(text=text, horizon="today", skill=skill, track=track, goal=goal)


def _build_factory(repo, runtime):
    """Create the lazy service factory for one CLI invocation."""
    from pb.sessions.repo import SessionRepoAdapter
    from pb.sessions.service import SessionService

    return {
        'task_service': lambda: __import__('pb.tasks.service', fromlist=['TaskService']).TaskService(repo=repo),
        'session_service': lambda: SessionService(repo=SessionRepoAdapter(repo)),
        'goals_service': lambda: __import__('pb.goals.service', fromlist=['GoalService']).GoalService(repo=repo),
        'socratic_service': lambda: __import__(
            'pb.vault.socratic_service', fromlist=['SocraticService']
        ).SocraticService(vault_path=runtime.vault_path),
        'scoring_service': lambda: __import__(
            'pb.vault.scoring_service', fromlist=['ScoringService']
        ).ScoringService(vault_path=runtime.vault_path),
        'study_service': lambda: __import__(
            'pb.study_service', fromlist=['StudyService']
        ).StudyService(
            vault_path=runtime.vault_path,
            config=runtime.config,
        ),
        'anki_service': lambda: __import__(
            'pb.vault.anki_service', fromlist=['AnkiService']
        ).AnkiService(
            vault_path=runtime.vault_path,
            repo=repo,
        ),
    }


def _render_home_screen(ctx: typer.Context) -> None:
    """Render the ProductiveBrain home screen."""
    from pb.cli.console import get_console
    from pb.core.action_routing import build_next_candidates
    from pb.llm.runtime import LLMRuntime

    runtime = ctx.obj["runtime"]
    repo = ctx.obj["repo"]
    console = get_console()

    next_candidates = build_next_candidates(repo, limit=5)
    next_action = next_candidates[0] if next_candidates else None
    active_session = repo.get_active_session()
    recent_goals = repo.list_goal_arcs()[:3]

    try:
        from pb.vault.anki_client import get_cards_by_status, get_pending_card_count

        recall_debt = len(get_cards_by_status("suggested"))
        export_ready = get_pending_card_count()
    except Exception:
        recall_debt = 0
        export_ready = 0

    practise_gap = 0
    for goal in repo.list_goal_arcs():
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        target = getattr(goal, "target_practice_stage", None)
        current = getattr(goal, "current_practice_stage", None)
        if mode in {"mixed", "practise", "practice"} and target and current != target:
            practise_gap += 1

    health = LLMRuntime(runtime.config).health()

    console.rule("[header]ProductiveBrain[/]")
    console.print(f"Vault: {runtime.vault_name} -> {runtime.vault_path}")
    console.print(
        f"Model: {health.provider}:{health.default_model} "
        f"({'ready' if health.available else 'needs credentials'})"
    )
    if next_action is not None:
        console.print(f"Next action: {next_action.human_label}")
        console.print(f"[dim]{next_action.short_reason}[/]")
        if ctx.obj.get("verbose"):
            console.print(f"[dim]pb {next_action.backing_command}[/]")
    else:
        console.print("Next action: Run `pb do` to choose the best next study or practise block.")

    if active_session is not None:
        subject = getattr(active_session, "subject_scope", "") or "current focus"
        branch = getattr(active_session, "branch", "study") or "study"
        console.print(f"Active session: {branch} on {subject}")
    else:
        console.print("Active session: none")

    console.print(f"Recall debt: {recall_debt} suggested cards, {export_ready} export-ready")
    console.print(f"Practise gap: {practise_gap} goal(s) still need deliberate practice")
    if recent_goals:
        console.print("Recent goals:")
        for goal in recent_goals:
            console.print(f"  - {goal.title}")
    else:
        console.print("Recent goals: none yet")
    console.print("")
    console.print("Suggested commands:")
    console.print("  pb do")
    console.print('  pb learn "Rust async"')
    console.print("  pb finish")
    console.print("  pb review week")
    console.print("  pb notes inbox")


def _handle_no_config_degradation(ctx: typer.Context) -> None:
    """D-01: No config present -- show welcome + auto-launch pb init."""
    from pb.cli.console import get_console
    from pb.cli.commands.init import init_command

    console = get_console()
    console.print()
    console.print("[bold]Welcome to ProductiveBrain[/bold] -- a terminal-native study and practice system.")
    console.print()
    console.print("No configuration found. Starting setup now...")
    console.print()
    init_command()


def _handle_vault_missing_degradation(ctx: typer.Context, exc: FileNotFoundError) -> None:
    """D-02: Config present but vault path bad -- print message only, exit 0.

    T-01-01: Do NOT interpolate exc into the message (could leak filesystem paths).
    """
    from pb.cli.console import get_console

    console = get_console()
    console.print()
    console.print("[yellow]Vault not found.[/yellow] Your configuration exists but the vault directory is missing.")
    console.print()
    console.print("Run [bold]pb init[/bold] to update your vault path.")


@app.command("home", hidden=True)
def show_home(ctx: typer.Context):
    """Render the dashboard without entering the shell."""
    _render_home_screen(ctx)


@app.callback(invoke_without_command=True)
def main(
    ctx: typer.Context,
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose output"),
    config_path: Optional[Path] = typer.Option(None, "--config", exists=False, dir_okay=False, help="Explicit config.toml path."),
    vault: Optional[str] = typer.Option(None, "--vault", help="Use a named vault profile for this invocation."),
    yes: bool = typer.Option(False, "--yes", help="Accept previews and confirmations for this invocation."),
    dryrun: bool = typer.Option(False, "--dryrun", help="Sandbox mode: all writes go to a temp directory"),
):
    """ProductiveBrain - terminal-native study and deliberate-practice system."""
    # Display is always rich (Glow-rendered Markdown) — there is no --plain/--rich
    # flag. Rich is the only mode; plain rendering has been removed entirely.
    global _active_command, _active_data_dir
    _active_command = ctx.invoked_subcommand or ("shell" if sys.stdin.isatty() else "home")
    _active_data_dir = None

    if getattr(ctx, "meta", {}).get("help_requested"):
        return

    from pb.cli.console import install_prompt_abort, set_plain_mode
    install_prompt_abort()

    _configure_structlog(verbose=verbose)

    ctx.ensure_object(dict)
    ctx.obj["yes"] = yes
    ctx.obj["verbose"] = verbose
    ctx.obj["dryrun"] = dryrun
    ctx.obj["config_path"] = config_path
    ctx.obj["vault_override"] = vault

    if config_path is not None:
        os.environ["PRODUCTIVEBRAIN_CONFIG_PATH"] = str(config_path)
    if yes:
        os.environ["PRODUCTIVEBRAIN_AUTO_YES"] = "1"

    if not dryrun:
        from pb.core.dryrun import cleanup_stale_dryrun_dirs
        try:
            cleanup_stale_dryrun_dirs()
        except Exception:
            pass

    skip_runtime_bootstrap = ctx.invoked_subcommand in {"init", "mcp"}
    allow_missing_config = ctx.invoked_subcommand in {"doctor", "config", "model", "mcp", "init"}
    runtime = None

    if not skip_runtime_bootstrap:
        if config_module._config is not None and config_path is None and vault is None:
            runtime = runtime_from_config(config_module._config, yes=yes)
        else:
            try:
                runtime = build_runtime_context(config_path=config_path, vault=vault, yes=yes, force_reload=True)
            except FileNotFoundError as exc:
                if not allow_missing_config:
                    from pb.storage.config import get_config_path
                    config_exists = get_config_path(config_path).exists()
                    if ctx.invoked_subcommand is not None:
                        # Subcommand invoked — show targeted error, not setup wizard
                        from pb.cli.console import get_err_console
                        if config_exists:
                            get_err_console().print(
                                "[error]Vault not found. Run `pb init` to update your vault path.[/]"
                            )
                        else:
                            get_err_console().print("[error]Config not found. Run `pb init` to create one.[/]")
                        raise typer.Exit(code=ExitCode.CONFIG_ERROR)
                    # Bare `pb` with no subcommand -- guided degradation
                    if config_exists:
                        # Scenario 2: config present, vault path bad (per D-02)
                        _handle_vault_missing_degradation(ctx, exc)
                    else:
                        # Scenario 1: no config at all (per D-01)
                        _handle_no_config_degradation(ctx)
                    raise typer.Exit(code=ExitCode.SUCCESS)  # D-04: always exit 0
            else:
                config_module._config = runtime.config

        if runtime is not None:
            if dryrun:
                from pb.core.dryrun import create_dryrun_sandbox
                from pb.cli.console import get_console
                sandbox = create_dryrun_sandbox(runtime.data_dir)
                runtime.vault_path = sandbox.vault_path
                runtime.data_dir = sandbox.data_dir
                runtime.db_path = sandbox.db_path
                runtime.quarantine_path = sandbox.vault_path / "quarantine"
                runtime.quarantine_path.mkdir(exist_ok=True)
                os.environ["PRODUCTIVEBRAIN_DRYRUN"] = "1"
                get_console().print(f"[dim][dryrun] outputs → {sandbox.root}[/]")

            set_db_path(runtime.db_path)
            init_db(runtime.db_path)
            ctx.obj["runtime"] = runtime
            _active_data_dir = runtime.data_dir

    if runtime is None and not skip_runtime_bootstrap:
        ctx.obj["config"] = None
        ctx.obj["repo"] = None
        ctx.obj["factory"] = {}
        return

    # Rich is the only display mode (Glow-rendered Markdown). No plain fallback.
    set_plain_mode(False)

    if runtime is not None:
        from pb.storage.repository import Repository

        repo = Repository()
        ctx.obj['config'] = runtime.config
        ctx.obj['repo'] = repo
        ctx.obj['vault_cwd'] = None
        ctx.obj['factory'] = _build_factory(repo, runtime)
        ctx.obj['yes'] = yes or get_session_auto_yes(runtime.config)

        if not dryrun:
            try:
                from pb.cli.console import get_console
                from pb.core.anki_bootstrap import bootstrap_anki_if_approved

                bootstrap_anki_if_approved(
                    runtime.config,
                    config_path=config_path,
                    console=get_console(),
                    interactive=sys.stdin.isatty(),
                )
            except Exception:
                pass  # Non-fatal: Anki availability must never block pb startup.

    # Phase 23: Check for expired timer before executing any command (D-13/D-14)
        try:
            _check_timer_expiry(ctx)
        except Exception:
            pass  # Non-fatal

        if ctx.invoked_subcommand is None:
            if sys.stdin.isatty():
                _interactive_shell(ctx)
            else:
                _render_home_screen(ctx)
            return

        if dryrun and sys.stdin.isatty():
            ctx.call_on_close(lambda: _interactive_shell(ctx))


def _check_timer_expiry(ctx: typer.Context) -> None:
    """Intercept any pb command when a timer has expired (D-13/D-14).

    Checks for timer_expired.flag. If found and a session is active,
    shows a picker to extend, add time, finish, or cancel.
    Non-TTY: flag is cleaned up silently.
    Entire function is non-fatal — never blocks the requested command.
    """
    import sys as _sys
    from pb.core.timer import TIMER_EXPIRED_FLAG

    FLAG = TIMER_EXPIRED_FLAG
    if not FLAG.exists():
        return

    try:
        if not _sys.stdin.isatty():
            FLAG.unlink(missing_ok=True)
            return

        repo = ctx.obj.get('repo') if ctx.obj else None
        if repo is None:
            FLAG.unlink(missing_ok=True)
            return

        session = repo.get_active_session()
        if session is None:
            FLAG.unlink(missing_ok=True)
            return

        task = repo.get_task(session.task_id)
        task_title = task.title if task else "Unknown"

        # Compute overtime
        from datetime import datetime as _dt
        elapsed_min = int((_dt.utcnow() - session.start_at).total_seconds() / 60)
        duration = getattr(session, 'duration_minutes', None) or 0
        overtime_min = max(0, elapsed_min - duration)

        from pb.cli.pickers import timer_expiry_picker
        choice = timer_expiry_picker(task_title, overtime_min)

        if choice == "extend_10":
            # Extend session duration by 10 minutes in the session record
            new_duration = duration + 10
            session.duration_minutes = new_duration
            repo.update_session(session)
            from pb.cli.console import get_console
            get_console().print(f"[success]Extended by 10m (now {new_duration}m total)[/]")

        elif choice == "add_time":
            # Prompt for custom minutes (T-23-11: cast to int with try/except)
            from pb.cli.console import get_console, get_err_console
            try:
                raw = input("Additional minutes: ").strip()
                extra = int(raw)
                new_duration = duration + extra
                session.duration_minutes = new_duration
                repo.update_session(session)
                get_console().print(f"[success]Extended by {extra}m (now {new_duration}m total)[/]")
            except (ValueError, EOFError):
                get_err_console().print("[warn]Invalid input. Extension cancelled.[/]")

        elif choice == "finish":
            # Finish the session via SessionService
            from pb.sessions.service import SessionService
            from pb.sessions.repo import SessionRepoAdapter
            svc = SessionService(repo=SessionRepoAdapter(repo))
            svc.finish_session(note=None, completion_pct=100)
            from pb.cli.console import get_console
            get_console().print(f"[success]Finished: {task_title}[/]")

        # "cancel" or None: do nothing, session continues in overtime

    except Exception:
        pass  # Non-fatal: never block the requested command

    finally:
        FLAG.unlink(missing_ok=True)


def _interactive_shell(ctx: typer.Context):
    """Interactive shell mode -- delegates to shell.run_shell() (Phase 10 upgrade)."""
    from pb.cli.shell import run_shell
    from pb.storage.config import get_vault_path

    click_app = typer.main.get_command(app)
    vault_root = get_vault_path()
    repo = ctx.obj['repo']  # Use repo from ctx.obj, not a fresh Repository()
    run_shell(click_app, vault_root, repo, runtime_ctx=ctx.obj["runtime"])


def run_app():
    """Run the CLI app with exception handling."""
    from pb.cli.console import get_err_console
    from rich.markup import escape

    def _record_error(exc: BaseException, *, status: int) -> None:
        try:
            log_error(
                event="cli.exception",
                message=str(exc),
                exc=exc,
                data_dir=_active_data_dir,
                command=_active_command or "",
                status=status,
                extra={"exception_type": exc.__class__.__name__},
            )
        except Exception:
            pass

    exit_code = 0
    error_msg = ""
    try:
        sys.argv = _normalize_global_option_order(list(sys.argv))
        sys.argv = _normalize_topic_command_option_order(list(sys.argv))
        app()
    except RuleViolation as e:
        _record_error(e, status=ExitCode.CONFLICT)
        err = get_err_console()
        err.print(f"[error]Error: {escape(str(e))}[/]")
        exit_code = ExitCode.CONFLICT
        error_msg = str(e).splitlines()[0] if str(e) else ""
        raise SystemExit(exit_code)
    except sqlite3.Error as e:
        _record_error(e, status=ExitCode.DB_ERROR)
        err = get_err_console()
        err.print(f"[error]Database error: {escape(str(e))}[/]")
        exit_code = ExitCode.DB_ERROR
        error_msg = str(e).splitlines()[0] if str(e) else ""
        raise SystemExit(exit_code)
    except UserError as e:
        _record_error(e, status=e.exit_code)
        err = get_err_console()
        err.print(f"[error]Error: {escape(str(e))}[/]")
        exit_code = e.exit_code
        error_msg = str(e).splitlines()[0] if str(e) else ""
        raise SystemExit(exit_code)
    except PbSystemError as e:
        _record_error(e, status=e.exit_code)
        err = get_err_console()
        err.print(f"[error]System error: {escape(str(e))}[/]")
        exit_code = e.exit_code
        error_msg = str(e).splitlines()[0] if str(e) else ""
        raise SystemExit(exit_code)
    except Exception as e:
        exit_code = ExitCode.INTERNAL
        error_msg = str(e).splitlines()[0] if str(e) else e.__class__.__name__
        log_ref = log_error(
            event="cli.unhandled_exception",
            message=str(e),
            exc=e,
            data_dir=_active_data_dir,
            command=_active_command or "",
            status=exit_code,
            extra={"exception_type": e.__class__.__name__},
        )
        err = get_err_console()
        err.print(f"[error]Error: {escape(format_logged_exception(e, log_ref))}[/]")
        raise SystemExit(exit_code)
    finally:
        # D-01: Log every command invocation
        if _active_command:
            try:
                from pb.storage.database import log_usage
                log_usage(_active_command, exit_code, error_msg)
            except Exception:
                pass  # Non-fatal


if __name__ == "__main__":
    run_app()
