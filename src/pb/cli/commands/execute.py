# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Execution commands - start/pause/resume/finish/now/interrupt."""

import sys
from datetime import datetime
from typing import Optional

import typer
from rich.markup import escape

from pb.cli.active_session import resolve_active_session_preflight
from pb.cli.console import get_console, get_err_console
from pb.cli.display import format_date_local
from pb.cli.helpers import confirm_choice, parse_duration
from pb.cli.llm_guard import runtime_for_ctx
from pb.cli.markdown import render_markdown
from pb.cli.pickers import pick_or_prompt, pick_single_choice
from pb.cli.preview import markdown_step_lines, preview_decision, render_styled_preview
from pb.core.closeout import CloseoutService
from pb.core.feedback_proposals import FeedbackProposalService
from pb.core.enums import EnergyType, Horizon, TaskState
from pb.core.entity_refs import display_ref
from pb.core.goal_roadmaps import (
    materialize_next_frontier_tasks,
    preview_rows_for_follow_on_specs,
    roadmap_follow_on_specs,
)
from pb.core.models import Task, utc_now
from pb.core.naming import stored_short_title
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.domain.exceptions import ExitCode
from pb.domain.rules import RuleViolation


app = typer.Typer()


def _next_session_recommendation(repo, finished_task_id: str) -> str | None:
    """Suggest the next pb study command after finishing a session."""
    from pb.cli.commands.study import _planned_study_rows

    items = _planned_study_rows(repo, include_completed=True)
    if not items:
        return None

    found_finished = False
    for row in items:
        if row.task.id == finished_task_id:
            found_finished = True
            continue
        if found_finished and row.task.completion < 100:
            return f"pb study {row.display_code}"

    for row in items:
        if row.task.id != finished_task_id and row.task.completion < 100:
            return f"pb study {row.display_code}"

    return None


def _detect_session_overrun(repo, session) -> int | None:
    """Return minutes over plan if session ran long, else None."""
    if session is None or not hasattr(session, "end_at") or session.end_at is None:
        return None
    blocks = repo.list_time_blocks_for_date(session.start_at)
    for block in blocks:
        if block.task_id == session.task_id:
            elapsed = (session.end_at - session.start_at).total_seconds() / 60
            overrun = int(elapsed - block.duration_minutes)
            return overrun if overrun > 5 else None
    return None


def _task_session_defaults(task) -> dict[str, object]:
    """Recover branch-specific session defaults from task metadata."""
    meta = parse_learning_task_metadata(task)
    branch = meta.branch or "study"
    return {
        "branch": branch,
        "goal_id": getattr(task, "linked_goal_arc_ids", [None])[0] if getattr(task, "linked_goal_arc_ids", None) else None,
        "track_id": getattr(task, "linked_track_ids", [None])[0] if getattr(task, "linked_track_ids", None) else None,
        "subject_scope": meta.domain or meta.scope,
        "target_bloom_stage": meta.bloom_target or None,
        "practice_stage": meta.practice_stage or None,
        "drill_type": meta.drill or None,
        "constraint": meta.constraint or None,
        "feedback_source": meta.feedback_source or None,
        "evidence_target": meta.evidence_target or None,
        "coach_cues": meta.cues or None,
        "domain_pack_id": meta.domain_pack_id or "",
        "session_blueprint": dict(meta.session_blueprint or {}) if meta.session_blueprint else None,
    }


def _advance_goal_progress_from_session(repo, session, completion_pct: int) -> None:
    """Promote goal progress when a finished session provides enough evidence."""

    goal_id = getattr(session, "goal_id", None)
    if not goal_id or completion_pct < 70:
        return
    goal = repo.get_goal_arc(goal_id)
    if goal is None:
        return

    branch = (getattr(session, "branch", "") or "").lower()
    changed = False
    if branch == "study" and getattr(session, "target_bloom_stage", None) is not None:
        if goal.current_bloom_stage != session.target_bloom_stage:
            goal.current_bloom_stage = session.target_bloom_stage
            changed = True
    if branch in {"practise", "practice"} and getattr(session, "practice_stage", None) is not None:
        if goal.current_practice_stage != session.practice_stage:
            goal.current_practice_stage = session.practice_stage
            changed = True

    if changed:
        repo.update_goal_arc(goal)


def _discard_active_session(repo, session_service, session, task) -> None:
    """Remove an unhelpful active session and its empty generated task when safe."""
    try:
        session_service.timer.stop_session_timers()
    except Exception:
        pass
    repo.delete_session(session.id)
    if task is None:
        return
    if not repo.list_sessions_for_task(task.id) and task.completion <= 0:
        repo.hard_delete_task(task.id)


def _create_recovery_task(repo, task, *, note: str, branch: str):
    title = f"Recovery: {stored_short_title(task) or task.title}"
    recovery = Task(
        title=title,
        description=note,
        horizon=Horizon.TODAY,
        state=TaskState.ACTIVE,
        created_at=utc_now(),
        energy_type=EnergyType.DEEP if branch == "study" else EnergyType.PRACTICE,
        work_type="study" if branch == "study" else "practice",
        linked_goal_arc_ids=list(getattr(task, "linked_goal_arc_ids", []) or []),
        linked_track_ids=list(getattr(task, "linked_track_ids", []) or []),
    )
    repo.create_task(recovery)
    return recovery


def _maybe_create_roadmap_follow_ons(ctx: typer.Context, repo, task, assessment) -> list[Task]:
    """Preview and optionally create the next roadmap/project tasks."""
    if task is None or not getattr(task, "linked_goal_arc_ids", None):
        return []
    goal_id = task.linked_goal_arc_ids[0]
    goal = repo.get_goal_arc(goal_id)
    if goal is None:
        return []
    specs = roadmap_follow_on_specs(repo, goal, task, assessment=assessment)
    if not specs:
        return []
    if not sys.stdin.isatty():
        return []

    render_styled_preview(
        title="Next Project Tasks",
        rows=[
            ("Project", (getattr(goal, "generated_names", {}) or {}).get("goal_project_title") or goal.title),
            ("Count", str(len(specs))),
        ],
        sections=preview_rows_for_follow_on_specs(specs),
    )

    while True:
        decision = preview_decision(yes=False, action_label="Create these next task(s)")
        if decision.kind == "accept":
            return materialize_next_frontier_tasks(
                repo,
                goal,
                max_new=len(specs),
                remediation_specs=specs if any(spec.get("title", "").startswith("Reinforce ") for spec in specs) else None,
            ) if any(spec.get("title", "").startswith("Reinforce ") for spec in specs) else materialize_next_frontier_tasks(repo, goal, max_new=len(specs))
        if decision.kind == "cancel":
            return []
        if any(token in decision.text.lower() for token in ("one", "1", "single", "first")):
            trimmed = specs[:1]
        elif any(token in decision.text.lower() for token in ("two", "2")):
            trimmed = specs[:2]
        else:
            trimmed = specs
        specs = trimmed
        render_styled_preview(
            title="Next Project Tasks",
            rows=[
                ("Project", (getattr(goal, "generated_names", {}) or {}).get("goal_project_title") or goal.title),
                ("Count", str(len(specs))),
            ],
            sections=preview_rows_for_follow_on_specs(specs),
        )


def _find_task(repo, task_id: str):
    """
    Find a task by ID or prefix with disambiguation per QUAL-03.

    Per D-05: Numbered selection prompt when multiple match
    Per D-06: Each option shows creation date, name, state, deadline
    """
    err_console = get_err_console()

    task = repo.resolve_task_ref(task_id, include_archived=True)
    if task is not None:
        # Check if archived
        if task.archived_at is not None:
            err_console.print(
                f"[error]Task {escape(display_ref(task, 'task'))} is archived. Use 'pb task restore' to unarchive.[/]"
            )
            raise typer.Exit(code=ExitCode.NOT_FOUND)
        return task

    err_console.print(f"[error]Task not found: {escape(task_id)}[/]")
    raise typer.Exit(code=ExitCode.NOT_FOUND)


@app.command("start")
def start_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID (omit for task picker)"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g., 30, 30m, 1h)"),
    suggest: bool = typer.Option(False, "--suggest", help="Show duration suggestion from history"),
):
    """Start a focus session on a task."""
    start_task_internal(ctx, task_id=task_id, duration=duration, suggest=suggest)


def start_task_internal(
    ctx: typer.Context,
    *,
    task_id: Optional[str],
    duration: Optional[str],
    suggest: bool,
    pre_session_markdown: Optional[str] = None,
    skip_clock: bool = False,
):
    """Shared task start implementation for learner flows."""
    session_service = ctx.obj['factory']['session_service']()
    repo = ctx.obj['repo']
    console = get_console()
    err_console = get_err_console()

    # Build resumable task set (tasks with prior sessions but no current active session)
    active_session = repo.get_active_session()
    active_tid = active_session.task_id if active_session else None

    all_tasks = repo.list_tasks()
    resumable_ids = {
        t.id for t in all_tasks
        if repo.list_sessions_for_task(t.id) and t.id != active_tid
    }

    # D-01: picker shows active-state tasks + resumable tasks with [paused] indicator
    # TaskState.PAUSED (snooze) tasks are NOT shown — they are excluded below
    if task_id is None:
        tasks = [
            t for t in all_tasks
            if t.completion < 100
            and t.state.value == "active"
            and t.archived_at is None
        ]
        if not tasks:
            err_console.print("[error]No tasks to start. Use `pb add` to create one.[/]")
            raise typer.Exit(code=ExitCode.NOT_FOUND)

        task = pick_or_prompt(
            tasks,
            find_fn=lambda tid: _find_task(repo, tid),
            title="Select task",
            active_task_id=active_tid,
            paused_task_ids=resumable_ids,
        )
        if not task:
            raise typer.Exit(code=ExitCode.SUCCESS)
    else:
        task = _find_task(repo, task_id)

    # Pre-task verification for priority dependencies
    try:
        from pb.core.verification import needs_verification
        if needs_verification(repo, task) and sys.stdin.isatty():
            console.print("[yellow]Prerequisite verification needed before starting this task.[/]")
            console.print("[dim]Run `pb study verify` to take the quiz, or --skip-verify to bypass.[/]")
    except Exception:
        pass

    session_defaults = _task_session_defaults(task)
    if not resolve_active_session_preflight(
        ctx,
        new_intent=task.title,
        new_branch=str(session_defaults["branch"]),
    ):
        return
    from pb.cli.commands.clarify import maybe_expand_learning_todo

    if maybe_expand_learning_todo(ctx, task=task):
        return

    # D-02: Resolve duration — CLI flag overrides TimeBlock; no prompt if neither present = stopwatch
    duration_minutes = None

    # SESS-08: --suggest flag returns median duration from history (silent if insufficient)
    if suggest:
        suggestion = session_service.suggest_duration(task.id)
        if suggestion is not None:
            console.print(f"[dim]Suggested duration: {suggestion}m (based on history)[/]")
            duration_minutes = suggestion

    # CLI -d flag overrides suggestion
    if duration is not None:
        parsed = parse_duration(duration)
        if parsed is None:
            err_console.print(f"[error]Invalid duration: {escape(duration)}. Use format: 30, 30m, 1h, 1.5h[/]")
            raise typer.Exit(code=ExitCode.BAD_INPUT)
        duration_minutes = parsed
    elif duration_minutes is None:
        # TimeBlock pre-set from pb plan day (per D-02)
        blocks = repo.list_time_blocks_for_date(datetime.utcnow())
        for block in blocks:
            if block.task_id == task.id:
                duration_minutes = block.duration_minutes
                break

    # Determine timer mode (per D-02/D-03)
    timer_mode = "timer" if duration_minutes is not None else "stopwatch"

    # D-03: Zero interrogation — no expectation prompt, no duration prompt
    try:
        session_defaults = _task_session_defaults(task)
        learning_branch = str(session_defaults["branch"]).lower() in {"study", "practise", "practice"}
        session = session_service.start_session(
            task_id=task.id,
            mode="focus",
            duration_minutes=duration_minutes,
            timer_mode=timer_mode,
            branch=str(session_defaults["branch"]),
            goal_id=session_defaults["goal_id"],
            track_id=session_defaults["track_id"],
            subject_scope=str(session_defaults["subject_scope"]),
            target_bloom_stage=session_defaults["target_bloom_stage"],
            practice_stage=session_defaults["practice_stage"],
            drill_type=session_defaults["drill_type"],
            constraint=session_defaults["constraint"],
            feedback_source=session_defaults["feedback_source"],
            evidence_target=session_defaults["evidence_target"],
            coach_cues=session_defaults["coach_cues"],
        )
        if task.id in resumable_ids:
            console.print(f"[success]Resumed: {escape(task.title)}[/]")
        elif duration_minutes:
            console.print(f"[success]Started: {escape(task.title)} ({duration_minutes}m)[/]")
        else:
            console.print(f"[success]Started: {escape(task.title)}[/]")
        generated = dict(getattr(session, "generated_names", {}) or {})
        if session_defaults.get("domain_pack_id"):
            generated["domain_pack_id"] = session_defaults["domain_pack_id"]
        if session_defaults.get("session_blueprint"):
            generated["session_blueprint"] = session_defaults["session_blueprint"]
        if generated != dict(getattr(session, "generated_names", {}) or {}):
            session.generated_names = generated
            repo.update_session(session)
    except RuleViolation as e:
        err_console.print(f"[error]{escape(str(e))}[/]")
        raise typer.Exit(code=ExitCode.CONFLICT)

    step_markdown = pre_session_markdown or _task_steps_markdown(task)
    if step_markdown and sys.stdin.isatty():
        render_markdown(step_markdown)

    # Learning sessions return directly to the prompt so the timer can keep
    # running without trapping the user in a blocking clock view.
    if sys.stdin.isatty() and not skip_clock and not learning_branch:
        from pb.cli.display import live_session_clock
        live_session_clock(session, task, duration_minutes=duration_minutes)
    return session, task


def _task_steps_markdown(task) -> Optional[str]:
    """Build a once-per-session step guide from persisted task metadata."""
    meta = parse_learning_task_metadata(task)
    if not meta.steps:
        return None
    lines = [f"# Session Guide", "", f"**{task.title}**", ""]
    lines.extend(markdown_step_lines(meta.steps))
    return "\n".join(lines).rstrip() + "\n"


@app.command("pause")
def pause_task(
    ctx: typer.Context,
    note: Optional[str] = typer.Option(None, "--note", "-n", help="Outcome note"),
):
    """Pause the current session."""
    session_service = ctx.obj['factory']['session_service']()
    console = get_console()
    err_console = get_err_console()

    session = session_service.pause_session(outcome=note)
    if session is None:
        err_console.print("[error]No active session to pause.[/]")
        raise typer.Exit(code=ExitCode.NOT_FOUND)

    task = ctx.obj['repo'].get_task(session.task_id)
    task_title = task.title if task else "Unknown"
    console.print(f"[success]Paused: {escape(task_title)}[/]")


@app.command("resume")
def resume_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to resume (omit for picker)"),
):
    """Resume working on a task that has prior sessions.

    With no args, shows a picker of tasks with previous sessions.
    """
    session_service = ctx.obj['factory']['session_service']()
    repo = ctx.obj['repo']
    console = get_console()
    err_console = get_err_console()

    # D-04: resume shows only resumable (paused) tasks — filtered view
    if task_id is None:
        active_session = repo.get_active_session()
        active_tid = active_session.task_id if active_session else None
        all_tasks = repo.list_tasks()
        resumable = [
            t for t in all_tasks
            if t.completion < 100
            and t.archived_at is None
            and t.id != active_tid
            and (
                (t.state.value == "active" and repo.list_sessions_for_task(t.id))
                or t.state.value == "paused"
            )
        ]
        if not resumable:
            err_console.print("[error]No resumable tasks.[/]")
            raise typer.Exit(code=ExitCode.NOT_FOUND)

        task = pick_or_prompt(
            resumable,
            find_fn=lambda tid: _find_task(repo, tid),
            title="Resume task",
            active_task_id=active_tid,
            paused_task_ids={t.id for t in resumable},
        )
        if not task:
            raise typer.Exit(code=ExitCode.SUCCESS)
    else:
        task = _find_task(repo, task_id)

    if task.state.value == "paused":
        task.state = type(task.state).ACTIVE
        task.paused_until = None
        task.pause_reason = None
        repo.update_task(task)

    session_defaults = _task_session_defaults(task)
    try:
        session = session_service.start_session(
            task_id=task.id,
            mode="focus",
            duration_minutes=None,
            timer_mode="stopwatch",
            branch=str(session_defaults["branch"]),
            goal_id=session_defaults["goal_id"],
            track_id=session_defaults["track_id"],
            subject_scope=str(session_defaults["subject_scope"]),
            target_bloom_stage=session_defaults["target_bloom_stage"],
            practice_stage=session_defaults["practice_stage"],
            drill_type=session_defaults["drill_type"],
            constraint=session_defaults["constraint"],
            feedback_source=session_defaults["feedback_source"],
            evidence_target=session_defaults["evidence_target"],
            coach_cues=session_defaults["coach_cues"],
        )
        console.print(f"[success]Resumed: {escape(task.title)}[/]")
        generated = dict(getattr(session, "generated_names", {}) or {})
        if session_defaults.get("domain_pack_id"):
            generated["domain_pack_id"] = session_defaults["domain_pack_id"]
        if session_defaults.get("session_blueprint"):
            generated["session_blueprint"] = session_defaults["session_blueprint"]
        if generated != dict(getattr(session, "generated_names", {}) or {}):
            session.generated_names = generated
            repo.update_session(session)
    except RuleViolation as e:
        err_console.print(f"[error]{escape(str(e))}[/]")
        raise typer.Exit(code=ExitCode.CONFLICT)

    learning_branch = str(session_defaults["branch"]).lower() in {"study", "practise", "practice"}
    if sys.stdin.isatty() and not learning_branch:
        from pb.cli.display import live_session_clock
        live_session_clock(session, task, duration_minutes=None)


@app.command("finish")
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
    session_service = ctx.obj['factory']['session_service']()
    repo = ctx.obj['repo']
    console = get_console()
    err_console = get_err_console()

    # D-05: Finish operates on active session only
    current_session = session_service.get_current_session()
    if current_session is None:
        err_console.print("[error]No active session.[/] Use [code]pb session list[/] to see recent sessions or [code]pb resume[/] to continue a paused task.")
        raise typer.Exit(code=ExitCode.NOT_FOUND)

    task = repo.get_task(current_session.task_id)
    task_title = task.title if task else "Unknown"
    task_completion = task.completion if task is not None else 0

    outcome = "done"
    inline_words = list(note_words or [])
    if completion is None and inline_words and inline_words[0].lower() in {"done", "partial", "blocked"}:
        outcome = inline_words.pop(0).lower()

    note = " ".join(inline_words).strip() if inline_words else None

    if completion is not None:
        completion_pct = completion
    elif outcome == "partial":
        completion_pct = max(task_completion, 50)
    elif outcome == "blocked":
        completion_pct = task_completion
    else:
        completion_pct = 100

    finish_note = note or outcome
    branch_value = getattr(current_session, "branch", None)
    branch = branch_value.lower() if isinstance(branch_value, str) and branch_value else "study"

    closeout = CloseoutService().generate_closeout(
        current_session,
        finish_note,
        {"title": task_title},
    )
    feedback_service = FeedbackProposalService()
    if branch in {"study", "practise", "practice"} and closeout.status in {
        "no_progress",
        "frustration_feedback",
        "accidental_start",
    }:
        proposal = None
        proposal_path = None
        if closeout.feedback_note or feedback_service.looks_like_feedback(finish_note):
            proposal = feedback_service.generate_proposal(closeout.feedback_note or finish_note, scope="learn")
            proposal_path = feedback_service.write_proposal(
                ctx.obj["runtime"].vault_path,
                closeout.feedback_note or finish_note,
                proposal,
                scope="learn",
            )
        if sys.stdin.isatty() and not yes:
            console.print(closeout.summary)
            if proposal_path is not None:
                console.print(f"[dim]Feedback proposal:[/] {proposal_path.relative_to(ctx.obj['runtime'].vault_path)}")
            choice = pick_single_choice(
                [
                    ("discard", "Discard session"),
                    ("keep_failed", "Keep as failed attempt"),
                    ("feedback", "Convert to feedback"),
                    ("recovery", "Create recovery step"),
                ],
                title="No-progress closeout",
                text="Choose how to close this session.",
            )
            if choice == "discard":
                _discard_active_session(repo, session_service, current_session, task)
                console.print(f"[success]Discarded: {escape(task_title)}[/]")
                return
            if choice == "feedback":
                _discard_active_session(repo, session_service, current_session, task)
                if proposal is not None and proposal.preference_patches:
                    console.print(proposal.summary)
                console.print("[success]Captured as product feedback.[/]")
                return
            if choice == "recovery":
                completion_pct = 0
                recovery = _create_recovery_task(
                    repo,
                    task,
                    note=closeout.recovery_step or finish_note,
                    branch=branch,
                ) if task is not None else None
                if recovery is not None:
                    console.print(f"[success]Recovery step created:[/] {escape(recovery.title)}")
            else:
                completion_pct = 0
        else:
            completion_pct = 0

    # Finish the session (DB write first — per A4 in RESEARCH.md)
    session = session_service.finish_session(note=finish_note, completion_pct=completion_pct)
    if session is None:
        err_console.print("[error]No active session.[/] Use [code]pb session list[/] to see recent sessions or [code]pb resume[/] to continue a paused task.")
        raise typer.Exit(code=ExitCode.NOT_FOUND)

    branch_value = getattr(session, "branch", None) or getattr(current_session, "branch", None)
    branch = branch_value.lower() if isinstance(branch_value, str) and branch_value else "study"

    try:
        from pb.core.lesson_engine import LessonNoteWriter

        lesson_run = repo.get_lesson_run(session.id)
        if lesson_run is not None:
            runtime = runtime_for_ctx(ctx)
            meta = parse_learning_task_metadata(task)
            topic = getattr(session, "subject_scope", "") or meta.scope or getattr(task, "title", "") or "lesson"
            domain = meta.domain or getattr(session, "subject_scope", "") or topic
            lesson_note_path = LessonNoteWriter(ctx.obj["runtime"], runtime).write_note(
                repo=repo,
                run=lesson_run,
                task=task,
                session=session,
                topic=topic,
                domain=domain,
            )
            if lesson_note_path is not None:
                lesson_run.note_path = str(lesson_note_path)
                lesson_run.updated_at = datetime.utcnow().isoformat()
                repo.update_lesson_run(lesson_run)
                generated_names = dict(getattr(session, "generated_names", {}) or {})
                generated_names["learning_partner_dossier_path"] = str(lesson_note_path)
                session.generated_names = generated_names
                repo.update_session(session)
                console.print(
                    f"[dim]Learning dossier:[/] {lesson_note_path.relative_to(ctx.obj['runtime'].vault_path)}"
                )
    except Exception:
        pass

    try:
        from pb.core.learner_memory import append_partner_session_memory

        runtime = runtime_for_ctx(ctx)
        memory_path = append_partner_session_memory(
            runtime=runtime,
            runtime_ctx=ctx.obj["runtime"],
            repo=repo,
            session=session,
            task=task,
        )
        if memory_path is not None:
            console.print(
                f"[dim]Learning dossier updated:[/] {memory_path.relative_to(ctx.obj['runtime'].vault_path)}"
            )
    except Exception:
        pass

    _advance_goal_progress_from_session(repo, session, completion_pct)
    unlocked_tasks = []
    if completion_pct >= 100:
        try:
            from pb.core.learning_curriculum import unlock_ready_curriculum_tasks

            unlocked_tasks = unlock_ready_curriculum_tasks(repo)
        except Exception:
            unlocked_tasks = []

    console.print(f"[success]Finished: {escape(task_title)}[/]")
    if completion_pct != 100:
        console.print(f"[dim]Completion: {completion_pct}%[/]")
    if unlocked_tasks:
        unlocked_titles = ", ".join(escape(item.title) for item in unlocked_tasks[:3])
        console.print(f"[dim]Unlocked next task(s): {unlocked_titles}[/]")

    # D-11: --debrief flag triggers Socratic debrief (opt-in only; no prompt without flag)
    if debrief:
        if branch != "study":
            console.print("[warn]`pb finish --debrief` is only available for study sessions.[/]")
            debrief = False
        else:
            try:
                socratic_service = ctx.obj['factory']['socratic_service']()
                from pb.cli.console import get_console as _get_console
                from pb.core.graph_writer import make_slug as _make_slug
                from pb.llm.gemini import FLASH_LITE_MODEL as _FLASH_LITE
                console_local = _get_console()
                qa_pairs = socratic_service.run_finish_debrief(
                    session=session, task=task, console=console_local
                )
                if qa_pairs:
                    domain = getattr(task, "domain", None) or getattr(session, "domain", None)
                    if domain:
                        all_answers = " ".join(a for _, a in qa_pairs)
                        slug = _make_slug(all_answers[:60]) or "finish-debrief"
                        socratic_service.build_and_submit(
                            qa_pairs=qa_pairs,
                            domain=domain,
                            slug=slug,
                            template="brief",
                            sync=False,
                            model=_FLASH_LITE,
                            console=console_local,
                        )
            except Exception as exc:
                # Non-fatal: --debrief failure must not block finish
                try:
                    from pb.cli.console import get_console as _gc
                    _gc().print(f"[warn]Debrief skipped: {exc}[/]")
                except Exception:
                    pass

    # Phase 17 GRPH-05, D-06: Update domain _state.md on finish
    try:
        from pb.vault import get_vault_path as _get_vault
        from pb.core.graph_writer import GraphWriter as _GraphWriter
        _vault = _get_vault()
        _writer = _GraphWriter(vault_path=_vault)
        knowledge_dir = _vault / "knowledge"
        if knowledge_dir.exists():
            for domain_dir in knowledge_dir.iterdir():
                if domain_dir.is_dir() and (domain_dir / "_state.md").exists():
                    _summary = f"{task_title}: {finish_note}"[:80]
                    _writer.update_state_md(domain_dir, _summary, _vault)
    except Exception:
        pass  # Non-fatal

    # Write GraphWriter task note for Obsidian compatibility
    try:
        from pb.vault.config import get_vault_path as _get_vault
        from pb.core.graph_writer import GraphWriter as _GraphWriter
        _vault = _get_vault()
        _writer = _GraphWriter(vault_path=_vault)
        project = None
        if task.project_id:
            project = repo.get_project(task.project_id)

        _writer.write_task_note(
            session=session,
            task=task,
            project=project,
            next_steps=[]
        )
    except Exception:
        pass  # Non-fatal

    # Phase 2: Evidence note (replaces SessionLogWriter per D-01)
    import structlog as _structlog
    evidence_path = None
    assessment = None
    try:
        from pb.core.domain_templates import _resolve_domain
        from pb.core.evidence_writer import EvidenceWriter, index_evidence_note

        domain = _resolve_domain(session, task)

        # Run assessment unless --skip/-q flag or non-learning branch
        if not skip and branch in {"study", "practise", "practice"}:
            try:
                from pb.core.finish_assessment import FinishAssessmentAgent
                agent = FinishAssessmentAgent()
                if agent.is_available():
                    assessment = agent.run(session, task, domain)
                    if assessment is not None:
                        generated_names = dict(getattr(session, "generated_names", {}) or {})
                        generated_names["finish_assessment"] = {
                            "critique": getattr(assessment, "critique", ""),
                            "sub_skill_scores": [
                                {
                                    "name": item.name,
                                    "score": item.score,
                                    "is_weak": item.is_weak,
                                }
                                for item in getattr(assessment, "sub_skill_scores", [])
                            ],
                            "retry_items": list(getattr(assessment, "retry_items", []) or []),
                        }
                        session.generated_names = generated_names
                        repo.update_session(session)
            except Exception as _assess_err:
                _structlog.get_logger().warning("finish_assessment.skipped", error=str(_assess_err))
        elif skip:
            # Per D-04: --skip prompts before writing bare evidence note in
            # interactive/manual mode, but --yes and session auto-yes must stay
            # non-blocking for evals, MCP, and shell automation.
            auto_yes = bool(yes or (ctx.obj and ctx.obj.get("yes")))
            if not auto_yes and not confirm_choice(
                "Skip AI assessment and save a lightweight learning summary instead?",
                default=False,
            ):
                raise typer.Exit(code=0)

        # Write evidence note
        _vault_ev = None
        try:
            from pb.vault.config import get_vault_path as _get_vault_path
            _vault_ev = _get_vault_path()
        except Exception:
            pass

        writer = EvidenceWriter(vault_path=_vault_ev)
        evidence_path = writer.write_evidence(session, task, assessment, domain)

        # Index in SQLite (write-through cache)
        if evidence_path:
            index_evidence_note(session, task, assessment, evidence_path, domain)

        # Enqueue retry items from assessment (D-10)
        if assessment and getattr(assessment, "retry_items", None):
            try:
                from pb.core.retry_queue import RetryQueueWriter
                rq = RetryQueueWriter()
                rq.enqueue_from_assessment(
                    domain=domain,
                    retry_items=assessment.retry_items,
                    evidence_id=session.id,
                )
            except Exception as _rq_err:
                _structlog.get_logger().warning("retry_queue.enqueue_skipped", error=str(_rq_err))

        # Auto-queue incomplete session goal (per D-14, EVID-10)
        # completion_pct is in scope from the outcome resolution block above.
        if completion_pct < 100:
            try:
                from pb.core.retry_queue import RetryQueueWriter as _RQW
                rq_incomplete = _RQW()
                session_goal = getattr(task, "title", None) or getattr(task, "name", None) or "Incomplete session"
                rq_incomplete.enqueue(
                    domain=domain,
                    item_text=session_goal,
                    source="incomplete",
                    priority=2,
                    evidence_id=session.id,
                )
            except Exception as _rq_inc_err:
                _structlog.get_logger().warning(
                    "retry_queue.incomplete_enqueue_skipped", error=str(_rq_inc_err)
                )

    except Exception as _ev_err:
        _structlog.get_logger().warning("evidence_writer.skipped", error=str(_ev_err))

    # Display evidence creation feedback and render note content (per D-21)
    if evidence_path is not None:
        try:
            _vault_for_display = _vault_ev if _vault_ev else evidence_path.parent
            display_path = evidence_path.relative_to(_vault_for_display)
        except Exception:
            display_path = evidence_path
        console.print("[dim]Evidence created:[/]")
        console.print(f"  - {display_path}")
        if assessment:
            weak_count = sum(1 for ss in assessment.sub_skill_scores if ss.is_weak)
            if weak_count:
                console.print(f"  - [yellow]{weak_count} weak sub-skill(s) queued for retry[/]")
        if completion_pct < 100:
            console.print(f"  - [cyan]Session goal queued for retry (incomplete: {completion_pct}%)[/]")

        # Per D-21: Render evidence note content with Glow when available, plain text fallback
        try:
            from pb.cli.markdown import render_markdown
            note_content = evidence_path.read_text()
            render_markdown(note_content)
        except Exception:
            pass  # Non-fatal: evidence was already written successfully

    created_follow_ons = []
    if completion_pct >= 100:
        try:
            created_follow_ons = _maybe_create_roadmap_follow_ons(ctx, repo, task, assessment)
        except Exception:
            created_follow_ons = []
    if created_follow_ons:
        created_titles = ", ".join(item.title for item in created_follow_ons[:3])
        console.print(f"[dim]Created next project task(s):[/] {escape(created_titles)}")

    next_cmd = _next_session_recommendation(repo, current_session.task_id)
    if next_cmd:
        console.print(f"[bold]Next:[/] [green]{next_cmd}[/]")

    overrun = _detect_session_overrun(repo, session)
    if overrun and overrun > 5:
        console.print(f"[yellow]Ran {overrun}min over — consider `pb study delete` to adjust today's plan[/]")


@app.command("now")
def show_now(
    ctx: typer.Context,
    json_out: bool = typer.Option(False, "--json", help="Output full session JSON"),
    plain_out: bool = typer.Option(False, "--plain", help="Pipe-friendly: task_name elapsed_min"),
):
    """Show current session status."""
    from pb.cli.display import format_now_output
    session_service = ctx.obj['factory']['session_service']()
    repo = ctx.obj['repo']
    console = get_console()
    err_console = get_err_console()

    session = session_service.get_current_session()
    task = None
    if session is not None:
        task = repo.get_task(session.task_id)

    mode = "json" if json_out else ("plain" if plain_out else "rich")
    output = format_now_output(session, task, mode=mode)

    if json_out or plain_out:
        # Plain/JSON output: use raw print for machine-parseable output
        import sys as _sys
        _sys.stdout.write(output + "\n")
        _sys.stdout.flush()
    else:
        if session is None:
            err_console.print("[error]No active session.[/]")
            raise typer.Exit(code=ExitCode.NOT_FOUND)
        console.print(output)
        # Show metadata line if interruptions or completion are non-zero
        interruptions = getattr(session, "interruption_count", 0)
        completion = getattr(session, "completion_pct", None)
        if interruptions or (completion is not None and completion != 100):
            parts = []
            if completion is not None:
                parts.append(f"Completion: {completion}%")
            if interruptions:
                parts.append(f"Interruptions: {interruptions}")
            console.print(f"[dim]{' | '.join(parts)}[/]")


@app.command("redo")
def redo_task(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to reopen (omit for picker)"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g., 30, 30m, 1h)"),
):
    """Reopen a finished task and start a new session on it.

    Shows completed tasks (100%) in a picker. Selecting one resets it
    to active and starts a focus session, same as pb start.
    """
    session_service = ctx.obj['factory']['session_service']()
    repo = ctx.obj['repo']
    console = get_console()
    err_console = get_err_console()
    from pb.domain.enums import TaskState

    if task_id is None:
        done_tasks = [
            t for t in repo.list_tasks()
            if t.completion >= 100 and t.archived_at is None
        ]
        if not done_tasks:
            err_console.print("[error]No finished tasks to redo.[/]")
            raise typer.Exit(code=ExitCode.NOT_FOUND)

        task = pick_or_prompt(
            done_tasks,
            title="Redo task",
        )
        if not task:
            raise typer.Exit(code=ExitCode.SUCCESS)
    else:
        task = _find_task(repo, task_id)

    task.state = TaskState.ACTIVE
    task.completion = 0
    repo.update_task(task)

    duration_minutes = None
    if duration:
        parsed = parse_duration(duration)
        if parsed is None:
            err_console.print(f"[error]Invalid duration format: {escape(duration)}[/]")
            raise typer.Exit(code=ExitCode.BAD_INPUT)
        duration_minutes = parsed

    try:
        session_service.start_session(
            task_id=task.id,
            mode="focus",
            duration_minutes=duration_minutes,
            timer_mode="timer" if duration_minutes else "stopwatch",
        )
        console.print(f"[success]Reopened + started: {escape(task.title)}[/]")
        if duration_minutes:
            console.print(f"[dim]Duration: {duration_minutes}m[/]")
    except RuleViolation as e:
        err_console.print(f"[error]{escape(str(e))}[/]")
        raise typer.Exit(code=ExitCode.CONFLICT)
