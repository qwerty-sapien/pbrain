# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Planning commands - day/week planning, time blocks."""

from dataclasses import dataclass, field
import re
import sys
from datetime import datetime, timedelta
from typing import Optional

import typer
from rich.markup import escape
from rich.table import Table

from pb.cli.llm_guard import runtime_for_ctx
from pb.cli.preview import (
    markdown_learning_plan_lines,
    preview_decision,
    render_markdown_preview,
)
from pb.cli.console import get_console, get_err_console
from pb.cli.helpers import (
    confirm_choice,
    format_block_for_selection,
    format_task_for_selection,
    parse_duration,
    prompt_text,
    select_from_numbered_list,
)
from pb.cli.pickers import pick_many_choices, pick_single_choice
from pb.cli.task_scoring import score_task_interactively, task_missing_planning_scores
from pb.core.goal_roadmaps import ensure_goal_seed_tasks
from pb.core.intake import create_task
from pb.core.learning_tasks import ensure_time_block, materialize_learning_task
from pb.core.models import Task
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.core.staging import build_assumptions, build_learning_context, build_reflection
from pb.llm.drafts import LearningPlanBlockDraft, MixedPlanDraft, artifact_presentation_prompt
from pb.llm.runtime import DraftGenerationError
from pb.core.planner import Planner
from pb.storage.repository import Repository


def parse_time(raw: str) -> datetime:
    """
    Parse HH:MM, H:MM, or HHMM into today's datetime at that wall-clock time.

    Per D-01: Accept flexible numeric formats (09:00, 9:00, 0900).
    Raises ValueError on unrecognized format.
    """
    raw = raw.strip()
    if re.fullmatch(r"\d{3,4}", raw):
        raw = raw.zfill(4)
        raw = raw[:2] + ":" + raw[2:]
    m = re.fullmatch(r"(\d{1,2}):(\d{2})", raw)
    if not m:
        raise ValueError(raw)
    hour, minute = int(m.group(1)), int(m.group(2))
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        raise ValueError(raw)
    return datetime.utcnow().replace(hour=hour, minute=minute, second=0, microsecond=0)


def parse_budget(raw: str) -> int:
    """Parse a duration string into total minutes.

    Accepts: 4h, 240m, 2h30m, 1.5h, 90 (bare number = minutes).
    Returns total minutes as int. Raises ValueError on invalid or non-positive input.
    """
    normalized = (raw or "").strip()
    if not normalized:
        raise ValueError("Empty budget string")
    parsed = parse_duration(normalized)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Invalid budget format: {raw}")
    return parsed


_COMMITMENT_RE = re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*hours?\s*/\s*(?P<unit>day|week)\b", re.IGNORECASE)


def _goal_commitment_hours(goals) -> tuple[float, float]:
    """Return (hours_per_day, hours_per_week) parsed from active goal metrics."""
    per_day = 0.0
    per_week = 0.0
    for goal in goals:
        metric = getattr(goal, "primary_metric", None)
        if not metric:
            continue
        match = _COMMITMENT_RE.search(str(metric))
        if not match:
            continue
        value = float(match.group("value"))
        if match.group("unit").lower() == "day":
            per_day += value
        else:
            per_week += value
    return per_day, per_week


def _default_day_hours(repo: Repository) -> float:
    goals = repo.list_goal_arcs(status=None)
    per_day, per_week = _goal_commitment_hours(goals)
    total = per_day + (per_week / 7.0)
    return round(total if total > 0 else 4.0, 2)


def _default_week_hours(repo: Repository) -> float:
    goals = repo.list_goal_arcs(status=None)
    per_day, per_week = _goal_commitment_hours(goals)
    total = per_week + (per_day * 7.0)
    return round(total if total > 0 else 40.0, 2)


@dataclass
class DayPlanConsultation:
    priority_note: str = ""
    selected_task_ids: list[str] = field(default_factory=list)
    added_todos: list[str] = field(default_factory=list)
    cleared_todos: int = 0
    created_goal_titles: list[str] = field(default_factory=list)
    created_task_ids: list[str] = field(default_factory=list)


def _active_plan_candidates(repo: Repository) -> list[Task]:
    return [
        task
        for task in repo.list_tasks()
        if task.archived_at is None and task.completion < 100
    ]


def _task_goal_label(repo: Repository, task: Task) -> str:
    goal_ids = getattr(task, "linked_goal_arc_ids", []) or []
    if not goal_ids:
        return "unlinked"
    goal = repo.get_goal_arc(goal_ids[0])
    return goal.title if goal is not None else goal_ids[0]


def _pick_day_plan_tasks(repo: Repository, *, preselected_ids: list[str] | None = None) -> list[str]:
    tasks = _active_plan_candidates(repo)
    if not tasks or not sys.stdin.isatty():
        return preselected_ids or []
    preselected = set(preselected_ids or [])
    options = []
    details = []
    for task in tasks:
        label = task.title
        if task.id in preselected:
            label += " *"
        options.append((task.id, label))
        details.append(
            "\n".join(
                [
                    f"Goal/Project: {_task_goal_label(repo, task)}",
                    f"Type: {getattr(task, 'work_type', '') or 'task'}",
                    f"Description: {task.description or '-'}",
                ]
            )
        )
    picked = pick_many_choices(
        options,
        title="Plan inputs",
        text="Select the tasks or todos you want today's plan to explicitly consider.",
        details=details,
    )
    return picked or (preselected_ids or [])


def _create_goal_linked_task(ctx: typer.Context, repo: Repository) -> Task | None:
    goals = repo.list_goal_arcs(status=None)
    if not goals or not sys.stdin.isatty():
        return None
    selected_goal_id = pick_single_choice(
        [(goal.id, goal.title) for goal in goals],
        title="Assign new task",
        text="Choose the existing project/goal this new task belongs to.",
    )
    if not selected_goal_id:
        return None
    goal = next((item for item in goals if item.id == selected_goal_id), None)
    if goal is None:
        return None
    title = prompt_text("New task title", default="").strip()
    if not title:
        return None
    task = Task(
        title=title,
        description=f"Created during day-plan intake for goal {goal.title}.",
        work_type="todo",
        linked_goal_arc_ids=[goal.id],
    )
    repo.create_task(task)
    get_console().print(f"[success]Task added to {escape(goal.title)}:[/] {escape(task.title)}")
    return task


def _archive_active_todos(repo: Repository) -> int:
    todos = [
        task
        for task in repo.list_tasks()
        if task.archived_at is None
        and task.completion < 100
        and (getattr(task, "work_type", "") or "").lower() == "todo"
    ]
    for task in todos:
        repo.archive_task(task.id)
    return len(todos)


def _consult_day_plan(ctx: typer.Context, repo: Repository) -> DayPlanConsultation:
    consultation = DayPlanConsultation()
    if not sys.stdin.isatty():
        return consultation

    consultation.selected_task_ids = _pick_day_plan_tasks(repo)

    while True:
        action = pick_single_choice(
            [
                ("continue", "Continue to draft"),
                ("pick_tasks", "Re-pick tasks"),
                ("add_task", "Add goal-linked task"),
                ("add_todo", "Add ad-hoc todo"),
                ("clear_todo", "Clear todo list"),
                ("create_project", "Create new project"),
            ],
            title="Planning prep",
            text="Adjust what the planner should consider before the draft is generated.",
        )
        if action in {None, "continue"}:
            break
        if action == "pick_tasks":
            consultation.selected_task_ids = _pick_day_plan_tasks(
                repo,
                preselected_ids=consultation.selected_task_ids,
            )
            continue
        if action == "add_task":
            task = _create_goal_linked_task(ctx, repo)
            if task is not None:
                consultation.created_task_ids.append(task.id)
                consultation.selected_task_ids.append(task.id)
            continue
        if action == "add_todo":
            todo_text = prompt_text("Todo", default="").strip()
            if todo_text:
                task = create_task(repo, todo_text)
                consultation.added_todos.append(task.title)
                consultation.selected_task_ids.append(task.id)
                get_console().print(f"[success]Todo captured:[/] {escape(task.title)}")
            continue
        if action == "clear_todo":
            count = _archive_active_todos(repo)
            consultation.cleared_todos += count
            if count:
                get_console().print(f"[success]Archived {count} todo item(s).[/]")
            else:
                get_console().print("[dim]No active todos to clear.[/]")
            consultation.selected_task_ids = [
                task_id
                for task_id in consultation.selected_task_ids
                if (repo.get_task(task_id) is not None)
            ]
            continue
        if action == "create_project":
            from pb.cli.commands.goals import _create_goal_via_llm

            raw_goal = prompt_text("New project", default="").strip()
            if not raw_goal:
                continue
            goal = _create_goal_via_llm(ctx, raw_goal, yes=False)
            if goal is not None:
                consultation.created_goal_titles.append(goal.title)
                get_console().print(f"[success]Project created:[/] {escape(goal.title)}")
            continue
    return consultation


def _format_minutes_label(minutes: int) -> str:
    hours, mins = divmod(max(0, minutes), 60)
    if hours and mins:
        return f"{hours}h {mins}min"
    if hours:
        return f"{hours}h"
    return f"{mins}min"


def _parse_work_hours_input(raw: str) -> int:
    normalized = (raw or "").strip().lower()
    if not normalized:
        raise ValueError("Empty work-hours string")
    if re.fullmatch(r"\d+(?:\.\d+)?", normalized):
        hours = float(normalized)
        minutes = int(round(hours * 60))
        if minutes <= 0:
            raise ValueError("Work hours must be positive.")
        return minutes
    parsed = parse_duration(normalized)
    if parsed is None or parsed <= 0:
        raise ValueError(f"Invalid work-hours format: {raw}")
    return parsed


def _prompt_work_budget_minutes(label: str, default_hours: float) -> int:
    raw = prompt_text(label, default=f"{default_hours:g}")
    try:
        minutes = _parse_work_hours_input(raw)
    except ValueError:
        minutes = int(round(default_hours * 60))
    return minutes if minutes > 0 else int(round(default_hours * 60))


def _ensure_weekly_task_scores(repo: Repository, *, skip: bool = False) -> bool:
    pending = [
        task
        for task in repo.list_tasks()
        if task.archived_at is None
        and task.completion < 100
        and str(getattr(task, "state", "")).lower().endswith("active")
        and task_missing_planning_scores(task)
    ]
    if not pending:
        return True
    if skip:
        return True
    console = get_console()
    console.print(f"[warn]{len(pending)} task(s) need scoring before the weekly plan can be generated.[/]")
    for task in pending:
        if not score_task_interactively(repo, task):
            return False
    return True


def _scale_plan_blocks(draft: MixedPlanDraft, factor: float, *, budget_minutes: int | None) -> MixedPlanDraft:
    scaled = draft.model_copy(deep=True)
    for block in scaled.blocks:
        block.duration_minutes = max(15, min(240, int(round(block.duration_minutes * factor))))
    if budget_minutes is not None:
        _cap_plan_to_budget(scaled, budget_minutes)
    return scaled


def _cap_plan_to_budget(draft: MixedPlanDraft, budget_minutes: int) -> MixedPlanDraft:
    remaining = max(0, budget_minutes)
    kept: list[LearningPlanBlockDraft] = []
    for block in draft.blocks:
        if remaining <= 0:
            break
        block_copy = block.model_copy(deep=True)
        block_copy.duration_minutes = min(block_copy.duration_minutes, remaining)
        kept.append(block_copy)
        remaining -= block_copy.duration_minutes
    draft.blocks = kept
    return draft


def _apply_day_plan_refinement(
    draft: MixedPlanDraft,
    instruction: str,
    *,
    budget_minutes: int | None,
) -> MixedPlanDraft | None:
    lowered = " ".join((instruction or "").lower().split())
    if not lowered:
        return None
    if "half" in lowered and any(token in lowered for token in ("time", "duration", "minutes", "hours", "long")):
        return _scale_plan_blocks(draft, 0.5, budget_minutes=budget_minutes)
    if "double" in lowered and any(token in lowered for token in ("time", "duration", "minutes", "hours", "long")):
        return _scale_plan_blocks(draft, 2.0, budget_minutes=budget_minutes)
    hours_match = re.search(r"(\d+(?:\.\d+)?)\s*hours?", lowered)
    if hours_match:
        minutes = int(float(hours_match.group(1)) * 60)
        updated = draft.model_copy(deep=True)
        return _cap_plan_to_budget(updated, minutes)
    minutes_match = re.search(r"(\d+)\s*(?:minutes|min)", lowered)
    if minutes_match:
        updated = draft.model_copy(deep=True)
        return _cap_plan_to_budget(updated, int(minutes_match.group(1)))
    block_match = re.search(r"(\d+)\s*blocks?", lowered)
    if block_match:
        target = max(1, int(block_match.group(1)))
        updated = draft.model_copy(deep=True)
        updated.blocks = updated.blocks[:target]
        if budget_minutes is not None:
            _cap_plan_to_budget(updated, budget_minutes)
        return updated
    if any(token in lowered for token in ("less", "smaller", "shorter", "reduce")) and any(
        token in lowered for token in ("time", "duration", "minutes", "hours", "budget")
    ):
        return _scale_plan_blocks(draft, 0.75, budget_minutes=budget_minutes)
    if any(token in lowered for token in ("more", "longer", "expand")) and any(
        token in lowered for token in ("time", "duration", "minutes", "hours", "budget")
    ):
        return _scale_plan_blocks(draft, 1.25, budget_minutes=budget_minutes)
    return None


def _show_people_prompts():
    """Show proactive relationship prompts. Non-fatal: vault errors silently skipped."""
    try:
        from pb.core.prompts import ProactivePromptsEngine

        engine = ProactivePromptsEngine()
        prompts = engine.get_prompts()
        if prompts:
            console = get_console()
            console.rule("[header]Relationship Reminders[/]")
            for p in prompts[:5]:  # Show top 5 most urgent
                icons = {"overdue_commitment": "!", "birthday": "*",
                         "gift_reminder": "~", "decay_warning": "?"}
                icon = icons.get(p.prompt_type, "-")
                console.print(f"  [{icon}] {escape(p.person_name)}: {escape(p.message)}")
            if len(prompts) > 5:
                console.print(f"  [dim]... and {len(prompts) - 5} more (run 'pb prompts' for all)[/]")
            console.print("")
    except Exception:
        pass  # Non-fatal: prompt failure does not affect daily plan


app = typer.Typer()


def _resolve_threshold(domain_name: str, thresholds: dict) -> int:
    """Resolve decay threshold: exact match -> prefix match -> _default.

    Phase 19 D-15: Per-domain decay thresholds from config.toml learning.decay_thresholds.
    """
    lower = domain_name.lower()
    if lower in thresholds:
        return thresholds[lower]
    for key, val in thresholds.items():
        if key != "_default" and lower.startswith(key):
            return val
    return thresholds.get("_default", 5)


def _recent_learning_sessions(repo: Repository, limit: int = 8) -> list[str]:
    rows: list[tuple[datetime, str]] = []
    for task in repo.list_tasks():
        for session in repo.list_sessions_for_task(task.id):
            summary = f"{session.branch}:{session.subject_scope}:{session.actual_outcome or ''}"
            rows.append((session.start_at, summary))
    rows.sort(key=lambda item: item[0], reverse=True)
    return [summary for _, summary in rows[:limit]]


def _selected_consultation_tasks(repo: Repository, consultation: DayPlanConsultation | None) -> list[Task]:
    selected: list[Task] = []
    seen: set[str] = set()
    for task_id in consultation.selected_task_ids if consultation is not None else []:
        task = repo.get_task(task_id)
        if task is None or task.id in seen:
            continue
        seen.add(task.id)
        selected.append(task)
    return selected


def _consultation_summary(consultation: DayPlanConsultation | None, repo: Repository) -> str:
    if consultation is None:
        return ""
    lines: list[str] = []
    if consultation.priority_note:
        lines.append(f"- Priority note: {consultation.priority_note}")
    selected_tasks = _selected_consultation_tasks(repo, consultation)
    if selected_tasks:
        lines.append("- Explicitly selected tasks:")
        for task in selected_tasks:
            lines.append(f"  - {task.title} | goal={_task_goal_label(repo, task)} | type={getattr(task, 'work_type', '') or 'task'}")
    if consultation.added_todos:
        lines.append("- Added todos:")
        for item in consultation.added_todos:
            lines.append(f"  - {item}")
    if consultation.cleared_todos:
        lines.append(f"- Cleared todos: {consultation.cleared_todos}")
    if consultation.created_goal_titles:
        lines.append("- Created projects:")
        for title in consultation.created_goal_titles:
            lines.append(f"  - {title}")
    return "\n".join(lines)


def _plan_day_prompt(
    repo: Repository,
    budget_minutes: int | None,
    *,
    consultation: DayPlanConsultation | None = None,
) -> str:
    goals = repo.list_goal_arcs(status=None)
    goal_lines = []
    for goal in goals:
        goal_lines.append(
            f"- id={goal.id} | title={goal.title} | domain={goal.domain} | mode={goal.execution_mode} | "
            f"study_target={goal.target_bloom_stage.value if goal.target_bloom_stage else ''} | "
            f"practice_target={goal.target_practice_stage.value if goal.target_practice_stage else ''} | "
            f"success={goal.success_definition}"
        )
    recent = "\n".join(f"- {row}" for row in _recent_learning_sessions(repo))
    try:
        from pb.vault.anki_client import get_pending_card_count

        pending_anki = get_pending_card_count()
    except Exception:
        pending_anki = 0
    active_tasks = _active_plan_candidates(repo)
    task_lines = [
        f"- {task.title} | goal={_task_goal_label(repo, task)} | type={getattr(task, 'work_type', '') or 'task'} | est={getattr(task, 'estimated_minutes', None) or '-'}"
        for task in active_tasks[:12]
    ]
    consultation_text = _consultation_summary(consultation, repo)
    if not goals and not active_tasks and not consultation_text:
        return ""
    return (
        "Create today's executable learning plan.\n"
        "Return a serious but low-ceremony set of study and practise blocks with concrete evidence.\n"
        "Prefer prerequisite-first sequencing when the learner sounds underprepared.\n"
        "Every block's `subject_scope` must name the exact competency slice for that block, not merely paraphrase the goal or task title.\n"
        "For advanced or long-horizon goals, distinguish prerequisite progress from goal progress.\n"
        "If readiness is unproven, schedule prerequisite-building blocks before target-layer application.\n"
        "Do not reallocate time arbitrarily. Preserve the total budget unless the user explicitly asked to change it.\n"
        "If the user selected tasks, anchor the plan to those tasks instead of drifting to adjacent topics.\n"
        f"Total time budget minutes: {budget_minutes or 240}\n"
        f"Accepted/edited Anki candidates awaiting export: {pending_anki}\n\n"
        "Active goals:\n"
        f"{chr(10).join(goal_lines) or '- none'}\n\n"
        "Active tasks and todos:\n"
        f"{chr(10).join(task_lines) or '- none'}\n\n"
        "Planning consultation:\n"
        f"{consultation_text or '- none'}\n\n"
        "Recent sessions:\n"
        f"{recent or '- none'}\n"
        "\n"
        "MICROTASK DECOMPOSITION:\n"
        "When a block covers multiple sub-topics or would benefit from smaller focused sessions, "
        "split it into sub-blocks with `sub_index` values like '2a', '2b', '2c'.\n"
        "Example: 'Multivariable calculus and linalg in semi-Riemannian geometry' becomes:\n"
        "  - 2a: Multivariable calculus core (30 min)\n"
        "  - 2b: Linear algebra for semi-Riemannian geometry (20 min)\n"
        "  - 2c: Connecting calc + linalg in the geometric context (10 min)\n"
        "Only decompose when scope spans distinct sub-topics or prerequisite chains. "
        "A single focused topic should remain one block.\n"
        f"{artifact_presentation_prompt()}"
    )


def _build_day_plan_refinement_prompt(
    *,
    base_prompt: str,
    draft: MixedPlanDraft,
    instruction: str,
    budget_minutes: int | None,
) -> str:
    return (
        f"{base_prompt}\n\n"
        "Refine the existing draft below using the learner's latest correction.\n"
        "Only change durations when the learner explicitly asked for more or less time, or when a prerequisite shift makes a small rebalance necessary.\n"
        "If the learner says they are missing foundations, make the plan more prerequisite-first and reduce premature application.\n"
        f"Budget minutes: {budget_minutes or 0}\n"
        f"Existing draft JSON: {draft.model_dump_json()}\n"
        f"Learner refinement request: {instruction.strip()}\n"
    )


def _default_block_duration(budget_minutes: int | None, slots: int = 1) -> int:
    baseline = budget_minutes or 90
    return max(20, min(60, int(baseline / max(1, slots))))


def _build_quick_plan(repo: Repository, budget_minutes: int | None) -> MixedPlanDraft:
    """Create a deterministic low-ceremony plan without an LLM."""

    goals = repo.list_goal_arcs(status=None)
    blocks: list[LearningPlanBlockDraft] = []
    if not goals:
        active_tasks = _active_plan_candidates(repo)
        remaining = budget_minutes or 90
        for task in active_tasks[:3]:
            if remaining <= 0:
                break
            duration = min(_default_block_duration(remaining, slots=max(1, len(active_tasks[:3]))), remaining)
            branch = "practise" if (getattr(task, "work_type", "") or "").lower() == "practice" else "study"
            blocks.append(
                LearningPlanBlockDraft(
                    branch=branch,
                    subject_scope=task.title,
                    duration_minutes=duration,
                    target_bloom_stage="apply" if branch == "study" else None,
                    practice_stage="integrate" if branch == "practise" else None,
                    drill_type=task.title if branch == "practise" else None,
                    success_check=f"Make concrete progress on {task.title}.",
                    reason=f"Deterministic quick-plan block derived from the active task {task.title}.",
                )
            )
            remaining -= duration
        return MixedPlanDraft(summary="Deterministic quick plan generated from active tasks.", blocks=blocks)

    remaining = budget_minutes or 90
    for goal in goals[:3]:
        focus = goal.domain or goal.title
        per_block = _default_block_duration(remaining, slots=max(1, len(goals)))
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        if mode in {"study", "mixed"} and remaining > 0:
            duration = min(per_block, remaining)
            blocks.append(
                LearningPlanBlockDraft(
                    goal_id=goal.id,
                    branch="study",
                    subject_scope=focus,
                    duration_minutes=duration,
                    target_bloom_stage=getattr(goal, "target_bloom_stage", None) or "apply",
                    study_mode="active recall",
                    success_check=f"Summarize or retrieve the core ideas behind {focus} without rereading.",
                    reason=f"Advance the conceptual side of {goal.title}.",
                )
            )
            remaining -= duration
        if mode in {"practise", "practice", "mixed"} and remaining > 0:
            duration = min(per_block, remaining)
            blocks.append(
                LearningPlanBlockDraft(
                    goal_id=goal.id,
                    branch="practise",
                    subject_scope=focus,
                    duration_minutes=duration,
                    practice_stage=getattr(goal, "target_practice_stage", None) or "integrate",
                    drill_type=focus,
                    feedback_source=getattr(goal, "feedback_source", None) or "artifact",
                    evidence_target=f"One concrete artifact or clean rep set for {focus}.",
                    success_check=f"Finish one deliberate-practice block for {focus}.",
                    reason=f"Advance the embodied side of {goal.title}.",
                )
            )
            remaining -= duration
        if remaining <= 0:
            break

    summary = "Deterministic quick plan generated from active goals."
    return MixedPlanDraft(summary=summary, blocks=blocks)


def _manual_plan_day_wizard(repo: Repository, budget_minutes: int | None) -> MixedPlanDraft:
    """Collect a few executable blocks when model synthesis is unavailable."""

    typer.echo("Manual day planning")
    typer.echo("Tip: keep each block concrete. One scope, one branch, one success check.")

    blocks: list[LearningPlanBlockDraft] = []
    remaining = budget_minutes or 120
    while remaining > 0:
        scope = prompt_text("Block focus", default="")
        if not scope.strip():
            break
        branch = prompt_text("Branch (study/practise)", default="study").strip().lower() or "study"
        minutes_default = str(min(remaining, 45))
        raw_minutes = prompt_text("Duration minutes", default=minutes_default)
        try:
            duration_minutes = max(5, min(240, int(raw_minutes)))
        except ValueError:
            duration_minutes = min(remaining, 45)
        success = prompt_text("Success check", default=f"Finish one concrete {branch} block for {scope}.")
        if branch in {"practise", "practice"}:
            blocks.append(
                LearningPlanBlockDraft(
                    branch="practise",
                    subject_scope=scope,
                    duration_minutes=min(duration_minutes, remaining),
                    practice_stage="integrate",
                    drill_type=prompt_text("Drill type", default=scope) or scope,
                    feedback_source="artifact",
                    evidence_target=prompt_text("Evidence target", default=f"One artifact or clean rep set for {scope}."),
                    success_check=success,
                    reason=f"Manual deliberate-practice block for {scope}.",
                )
            )
        else:
            blocks.append(
                LearningPlanBlockDraft(
                    branch="study",
                    subject_scope=scope,
                    duration_minutes=min(duration_minutes, remaining),
                    target_bloom_stage="apply",
                    study_mode=prompt_text("Study mode", default="active recall") or "active recall",
                    success_check=success,
                    reason=f"Manual study block for {scope}.",
                )
            )
        remaining -= min(duration_minutes, remaining)
        if remaining <= 0:
            break
        if not confirm_choice("Add another block?", default=False):
            break
    return MixedPlanDraft(summary="Manual day plan.", blocks=blocks)


def _succinct_task_title(block, task) -> str:
    """Derive a short task label from available metadata."""
    if block.title:
        return block.title
    scope = block.subject_scope or ""
    if ":" in scope:
        return scope.split(":")[0].strip()
    words = scope.split()
    if len(words) <= 4:
        return scope
    return " ".join(words[:4]) + "..."


def _render_plan_rows(rows: list[tuple], total_minutes: int | None) -> None:
    console = get_console()
    title = f"Day Plan (budget {total_minutes} min) - pb study <code>" if total_minutes else "Day Plan - pb study <code>"
    table = Table(title=title, show_header=True, header_style="bold", show_edge=False, box=None)
    table.add_column("CODE", no_wrap=True, style="green bold")
    table.add_column("MODE", no_wrap=True)
    table.add_column("MIN", justify="right")
    table.add_column("SCOPE", ratio=3, overflow="fold")
    table.add_column("TASK", ratio=2, overflow="fold")
    elapsed = 0
    for idx, (block, task) in enumerate(rows, start=1):
        code = getattr(block, "sub_index", None) or str(idx)
        task_label = block.title or _succinct_task_title(block, task)
        table.add_row(
            code,
            block.branch,
            str(block.duration_minutes),
            escape(block.subject_scope),
            escape(task_label),
        )
        elapsed += block.duration_minutes
    console.print(table)
    if total_minutes:
        remaining = total_minutes - elapsed
        if remaining == 0:
            console.print(f"[dim]Planned {elapsed} minutes, using the full {total_minutes}-minute budget.[/]")
        elif remaining > 0:
            console.print(f"[dim]Planned {elapsed} of {total_minutes} budgeted minutes; {remaining} minutes remain unscheduled.[/]")
        else:
            console.print(f"[dim]Planned {elapsed} minutes, {abs(remaining)} over the {total_minutes}-minute budget.[/]")
    else:
        console.print(f"[dim]Planned {elapsed} minutes of goal-aligned learning work.[/]")
    console.print("[dim]Start the next planned block with `pb study plan`, or jump to a code with `pb study <code>`.[/]")


@app.command("day")
def plan_day(
    ctx: typer.Context,
    budget: Optional[str] = typer.Option(None, "--budget", help="Time budget (e.g., 4h, 240m, 2h30m)"),
    hours: Optional[str] = typer.Option(None, "--hours", help="Available work time today (e.g., 1h 10min, 5 hours, 2 minutes, 1.5)"),
    quick: bool = typer.Option(False, "--quick", "-q", help="Skip interactive prompts, use defaults"),
    skip_scoring: bool = typer.Option(False, "--skip-scoring", help="Compatibility alias for --quick in non-interactive evals"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and materialize the plan"),
):
    """Draft and materialize today's learning plan."""
    if skip_scoring:
        quick = True
    runtime = runtime_for_ctx(ctx)
    repo = ctx.obj["repo"] if ctx.obj and ctx.obj.get("repo") is not None else Repository()
    ensure_goal_seed_tasks(repo, repo.list_goal_arcs(status=None), vault_path=ctx.obj["runtime"].vault_path)
    budget_minutes = None
    if budget is not None and hours is not None:
        err_console = get_err_console()
        err_console.print("[error]Use either --budget or --hours, not both.[/]")
        raise typer.Exit(code=1)
    if budget is not None:
        try:
            budget_minutes = parse_budget(budget)
        except ValueError:
            err_console = get_err_console()
            err_console.print(f"[error]Invalid budget format: {escape(budget)}. Use e.g. 4h, 240m, 2h30m[/]")
            raise typer.Exit(code=1)
    elif hours is not None:
        try:
            budget_minutes = _parse_work_hours_input(hours)
        except ValueError:
            get_err_console().print(
                f"[error]Invalid hours format: {escape(hours)}. Use e.g. 1h 10min, 5 hours, 2 minutes, or 1.5[/]"
            )
            raise typer.Exit(code=1)
    else:
        default_hours = _default_day_hours(repo)
        resolved_hours = (
            _prompt_work_budget_minutes("Available work hours today", default_hours)
            if sys.stdin.isatty()
            else int(round(default_hours * 60))
        )
        budget_minutes = resolved_hours

    consultation = DayPlanConsultation()
    if not quick:
        consultation = _consult_day_plan(ctx, repo)

    prompt = _plan_day_prompt(repo, budget_minutes, consultation=consultation)
    if not prompt:
        planner = Planner(repo)
        console = get_console()
        console.print(planner.generate_daily_plan(budget_minutes=budget_minutes))
        return

    runtime_ctx = ctx.obj["runtime"]
    prompt += feedback_prompt_suffix(runtime_ctx.vault_path, "plan")
    recorder = runtime.make_stage_recorder("plan_day", budget or "today", route_hint="plan day")
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("plan_day", budget or "today", context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("plan_day", budget or "today", context))
    recorder.add(
        "consult",
        {
            "priority_note": consultation.priority_note,
            "selected_task_ids": consultation.selected_task_ids,
            "added_todos": consultation.added_todos,
            "cleared_todos": consultation.cleared_todos,
            "created_goals": consultation.created_goal_titles,
        },
    )
    if sys.stdin.isatty():
        get_console().print(f"[dim]{reflection}[/]")

    draft_result = None
    if quick:
        draft = _build_quick_plan(repo, budget_minutes)
        recorder.add("draft", {"mode": "quick", "blocks": len(draft.blocks)})
    else:
        try:
            draft_result = runtime.generate_draft(
                MixedPlanDraft,
                prompt,
                source_scope="plan_day",
            )
            draft = draft_result.payload
            recorder.add(
                "draft",
                {
                    "model": draft_result.model,
                    "attempts": [attempt.__dict__ for attempt in draft_result.attempts],
                },
            )
        except DraftGenerationError as exc:
            recorder.add(
                "draft",
                {
                    "error": exc.to_user_message(),
                    "attempts": [attempt.__dict__ for attempt in exc.attempts],
                },
                status="error",
            )
            if sys.stdin.isatty():
                get_err_console().print(
                    f"[warn]{exc.to_user_message()}[/]\n"
                    "[dim]Falling back to a deterministic quick plan instead of the old manual wizard.[/]"
                )
            draft = _build_quick_plan(repo, budget_minutes)
            recorder.add("fallback", {"mode": "quick_plan", "blocks": len(draft.blocks)})

    if not draft.blocks:
        get_err_console().print("[error]No plan blocks available.[/]")
        recorder.finalize("empty")
        raise typer.Exit(code=1)

    recorder.add("verify", {"preview": draft.model_dump(mode="json")})
    _total_min = sum(b.duration_minutes for b in draft.blocks)
    render_markdown_preview(
        title="Day Plan Draft",
        rows=[
            ("Planned time", f"{_total_min} min"),
        ],
        sections=[
            ("Plan", markdown_learning_plan_lines(draft.blocks, presentation=draft.presentation)),
        ],
    )
    if quick and yes:
        accepted = True
    else:
        accepted = False
        while True:
            decision = preview_decision(yes=yes, action_label="Materialize today's plan")
            if decision.kind == "accept":
                accepted = True
                break
            if decision.kind == "cancel":
                break
            refined = _apply_day_plan_refinement(draft, decision.text, budget_minutes=budget_minutes)
            if refined is not None:
                draft = refined
            elif not quick:
                try:
                    draft = runtime.generate_draft(
                        MixedPlanDraft,
                        _build_day_plan_refinement_prompt(
                            base_prompt=prompt,
                            draft=draft,
                            instruction=decision.text,
                            budget_minutes=budget_minutes,
                        ),
                        source_scope="plan_day_refine",
                    ).payload
                    if budget_minutes is not None:
                        _cap_plan_to_budget(draft, budget_minutes)
                except DraftGenerationError:
                    get_err_console().print("[warn]Couldn't refine that draft automatically; keeping the current plan.[/]")
            _total_min = sum(b.duration_minutes for b in draft.blocks)
            render_markdown_preview(
                title="Day Plan Draft",
                rows=[
                    ("Planned time", f"{_total_min} min"),
                ],
                sections=[
                    ("Plan", markdown_learning_plan_lines(draft.blocks, presentation=draft.presentation)),
                ],
            )

    if not accepted:
        if draft_result is not None:
            repo.create_generation_provenance(
                runtime.build_provenance(
                    artifact_kind="plan_day",
                    artifact_id="today",
                    generated_draft=draft_result,
                    accepted_by_user=False,
                )
            )
        recorder.finalize("cancelled", artifact_kind="plan_day", artifact_id="today")
        raise typer.Exit(code=0)

    rows: list[tuple] = []
    remaining = budget_minutes
    for block in draft.blocks:
        if remaining is not None and remaining <= 0:
            break
        if remaining is not None and block.duration_minutes > remaining:
            block.duration_minutes = remaining
        if not block.subject_scope:
            goal = repo.get_goal_arc(block.goal_id) if block.goal_id else None
            block.subject_scope = goal.domain if goal and goal.domain else (goal.title if goal else "")
        task, _ = materialize_learning_task(repo, block)
        ensure_time_block(repo, task, block)
        rows.append((block, task))
        if remaining is not None:
            remaining -= block.duration_minutes

    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="plan_day",
                artifact_id="today",
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add("materialize", {"rows": len(rows), "budget_minutes": budget_minutes})
    recorder.finalize("persisted", artifact_kind="plan_day", artifact_id="today")
    _render_plan_rows(rows, budget_minutes)


@app.command("week")
def plan_week(
    ctx: typer.Context,
    hours: Optional[str] = typer.Option(None, "--hours", help="Available work time this week (e.g., 6h, 12 hours, 90 minutes, 40)"),
    skip_scoring: bool = typer.Option(False, "--skip-scoring", help="Skip interactive task scoring and proceed with existing scores"),
):
    """Generate capacity-aware weekly plan (D-27)."""
    repo = ctx.obj["repo"] if ctx.obj and ctx.obj.get("repo") is not None else Repository()
    vault_path = ctx.obj["runtime"].vault_path if ctx.obj and ctx.obj.get("runtime") is not None else None
    ensure_goal_seed_tasks(repo, repo.list_goal_arcs(status=None), vault_path=vault_path)
    planner = Planner(repo)

    if hours is None:
        default_hours = _default_week_hours(repo)
        minutes = _prompt_work_budget_minutes("Available work hours this week", default_hours) if sys.stdin.isatty() else int(round(default_hours * 60))
    else:
        try:
            minutes = _parse_work_hours_input(hours)
        except ValueError:
            get_err_console().print(
                f"[error]Invalid hours format: {escape(hours)}. Use e.g. 6h, 12 hours, 90 minutes, or 40[/]"
            )
            raise typer.Exit(code=1)

    if minutes <= 0:
        err_console = get_err_console()
        err_console.print("[error]Hours must be positive[/]")
        raise typer.Exit(code=1)

    if not _ensure_weekly_task_scores(repo, skip=skip_scoring):
        get_err_console().print("[error]Weekly planning requires scoring the remaining unscored tasks first.[/]")
        raise typer.Exit(code=1)

    plan = planner.generate_weekly_plan(available_hours=(minutes / 60.0))
    console = get_console()
    console.print(plan)


block_app = typer.Typer()
app.add_typer(block_app, name="block", hidden=True, help="Time block commands")


@block_app.command("add")
def add_block(
    duration_or_task: Optional[str] = typer.Argument(None, help="Duration (minutes) or task ID/name"),
    task_or_start: Optional[str] = typer.Argument(None, help="Task ID/name, or start time (HH:MM)"),
    start_or_duration: Optional[str] = typer.Argument(None, help="Start time or duration"),
    repeat: Optional[str] = typer.Option(None, "--repeat", help="Recurrence: daily or weekly"),
):
    """
    Add a time block for a task.

    Three modes (per D-08, D-10, D-11):

    1. Interactive: `pb plan block add`
       - Shows numbered task list, prompts for selection and duration

    2. Inline shorthand: `pb plan block add 60 task-name`
       - Duration first, then task prefix/name
       - Start time is optional (duration-only block)

    3. Full spec: `pb plan block add 60 task-name 09:00`
       - Duration, task, and optional start time
    """
    # Validate --repeat value (D-10)
    if repeat and repeat not in ("daily", "weekly"):
        err_console = get_err_console()
        err_console.print(f"[error]Invalid repeat value: {escape(repeat)}. Use 'daily' or 'weekly'.[/]")
        raise typer.Exit(code=1)

    repo = Repository()

    # Mode 1: Interactive (no arguments)
    if duration_or_task is None:
        return _add_block_interactive(repo, repeat=repeat)

    # Try to parse first arg as duration (numeric)
    try:
        duration = int(duration_or_task)
        # Mode 2/3: Inline shorthand - duration first
        if task_or_start is None:
            err_console = get_err_console()
            err_console.print("[error]Task required. Usage: pb plan block add <duration> <task>[/]")
            raise typer.Exit(code=1)

        task = _resolve_task_by_prefix_or_name(repo, task_or_start)
        start_time = None

        # Check if start time provided
        if start_or_duration is not None:
            try:
                start_time = parse_time(start_or_duration)
            except ValueError:
                err_console = get_err_console()
                err_console.print("[error]Invalid time format. Use HH:MM (e.g., 09:00 or 9:00)[/]")
                raise typer.Exit(code=1)

        return _create_block(repo, task, duration, start_time, repeat=repeat)

    except ValueError:
        # First arg is not numeric - treat as legacy task_id format
        # Legacy: pb plan block add <task_id> <start> <duration>
        task = _resolve_task_by_prefix_or_name(repo, duration_or_task)

        if task_or_start is None:
            err_console = get_err_console()
            err_console.print("[error]Start time or duration required.[/]")
            raise typer.Exit(code=1)

        try:
            start_time = parse_time(task_or_start)
        except ValueError:
            err_console = get_err_console()
            err_console.print("[error]Invalid time format. Use HH:MM (e.g., 09:00 or 9:00)[/]")
            raise typer.Exit(code=1)

        if start_or_duration is None:
            err_console = get_err_console()
            err_console.print("[error]Duration required. Usage: pb plan block add <task> <start> <duration>[/]")
            raise typer.Exit(code=1)

        try:
            duration = int(start_or_duration)
        except ValueError:
            err_console = get_err_console()
            err_console.print("[error]Duration must be a number[/]")
            raise typer.Exit(code=1)

        return _create_block(repo, task, duration, start_time, repeat=repeat)


def _resolve_task_by_prefix_or_name(repo: Repository, query: str):
    """Resolve task by exact ID, prefix, or title substring."""
    task = repo.resolve_task_ref(query)
    if task is not None:
        return task

    tasks = repo.list_tasks()
    matches = []
    for t in tasks:
        if t.id.startswith(query) or query.lower() in t.title.lower():
            matches.append(t)

    if len(matches) == 0:
        err_console = get_err_console()
        err_console.print(f"[error]Task not found: {escape(query)}[/]")
        raise typer.Exit(code=1)

    if len(matches) == 1:
        return matches[0]

    # Multiple matches - use numbered selection
    return select_from_numbered_list(
        matches,
        format_task_for_selection,
        prompt="Select task",
        header=f"Multiple tasks match '{query}':",
    )


def _add_block_interactive(repo: Repository, repeat: Optional[str] = None):
    """Interactive mode for adding a block per D-10."""
    tasks = [
        t
        for t in repo.list_tasks()
        if t.completion < 100 and t.state.value == "active"
    ]

    if not tasks:
        console = get_console()
        console.print("[dim]No tasks available for blocking. Capture or ready some tasks first.[/]")
        raise typer.Exit(code=0)

    task = select_from_numbered_list(
        tasks,
        format_task_for_selection,
        prompt="Select task",
        header="Tasks available for time blocking:",
    )

    # Prompt for duration
    try:
        duration = typer.prompt("Duration (minutes)", type=int)
    except typer.Abort:
        raise typer.Exit(code=0)

    if duration <= 0:
        err_console = get_err_console()
        err_console.print("[error]Duration must be positive[/]")
        raise typer.Exit(code=1)

    # Optional start time
    start_input = typer.prompt("Start time (HH:MM, or press Enter for no start time)", default="", show_default=False)
    start_time = None
    if start_input.strip():
        try:
            start_time = parse_time(start_input)
        except ValueError:
            err_console = get_err_console()
            err_console.print("[error]Invalid time format. Use HH:MM (e.g., 09:00 or 9:00)[/]")
            raise typer.Exit(code=1)

    return _create_block(repo, task, duration, start_time, repeat=repeat)


def _create_block(repo: Repository, task, duration: int, start_time, repeat: Optional[str] = None):
    """Create a block and display result. Optionally set up recurrence (D-10)."""
    if duration <= 0:
        err_console = get_err_console()
        err_console.print("[error]Duration must be positive[/]")
        raise typer.Exit(code=1)

    planner = Planner(repo)
    block, overlap = planner.schedule_block(task, start_time, duration)

    console = get_console()

    # Per D-03, D-04: Warn on overlap but continue
    if overlap and overlap.start_time:
        overlap_start = overlap.start_time.strftime("%H:%M")
        overlap_end = (overlap.start_time + timedelta(minutes=overlap.duration_minutes)).strftime("%H:%M")
        console.print(f"[warn]Note: overlaps with existing block {overlap_start}-{overlap_end}[/]")

    # D-10: Handle recurrence
    if repeat:
        from pb.domain.models import generate_internal_id
        block.series_id = generate_internal_id()
        block.recurrence_rule = repeat
        repo.update_time_block(block)
        instances = planner.generate_recurrence_instances(block)
        console.print(f"[success]Created recurring block ({escape(repeat)}): {escape(task.title)} - {duration}m. Generated {len(instances)} future instances.[/]")
    else:
        if start_time:
            time_str = f"at {start_time.strftime('%H:%M')}"
        else:
            time_str = "(no start time)"
        console.print(f"[success]Scheduled: {escape(task.title)} {time_str} for {duration}m[/]")


@block_app.command("list")
def list_blocks():
    """List today's time blocks with gap detection per D-12."""
    repo = Repository()
    planner = Planner(repo)
    blocks = planner.get_today_blocks()

    if not blocks:
        console = get_console()
        console.print("[dim]No blocks scheduled for today.[/]")
        return

    console = get_console()
    scheduled = [b for b in blocks if b.start_time is not None]
    unscheduled = [b for b in blocks if b.start_time is None]

    # Sort scheduled blocks by start time
    scheduled.sort(key=lambda b: b.start_time)

    if scheduled:
        console.rule("[header]Today's Scheduled Blocks[/]")
        t = Table(show_header=True, header_style="table.header",
                  show_edge=False, show_lines=False, pad_edge=False, box=None)
        t.add_column("#", justify="right", no_wrap=True)
        t.add_column("TIME", no_wrap=True)
        t.add_column("DUR", justify="right", no_wrap=True)
        t.add_column("TASK")
        t.add_column("STATE", justify="center", no_wrap=True)

        prev_end = None
        for i, block in enumerate(scheduled, 1):
            task = repo.get_task(block.task_id)
            task_title = task.title if task else "Unknown"
            task_state = task.state.value if task else ""
            start = block.start_time.strftime("%H:%M")
            end_time = block.start_time + timedelta(minutes=block.duration_minutes)
            end = end_time.strftime("%H:%M")

            # Gap detection per D-12: show gaps > 15 minutes
            if prev_end is not None:
                gap_minutes = (block.start_time - prev_end).total_seconds() / 60
                if gap_minutes > 15:
                    t.add_row("", f"[dim]--- Gap: {int(gap_minutes)}m ---[/]", "", "", "")

            t.add_row(
                str(i),
                f"{start}-{end}",
                f"[dim]{block.duration_minutes}m[/]",
                escape(task_title),
                f"[value.med]{escape(task_state)}[/]" if task_state else "",
            )
            prev_end = end_time

        console.print(t)

    if unscheduled:
        if scheduled:
            console.print("")
        console.rule("[header]Unscheduled Commitments[/]")
        t2 = Table(show_header=True, header_style="table.header",
                   show_edge=False, show_lines=False, pad_edge=False, box=None)
        t2.add_column("#", justify="right", no_wrap=True)
        t2.add_column("DUR", justify="right", no_wrap=True)
        t2.add_column("TASK")
        t2.add_column("STATE", justify="center", no_wrap=True)

        offset = len(scheduled)
        for i, block in enumerate(unscheduled, 1):
            task = repo.get_task(block.task_id)
            task_title = task.title if task else "Unknown"
            task_state = task.state.value if task else ""
            t2.add_row(
                str(offset + i),
                f"[dim]{block.duration_minutes}m[/]",
                escape(task_title),
                f"[value.med]{escape(task_state)}[/]" if task_state else "",
            )
        console.print(t2)


@block_app.command("rm")
def rm_block(
    block_number: Optional[int] = typer.Argument(None, help="Block number from list"),
):
    """
    Remove a scheduled time block (per D-05, D-09).

    Usage:
      pb plan block rm      # Interactive: shows numbered list
      pb plan block rm 2    # Direct: removes block #2 from today's list
    """
    repo = Repository()
    blocks = repo.list_time_blocks_for_date(datetime.utcnow())

    if not blocks:
        console = get_console()
        console.print("[dim]No blocks scheduled for today.[/]")
        return

    if block_number is None:
        # Interactive selection
        block = select_from_numbered_list(
            blocks,
            lambda b, i: format_block_for_selection(b, repo, i),
            prompt="Select block to remove",
            header="Today's blocks:",
        )
    else:
        # Direct by number
        if block_number < 1 or block_number > len(blocks):
            err_console = get_err_console()
            err_console.print(f"[error]Invalid block number: {block_number} (valid: 1-{len(blocks)})[/]")
            raise typer.Exit(code=1)
        block = blocks[block_number - 1]

    deleted = repo.delete_time_block(block.id)
    if deleted:
        task = repo.get_task(block.task_id)
        task_title = task.title if task else "Unknown"
        console = get_console()
        console.print(f"[success]Removed: {escape(task_title)} ({block.duration_minutes}m)[/]")
    else:
        err_console = get_err_console()
        err_console.print("[error]Failed to remove block[/]")
        raise typer.Exit(code=1)


@block_app.command("edit")
def edit_block(
    block_number: Optional[int] = typer.Argument(None, help="Block number from list"),
    start: Optional[str] = typer.Option(None, "--start", help="New start time (HH:MM, or 'none' to clear)"),
    duration: Optional[int] = typer.Option(None, "--duration", help="New duration in minutes"),
):
    """
    Edit a scheduled time block (per D-06, D-07, D-09).

    Usage:
      pb plan block edit            # Interactive selection, then prompts
      pb plan block edit 2          # Edit block #2, then prompts
      pb plan block edit 2 --start 10:00 --duration 90
    """
    repo = Repository()
    blocks = repo.list_time_blocks_for_date(datetime.utcnow())

    if not blocks:
        console = get_console()
        console.print("[dim]No blocks scheduled for today.[/]")
        return

    if block_number is None:
        block = select_from_numbered_list(
            blocks,
            lambda b, i: format_block_for_selection(b, repo, i),
            prompt="Select block to edit",
            header="Today's blocks:",
        )
    else:
        if block_number < 1 or block_number > len(blocks):
            err_console = get_err_console()
            err_console.print(f"[error]Invalid block number: {block_number} (valid: 1-{len(blocks)})[/]")
            raise typer.Exit(code=1)
        block = blocks[block_number - 1]

    console = get_console()

    # If no options provided, prompt interactively
    if start is None and duration is None:
        # Show current values and prompt
        current_start = block.start_time.strftime("%H:%M") if block.start_time else "none"
        console.print(f"[dim]Current:[/] {current_start}, {block.duration_minutes}m")

        start = typer.prompt("New start time (HH:MM, 'none', or Enter to keep)", default="", show_default=False)
        dur_input = typer.prompt(
            f"New duration (or Enter to keep {block.duration_minutes})",
            default="",
            show_default=False,
        )

        if dur_input.strip():
            try:
                duration = int(dur_input)
            except ValueError:
                err_console = get_err_console()
                err_console.print("[error]Duration must be a number[/]")
                raise typer.Exit(code=1)

    # Apply changes
    if start is not None and start.strip():
        if start.lower() == "none":
            block.start_time = None
        else:
            try:
                block.start_time = parse_time(start)
            except ValueError:
                err_console = get_err_console()
                err_console.print("[error]Invalid time format. Use HH:MM (e.g., 09:00 or 9:00)[/]")
                raise typer.Exit(code=1)

    if duration is not None:
        if duration <= 0:
            err_console = get_err_console()
            err_console.print("[error]Duration must be positive[/]")
            raise typer.Exit(code=1)
        block.duration_minutes = duration

    planner = Planner(repo)
    updated, overlap = planner.update_block(block)

    if overlap and overlap.start_time:
        overlap_start = overlap.start_time.strftime("%H:%M")
        overlap_end = (overlap.start_time + timedelta(minutes=overlap.duration_minutes)).strftime("%H:%M")
        console.print(f"[warn]Note: overlaps with existing block {overlap_start}-{overlap_end}[/]")

    if updated.start_time:
        time_str = updated.start_time.strftime("%H:%M")
    else:
        time_str = "unscheduled"

    console.print(f"[success]Updated: {time_str} for {updated.duration_minutes}m[/]")
