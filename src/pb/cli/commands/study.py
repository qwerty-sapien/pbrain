# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Study commands for conceptual and internalisation work."""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import structlog
import typer
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from pb.cli.active_session import resolve_active_session_preflight
from pb.cli.context_args import parse_context_argv
from pb.cli.context_runtime import (
    attach_active_context,
    context_prompt_contract,
    prepare_context_scope,
    raise_for_blocking_context,
)
from pb.cli.commands.clarify import maybe_start_clarification_plan
from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console
from pb.cli.helpers import parse_duration, prompt_text
from pb.cli.learning_flow import (
    build_learning_session_markdown,
    choose_learning_block_action,
    fetch_grounded_learning_resources,
    resource_preview_sections,
)
from pb.cli.llm_guard import print_llm_error, runtime_for_ctx
from pb.cli.markdown import render_markdown
from pb.cli.pickers import pick_single_choice
from pb.cli.preview import build_step_table, confirm_preview, markdown_step_lines, render_markdown_preview
from pb.cli.normalize import join_words_safe
from pb.cli.topic_group import TopicFallbackGroup
from pb.core.learning_tasks import materialize_learning_task
from pb.core.models import utc_now
from pb.core.renderables import renderable_cli_text, renderable_markdown_text
from pb.core.learning_tasks import infer_learning_duration_minutes
from pb.core.enums import BloomStage, EnergyType, TaskState
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.core.learning_block_flow import collect_revision_feedback, learner_profile_suffix
from pb.core.clarifier import (
    ClarifierService,
    ask_clarifier_questions,
    build_clarifier_context,
    clarifier_prompt_block,
    learning_intent_style_guidance,
    persist_clarifier_answers,
)
from pb.core.product_control import ProductControlEngine
from pb.core.scope_resolution import (
    list_knowledge_domains as resolved_knowledge_domains,
    match_domain_name as resolved_domain_name,
    match_goal as resolved_goal,
    match_track as resolved_track,
)
from pb.core.learning_metadata import build_learning_task_description, parse_learning_task_metadata
from pb.core.learning_partner import LearningPartnerSession
from pb.core.naming import (
    NameService,
    apply_generated_names,
    apply_generated_title,
)
from pb.core.session_blueprints import pack_display_label, resolve_learning_session_blueprint
from pb.core.staging import build_assumptions, build_learning_context, build_reflection
from pb.domain.models import Task
from pb.llm.drafts import RecallPromptDraft, StudyPlanDraft, artifact_presentation_prompt
from pb.llm.runtime import DraftGenerationError
from pb.study_service import StudyBlock

logger = structlog.get_logger()


def _scope_bullets_with_freshness(scope: str, vault_path: str | None) -> list[str]:
    """Return markdown bullets for scope concepts, bolding items absent from the vault."""
    items = [item.strip() for item in re.split(r"[,;]+", scope) if item.strip()]
    if not items:
        return []
    known_titles: set[str] = set()
    if vault_path:
        try:
            for root, _, files in os.walk(vault_path):
                for fname in files:
                    if fname.endswith(".md"):
                        known_titles.add(fname[:-3].lower())
        except Exception:
            pass
    lines = []
    for item in items:
        item_lower = item.lower()
        in_vault = any(item_lower in title or title in item_lower for title in known_titles)
        lines.append(f"- {item}" if in_vault else f"- **{item}**")
    return lines


def _success_lines(success: str) -> list[str]:
    """Return numbered list lines when multiple goals, single line otherwise."""
    if not success:
        return []
    parts = [s.strip().rstrip(".") for s in re.split(r"(?<=[.!?])\s+|;\s*", success) if s.strip()]
    if len(parts) <= 1:
        return [success]
    return [f"{i + 1}. {part}" for i, part in enumerate(parts)]


app = typer.Typer(
    cls=TopicFallbackGroup,
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


@dataclass
class MaterializedStudyRow:
    block: StudyBlock
    task: Task
    command: str
    task_created: bool
    block_created: bool


@dataclass(frozen=True)
class PlannedStudyBlockRow:
    display_code: str
    task: Task
    block: object


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _resolve_study_block_blueprint(*, block, runtime_ctx, domain_hint: str, topic_text: str) -> None:
    """Resolve a durable session blueprint and optionally swap to a nearby pack."""

    resolution = resolve_learning_session_blueprint(
        branch="study",
        domain=domain_hint,
        topic=block.subject_scope or topic_text,
        drill=block.study_mode or "",
        domain_pack_id=block.domain_pack_id,
        vault_path=runtime_ctx.vault_path,
        allow_custom_init=True,
    )
    block.domain_pack_id = resolution.pack_id
    block.session_blueprint = resolution.blueprint

    if not (_is_interactive() and resolution.source == "custom" and resolution.suggested_pack_ids):
        return

    options = [(resolution.pack_id, f"Use new custom blueprint · {pack_display_label(resolution.pack_id)}")]
    options.extend((pack_id, f"Use nearby blueprint · {pack_display_label(pack_id)}") for pack_id in resolution.suggested_pack_ids)
    choice = pick_single_choice(
        options,
        title="Session blueprint",
        text="No precise session pack matched this study block. Keep the new custom blueprint or reuse a nearby one.",
    )
    if not choice or choice == resolution.pack_id:
        return

    selected = resolve_learning_session_blueprint(
        branch="study",
        domain=domain_hint,
        topic=block.subject_scope or topic_text,
        drill=block.study_mode or "",
        domain_pack_id=choice,
        vault_path=runtime_ctx.vault_path,
        allow_custom_init=False,
    )
    block.domain_pack_id = selected.pack_id
    block.session_blueprint = selected.blueprint


def _coerce_bloom_stage(raw: str | None) -> Optional[BloomStage]:
    lowered = (raw or "").strip().lower()
    if not lowered:
        return None
    aliases = {
        "remember": BloomStage.REMEMBER,
        "recall": BloomStage.REMEMBER,
        "understand": BloomStage.UNDERSTAND,
        "understanding": BloomStage.UNDERSTAND,
        "apply": BloomStage.APPLY,
        "application": BloomStage.APPLY,
        "analyze": BloomStage.ANALYZE,
        "analyse": BloomStage.ANALYZE,
        "analysis": BloomStage.ANALYZE,
        "evaluate": BloomStage.EVALUATE,
        "evaluation": BloomStage.EVALUATE,
        "create": BloomStage.CREATE,
        "creation": BloomStage.CREATE,
    }
    if lowered in aliases:
        return aliases[lowered]
    for token, stage in aliases.items():
        if token in lowered:
            return stage
    return None


def resolve_stage_override(
    *,
    stage: Optional[str],
    legacy_level: Optional[str],
    apply_stage: bool,
    understand_stage: bool,
    evaluate_stage: bool,
    create_stage: bool,
) -> Optional[BloomStage]:
    """Resolve explicit stage overrides while keeping model inference as the default."""
    flag_count = sum(bool(value) for value in (apply_stage, understand_stage, evaluate_stage, create_stage))
    text_count = int(bool((stage or "").strip())) + int(bool((legacy_level or "").strip()))
    if flag_count + text_count > 1:
        raise typer.BadParameter("Choose only one stage override.")
    if apply_stage:
        return BloomStage.APPLY
    if understand_stage:
        return BloomStage.UNDERSTAND
    if evaluate_stage:
        return BloomStage.EVALUATE
    if create_stage:
        return BloomStage.CREATE
    resolved = _coerce_bloom_stage(stage or legacy_level)
    if (stage or legacy_level) and resolved is None:
        raise typer.BadParameter(
            "Stage must resolve to remember, understand, apply, analyze, evaluate, or create."
        )
    return resolved


def _match_goal(repo, subject: str):
    return resolved_goal(repo, subject, allowed_modes={"mixed", "study"})


def _match_track(repo, subject: str):
    return resolved_track(repo, subject)


def _list_knowledge_domains() -> list[str]:
    return resolved_knowledge_domains()


def _match_domain_name(subject: str) -> str:
    return resolved_domain_name(subject)


def _normalize_domain(raw: str) -> str:
    """Normalize a domain name for dedup: lowercase, strip separators."""
    return raw.strip().lower().replace("_", " ").replace("-", " ")


def _resolve_broad_category(candidate: str, categories: list[str]) -> str:
    """Map a free-form domain/scope hint onto one known broad category."""

    if not candidate.strip():
        return ""
    matched = resolved_domain_name(candidate, domains=categories)
    if matched:
        return matched
    normalized_candidate = _normalize_domain(candidate)
    for category in categories:
        normalized_category = _normalize_domain(category)
        if normalized_candidate == normalized_category:
            return category
    return ""


def _collect_broad_categories(repo) -> dict[str, list[str]]:
    """Build broad_category -> [specific_topic, ...] using explicit linkage first."""

    from collections import defaultdict

    categories: dict[str, set[str]] = defaultdict(set)
    goal_categories: dict[str, str] = {}
    track_categories: dict[str, str] = {}

    def add_specific(category: str, specific: str) -> None:
        norm_cat = _normalize_domain(category)
        norm_spec = _normalize_domain(specific)
        if norm_spec and norm_spec != norm_cat:
            categories[category].add(specific)

    seen_normalized: set[str] = set()

    def add_broad(name: str) -> None:
        if not name.strip():
            return
        norm = _normalize_domain(name)
        if norm in seen_normalized:
            return
        seen_normalized.add(norm)
        if name not in categories:
            categories[name] = set()

    for goal in repo.list_goal_arcs(status=None):
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        if mode in {"mixed", "study"}:
            broad = goal.domain or goal.title
            add_broad(broad)
            goal_categories[goal.id] = broad
            if goal.title.strip():
                add_specific(broad, goal.title)

    for track in repo.list_tracks(active_only=True):
        add_broad(track.name)
        track_categories[track.id] = track.name

    for domain in _list_knowledge_domains():
        add_broad(domain)

    for task in repo.list_tasks():
        if task.archived_at is not None:
            continue
        meta = parse_learning_task_metadata(task)
        scope = meta.scope or task.title
        broad_names = list(categories.keys())
        category = ""

        for goal_id in getattr(task, "linked_goal_arc_ids", []) or []:
            category = goal_categories.get(goal_id, "")
            if category:
                break
        if not category:
            for track_id in getattr(task, "linked_track_ids", []) or []:
                category = track_categories.get(track_id, "")
                if category:
                    break
        if not category and meta.domain:
            category = _resolve_broad_category(meta.domain, broad_names)
        if not category and scope:
            matched_scope_category = _resolve_broad_category(scope, broad_names)
            if matched_scope_category and _normalize_domain(matched_scope_category) != _normalize_domain(scope):
                category = matched_scope_category
        if not category and meta.domain.strip():
            category = meta.domain.strip()
            add_broad(category)

        if category:
            add_specific(category, scope)
            for session in repo.list_sessions_for_task(task.id):
                if getattr(session, "branch", "") == "study":
                    add_specific(category, getattr(session, "subject_scope", "") or scope)

    return {cat: sorted(specs) for cat, specs in categories.items() if cat.strip()}


def _pick_study_target(repo) -> Optional[str]:
    from pb.cli.pickers import pick_single_choice

    broad = _collect_broad_categories(repo)
    if not broad:
        if not _is_interactive():
            return None
        return typer.prompt("Study focus", default="", show_default=False).strip() or None

    # Layer 1: broad categories
    broad_choices = [(cat, cat) for cat in sorted(broad.keys())]
    selected_cat = pick_single_choice(broad_choices, title="Select study focus")
    if not selected_cat:
        return None

    specifics = broad.get(selected_cat, [])
    if not specifics:
        return selected_cat

    # Layer 2: specific topics within the category
    specific_choices = [(topic, topic) for topic in specifics]
    specific_choices.insert(0, (selected_cat, f"{selected_cat} (general)"))
    selected_topic = pick_single_choice(specific_choices, title=f"{selected_cat} — pick topic")
    return selected_topic or selected_cat


def _planned_study_rows(
    repo,
    *,
    include_completed: bool = False,
    include_archived: bool = False,
) -> list[PlannedStudyBlockRow]:
    """Return today's planned study blocks in one canonical display order."""
    from pb.core.models import utc_now

    blocks = repo.list_time_blocks_created_for_date(utc_now())
    if not blocks:
        return []

    counter = 1
    result: list[PlannedStudyBlockRow] = []
    for block in blocks:
        task = repo.get_task(block.task_id)
        display = getattr(block, "sub_index", None) or str(counter)
        counter += 1
        if task is None:
            continue
        if task.archived_at is not None and not include_archived:
            continue
        if task.completion >= 100 and not include_completed:
            continue
        result.append(
            PlannedStudyBlockRow(
                display_code=str(display),
                task=task,
                block=block,
            )
        )
    return result


def _todays_plan_blocks(repo) -> list[tuple[str, object, object]]:
    """Return today's unfinished plan rows as legacy tuples for callers."""
    return [(row.display_code, row.task, row.block) for row in _planned_study_rows(repo)]


def _resolve_planned_study_block(repo, index: Optional[str]) -> PlannedStudyBlockRow | None:
    rows = _planned_study_rows(repo)
    if not rows:
        return None
    if index is not None:
        return next((row for row in rows if row.display_code == index), None)

    active_session = repo.get_active_session()
    active_tid = active_session.task_id if active_session else None
    return next((row for row in rows if row.task.id != active_tid and row.task.completion < 100), None)


def _start_planned_study_block(ctx: typer.Context, index: Optional[str]) -> None:
    """Start the next or requested study block from today's planned rows."""
    from pb.cli.commands.execute import start_task_internal

    repo = ctx.obj["repo"]
    console = get_console()
    rows = _planned_study_rows(repo)
    if not rows:
        console.print("[dim]No plan for today. Run `pb plan day` first.[/]")
        raise typer.Exit(code=0)

    row = _resolve_planned_study_block(repo, index)
    if row is None:
        if index is None:
            console.print("[success]All today's sessions complete![/]")
            raise typer.Exit(code=0)
        console.print(f"[error]No block {index} in today's plan.[/]")
        raise typer.Exit(code=1)

    if not resolve_active_session_preflight(
        ctx, new_intent=row.task.title, new_branch="study",
    ):
        return

    start_task_internal(
        ctx,
        task_id=row.task.id,
        duration=f"{row.block.duration_minutes}m",
        suggest=False,
    )


def _mode_to_bloom(mode: str) -> str:
    return {
        "re-engage": "understand",
        "consolidate": "apply",
        "explore": "understand",
        "review": "remember",
    }.get(mode, "apply")


def _find_existing_study_task(repo, title: str, scope: str, mode: str):
    for task in repo.list_tasks():
        if task.archived_at is not None or task.completion >= 100:
            continue
        if task.title != title:
            continue
        meta = parse_learning_task_metadata(task)
        if meta.branch == "study" and meta.scope == scope and meta.study_mode == mode:
            return task
    return None


def _ensure_study_task(repo, block: StudyBlock) -> tuple[Task, bool]:
    title = f"Study: {block.domain} [{block.mode.capitalize()}]"
    existing = _find_existing_study_task(repo, title, block.domain, block.mode)
    if existing is not None:
        return existing, False

    matched_goal = _match_goal(repo, block.domain)
    matched_track = _match_track(repo, block.domain)
    description = build_learning_task_description(
        "Auto-created by pb study plan.",
        branch="study",
        scope=block.domain,
        domain=block.domain,
        bloom_target=_mode_to_bloom(block.mode),
        study_mode=block.mode,
    )
    task = Task(
        title=title,
        description=description,
        state=TaskState.ACTIVE,
        created_at=utc_now(),
        energy_type=EnergyType.DEEP,
        work_type="study",
        linked_goal_arc_ids=[matched_goal.id] if matched_goal else [],
        linked_track_ids=[matched_track.id] if matched_track else [],
    )
    repo.create_task(task)
    return task, True


def _ensure_time_block(repo, task: Task, minutes: int) -> bool:
    today_blocks = repo.list_time_blocks_created_for_date(utc_now())
    for block in today_blocks:
        if block.task_id == task.id and block.duration_minutes == minutes and (block.block_kind or "study") == "study":
            return False

    from pb.core.planner import Planner

    planner = Planner(repo)
    created_block, _ = planner.schedule_block(task, None, minutes)
    if getattr(created_block, "block_kind", "") != "study":
        created_block.block_kind = "study"
        repo.update_time_block(created_block)
    return True


def _render_domain_status(statuses: list) -> None:
    console = get_console()
    table = Table(
        title="Domain Status",
        show_header=True,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        box=None,
    )
    table.add_column("DOMAIN", style="cyan")
    table.add_column("#new", justify="right")
    table.add_column("#learning", justify="right")
    table.add_column("#learnt", justify="right")
    table.add_column("#stale", justify="right")
    table.add_column("PRESSURE", justify="right")
    for status in statuses:
        table.add_row(
            status.name,
            str(status.stage_new),
            str(status.stage_learning),
            str(status.stage_learnt),
            str(status.stage_stale),
            f"{status.decay_pressure:.0f}d" if status.is_stale else "-",
        )
    console.print(table)
    console.print()


def _render_materialized_study_plan(rows: list[MaterializedStudyRow], total_minutes: int, console) -> None:
    if not rows:
        console.print("[dim]No study blocks to schedule. Add knowledge domain notes first.[/]")
        return

    mode_styles = {
        "re-engage": "[bold red]re-engage[/]",
        "consolidate": "[bold yellow]consolidate[/]",
        "explore": "[bold blue]explore[/]",
        "review": "[bold magenta]review[/]",
    }
    table = Table(
        title=f"Study Plan ({total_minutes} min)",
        show_header=True,
        header_style="bold",
        show_edge=False,
        show_lines=True,
        pad_edge=False,
        box=None,
    )
    table.add_column("TIME", no_wrap=True, style="dim")
    table.add_column("MIN", justify="right", no_wrap=True)
    table.add_column("DOMAIN", style="cyan")
    table.add_column("MODE", no_wrap=True)
    table.add_column("TASK")
    table.add_column("COMMAND", style="green")
    table.add_column("STATUS", no_wrap=True)

    elapsed = 0
    for row in rows:
        block = row.block
        time_label = f"{elapsed}-{elapsed + block.minutes}"
        status_bits = []
        if row.task_created:
            status_bits.append("task")
        if row.block_created:
            status_bits.append("block")
        status_label = "new" if status_bits else "reused"
        table.add_row(
            time_label,
            str(block.minutes),
            escape(block.domain),
            mode_styles.get(block.mode, block.mode),
            escape(row.task.title),
            escape(row.command),
            status_label,
        )
        elapsed += block.minutes

    console.print()
    console.print(table)
    console.print()
    console.print(
        Panel(
            "[dim]These study tasks were added to today's queue. Use the shown learner-facing command "
            "to start the next conceptual block. Only active engagement is scheduled.[/]",
            title="[bold]How to use[/]",
            expand=False,
        )
    )


def _seed_study_block(
    *,
    topic: str,
    duration: Optional[int],
    level: Optional[BloomStage],
) -> StudyPlanDraft:
    scope = topic.strip() or "study topic"
    return StudyPlanDraft(
        summary="Deterministic study block.",
        blocks=[
            {
                "branch": "study",
                "subject_scope": scope,
                "duration_minutes": duration or infer_learning_duration_minutes("study", scope),
                "target_bloom_stage": (level or BloomStage.APPLY).value,
                "study_mode": "active recall",
                "success_check": f"Explain {scope} from memory and work one concrete example.",
                "reason": f"Keep progress moving on {scope} even without a live model.",
            }
        ],
    )


def _build_study_prompt(
    *,
    topic_text: str,
    domain_hint: str,
    matched_goal,
    requested_minutes: Optional[int],
    stage_hint: Optional[BloomStage],
    steps: bool,
    vault_path: Path,
    revision_note: str = "",
    prior_block: Optional[dict[str, object]] = None,
    clarifier_bundle=None,
    context_contract: str = "",
) -> str:
    stage_instruction = (
        f"Stage override: {stage_hint.value}. Use that exact `target_bloom_stage`.\n"
        if stage_hint is not None
        else (
            "Infer `target_bloom_stage` from the user's existing context, prior sessions, goal state, and adjacent knowledge. "
            "If they seem fluent in nearby material, step up the conceptual demand even for a new adjacent topic.\n"
        )
    )
    duration_instruction = (
        f"Requested duration minutes: {requested_minutes}. Use that exact `duration_minutes`.\n"
        if requested_minutes is not None
        else (
            "Choose an appropriate `duration_minutes` for a single high-value study block. "
            "Use the topic complexity, likely familiarity, and the best active-recall span to decide the timebox.\n"
        )
    )
    prompt = (
        "Create a single conceptual study block for the learning system.\n"
        "Return exactly one block.\n"
        + learning_intent_style_guidance()
        + f"Topic: {topic_text}\n"
        + f"Domain hint: {domain_hint}\n"
        + f"Goal title: {matched_goal.title if matched_goal else ''}\n"
        + f"{stage_instruction}"
        + f"{duration_instruction}"
        + "The block should be concrete, active, and retrieval-oriented.\n"
        + "`subject_scope` must name the exact competency slice for this block, not a vague paraphrase of the topic.\n"
        + "For ambitious topics, separate prerequisite progress from target progress.\n"
        + "If the learner's prerequisite readiness is unproven, lower the scope to the earliest useful missing layer and name the exact concepts or capabilities to cover.\n"
        + "Good scope example: 'Concrete fluency with vectors, matrices, systems of equations, span, basis, dimension, and linear maps for geometry in R^2 and R^3.'\n"
        + "Bad scope example: 'Linear algebra in a geometry context.'\n"
    )
    if prior_block is not None:
        prompt += (
            "Use the existing draft as the starting point and only change what was requested.\n"
            f"Existing draft JSON: {json.dumps(prior_block, ensure_ascii=True)}\n"
        )
    if revision_note.strip():
        prompt += f"User revision request: {revision_note.strip()}\n"
    if context_contract.strip():
        prompt += context_contract
    if steps:
        prompt += (
            "Include 4-8 ordered steps in `steps`.\n"
            "Each step must include `title`, `instruction`, and `success_check`.\n"
            "Use the steps to sequence concepts, formulae, and checks in the most effective study order.\n"
            "If any step instruction or check contains LaTeX that should be treated as math, "
            "return it as an object with `text` and `is_latex: true`.\n"
        )
    else:
        prompt += "Leave `steps` as an empty list unless stepwise guidance is explicitly requested.\n"
    prompt += clarifier_prompt_block(clarifier_bundle)
    prompt += artifact_presentation_prompt()
    prompt += feedback_prompt_suffix(vault_path, "study")
    return prompt


def _run_pre_gen_diagnostic(
    *,
    concept_id: str,
    topic: str,
    domain: str,
    runtime,
    runtime_ctx,
    console,
) -> str:
    """Run 3 MCQ probe questions to determine difficulty tier (D-16-19/D-16-20).

    Returns: "easier" | "current" | "harder"
    """
    # If LLM unavailable, fall back gracefully — do not block the session.
    if not runtime.health().available:
        return "current"

    try:
        import json
        import re as _re
        from pb.llm.drafts import LessonPlanDraft
        from pb.llm.runtime import DraftGenerationError

        probe_prompt = (
            f"Generate exactly 3 multiple-choice diagnostic questions about '{topic}' "
            f"in the domain '{domain}'. Each question should test conceptual understanding "
            f"at default difficulty. Format as JSON: "
            '{{"questions": [{{"question": "...", "options": ["A...", "B...", "C...", "D..."], "correct": "A..."}}]}}'
        )
        # Use generate_draft with a raw probe — get text back and parse JSON ourselves.
        # We abuse a simple Pydantic-compatible format: call client directly via runtime._client_for_provider.
        provider_name, selected_model = runtime._resolve_provider_and_model(None)
        client = runtime._client_for_provider(provider_name)
        result = client.generate_with_model(
            probe_prompt,
            selected_model,
            timeout=30,
            max_output_tokens=4000,
        )
        raw_text = (result.text or "").strip() if result.error is None else ""
        if not raw_text:
            return "current"
        match = _re.search(r'\{.*\}', raw_text, _re.DOTALL)
        if not match:
            return "current"
        data = json.loads(match.group(0))
        questions = data.get("questions", [])[:3]
    except Exception:
        return "current"

    if not questions:
        return "current"

    console.print(f"[dim]Diagnostic: 3 questions to calibrate difficulty for '{topic}'[/]")
    correct_count = 0
    for q in questions:
        question_text = q.get("question", "")
        options = q.get("options", [])
        correct_answer = q.get("correct", "")
        if not question_text or not options:
            continue
        console.print(f"\n[bold]{question_text}[/bold]")
        try:
            chosen = pick_single_choice(
                [(opt, opt) for opt in options],
                title="Choose one",
                text="Use arrows or digit keys.",
            )
        except Exception:
            continue
        if chosen and str(chosen).strip() == str(correct_answer).strip():
            correct_count += 1
            console.print("[green]Correct[/green]")
        else:
            console.print(f"[dim]Answer: {correct_answer}[/dim]")

    if correct_count >= 3:
        return "harder"
    elif correct_count >= 2:
        return "current"
    else:
        return "easier"


def launch_study_session(
    ctx: typer.Context,
    *,
    topic: Optional[str] = None,
    duration: Optional[str] = None,
    stage_hint: Optional[BloomStage] = None,
    yes: bool = False,
    steps: bool = False,
) -> None:
    """Create a study task, start it, and annotate the session."""
    from pb.cli.commands.execute import start_task_internal

    repo = ctx.obj["repo"]
    console = get_console()
    auto_yes = bool(yes or ((ctx.obj or {}).get("yes")))
    topic_text = (topic or "").strip()
    if not topic_text:
        topic_text = _pick_study_target(repo) or ""
    if not topic_text:
        raise typer.BadParameter("A study topic is required.")
    if not resolve_active_session_preflight(
        ctx,
        new_intent=topic_text,
        new_branch="study",
    ):
        return
    if maybe_start_clarification_plan(
        ctx,
        topic=topic_text,
        preferred_branch="study",
        yes=auto_yes,
    ):
        return

    runtime = runtime_for_ctx(ctx)
    control_engine = ProductControlEngine(repo=repo, runtime=runtime)
    prepared_context = ctx.obj.get("_prepared_context_scope")
    active_context_scope = getattr(prepared_context, "scope", None)
    matched_goal = _match_goal(repo, topic_text)
    matched_track = _match_track(repo, topic_text)
    domain_hint = (
        _match_domain_name(topic_text)
        or getattr(matched_goal, "domain", "")
        or getattr(matched_track, "name", "")
        or topic_text
    )
    requested_minutes = parse_duration(duration) if duration else None
    runtime_ctx = ctx.obj["runtime"]
    _, control_state = control_engine.load_state(
        scope="artifact",
        artifact_kind="study_block",
        artifact_id=topic_text,
        goal_id=getattr(matched_goal, "id", "") or "",
    )
    clarifier_bundle = None
    clarifier_answers: dict[str, str] = {}
    if sys.stdin.isatty() and not auto_yes:
        clarifier_context = build_clarifier_context(
            repo,
            runtime_ctx,
            raw_request=topic_text,
            scope="study",
            mode="study",
            domain=domain_hint,
            control_state=control_state,
        )
        questions = ClarifierService(runtime).generate_questions(
            topic_text,
            clarifier_context,
            max_questions=2,
            scope="study",
            control_state=control_state,
        )
        clarifier_bundle = ask_clarifier_questions(questions) if questions else None
        clarifier_answers = clarifier_bundle.answers if clarifier_bundle is not None else {}
    prompt = _build_study_prompt(
        topic_text=topic_text,
        domain_hint=domain_hint,
        matched_goal=matched_goal,
        requested_minutes=requested_minutes,
        stage_hint=stage_hint,
        steps=steps,
        vault_path=runtime_ctx.vault_path,
        clarifier_bundle=clarifier_bundle,
        context_contract=context_prompt_contract(active_context_scope),
    )
    prompt += learner_profile_suffix(repo, runtime_ctx)
    recorder = runtime.make_stage_recorder("study", topic_text, route_hint="study")
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("study", topic_text, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("study", topic_text, context))
    recorder.add("clarify", clarifier_answers)
    if sys.stdin.isatty() and bool(ctx.obj.get("verbose")):
        console.print(f"[dim]{reflection}[/]")

    draft_result = None
    try:
        draft_result = runtime.generate_draft(
            StudyPlanDraft,
            prompt,
            source_scope=f"study:{topic_text}",
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
        print_llm_error(exc)
        draft = _seed_study_block(topic=topic_text, duration=requested_minutes, level=stage_hint)

    if not draft.blocks:
        recorder.finalize("empty")
        raise typer.BadParameter("No study block was generated.")
    block = draft.blocks[0]
    block.branch = "study"
    block.subject_scope = block.subject_scope or topic_text
    block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
    block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
        "study",
        block.subject_scope or topic_text,
        study_mode=block.study_mode or "",
    )
    block.target_bloom_stage = block.target_bloom_stage or stage_hint or BloomStage.APPLY
    block.study_mode = block.study_mode or "manual"
    _resolve_study_block_blueprint(
        block=block,
        runtime_ctx=runtime_ctx,
        domain_hint=domain_hint,
        topic_text=topic_text,
    )
    recorder.add("verify", {"preview": block.model_dump(mode="json")})
    resources = None

    def _cancel_preview() -> None:
        if draft_result is not None:
            repo.create_generation_provenance(
                runtime.build_provenance(
                    artifact_kind="study_block",
                    artifact_id=topic_text,
                    generated_draft=draft_result,
                    accepted_by_user=False,
                )
            )
        recorder.finalize("cancelled", artifact_kind="study_block", artifact_id=topic_text)
        raise typer.Exit(code=0)

    while True:
        preview_sections: list[tuple[str, list[str] | object]] = []
        scope_bullets = _scope_bullets_with_freshness(
            block.subject_scope or topic_text,
            getattr(runtime_ctx, "vault_path", None),
        )
        success_bullets = _success_lines(block.success_check or "")
        if block.steps:
            preview_sections.append(("Steps", build_step_table(block.steps, presentation=draft.presentation)))
        preview_sections.extend(resource_preview_sections(resources))
        render_markdown_preview(
            title="Study Block Draft",
            rows=[
                ("Title", block.title),
                ("Planned time", f"{block.duration_minutes} min"),
            ],
            sections=[
                ("Scope", scope_bullets),
                ("Success", success_bullets),
                *preview_sections,
            ],
        )
        if auto_yes or not sys.stdin.isatty():
            accepted = confirm_preview(yes=auto_yes, action_label="Start this study block")
            if not accepted:
                _cancel_preview()
            break

        action = choose_learning_block_action("Start this study block")
        if action in {None, "cancel"}:
            _cancel_preview()
        if action == "start":
            break
        if action == "resources":
            resources = fetch_grounded_learning_resources(
                topic=topic_text,
                branch="study",
                block_payload=block.model_dump(mode="json"),
            )
            recorder.add(
                "resources",
                {
                    "warning": resources.warning,
                    "bundle": resources.bundle.model_dump(mode="json") if resources.bundle is not None else None,
                    "qc_notes": resources.qc_notes,
                },
                status="warn" if resources.warning else "ok",
            )
            if resources.warning:
                console.print(f"[warn]{resources.warning}[/]")
            continue

        revision_feedback = collect_revision_feedback(
            engine=control_engine,
            repo=repo,
            runtime_ctx=runtime_ctx,
            mode="study",
            artifact_kind="study_block",
            artifact_id=topic_text,
            current_artifact=json.dumps(block.model_dump(mode="json"), ensure_ascii=True),
            domain=domain_hint,
            target=block.subject_scope or topic_text,
            goal_id=getattr(matched_goal, "id", "") or "",
            title="Revise study block",
        )
        if revision_feedback is None:
            continue

        revision_note = revision_feedback.free_text

        topic_text = block.subject_scope or topic_text
        requested_minutes = block.duration_minutes
        matched_goal = _match_goal(repo, topic_text)
        matched_track = _match_track(repo, topic_text)
        domain_hint = (
            _match_domain_name(topic_text)
            or getattr(matched_goal, "domain", "")
            or getattr(matched_track, "name", "")
            or topic_text
        )
        prompt = _build_study_prompt(
            topic_text=topic_text,
            domain_hint=domain_hint,
            matched_goal=matched_goal,
            requested_minutes=requested_minutes,
            stage_hint=stage_hint,
            steps=steps,
            vault_path=runtime_ctx.vault_path,
            revision_note=revision_note,
            prior_block=block.model_dump(mode="json"),
            clarifier_bundle=clarifier_bundle,
            context_contract=context_prompt_contract(active_context_scope),
        )
        prompt += revision_feedback.prompt_suffix
        try:
            draft_result = runtime.generate_draft(
                StudyPlanDraft,
                prompt,
                source_scope=f"study:{topic_text}",
            )
            draft = draft_result.payload
            recorder.add(
                "revise",
                {
                    "scope": topic_text,
                    "duration_minutes": requested_minutes,
                    "note": revision_note,
                    "model": draft_result.model,
                },
            )
        except DraftGenerationError as exc:
            recorder.add(
                "revise",
                {
                    "scope": topic_text,
                    "duration_minutes": requested_minutes,
                    "note": revision_note,
                    "error": exc.to_user_message(),
                },
                status="error",
            )
            recorder.finalize("error")
            print_llm_error(exc)
            raise typer.Exit(code=1)

        if not draft.blocks:
            recorder.finalize("empty")
            raise typer.BadParameter("No study block was generated.")
        block = draft.blocks[0]
        block.branch = "study"
        block.subject_scope = block.subject_scope or topic_text
        block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
        block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
            "study",
            block.subject_scope or topic_text,
            study_mode=block.study_mode or "",
        )
        block.target_bloom_stage = block.target_bloom_stage or stage_hint or BloomStage.APPLY
        block.study_mode = block.study_mode or "manual"
        _resolve_study_block_blueprint(
            block=block,
            runtime_ctx=runtime_ctx,
            domain_hint=domain_hint,
            topic_text=topic_text,
        )
        resources = None

    task_names = NameService(runtime).generate_names(
        "study_task",
        topic_text,
        {
            "domain": domain_hint,
            "subject": block.subject_scope or topic_text,
            "goal": matched_goal.title if matched_goal else "",
            "activity_type": "study",
            "study_mode": block.study_mode or "",
        },
    )
    task, _ = materialize_learning_task(repo, block)
    apply_generated_title(task, task_names, title_key="task_title")
    attach_active_context(task, active_context_scope)
    task.linked_goal_arc_ids = [matched_goal.id] if matched_goal else []
    task.linked_track_ids = [matched_track.id] if matched_track else []
    if clarifier_bundle is not None:
        persist_clarifier_answers(task, clarifier_bundle)
    repo.update_task(task)

    pre_session_markdown = build_learning_session_markdown(
        task_title=task.title,
        steps=block.steps,
        resources=resources,
    )
    start_task_internal(
        ctx,
        task_id=task.id,
        duration=f"{block.duration_minutes}m",
        suggest=False,
        pre_session_markdown=pre_session_markdown,
    )
    active_session = repo.get_active_session()
    if active_session is not None and active_session.task_id == task.id:
        active_session.branch = "study"
        active_session.goal_id = matched_goal.id if matched_goal else None
        active_session.track_id = matched_track.id if matched_track else None
        active_session.subject_scope = block.subject_scope or domain_hint
        active_session.target_bloom_stage = block.target_bloom_stage or stage_hint or BloomStage.APPLY
        active_session.bloom_stage = None
        apply_generated_names(active_session, task_names)
        attach_active_context(active_session, active_context_scope)
        if clarifier_bundle is not None:
            persist_clarifier_answers(active_session, clarifier_bundle)
        repo.update_session(active_session)

    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="study_task",
                artifact_id=task.id,
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add(
        "materialize",
        {
            "task_id": task.id,
            "subject_scope": block.subject_scope or topic_text,
            "goal_id": matched_goal.id if matched_goal else None,
        },
    )
    recorder.finalize("persisted", artifact_kind="study_task", artifact_id=task.id)

    # D-16-19/D-16-20: pre-gen diagnostic gate — fires ONLY for confidence strictly < 0.3
    from pb.core.graph_writer import make_slug
    from pb.core.confidence_model import THRESHOLD_NONE, DELTA_DIAGNOSTIC_CORRECT, THRESHOLD_FULL, clamp_score

    _study_concept_id = f"concept:{domain_hint.lower()}:{make_slug(block.subject_scope or topic_text)}"
    _study_records = repo.list_concept_confidence(_study_concept_id)
    _study_score = getattr(_study_records[0], "confidence_score", 0.0) if _study_records else 0.0
    _difficulty_tier = "current"
    if _study_score < THRESHOLD_NONE and not auto_yes and sys.stdin.isatty():   # strictly less than 0.3 (not <= 0.3)
        _difficulty_tier = _run_pre_gen_diagnostic(
            concept_id=_study_concept_id,
            topic=block.subject_scope or topic_text,
            domain=domain_hint,
            runtime=runtime,
            runtime_ctx=runtime_ctx,
            console=console,
        )

    goal_label = matched_goal.title if matched_goal else (f"track {matched_track.name}" if matched_track else "free study")
    console.print(
        f"[dim]Study route:[/] scope `{block.title or topic_text}` | domain `{domain_hint}` | goal `{goal_label}`"
    )
    if sys.stdin.isatty() and not auto_yes and os.environ.get("PB_IN_SHELL") != "1":
        while True:
            active_session = repo.get_active_session()
            if active_session is None or active_session.task_id != task.id:
                break
            try:
                partner = LearningPartnerSession(
                    runtime=runtime,
                    runtime_ctx=runtime_ctx,
                    repo=repo,
                    task=task,
                    session=active_session,
                    branch="study",
                    objective=block.success_check or block.reason or topic_text,
                    topic=block.subject_scope or topic_text,
                    domain=domain_hint,
                    clarifier_answers=clarifier_answers,
                    mode=block.study_mode or "study_partner",
                    verbose=bool(ctx.obj.get("verbose")),
                    confidence_level=_study_score,
                )
            except DraftGenerationError as exc:
                print_llm_error(exc)
                raise typer.Exit(code=1)
            result = partner.start()
            if result.note_path is not None:
                console.print(f"[dim]Partner note:[/] {result.note_path.relative_to(runtime_ctx.vault_path)}")
            if result.action == "command" and result.command:
                run_internal_command(ctx, result.command)
                active_session = repo.get_active_session()
                if active_session is not None and active_session.task_id == task.id:
                    continue
                return
            if result.action == "finish":
                # D-16-19: update confidence from study session outcome
                # Re-fetch current score at finish time (pre-session _study_records is stale)
                from datetime import datetime, timedelta
                _current_records = repo.list_concept_confidence(_study_concept_id)
                _current_score = _current_records[0].confidence_score if _current_records else 0.0
                _study_new = clamp_score(_current_score + DELTA_DIAGNOSTIC_CORRECT)
                _now_iso = datetime.utcnow().isoformat()
                repo.upsert_concept_confidence(
                    _study_concept_id,
                    confidence_score=_study_new,
                    last_evidence_at=_now_iso,
                )
                # D-16-26 (Phase 16 scope: maintenance scheduling only):
                # If score rises to full, schedule low-frequency maintenance review (14 days).
                if _study_new > THRESHOLD_FULL:
                    _maintenance_date = (datetime.utcnow() + timedelta(days=14)).isoformat()
                    repo.upsert_concept_confidence(
                        _study_concept_id,
                        confidence_score=_study_new,
                        next_review_at=_maintenance_date,
                    )
                from pb.cli.commands.execute import finish_task

                finish_task(ctx, note_words=[result.summary], completion=100, debrief=False, skip=False)
                return
            if result.action == "pause":
                paused = ctx.obj["factory"]["session_service"]().pause_session(outcome=result.summary)
                if paused is not None:
                    console.print(f"[success]Paused: {escape(task.title)}[/]")
                return


@app.command("day")
def study_day(
    ctx: typer.Context,
    index: Optional[str] = typer.Argument(None, help="Block code (1, 2, 2a) or omit for next"),
):
    """Start the next session from today's plan, or a specific block by code."""
    _start_planned_study_block(ctx, index)


@app.command("skip")
def study_skip(ctx: typer.Context):
    """Skip the next planned session and start the one after it."""
    from pb.cli.commands.execute import start_task_internal

    repo = ctx.obj["repo"]
    console = get_console()
    plan_items = _todays_plan_blocks(repo)

    if not plan_items:
        console.print("[dim]No plan for today. Run `pb plan day` first.[/]")
        raise typer.Exit(code=0)

    active_session = repo.get_active_session()
    active_tid = active_session.task_id if active_session else None
    unfinished = [
        (code, t, b) for code, t, b in plan_items
        if t.id != active_tid and t.completion < 100
    ]

    if len(unfinished) < 1:
        console.print("[success]All today's sessions complete![/]")
        raise typer.Exit(code=0)

    console.print(f"[dim]Skipped: {escape(unfinished[0][1].title)}[/]")

    if len(unfinished) < 2:
        console.print("[dim]No more sessions after the skipped one.[/]")
        raise typer.Exit(code=0)

    task, block = unfinished[1][1], unfinished[1][2]
    if not resolve_active_session_preflight(
        ctx, new_intent=task.title, new_branch="study",
    ):
        return

    start_task_internal(
        ctx,
        task_id=task.id,
        duration=f"{block.duration_minutes}m",
        suggest=False,
    )


@app.command("delete")
def study_delete(
    ctx: typer.Context,
    codes: Optional[list[str]] = typer.Argument(None, help="Block codes to delete (e.g. 2 3 5)"),
    all_blocks: bool = typer.Option(False, "--all", help="Delete all planned sessions"),
):
    """Remove sessions from today's plan by code, --all, or picker."""
    from pb.cli.helpers import pick_task
    from pb.core.models import utc_now

    repo = ctx.obj["repo"]
    console = get_console()
    plan_items = _todays_plan_blocks(repo)

    if not plan_items:
        console.print("[dim]No plan for today.[/]")
        raise typer.Exit(code=0)

    if all_blocks:
        target_ids = {t.id for _, t, _ in plan_items}
    elif codes:
        code_set = set(codes)
        target_ids = set()
        for code, task, _ in plan_items:
            if code in code_set:
                target_ids.add(task.id)
        if not target_ids:
            console.print(f"[error]No matching codes: {', '.join(codes)}[/]")
            raise typer.Exit(code=1)
    else:
        tasks = [t for _, t, _ in plan_items]
        selected = pick_task(tasks, prompt_text="Select sessions to remove", multi_select=True)
        if not selected:
            raise typer.Exit(code=0)
        target_ids = {t.id for t in selected}

    blocks = repo.list_time_blocks_created_for_date(utc_now())
    deleted_count = 0
    deleted_titles = []
    for block in blocks:
        if block.task_id in target_ids:
            repo.delete_time_block(block.id)
            deleted_count += 1
            task = repo.get_task(block.task_id)
            if task and task.title not in deleted_titles:
                deleted_titles.append(task.title)

    console.print(f"[success]Removed {deleted_count} block(s): {', '.join(escape(t) for t in deleted_titles)}[/]")


@app.callback(invoke_without_command=True)
def study_command(
    ctx: typer.Context,
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g. 30m, 45m, 1h)"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Optional stage hint; otherwise inferred from context"),
    apply_stage: bool = typer.Option(False, "--apply", "-a", help="Override toward apply/analyze"),
    understand_stage: bool = typer.Option(False, "--understand", "-u", help="Override toward understand"),
    evaluate_stage: bool = typer.Option(False, "--evaluate", "-e", help="Override toward evaluate"),
    create_stage: bool = typer.Option(False, "--create", "-c", help="Override toward create"),
    level: Optional[str] = typer.Option(None, "--level", "-l", hidden=True),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise study sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
    legacy_time: Optional[int] = typer.Option(None, "--time", help="Compatibility alias for `pb study plan --time`", hidden=True),
    legacy_domain: Optional[str] = typer.Option(None, "--domain", help="Compatibility alias for `pb study plan --domain`", hidden=True),
    legacy_verbose: bool = typer.Option(False, "--verbose", hidden=True),
):
    """Start a conceptual study block, or route legacy planner flags to `pb study plan`."""
    if ctx.invoked_subcommand is not None:
        return
    if legacy_time is not None or legacy_domain is not None or legacy_verbose:
        study_plan(ctx, time=legacy_time or 60, domain=legacy_domain, verbose=legacy_verbose)
        return
    import re as _re
    parsed_args = parse_context_argv(ctx.args)
    prepared_context = prepare_context_scope(
        ctx,
        [Path(token).expanduser() for token in parsed_args.context_tokens],
    )
    raise_for_blocking_context(prepared_context)
    raw_topic = join_words_safe(parsed_args.topic_tokens)
    if raw_topic:
        if raw_topic == "day":
            _start_planned_study_block(ctx, None)
            return
        if raw_topic == "skip":
            ctx.invoke(study_skip)
            return
        if raw_topic.startswith("delete"):
            parts = raw_topic.split()[1:]
            if "--all" in parts:
                ctx.invoke(study_delete, codes=None, all_blocks=True)
            elif parts:
                ctx.invoke(study_delete, codes=parts, all_blocks=False)
            else:
                ctx.invoke(study_delete, codes=None, all_blocks=False)
            return
        if _re.match(r"^\d+[a-z]?$", raw_topic):
            _start_planned_study_block(ctx, raw_topic)
            return

    stage_hint = resolve_stage_override(
        stage=stage,
        legacy_level=level,
        apply_stage=apply_stage,
        understand_stage=understand_stage,
        evaluate_stage=evaluate_stage,
        create_stage=create_stage,
    )
    launch_study_session(
        ctx,
        topic=raw_topic or None,
        duration=duration,
        stage_hint=stage_hint,
        yes=yes,
        steps=steps,
    )


@app.command("plan", hidden=True)
def study_plan(
    ctx: typer.Context,
    time: int = typer.Option(60, "--time", "-t", help="Total study minutes"),
    domain: Optional[str] = typer.Option(None, "--domain", help="Filter to one domain"),
    verbose: bool = typer.Option(False, "--verbose", help="Show domain status breakdown"),
):
    """Create actionable study tasks and time blocks from vault state."""
    from pb.vault.embeddings import EmbeddingUnavailableError

    study_svc = ctx.obj["factory"]["study_service"]()
    repo = ctx.obj["repo"]
    console = get_console()

    try:
        blocks = study_svc.generate_plan(total_minutes=time, domain_filter=domain)
    except EmbeddingUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if not blocks:
        typer.echo("No study blocks generated. Check vault has notes in knowledge/.")
        return

    if verbose:
        try:
            _render_domain_status(study_svc.get_domain_statuses())
        except Exception:
            pass

    rows: list[MaterializedStudyRow] = []
    for block in blocks:
        task, task_created = _ensure_study_task(repo, block)
        block_created = _ensure_time_block(repo, task, block.minutes)
        rows.append(
            MaterializedStudyRow(
                block=block,
                task=task,
                command=block.pb_command,
                task_created=task_created,
                block_created=block_created,
            )
        )

    _render_materialized_study_plan(rows, time, console)
    created_count = sum(1 for row in rows if row.task_created or row.block_created)
    console.print(f"[dim]{created_count} study planning actions created or refreshed.[/]")


@app.command("start", hidden=True)
def study_start(
    ctx: typer.Context,
    topic_words: Optional[list[str]] = typer.Argument(None, help="Study topic"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g., 30m, 45m, 1h)"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Optional stage hint; otherwise inferred from context"),
    apply_stage: bool = typer.Option(False, "--apply", "-a", help="Override toward apply/analyze"),
    understand_stage: bool = typer.Option(False, "--understand", "-u", help="Override toward understand"),
    evaluate_stage: bool = typer.Option(False, "--evaluate", "-e", help="Override toward evaluate"),
    create_stage: bool = typer.Option(False, "--create", "-c", help="Override toward create"),
    level: Optional[str] = typer.Option(None, "--level", "-l", hidden=True),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise study sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Compatibility alias for the top-level study flow."""
    stage_hint = resolve_stage_override(
        stage=stage,
        legacy_level=level,
        apply_stage=apply_stage,
        understand_stage=understand_stage,
        evaluate_stage=evaluate_stage,
        create_stage=create_stage,
    )
    launch_study_session(
        ctx,
        topic=" ".join(topic_words or []).strip() or None,
        duration=duration,
        stage_hint=stage_hint,
        yes=yes,
        steps=steps,
    )


@app.command("debrief", hidden=True)
def study_debrief(
    ctx: typer.Context,
    topic: Optional[list[str]] = typer.Argument(None, help="Topic to focus the debrief on (no quotes needed)"),
    domain: Optional[str] = typer.Option(None, "--domain", help="Target domain (defaults to cwd inference)"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Optional stage hint; otherwise inferred from context"),
    apply_stage: bool = typer.Option(False, "--apply", "-a", help="Override toward apply/analyze"),
    understand_stage: bool = typer.Option(False, "--understand", "-u", help="Override toward understand"),
    evaluate_stage: bool = typer.Option(False, "--evaluate", "-e", help="Override toward evaluate"),
    create_stage: bool = typer.Option(False, "--create", "-c", help="Override toward create"),
    level: Optional[str] = typer.Option(None, "--level", "-l", hidden=True),
    sync: bool = typer.Option(False, "--sync", help="Bypass Vertex Batch; create note synchronously"),
    flash: bool = typer.Option(False, "--flash", help="Use Flash model for note structuring (default Flash Lite)"),
):
    """Run a Socratic debrief for the current study context."""
    from pb.cli.commands.learn import _pick_domain
    from pb.core.graph_writer import make_slug
    from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL
    from pb.vault import get_vault_path

    console = get_console()
    if not _is_interactive():
        console.print("[error]pb study debrief requires an interactive terminal[/]")
        raise typer.Exit(code=1)

    try:
        vault_path = get_vault_path()
    except Exception:
        console.print("[error]Vault not configured. Run `pb init` first.[/]")
        raise typer.Exit(code=1)

    socratic_service = ctx.obj["factory"]["socratic_service"]()
    knowledge_dir = vault_path / "knowledge"
    if not domain:
        domain = socratic_service.detect_domain(knowledge_dir)
    if not domain:
        domain = _pick_domain(knowledge_dir, console)
    if not domain:
        console.print("[error]No domain detected or selected; aborting.[/]")
        raise typer.Exit(code=1)

    topic_str = " ".join(topic) if topic else ""

    import os

    stage_hint = resolve_stage_override(
        stage=stage,
        legacy_level=level,
        apply_stage=apply_stage,
        understand_stage=understand_stage,
        evaluate_stage=evaluate_stage,
        create_stage=create_stage,
    )
    os.environ["PB_BLOOMS_LEVEL"] = (stage_hint or BloomStage.APPLY).value

    qa_pairs = socratic_service.run_study_debrief(domain=domain, console=console, topic=topic_str)
    if not qa_pairs:
        console.print("[warn]No Q&A captured; nothing to save.[/]")
        return

    all_answers = " ".join(answer for _, answer in qa_pairs)
    slug = make_slug(all_answers[:60]) or "study-note"
    model = FLASH_MODEL if flash else FLASH_LITE_MODEL
    result = socratic_service.build_and_submit(
        qa_pairs=qa_pairs,
        domain=domain,
        slug=slug,
        template="deep",
        sync=sync,
        model=model,
        console=console,
    )
    if result:
        console.print(f"[success]{'Saved' if sync else 'Submitted'}: {result}[/]")


@app.command("resume")
def study_resume(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to resume"),
):
    """Resume a paused study task."""
    from pb.cli.commands.execute import resume_task

    resume_task(ctx, task_id=task_id)


@app.command("recall")
def study_recall(
    ctx: typer.Context,
    scope_words: Optional[list[str]] = typer.Argument(None, help="Scoped domain or concept"),
    limit: int = typer.Option(8, "--limit", help="Maximum notes to preview"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and persist recall prompts"),
):
    """Generate a scoped recall draft from the current study context."""
    from pb.vault.config import get_vault_path

    repo = ctx.obj["repo"]
    scope = " ".join(scope_words or []).strip()
    if not scope:
        active = repo.get_active_session()
        scope = getattr(active, "subject_scope", "") if active is not None else ""
    if not scope:
        goals = repo.list_goal_arcs()
        if goals:
            scope = goals[0].domain or goals[0].title
    if not scope:
        raise typer.BadParameter("A recall scope is required. Try `pb study recall math`.")

    vault = get_vault_path()
    knowledge_dir = vault / "knowledge"
    scope_lower = scope.lower()
    matches: list[Path] = []
    for md_file in knowledge_dir.rglob("*.md"):
        rel = md_file.relative_to(knowledge_dir)
        haystack = f"{rel} {md_file.stem}".lower()
        if scope_lower in haystack:
            matches.append(md_file)
    if not matches:
        typer.echo(f"No scoped recall notes found for: {scope}")
        return

    runtime = runtime_for_ctx(ctx)
    note_context = []
    for note_path in matches[:limit]:
        rel = note_path.relative_to(vault)
        try:
            note_context.append(f"## {rel}\n{note_path.read_text(encoding='utf-8', errors='ignore')[:1800]}")
        except OSError:
            continue
    prompt = (
        "Create scoped active-recall prompts for the study loop.\n"
        f"Scope: {scope}\n"
        "If a prompt or answer contains mathematical TeX/LaTeX that should be treated as math, "
        "return it as {text: ..., is_latex: true}. Plain strings are always plain text.\n"
        "Use only the provided notes.\n\n"
        f"{chr(10).join(note_context)}"
    )
    draft_result = runtime.generate_draft(
        RecallPromptDraft,
        prompt,
        source_scope=f"recall:{scope}",
    )
    _recall = draft_result.payload
    render_markdown_preview(
        title="Recall Draft",
        rows=[
            ("Scope", _recall.scope),
            ("Summary", _recall.summary),
            ("Prompts", str(len(_recall.prompts))),
        ],
        sections=[
            (
                f"Prompt {i + 1}",
                [
                    f"- **Prompt:** {renderable_cli_text(item.prompt)}",
                    f"- **Answer:** {renderable_cli_text(item.answer)}",
                    f"- **Difficulty:** {item.difficulty}" if item.difficulty else "",
                    f"- **Source:** `{item.source_note}`" if item.source_note else "",
                ],
            )
            for i, item in enumerate(_recall.prompts)
        ],
    )
    accepted = confirm_preview(yes=yes, action_label="Persist these recall prompts")
    if not accepted:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="recall_draft",
                artifact_id=scope,
                generated_draft=draft_result,
                accepted_by_user=False,
            )
        )
        raise typer.Exit(code=0)

    slug = scope.lower().replace("/", "-").replace(" ", "-")
    recall_dir = vault / "30-recall"
    recall_dir.mkdir(parents=True, exist_ok=True)
    recall_path = recall_dir / f"{slug}.md"
    lines = [f"# Recall: {draft_result.payload.scope}", ""]
    if draft_result.payload.summary:
        lines.extend([draft_result.payload.summary, ""])
    for idx, prompt_item in enumerate(draft_result.payload.prompts, start=1):
        lines.append(f"## Prompt {idx}")
        lines.append(renderable_markdown_text(prompt_item.prompt))
        if prompt_item.answer and prompt_item.answer.text:
            lines.extend(["", f"Answer: {renderable_markdown_text(prompt_item.answer)}"])
        if prompt_item.source_note:
            lines.extend(["", f"Source: `{prompt_item.source_note}`"])
        lines.append("")
    recall_path.write_text("\n".join(lines).strip() + "\n", encoding="utf-8")
    repo.create_generation_provenance(
        runtime.build_provenance(
            artifact_kind="recall_note",
            artifact_id=str(recall_path),
            generated_draft=draft_result,
            accepted_by_user=True,
        )
    )
    render_markdown(recall_path.read_text(encoding="utf-8"))


@app.command("vocab", hidden=True)
def study_vocab(
    ctx: typer.Context,
    term: Optional[str] = typer.Argument(None, help="Vocabulary term"),
    definition_words: Optional[list[str]] = typer.Argument(None, help="Definition or translation"),
    domain: str = typer.Option("german-a1-to-b1", "--domain", help="Knowledge domain"),
):
    """Capture vocabulary inline without leaving the study flow."""
    from pb.cli.commands.vocab import add_vocab

    if not term:
        raise typer.BadParameter("Provide at least a vocabulary term.")
    definition = " ".join(definition_words or []).strip()
    if not definition and sys.stdin.isatty():
        definition = typer.prompt("Definition", default="", show_default=False).strip()
    if not definition:
        raise typer.BadParameter("A definition or translation is required.")
    add_vocab(ctx, term=term, definition=definition, domain=domain)
