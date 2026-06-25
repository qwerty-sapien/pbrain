# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Goal commands for the learning system."""

from __future__ import annotations

import re
import shlex
import sys
from contextlib import contextmanager
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import typer

from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console, get_err_console, resolve_render_width
from pb.cli.helpers import confirm_choice, prompt_text
from pb.cli.llm_guard import runtime_for_ctx
from pb.cli.pickers import pick_boolean, pick_many_choices, pick_single_choice
from pb.cli.preview import preview_decision, render_markdown_preview, render_styled_preview
from rich.console import Group
from rich.markup import escape
from rich.text import Text
from pb.core.clarifier import (
    ClarifierService,
    ask_clarifier_questions,
    build_clarifier_context,
    clarifier_prompt_block,
    learning_intent_style_guidance,
)
from pb.core.enums import BloomStage, EvidenceType, FeedbackSource, PracticeStage
from pb.core.entity_refs import display_ref
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.core.goal_roadmaps import (
    attach_roadmap_to_goal,
    build_goal_roadmap_prompt,
    ensure_goal_seed_tasks,
    ensure_roadmap_populated,
    fallback_goal_roadmap,
    materialize_next_frontier_tasks,
    project_title_for_goal,
    roadmap_from_goal,
    write_goal_roadmap_note,
)
from pb.core.learning_block_flow import collect_revision_feedback, learner_profile_suffix
from pb.core.graph_writer import make_slug
from pb.core.naming import (
    NameService,
    apply_generated_title,
    stored_display_title,
)
from pb.core.product_control import ProductControlEngine
from pb.core.roadmap_dag import (
    build_symbolic_dag,
    render_legend_lines,
    render_symbolic_node_lines,
    render_unicode_dependency_lines,
)
from pb.core.renderables import renderable_cli_text
from pb.core.resources import read_template_text
from pb.core.scope_resolution import matching_goals
from pb.core.staging import build_assumptions, build_learning_context, build_reflection, needs_single_clarification
from pb.domain.models import GoalArc, Track
from pb.llm.drafts import GoalDraft, GoalRoadmapDraft, GoalRoadmapNodeDraft
from pb.llm.runtime import DraftGenerationError, GeneratedDraft
from pb.storage.repository import Repository


app = typer.Typer(no_args_is_help=False)

_DOMAIN_STOPWORDS: frozenset[str] = frozenset({
    "get", "be", "become", "learn", "study", "understand", "know", "improve",
    "better", "at", "in", "on", "for", "a", "an", "the", "to", "and", "or",
    "how", "why", "what", "make", "build", "do", "use", "master", "practise",
    "practice", "achieve", "reach", "complete", "pass", "ace", "pursue",
})


def _extract_domain(phrase: str) -> str:
    """Return first non-stopword token from phrase, fallback to first token."""
    tokens = phrase.lower().split()
    for token in tokens:
        if token not in _DOMAIN_STOPWORDS and len(token) > 2:
            return token
    return tokens[0] if tokens else "learning"


def _default_goal_fields(draft: GoalDraft) -> GoalDraft:
    """Apply required defaults by execution mode."""
    payload = draft.model_copy(deep=True)
    if payload.execution_mode in {"study", "mixed"}:
        payload.study_framework = payload.study_framework or "bloom_retrieval"
        payload.target_bloom_stage = payload.target_bloom_stage or BloomStage.APPLY
    else:
        payload.study_framework = None
        payload.current_bloom_stage = None
        payload.target_bloom_stage = None
    if payload.execution_mode in {"practise", "mixed"}:
        payload.practice_framework = payload.practice_framework or "deliberate_practice"
        payload.target_practice_stage = payload.target_practice_stage or PracticeStage.INTEGRATE
        payload.feedback_source = payload.feedback_source or FeedbackSource.ARTIFACT
    else:
        payload.practice_framework = None
        payload.current_practice_stage = None
        payload.target_practice_stage = None
    return payload


def _build_goal_prompt(
    raw_focus: str,
    *,
    existing_goal: GoalArc | None = None,
    clarifier_bundle=None,
) -> str:
    """Build the structured goal prompt for the required LLM runtime."""
    context = ""
    if existing_goal is not None:
        context = (
            f"Existing goal title: {existing_goal.title}\n"
            f"Existing domain: {existing_goal.domain}\n"
            f"Existing mode: {existing_goal.execution_mode}\n"
            f"Existing success definition: {existing_goal.success_definition}\n"
            f"Existing description: {existing_goal.description}\n"
        )
    return (
        "Turn the messy learning request into a structured goal for a CLI-first learning system.\n"
        "Prefer study for conceptual work, practise for deliberate drills, mixed when both matter.\n"
        "Use Bloom for study targets and deliberate-practice stages for practice targets.\n"
        + learning_intent_style_guidance()
        + "Return concise but concrete fields.\n\n"
        + f"{context}"
        + f"User request:\n{raw_focus}\n"
        + clarifier_prompt_block(clarifier_bundle)
    )


def _should_clarify_goal(raw_focus: str) -> bool:
    return needs_single_clarification(raw_focus)


def _append_clarification(raw_focus: str, clarification: str) -> str:
    if not clarification.strip():
        return raw_focus
    return f"{raw_focus}\nClarification:\n{clarification.strip()}"


def _goal_route_options(goal: GoalArc) -> list[tuple[str, str]]:
    focus = goal.domain or stored_display_title(goal) or goal.title
    return [
        ("save", "Save only"),
        ("plan day --quick", "Plan day"),
        (f"study {shlex.quote(focus)}", "Start study now"),
        (f"practise {shlex.quote(focus)}", "Start practise now"),
    ]


def _goal_route_default(goal_like) -> str:
    mode = getattr(goal_like, "execution_mode", "mixed")
    if mode == "study":
        return "study"
    if mode in {"practise", "practice"}:
        return "practise"
    return "plan"


def _manual_goal_wizard(
    raw_focus: str,
    *,
    existing: GoalArc | None = None,
    clarification: str = "",
) -> GoalDraft:
    """Collect every required goal field without an LLM."""

    seed = existing.title if existing is not None else raw_focus.strip()
    default_mode = getattr(existing, "execution_mode", "") or ("mixed" if len(seed.split()) <= 2 else "study")
    default_domain = getattr(existing, "domain", "") or seed.split()[0].lower().replace(" ", "-")
    default_success = getattr(existing, "success_definition", "") or clarification or (
        f"Be able to use {seed} in a concrete learning block."
    )
    default_description = getattr(existing, "description", "") or (
        f"Turn {seed} into a learnable, executable direction."
    )
    default_horizon = getattr(getattr(existing, "horizon", None), "value", None) or "quarter"
    default_framework = getattr(existing, "framework", "") or "Bloom-first learning loop"

    typer.echo("Manual goal setup")
    typer.echo("Tip: keep the title short, the success definition testable, and the mode honest.")

    title = prompt_text("Goal title", default=seed or "Learning goal") or seed or "Learning goal"
    domain = prompt_text("Domain", default=default_domain or title.lower()) or default_domain or title.lower()
    mode = (
        prompt_text("Mode (study/practise/mixed)", default=default_mode or "mixed").strip().lower()
        or default_mode
        or "mixed"
    )
    description = (
        prompt_text("Why this matters", default=default_description or title).strip()
        or default_description
        or title
    )
    success_definition = (
        prompt_text("Success definition", default=default_success).strip()
        or default_success
    )
    framework = prompt_text("Planning basis", default=default_framework).strip() or default_framework
    horizon = (
        prompt_text("Horizon (month/quarter/six_month)", default=default_horizon).strip().lower()
        or default_horizon
    )

    study_target = getattr(existing, "target_bloom_stage", None)
    practice_target = getattr(existing, "target_practice_stage", None)
    feedback = getattr(existing, "feedback_source", None)
    evidence = getattr(existing, "evidence_type", None)

    study_value = study_target.value if study_target else "apply"
    practice_value = practice_target.value if practice_target else "integrate"
    feedback_value = feedback.value if feedback else "artifact"
    evidence_value = evidence.value if evidence else "artifact"

    if mode in {"study", "mixed"}:
        study_value = prompt_text(
            "Target Bloom stage",
            default=study_value,
        ).strip().lower() or study_value
    if mode in {"practise", "practice", "mixed"}:
        practice_value = prompt_text(
            "Target practice stage",
            default=practice_value,
        ).strip().lower() or practice_value
        feedback_value = prompt_text(
            "Feedback source",
            default=feedback_value,
        ).strip().lower() or feedback_value
        evidence_value = prompt_text(
            "Evidence type",
            default=evidence_value,
        ).strip().lower() or evidence_value

    draft = GoalDraft(
        title=title,
        description=description,
        domain=domain,
        execution_mode="practise" if mode == "practice" else mode,
        horizon=horizon,
        framework=framework,
        study_framework="bloom_retrieval" if mode in {"study", "mixed"} else None,
        current_bloom_stage=getattr(existing, "current_bloom_stage", None),
        target_bloom_stage=BloomStage(study_value) if mode in {"study", "mixed"} else None,
        practice_framework="deliberate_practice" if mode in {"practise", "practice", "mixed"} else None,
        current_practice_stage=getattr(existing, "current_practice_stage", None),
        target_practice_stage=PracticeStage(practice_value) if mode in {"practise", "practice", "mixed"} else None,
        success_definition=success_definition,
        primary_metric=getattr(existing, "primary_metric", None),
        feedback_source=FeedbackSource(feedback_value) if mode in {"practise", "practice", "mixed"} else None,
        evidence_type=EvidenceType(evidence_value) if mode in {"practise", "practice", "mixed"} else None,
    )
    return _default_goal_fields(draft)


def _seed_goal_draft(raw_focus: str, *, clarification: str = "") -> GoalDraft:
    """Produce a deterministic goal draft when prompts are unavailable."""

    normalized = " ".join(raw_focus.split()) or "Learning goal"
    domain = _extract_domain(normalized)
    mode = "mixed" if len(normalized.split()) <= 2 else "study"
    draft = GoalDraft(
        title=normalized.title(),
        description=f"Turn {normalized} into a concrete learning loop.",
        domain=domain,
        execution_mode=mode,
        horizon="quarter",
        framework="Bloom-first learning loop",
        study_framework="bloom_retrieval" if mode in {"study", "mixed"} else None,
        target_bloom_stage=BloomStage.APPLY if mode in {"study", "mixed"} else None,
        practice_framework="deliberate_practice" if mode in {"practise", "mixed"} else None,
        target_practice_stage=PracticeStage.INTEGRATE if mode in {"practise", "mixed"} else None,
        success_definition=clarification or f"Be able to make measurable progress in {normalized}.",
        feedback_source=FeedbackSource.ARTIFACT if mode in {"practise", "mixed"} else None,
        evidence_type=EvidenceType.ARTIFACT if mode in {"practise", "mixed"} else None,
    )
    return _default_goal_fields(draft)


def _resolve_goal_draft(
    ctx: typer.Context,
    *,
    raw_goal: str,
    prompt: str,
    source_scope: str,
    artifact_kind: str,
    artifact_id: str,
    existing_goal: GoalArc | None = None,
) -> tuple[GoalDraft, GeneratedDraft | None, object]:
    """Run staged goal drafting with one clarification and a manual fallback."""

    runtime = runtime_for_ctx(ctx)
    repo = ctx.obj["repo"]
    runtime_ctx = ctx.obj["runtime"]
    recorder = runtime.make_stage_recorder("goal", raw_goal, route_hint="goal")
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("goal", raw_goal, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("goal", raw_goal, context))
    if sys.stdin.isatty() and bool(ctx.obj.get("verbose")):
        get_console().print(f"[dim]{reflection}[/]")

    clarified_goal = raw_goal
    clarification = ""
    clarifier_bundle = None
    if sys.stdin.isatty() and _should_clarify_goal(raw_goal):
        clarifier_context = build_clarifier_context(
            repo,
            runtime_ctx,
            raw_request=raw_goal,
            scope="goal",
            mode="goal",
            domain=getattr(existing_goal, "domain", "") if existing_goal else "",
        )
        questions = ClarifierService(runtime).generate_questions(
            raw_goal,
            clarifier_context,
            max_questions=1,
            scope="goal",
        )
        clarifier_bundle = ask_clarifier_questions(questions) if questions else None
        answers = clarifier_bundle.answers if clarifier_bundle is not None else {}
        clarification = next((answer for answer in answers.values() if answer.strip()), "")
        clarified_goal = _append_clarification(raw_goal, clarification)
        recorder.add(
            "clarify",
            {
                "questions": answers,
            },
            status="ok" if clarification else "skipped",
        )
        prompt = _build_goal_prompt(
            clarified_goal,
            existing_goal=existing_goal,
            clarifier_bundle=clarifier_bundle,
        )
    prompt += feedback_prompt_suffix(runtime_ctx.vault_path, "goal")

    try:
        draft_result = runtime.generate_draft(
            GoalDraft,
            prompt,
            source_scope=source_scope,
        )
        draft = _default_goal_fields(draft_result.payload)
        recorder.add(
            "draft",
            {
                "model": draft_result.model,
                "attempts": [attempt.__dict__ for attempt in draft_result.attempts],
            },
        )
        return draft, draft_result, recorder
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
            get_err_console().print(f"[warn]{exc.to_user_message()}[/]")
            draft = _manual_goal_wizard(raw_goal, existing=existing_goal, clarification=clarification)
        else:
            draft = _seed_goal_draft(raw_goal, clarification=clarification)
        recorder.add(
            "verify",
            {
                "mode": "manual",
                "title": draft.title,
                "execution_mode": draft.execution_mode,
            },
        )
        return draft, None, recorder


def _route_after_goal(ctx: typer.Context, goal: GoalArc, *, auto_yes: bool = False) -> None:
    """Offer immediate routing into the canonical learning loop."""

    options = _goal_route_options(goal)
    default_key = _goal_route_default(goal)
    default_command = next(
        (value for value, _ in options if value == default_key or value.startswith(f"{default_key} ")),
        options[0][0],  # fallback to first option ("save")
    )

    if not sys.stdin.isatty():
        typer.echo(f"Next recommended command: pb {default_command}")
        return

    selected = pick_single_choice(
        [(value, label) for value, label in options],
        title="What should happen next?",
        text="You can save the goal and stop here, or route directly into planning, study, or practise.",
    )
    command = selected or default_command
    if command == "save":
        return
    if auto_yes and not command.endswith("--yes"):
        routed = f"{command} --yes"
    else:
        routed = command
    run_internal_command(ctx, routed)


def _draft_to_goal(draft: GoalDraft, existing: GoalArc | None = None) -> GoalArc:
    """Convert a validated goal draft into the durable GoalArc model."""
    from pb.domain.enums import Horizon

    horizon_enum = Horizon(draft.horizon)
    if existing is None:
        return GoalArc(
            title=draft.title,
            domain=draft.domain,
            execution_mode=draft.execution_mode,
            study_framework=draft.study_framework,
            current_bloom_stage=draft.current_bloom_stage,
            target_bloom_stage=draft.target_bloom_stage,
            practice_framework=draft.practice_framework,
            current_practice_stage=draft.current_practice_stage,
            target_practice_stage=draft.target_practice_stage,
            horizon=horizon_enum,
            description=draft.description,
            success_definition=draft.success_definition,
            framework=draft.framework,
            primary_metric=draft.primary_metric,
            feedback_source=draft.feedback_source,
            evidence_type=draft.evidence_type,
        )

    existing.title = draft.title
    existing.domain = draft.domain
    existing.execution_mode = draft.execution_mode
    existing.study_framework = draft.study_framework
    existing.current_bloom_stage = draft.current_bloom_stage
    existing.target_bloom_stage = draft.target_bloom_stage
    existing.practice_framework = draft.practice_framework
    existing.current_practice_stage = draft.current_practice_stage
    existing.target_practice_stage = draft.target_practice_stage
    existing.horizon = horizon_enum
    existing.description = draft.description
    existing.success_definition = draft.success_definition
    existing.framework = draft.framework
    existing.primary_metric = draft.primary_metric
    existing.feedback_source = draft.feedback_source
    existing.evidence_type = draft.evidence_type
    return existing


def _write_goal_note(goal: GoalArc) -> None:
    """Mirror the durable goal state into the vault note."""
    from datetime import datetime as _dt

    import yaml as _yaml
    from pb.vault.config import get_vault_path

    vault = get_vault_path()
    goals_dir = vault / "direction" / "goals"
    goals_dir.mkdir(parents=True, exist_ok=True)
    display_title = stored_display_title(goal) or goal.title
    generated = getattr(goal, "generated_names", {}) or {}
    slug = generated.get("slug") or make_slug(display_title)
    goal_note_path = goals_dir / f"{slug}.md"
    horizon_str = goal.horizon.value if goal.horizon else "six_month"
    fm = {
        "type": "goal",
        "title": display_title,
        "status": goal.status,
        "project_title": generated.get("goal_project_title"),
        "roadmap_note_path": generated.get("roadmap_note_path"),
        "horizon": horizon_str,
        "domain": goal.domain,
        "execution_mode": goal.execution_mode,
        "study_framework": goal.study_framework,
        "current_bloom_stage": goal.current_bloom_stage.value if goal.current_bloom_stage else None,
        "target_bloom_stage": goal.target_bloom_stage.value if goal.target_bloom_stage else None,
        "practice_framework": goal.practice_framework,
        "current_practice_stage": goal.current_practice_stage.value if goal.current_practice_stage else None,
        "target_practice_stage": goal.target_practice_stage.value if goal.target_practice_stage else None,
        "success_definition": goal.success_definition,
        "framework": goal.framework,
        "description": goal.description,
        "primary_metric": goal.primary_metric,
        "feedback_source": goal.feedback_source.value if goal.feedback_source else None,
        "evidence_type": goal.evidence_type.value if goal.evidence_type else None,
        "target_date": goal.target_date.strftime("%Y-%m-%d") if goal.target_date else None,
        "updated": _dt.utcnow().strftime("%Y-%m-%d"),
    }
    frontmatter = _yaml.dump({k: v for k, v in fm.items() if v not in (None, "")}, default_flow_style=False, allow_unicode=True)
    body = (
        f"# {display_title}\n\n"
        f"{goal.description}\n\n"
        f"## Success Definition\n\n{goal.success_definition or '-'}\n\n"
        f"## Project\n\n{generated.get('goal_project_title') or '-'}\n\n"
        f"## Planning Basis\n\n{goal.framework or '-'}\n"
    )
    goal_note_path.write_text(f"---\n{frontmatter}---\n\n{body}")

    domain_folder = generated.get("folder_name") or make_slug(goal.domain or display_title)
    domain_dir = vault / "knowledge" / str(domain_folder)
    domain_dir.mkdir(parents=True, exist_ok=True)

    index_path = domain_dir / "_index.md"
    if not index_path.exists():
        from string import Template as _T

        _INDEX_FALLBACK = (
            "---\ntype: domain_index\ntitle: $goal_title\nslug: $slug\ncreated: $date\n---\n\n"
            "# $goal_title\n\nDomain index. Add notes as [[note-title]] links below.\n\n## Notes\n\n_No notes yet._\n"
        )
        try:
            index_tmpl_text = read_template_text("index_md.md")
        except (FileNotFoundError, OSError, ModuleNotFoundError):
            index_tmpl_text = _INDEX_FALLBACK
        index_tmpl = _T(index_tmpl_text)
        index_path.write_text(index_tmpl.safe_substitute(
            goal_title=goal.title,
            slug=str(domain_folder),
            date=_dt.utcnow().strftime("%Y-%m-%d"),
        ))

    state_path = domain_dir / "_state.md"
    if not state_path.exists():
        from string import Template as _T

        _STATE_FALLBACK = (
            "---\ntype: domain_state\nupdated: $date\n"
            "stage_counts:\n  new: 0\n  learning: 0\n  learnt: 0\n  stale: 0\n"
            "session_summaries: []\n---\n\n# $goal_title -- Learning State\n\nNo sessions yet.\n"
        )
        try:
            state_tmpl_text = read_template_text("state_md.md")
        except (FileNotFoundError, OSError, ModuleNotFoundError):
            state_tmpl_text = _STATE_FALLBACK
        state_tmpl = _T(state_tmpl_text)
        state_path.write_text(state_tmpl.safe_substitute(
            goal_title=goal.title,
            date=_dt.utcnow().strftime("%Y-%m-%d"),
        ))


@contextmanager
def _transactional_goal_write(repo, goal_id: str):
    """Roll back the SQLite goal_arcs row if the vault write raises."""
    try:
        yield
    except Exception:
        try:
            repo.hard_delete_goal_arc(goal_id)
        except Exception:
            pass  # Don't mask the original error
        raise


def _persist_goal(
    repo: Repository,
    draft: GoalDraft,
    *,
    existing: GoalArc | None = None,
    track_name: str = "",
    target_date: Optional[datetime] = None,
    primary_metric_override: Optional[str] = None,
    goal_names=None,
) -> GoalArc:
    """Write the goal state to SQLite and the vault."""
    goal = _draft_to_goal(draft, existing=existing)
    if goal_names is not None:
        apply_generated_title(goal, goal_names, title_key="goal_title")
    if target_date is not None:
        goal.target_date = target_date
    if primary_metric_override:
        goal.primary_metric = primary_metric_override
    if existing is None:
        repo.create_goal_arc(goal)
    else:
        repo.update_goal_arc(goal)

    if track_name:
        tracks = repo.list_tracks()
        track_obj = next((t for t in tracks if t.name == track_name or t.name.startswith(track_name)), None)
        if track_obj and goal.id not in track_obj.linked_goal_arc_ids:
            track_obj.linked_goal_arc_ids.append(goal.id)
            repo.update_track(track_obj)

    # STATE-01: wrap vault write so SQLite row is rolled back on failure
    if existing is None:
        with _transactional_goal_write(repo, goal.id):
            _write_goal_note(goal)
    else:
        _write_goal_note(goal)
    return goal


_TIMEFRAME_RE = re.compile(
    r"^(\d+(?:\.\d+)?)\s*(w|wk|wks|week|weeks|m|mo|mos|month|months|y|yr|yrs|year|years)$"
)
_TIMEFRAME_DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d", "%Y/%m/%d", "%d/%m/%y", "%d-%m-%y")


def _horizon_for_days(days: int) -> str:
    if days <= 35:
        return "month"
    if days <= 100:
        return "quarter"
    return "six_month"


def _parse_timeframe(raw: str) -> tuple[Optional[datetime], str]:
    """Parse '3 weeks' / '6 months' / '1 year' / 'DD/MM/YYYY' into (target_date, horizon)."""
    s = (raw or "").strip()
    if not s:
        return None, "six_month"

    for fmt in _TIMEFRAME_DATE_FORMATS:
        try:
            dt = datetime.strptime(s, fmt)
            delta_days = max(1, (dt - datetime.utcnow()).days)
            return dt, _horizon_for_days(delta_days)
        except ValueError:
            continue

    match = _TIMEFRAME_RE.match(s.lower())
    if match:
        n = float(match.group(1))
        unit = match.group(2)
        if unit.startswith("w"):
            days = int(n * 7)
        elif unit.startswith("y"):
            days = int(n * 365)
        else:
            days = int(n * 30)
        return datetime.utcnow() + timedelta(days=max(1, days)), _horizon_for_days(days)

    return None, "six_month"


def _render_goal_preview(draft: GoalDraft, *, title: str = "Goal Draft") -> None:
    """Render a minimal preview surfacing only the user-facing fields."""
    render_styled_preview(
        title=title,
        rows=[
            ("Title", draft.title),
            ("Description", draft.description),
            ("Domain", draft.domain),
            ("Success", draft.success_definition),
        ],
    )


def _prerequisites_lines(draft: GoalDraft, roadmap: GoalRoadmapDraft) -> list[str]:
    """Return prerequisite point-form list, or empty if straightforward topic."""
    prereqs = getattr(draft, "prerequisites", None) or []
    if isinstance(prereqs, str):
        prereqs = [p.strip() for p in prereqs.split(",") if p.strip()]
    if not prereqs:
        return []
    from pb.core.renderables import renderable_cli_text as _rcli
    return [f"- {_rcli(p)}" for p in prereqs]


def _roadmap_dag(roadmap: GoalRoadmapDraft):
    return build_symbolic_dag(roadmap.nodes)


def _render_dag_chart_lines(roadmap: GoalRoadmapDraft) -> list[str]:
    """Return compact DAG lines followed by the wrapped legend."""
    dag = _roadmap_dag(roadmap)
    lines = render_unicode_dependency_lines(dag)
    legend_lines = render_legend_lines(dag, width=max(40, resolve_render_width() - 6))
    if legend_lines:
        lines.extend(["", *legend_lines])
    return lines


def _roadmap_task_lines(roadmap: GoalRoadmapDraft) -> list[str]:
    """Compact roadmap lines with symbol indexes and succinct sub-bullets."""
    dag = _roadmap_dag(roadmap)
    symbol_by_id = dag.symbol_by_id
    node_lookup = {node.node_id: node for node in roadmap.nodes}
    lines: list[str] = []
    for index, dag_node in enumerate(dag.nodes[:10], start=1):
        node = node_lookup.get(dag_node.node_id)
        if node is None:
            continue
        title = renderable_cli_text(node.title).strip() or node.title
        scope = renderable_cli_text(node.scope or "").strip()
        milestone = renderable_cli_text(node.milestone or "").strip()
        success_check = renderable_cli_text(node.success_check or "").strip()
        symbol = symbol_by_id.get(node.node_id, str(index))
        lines.append(f"[{symbol}] **{title}**")
        if scope:
            lines.append(f"- Scope: {scope}")
        if milestone:
            lines.append(f"- Milestone: {milestone}")
        if success_check:
            lines.append(f"- Check: {success_check}")
        lines.append("")
    if len(roadmap.nodes) > 10:
        lines.append(f"*... and {len(roadmap.nodes) - 10} more tasks*")
    elif lines and lines[-1] == "":
        lines.pop()
    return lines


def _goal_header_renderable(draft: GoalDraft) -> Group:
    """Return a plain left-aligned goal header without markdown heading centering."""
    title = renderable_cli_text(draft.title).strip() or "Untitled goal"
    domain = renderable_cli_text(draft.domain).strip()
    parts: list[object] = [Text(title, style="bold white")]
    if domain:
        parts.extend([Text(""), Text(domain, style="bold magenta")])
    return Group(*parts)


def _dag_chart_renderable(roadmap: GoalRoadmapDraft) -> Text:
    """Render the compact DAG and legend with lightweight symbol styling."""
    dag = _roadmap_dag(roadmap)
    accent = getattr(getattr(roadmap, "presentation", None), "accent", "cyan")
    text = Text()
    graph_lines = render_unicode_dependency_lines(dag)
    legend_lines = render_legend_lines(dag, width=max(40, resolve_render_width() - 6))
    for line_index, line in enumerate(graph_lines):
        if line_index > 0:
            text.append("\n")
        text.append_text(_styled_graph_line(line, accent=accent))
    if legend_lines:
        text.append("\n\n")
        for line_index, line in enumerate(legend_lines):
            if line_index > 0:
                text.append("\n")
            text.append_text(_styled_legend_line(line, accent=accent))
    return text


def _roadmap_task_renderable(roadmap: GoalRoadmapDraft) -> Text:
    """Render compact roadmap blocks with spacing and terminal-safe accents."""
    dag = _roadmap_dag(roadmap)
    symbol_by_id = dag.symbol_by_id
    node_lookup = {node.node_id: node for node in roadmap.nodes}
    presentation = getattr(roadmap, "presentation", None)
    accent = getattr(presentation, "accent", "cyan")
    density = getattr(presentation, "density", "balanced")
    gap_lines = 1 if density == "compact" else 2 if density == "relaxed" else 1
    text = Text()
    for index, dag_node in enumerate(dag.nodes[:10], start=1):
        node = node_lookup.get(dag_node.node_id)
        if node is None:
            continue
        title = renderable_cli_text(node.title).strip() or node.title
        scope = renderable_cli_text(node.scope or "").strip()
        milestone = renderable_cli_text(node.milestone or "").strip()
        success_check = renderable_cli_text(node.success_check or "").strip()
        if index > 1:
            text.append("\n" * gap_lines)
        text.append("[", style="roadmap.bracket")
        text.append(symbol_by_id.get(node.node_id, str(index)), style=f"bold {accent}")
        text.append("] ", style="roadmap.bracket")
        text.append(title, style="roadmap.title")
        if node.branch:
            text.append(" ")
            text.append(node.branch, style=f"branch.{node.branch}")
        if scope:
            text.append("\n")
            text.append("• ", style="roadmap.bullet")
            text.append("Scope: ", style="roadmap.label")
            text.append(scope, style="roadmap.meta")
        if milestone:
            text.append("\n")
            text.append("• ", style="roadmap.bullet")
            text.append("Milestone: ", style="roadmap.label")
            text.append(milestone, style="roadmap.meta")
        if success_check:
            text.append("\n")
            text.append("• ", style="roadmap.bullet")
            text.append("Check: ", style="roadmap.label")
            text.append(success_check, style="roadmap.check")
    if len(roadmap.nodes) > 10:
        if text:
            text.append("\n")
        text.append(f"... and {len(roadmap.nodes) - 10} more tasks", style="dim italic")
    return text


def _styled_graph_line(line: str, *, accent: str) -> Text:
    text = Text()
    if "─▶" not in line:
        text.append(line, style=f"bold {accent}")
        return text
    left, right = line.split("─▶", 1)
    left_parts = [part.strip() for part in left.split("+")]
    right_parts = [part.strip() for part in right.split("+")]
    for index, part in enumerate(left_parts):
        if index > 0:
            text.append(" + ", style="graph.edge")
        text.append(part, style=f"bold {accent}")
    text.append(" ─▶ ", style="graph.edge")
    for index, part in enumerate(right_parts):
        if index > 0:
            text.append(" + ", style="graph.edge")
        text.append(part, style=f"bold {accent}")
    return text


def _styled_legend_line(line: str, *, accent: str) -> Text:
    if line == "Legend":
        return Text(line, style="legend.heading")
    match = re.match(r"^([A-Z]+)(\s{2,})(.*)$", line)
    if not match:
        return Text(line, style="legend.text")
    symbol, spacing, remainder = match.groups()
    text = Text()
    text.append(symbol, style=f"bold {accent}")
    text.append(spacing, style="legend.text")
    text.append(remainder, style="legend.text")
    return text


def _render_goal_bundle_preview(
    draft: GoalDraft,
    roadmap: GoalRoadmapDraft,
    *,
    title: str = "Goal Draft",
) -> None:
    """Render the goal plus roadmap preview as clean learner-facing output."""
    prereq_lines = _prerequisites_lines(draft, roadmap)
    sections: list[tuple[str, object]] = [("", _goal_header_renderable(draft))]
    if prereq_lines:
        sections.append(("Prerequisites", prereq_lines))
    sections.append(("Dependency Chart", _dag_chart_renderable(roadmap)))
    sections.append(("Roadmap", _roadmap_task_renderable(roadmap)))

    render_markdown_preview(
        title=title,
        sections=sections,
    )


def _pick_confident_roadmap_nodes(roadmap: GoalRoadmapDraft) -> list[str]:
    """Let the learner flag roadmap topics they already feel confident in."""
    if not sys.stdin.isatty() or not roadmap.nodes:
        return []

    dag = _roadmap_dag(roadmap)
    options = [
        (node_id, f"[{symbol}] {title}")
        for node_id, symbol, title in render_symbolic_node_lines(dag)
    ]
    selected = pick_many_choices(
        options,
        title="Confidence check",
        text=(
            "Tick any roadmap topics you already feel confident in. "
            "Use the DAG legend order to navigate, then confirm your selections."
        ),
    )
    if selected:
        selected_set = set(selected)
        console = get_console()
        console.print("[dim]Your confidence selections:[/]")
        for node_id, symbol, title in render_symbolic_node_lines(dag):
            line = f"[{symbol}] {title}"
            if node_id in selected_set:
                line = f"[{symbol}] ✓ {title}"
            console.print(line, markup=False)
    return selected


def _resolve_goal_roadmap(
    ctx: typer.Context,
    *,
    raw_goal: str,
    goal_draft: GoalDraft,
    existing_goal: GoalArc | None = None,
) -> GoalRoadmapDraft:
    """Generate a structured roadmap for a goal, with deterministic fallback."""
    runtime = runtime_for_ctx(ctx)
    prompt = build_goal_roadmap_prompt(goal_draft, raw_goal, existing_goal=existing_goal)
    prompt += feedback_prompt_suffix(ctx.obj["runtime"].vault_path, "goal_roadmap")
    prompt += learner_profile_suffix(ctx.obj["repo"], ctx.obj["runtime"])
    try:
        return runtime.generate_draft(
            GoalRoadmapDraft,
            prompt,
            source_scope=f"goal_roadmap:{goal_draft.title}",
        ).payload
    except DraftGenerationError as exc:
        get_err_console().print(f"[warn]{exc.to_user_message()}[/]")
        return fallback_goal_roadmap(goal_draft)


def _refine_goal_roadmap(
    ctx: typer.Context,
    *,
    raw_goal: str,
    goal_draft: GoalDraft,
    roadmap: GoalRoadmapDraft,
    instruction: str,
) -> GoalRoadmapDraft:
    """Refine an existing roadmap from user feedback."""
    runtime = runtime_for_ctx(ctx)
    prompt = (
        build_goal_roadmap_prompt(goal_draft, raw_goal)
        + "\nCurrent roadmap JSON:\n"
        + str(roadmap.model_dump(mode="json"))
        + "\n\nUser refinement:\n"
        + instruction.strip()
        + "\n\n"
        + learner_profile_suffix(ctx.obj["repo"], ctx.obj["runtime"])
    )
    try:
        return runtime.generate_draft(
            GoalRoadmapDraft,
            prompt,
            source_scope=f"goal_roadmap_refine:{goal_draft.title}",
        ).payload
    except DraftGenerationError:
        return roadmap


def _review_goal_roadmap(
    ctx: typer.Context,
    *,
    raw_goal: str,
    goal_draft: GoalDraft,
    roadmap: GoalRoadmapDraft,
    yes: bool,
) -> GoalRoadmapDraft:
    """Walk the user through the roadmap one task at a time."""
    if yes or not sys.stdin.isatty():
        return roadmap
    control_engine = ProductControlEngine(repo=ctx.obj["repo"], runtime=runtime_for_ctx(ctx))
    current = roadmap
    for index, node in enumerate(list(current.nodes), start=1):
        choice = pick_single_choice(
            [
                ("keep", "Keep this task as drafted"),
                ("revise", "Revise this task"),
                ("stop", "Stop roadmap review"),
            ],
            title=f"Task {index}: {renderable_cli_text(node.title)}",
            text=renderable_cli_text(node.success_check or node.milestone or node.scope),
        )
        if choice is None or choice == "keep":
            continue
        if choice == "stop":
            break
        feedback = collect_revision_feedback(
            engine=control_engine,
            repo=ctx.obj["repo"],
            runtime_ctx=ctx.obj["runtime"],
            mode=node.branch,
            artifact_kind="goal_roadmap",
            artifact_id=goal_draft.title,
            current_artifact=str(node.model_dump(mode="json")),
            domain=goal_draft.domain,
            target=node.scope or node.title,
            current_node=node,
            title=f"Task {index}: {node.title}",
        )
        if feedback is None:
            continue
        instruction = (
            f"Refine roadmap node {node.node_id}.\n"
            f"Learner note: {feedback.free_text}\n"
            f"Control action: {feedback.decision.action}\n"
            f"Reason: {feedback.decision.reason}\n"
            f"Instruction: {feedback.decision.instruction}"
        )
        current = _refine_goal_roadmap(
            ctx,
            raw_goal=raw_goal,
            goal_draft=goal_draft,
            roadmap=current,
            instruction=instruction,
        )
    return current


def _guided_goal_flow(ctx: typer.Context) -> None:
    """Prompt for a focused goal description and immediately route to LLM draft."""
    repo = ctx.obj["repo"]
    ensure_goal_seed_tasks(repo, repo.list_goal_arcs(status=None), vault_path=ctx.obj["runtime"].vault_path)
    current_goals = repo.list_goal_arcs(status=None)
    if current_goals:
        typer.echo("Current goals:")
        for goal in current_goals[:5]:
            typer.echo(f"  {display_ref(goal, 'goal')}  {stored_display_title(goal) or goal.title}")
        typer.echo("")

    try:
        raw_focus = prompt_text("What are you working toward?").strip()
        if not raw_focus:
            list_goals()
            return
    except (typer.Abort, EOFError, KeyboardInterrupt):
        raise typer.Exit(code=0)

    goal = _create_goal_via_llm(ctx, raw_focus, horizon="six_month")
    if goal is not None:
        _route_after_goal(ctx, goal, auto_yes=False)


@app.command("delete")
def goal_delete(ctx: typer.Context):
    """Delete goals via multiselect picker."""
    repo = ctx.obj["repo"]
    goals = repo.list_goal_arcs(status=None)
    if not goals:
        get_console().print("[dim]No goals to delete.[/]")
        raise typer.Exit(code=0)

    selected_ids = pick_many_choices(
        [(g.id, g.title) for g in goals],
        title="Delete goals",
        text="Select goals to remove.",
    )
    if not selected_ids:
        raise typer.Exit(code=0)

    for goal_id in selected_ids:
        repo.archive_goal_arc(goal_id)

    titles = ", ".join(
        escape(g.title) for g in goals if g.id in set(selected_ids)
    )
    get_console().print(f"[success]Archived: {titles}[/]")


@app.callback(invoke_without_command=True)
def goals_callback(ctx: typer.Context):
    """Run the compact goal flow or list all goals in non-interactive mode."""
    if ctx.invoked_subcommand is not None:
        return
    if ctx.obj and ctx.obj.get("repo") is not None:
        ensure_goal_seed_tasks(ctx.obj["repo"], ctx.obj["repo"].list_goal_arcs(status=None), vault_path=ctx.obj["runtime"].vault_path)
    if sys.stdin.isatty():
        _guided_goal_flow(ctx)
    else:
        list_goals()

@app.command("list")
def list_goals_cmd():
    """List all goal arcs."""
    list_goals()

def list_goals():
    repo = Repository()
    goals = repo.list_goal_arcs()

    if not goals:
        typer.echo("No goals defined.")
        typer.echo("Run 'pb goal' to create one.")
        return

    typer.echo("Goals:")
    for goal in goals:
        horizon = goal.horizon.value if goal.horizon else "N/A"
        domain = f" · {goal.domain}" if getattr(goal, "domain", "") else ""
        mode = getattr(goal, "execution_mode", "mixed")
        typer.echo(f"  {display_ref(goal, 'goal')}  [{horizon} · {mode}{domain}] {stored_display_title(goal)}")


def _create_goal_via_llm(
    ctx: typer.Context,
    raw_goal: str,
    *,
    horizon: str = "six_month",
    track: str = "",
    yes: bool = False,
    target_date: Optional[datetime] = None,
    primary_metric_override: Optional[str] = None,
) -> Optional[GoalArc]:
    """Shared LLM-backed goal creation. Returns the persisted goal or None when cancelled."""
    runtime = runtime_for_ctx(ctx)
    repo = ctx.obj["repo"]

    # Specificity gate — warn on vague (1-2 word) input in TTY mode only
    if needs_single_clarification(raw_goal):
        if sys.stdin.isatty() and not yes:
            get_console().print(
                f"[yellow]Goal looks vague ({len(raw_goal.split())} word(s)). "
                "Add more context for a useful draft.[/]"
            )

    draft, draft_result, recorder = _resolve_goal_draft(
        ctx,
        raw_goal=raw_goal,
        prompt=_build_goal_prompt(raw_goal),
        source_scope=f"goal:{raw_goal}",
        artifact_kind="goal",
        artifact_id=raw_goal,
    )
    draft.horizon = horizon
    roadmap = _resolve_goal_roadmap(ctx, raw_goal=raw_goal, goal_draft=draft)
    roadmap = _review_goal_roadmap(ctx, raw_goal=raw_goal, goal_draft=draft, roadmap=roadmap, yes=yes)
    roadmap = ensure_roadmap_populated(roadmap, draft)
    recorder.add(
        "verify",
        {
            "goal_preview": draft.model_dump(mode="json"),
            "roadmap_preview": roadmap.model_dump(mode="json"),
        },
    )
    _render_goal_bundle_preview(draft, roadmap)

    accepted = yes
    while not accepted:
        decision = preview_decision(yes=False, action_label="Create this goal")
        if decision.kind == "accept":
            accepted = True
            break
        if decision.kind == "cancel":
            break
        roadmap = _refine_goal_roadmap(
            ctx,
            raw_goal=raw_goal,
            goal_draft=draft,
            roadmap=roadmap,
            instruction=decision.text,
        )
        _render_goal_bundle_preview(draft, roadmap)

    if not accepted:
        if draft_result is not None:
            repo.create_generation_provenance(
                runtime.build_provenance(
                    artifact_kind="goal_draft",
                    artifact_id=raw_goal,
                    generated_draft=draft_result,
                    accepted_by_user=False,
                )
            )
        recorder.finalize("cancelled", artifact_kind="goal_draft", artifact_id=raw_goal)
        return None

    # Near-duplicate detection — only for new goals, not refine path
    similar = matching_goals(repo, raw_goal, limit=3)
    similar = [m for m in similar if m["title"].lower() != raw_goal.lower()]
    if similar and sys.stdin.isatty() and not yes:
        get_console().print("[yellow]Similar goals already exist:[/]")
        for m in similar:
            get_console().print(f"  [dim]{m['title']}[/]")
        if not pick_boolean("Create new goal anyway?"):
            recorder.finalize("cancelled", reason="duplicate_detected")
            return None

    goal_names = NameService(runtime).generate_names(
        "goal",
        raw_goal,
        {
            "domain": draft.domain,
            "subject": draft.domain or draft.title,
            "activity_type": "goal",
            "execution_mode": draft.execution_mode,
            "success_definition": draft.success_definition,
        },
    )

    goal = _persist_goal(
        repo,
        draft,
        track_name=track,
        target_date=target_date,
        primary_metric_override=primary_metric_override,
        goal_names=goal_names,
    )
    confident_node_ids = [] if yes else _pick_confident_roadmap_nodes(roadmap)
    roadmap_path = write_goal_roadmap_note(
        ctx.obj["runtime"].vault_path,
        goal,
        roadmap,
        confident_node_ids=confident_node_ids,
    )
    attach_roadmap_to_goal(
        goal,
        roadmap,
        roadmap_path=roadmap_path,
        confident_node_ids=confident_node_ids,
    )
    repo.update_goal_arc(goal)
    _write_goal_note(goal)
    seed_tasks = materialize_next_frontier_tasks(repo, goal, roadmap=roadmap, max_new=1)
    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="goal",
                artifact_id=goal.id,
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add(
        "materialize",
        {
            "goal_id": goal.id,
            "title": goal.title,
            "execution_mode": goal.execution_mode,
            "project_title": project_title_for_goal(goal, roadmap),
            "seed_tasks": [task.id for task in seed_tasks],
        },
    )
    recorder.finalize("persisted", artifact_kind="goal", artifact_id=goal.id)
    typer.echo(f"Created goal: {stored_display_title(goal) or goal.title}")
    if confident_node_ids:
        typer.echo(
            f"Marked {len(confident_node_ids)} roadmap topic(s) as confidence claims pending diagnostic confirmation."
        )
    if seed_tasks:
        typer.echo(f"Seed task: {seed_tasks[0].title}")
    return goal


@app.command("add")
def add_goal(
    ctx: typer.Context,
    goal_words: Optional[list[str]] = typer.Argument(None, help="Messy goal description. Multi-word OK."),
    horizon: str = typer.Option("six_month", "--horizon", "-h", help="Horizon: month|quarter|six_month"),
    track: str = typer.Option("", "--track", "-t", help="Link goal to track by name"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and persist it"),
):
    """Draft, preview, and persist a structured learning goal."""
    raw_goal = " ".join(goal_words or []).strip()
    if not raw_goal and sys.stdin.isatty():
        raw_goal = prompt_text("Describe the goal", default="")
    if not raw_goal:
        raise typer.BadParameter("A goal description is required.")

    goal = _create_goal_via_llm(ctx, raw_goal, horizon=horizon, track=track, yes=yes)
    if goal is None:
        raise typer.Exit(code=0)
    _route_after_goal(ctx, goal, auto_yes=yes)


@app.command("refine")
def refine_goal(
    ctx: typer.Context,
    goal_id: str = typer.Argument(..., help="Goal ID or prefix"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and update the goal"),
):
    """Re-draft an existing goal from its stored state and current evidence."""
    runtime = runtime_for_ctx(ctx)
    repo = ctx.obj["repo"]
    goal = repo.resolve_goal_ref(goal_id)
    if goal is None:
        raise typer.BadParameter(f"Goal not found: {goal_id}")

    sessions = []
    for task in repo.list_tasks():
        if goal.id in getattr(task, "linked_goal_arc_ids", []):
            sessions.extend(repo.list_sessions_for_task(task.id)[-3:])
    session_context = "\n".join(
        f"- {sess.branch} | {sess.subject_scope} | outcome={sess.actual_outcome or ''}"
        for sess in sessions[-6:]
    )
    prompt = _build_goal_prompt(
        f"{goal.title}\n{goal.description}\nRecent evidence:\n{session_context}",
        existing_goal=goal,
    )
    draft, draft_result, recorder = _resolve_goal_draft(
        ctx,
        raw_goal=goal.title,
        prompt=prompt,
        source_scope=f"goal_refine:{goal.id}",
        artifact_kind="goal_refine",
        artifact_id=goal.id,
        existing_goal=goal,
    )
    recorder.add("verify", {"preview": draft.model_dump(mode="json")})
    _render_goal_preview(draft, title="Refined Goal Draft")
    accepted = yes
    if not accepted:
        accepted = preview_decision(yes=False, action_label="Update this goal").kind == "accept"
    if not accepted:
        if draft_result is not None:
            repo.create_generation_provenance(
                runtime.build_provenance(
                    artifact_kind="goal_refine",
                    artifact_id=goal.id,
                    generated_draft=draft_result,
                    accepted_by_user=False,
                )
            )
        recorder.finalize("cancelled", artifact_kind="goal_refine", artifact_id=goal.id)
        raise typer.Exit(code=0)

    goal_names = NameService(runtime).generate_names(
        "goal",
        goal.title,
        {
            "domain": draft.domain,
            "subject": draft.domain or draft.title,
            "activity_type": "goal",
            "execution_mode": draft.execution_mode,
            "success_definition": draft.success_definition,
        },
    )
    goal = _persist_goal(repo, draft, existing=goal, goal_names=goal_names)
    ensure_goal_seed_tasks(repo, [goal], vault_path=ctx.obj["runtime"].vault_path)
    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="goal",
                artifact_id=goal.id,
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add("materialize", {"goal_id": goal.id, "title": goal.title})
    recorder.finalize("persisted", artifact_kind="goal", artifact_id=goal.id)
    typer.echo(f"Updated goal: {stored_display_title(goal) or goal.title}")


@app.command("tracks", hidden=True)
def tracks_command():
    """List all tracks (relocated from top-level)."""
    list_tracks()


def list_tracks():
    """List all tracks (called from main.py)."""
    repo = Repository()
    tracks = repo.list_tracks()

    if not tracks:
        typer.echo("No tracks defined.")
        typer.echo("Use 'pb goal track add <name>' to create one.")
        return

    typer.echo("Tracks:")
    for track in tracks:
        status = "active" if track.active else "inactive"
        typer.echo(f"  {display_ref(track, 'track')}  {track.name} ({status})")


track_app = typer.Typer()
app.add_typer(track_app, name="track", hidden=True, help="Track commands")


@track_app.command("add")
def add_track(
    name: str = typer.Argument(..., help="Track name"),
    description: str = typer.Option("", "--desc", "-d", help="Track description"),
):
    """Add a new track."""
    repo = Repository()
    track = Track(
        name=name,
        description=description,
    )
    repo.create_track(track)
    typer.echo(f"Created track: {display_ref(track, 'track')} {track.name}")


@track_app.command("list")
def track_list():
    """List all tracks."""
    list_tracks()


def _infer_domain_from_goal(goal, vault_path) -> Optional[str]:
    """Fuzzy-match goal title/description against vault domain directory names. D-16, Pitfall A3."""
    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return None
    try:
        domains = [d.name for d in knowledge_dir.iterdir() if d.is_dir()]
    except Exception:
        return None
    title_lower = (goal.title + " " + getattr(goal, "description", "")).lower()
    for domain in domains:
        if domain.lower() in title_lower or domain.lower().replace("-", " ") in title_lower:
            return domain
    return None


def _count_goal_domain_stages(vault_path, domain: str) -> dict:
    """Count notes by learning stage tag in vault domain directory. D-16."""
    import re
    domain_dir = vault_path / "knowledge" / domain
    if not domain_dir.exists():
        return {}
    stage_counts: dict = {"new": 0, "learning": 0, "learnt": 0, "stale": 0}
    try:
        for md_file in domain_dir.rglob("*.md"):
            try:
                content = md_file.read_text(encoding="utf-8", errors="ignore")
                # Look for learning_stage: in frontmatter OR #new/#learning tags
                if re.search(r"learning_stage:\s*new|#new\b", content):
                    stage_counts["new"] += 1
                elif re.search(r"learning_stage:\s*learning|#learning\b", content):
                    stage_counts["learning"] += 1
                elif re.search(r"learning_stage:\s*learnt|#learnt\b", content):
                    stage_counts["learnt"] += 1
                elif re.search(r"learning_stage:\s*stale|#stale\b", content):
                    stage_counts["stale"] += 1
            except Exception:
                continue
    except Exception:
        return {}
    return stage_counts


@app.command("report", hidden=True)
def goals_report(
    ctx: typer.Context,
    goal_id: Optional[str] = typer.Option(None, "--goal", help="Filter to single goal ID"),
):
    """Per-goal progress: tasks, sessions, and domain note stage distribution (ALGN-02)."""
    from pb.cli.console import get_console
    from pb.vault.config import get_vault_path

    console = get_console()
    repo = Repository()
    vault = get_vault_path()

    goals = repo.list_goal_arcs(status=None)  # include all statuses
    if goal_id:
        goals = [g for g in goals if g.id == goal_id]

    if not goals:
        console.print("No goals defined.")
        console.print("[dim]Run 'pb goal' to create one.[/]")
        return

    console.rule("[header]Goal Report[/]")

    for goal in goals:
        console.print()
        cadence = getattr(goal, "cadence", None) or getattr(goal, "horizon", None)
        cadence_str = cadence.value if hasattr(cadence, "value") else str(cadence) if cadence else "ongoing"
        console.print(f"[header]{goal.title}[/header]  [dim][{cadence_str} · {goal.status}][/dim]")
        console.print()

        # Tasks linked to this goal
        all_tasks = repo.list_tasks()
        linked_tasks = [t for t in all_tasks if goal.id in getattr(t, "linked_goal_arc_ids", [])]

        console.print("  [subheader]Tasks[/subheader]")
        if linked_tasks:
            display_tasks = linked_tasks[:5]
            for t in display_tasks:
                from pb.core.enums import TaskState
                task_state = getattr(t, "state", None)
                is_done = task_state == TaskState.DONE or str(task_state).lower() in ("done", "completed")
                bullet = "[value.low]✓[/]" if is_done else "[dim]○[/]"
                title = str(t.title)[:60]
                console.print(f"    {bullet} {title}")
            if len(linked_tasks) > 5:
                console.print(f"    [dim]... and {len(linked_tasks) - 5} more[/dim]")
        else:
            console.print("    [dim]No tasks linked to this goal.[/dim]")

        # Sessions linked through tasks
        task_ids = {t.id for t in linked_tasks}
        all_sessions = []
        for t_id in task_ids:
            try:
                all_sessions.extend(repo.list_sessions_for_task(t_id))
            except Exception:
                pass

        console.print()
        console.print("  [subheader]Sessions[/subheader]")
        if all_sessions:
            total_minutes = 0
            for s in all_sessions:
                if s.end_at and s.start_at:
                    try:
                        diff = (s.end_at - s.start_at).total_seconds() / 60
                        total_minutes += int(diff)
                    except Exception:
                        pass
            hours = total_minutes // 60
            mins = total_minutes % 60
            time_str = f"{hours}h {mins:02d}m" if hours else f"{mins}m"
            console.print(f"    [dim]{len(all_sessions)} sessions · {time_str} total[/dim]")
        else:
            console.print("    [dim]No sessions recorded.[/dim]")

        # Domain note stage distribution
        console.print()
        inferred_domain = _infer_domain_from_goal(goal, vault)
        stage_counts = _count_goal_domain_stages(vault, inferred_domain) if inferred_domain else {}
        domain_label = f" ({inferred_domain} domain)" if inferred_domain else ""
        console.print(f"  [subheader]Notes{domain_label}[/subheader]")
        if stage_counts:
            total_notes = sum(stage_counts.values())
            new_c = stage_counts.get("new", 0)
            learning_c = stage_counts.get("learning", 0)
            learnt_c = stage_counts.get("learnt", 0)
            stale_c = stage_counts.get("stale", 0)
            console.print(f"    [dim]{total_notes} notes total[/dim]")
            console.print(f"    [dim]#new: {new_c}  #learning: {learning_c}  #learnt: {learnt_c}  #stale: {stale_c}[/dim]")
        else:
            if inferred_domain:
                console.print(f"    [dim]No notes found in {inferred_domain} domain.[/dim]")
            else:
                console.print("    [dim]Domain not matched — no note data.[/dim]")

        console.rule()
