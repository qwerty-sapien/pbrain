# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Practise commands for deliberate practice and drill work."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from typing import Optional

import typer

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
from pb.cli.pickers import pick_single_choice
from pb.cli.preview import build_step_table, confirm_preview, markdown_step_lines, render_markdown_preview
from pb.cli.normalize import join_words_safe
from pb.cli.topic_group import TopicFallbackGroup
from pb.core.clarifier import (
    ClarifierService,
    ask_clarifier_questions,
    build_clarifier_context,
    clarifier_prompt_block,
    learning_intent_style_guidance,
    persist_clarifier_answers,
)
from pb.core.enums import SessionMode
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.core.learning_block_flow import collect_revision_feedback, learner_profile_suffix
from pb.core.learning_partner import LearningPartnerSession
from pb.core.learning_tasks import infer_learning_duration_minutes, materialize_learning_task
from pb.core.naming import (
    NameService,
    apply_generated_names,
    apply_generated_title,
)
from pb.core.product_control import ProductControlEngine
from pb.core.session_blueprints import pack_display_label, resolve_learning_session_blueprint
from pb.core.scope_resolution import match_goal as resolved_goal, match_track as resolved_track
from pb.core.staging import build_assumptions, build_learning_context, build_reflection
from pb.llm.drafts import PractisePlanDraft, artifact_presentation_prompt
from pb.llm.runtime import DraftGenerationError

app = typer.Typer(
    cls=TopicFallbackGroup,
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def _is_interactive() -> bool:
    return sys.stdin.isatty()


def _resolve_practise_block_blueprint(*, block, runtime_ctx, domain_hint: str, skill_text: str) -> None:
    """Resolve a durable practise blueprint, optionally swapping to a nearby pack."""

    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain=domain_hint,
        topic=block.subject_scope or skill_text,
        drill=block.drill_type or block.constraint or "",
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
        text="No precise session pack matched this practice block. Keep the new custom blueprint or reuse a nearby one.",
    )
    if not choice or choice == resolution.pack_id:
        return

    selected = resolve_learning_session_blueprint(
        branch="practise",
        domain=domain_hint,
        topic=block.subject_scope or skill_text,
        drill=block.drill_type or block.constraint or "",
        domain_pack_id=choice,
        vault_path=runtime_ctx.vault_path,
        allow_custom_init=False,
    )
    block.domain_pack_id = selected.pack_id
    block.session_blueprint = selected.blueprint


def _match_goal(repo, skill: str):
    return resolved_goal(repo, skill, allowed_modes={"mixed", "practise", "practice"})


def _match_track(repo, skill: str):
    return resolved_track(repo, skill)


def _recent_practise_scopes(repo, limit: int = 5) -> list[str]:
    rows: list[tuple[datetime, str]] = []
    for task in repo.list_tasks():
        for session in repo.list_sessions_for_task(task.id):
            if session.branch == "practise" and session.subject_scope:
                rows.append((session.start_at, session.subject_scope))
    rows.sort(key=lambda item: item[0], reverse=True)
    seen: list[str] = []
    for _, scope in rows:
        if scope not in seen:
            seen.append(scope)
        if len(seen) >= limit:
            break
    return seen


def _collect_practise_targets(repo) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(value: str) -> None:
        clean = value.strip()
        if clean and clean not in seen:
            seen.add(clean)
            candidates.append(clean)

    active_session = repo.get_active_session()
    if active_session is not None and getattr(active_session, "branch", "") == "practise":
        add(getattr(active_session, "subject_scope", ""))

    for goal in repo.list_goal_arcs(status=None):
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        if mode in {"mixed", "practise", "practice"}:
            add(goal.domain or goal.title)

    for track in repo.list_tracks(active_only=True):
        add(track.name)

    for scope in _recent_practise_scopes(repo):
        add(scope)

    return candidates


def _pick_practise_target(repo) -> Optional[str]:
    from pb.cli.pickers import pick_single_choice

    choices = _collect_practise_targets(repo)
    if choices:
        selected = pick_single_choice(
            [(choice, choice) for choice in choices],
            title="Select practice target",
        )
        if selected:
            return selected
    if not _is_interactive():
        return None
    return typer.prompt("Practice target", default="", show_default=False).strip() or None


def _set_practise_session_metadata(repo, task_id: str, skill: str, drill: str | None, cues: str | None) -> None:
    active_session = repo.get_active_session()
    if active_session is None or active_session.task_id != task_id:
        return
    active_session.branch = "practise"
    active_session.subject_scope = skill
    active_session.drill_type = drill
    active_session.coach_cues = cues
    active_session.mode = SessionMode.PRACTICE
    repo.update_session(active_session)


def _manual_practise_block(
    *,
    skill: str,
    duration: Optional[int],
    drill: str | None,
    cues: str | None,
) -> PractisePlanDraft:
    """Collect a viable deliberate-practice block without an LLM."""

    typer.echo("Manual practise setup")
    typer.echo("Tip: choose one drill, one feedback source, and one clear success check.")
    scope = prompt_text("Practice target", default=skill or "")
    if not scope.strip():
        raise typer.BadParameter("A practice target is required.")
    fallback_duration = duration or infer_learning_duration_minutes("practise", scope)
    drill_value = prompt_text("Drill type", default=drill or scope).strip() or drill or scope
    constraint = prompt_text("Constraint", default="Slow down before speeding up.").strip() or "Slow down before speeding up."
    evidence = prompt_text("Evidence target", default=f"One clean rep set for {scope}.").strip() or f"One clean rep set for {scope}."
    success = prompt_text("Success check", default=f"Finish one deliberate-practice block for {scope}.").strip() or f"Finish one deliberate-practice block for {scope}."
    cue_text = prompt_text("Coach cues", default=cues or "").strip() or cues or ""
    return PractisePlanDraft(
        summary="Manual deliberate-practice block.",
        blocks=[
            {
                "branch": "practise",
                "subject_scope": scope,
                "duration_minutes": fallback_duration,
                "practice_stage": "integrate",
                "drill_type": drill_value,
                "constraint": constraint,
                "feedback_source": "artifact",
                "evidence_target": evidence,
                "coach_cues": cue_text,
                "success_check": success,
                "reason": f"Manual deliberate-practice block for {scope}.",
            }
        ],
    )


def _seed_practise_block(
    *,
    skill: str,
    duration: Optional[int],
    drill: str | None,
    cues: str | None,
) -> PractisePlanDraft:
    scope = skill.strip() or "practice target"
    drill_value = drill or scope
    return PractisePlanDraft(
        summary="Deterministic deliberate-practice block.",
        blocks=[
            {
                "branch": "practise",
                "subject_scope": scope,
                "duration_minutes": duration or infer_learning_duration_minutes("practise", scope),
                "practice_stage": "integrate",
                "drill_type": drill_value,
                "constraint": "Slow down before speeding up.",
                "feedback_source": "artifact",
                "evidence_target": f"One clean rep set for {scope}.",
                "coach_cues": cues or "",
                "success_check": f"Finish one deliberate-practice block for {scope}.",
                "reason": f"Keep deliberate practice moving on {scope} even without a live model.",
            }
        ],
    )


def _build_practise_prompt(
    *,
    skill_text: str,
    domain_hint: str,
    matched_goal,
    requested_minutes: Optional[int],
    drill: Optional[str],
    cues: Optional[str],
    steps: bool,
    vault_path,
    revision_note: str = "",
    prior_block: Optional[dict[str, object]] = None,
    clarifier_bundle=None,
    context_contract: str = "",
) -> str:
    duration_instruction = (
        f"Requested duration minutes: {requested_minutes}. Use that exact `duration_minutes`.\n"
        if requested_minutes is not None
        else (
            "Choose an appropriate `duration_minutes` for a single deliberate-practice block. "
            "Use the drill, feedback loop, and likely recovery/attention needs to decide the timebox.\n"
        )
    )
    prompt = (
        "Create a single deliberate-practice block for the learning system.\n"
        "Return exactly one block.\n"
        + learning_intent_style_guidance()
        + f"Skill: {skill_text}\n"
        + f"Domain hint: {domain_hint}\n"
        + f"Goal title: {matched_goal.title if matched_goal else ''}\n"
        + f"{duration_instruction}"
        + f"Requested drill hint: {drill or skill_text}\n"
        + f"Coach cues hint: {cues or ''}\n"
        + "The block must include a practice stage, drill type, constraint, feedback source, evidence target, and success check.\n"
        + "`subject_scope` must identify the exact capability being trained, not a loose restatement of the overall goal.\n"
        + "If the requested goal is too advanced for one useful rep, move down to the nearest prerequisite capability that can actually be drilled today.\n"
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
            "Use the steps to sequence drills, constraints, and checkpoints in the most effective practice order.\n"
            "If any step instruction or check contains LaTeX that should be treated as math, "
            "return it as an object with `text` and `is_latex: true`.\n"
        )
    else:
        prompt += "Leave `steps` as an empty list unless stepwise guidance is explicitly requested.\n"
    prompt += clarifier_prompt_block(clarifier_bundle)
    prompt += artifact_presentation_prompt()
    prompt += feedback_prompt_suffix(vault_path, "practise")
    return prompt


def launch_practise_session(
    ctx: typer.Context,
    *,
    skill: Optional[str] = None,
    duration: Optional[str] = None,
    drill: Optional[str] = None,
    cues: Optional[str] = None,
    yes: bool = False,
    steps: bool = False,
) -> None:
    """Create a practise task, start it, and annotate the session."""
    from pb.cli.commands.execute import start_task_internal

    repo = ctx.obj["repo"]
    console = get_console()
    auto_yes = bool(yes or ((ctx.obj or {}).get("yes")))
    skill_text = (skill or "").strip()
    if not skill_text:
        skill_text = _pick_practise_target(repo) or ""
    if not skill_text:
        raise typer.BadParameter("A practice target is required.")
    if not resolve_active_session_preflight(
        ctx,
        new_intent=skill_text,
        new_branch="practise",
    ):
        return
    if maybe_start_clarification_plan(
        ctx,
        topic=skill_text,
        preferred_branch="practise",
        yes=auto_yes,
    ):
        return

    runtime = runtime_for_ctx(ctx)
    control_engine = ProductControlEngine(repo=repo, runtime=runtime)
    prepared_context = ctx.obj.get("_prepared_context_scope")
    active_context_scope = getattr(prepared_context, "scope", None)
    matched_goal = _match_goal(repo, skill_text)
    matched_track = _match_track(repo, skill_text)
    domain_hint = (
        getattr(matched_goal, "domain", "")
        or getattr(matched_track, "name", "")
        or skill_text
    )

    # D-16-24: soft gate — warn if no study history for this concept
    from pb.core.confidence_model import THRESHOLD_NONE
    from pb.core.graph_writer import make_slug

    _practise_concept_id = f"concept:{domain_hint.lower()}:{make_slug(skill_text)}"
    _practise_records = repo.list_concept_confidence(_practise_concept_id)
    _practise_score = getattr(_practise_records[0], "confidence_score", 0.0) if _practise_records else 0.0
    if _practise_score < THRESHOLD_NONE:
        console.print(
            f"[yellow]No study history for '{skill_text}' (confidence: none). "
            f"Consider `pb study {skill_text}` first.[/yellow]"
        )
        # Soft gate: warn only, do NOT raise typer.Exit

    requested_minutes = parse_duration(duration) if duration else None
    runtime_ctx = ctx.obj["runtime"]
    _, control_state = control_engine.load_state(
        scope="artifact",
        artifact_kind="practise_block",
        artifact_id=skill_text,
        goal_id=getattr(matched_goal, "id", "") or "",
    )
    clarifier_bundle = None
    clarifier_answers: dict[str, str] = {}
    if sys.stdin.isatty() and not auto_yes:
        clarifier_context = build_clarifier_context(
            repo,
            runtime_ctx,
            raw_request=skill_text,
            scope="practise",
            mode="practise",
            domain=domain_hint,
            control_state=control_state,
        )
        questions = ClarifierService(runtime).generate_questions(
            skill_text,
            clarifier_context,
            max_questions=2,
            scope="practise",
            control_state=control_state,
        )
        clarifier_bundle = ask_clarifier_questions(questions) if questions else None
        clarifier_answers = clarifier_bundle.answers if clarifier_bundle is not None else {}
    prompt = _build_practise_prompt(
        skill_text=skill_text,
        domain_hint=domain_hint,
        matched_goal=matched_goal,
        requested_minutes=requested_minutes,
        drill=drill,
        cues=cues,
        steps=steps,
        vault_path=runtime_ctx.vault_path,
        clarifier_bundle=clarifier_bundle,
        context_contract=context_prompt_contract(active_context_scope),
    )
    prompt += learner_profile_suffix(repo, runtime_ctx)
    recorder = runtime.make_stage_recorder("practise", skill_text, route_hint="practise")
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("practise", skill_text, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("practise", skill_text, context))
    recorder.add("clarify", clarifier_answers)
    if sys.stdin.isatty() and bool(ctx.obj.get("verbose")):
        console.print(f"[dim]{reflection}[/]")

    draft_result = None
    try:
        draft_result = runtime.generate_draft(
            PractisePlanDraft,
            prompt,
            source_scope=f"practise:{skill_text}",
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
        console.print(f"[warn]{exc.to_user_message()}[/]")
        draft = _seed_practise_block(skill=skill_text, duration=requested_minutes, drill=drill, cues=cues)

    if not draft.blocks:
        raise typer.BadParameter("No practise block was generated.")
    block = draft.blocks[0]
    block.branch = "practise"
    block.subject_scope = block.subject_scope or skill_text
    block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
    block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
        "practise",
        block.subject_scope or skill_text,
    )
    block.drill_type = block.drill_type or drill or skill_text
    block.coach_cues = block.coach_cues or cues or ""
    _resolve_practise_block_blueprint(
        block=block,
        runtime_ctx=runtime_ctx,
        domain_hint=domain_hint,
        skill_text=skill_text,
    )
    recorder.add("verify", {"preview": block.model_dump(mode="json")})
    resources = None

    def _cancel_preview() -> None:
        if draft_result is not None:
            repo.create_generation_provenance(
                runtime.build_provenance(
                    artifact_kind="practise_block",
                    artifact_id=skill_text,
                    generated_draft=draft_result,
                    accepted_by_user=False,
                )
            )
        recorder.finalize("cancelled", artifact_kind="practise_block", artifact_id=skill_text)
        raise typer.Exit(code=0)

    while True:
        preview_sections: list[tuple[str, list[str] | object]] = []
        if block.steps:
            preview_sections.append(("Steps", build_step_table(block.steps, presentation=draft.presentation)))
        preview_sections.extend(resource_preview_sections(resources))
        render_markdown_preview(
            title="Practise Block Draft",
            rows=[
                ("Title", block.title),
                ("Scope", block.subject_scope),
                ("Planned time", f"{block.duration_minutes} min"),
                ("Drill", block.drill_type),
                ("Constraint", block.constraint),
                ("Success", block.success_check),
            ],
            sections=preview_sections,
        )
        if auto_yes or not sys.stdin.isatty():
            accepted = confirm_preview(yes=auto_yes, action_label="Start this practise block")
            if not accepted:
                _cancel_preview()
            break

        action = choose_learning_block_action("Start this practise block")
        if action in {None, "cancel"}:
            _cancel_preview()
        if action == "start":
            break
        if action == "resources":
            resources = fetch_grounded_learning_resources(
                topic=skill_text,
                branch="practise",
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
            mode="practise",
            artifact_kind="practise_block",
            artifact_id=skill_text,
            current_artifact=json.dumps(block.model_dump(mode="json"), ensure_ascii=True),
            domain=domain_hint,
            target=block.subject_scope or skill_text,
            goal_id=getattr(matched_goal, "id", "") or "",
            title="Revise practise block",
        )
        if revision_feedback is None:
            continue

        revision_note = revision_feedback.free_text

        skill_text = block.subject_scope or skill_text
        drill = block.drill_type or drill or skill_text
        requested_minutes = block.duration_minutes
        matched_goal = _match_goal(repo, skill_text)
        matched_track = _match_track(repo, skill_text)
        domain_hint = (
            getattr(matched_goal, "domain", "")
            or getattr(matched_track, "name", "")
            or skill_text
        )
        prompt = _build_practise_prompt(
            skill_text=skill_text,
            domain_hint=domain_hint,
            matched_goal=matched_goal,
            requested_minutes=requested_minutes,
            drill=drill,
            cues=cues,
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
                PractisePlanDraft,
                prompt,
                source_scope=f"practise:{skill_text}",
            )
            draft = draft_result.payload
            recorder.add(
                "revise",
                {
                    "scope": skill_text,
                    "duration_minutes": requested_minutes,
                    "drill": drill,
                    "note": revision_note,
                    "model": draft_result.model,
                },
            )
        except DraftGenerationError as exc:
            recorder.add(
                "revise",
                {
                    "scope": skill_text,
                    "duration_minutes": requested_minutes,
                    "drill": drill,
                    "note": revision_note,
                    "error": exc.to_user_message(),
                },
                status="error",
            )
            console.print(f"[warn]{exc.to_user_message()}[/]")
            draft_result = None
            draft = _seed_practise_block(skill=skill_text, duration=requested_minutes, drill=drill, cues=cues)

        if not draft.blocks:
            raise typer.BadParameter("No practise block was generated.")
        block = draft.blocks[0]
        block.branch = "practise"
        block.subject_scope = block.subject_scope or skill_text
        block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
        block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
            "practise",
            block.subject_scope or skill_text,
        )
        block.drill_type = block.drill_type or drill or skill_text
        block.coach_cues = block.coach_cues or cues or ""
        _resolve_practise_block_blueprint(
            block=block,
            runtime_ctx=runtime_ctx,
            domain_hint=domain_hint,
            skill_text=skill_text,
        )
        resources = None

    task_names = NameService(runtime).generate_names(
        "practise_task",
        skill_text,
        {
            "domain": domain_hint,
            "subject": block.subject_scope or skill_text,
            "goal": matched_goal.title if matched_goal else "",
            "activity_type": "practise",
            "drill_type": block.drill_type or drill or "",
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
    _set_practise_session_metadata(repo, task.id, block.subject_scope or skill_text, block.drill_type or drill or skill_text, block.coach_cues or cues)

    active_session = repo.get_active_session()
    if active_session is not None and active_session.task_id == task.id:
        active_session.goal_id = matched_goal.id if matched_goal else None
        active_session.track_id = matched_track.id if matched_track else None
        active_session.practice_stage = block.practice_stage
        active_session.constraint = block.constraint or None
        active_session.feedback_source = block.feedback_source
        active_session.evidence_target = block.evidence_target or None
        apply_generated_names(active_session, task_names)
        attach_active_context(active_session, active_context_scope)
        if clarifier_bundle is not None:
            persist_clarifier_answers(active_session, clarifier_bundle)
        repo.update_session(active_session)

    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="practise_task",
                artifact_id=task.id,
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add(
        "materialize",
        {
            "task_id": task.id,
            "subject_scope": block.subject_scope or skill_text,
            "goal_id": matched_goal.id if matched_goal else None,
        },
    )
    recorder.finalize("persisted", artifact_kind="practise_task", artifact_id=task.id)

    goal_label = matched_goal.title if matched_goal else "free practise"
    drill_label = block.drill_type or drill or skill_text
    console.print(
        f"[dim]Practise route:[/] scope `{block.subject_scope or skill_text}` | goal `{goal_label}` | drill `{drill_label}`"
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
                    branch="practise",
                    objective=block.success_check or block.reason or skill_text,
                    topic=block.subject_scope or skill_text,
                    domain=domain_hint,
                    clarifier_answers=clarifier_answers,
                    mode=block.practice_stage.value if block.practice_stage else "practise_coach",
                    verbose=bool(ctx.obj.get("verbose")),
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
                # D-16-27: drill-burst streak management at session end
                from pb.core.confidence_model import THRESHOLD_NONE, BURST_N, clamp_score
                _prac_records = repo.list_concept_confidence(_practise_concept_id)
                if _prac_records and _prac_records[0].burst_active:
                    # Determine if session was successful (detected_gaps empty = correct; else wrong)
                    _detected_gaps = getattr(result, "detected_gaps", []) or []
                    _answer_correct = not _detected_gaps
                    if _answer_correct:
                        _new_streak = _prac_records[0].burst_streak + 1
                        if _new_streak >= BURST_N:  # BURST_N = 3 from confidence_model
                            # Burst complete — restore to partial threshold
                            repo.upsert_concept_confidence(
                                _practise_concept_id,
                                confidence_score=THRESHOLD_NONE,  # restore to 0.3 (partial floor)
                                burst_active=0,
                                burst_streak=0,
                            )
                        else:
                            repo.upsert_concept_confidence(
                                _practise_concept_id,
                                confidence_score=_prac_records[0].confidence_score,
                                burst_streak=_new_streak,
                            )
                    else:
                        # Wrong answer — RESET streak to 0 (not decrement — full reset per D-16-27)
                        repo.upsert_concept_confidence(
                            _practise_concept_id,
                            confidence_score=_prac_records[0].confidence_score,
                            burst_active=1,
                            burst_streak=0,
                        )
                from pb.cli.commands.execute import finish_task

                finish_task(ctx, note_words=[result.summary], completion=100, debrief=False, skip=False)
                return
            if result.action == "pause":
                paused = ctx.obj["factory"]["session_service"]().pause_session(outcome=result.summary)
                if paused is not None:
                    console.print(f"[success]Paused: {task.title}[/]")
                return


@app.callback(invoke_without_command=True)
def practise_callback(
    ctx: typer.Context,
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g. 30m, 45m, 1h)"),
    drill: Optional[str] = typer.Option(None, "--drill", help="Specific drill or rep focus"),
    cues: Optional[str] = typer.Option(None, "--cues", help="Short coach cues"),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise practice sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
) -> None:
    """Start a practise block."""
    if ctx.invoked_subcommand is not None:
        return
    parsed_args = parse_context_argv(ctx.args)
    prepared_context = prepare_context_scope(
        ctx,
        [Path(token).expanduser() for token in parsed_args.context_tokens],
    )
    raise_for_blocking_context(prepared_context)
    launch_practise_session(
        ctx,
        skill=join_words_safe(parsed_args.topic_tokens) or None,
        duration=duration,
        drill=drill,
        cues=cues,
        yes=yes,
        steps=steps,
    )


@app.command("start", hidden=True)
def start_practise(
    ctx: typer.Context,
    skill_words: Optional[list[str]] = typer.Argument(None, help="Skill or practice target"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g. 30m, 45m, 1h)"),
    drill: Optional[str] = typer.Option(None, "--drill", help="Specific drill or rep focus"),
    cues: Optional[str] = typer.Option(None, "--cues", help="Short coach cues"),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise practice sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Compatibility alias for the top-level practise flow."""
    launch_practise_session(
        ctx,
        skill=" ".join(skill_words or []).strip() or None,
        duration=duration,
        drill=drill,
        cues=cues,
        yes=yes,
        steps=steps,
    )


@app.command("resume")
def resume_practise(
    ctx: typer.Context,
    task_id: Optional[str] = typer.Argument(None, help="Task ID to resume"),
):
    """Resume a paused practice task."""
    from pb.cli.commands.execute import resume_task

    resume_task(ctx, task_id=task_id)


@app.command("drill", hidden=True)
def practise_drill(
    ctx: typer.Context,
    drill_words: list[str] = typer.Argument(..., help="Drill focus"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration"),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise practice sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Start a drill-oriented practice session."""
    drill = " ".join(drill_words).strip()
    launch_practise_session(ctx, skill=drill, duration=duration, drill=drill, cues=None, yes=yes, steps=steps)


@app.command("session", hidden=True)
def practise_session(
    ctx: typer.Context,
    skill_words: Optional[list[str]] = typer.Argument(None, help="Skill or practice target"),
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration"),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise practice sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Alias for the top-level practise flow."""
    launch_practise_session(
        ctx,
        skill=" ".join(skill_words or []).strip() or None,
        duration=duration,
        drill=None,
        cues=None,
        yes=yes,
        steps=steps,
    )


@app.command("log", hidden=True)
def practise_log(
    ctx: typer.Context,
    note_words: list[str] = typer.Argument(..., help="Short performance note"),
):
    """Attach a terse note to the active practise session."""
    repo = ctx.obj["repo"]
    console = get_console()
    active_session = repo.get_active_session()
    if active_session is None or getattr(active_session, "branch", "study") != "practise":
        typer.echo("No active practise session.")
        raise typer.Exit(code=1)
    note = " ".join(note_words).strip()
    active_session.actual_outcome = note
    repo.update_session(active_session)
    console.print(f"[success]Practise note saved:[/] {note}")
