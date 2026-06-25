# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interactive teaching flow for new concepts."""

from __future__ import annotations

import sys
from pathlib import Path
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
from pb.cli.llm_guard import print_llm_error, runtime_for_ctx
from pb.cli.pickers import pick_single_choice
from pb.cli.preview import build_step_table, confirm_preview, markdown_step_lines, render_markdown_preview
from pb.cli.topic_group import TopicFallbackGroup
from pb.core.clarifier import (
    ClarifierService,
    ask_clarifier_questions,
    build_clarifier_context,
    clarifier_prompt_block,
    learning_intent_style_guidance,
    persist_clarifier_answers,
)
from pb.core.enums import BloomStage
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
from pb.core.scope_resolution import match_domain_name, match_goal, match_track
from pb.core.staging import build_assumptions, build_learning_context, build_reflection
from pb.llm.drafts import InstructionStep, StudyPlanDraft, artifact_presentation_prompt
from pb.llm.runtime import DraftGenerationError

app = typer.Typer(
    cls=TopicFallbackGroup,
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def _clarifier_answer_block(bundle) -> str:
    return clarifier_prompt_block(bundle)


def launch_teach_session(
    ctx: typer.Context,
    *,
    concept: Optional[str] = None,
    duration: Optional[str] = None,
    stage_hint: Optional[BloomStage] = None,
    yes: bool = False,
    steps: bool = True,
) -> None:
    """Create a tracked teaching session, then start the lesson runtime."""
    from pb.cli.commands.execute import start_task_internal
    from pb.cli.commands.study import _seed_study_block

    repo = ctx.obj["repo"]
    console = get_console()
    auto_yes = bool(yes or ((ctx.obj or {}).get("yes")))
    concept_text = (concept or "").strip()
    if not concept_text:
        raise typer.BadParameter("A concept is required. Try `pb teach linear algebra`.")
    if not resolve_active_session_preflight(
        ctx,
        new_intent=concept_text,
        new_branch="teach",
    ):
        return
    if maybe_start_clarification_plan(
        ctx,
        topic=concept_text,
        preferred_branch="teach",
        yes=auto_yes,
    ):
        return

    runtime = runtime_for_ctx(ctx)
    control_engine = ProductControlEngine(repo=repo, runtime=runtime)
    prepared_context = ctx.obj.get("_prepared_context_scope")
    active_context_scope = getattr(prepared_context, "scope", None)
    matched_goal = match_goal(repo, concept_text, allowed_modes={"mixed", "study"})
    matched_track = match_track(repo, concept_text)
    requested_minutes = parse_duration(duration) if duration else None
    resolved_domain = (
        match_domain_name(concept_text)
        or match_domain_name(getattr(matched_goal, "domain", ""))
        or match_domain_name(getattr(matched_track, "name", ""))
    )
    domain_hint = resolved_domain or getattr(matched_goal, "domain", "") or getattr(matched_track, "name", "") or concept_text
    stage_instruction = (
        f"Stage override: {stage_hint.value}. Use that exact `target_bloom_stage`.\n"
        if stage_hint is not None
        else (
            "Infer `target_bloom_stage` from the user's local context, prior teach/study sessions, and adjacent knowledge. "
            "If they already show fluency in nearby material, step up the complexity appropriately.\n"
        )
    )
    duration_instruction = (
        f"Requested duration minutes: {requested_minutes}. Use that exact `duration_minutes`.\n"
        if requested_minutes is not None
        else (
            "Choose an appropriate `duration_minutes` for a single guided lesson block. "
            "Use the concept complexity, likely familiarity, and the best explanation-to-application span to decide the timebox.\n"
        )
    )
    prompt = (
        "Create a single tracked Socratic teaching block for the learning system.\n"
        "Return exactly one block.\n"
        + learning_intent_style_guidance()
        + f"Concept: {concept_text}\n"
        + f"Domain hint: {domain_hint}\n"
        + f"Goal title: {matched_goal.title if matched_goal else ''}\n"
        + f"{stage_instruction}"
        + f"{duration_instruction}"
        + "Set `study_mode` to `feynman_teach`.\n"
        + "The block should feel like an interactive guided lesson, not a diagnostic.\n"
        + "Do not start by asking the learner to choose expert subtopics or application contexts. "
        + "The first interactive move in teach mode is always the learner explaining what they know; "
        + "use later turns to identify gaps, choose examples, and adjust depth.\n"
    )
    if steps:
        prompt += (
            "Include 4-8 ordered steps in `steps`.\n"
            "Each step must include `title`, `instruction`, and `success_check`.\n"
            "If any step instruction or check contains mathematical TeX/LaTeX, "
            "return it as an object with `text` and `is_latex: true`.\n"
        )
    else:
        prompt += "Leave `steps` as an empty list unless stepwise guidance is explicitly requested.\n"
    prompt += artifact_presentation_prompt()
    runtime_ctx = ctx.obj["runtime"]
    _, control_state = control_engine.load_state(
        scope="artifact",
        artifact_kind="teach_block",
        artifact_id=concept_text,
        goal_id=getattr(matched_goal, "id", "") or "",
    )
    clarifier_bundle = None
    clarifier_answers: dict[str, str] = {}
    # Teach mode is a Feynman-style loop: the learner explains first, then the
    # assistant diagnoses gaps. Pre-start clarifiers made teach feel identical
    # to study and often asked expert subtopic questions too early.
    recorder_clarify_note = {
        "strategy": "defer_clarification_until_after_first_explanation",
    }
    prompt += context_prompt_contract(active_context_scope)
    prompt += feedback_prompt_suffix(runtime_ctx.vault_path, "teach")
    prompt += learner_profile_suffix(repo, runtime_ctx)
    recorder = runtime.make_stage_recorder("teach", concept_text, route_hint="teach")
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("teach", concept_text, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("teach", concept_text, context))
    recorder.add("clarify", recorder_clarify_note)
    if sys.stdin.isatty() and bool(ctx.obj.get("verbose")):
        console.print(f"[dim]{reflection}[/]")

    draft_result = None
    try:
        draft_result = runtime.generate_draft(
            StudyPlanDraft,
            prompt,
            source_scope=f"teach:{concept_text}",
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
        draft = (
            StudyPlanDraft(
                summary="Deterministic teaching block.",
                blocks=[
                    {
                        "branch": "study",
                        "subject_scope": concept_text,
                        "duration_minutes": requested_minutes or infer_learning_duration_minutes(
                            "study",
                            concept_text,
                            study_mode="feynman_teach",
                        ),
                        "target_bloom_stage": (stage_hint or BloomStage.APPLY).value,
                        "study_mode": "feynman_teach",
                        "success_check": f"Explain {concept_text} and answer one transfer question.",
                        "reason": f"Keep guided teaching moving on {concept_text} even without a live model.",
                    }
                ],
            )
            if not runtime.health().available
            else _seed_study_block(topic=concept_text, duration=requested_minutes, level=stage_hint)
        )

    if not draft.blocks:
        recorder.finalize("empty")
        raise typer.BadParameter("No teaching block was generated.")
    block = draft.blocks[0]
    block.branch = "study"
    block.subject_scope = block.subject_scope or concept_text
    block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
    block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
        "study",
        block.subject_scope or concept_text,
        study_mode="feynman_teach",
    )
    block.target_bloom_stage = block.target_bloom_stage or stage_hint or BloomStage.APPLY
    block.study_mode = "feynman_teach"
    if steps and not block.steps:
        block.steps = _fallback_teach_steps(block.subject_scope or concept_text, block.target_bloom_stage or BloomStage.APPLY)
    recorder.add("verify", {"preview": block.model_dump(mode="json")})

    while True:
        preview_sections: list[tuple[str, list[str] | object]] = []
        if block.steps:
            preview_sections.append(("Steps", build_step_table(block.steps, presentation=draft.presentation)))
        render_markdown_preview(
            title="Teach Session Draft",
            rows=[
                ("Title", block.title),
                ("Concept", block.subject_scope),
                ("Planned time", f"{block.duration_minutes} min"),
                ("Mode", block.study_mode),
                ("Success", block.success_check),
            ],
            sections=preview_sections,
        )
        if auto_yes or not sys.stdin.isatty():
            accepted = confirm_preview(yes=auto_yes, action_label="Start this teaching session")
            if not accepted:
                if draft_result is not None:
                    repo.create_generation_provenance(
                        runtime.build_provenance(
                            artifact_kind="teach_block",
                            artifact_id=concept_text,
                            generated_draft=draft_result,
                            accepted_by_user=False,
                        )
                    )
                recorder.finalize("cancelled", artifact_kind="teach_block", artifact_id=concept_text)
                raise typer.Exit(code=0)
            break

        action = pick_single_choice(
            [
                ("start", "Start this teaching session"),
                ("revise", "Revise this teaching block"),
                ("cancel", "Cancel"),
            ],
            title="Teach preview options",
        )
        if action in {None, "cancel"}:
            if draft_result is not None:
                repo.create_generation_provenance(
                    runtime.build_provenance(
                        artifact_kind="teach_block",
                        artifact_id=concept_text,
                        generated_draft=draft_result,
                        accepted_by_user=False,
                    )
                )
            recorder.finalize("cancelled", artifact_kind="teach_block", artifact_id=concept_text)
            raise typer.Exit(code=0)
        if action == "start":
            break

        revision_feedback = collect_revision_feedback(
            engine=control_engine,
            repo=repo,
            runtime_ctx=runtime_ctx,
            mode="teach",
            artifact_kind="teach_block",
            artifact_id=concept_text,
            current_artifact=str(block.model_dump(mode="json")),
            domain=domain_hint,
            target=block.subject_scope or concept_text,
            goal_id=getattr(matched_goal, "id", "") or "",
            title="Revise teaching block",
        )
        if revision_feedback is None:
            continue

        concept_text = block.subject_scope or concept_text
        requested_minutes = block.duration_minutes
        matched_goal = match_goal(repo, concept_text, allowed_modes={"mixed", "study"})
        matched_track = match_track(repo, concept_text)
        resolved_domain = (
            match_domain_name(concept_text)
            or match_domain_name(getattr(matched_goal, "domain", ""))
            or match_domain_name(getattr(matched_track, "name", ""))
        )
        domain_hint = resolved_domain or getattr(matched_goal, "domain", "") or getattr(matched_track, "name", "") or concept_text
        prompt = (
            "Create a single tracked Socratic teaching block for the learning system.\n"
            "Return exactly one block.\n"
            + learning_intent_style_guidance()
            + f"Concept: {concept_text}\n"
            + f"Domain hint: {domain_hint}\n"
            + f"Goal title: {matched_goal.title if matched_goal else ''}\n"
            + f"{stage_instruction}"
            + f"Requested duration minutes: {requested_minutes}. Use that exact `duration_minutes`.\n"
            + "Set `study_mode` to `feynman_teach`.\n"
            + "The block should feel like an interactive guided lesson, not a diagnostic.\n"
            + "Do not ask pre-start expert subtopic clarifiers. The learner should explain first.\n"
            + "Use the existing draft as the starting point and only change what was requested.\n"
            + f"Existing draft JSON: {block.model_dump(mode='json')}\n"
            + f"User revision request: {revision_feedback.free_text}\n"
        )
        if steps:
            prompt += (
                "Include 4-8 ordered steps in `steps`.\n"
                "Each step must include `title`, `instruction`, and `success_check`.\n"
                "If any step instruction or check contains mathematical TeX/LaTeX, "
                "return it as an object with `text` and `is_latex: true`.\n"
            )
        else:
            prompt += "Leave `steps` as an empty list unless stepwise guidance is explicitly requested.\n"
        prompt += artifact_presentation_prompt()
        prompt += _clarifier_answer_block(clarifier_bundle)
        prompt += feedback_prompt_suffix(runtime_ctx.vault_path, "teach")
        prompt += learner_profile_suffix(repo, runtime_ctx)
        prompt += revision_feedback.prompt_suffix
        try:
            draft_result = runtime.generate_draft(
                StudyPlanDraft,
                prompt,
                source_scope=f"teach:{concept_text}",
            )
            draft = draft_result.payload
        except DraftGenerationError as exc:
            console.print(f"[warn]{exc.to_user_message()}[/]")
            draft_result = None
            draft = (
                StudyPlanDraft(
                    summary="Deterministic teaching block.",
                    blocks=[
                        {
                            "branch": "study",
                            "subject_scope": concept_text,
                            "duration_minutes": requested_minutes or infer_learning_duration_minutes(
                                "study",
                                concept_text,
                                study_mode="feynman_teach",
                            ),
                            "target_bloom_stage": (stage_hint or BloomStage.APPLY).value,
                            "study_mode": "feynman_teach",
                            "success_check": f"Explain {concept_text} and answer one transfer question.",
                            "reason": f"Keep guided teaching moving on {concept_text} even without a live model.",
                        }
                    ],
                )
                if not runtime.health().available
                else _seed_study_block(topic=concept_text, duration=requested_minutes, level=stage_hint)
            )
        if not draft.blocks:
            recorder.finalize("empty")
            raise typer.BadParameter("No teaching block was generated.")
        block = draft.blocks[0]
        block.branch = "study"
        block.subject_scope = block.subject_scope or concept_text
        block.goal_id = block.goal_id or (matched_goal.id if matched_goal else None)
        block.duration_minutes = requested_minutes or block.duration_minutes or infer_learning_duration_minutes(
            "study",
            block.subject_scope or concept_text,
            study_mode="feynman_teach",
        )
        block.target_bloom_stage = block.target_bloom_stage or stage_hint or BloomStage.APPLY
        block.study_mode = "feynman_teach"
        if steps and not block.steps:
            block.steps = _fallback_teach_steps(block.subject_scope or concept_text, block.target_bloom_stage or BloomStage.APPLY)

    task_names = NameService(runtime).generate_names(
        "teach_task",
        concept_text,
        {
            "domain": domain_hint,
            "subject": block.subject_scope or concept_text,
            "goal": matched_goal.title if matched_goal else "",
            "activity_type": "teach",
            "study_mode": block.study_mode or "feynman_teach",
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

    start_task_internal(
        ctx,
        task_id=task.id,
        duration=duration or f"{block.duration_minutes}m",
        suggest=False,
        skip_clock=True,
    )
    active_session = repo.get_active_session()
    if active_session is not None and active_session.task_id == task.id:
        apply_generated_names(active_session, task_names)
        attach_active_context(active_session, active_context_scope)
        if clarifier_bundle is not None:
            persist_clarifier_answers(active_session, clarifier_bundle)
        repo.update_session(active_session)

    if draft_result is not None:
        repo.create_generation_provenance(
            runtime.build_provenance(
                artifact_kind="teach_task",
                artifact_id=task.id,
                generated_draft=draft_result,
                accepted_by_user=True,
            )
        )
    recorder.add(
        "materialize",
        {
            "task_id": task.id,
            "subject_scope": block.subject_scope or concept_text,
            "goal_id": matched_goal.id if matched_goal else None,
        },
    )
    recorder.finalize("persisted", artifact_kind="teach_task", artifact_id=task.id)

    if auto_yes or not sys.stdin.isatty():
        console.print("[dim]Teach session is ready. Run `pb finish <what you learned> --skip` when done.[/]")
        return

    console.print("[dim]Interactive lesson starting. Use `finish` when the lesson is ready to close.[/]")
    if sys.stdin.isatty():
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
                    branch="teach",
                    objective=block.success_check or block.reason or concept_text,
                    topic=block.subject_scope or concept_text,
                    domain=resolved_domain or domain_hint,
                    clarifier_answers=clarifier_answers,
                    mode=block.study_mode or "feynman_teach",
                    verbose=bool(ctx.obj.get("verbose")),
                )
            except DraftGenerationError as exc:
                print_llm_error(exc)
                topic = block.subject_scope or concept_text
                console.print("[success]Teach session is ready for manual explain-back.[/]")
                console.print(
                    f"Please explain {topic} in your own words — as if teaching someone who has not seen it before."
                )
                console.print("[dim]When you are done, run `pb finish <what you learned> --skip`.[/]")
                return
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
                # D-16-23: update confidence on teach session end
                from pb.core.confidence_model import (
                    DELTA_TEACH_FULL_COVERAGE, DELTA_WRONG, clamp_score
                )
                from pb.core.graph_writer import make_slug

                _concept_id = f"concept:{(resolved_domain or domain_hint or '').lower()}:{make_slug(block.subject_scope or concept_text)}"
                _detected_gaps = getattr(result, "detected_gaps", []) or []
                _prev_records = repo.list_concept_confidence(_concept_id)
                _prev_score = getattr(_prev_records[0], "confidence_score", 0.0) if _prev_records else 0.0
                if not _detected_gaps:
                    # Full coverage — no gaps detected
                    _new_score = clamp_score(_prev_score + DELTA_TEACH_FULL_COVERAGE)
                else:
                    # Gaps remain — partial/failed teach-back
                    _new_score = clamp_score(_prev_score + DELTA_WRONG)
                repo.upsert_concept_confidence(_concept_id, confidence_score=_new_score)
                from pb.cli.commands.execute import finish_task
                finish_task(ctx, note_words=[result.summary], completion=100, debrief=False, skip=False)
                return
            if result.action == "pause":
                paused = ctx.obj["factory"]["session_service"]().pause_session(outcome=result.summary)
                if paused is not None:
                    console.print(f"[success]Paused: {task.title}[/]")
                return


def _fallback_teach_steps(concept: str, bloom_target: BloomStage) -> list[InstructionStep]:
    """Provide a deterministic teaching skeleton when drafting is unavailable."""
    return [
        InstructionStep(
            title="Anchor the concept",
            instruction=f"What do you already know that connects to {concept}?",
            success_check="Name one prior idea, skill, or example that can anchor the lesson.",
        ),
        InstructionStep(
            title="State the core idea",
            instruction=f"Say what {concept} is in one clear sentence before adding detail.",
            success_check="You can explain the core idea without reading from notes.",
        ),
        InstructionStep(
            title="Walk one example",
            instruction=f"Work through one representative example of {concept}.",
            success_check="You can narrate each step of the example aloud.",
        ),
        InstructionStep(
            title="Apply it",
            instruction=f"Use {concept} at the {bloom_target.value} level on a fresh prompt or variation.",
            success_check="You can solve or explain a nearby variant without copying the example.",
        ),
        InstructionStep(
            title="Check retention",
            instruction="Summarize the idea, the trigger for using it, and the common mistake to avoid.",
            success_check="You can retrieve the concept, use case, and pitfall from memory.",
        ),
    ]


@app.callback(invoke_without_command=True)
def teach_command(
    ctx: typer.Context,
    duration: Optional[str] = typer.Option(None, "--duration", "-d", help="Duration (e.g. 30m, 45m, 1h)"),
    stage: Optional[str] = typer.Option(None, "--stage", help="Optional stage hint; otherwise inferred from context"),
    apply_stage: bool = typer.Option(False, "--apply", "-a", help="Override toward apply/analyze"),
    understand_stage: bool = typer.Option(False, "--understand", "-u", help="Override toward understand"),
    evaluate_stage: bool = typer.Option(False, "--evaluate", "-e", help="Override toward evaluate"),
    create_stage: bool = typer.Option(False, "--create", "-c", help="Override toward create"),
    level: Optional[str] = typer.Option(None, "--level", "-l", hidden=True),
    steps: bool = typer.Option(True, "--steps/--no-steps", help="Include a stepwise guided lesson plan"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Teach a concept through a tracked Socratic lesson."""
    if ctx.invoked_subcommand is not None:
        return
    parsed_args = parse_context_argv(ctx.args)
    prepared_context = prepare_context_scope(
        ctx,
        [Path(token).expanduser() for token in parsed_args.context_tokens],
    )
    raise_for_blocking_context(prepared_context)
    from pb.cli.commands.study import resolve_stage_override

    stage_hint = resolve_stage_override(
        stage=stage,
        legacy_level=level,
        apply_stage=apply_stage,
        understand_stage=understand_stage,
        evaluate_stage=evaluate_stage,
        create_stage=create_stage,
    )
    launch_teach_session(
        ctx,
        concept=" ".join(parsed_args.topic_tokens).strip() or None,
        duration=duration,
        stage_hint=stage_hint,
        steps=steps,
        yes=yes,
    )
