# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared broad-topic clarification flow for multi-step learning plans."""

from __future__ import annotations

import re
import os
import sys

import typer

from pb.cli.active_session import resolve_active_session_preflight
from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console
from pb.cli.llm_guard import llm_requirement_message, runtime_for_ctx
from pb.cli.preview import confirm_preview, markdown_learning_plan_lines, render_markdown_preview
from pb.core.action_routing import route_learning_intent
from pb.core.clarifier import (
    ClarifierService,
    ask_clarifier_questions,
    build_clarifier_context,
    clarifier_prompt_block,
    learning_intent_style_guidance,
    persist_clarifier_answers,
)
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.core.learning_block_flow import learner_profile_suffix
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.learning_partner import LearningPartnerSession
from pb.core.learning_curriculum import (
    curriculum_roots,
    fallback_curriculum_plan,
    materialize_curriculum_plan,
    needs_curriculum_clarification,
    write_curriculum_note,
)
from pb.core.product_control import ProductControlEngine
from pb.core.scope_resolution import match_goal, match_track
from pb.core.staging import build_assumptions, build_learning_context, build_reflection
from pb.llm.drafts import CurriculumPlanDraft, GoalDraft, artifact_presentation_prompt
from pb.llm.runtime import DraftGenerationError


_LEARNING_INTENT_RE = re.compile(
    r"\b(learn|study|practise|practice|master|understand|internali[sz]e|apply|teach|proof|equation|theory)\b",
    re.IGNORECASE,
)
_LEARNING_DOMAIN_RE = re.compile(
    r"\b(algebra|calculus|geometry|physics|ricci|tensor|manifold|integral|equations?|proofs?|grammar|vocab|language)\b",
    re.IGNORECASE,
)


def _preferred_branch_for_request(repo, topic: str) -> str:
    lowered = " ".join((topic or "").lower().split())
    if lowered.startswith("study "):
        return "study"
    if lowered.startswith("teach "):
        return "teach"
    if lowered.startswith("practise ") or lowered.startswith("practice "):
        return "practise"
    if lowered.startswith("learn "):
        return "mixed"
    decision = route_learning_intent(repo, topic)
    return decision.branch if decision.confidence >= 0.18 else "mixed"


def _looks_like_learning_request(topic: str) -> bool:
    normalized = " ".join((topic or "").split())
    if not normalized:
        return False
    if _LEARNING_INTENT_RE.search(normalized):
        return True
    return bool(_LEARNING_DOMAIN_RE.search(normalized))


def _maybe_create_goal(ctx: typer.Context, *, topic: str, yes: bool):
    from pb.cli.commands.goals import _create_goal_via_llm

    goal = _create_goal_via_llm(ctx, topic, yes=yes)
    if goal is None:
        raise typer.Exit(code=0)
    return goal


def _build_curriculum_prompt(
    *,
    topic: str,
    preferred_branch: str,
    matched_goal,
    matched_track,
    clarifier_bundle,
    clarifications: dict[str, str],
    feedback_suffix: str,
) -> str:
    clarification_lines = "".join(
        f"- {question}: {answer}\n"
        for question, answer in clarifications.items()
        if answer.strip()
    )
    return (
        "You are building a clarified multi-step learning plan for a CLI-first learning system.\n"
        "The plan may mix study and deliberate practice, but every block must be directly executable.\n"
        "Return a dependency-aware curriculum with 2-6 blocks.\n"
        "Each block must use a stable `node_id` and list prerequisite node ids in `depends_on`.\n"
        "Prefer exactly one root block when possible.\n"
        "Keep block scopes concrete and evidence-driven.\n"
        "`subject_scope` must specify the exact concepts, proofs, drills, or capabilities to cover, not just paraphrase the topic.\n"
        "For advanced or long-horizon goals, distinguish prerequisite progress from target progress.\n"
        "If prior evidence is weak or absent, assume the safer lower starting point and materialize prerequisite blocks first.\n"
        "Do not pretend the learner can jump straight to the terminal topic unless the context clearly proves the prerequisites are already solid.\n"
        + learning_intent_style_guidance()
        + f"Topic: {topic}\n"
        + f"Preferred branch: {preferred_branch}\n"
        + f"Matched goal: {matched_goal.title if matched_goal else ''}\n"
        + f"Matched track: {matched_track.name if matched_track else ''}\n"
        + clarifier_prompt_block(clarifier_bundle)
        + (f"Clarifications:\n{clarification_lines}" if clarification_lines else "")
        + f"{feedback_suffix}"
        + "Use Bloom targets for study blocks and practice stages for practice blocks.\n"
        + f"{artifact_presentation_prompt(include_dependency_layout=True)}"
    )


def _persist_lightweight_goal(
    repo,
    runtime,
    *,
    topic: str,
    draft: CurriculumPlanDraft,
    matched_track,
    allow_model_names: bool = True,
) -> object:
    from pb.cli.commands.goals import _default_goal_fields, _extract_domain, _persist_goal
    from pb.core.naming import NameService, deterministic_names

    normalized_topic = " ".join((topic or "").split()) or "Learning goal"
    branches = {
        ("practise" if (block.branch or "").strip().lower() == "practice" else (block.branch or "").strip().lower())
        for block in draft.blocks
        if (block.branch or "").strip()
    }
    if branches == {"study"}:
        execution_mode = "study"
    elif branches == {"practise"}:
        execution_mode = "practise"
    else:
        execution_mode = "mixed"

    final_check = next(
        (
            (block.success_check or "").strip()
            for block in reversed(draft.blocks)
            if (block.success_check or "").strip()
        ),
        "",
    )
    domain = (getattr(matched_track, "name", "") or _extract_domain(normalized_topic) or normalized_topic).strip()
    goal_draft = _default_goal_fields(
        GoalDraft(
            title=normalized_topic.title(),
            description=draft.summary.strip() or f"Turn {normalized_topic} into a concrete learning loop.",
            domain=domain,
            execution_mode=execution_mode,
            horizon="quarter",
            framework="Bloom-first learning loop",
            success_definition=final_check or draft.summary.strip() or f"Make concrete progress in {normalized_topic}.",
        )
    )
    naming_context = {
        "domain": goal_draft.domain,
        "subject": goal_draft.domain or goal_draft.title,
        "activity_type": "goal",
        "execution_mode": goal_draft.execution_mode,
        "success_definition": goal_draft.success_definition,
    }
    goal_names = (
        NameService(runtime).generate_names("goal", normalized_topic, naming_context)
        if allow_model_names
        else deterministic_names("goal", normalized_topic, naming_context)
    )
    return _persist_goal(
        repo,
        goal_draft,
        track_name=getattr(matched_track, "name", "") or "",
        goal_names=goal_names,
    )


def _generate_curriculum_plan(
    runtime,
    *,
    prompt: str,
    source_scope: str,
) -> tuple[CurriculumPlanDraft, dict[str, object]]:
    generated = runtime.generate_draft(
        CurriculumPlanDraft,
        prompt,
        source_scope=source_scope,
        model=runtime.config.model_roles.fast_inference or runtime.config.model_roles.default,
        max_output_tokens=30000,
    )
    return generated.payload, {
        "model": generated.model,
        "attempts": [
            {
                "model": f"{attempt.provider}:{attempt.model}",
                "status": attempt.status,
                "error": attempt.raw_message,
            }
            for attempt in generated.attempts
        ],
        "source_scope": source_scope,
        "raw_response": generated.raw_response,
    }


def _warn_and_fallback_curriculum(
    *,
    workflow: str,
    topic: str,
    preferred_branch: str,
    detail: str,
) -> CurriculumPlanDraft:
    console = get_console()
    console.print(
        "[warn]"
        + llm_requirement_message(
            workflow,
            detail=detail,
            fallback_available=True,
        )
        + "[/]"
    )
    console.print("[dim]Using a local fallback learning plan so you can keep moving.[/]")
    return fallback_curriculum_plan(topic, preferred_branch=preferred_branch)


def maybe_start_clarification_plan(
    ctx: typer.Context,
    *,
    topic: str,
    preferred_branch: str,
    yes: bool,
) -> bool:
    """Start a clarification-driven plan when the topic is broad enough."""
    if not needs_curriculum_clarification(topic, preferred_branch=preferred_branch):
        return False
    if not resolve_active_session_preflight(
        ctx,
        new_intent=topic,
        new_branch=preferred_branch,
    ):
        return True
    _launch_clarification_plan(
        ctx,
        topic=topic,
        preferred_branch=preferred_branch,
        yes=yes,
    )
    return True


def _launch_clarification_plan(
    ctx: typer.Context,
    *,
    topic: str,
    preferred_branch: str,
    yes: bool,
) -> None:
    from pb.cli.commands.execute import start_task_internal

    repo = ctx.obj["repo"]
    runtime = runtime_for_ctx(ctx)
    runtime_ctx = ctx.obj["runtime"]
    console = get_console()
    matched_goal = match_goal(repo, topic)
    matched_track = match_track(repo, topic)
    control_engine = ProductControlEngine(repo=repo, runtime=runtime)
    _, control_state = control_engine.load_state(
        scope="artifact",
        artifact_kind="curriculum_plan",
        artifact_id=topic,
        goal_id=getattr(matched_goal, "id", "") or "",
    )

    recorder = runtime.make_stage_recorder("clarify", topic, route_hint=preferred_branch)
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("learn", topic, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("learn", topic, context))
    if bool(ctx.obj.get("verbose")):
        console.print(f"[dim]{reflection}[/]")
    probe = runtime.live_probe()
    recorder.add(
        "probe",
        {
            "provider": probe.provider,
            "model": probe.model,
            "category": probe.category,
            "message": probe.message,
        },
        status="ok" if probe.available else "error",
    )

    clarifier_bundle = None
    clarifications: dict[str, str] = {}
    preferred_mix = preferred_branch if preferred_branch in {"study", "practise", "mixed"} else "mixed"
    draft_result = None

    if probe.available:
        clarifier_context = build_clarifier_context(
            repo,
            runtime_ctx,
            raw_request=topic,
            scope="learn",
            mode=preferred_branch,
            domain=getattr(matched_goal, "domain", "") or getattr(matched_track, "name", "") or topic,
            control_state=control_state,
        )
        questions = ClarifierService(runtime).generate_questions(
            topic,
            clarifier_context,
            max_questions=3,
            scope="learn",
            control_state=control_state,
        )
        clarifier_bundle = ask_clarifier_questions(questions) if questions and sys.stdin.isatty() else None
        clarifications = clarifier_bundle.answers if clarifier_bundle is not None else {}
        preferred_mix = (
            next(
                (
                    answer.strip().lower()
                    for answer in clarifications.values()
                    if answer.strip().lower() in {"study", "practise", "practice", "mixed"}
                ),
                preferred_mix,
            )
            or "mixed"
        )
        recorder.add("clarify", clarifications)

        prompt = _build_curriculum_prompt(
            topic=topic,
            preferred_branch=preferred_branch,
            matched_goal=matched_goal,
            matched_track=matched_track,
            clarifier_bundle=clarifier_bundle,
            clarifications=clarifications,
            feedback_suffix=feedback_prompt_suffix(runtime_ctx.vault_path, "learn") + learner_profile_suffix(repo, runtime_ctx),
        )

        try:
            draft, draft_result = _generate_curriculum_plan(
                runtime,
                prompt=prompt,
                source_scope=f"clarify:{preferred_branch}:{topic}",
            )
            recorder.add(
                "draft",
                {
                    "model": draft_result["model"],
                    "attempts": draft_result["attempts"],
                },
            )
        except DraftGenerationError as exc:
            recorder.add(
                "draft",
                {
                    "error": exc.to_user_message(),
                },
                status="error",
            )
            draft = _warn_and_fallback_curriculum(
                workflow="clarified learning plan",
                topic=topic,
                preferred_branch=preferred_mix,
                detail=exc.to_user_message(),
            )
            recorder.add("fallback", {"source": "draft_error", "preferred_branch": preferred_mix})
    else:
        draft = _warn_and_fallback_curriculum(
            workflow="clarified learning plan",
            topic=topic,
            preferred_branch=preferred_mix,
            detail=probe.message,
        )
        recorder.add("fallback", {"source": "probe", "preferred_branch": preferred_mix})

    if not draft.blocks:
        recorder.finalize("empty")
        raise typer.BadParameter("No clarified learning plan was generated.")

    sections = [
        (
            "Plan",
            markdown_learning_plan_lines(draft.blocks, presentation=draft.presentation),
        )
    ]
    render_markdown_preview(
        title="Clarified Learning Plan",
        rows=[
            ("Topic", topic),
            ("Learner state", draft.learner_state),
        ],
        sections=sections,
    )
    if not confirm_preview(yes=yes, action_label="Create this learning plan"):
        recorder.finalize("cancelled")
        raise typer.Exit(code=0)

    if matched_goal is None:
        matched_goal = _persist_lightweight_goal(
            repo,
            runtime,
            topic=topic,
            draft=draft,
            matched_track=matched_track,
            allow_model_names=probe.available,
        )

    plan_id, tasks = materialize_curriculum_plan(
        repo,
        draft,
        topic=topic,
        goal_id=matched_goal.id if matched_goal else None,
        track_id=matched_track.id if matched_track else None,
    )
    if clarifier_bundle is not None and tasks:
        persist_clarifier_answers(tasks[0], clarifier_bundle)
        repo.update_task(tasks[0])
    note_path = write_curriculum_note(
        runtime_ctx.vault_path,
        plan_id=plan_id,
        topic=topic,
        summary=draft.summary,
        learner_state=draft.learner_state,
        clarifications=clarifications,
        tasks=tasks,
    )
    roots = curriculum_roots(tasks)
    recorder.add(
        "materialize",
        {
            "plan_id": plan_id,
            "task_ids": [task.id for task in tasks],
            "root_ids": [task.id for task in roots],
        },
    )
    recorder.finalize("persisted", artifact_kind="learning_plan", artifact_id=plan_id)

    console.print(f"[success]Learning plan saved:[/] {note_path.relative_to(runtime_ctx.vault_path)}")
    if not roots:
        console.print("[warn]No root task was created. Review the generated plan note.[/]")
        raise typer.Exit(code=0)

    first_task = roots[0]
    start_task_internal(
        ctx,
        task_id=first_task.id,
        duration=None,
        suggest=False,
        skip_clock=True,
    )
    active_session = repo.get_active_session()
    if clarifier_bundle is not None and active_session is not None and active_session.task_id == first_task.id:
        persist_clarifier_answers(active_session, clarifier_bundle)
        repo.update_session(active_session)
    if sys.stdin.isatty() and os.environ.get("PB_IN_SHELL") != "1":
        while True:
            active_session = repo.get_active_session()
            if active_session is None or active_session.task_id != first_task.id:
                break
            meta = parse_learning_task_metadata(first_task)
            partner = LearningPartnerSession(
                runtime=runtime,
                runtime_ctx=runtime_ctx,
                repo=repo,
                task=first_task,
                session=active_session,
                branch=meta.branch or "study",
                objective=meta.success_check or draft.summary or topic,
                topic=meta.scope or topic,
                domain=meta.domain or topic,
                clarifier_answers=clarifications,
                mode=meta.study_mode or meta.practice_stage or preferred_mix,
                verbose=bool(ctx.obj.get("verbose")),
            )
            result = partner.start()
            if result.note_path is not None:
                console.print(f"[dim]Partner note:[/] {result.note_path.relative_to(runtime_ctx.vault_path)}")
            if result.action == "command" and result.command:
                run_internal_command(ctx, result.command)
                active_session = repo.get_active_session()
                if active_session is not None and active_session.task_id == first_task.id:
                    continue
                return
            if result.action == "finish":
                from pb.cli.commands.execute import finish_task

                finish_task(ctx, note_words=[result.summary], completion=100, debrief=False, skip=False)
                return
            if result.action == "pause":
                paused = ctx.obj["factory"]["session_service"]().pause_session(outcome=result.summary)
                if paused is not None:
                    console.print(f"[success]Paused: {first_task.title}[/]")


def maybe_expand_learning_todo(ctx: typer.Context, *, task) -> bool:
    """Expand a broad learning todo into a dependency-aware curriculum before starting it."""
    if (getattr(task, "work_type", "") or "").lower() != "todo":
        return False

    topic = (getattr(task, "description", "") or task.title or "").strip()
    if not _looks_like_learning_request(topic):
        return False

    repo = ctx.obj["repo"]
    preferred_branch = _preferred_branch_for_request(repo, topic)
    if not needs_curriculum_clarification(topic, preferred_branch=preferred_branch):
        return False

    runtime = runtime_for_ctx(ctx)
    runtime_ctx = ctx.obj["runtime"]
    console = get_console()
    matched_goal = match_goal(repo, topic)
    matched_track = match_track(repo, topic)

    recorder = runtime.make_stage_recorder("todo_expand", topic, route_hint=preferred_branch)
    context = build_learning_context(repo, runtime_ctx)
    recorder.add("prepare", context)
    recorder.add("reflect", build_reflection("learn", topic, context))
    recorder.add("assume", build_assumptions("learn", topic, context))

    probe = runtime.live_probe()
    recorder.add(
        "probe",
        {
            "provider": probe.provider,
            "model": probe.model,
            "category": probe.category,
            "message": probe.message,
        },
        status="ok" if probe.available else "error",
    )

    if probe.available:
        if matched_goal is None and matched_track is None and (sys.stdin.isatty() or sys.stdout.isatty()):
            matched_goal = _maybe_create_goal(ctx, topic=topic, yes=False)
        prompt = _build_curriculum_prompt(
            topic=topic,
            preferred_branch=preferred_branch,
            matched_goal=matched_goal,
            matched_track=matched_track,
            clarifier_bundle=None,
            clarifications={},
            feedback_suffix=feedback_prompt_suffix(runtime_ctx.vault_path, "learn") + learner_profile_suffix(repo, runtime_ctx),
        )
        try:
            draft, metadata = _generate_curriculum_plan(
                runtime,
                prompt=prompt,
                source_scope=f"todo_expand:{preferred_branch}:{topic}",
            )
            recorder.add(
                "draft",
                {
                    "model": metadata["model"],
                    "attempts": metadata["attempts"],
                },
            )
        except DraftGenerationError as exc:
            recorder.add("draft", {"error": exc.to_user_message()}, status="error")
            draft = _warn_and_fallback_curriculum(
                workflow="broad todo expansion",
                topic=topic,
                preferred_branch=preferred_branch,
                detail=exc.to_user_message(),
            )
            recorder.add("fallback", {"source": "draft_error", "preferred_branch": preferred_branch})
    else:
        draft = _warn_and_fallback_curriculum(
            workflow="broad todo expansion",
            topic=topic,
            preferred_branch=preferred_branch,
            detail=probe.message,
        )
        recorder.add("fallback", {"source": "probe", "preferred_branch": preferred_branch})

    plan_id, tasks = materialize_curriculum_plan(
        repo,
        draft,
        topic=topic,
        goal_id=matched_goal.id if matched_goal else None,
        track_id=matched_track.id if matched_track else None,
    )
    note_path = write_curriculum_note(
        runtime_ctx.vault_path,
        plan_id=plan_id,
        topic=topic,
        summary=draft.summary,
        learner_state=draft.learner_state,
        clarifications={},
        tasks=tasks,
    )
    roots = curriculum_roots(tasks)

    expansion_summary = (
        f"Expanded into learning plan {plan_id} with tasks: "
        + ", ".join(created.title for created in tasks[:4])
    )
    task.description = "\n\n".join(
        part
        for part in [task.description.strip(), expansion_summary]
        if part
    )
    repo.update_task(task)
    repo.archive_task(task.id)

    recorder.add(
        "materialize",
        {
            "source_task_id": task.id,
            "plan_id": plan_id,
            "task_ids": [created.id for created in tasks],
            "root_ids": [root.id for root in roots],
        },
    )
    recorder.finalize("persisted", artifact_kind="learning_plan", artifact_id=plan_id)

    console.print(f"[success]Expanded todo into learning plan:[/] {note_path.relative_to(runtime_ctx.vault_path)}")
    if not roots:
        console.print("[warn]No root task was created. Review the generated plan note.[/]")
        return True

    from pb.cli.commands.execute import start_task_internal

    first_root = roots[0]
    start_task_internal(
        ctx,
        task_id=first_root.id,
        duration=None,
        suggest=False,
        skip_clock=True,
    )
    return True
