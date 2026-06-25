# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Helpers for materializing learning plan blocks into tasks and time blocks."""

from __future__ import annotations

from pb.core.enums import EnergyType, TaskState
from pb.core.learning_metadata import build_learning_task_description, parse_learning_task_metadata
from pb.core.models import Task, utc_now
from pb.core.naming import apply_generated_title, deterministic_names, stored_display_title
from pb.core.session_blueprints import blueprint_payload, hydrate_learning_block_blueprint
from pb.llm.drafts import LearningPlanBlockDraft


def learning_task_title(block: LearningPlanBlockDraft) -> str:
    """Return the canonical task title for a study or practise block."""
    if block.branch == "practise":
        return f"Practise: {block.subject_scope}"
    if (block.study_mode or "").strip().lower() == "socratic_teach":
        return f"Teach: {block.subject_scope}"
    return f"Study: {block.subject_scope}"


def infer_learning_duration_minutes(
    branch: str,
    subject_scope: str,
    *,
    study_mode: str = "",
) -> int:
    """Return a deterministic fallback duration when the model cannot choose one."""
    lowered = (subject_scope or "").lower()
    mode = (study_mode or "").strip().lower()

    if branch == "practise":
        long_block_keywords = (
            "archery",
            "swim",
            "running",
            "lift",
            "guitar",
            "piano",
            "violin",
            "drums",
            "tennis",
            "basketball",
            "dance",
            "climb",
        )
        if any(token in lowered for token in long_block_keywords):
            return 60
        return 45

    if mode == "socratic_teach":
        return 40
    if any(token in lowered for token in ("foundations", "theory", "proof", "derivation")):
        return 40
    return 35


def materialize_learning_task(repo, block: LearningPlanBlockDraft):
    """Create or reuse an active learning task for the given block."""
    block = hydrate_learning_block_blueprint(block)
    title = learning_task_title(block)
    for task in repo.list_tasks():
        if task.archived_at is not None or task.completion >= 100:
            continue
        meta = parse_learning_task_metadata(task)
        if meta.branch == block.branch and meta.scope == block.subject_scope:
            return task, False

    description = build_learning_task_description(
        block.reason or ("AI-planned learning block." if block.branch == "study" else "AI-planned deliberate-practice block."),
        branch=block.branch,
        scope=block.subject_scope,
        domain=block.subject_scope,
        bloom_target=block.target_bloom_stage.value if block.target_bloom_stage else "",
        study_mode=block.study_mode or "",
        practice_stage=block.practice_stage.value if block.practice_stage else "",
        drill=block.drill_type or "",
        constraint=block.constraint,
        feedback_source=block.feedback_source.value if block.feedback_source else "",
        evidence_target=block.evidence_target,
        success_check=block.success_check,
        cues=block.coach_cues,
        domain_pack_id=block.domain_pack_id,
        session_blueprint=blueprint_payload(block.session_blueprint),
        steps=block.steps,
    )
    task = Task(
        title=title,
        description=description,
        state=TaskState.ACTIVE,
        created_at=utc_now(),
        energy_type=EnergyType.PRACTICE if block.branch == "practise" else EnergyType.DEEP,
        work_type="practice" if block.branch == "practise" else "study",
        linked_goal_arc_ids=[block.goal_id] if block.goal_id else [],
    )
    names = deterministic_names(
        "learning_task",
        block.subject_scope or title,
        {
            "topic": block.subject_scope or title,
            "domain": block.subject_scope or title,
            "branch": block.branch,
            "activity_type": block.branch,
        },
    )
    apply_generated_title(task, names, title_key="task_title")
    repo.create_task(task)
    return task, True


def ensure_time_block(repo, task: Task, block: LearningPlanBlockDraft) -> bool:
    """Ensure a matching time block exists for today."""
    today_blocks = repo.list_time_blocks_created_for_date(utc_now())
    for row in today_blocks:
        if row.task_id == task.id and row.duration_minutes == block.duration_minutes:
            return False

    from pb.core.planner import Planner

    planner = Planner(repo)
    created_block, _ = planner.schedule_block(task, None, block.duration_minutes)
    created_block.block_kind = block.branch
    repo.update_time_block(created_block)
    return True
