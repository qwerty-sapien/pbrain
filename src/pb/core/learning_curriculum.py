# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Helpers for clarified multi-step learning plans."""

from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path

from pb.core.enums import EnergyType, TaskState
from pb.core.learning_metadata import (
    build_learning_task_description,
    parse_learning_task_metadata,
)
from pb.core.session_blueprints import blueprint_payload, hydrate_learning_block_blueprint
from pb.core.learning_tasks import learning_task_title
from pb.core.models import Task, generate_internal_id, utc_now
from pb.llm.drafts import CurriculumPlanDraft, LearningPlanBlockDraft
from pb.vault.lifecycle import write_frontmatter


PLAN_WAITING_REASON = "Awaiting prerequisite task completion."

_BROAD_TOPIC_HINTS = {
    "algebra",
    "biology",
    "calculus",
    "chemistry",
    "chess",
    "economics",
    "german",
    "golang",
    "history",
    "javascript",
    "machine learning",
    "math",
    "physics",
    "piano",
    "python",
    "ricci flow",
    "rust",
    "statistics",
    "vector calculus",
}

_FOCUSED_TOPIC_HINTS = {
    "eigendecomposition",
    "conjugation",
    "listening",
    "speaking",
    "grammar",
    "proofs",
    "eigenvalues",
    "transposition",
    "scales",
    "rhythm",
    "verb",
    "verbs",
}

_PREREQUISITE_HEAVY_HINTS = {
    "ricci flow",
    "general relativity",
    "riemannian manifold",
    "riemannian manifolds",
    "levi-civita connection",
    "vector calculus",
    "maxwell equations",
    "lebesgue integral",
    "lebesgue integrals",
    "measure theory",
}

_LEARNING_VERBS = (
    "learn",
    "study",
    "practise",
    "practice",
    "apply",
    "master",
    "understand",
    "cover",
    "work through",
)


def needs_curriculum_clarification(text: str, *, preferred_branch: str = "") -> bool:
    """Return True when a request is broad enough to justify a clarification plan."""
    normalized = " ".join((text or "").lower().split())
    if not normalized:
        return False
    tokens = normalized.split()
    if any(token in _FOCUSED_TOPIC_HINTS for token in tokens):
        return False
    if any(phrase in normalized for phrase in _PREREQUISITE_HEAVY_HINTS):
        return True
    if len(tokens) <= 2:
        if preferred_branch == "practise":
            return False
        return normalized in _BROAD_TOPIC_HINTS
    if "," in normalized or ";" in normalized:
        return True
    if " and " in normalized and len(tokens) >= 4:
        return True
    verb_hits = sum(1 for verb in _LEARNING_VERBS if verb in normalized)
    if verb_hits >= 2:
        return True
    if re.search(r"\b(?:learn|study|master|understand|apply)\b.+\b(?:and|plus|including|covering)\b", normalized):
        return True
    if re.search(r"\b(?:its|their)\s+application\b", normalized):
        return True
    if any(
        phrase in normalized
        for phrase in (
            "from scratch",
            "get better at",
            "get good at",
            "learn ",
            "master ",
            "understand ",
            "be better at",
        )
    ):
        return True
    if re.search(r"\b[a-c]\d\b", normalized) and any(token in normalized for token in ("month", "months", "week", "weeks")):
        return True
    if len(tokens) <= 4 and normalized in _BROAD_TOPIC_HINTS:
        return True
    if len(tokens) >= 6 and any(marker in normalized for marker in (" and ", " with ", " including ", " plus ")):
        return True
    if preferred_branch == "teach" and normalized in _BROAD_TOPIC_HINTS:
        return True
    return False


def fallback_curriculum_plan(
    topic: str,
    *,
    preferred_branch: str = "mixed",
) -> CurriculumPlanDraft:
    """Build a deterministic small curriculum when structured drafting fails."""
    normalized = " ".join((topic or "").split()) or "this topic"
    branch = "practise" if preferred_branch == "practise" else "study"
    follow_on_branch = "practise" if preferred_branch in {"mixed", "practise"} else "study"
    return CurriculumPlanDraft(
        summary=f"Clarify fundamentals first, then apply them in a concrete block for {normalized}.",
        learner_state="No prior curriculum history was available, so this starts with a fundamentals pass.",
        blocks=[
            LearningPlanBlockDraft(
                node_id="root",
                branch=branch,
                subject_scope=f"{normalized} fundamentals",
                duration_minutes=35,
                target_bloom_stage="understand" if branch == "study" else None,
                study_mode="active recall" if branch == "study" else None,
                practice_stage="integrate" if branch == "practise" else None,
                drill_type=normalized if branch == "practise" else None,
                success_check=f"Explain the core ideas behind {normalized} without rereading.",
                reason=f"Establish a usable baseline before widening the plan for {normalized}.",
            ),
            LearningPlanBlockDraft(
                node_id="apply",
                depends_on=["root"],
                branch=follow_on_branch,
                subject_scope=normalized,
                duration_minutes=40,
                target_bloom_stage="apply" if follow_on_branch == "study" else None,
                study_mode="worked example" if follow_on_branch == "study" else None,
                practice_stage="integrate" if follow_on_branch == "practise" else None,
                drill_type=normalized if follow_on_branch == "practise" else None,
                success_check=f"Produce one concrete attempt that proves progress in {normalized}.",
                reason=f"Convert clarified understanding into deliberate progress on {normalized}.",
            ),
        ],
    )


def materialize_curriculum_plan(
    repo,
    draft: CurriculumPlanDraft,
    *,
    topic: str,
    goal_id: str | None = None,
    track_id: str | None = None,
) -> tuple[str, list[Task]]:
    """Persist a linked curriculum plan as executable tasks."""
    plan_id = generate_internal_id()
    title_seed = " ".join((topic or "").split()) or "Learning plan"
    normalized_title = re.sub(r"\s+", " ", title_seed).strip()
    block_specs: list[tuple[LearningPlanBlockDraft, str]] = []
    task_by_node: dict[str, Task] = {}

    for index, block in enumerate(draft.blocks, start=1):
        block = hydrate_learning_block_blueprint(block)
        node_id = (block.node_id or f"node-{index}").strip() or f"node-{index}"
        block_specs.append((block, node_id))
        initial_state = TaskState.PAUSED if block.depends_on else TaskState.ACTIVE
        description = build_learning_task_description(
            block.reason or f"Curriculum task for {normalized_title}.",
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
            plan_id=plan_id,
            plan_title=normalized_title,
            plan_node_id=node_id,
        )
        task = Task(
            title=block.title.strip() or learning_task_title(block),
            description=description,
            state=initial_state,
            pause_reason=PLAN_WAITING_REASON if initial_state == TaskState.PAUSED else None,
            created_at=utc_now(),
            energy_type=EnergyType.PRACTICE if block.branch == "practise" else EnergyType.DEEP,
            work_type="practice" if block.branch == "practise" else "study",
            linked_goal_arc_ids=[goal_id] if goal_id else [],
            linked_track_ids=[track_id] if track_id else [],
        )
        repo.create_task(task)
        task_by_node[node_id] = task

    created_tasks: list[Task] = []
    for block, node_id in block_specs:
        task = task_by_node[node_id]
        dependency_ids = [
            task_by_node[dependency].id
            for dependency in block.depends_on
            if dependency in task_by_node
        ]
        task.description = build_learning_task_description(
            block.reason or f"Curriculum task for {normalized_title}.",
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
            plan_id=plan_id,
            plan_title=normalized_title,
            plan_node_id=node_id,
            depends_on_task_ids=dependency_ids,
        )
        repo.update_task(task)
        created_tasks.append(task)

    return plan_id, created_tasks


def unlock_ready_curriculum_tasks(repo) -> list[Task]:
    """Activate paused curriculum tasks whose prerequisites are all complete."""
    unlocked: list[Task] = []
    for task in repo.list_tasks(include_archived=True):
        if task.archived_at is not None or task.state != TaskState.PAUSED:
            continue
        if (task.pause_reason or "") != PLAN_WAITING_REASON:
            continue
        meta = parse_learning_task_metadata(task)
        dependencies = meta.depends_on_task_ids or []
        if not dependencies:
            continue
        if not all((repo.get_task(task_id) and repo.get_task(task_id).completion >= 100) for task_id in dependencies):
            continue
        task.state = TaskState.ACTIVE
        task.paused_until = None
        task.pause_reason = None
        repo.update_task(task)
        unlocked.append(task)
    return unlocked


def curriculum_descendants(repo, task_id: str) -> list[Task]:
    """Return dependent curriculum tasks reachable from the given task."""
    descendants: list[Task] = []
    pending = [task_id]
    seen: set[str] = {task_id}
    all_tasks = repo.list_tasks(include_archived=True)
    while pending:
        current = pending.pop()
        for task in all_tasks:
            if task.id in seen:
                continue
            meta = parse_learning_task_metadata(task)
            dependencies = meta.depends_on_task_ids or []
            if current not in dependencies:
                continue
            descendants.append(task)
            pending.append(task.id)
            seen.add(task.id)
    return descendants


def pause_curriculum_descendants(repo, task_id: str) -> list[Task]:
    """Pause downstream curriculum tasks until the prerequisite is complete again."""
    paused: list[Task] = []
    for task in curriculum_descendants(repo, task_id):
        if task.archived_at is not None or task.completion >= 100:
            continue
        task.state = TaskState.PAUSED
        task.paused_until = None
        task.pause_reason = PLAN_WAITING_REASON
        repo.update_task(task)
        paused.append(task)
    return paused


def curriculum_roots(tasks: list[Task]) -> list[Task]:
    """Return the tasks that can start immediately."""
    roots: list[Task] = []
    for task in tasks:
        meta = parse_learning_task_metadata(task)
        if not (meta.depends_on_task_ids or []):
            roots.append(task)
    return roots


def write_curriculum_note(
    vault_path: Path,
    *,
    plan_id: str,
    topic: str,
    summary: str,
    learner_state: str,
    clarifications: dict[str, str],
    tasks: list[Task],
) -> Path:
    """Persist the clarification transcript and curriculum skeleton into the vault."""
    plan_dir = vault_path / "direction" / "plans"
    plan_dir.mkdir(parents=True, exist_ok=True)
    slug = re.sub(r"[^a-z0-9]+", "-", topic.lower()).strip("-") or "learning-plan"
    note_path = plan_dir / f"{slug}-{plan_id[:8]}.md"
    fm = {
        "type": "learning_plan",
        "plan_id": plan_id,
        "topic": topic,
        "updated": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    lines = [
        f"# Learning Plan: {topic}",
        "",
        "## Summary",
        "",
        summary.strip() or "-",
        "",
        "## Learner State",
        "",
        learner_state.strip() or "-",
        "",
        "## Clarification",
        "",
    ]
    for label, answer in clarifications.items():
        lines.append(f"- {label}: {answer.strip() or '-'}")
    lines.extend(["", "## Tasks", ""])
    task_index = {task.id: index for index, task in enumerate(tasks, start=1)}
    for index, task in enumerate(tasks, start=1):
        meta = parse_learning_task_metadata(task)
        deps = meta.depends_on_task_ids or []
        dependency_labels = [f"[{task_index[dependency]}]" for dependency in deps if dependency in task_index]
        dependency_text = ", ".join(dependency_labels) if dependency_labels else "start now"
        lines.append(f"{index}. {task.title} [{meta.branch or task.work_type or 'study'}] -> {dependency_text}")

    note_path.write_text(
        write_frontmatter(fm, "\n".join(lines).rstrip() + "\n"),
        encoding="utf-8",
    )
    return note_path
