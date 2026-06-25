# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Goal-backed roadmap/project helpers."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import re
from typing import Iterable, Optional

import yaml

from pb.core.enums import EnergyType, TaskState
from pb.core.learning_metadata import build_learning_task_description, parse_learning_task_metadata
from pb.core.learning_prompting import learning_intent_style_guidance
from pb.core.models import GoalArc, Task, utc_now
from pb.core.roadmap_dag import build_symbolic_dag, render_legend_lines, render_mermaid_flowchart_lines
from pb.llm.drafts import GoalRoadmapDraft, GoalRoadmapNodeDraft, artifact_presentation_prompt


MAX_CONCURRENT_FRONTIER = 3


def build_goal_roadmap_prompt(goal_draft, raw_goal: str, *, existing_goal: GoalArc | None = None) -> str:
    """Build the roadmap-generation prompt for goal setup."""
    existing_text = ""
    if existing_goal is not None:
        existing_text = (
            f"Existing title: {existing_goal.title}\n"
            f"Existing domain: {existing_goal.domain}\n"
            f"Existing mode: {existing_goal.execution_mode}\n"
        )
    return (
        "Turn this learning goal into a serious project-style roadmap for a CLI-first learning system.\n"
        "Return 4 to 8 narrow tasks/phases.\n"
        "Each task must build concrete competency for later tasks, with a milestone and success check.\n"
        "For ambitious goals, distinguish prerequisite progress from target progress.\n"
        "If the learner's current readiness is unproven, front-load prerequisite phases instead of pretending the terminal topic is immediately executable.\n"
        "Each phase scope should name the exact concepts or capabilities to build, not just paraphrase the raw goal.\n"
        "Use prerequisites to reflect the true dependency structure, not an arbitrary sequence.\n"
        "When several later phases share the same foundation, make them sibling branches instead of forcing a fake ladder.\n"
        "Do not make every task depend on the immediately previous task unless that dependency is genuinely required.\n"
        "Keep the roadmap compact, but allow real branching when the topic naturally splits into sub-skills.\n"
        "Use 'study' for conceptual phases and 'practise' for deliberate drills.\n"
        "Default progression_mode should be adaptive unless the request strongly implies otherwise.\n\n"
        + learning_intent_style_guidance()
        + f"{existing_text}"
        + f"Goal title: {goal_draft.title}\n"
        + f"Domain: {goal_draft.domain}\n"
        + f"Execution mode: {goal_draft.execution_mode}\n"
        + f"Success definition: {goal_draft.success_definition}\n"
        + f"Description: {goal_draft.description}\n"
        + f"Raw request: {raw_goal}\n"
        + f"{artifact_presentation_prompt(include_dependency_layout=True)}"
    )


def ensure_roadmap_populated(roadmap: GoalRoadmapDraft, goal_like) -> GoalRoadmapDraft:
    """Return roadmap as-is if non-empty, otherwise fall back to deterministic generation."""
    if roadmap and roadmap.project_title and roadmap.nodes:
        return roadmap
    return fallback_goal_roadmap(goal_like)


def fallback_goal_roadmap(goal_like) -> GoalRoadmapDraft:
    """Deterministic fallback roadmap when structured drafting is unavailable."""
    title = getattr(goal_like, "title", "") or "Learning project"
    domain = getattr(goal_like, "domain", "") or title
    mode = getattr(goal_like, "execution_mode", "") or "study"
    project_title = f"{title} Project"

    def branch_for(index: int) -> str:
        if mode == "study":
            return "study"
        if mode in {"practise", "practice"}:
            return "practise"
        return "study" if index in {1, 2, 4} else "practise"

    nodes = [
        GoalRoadmapNodeDraft(
            node_id="phase-1",
            title=f"{title}: foundations",
            branch=branch_for(1),
            scope=f"{domain} fundamentals",
            milestone=f"Build a usable baseline in {domain}.",
            success_check=f"Explain or demonstrate the core building blocks of {domain} without help.",
            clarification_prompts=[f"What does 'baseline' mean for {domain} in your own words?"],
        ),
        GoalRoadmapNodeDraft(
            node_id="phase-2",
            title=f"{title}: core patterns",
            branch=branch_for(2),
            scope=f"{domain} core patterns",
            milestone=f"Recognize and reproduce the core patterns behind {domain}.",
            success_check=f"Handle a guided problem or drill using the main patterns in {domain}.",
            prerequisites=["phase-1"],
            clarification_prompts=[f"Which core patterns matter most in {domain}?"],
        ),
        GoalRoadmapNodeDraft(
            node_id="phase-3",
            title=f"{title}: guided application",
            branch=branch_for(3),
            scope=f"guided {domain} application",
            milestone=f"Apply {domain} in a concrete guided setting.",
            success_check=f"Produce one concrete application attempt in {domain}.",
            prerequisites=["phase-2"],
            clarification_prompts=[f"What kind of application should prove progress in {domain}?"],
        ),
        GoalRoadmapNodeDraft(
            node_id="phase-4",
            title=f"{title}: independent milestone",
            branch=branch_for(4),
            scope=f"independent {domain} milestone",
            milestone=getattr(goal_like, "success_definition", "") or f"Reach a meaningful milestone in {domain}.",
            success_check=getattr(goal_like, "success_definition", "") or f"Complete an independently chosen milestone in {domain}.",
            prerequisites=["phase-3"],
            clarification_prompts=[f"What final milestone would make {domain} feel real rather than theoretical?"],
        ),
    ]
    return GoalRoadmapDraft(
        summary=f"Roadmap from foundations to an independent milestone in {domain}.",
        progression_mode="adaptive",
        project_title=project_title,
        nodes=nodes,
    )


def roadmap_from_goal(goal: GoalArc) -> GoalRoadmapDraft | None:
    """Load the stored roadmap draft from goal metadata."""
    generated = getattr(goal, "generated_names", {}) or {}
    raw = generated.get("roadmap")
    if not isinstance(raw, dict):
        return None
    try:
        return GoalRoadmapDraft.model_validate(raw)
    except Exception:
        return None


def project_title_for_goal(goal: GoalArc, roadmap: GoalRoadmapDraft | None = None) -> str:
    """Resolve the user-facing project title for a goal-backed sequence."""
    generated = getattr(goal, "generated_names", {}) or {}
    explicit = generated.get("goal_project_title")
    if isinstance(explicit, str) and explicit.strip():
        return explicit.strip()
    if roadmap is not None and roadmap.project_title.strip():
        return roadmap.project_title.strip()
    return f"{goal.title} Project"


def attach_roadmap_to_goal(
    goal: GoalArc,
    roadmap: GoalRoadmapDraft,
    *,
    roadmap_path: Path | None = None,
    confident_node_ids: list[str] | None = None,
) -> GoalArc:
    """Store roadmap metadata on the goal without changing schema."""
    generated = dict(getattr(goal, "generated_names", {}) or {})
    generated["roadmap"] = roadmap.model_dump(mode="json")
    generated["goal_project_title"] = project_title_for_goal(goal, roadmap)
    generated["goal_progress_mode"] = roadmap.progression_mode
    if confident_node_ids is not None:
        generated["roadmap_confident"] = [node_id for node_id in confident_node_ids if node_id]
    if roadmap_path is not None:
        generated["roadmap_note_path"] = str(roadmap_path)
    goal.generated_names = generated
    return goal


def write_goal_roadmap_note(
    vault_path: Path,
    goal: GoalArc,
    roadmap: GoalRoadmapDraft,
    *,
    confident_node_ids: list[str] | None = None,
) -> Path:
    """Persist the roadmap spec as a durable project note."""
    projects_dir = vault_path / "direction" / "projects"
    projects_dir.mkdir(parents=True, exist_ok=True)
    slug = _slugify(project_title_for_goal(goal, roadmap))
    note_path = projects_dir / f"{slug}.md"
    frontmatter = {
        "type": "goal_project",
        "goal_id": goal.id,
        "goal_title": goal.title,
        "project_title": project_title_for_goal(goal, roadmap),
        "progression_mode": roadmap.progression_mode,
        "updated": datetime.utcnow().strftime("%Y-%m-%d"),
    }
    confident_set = {node_id for node_id in (confident_node_ids or []) if node_id}
    dag = build_symbolic_dag(roadmap.nodes)
    symbol_by_id = dag.symbol_by_id
    mermaid_lines = render_mermaid_flowchart_lines(dag)
    legend_lines = render_legend_lines(dag, width=72)
    node_lookup = {node.node_id: node for node in roadmap.nodes}
    lines = [
        f"# {project_title_for_goal(goal, roadmap)}",
        "",
        "## Goal",
        "",
        goal.title,
        "",
        "## Goal Summary",
        "",
        roadmap.summary or goal.description or "-",
        "",
        "## Dependency DAG",
        "",
        "```mermaid",
        *mermaid_lines,
        "```",
        "",
        *legend_lines,
        "",
        "## Roadmap",
        "",
    ]
    for dag_node in dag.nodes:
        node = node_lookup.get(dag_node.node_id)
        if node is None:
            continue
        prereq_text = " + ".join(symbol_by_id.get(dep, dep) for dep in node.prerequisites) or "start"
        symbol = symbol_by_id.get(node.node_id, node.node_id)
        lines.extend(
            [
                f"[{symbol}] {node.title}",
                f"- Branch: {node.branch}",
                f"- Scope: {node.scope or node.title}",
                f"- Milestone: {node.milestone or '-'}",
                f"- Check: {node.success_check or '-'}",
                f"- Prerequisites: {prereq_text}",
                "",
            ]
        )
    if confident_set:
        lines.extend(
            [
                "## Confidence Claims Pending Diagnostic",
                "",
            ]
        )
        for node in roadmap.nodes:
            if node.node_id not in confident_set:
                continue
            lines.append(f"- [ ] {node.title}")
        lines.append("")
    note_path.write_text(
        f"---\n{yaml.safe_dump(frontmatter, sort_keys=False).strip()}\n---\n\n" + "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )
    return note_path


def ensure_goal_seed_tasks(repo, goals: Iterable[GoalArc], *, vault_path: Path | None = None) -> list[Task]:
    """Backfill missing frontier tasks for goals that currently have none."""
    created: list[Task] = []
    for goal in goals:
        created.extend(ensure_goal_frontier(repo, goal, vault_path=vault_path))
    return created


def ensure_goal_frontier(repo, goal: GoalArc, *, vault_path: Path | None = None) -> list[Task]:
    """Ensure a goal has at least one linked incomplete task."""
    incomplete = _linked_incomplete_tasks(repo, goal.id)
    if incomplete:
        return []

    roadmap = roadmap_from_goal(goal)
    if roadmap is None:
        roadmap = fallback_goal_roadmap(goal)
        if vault_path is not None:
            roadmap_path = write_goal_roadmap_note(vault_path, goal, roadmap)
            attach_roadmap_to_goal(goal, roadmap, roadmap_path=roadmap_path)
            repo.update_goal_arc(goal)
        else:
            attach_roadmap_to_goal(goal, roadmap)
            repo.update_goal_arc(goal)

    return materialize_next_frontier_tasks(repo, goal, roadmap=roadmap, max_new=1)


def materialize_next_frontier_tasks(
    repo,
    goal: GoalArc,
    *,
    roadmap: GoalRoadmapDraft | None = None,
    max_new: int = MAX_CONCURRENT_FRONTIER,
    sequence_status: str = "frontier",
    remediation_specs: list[dict[str, str]] | None = None,
) -> list[Task]:
    """Create the next reachable roadmap tasks, respecting the frontier cap."""
    roadmap = roadmap or roadmap_from_goal(goal)
    if roadmap is None:
        return []

    active_count = len(_linked_incomplete_tasks(repo, goal.id))
    available_slots = max(0, MAX_CONCURRENT_FRONTIER - active_count)
    if max_new > 0:
        available_slots = min(available_slots, max_new)
    if available_slots <= 0:
        return []

    if remediation_specs:
        return [
            _create_ad_hoc_task(
                repo,
                goal,
                roadmap,
                title=spec["title"],
                scope=spec["scope"],
                milestone=spec["milestone"],
                branch=spec["branch"],
                sequence_status="remediation",
                parent_node_id=spec.get("parent_node_id", ""),
            )
            for spec in remediation_specs[:available_slots]
        ]

    completed_node_ids = _completed_node_ids(repo, goal.id)
    existing_node_ids = _existing_node_ids(repo, goal.id)
    bypassed = set(_goal_generated_list(goal, "roadmap_bypassed"))
    created: list[Task] = []

    for node in roadmap.nodes:
        if node.node_id in existing_node_ids or node.node_id in bypassed:
            continue
        prereqs = node.prerequisites or []
        if prereqs and not all(prereq in completed_node_ids or prereq in bypassed for prereq in prereqs):
            continue
        created.append(_create_task_for_node(repo, goal, roadmap, node, sequence_status=sequence_status))
        if len(created) >= available_slots:
            break
    return created


def roadmap_follow_on_specs(repo, goal: GoalArc, task: Task, assessment=None) -> list[dict[str, str]]:
    """Build next-task specs after a roadmap task is completed."""
    meta = parse_learning_task_metadata(task)
    roadmap = roadmap_from_goal(goal)
    if roadmap is None:
        return []

    if assessment is not None:
        remediation = _remediation_specs(goal, task, assessment)
        if remediation:
            return remediation

    completed = _completed_node_ids(repo, goal.id)
    completed.add(meta.roadmap_node_id or meta.plan_node_id)
    existing = _existing_node_ids(repo, goal.id)
    bypassed = set(_goal_generated_list(goal, "roadmap_bypassed"))
    specs: list[dict[str, str]] = []
    for node in roadmap.nodes:
        if node.node_id in existing or node.node_id in completed or node.node_id in bypassed:
            continue
        prereqs = node.prerequisites or []
        if prereqs and not all(prereq in completed or prereq in bypassed for prereq in prereqs):
            continue
        specs.append(
            {
                "title": node.title,
                "scope": node.scope or node.title,
                "milestone": node.success_check or node.milestone,
                "branch": node.branch,
                "parent_node_id": meta.roadmap_node_id or meta.plan_node_id,
            }
        )
        if len(specs) >= MAX_CONCURRENT_FRONTIER:
            break
    return specs


def preview_rows_for_follow_on_specs(specs: list[dict[str, str]]) -> list[tuple[str, list[tuple[str, str]]]]:
    """Build preview rows for next-task creation."""
    rows: list[tuple[str, list[tuple[str, str]]]] = []
    for index, spec in enumerate(specs, start=1):
        rows.append(
            (
                f"Task {index}",
                [
                    ("Title", spec.get("title", "")),
                    ("Scope", spec.get("scope", "")),
                    ("Branch", spec.get("branch", "")),
                    ("Milestone", spec.get("milestone", "")),
                ],
            )
        )
    return rows


def _create_task_for_node(repo, goal: GoalArc, roadmap: GoalRoadmapDraft, node: GoalRoadmapNodeDraft, *, sequence_status: str) -> Task:
    branch = "practise" if node.branch == "practise" else "study"
    description = build_learning_task_description(
        node.milestone or f"Roadmap task for {goal.title}.",
        branch=branch,
        scope=node.scope or node.title,
        domain=goal.domain or node.scope or goal.title,
        bloom_target=goal.target_bloom_stage.value if branch == "study" and goal.target_bloom_stage else "",
        practice_stage=goal.target_practice_stage.value if branch == "practise" and goal.target_practice_stage else "",
        success_check=node.success_check or node.milestone,
        roadmap_id=_roadmap_id(goal, roadmap),
        roadmap_node_id=node.node_id,
        parent_node_id=(node.prerequisites[-1] if node.prerequisites else ""),
        sequence_level=_node_level(roadmap, node.node_id),
        sequence_status=sequence_status,
        goal_project_title=project_title_for_goal(goal, roadmap),
        goal_progress_mode=roadmap.progression_mode,
    )
    task = Task(
        title=node.title,
        description=description,
        state=TaskState.ACTIVE,
        created_at=utc_now(),
        energy_type=EnergyType.PRACTICE if branch == "practise" else EnergyType.DEEP,
        work_type="practice" if branch == "practise" else "study",
        linked_goal_arc_ids=[goal.id],
    )
    task.generated_names["goal_project_title"] = project_title_for_goal(goal, roadmap)
    task.generated_names["roadmap_node_id"] = node.node_id
    repo.create_task(task)
    return task


def _create_ad_hoc_task(
    repo,
    goal: GoalArc,
    roadmap: GoalRoadmapDraft,
    *,
    title: str,
    scope: str,
    milestone: str,
    branch: str,
    sequence_status: str,
    parent_node_id: str,
) -> Task:
    description = build_learning_task_description(
        milestone,
        branch=branch,
        scope=scope,
        domain=goal.domain or scope or goal.title,
        success_check=milestone,
        roadmap_id=_roadmap_id(goal, roadmap),
        roadmap_node_id=f"dynamic-{_slugify(title)}",
        parent_node_id=parent_node_id,
        sequence_level=1,
        sequence_status=sequence_status,
        goal_project_title=project_title_for_goal(goal, roadmap),
        goal_progress_mode=roadmap.progression_mode,
    )
    task = Task(
        title=title,
        description=description,
        state=TaskState.ACTIVE,
        created_at=utc_now(),
        energy_type=EnergyType.PRACTICE if branch == "practise" else EnergyType.DEEP,
        work_type="practice" if branch == "practise" else "study",
        linked_goal_arc_ids=[goal.id],
    )
    task.generated_names["goal_project_title"] = project_title_for_goal(goal, roadmap)
    repo.create_task(task)
    return task


def _linked_incomplete_tasks(repo, goal_id: str) -> list[Task]:
    return [
        task
        for task in repo.list_tasks()
        if goal_id in (getattr(task, "linked_goal_arc_ids", []) or [])
        and task.archived_at is None
        and task.completion < 100
    ]


def _completed_node_ids(repo, goal_id: str) -> set[str]:
    completed: set[str] = set()
    for task in repo.list_tasks(include_archived=True):
        if goal_id not in (getattr(task, "linked_goal_arc_ids", []) or []):
            continue
        if task.completion < 100:
            continue
        meta = parse_learning_task_metadata(task)
        node_id = meta.roadmap_node_id or meta.plan_node_id
        if node_id:
            completed.add(node_id)
    return completed


def _existing_node_ids(repo, goal_id: str) -> set[str]:
    existing: set[str] = set()
    for task in repo.list_tasks(include_archived=True):
        if goal_id not in (getattr(task, "linked_goal_arc_ids", []) or []):
            continue
        meta = parse_learning_task_metadata(task)
        node_id = meta.roadmap_node_id or meta.plan_node_id
        if node_id:
            existing.add(node_id)
    return existing


def _node_level(roadmap: GoalRoadmapDraft, node_id: str, memo: Optional[dict[str, int]] = None) -> int:
    memo = memo or {}
    if node_id in memo:
        return memo[node_id]
    node = next((item for item in roadmap.nodes if item.node_id == node_id), None)
    if node is None:
        return 1
    prereqs = node.prerequisites or []
    if not prereqs:
        memo[node_id] = 1
        return 1
    memo[node_id] = 1 + max(_node_level(roadmap, prereq, memo) for prereq in prereqs)
    return memo[node_id]


def _goal_generated_list(goal: GoalArc, key: str) -> list[str]:
    generated = getattr(goal, "generated_names", {}) or {}
    raw = generated.get(key, [])
    if not isinstance(raw, list):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def _remediation_specs(goal: GoalArc, task: Task, assessment) -> list[dict[str, str]]:
    weak = [item for item in getattr(assessment, "sub_skill_scores", []) if getattr(item, "is_weak", False)]
    if not weak:
        return []
    meta = parse_learning_task_metadata(task)
    branch = meta.branch or "study"
    specs: list[dict[str, str]] = []
    for item in weak[:3]:
        skill = str(getattr(item, "name", "")).replace("_", " ").strip() or "foundation"
        specs.append(
            {
                "title": f"Reinforce {skill}",
                "scope": f"{goal.domain or goal.title}: {skill}",
                "milestone": f"Repair the weak spot in {skill} before moving deeper.",
                "branch": branch,
                "parent_node_id": meta.roadmap_node_id or meta.plan_node_id,
            }
        )
    return specs


def _roadmap_id(goal: GoalArc, roadmap: GoalRoadmapDraft) -> str:
    generated = getattr(goal, "generated_names", {}) or {}
    explicit = generated.get("roadmap_note_path")
    if isinstance(explicit, str) and explicit.strip():
        return explicit
    return _slugify(project_title_for_goal(goal, roadmap))


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-") or "project"
