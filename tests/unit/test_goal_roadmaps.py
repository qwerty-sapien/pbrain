from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from pb.core.goal_roadmaps import (
    attach_roadmap_to_goal,
    ensure_goal_frontier,
    fallback_goal_roadmap,
    refine_follow_on_specs,
    roadmap_follow_on_specs,
)
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.domain.models import GoalArc


def test_ensure_goal_frontier_backfills_seed_task(repo, temp_config):
    goal = GoalArc(title="Ricci Flow in General Relativity", domain="geometry", execution_mode="study")
    repo.create_goal_arc(goal)

    created = ensure_goal_frontier(repo, goal, vault_path=Path(temp_config.general.vault_path))

    assert len(created) == 1
    task = created[0]
    assert goal.id in task.linked_goal_arc_ids
    meta = parse_learning_task_metadata(task)
    assert meta.roadmap_node_id
    assert meta.goal_project_title


def test_roadmap_follow_on_specs_unlock_next_task_after_completion(repo):
    goal = GoalArc(title="Ricci Flow in General Relativity", domain="geometry", execution_mode="study")
    roadmap = fallback_goal_roadmap(goal)
    attach_roadmap_to_goal(goal, roadmap)
    repo.create_goal_arc(goal)

    seed = ensure_goal_frontier(repo, goal)[0]
    seed.completion = 100
    repo.update_task(seed)

    specs = roadmap_follow_on_specs(repo, goal, seed, assessment=None)

    assert specs
    assert specs[0]["title"] == roadmap.nodes[1].title


def test_roadmap_follow_on_specs_insert_remediation_for_weak_assessment(repo):
    goal = GoalArc(title="Ricci Flow in General Relativity", domain="geometry", execution_mode="study")
    roadmap = fallback_goal_roadmap(goal)
    attach_roadmap_to_goal(goal, roadmap)
    repo.create_goal_arc(goal)

    seed = ensure_goal_frontier(repo, goal)[0]
    weak_assessment = SimpleNamespace(
        sub_skill_scores=[
            SimpleNamespace(name="tensor algebra", is_weak=True),
            SimpleNamespace(name="christoffel symbols", is_weak=True),
        ]
    )

    specs = roadmap_follow_on_specs(repo, goal, seed, assessment=weak_assessment)

    assert specs
    assert specs[0]["title"].startswith("Reinforce ")


def test_refine_follow_on_specs_makes_remediation_more_descriptive():
    specs = [
        {
            "title": "Reinforce problem setup",
            "scope": "Physics and Mathematics: problem setup",
            "milestone": "Repair the weak spot in problem setup before moving deeper.",
            "branch": "study",
            "parent_node_id": "phase-1",
        }
    ]

    refined = refine_follow_on_specs(specs, "more descriptive")

    assert refined != specs
    assert refined[0]["title"] == "Reinforce problem setup: concrete correction and check"
    assert "givens" in refined[0]["scope"]
    assert "previous failure mode" in refined[0]["milestone"]


def test_refine_follow_on_specs_carries_unknown_refinement_into_preview():
    specs = [
        {
            "title": "Reinforce verification",
            "scope": "Physics and Mathematics: verification",
            "milestone": "Repair the weak spot in verification before moving deeper.",
            "branch": "study",
            "parent_node_id": "phase-1",
        }
    ]

    refined = refine_follow_on_specs(specs, "make this about Schwarzschild coordinates")

    assert refined != specs
    assert "Schwarzschild coordinates" in refined[0]["milestone"]
