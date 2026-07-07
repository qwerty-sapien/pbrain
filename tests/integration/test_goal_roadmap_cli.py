"""Integration tests for roadmap-backed goal creation and planning gates."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.naming import deterministic_names
from pb.domain.models import Task
from pb.llm.drafts import (
    ArtifactPresentationDraft,
    GeneratedNamesDraft,
    GoalDraft,
    GoalRoadmapDraft,
    GoalRoadmapNodeDraft,
)
from pb.llm.runtime import GeneratedDraft
from pb.storage.repository import Repository


runner = CliRunner()


def _mock_goal_runtime():
    def fake_require(self, purpose):
        return SimpleNamespace(
            configured=True,
            available=True,
            provider="gemini",
            backend="auto",
            default_model="gemini-3-flash-preview",
            structured_output=True,
            credential_source="test",
            message="ready",
        )

    def fake_generate(self, schema_cls, prompt, **kwargs):
        if schema_cls is GoalDraft:
            payload = GoalDraft(
                title="Ricci Flow in General Relativity",
                description="Apply Ricci flow tools to Einstein field equations.",
                domain="geometry",
                execution_mode="study",
                success_definition="Use Ricci flow ideas in a concrete relativity derivation.",
            )
        elif schema_cls is GoalRoadmapDraft:
            payload = GoalRoadmapDraft(
                summary="Foundations first, then guided application, then an independent milestone.",
                progression_mode="adaptive",
                project_title="Ricci Flow Project",
                presentation=ArtifactPresentationDraft(
                    variant="operator",
                    density="balanced",
                    accent="cyan",
                    roadmap_layout="dag_legend",
                ),
                nodes=[
                    GoalRoadmapNodeDraft(
                        node_id="phase-1",
                        title="Ricci flow foundations",
                        branch="study",
                        scope="ricci flow basics",
                        milestone="Understand the core objects.",
                        success_check="Explain the geometric flow ingredients without notes.",
                    ),
                    GoalRoadmapNodeDraft(
                        node_id="phase-2",
                        title="Ricci flow for relativity tensors",
                        branch="study",
                        scope="ricci flow and curvature tensors",
                        milestone="Connect Ricci flow language to relativity objects.",
                        success_check="Work through a guided relativity tensor example.",
                        prerequisites=["phase-1"],
                    ),
                    GoalRoadmapNodeDraft(
                        node_id="phase-3",
                        title="Tensor drill set",
                        branch="practise",
                        scope="tensor drills",
                        milestone="Apply the roadmap concepts in short deliberate drills.",
                        success_check="Complete a tensor drill set with one self-correction pass.",
                        prerequisites=["phase-1"],
                    ),
                    GoalRoadmapNodeDraft(
                        node_id="phase-4",
                        title="Independent relativity milestone",
                        branch="study",
                        scope="independent relativity derivation",
                        milestone="Combine the theory and drill branches in one derivation.",
                        success_check="Complete a guided-to-independent relativity derivation.",
                        prerequisites=["phase-2", "phase-3"],
                    ),
                ],
            )
        else:
            payload = deterministic_names("goal", "ricci flow", {}).model_copy()
            if schema_cls is GeneratedNamesDraft:
                payload = deterministic_names("goal", "ricci flow", {})
        return GeneratedDraft(
            payload=payload,
            model="gemini-3-flash-preview",
            source_scope=kwargs.get("source_scope", "test"),
            prompt_template_version="test",
            raw_response="{}",
        )

    return (
        patch("pb.llm.runtime.LLMRuntime.require", new=fake_require),
        patch("pb.llm.runtime.LLMRuntime.generate_draft", new=fake_generate),
    )


def test_goal_add_persists_roadmap_and_seed_task(temp_db, temp_config):
    require_patch, generate_patch = _mock_goal_runtime()
    with require_patch, generate_patch, patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["goal", "add", "--yes", "ricci", "flow"])

    assert result.exit_code == 0, result.output
    assert "Dependency Chart" in result.output
    assert "Legend" in result.output
    assert "[A] Ricci flow foundations" in result.output
    assert "Scope: ricci flow basics" in result.output
    assert "Milestone: Understand the core objects." in result.output
    assert "Check: Explain the geometric flow ingredients without notes." in result.output
    assert "Prereqs:" not in result.output
    assert "START ->" not in result.output
    assert "Seed task:" in result.output

    repo = Repository()
    goals = repo.list_goal_arcs(status=None)
    assert len(goals) == 1
    goal = goals[0]
    assert "roadmap" in goal.generated_names
    roadmap_path = Path(goal.generated_names["roadmap_note_path"])
    assert roadmap_path.exists()
    content = roadmap_path.read_text(encoding="utf-8")
    assert "```mermaid" in content
    assert "flowchart LR" in content
    assert 'A["A"]' in content
    assert "Legend" in content
    assert "Surface" not in content.split("```mermaid", 1)[1].split("```", 1)[0]

    tasks = [task for task in repo.list_tasks() if goal.id in task.linked_goal_arc_ids]
    assert len(tasks) == 1
    meta = parse_learning_task_metadata(tasks[0])
    assert meta.roadmap_node_id == "phase-1"
    assert meta.goal_project_title == "Ricci Flow Project"


def test_goal_add_records_confident_topics_for_later_diagnostic_confirmation(temp_db, temp_config):
    require_patch, generate_patch = _mock_goal_runtime()
    with require_patch, generate_patch, patch(
        "pb.cli.commands.goals.preview_decision",
        return_value=SimpleNamespace(kind="accept", text=""),
    ), patch(
        "pb.cli.commands.goals._pick_confident_roadmap_nodes",
        return_value=["phase-2"],
    ), patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["goal", "add", "ricci", "flow"])

    assert result.exit_code == 0, result.output
    assert "Marked 1 roadmap topic(s) as confidence claims pending diagnostic confirmation." in result.output

    repo = Repository()
    goal = repo.list_goal_arcs(status=None)[0]
    assert goal.generated_names["roadmap_confident"] == ["phase-2"]
    roadmap_path = Path(goal.generated_names["roadmap_note_path"])
    content = roadmap_path.read_text(encoding="utf-8")
    assert "Confidence Claims Pending Diagnostic" in content
    assert "- [ ] Ricci flow for relativity tensors" in content


def test_confidence_picker_reprints_selected_dag_nodes_with_checkmarks():
    roadmap = GoalRoadmapDraft(
        nodes=[
            GoalRoadmapNodeDraft(
                node_id="phase-1",
                title="Ricci flow foundations",
                branch="study",
                scope="ricci flow basics",
            ),
            GoalRoadmapNodeDraft(
                node_id="phase-2",
                title="Ricci flow for relativity tensors",
                branch="study",
                scope="ricci flow and curvature tensors",
                prerequisites=["phase-1"],
            ),
        ]
    )

    with patch("pb.cli.commands.goals.sys.stdin.isatty", return_value=True), patch(
        "pb.cli.commands.goals.pick_many_choices",
        return_value=["phase-2"],
    ), patch("pb.cli.commands.goals.get_console") as mock_console:
        from pb.cli.commands.goals import _pick_confident_roadmap_nodes

        selected = _pick_confident_roadmap_nodes(roadmap)

    assert selected == ["phase-2"]
    printed = [call.args[0] for call in mock_console().print.call_args_list if call.args]
    assert any("[B] ✓ Ricci flow for relativity tensors" in str(item) for item in printed)


def test_plan_week_scores_unscored_tasks_before_generating_plan(temp_db, temp_config):
    repo = Repository()
    repo.create_task(Task(title="Ricci flow foundations", description="Unscored task for weekly planning.", work_type="study"))

    result = runner.invoke(
        app,
        ["plan", "week", "--hours", "5"],
        input="4\n3\n5\n2\ny\nn\n3\nstudy\n",
    )

    assert result.exit_code == 0, result.output
    assert "Weekly Plan" in result.output
    task = repo.list_tasks()[0]
    assert task.impact == 4
    assert task.work_type == "study"


def test_task_help_hides_score_command(temp_db, temp_config):
    result = runner.invoke(app, ["task", "--help"])

    assert result.exit_code == 0, result.output
    assert "score" not in result.output.lower()
