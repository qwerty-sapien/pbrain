"""Integration coverage for later/delete/feedback and clarified curricula."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.feedback_profile import load_feedback_guidance, save_feedback_profile
from pb.core.learning_curriculum import PLAN_WAITING_REASON
from pb.core.models import GenerationProvenance, Session, Task, TimeBlock
from pb.core.naming import deterministic_names
from pb.core.enums import SessionMode, TaskState
from pb.llm.drafts import CurriculumPlanDraft, GeneratedNamesDraft, LearningPlanBlockDraft, StudyPlanDraft
from pb.llm.runtime import DraftAttempt, GeneratedDraft, LLMProbeResult
from pb.storage.repository import Repository


def _runner() -> CliRunner:
    return CliRunner()


def _session_patches():
    return (
        patch("pb.sessions.service.TimerManager.start_session_timers", return_value=None),
        patch("pb.sessions.service.TimerManager.stop_session_timers", return_value=None),
    )


def _mock_curriculum_runtime(payload: CurriculumPlanDraft):
    def fake_generate(self, schema_cls, prompt, **kwargs):
        if schema_cls is GeneratedNamesDraft:
            return GeneratedDraft(
                payload=deterministic_names("goal", "placeholder", {}),
                model="gemini-3-flash-preview",
                source_scope=kwargs.get("source_scope", "test"),
                prompt_template_version="test",
                raw_response="{}",
            )
        if schema_cls is CurriculumPlanDraft:
            return GeneratedDraft(
                payload=payload,
                model="gemini:gemini-3-flash-preview",
                source_scope=kwargs.get("source_scope", "test"),
                prompt_template_version="test",
                raw_response="{}",
                attempts=(
                    DraftAttempt(
                        provider="gemini",
                        model="gemini-3-flash-preview",
                        prompt_kind="draft",
                        status="ok",
                    ),
                ),
            )
        raise AssertionError(f"Unexpected schema: {schema_cls}")

    return patch("pb.llm.runtime.LLMRuntime.generate_draft", new=fake_generate)


def test_later_resets_active_task_and_postpones_it(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    task = Task(title="Study: Calculus", state=TaskState.ACTIVE, completion=40)
    repo.create_task(task)
    session = Session(task_id=task.id, mode=SessionMode.FOCUS)
    repo.create_session(session)

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        result = runner.invoke(app, ["later"])

    assert result.exit_code == 0, result.output
    assert "Moved to later" in result.output

    refreshed = repo.get_task(task.id)
    assert refreshed is not None
    assert refreshed.state == TaskState.PAUSED
    assert refreshed.completion == 0
    assert refreshed.completed_at is None
    assert refreshed.paused_until is None
    assert repo.list_sessions_for_task(task.id) == []


def test_delete_removes_active_task_and_runtime_history(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    task = Task(title="Teach: Bayes Rule", state=TaskState.ACTIVE)
    repo.create_task(task)
    session = Session(task_id=task.id, mode=SessionMode.FOCUS)
    repo.create_session(session)
    block = TimeBlock(task_id=task.id, duration_minutes=30, start_time=datetime.utcnow())
    repo.create_time_block(block)
    repo.create_generation_provenance(
        GenerationProvenance(
            artifact_kind="teach_task",
            artifact_id=task.id,
            generated_by_model="gemini-3-flash-preview",
            prompt_template_version="test",
            source_scope="test",
            accepted_by_user=True,
        )
    )

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        result = runner.invoke(app, ["delete"])

    assert result.exit_code == 0, result.output
    assert "Deleted:" in result.output
    assert repo.get_task(task.id) is None
    assert repo.get_session(session.id) is None
    assert repo.get_time_block(block.id) is None
    assert repo.list_generation_provenance(artifact_id=task.id) == []


def test_feedback_command_writes_scoped_guidance_note(temp_db, temp_config):
    runner = _runner()
    vault = Path(temp_config.general.vault_path)

    result = runner.invoke(app, ["feedback", "study", "worked", "examples"])

    assert result.exit_code == 0, result.output
    note_path = vault / "direction" / "preferences" / "study.md"
    assert note_path.exists()
    content = note_path.read_text(encoding="utf-8")
    assert "worked examples" in content


def test_feedback_guidance_is_injected_into_study_prompt(temp_db, temp_config):
    runner = _runner()
    runtime_vault = Path(temp_config.general.vault_path)
    save_feedback_profile(
        runtime_vault,
        "study",
        more_of="Use worked examples before summaries.",
        less_of="Avoid abstract jargon dumps.",
    )

    captured: dict[str, str] = {}

    def fake_generate(self, schema_cls, prompt, **kwargs):
        if schema_cls is GeneratedNamesDraft:
            return GeneratedDraft(
                payload=deterministic_names("task", "placeholder", {}),
                model="gemini-3-flash-preview",
                source_scope=kwargs.get("source_scope", "test"),
                prompt_template_version="test",
                raw_response="{}",
            )
        captured["prompt"] = prompt
        payload = StudyPlanDraft(
            summary="",
            blocks=[
                LearningPlanBlockDraft(
                    branch="study",
                    subject_scope="eigenvalues",
                    duration_minutes=25,
                    target_bloom_stage="apply",
                    study_mode="worked example",
                    success_check="Solve one example cleanly.",
                    reason="Ground the concept in a concrete example.",
                )
            ],
        )
        return GeneratedDraft(
            payload=payload,
            model="gemini-3-flash-preview",
            source_scope=kwargs.get("source_scope", "test"),
            prompt_template_version="test",
            raw_response="{}",
        )

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch, patch("pb.llm.runtime.LLMRuntime.generate_draft", new=fake_generate):
        result = runner.invoke(app, ["study", "--yes", "eigenvalues"])

    assert result.exit_code == 0, result.output
    assert "Use worked examples before summaries." in captured["prompt"]
    assert "Avoid abstract jargon dumps." in captured["prompt"]


def test_learn_broad_topic_creates_linked_curriculum_and_unlocks_next_task(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    runtime_vault = Path(temp_config.general.vault_path)

    payload = CurriculumPlanDraft(
        summary="Start with fundamentals, then apply them.",
        learner_state="Beginner with some motivation but weak recall.",
        blocks=[
            LearningPlanBlockDraft(
                node_id="root",
                branch="study",
                subject_scope="calculus limits",
                duration_minutes=30,
                target_bloom_stage="understand",
                study_mode="active recall",
                success_check="Explain a limit in plain language.",
                reason="Build the conceptual footing first.",
            ),
            LearningPlanBlockDraft(
                node_id="practice",
                depends_on=["root"],
                branch="practise",
                subject_scope="calculus limit drills",
                duration_minutes=35,
                practice_stage="integrate",
                drill_type="limit drills",
                success_check="Solve three progressively harder limit drills.",
                reason="Translate understanding into deliberate reps.",
            ),
        ],
    )

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch, _mock_curriculum_runtime(payload), patch(
        "pb.cli.commands.clarify._maybe_create_goal",
        side_effect=AssertionError("broad learn flow should not create the goal before clarification"),
    ):
        result = runner.invoke(app, ["learn", "--yes", "calculus"])

    assert result.exit_code == 0, result.output
    assert "Learning plan saved:" in result.output

    goals = repo.list_goal_arcs(status=None)
    assert len(goals) == 1
    lightweight_goal = goals[0]

    active_session = repo.get_active_session()
    assert active_session is not None
    root_task = repo.get_task(active_session.task_id)
    assert root_task is not None
    assert root_task.title.startswith("Study:")
    assert root_task.linked_goal_arc_ids == [lightweight_goal.id]

    paused_children = [
        task for task in repo.list_tasks(include_archived=True)
        if task.id != root_task.id and task.state == TaskState.PAUSED
    ]
    assert len(paused_children) == 1
    child = paused_children[0]
    assert child.pause_reason == PLAN_WAITING_REASON

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        finish_result = runner.invoke(app, ["finish"])

    assert finish_result.exit_code == 0, finish_result.output
    refreshed_child = repo.get_task(child.id)
    assert refreshed_child is not None
    assert refreshed_child.state == TaskState.ACTIVE
    assert load_feedback_guidance(runtime_vault, "learn") == ""


def test_learn_broad_topic_falls_back_to_local_curriculum_when_llm_probe_fails(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    probe = LLMProbeResult(
        available=False,
        provider="gemini",
        backend="vertex",
        model="gemini-3-flash-preview",
        credential_source="vertex",
        category="config",
        message="gemini rejected the configured request or model settings.",
        debug_message="404 NOT_FOUND projects/my-gcp-project-12345/... gemini-3-flash",
        http_status=404,
        retryable=False,
    )

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch, patch(
        "pb.llm.runtime.LLMRuntime.live_probe",
        return_value=probe,
    ), patch(
        "pb.llm.runtime.LLMRuntime.generate_draft",
        side_effect=AssertionError("Clarify should not call the LLM after a failed live probe"),
    ), patch(
        "pb.cli.commands.clarify._maybe_create_goal",
        side_effect=AssertionError("broad learn flow should not create the goal before clarification"),
    ):
        result = runner.invoke(app, ["learn", "--yes", "calculus"])

    assert result.exit_code == 0, result.output
    assert "needs live LLM output" in result.output
    assert "pb doctor --llm" in result.output
    assert "Using a local fallback learning plan" in result.output
    assert "projects/my-gcp-project-12345" not in result.output

    active_session = repo.get_active_session()
    assert active_session is not None
    tasks = repo.list_tasks(include_archived=True)
    assert len(tasks) >= 2
    assert len(repo.list_goal_arcs(status=None)) == 1


def test_start_broad_learning_todo_expands_into_curriculum_and_archives_source(temp_db, temp_config):
    runner = _runner()
    repo = Repository()

    payload = CurriculumPlanDraft(
        summary="Split the request into prerequisites and application.",
        learner_state="Broad request with multiple dependent topics.",
        blocks=[
            LearningPlanBlockDraft(
                node_id="root",
                branch="study",
                subject_scope="vector calculus prerequisites",
                duration_minutes=35,
                target_bloom_stage="understand",
                study_mode="active recall",
                success_check="Explain the prerequisite stack clearly.",
                reason="Start with the foundations the later topics depend on.",
            ),
            LearningPlanBlockDraft(
                node_id="apply",
                depends_on=["root"],
                branch="practise",
                subject_scope="maxwell equation application drills",
                duration_minutes=40,
                practice_stage="integrate",
                drill_type="worked physics drills",
                success_check="Solve one constrained application cleanly.",
                reason="Turn the prerequisite pass into a concrete rep.",
            ),
        ],
    )

    todo = Task(
        title="learn vector calc, lebesgue integrals, line integrals, apply maxwell equations, and learn ricci flow",
        work_type="todo",
        state=TaskState.ACTIVE,
    )
    repo.create_task(todo)

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch, _mock_curriculum_runtime(payload):
        result = runner.invoke(app, ["start", todo.id])

    assert result.exit_code == 0, result.output
    assert "Expanded todo into learning plan:" in result.output

    archived_source = repo.get_task(todo.id)
    assert archived_source is not None
    assert archived_source.archived_at is not None
    assert "Expanded into learning plan" in archived_source.description

    active_session = repo.get_active_session()
    assert active_session is not None
    root_task = repo.get_task(active_session.task_id)
    assert root_task is not None
    assert root_task.id != todo.id
    assert root_task.title.startswith("Study:")

    paused_children = [
        task for task in repo.list_tasks(include_archived=True)
        if task.id not in {todo.id, root_task.id} and task.state == TaskState.PAUSED
    ]
    assert len(paused_children) == 1
    assert paused_children[0].pause_reason == PLAN_WAITING_REASON

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        finish_result = runner.invoke(app, ["finish"])

    assert finish_result.exit_code == 0, finish_result.output
    refreshed_child = repo.get_task(paused_children[0].id)
    assert refreshed_child is not None
    assert refreshed_child.state == TaskState.ACTIVE
