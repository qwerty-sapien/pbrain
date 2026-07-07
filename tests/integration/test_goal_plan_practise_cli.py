"""Focused integration tests for the goal-plan-study/practise CLI surfaces."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.naming import deterministic_names
from pb.llm.drafts import GeneratedNamesDraft, GoalDraft, InstructionStep, LearningPlanBlockDraft, PractisePlanDraft, StudyPlanDraft
from pb.llm.runtime import GeneratedDraft
from pb.core.models import ActionReminder, Session, Task, TimeBlock
from pb.core.enums import SessionMode, TaskState
from pb.storage.repository import Repository


def _runner() -> CliRunner:
    return CliRunner()


def _session_patches():
    return (
        patch("pb.sessions.service.TimerManager.start_session_timers", return_value=None),
        patch("pb.sessions.service.TimerManager.stop_session_timers", return_value=None),
    )


def _mock_draft_runtime(payload):
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
        if schema_cls is GeneratedNamesDraft:
            raw_intent = "placeholder"
            for line in prompt.splitlines():
                if line.startswith("Raw intent:"):
                    raw_intent = line.split(":", 1)[1].strip() or raw_intent
                    break
            payload_obj = deterministic_names("task", raw_intent, {})
        elif schema_cls is GoalDraft:
            payload_obj = GoalDraft(
                title="Placeholder Goal",
                domain="math",
                execution_mode="study",
                success_definition="Make measurable progress.",
            )
        else:
            payload_obj = payload
        return GeneratedDraft(
            payload=payload_obj,
            model="gemini-3-flash-preview",
            source_scope=kwargs.get("source_scope", "test"),
            prompt_template_version="test",
            raw_response="{}",
        )

    return (
        patch("pb.llm.runtime.LLMRuntime.require", new=fake_require),
        patch("pb.llm.runtime.LLMRuntime.generate_draft", new=fake_generate),
    )


def test_practise_direct_creates_practise_session(temp_db, temp_config):
    """Bare `pb practise <topic>` stamps the active session as practise."""
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    practise_draft = PractisePlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="practise",
                subject_scope="Piano",
                duration_minutes=25,
                practice_stage="integrate",
                drill_type="reps",
                constraint="No sheet music after the first two reps",
                feedback_source="self",
                evidence_target="clean runs",
                coach_cues="Slow down before speeding up",
                success_check="Three clean repetitions",
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(practise_draft)
    with start_patch, stop_patch, require_patch, generate_patch:
        result = runner.invoke(app, ["practise", "--yes", "Piano"])
    assert result.exit_code == 0, result.output
    assert "Started:" in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    assert session.branch == "practise"
    assert session.subject_scope == "Piano"

    task = repo.get_task(session.task_id)
    assert task is not None
    assert task.work_type == "practice"


def test_study_direct_creates_study_session(temp_db, temp_config):
    """Bare `pb study <topic>` starts a study session instead of treating the first word as a subcommand."""
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    study_draft = StudyPlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="study",
                subject_scope="jazz music theory bossa nova",
                duration_minutes=30,
                target_bloom_stage="apply",
                study_mode="active recall",
                reason="Focus on the conceptual layer first.",
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(study_draft)
    with start_patch, stop_patch, require_patch, generate_patch:
        result = runner.invoke(app, ["study", "--yes", "jazz", "music", "theory", "bossa", "nova"])
    assert result.exit_code == 0, result.output
    assert "Started:" in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    assert session.branch == "study"
    assert session.subject_scope

    task = repo.get_task(session.task_id)
    assert task is not None
    assert task.generated_names["short_title"]
    assert task.work_type == "study"


def _create_planned_study_task(repo: Repository, title: str, minutes: int) -> Task:
    task = Task(title=title, state=TaskState.ACTIVE, work_type="study")
    repo.create_task(task)
    repo.create_time_block(
        TimeBlock(
            task_id=task.id,
            duration_minutes=minutes,
            block_kind="study",
        )
    )
    return task


def test_study_numeric_shortcut_starts_planned_block(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    planned = _create_planned_study_task(repo, "Verify Binary Classification Metrics", 45)

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        result = runner.invoke(app, ["study", "1"])

    assert result.exit_code == 0, result.output
    assert "Started: Verify Binary Classification Metrics (45m)" in result.output

    session = repo.get_active_session()
    assert session is not None
    assert session.task_id == planned.id
    assert session.branch == "study"


def test_study_day_command_starts_requested_block(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    planned = _create_planned_study_task(repo, "Precision Recall Drill", 30)

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        result = runner.invoke(app, ["study", "day", "1"])

    assert result.exit_code == 0, result.output
    assert "Started: Precision Recall Drill (30m)" in result.output

    session = repo.get_active_session()
    assert session is not None
    assert session.task_id == planned.id


def test_study_invalid_planned_code_returns_clean_error(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    _create_planned_study_task(repo, "Only planned block", 25)

    result = runner.invoke(app, ["study", "9"])

    assert result.exit_code == 1, result.output
    assert "No block 9 in today's plan." in result.output
    assert "missing 1 required positional argument" not in result.output


def test_finish_recommendation_uses_valid_study_command(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    first = _create_planned_study_task(repo, "First planned study block", 20)
    second = _create_planned_study_task(repo, "Second planned study block", 35)
    repo.create_session(
        Session(
            task_id=first.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="first scope",
        )
    )

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        finish_result = runner.invoke(app, ["finish", "wrapped", "up"])

    assert finish_result.exit_code == 0, finish_result.output
    assert "Next:" in finish_result.output
    assert "pb study 2" in finish_result.output

    with start_patch, stop_patch:
        next_result = runner.invoke(app, ["study", "2"])

    assert next_result.exit_code == 0, next_result.output
    assert "Started: Second planned study block (35m)" in next_result.output

    session = repo.get_active_session()
    assert session is not None
    assert session.task_id == second.id


def test_learn_preflights_active_session_before_llm_calls(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    active_task = Task(title="German conjugation recovery", state=TaskState.ACTIVE)
    repo.create_task(active_task)
    repo.create_session(
        Session(
            task_id=active_task.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="German conjugation",
        )
    )

    with patch(
        "pb.llm.runtime.LLMRuntime.generate_draft",
        side_effect=AssertionError("LLM should not run before active-session preflight"),
    ):
        result = runner.invoke(app, ["learn", "linear", "algebra"])

    assert result.exit_code != 0
    assert "Session active: German conjugation recovery." in result.output
    assert "pb finish --skip" in result.output
    assert "pb pause" in result.output


def test_study_start_skips_blocking_live_clock_for_learning_sessions(temp_db, temp_config):
    """Study sessions should return to the prompt instead of entering the blocking clock UI."""
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    study_draft = StudyPlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="study",
                subject_scope="signal processing",
                duration_minutes=35,
                target_bloom_stage="apply",
                study_mode="active recall",
                reason="Work from concept to example.",
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(study_draft)
    with start_patch, stop_patch, require_patch, generate_patch, patch(
        "pb.cli.commands.execute.sys.stdin.isatty",
        return_value=True,
    ), patch(
        "pb.cli.display.live_session_clock",
    ) as mock_clock:
        result = runner.invoke(app, ["study", "--yes", "signal", "processing"])

    assert result.exit_code == 0, result.output
    mock_clock.assert_not_called()


def test_finish_partial_defaults_to_mid_completion(temp_db, temp_config):
    """pb finish partial ... sets a mid-range completion without prompting."""
    runner = _runner()
    repo = Repository()
    task = Task(title="Partial completion task", state=TaskState.ACTIVE)
    repo.create_task(task)
    session = Session(task_id=task.id, mode=SessionMode.FOCUS)
    repo.create_session(session)

    start_patch, stop_patch = _session_patches()
    with start_patch, stop_patch:
        result = runner.invoke(app, ["finish", "partial", "needs", "more", "reps"])
    assert result.exit_code == 0, result.output
    assert "50%" in result.output

    finished = repo.get_session(session.id)
    assert finished is not None
    assert finished.completion_pct == 50
    assert finished.actual_outcome == "needs more reps"


def test_next_schedule_creates_reminder(temp_db, temp_config):
    """pb next --schedule stores an actionable reminder and schedules a notification."""
    runner = _runner()
    with patch("pb.cli.commands.next.schedule_actionable_notification", return_value=True) as mock_schedule:
        result = runner.invoke(app, ["next", "--schedule", "15"])
    assert result.exit_code == 0, result.output
    assert "Reminder scheduled" in result.output
    mock_schedule.assert_called_once()

    repo = Repository()
    reminders = repo.list_action_reminders()
    assert len(reminders) == 1
    assert reminders[0].status == "pending"


def test_next_reminder_skip_marks_skipped(temp_db, temp_config):
    """pb next --reminder ... supports Skip without side effects."""
    import pb.cli.commands.next as next_mod

    repo = Repository()
    reminder = ActionReminder(
        title="pb study start math",
        message="Start the planned study block.",
        target_command="study start math",
    )
    repo.create_action_reminder(reminder)
    ctx = SimpleNamespace(obj={"repo": repo})

    with patch("pb.cli.commands.next.sys.stdin.isatty", return_value=True), patch(
        "pb.cli.commands.next.pick_single_choice",
        return_value="skip",
    ):
        next_mod._run_reminder_action(ctx, reminder.id)

    updated = repo.get_action_reminder(reminder.id)
    assert updated is not None
    assert updated.status == "skipped"


def test_do_routes_to_practise_for_skill_intent(temp_db, temp_config):
    """pb do uses the new router and can choose a practise command."""
    import pb.cli.commands.do as do_mod

    repo = Repository()
    ctx = SimpleNamespace(obj={"repo": repo})

    with patch("pb.cli.normalize.sys.stdin.isatty", return_value=True), patch(
        "pb.cli.commands.do.pick_single_choice",
        return_value="practise 'practice piano scales'",
    ), patch("pb.cli.commands.do.run_internal_command") as mock_run:
        do_mod.do_command(ctx, ["practice", "piano", "scales"])

    mock_run.assert_called_once()
    invoked = mock_run.call_args[0][1]
    assert invoked.startswith("practise ")


def test_learn_force_practise_routes_to_practise_session(temp_db, temp_config):
    """`pb learn -p ...` forces the practise branch."""
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    practise_draft = PractisePlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="practise",
                subject_scope="leetcode arrays",
                duration_minutes=20,
                practice_stage="integrate",
                drill_type="timed problem",
                constraint="No hints for the first attempt",
                feedback_source="tests",
                evidence_target="passing solution",
                coach_cues="Explain the invariant aloud",
                success_check="One correct solution",
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(practise_draft)
    with start_patch, stop_patch, require_patch, generate_patch:
        result = runner.invoke(app, ["learn", "--yes", "-p", "leetcode", "arrays"])
    assert result.exit_code == 0, result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    assert session.branch == "practise"


def test_study_time_compat_routes_to_study_plan(temp_db, temp_config):
    """Legacy `pb study --time` still works through the new `study plan` surface."""
    vault = Path(temp_config.general.vault_path)
    domain_dir = vault / "knowledge" / "math"
    domain_dir.mkdir(parents=True, exist_ok=True)
    today = datetime.utcnow().date().isoformat()
    (domain_dir / "_state.md").write_text(f"---\nlast_activity: {today}\n---\n")
    (domain_dir / "note-learning.md").write_text("---\nlearning_stage: '#learning'\n---\n\nContent.")

    runner = _runner()
    result = runner.invoke(app, ["study", "--time", "10"])
    assert result.exit_code == 0, result.output
    assert "Study Plan (10 min)" in result.output
    assert "pb study math" in result.output


def test_study_steps_preview_persists_metadata(temp_db, temp_config):
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    study_draft = StudyPlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="study",
                subject_scope="linear algebra",
                duration_minutes=30,
                target_bloom_stage="apply",
                study_mode="active recall",
                success_check="Explain the core idea without rereading.",
                reason="Lock in the conceptual structure first.",
                steps=[
                    InstructionStep(
                        title="Recall definitions",
                        instruction="Define eigenvalue and eigenvector from memory.",
                        success_check="You can say both definitions cleanly.",
                    ),
                    InstructionStep(
                        title="Work one example",
                        instruction={"text": "$Ax = \\lambda x$", "is_latex": True},
                        success_check="Explain what each symbol means.",
                    ),
                ],
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(study_draft)
    with start_patch, stop_patch, require_patch, generate_patch, patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["study", "--yes", "--steps", "linear", "algebra"])
    assert result.exit_code == 0, result.output
    assert "Recall definitions" in result.output
    assert "Work one example" in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    task = repo.get_task(session.task_id)
    assert task is not None
    assert "PB_STEPS_JSON:" in task.description
    assert "## Steps" in task.description


def test_practise_steps_preview_persists_metadata(temp_db, temp_config):
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    practise_draft = PractisePlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="practise",
                subject_scope="jazz transposition",
                duration_minutes=25,
                practice_stage="integrate",
                drill_type="Barry Harris drill",
                constraint="Stay in one key center before modulating.",
                feedback_source="self",
                evidence_target="One clean transposition cycle.",
                coach_cues="Name the chord function aloud.",
                success_check="One clean deliberate-practice block.",
                steps=[
                    InstructionStep(
                        title="Anchor the scale",
                        instruction="Start from Cdim7 and sing each leading tone.",
                        success_check="You can name each tone before playing it.",
                    ),
                ],
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(practise_draft)
    with start_patch, stop_patch, require_patch, generate_patch, patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["practise", "--yes", "--steps", "jazz", "transposition"])
    assert result.exit_code == 0, result.output
    assert "Anchor the scale" in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    task = repo.get_task(session.task_id)
    assert task is not None
    assert "PB_STEPS_JSON:" in task.description


def test_learn_steps_routes_through_and_persists_step_metadata(temp_db, temp_config):
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    study_draft = StudyPlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="study",
                subject_scope="bayes rule",
                duration_minutes=20,
                target_bloom_stage="apply",
                study_mode="active recall",
                success_check="Apply Bayes rule once from memory.",
                reason="Move from recognition to application.",
                steps=[
                    InstructionStep(
                        title="State the formula",
                        instruction={"text": "$P(A|B)=\\\\frac{P(B|A)P(A)}{P(B)}$", "is_latex": True},
                        success_check="You can explain each term aloud.",
                    ),
                ],
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(study_draft)
    with start_patch, stop_patch, require_patch, generate_patch, patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["learn", "--yes", "--steps", "bayes", "rule"])
    assert result.exit_code == 0, result.output
    assert "State the formula" in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None
    task = repo.get_task(session.task_id)
    assert task is not None
    assert "PB_STEPS_JSON:" in task.description


def test_teach_creates_tracked_session_and_runs_teaching_loop(temp_db, temp_config):
    runner = _runner()
    start_patch, stop_patch = _session_patches()
    vault = Path(temp_config.general.vault_path)
    domain_dir = vault / "knowledge" / "bayes-rule"
    domain_dir.mkdir(parents=True, exist_ok=True)
    (domain_dir / "_state.md").write_text("# Bayes Rule\n")
    teach_draft = StudyPlanDraft(
        summary="",
        blocks=[
            LearningPlanBlockDraft(
                branch="study",
                subject_scope="bayes rule",
                duration_minutes=30,
                target_bloom_stage="apply",
                study_mode="feynman_teach",
                success_check="Answer one transfer question.",
                reason="Teach the concept interactively.",
                steps=[
                    InstructionStep(
                        title="Anchor prior knowledge",
                        instruction="Ask what conditional probability already feels familiar.",
                        success_check="The learner names one anchor concept.",
                    ),
                ],
            )
        ],
    )
    require_patch, generate_patch = _mock_draft_runtime(teach_draft)
    with start_patch, stop_patch, require_patch, generate_patch, patch(
        "pb.cli.markdown.resolve_glow_binary",
        return_value=None,
    ):
        result = runner.invoke(app, ["teach", "--yes", "bayes", "rule"])
    assert result.exit_code == 0, result.output
    # In --yes / non-TTY mode, teach exits after session creation without entering the loop.
    assert any(
        phrase in result.output
        for phrase in ("Teach session is ready", "Interactive lesson starting")
    ), f"Expected session confirmation in output, got: {result.output}"
    assert "[Apply]" not in result.output

    repo = Repository()
    session = repo.get_active_session()
    assert session is not None

    tasks = [
        task
        for task in repo.list_tasks(include_archived=True)
        if task.generated_names.get("display_title") == "Bayes rule"
    ]
    assert len(tasks) == 1
    task = tasks[0]
    assert task.title == "Bayes rule"
    assert task.state != TaskState.DONE
    assert "PB_STUDY_MODE: feynman_teach" in task.description

    lesson_notes = list(domain_dir.glob("*.md"))
    assert all(note.name == "_state.md" for note in lesson_notes)


def test_do_surfaces_teach_for_teach_like_intent(temp_db, temp_config):
    runner = _runner()
    with patch("pb.core.action_routing.rerank_candidates_with_gemini", side_effect=lambda intent, items: list(items)):
        result = runner.invoke(app, ["do", "teach", "me", "bayes", "rule"])
    assert result.exit_code == 0, result.output
    assert "pb teach" in result.output


def test_next_renders_human_labels_for_learning_actions(temp_db, temp_config):
    runner = _runner()
    repo = Repository()
    from pb.core.models import GoalArc

    repo.create_goal_arc(
        GoalArc(
            title="Bayes mastery",
            domain="bayes rule",
            execution_mode="study",
            target_bloom_stage="apply",
        )
    )

    result = runner.invoke(app, ["next"])
    assert result.exit_code == 0, result.output
    assert "Next directions" in result.output
    assert "because" in result.output
    assert "pb teach" not in result.output
