from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from pb.cli.commands.execute import finish_task
from pb.core.session_blueprints import resolve_learning_session_blueprint


def _ctx_obj(branch: str = "study"):
    session = MagicMock()
    session.task_id = "task-123"
    session.branch = branch
    session.goal_id = None
    session.target_bloom_stage = None
    session.practice_stage = None
    session.actual_outcome = None
    session.observed_errors = None
    session.next_adjustment = None

    task = MagicMock()
    task.title = "Study charisma"
    task.completion = 0
    task.project_id = None

    session_service = MagicMock()
    session_service.get_current_session.return_value = session
    session_service.finish_session.return_value = session

    repo = MagicMock()
    repo.get_task.return_value = task

    return {
        "factory": {"session_service": lambda: session_service},
        "repo": repo,
        "runtime": SimpleNamespace(vault_path=Path("/tmp/vault"), data_dir=Path("/tmp/data")),
    }, session, repo


def test_finish_study_does_not_ask_hardcoded_checkin_prompts():
    ctx_obj, session, repo = _ctx_obj("study")
    ctx = SimpleNamespace(obj=ctx_obj)

    with patch("pb.cli.commands.execute.sys.stdin.isatty", return_value=True), \
         patch("pb.cli.commands.execute.typer.prompt") as prompt_mock, \
         patch("pb.core.finish_assessment.FinishAssessmentAgent.is_available", return_value=False), \
         patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
         patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
         patch("pb.core.graph_writer.GraphWriter.write_task_note", return_value=None), \
         patch("pb.core.session_log_writer.SessionLogWriter.write_session_log", return_value=None):
        finish_task(ctx, note_words=None, completion=None, yes=False, debrief=False, skip=False)

    prompt_mock.assert_not_called()
    assert "finish_checkin_qa" not in (getattr(session, "generated_names", {}) or {})
    repo.update_session.assert_not_called()


def test_finish_with_inline_note_uses_note_without_followup_prompts():
    ctx_obj, session, repo = _ctx_obj("practise")
    ctx = SimpleNamespace(obj=ctx_obj)

    with patch("pb.cli.commands.execute.sys.stdin.isatty", return_value=True), \
         patch("pb.cli.commands.execute.typer.prompt") as prompt_mock, \
         patch("pb.core.finish_assessment.FinishAssessmentAgent.is_available", return_value=False), \
         patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
         patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
         patch("pb.core.graph_writer.GraphWriter.write_task_note", return_value=None), \
         patch("pb.core.session_log_writer.SessionLogWriter.write_session_log", return_value=None):
        finish_task(
            ctx,
            note_words=["locked", "in", "the", "slower", "tempo"],
            completion=None,
            yes=False,
            debrief=False,
            skip=False,
        )

    prompt_mock.assert_not_called()
    ctx_obj["factory"]["session_service"]().finish_session.assert_called_once_with(
        note="locked in the slower tempo",
        completion_pct=100,
    )
    assert "finish_checkin_qa" not in (getattr(session, "generated_names", {}) or {})
    repo.update_session.assert_not_called()


def test_finish_skip_yes_does_not_prompt_for_lightweight_summary():
    ctx_obj, session, repo = _ctx_obj("study")
    ctx_obj["yes"] = True
    ctx = SimpleNamespace(obj=ctx_obj)

    with patch("pb.cli.commands.execute.sys.stdin.isatty", return_value=True), \
         patch("pb.cli.commands.execute.confirm_choice", side_effect=AssertionError("skip --yes must not prompt")), \
         patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
         patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
         patch("pb.core.graph_writer.GraphWriter.write_task_note", return_value=None), \
         patch("pb.core.session_log_writer.SessionLogWriter.write_session_log", return_value=None):
        finish_task(
            ctx,
            note_words=["kept", "manual", "evidence"],
            completion=100,
            yes=True,
            debrief=False,
            skip=True,
        )

    ctx_obj["factory"]["session_service"]().finish_session.assert_called_once_with(
        note="kept manual evidence",
        completion_pct=100,
    )


def test_finish_triggers_partner_memory_compaction_before_evidence():
    ctx_obj, session, repo = _ctx_obj("study")
    session.generated_names = {"learning_partner_used": True}
    ctx = SimpleNamespace(obj=ctx_obj)

    with patch("pb.cli.commands.execute.runtime_for_ctx", return_value=MagicMock()), \
         patch("pb.cli.commands.execute.sys.stdin.isatty", return_value=False), \
         patch("pb.core.learner_memory.append_partner_session_memory", return_value=Path("/tmp/vault/80-logs/task-memory/gr.tsv")) as compact_mock, \
         patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
         patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
         patch("pb.core.graph_writer.GraphWriter.write_task_note", return_value=None), \
         patch("pb.core.session_log_writer.SessionLogWriter.write_session_log", return_value=None):
        finish_task(ctx, note_words=None, completion=None, yes=False, debrief=False, skip=False)

    compact_mock.assert_called_once()


def test_finish_ignores_blueprint_specific_closeout_prompts():
    ctx_obj, session, repo = _ctx_obj("practise")
    ctx = SimpleNamespace(obj=ctx_obj)
    resolution = resolve_learning_session_blueprint(
        branch="practise",
        domain="cardistry",
        topic="Biddle grip",
        drill="static hold",
    )
    session.generated_names = {
        "domain_pack_id": resolution.pack_id,
        "session_blueprint": resolution.blueprint.model_dump(mode="json"),
        "learning_partner_evidence": [],
    }
    with patch("pb.cli.commands.execute.sys.stdin.isatty", return_value=True), \
         patch("pb.cli.commands.execute.typer.prompt") as prompt_mock, \
         patch("pb.core.finish_assessment.FinishAssessmentAgent.is_available", return_value=False), \
         patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
         patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
         patch("pb.core.graph_writer.GraphWriter.write_task_note", return_value=None), \
         patch("pb.core.session_log_writer.SessionLogWriter.write_session_log", return_value=None):
        finish_task(ctx, note_words=None, completion=None, yes=False, debrief=False, skip=False)

    prompt_mock.assert_not_called()
    assert "finish_checkin_qa" not in (getattr(session, "generated_names", {}) or {})
