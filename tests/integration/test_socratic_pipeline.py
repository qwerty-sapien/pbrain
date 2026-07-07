"""Integration tests for the Socratic capture pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pb.cli.commands.note import app as note_app
from pb.cli.commands.study import app as study_app


def _make_vault(tmp_path: Path, domains=("ml", "piano")) -> Path:
    for domain in domains:
        domain_dir = tmp_path / "knowledge" / domain
        domain_dir.mkdir(parents=True, exist_ok=True)
        (domain_dir / "_state.md").write_text(f"# {domain} state\nKey concepts: ...")
    return tmp_path


def _make_fake_service(*, domain="ml", qa_pairs=None, note_path="knowledge/ml/some-slug.md"):
    if qa_pairs is None:
        qa_pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3")]
    svc = MagicMock()
    svc.detect_domain.return_value = domain
    svc.run_note_debrief.return_value = qa_pairs
    svc.run_study_debrief.return_value = qa_pairs
    svc.run_finish_debrief.return_value = qa_pairs
    svc.build_and_submit.return_value = note_path
    svc.suggest_bridge.return_value = None
    return svc


def _make_ctx_obj(svc):
    return {
        "factory": {"socratic_service": lambda: svc},
        "repo": MagicMock(),
        "config": MagicMock(),
        "vault_cwd": None,
    }


class TestPbNoteSyncEndToEnd:
    def test_pb_note_end_to_end_sync(self, tmp_path):
        vault = _make_vault(tmp_path)
        svc = _make_fake_service(domain="ml", note_path="knowledge/ml/vae-note.md")

        runner = CliRunner()
        with (
            patch("pb.vault.get_vault_path", return_value=vault),
            patch("pb.cli.commands.note._pick_session_depth", return_value=3),
            patch("pb.cli.commands.note._pick_domain_for_note", return_value="ml"),
            patch("pb.core.graph_writer.make_slug", return_value="vae-note"),
            patch("pb.cli.commands.note._is_interactive", return_value=True),
            patch("subprocess.run"),
        ):
            result = runner.invoke(note_app, ["--sync", "VAE"], catch_exceptions=False, obj=_make_ctx_obj(svc))

        assert result.exit_code == 0, result.output
        svc.run_note_debrief.assert_called_once()
        svc.build_and_submit.assert_called_once()
        assert svc.build_and_submit.call_args.kwargs.get("sync") is True
        assert svc.build_and_submit.call_args.kwargs.get("template") == "brief"


class TestPbStudyDebriefSyncEndToEnd:
    def test_pb_study_debrief_end_to_end_sync(self, tmp_path):
        vault = _make_vault(tmp_path)
        qa_pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4"), ("Q5", "A5")]
        svc = _make_fake_service(domain="piano", qa_pairs=qa_pairs, note_path="knowledge/piano/deep-note.md")
        svc.detect_domain.return_value = None

        runner = CliRunner()
        with (
            patch("pb.vault.get_vault_path", return_value=vault),
            patch("pb.cli.commands.learn._pick_domain", return_value="piano"),
            patch("pb.core.graph_writer.make_slug", return_value="deep-note"),
            patch("pb.cli.commands.study._is_interactive", return_value=True),
            patch("subprocess.run"),
        ):
            result = runner.invoke(
                study_app,
                ["debrief", "--domain", "piano", "--sync"],
                catch_exceptions=False,
                obj=_make_ctx_obj(svc),
            )

        assert result.exit_code == 0, result.output
        svc.run_study_debrief.assert_called_once()
        assert svc.run_study_debrief.call_args.kwargs.get("domain") == "piano"
        svc.build_and_submit.assert_called_once()
        assert svc.build_and_submit.call_args.kwargs.get("template") == "deep"
        assert svc.build_and_submit.call_args.kwargs.get("sync") is True


class TestPbNoteQuickSkipsDebrief:
    def test_pb_note_quick_skips_debrief(self, tmp_path):
        vault = _make_vault(tmp_path)
        svc = _make_fake_service(domain="ml", qa_pairs=[], note_path="knowledge/ml/stub.md")

        def _must_not_be_called(*args, **kwargs):
            raise AssertionError("run_note_debrief must NOT be called with --quick")

        svc.run_note_debrief.side_effect = _must_not_be_called

        runner = CliRunner()
        with (
            patch("pb.vault.get_vault_path", return_value=vault),
            patch("pb.cli.commands.note._pick_session_depth", return_value=3),
            patch("pb.cli.commands.note._pick_domain_for_note", return_value="ml"),
            patch("pb.core.graph_writer.make_slug", return_value="stub"),
            patch("pb.cli.commands.note._is_interactive", return_value=True),
            patch("subprocess.run"),
        ):
            result = runner.invoke(note_app, ["--quick", "Some idea"], catch_exceptions=False, obj=_make_ctx_obj(svc))

        assert result.exit_code == 0, result.output
        svc.run_note_debrief.assert_not_called()
        assert svc.build_and_submit.call_args.kwargs.get("qa_pairs") == []


class TestPbNoteBridgeGate:
    def test_pb_note_long_mode_triggers_bridge_when_qa_pairs_ge_4(self, tmp_path):
        vault = _make_vault(tmp_path)
        qa_pairs = [("Q1", "A1"), ("Q2", "A2"), ("Q3", "A3"), ("Q4", "A4")]
        svc = _make_fake_service(domain="piano", qa_pairs=qa_pairs, note_path="knowledge/piano/bridge-test.md")

        runner = CliRunner()
        with (
            patch("pb.vault.get_vault_path", return_value=vault),
            patch("pb.cli.commands.note._pick_session_depth", return_value=12),
            patch("pb.cli.commands.note._pick_domain_for_note", return_value="piano"),
            patch("pb.core.graph_writer.make_slug", return_value="bridge-test"),
            patch("pb.cli.commands.note._is_interactive", return_value=True),
            patch("subprocess.run"),
        ):
            result = runner.invoke(note_app, ["--sync", "Counterpoint"], catch_exceptions=False, obj=_make_ctx_obj(svc))

        assert result.exit_code == 0, result.output
        svc.suggest_bridge.assert_called_once()


class TestPbFinishDebriefEndToEnd:
    def test_pb_finish_debrief_end_to_end(self, tmp_path):
        from pb.cli.commands.execute import app as execute_app

        vault = _make_vault(tmp_path)
        svc = _make_fake_service(domain="ml", note_path="knowledge/ml/debrief-note.md")

        fake_task = MagicMock()
        fake_task.id = 1
        fake_task.title = "ML task"
        fake_task.domain = "ml"
        fake_session = MagicMock()
        fake_session.task_id = 1
        fake_session.domain = "ml"
        fake_session.branch = "study"

        fake_session_service = MagicMock()
        fake_session_service.get_current_session.return_value = fake_session
        fake_session_service.finish_session.return_value = fake_session

        fake_repo = MagicMock()
        fake_repo.get_task.return_value = fake_task

        ctx_obj = {
            "factory": {
                "socratic_service": lambda: svc,
                "session_service": lambda: fake_session_service,
            },
            "repo": fake_repo,
            "config": MagicMock(),
            "vault_cwd": None,
        }

        runner = CliRunner()
        with (
            patch("pb.vault.config.get_vault_path", return_value=vault),
            patch("pb.core.graph_writer.make_slug", return_value="debrief-note"),
            patch("subprocess.run"),
        ):
            result = runner.invoke(execute_app, ["finish", "--debrief"], catch_exceptions=False, obj=ctx_obj)

        assert result.exit_code == 0, result.output
        svc.run_finish_debrief.assert_called_once()
        assert svc.build_and_submit.call_args.kwargs.get("template") == "brief"
