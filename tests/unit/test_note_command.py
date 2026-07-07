"""Unit tests for reworked pb note command (Plan 24-03, Task 2).

Tests verify note.py is a top-level verb (not a subcommand group) with:
- --concept/--person/--opp flag routes to QuestionTreeEngine flow
- topic argument triggers SocraticService debrief
- --quick, --sync, --flash, --trust flag wiring
- Old socratic and book subcommands removed
- Back-compat concept/person/opp subcommands retained

All 11 tests per plan spec.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pb.cli.commands.note import app


# ---------------------------------------------------------------------------
# Shared fake factory / service setup
# ---------------------------------------------------------------------------

def _make_fake_service():
    """Return a MagicMock with the SocraticService interface."""
    svc = MagicMock()
    svc.detect_domain.return_value = "ml"
    svc.run_note_debrief.return_value = [("Q1", "A1"), ("Q2", "A2")]
    svc.build_and_submit.return_value = "knowledge/ml/some-slug.md"
    svc.suggest_bridge.return_value = None
    return svc


def _make_ctx_obj(svc=None):
    """Return a ctx.obj dict with a fake socratic_service factory."""
    if svc is None:
        svc = _make_fake_service()
    return {
        "factory": {
            "socratic_service": lambda: svc,
        },
        "repo": MagicMock(),
        "config": MagicMock(),
        "vault_cwd": None,
    }


def _note_patches(tmp_path):
    """Return context manager stack for the common note capture path.

    Patches are applied at source modules since note.py uses lazy in-function imports.
    sys.stdin.isatty is patched to True so the non-TTY guard in note_capture is bypassed.
    """
    return [
        patch("pb.vault.config.get_vault_path", return_value=tmp_path),
        patch("pb.cli.commands.note._pick_session_depth", return_value=3),
        patch("pb.cli.commands.note._pick_domain_for_note", return_value="ml"),
        patch("pb.core.graph_writer.make_slug", return_value="test-slug"),
        patch("pb.cli.commands.note._is_interactive", return_value=True),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNoteFlagRoutes:
    """Tests 1-3: Flag routes to _run_question_tree."""

    def _invoke_with_flag(self, flag):
        runner = CliRunner()
        with patch("pb.cli.commands.note._run_question_tree") as mock_tree:
            result = runner.invoke(app, [flag], catch_exceptions=False,
                                   obj=_make_ctx_obj())
            return result, mock_tree

    def test_concept_flag_routes_to_question_tree(self):
        """Test 1: --concept routes to _run_question_tree('concept')."""
        result, mock_tree = self._invoke_with_flag("--concept")
        mock_tree.assert_called_once_with("concept")

    def test_person_flag_routes_to_question_tree(self):
        """Test 2: --person routes to _run_question_tree('person')."""
        result, mock_tree = self._invoke_with_flag("--person")
        mock_tree.assert_called_once_with("person")

    def test_opp_flag_routes_to_question_tree(self):
        """Test 3: --opp routes to _run_question_tree('opportunity')."""
        result, mock_tree = self._invoke_with_flag("--opp")
        mock_tree.assert_called_once_with("opportunity")


class TestNoteTopicDebrief:
    """Tests 4-8: Topic argument triggers SocraticService debrief."""

    def test_topic_calls_run_note_debrief(self, tmp_path):
        """Test 4: pb note VAE calls socratic_service.run_note_debrief(topic='VAE', ...)."""
        svc = _make_fake_service()
        runner = CliRunner()
        patches = _note_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = runner.invoke(app, ["VAE"], catch_exceptions=False,
                                   obj=_make_ctx_obj(svc))
        svc.run_note_debrief.assert_called_once()
        call_kwargs = svc.run_note_debrief.call_args
        # topic may be passed as positional or keyword
        topic_val = call_kwargs.kwargs.get("topic")
        if topic_val is None and call_kwargs.args:
            topic_val = call_kwargs.args[0]
        assert topic_val == "VAE", f"Expected topic='VAE', got call: {call_kwargs}"

    def test_quick_flag_skips_debrief(self, tmp_path):
        """Test 5: --quick skips debrief; build_and_submit called with qa_pairs=[]."""
        svc = _make_fake_service()
        runner = CliRunner()
        patches = _note_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            # Flags must precede the variadic topic arg (Typer list arg absorbs trailing tokens)
            result = runner.invoke(app, ["--quick", "VAE"], catch_exceptions=False,
                                   obj=_make_ctx_obj(svc))
        # run_note_debrief must NOT be called
        svc.run_note_debrief.assert_not_called()
        # build_and_submit must be called with empty qa_pairs
        svc.build_and_submit.assert_called_once()
        call_kwargs = svc.build_and_submit.call_args
        qa = call_kwargs.kwargs.get("qa_pairs")
        if qa is None and call_kwargs.args:
            qa = call_kwargs.args[0]
        assert qa == [], f"Expected qa_pairs=[], got: {call_kwargs}"

    def test_sync_flag_passes_sync_true(self, tmp_path):
        """Test 6: --sync results in build_and_submit(sync=True)."""
        svc = _make_fake_service()
        runner = CliRunner()
        patches = _note_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            # Flag before topic (Typer list arg absorbs trailing tokens as part of topic)
            result = runner.invoke(app, ["--sync", "VAE"], catch_exceptions=False,
                                   obj=_make_ctx_obj(svc))
        svc.build_and_submit.assert_called_once()
        call_kwargs = svc.build_and_submit.call_args
        sync_val = call_kwargs.kwargs.get("sync")
        assert sync_val is True, f"Expected sync=True, got: {call_kwargs}"

    def test_flash_flag_uses_flash_model(self, tmp_path):
        """Test 7: --flash passes FLASH_MODEL; default uses FLASH_LITE_MODEL."""
        from pb.llm.gemini import FLASH_MODEL, FLASH_LITE_MODEL
        svc = _make_fake_service()
        runner = CliRunner()
        patches = _note_patches(tmp_path)

        # With --flash (flag before topic to avoid absorption by variadic arg)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            runner.invoke(app, ["--flash", "VAE"], catch_exceptions=False,
                          obj=_make_ctx_obj(svc))
        call_kwargs = svc.build_and_submit.call_args
        model_val = call_kwargs.kwargs.get("model")
        assert model_val == FLASH_MODEL, f"Expected FLASH_MODEL, got: {model_val}"

        # Without --flash: FLASH_LITE_MODEL
        svc2 = _make_fake_service()
        patches2 = _note_patches(tmp_path)
        with patches2[0], patches2[1], patches2[2], patches2[3], patches2[4]:
            runner.invoke(app, ["VAE"], catch_exceptions=False,
                          obj=_make_ctx_obj(svc2))
        call_kwargs2 = svc2.build_and_submit.call_args
        model_val2 = call_kwargs2.kwargs.get("model")
        assert model_val2 == FLASH_LITE_MODEL, f"Expected FLASH_LITE_MODEL, got: {model_val2}"

    def test_bare_invocation_calls_debrief_with_empty_topic(self, tmp_path):
        """Test 8: Bare `pb note` (no topic) calls run_note_debrief with topic=''."""
        svc = _make_fake_service()
        runner = CliRunner()
        patches = _note_patches(tmp_path)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            result = runner.invoke(app, [], catch_exceptions=False,
                                   obj=_make_ctx_obj(svc))
        svc.run_note_debrief.assert_called_once()
        call_kwargs = svc.run_note_debrief.call_args
        topic_val = call_kwargs.kwargs.get("topic")
        if topic_val is None and call_kwargs.args:
            topic_val = call_kwargs.args[0]
        assert topic_val == "", f"Expected topic='', got: {call_kwargs}"


class TestNoteSubcommandRemoval:
    """Tests 9-10: Old subcommands removed."""

    def test_socratic_subcommand_removed(self):
        """Test 9: 'socratic' is no longer a registered subcommand."""
        runner = CliRunner()
        result = runner.invoke(app, ["socratic"], obj=_make_ctx_obj())
        # 'socratic' should not be found as a subcommand
        registered_names = [cmd.name for cmd in app.registered_commands]
        assert "socratic" not in registered_names
        # Invoking it should fail (no such command)
        assert result.exit_code != 0 or "No such command" in result.output

    def test_book_subcommand_removed(self):
        """Test 10: 'book' is no longer a registered subcommand."""
        runner = CliRunner()
        result = runner.invoke(app, ["book"], obj=_make_ctx_obj())
        registered_names = [cmd.name for cmd in app.registered_commands]
        assert "book" not in registered_names
        assert result.exit_code != 0 or "No such command" in result.output


class TestNoteBackcompatSubcommands:
    """Test 11: Back-compat concept/person/opp subcommands still exist."""

    def test_concept_person_opp_subcommands_registered(self):
        """Test 11: concept, person, opp are still registered as subcommands."""
        registered = [cmd.name for cmd in app.registered_commands]
        assert "concept" in registered, f"'concept' missing from {registered}"
        assert "person" in registered, f"'person' missing from {registered}"
        assert "opp" in registered, f"'opp' missing from {registered}"
