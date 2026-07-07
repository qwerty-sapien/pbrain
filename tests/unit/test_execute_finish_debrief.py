"""Unit tests for execute.py finish --debrief service delegation (Plan 24-03, Task 3).

Tests verify the --debrief block in finish_task() delegates to SocraticService:
- Test 6: run_finish_debrief called when --debrief flag set and domain session active
- Test 7: if run_finish_debrief raises, finish does not crash (non-fatal)
- Test 8: old broken SocraticDebriefEngine() no-arg call is gone from execute.py

INV-5 compliance: no engine = SocraticDebriefEngine() in execute.py debrief block.
"""
from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

import pb.cli.commands.execute as execute_module
from pb.cli.commands.execute import app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fake_session(task_id="task-abc"):
    session = MagicMock()
    session.task_id = task_id
    session.id = "session-xyz"
    session.domain = "ml"
    return session


def _make_fake_task(task_id="task-abc"):
    task = MagicMock()
    task.id = task_id
    task.title = "Study VAE"
    task.domain = "ml"
    task.archived_at = None
    return task


def _make_socratic_svc():
    svc = MagicMock()
    svc.run_finish_debrief.return_value = [("Q1", "A1"), ("Q2", "A2")]
    return svc


def _make_ctx_obj(session=None, task=None, socratic_svc=None):
    """Build a minimal ctx.obj for finish_task invocation."""
    if session is None:
        session = _make_fake_session()
    if task is None:
        task = _make_fake_task()
    if socratic_svc is None:
        socratic_svc = _make_socratic_svc()

    session_service = MagicMock()
    session_service.get_current_session.return_value = session
    session_service.finish_session.return_value = session
    session_service.get_elapsed_minutes.return_value = 30

    repo = MagicMock()
    repo.get_task.return_value = task

    return {
        "factory": {
            "session_service": lambda: session_service,
            "socratic_service": lambda: socratic_svc,
        },
        "repo": repo,
        "config": MagicMock(),
        "vault_cwd": None,
    }


def _finish_patches():
    """Patches to suppress side effects in finish_task that aren't under test."""
    return [
        patch("pb.llm.gemini.GeminiClient.is_available", return_value=False),
        patch("pb.cli.commands.execute._check_timer_expiry", side_effect=None),
    ]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestFinishDebriefDelegation:
    """Tests 6-7: --debrief delegates to SocraticService."""

    def test_debrief_calls_run_finish_debrief(self):
        """Test 6: pb finish --debrief calls socratic_service.run_finish_debrief(...)."""
        socratic_svc = _make_socratic_svc()
        ctx_obj = _make_ctx_obj(socratic_svc=socratic_svc)
        runner = CliRunner()

        # write_session_packet doesn't exist yet (packet.py is empty); the try/except
        # in finish_task absorbs the ImportError non-fatally — no need to patch it.
        with patch("pb.vault.graph_store.open_vault_db", side_effect=Exception("skip")), \
             patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
             patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
             patch("pb.core.graph_writer.make_slug", return_value="finish-slug"):
            result = runner.invoke(app, ["finish", "--debrief"],
                                   catch_exceptions=False, obj=ctx_obj)

        # The debrief service call must have been attempted
        socratic_svc.run_finish_debrief.assert_called_once()

    def test_debrief_exception_does_not_crash_finish(self):
        """Test 7: if run_finish_debrief raises, finish exits 0 (non-fatal)."""
        socratic_svc = _make_socratic_svc()
        socratic_svc.run_finish_debrief.side_effect = RuntimeError("LLM unavailable")
        ctx_obj = _make_ctx_obj(socratic_svc=socratic_svc)
        runner = CliRunner()

        with patch("pb.vault.graph_store.open_vault_db", side_effect=Exception("skip")), \
             patch("pb.vault.config.get_vault_path", return_value=Path("/tmp/vault")), \
             patch("pb.core.graph_writer.GraphWriter.update_state_md", return_value=None), \
             patch("pb.core.graph_writer.make_slug", return_value="finish-slug"):
            result = runner.invoke(app, ["finish", "--debrief"],
                                   catch_exceptions=False, obj=ctx_obj)

        # finish must succeed regardless of debrief failure
        assert result.exit_code == 0, f"finish crashed: {result.output}"


class TestFinishDebriefInv5:
    """Test 8: Broken SocraticDebriefEngine() stub is gone from execute.py."""

    def test_broken_engine_stub_removed(self):
        """Test 8: 'engine = SocraticDebriefEngine()' not in execute.py."""
        source = inspect.getsource(execute_module)
        assert "engine = SocraticDebriefEngine()" not in source, (
            "execute.py still contains the broken no-arg SocraticDebriefEngine() stub"
        )
        assert "engine.run_debrief(" not in source, (
            "execute.py still contains the broken engine.run_debrief() call"
        )
