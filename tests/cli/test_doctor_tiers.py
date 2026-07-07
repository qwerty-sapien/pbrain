# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Deterministic offline tests for pb doctor --tiers per-tier iteration and surgical failure reporting.

Tests:
- test_tiers_all_ok: all bindings probe OK, exit 0, each distinct binding probed exactly once
- test_tiers_one_fails_surgical: one tier fails, exit 53, only broken tier surfaced, no save_config
- test_default_doctor_unaffected: no --tiers flag, healthy vault+SQLite, exit 0 (Phase-4 contract)
"""

from __future__ import annotations

import tempfile
import json
from dataclasses import dataclass
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pb.cli.commands.doctor import _verify_tiers, doctor_command
from pb.core.exceptions import ExitCode
from pb.storage.config import Config, GeneralConfig, ModelRolesConfig


# ---------------------------------------------------------------------------
# Fake LLMProbeResult — avoids importing the dataclass directly and keeps
# tests insulated from runtime changes.
# ---------------------------------------------------------------------------

@dataclass
class FakeProbeResult:
    available: bool
    provider: str = "gemini"
    backend: str = "gemini"
    model: str = "gemini-3-flash-preview"
    credential_source: str = "env"
    category: str = "ok"
    message: str = "Live gemini probe succeeded."
    debug_message: str = ""
    http_status: int | None = None
    retryable: bool = False


def _make_config(
    default: str = "",
    planner: str = "",
    reviewer: str = "",
    recall: str = "",
    fast: str = "",
    fast_inference: str = "",
    namer: str = "",
    vault_path: str = "",
) -> Config:
    """Build a minimal Config with the given model_roles bindings."""
    return Config(
        general=GeneralConfig(vault_path=vault_path or None),
        model_roles=ModelRolesConfig(
            default=default,
            planner=planner,
            reviewer=reviewer,
            recall=recall,
            fast=fast,
            fast_inference=fast_inference,
            namer=namer,
        ),
    )


def _make_console() -> MagicMock:
    """Return a console stub that captures print calls."""
    console = MagicMock()
    console.rule = MagicMock()
    console.print = MagicMock()
    return console


# ---------------------------------------------------------------------------
# Test 1: All tiers probe OK — exit 0, each distinct binding probed once
# ---------------------------------------------------------------------------

def test_tiers_all_ok(monkeypatch):
    """All configured model-role bindings probe OK: exit code 0, de-dup proven."""
    # Three distinct bindings — fast and fast_inference share one; default is unique
    config = _make_config(
        default="gemini:gemini-3.1-pro-preview",
        planner="gemini:gemini-3.1-pro-preview",     # same as default → de-dup
        fast="gemini:gemini-3.1-flash-lite-preview",
        fast_inference="gemini:gemini-3.1-flash-lite-preview",  # same as fast → de-dup
        namer="gemini:gemini-3-flash-preview",
    )
    # Distinct bindings: pro (default+planner), flash-lite (fast+fast_inference), flash (namer) = 3
    expected_distinct = 3

    probe_calls: list[str] = []

    def fake_live_probe(*, model: str | None = None, timeout: int = 12) -> FakeProbeResult:
        probe_calls.append(model or "")
        return FakeProbeResult(available=True, model=model or "")

    runtime = MagicMock()
    runtime.live_probe = fake_live_probe

    console = _make_console()
    result = _verify_tiers(config, runtime, json_out=False, console=console)

    assert result == ExitCode.SUCCESS, f"Expected exit 0, got {result}"
    # De-dup: each distinct binding probed exactly once
    assert len(probe_calls) == expected_distinct, (
        f"Expected {expected_distinct} probes (de-duped), got {len(probe_calls)}: {probe_calls}"
    )


# ---------------------------------------------------------------------------
# Test 2: One tier fails — exit 53, only broken tier surfaced, save_config NOT called
# ---------------------------------------------------------------------------

def test_tiers_one_fails_surgical(monkeypatch):
    """Planner tier fails: exit 53, output names the broken tier, no save_config side effect."""
    config = _make_config(
        default="gemini:gemini-3-flash-preview",
        planner="gemini:gemini-3.1-pro-preview",  # this one fails
        fast="gemini:gemini-3.1-flash-lite-preview",
    )

    PLANNER_BINDING = "gemini:gemini-3.1-pro-preview"

    def fake_live_probe(*, model: str | None = None, timeout: int = 12) -> FakeProbeResult:
        if model == PLANNER_BINDING:
            return FakeProbeResult(
                available=False,
                model=model or "",
                category="auth",
                message="API key missing or invalid.",
            )
        return FakeProbeResult(available=True, model=model or "")

    runtime = MagicMock()
    runtime.live_probe = fake_live_probe

    console = _make_console()

    with patch("pb.cli.commands.doctor.save_config") as mock_save:
        result = _verify_tiers(config, runtime, json_out=False, console=console)
        # save_config must NOT be called (T-07-07 / D-04 surgical — no side effects)
        mock_save.assert_not_called()

    assert result == ExitCode.CONFIG_ERROR, f"Expected exit 53, got {result}"

    # Verify the broken tier name ("planner") appears in output
    all_print_calls = " ".join(
        str(arg) for call_args in console.print.call_args_list for arg in call_args[0]
    )
    assert "planner" in all_print_calls, (
        f"Expected 'planner' in output, got: {all_print_calls!r}"
    )

    # Ensure no "reconfigure" / full-reset phrase is in the output (surgical only)
    reconfigure_phrases = ["reconfigure everything", "full reconfigure", "reconfigure all"]
    for phrase in reconfigure_phrases:
        assert phrase.lower() not in all_print_calls.lower(), (
            f"Found forbidden phrase '{phrase}' in output — must be surgical, not a full reconfigure"
        )


# ---------------------------------------------------------------------------
# Test 3: Default pb doctor (no --tiers) exits 0 when vault+SQLite healthy
#          even with absent LLM creds (Phase-4 release-gate contract)
# ---------------------------------------------------------------------------

def test_default_doctor_unaffected(tmp_path, monkeypatch):
    """pb doctor with no flags exits 0 on healthy vault+SQLite even without LLM creds."""
    import typer

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()

    config = _make_config(vault_path=str(vault_dir))

    # ctx stub — provides the config via ctx.obj
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = {"config": config}

    # LLMRuntime.health() → not available (no creds) — must not affect exit code
    fake_health = MagicMock()
    fake_health.available = False
    fake_health.provider = ""
    fake_health.credential_source = "none"
    fake_health.default_model = ""
    fake_health.structured_output = False
    fake_health.message = "No provider configured."

    # data_dir and db must exist for the required checks to pass
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_parent = tmp_path / "db"
    db_parent.mkdir()

    with (
        patch("pb.cli.commands.doctor.LLMRuntime") as mock_runtime_cls,
        patch("pb.cli.commands.doctor.get_vault_path", return_value=vault_dir),
        patch("pb.cli.commands.doctor.get_data_dir", return_value=data_dir),
        patch("pb.cli.commands.doctor.get_quarantine_path", return_value=vault_dir / ".quarantine"),
        patch("pb.cli.commands.doctor.get_db_path", return_value=db_parent / "brain.db"),
        patch("pb.cli.commands.doctor.get_console", return_value=_make_console()),
        patch("pb.cli.commands.doctor.get_err_console", return_value=_make_console()),
    ):
        mock_runtime = MagicMock()
        mock_runtime.health.return_value = fake_health
        mock_runtime.live_probe.return_value = None  # not called when llm=False
        mock_runtime_cls.return_value = mock_runtime

        # Patch anki so it doesn't fail
        with patch("pb.vault.anki_client.is_anki_available", return_value=False, create=True):
            try:
                doctor_command(ctx, fix=False, json_out=False, llm=False, tiers=False, debug=False)
                exit_code = 0  # no exception → success
            except typer.Exit as exc:
                exit_code = exc.exit_code

    assert exit_code == ExitCode.SUCCESS, (
        f"Default pb doctor must exit 0 when vault+SQLite healthy (Phase-4 contract). Got: {exit_code}"
    )


def test_default_doctor_json_marks_llm_setup_not_live_verified(tmp_path, capsys):
    """Default doctor should not imply live LLM requests are verified."""
    import typer

    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    db_parent = tmp_path / "db"
    db_parent.mkdir()
    config = _make_config(vault_path=str(vault_dir))
    ctx = MagicMock(spec=typer.Context)
    ctx.obj = {"config": config}

    fake_health = MagicMock()
    fake_health.available = True
    fake_health.provider = "gemini"
    fake_health.credential_source = "vertex"
    fake_health.default_model = "gemini-3-flash-preview"
    fake_health.structured_output = True
    fake_health.message = "request can be constructed"

    with (
        patch("pb.cli.commands.doctor.LLMRuntime") as mock_runtime_cls,
        patch("pb.cli.commands.doctor.get_vault_path", return_value=vault_dir),
        patch("pb.cli.commands.doctor.get_data_dir", return_value=data_dir),
        patch("pb.cli.commands.doctor.get_quarantine_path", return_value=vault_dir / ".quarantine"),
        patch("pb.cli.commands.doctor.get_db_path", return_value=db_parent / "brain.db"),
        patch("pb.cli.commands.doctor.get_console", return_value=_make_console()),
        patch("pb.cli.commands.doctor.get_err_console", return_value=_make_console()),
        patch("pb.vault.anki_client.is_anki_available", return_value=False, create=True),
    ):
        mock_runtime = MagicMock()
        mock_runtime.health.return_value = fake_health
        mock_runtime_cls.return_value = mock_runtime
        doctor_command(ctx, fix=False, json_out=True, llm=False, tiers=False, debug=False)

    payload = json.loads(capsys.readouterr().out)
    llm_setup = next(check for check in payload["checks"] if check["label"] == "LLM request setup")
    assert llm_setup["status"] == "WARN"
    assert llm_setup["ok"] is False
    assert "pb doctor --llm" in llm_setup["detail"]
