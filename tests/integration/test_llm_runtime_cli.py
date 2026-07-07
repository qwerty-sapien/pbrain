"""Integration coverage for the LLM runtime management commands."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import call, patch

from typer.testing import CliRunner

from pb.cli.main import app
from pb.llm.runtime import LLMHealth, LLMProbeResult


runner = CliRunner()


def _healthy_runtime() -> LLMHealth:
    return LLMHealth(
        configured=True,
        available=True,
        provider="gemini",
        backend="aistudio",
        default_model="gemini-3-flash-preview",
        structured_output=True,
        credential_source="aistudio",
        message="Gemini credentials are configured.",
    )


def test_doctor_reports_llm_health(temp_db, temp_config):
    """`pb doctor` succeeds when the runtime and local state are healthy."""
    with patch("pb.cli.commands.doctor.LLMRuntime.health", return_value=_healthy_runtime()), patch(
        "pb.vault.anki_client.is_anki_available",
        return_value=True,
    ):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "Doctor" in result.output
    assert "provider configured" in result.output
    assert "LLM request setup" in result.output
    assert "OK Anki integration: AnkiConnect online" in result.output


def test_doctor_fails_when_anki_is_offline(temp_db, temp_config):
    with patch("pb.cli.commands.doctor.LLMRuntime.health", return_value=_healthy_runtime()), patch(
        "pb.vault.anki_client.is_anki_available",
        return_value=False,
    ):
        result = runner.invoke(app, ["doctor"])

    assert result.exit_code == 0, result.output
    assert "WARN Anki integration: AnkiConnect offline" in result.output


def test_model_status_reports_current_model(temp_db, temp_config):
    """`pb model status` shows the configured Gemini model state."""
    with patch("pb.cli.commands.model.LLMRuntime.health", return_value=_healthy_runtime()):
        result = runner.invoke(app, ["model", "status"])

    assert result.exit_code == 0, result.output
    assert "Model Status" in result.output
    assert "gemini-3-flash-preview" in result.output
    assert "aistudio" in result.output


def test_doctor_llm_probe_hides_raw_details_and_skips_anki_blocker(temp_db, temp_config):
    probe = LLMProbeResult(
        available=False,
        provider="gemini",
        backend="vertex",
        model="gemini-3-flash-preview",
        credential_source="vertex",
        category="config",
        message="gemini rejected the configured request or model settings.",
        debug_message="404 NOT_FOUND projects/my-gcp-project-12345/... gemini-3-flash-preview",
        http_status=404,
        retryable=False,
    )

    with patch("pb.cli.commands.doctor.LLMRuntime.health", return_value=_healthy_runtime()), patch(
        "pb.cli.commands.doctor.LLMRuntime.live_probe",
        return_value=probe,
    ), patch(
        "pb.vault.anki_client.is_anki_available",
        return_value=False,
    ):
        result = runner.invoke(app, ["doctor", "--llm"])

    assert result.exit_code == 53, result.output
    assert "LLM live probe" in result.output
    assert "Anki integration" not in result.output
    assert "projects/my-gcp-project-12345" not in result.output
    assert "pb doctor --llm --debug" in result.output

    with patch("pb.cli.commands.doctor.LLMRuntime.health", return_value=_healthy_runtime()), patch(
        "pb.cli.commands.doctor.LLMRuntime.live_probe",
        return_value=probe,
    ):
        debug_result = runner.invoke(app, ["doctor", "--llm", "--debug"])

    assert debug_result.exit_code == 53, debug_result.output
    assert "projects/my-gcp-project-12345" in debug_result.output


def test_init_llm_updates_config_values(tmp_path):
    """`pb init llm` writes the expected Gemini config keys."""
    config_path = tmp_path / "config.toml"
    config_path.write_text("[general]\nvault_path = '/tmp/vault'\n")

    with patch("pb.cli.commands.init.get_config_path", return_value=config_path), patch(
        "pb.cli.commands.init.set_config_value"
    ) as mock_set:
        result = runner.invoke(
            app,
            ["init", "llm", "--backend", "aistudio", "--model", "gemini-3-flash-preview"],
        )

    assert result.exit_code == 0, result.output
    assert "LLM configuration saved." in result.output
    mock_set.assert_has_calls(
        [
            call("llm", "provider", "gemini"),
            call("llm", "backend", "aistudio"),
            call("llm", "default_model", "gemini-3-flash-preview"),
            call("llm", "prompt_template_version", "v3"),
            call("llm", "require_llm_for_core_workflows", True),
        ]
    )
