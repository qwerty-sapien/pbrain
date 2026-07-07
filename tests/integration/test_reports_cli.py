"""Integration tests for review CLI commands (D-39).

Tests: pb review day/week/energy/friction/priority/month/track
       pb goal report
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.llm.drafts import DailyReviewDraft, WeeklyReviewDraft
from pb.llm.runtime import GeneratedDraft
from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, StorageConfig
from pb.storage.database import init_db, set_db_path


runner = CliRunner()


@pytest.fixture(autouse=True)
def mock_review_runtime():
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
        payload = (
            DailyReviewDraft(
                summary="Daily progress moved at least one goal forward.",
                progress_signals=["Study sessions completed."],
                friction_patterns=["Energy dipped in one block."],
                evidence_captured=["Anki candidates prepared."],
                next_adjustments=["Protect the next planned block."],
            )
            if schema_cls is DailyReviewDraft
            else WeeklyReviewDraft(
                summary="Weekly progress was visible and specific.",
                wins=["Study consistency improved."],
                stalls=["Export debt remained."],
                evidence_progress=["Recall and Anki evidence accumulated."],
                friction_patterns=["Energy dipped mid-week."],
                next_week_focus=["Export cards and continue practice."],
            )
        )
        return GeneratedDraft(
            payload=payload,
            model="gemini-3-flash-preview",
            source_scope=kwargs.get("source_scope", "review"),
            prompt_template_version="test",
            raw_response="{}",
        )

    with patch("pb.llm.runtime.LLMRuntime.require", new=fake_require), patch(
        "pb.llm.runtime.LLMRuntime.generate_draft",
        new=fake_generate,
    ):
        yield


@pytest.fixture
def cli_env(tmp_path):
    """Set up CLI test environment with config and fresh DB."""
    vault_path = tmp_path / "vault"
    vault_path.mkdir()

    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)

    config = Config(
        general=GeneralConfig(vault_path=str(vault_path)),
        storage=StorageConfig(data_dir=str(tmp_path)),
    )
    config_module._config = config

    yield tmp_path

    config_module._config = None
    db_module._db_path = None


def test_review_day(cli_env):
    """pb review day succeeds and shows Daily Review."""
    result = runner.invoke(app, ["review", "day"])
    assert result.exit_code == 0, result.output
    assert "Daily Review" in result.output


def test_review_week(cli_env):
    """pb review week succeeds and shows Weekly Reflection."""
    result = runner.invoke(app, ["review", "week"])
    assert result.exit_code == 0, result.output
    assert "Weekly Reflection" in result.output


def test_review_energy(cli_env):
    """pb review energy succeeds (empty or data message)."""
    result = runner.invoke(app, ["review", "energy"])
    assert result.exit_code == 0, result.output
    assert "Energy Report" in result.output


def test_review_energy_with_days_flag(cli_env):
    """pb review energy --days 14 is accepted."""
    result = runner.invoke(app, ["review", "energy", "--days", "14"])
    assert result.exit_code == 0, result.output
    assert "14 days" in result.output


def test_review_friction(cli_env):
    """pb review friction succeeds."""
    result = runner.invoke(app, ["review", "friction"])
    assert result.exit_code == 0, result.output
    assert "Friction Report" in result.output


def test_review_priority(cli_env):
    """pb review priority succeeds and shows Eisenhower Distribution."""
    result = runner.invoke(app, ["review", "priority"])
    assert result.exit_code == 0, result.output
    assert "Priority Report" in result.output
    assert "Eisenhower Distribution" in result.output


def test_review_month(cli_env):
    """pb review month succeeds and shows Month Report."""
    result = runner.invoke(app, ["review", "month"])
    assert result.exit_code == 0, result.output
    assert "Month Report" in result.output
    assert "Month-over-Month" in result.output


def test_review_track_not_found(cli_env):
    """pb review track nonexistent shows Track not found."""
    result = runner.invoke(app, ["review", "track", "nonexistent"])
    assert result.exit_code == 0, result.output
    assert "Track not found" in result.output


def test_goals_report_no_goals(cli_env):
    """pb goal report shows helpful message when no goals defined."""
    result = runner.invoke(app, ["goal", "report"])
    assert result.exit_code == 0, result.output
    assert "Goals Report" in result.output or "No goals defined" in result.output
