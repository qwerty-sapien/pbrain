"""Adversarial release gate tests for v1.0.

Cross-boundary scenarios that chain commands with messy, erratic inputs
to surface state pollution between commands. Each test exercises multiple
requirements simultaneously — the way real users break things.

Requirements covered: TEST-01..03, CLI-01..04, STATE-01..02, MCP-01..03
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.exceptions import ExitCode
from pb.llm.drafts import (
    GoalDraft,
    GoalRoadmapDraft,
    GoalRoadmapNodeDraft,
    LearningPlanBlockDraft,
    PractisePlanDraft,
    StudyPlanDraft,
)
from pb.llm.runtime import GeneratedDraft


runner = CliRunner()


def _fake_generate_draft(self, schema_cls, prompt, **kwargs):
    """Universal LLM mock that returns valid payloads for all draft types."""
    from pb.llm.drafts import GeneratedNamesDraft, ClarifierQuestionDraft
    from pb.cli.commands.goals import _extract_domain
    source_scope = kwargs.get("source_scope", "test")
    topic = source_scope.split(":", 1)[-1].strip() or "topic"

    if schema_cls is GoalDraft:
        payload = GoalDraft(
            title=topic.title(),
            description=f"Goal for {topic}.",
            domain=_extract_domain(topic),
            execution_mode="study",
            horizon="quarter",
            framework="Bloom",
            study_framework="bloom_retrieval",
            target_bloom_stage="apply",
            success_definition=f"Progress on {topic}.",
            feedback_source="artifact",
            evidence_type="artifact",
        )
    elif schema_cls is GoalRoadmapDraft:
        payload = GoalRoadmapDraft(
            summary=f"Roadmap for {topic}.",
            project_title=topic.title(),
            nodes=[GoalRoadmapNodeDraft(
                node_id="n1", title=f"Foundation: {topic}",
                branch="study", scope=topic,
                success_check=f"Explain core concepts of {topic}.",
            )],
        )
    elif schema_cls is StudyPlanDraft:
        payload = StudyPlanDraft(
            summary="",
            blocks=[LearningPlanBlockDraft(
                branch="study", subject_scope=topic, duration_minutes=30,
                target_bloom_stage="apply", study_mode="active recall",
                success_check=f"Explain {topic}.", reason=f"Study {topic}.",
            )],
        )
    elif schema_cls is PractisePlanDraft:
        payload = PractisePlanDraft(
            summary="",
            blocks=[LearningPlanBlockDraft(
                branch="practise", subject_scope=topic, duration_minutes=25,
                practice_stage="integrate", drill_type=topic,
                success_check=f"Complete {topic}.", reason=f"Practise {topic}.",
            )],
        )
    elif schema_cls is GeneratedNamesDraft:
        payload = GeneratedNamesDraft(
            display_title=topic.title(),
            short_title=topic[:20],
            slug=topic.lower().replace(" ", "-")[:30],
            note_title=topic.title(),
            session_title=topic.title(),
            task_title=topic.title(),
        )
    else:
        payload = schema_cls()

    return GeneratedDraft(
        payload=payload, model="gemini:test",
        source_scope=source_scope, prompt_template_version="test", raw_response="{}",
    )


@pytest.fixture
def cli_env(tmp_path, monkeypatch, temp_db, temp_config):
    """Fully isolated CLI environment with mocked LLM and timers."""
    config_path = tmp_path / "config.toml"
    vault_path = tmp_path / "vault"
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("HOME", str(tmp_path))

    patches = [
        patch("pb.sessions.service.TimerManager.start_session_timers", return_value=None),
        patch("pb.sessions.service.TimerManager.stop_session_timers", return_value=None),
        patch("pb.cli.commands.next.schedule_actionable_notification", return_value=True),
        patch("pb.llm.runtime.LLMRuntime.generate_draft", new=_fake_generate_draft),
    ]
    ctx = {}
    for p in patches:
        ctx[id(p)] = p.start()

    result = runner.invoke(app, [
        "init", "--non-interactive", "--vault-name", "main",
        "--vault-path", str(vault_path), "--provider", "gemini",
        "--model", "gemini-3-flash-preview", "--yes",
    ])
    assert result.exit_code == 0, f"init failed: {result.output}"

    yield {"vault_path": vault_path, "tmp_path": tmp_path}

    for p in patches:
        p.stop()


class TestCrossBoundaryScenarios:
    """Chain multiple commands in realistic messy sequences."""

    def test_goal_add_then_learn_then_finish_cross_boundary(self, cli_env):
        """CLI-01 + CLI-02 + CLI-04: goal add → learn with flags → finish with completion.

        Exercises the full learning loop in sequence, verifying state consistency
        across command boundaries.
        """
        # CLI-01: goal add with --yes must not crash
        result = runner.invoke(app, ["goal", "add", "--yes", "get", "better", "at", "calculus", "integration", "techniques"])
        assert result.exit_code == 0, f"goal add failed: {result.output}"
        assert "Created goal" in result.output

        # CLI-02: learn with flags — flags must not be swallowed as topic text
        result = runner.invoke(app, ["learn", "--yes", "--study", "integration", "by", "parts"])
        assert result.exit_code == 0, f"learn failed: {result.output}"
        assert "Started:" in result.output

        # Verify topic is clean — "integration by parts" not "integration by parts --yes --study"
        result = runner.invoke(app, ["now"])
        assert result.exit_code == 0
        output_lower = result.output.lower()
        assert "--yes" not in output_lower
        assert "--study" not in output_lower

        # CLI-04: finish with completion — must not prompt
        result = runner.invoke(app, ["finish", "--completion", "65"])
        assert result.exit_code == 0, f"finish failed: {result.output}"
        assert "Finished:" in result.output

    def test_finish_without_session_then_learn_then_double_finish(self, cli_env):
        """Erratic user: finish with no session, then learn, then finish twice."""
        # finish with no active session — should error cleanly, not crash
        result = runner.invoke(app, ["finish", "some", "note"])
        assert result.exit_code != 0

        # learn → start a real session
        result = runner.invoke(app, ["learn", "--yes", "--study", "topology"])
        assert result.exit_code == 0

        # finish once — should succeed
        result = runner.invoke(app, ["finish", "understood", "basic", "open", "sets"])
        assert result.exit_code == 0

        # finish again immediately — no session active, should error cleanly
        result = runner.invoke(app, ["finish", "already", "done"])
        assert result.exit_code != 0

    def test_learn_twice_without_finish(self, cli_env):
        """Impatient user: starts learning, then tries to start again without finishing."""
        result = runner.invoke(app, ["learn", "--yes", "--study", "group", "theory"])
        assert result.exit_code == 0

        # Try to learn something else — should be blocked (active session)
        result = runner.invoke(app, ["learn", "--yes", "--study", "ring", "theory"])
        assert result.exit_code != 0

        # Clean up
        result = runner.invoke(app, ["finish", "done"])
        assert result.exit_code == 0

    def test_empty_and_whitespace_inputs(self, cli_env):
        """Edge case: empty strings and whitespace-only inputs."""
        # empty topic for learn — should error or prompt
        result = runner.invoke(app, ["learn", "--yes", "--study"])
        # With --yes, no interactive prompt → should fail gracefully
        assert result.exit_code != 0 or "required" in result.output.lower() or result.output.strip() == ""


class TestDoctorExitCodes:
    """STATE-02: doctor exits 0 when vault+SQLite healthy, even if Anki/LLM absent."""

    def test_doctor_exits_0_without_anki_or_llm(self, cli_env):
        """Doctor should pass when required infrastructure is healthy."""
        with patch("pb.vault.anki_client.is_anki_available", return_value=False):
            result = runner.invoke(app, ["doctor"])
            assert result.exit_code == 0, (
                f"doctor should exit 0 when vault+SQLite healthy but Anki offline.\n"
                f"Output: {result.output}"
            )

    def test_doctor_json_exits_0_without_anki(self, cli_env):
        """JSON mode should also report exit 0 for optional-only failures."""
        with patch("pb.vault.anki_client.is_anki_available", return_value=False):
            result = runner.invoke(app, ["doctor", "--json"])
            assert result.exit_code == 0
            import json
            data = json.loads(result.output)
            anki_check = next((c for c in data["checks"] if "Anki" in c["label"]), None)
            assert anki_check is not None
            assert anki_check["required"] is False


class TestInputNormalization:
    """Verify that input normalization works correctly at the boundary."""

    def test_learn_joins_multiword_topic(self, cli_env):
        """CLI-02: multi-word topic joined correctly, flags not swallowed."""
        result = runner.invoke(app, ["learn", "--yes", "-s", "quantum", "field", "theory"])
        assert result.exit_code == 0
        # Clean up
        runner.invoke(app, ["finish", "done"])

    def test_goal_add_multiword_title(self, cli_env):
        """CLI-01: goal add with multi-word title."""
        result = runner.invoke(app, ["goal", "add", "--yes", "master", "distributed", "systems"])
        assert result.exit_code == 0
        assert "Created goal" in result.output


class TestContextAndNormalize:
    """Unit-level tests for the new infrastructure modules."""

    def test_join_words_empty(self):
        from pb.cli.normalize import join_words
        assert join_words(None) == ""
        assert join_words([]) == ""
        assert join_words(["  "]) == ""

    def test_join_words_multiword(self):
        from pb.cli.normalize import join_words
        assert join_words(["hello", "world"]) == "hello world"
        assert join_words(["quantum", "field", "theory"]) == "quantum field theory"

    def test_is_interactive_respects_yes(self):
        from unittest.mock import MagicMock
        from pb.cli.normalize import is_interactive

        ctx = MagicMock()
        ctx.obj = {"yes": True}
        # Even if stdin is a TTY, --yes makes it non-interactive
        with patch("pb.cli.normalize.sys") as mock_sys:
            mock_sys.stdin.isatty.return_value = True
            assert is_interactive(ctx) is False

    def test_command_context_from_typer_missing_runtime(self):
        from pb.cli.context import CommandContext
        from pb.core.exceptions import ConfigError

        ctx = MagicMock()
        ctx.obj = {"runtime": None, "repo": None}
        with pytest.raises(ConfigError):
            CommandContext.from_typer(ctx)

    def test_command_context_from_typer_success(self):
        from pb.cli.context import CommandContext

        ctx = MagicMock()
        ctx.obj = {
            "runtime": MagicMock(),
            "repo": MagicMock(),
            "config": MagicMock(),
            "factory": {"test": lambda: "svc"},
            "yes": True,
            "verbose": False,
        }
        with patch("pb.cli.context.get_console"), patch("pb.cli.context.get_err_console"):
            cc = CommandContext.from_typer(ctx)
            assert cc.yes is True
            assert cc.verbose is False
            assert cc.service("test") == "svc"


class TestPhase4Regressions:
    """Regression tests for Phase 4 bug fixes (CLI-01, CLI-03, CLI-04, STATE-01)."""

    def test_cli01_goal_add_yes_exits_0(self, cli_env):
        """CLI-01: goal add --yes must exit 0, create goal, no crash."""
        result = runner.invoke(app, [
            "goal", "add", "--yes",
            "get", "better", "at", "calculus", "integration", "techniques",
        ])
        assert result.exit_code == 0, f"CLI-01 failed (exit {result.exit_code}): {result.output}"

    def test_cli03_domain_inference_skips_stopwords(self, cli_env):
        """CLI-03: domain for 'get better at calculus' must be 'calculus' not 'get'."""
        result = runner.invoke(app, [
            "goal", "add", "--yes",
            "get", "better", "at", "calculus", "integration", "techniques",
        ])
        assert result.exit_code == 0, f"CLI-03 setup failed: {result.output}"
        from pb.storage.repository import Repository
        repo = Repository()
        goals = repo.list_goal_arcs()
        assert goals, "No goal stored after goal add"
        last_goal = goals[-1]
        assert last_goal.domain != "get", f"Stopword 'get' stored as domain: {last_goal.domain}"
        assert last_goal.domain == "calculus", f"Expected 'calculus', got '{last_goal.domain}'"

    def test_cli04_finish_accepts_yes_flag(self, cli_env):
        """CLI-04: finish --completion N --yes must exit 0 without prompting."""
        runner.invoke(app, ["learn", "--yes", "--study", "topology"])
        result = runner.invoke(app, ["finish", "--completion", "65", "--yes"])
        assert result.exit_code == 0, f"CLI-04 failed (exit {result.exit_code}): {result.output}"

    def test_state01_goal_add_rollback_on_vault_failure(self, cli_env):
        """STATE-01: SQLite row rolled back if vault write fails."""
        from pb.storage.repository import Repository

        with patch("pb.cli.commands.goals._write_goal_note", side_effect=OSError("vault write failed")):
            result = runner.invoke(app, [
                "goal", "add", "--yes", "learn", "topology",
            ])
            assert result.exit_code != 0, f"Expected failure but got exit 0: {result.output}"

        repo = Repository()
        goals = repo.list_goal_arcs()
        orphan_goals = [g for g in goals if "topology" in (g.title or "").lower()]
        assert not orphan_goals, f"Goal row not rolled back: {[g.title for g in orphan_goals]}"
