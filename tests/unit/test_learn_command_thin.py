"""Unit tests for the thin `pb learn` router."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock, patch

from typer.testing import CliRunner

from pb.cli.commands.learn import _pick_domain, app, learn_command
from pb.core.action_routing import LearningRouteDecision


def _make_ctx_obj():
    repo = MagicMock()
    repo.get_active_session.return_value = None
    return {
        "factory": {},
        "repo": repo,
        "config": MagicMock(),
        "vault_cwd": None,
        "runtime": MagicMock(),
    }


class TestLearnRouting:
    def test_force_practise_skips_router(self):
        runner = CliRunner()
        with patch("pb.cli.commands.learn._dispatch_to_branch") as mock_dispatch, patch(
            "pb.cli.commands.learn.route_learning_intent"
        ) as mock_route:
            result = runner.invoke(
                app,
                ["--practise", "leetcode", "arrays"],
                catch_exceptions=False,
                obj=_make_ctx_obj(),
            )
        assert result.exit_code == 0, result.output
        mock_route.assert_not_called()
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.args[1:] == ("practise", "leetcode arrays")

    def test_auto_route_dispatches_router_decision(self):
        runner = CliRunner()
        decision = LearningRouteDecision(branch="study", reason="matches conceptual keywords", confidence=0.4)
        with patch("pb.cli.commands.learn._dispatch_to_branch") as mock_dispatch, patch(
            "pb.cli.commands.learn.route_learning_intent",
            return_value=decision,
        ) as mock_route:
            result = runner.invoke(
                app,
                ["jazz", "harmony"],
                catch_exceptions=False,
                obj=_make_ctx_obj(),
            )
        assert result.exit_code == 0, result.output
        mock_route.assert_called_once()
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.args[1:] == ("study", "jazz harmony")

    def test_empty_non_tty_invocation_errors(self):
        runner = CliRunner()
        result = runner.invoke(app, [], obj=_make_ctx_obj())
        assert result.exit_code != 0
        assert "learning target is required" in result.output.lower()

    def test_empty_interactive_invocation_prompts_for_topic(self):
        runner = CliRunner()
        decision = LearningRouteDecision(branch="study", reason="default", confidence=0.0)
        with patch("pb.cli.commands.learn.is_interactive", return_value=True), patch(
            "pb.cli.commands.learn.typer.prompt",
            return_value="music theory",
        ), patch("pb.cli.commands.learn.route_learning_intent", return_value=decision), patch(
            "pb.cli.commands.learn._dispatch_to_branch"
        ) as mock_dispatch:
            result = runner.invoke(app, [], catch_exceptions=False, obj=_make_ctx_obj())
        assert result.exit_code == 0, result.output
        mock_dispatch.assert_called_once()
        assert mock_dispatch.call_args.args[1:] == ("study", "music theory")


class TestLearnDomainPicker:
    def test_pick_domain_uses_picker_selection(self, tmp_path):
        knowledge_dir = tmp_path / "knowledge"
        for domain in ("piano", "math"):
            domain_dir = knowledge_dir / domain
            domain_dir.mkdir(parents=True)
            (domain_dir / "_state.md").write_text("---\n---\n")

        with patch("pb.cli.pickers.pick_single_choice", return_value="piano"):
            assert _pick_domain(knowledge_dir, console=None) == "piano"


class TestLearnThinness:
    def test_learn_command_has_no_direct_socratic_calls(self):
        source = inspect.getsource(learn_command)
        forbidden = [
            "SocraticDebriefEngine(",
            "build_socratic_note(",
            "infer_wikilinks(",
            "extract_socratic_cards(",
        ]
        for pattern in forbidden:
            assert pattern not in source, f"`pb learn` should stay thin; found {pattern}"
