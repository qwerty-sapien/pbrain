# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import typer.main

from pb.cli import shell
from pb.cli.input_router import PbCommandResolver, RoutedInput, classify_interactive_input
from pb.cli.main import app as pb_app
from pb.core.registry import CommandHandler, CommandRegistry
from pb.llm.drafts import LearningPartnerTurnDraft


class _FakeClickApp:
    def __init__(self):
        self.commands = {"finish": object(), "pause": object(), "next": object()}
        self.params = []
        self.calls: list[list[str]] = []

    def __call__(self, args, standalone_mode=False):
        self.calls.append(list(args))


def _slash_registry() -> CommandRegistry:
    registry = CommandRegistry()
    for command in ("/hint", "/harder"):
        registry.register(CommandHandler(name=command, help_text=command, handler=lambda args, ctx: None))
    return registry


def test_real_pb_commands_take_priority_over_learning_answers():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))
    registry = _slash_registry()

    decision = classify_interactive_input(
        "finish",
        pb_command_resolver=resolver,
        slash_registry=registry,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "pb_command"
    assert decision.text == "finish"


def test_tab_routes_to_back_navigation_during_learning():
    decision = classify_interactive_input(
        "\t",
        pb_command_resolver=None,
        slash_registry=None,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "navigation"
    assert decision.argv == ("back",)
    assert decision.command == "back"


def test_root_flags_take_priority_over_learning_answers():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))

    decision = classify_interactive_input(
        "--help",
        pb_command_resolver=resolver,
        slash_registry=_slash_registry(),
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "pb_command"
    assert decision.text == "--help"


def test_shell_and_standalone_learning_use_consistent_pb_and_slash_routing():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))
    registry = _slash_registry()

    shell_decision = classify_interactive_input(
        "/hint",
        pb_command_resolver=resolver,
        slash_registry=registry,
        active_learning=True,
        allow_shell_commands=True,
        allow_nl_dispatch=False,
    )
    partner_decision = classify_interactive_input(
        "/hint",
        pb_command_resolver=resolver,
        slash_registry=registry,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert shell_decision.kind == "slash_command"
    assert partner_decision.kind == "slash_command"
    assert shell_decision.command == partner_decision.command == "/hint"


def test_unknown_slash_command_is_not_treated_as_an_answer():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))
    registry = _slash_registry()

    decision = classify_interactive_input(
        "/unknown-command",
        pb_command_resolver=resolver,
        slash_registry=registry,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "slash_unknown"
    assert decision.text == "/unknown-command"


def test_contextual_slash_commands_do_not_reach_coaching_turn(monkeypatch, tmp_path):
    click_app = _FakeClickApp()
    resolver = PbCommandResolver(click_app)
    partner = MagicMock()
    partner.command_registry = _slash_registry()
    partner.run_contextual_command.return_value = None
    monkeypatch.setattr(shell, "_learning_partner_for_session", lambda *args, **kwargs: partner)
    coaching_turn = MagicMock()
    monkeypatch.setattr(shell, "_coaching_turn", coaching_turn)

    repo = SimpleNamespace(get_active_session=lambda: SimpleNamespace(id="sess-1", task_id="task-1"))
    runtime = SimpleNamespace()
    runtime_ctx = SimpleNamespace()

    shell._dispatch(
        ["/hint"],
        click_app,
        Path(tmp_path),
        [Path(tmp_path)],
        repo=repo,
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        raw_input="/hint",
        pb_command_resolver=resolver,
    )

    coaching_turn.assert_not_called()
    partner.run_contextual_command.assert_called_once_with("/hint")
    assert click_app.calls == []


def test_hint_during_active_question_runs_contextual_command(monkeypatch, tmp_path):
    partner = MagicMock()
    next_turn = LearningPartnerTurnDraft(reply="Use the chain rule on the outer function.")
    partner.run_contextual_command.return_value = next_turn
    partner._render_question_input.side_effect = [
        RoutedInput(kind="slash_command", command="/hint"),
        None,
    ]

    shell._render_partner_turn_chain(partner, LearningPartnerTurnDraft(reply="Differentiate this."))

    partner.run_contextual_command.assert_called_once_with("/hint")


def test_global_pb_command_inside_active_question_returns_to_command_path(monkeypatch, tmp_path):
    click_app = _FakeClickApp()
    resolver = PbCommandResolver(click_app)
    partner = MagicMock()
    partner.command_registry = _slash_registry()
    partner.respond_once.return_value = LearningPartnerTurnDraft(reply="Choose one.")
    partner._render_question_input.side_effect = [RoutedInput(kind="pb_command", text="finish", argv=("finish",))]
    monkeypatch.setattr(shell, "_learning_partner_for_session", lambda *args, **kwargs: partner)

    repo = SimpleNamespace(get_active_session=lambda: SimpleNamespace(id="sess-1", task_id="task-1"))
    runtime = SimpleNamespace()
    runtime_ctx = SimpleNamespace()

    shell._dispatch(
        ["answer"],
        click_app,
        Path(tmp_path),
        [Path(tmp_path)],
        repo=repo,
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        raw_input="answer",
        pb_command_resolver=resolver,
    )

    assert click_app.calls == [["finish"]]


def test_nested_partner_command_preserves_then_follow_up(monkeypatch, tmp_path):
    click_app = _FakeClickApp()
    resolver = PbCommandResolver(click_app)
    nested = RoutedInput(
        kind="pb_command",
        text="finish --skip --yes; next --run",
        argv=("finish", "--skip", "--yes", "Next session preference: More theoretical"),
        command="finish",
        args="then:next --run",
    )
    monkeypatch.setattr(shell, "_learning_partner_for_session", lambda *args, **kwargs: None)
    monkeypatch.setattr(shell, "_coaching_turn", lambda *args, **kwargs: nested)

    repo = SimpleNamespace(get_active_session=lambda: SimpleNamespace(id="sess-1", task_id="task-1"))
    runtime = SimpleNamespace()
    runtime_ctx = SimpleNamespace()

    shell._dispatch(
        ["answer"],
        click_app,
        Path(tmp_path),
        [Path(tmp_path)],
        repo=repo,
        runtime=runtime,
        runtime_ctx=runtime_ctx,
        raw_input="answer",
        pb_command_resolver=resolver,
    )

    assert click_app.calls == [
        ["finish", "--skip", "--yes", "Next session preference: More theoretical"],
        ["next", "--run"],
    ]
