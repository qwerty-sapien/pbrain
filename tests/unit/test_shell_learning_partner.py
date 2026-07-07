# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

from pb.cli import shell
from pb.cli.input_router import PbCommandResolver, RoutedInput, classify_interactive_input
from pb.cli.main import app as pb_app
from pb.core.learning_metadata import build_learning_task_description
from pb.core.learning_partner import LearningPartnerSession, PartnerRunResult
from pb.core.models import Session, Task
from pb.domain.enums import SessionMode, TaskState
from pb.llm.drafts import LearningPartnerTurnDraft
import typer.main


def _runtime(tmp_path: Path):
    learning_policy = SimpleNamespace(
        answer_check="",
        small_retry="",
        session_explain="",
        lesson_hint_intuitive="",
        drill_generation="",
        complex_free_response_eval="",
        lesson_planning="",
    )
    return SimpleNamespace(
        health=lambda: SimpleNamespace(available=False),
        role_bindings=lambda: {},
        config=SimpleNamespace(
            model_roles=SimpleNamespace(default="gemini:test", fast_inference="gemini:test"),
            learning=SimpleNamespace(model_policy=learning_policy),
            preferences={},
        ),
    )


def _make_partner(repo, temp_config, tmp_path: Path) -> LearningPartnerSession:
    task = repo.create_task(
        Task(
            title="Biddle grip",
            state=TaskState.ACTIVE,
            work_type="practice",
            description=build_learning_task_description(
                branch="practise",
                scope="Biddle grip",
                success_check="Hold the grip cleanly for 3 seconds.",
                domain="cardistry",
                practice_stage="integrate",
            ),
            generated_names={"display_title": "Biddle grip"},
        )
    )
    session = repo.create_session(
        Session(
            task_id=task.id,
            mode=SessionMode.PRACTICE,
            branch="practise",
            subject_scope="Biddle grip",
            intended_outcome="Hold the grip cleanly for 3 seconds.",
        )
    )
    runtime_ctx = SimpleNamespace(
        vault_path=Path(temp_config.general.vault_path),
        quarantine_path=tmp_path / "Learning" / "Inbox" / "pb",
        data_dir=tmp_path / ".pb-data",
    )
    runtime_ctx.quarantine_path.mkdir(parents=True, exist_ok=True)
    runtime_ctx.data_dir.mkdir(parents=True, exist_ok=True)
    return LearningPartnerSession(
        runtime=_runtime(tmp_path),
        runtime_ctx=runtime_ctx,
        repo=repo,
        task=task,
        session=session,
        branch="practise",
        objective="Hold the grip cleanly for 3 seconds.",
        topic="Biddle grip",
        domain="cardistry",
        mode="integrate",
    )


def test_render_partner_turn_chain_feeds_picker_answers_back_into_partner(monkeypatch, repo, temp_config, temp_dir):
    partner = _make_partner(repo, temp_config, temp_dir)
    initial_turn = partner.open_with_first_move()
    answers = iter(["1 2 3", None])

    monkeypatch.setattr(partner, "_render_session_frame", lambda turn: None)
    monkeypatch.setattr(partner, "_render_question_input", lambda turn: next(answers))

    shell._render_partner_turn_chain(partner, initial_turn)

    assert any(
        item["role"] == "user" and item["content"] == "1 2 3"
        for item in partner.transcript
    )
    assert any(item.get("note") == "1 2 3" for item in partner.evidence_log)


def test_maybe_open_learning_session_uses_opening_move_once(monkeypatch, tmp_path):
    fake_partner = MagicMock()
    fake_partner.transcript = []
    fake_partner.open_with_first_move.return_value = LearningPartnerTurnDraft(reply="Start with one slow rep.")

    render_mock = MagicMock()
    monkeypatch.setattr(shell, "_learning_partner_for_session", lambda *args, **kwargs: fake_partner)
    monkeypatch.setattr(shell, "_render_partner_turn_chain", render_mock)

    shell._maybe_open_learning_session(
        repo=SimpleNamespace(),
        runtime=SimpleNamespace(),
        runtime_ctx=SimpleNamespace(),
        active_session=SimpleNamespace(),
    )

    fake_partner.open_with_first_move.assert_called_once()
    render_mock.assert_called_once()


def test_unique_note_path_appends_numeric_suffix(tmp_path):
    base = tmp_path / "note.md"
    base.write_text("first", encoding="utf-8")
    second = shell._unique_note_path(base)
    second.write_text("second", encoding="utf-8")

    assert second.name == "note-2.md"
    assert shell._unique_note_path(base).name == "note-3.md"


def test_classify_interactive_input_turns_arrow_sequences_into_navigation():
    decision = classify_interactive_input(
        "\x1b[D\x1b[D\x1b[C",
        pb_command_resolver=None,
        slash_registry=None,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "navigation"
    assert decision.argv == ("left", "left", "right")
    assert decision.command == "right"


def test_model_slash_alias_works_during_active_sessions_too():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))

    decision = classify_interactive_input(
        "/model use gemini:gemini-3.1-pro-preview",
        pb_command_resolver=resolver,
        slash_registry=None,
        active_learning=True,
        allow_shell_commands=False,
        allow_nl_dispatch=False,
    )

    assert decision.kind == "pb_command"
    assert decision.command == "model"


def test_next_slash_inside_active_question_returns_finish_then_next():
    partner = MagicMock()
    partner.run_contextual_command.return_value = PartnerRunResult(
        action="next",
        summary="Next session preference: proofs",
        follow_up_command="next --run",
        skip_finish_assessment=True,
    )
    partner._render_question_input.side_effect = [
        RoutedInput(kind="slash_command", command="/next", args="proofs"),
    ]

    routed = shell._render_partner_turn_chain(partner, LearningPartnerTurnDraft(reply="Explain it."))

    partner.run_contextual_command.assert_called_once_with("/next", "proofs")
    assert routed is not None
    assert routed.kind == "pb_command"
    assert routed.argv == ("finish", "--skip", "--yes", "Next session preference: proofs")
    assert routed.args == "then:next --run"


def test_finish_slash_inside_active_question_routes_through_finish_command():
    partner = MagicMock()
    partner.run_contextual_command.return_value = PartnerRunResult(
        action="finish",
        summary="Finished the lesson session.",
    )
    partner._render_question_input.side_effect = [
        RoutedInput(kind="slash_command", command="/finish", args=""),
    ]

    routed = shell._render_partner_turn_chain(partner, LearningPartnerTurnDraft(reply="Explain it."))

    partner.run_contextual_command.assert_called_once_with("/finish")
    assert routed is not None
    assert routed.kind == "pb_command"
    assert routed.argv == ("finish", "Finished the lesson session.")
    assert "outcome" not in routed.text
