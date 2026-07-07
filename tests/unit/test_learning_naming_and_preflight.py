from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import typer
from click import Command, Context

from pb.cli.active_session import resolve_active_session_preflight
from pb.core.models import GoalArc, Session, Task
from pb.domain.enums import SessionMode
from pb.domain.exceptions import ExitCode


def test_repository_round_trips_generated_names(repo):
    task = Task(
        title="German conjugation recovery",
        generated_names={
            "short_title": "German Recovery",
            "display_title": "German conjugation recovery",
            "folder_name": "german_lang",
        },
    )
    repo.create_task(task)

    saved = repo.get_task(task.id)

    assert saved is not None
    assert saved.generated_names["short_title"] == "German Recovery"
    assert saved.generated_names["folder_name"] == "german_lang"


def test_active_session_preflight_blocks_with_one_line_guard(repo, monkeypatch, capsys):
    active_task = Task(title="German conjugation recovery")
    repo.create_task(active_task)
    repo.create_session(
        Session(
            task_id=active_task.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="German conjugation",
        )
    )

    ctx = Context(Command("pb"))
    ctx.obj = {
        "repo": repo,
        "factory": {"session_service": lambda: SimpleNamespace()},
        "runtime": SimpleNamespace(),
    }

    monkeypatch.setattr("sys.stdin.isatty", lambda: False)

    with pytest.raises(typer.Exit) as exc:
        resolve_active_session_preflight(
            ctx,
            new_intent="linear algebra proofs",
            new_branch="study",
        )

    assert exc.value.exit_code == ExitCode.CONFLICT
    rendered = capsys.readouterr()
    output = rendered.out + rendered.err
    assert "Session active: German conjugation recovery." in output
    assert "Re-run with `--yes` to pause it and" in output
    assert "start the new session." in output


def test_active_session_preflight_single_yes_pauses_current_session(repo, monkeypatch, capsys):
    active_task = Task(title="German conjugation recovery")
    repo.create_task(active_task)
    active_session = repo.create_session(
        Session(
            task_id=active_task.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="German conjugation",
        )
    )

    ctx = Context(Command("pb"))
    ctx.obj = {
        "repo": repo,
        "factory": {"session_service": lambda: SimpleNamespace()},
        "runtime": SimpleNamespace(),
    }
    prompts: list[str] = []

    def accept_prompt(label: str, **kwargs) -> bool:
        prompts.append(label)
        return True

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr("pb.cli.active_session.confirm_choice", accept_prompt)

    allowed = resolve_active_session_preflight(
        ctx,
        new_intent="linear algebra proofs",
        new_branch="study",
    )

    assert allowed is True
    assert prompts == [
        "Session active: German conjugation recovery. Pause it and start study session: linear algebra proofs?"
    ]
    saved = repo.get_session(active_session.id)
    assert saved is not None
    assert saved.end_at is not None
    assert saved.actual_outcome == "Paused to start study session: linear algebra proofs"
    rendered = capsys.readouterr()
    output = rendered.out + rendered.err
    assert "Paused: German conjugation recovery" in output


def test_learning_question_banlist_removed_from_learning_flows():
    banned_strings = [
        "What outcome would make",
        "Where are you starting from:",
        "What feels hardest right now:",
        "Should the next stretch lean study, practise, or mixed?",
        "Quick learning check-in.",
    ]
    paths = [
        Path("src/pb/cli/commands/clarify.py"),
        Path("src/pb/cli/commands/goals.py"),
        Path("src/pb/cli/commands/execute.py"),
    ]

    for path in paths:
        text = path.read_text(encoding="utf-8")
        for banned in banned_strings:
            assert banned not in text, f"{banned!r} still present in {path}"


def test_mcp_start_tools_report_active_session_conflict(monkeypatch, temp_config, repo):
    from pb.mcp.tools import productivebrain as tools

    runtime = SimpleNamespace(
        vault_path=Path(temp_config.vaults[temp_config.general.active_vault].path),
        quarantine_path=Path(temp_config.vaults[temp_config.general.active_vault].path) / "Learning" / "Inbox" / "pb",
        vault_name=temp_config.general.active_vault,
    )
    active_task = Task(
        title="German conjugation recovery",
        generated_names={"display_title": "German conjugation recovery"},
    )
    repo.create_task(active_task)
    repo.create_session(
        Session(
            task_id=active_task.id,
            mode=SessionMode.FOCUS,
            branch="study",
            subject_scope="German conjugation",
        )
    )

    monkeypatch.setattr(tools, "_bootstrap_repo", lambda: (runtime, repo))

    result = tools._do_start_learning_session("study", "linear algebra")

    assert result["started"] is False
    assert result["active_session"]["title"] == "German conjugation recovery"
    assert "pause_current_session_and_switch" in result["options"]


def test_goal_round_trips_generated_names(repo):
    goal = GoalArc(
        title="German B1",
        domain="german",
        generated_names={
            "display_title": "German B1 recovery",
            "short_title": "German B1",
            "folder_name": "german_lang",
        },
    )
    repo.create_goal_arc(goal)

    saved = repo.get_goal_arc(goal.id)

    assert saved is not None
    assert saved.generated_names["display_title"] == "German B1 recovery"
    assert saved.generated_names["folder_name"] == "german_lang"


def test_deterministic_names_keep_short_title_compact_and_rendered():
    from pb.core.naming import deterministic_names

    names = deterministic_names(
        "study_task",
        r"Concrete Fluency with $\mathbb{R}^n$ Open Balls and C^(Δ 7) Fsharp",
    )

    assert len(names.short_title) < 20
    assert "$" not in names.short_title
    assert r"\mathbb" not in names.short_title
    assert "Fsharp" not in names.display_title
    assert names.slug
