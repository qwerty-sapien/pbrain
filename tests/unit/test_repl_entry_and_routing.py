"""Tests for the REPL-first entry flow and chooser-first routing.

Note: Two tests removed in phase 14:
  - test_bare_pb_uses_shell_in_tty: monkeypatched _handle_inline_capture_flags
    which was deleted (phase 14-03 severed inline capture flags from main.py)
  - test_next_interactive_chooser_includes_thought_and_todo: pb next no longer
    includes thought/todo options (moved to memo/); those are memo's concern now
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

from pb.core.dispatch_models import InteractionEnvelope


def test_do_interactive_still_uses_chooser_for_single_candidate(monkeypatch, repo):
    from pb.cli.commands import do as do_mod

    picker = MagicMock(return_value=None)
    runner_mock = MagicMock()
    ctx = SimpleNamespace(obj={"repo": repo})

    import pb.cli.normalize as _norm_mod
    monkeypatch.setattr(_norm_mod.sys.stdin, "isatty", lambda: True)
    async def fake_dispatch(*args, **kwargs):
        return InteractionEnvelope(
            session_id="sess-1",
            status="complete",
            prompt="Pick one",
            options=["thought note this"],
        )

    monkeypatch.setattr(do_mod, "dispatch", fake_dispatch)
    monkeypatch.setattr(do_mod, "pick_single_choice", picker)
    monkeypatch.setattr(do_mod, "run_internal_command", runner_mock)

    do_mod.do_command(ctx, ["note", "this"])

    picker.assert_called_once()
    runner_mock.assert_not_called()
