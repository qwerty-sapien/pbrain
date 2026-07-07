# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from types import SimpleNamespace


def test_next_auto_yes_prints_recommendations_without_opening_picker(monkeypatch):
    from pb.cli.commands import next as next_cmd

    candidate = SimpleNamespace(
        backing_command="study Bayes",
        human_label="Study Bayes",
        short_reason="It is the clearest next learning block.",
        source="active_session",
        agent_id="",
    )
    printed: list[str] = []

    class FakeConsole:
        def print(self, *parts, **_kwargs):
            printed.append(" ".join(str(part) for part in parts))

    def fail_picker(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("auto-yes should not open the next-action picker")

    def fail_runner(*_args, **_kwargs):  # pragma: no cover - failure path
        raise AssertionError("auto-yes should not run the recommendation without --run")

    monkeypatch.setattr("sys.stdin.isatty", lambda: True)
    monkeypatch.setattr(next_cmd, "build_next_candidates", lambda _repo, limit=5: [candidate])
    monkeypatch.setattr(next_cmd, "pick_single_choice", fail_picker)
    monkeypatch.setattr(next_cmd, "run_internal_command", fail_runner)
    monkeypatch.setattr(next_cmd, "get_console", lambda: FakeConsole())

    ctx = SimpleNamespace(obj={"repo": object(), "yes": True})

    next_cmd.next_action(ctx, run=False, schedule=None, reminder=None)

    assert any("Next directions" in line for line in printed)
    assert any("Study Bayes" in line for line in printed)
