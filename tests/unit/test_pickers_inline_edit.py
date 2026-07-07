# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.cli import pickers
from pb.cli.pickers import PickerResult


def test_pick_single_choice_inline_edit_returns_custom_value(monkeypatch):
    monkeypatch.setattr(pickers.sys.stdin, "isatty", lambda: True)

    def fake_pick(labels, *args, editable_state=None, **kwargs):
        assert editable_state is not None
        editable_state["text"] = "custom answer"
        return [len(labels) - 1]

    monkeypatch.setattr("pb.cli.helpers._interactive_pick", fake_pick)

    value = pickers.pick_single_choice(
        [("a", "Alpha"), ("b", "Beta")],
        title="Choose one",
        allow_inline_edit=True,
    )

    assert value == "custom answer"


def test_pick_many_choices_inline_edit_appends_custom_value(monkeypatch):
    monkeypatch.setattr(pickers.sys.stdin, "isatty", lambda: True)

    def fake_pick(labels, *args, editable_state=None, **kwargs):
        assert editable_state is not None
        editable_state["text"] = "custom option"
        return [0, len(labels) - 1]

    monkeypatch.setattr("pb.cli.helpers._interactive_pick", fake_pick)

    values = pickers.pick_many_choices(
        [("a", "Alpha"), ("b", "Beta")],
        title="Choose many",
        allow_inline_edit=True,
    )

    assert values == ["a", "custom option"]


def test_pick_many_choices_inline_edit_returns_inline_result_when_requested(monkeypatch):
    monkeypatch.setattr(pickers.sys.stdin, "isatty", lambda: True)

    def fake_pick(labels, *args, editable_state=None, **kwargs):
        assert editable_state is not None
        editable_state["text"] = "custom option"
        return [0, len(labels) - 1]

    monkeypatch.setattr("pb.cli.helpers._interactive_pick", fake_pick)

    result = pickers.pick_many_choices(
        [("a", "Alpha"), ("b", "Beta")],
        title="Choose many",
        allow_inline_edit=True,
        return_result=True,
    )

    assert result == PickerResult(kind="inline_text", value=["a", "custom option"])
