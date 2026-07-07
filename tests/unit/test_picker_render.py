"""Regression tests for the shared arrow-picker renderer.

Every TUI picker in the CLI (task pickers, single-choice, multi-select,
inbox/init pickers, numbered-list selection) funnels through
``_render_picker``/``_draw`` in ``pb.cli.helpers``. The redraw routine
moves the cursor up one row per *rendered line*, so a line wider than the
terminal — which the terminal wraps onto extra physical rows — corrupts
the cursor accounting and leaves stale rows behind. That is the reported
"`Choose next action` printed four times" bug.
"""

from __future__ import annotations

import os

from pb.cli.helpers import _display_width, _picker_formatted_text, _render_picker, _screen_line_count


def _fake_size(monkeypatch, cols: int) -> None:
    monkeypatch.setattr(
        "shutil.get_terminal_size",
        lambda *a, **k: os.terminal_size((cols, 24)),
    )


def test_render_picker_lines_never_exceed_terminal_width(monkeypatch):
    _fake_size(monkeypatch, 80)
    long_label = "pb study 'Machine Learning' — " + "detail " * 40
    labels = [long_label, "short option", long_label]
    lines = _render_picker(
        labels,
        cursor=0,
        checked=set(),
        header="Choose next action\n  Pick what to do now",
        multi=False,
    )
    for line in lines:
        assert len(line) <= 80, f"line wraps the terminal ({len(line)} cols): {line!r}"


def test_screen_line_count_matches_rendered_rows(monkeypatch):
    # The redraw cursor-up count (_screen_line_count) must equal the physical
    # rows actually drawn; this only holds when no line wraps.
    _fake_size(monkeypatch, 80)
    lines = _render_picker(
        ["x" * 300] * 3,
        cursor=1,
        checked=set(),
        header="Header",
        multi=True,
    )
    assert _screen_line_count(lines) == len(lines)
    for line in lines:
        assert len(line) <= 80


def test_cursor_and_multiselect_decorations_stay_within_width(monkeypatch):
    # The cursor line gains a "[ ... ]" wrapper and multi mode adds "[x]";
    # truncation must clip the fully-decorated line, not the raw label.
    _fake_size(monkeypatch, 60)
    labels = ["z" * 200 for _ in range(3)]
    lines = _render_picker(labels, cursor=0, checked={0, 1}, header="Pick", multi=True)
    for line in lines:
        assert len(line) <= 60
    assert any("Submit selections" in line for line in lines)


def test_render_picker_width_param_overrides_terminal(monkeypatch):
    _fake_size(monkeypatch, 200)
    lines = _render_picker(
        ["y" * 120],
        cursor=0,
        checked=set(),
        header="Header",
        multi=False,
        width=40,
    )
    for line in lines:
        assert len(line) <= 40


def test_render_picker_lines_never_exceed_terminal_cell_width(monkeypatch):
    _fake_size(monkeypatch, 36)
    lines = _render_picker(
        ["Option with 𝐶^(Δ 7) and Fsharp harmony"] * 2,
        cursor=0,
        checked=set(),
        header="Choose one\n  Theory focus with 𝐶^(Δ 7)",
        multi=False,
        width=36,
    )

    assert any("F#" in line for line in lines)
    for line in lines:
        assert _display_width(line) <= 36


def test_prompt_toolkit_highlight_spans_wrapped_active_choice(monkeypatch):
    _fake_size(monkeypatch, 48)
    lines = _render_picker(
        ["A long active option with enough words to wrap across several terminal rows"],
        cursor=0,
        checked=set(),
        header="Header",
        multi=False,
        width=48,
    )
    fragments = [
        (style, text)
        for style, text in _picker_formatted_text(lines)
        if text != "\n"
    ]
    highlighted = [text for style, text in fragments if style == "reverse bold"]

    assert len(highlighted) >= 2
    assert highlighted[0].lstrip().startswith("❯")
    assert any("]" in line for line in highlighted)
