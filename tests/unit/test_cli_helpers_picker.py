"""Tests for the inline picker preview and verbose shortcuts."""

from __future__ import annotations


def test_render_picker_includes_preview_and_verbose_content():
    from pb.cli.helpers import _render_picker

    lines = _render_picker(
        ["Short label"],
        0,
        set(),
        "Select thought",
        False,
        width=42,
        details=["Full detail text that should appear in preview."],
        verbose_labels=["A much longer verbose label for the selected thought."],
        preview_open=True,
        verbose=True,
    )

    rendered = "\n".join(lines)
    assert "Preview:" in rendered
    assert "Full detail text" in rendered
    assert "longer verbose label" in rendered
    assert "Controls:" in rendered
    assert "digits jump" in rendered


def test_render_picker_preview_uses_chunked_double_spaced_body():
    from pb.cli.helpers import _render_picker

    lines = _render_picker(
        ["Short label"],
        0,
        set(),
        "Select thought",
        False,
        width=54,
        details=[
            "One two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen. "
            "Second paragraph starts here and should stay visibly separated."
        ],
        preview_open=True,
    )

    preview_start = lines.index("  Preview:") + 1
    controls_start = next(index for index, line in enumerate(lines) if "Controls:" in line)
    preview_body = lines[preview_start:controls_start]
    chunks: list[list[str]] = []
    current: list[str] = []
    for line in preview_body:
        if line == "":
            if current:
                chunks.append(current)
                current = []
            continue
        current.append(line)
    if current:
        chunks.append(current)

    assert "" in preview_body
    assert chunks
    assert all(len(chunk) <= 5 for chunk in chunks)


def test_interactive_pick_right_opens_preview_without_cancelling(monkeypatch):
    from pb.cli import helpers

    renders: list[list[str]] = []
    keys = iter(["right", "enter"])

    monkeypatch.setattr(helpers, "_is_real_tty", lambda: True)
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda lines, prev: renders.append(list(lines)))

    result = helpers._interactive_pick(
        ["Short label"],
        "Select thought",
        multi=False,
        details=["Full preview body"],
    )

    assert result == [0]
    assert any("Preview:" in "\n".join(lines) for lines in renders)


def test_interactive_pick_ctrl_o_toggles_verbose_mode(monkeypatch):
    from pb.cli import helpers

    renders: list[list[str]] = []
    keys = iter(["ctrl-o", "q"])

    monkeypatch.setattr(helpers, "_is_real_tty", lambda: True)
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda lines, prev: renders.append(list(lines)))

    result = helpers._interactive_pick(
        ["Short"],
        "Select thought",
        multi=False,
        verbose_labels=["Expanded verbose label that should be visible after toggling verbose mode."],
    )

    assert result is None
    assert any("Expanded verbose label" in "\n".join(lines) for lines in renders[1:])


def test_simple_tty_pick_number_selects_first_option(monkeypatch):
    from pb.cli import helpers

    monkeypatch.setattr(helpers, "_read_key", lambda: "1")
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(["one", "two"], "Header", multi=False)

    assert result == [0]


def test_simple_tty_pick_q_cancels(monkeypatch):
    from pb.cli import helpers

    monkeypatch.setattr(helpers, "_read_key", lambda: "q")
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(["one", "two"], "Header", multi=False)

    assert result is None


def test_simple_tty_pick_arrow_down_then_enter_selects_second_option(monkeypatch):
    from pb.cli import helpers

    keys = iter(["down", "enter"])
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(["one", "two"], "Header", multi=False)

    assert result == [1]


def test_simple_tty_pick_multi_uses_enter_to_confirm(monkeypatch):
    from pb.cli import helpers

    keys = iter(["space", "down", "space", "down", "enter"])
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(["one", "two"], "Header", multi=True)

    assert result == [0, 1]


def test_simple_tty_pick_multi_enter_toggles_before_submit(monkeypatch):
    from pb.cli import helpers

    keys = iter(["enter", "down", "down", "enter"])
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(["one", "two"], "Header", multi=True)

    assert result == [0]


def test_simple_tty_pick_inline_row_accepts_typing_and_backspace(monkeypatch):
    from pb.cli import helpers

    keys = iter(["down", "a", "space", "2", "backspace", "enter"])
    editable_state = {"text": ""}
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(
        ["one", ""],
        "Header",
        multi=False,
        editable_index=1,
        editable_state=editable_state,
        editable_placeholder="Type here",
    )

    assert result == [1]
    assert editable_state["text"] == "a "


def test_simple_tty_pick_slash_enters_command_mode_and_returns_command(monkeypatch):
    from pb.cli import helpers
    from pb.cli.input_router import QuestionCommandBuffer

    keys = iter(["/", "e", "a", "s", "i", "e", "r", "enter"])
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(
        ["one", "two"],
        "Header",
        multi=False,
        allow_slash_commands=True,
        command_buffer_state=QuestionCommandBuffer(),
    )

    assert result == "/easier"


def test_simple_tty_pick_backspace_after_only_slash_cancels_command_mode(monkeypatch):
    from pb.cli import helpers
    from pb.cli.input_router import QuestionCommandBuffer

    keys = iter(["/", "backspace", "down", "enter"])
    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda *args, **kwargs: None)

    result = helpers._simple_tty_pick(
        ["one", "two"],
        "Header",
        multi=False,
        allow_slash_commands=True,
        command_buffer_state=QuestionCommandBuffer(),
    )

    assert result == [1]


def test_simple_tty_pick_arrow_redraw_keeps_single_header(monkeypatch):
    from pb.cli import helpers

    prompt = "To focus the theory, which conceptual layer should we clarify first?"
    renders: list[list[str]] = []
    keys = iter(["down", "down", "q"])

    monkeypatch.setattr(helpers, "_read_key", lambda: next(keys))
    monkeypatch.setattr(helpers, "_draw", lambda lines, prev: renders.append(list(lines)))

    result = helpers._simple_tty_pick(
        ["Core definitions", "Why the mechanism works", "One guided example"],
        f"Choose one\n  {prompt}",
        multi=False,
    )

    assert result is None
    assert len(renders) >= 3
    for render in renders:
        joined = "\n".join(render)
        assert joined.count("Choose one") == 1
        assert joined.count("To focus the theory") == 1


def test_prompt_toolkit_picker_accepts_back_navigation_keyword():
    import inspect

    from pb.cli import helpers

    assert "allow_back_navigation" in inspect.signature(helpers._prompt_toolkit_pick).parameters
