# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""CLI helper functions for user interaction."""

from dataclasses import dataclass
import re
import shutil
import sys
import textwrap
import os
from datetime import timedelta
from typing import Callable, Optional, TypeVar

import typer

from pb.cli.input_router import QuestionCommandBuffer
from pb.core.matching import MatchCandidate, resolve_strict_match
from pb.core.naming import stored_short_title

try:
    from prompt_toolkit.application import Application
    from prompt_toolkit.formatted_text import FormattedText
    from prompt_toolkit.keys import Keys
    from prompt_toolkit.key_binding import KeyBindings
    from prompt_toolkit.layout import Layout, Window
    from prompt_toolkit.layout.controls import FormattedTextControl
    _PROMPT_TOOLKIT_AVAILABLE = True
except ImportError:  # pragma: no cover - prompt_toolkit is a declared dependency
    Application = None  # type: ignore[assignment]
    FormattedText = list  # type: ignore[assignment]
    Keys = None  # type: ignore[assignment]
    KeyBindings = None  # type: ignore[assignment]
    Layout = None  # type: ignore[assignment]
    Window = None  # type: ignore[assignment]
    FormattedTextControl = None  # type: ignore[assignment]
    _PROMPT_TOOLKIT_AVAILABLE = False

try:
    import readline  # noqa: F401 — enables arrow-key editing in input()/typer.prompt()
except ImportError:
    pass

from pb.domain.exceptions import ExitCode

T = TypeVar("T")
_SLASH_COMMAND_SENTINEL = "__PB_SLASH_COMMAND__"


@dataclass(frozen=True)
class ConfirmationDecision:
    """Parsed intent for a yes/no style prompt."""

    kind: str
    text: str = ""
    action: str = ""


_YES_WORDS = {
    "y",
    "yes",
    "yeah",
    "yep",
    "sure",
    "ok",
    "okay",
    "go",
    "continue",
    "proceed",
    "do it",
    "go ahead",
    "apply",
    "accept",
    "create it",
    "run it",
    "execute",
}
_NO_WORDS = {
    "n",
    "no",
    "nope",
    "nah",
    "cancel",
    "stop",
    "skip",
    "later",
    "not now",
    "never mind",
}
_MODIFY_HINTS = ("change", "edit", "refine", "modify", "different", "half", "double", "reduce", "less", "more")
_CLARIFY_HINTS = ("why", "what", "explain", "clarify", "help")

def _read_key() -> str:
    """Read a single keypress, returning a named action string."""
    import termios
    import tty
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            seq = sys.stdin.read(1)
            if seq == "[":
                code = sys.stdin.read(1)
                if code == "A":
                    return "up"
                if code == "B":
                    return "down"
                if code == "C":
                    return "right"
                if code == "D":
                    return "left"
            return "esc"
        if ch in ("\r", "\n"):
            return "enter"
        if ch == " ":
            return "space"
        if ch == "\x0f":
            return "ctrl-o"
        if ch in ("\x08", "\x7f"):
            return "backspace"
        if ch == "\x03":
            return "ctrl-c"
        if ch == "\x04":
            return "ctrl-d"
        if ch in ("q", "Q"):
            return "q"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _task_label(task, active_task_id: Optional[str] = None) -> str:
    """Format task as '{title}  [{completion}%]' or '{title}  [working]' if active."""
    title = task.title
    completion = task.completion if hasattr(task, "completion") else 0
    if active_task_id and task.id == active_task_id:
        suffix = "  [working]"
    elif completion > 0:
        suffix = f"  [{completion}%]"
    else:
        suffix = ""
    return f"{title}{suffix}"


_SCROLL_THRESHOLD = 30


def _truncate_to_width(line: str, width: int) -> str:
    """Clip a rendered line so it occupies exactly one terminal row.

    The redraw routine (`_draw`/`_screen_line_count`) moves the cursor up
    one row per rendered line. A line wider than the terminal wraps onto
    extra physical rows, so the cursor-up count falls short on redraw and
    stale rows accumulate. Clipping to `width - 1` keeps every line on a
    single row regardless of the terminal's autowrap behaviour.
    """
    limit = max(1, width - 1)
    if len(line) <= limit:
        return line
    return line[: max(0, limit - 1)] + "…"


def _wrap_picker_text(text: str, width: int, prefix: str) -> list[str]:
    """Wrap text into explicit terminal-width-safe lines."""
    available = max(8, width - len(prefix) - 1)
    paragraphs = str(text or "").splitlines() or [""]
    lines: list[str] = []
    continuation_prefix = " " * len(prefix)
    for paragraph in paragraphs:
        wrapped = textwrap.wrap(
            paragraph,
            width=available,
            break_long_words=True,
            break_on_hyphens=False,
        ) or [""]
        for idx, chunk in enumerate(wrapped):
            active_prefix = prefix if idx == 0 else continuation_prefix
            lines.append(_truncate_to_width(f"{active_prefix}{chunk}", width))
    return lines


def _append_picker_entry(
    lines: list[str],
    prefix: str,
    text: str,
    *,
    width: int,
    verbose: bool,
) -> None:
    """Append a picker row with explicit wrapping so text stays visible."""
    lines.extend(_wrap_picker_text(text, width, prefix))


def _render_picker(
    labels: list[str],
    cursor: int,
    checked: set[int],
    header: str,
    multi: bool,
    width: Optional[int] = None,
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    preview_open: bool = False,
    verbose: bool = False,
    editable_index: Optional[int] = None,
    editable_text: str = "",
    editable_placeholder: str = "Type here",
    command_mode: bool = False,
    command_buffer_text: str = "",
) -> list[str]:
    """Build display lines for the picker, with scrollable viewport for >30 items.

    Text is explicitly wrapped to the terminal width so long labels stay fully
    visible without relying on terminal autowrap.
    """
    if width is None:
        width = shutil.get_terminal_size().columns
    total_items = len(labels) + (1 if multi else 0)
    use_viewport = len(labels) > _SCROLL_THRESHOLD
    if use_viewport:
        viewport_size = _SCROLL_THRESHOLD
        half = viewport_size // 2
        start = max(0, cursor - half)
        end = start + viewport_size
        if end > len(labels):
            end = len(labels)
            start = max(0, end - viewport_size)
        visible_range = range(start, end)
    else:
        visible_range = range(len(labels))

    header_lines = [part.rstrip() for part in str(header or "").split("\n") if part.strip()]
    lines = [f"  {header_lines[0]}" if header_lines else "  Choose"]
    for extra in header_lines[1:]:
        lines.extend(_wrap_picker_text(extra, width, "    "))
    lines.extend(_wrap_picker_text("─" * max(8, min(width - 4, 36)), width, "  "))
    if use_viewport and visible_range.start > 0:
        lines.append(f"  ▲ {visible_range.start} more above")
    for i in visible_range:
        arrow = "❯" if i == cursor else "•"
        label = labels[i]
        if editable_index is not None and i == editable_index:
            placeholder = editable_placeholder.strip() or "Type here"
            label = f"{editable_text}" if editable_text else f"<{placeholder}>"
        if verbose and verbose_labels and i < len(verbose_labels):
            label = verbose_labels[i]
        if i == cursor:
            label = f"[ {label} ]"
        if multi:
            mark = "■" if i in checked else "□"
            _append_picker_entry(
                lines,
                f"  {arrow} {i + 1}. {mark} ",
                label,
                width=width,
                verbose=verbose,
            )
        else:
            _append_picker_entry(
                lines,
                f"  {arrow} {i + 1}. ",
                label,
                width=width,
                verbose=verbose,
            )
    if use_viewport and visible_range.stop < len(labels):
        lines.append(f"  ▼ {len(labels) - visible_range.stop} more below")
    if multi:
        arrow = "❯" if cursor == len(labels) else "•"
        n = len(checked)
        lines.extend(_wrap_picker_text(f"Submit selections ({n} selected)", width, f"  {arrow}    "))
    if preview_open and 0 <= cursor < len(labels):
        preview_text = labels[cursor]
        if details and cursor < len(details):
            preview_text = details[cursor]
        lines.extend(_wrap_picker_text("Preview:", width, "  "))
        lines.extend(_wrap_picker_text(preview_text, width, "    "))
    if command_mode:
        lines.extend(_wrap_picker_text("Command:", width, "  "))
        lines.extend(_wrap_picker_text(command_buffer_text or "/", width, "    "))
    if multi:
        controls = "  Controls: digits toggle  Space/Enter toggle  ↓ to submit  → preview  Ctrl+O details"
    else:
        controls = "  Controls: digits jump  Enter select  → preview  Ctrl+O details"
    if editable_index is not None:
        controls += "  type edits inline  Backspace delete"
    controls += "  Q cancel"
    lines.extend(_wrap_picker_text(controls, width, "  "))
    return lines


def _screen_line_count(lines: list[str]) -> int:
    """Count actual screen lines, accounting for embedded newlines."""
    return sum(1 + line.count("\n") for line in lines)


def _draw(lines: list[str], prev_count: int) -> None:
    """Redraw the picker, clearing previous output first."""
    if prev_count > 0:
        sys.stdout.write(f"\033[{prev_count}A")
        for _ in range(prev_count):
            sys.stdout.write("\033[2K\n")
        sys.stdout.write(f"\033[{prev_count}A")
    for line in lines:
        sys.stdout.write(line + "\n")
    sys.stdout.flush()


def select_from_numbered_list(
    items: list[T],
    format_item: Callable[[T, int], str],
    prompt: str = "Select number",
    header: Optional[str] = None,
) -> T:
    """Interactive numbered list selection with arrow navigation.

    Per D-09: Numbered selection for all block commands.
    """
    if not items:
        typer.echo("No items available.", err=True)
        raise typer.Exit(code=ExitCode.NOT_FOUND)

    labels = [format_item(item, i + 1) for i, item in enumerate(items)]
    result = _interactive_pick(labels, header or prompt, multi=False)
    if result is None:
        raise typer.Exit(code=ExitCode.SUCCESS)
    return items[result[0]]


def format_task_for_selection(task, index: int) -> str:
    """Format a task for numbered selection display."""
    title = task.title
    completion = task.completion if hasattr(task, "completion") else 0
    suffix = f"  [{completion}%]" if completion > 0 else ""
    return f"{title}{suffix}"


def format_block_for_selection(block, repo, index: int) -> str:
    """Format a block for numbered selection display per D-09."""
    task = repo.get_task(block.task_id)
    task_title = stored_short_title(task) if task else "Unknown"

    if block.start_time:
        start = block.start_time.strftime("%H:%M")
        end = (block.start_time + timedelta(minutes=block.duration_minutes)).strftime("%H:%M")
        time_str = f"{start}-{end}"
    else:
        time_str = "unscheduled"

    return f"{time_str} ({block.duration_minutes}m): {task_title}"


def _is_real_tty() -> bool:
    """Check if stdin is a real terminal (not CliRunner or pipe)."""
    try:
        fd = sys.stdin.fileno()
        return os.isatty(fd)
    except Exception:
        return False


def _fallback_pick(
    labels: list[str],
    header: str,
    multi: bool,
) -> Optional[list[int]]:
    """Simple input()-based fallback for non-TTY contexts (tests, pipes)."""
    print(f"  {header}:")
    for i, label in enumerate(labels):
        print(f"  {i + 1}. {label}")

    try:
        raw = input("Enter number: ").strip()
    except (EOFError, KeyboardInterrupt):
        return None

    if not raw:
        return None
    if raw[0].lower() == "q":
        return None
    if not raw[0].isdigit():
        return None
    idx = int(raw[0]) - 1
    if 0 <= idx < len(labels):
        return [idx]
    return None


def prompt_text(label: str, *, default: str = "", err: bool = False) -> str:
    """Prompt without Click-style [default] decorations."""
    return str(typer.prompt(label, default=default, show_default=False, err=err)).strip()


def interpret_confirmation(raw: str, *, default: bool = False, mode: str = "standard") -> ConfirmationDecision:
    """Interpret a yes/no prompt response without forcing exact tokens."""
    normalized = " ".join((raw or "").strip().lower().split())
    if not normalized:
        return ConfirmationDecision("accept" if default else "cancel")
    if normalized in _YES_WORDS:
        return ConfirmationDecision("accept")
    if normalized in _NO_WORDS:
        return ConfirmationDecision("cancel")
    if any(normalized.startswith(word) for word in ("yes ", "sure ", "ok ", "okay ")):
        return ConfirmationDecision("accept")
    if any(normalized.startswith(word) for word in ("no ", "don't", "do not", "cancel ", "skip ")):
        return ConfirmationDecision("cancel")

    if mode == "preview":
        return ConfirmationDecision("modify", text=raw.strip())

    if any(hint in normalized for hint in _MODIFY_HINTS):
        return ConfirmationDecision("alternative", text=raw.strip(), action="modify")
    if any(hint in normalized for hint in _CLARIFY_HINTS):
        return ConfirmationDecision("alternative", text=raw.strip(), action="clarify")
    return ConfirmationDecision("cancel", text=raw.strip())


def prompt_confirmation(
    label: str,
    *,
    default: bool = False,
    err: bool = False,
    mode: str = "standard",
) -> ConfirmationDecision:
    """Prompt and parse a yes/no-style reply with light intent inference."""
    hint = "Y/n" if default else "y/N"
    default_value = "y" if default else "n"
    raw = prompt_text(f"{label} ({hint})", default=default_value, err=err)
    return interpret_confirmation(raw, default=default, mode=mode)


def confirm_choice(label: str, *, default: bool = False, err: bool = False) -> bool:
    """Confirm without forcing the user to type literal y/n."""
    decision = prompt_confirmation(label, default=default, err=err, mode="standard")
    return decision.kind == "accept"


def _interactive_pick(
    labels: list[str],
    header: str,
    multi: bool,
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    allow_slash_commands: bool = False,
    editable_index: Optional[int] = None,
    editable_state: Optional[dict[str, str]] = None,
    editable_placeholder: str = "Type here",
    command_buffer_state: Optional[QuestionCommandBuffer] = None,
) -> Optional[list[int] | str]:
    """Arrow-navigated picker. Returns list of selected indices or None.

    Uses prompt_toolkit for TTY selection and falls back to simple numbered
    input when not running in a real terminal.
    """
    if not _is_real_tty():
        return _fallback_pick(labels, header, multi)
    if not _PROMPT_TOOLKIT_AVAILABLE:
        return _simple_tty_pick(
            labels,
            header,
            multi,
            details=details,
            verbose_labels=verbose_labels,
            editable_index=editable_index,
            editable_state=editable_state,
            editable_placeholder=editable_placeholder,
        )

    try:
        return _prompt_toolkit_pick(
            labels,
            header,
            multi,
            details=details,
            verbose_labels=verbose_labels,
            allow_slash_commands=allow_slash_commands,
            editable_index=editable_index,
            editable_state=editable_state,
            editable_placeholder=editable_placeholder,
            command_buffer_state=command_buffer_state,
        )
    except Exception:
        return _simple_tty_pick(
            labels,
            header,
            multi,
            details=details,
            verbose_labels=verbose_labels,
            allow_slash_commands=allow_slash_commands,
            editable_index=editable_index,
            editable_state=editable_state,
            editable_placeholder=editable_placeholder,
            command_buffer_state=command_buffer_state,
        )


def _picker_formatted_text(lines: list[str]) -> FormattedText:
    """Convert picker lines into prompt_toolkit fragments."""
    fragments: list[tuple[str, str]] = []
    active_wrap = False
    for index, line in enumerate(lines):
        style = ""
        stripped = line.lstrip()
        if "Controls:" in line or stripped.startswith("Preview:"):
            style = "bold"
        elif stripped.startswith("❯"):
            style = "reverse bold"
            active_wrap = "[" in line and "]" not in line
        elif active_wrap:
            style = "reverse bold"
            if "]" in line:
                active_wrap = False
        elif stripped.startswith("─"):
            style = "ansibrightblack"
        fragments.append((style, line))
        if index < len(lines) - 1:
            fragments.append(("", "\n"))
    return FormattedText(fragments)


def _prompt_toolkit_pick(
    labels: list[str],
    header: str,
    multi: bool,
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    allow_slash_commands: bool = False,
    editable_index: Optional[int] = None,
    editable_state: Optional[dict[str, str]] = None,
    editable_placeholder: str = "Type here",
    command_buffer_state: Optional[QuestionCommandBuffer] = None,
) -> Optional[list[int] | str]:
    """Run a simple picker that cooperates with prompt_toolkit REPL sessions."""
    cursor = 0
    checked: set[int] = set()
    total = len(labels) + (1 if multi else 0)
    preview_open = False
    verbose = False
    result: Optional[list[int] | str] = None

    def _editable_text() -> str:
        return str((editable_state or {}).get("text", ""))

    def _set_editable_text(text: str) -> None:
        if editable_state is None or editable_index is None:
            return
        editable_state["text"] = text
        if multi:
            if text:
                checked.add(editable_index)
            else:
                checked.discard(editable_index)

    def _editing_inline() -> bool:
        return editable_index is not None and editable_state is not None and cursor == editable_index

    def _command_active() -> bool:
        return bool(command_buffer_state is not None and command_buffer_state.active)

    def _command_text() -> str:
        return command_buffer_state.text if command_buffer_state is not None else ""

    def _activate_command() -> None:
        if command_buffer_state is not None:
            command_buffer_state.activate()

    def _append_command(data: str) -> None:
        if command_buffer_state is None:
            return
        if not command_buffer_state.active:
            command_buffer_state.activate()
        command_buffer_state.append(data)

    def _backspace_command() -> None:
        if command_buffer_state is not None:
            command_buffer_state.backspace()

    def render() -> FormattedText:
        return _picker_formatted_text(
            _render_picker(
                labels,
                cursor,
                checked,
                header,
                multi,
                details=details,
                verbose_labels=verbose_labels,
                preview_open=preview_open,
                verbose=verbose,
                editable_index=editable_index,
                editable_text=_editable_text(),
                editable_placeholder=editable_placeholder,
                command_mode=_command_active(),
                command_buffer_text=_command_text(),
            )
        )

    def finish(selection: Optional[list[int] | str]) -> None:
        nonlocal result
        result = selection
        app.exit()

    control = FormattedTextControl(render, focusable=True, show_cursor=False)
    bindings = KeyBindings()

    @bindings.add("up")
    def _move_up(event) -> None:
        nonlocal cursor
        if _command_active():
            return
        cursor = (cursor - 1) % total
        event.app.invalidate()

    @bindings.add("down")
    def _move_down(event) -> None:
        nonlocal cursor
        if _command_active():
            return
        cursor = (cursor + 1) % total
        event.app.invalidate()

    @bindings.add("left")
    def _hide_preview(event) -> None:
        nonlocal preview_open
        if _command_active():
            return
        preview_open = False
        event.app.invalidate()

    @bindings.add("right")
    def _show_preview(event) -> None:
        nonlocal preview_open
        if _command_active():
            return
        preview_open = cursor < len(labels)
        event.app.invalidate()

    @bindings.add("c-o")
    def _toggle_verbose(event) -> None:
        nonlocal verbose
        if _command_active():
            return
        verbose = not verbose
        event.app.invalidate()

    @bindings.add("q")
    def _cancel_q(event) -> None:
        if _command_active():
            _append_command("q")
            event.app.invalidate()
            return
        finish(None)

    @bindings.add("escape")
    def _cancel_escape(event) -> None:
        if _command_active():
            if command_buffer_state is not None:
                command_buffer_state.clear()
            event.app.invalidate()
            return
        finish(None)

    @bindings.add("c-c")
    @bindings.add("c-d")
    def _cancel(_event) -> None:
        finish(None)

    if allow_slash_commands:
        @bindings.add("/")
        def _slash_command(event) -> None:
            if _command_active():
                _append_command("/")
            else:
                _activate_command()
            event.app.invalidate()

    def accept_current(event) -> None:
        if _command_active():
            finish(_command_text() or _SLASH_COMMAND_SENTINEL)
            return
        if multi:
            if cursor == len(labels):
                if checked:
                    finish(sorted(checked))
                return
            if cursor in checked:
                checked.discard(cursor)
                if cursor == editable_index:
                    _set_editable_text("")
            else:
                checked.add(cursor)
            event.app.invalidate()
            return
        if cursor < len(labels):
            finish([cursor])

    @bindings.add("enter")
    def _accept_enter(event) -> None:
        accept_current(event)

    @bindings.add(" ")
    def _accept_space(event) -> None:
        if _command_active():
            _append_command(" ")
            event.app.invalidate()
            return
        if _editing_inline():
            _set_editable_text(_editable_text() + " ")
            event.app.invalidate()
            return
        if multi:
            if cursor < len(labels):
                if cursor in checked:
                    checked.discard(cursor)
                    if cursor == editable_index:
                        _set_editable_text("")
                else:
                    checked.add(cursor)
                event.app.invalidate()
            return
        accept_current(event)

    @bindings.add("backspace")
    def _inline_backspace(event) -> None:
        if _command_active():
            _backspace_command()
            event.app.invalidate()
            return
        if not _editing_inline():
            return
        _set_editable_text(_editable_text()[:-1])
        event.app.invalidate()

    if Keys is not None:
        @bindings.add(Keys.Any)
        def _inline_any(event) -> None:
            if _command_active():
                data = event.data or ""
                if not data or not data.isprintable():
                    return
                _append_command(data)
                event.app.invalidate()
                return
            if not _editing_inline():
                return
            data = event.data or ""
            if not data or not data.isprintable():
                return
            _set_editable_text(_editable_text() + data)
            event.app.invalidate()

    for digit in "123456789":
        @bindings.add(digit)
        def _pick_digit(event, digit=digit) -> None:
            if _command_active():
                _append_command(digit)
                event.app.invalidate()
                return
            if _editing_inline():
                _set_editable_text(_editable_text() + digit)
                event.app.invalidate()
                return
            index = int(digit) - 1
            if not 0 <= index < len(labels):
                return
            if multi:
                if index in checked:
                    checked.discard(index)
                else:
                    checked.add(index)
                event.app.invalidate()
                return
            finish([index])

    app = Application(
        layout=Layout(Window(content=control, wrap_lines=True)),
        key_bindings=bindings,
        full_screen=False,
        mouse_support=False,
        erase_when_done=True,
    )
    app.run()
    return result


def _simple_tty_pick(
    labels: list[str],
    header: str,
    multi: bool,
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    allow_slash_commands: bool = False,
    editable_index: Optional[int] = None,
    editable_state: Optional[dict[str, str]] = None,
    editable_placeholder: str = "Type here",
    command_buffer_state: Optional[QuestionCommandBuffer] = None,
) -> Optional[list[int] | str]:
    """Fallback interactive picker for real terminals when prompt_toolkit fails."""
    cursor = 0
    checked: set[int] = set()
    total = len(labels) + (1 if multi else 0)
    prev_count = 0
    preview_open = False
    verbose = False

    def _editable_text() -> str:
        return str((editable_state or {}).get("text", ""))

    def _set_editable_text(text: str) -> None:
        if editable_state is None or editable_index is None:
            return
        editable_state["text"] = text
        if multi:
            if text:
                checked.add(editable_index)
            else:
                checked.discard(editable_index)

    def _editing_inline() -> bool:
        return editable_index is not None and editable_state is not None and cursor == editable_index

    def _command_active() -> bool:
        return bool(command_buffer_state is not None and command_buffer_state.active)

    def _command_text() -> str:
        return command_buffer_state.text if command_buffer_state is not None else ""

    def _activate_command() -> None:
        if command_buffer_state is not None:
            command_buffer_state.activate()

    def _append_command(data: str) -> None:
        if command_buffer_state is None:
            return
        if not command_buffer_state.active:
            command_buffer_state.activate()
        command_buffer_state.append(data)

    def _backspace_command() -> None:
        if command_buffer_state is not None:
            command_buffer_state.backspace()

    lines = _render_picker(
        labels,
        cursor,
        checked,
        header,
        multi,
        details=details,
        verbose_labels=verbose_labels,
        preview_open=preview_open,
        verbose=verbose,
        editable_index=editable_index,
        editable_text=_editable_text(),
        editable_placeholder=editable_placeholder,
        command_mode=_command_active(),
        command_buffer_text=_command_text(),
    )
    _draw(lines, 0)
    prev_count = _screen_line_count(lines)

    while True:
        key = _read_key()

        if key in ("esc", "q", "ctrl-c", "ctrl-d"):
            if _command_active() and key == "q":
                _append_command("q")
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            if _command_active() and key == "esc":
                if command_buffer_state is not None:
                    command_buffer_state.clear()
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            return None

        if key == "up":
            if _command_active():
                continue
            cursor = (cursor - 1) % total
        elif key == "down":
            if _command_active():
                continue
            cursor = (cursor + 1) % total
        elif key == "right":
            if _command_active():
                continue
            preview_open = cursor < len(labels)
        elif key == "left":
            if _command_active():
                continue
            preview_open = False
        elif key == "ctrl-o":
            if _command_active():
                continue
            verbose = not verbose
        elif key == "backspace":
            if _command_active():
                _backspace_command()
            elif _editing_inline():
                _set_editable_text(_editable_text()[:-1])
        elif key in ("enter", "space"):
            if _command_active():
                if key == "space":
                    _append_command(" ")
                else:
                    return _command_text() or _SLASH_COMMAND_SENTINEL
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            if key == "space" and _editing_inline():
                _set_editable_text(_editable_text() + " ")
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            if multi:
                if cursor == len(labels):
                    if checked:
                        return sorted(checked)
                    continue
                if cursor in checked:
                    checked.discard(cursor)
                    if cursor == editable_index:
                        _set_editable_text("")
                else:
                    checked.add(cursor)
            elif cursor < len(labels):
                return [cursor]
        elif key.isdigit():
            if _command_active():
                _append_command(key)
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            if _editing_inline():
                _set_editable_text(_editable_text() + key)
                lines = _render_picker(
                    labels,
                    cursor,
                    checked,
                    header,
                    multi,
                    details=details,
                    verbose_labels=verbose_labels,
                    preview_open=preview_open,
                    verbose=verbose,
                    editable_index=editable_index,
                    editable_text=_editable_text(),
                    editable_placeholder=editable_placeholder,
                    command_mode=_command_active(),
                    command_buffer_text=_command_text(),
                )
                _draw(lines, prev_count)
                prev_count = _screen_line_count(lines)
                continue
            idx = int(key) - 1
            if 0 <= idx < len(labels):
                if multi:
                    if idx in checked:
                        checked.discard(idx)
                    else:
                        checked.add(idx)
                else:
                    return [idx]
        elif allow_slash_commands and key == "/":
            if _command_active():
                _append_command("/")
            else:
                _activate_command()
        elif _editing_inline() and len(key) == 1 and key.isprintable():
            _set_editable_text(_editable_text() + key)
        elif _command_active() and len(key) == 1 and key.isprintable():
            _append_command(key)

        lines = _render_picker(
            labels,
            cursor,
            checked,
            header,
            multi,
            details=details,
            verbose_labels=verbose_labels,
            preview_open=preview_open,
            verbose=verbose,
            editable_index=editable_index,
            editable_text=_editable_text(),
            editable_placeholder=editable_placeholder,
            command_mode=_command_active(),
            command_buffer_text=_command_text(),
        )
        _draw(lines, prev_count)
        prev_count = _screen_line_count(lines)


# NOTE: pick_task is kept for backward compatibility. New code should use
# pb.cli.pickers.pick_task_dialog or pick_or_prompt directly.
def pick_task(
    tasks: list,
    prompt_text: str = "Select a task",
    multi_select: bool = False,
    allow_nlp: bool = True,
) -> Optional[list]:
    """Interactive task picker with arrow navigation (D-18, D-19).

    Arrow keys navigate, Enter selects/toggles, number keys select directly.
    q/Esc/Ctrl-C/Ctrl-D cancels.

    Returns selected task(s) as a list, or None if cancelled.
    """
    if not tasks:
        typer.echo("No tasks available.", err=True)
        return None

    labels = [_task_label(t) for t in tasks]
    if allow_nlp:
        labels.append("[search by description]")

    result = _interactive_pick(labels, prompt_text, multi=multi_select)
    if result is None:
        return None

    selected = []
    nlp_idx = len(tasks) if allow_nlp else -1
    for idx in result:
        if idx == nlp_idx:
            try:
                desc = input("  Describe the task: ").strip()
            except (EOFError, KeyboardInterrupt):
                continue
            match_result = _nlp_match_task_result(desc, tasks)
            if match_result.accepted and match_result.matched_index is not None:
                match = tasks[match_result.matched_index]
                selected.append(match)
                typer.echo(f"  Matched: {match.title}")
            else:
                typer.echo("  I don't know which task you mean.", err=True)
                for suggestion_index in match_result.suggestions[:3]:
                    typer.echo(f"    - {tasks[suggestion_index].title}", err=True)
        elif idx < len(tasks):
            selected.append(tasks[idx])

    return selected if selected else None


def _nlp_match_task_result(description: str, tasks: list):
    """Strict-match a user description to a task or decline to guess."""
    candidates = []
    for task in tasks:
        summary = (getattr(task, "description", "") or "").splitlines()[0] if getattr(task, "description", "") else ""
        candidates.append(
            MatchCandidate(
                key=getattr(task, "id", ""),
                label=getattr(task, "title", ""),
                text=" | ".join(
                    part for part in [getattr(task, "title", ""), summary] if part
                ),
            )
        )

    return resolve_strict_match(description, candidates)


def _nlp_match_task(description: str, tasks: list):
    """Backward-compatible task matcher returning a task or None."""
    result = _nlp_match_task_result(description, tasks)
    if result.accepted and result.matched_index is not None:
        return tasks[result.matched_index]
    return None


def parse_duration(raw: str) -> Optional[int]:
    """Parse duration string into minutes.

    Per D-02: Tasks without duration prompt for duration before starting.
    Accepts flexible formats per Claude's Discretion area.

    Accepts:
        "30" - plain integer (minutes)
        "30m", "30 min", "30min", "30 minutes" - explicit minutes
        "0.5h", "1h", "1.5h", "1 hr" - hours converted to minutes

    Returns:
        Duration in minutes, or None if invalid format

    Examples:
        >>> parse_duration("30")
        30
        >>> parse_duration("30m")
        30
        >>> parse_duration("1.5h")
        90
        >>> parse_duration("invalid")
        None
    """
    raw = raw.strip().lower()

    if not raw:
        return None

    # Plain integer = minutes
    if raw.isdigit():
        return int(raw)

    token_re = re.compile(
        r"(\d+(?:\.\d+)?)\s*(hours?|hrs?|hr|h|minutes?|mins?|min|m)",
        re.IGNORECASE,
    )
    matches = list(token_re.finditer(raw))
    if not matches:
        return None

    cursor = 0
    total_minutes = 0.0
    for match in matches:
        gap = raw[cursor:match.start()]
        if gap.strip():
            return None
        value = float(match.group(1))
        unit = match.group(2).lower()
        if unit.startswith("h"):
            total_minutes += value * 60.0
        else:
            total_minutes += value
        cursor = match.end()

    if raw[cursor:].strip():
        return None

    minutes = int(round(total_minutes))
    return minutes if minutes > 0 else None
