# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Interactive context-file browser."""

from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

from pb.cli.helpers import _draw, _read_key, _screen_line_count, _wrap_picker_text
from pb.core.context_file_intake import canonical_class_from_path, sniff_mime_type


_SUITABLE_CONTEXT_CLASSES = {
    "archive.bundle",
    "document.pdf",
    "image.raster",
    "table.delimited",
    "text.code",
    "text.markup",
    "text.plain",
}
_SKIP_DIR_NAMES = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    ".venv",
    "venv",
    "node_modules",
}


@dataclass(frozen=True)
class _BrowserEntry:
    path: Path
    kind: str
    label: str
    detail: str = ""


def _file_size_label(path: Path) -> str:
    try:
        size = path.stat().st_size
    except OSError:
        return "?"
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"


def _visible_entries(directory: Path) -> tuple[list[Path], str]:
    try:
        entries = list(directory.iterdir())
    except PermissionError:
        return [], (
            "Access to this folder was denied by macOS. Allow your terminal app under "
            "System Settings > Privacy & Security > Files & Folders, then reopen this folder."
        )
    except OSError as error:
        return [], f"Could not read this folder: {error.strerror or error}"
    visible = sorted(
        [entry for entry in entries if not entry.name.startswith(".")],
        key=lambda item: (not item.is_dir(), item.name.lower()),
    )
    return visible, ""


def _is_suitable_context_file(path: Path) -> bool:
    if not path.is_file():
        return False
    canonical_class = canonical_class_from_path(path, mime_type=sniff_mime_type(path))
    return canonical_class in _SUITABLE_CONTEXT_CLASSES


def _recursive_suitable_files(root: Path) -> list[Path]:
    if root.is_file():
        return [root.resolve()] if _is_suitable_context_file(root) else []
    if not root.is_dir():
        return []
    found: list[Path] = []
    stack = [root]
    while stack:
        current = stack.pop()
        visible, _ = _visible_entries(current)
        for entry in visible:
            if entry.is_dir():
                if entry.name.lower() not in _SKIP_DIR_NAMES:
                    stack.append(entry)
                continue
            if _is_suitable_context_file(entry):
                found.append(entry.resolve())
    return sorted(dict.fromkeys(found), key=lambda item: str(item).lower())


def _entries_for(directory: Path) -> tuple[list[_BrowserEntry], str]:
    entries: list[_BrowserEntry] = []
    if directory.parent != directory:
        entries.append(_BrowserEntry(directory.parent, "parent", "../", str(directory.parent)))
    visible, notice = _visible_entries(directory)
    for entry in visible:
        if entry.is_dir():
            entries.append(_BrowserEntry(entry, "dir", f"{entry.name}/", f"Open {entry}"))
        elif entry.is_file():
            label = f"{entry.name}  {_file_size_label(entry)}"
            entries.append(_BrowserEntry(entry, "file", label, str(entry)))
    return entries, notice


def _render_context_browser(
    entries: list[_BrowserEntry],
    *,
    current: Path,
    cursor: int,
    selected: set[Path],
    notice: str = "",
) -> list[str]:
    width = shutil.get_terminal_size().columns
    total_items = len(entries) + 1
    viewport_size = 30
    if len(entries) > viewport_size:
        half = viewport_size // 2
        start = max(0, cursor - half)
        stop = start + viewport_size
        if stop > len(entries):
            stop = len(entries)
            start = max(0, stop - viewport_size)
        visible_range = range(start, stop)
    else:
        visible_range = range(len(entries))

    lines: list[str] = []
    lines.extend(_wrap_picker_text(f"Select context files from {current}", width, "  "))
    lines.extend(_wrap_picker_text(f"Selected files: {len(selected)}", width, "  "))
    lines.extend(_wrap_picker_text("-" * max(8, min(width - 4, 36)), width, "  "))
    if notice:
        lines.extend(_wrap_picker_text(notice, width, "  ! "))
    if visible_range.start > 0:
        lines.append(f"  ^ {visible_range.start} more above")
    for index in visible_range:
        entry = entries[index]
        arrow = ">" if cursor == index else " "
        if entry.kind == "file":
            mark = "[x]" if entry.path.resolve() in selected else "[ ]"
        else:
            mark = "   "
        lines.extend(_wrap_picker_text(entry.label, width, f"  {arrow} {index + 1}. {mark} "))
    if visible_range.stop < len(entries):
        lines.append(f"  v {len(entries) - visible_range.stop} more below")
    submit_arrow = ">" if cursor == len(entries) else " "
    lines.extend(_wrap_picker_text(f"Submit selected ({len(selected)})", width, f"  {submit_arrow}    "))
    controls = "Controls: Enter opens folders/toggles files  Space toggles files  ` submit  A select all here  C clear  Left parent  Q cancel"
    lines.extend(_wrap_picker_text(controls, width, "  "))
    return lines


def _activate_entry(entry: _BrowserEntry, selected: set[Path]) -> Path | None:
    if entry.kind in {"dir", "parent"}:
        return entry.path.resolve()
    if entry.kind == "file":
        resolved = entry.path.resolve()
        if resolved in selected:
            selected.remove(resolved)
        else:
            selected.add(resolved)
    return None


def pick_context_files(start_dir: Path | None = None) -> list[Path]:
    """Pick one or more files, navigating directories from `start_dir`."""

    if not sys.stdin.isatty():
        return []
    current = (start_dir or Path.cwd()).expanduser().resolve()
    selected: set[Path] = set()
    cursor = 0
    prev_count = 0

    while True:
        entries, notice = _entries_for(current)
        total = len(entries) + 1
        cursor = min(cursor, max(0, total - 1))
        lines = _render_context_browser(entries, current=current, cursor=cursor, selected=selected, notice=notice)
        _draw(lines, prev_count)
        prev_count = _screen_line_count(lines)
        key = _read_key()

        if key in {"q", "esc", "ctrl-c", "ctrl-d"}:
            return []
        if key == "up":
            cursor = (cursor - 1) % total
            continue
        if key == "down":
            cursor = (cursor + 1) % total
            continue
        if key == "left":
            if current.parent != current:
                current = current.parent.resolve()
                cursor = 0
            continue
        if key == "right" and cursor < len(entries) and entries[cursor].kind == "dir":
            current = entries[cursor].path.resolve()
            cursor = 0
            continue
        if key in {"a", "A"}:
            root = current
            if cursor < len(entries) and entries[cursor].kind == "dir":
                root = entries[cursor].path
            selected.update(_recursive_suitable_files(root))
            continue
        if key in {"c", "C"}:
            selected.clear()
            continue
        if key.isdigit() and key != "0":
            index = int(key) - 1
            if not 0 <= index < len(entries):
                continue
            new_dir = _activate_entry(entries[index], selected)
            if new_dir is not None:
                current = new_dir
                cursor = 0
            continue
        if key == "backtick":
            if selected:
                return sorted(selected, key=lambda item: str(item).lower())
            continue
        if key in {"enter", "space"}:
            if cursor == len(entries):
                if selected:
                    return sorted(selected, key=lambda item: str(item).lower())
                continue
            if not entries:
                continue
            if key == "space" and entries[cursor].kind != "file":
                continue
            new_dir = _activate_entry(entries[cursor], selected)
            if new_dir is not None:
                current = new_dir
                cursor = 0
