# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Simple prompt-toolkit-friendly picker helpers."""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

import typer

if TYPE_CHECKING:
    from pb.domain.models import Task

T = TypeVar("T")

# Sentinel value indicating the user pressed "0" for manual input
_MANUAL_INPUT_SENTINEL = "__MANUAL_INPUT__"
_SLASH_COMMAND_SENTINEL = "__PB_SLASH_COMMAND__"


@dataclass(frozen=True)
class PickerResult:
    """Structured picker result for selection, inline text, commands, or cancel."""

    kind: str
    value: object | None = None

def _task_label(task: "Task", active_task_id: Optional[str] = None,
                paused_task_ids: Optional[set] = None) -> str:
    """Format a task for picker display.

    Shows completion percentage (only if >0), [working] if the task has an active session,
    and [paused] if the task is in the paused_task_ids set (resumable tasks).
    """
    title = task.title
    completion = task.completion if hasattr(task, "completion") else 0
    if active_task_id and task.id == active_task_id:
        suffix = "  [working]"
    elif paused_task_ids and task.id in paused_task_ids:
        suffix = "  [paused]"
    elif completion > 0:
        suffix = f"  [{completion}%]"
    else:
        suffix = ""
    return f"{title}{suffix}"


def pick_task_dialog(
    tasks: list,
    title: str = "Select task",
    active_task_id: Optional[str] = None,
    paused_task_ids: Optional[set] = None,
) -> Optional[object]:
    """Single-select inline picker using _interactive_pick.

    Returns the selected task object, or None if cancelled.

    Args:
        tasks: List of Task objects to display.
        title: Dialog title text.
        active_task_id: Task ID currently being worked on (shown with [working] label).
        paused_task_ids: Set of task IDs that are resumable (shown with [paused] label).

    Returns:
        Selected Task, or None if cancelled.
    """
    if not tasks:
        return None

    if not sys.stdin.isatty():
        return None  # Non-TTY: caller handles fallback

    from pb.cli.helpers import _interactive_pick
    labels = [_task_label(t, active_task_id, paused_task_ids=paused_task_ids) for t in tasks]
    result = _interactive_pick(labels, title, multi=False)
    if result is None:
        return None
    if result == _MANUAL_INPUT_SENTINEL:
        return _MANUAL_INPUT_SENTINEL
    return tasks[result[0]]


def pick_tasks_dialog(
    tasks: list,
    title: str = "Select tasks",
) -> list:
    """Multi-select inline picker using _interactive_pick.

    Args:
        tasks: List of Task objects to display.
        title: Dialog title text.

    Returns:
        List of selected Task objects (empty if cancelled).
    """
    if not tasks:
        return []

    if not sys.stdin.isatty():
        return []

    from pb.cli.helpers import _interactive_pick
    labels = [_task_label(t) for t in tasks]
    result = _interactive_pick(labels, title, multi=True)
    if result is None:
        return []
    return [tasks[i] for i in result]


def pick_skills_dialog(
    all_skills: list[str],
    pre_checked: list[str],
    title: str = "Tag skills for this task",
) -> list[str]:
    """Multi-select inline skill picker using _interactive_pick.

    Displays all known skills. An "Add new skill" sentinel lets users create a skill inline.
    Note: _interactive_pick doesn't currently support pre-checked defaults, so users will
    need to manually select them.

    Args:
        all_skills: All known skill names to display.
        pre_checked: Skill names that should ideally appear pre-selected (informational).
        title: Dialog title text.

    Returns:
        List of selected skill name strings (empty if cancelled or non-TTY).
    """
    if not all_skills and not pre_checked:
        return []

    if not sys.stdin.isatty():
        return []

    from pb.cli.helpers import _interactive_pick

    # Any pre_checked items not in all_skills are prepended
    extra_pre = [s for s in pre_checked if s not in all_skills]
    merged = list(dict.fromkeys(extra_pre + all_skills))

    labels = [f"{skill} *" if skill in pre_checked else skill for skill in merged]
    labels.append("+ Add new skill...")
    
    # We can use the header to inform users about pre-checked items
    header = title
    if pre_checked:
        header += f"\n  (Suggested: {', '.join(pre_checked)})"

    result = _interactive_pick(labels, header, multi=True)
    if not result:
        return []

    selected = []
    add_new_idx = len(labels) - 1
    
    for idx in result:
        if idx == add_new_idx:
            new_name = typer.prompt("New skill name", default="", show_default=False).strip()
            if new_name:
                selected.append(new_name)
        elif 0 <= idx < len(merged):
            selected.append(merged[idx])

    return selected


def pick_single_choice(
    options: list[tuple[str, str]],
    title: str = "Select",
    text: str = "",
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    allow_inline_edit: bool = False,
    inline_prompt: str = "Type your answer",
    return_result: bool = False,
    slash_registry=None,
    pb_command_resolver=None,
) -> Optional[str] | PickerResult:
    """Single-select inline picker using _interactive_pick."""
    if not sys.stdin.isatty():
        return PickerResult(kind="cancel") if return_result else None

    from pb.cli.helpers import _interactive_pick
    from pb.cli.input_router import QuestionCommandBuffer, classify_interactive_input
    labels = [label for _, label in options]
    inline_index = -1
    editable_state: dict[str, str] | None = None
    command_buffer = QuestionCommandBuffer() if slash_registry is not None else None
    if allow_inline_edit:
        inline_index = len(labels)
        labels.append("")
        editable_state = {"text": ""}
    header = title
    if text:
        header = f"{title}\n  {text}"
    while True:
        res = _interactive_pick(
            labels,
            header,
            multi=False,
            details=details,
            verbose_labels=verbose_labels,
            allow_slash_commands=slash_registry is not None,
            editable_index=inline_index if allow_inline_edit else None,
            editable_state=editable_state,
            editable_placeholder=inline_prompt,
            command_buffer_state=command_buffer,
        )
        if isinstance(res, str) and res.startswith("/"):
            routed = classify_interactive_input(
                res,
                pb_command_resolver=pb_command_resolver,
                slash_registry=slash_registry,
                active_learning=True,
                allow_shell_commands=False,
                allow_nl_dispatch=False,
            )
            return PickerResult(kind="command", value=routed) if return_result else None
        if isinstance(res, list) and res:
            if allow_inline_edit and res[0] == inline_index:
                typed = str((editable_state or {}).get("text", "")).strip()
                return PickerResult(kind="inline_text", value=typed or None) if return_result else (typed or None)
            selected = options[res[0]][0]
            return PickerResult(kind="selection", value=selected) if return_result else selected
        return PickerResult(kind="cancel") if return_result else None


def pick_many_choices(
    options: list[tuple[str, str]],
    title: str = "Select",
    text: str = "",
    *,
    details: Optional[list[str]] = None,
    verbose_labels: Optional[list[str]] = None,
    allow_inline_edit: bool = False,
    inline_prompt: str = "Type your answer",
    return_result: bool = False,
    slash_registry=None,
    pb_command_resolver=None,
) -> list[str] | PickerResult:
    """Multi-select inline picker returning the selected option values."""
    if not sys.stdin.isatty():
        return PickerResult(kind="cancel") if return_result else []

    from pb.cli.helpers import _interactive_pick
    from pb.cli.input_router import QuestionCommandBuffer, classify_interactive_input

    labels = [label for _, label in options]
    inline_index = -1
    editable_state: dict[str, str] | None = None
    command_buffer = QuestionCommandBuffer() if slash_registry is not None else None
    if allow_inline_edit:
        inline_index = len(labels)
        labels.append("")
        editable_state = {"text": ""}
    header = title
    if text:
        header = f"{title}\n  {text}"
    while True:
        res = _interactive_pick(
            labels,
            header,
            multi=True,
            details=details,
            verbose_labels=verbose_labels,
            allow_slash_commands=slash_registry is not None,
            editable_index=inline_index if allow_inline_edit else None,
            editable_state=editable_state,
            editable_placeholder=inline_prompt,
            command_buffer_state=command_buffer,
        )
        if isinstance(res, str) and res.startswith("/"):
            routed = classify_interactive_input(
                res,
                pb_command_resolver=pb_command_resolver,
                slash_registry=slash_registry,
                active_learning=True,
                allow_shell_commands=False,
                allow_nl_dispatch=False,
            )
            return PickerResult(kind="command", value=routed) if return_result else []
        if not isinstance(res, list) or not res:
            return PickerResult(kind="cancel") if return_result else []
        selected = [options[index][0] for index in res if 0 <= index < len(options)]
        if allow_inline_edit and inline_index in res:
            typed = str((editable_state or {}).get("text", "")).strip()
            if typed:
                selected.append(typed)
            if return_result and typed:
                return PickerResult(kind="inline_text", value=selected)  # includes typed value at the end
            return PickerResult(kind="selection", value=selected) if return_result else selected
        return PickerResult(kind="selection", value=selected) if return_result else selected


def pick_boolean(title: str, text: str = "") -> bool:
    """Yes/No TUI picker returning True/False."""
    if os.environ.get("PRODUCTIVEBRAIN_AUTO_YES", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    options = [("yes", "Yes"), ("no", "No")]
    res = pick_single_choice(options, title=title, text=text)
    return res == "yes"


def pick_deck(title: str = "Select Deck") -> Optional[str]:
    """TUI picker for selecting an Anki deck/domain, with manual fallback."""

    # We could query knowledge domains from the vault or Anki.
    # For now, let's use the known mapped domains in anki.py or just basic folders.
    try:
        from pb.vault import get_vault_path
        vault = get_vault_path()
        k_dir = vault / "knowledge"
        dirs = [d.name for d in k_dir.iterdir() if d.is_dir() and not d.name.startswith(".")]
    except Exception:
        dirs = []
        
    options = [(d, d) for d in sorted(dirs)]
    options.append(("__ADD_NEW__", "+ Type deck name manually..."))
    
    res = pick_single_choice(options, title=title, text="Select a domain/deck")
    if res == "__ADD_NEW__":
        import typer
        return typer.prompt("Deck name").strip()
    return res


def _pick_numbered_fallback(
    tasks: list,
    title: str = "Select task",
    active_task_id: Optional[str] = None,
    paused_task_ids: Optional[set] = None,
) -> Optional[object]:
    """Arrow-navigated fallback when radiolist_dialog is unavailable."""
    from pb.cli.helpers import _interactive_pick

    labels = [_task_label(task, active_task_id, paused_task_ids=paused_task_ids) for task in tasks]
    result = _interactive_pick(labels, title, multi=False)
    if result is None:
        return None
    return tasks[result[0]]


def pick_or_prompt(
    tasks: list,
    find_fn: Callable[[str], Optional[object]] = None,
    title: str = "Select task",
    active_task_id: Optional[str] = None,
    paused_task_ids: Optional[set] = None,
) -> Optional[object]:
    """TUI picker with numbered-list fallback.

    Tries the full radiolist dialog first. If it fails or is unavailable,
    falls back to a simple numbered list — never asks for raw task IDs.

    Args:
        tasks: List of Task objects.
        find_fn: Optional function to find a task by ID (unused, kept for compat).
        title: Dialog title text.
        active_task_id: Task ID currently being worked on (shown with [working] label).
        paused_task_ids: Set of task IDs that are resumable (shown with [paused] label).
    """
    result = pick_task_dialog(tasks, title=title, active_task_id=active_task_id,
                              paused_task_ids=paused_task_ids)

    if result is not None and result != _MANUAL_INPUT_SENTINEL:
        return result

    return _pick_numbered_fallback(tasks, title=title, active_task_id=active_task_id,
                                   paused_task_ids=paused_task_ids)


def timer_expiry_picker(task_title: str, overtime_min: int) -> Optional[str]:
    """Inline picker with 4 fixed choices for an expired timer.

    Returns one of: "extend_10", "add_time", "finish", "cancel".
    Non-TTY: returns None (caller skips).
    """
    if not sys.stdin.isatty():
        return None

    options = [
        ("extend_10", "[+10m] Extend 10 minutes"),
        ("add_time", "[Add Time] Custom amount"),
        ("finish", "[Finish] End session now"),
        ("cancel", "[Cancel] Dismiss (session continues)"),
    ]
    
    header = f"Timer expired ({overtime_min}m overtime) — {task_title}"
    res = pick_single_choice(options, title=header)
    return res if res is not None else "cancel"


def suggestion_picker(
    next_steps: list[str],
    skills: list[str],
    title: str = "Next steps + skills",
) -> tuple[list[str], list[str]]:
    """Inline multiselect picker for Flash Lite next-steps + skill suggestions.

    Returns (selected_next_steps, selected_skills).
    Non-TTY: returns ([], []).
    """
    if not sys.stdin.isatty():
        return [], []

    values = (
        [(f"next:{s}", f"[next] {s}") for s in next_steps]
        + [(f"skill:{s}", f"[skill] {s}") for s in skills]
    )
    if not values:
        return [], []

    from pb.cli.helpers import _interactive_pick
    labels = [label for _, label in values]
    
    result_idx = _interactive_pick(labels, title, multi=True)
    if not result_idx:
        return [], []

    selected_vals = [values[i][0] for i in result_idx]

    sel_next = [v[len("next:"):] for v in selected_vals if v.startswith("next:")]
    sel_skills = [v[len("skill:"):] for v in selected_vals if v.startswith("skill:")]
    return sel_next, sel_skills
