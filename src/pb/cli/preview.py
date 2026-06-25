# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Preview helpers for AI-generated drafts before durable writes."""

from __future__ import annotations

import os
import re
import sys
from typing import Optional

import typer
from rich.console import Group
from rich.markdown import Markdown as RichMarkdown
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text

from pb.cli.helpers import ConfirmationDecision, confirm_choice, prompt_confirmation
from pb.cli.console import get_console
from pb.core.renderables import renderable_cli_text


_ENUM_TOKENS = {
    # Bloom + practice stages
    "remember", "understand", "apply", "analyze", "evaluate", "create",
    "orient", "explore", "isolate", "integrate", "perform",
    # Modes / branches
    "study", "practise", "practice", "mixed", "manual", "auto", "focus",
    # Frameworks
    "bloom_retrieval", "deliberate_practice",
    # Difficulty
    "easy", "medium", "hard",
    # Feedback / evidence
    "artifact", "self", "tests", "peer", "coach",
    # Horizons
    "month", "quarter", "six_month", "today", "week",
}

_NUMERIC_RE = re.compile(r"^-?\d+(?:[.,]\d+)?(?:\s*[A-Za-z/]+)?$")


def _row_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, dict)):
        return renderable_cli_text(value)
    if hasattr(value, "text") and hasattr(value, "is_latex"):
        return renderable_cli_text(value)
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return str(getattr(value, "value"))
    return str(value)


def _styled_value(value: str) -> Text:
    """Color-code a value cell based on its shape (number / enum / prose)."""
    stripped = value.strip()
    if _NUMERIC_RE.match(stripped):
        return Text(value, style="yellow")
    if stripped.lower() in _ENUM_TOKENS:
        return Text(value, style="magenta")
    return Text(value, style="white")


def _build_kv_table(rows: list[tuple[str, object]]) -> Optional[Table]:
    cleaned = [(label, _row_value(v)) for label, v in rows if _row_value(v).strip()]
    if not cleaned:
        return None
    longest_label = max(len(label) for label, _ in cleaned)
    table = Table(show_header=False, box=None, padding=(0, 3), expand=True)
    table.add_column(style="dim bold", no_wrap=True, width=longest_label)
    table.add_column(overflow="fold")
    for label, value in cleaned:
        table.add_row(label, _styled_value(value))
    return table


def render_styled_preview(
    *,
    title: str,
    rows: Optional[list[tuple[str, object]]] = None,
    sections: Optional[list[tuple[str, list[tuple[str, object]]]]] = None,
    border_style: str = "panel.border",
) -> None:
    """Render a draft as a color-coded, styled tabular preview."""
    elements: list[object] = []
    if rows:
        table = _build_kv_table(rows)
        if table is not None:
            elements.append(table)
    for subtitle, sub_rows in (sections or []):
        sub_table = _build_kv_table(sub_rows)
        if sub_table is None:
            continue
        if elements:
            elements.append(Text())
            elements.append(Rule(subtitle, style="section.rule"))
            elements.append(Text())
            elements.append(sub_table)
    if not elements:
        elements.append(Text("(empty preview)", style="dim italic"))
    get_console().print(
        Panel(
            Group(*elements),
            title=f"[bold]{title}[/]",
            border_style=border_style,
            padding=(1, 2),
            expand=True,
        )
    )


def render_markdown_preview(
    *,
    title: str,
    rows: Optional[list[tuple[str, object]]] = None,
    sections: Optional[list[tuple[str, list[str] | object]]] = None,
    divider: str = "",
) -> None:
    """Render learner-facing previews as styled Rich panels.

    Section content can be a list of markdown strings (rendered via Rich
    Markdown) or any Rich renderable (Table, Group, etc.) for full control.
    """
    console = get_console()
    elements: list[object] = []

    if rows:
        kv = [(label, _markdown_preview_value(v)) for label, v in rows]
        kv = [(label, v) for label, v in kv if v.strip()]
        if kv:
            longest = max(len(label) for label, _ in kv)
            table = Table(show_header=False, box=None, padding=(0, 3), expand=True)
            table.add_column(style="dim bold", no_wrap=True, width=longest)
            table.add_column(overflow="fold")
            for label, value in kv:
                table.add_row(label, Text(value))
            elements.append(table)

    for section_title, section_content in (sections or []):
        if elements:
            elements.append(Text())
        if section_title:
            elements.append(Rule(section_title, style="section.rule"))
            elements.append(Text())
        if isinstance(section_content, list):
            clean_lines = [line.rstrip() for line in section_content if str(line).strip()]
            if not clean_lines:
                continue
            elements.append(RichMarkdown("\n".join(clean_lines)))
        else:
            elements.append(section_content)

    if not elements:
        elements.append(Text("(empty preview)", style="dim italic"))

    console.print(
        Panel(
            Group(*elements),
            title=f"[bold]{title}[/]",
            border_style="panel.border",
            padding=(1, 2),
            expand=True,
        )
    )


def markdown_learning_plan_lines(blocks: list[object], *, presentation: object | None = None) -> Group:
    """Format curriculum-plan blocks as compact learner-facing outline rows."""

    accent = _accent_style(presentation)
    gap_lines = _gap_lines(presentation)
    node_index = {
        str(getattr(block, "node_id", "") or "").strip(): index
        for index, block in enumerate(blocks, start=1)
        if str(getattr(block, "node_id", "") or "").strip()
    }
    elements: list[object] = []
    for index, block in enumerate(blocks, start=1):
        title = renderable_cli_text(
            str(getattr(block, "title", "") or getattr(block, "subject_scope", "") or "").strip()
        ).strip() or "Untitled block"
        branch = str(getattr(block, "branch", "") or "").strip()
        duration = getattr(block, "duration_minutes", None)
        depends_on = getattr(block, "depends_on", None) or []
        current_node_id = str(getattr(block, "node_id", "") or "").strip()
        success = renderable_cli_text(getattr(block, "success_check", "") or "").strip()
        reason = renderable_cli_text(getattr(block, "reason", "") or "").strip()

        line = Text()
        line.append("[", style="plan.bracket")
        line.append(str(index), style=f"bold {accent}")
        line.append("] ", style="plan.bracket")
        line.append(title, style="plan.title")
        meta: list[str] = []
        if branch:
            meta.append(branch)
        if duration:
            meta.append(f"{duration} min")
        if meta:
            line.append("  ", style="plan.meta")
            line.append(" · ".join(meta), style="plan.meta")
        elements.append(line)

        if success:
            detail = Text()
            detail.append("• ", style="plan.bullet")
            detail.append("Check: ", style="plan.label")
            detail.append(success, style="plan.detail")
            elements.append(detail)
        if reason:
            detail = Text()
            detail.append("• ", style="plan.bullet")
            detail.append("Why: ", style="plan.label")
            detail.append(reason, style="plan.detail")
            elements.append(detail)
        previous_node_id = next(
            (
                str(getattr(blocks[index - 2], "node_id", "") or "").strip()
                for _ in [0]
                if index > 1
            ),
            "",
        )
        dependency_labels = [
            f"Step {node_index[dependency]}"
            for dependency in depends_on
            if dependency in node_index
        ]
        should_show_dependencies = bool(depends_on)
        if len(depends_on) == 1 and previous_node_id and depends_on[0] == previous_node_id:
            should_show_dependencies = False
        if should_show_dependencies and dependency_labels:
            detail = Text()
            detail.append("• ", style="plan.bullet")
            detail.append("Depends: ", style="plan.label")
            detail.append(", ".join(dependency_labels), style="plan.detail")
            elements.append(detail)
        if index < len(blocks):
            elements.append(Text("\n" * gap_lines))
    return Group(*elements)


def _markdown_preview_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, dict)):
        return renderable_cli_text(value).strip()
    if hasattr(value, "text") and hasattr(value, "is_latex"):
        return renderable_cli_text(value).strip()
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        return str(getattr(value, "value")).strip()
    return str(value).strip()


def markdown_step_lines(steps: list[object]) -> list[str]:
    """Format ordered steps into Markdown lines (for glow/markdown contexts)."""
    lines: list[str] = []
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            title = str(step.get("title", "")).strip()
            instruction_value = step.get("instruction", "")
            success_value = step.get("success_check", "")
        else:
            title = str(getattr(step, "title", "") or "").strip()
            instruction_value = getattr(step, "instruction", "")
            success_value = getattr(step, "success_check", "")
        title = renderable_cli_text(title).strip()
        instruction = renderable_cli_text(instruction_value).strip()
        success_check = renderable_cli_text(success_value).strip()
        if not title:
            continue
        lines.append(f"{index}. **{title}**")
        if instruction:
            lines.append(f"   - Do: {instruction}")
        if success_check:
            lines.append(f"   - Check: {success_check}")
        lines.append("")
    if lines and lines[-1] == "":
        lines.pop()
    return lines


def build_step_table(steps: list[object], *, presentation: object | None = None) -> Group:
    """Format ordered steps as compact preview blocks instead of dense tables."""

    accent = _accent_style(presentation)
    gap_lines = _gap_lines(presentation)
    elements: list[object] = []
    for index, step in enumerate(steps, start=1):
        if isinstance(step, dict):
            title = str(step.get("title", "")).strip()
            instruction_value = step.get("instruction", "")
            success_value = step.get("success_check", "")
        else:
            title = str(getattr(step, "title", "") or "").strip()
            instruction_value = getattr(step, "instruction", "")
            success_value = getattr(step, "success_check", "")
        title = renderable_cli_text(title).strip()
        instruction = renderable_cli_text(instruction_value).strip()
        success_check = renderable_cli_text(success_value).strip()
        if not title:
            continue
        step_text = Text()
        step_text.append("[", style="step.bracket")
        step_text.append(str(index), style=f"bold {accent}")
        step_text.append("] ", style="step.bracket")
        step_text.append(title, style="step.title")
        if instruction:
            step_text.append("\n")
            step_text.append("• ", style="step.bullet")
            step_text.append("Do: ", style="step.label")
            step_text.append(instruction, style="step.detail")
        if success_check:
            step_text.append("\n")
            step_text.append("• ", style="step.bullet")
            step_text.append("Check: ", style="step.label")
            step_text.append(success_check, style="step.check")
        elements.append(step_text)
        if index < len(steps):
            elements.append(Text("\n" * gap_lines))
    return Group(*elements)


def _accent_style(presentation: object | None) -> str:
    accent = str(getattr(presentation, "accent", "") or "cyan").strip().lower()
    return accent if accent in {"cyan", "blue", "green", "yellow", "magenta"} else "cyan"


def _gap_lines(presentation: object | None) -> int:
    density = str(getattr(presentation, "density", "") or "balanced").strip().lower()
    if density == "compact":
        return 1
    if density == "relaxed":
        return 2
    return 1


# Back-compat shim: legacy callers that still import render_json_preview get
# the styled renderer with all top-level keys as rows. New code should call
# render_styled_preview directly with a curated field list.
def render_json_preview(title: str, payload: object) -> None:
    """Deprecated: use render_styled_preview with curated fields instead."""
    rows: list[tuple[str, object]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            rows.append((key.replace("_", " ").title(), value))
    else:
        rows.append(("Value", payload))
    render_styled_preview(title=title, rows=rows)


def preview_decision(*, yes: bool, action_label: str) -> ConfirmationDecision:
    """Return the user's intent for a draft preview."""
    auto_yes = (
        yes
        or os.environ.get("PRODUCTIVEBRAIN_AUTO_YES", "").strip().lower() in {"1", "true", "yes", "on"}
    )
    if not auto_yes:
        try:
            from pb.runtime import get_session_auto_yes

            auto_yes = get_session_auto_yes()
        except Exception:
            auto_yes = False

    if auto_yes:
        return ConfirmationDecision("accept")
    if not sys.stdin.isatty():
        console = get_console()
        console.print(f"[warn]Preview only. Re-run with `--yes` to {action_label.lower()}.[/]")
        return ConfirmationDecision("cancel")
    return prompt_confirmation(f"{action_label}?", default=True, mode="preview")


def confirm_preview(*, yes: bool, action_label: str) -> bool:
    """Backward-compatible preview confirmation returning a boolean."""
    return preview_decision(yes=yes, action_label=action_label).kind == "accept"
