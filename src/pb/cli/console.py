# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared console factory for Rich CLI rendering.

Provides get_console() and get_err_console() with theme injection.
_plain_mode is set by main.py callback when --plain is passed.

Per D-05, D-06, D-07, D-08.
"""

import copy
import shutil
from typing import Any

from rich.console import Console, Group
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich.theme import Theme

from pb.core.renderables import renderable_cli_text

_plain_mode: bool = False
_prompt_abort_installed: bool = False
_ABORT_TOKENS = {"q", "quit", "exit"}
_DEFAULT_MAX_CONTENT_WIDTH = 80
_DEFAULT_CONTENT_WIDTH_RATIO = 0.70
_DEFAULT_TERMINAL_WIDTH = 120
_LATEX_RENDERED_META = "pb_latex_rendered"


def mark_latex_rendered_text(text: Text) -> Text:
    """Mark a Text object as already processed by the terminal math renderer."""
    if text.plain:
        text.apply_meta({_LATEX_RENDERED_META: True})
    return text


class LatexAwareConsole(Console):
    """Rich Console that applies terminal LaTeX rendering to printed content."""

    def print(self, *objects, **kwargs) -> None:  # type: ignore[override]
        rendered_objects = tuple(self._render_latex_object(obj) for obj in objects)
        super().print(*rendered_objects, **kwargs)

    def rule(self, title: str | Text = "", **kwargs) -> None:  # type: ignore[override]
        super().rule(self._render_latex_title(title), **kwargs)

    def _render_latex_object(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return renderable_cli_text(obj)
        if isinstance(obj, Text):
            return self._render_latex_text(obj)
        if isinstance(obj, Group):
            return Group(
                *(self._render_latex_object(item) for item in getattr(obj, "_renderables", ())),
                fit=getattr(obj, "fit", True),
            )
        if isinstance(obj, Panel):
            return Panel(
                self._render_latex_object(obj.renderable),
                obj.box,
                title=self._render_latex_title(obj.title),
                title_align=obj.title_align,
                subtitle=self._render_latex_title(obj.subtitle),
                subtitle_align=obj.subtitle_align,
                safe_box=obj.safe_box,
                expand=obj.expand,
                style=obj.style,
                border_style=obj.border_style,
                width=obj.width,
                height=obj.height,
                padding=obj.padding,
                highlight=obj.highlight,
            )
        if isinstance(obj, Rule):
            return Rule(
                self._render_latex_title(obj.title),
                characters=obj.characters,
                style=obj.style,
                end=obj.end,
                align=obj.align,
            )
        if isinstance(obj, Table):
            return self._render_latex_table(obj)
        return obj

    def _render_latex_title(self, title: str | Text | None) -> str | Text | None:
        if isinstance(title, str):
            return renderable_cli_text(title)
        if isinstance(title, Text):
            return self._render_latex_text(title)
        return title

    def _render_latex_text(self, text: Text) -> Text:
        if _text_is_latex_rendered(text):
            return text
        plain = text.plain
        rendered_plain = renderable_cli_text(plain)
        if rendered_plain == plain:
            return text

        if not text.spans:
            return Text(
                rendered_plain,
                style=text.style,
                justify=text.justify,
                overflow=text.overflow,
                no_wrap=text.no_wrap,
                end=text.end,
                tab_size=text.tab_size,
            )

        rendered = self._render_latex_text_by_style(text)
        if rendered.plain == rendered_plain:
            return rendered
        return Text(
            rendered_plain,
            style=text.style,
            justify=text.justify,
            overflow=text.overflow,
            no_wrap=text.no_wrap,
            end=text.end,
            tab_size=text.tab_size,
        )

    def _render_latex_text_by_style(self, text: Text) -> Text:
        plain = text.plain
        rendered = Text(
            style=text.style,
            justify=text.justify,
            overflow=text.overflow,
            no_wrap=text.no_wrap,
            end=text.end,
            tab_size=text.tab_size,
        )
        start = 0
        while start < len(plain):
            style = text.get_style_at_offset(self, start)
            end = start + 1
            while end < len(plain) and text.get_style_at_offset(self, end) == style:
                end += 1
            rendered.append(renderable_cli_text(plain[start:end]), style=style)
            start = end
        return rendered

    def _render_latex_table(self, table: Table) -> Table:
        try:
            rendered = copy.deepcopy(table)
        except Exception:
            rendered = table
        rendered.title = self._render_latex_title(rendered.title)
        rendered.caption = self._render_latex_title(rendered.caption)
        for column in rendered.columns:
            column.header = self._render_latex_object(column.header)
            column.footer = self._render_latex_object(column.footer)
            column._cells = [self._render_latex_object(cell) for cell in column._cells]
        return rendered


def _text_is_latex_rendered(text: Text) -> bool:
    plain_length = len(text.plain)
    if plain_length == 0:
        return False
    for span in text.spans:
        meta = getattr(span.style, "meta", None) or {}
        if meta.get(_LATEX_RENDERED_META) and span.start <= 0 and span.end >= plain_length:
            return True
    return False


def set_plain_mode(enabled: bool) -> None:
    """Called from main.py callback when --plain is set."""
    global _plain_mode
    _plain_mode = enabled


def install_prompt_abort() -> None:
    """Patch click's visible_prompt_func so q/quit/exit aborts any typer.prompt.

    Only affects click/typer prompts — the shell's own input() is untouched.
    Guarded to apply exactly once even when callback fires repeatedly in shell.
    """
    global _prompt_abort_installed
    if _prompt_abort_installed:
        return
    _prompt_abort_installed = True

    import click
    import click.termui

    _original = click.termui.visible_prompt_func

    def _prompt_with_abort(prompt: str) -> str:
        result = _original(prompt)
        if result.strip().lower() in _ABORT_TOKENS:
            raise click.Abort()
        if result.strip() == "\x1b":
            raise click.Abort()
        return result

    click.termui.visible_prompt_func = _prompt_with_abort


def _configured_max_content_width() -> int:
    try:
        from pb.storage.config import get_config

        ui = getattr(get_config(), "ui", None)
        value = int(getattr(ui, "max_content_width", 0) or 0)
    except Exception:
        value = 0
    return max(0, value)


def _configured_content_width_ratio() -> float:
    try:
        from pb.storage.config import get_config

        ui = getattr(get_config(), "ui", None)
        value = float(getattr(ui, "content_width_ratio", _DEFAULT_CONTENT_WIDTH_RATIO))
    except Exception:
        value = _DEFAULT_CONTENT_WIDTH_RATIO
    return min(1.0, max(0.40, value))


def resolve_render_width() -> int:
    """Return the wrapped render width capped by UI configuration."""
    terminal_width = shutil.get_terminal_size((_DEFAULT_TERMINAL_WIDTH, 24)).columns
    ratio_target = max(40, int(terminal_width * _configured_content_width_ratio()))
    max_width = _configured_max_content_width()
    if max_width > 0:
        ratio_target = min(ratio_target, max(40, max_width))
    return max(40, min(terminal_width, ratio_target))


def get_console() -> Console:
    """Return a themed Console, or plain Console when --plain is active.

    Per D-07: --plain returns Console(no_color=True, highlight=False).
    Per D-08: Rich auto-detects non-TTY and strips color independently.
    """
    width = resolve_render_width()
    if _plain_mode:
        return LatexAwareConsole(no_color=True, highlight=False, width=width)
    from pb.cli.themes import load_active_theme

    return LatexAwareConsole(theme=Theme(load_active_theme()), width=width)


def get_err_console() -> Console:
    """Error console -- always styled, always stderr. Per D-06, D-19."""
    from pb.cli.themes import load_active_theme

    return LatexAwareConsole(stderr=True, theme=Theme(load_active_theme()), width=resolve_render_width())
