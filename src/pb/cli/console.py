# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared console factory for Rich CLI rendering.

Provides get_console() and get_err_console() with theme injection.
_plain_mode is set by main.py callback when --plain is passed.

Per D-05, D-06, D-07, D-08.
"""

import shutil

from rich.console import Console
from rich.theme import Theme

_plain_mode: bool = False
_prompt_abort_installed: bool = False
_ABORT_TOKENS = {"q", "quit", "exit"}
_DEFAULT_MAX_CONTENT_WIDTH = 80
_DEFAULT_CONTENT_WIDTH_RATIO = 0.70
_DEFAULT_TERMINAL_WIDTH = 120


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
        return Console(no_color=True, highlight=False, width=width)
    from pb.cli.themes import load_active_theme

    return Console(theme=Theme(load_active_theme()), width=width)


def get_err_console() -> Console:
    """Error console -- always styled, always stderr. Per D-06, D-19."""
    from pb.cli.themes import load_active_theme

    return Console(stderr=True, theme=Theme(load_active_theme()), width=resolve_render_width())
