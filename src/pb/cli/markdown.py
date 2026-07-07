# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Markdown rendering helpers for CLI output."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.ansi import AnsiDecoder
from rich.console import Group
from rich.markdown import Markdown
from rich.text import Text

from pb.cli.console import get_console, mark_latex_rendered_text, resolve_render_width
from pb.core.renderables import renderable_cli_text
from pb.core.resources import resource_path


def _iter_glow_candidates():
    """Yield likely glow binary locations in descending preference order."""
    env_path = os.environ.get("PB_GLOW_PATH")
    if env_path:
        yield Path(env_path).expanduser()

    on_path = shutil.which("glow")
    if on_path:
        yield Path(on_path)

    yield Path("/opt/homebrew/bin/glow")
    yield Path("/usr/local/bin/glow")
    yield Path.home() / ".local" / "bin" / "glow"

    for cellar in (Path("/opt/homebrew/Cellar/glow"), Path("/usr/local/Cellar/glow")):
        if not cellar.is_dir():
            continue
        for version_dir in sorted(cellar.iterdir(), reverse=True):
            yield version_dir / "bin" / "glow"


def resolve_glow_binary() -> str | None:
    """Return the first usable glow binary path, or None."""
    seen: set[str] = set()
    for candidate in _iter_glow_candidates():
        candidate_str = str(candidate)
        if candidate_str in seen:
            continue
        seen.add(candidate_str)
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate_str
    return None


def resolve_glow_style_path() -> str | None:
    """Return the bundled Glow/Glamour stylesheet, or a caller override."""
    env_path = os.environ.get("PB_GLOW_STYLE")
    if env_path:
        candidate = Path(env_path).expanduser()
        if candidate.is_file():
            return str(candidate)

    try:
        with resource_path("cli", "glow_style.json") as bundled:
            if bundled.is_file():
                return str(bundled)
    except (FileNotFoundError, ModuleNotFoundError):
        pass
    return None


def render_markdown(text: str) -> bool:
    """Render Markdown with glow when available, else use Rich markdown."""
    rendered_text = render_markdown_for_terminal(text)
    glow = resolve_glow_binary()
    width = resolve_render_width()
    if glow:
        try:
            args = [glow]
            style_path = resolve_glow_style_path()
            if style_path:
                args.extend(["-s", style_path])
            args.extend(["-w", str(width)])
            args.append("-")
            env = dict(os.environ)
            config_home = Path(
                env.get("PB_GLOW_CONFIG_HOME")
                or env.get("GLOW_CONFIG_HOME")
                or (Path(tempfile.gettempdir()) / "pb-glow")
            )
            config_home.mkdir(parents=True, exist_ok=True)
            env["GLOW_CONFIG_HOME"] = str(config_home)
            subprocess.run(args, input=rendered_text, text=True, check=False, env=env)
            return True
        except OSError:
            pass

    try:
        get_console().print(Markdown(rendered_text), soft_wrap=True)
        return True
    except Exception:
        print(rendered_text)
        return False


def render_markdown_to_rich(text: str, *, width: int | None = None) -> object:
    """Return a Rich renderable for Markdown, using Glow as the primary formatter."""
    return Group(*render_markdown_to_rich_blocks(text, width=width))


def render_markdown_to_rich_blocks(text: str, *, width: int | None = None) -> list[object]:
    """Render Markdown to Rich blocks, preferring Glow and falling back to Rich."""
    rendered_text = render_markdown_for_terminal(text)
    glow_output = _capture_glow(rendered_text, width=width)
    if glow_output is not None:
        blocks = [
            mark_latex_rendered_text(line)
            for line in AnsiDecoder().decode(glow_output.rstrip("\n"))
        ]
        while blocks and not blocks[-1].plain.strip():
            blocks.pop()
        while blocks and not blocks[0].plain.strip():
            blocks.pop(0)
        return blocks or [Text()]
    return [Markdown(rendered_text)]


def render_markdown_for_terminal(text: str) -> str:
    """Render LaTeX in Markdown text without touching fenced code."""
    rendered: list[str] = []
    pending: list[str] = []
    in_fence = False
    fence_marker = ""

    def flush_pending() -> None:
        if pending:
            rendered.append(renderable_cli_text("".join(pending)))
            pending.clear()

    for line in (text or "").splitlines(keepends=True):
        stripped = line.lstrip()
        if stripped.startswith(("```", "~~~")):
            marker = stripped[:3]
            if not in_fence:
                flush_pending()
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ""
            rendered.append(line)
            continue
        if in_fence:
            rendered.append(line)
            continue
        pending.append(line)
    flush_pending()
    return "".join(rendered)


def _capture_glow(text: str, *, width: int | None = None) -> str | None:
    glow = resolve_glow_binary()
    if not glow:
        return None
    args = [glow]
    style_path = resolve_glow_style_path()
    if style_path:
        args.extend(["-s", style_path])
    args.extend(["-w", str(width or resolve_render_width())])
    args.append("-")
    env = _glow_env()
    try:
        completed = subprocess.run(
            args,
            input=text,
            text=True,
            check=False,
            capture_output=True,
            env=env,
        )
    except OSError:
        return None
    if completed.returncode != 0 and not completed.stdout:
        return None
    return completed.stdout


def _glow_env() -> dict[str, str]:
    env = dict(os.environ)
    config_home = Path(
        env.get("PB_GLOW_CONFIG_HOME")
        or env.get("GLOW_CONFIG_HOME")
        or (Path(tempfile.gettempdir()) / "pb-glow")
    )
    config_home.mkdir(parents=True, exist_ok=True)
    env["GLOW_CONFIG_HOME"] = str(config_home)
    return env
