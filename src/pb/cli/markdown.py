# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Markdown rendering helpers for CLI output."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

from rich.markdown import Markdown

from pb.cli.console import get_console, resolve_render_width
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
            subprocess.run(args, input=text, text=True, check=False, env=env)
            return True
        except OSError:
            pass

    try:
        get_console().print(Markdown(text), soft_wrap=True)
        return True
    except Exception:
        print(text)
        return False
