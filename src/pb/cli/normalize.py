# SPDX-License-Identifier: AGPL-3.0-or-later
"""Input normalization at the CLI boundary.

All raw argv → clean domain value conversions live here.
After normalization, internal functions receive typed values.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer


def join_words(words: Optional[list[str]]) -> str:
    """Join a variadic word list into a single trimmed string."""
    return " ".join(words or []).strip()


def join_words_safe(words: Optional[list[str]]) -> str:
    """Join topic words, dropping any --flag tokens that slipped through argument parsing.

    Only strips tokens that start with '--' (two hyphens).
    Hyphenated topics like 'step-by-step' and single-hyphen tokens are preserved.
    """
    return " ".join(w for w in (words or []) if not w.startswith("--")).strip()


def is_interactive(ctx: typer.Context | None = None) -> bool:
    """True when the session is interactive (TTY and --yes not set)."""
    if not sys.stdin.isatty():
        return False
    if ctx is not None:
        obj = ctx.obj or {}
        if obj.get("yes", False):
            return False
    return True


def require_topic(
    words: Optional[list[str]],
    *,
    prompt_text: str = "Topic",
    ctx: typer.Context | None = None,
) -> str:
    """Normalize word list into a topic string, prompting if interactive and empty."""
    topic = join_words(words)
    if topic:
        return topic
    if is_interactive(ctx):
        topic = typer.prompt(prompt_text, default="", show_default=False).strip()
    if not topic:
        raise typer.BadParameter(f"A topic is required. Example: pb learn jazz harmony")
    return topic
