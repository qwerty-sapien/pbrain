# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AI-powered command suggestion -- pb suggest <intent> (D-03).

CLI entry point for ? <intent> outside the shell context.
When no vault_cwd is available, cwd context is omitted gracefully.
"""
from __future__ import annotations

import typer

from pb.core.suggestions import SuggestionEngine
from pb.domain.exceptions import ExitCode


def suggest_command(
    intent: list[str] = typer.Argument(..., help="Natural language description of what you want to do"),
):
    """Get an AI command suggestion for a given intent.

    Flash Lite analyzes your intent with context from your active task and
    recent commands, then suggests the best pb command to run.

    Examples:
      pb suggest start working on my project
      pb suggest show me what I did today
      pb suggest find notes about machine learning
    """
    from pb.llm.gemini import get_client

    client = get_client()
    if not client.is_available():
        typer.echo("  AI suggestions unavailable -- set GEMINI_API_KEY to enable.")
        raise typer.Exit(code=ExitCode.CONFIG_ERROR)

    intent_text = " ".join(intent)
    engine = SuggestionEngine()  # No vault_cwd in CLI context
    result = engine.suggest(intent_text)
    if not result:
        typer.echo("  No suggestion available. Try rephrasing.")
        raise typer.Exit(code=1)

    command, explanation = result
    # Strip a legacy "pb " prefix if present — suggestion engine now omits it,
    # but guard against older prompts or model drift
    if command.lower().startswith("pb "):
        command = command[3:]
    typer.echo(f"\n  Suggestion: pb {command}")
    if explanation:
        typer.echo(f"  {explanation}")
    typer.echo(f"\n  Run it: pb {command}")
