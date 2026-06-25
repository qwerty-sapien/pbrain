# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""User-facing preferences: pb set model|language."""

from __future__ import annotations

import typer

from pb.cli.console import get_console
from pb.storage.config import get_config, set_model_role, set_ui_language

app = typer.Typer(no_args_is_help=True, help="Adjust model tiers, language, and other preferences.")

_TIER_ROLES: dict[str, tuple[str, ...]] = {
    "fast": ("fast", "fast_inference", "namer"),
    "balanced": ("default",),
    "pro": ("planner", "reviewer", "recall"),
}

_TIER_DESCRIPTIONS = {
    "fast": "Quick responses, low latency. Used for routing, suggestions, and inline checks.",
    "balanced": "Default generation model. Used for most drafting tasks.",
    "pro": "Deep reasoning. Used for lesson planning, review, and recall generation.",
}


@app.command("model")
def set_model(
    tier: str = typer.Argument(..., help="Tier to configure: fast, balanced, or pro"),
    model_id: str = typer.Argument(..., help="Full model ID, e.g. gemini-3-flash-preview"),
) -> None:
    """Assign a model ID to a tier.

    Examples:
      pb set model fast gemini-3.1-flash-lite-preview
      pb set model balanced gemini-3-flash-preview
      pb set model pro gemini-3.1-pro-preview
    """
    console = get_console()
    tier = tier.strip().lower()
    if tier not in _TIER_ROLES:
        console.print(f"[error]Unknown tier '{tier}'.[/] Valid tiers: fast, balanced, pro")
        raise typer.Exit(1)

    roles = _TIER_ROLES[tier]
    for role in roles:
        set_model_role(role, model_id)

    console.print(f"[success]{tier}[/] → [code]{model_id}[/]")
    console.print(f"[dim]Updated roles: {', '.join(roles)}[/]")
    console.print(f"[dim]{_TIER_DESCRIPTIONS[tier]}[/]")


@app.command("language")
def set_language(
    lang: str = typer.Argument(
        ...,
        help="Language code or 'auto'. Examples: auto, en, zh, es, fr, ja, de",
    ),
) -> None:
    """Set the response language for LLM-based commands.

    'auto' (default) matches the language of your input.
    Any other value forces that language for all responses.

    Examples:
      pb set language auto
      pb set language zh
      pb set language en
    """
    console = get_console()
    lang = lang.strip().lower()
    set_ui_language(lang)
    if lang == "auto":
        console.print("[success]Language[/] → [code]auto[/] (responses will match your input language)")
    else:
        console.print(f"[success]Language[/] → [code]{lang}[/] (all LLM responses will use this language)")


@app.command("status")
def set_status() -> None:
    """Show current tier assignments and language preference."""
    console = get_console()
    cfg = get_config()
    roles = cfg.model_roles

    console.print("[bold]Model tiers[/]")
    fast_model = roles.fast or roles.default or "[dim]not set[/]"
    balanced_model = roles.default or "[dim]not set[/]"
    pro_model = roles.planner or roles.default or "[dim]not set[/]"
    console.print(f"  fast     [code]{fast_model}[/]")
    console.print(f"  balanced [code]{balanced_model}[/]")
    console.print(f"  pro      [code]{pro_model}[/]")

    lang = cfg.ui.language or "auto"
    console.print(f"\n[bold]Language[/]  [code]{lang}[/]")
