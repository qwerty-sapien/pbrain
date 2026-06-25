# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Proactive prompts command -- relationship + cross-domain (D-15)."""
from __future__ import annotations

import typer


PROMPT_ICONS = {
    "overdue_commitment": "!",
    "birthday": "*",
    "gift_reminder": "~",
    "decay_warning": "?",
    "goal_deadline": "#",
    "event_prep": "@",
    "skill_gap": "^",
    "stale_inbox": ">",
}

PROMPT_HEADERS = {
    "overdue_commitment": "Overdue Commitments",
    "birthday": "Upcoming Birthdays",
    "gift_reminder": "Gift Reminders",
    "decay_warning": "Relationship Decay Warnings",
    "goal_deadline": "Goal Deadlines Approaching",
    "event_prep": "Events Coming Up",
    "skill_gap": "Skill Gap Nudges",
    "stale_inbox": "Stale Inbox Items",
}


def prompts_command():
    """Show proactive prompts across all domains."""
    from pb.core.prompts import ProactivePromptsEngine

    engine = ProactivePromptsEngine()
    prompts = engine.get_prompts()
    if not prompts:
        typer.echo("No proactive prompts right now.")
        return
    # Group by type, display each group
    grouped: dict[str, list] = {}
    for p in prompts:
        grouped.setdefault(p.prompt_type, []).append(p)
    for ptype in [
        "overdue_commitment", "birthday", "gift_reminder", "decay_warning",
        "goal_deadline", "event_prep", "skill_gap", "stale_inbox",
    ]:
        items = grouped.get(ptype, [])
        if not items:
            continue
        header = PROMPT_HEADERS[ptype]
        icon = PROMPT_ICONS[ptype]
        typer.echo(f"\n[{icon}] {header}:")
        for p in items:
            typer.echo(f"  {p.person_name} — {p.message}")
