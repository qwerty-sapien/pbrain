# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Preflight handling for conflicting active learning sessions."""

from __future__ import annotations

import typer

from pb.cli.console import get_err_console
from pb.core.naming import stored_display_title
from pb.domain.exceptions import ExitCode


def resolve_active_session_preflight(
    ctx: typer.Context,
    *,
    new_intent: str,
    new_branch: str = "",
) -> bool:
    """Resolve active-session conflicts before starting a new learning flow."""
    del new_intent, new_branch
    repo = ctx.obj["repo"]
    active_session = repo.get_active_session()
    if active_session is None:
        return True

    active_task = repo.get_task(active_session.task_id)
    active_title = stored_display_title(active_task) or getattr(active_session, "subject_scope", "") or "Current session"
    message = (
        f"Session active: {active_title}. "
        "Finish it with `pb finish --skip`, pause it with `pb pause`, "
        "or inspect recent sessions with `pb session list` before starting another block."
    )
    get_err_console().print(f"[error]{message}[/]")
    raise typer.Exit(code=ExitCode.CONFLICT)
