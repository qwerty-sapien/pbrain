# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Preflight handling for conflicting active learning sessions."""

from __future__ import annotations

import os
import sys
from datetime import datetime

import click
import typer
from rich.markup import escape

from pb.cli.console import get_err_console
from pb.cli.helpers import confirm_choice
from pb.core.naming import stored_display_title
from pb.domain.exceptions import ExitCode


_AUTO_YES_VALUES = {"1", "true", "yes", "on"}


def _auto_yes_enabled(ctx: typer.Context) -> bool:
    obj = ctx.obj or {}
    if obj.get("yes"):
        return True
    if os.environ.get("PRODUCTIVEBRAIN_AUTO_YES", "").strip().lower() in _AUTO_YES_VALUES:
        return True
    try:
        from pb.runtime import get_session_auto_yes

        runtime = obj.get("runtime")
        config = getattr(runtime, "config", None) or obj.get("config")
        return bool(get_session_auto_yes(config))
    except Exception:
        return False


def _pause_active_session(ctx: typer.Context, *, outcome: str) -> object | None:
    obj = ctx.obj or {}
    factory = obj.get("factory") or {}
    service_factory = factory.get("session_service") if isinstance(factory, dict) else None
    if service_factory is not None:
        service = service_factory()
        pause_session = getattr(service, "pause_session", None)
        if callable(pause_session):
            return pause_session(outcome=outcome)

    repo = obj["repo"]
    active_session = repo.get_active_session()
    if active_session is None:
        return None
    active_session.end_at = datetime.utcnow()
    active_session.actual_outcome = outcome
    return repo.update_session(active_session)


def resolve_active_session_preflight(
    ctx: typer.Context,
    *,
    new_intent: str,
    new_branch: str = "",
) -> bool:
    """Resolve active-session conflicts before starting a new learning flow."""
    repo = ctx.obj["repo"]
    active_session = repo.get_active_session()
    if active_session is None:
        return True

    active_task = repo.get_task(active_session.task_id)
    active_title = (
        stored_display_title(active_task)
        or getattr(active_session, "subject_scope", "")
        or "Current session"
    )
    branch_label = (
        f"{new_branch.strip().lower()} session"
        if new_branch.strip()
        else "new session"
    )
    target = new_intent.strip() or "the requested block"
    prompt = f"Session active: {active_title}. Pause it and start {branch_label}: {target}?"

    if not _auto_yes_enabled(ctx):
        if os.environ.get("PRODUCTIVEBRAIN_SHELL_TEST_MODE", "").strip().lower() in _AUTO_YES_VALUES:
            get_err_console().print(
                "[error]"
                f"Session active: {escape(active_title)}. "
                "Re-run with `--yes` to pause it and start the new session. "
                "You can also use `pb finish --skip` or `pb pause`."
                "[/]"
            )
            return False
        try:
            accepted = confirm_choice(prompt, default=False, err=True)
        except (click.exceptions.Abort, EOFError, OSError):
            if not sys.stdin.isatty():
                get_err_console().print(
                    "[error]"
                    f"Session active: {escape(active_title)}. "
                    "Re-run with `--yes` to pause it and start the new session. "
                    "You can also use `pb finish --skip` or `pb pause`."
                    "[/]"
                )
                raise typer.Exit(code=ExitCode.CONFLICT)
            raise
        if not accepted:
            get_err_console().print("[dim]No changes. Current session is still active.[/]")
            return False

    outcome = f"Paused to start {branch_label}: {target}"
    paused = _pause_active_session(ctx, outcome=outcome)
    if paused is not None:
        get_err_console().print(f"[dim]Paused: {escape(active_title)}[/]")
    return True
