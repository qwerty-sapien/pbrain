# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""CLI flow for deferred learning transitions."""

from __future__ import annotations

import sys

import typer

from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console
from pb.cli.preview import preview_decision
from pb.core.learning_transitions import LearningTransitionService, LearningTransitionUnavailable


def process_pending_learning_transitions(ctx: typer.Context, *, interactive: bool = True) -> bool:
    """Process the oldest pending learning transition when the LLM is available."""
    if not interactive or not sys.stdin.isatty():
        return False
    repo = (ctx.obj or {}).get("repo")
    runtime_ctx = (ctx.obj or {}).get("runtime")
    if repo is None or runtime_ctx is None:
        return False
    runtime = (ctx.obj or {}).get("_llm_runtime")
    if runtime is None:
        try:
            from pb.cli.llm_guard import runtime_for_ctx

            runtime = runtime_for_ctx(ctx)
        except Exception:
            runtime = None
    service = LearningTransitionService(repo, runtime, runtime_ctx)
    if not service.llm_available():
        return False
    pending = repo.list_pending_learning_transitions(limit=1)
    if not pending:
        return False
    transition = pending[0]
    route_kind = "next_session" if transition.route_kind in {"next_session", "continue"} else transition.route_kind
    console = get_console()
    try:
        draft = service.generate(route_kind=route_kind, transition=transition)
        if draft.next_session is None:
            raise LearningTransitionUnavailable("The model did not return a next-session plan.")
        service.render_next_session_preview(draft)
        decision = preview_decision(yes=False, action_label="Start queued next session", allow_refinement=False)
        if decision.kind != "accept":
            repo.mark_learning_transition_completed(transition.id)
            console.print("[dim]Queued learning transition dismissed.[/]")
            return True
        command = service.command_for_block(draft.next_session)
        repo.mark_learning_transition_completed(transition.id)
        run_internal_command(ctx, command)
        active = repo.get_active_session()
        if active is not None:
            task = repo.get_task(active.task_id)
            service.maybe_offer_agent_spawn(session=active, task=task, draft=draft)
        return True
    except Exception as exc:
        repo.mark_learning_transition_failed(transition.id, str(exc))
        console.print("[warn]Queued learning transition still pending; LLM route failed this time.[/]")
        return False
