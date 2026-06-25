# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Natural-language command router for the canonical CLI surface."""

from __future__ import annotations

import asyncio
import os

import structlog
import typer

from pb.cli.command_runner import run_internal_command
from pb.cli.console import get_console
from pb.cli.normalize import is_interactive, join_words
from pb.cli.pickers import pick_single_choice
from pb.core.action_routing import suggest_commands_for_intent
from pb.core.dispatch_models import InteractionEnvelope
from pb.core.dispatcher import dispatch
from pb.core.goal_roadmaps import ensure_goal_seed_tasks

app = typer.Typer(no_args_is_help=False)

_logger = structlog.get_logger()


def _prefer_plain_suggestions() -> bool:
    return os.environ.get("PRODUCTIVEBRAIN_SHELL_TEST_MODE", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }



def _render_envelope(ctx: typer.Context, envelope: InteractionEnvelope, repo, *, first_turn: bool = True) -> None:
    """Render an InteractionEnvelope to the console. Handles all status variants."""
    console = get_console()

    # D-01: print agent label once on first turn (dim, does not dominate output)
    # Suppress label for transparent agents (capture is a one-shot wrapper, not a real mode)
    _TRANSPARENT_AGENTS = {"capture"}
    if first_turn and envelope.session_id:
        agent_label = envelope.fields.get("agent_id", "")
        if not agent_label:
            try:
                from pb.mcp.protocol import get_session
                sess = get_session(envelope.session_id)
                if sess:
                    raw = sess.agent_id
                    agent_label = raw.replace("domain_", "").replace("_agent", "").replace("_", "-")
            except Exception:
                pass
        if agent_label and agent_label not in _TRANSPARENT_AGENTS:
            from pb.agents.colors import color_for_agent
            label_color = color_for_agent(agent_label)
            console.print(f"[{label_color}]{agent_label}[/{label_color}]")

    if envelope.status == "complete" and not envelope.options:
        # One-shot action — display response and return
        if envelope.prompt:
            console.print(envelope.prompt)
        return

    if envelope.status == "complete" and envelope.options:
        # Complete with follow-up options — show them
        if envelope.prompt:
            console.print(envelope.prompt)
        if not is_interactive(ctx) or _prefer_plain_suggestions():
            console.print("[header]Do[/]")
            for i, opt in enumerate(envelope.options, 1):
                console.print(f"{i}. {opt}")
            return
        selected = pick_single_choice(
            [(opt, opt) for opt in envelope.options],
            title="Choose action",
            text=envelope.prompt or "Choose the next step.",
        )
        if selected:
            run_internal_command(ctx, selected)
        return

    if envelope.status == "active" and envelope.options:
        if not is_interactive(ctx) or _prefer_plain_suggestions():
            console.print("[header]Do[/]")
            if envelope.prompt:
                console.print(envelope.prompt)
            for i, opt in enumerate(envelope.options, 1):
                console.print(f"{i}. {opt}")
            return
        # Interactive: let user pick an option then advance the session
        selected = pick_single_choice(
            [(opt, opt) for opt in envelope.options],
            title="Choose action",
            text=envelope.prompt or "Choose the next step.",
        )
        if selected:
            try:
                next_envelope = asyncio.run(
                    dispatch(repo, selected, session_id=envelope.session_id)
                )
                _render_envelope(ctx, next_envelope, repo, first_turn=False)
            except Exception as exc:
                _logger.warning("do.envelope_continuation_error", error=str(exc))
                console.print("Something went wrong. Please try again.")
        return

    if envelope.status == "active" and not envelope.options:
        # Active with no options — show prompt
        if envelope.prompt:
            console.print(envelope.prompt)
        return

    if envelope.status == "blocked":
        if envelope.prompt:
            console.print(envelope.prompt)
        return

    if envelope.status == "error":
        # T-10-15: log internally; show clean message (REL-01, T-10-16)
        _logger.warning("do.envelope_error", prompt=envelope.prompt)
        console.print(envelope.prompt or "Something went wrong. Please try again.")
        return

    # Fallback: just print whatever is in the envelope
    if envelope.prompt:
        console.print(envelope.prompt)


def _command_router_candidates(repo, intent: str):
    """Return non-capture command suggestions when the direct router has a clear plan."""
    candidates = suggest_commands_for_intent(repo, intent, limit=5)
    routed = [candidate for candidate in candidates if candidate.kind != "capture"]
    return routed or []


@app.callback(invoke_without_command=True)
def do_command(
    ctx: typer.Context,
    intent_words: list[str] = typer.Argument(None, help="Natural-language request"),
):
    """Route a natural-language request to one or more likely pb commands."""
    try:
        repo = ctx.obj["repo"]
        runtime = (ctx.obj or {}).get("runtime")
        ensure_goal_seed_tasks(repo, repo.list_goal_arcs(status=None), vault_path=getattr(runtime, "vault_path", None))
        console = get_console()
        intent = join_words(intent_words)

        if not intent:
            from pb.core.action_routing import build_next_candidates
            candidates = build_next_candidates(repo, limit=5)
            if candidates:
                if not is_interactive(ctx) or _prefer_plain_suggestions():
                    for i, c in enumerate(candidates, 1):
                        console.print(f"{i}. {c.human_label}")
                else:
                    selected = pick_single_choice(
                        [(c.backing_command, c.human_label) for c in candidates],
                        title="Choose action",
                        text="What would you like to do?",
                        details=[c.short_reason for c in candidates],
                    )
                    if selected:
                        run_internal_command(ctx, selected)
            else:
                console.print("Nothing to do right now.")
            return

        routed_candidates = _command_router_candidates(repo, intent)
        if routed_candidates:
            if not is_interactive(ctx) or _prefer_plain_suggestions():
                console.print("[header]Do[/]")
                for i, candidate in enumerate(routed_candidates, 1):
                    console.print(f"{i}. {candidate.human_label}")
                    console.print(f"   [dim]Run:[/] pb {candidate.backing_command}")
                    console.print(f"   [dim]Why:[/] {candidate.short_reason}")
                return

            selected = pick_single_choice(
                [(c.backing_command, c.human_label) for c in routed_candidates],
                title="Choose action",
                text=intent,
                details=[c.short_reason for c in routed_candidates],
            )
            if selected:
                run_internal_command(ctx, selected)
            return

        # Machine path: produce clean structured output (PRODUCTIVEBRAIN_SHELL_TEST_MODE)
        if not is_interactive(ctx) or _prefer_plain_suggestions():
            envelope = asyncio.run(dispatch(repo, intent))
            if envelope.prompt:
                console.print(envelope.prompt)
            if envelope.options:
                for i, opt in enumerate(envelope.options, 1):
                    console.print(f"{i}. {opt}")
            return

        # Interactive path: full envelope rendering
        envelope = asyncio.run(dispatch(repo, intent))
        _render_envelope(ctx, envelope, repo, first_turn=True)

    except Exception as exc:
        # REL-01 / T-10-15: catch all dispatch errors, log internally, show clean message
        _logger.warning("do.dispatch_error", error=str(exc))
        get_console().print("Something went wrong. Please try again.")
        return
