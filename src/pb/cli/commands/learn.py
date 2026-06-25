# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Umbrella learning router for choosing study vs practise."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from pb.cli.active_session import resolve_active_session_preflight
from pb.cli.context_args import parse_context_argv
from pb.cli.context_runtime import prepare_context_scope, raise_for_blocking_context
from pb.cli.commands.clarify import maybe_start_clarification_plan
from pb.cli.console import get_console
from pb.cli.llm_guard import runtime_for_ctx
from pb.cli.normalize import is_interactive, join_words, join_words_safe
from pb.core.action_routing import route_learning_intent
from pb.core.staging import build_assumptions, build_learning_context, build_reflection

app = typer.Typer(
    no_args_is_help=False,
    invoke_without_command=True,
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)


def _pick_domain(knowledge_dir: Path, console) -> Optional[str]:
    """Choose a knowledge domain from the vault, with a manual fallback."""
    try:
        domains = sorted(
            domain_dir.name
            for domain_dir in knowledge_dir.iterdir()
            if domain_dir.is_dir() and not domain_dir.name.startswith(".") and (domain_dir / "_state.md").exists()
        )
    except Exception:
        domains = []

    if domains:
        from pb.cli.pickers import pick_single_choice

        choice = pick_single_choice(
            [(domain, domain) for domain in domains],
            title="Select knowledge domain",
        )
        if choice:
            return choice

    if not is_interactive():
        return None
    typed = typer.prompt("Domain", default="", show_default=False).strip()
    return typed or None


def _dispatch_to_branch(
    ctx: typer.Context,
    branch: str,
    topic: str,
    *,
    yes: bool = False,
    steps: bool = False,
) -> None:
    if branch == "practise":
        from pb.cli.commands.practise import launch_practise_session

        launch_practise_session(ctx, skill=topic, yes=yes, steps=steps)
        return

    from pb.cli.commands.study import launch_study_session

    launch_study_session(ctx, topic=topic, yes=yes, steps=steps)


@app.callback(invoke_without_command=True)
def learn_command(
    ctx: typer.Context,
    topic_words: Optional[list[str]] = typer.Argument(None, help="Learning target. Multi-word OK."),
    force_study: bool = typer.Option(False, "--study", "-s", help="Force the study branch"),
    force_practise: bool = typer.Option(False, "--practise", "--practice", "-p", help="Force the practise branch"),
    steps: bool = typer.Option(False, "--steps", help="Include a stepwise study or practice sequence"),
    yes: bool = typer.Option(False, "--yes", help="Accept the AI draft and start immediately"),
):
    """Auto-route a learning request to study or practise."""
    if ctx.invoked_subcommand is not None:
        return

    console = get_console()
    if force_study and force_practise:
        raise typer.BadParameter("Choose only one forced branch: --study or --practise.")

    raw_tokens = [*(topic_words or []), *ctx.args]
    parsed_args = parse_context_argv(raw_tokens)
    prepared_context = prepare_context_scope(
        ctx,
        [Path(token).expanduser() for token in parsed_args.context_tokens],
    )
    raise_for_blocking_context(prepared_context)
    topic = join_words_safe(parsed_args.topic_tokens)
    if not topic:
        if not is_interactive(ctx):
            raise typer.BadParameter("A learning target is required. Try `pb learn jazz harmony`.")
        topic = typer.prompt("What do you want to learn?", default="", show_default=False).strip()
    if not topic:
        raise typer.Exit(code=0)

    preferred_branch = "study" if force_study else "practise" if force_practise else "mixed"
    if not resolve_active_session_preflight(
        ctx,
        new_intent=topic,
        new_branch=preferred_branch,
    ):
        return
    if maybe_start_clarification_plan(
        ctx,
        topic=topic,
        preferred_branch=preferred_branch,
        yes=yes,
    ):
        return

    if force_study:
        console.print("[dim]Routing to study: explicit study flag.[/]")
        _dispatch_to_branch(ctx, "study", topic, yes=yes, steps=steps)
        return
    if force_practise:
        console.print("[dim]Routing to practise: explicit practise flag.[/]")
        _dispatch_to_branch(ctx, "practise", topic, yes=yes, steps=steps)
        return

    runtime = runtime_for_ctx(ctx)
    recorder = runtime.make_stage_recorder("learn", topic, route_hint="learn")
    runtime_ctx = ctx.obj.get("runtime", runtime)
    context = build_learning_context(ctx.obj["repo"], runtime_ctx)
    recorder.add("prepare", context)
    reflection = build_reflection("learn", topic, context)
    recorder.add("reflect", reflection)
    recorder.add("assume", build_assumptions("learn", topic, context))
    console.print(f"[dim]{reflection}[/]")

    decision = route_learning_intent(ctx.obj["repo"], topic)
    recorder.add("verify", {"branch": decision.branch, "reason": decision.reason})
    recorder.finalize("routed", branch=decision.branch, reason=decision.reason)
    console.print(f"[dim]Routing to {decision.branch}: {decision.reason}[/]")
    _dispatch_to_branch(ctx, decision.branch, topic, yes=yes, steps=steps)
