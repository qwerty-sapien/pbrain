# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Helpers for enforcing the LLM-required product contract."""

from __future__ import annotations

import typer

from pb.cli.console import get_console, get_err_console
from pb.core.exceptions import ExitCode
from pb.llm.runtime import DraftGenerationError, LLMRuntime


def print_llm_error(exc: DraftGenerationError) -> None:
    """Print a full LLM failure report with per-attempt details."""
    console = get_console()
    console.print()
    console.print("[red bold]LLM returned no usable draft[/]")
    console.print(f"  {exc.to_user_message()}")
    if exc.attempts:
        console.print()
        console.print("  [dim]Attempt log:[/]")
        for attempt in exc.attempts:
            status_tag = f"HTTP {attempt.http_status}" if attempt.http_status is not None else attempt.status
            console.print(f"    {attempt.provider}:{attempt.model} [{attempt.prompt_kind}] → {status_tag}")
            if attempt.raw_message:
                console.print(f"      [dim]{attempt.raw_message}[/]")
    else:
        details = exc.debug_details()
        if details:
            console.print(f"  [dim]{details}[/]")
    console.print()


def llm_requirement_message(
    workflow: str,
    *,
    detail: str | None = None,
    fallback_available: bool,
) -> str:
    lines = [
        f"`{workflow}` needs live LLM output.",
        "pb could not complete that step with the current LLM setup.",
    ]
    if detail:
        lines.append(detail)
    lines.append("Run `pb doctor --llm` to verify credentials, model access, quota, and live request health.")
    if fallback_available:
        lines.append("pb can keep you moving with a local fallback plan for this step.")
    else:
        lines.append("No safe local fallback is available for this step.")
    return "\n".join(lines)


def runtime_for_ctx(ctx: typer.Context) -> LLMRuntime:
    config = ctx.obj.get("config") if ctx.obj else None
    return LLMRuntime(config)


def require_llm(ctx: typer.Context, purpose: str | None = None, *, workflow: str | None = None) -> LLMRuntime:
    runtime = runtime_for_ctx(ctx)
    label = workflow or purpose or "llm workflow"
    try:
        runtime.require(label)
        return runtime
    except RuntimeError as exc:
        get_err_console().print(
            "[error]"
            + llm_requirement_message(label, detail=str(exc), fallback_available=False)
            + "[/]"
        )
        raise typer.Exit(code=ExitCode.CONFIG_ERROR)
