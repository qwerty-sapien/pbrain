# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Scoped feedback capture for learner-facing command surfaces."""

from __future__ import annotations

import asyncio
import sys
from typing import Optional

import typer

from pb.cli.console import get_console, get_err_console
from pb.cli.helpers import confirm_choice, prompt_text
from pb.core.agent_weights import record_agent_weight_event, sort_sessions_by_weight
from pb.core.agent_instruction_judge import (
    format_patch_announcement,
    judge_agent_instruction_fit,
)
from pb.core.feedback_proposals import FeedbackProposalService
from pb.core.feedback_profile import (
    SUPPORTED_FEEDBACK_SCOPES,
    append_learner_level_assertion,
    normalize_feedback_scope,
    save_feedback_profile,
)
from pb.mcp.protocol import list_active_sessions


app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


def _render_proposal(console, proposal) -> None:
    console.print("[subheader]Proposed patches[/]")
    console.print(proposal.summary or "No summary.")
    if proposal.preference_patches:
        console.print("Preference patches:")
        for key, value in proposal.preference_patches.items():
            console.print(f"  - {key} = {value}")
    if proposal.workflow_patches:
        console.print("Workflow patches:")
        for item in proposal.workflow_patches:
            console.print(f"  - {item}")


def _capture_feedback_proposal(ctx: typer.Context, text: str, *, scope: str = "general") -> None:
    runtime = ctx.obj["runtime"]
    console = get_console()
    service = FeedbackProposalService()
    proposal = service.generate_proposal(text, scope=scope)
    proposal_path = service.write_proposal(runtime.vault_path, text, proposal, scope=scope)
    console.print(f"[success]Saved feedback proposal:[/] {proposal_path.relative_to(runtime.vault_path)}")
    _render_proposal(console, proposal)
    if sys.stdin.isatty() and proposal.preference_patches:
        if confirm_choice("Apply these preference patches now?", default=False):
            service.apply_proposal(proposal, config_path=ctx.obj.get("config_path"))
            console.print("[success]Applied preference patches to config.[/]")


def _scope_specific_prompt(scope: str) -> str:
    prompts = {
        "anki": "What should Anki generation optimize for: card style, difficulty, phrasing, or deck structure?",
        "diagnostic": "What should diagnostic or Socratic probing optimize for: confidence-building, depth, gap-finding, pacing, or something else?",
        "goal": "What should goal-shaping preserve or avoid when it turns a rough ambition into a durable goal?",
        "learn": "When `pb learn` decides the route, what should it prioritize: speed, confidence-building, challenge, or something else?",
        "plan": "What should planning bias toward: shorter blocks, mixed blocks, tighter sequencing, more retrieval, or something else?",
        "practise": "What kind of drills, cues, or feedback loops help you most during practice?",
        "review": "What kind of reflection is useful, and what kind feels like overhead?",
        "study": "What teaching style helps most during study: direct explanation, Socratic prompts, examples, recall pressure, or something else?",
        "teach": "What should guided teaching sessions do more or less of?",
    }
    return prompts.get(scope, "What specific instruction should this workflow keep in mind?")


def _bounded_optional_score(value: int | None, *, label: str) -> int | None:
    if value is None:
        return None
    if not 1 <= value <= 5:
        get_err_console().print(f"[error]{label} must be between 1 and 5.[/]")
        raise typer.Exit(code=1)
    return value


def _next_flag_index(words: list[str], start: int) -> int:
    flags = {"--level", "-l", "--confidence", "-c", "--evidence", "-e", "--note", "-n"}
    for index in range(start, len(words)):
        if words[index] in flags:
            return index
    return len(words)


def _parse_level_words(words: list[str]) -> tuple[list[str], int | None, int | None, str, str]:
    topic_words: list[str] = []
    level: int | None = None
    confidence: int | None = None
    evidence = ""
    note = ""
    index = 0
    while index < len(words):
        token = words[index]
        if token in {"--level", "-l"}:
            index += 1
            if index >= len(words):
                get_err_console().print("[error]--level needs a value from 1 to 5.[/]")
                raise typer.Exit(code=1)
            try:
                level = int(words[index])
            except ValueError:
                get_err_console().print("[error]--level needs a value from 1 to 5.[/]")
                raise typer.Exit(code=1)
            index += 1
            continue
        if token in {"--confidence", "-c"}:
            index += 1
            if index >= len(words):
                get_err_console().print("[error]--confidence needs a value from 1 to 5.[/]")
                raise typer.Exit(code=1)
            try:
                confidence = int(words[index])
            except ValueError:
                get_err_console().print("[error]--confidence needs a value from 1 to 5.[/]")
                raise typer.Exit(code=1)
            index += 1
            continue
        if token in {"--evidence", "-e"}:
            end = _next_flag_index(words, index + 1)
            evidence = " ".join(words[index + 1 : end]).strip()
            index = end
            continue
        if token in {"--note", "-n"}:
            end = _next_flag_index(words, index + 1)
            note = " ".join(words[index + 1 : end]).strip()
            index = end
            continue
        topic_words.append(token)
        index += 1
    return topic_words, level, confidence, evidence, note


def _record_feedback_level(
    ctx: typer.Context,
    *,
    topic_words: Optional[list[str]],
    level: Optional[int],
    confidence: Optional[int],
    evidence: str,
    note: str,
) -> None:
    topic = " ".join(topic_words or []).strip()
    if not topic:
        get_err_console().print("[error]`pb feedback level` needs a topic or concept.[/]")
        raise typer.Exit(code=1)
    level = _bounded_optional_score(level, label="Level")
    confidence = _bounded_optional_score(confidence, label="Confidence")
    if level is None and confidence is None and not evidence.strip() and not note.strip():
        get_err_console().print(
            "[error]Add --level, --confidence, --evidence, or --note so the assertion has signal.[/]"
        )
        raise typer.Exit(code=1)

    runtime = ctx.obj["runtime"]
    path, record = append_learner_level_assertion(
        runtime.vault_path,
        topic=topic,
        level=level,
        confidence=confidence,
        evidence=evidence,
        note=note,
    )

    repo = ctx.obj.get("repo") if ctx.obj else None
    if repo is not None:
        try:
            session = repo.get_active_session()
            if session is not None:
                generated = dict(getattr(session, "generated_names", {}) or {})
                rows = list(generated.get("learner_self_reports") or [])
                rows.append(record)
                generated["learner_self_reports"] = rows
                session.generated_names = generated
                repo.update_session(session)
        except Exception:
            pass

    console = get_console()
    console.print(f"[success]Saved learner level assertion:[/] {path.relative_to(runtime.vault_path)}")


@app.callback(invoke_without_command=True)
def feedback_command(
    ctx: typer.Context,
    surface: str = typer.Argument(..., help="Primary command surface: learn, study, practise, teach, diagnostic, anki, goal, plan, review, or general."),
    note_words: Optional[list[str]] = typer.Argument(None, help="Optional scope note to seed the conversation."),
):
    """Capture durable workflow guidance for one primary learning surface."""
    if ctx.invoked_subcommand is not None:
        return

    if surface.strip().lower() == "wrong":
        feedback_wrong(ctx, intent_words=note_words)
        return

    if surface.strip().lower() == "level":
        topic_words, level, confidence, evidence, note = _parse_level_words(list(note_words or []))
        _record_feedback_level(
            ctx,
            topic_words=topic_words,
            level=level,
            confidence=confidence,
            evidence=evidence,
            note=note,
        )
        return

    normalized = normalize_feedback_scope(surface)
    if normalized not in SUPPORTED_FEEDBACK_SCOPES:
        proposal_text = " ".join([surface, *list(note_words or [])]).strip()
        if not proposal_text:
            get_err_console().print("[error]`pb feedback` needs feedback text.[/]")
            raise typer.Exit(code=1)
        _capture_feedback_proposal(ctx, proposal_text, scope="general")
        return

    focus_note = " ".join(note_words or []).strip()
    if not sys.stdin.isatty() and not focus_note:
        get_err_console().print("[error]`pb feedback` needs an interactive terminal or an inline scope note.[/]")
        raise typer.Exit(code=1)

    console = get_console()
    runtime = ctx.obj["runtime"]
    console.print(f"[dim]Capturing guidance for `pb {normalized}`.[/]")

    more_of = prompt_text(
        f"What should `pb {normalized}` do more of?",
        default="",
    ) if sys.stdin.isatty() else ""
    less_of = prompt_text(
        f"What should `pb {normalized}` do less of or avoid?",
        default="",
    ) if sys.stdin.isatty() else ""
    learner_context = prompt_text(
        "What should the system remember about your level, background, or constraints?",
        default="",
    ) if sys.stdin.isatty() else ""
    keep_in_mind = prompt_text(
        _scope_specific_prompt(normalized),
        default=focus_note,
    ) if sys.stdin.isatty() else focus_note

    note_path = save_feedback_profile(
        runtime.vault_path,
        normalized,
        more_of=more_of,
        less_of=less_of,
        learner_context=learner_context,
        keep_in_mind=keep_in_mind,
        focus_note=focus_note,
    )
    console.print(f"[success]Saved feedback:[/] {note_path.relative_to(runtime.vault_path)}")
    combined_feedback = " ".join(
        part for part in (focus_note, more_of, less_of, learner_context, keep_in_mind) if part.strip()
    ).strip()
    if combined_feedback:
        _capture_feedback_proposal(ctx, combined_feedback, scope=normalized)


@app.command("wrong")
def feedback_wrong(
    ctx: typer.Context,
    intent_words: Optional[list[str]] = typer.Argument(None, help="Optional replacement intent to redispatch"),
):
    """Mark the active dispatch agent as wrong and optionally redispatch."""
    console = get_console()
    err_console = get_err_console()
    active_sessions = sort_sessions_by_weight(list_active_sessions())
    if not active_sessions:
        err_console.print(
            "[error]No active dispatch session to mark as wrong.[/] "
            "Use `pb feedback general <what should change>` for durable guidance, "
            "or run `pb do <intent>` first if you want to correct a routed action."
        )
        raise typer.Exit(code=40)

    session = active_sessions[0]
    intent = " ".join(intent_words or []).strip()
    record_agent_weight_event(
        session.agent_id,
        "wrong",
        source_kind="human",
        session_id=session.id,
        metadata={"replacement_intent": intent},
    )
    patch_record = asyncio.run(
        judge_agent_instruction_fit(
            agent_id=session.agent_id,
            session_id=session.id,
            feedback_text=(
                f"User marked this agent as wrong."
                + (f" Replacement intent: {intent}" if intent else "")
            ),
            evidence=[f"session_id: {session.id}", f"agent_id: {session.agent_id}"],
            trigger_kind="feedback_wrong",
            auto_apply=True,
        )
    )
    if not intent:
        console.print(f"[dim]Marked {session.agent_id} as wrong.[/]")
        if patch_record is not None:
            console.print(format_patch_announcement(patch_record))
        return
    if patch_record is not None:
        console.print(format_patch_announcement(patch_record))

    repo = ctx.obj["repo"]
    from pb.cli.commands.do import _render_envelope
    from pb.core.dispatcher import dispatch
    envelope = asyncio.run(dispatch(repo, intent, excluded_agent_ids={session.agent_id}))
    _render_envelope(ctx, envelope, repo, first_turn=True)


@app.command("level")
def feedback_level(
    ctx: typer.Context,
    topic_words: Optional[list[str]] = typer.Argument(None, help="Concept, subskill, or topic being self-assessed."),
    level: Optional[int] = typer.Option(None, "--level", "-l", help="Current understanding level, 1-5."),
    confidence: Optional[int] = typer.Option(None, "--confidence", "-c", help="Confidence in that estimate, 1-5."),
    evidence: str = typer.Option("", "--evidence", "-e", help="Concrete evidence for the self-assessment."),
    note: str = typer.Option("", "--note", "-n", help="Additional context to remember."),
) -> None:
    """Record an explicit learner self-report without adding finish-time prompts."""

    _record_feedback_level(
        ctx,
        topic_words=topic_words,
        level=level,
        confidence=confidence,
        evidence=evidence,
        note=note,
    )
