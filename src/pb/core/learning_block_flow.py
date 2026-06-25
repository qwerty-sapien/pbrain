# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared revision-control helpers for study, practise, and teach flows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from pb.cli.console import get_console
from pb.cli.helpers import prompt_text
from pb.cli.pickers import pick_single_choice
from pb.core.learner_memory import build_global_learner_profile, learner_profile_prompt
from pb.core.product_control import AdaptiveOption, ControlDecision, ControlState, ProductControlEngine


@dataclass
class RevisionFeedbackResult:
    event_kind: str
    free_text: str
    decision: ControlDecision
    state: ControlState
    prompt_suffix: str


def learner_profile_suffix(repo, runtime_ctx) -> str:
    """Return the prompt suffix for the global learner profile."""

    return learner_profile_prompt(build_global_learner_profile(repo, runtime_ctx))


def adaptive_revision_options(
    engine: ProductControlEngine,
    *,
    mode: str,
    state: ControlState,
) -> list[AdaptiveOption]:
    """Return either escalated adaptive options or the base per-mode menu."""

    last_decision = dict(getattr(state, "last_decision", {}) or {})
    options = last_decision.get("adaptive_options")
    if isinstance(options, list) and options:
        try:
            return [AdaptiveOption.model_validate(item) for item in options]
        except Exception:
            return engine.default_options(mode=mode)
    return engine.default_options(mode=mode)


def collect_revision_feedback(
    *,
    engine: ProductControlEngine,
    repo,
    runtime_ctx,
    mode: str,
    artifact_kind: str,
    artifact_id: str,
    current_artifact: str,
    domain: str,
    target: str,
    goal_id: str = "",
    task_id: str = "",
    session_id: str = "",
    goal=None,
    current_node=None,
    title: str = "Revise draft",
) -> RevisionFeedbackResult | None:
    """Prompt for revision feedback, interpret it, and persist local/global control state."""

    _, state = engine.load_state(
        scope="artifact",
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        goal_id=goal_id,
        task_id=task_id,
        session_id=session_id,
    )
    adaptive_options = adaptive_revision_options(engine, mode=mode, state=state)
    menu_options = [option for option in adaptive_options if option.key not in {"custom_revision", "chat"}]
    menu_options.append(
        AdaptiveOption(
            key="chat",
            label="Let's discuss this",
            description="Talk through the mismatch before redrawing the draft.",
            control_signal="chat",
        )
    )
    selected = pick_single_choice(
        [(option.key, option.label) for option in menu_options],
        title=title,
        text="Choose a revision direction, type your own revision inline, or open a short discussion.",
        details=[option.description for option in menu_options],
        allow_inline_edit=True,
        inline_prompt="Revision request",
        return_result=True,
    )
    if getattr(selected, "kind", "") == "cancel" or selected is None:
        return None
    chosen = None
    free_text = ""
    if getattr(selected, "kind", "") == "inline_text":
        free_text = str(getattr(selected, "value", "") or "").strip()
        if not free_text:
            return None
        chosen = AdaptiveOption(
            key="custom_revision",
            label="Custom revision",
            description="Apply the typed revision request.",
            control_signal="custom_revision",
        )
    else:
        selected_key = str(getattr(selected, "value", "") or "").strip()
        chosen = next((option for option in menu_options if option.key == selected_key), None)
    if chosen is None:
        return None

    if chosen.key == "chat":
        free_text = prompt_text("Discuss", default="").strip()
        if not free_text:
            return None
    elif chosen.control_signal in {"needs_prerequisite", "too_abstract", "too_basic", "too_applied"}:
        if chosen.payload.get("restart_point"):
            free_text = f"Restart from {chosen.payload['restart_point']}."

    event, state, decision = engine.record_feedback(
        artifact_kind=artifact_kind,
        artifact_id=artifact_id,
        label=chosen.label,
        free_text=free_text,
        current_artifact=current_artifact,
        domain=domain,
        target=target,
        goal_id=goal_id,
        task_id=task_id,
        session_id=session_id,
        explicit_kind=chosen.control_signal or chosen.key,
        goal=goal,
        current_node=current_node,
    )
    console = get_console()
    if decision.action == "branch_rebase":
        console.print(
            "[dim]You've asked for a more foundational path more than once. "
            "I'm rebasing from a nearer prerequisite instead of just lightly simplifying.[/]"
        )
    elif decision.action == "global_rebuild":
        console.print(
            "[dim]Repeated feedback suggests the whole path starts too advanced. "
            "I'm rebuilding from an earlier prerequisite floor.[/]"
        )
    prompt_suffix = (
        f"\nUser control signal: {event.kind}\n"
        f"User feedback text: {free_text or chosen.label}\n"
        f"Control decision: {decision.action}\n"
        f"Decision reason: {decision.reason}\n"
        f"Revision instruction: {decision.instruction}\n"
        f"Current knowns: {state.knowns}\n"
        f"Current unknowns: {state.unknowns}\n"
        f"Current floor: {state.current_floor}\n"
        f"Global profile:\n{learner_profile_prompt(build_global_learner_profile(repo, runtime_ctx))}"
    )
    return RevisionFeedbackResult(
        event_kind=event.kind,
        free_text=free_text or chosen.label,
        decision=decision,
        state=state,
        prompt_suffix=prompt_suffix,
    )
