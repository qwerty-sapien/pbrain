# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Reusable learner-control engine across learner-facing workflows."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field

from pb.core.prerequisites import (
    build_prerequisite_chain,
    infer_knowns_unknowns,
    suggest_restart_points,
)


SUPPORTED_FEEDBACK_KINDS = {
    "accept",
    "cancel",
    "too_advanced",
    "too_basic",
    "too_abstract",
    "too_applied",
    "wrong_scope",
    "needs_prerequisite",
    "custom_revision",
    "chat",
}


class FeedbackEvent(BaseModel):
    id: str = Field(default_factory=lambda: str(uuid4()))
    scope_key: str = ""
    scope: str = "artifact"
    kind: str
    artifact_kind: str
    artifact_id: str
    node_id: str = ""
    label: str = ""
    free_text: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class ControlState(BaseModel):
    scope: str
    goal_id: str = ""
    task_id: str = ""
    session_id: str = ""
    feedback_events: list[FeedbackEvent] = Field(default_factory=list)
    signal_counts: dict[str, int] = Field(default_factory=dict)
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    current_floor: str = ""
    last_decision: dict[str, Any] = Field(default_factory=dict)
    escalation_level: int = 0


class AdaptiveOption(BaseModel):
    key: str
    label: str
    description: str = ""
    control_signal: str = ""
    payload: dict[str, Any] = Field(default_factory=dict)
    is_default: bool = False


class ControlDecision(BaseModel):
    action: str
    reason: str
    instruction: str
    adaptive_options: list[AdaptiveOption] = Field(default_factory=list)
    requires_global_rebuild: bool = False


class FeedbackInterpretationDraft(BaseModel):
    kind: str = "custom_revision"
    label: str = ""
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    notes: str = ""


def scope_key_for(*, scope: str, artifact_kind: str = "", artifact_id: str = "", task_id: str = "", session_id: str = "") -> str:
    if scope == "global":
        return "global:learner"
    if scope == "session" and session_id:
        return f"session:{session_id}"
    if task_id:
        return f"{scope}:{artifact_kind}:{artifact_id}:task:{task_id}"
    return f"{scope}:{artifact_kind}:{artifact_id}"


def _signal_family(kind: str) -> str:
    normalized = (kind or "").strip().lower()
    if normalized in {"too_advanced", "needs_prerequisite"}:
        return "foundational"
    if normalized in {"too_abstract"}:
        return "concrete"
    if normalized in {"too_basic"}:
        return "advance"
    if normalized in {"wrong_scope"}:
        return "scope"
    if normalized in {"too_applied"}:
        return "conceptual"
    return normalized or "custom_revision"


def _heuristic_kind(text: str) -> str:
    lowered = " ".join((text or "").lower().split())
    if any(token in lowered for token in ("more foundational", "too advanced", "work on the basics", "start earlier", "prerequisite", "dont know", "don't know", "what is ")):
        return "needs_prerequisite"
    if any(token in lowered for token in ("too abstract", "more concrete", "give me drills", "worked example", "example first")):
        return "too_abstract"
    if any(token in lowered for token in ("too basic", "already know", "skip ahead", "move faster", "harder")):
        return "too_basic"
    if any(token in lowered for token in ("wrong scope", "not what i meant", "different topic", "focus on")):
        return "wrong_scope"
    if any(token in lowered for token in ("too applied", "more conceptual", "less application", "theory first")):
        return "too_applied"
    if "chat" in lowered:
        return "chat"
    return "custom_revision"


def _merge_unique(existing: list[str], additions: list[str]) -> list[str]:
    merged = list(existing or [])
    for item in additions:
        clean = str(item).strip()
        if clean and clean not in merged:
            merged.append(clean)
    return merged


class ProductControlEngine:
    """Structured learner-feedback accumulation and escalation."""

    def __init__(self, *, repo=None, runtime=None):
        self.repo = repo
        self.runtime = runtime

    def load_state(
        self,
        *,
        scope: str,
        artifact_kind: str,
        artifact_id: str,
        goal_id: str = "",
        task_id: str = "",
        session_id: str = "",
    ) -> tuple[str, ControlState]:
        scope_key = scope_key_for(
            scope=scope,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            task_id=task_id,
            session_id=session_id,
        )
        payload = self.repo.get_control_state_snapshot(scope_key) if self.repo is not None else None
        if payload:
            try:
                state = ControlState.model_validate(payload)
            except Exception:
                state = ControlState(scope=scope, goal_id=goal_id, task_id=task_id, session_id=session_id)
        else:
            state = ControlState(scope=scope, goal_id=goal_id, task_id=task_id, session_id=session_id)
        events = self.repo.list_feedback_events(scope_key=scope_key, limit=100) if self.repo is not None else []
        if events:
            state.feedback_events = [FeedbackEvent.model_validate(item) for item in events]
        return scope_key, state

    def interpret_feedback(
        self,
        *,
        artifact_kind: str,
        artifact_id: str,
        label: str,
        free_text: str,
        current_artifact: str,
        prior_state: ControlState,
        explicit_kind: str = "",
    ) -> FeedbackEvent:
        kind = explicit_kind or _heuristic_kind(f"{label}\n{free_text}")
        knowns, unknowns = infer_knowns_unknowns(
            free_text or label,
            current_artifact,
            prior_state,
            runtime=self.runtime,
        )
        metadata: dict[str, Any] = {
            "knowns": knowns,
            "unknowns": unknowns,
        }
        if explicit_kind not in SUPPORTED_FEEDBACK_KINDS and free_text.strip() and self.runtime is not None and self.runtime.health().available:
            prompt = (
                "Interpret this learner feedback into one supported control signal.\n"
                f"Supported kinds: {sorted(SUPPORTED_FEEDBACK_KINDS)}\n"
                f"Current artifact: {current_artifact[:2500]}\n"
                f"Prior unknowns: {prior_state.unknowns}\n"
                f"Learner feedback: {free_text or label}\n"
            )
            try:
                interpreted = self.runtime.generate_draft(
                    FeedbackInterpretationDraft,
                    prompt,
                    source_scope=f"product_control:interpret:{artifact_kind}:{artifact_id}",
                    model=self.runtime.config.model_roles.fast_inference or self.runtime.config.model_roles.default,
                    max_output_tokens=4000,
                ).payload
                if interpreted.kind in SUPPORTED_FEEDBACK_KINDS:
                    kind = interpreted.kind
                metadata["knowns"] = _merge_unique(knowns, interpreted.knowns)
                metadata["unknowns"] = _merge_unique(unknowns, interpreted.unknowns)
            except Exception:
                pass
        return FeedbackEvent(
            scope_key="",
            scope="artifact",
            kind=kind,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            label=label,
            free_text=free_text,
            metadata=metadata,
        )

    def apply_feedback(
        self,
        *,
        scope_key: str,
        state: ControlState,
        event: FeedbackEvent,
        domain: str,
        target: str,
        goal=None,
        current_node=None,
    ) -> tuple[ControlState, ControlDecision]:
        event.scope_key = scope_key
        state.feedback_events.append(event)
        family = _signal_family(event.kind)
        count = int(state.signal_counts.get(family, 0) or 0) + 1
        state.signal_counts[family] = count
        state.knowns = _merge_unique(state.knowns, list(event.metadata.get("knowns", []) or []))
        state.unknowns = _merge_unique(state.unknowns, list(event.metadata.get("unknowns", []) or []))

        decision = self._decision_for(
            family=family,
            count=count,
            state=state,
            domain=domain,
            target=target,
            goal=goal,
            current_node=current_node,
        )
        state.escalation_level = max(state.escalation_level, count if family in {"foundational", "concrete", "advance", "scope"} else state.escalation_level)
        state.last_decision = decision.model_dump(mode="json")

        if self.repo is not None:
            self.repo.append_feedback_event(event.model_dump(mode="json"))
            self.repo.save_control_state_snapshot(scope_key, state.scope, state.model_dump(mode="json"))
        return state, decision

    def record_feedback(
        self,
        *,
        artifact_kind: str,
        artifact_id: str,
        label: str,
        free_text: str,
        current_artifact: str,
        domain: str,
        target: str,
        scope: str = "artifact",
        goal_id: str = "",
        task_id: str = "",
        session_id: str = "",
        explicit_kind: str = "",
        goal=None,
        current_node=None,
    ) -> tuple[FeedbackEvent, ControlState, ControlDecision]:
        artifact_scope_key, artifact_state = self.load_state(
            scope=scope,
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            goal_id=goal_id,
            task_id=task_id,
            session_id=session_id,
        )
        interpreted = self.interpret_feedback(
            artifact_kind=artifact_kind,
            artifact_id=artifact_id,
            label=label,
            free_text=free_text,
            current_artifact=current_artifact,
            prior_state=artifact_state,
            explicit_kind=explicit_kind,
        )
        artifact_state, decision = self.apply_feedback(
            scope_key=artifact_scope_key,
            state=artifact_state,
            event=interpreted,
            domain=domain,
            target=target,
            goal=goal,
            current_node=current_node,
        )

        if self.repo is not None:
            global_scope_key, global_state = self.load_state(
                scope="global",
                artifact_kind=artifact_kind,
                artifact_id="learner",
            )
            global_event = interpreted.model_copy(
                update={
                    "id": str(uuid4()),
                    "scope": "global",
                    "scope_key": global_scope_key,
                    "artifact_id": "learner",
                }
            )
            self.apply_feedback(
                scope_key=global_scope_key,
                state=global_state,
                event=global_event,
                domain=domain,
                target=target,
                goal=goal,
                current_node=current_node,
            )
            if session_id:
                session_scope_key, session_state = self.load_state(
                    scope="session",
                    artifact_kind=artifact_kind,
                    artifact_id=artifact_id,
                    goal_id=goal_id,
                    task_id=task_id,
                    session_id=session_id,
                )
                session_event = interpreted.model_copy(
                    update={
                        "id": str(uuid4()),
                        "scope": "session",
                        "scope_key": session_scope_key,
                    }
                )
                self.apply_feedback(
                    scope_key=session_scope_key,
                    state=session_state,
                    event=session_event,
                    domain=domain,
                    target=target,
                    goal=goal,
                    current_node=current_node,
                )
        return interpreted, artifact_state, decision

    def default_options(self, *, mode: str = "study") -> list[AdaptiveOption]:
        options = [
            AdaptiveOption(
                key="needs_prerequisite",
                label="Start earlier / prerequisites first",
                description="Rebase from missing prerequisites instead of lightly simplifying.",
                control_signal="needs_prerequisite",
                is_default=True,
            ),
            AdaptiveOption(
                key="wrong_scope",
                label="Wrong scope",
                description="The target should shift to a different subtopic or capability.",
                control_signal="wrong_scope",
            ),
            AdaptiveOption(
                key="too_basic",
                label="I already know this",
                description="Skip forward, but verify with a diagnostic checkpoint.",
                control_signal="too_basic",
            ),
            AdaptiveOption(
                key="custom_revision",
                label="Custom revision",
                description="Type the exact change you want.",
                control_signal="custom_revision",
            ),
            AdaptiveOption(
                key="chat",
                label="Let's discuss this",
                description="Talk through what feels off before deciding how to revise.",
                control_signal="chat",
            ),
        ]
        if mode == "practise":
            options.insert(
                1,
                AdaptiveOption(
                    key="too_abstract",
                    label="Make this more concrete",
                    description="Convert abstraction into drills or worked reps.",
                    control_signal="too_abstract",
                ),
            )
        elif mode == "teach":
            options.insert(
                1,
                AdaptiveOption(
                    key="too_abstract",
                    label="This is confusing",
                    description="Switch to diagnosis and explanation checkpoints.",
                    control_signal="too_abstract",
                ),
            )
        else:
            options.insert(
                1,
                AdaptiveOption(
                    key="too_abstract",
                    label="Make this more concrete",
                    description="Lead with examples or concrete drills.",
                    control_signal="too_abstract",
                ),
            )
            options.insert(
                2,
                AdaptiveOption(
                    key="too_applied",
                    label="Make this more conceptual",
                    description="Pull back from application and explain the structure first.",
                    control_signal="too_applied",
                ),
            )
        return options

    def _decision_for(
        self,
        *,
        family: str,
        count: int,
        state: ControlState,
        domain: str,
        target: str,
        goal=None,
        current_node=None,
    ) -> ControlDecision:
        if family == "accept":
            return ControlDecision(action="accept", reason="The learner accepted the draft.", instruction="Keep the current artifact unchanged.")
        if family == "cancel":
            return ControlDecision(action="cancel", reason="The learner canceled this draft.", instruction="Stop and keep nothing.")
        if family == "foundational":
            chain = build_prerequisite_chain(domain, target, state.knowns, state.unknowns, runtime=self.runtime)
            restart_points = suggest_restart_points(goal, current_node, state, domain=domain, target=target, runtime=self.runtime)
            state.current_floor = restart_points[0] if restart_points else (chain[0] if chain else "")
            if count <= 1:
                action = "local_refine"
                reason = "First foundational/prerequisite signal; refine the current branch locally."
                instruction = (
                    "Refine this artifact from a nearer prerequisite. "
                    f"Keep the learner's knowns {state.knowns} intact, but address unknowns {state.unknowns} before returning to `{target}`."
                )
            elif count == 2:
                action = "branch_rebase"
                reason = "Repeated foundational feedback; rebase from a nearer prerequisite instead of lightly simplifying."
                instruction = (
                    "Rebase the current branch from a nearer prerequisite restart point. "
                    f"Offer a restart from one of: {restart_points or chain[:5]}."
                )
            else:
                action = "global_rebuild"
                reason = "Repeated foundational feedback indicates the whole path starts too advanced."
                instruction = (
                    "Rebuild the entire roadmap or block sequence from the inferred foundational floor. "
                    f"Start from `{state.current_floor or (restart_points[0] if restart_points else target)}` and reconnect to `{target}` only after prerequisites."
                )
            options = [
                AdaptiveOption(
                    key=f"restart_{index}",
                    label=point,
                    description=f"Restart from {point}.",
                    control_signal="needs_prerequisite",
                    payload={"restart_point": point},
                    is_default=index == 0,
                )
                for index, point in enumerate(restart_points)
            ]
            options.extend(
                [
                    AdaptiveOption(key="custom_revision", label="Custom revision", description="Type another prerequisite shift.", control_signal="custom_revision"),
                    AdaptiveOption(key="chat", label="Let's discuss this", description="Talk through what feels mismatched.", control_signal="chat"),
                ]
            )
            return ControlDecision(
                action=action,
                reason=reason,
                instruction=instruction,
                adaptive_options=options,
                requires_global_rebuild=action == "global_rebuild",
            )
        if family == "concrete":
            action = "concretize" if count == 1 else ("drillify" if count == 2 else "rebuild_concrete")
            reason = "The learner wants less abstraction and more concrete execution."
            instruction = (
                "Replace abstract exposition with concrete examples, drills, or worked applications. "
                "Keep the same target unless the learner also signals wrong scope."
            )
            options = [
                AdaptiveOption(key="example_first", label="Worked example first", description="Lead with one worked example.", control_signal="too_abstract", is_default=True),
                AdaptiveOption(key="drill_first", label="Concrete drill first", description="Move directly to a drill or rep.", control_signal="too_abstract"),
                AdaptiveOption(key="custom_revision", label="Custom revision", description="Type a different concrete shift.", control_signal="custom_revision"),
            ]
            return ControlDecision(action=action, reason=reason, instruction=instruction, adaptive_options=options)
        if family == "advance":
            action = "diagnostic_skip" if count == 1 else ("rescope_higher" if count == 2 else "rebuild_higher")
            reason = "The learner says this is too basic or already known."
            instruction = (
                "Skip forward with diagnostic confirmation. Preserve the learner's likely known concepts, and move to the next unproven layer."
            )
            return ControlDecision(
                action=action,
                reason=reason,
                instruction=instruction,
                adaptive_options=[
                    AdaptiveOption(key="diagnostic", label="Diagnostic checkpoint", description="Verify readiness before skipping.", control_signal="too_basic", is_default=True),
                    AdaptiveOption(key="harder", label="Move one layer up", description="Advance to the next stronger target.", control_signal="too_basic"),
                ],
            )
        if family == "scope":
            action = "reanchor_scope" if count == 1 else "rebuild_scope"
            reason = "The learner says the artifact targets the wrong scope."
            instruction = "Re-anchor the artifact to the requested subtopic or goal slice before proceeding."
            return ControlDecision(
                action=action,
                reason=reason,
                instruction=instruction,
                adaptive_options=[
                    AdaptiveOption(key="custom_scope", label="Describe the intended scope", description="State the exact concept or capability to target.", control_signal="wrong_scope", is_default=True),
                ],
                requires_global_rebuild=action == "rebuild_scope",
            )
        if family == "conceptual":
            return ControlDecision(
                action="make_conceptual",
                reason="The learner wants less premature application.",
                instruction="Shift from applied drills toward concept-first explanation and structure.",
                adaptive_options=[
                    AdaptiveOption(key="concept_first", label="Concepts before applications", description="Explain the structure first.", control_signal="too_applied", is_default=True),
                    AdaptiveOption(key="custom_revision", label="Custom revision", description="Type the conceptual shift you want.", control_signal="custom_revision"),
                ],
            )
        if family == "chat":
            return ControlDecision(
                action="chat",
                reason="The learner wants to discuss the mismatch before revising.",
                instruction="Discuss the mismatch explicitly, then refine based on what they reveal.",
                adaptive_options=[
                    AdaptiveOption(key="custom_revision", label="Turn this into a change", description="After discussing, convert it into a structured revision.", control_signal="custom_revision", is_default=True),
                ],
            )
        return ControlDecision(
            action="local_refine",
            reason="Apply the learner's custom revision directly.",
            instruction="Refine the artifact using the learner's requested change, preserving the validated parts.",
            adaptive_options=[
                AdaptiveOption(key="custom_revision", label="Custom revision", description="Apply the requested change.", control_signal="custom_revision", is_default=True),
            ],
        )
