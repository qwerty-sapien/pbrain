# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Thin staged-intake helpers inspired by GSD's discuss architecture.

The goal here is not to recreate a heavyweight planning filesystem. We keep
just enough structured runtime state to explain what the CLI inferred, what it
asked, and what eventually got persisted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pb.core.models import generate_internal_id, utc_now


def _iso_now() -> str:
    return utc_now().isoformat()


@dataclass
class StageEntry:
    """One recorded step in a staged intake flow."""

    stage: str
    content: Any
    status: str = "ok"
    created_at: str = field(default_factory=_iso_now)


@dataclass
class StageRecorder:
    """Collect and persist a lightweight stage transcript."""

    data_dir: Path
    workflow: str
    intent: str
    run_id: str = field(default_factory=generate_internal_id)
    route_hint: str = ""
    entries: list[StageEntry] = field(default_factory=list)
    outcome: str = "pending"
    metadata: dict[str, Any] = field(default_factory=dict)

    def add(self, stage: str, content: Any, *, status: str = "ok") -> None:
        self.entries.append(StageEntry(stage=stage, content=content, status=status))

    def finalize(self, outcome: str, **metadata: Any) -> Path:
        self.outcome = outcome
        if metadata:
            self.metadata.update(metadata)
        return self.persist()

    def persist(self) -> Path:
        intake_dir = self.data_dir / "intake"
        intake_dir.mkdir(parents=True, exist_ok=True)
        timestamp = utc_now().strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp}-{self.workflow}-{self.run_id[:8]}.json"
        path = intake_dir / filename
        payload = {
            "id": self.run_id,
            "workflow": self.workflow,
            "intent": self.intent,
            "route_hint": self.route_hint,
            "outcome": self.outcome,
            "metadata": self.metadata,
            "entries": [
                {
                    "stage": entry.stage,
                    "status": entry.status,
                    "created_at": entry.created_at,
                    "content": entry.content,
                }
                for entry in self.entries
            ],
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True, default=str) + "\n", encoding="utf-8")
        return path


def build_learning_context(repo, runtime, *, limit: int = 3) -> dict[str, Any]:
    """Gather a small local snapshot before asking the user anything."""

    active_session = repo.get_active_session()
    active_goals = repo.list_goal_arcs(status=None)[:limit]
    recent_sessions = []
    if repo is not None:
        rows = []
        for task in repo.list_tasks():
            for session in repo.list_sessions_for_task(task.id):
                rows.append((session.start_at, session.branch or "study", session.subject_scope or task.title))
        rows.sort(key=lambda item: item[0], reverse=True)
        for _, branch, scope in rows[:limit]:
            recent_sessions.append({"branch": branch, "scope": scope})

    try:
        health = runtime.health()
        provider_health = {
            "provider": health.provider,
            "model": health.default_model,
            "available": health.available,
            "message": health.message,
        }
    except Exception as exc:  # pragma: no cover - defensive fallback
        provider_health = {"available": False, "message": str(exc)}

    quarantine_root = getattr(runtime, "quarantine_path", None)
    pending_thoughts = 0
    if quarantine_root is not None:
        try:
            pending_thoughts = len(list((quarantine_root / "thoughts").glob("*.md")))
        except Exception:
            pending_thoughts = 0

    return {
        "vault": getattr(runtime, "vault_name", ""),
        "active_session": {
            "branch": getattr(active_session, "branch", ""),
            "scope": getattr(active_session, "subject_scope", ""),
        }
        if active_session is not None
        else None,
        "active_goals": [
            {
                "title": goal.title,
                "domain": getattr(goal, "domain", ""),
                "mode": getattr(goal, "execution_mode", "mixed"),
            }
            for goal in active_goals
        ],
        "recent_sessions": recent_sessions,
        "provider_health": provider_health,
        "pending_thoughts": pending_thoughts,
    }


def build_reflection(workflow: str, intent: str, context: dict[str, Any]) -> str:
    """Summarize what the system thinks the user wants before clarifying."""

    normalized = " ".join((intent or "").split()) or "the current learning request"
    goal_count = len(context.get("active_goals", []))
    active = context.get("active_session") or {}
    if workflow == "goal":
        return (
            f"You want to turn `{normalized}` into a concrete learning direction. "
            f"I found {goal_count} active goal(s)"
            f"{' and an active ' + active.get('branch', 'study') + ' session' if active else ''}, "
            "so the next step is to infer the goal shape, check any ambiguity, then persist only after preview."
        )
    if workflow == "plan_day":
        return (
            f"You want to turn your current goals into executable learning blocks around `{normalized}`. "
            "I gathered goal, session, and provider context first so the plan can stay concrete and low-ceremony."
        )
    if workflow == "study":
        return (
            f"You want an immediate conceptual study block for `{normalized}`. "
            "I’ll assume the fastest route unless one high-impact clarification is still needed."
        )
    if workflow == "practise":
        return (
            f"You want an immediate deliberate-practice block for `{normalized}`. "
            "I’ll bias toward an executable drill, evidence target, and a clear success check."
        )
    return (
        f"You want to move `{normalized}` forward. "
        "I gathered local context first so the CLI can make assumptions before asking more from you."
    )


def build_assumptions(workflow: str, intent: str, context: dict[str, Any]) -> list[str]:
    """Generate structured assumptions from local context before clarification."""

    normalized = " ".join((intent or "").split())
    assumptions: list[str] = []
    if normalized:
        assumptions.append(f"The main focus is `{normalized}`.")
    if workflow == "goal":
        assumptions.append("The user prefers a goal that can route directly into study, practise, or planning.")
    if workflow == "study":
        assumptions.append("Conceptual understanding and active retrieval matter more than passive reading.")
    if workflow == "practise":
        assumptions.append("The next step should be a drill with feedback, not a vague aspiration.")
    active_goals = context.get("active_goals", [])
    if active_goals:
        assumptions.append(f"There are already {len(active_goals)} active goal(s) to align against.")
    active_session = context.get("active_session")
    if active_session:
        assumptions.append(
            f"There is an active {active_session.get('branch', 'study')} session"
            f" on `{active_session.get('scope', 'current focus')}`."
        )
    return assumptions


def needs_single_clarification(text: str) -> bool:
    """Return True for terse inputs that benefit from one focused follow-up."""

    tokens = [token for token in (text or "").split() if token.strip()]
    normalized = " ".join(tokens)
    return len(tokens) <= 2 or len(normalized) <= 6
