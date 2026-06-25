# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Durable global profile building over canonical learning dossiers."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel, Field

from pb.core.learning_dossier import (
    LearningDossierUpdater,
    PartnerDossierSignals,
    list_learning_dossiers,
    resolve_subtopic_dossier_key,
)
from pb.core.anki_bootstrap import ANKI_RECENT_REVIEW_SIGNAL_PREF, ANKI_REVIEW_THRESHOLD
from pb.core.feedback_profile import load_learner_level_assertions
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.learning_partner import load_session_transcript
from pb.llm.runtime import LLMRuntime


class PartnerSessionCompactDraft(BaseModel):
    summary: str
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    detected_gaps: list[str] = Field(default_factory=list)
    recall_candidates: list[str] = Field(default_factory=list)
    corrections: list[str] = Field(default_factory=list)
    next_drill: str = ""
    next_action: str = ""
    control_signals: list[str] = Field(default_factory=list)
    escalation_level: int = 0

def _relative_path(base: Path, value: Optional[Path]) -> str:
    if value is None:
        return ""
    try:
        return str(value.relative_to(base))
    except Exception:
        return str(value)


def append_partner_session_memory(
    *,
    runtime: LLMRuntime,
    runtime_ctx,
    repo,
    session,
    task,
) -> Path | None:
    generated = dict(getattr(session, "generated_names", {}) or {})
    if not generated.get("learning_partner_used"):
        return None
    transcript = load_session_transcript(Path(runtime_ctx.data_dir), getattr(session, "id", ""))
    if not transcript or not runtime.health().available:
        return None

    transcript_path = Path(runtime_ctx.data_dir) / "transcripts" / f"{getattr(session, 'id', '')}.json"
    prompt = (
        "Compact this learning-partner session into durable learner memory.\n"
        "Be concrete, not flattering. Extract only specific knowns, unknowns, gaps, corrections, and next moves.\n"
        f"Task title: {getattr(task, 'title', '')}\n"
        f"Branch: {getattr(session, 'branch', '')}\n"
        f"Objective: {getattr(session, 'intended_outcome', '') or getattr(session, 'subject_scope', '')}\n"
        f"Observed errors: {getattr(session, 'observed_errors', '')}\n"
        f"Next adjustment: {getattr(session, 'next_adjustment', '')}\n"
        f"Partner closeout: {generated.get('learning_partner_closeout', {})}\n"
        f"Recent control state: {generated.get('control_state_snapshot', {})}\n"
        f"Transcript: {transcript[-16:]}\n"
    )
    draft = runtime.generate_draft(
        PartnerSessionCompactDraft,
        prompt,
        source_scope=f"learner_memory:{getattr(session, 'id', '')}",
        model=runtime.config.model_roles.fast_inference or runtime.config.model_roles.default,
        max_output_tokens=4000,
    ).payload

    control_state = generated.get("control_state_snapshot") if isinstance(generated.get("control_state_snapshot"), dict) else {}
    signal_counts = control_state.get("signal_counts", {}) if isinstance(control_state, dict) else {}
    signal_labels = []
    if isinstance(signal_counts, dict):
        for key, value in signal_counts.items():
            signal_labels.extend([str(key)] * int(value or 0))

    metadata = parse_learning_task_metadata(task)
    updater = LearningDossierUpdater(Path(runtime_ctx.vault_path))
    key = resolve_subtopic_dossier_key(
        session=session,
        task=task,
        domain=str(metadata.domain or getattr(session, "subject_scope", "") or getattr(task, "title", "") or ""),
        subtopic=str(getattr(session, "subject_scope", "") or metadata.scope or getattr(task, "title", "") or ""),
    )
    path = updater.upsert(
        key=key,
        session=session,
        task=task,
        partner=PartnerDossierSignals(
            summary=draft.summary,
            knowns=tuple(draft.knowns),
            unknowns=tuple(draft.unknowns),
            detected_gaps=tuple(draft.detected_gaps),
            recall_candidates=tuple(draft.recall_candidates),
            corrections=tuple(draft.corrections),
            next_drill=draft.next_drill,
            next_action=draft.next_action,
            control_signals=tuple(draft.control_signals or signal_labels),
            escalation_level=draft.escalation_level,
        ),
        transcript_path=transcript_path,
    )

    generated["learning_partner_compact"] = draft.model_dump(mode="json")
    generated["learning_partner_dossier_path"] = str(path)
    session.generated_names = generated
    repo.update_session(session)
    return path


def build_global_learner_profile(repo, runtime_ctx, *, limit: int = 20) -> dict[str, Any]:
    feedback_events = repo.list_feedback_events(scope_key="global:learner", limit=limit)
    signal_counter = Counter(str(item.get("kind", "")) for item in feedback_events if item.get("kind"))
    dossiers = list_learning_dossiers(Path(runtime_ctx.vault_path))
    dossiers.sort(
        key=lambda item: (
            str(item.get("updated", "") or ""),
            str(item.get("title", "") or ""),
        ),
        reverse=True,
    )
    recent_unknowns: list[str] = []
    recent_knowns: list[str] = []
    next_drills: list[str] = []
    for dossier in dossiers[:limit]:
        for key, bucket in (("weaknesses", recent_unknowns), ("strengths", recent_knowns), ("next_drills", next_drills)):
            values = dossier.get(key, [])
            if isinstance(values, list):
                for item in values:
                    text = str(item).strip()
                    if text and text not in bucket:
                        bucket.append(text)
    config = getattr(runtime_ctx, "config", None)
    preferences = dict(getattr(config, "preferences", {}) or {})
    raw_anki_signal = preferences.get(ANKI_RECENT_REVIEW_SIGNAL_PREF)
    anki_signal = raw_anki_signal if isinstance(raw_anki_signal, dict) else {}
    try:
        anki_reviews = int(anki_signal.get("reviews_since_last_check", 0) or 0)
    except (TypeError, ValueError):
        anki_reviews = 0
    if anki_reviews < ANKI_REVIEW_THRESHOLD:
        anki_signal = {}

    try:
        learner_level_assertions = load_learner_level_assertions(Path(runtime_ctx.vault_path), limit=5)
    except Exception:
        learner_level_assertions = []

    return {
        "top_signals": dict(signal_counter.most_common(6)),
        "recent_unknowns": recent_unknowns[:8],
        "recent_knowns": recent_knowns[:8],
        "next_drills": next_drills[:5],
        "partner_memory_rows": len(dossiers),
        "anki_review_signal": anki_signal,
        "learner_level_assertions": learner_level_assertions,
    }


def learner_profile_prompt(profile: dict[str, Any]) -> str:
    if not profile:
        return ""
    return (
        "Global learner profile:\n"
        f"- Top feedback signals: {profile.get('top_signals', {})}\n"
        f"- Recent unknowns: {profile.get('recent_unknowns', [])}\n"
        f"- Recent knowns: {profile.get('recent_knowns', [])}\n"
        f"- Recent next drills: {profile.get('next_drills', [])}\n"
        f"- Partner memory rows: {profile.get('partner_memory_rows', 0)}\n"
        f"- Qualifying Anki review signal: {profile.get('anki_review_signal', {})}\n"
        f"- Learner self-reports: {profile.get('learner_level_assertions', [])}\n"
    )
