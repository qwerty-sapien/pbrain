# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Adaptive finish classification for learning sessions."""

from __future__ import annotations

from pb.llm.drafts import CloseoutDecisionDraft


class CloseoutService:
    """Classify finish notes before the CLI decides how to close the session."""

    def generate_closeout(self, session, user_note: str, context: dict | None = None) -> CloseoutDecisionDraft:
        note = (user_note or "").strip()
        lowered = note.lower()
        context = context or {}

        if any(token in lowered for token in ("accident", "misclick", "mistake", "wrong session")):
            return CloseoutDecisionDraft(
                status="accidental_start",
                summary="This looks like an accidental session start.",
                discard_recommended=True,
                recovery_step="Discard it and restart only when the target is clear.",
            )
        if any(token in lowered for token in ("didnt do anything", "didn't do anything", "nothing", "no progress")):
            status = "frustration_feedback" if any(token in lowered for token in ("useless", "broken", "tool")) else "no_progress"
            return CloseoutDecisionDraft(
                status=status,
                summary="No meaningful learning evidence was recorded.",
                recovery_step=f"Define one tiny next action for {(context.get('title') or getattr(session, 'subject_scope', '') or 'this topic')}.",
                feedback_note=note if status == "frustration_feedback" else "",
                discard_recommended=True,
            )
        if any(token in lowered for token in ("useless", "broken", "tool")):
            return CloseoutDecisionDraft(
                status="frustration_feedback",
                summary="This note reads as product frustration rather than learning evidence.",
                recovery_step="Either discard the session or capture one recovery step after the product issue is addressed.",
                feedback_note=note,
                discard_recommended=True,
            )
        if any(token in lowered for token in ("blocked", "stuck", "couldn't", "couldnt", "can't", "cant")):
            return CloseoutDecisionDraft(
                status="blocked",
                summary="The session ended blocked rather than completed.",
                recovery_step="Name the exact blocker and turn it into the next drill or question.",
            )
        if any(token in lowered for token in ("partial", "some", "half")):
            return CloseoutDecisionDraft(
                status="partial",
                summary="The session made partial progress.",
                recovery_step="Reuse the strongest cue and target the unresolved gap next.",
            )
        if not note:
            return CloseoutDecisionDraft(
                status="completed",
                summary="No finish note was given, so this is treated as a normal closeout.",
                recovery_step="Write the next drill while the weak point is still obvious.",
            )
        return CloseoutDecisionDraft(
            status="completed",
            summary="The session has enough signal to close as completed.",
            recovery_step="Convert the clearest gap into the next short drill.",
        )
