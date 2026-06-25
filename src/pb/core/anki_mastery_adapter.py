# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Thin swappable mastery→weighting adapter for Anki suggested-card reordering.

D-04: Signal was LessonSkillStateRecord.overall_status (demonstrated per-skill
      performance) in Phase 15. D-06 seam closed in Phase 16.
D-05: Influences review WEIGHTING ONLY — card generation is never touched.
D-06: Seam closed in Phase 16 — now reads concept_confidence (D-16-17).
      _weak_skill_slugs body replaced; anki.py call site unchanged.

# NOTE: from pb.core.confidence_model import THRESHOLD_FULL is imported locally
# inside _weak_skill_slugs to avoid circular import risk.
"""

from __future__ import annotations

from typing import Any


def _weak_skill_slugs(repo: Any, lesson_run_id: str) -> set[str]:
    """Return concept slugs where confidence is not full (Phase 16, D-16-17/D-16-18).

    lesson_run_id is accepted for signature compatibility but IGNORED in Phase 16 —
    concept_confidence is concept-global, not run-scoped. The D-06 seam is now closed.
    """
    from pb.core.confidence_model import THRESHOLD_FULL

    records = repo.list_concept_confidence()  # all records, no filter
    weak: set[str] = set()
    for record in records or []:
        score = getattr(record, "confidence_score", 1.0)
        concept_id = getattr(record, "concept_id", "") or ""
        if not concept_id:
            continue
        parts = concept_id.split(":")
        if len(parts) < 3:
            # L1 landmine guard: concept_id "concept:cs" has no slug segment — skip
            continue
        if score < THRESHOLD_FULL:  # not "full" means weak (none or partial)
            slug = parts[-1].lower()
            if slug:
                weak.add(slug)
    return weak


def _card_matches_weak_skill(card: dict, weak_slugs: set[str]) -> bool:
    """Heuristic card→skill match over note_slug / tags / domain (no skill_slug column exists)."""
    if not weak_slugs:
        return False
    haystacks: list[str] = []
    for key in ("note_slug", "domain"):
        value = str(card.get(key, "") or "").lower()
        if value:
            haystacks.append(value)
    # tags is a JSON-encoded string list (default '[]')
    raw_tags = card.get("tags", "")
    if raw_tags:
        haystacks.append(str(raw_tags).lower())
    blob = " ".join(haystacks)
    return any(slug in blob for slug in weak_slugs)


def reorder_by_mastery(cards: list[dict], *, repo: Any) -> list[dict]:
    """Reorder cards so weak-concept slugs surface first (D-05, D-06 seam closed Phase 16).

    Weighting ONLY — does not generate, drop, or mutate cards; it returns the same
    card dicts in a new order. Weak concept = confidence_score < THRESHOLD_FULL (D-16-18).

    D-06 seam closed: _weak_skill_slugs now reads repo.list_concept_confidence() (D-16-17).
    anki.py call site is unchanged.
    The seam is this function's (cards, *, repo) -> list[dict] contract.
    """
    if not cards:
        return cards
    # Resolve the lesson run id from the cards' run_id field; if absent, no signal → passthrough.
    run_id = ""
    for card in cards:
        candidate = str(card.get("run_id", "") or "")
        if candidate:
            run_id = candidate
            break
    weak_slugs = _weak_skill_slugs(repo, run_id)
    if not weak_slugs:
        return cards  # graceful no-signal passthrough — order unchanged
    # Stable partition: weak-matching cards first, preserving input order within each tier.
    weak_cards = [c for c in cards if _card_matches_weak_skill(c, weak_slugs)]
    rest = [c for c in cards if not _card_matches_weak_skill(c, weak_slugs)]
    return weak_cards + rest
