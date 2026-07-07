# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

"""Behavioral tests for pb.core.anki_mastery_adapter.reorder_by_mastery.

Phase 16 (D-16-17/D-16-18) — tests cover concept_confidence signal, NOT
lesson_skill_states.

Tests cover:
  1. Weak-concept-first ordering (confidence_score < THRESHOLD_FULL surfaces first).
  2. Graceful no-signal passthrough (repo returns [] → order unchanged).
  3. Empty cards input → empty list returned.
  4. Unmappable cards (no matching slug) are treated as neutral — no crash.
  5. Signature stability: callable as reorder_by_mastery(cards, repo=repo) → list[dict].
  6. Phase 16 slug extraction from concept_id (split(":")[-1], len >= 3 guard).
  7. lesson_run_id parameter accepted but ignored (no effect on behaviour).
  8. THRESHOLD_FULL boundary: score >= THRESHOLD_FULL means NOT weak.
  9. 2-segment concept_id ("concept:cs") is skipped — len(parts) < 3 guard.
"""

from __future__ import annotations

from pb.core.confidence_model import ConceptConfidenceRecord, THRESHOLD_FULL

import pytest

from pb.core.anki_mastery_adapter import reorder_by_mastery


# ---------------------------------------------------------------------------
# Helpers — tiny fake repo, no DB
# ---------------------------------------------------------------------------

def _make_confidence_record(concept_id: str, score: float) -> ConceptConfidenceRecord:
    """Return a ConceptConfidenceRecord with the given concept_id and confidence_score."""
    return ConceptConfidenceRecord(concept_id=concept_id, confidence_score=score)


class FakeRepo:
    """Minimal duck-typed repo that returns configurable concept_confidence records."""

    def __init__(self, records: list[ConceptConfidenceRecord] | None = None) -> None:
        self._records = records or []

    def list_concept_confidence(self, concept_id: str | None = None) -> list[ConceptConfidenceRecord]:
        """Return all confidence records (concept_id arg ignored — full-table scan)."""
        return self._records


def _make_card(note_slug: str, run_id: str = "run-001", tags: str = "[]") -> dict:
    return {
        "id": note_slug,
        "note_slug": note_slug,
        "front": f"Front {note_slug}",
        "back": f"Back {note_slug}",
        "card_type": "Basic",
        "status": "suggested",
        "deck": "Default",
        "tags": tags,
        "anki_model": "Basic",
        "created_at": "2026-01-01T00:00:00Z",
        "updated_at": "2026-01-01T00:00:00Z",
        "domain": "",
        "exported_at": None,
        "anki_note_id": None,
        "run_id": run_id,
    }


# ---------------------------------------------------------------------------
# Test 1 — weak-concept-first ordering
# ---------------------------------------------------------------------------

def test_weak_concept_cards_surface_before_full_confidence():
    """Cards mapped to low-confidence concepts appear before high-confidence cards."""
    records = [
        _make_confidence_record("concept:cs:recursion", 0.2),      # weak
        _make_confidence_record("concept:cs:sorting", 0.9),         # full (not weak)
        _make_confidence_record("concept:cs:binary-search", 0.15),  # weak
    ]
    repo = FakeRepo(records)

    card_recursion = _make_card("recursion")       # weak concept
    card_sorting = _make_card("sorting")           # full concept (not weak)
    card_binary_search = _make_card("binary-search")  # weak concept

    result = reorder_by_mastery([card_sorting, card_recursion, card_binary_search], repo=repo)

    assert len(result) == 3
    # Weak cards (recursion, binary-search) must come before full-confidence card (sorting)
    weak_indices = [i for i, c in enumerate(result) if c["note_slug"] in ("recursion", "binary-search")]
    full_indices = [i for i, c in enumerate(result) if c["note_slug"] == "sorting"]
    assert max(weak_indices) < min(full_indices), (
        f"Expected weak-concept cards before full-confidence card; got order: {[c['note_slug'] for c in result]}"
    )


def test_relative_order_within_tier_is_stable():
    """Within the weak tier, cards preserve their original input order."""
    records = [
        _make_confidence_record("concept:cs:recursion", 0.2),
        _make_confidence_record("concept:cs:binary-search", 0.15),
        _make_confidence_record("concept:cs:sorting", 0.9),
    ]
    repo = FakeRepo(records)

    card_recursion = _make_card("recursion")
    card_binary_search = _make_card("binary-search")
    card_sorting = _make_card("sorting")

    result = reorder_by_mastery([card_recursion, card_binary_search, card_sorting], repo=repo)
    slugs = [c["note_slug"] for c in result]
    # recursion then binary-search (input order preserved within weak tier)
    assert slugs.index("recursion") < slugs.index("binary-search")


# ---------------------------------------------------------------------------
# Test 2 — graceful no-signal passthrough
# ---------------------------------------------------------------------------

def test_no_signal_passthrough_returns_same_order():
    """When repo returns [] (no concept_confidence rows), cards are returned in original order."""
    repo = FakeRepo(records=[])  # no records → no signal

    card_a = _make_card("topic-a")
    card_b = _make_card("topic-b")
    result = reorder_by_mastery([card_a, card_b], repo=repo)

    assert [c["note_slug"] for c in result] == ["topic-a", "topic-b"]


def test_no_run_id_on_cards_is_irrelevant_in_phase_16():
    """Phase 16: run_id on cards is ignored — concept_confidence is concept-global.

    Even without a run_id, weak-concept cards should surface first because the Phase 16
    adapter reads list_concept_confidence() with no filter.
    """
    records = [
        _make_confidence_record("concept:cs:recursion", 0.2),   # weak
    ]
    repo = FakeRepo(records)

    # run_id is empty — but Phase 16 adapter does NOT gate on run_id
    card_recursion = {
        "id": "recursion",
        "note_slug": "recursion",
        "front": "Q",
        "back": "A",
        "run_id": "",  # empty run_id — should be ignored
    }
    card_other = _make_card("other-topic", run_id="")

    result = reorder_by_mastery([card_other, card_recursion], repo=repo)
    # recursion (weak) should come first despite no run_id
    assert result[0]["note_slug"] == "recursion"
    assert result[1]["note_slug"] == "other-topic"


# ---------------------------------------------------------------------------
# Test 3 — empty cards list
# ---------------------------------------------------------------------------

def test_empty_cards_returns_empty():
    """reorder_by_mastery([], repo=...) returns []."""
    repo = FakeRepo(records=[_make_confidence_record("concept:cs:recursion", 0.2)])
    result = reorder_by_mastery([], repo=repo)
    assert result == []


# ---------------------------------------------------------------------------
# Test 4 — unmappable cards do not crash
# ---------------------------------------------------------------------------

def test_unmappable_cards_do_not_crash():
    """Cards that match no known concept slug are treated as neutral (non-weak tier)."""
    records = [_make_confidence_record("concept:cs:recursion", 0.2)]
    repo = FakeRepo(records)

    card_unknown = _make_card("totally-unrelated-topic")  # matches no concept slug
    card_recursion = _make_card("recursion")

    # Must not raise; card_recursion (weak) should be first, unknown card second
    result = reorder_by_mastery([card_unknown, card_recursion], repo=repo)
    assert len(result) == 2
    assert result[0]["note_slug"] == "recursion"
    assert result[1]["note_slug"] == "totally-unrelated-topic"


# ---------------------------------------------------------------------------
# Test 5 — Phase 16 signature stability
# ---------------------------------------------------------------------------

def test_signature_is_keyword_only_repo():
    """reorder_by_mastery must be callable with keyword-only repo param."""
    repo = FakeRepo()
    cards = [_make_card("recursion")]
    # Must not raise TypeError
    result = reorder_by_mastery(cards, repo=repo)
    assert isinstance(result, list)
    assert len(result) == 1


def test_return_type_is_list_of_dicts():
    """Return value must be list[dict] — no Anki-specific types."""
    repo = FakeRepo()
    cards = [_make_card("topic-a"), _make_card("topic-b")]
    result = reorder_by_mastery(cards, repo=repo)
    assert isinstance(result, list)
    for item in result:
        assert isinstance(item, dict)


def test_card_content_not_mutated():
    """The adapter reorders only; it must not modify card dicts."""
    records = [_make_confidence_record("concept:cs:recursion", 0.2)]
    repo = FakeRepo(records)

    original_front = "My front text"
    card = _make_card("recursion")
    card["front"] = original_front

    result = reorder_by_mastery([card], repo=repo)
    assert result[0]["front"] == original_front


# ---------------------------------------------------------------------------
# Test 6 — Phase 16 slug extraction from concept_id
# ---------------------------------------------------------------------------

def test_slug_extracted_from_full_concept_id():
    """concept_id 'concept:cs:recursion' yields slug 'recursion' for card matching."""
    records = [
        _make_confidence_record("concept:cs:recursion", 0.2),  # weak
    ]
    repo = FakeRepo(records)

    card_weak = _make_card("recursion")
    card_strong = _make_card("other")

    result = reorder_by_mastery([card_strong, card_weak], repo=repo)
    assert result[0]["note_slug"] == "recursion"


def test_long_concept_id_slug_extracted_from_last_segment():
    """concept_id 'concept:organometallic-catalysis:oxidative-addition' → slug 'oxidative-addition'."""
    records = [
        _make_confidence_record("concept:organometallic-catalysis:oxidative-addition", 0.1),
    ]
    repo = FakeRepo(records)

    card_oa = _make_card("oxidative-addition")
    card_other = _make_card("something-else")

    result = reorder_by_mastery([card_other, card_oa], repo=repo)
    assert result[0]["note_slug"] == "oxidative-addition"


def test_two_segment_concept_id_is_skipped():
    """concept_id 'concept:cs' (only 2 segments) is skipped — len(parts) < 3 guard.

    Since this record is skipped, no weak slugs are found and cards pass through unchanged.
    """
    records = [
        _make_confidence_record("concept:cs", 0.1),  # 2 segments only — should be skipped
    ]
    repo = FakeRepo(records)

    card_a = _make_card("cs")
    card_b = _make_card("other")

    # With the 2-segment concept_id skipped, no weak slugs → passthrough order
    result = reorder_by_mastery([card_a, card_b], repo=repo)
    assert [c["note_slug"] for c in result] == ["cs", "other"]


# ---------------------------------------------------------------------------
# Test 7 — lesson_run_id is accepted but ignored
# ---------------------------------------------------------------------------

def test_lesson_run_id_ignored_in_phase_16():
    """lesson_run_id parameter is accepted but does NOT affect Phase 16 behaviour.

    The adapter should return weak-concept cards first regardless of run_id value
    on the cards, because concept_confidence is concept-global (not run-scoped).
    """
    records = [
        _make_confidence_record("concept:cs:recursion", 0.2),
    ]
    repo = FakeRepo(records)

    # Cards have explicit run_id values but Phase 16 ignores them
    card_a = _make_card("recursion", run_id="any-run-id")
    card_b = _make_card("other-topic", run_id="any-run-id")

    result = reorder_by_mastery([card_b, card_a], repo=repo)
    assert result[0]["note_slug"] == "recursion"


# ---------------------------------------------------------------------------
# Test 8 — THRESHOLD_FULL boundary
# ---------------------------------------------------------------------------

def test_full_confidence_card_not_treated_as_weak():
    """A concept with confidence_score >= THRESHOLD_FULL (0.7) is NOT weak → not surfaced first."""
    records = [
        _make_confidence_record("concept:cs:recursion", 0.8),   # above THRESHOLD_FULL → not weak
    ]
    repo = FakeRepo(records)

    card_recursion = _make_card("recursion")
    card_other = _make_card("other-topic")

    # Since recursion is at full confidence, no weak slugs → passthrough order
    result = reorder_by_mastery([card_recursion, card_other], repo=repo)
    assert [c["note_slug"] for c in result] == ["recursion", "other-topic"]


def test_score_exactly_at_threshold_full_is_not_weak():
    """confidence_score == THRESHOLD_FULL (0.7) is 'not full' boundary check.

    The condition is score < THRESHOLD_FULL — exactly 0.7 is NOT < 0.7, so it
    is treated as full (not weak). Card passes through unchanged.
    """
    records = [
        _make_confidence_record("concept:cs:recursion", THRESHOLD_FULL),  # exactly 0.7
    ]
    repo = FakeRepo(records)

    card_recursion = _make_card("recursion")
    card_other = _make_card("other-topic")

    # At exactly threshold, concept is NOT weak → passthrough
    result = reorder_by_mastery([card_recursion, card_other], repo=repo)
    assert [c["note_slug"] for c in result] == ["recursion", "other-topic"]


def test_score_just_below_threshold_full_is_weak():
    """confidence_score just below THRESHOLD_FULL (e.g. 0.69) IS weak."""
    records = [
        _make_confidence_record("concept:cs:recursion", 0.69),
    ]
    repo = FakeRepo(records)

    card_recursion = _make_card("recursion")
    card_other = _make_card("other-topic")

    result = reorder_by_mastery([card_other, card_recursion], repo=repo)
    assert result[0]["note_slug"] == "recursion"
