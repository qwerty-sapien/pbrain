# SPDX-License-Identifier: AGPL-3.0-or-later
"""Tests for pb.core.confidence_model (Phase 16, D-16-17/D-16-18/D-16-27)."""
from __future__ import annotations

import pytest
from pb.core.confidence_model import (
    BURST_N,
    DELTA_DIAGNOSTIC_CORRECT,
    DELTA_PRACTICE_CORRECT,
    DELTA_TEACH_FULL_COVERAGE,
    DELTA_WRONG,
    THRESHOLD_FULL,
    THRESHOLD_NONE,
    ConceptConfidenceRecord,
    clamp_score,
    confidence_label,
)


class TestConfidenceLabel:
    def test_zero_is_none(self):
        assert confidence_label(0.0) == "none"

    def test_just_below_threshold_none_is_none(self):
        assert confidence_label(0.29) == "none"

    def test_at_threshold_none_is_partial(self):
        assert confidence_label(0.3) == "partial"

    def test_mid_range_is_partial(self):
        assert confidence_label(0.5) == "partial"

    def test_at_threshold_full_is_partial(self):
        assert confidence_label(0.7) == "partial"

    def test_just_above_threshold_full_is_full(self):
        assert confidence_label(0.71) == "full"

    def test_one_is_full(self):
        assert confidence_label(1.0) == "full"


class TestClampScore:
    def test_below_zero_clamps_to_zero(self):
        assert clamp_score(-0.1) == 0.0

    def test_above_one_clamps_to_one(self):
        assert clamp_score(1.1) == 1.0

    def test_mid_value_unchanged(self):
        assert clamp_score(0.5) == 0.5

    def test_exact_zero_unchanged(self):
        assert clamp_score(0.0) == 0.0

    def test_exact_one_unchanged(self):
        assert clamp_score(1.0) == 1.0


class TestConceptConfidenceRecord:
    def test_can_instantiate_with_concept_id_only(self):
        r = ConceptConfidenceRecord(concept_id="concept:cs:recursion")
        assert r.concept_id == "concept:cs:recursion"

    def test_has_all_nine_fields(self):
        r = ConceptConfidenceRecord(
            concept_id="concept:cs:recursion",
            confidence_score=0.5,
            card_weight=0.5,
            next_review_at="2026-07-01T00:00:00",
            last_evidence_at="2026-06-22T00:00:00",
            created_at="2026-06-01T00:00:00",
            updated_at="2026-06-22T00:00:00",
            burst_active=1,
            burst_streak=2,
        )
        assert r.concept_id == "concept:cs:recursion"
        assert r.confidence_score == 0.5
        assert r.card_weight == 0.5
        assert r.next_review_at == "2026-07-01T00:00:00"
        assert r.last_evidence_at == "2026-06-22T00:00:00"
        assert r.created_at == "2026-06-01T00:00:00"
        assert r.updated_at == "2026-06-22T00:00:00"
        assert r.burst_active == 1
        assert r.burst_streak == 2

    def test_defaults_are_sane(self):
        r = ConceptConfidenceRecord(concept_id="concept:x:y")
        assert r.confidence_score == 0.0
        assert r.card_weight == 1.0
        assert r.burst_active == 0
        assert r.burst_streak == 0


class TestConstants:
    def test_delta_diagnostic_correct(self):
        assert DELTA_DIAGNOSTIC_CORRECT == 0.08

    def test_delta_teach_full_coverage(self):
        assert DELTA_TEACH_FULL_COVERAGE == 0.15

    def test_delta_practice_correct(self):
        assert DELTA_PRACTICE_CORRECT == 0.10

    def test_delta_wrong(self):
        assert DELTA_WRONG == -0.05

    def test_burst_n(self):
        assert BURST_N == 3

    def test_threshold_none(self):
        assert THRESHOLD_NONE == 0.3

    def test_threshold_full(self):
        assert THRESHOLD_FULL == 0.7
