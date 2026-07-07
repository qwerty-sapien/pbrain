# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F
"""Tests for pb/vault/concept_note.py — D-16-07 schema, D-16-03 QC flag, D-16-09 aliases field.

TDD RED phase: all tests written before implementation.
"""
from __future__ import annotations

import pytest

# ---------------------------------------------------------------------------
# Smoke: module importable (will fail RED before concept_note.py exists)
# ---------------------------------------------------------------------------

def test_module_importable():
    from pb.vault.concept_note import (
        render_concept_frontmatter,
        concept_note_path,
        check_body_length,
        write_concept_note,
    )
    assert callable(render_concept_frontmatter)
    assert callable(concept_note_path)
    assert callable(check_body_length)
    assert callable(write_concept_note)


# ---------------------------------------------------------------------------
# concept_note_path
# ---------------------------------------------------------------------------

def test_concept_note_path_basic():
    from pb.vault.concept_note import concept_note_path
    result = concept_note_path("organometallic-catalysis", "oxidative-addition")
    assert result == "knowledge/organometallic-catalysis/oxidative-addition.md"


def test_concept_note_path_cs():
    from pb.vault.concept_note import concept_note_path
    assert concept_note_path("cs", "recursion") == "knowledge/cs/recursion.md"


# ---------------------------------------------------------------------------
# check_body_length
# ---------------------------------------------------------------------------

def test_body_length_under_threshold():
    from pb.vault.concept_note import check_body_length
    assert check_body_length("x" * 999) is False


def test_body_length_exactly_threshold():
    """Exactly 1000 chars is NOT a QC candidate — only strictly greater than."""
    from pb.vault.concept_note import check_body_length
    assert check_body_length("x" * 1000) is False


def test_body_length_over_threshold():
    from pb.vault.concept_note import check_body_length
    assert check_body_length("x" * 1001) is True


def test_body_length_well_over():
    from pb.vault.concept_note import check_body_length
    assert check_body_length("x" * 2000) is True


# ---------------------------------------------------------------------------
# render_concept_frontmatter — basic D-16-07 schema fields
# ---------------------------------------------------------------------------

def test_render_starts_and_ends_with_separator():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    lines = fm.split("\n")
    assert lines[0] == "---"
    assert lines[-1] == "---"


def test_render_contains_type_concept():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "type: concept" in fm


def test_render_contains_concept_id():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "id: concept:organometallic-catalysis:oxidative-addition" in fm


def test_render_contains_slug():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "slug: oxidative-addition" in fm


def test_render_contains_title():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "title: Oxidative addition" in fm


def test_render_contains_domain():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "domain: organometallic-catalysis" in fm


# ---------------------------------------------------------------------------
# render_concept_frontmatter — learning.* section (no confidence_record)
# ---------------------------------------------------------------------------

def test_render_learning_section_present():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "learning:" in fm


def test_render_confidence_none_when_no_record():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "confidence: none" in fm


def test_render_confidence_score_zero_when_no_record():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "confidence_score: 0.0" in fm


def test_render_card_weight_one_when_no_record():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "card_weight: 1.0" in fm


# ---------------------------------------------------------------------------
# render_concept_frontmatter — with a confidence_record
# ---------------------------------------------------------------------------

def test_render_learning_with_partial_record():
    """When a real ConceptConfidenceRecord is provided, learning.* reflects it."""
    from pb.vault.concept_note import render_concept_frontmatter
    from pb.core.confidence_model import ConceptConfidenceRecord

    record = ConceptConfidenceRecord(
        concept_id="concept:organometallic-catalysis:oxidative-addition",
        confidence_score=0.45,
        card_weight=0.55,
        last_evidence_at="2026-06-01T10:00:00Z",
        next_review_at="2026-06-08T10:00:00Z",
    )
    fm = render_concept_frontmatter(
        "Oxidative addition",
        "organometallic-catalysis",
        "oxidative-addition",
        confidence_record=record,
    )
    assert "confidence: partial" in fm
    assert "confidence_score: 0.45" in fm


def test_render_learning_with_full_record():
    from pb.vault.concept_note import render_concept_frontmatter
    from pb.core.confidence_model import ConceptConfidenceRecord

    record = ConceptConfidenceRecord(
        concept_id="concept:cs:recursion",
        confidence_score=0.85,
        card_weight=0.15,
    )
    fm = render_concept_frontmatter("Recursion", "cs", "recursion", confidence_record=record)
    assert "confidence: full" in fm
    assert "confidence_score: 0.85" in fm


# ---------------------------------------------------------------------------
# render_concept_frontmatter — QC flag (D-16-03)
# ---------------------------------------------------------------------------

def test_render_no_qc_status_by_default():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "qc_status" not in fm


def test_render_qc_status_candidate_when_flag_true():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter(
        "Oxidative addition", "organometallic-catalysis", "oxidative-addition",
        qc_candidate=True,
    )
    assert "qc_status: candidate" in fm


# ---------------------------------------------------------------------------
# render_concept_frontmatter — aliases[] field (D-16-09)
# ---------------------------------------------------------------------------

def test_render_aliases_empty_list_by_default():
    """D-16-09: aliases[] field must be present in schema (Phase 16 delivery)."""
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "aliases: []" in fm


def test_render_aliases_with_values():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter(
        "Oxidative addition", "organometallic-catalysis", "oxidative-addition",
        aliases=["OA", "oxidative addition step"],
    )
    assert "OA" in fm
    assert "oxidative addition step" in fm


# ---------------------------------------------------------------------------
# render_concept_frontmatter — relations block
# ---------------------------------------------------------------------------

def test_render_relations_block_present():
    from pb.vault.concept_note import render_concept_frontmatter
    fm = render_concept_frontmatter("Oxidative addition", "organometallic-catalysis", "oxidative-addition")
    assert "relations:" in fm
    for key in ("prerequisites", "prerequisite_for", "appears_in", "contrasts_with",
                "usually_precedes", "usually_follows", "examples", "failure_modes",
                "misconceptions", "related"):
        assert key in fm, f"Missing relation key: {key}"


# ---------------------------------------------------------------------------
# L6 landmine: no upsert_concept_confidence in module
# ---------------------------------------------------------------------------

def test_no_upsert_concept_confidence_in_module():
    """L6 guard: concept_note.py must NEVER call upsert_concept_confidence."""
    import inspect
    import pb.vault.concept_note as m
    source = inspect.getsource(m)
    assert "upsert_concept_confidence" not in source, (
        "L6 VIOLATION: concept_note.py must not call or reference upsert_concept_confidence"
    )


# ---------------------------------------------------------------------------
# D-16-02: MOC note type documented in module docstring
# ---------------------------------------------------------------------------

def test_module_docstring_mentions_moc():
    import pb.vault.concept_note as m
    doc = m.__doc__ or ""
    assert "moc" in doc.lower(), "Module docstring must mention MOC note type (D-16-02)"


# ---------------------------------------------------------------------------
# write_concept_note integration (temp vault)
# ---------------------------------------------------------------------------

def test_write_concept_note_creates_file(tmp_path):
    from pb.vault.concept_note import write_concept_note

    class FakeRepo:
        def list_concept_confidence(self, concept_id):
            return []

    rel_path, qc_candidate = write_concept_note(
        "Oxidative addition",
        "organometallic-catalysis",
        "A metal inserts into a bond.",
        repo=FakeRepo(),
        vault_root=str(tmp_path),
    )
    assert rel_path == "knowledge/organometallic-catalysis/oxidative-addition.md"
    assert qc_candidate is False
    assert (tmp_path / rel_path).exists()


def test_write_concept_note_qc_flag_for_long_body(tmp_path):
    from pb.vault.concept_note import write_concept_note

    class FakeRepo:
        def list_concept_confidence(self, concept_id):
            return []

    body = "x" * 1001
    rel_path, qc_candidate = write_concept_note(
        "Big Concept",
        "testing",
        body,
        repo=FakeRepo(),
        vault_root=str(tmp_path),
    )
    assert qc_candidate is True


def test_write_concept_note_file_content_has_correct_frontmatter(tmp_path):
    from pb.vault.concept_note import write_concept_note

    class FakeRepo:
        def list_concept_confidence(self, concept_id):
            return []

    rel_path, _ = write_concept_note(
        "Recursion",
        "cs",
        "A function that calls itself.",
        repo=FakeRepo(),
        vault_root=str(tmp_path),
    )
    content = (tmp_path / rel_path).read_text()
    assert "type: concept" in content
    assert "id: concept:cs:recursion" in content
    assert "confidence: none" in content
    assert "aliases: []" in content


def test_write_concept_note_no_confidence_write(tmp_path):
    """Verify write_concept_note does NOT call upsert on the repo (D-16-13)."""
    from pb.vault.concept_note import write_concept_note

    upsert_calls = []

    class StrictRepo:
        def list_concept_confidence(self, concept_id):
            return []
        def upsert_concept_confidence(self, *args, **kwargs):
            upsert_calls.append((args, kwargs))
            raise AssertionError("upsert_concept_confidence must NOT be called during note creation")

    write_concept_note(
        "Reductive elimination",
        "organometallic-catalysis",
        "Reductive elimination reverses oxidative addition.",
        repo=StrictRepo(),
        vault_root=str(tmp_path),
    )
    assert upsert_calls == [], "write_concept_note must never call upsert_concept_confidence"
