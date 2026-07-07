from __future__ import annotations

from pb.core.encoding_contract import (
    EncodingSuggestion,
    encoding_status,
    record_rejection,
    validate_encoding_suggestion,
    was_rejected,
)


def test_encoding_status_missing_fields_is_incomplete_not_failure():
    status = encoding_status({"type": "concept", "title": "Closure"})

    assert status["complete"] is False
    assert "cue" in status["missing"]


def test_encoding_validation_rejects_generic_suggestion():
    suggestion = EncodingSuggestion(
        target_concept="concept:cs:closure",
        facet="future_use",
        value="use this later",
        source_provenance="session:abc",
        evidence_text="The learner said this might be useful later.",
    )

    result = validate_encoding_suggestion(suggestion, title="Closure", body="A closure captures lexical scope.")

    assert result.ok is False
    assert "generic" in result.reason


def test_encoding_validation_rejects_redundant_body_text():
    suggestion = EncodingSuggestion(
        target_concept="concept:cs:closure",
        facet="failure_mode",
        value="captures lexical scope",
        source_provenance="session:abc",
        evidence_text="The user confused captures lexical scope with global state.",
    )

    result = validate_encoding_suggestion(
        suggestion,
        title="Closure",
        body="A closure captures lexical scope.",
    )

    assert result.ok is False
    assert "body" in result.reason


def test_relation_like_encoding_suggestion_is_converted_to_graph_edge():
    suggestion = EncodingSuggestion(
        target_concept="concept:cs:closure",
        facet="prior_connection",
        value="Lexical scope",
        source_provenance="session:abc",
        evidence_text="The learner linked closures to lexical scope.",
    )

    result = validate_encoding_suggestion(suggestion)

    assert result.ok is False
    assert result.convert_to_relation is True


def test_rejected_suggestion_is_not_retried_without_new_evidence(repo):
    suggestion = EncodingSuggestion(
        target_concept="concept:cs:closure",
        facet="example",
        value="Callback remembers stale state",
        source_provenance="session:abc",
        evidence_text="In session abc, callback remembered stale state.",
    )

    record_rejection(suggestion, "not specific enough")

    assert was_rejected(suggestion) is True
    changed = EncodingSuggestion(
        target_concept=suggestion.target_concept,
        facet=suggestion.facet,
        value=suggestion.value,
        source_provenance=suggestion.source_provenance,
        evidence_text="New session evidence changed the basis.",
    )
    assert was_rejected(changed) is False
