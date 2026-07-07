# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

import zipfile
from pathlib import Path

from pb.core.context_file_intake import (
    ContextFileIngestResult,
    answer_source_scope_preamble,
    compatibility_message,
    inspect_context_files,
    plan_context_file_response,
    scope_clarification_message,
)


def _base_payload() -> dict:
    return {
        "current_provider": "gemini",
        "current_model": "gemini-3-flash-preview",
        "dryrun": False,
        "status": "ok",
        "scope_mode": "corpus_first",
        "source_utility": "course_notes",
        "parsed_files": [
            {
                "filename": "lecture-1.md",
                "extension": "md",
                "mime_type": "text/markdown",
                "size_mb": 0.2,
                "canonical_class": "text.markup",
                "normalized_as": "markdown",
                "content_summary": "Lecture notes on limits and continuity.",
                "source_ref": "vault://uploads/lecture-1.md",
                "parse_confidence": "high",
            }
        ],
        "failed_files": [],
        "domain_resolution": {
            "status": "created_new",
            "domain_id": "dom_math_analysis",
            "domain_name": "Real analysis",
            "domain_granularity": "broad",
            "matched_existing_domains": [],
            "new_domain_name": "Real analysis",
            "new_domain_basis": "course_material",
            "source_bundle_id": "bundle_analysis_01",
            "source_bundle_name": "Week 1 lecture notes",
            "scope_boundary": "Limits and continuity from the uploaded lecture notes.",
            "requires_user_confirmation": False,
        },
        "scope_clarification": {
            "needed": False,
            "reason": "none",
            "suggested_question": None,
            "allowed_answers": [],
        },
        "recommended_targets": [
            {
                "provider": "openai",
                "model": "gpt-5",
                "support_mode": "native_text_extract",
                "supported_basis": "runtime_probe",
                "documented_support": True,
                "max_file_size_mb": 20,
                "conversion_required": None,
                "confidence": "high",
                "reason": "Verified support for the uploaded file class.",
            }
        ],
        "fallback_conversions": [
            {
                "from_class": "document.pdf",
                "to_format": "plain_text",
                "reason": "Plain text avoids parser limitations.",
            }
        ],
    }


def test_failed_status_returns_compatibility_only_message() -> None:
    payload = _base_payload()
    payload["status"] = "failed"
    payload["parsed_files"] = []
    payload["failed_files"] = [
        {
            "filename": "deck.pptx",
            "extension": "pptx",
            "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "size_mb": 12.5,
            "canonical_class": "document.office.presentation",
            "failure_stage": "model_could_not_read",
            "failure_reason_user_safe": "This presentation format is not readable in this run.",
        }
    ]
    result = ContextFileIngestResult.model_validate(payload)

    plan = plan_context_file_response(result)

    assert plan.action == "compatibility_only"
    assert plan.can_answer is False
    assert "I could not use one or more uploaded files with gemini gemini-3-flash-preview." in plan.user_message
    assert "* deck.pptx: This presentation format is not readable in this run." in plan.user_message
    assert "Use openai gpt-5." in plan.user_message
    assert "Convert document.pdf to plain_text." in plan.user_message


def test_ok_with_scope_clarification_asks_single_question() -> None:
    payload = _base_payload()
    payload["source_utility"] = "textbook"
    payload["scope_clarification"] = {
        "needed": True,
        "reason": "textbook_too_broad",
        "suggested_question": "Which chapters or topics from this textbook should define the learning scope?",
        "allowed_answers": ["use_chapters_or_sections", "use_topics_only", "use_entire_file"],
    }
    result = ContextFileIngestResult.model_validate(payload)

    plan = plan_context_file_response(result)

    assert plan.action == "ask_scope_clarification"
    assert plan.can_answer is False
    assert "durable source notes" in plan.user_message
    assert "Which chapters or topics from this textbook should define the learning scope?" in plan.user_message
    assert "* Entire file" in plan.user_message
    assert "* Reference only" in plan.user_message


def test_scope_clarification_mentions_temporary_files_in_dryrun() -> None:
    payload = _base_payload()
    payload["dryrun"] = True
    payload["scope_clarification"] = {
        "needed": True,
        "reason": "unclear_user_intent",
        "suggested_question": "Should I treat these notes as reference only or as the active study scope?",
        "allowed_answers": ["treat_as_reference_only", "use_topics_only"],
    }
    result = ContextFileIngestResult.model_validate(payload)

    message = scope_clarification_message(result)

    assert "temporarily for this dry run" in message
    assert "Should I treat these notes as reference only or as the active study scope?" in message


def test_partial_with_essential_failures_requests_retry_or_conversion() -> None:
    payload = _base_payload()
    payload["status"] = "partial"
    payload["failed_files"] = [
        {
            "filename": "worksheet.zip",
            "extension": "zip",
            "mime_type": "application/zip",
            "size_mb": 4.3,
            "canonical_class": "archive.bundle",
            "failure_stage": "unsupported_extension",
            "failure_reason_user_safe": "The archive format was not readable in this run.",
        }
    ]
    result = ContextFileIngestResult.model_validate(payload)

    plan = plan_context_file_response(result, failed_files_essential=True)

    assert plan.action == "ask_retry_or_conversion"
    assert plan.can_answer is False
    assert "may be essential to the requested scope" in plan.user_message
    assert "* worksheet.zip: The archive format was not readable in this run." in plan.user_message


def test_partial_nonessential_failures_can_answer_with_parsed_files_only() -> None:
    payload = _base_payload()
    payload["status"] = "partial"
    payload["failed_files"] = [
        {
            "filename": "scan.xyz",
            "extension": "xyz",
            "mime_type": "application/octet-stream",
            "size_mb": 2.1,
            "canonical_class": "unknown",
            "failure_stage": "unsupported_extension",
            "failure_reason_user_safe": "This file type is not supported yet.",
        }
    ]
    result = ContextFileIngestResult.model_validate(payload)

    plan = plan_context_file_response(result, failed_files_essential=False)
    preamble = answer_source_scope_preamble(result)

    assert plan.action == "answer_with_parsed_files"
    assert plan.can_answer is True
    assert plan.parsed_files_only is True
    assert "Using 1 parsed file(s) as durable source notes" in preamble
    assert "within the `Real analysis` domain" in preamble
    assert "Failed uploads are excluded from this answer." in preamble


def test_compatibility_message_prefers_documented_target() -> None:
    payload = _base_payload()
    payload["status"] = "failed"
    payload["parsed_files"] = []
    payload["failed_files"] = [
        {
            "filename": "deck.pptx",
            "extension": "pptx",
            "mime_type": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
            "size_mb": 8.8,
            "canonical_class": "document.office.presentation",
            "failure_stage": "model_could_not_read",
            "failure_reason_user_safe": "The current model could not read this presentation format.",
        }
    ]
    payload["recommended_targets"] = [
        {
            "provider": "example",
            "model": "unknown-model",
            "support_mode": "tool_required",
            "supported_basis": "canonical_class",
            "documented_support": False,
            "max_file_size_mb": None,
            "conversion_required": "Convert to PDF first",
            "confidence": "medium",
            "reason": "Unverified guess.",
        },
        {
            "provider": "openai",
            "model": "gpt-5",
            "support_mode": "native_multimodal",
            "supported_basis": "runtime_probe",
            "documented_support": True,
            "max_file_size_mb": 25,
            "conversion_required": None,
            "confidence": "high",
            "reason": "Documented support for this presentation workflow.",
        },
    ]
    result = ContextFileIngestResult.model_validate(payload)

    message = compatibility_message(result)

    assert "Use openai gpt-5. Reason: Documented support for this presentation workflow." in message


def test_inspect_context_files_accepts_pdf_and_marks_searchable(tmp_path: Path) -> None:
    pdf_path = tmp_path / "source.pdf"
    pdf_path.write_bytes(b"%PDF-1.4\n1 0 obj << /Type /Catalog >> endobj\n%%EOF\n")

    result = inspect_context_files(
        [pdf_path],
        provider="gemini",
        model="gemini-3-flash-preview",
        dryrun=True,
    )

    assert result.status == "ok"
    assert result.parsed_files
    assert result.parsed_files[0].canonical_class == "document.pdf"
    assert result.parsed_files[0].is_searchable is True
    payload = result.model_dump(mode="json")
    parsed = payload["parsed_files"][0]
    assert parsed["isSearchable"] is True
    assert "is_searchable" not in parsed


def test_non_pdf_serialization_omits_searchable_metadata(tmp_path: Path) -> None:
    note_path = tmp_path / "note.md"
    note_path.write_text("# Note\nActive recall.\n", encoding="utf-8")

    result = inspect_context_files(
        [note_path],
        provider="gemini",
        model="gemini-3-flash-preview",
        dryrun=True,
    )

    parsed = result.model_dump(mode="json")["parsed_files"][0]
    assert "isSearchable" not in parsed
    assert "is_searchable" not in parsed


def test_archive_inspection_skips_junk_and_extracts_safe_children(tmp_path: Path) -> None:
    archive_path = tmp_path / "course-pack.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("__MACOSX/notes.md", "# junk")
        archive.writestr("week1/lecture_notes.md", "# Lecture\nStokes theorem review")

    result = inspect_context_files(
        [archive_path],
        provider="gemini",
        model="gemini-3-flash-preview",
        dryrun=True,
    )

    assert result.status == "ok"
    assert any("lecture_notes.md" in item.filename for item in result.parsed_files)
    assert all("__MACOSX" not in item.filename for item in result.parsed_files)


def test_domain_resolution_maps_transition_metal_worksheet_to_broad_domain(tmp_path: Path) -> None:
    worksheet_path = tmp_path / "transition_metal_catalysis_worksheet.md"
    worksheet_path.write_text("# Transition metal catalysis\nLigand field review.\n", encoding="utf-8")

    result = inspect_context_files(
        [worksheet_path],
        provider="gemini",
        model="gemini-3-flash-preview",
        dryrun=True,
    )

    assert result.domain_resolution.new_domain_name == "inorganic chemistry"
    assert result.scope_mode == "syllabus_only"
