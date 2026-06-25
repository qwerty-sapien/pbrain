# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Typed handling for uploaded context-file intake results.

This module turns the uploaded-file intake contract into deterministic behavior:
validate the runtime payload, decide whether the assistant should answer,
ask for scope clarification, or stop with a compatibility message, and render
the user-facing helper messages for those cases.
"""

from __future__ import annotations

import mimetypes
import re
import zipfile
from datetime import datetime
from pathlib import Path, PurePosixPath
from dataclasses import dataclass
from typing import Iterable, Literal

from pydantic import BaseModel, Field

from pb.core.models import generate_internal_id
from pb.core.scope_resolution import list_knowledge_domains, match_domain_name


ActionType = Literal[
    "compatibility_only",
    "ask_scope_clarification",
    "answer_with_parsed_files",
    "ask_retry_or_conversion",
]


class ParsedContextFile(BaseModel):
    filename: str
    extension: str
    mime_type: str
    size_mb: float
    canonical_class: Literal[
        "image.raster",
        "document.pdf",
        "document.office.word",
        "document.office.presentation",
        "document.office.spreadsheet",
        "text.plain",
        "text.markup",
        "text.code",
        "table.delimited",
        "archive.bundle",
        "unknown",
    ]
    normalized_as: Literal[
        "image",
        "searchable_pdf",
        "plain_text",
        "csv",
        "markdown",
        "extracted_archive",
        "provider_native",
    ]
    content_summary: str
    source_ref: str
    parse_confidence: Literal["high", "medium", "low"]


class FailedContextFile(BaseModel):
    filename: str
    extension: str
    mime_type: str
    size_mb: float
    canonical_class: str
    failure_stage: Literal[
        "upload_rejected",
        "request_rejected",
        "model_could_not_read",
        "too_large",
        "unsupported_mime",
        "unsupported_extension",
        "encrypted_or_corrupt",
        "ocr_needed",
        "unknown",
    ]
    failure_reason_user_safe: str


class ExistingDomainMatch(BaseModel):
    domain_id: str
    domain_name: str
    confidence: Literal["high", "medium", "low"]
    reason: str


class DomainResolution(BaseModel):
    status: Literal[
        "matched_existing",
        "created_new",
        "proposed_new",
        "ambiguous",
        "not_applicable",
    ]
    domain_id: str | None = None
    domain_name: str | None = None
    domain_granularity: Literal["broad", "narrow", "unknown"]
    matched_existing_domains: list[ExistingDomainMatch] = Field(default_factory=list)
    new_domain_name: str | None = None
    new_domain_basis: Literal[
        "uploaded_files",
        "syllabus",
        "course_material",
        "project_material",
        "unknown",
    ]
    source_bundle_id: str | None = None
    source_bundle_name: str | None = None
    scope_boundary: str = ""
    requires_user_confirmation: bool = False


class ScopeClarification(BaseModel):
    needed: bool
    reason: Literal[
        "textbook_too_broad",
        "large_reference",
        "mixed_archive",
        "unclear_user_intent",
        "ambiguous_domain",
        "none",
    ]
    suggested_question: str | None = None
    allowed_answers: list[
        Literal[
            "use_entire_file",
            "use_page_range",
            "use_chapters_or_sections",
            "use_topics_only",
            "treat_as_reference_only",
            "extract_syllabus_scope",
        ]
    ] = Field(default_factory=list)


class RecommendedTarget(BaseModel):
    provider: str
    model: str
    support_mode: Literal[
        "native_multimodal",
        "native_text_extract",
        "native_table_extract",
        "tool_required",
        "conversion_required",
    ]
    supported_basis: Literal[
        "exact_extension",
        "exact_mime",
        "canonical_class",
        "runtime_probe",
    ]
    documented_support: bool
    max_file_size_mb: float | None = None
    conversion_required: str | None = None
    confidence: Literal["high", "medium", "low"]
    reason: str


class FallbackConversion(BaseModel):
    from_class: str
    to_format: Literal[
        "searchable_pdf",
        "plain_text",
        "csv",
        "markdown",
        "extracted_archive",
    ]
    reason: str


class FileSupportDecision(BaseModel):
    provider: str
    model: str
    endpoint: str
    delivery: str
    canonical_class: str
    exact_mimes: list[str] = Field(default_factory=list)
    exact_extensions: list[str] = Field(default_factory=list)
    max_file_size_mb: float | None = None
    documented_support: bool = False
    probe_status: Literal["pass", "fail", "unknown"] = "unknown"
    support_mode: Literal[
        "native_multimodal",
        "native_text_extract",
        "native_table_extract",
        "tool_required",
        "conversion_required",
        "unsupported",
    ]
    notes: str = ""


class SourceBundleItem(BaseModel):
    id: str = Field(default_factory=generate_internal_id)
    bundle_id: str
    source_id: str
    position: int = 0
    source_ref: str = ""
    filename: str = ""
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class SourceBundle(BaseModel):
    id: str = Field(default_factory=generate_internal_id)
    name: str
    domain_id: str | None = None
    domain_name: str | None = None
    scope_mode: Literal[
        "syllabus_only",
        "corpus_first",
        "reference_only",
        "general_allowed",
        "unclear",
    ] = "unclear"
    scope_boundary: str = ""
    source_refs: list[str] = Field(default_factory=list)
    items: list[SourceBundleItem] = Field(default_factory=list)
    created_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ActiveContextScope(BaseModel):
    mode: Literal["none", "direct_files", "bundle"] = "none"
    locked: bool = False
    label: str = ""
    label_max_chars: int = 20
    scope_mode: Literal[
        "syllabus_only",
        "corpus_first",
        "reference_only",
        "general_allowed",
        "unclear",
    ] = "unclear"
    source_bundle_id: str | None = None
    source_refs: list[str] = Field(default_factory=list)
    domain_id: str | None = None
    scope_boundary: str = ""


class ContextFileIngestResult(BaseModel):
    current_provider: str
    current_model: str
    dryrun: bool
    status: Literal["ok", "partial", "failed"]
    scope_mode: Literal[
        "syllabus_only",
        "corpus_first",
        "reference_only",
        "general_allowed",
        "unclear",
    ]
    source_utility: Literal[
        "syllabus",
        "course_notes",
        "worksheet",
        "slides",
        "textbook",
        "reference_manual",
        "project_context",
        "mixed_archive",
        "unknown",
    ]
    parsed_files: list[ParsedContextFile] = Field(default_factory=list)
    failed_files: list[FailedContextFile] = Field(default_factory=list)
    domain_resolution: DomainResolution
    scope_clarification: ScopeClarification
    recommended_targets: list[RecommendedTarget] = Field(default_factory=list)
    fallback_conversions: list[FallbackConversion] = Field(default_factory=list)


@dataclass(frozen=True)
class ContextFileResponsePlan:
    action: ActionType
    persistence_label: str
    can_answer: bool
    parsed_files_only: bool
    user_message: str = ""


_IMAGE_EXTENSIONS = {"png", "jpg", "jpeg", "webp"}
_WORD_EXTENSIONS = {"docx", "rtf"}
_PRESENTATION_EXTENSIONS = {"pptx"}
_SPREADSHEET_EXTENSIONS = {"xlsx"}
_TEXT_EXTENSIONS = {"txt"}
_MARKUP_EXTENSIONS = {"md", "markdown", "html"}
_CODE_EXTENSIONS = {
    "py",
    "js",
    "ts",
    "tsx",
    "jsx",
    "rs",
    "go",
    "java",
    "c",
    "cc",
    "cpp",
    "h",
    "hpp",
    "rb",
    "php",
    "swift",
    "kt",
    "scala",
    "sh",
    "bash",
    "zsh",
    "toml",
    "ini",
    "cfg",
    "conf",
    "env",
}
_STRUCTURED_TEXT_EXTENSIONS = {"json", "yaml", "yml", "sql"}
_TABLE_EXTENSIONS = {"csv", "tsv"}
_ARCHIVE_EXTENSIONS = {"zip"}
_JUNK_PATH_PARTS = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".cache",
    "tmp",
    "temp",
    "dist",
    "build",
    "target",
    "coverage",
    "__MACOSX",
}
_JUNK_SUFFIXES = {
    ".ds_store",
    ".pyc",
    ".pyo",
    ".class",
    ".o",
    ".so",
    ".dll",
    ".dylib",
    ".exe",
}
_STRICT_SCOPE_UTILITIES = {"syllabus", "course_notes", "worksheet", "slides"}
_DOMAIN_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("vector calculus", ("vector calculus", "stokes", "divergence theorem", "curl", "manifold")),
    ("linear algebra", ("linear algebra", "eigenvalue", "matrix", "vector space", "determinant")),
    ("inorganic chemistry", ("inorganic", "transition metal", "coordination", "catalysis", "ligand")),
    ("organic chemistry", ("organic chemistry", "stereochemistry", "alkene", "carbonyl", "mechanism")),
    ("backend web development", ("backend", "http", "rest api", "postgres", "server", "sql")),
    ("rust programming", ("rust", "cargo", "borrow checker", "ownership", "lifetimes")),
    ("spatial transcriptomics", ("spatial transcriptomics", "rna-seq", "visium", "single-cell", "transcriptomics")),
]


def canonical_class_from_path(path: Path, *, mime_type: str = "") -> str:
    """Return the canonical class for one file path."""

    extension = path.suffix.lower().lstrip(".")
    lowered_mime = (mime_type or "").lower()
    if lowered_mime.startswith("image/") or extension in _IMAGE_EXTENSIONS:
        return "image.raster"
    if lowered_mime == "application/pdf" or extension == "pdf":
        return "document.pdf"
    if extension in _WORD_EXTENSIONS:
        return "document.office.word"
    if extension in _PRESENTATION_EXTENSIONS:
        return "document.office.presentation"
    if extension in _SPREADSHEET_EXTENSIONS:
        return "document.office.spreadsheet"
    if extension in _TABLE_EXTENSIONS or lowered_mime in {"text/csv", "text/tab-separated-values"}:
        return "table.delimited"
    if extension in _TEXT_EXTENSIONS:
        return "text.plain"
    if extension in _MARKUP_EXTENSIONS:
        return "text.markup"
    if extension in _CODE_EXTENSIONS or extension in _STRUCTURED_TEXT_EXTENSIONS:
        return "text.code"
    if extension in _ARCHIVE_EXTENSIONS or lowered_mime == "application/zip":
        return "archive.bundle"
    return "unknown"


def normalized_target_for_class(canonical_class: str) -> str:
    """Return the normalized storage/transport target for a canonical class."""

    mapping = {
        "image.raster": "image",
        "document.pdf": "searchable_pdf",
        "document.office.word": "searchable_pdf",
        "document.office.presentation": "searchable_pdf",
        "document.office.spreadsheet": "csv",
        "text.plain": "plain_text",
        "text.markup": "markdown",
        "text.code": "plain_text",
        "table.delimited": "csv",
        "archive.bundle": "extracted_archive",
    }
    return mapping.get(canonical_class, "provider_native")


def default_scope_mode_for_utility(source_utility: str) -> Literal[
    "syllabus_only",
    "corpus_first",
    "reference_only",
    "general_allowed",
    "unclear",
]:
    """Map source utility to the default scope mode."""

    utility = (source_utility or "").strip().lower()
    if utility in _STRICT_SCOPE_UTILITIES:
        return "syllabus_only"
    if utility == "reference_manual":
        return "reference_only"
    if utility in {"project_context", "mixed_archive"}:
        return "general_allowed"
    if utility == "textbook":
        return "unclear"
    return "corpus_first"


def summarize_context_label(scope: ActiveContextScope | None, *, include_markers: bool = True) -> str:
    """Return the user-facing 20-char max context label for prompts."""

    if scope is None or scope.mode == "none":
        return ""
    label = str(scope.label or "").strip() or "context"
    marker_chunks: list[str] = []
    if include_markers and scope.locked:
        marker_chunks.append("🔒")
    if include_markers and scope.mode == "bundle":
        marker_chunks.append("🧺")
    prefix = " ".join(marker_chunks).strip()
    available = max(1, int(scope.label_max_chars or 20) - (len(prefix) + (1 if prefix else 0)))
    clipped = label[:available].rstrip()
    if prefix:
        return f"{prefix} {clipped}".strip()
    return clipped


def active_context_from_bundle(bundle: SourceBundle, *, locked: bool) -> ActiveContextScope:
    """Construct the runtime context scope object for one stored bundle."""

    return ActiveContextScope(
        mode="bundle",
        locked=locked,
        label=bundle.name,
        scope_mode=bundle.scope_mode,
        source_bundle_id=bundle.id,
        source_refs=list(bundle.source_refs),
        domain_id=bundle.domain_id,
        scope_boundary=bundle.scope_boundary,
    )


def active_context_from_sources(
    source_refs: Iterable[str],
    *,
    label: str,
    domain_id: str | None = None,
    scope_mode: str = "unclear",
    scope_boundary: str = "",
    locked: bool = False,
) -> ActiveContextScope:
    """Construct the runtime context scope for direct file sources."""

    return ActiveContextScope(
        mode="direct_files",
        locked=locked,
        label=label,
        scope_mode=scope_mode,
        source_refs=[str(item).strip() for item in source_refs if str(item).strip()],
        domain_id=domain_id,
        scope_boundary=scope_boundary,
    )


def persistence_label(result: ContextFileIngestResult) -> str:
    """Describe whether uploaded files should be treated as durable or temporary."""

    return "temporary files for this dry run" if result.dryrun else "durable source notes"


def plan_context_file_response(
    result: ContextFileIngestResult,
    *,
    failed_files_essential: bool = False,
) -> ContextFileResponsePlan:
    """Map one intake result to the required response behavior."""

    persistence = persistence_label(result)
    if result.status == "failed":
        return ContextFileResponsePlan(
            action="compatibility_only",
            persistence_label=persistence,
            can_answer=False,
            parsed_files_only=False,
            user_message=compatibility_message(result),
        )

    if result.status == "ok" and result.scope_clarification.needed:
        return ContextFileResponsePlan(
            action="ask_scope_clarification",
            persistence_label=persistence,
            can_answer=False,
            parsed_files_only=False,
            user_message=scope_clarification_message(result),
        )

    if result.status == "partial" and failed_files_essential:
        return ContextFileResponsePlan(
            action="ask_retry_or_conversion",
            persistence_label=persistence,
            can_answer=False,
            parsed_files_only=True,
            user_message=retry_or_conversion_message(result),
        )

    return ContextFileResponsePlan(
        action="answer_with_parsed_files",
        persistence_label=persistence,
        can_answer=True,
        parsed_files_only=result.status == "partial",
        user_message="",
    )


def compatibility_message(result: ContextFileIngestResult) -> str:
    """Render the compatibility-only message for status=failed."""

    affected_lines = _failed_file_lines(result.failed_files) or ["* Unknown file: unsupported by the current runtime"]
    recommended = _best_recommended_target(result)
    fallback = _best_fallback_conversion(result)
    if recommended is None:
        recommended_option = "Use a provider and model with verified support for these file types."
    else:
        recommended_option = f"Use {recommended.provider} {recommended.model}. Reason: {recommended.reason}"
    return (
        f"I could not use one or more uploaded files with {result.current_provider} {result.current_model}.\n\n"
        "Affected file(s):\n\n"
        f"{chr(10).join(affected_lines)}\n\n"
        "Recommended option:\n"
        f"{recommended_option}\n\n"
        "Fallback:\n"
        f"{fallback}"
    )


def scope_clarification_message(result: ContextFileIngestResult) -> str:
    """Render the single scope-clarification question required by the contract."""

    opener = (
        "I can use these files temporarily for this dry run, but their intended learning scope is unclear."
        if result.dryrun
        else "I can use these files as durable source notes, but their intended learning scope is unclear."
    )
    question = result.scope_clarification.suggested_question or "Which part of these files should define the learning scope?"
    return (
        f"{opener}\n\n"
        "Question:\n"
        f"{question}\n\n"
        "Useful answers:\n\n"
        "* Page range\n"
        "* Chapters or sections\n"
        "* Topic list\n"
        "* Entire file\n"
        "* Reference only"
    )


def retry_or_conversion_message(result: ContextFileIngestResult) -> str:
    """Explain that failed files may be essential and request a retry or conversion."""

    failed_lines = _failed_file_lines(result.failed_files) or ["* Unknown file: retry the upload or convert it to a supported format"]
    fallback = _best_fallback_conversion(result)
    scope_note = (
        "I can proceed with the successfully parsed files, but the failed upload(s) may be essential to the requested scope."
    )
    return (
        f"{scope_note}\n\n"
        "Failed file(s):\n\n"
        f"{chr(10).join(failed_lines)}\n\n"
        "Recommended next step:\n"
        "Retry those files or convert them, then run the request again.\n\n"
        "Fallback:\n"
        f"{fallback}"
    )


def answer_source_scope_preamble(result: ContextFileIngestResult) -> str:
    """Return a concise preamble for substantive answers using parsed files only."""

    parsed_names = ", ".join(file.filename for file in result.parsed_files[:3])
    if len(result.parsed_files) > 3:
        parsed_names += ", ..."
    domain_name = _resolved_domain_name(result)
    scope_boundary = (result.domain_resolution.scope_boundary or "").strip()
    parts = [f"Using {len(result.parsed_files)} parsed file(s) as {persistence_label(result)}"]
    if parsed_names:
        parts.append(f"from {parsed_names}")
    if domain_name:
        parts.append(f"within the `{domain_name}` domain")
    if scope_boundary:
        parts.append(f"and the scope boundary `{scope_boundary}`")
    sentence = " ".join(parts).strip()
    if result.status == "partial" and result.failed_files:
        sentence += ". Failed uploads are excluded from this answer."
    else:
        sentence += "."
    return sentence


def _failed_file_lines(failed_files: list[FailedContextFile]) -> list[str]:
    return [f"* {item.filename}: {item.failure_reason_user_safe}" for item in failed_files]


def _best_recommended_target(
    result: ContextFileIngestResult,
) -> RecommendedTarget | None:
    if not result.recommended_targets:
        return None
    documented = [item for item in result.recommended_targets if item.documented_support]
    return (documented or result.recommended_targets)[0]


def _best_fallback_conversion(result: ContextFileIngestResult) -> str:
    if not result.fallback_conversions:
        return "No documented conversion fallback is available."
    best = result.fallback_conversions[0]
    return f"Convert {best.from_class} to {best.to_format}. Reason: {best.reason}"


def _resolved_domain_name(result: ContextFileIngestResult) -> str:
    resolution = result.domain_resolution
    if resolution.status == "matched_existing":
        return (resolution.domain_name or "").strip()
    if resolution.status == "created_new":
        return (resolution.domain_name or resolution.new_domain_name or "").strip()
    if resolution.status == "proposed_new":
        return (resolution.new_domain_name or resolution.domain_name or "").strip()
    return (resolution.domain_name or "").strip()


def sniff_mime_type(path: Path) -> str:
    """Guess MIME using magic bytes first and extension second."""

    try:
        head = path.read_bytes()[:64]
    except OSError:
        head = b""
    if head.startswith(b"%PDF-"):
        return "application/pdf"
    if head.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if head.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if head[:4] == b"RIFF" and head[8:12] == b"WEBP":
        return "image/webp"
    if head.startswith(b"PK\x03\x04"):
        return "application/zip"
    guessed, _ = mimetypes.guess_type(str(path))
    return guessed or "application/octet-stream"


def inspect_context_files(
    paths: Iterable[Path | str],
    *,
    provider: str,
    model: str,
    dryrun: bool = False,
    existing_domains: list[str] | None = None,
) -> ContextFileIngestResult:
    """Inspect one or more context files and return the public intake contract."""

    parsed_files: list[ParsedContextFile] = []
    failed_files: list[FailedContextFile] = []
    utilities: list[str] = []
    inspected_classes: list[str] = []
    path_list = [Path(item).expanduser() for item in paths]

    for path in path_list:
        utility = infer_source_utility(path)
        utilities.append(utility)
        if not path.exists() or not path.is_file():
            failed_files.append(
                FailedContextFile(
                    filename=path.name or str(path),
                    extension=path.suffix.lower().lstrip("."),
                    mime_type="application/octet-stream",
                    size_mb=0.0,
                    canonical_class="unknown",
                    failure_stage="upload_rejected",
                    failure_reason_user_safe="This file does not exist or is not a regular file.",
                )
            )
            continue
        file_result = _inspect_one_path(
            path,
            provider=provider,
            model=model,
            utilities=utilities,
        )
        inspected_classes.extend(file_result["canonical_classes"])
        parsed_files.extend(file_result["parsed"])
        failed_files.extend(file_result["failed"])

    source_utility = _dominant_source_utility(utilities)
    domain_resolution = resolve_context_domain(path_list, parsed_files, existing_domains=existing_domains)
    scope_clarification = build_scope_clarification(path_list, parsed_files, source_utility)
    decisions = capability_decisions_for_classes(provider=provider, model=model, canonical_classes=inspected_classes)
    recommended_targets, fallback_conversions = build_compatibility_recommendations(decisions)
    status: Literal["ok", "partial", "failed"]
    if parsed_files and failed_files:
        status = "partial"
    elif parsed_files:
        status = "ok"
    else:
        status = "failed"
    return ContextFileIngestResult(
        current_provider=provider,
        current_model=model,
        dryrun=dryrun,
        status=status,
        scope_mode=default_scope_mode_for_utility(source_utility),
        source_utility=source_utility,
        parsed_files=parsed_files,
        failed_files=failed_files,
        domain_resolution=domain_resolution,
        scope_clarification=scope_clarification,
        recommended_targets=recommended_targets,
        fallback_conversions=fallback_conversions,
    )


def resolve_context_domain(
    paths: Iterable[Path],
    parsed_files: Iterable[ParsedContextFile],
    *,
    existing_domains: list[str] | None = None,
) -> DomainResolution:
    """Resolve a broad learning domain from filenames and parsed summaries."""

    candidate_parts = [path.stem.replace("_", " ").replace("-", " ") for path in paths]
    candidate_parts.extend(file.content_summary for file in parsed_files if file.content_summary)
    combined = " ".join(candidate_parts).lower()

    for domain_name, keywords in _DOMAIN_KEYWORDS:
        if any(keyword in combined for keyword in keywords):
            return DomainResolution(
                status="proposed_new",
                domain_name=domain_name,
                domain_granularity="broad",
                new_domain_name=domain_name,
                new_domain_basis="uploaded_files",
                scope_boundary=_default_scope_boundary(paths),
                requires_user_confirmation=False,
            )

    available = existing_domains or list_knowledge_domains()
    if combined:
        matched_existing = match_domain_name(combined, domains=available)
        if matched_existing:
            return DomainResolution(
                status="matched_existing",
                domain_name=matched_existing,
                domain_granularity="broad",
                new_domain_basis="uploaded_files",
                scope_boundary=_default_scope_boundary(paths),
                requires_user_confirmation=False,
            )

    fallback_name = _broad_domain_from_filename(paths)
    return DomainResolution(
        status="proposed_new",
        domain_name=fallback_name,
        domain_granularity="broad",
        new_domain_name=fallback_name,
        new_domain_basis="uploaded_files",
        scope_boundary=_default_scope_boundary(paths),
        requires_user_confirmation=False,
    )


def build_scope_clarification(
    paths: Iterable[Path],
    parsed_files: Iterable[ParsedContextFile],
    source_utility: str,
) -> ScopeClarification:
    """Return the single-scope-question contract when a source is too broad."""

    utility = (source_utility or "").strip().lower()
    if utility == "textbook":
        return ScopeClarification(
            needed=True,
            reason="textbook_too_broad",
            suggested_question="Which chapters, sections, pages, or topics from this textbook should define the study scope?",
            allowed_answers=[
                "use_page_range",
                "use_chapters_or_sections",
                "use_topics_only",
                "use_entire_file",
                "treat_as_reference_only",
            ],
        )
    if utility == "reference_manual":
        return ScopeClarification(
            needed=True,
            reason="large_reference",
            suggested_question="Should I treat this manual as reference-only, or should I restrict the scope to a specific section or topic?",
            allowed_answers=[
                "treat_as_reference_only",
                "use_chapters_or_sections",
                "use_topics_only",
            ],
        )
    if utility == "mixed_archive":
        return ScopeClarification(
            needed=True,
            reason="mixed_archive",
            suggested_question="This archive contains mixed materials. Which topic, section, or subset should define the active learning scope?",
            allowed_answers=[
                "use_topics_only",
                "use_chapters_or_sections",
                "treat_as_reference_only",
                "use_entire_file",
            ],
        )
    large_pdf = any(
        file.canonical_class == "document.pdf" and file.size_mb >= 15.0
        for file in parsed_files
    )
    if large_pdf:
        return ScopeClarification(
            needed=True,
            reason="unclear_user_intent",
            suggested_question="This PDF is broad. Which page range, section, or topic should define the learning scope?",
            allowed_answers=[
                "use_page_range",
                "use_chapters_or_sections",
                "use_topics_only",
                "treat_as_reference_only",
            ],
        )
    return ScopeClarification(
        needed=False,
        reason="none",
        suggested_question=None,
        allowed_answers=[],
    )


def infer_source_utility(path: Path) -> Literal[
    "syllabus",
    "course_notes",
    "worksheet",
    "slides",
    "textbook",
    "reference_manual",
    "project_context",
    "mixed_archive",
    "unknown",
]:
    """Infer source utility from deterministic filename signals."""

    lowered = str(path.name).lower()
    if "syllabus" in lowered or "module-outline" in lowered or "module_outline" in lowered:
        return "syllabus"
    if any(token in lowered for token in ("worksheet", "problem-set", "problem_set", "exam-spec", "exam_spec")):
        return "worksheet"
    if any(token in lowered for token in ("lecture", "notes", "tutorial")):
        return "course_notes"
    if any(token in lowered for token in ("slides", "deck")):
        return "slides"
    if any(token in lowered for token in ("textbook", "chapter")):
        return "textbook"
    if any(token in lowered for token in ("manual", "reference", "handbook")):
        return "reference_manual"
    if path.suffix.lower() == ".zip":
        return "mixed_archive"
    if any(token in lowered for token in ("readme", "spec", "design", "schema", "api", "service")):
        return "project_context"
    return "unknown"


def capability_decisions_for_classes(
    *,
    provider: str,
    model: str,
    canonical_classes: Iterable[str],
) -> list[FileSupportDecision]:
    """Return deterministic support decisions for each canonical class."""

    unique_classes: list[str] = []
    seen: set[str] = set()
    for canonical_class in canonical_classes:
        clean = str(canonical_class).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        unique_classes.append(clean)

    decisions: list[FileSupportDecision] = []
    lowered_provider = (provider or "").strip().lower()
    for canonical_class in unique_classes:
        support_mode = "unsupported"
        notes = "No deterministic compatibility rule matched this file class."
        exact_mimes: list[str] = []
        exact_extensions: list[str] = []
        documented_support = False
        endpoint = "chat.completions"
        delivery = "file"
        max_file_size_mb: float | None = 20.0

        if canonical_class in {"text.plain", "text.markup", "text.code"}:
            support_mode = "native_text_extract"
            notes = "Text-like material can be normalized into prompt-safe text."
            delivery = "inline_text"
            documented_support = True
        elif canonical_class == "table.delimited":
            support_mode = "native_table_extract"
            notes = "Delimited tables are normalized into CSV text."
            delivery = "inline_text"
            documented_support = True
        elif canonical_class == "image.raster":
            exact_mimes = ["image/png", "image/jpeg", "image/webp"]
            exact_extensions = [".png", ".jpg", ".jpeg", ".webp"]
            if lowered_provider in {"gemini", "vertex", "openai", "anthropic"}:
                support_mode = "native_multimodal"
                notes = "Raster images should stay as images for multimodal handling."
                documented_support = lowered_provider in {"gemini", "openai", "anthropic"}
            else:
                support_mode = "unsupported"
        elif canonical_class == "document.pdf":
            exact_mimes = ["application/pdf"]
            exact_extensions = [".pdf"]
            if lowered_provider in {"gemini", "anthropic"}:
                support_mode = "native_text_extract"
                notes = "Searchable PDFs can stay in PDF form or be normalized to text."
                documented_support = True
            else:
                support_mode = "tool_required"
                notes = "Treat PDFs as tool-normalized text before model delivery."
                documented_support = lowered_provider in {"openai", "openrouter"}
        elif canonical_class == "document.office.spreadsheet":
            support_mode = "conversion_required"
            notes = "Convert spreadsheets to CSV before use."
        elif canonical_class in {"document.office.word", "document.office.presentation"}:
            support_mode = "conversion_required"
            notes = "Convert slides or word-processing documents to PDF or extracted text first."
        elif canonical_class == "archive.bundle":
            support_mode = "tool_required"
            notes = "Archives must be safely extracted and each child routed by type."
            documented_support = True

        decisions.append(
            FileSupportDecision(
                provider=lowered_provider or "gemini",
                model=model,
                endpoint=endpoint,
                delivery=delivery,
                canonical_class=canonical_class,
                exact_mimes=exact_mimes,
                exact_extensions=exact_extensions,
                max_file_size_mb=max_file_size_mb,
                documented_support=documented_support,
                support_mode=support_mode,
                notes=notes,
            )
        )
    return decisions


def build_compatibility_recommendations(
    decisions: Iterable[FileSupportDecision],
) -> tuple[list[RecommendedTarget], list[FallbackConversion]]:
    """Build ranked recommended targets and fallback conversions."""

    recommended: list[RecommendedTarget] = []
    conversions: list[FallbackConversion] = []
    for decision in decisions:
        if decision.support_mode in {"native_multimodal", "native_text_extract", "native_table_extract", "tool_required"}:
            recommended.append(
                RecommendedTarget(
                    provider=decision.provider,
                    model=decision.model,
                    support_mode=decision.support_mode,
                    supported_basis="canonical_class",
                    documented_support=decision.documented_support,
                    max_file_size_mb=decision.max_file_size_mb,
                    conversion_required=None,
                    confidence="high" if decision.documented_support else "medium",
                    reason=decision.notes,
                )
            )
        if decision.support_mode == "conversion_required":
            target_format = "plain_text"
            if decision.canonical_class == "document.office.spreadsheet":
                target_format = "csv"
            elif decision.canonical_class in {"document.office.word", "document.office.presentation"}:
                target_format = "searchable_pdf"
            conversions.append(
                FallbackConversion(
                    from_class=decision.canonical_class,
                    to_format=target_format,  # type: ignore[arg-type]
                    reason=decision.notes,
                )
            )
            recommended.append(
                RecommendedTarget(
                    provider="openai",
                    model="gpt-5",
                    support_mode="conversion_required",
                    supported_basis="canonical_class",
                    documented_support=False,
                    max_file_size_mb=20.0,
                    conversion_required=target_format,
                    confidence="medium",
                    reason=decision.notes,
                )
            )
    recommended.sort(
        key=lambda item: (
            item.confidence != "high",
            not item.documented_support,
            item.support_mode == "conversion_required",
            item.support_mode == "tool_required",
        )
    )
    return recommended[:5], conversions[:5]


def _inspect_one_path(
    path: Path,
    *,
    provider: str,
    model: str,
    utilities: list[str],
) -> dict[str, list[ParsedContextFile] | list[FailedContextFile] | list[str]]:
    mime_type = sniff_mime_type(path)
    canonical_class = canonical_class_from_path(path, mime_type=mime_type)
    size_mb = _size_mb(path)
    source_ref = f"file://{path}"
    if canonical_class == "archive.bundle":
        return _inspect_archive(path, provider=provider, model=model)
    if canonical_class == "document.pdf" and _pdf_needs_ocr(path):
        return {
            "parsed": [],
            "failed": [
                FailedContextFile(
                    filename=path.name,
                    extension=path.suffix.lower().lstrip("."),
                    mime_type=mime_type,
                    size_mb=size_mb,
                    canonical_class=canonical_class,
                    failure_stage="ocr_needed",
                    failure_reason_user_safe="This PDF appears to need OCR or a searchable text layer before ProductiveBrain can use it.",
                )
            ],
            "canonical_classes": [canonical_class],
        }
    if canonical_class in {"document.office.word", "document.office.presentation", "document.office.spreadsheet", "unknown"}:
        failure_stage = "unsupported_extension" if canonical_class == "unknown" else "model_could_not_read"
        failure_reason = "This file type is not supported yet."
        if canonical_class == "document.office.word":
            failure_reason = "Convert this document to a searchable PDF or extracted text first."
        elif canonical_class == "document.office.presentation":
            failure_reason = "Convert this slide deck to PDF or extracted text first."
        elif canonical_class == "document.office.spreadsheet":
            failure_reason = "Convert this workbook to CSV first."
        return {
            "parsed": [],
            "failed": [
                FailedContextFile(
                    filename=path.name,
                    extension=path.suffix.lower().lstrip("."),
                    mime_type=mime_type,
                    size_mb=size_mb,
                    canonical_class=canonical_class,
                    failure_stage=failure_stage,  # type: ignore[arg-type]
                    failure_reason_user_safe=failure_reason,
                )
            ],
            "canonical_classes": [canonical_class],
        }
    summary = _content_summary(path, canonical_class=canonical_class)
    return {
        "parsed": [
            ParsedContextFile(
                filename=path.name,
                extension=path.suffix.lower().lstrip("."),
                mime_type=mime_type,
                size_mb=size_mb,
                canonical_class=canonical_class,  # type: ignore[arg-type]
                normalized_as=normalized_target_for_class(canonical_class),  # type: ignore[arg-type]
                content_summary=summary,
                source_ref=source_ref,
                parse_confidence="high" if summary else "medium",
            )
        ],
        "failed": [],
        "canonical_classes": [canonical_class],
    }


def _inspect_archive(path: Path, *, provider: str, model: str) -> dict[str, list[ParsedContextFile] | list[FailedContextFile] | list[str]]:
    parsed: list[ParsedContextFile] = []
    failed: list[FailedContextFile] = []
    canonical_classes = ["archive.bundle"]
    try:
        with zipfile.ZipFile(path) as archive:
            members = archive.infolist()
            total_size = 0
            extracted_count = 0
            for member in members:
                if member.is_dir():
                    continue
                safe_rel = _safe_archive_member(member.filename)
                if safe_rel is None:
                    continue
                total_size += int(member.file_size or 0)
                extracted_count += 1
                if extracted_count > 200 or total_size > 100 * 1024 * 1024:
                    failed.append(
                        FailedContextFile(
                            filename=path.name,
                            extension="zip",
                            mime_type="application/zip",
                            size_mb=_size_mb(path),
                            canonical_class="archive.bundle",
                            failure_stage="too_large",
                            failure_reason_user_safe="This archive exceeded the safe extraction limits for file count or total size.",
                        )
                    )
                    break
                child_name = safe_rel.name
                extension = safe_rel.suffix.lower().lstrip(".")
                mime_type = mimetypes.guess_type(str(safe_rel))[0] or "application/octet-stream"
                canonical_class = canonical_class_from_path(safe_rel, mime_type=mime_type)
                canonical_classes.append(canonical_class)
                if canonical_class == "document.pdf":
                    failed.append(
                        FailedContextFile(
                            filename=f"{path.name}:{child_name}",
                            extension=extension,
                            mime_type=mime_type,
                            size_mb=round((member.file_size or 0) / (1024 * 1024), 3),
                            canonical_class=canonical_class,
                            failure_stage="model_could_not_read",
                            failure_reason_user_safe="Extracted PDF children need to be added directly so OCR and text checks can run safely.",
                        )
                    )
                    continue
                if canonical_class in {"document.office.word", "document.office.presentation", "document.office.spreadsheet", "unknown"}:
                    failed.append(
                        FailedContextFile(
                            filename=f"{path.name}:{child_name}",
                            extension=extension,
                            mime_type=mime_type,
                            size_mb=round((member.file_size or 0) / (1024 * 1024), 3),
                            canonical_class=canonical_class,
                            failure_stage="unsupported_extension",
                            failure_reason_user_safe="This extracted file type needs conversion or direct upload outside the archive.",
                        )
                    )
                    continue
                parsed.append(
                    ParsedContextFile(
                        filename=f"{path.name}:{child_name}",
                        extension=extension,
                        mime_type=mime_type,
                        size_mb=round((member.file_size or 0) / (1024 * 1024), 3),
                        canonical_class=canonical_class,  # type: ignore[arg-type]
                        normalized_as=normalized_target_for_class(canonical_class),  # type: ignore[arg-type]
                        content_summary=f"Extracted from archive member `{child_name}`.",
                        source_ref=f"archive://{path.name}/{safe_rel.as_posix()}",
                        parse_confidence="medium",
                    )
                )
    except zipfile.BadZipFile:
        failed.append(
            FailedContextFile(
                filename=path.name,
                extension="zip",
                mime_type="application/zip",
                size_mb=_size_mb(path),
                canonical_class="archive.bundle",
                failure_stage="encrypted_or_corrupt",
                failure_reason_user_safe="This archive is corrupt or unreadable.",
            )
        )
    return {
        "parsed": parsed,
        "failed": failed,
        "canonical_classes": canonical_classes,
    }


def _safe_archive_member(name: str) -> PurePosixPath | None:
    clean = PurePosixPath(str(name).replace("\\", "/"))
    if clean.is_absolute() or ".." in clean.parts:
        return None
    lowered_parts = {part.lower() for part in clean.parts}
    if lowered_parts & _JUNK_PATH_PARTS:
        return None
    if clean.name.lower() in _JUNK_SUFFIXES or any(clean.name.lower().endswith(suffix) for suffix in _JUNK_SUFFIXES):
        return None
    if len(clean.parts) > 8:
        return None
    return clean


def _pdf_needs_ocr(path: Path) -> bool:
    try:
        data = path.read_bytes()
    except OSError:
        return True
    if b"/Font" not in data and b"BT" not in data:
        return True
    if re.search(rb"\(([^)]{4,})\)\s*Tj", data):
        return False
    if re.search(rb"\[(.*?)\]\s*TJ", data, flags=re.DOTALL):
        return False
    if re.search(rb"/ToUnicode", data):
        return False
    return True


def _content_summary(path: Path, *, canonical_class: str) -> str:
    if canonical_class == "document.pdf":
        return "Searchable PDF source material."
    if canonical_class == "image.raster":
        return "Raster image source material."
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return ""
    clean = re.sub(r"\s+", " ", text).strip()
    if not clean:
        return ""
    return clean[:240]


def _dominant_source_utility(
    utilities: Iterable[str],
) -> Literal[
    "syllabus",
    "course_notes",
    "worksheet",
    "slides",
    "textbook",
    "reference_manual",
    "project_context",
    "mixed_archive",
    "unknown",
]:
    items = [str(item).strip().lower() for item in utilities if str(item).strip()]
    if not items:
        return "unknown"
    priorities = [
        "syllabus",
        "worksheet",
        "course_notes",
        "slides",
        "textbook",
        "reference_manual",
        "mixed_archive",
        "project_context",
        "unknown",
    ]
    for utility in priorities:
        if utility in items:
            return utility  # type: ignore[return-value]
    return "unknown"


def _default_scope_boundary(paths: Iterable[Path]) -> str:
    labels = [path.name for path in paths if path.name]
    if not labels:
        return ""
    joined = ", ".join(labels[:3])
    if len(labels) > 3:
        joined += ", ..."
    return f"Use only the uploaded source material from {joined}."


def _broad_domain_from_filename(paths: Iterable[Path]) -> str:
    stems = [path.stem.replace("_", " ").replace("-", " ").strip() for path in paths if path.stem.strip()]
    combined = " ".join(stems).lower()
    if "chem" in combined:
        return "inorganic chemistry"
    if any(token in combined for token in ("rust", "cargo")):
        return "rust programming"
    if any(token in combined for token in ("calculus", "manifold", "stokes")):
        return "vector calculus"
    if any(token in combined for token in ("matrix", "eigen", "linear")):
        return "linear algebra"
    if any(token in combined for token in ("backend", "api", "server", "sql")):
        return "backend web development"
    if any(token in combined for token in ("transcriptomics", "visium", "rna")):
        return "spatial transcriptomics"
    return "general learning"


def _size_mb(path: Path) -> float:
    try:
        return round(path.stat().st_size / (1024 * 1024), 3)
    except OSError:
        return 0.0
