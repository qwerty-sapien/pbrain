# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Minimal multi-path encoding contract for concept notes."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pb.core.models import utc_now
from pb.storage.database import get_connection
from pb.vault.lifecycle import read_frontmatter, write_frontmatter

ENCODING_VERSION = 1
ALLOWED_ENCODING_FACETS = ("cue", "example", "source_context", "future_use", "failure_mode")
_GENERIC_VALUES = {
    "important",
    "review this",
    "understand this",
    "learn this",
    "example",
    "concept",
    "use this later",
    "common mistake",
}
_RELATIONLIKE_FACETS = {"prior_connection", "related", "prerequisite", "prerequisites", "contrast", "contrasts_with"}


def _norm(text: str) -> str:
    lowered = re.sub(r"[^a-z0-9+#& ]+", " ", (text or "").strip().lower())
    return re.sub(r"\s+", " ", lowered).strip()


def _tokens(text: str) -> set[str]:
    return {token for token in _norm(text).split() if len(token) >= 3}


@dataclass(frozen=True)
class EncodingSuggestion:
    target_concept: str
    facet: str
    value: str
    source_provenance: str
    evidence_text: str = ""

    @property
    def suggestion_hash(self) -> str:
        basis = "|".join(
            [
                self.target_concept,
                self.facet,
                _norm(self.value),
                self.source_provenance,
                evidence_hash(self.evidence_text),
            ]
        )
        return hashlib.sha256(basis.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class EncodingValidation:
    ok: bool
    reason: str = ""
    convert_to_relation: bool = False


def evidence_hash(evidence_text: str) -> str:
    return hashlib.sha256(_norm(evidence_text).encode("utf-8")).hexdigest()[:32]


def encoding_status(frontmatter: dict[str, Any]) -> dict[str, Any]:
    encoding = frontmatter.get("encoding")
    present = set()
    if isinstance(encoding, dict):
        present = {key for key in ALLOWED_ENCODING_FACETS if str(encoding.get(key, "")).strip()}
    missing = [key for key in ALLOWED_ENCODING_FACETS if key not in present]
    return {
        "version": int(frontmatter.get("encoding_version") or 0),
        "complete": not missing and int(frontmatter.get("encoding_version") or 0) == ENCODING_VERSION,
        "missing": missing,
    }


def validate_encoding_suggestion(
    suggestion: EncodingSuggestion,
    *,
    title: str = "",
    aliases: list[str] | None = None,
    body: str = "",
    relations: dict[str, list[str]] | None = None,
) -> EncodingValidation:
    facet = _norm(suggestion.facet).replace(" ", "_")
    value = " ".join((suggestion.value or "").split())
    if facet in _RELATIONLIKE_FACETS:
        return EncodingValidation(False, "better represented as a graph edge", convert_to_relation=True)
    if facet not in ALLOWED_ENCODING_FACETS:
        return EncodingValidation(False, f"unsupported encoding facet: {suggestion.facet}")
    if len(value) < 4 or _norm(value) in _GENERIC_VALUES:
        return EncodingValidation(False, "generic encoding suggestion")
    if not suggestion.source_provenance.strip() or not suggestion.evidence_text.strip():
        return EncodingValidation(False, "unsupported by provenance")
    title_tokens = _tokens(title)
    value_tokens = _tokens(value)
    if value_tokens and (value_tokens <= title_tokens or _norm(value) == _norm(title)):
        return EncodingValidation(False, "redundant with title")
    alias_text = " ".join(aliases or [])
    if value_tokens and value_tokens <= _tokens(alias_text):
        return EncodingValidation(False, "redundant with aliases")
    body_tokens = _tokens(body)
    if len(value_tokens) >= 3 and value_tokens <= body_tokens:
        return EncodingValidation(False, "redundant with note body")
    relation_text = " ".join(item for values in (relations or {}).values() for item in values)
    if value_tokens and value_tokens <= _tokens(relation_text):
        return EncodingValidation(False, "redundant with graph relations")
    if len(value_tokens) < 2 and facet != "cue":
        return EncodingValidation(False, "untestable as a retrieval path")
    return EncodingValidation(True)


def was_rejected(suggestion: EncodingSuggestion) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT evidence_hash FROM encoding_suggestion_rejections WHERE suggestion_hash = ?",
            (suggestion.suggestion_hash,),
        ).fetchone()
    return bool(row and row["evidence_hash"] == evidence_hash(suggestion.evidence_text))


def record_rejection(suggestion: EncodingSuggestion, reason: str) -> str:
    now = utc_now().isoformat()
    with get_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO encoding_suggestion_rejections (
                suggestion_hash, target_concept, facet, value, source_provenance,
                evidence_hash, reason, created_at, updated_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?,
                ?, COALESCE((SELECT created_at FROM encoding_suggestion_rejections WHERE suggestion_hash = ?), ?), ?
            )
            """,
            (
                suggestion.suggestion_hash,
                suggestion.target_concept,
                suggestion.facet,
                suggestion.value,
                suggestion.source_provenance,
                evidence_hash(suggestion.evidence_text),
                reason,
                suggestion.suggestion_hash,
                now,
                now,
            ),
        )
        conn.commit()
    return suggestion.suggestion_hash


def apply_encoding_suggestion(note_path: Path, suggestion: EncodingSuggestion) -> EncodingValidation:
    """Validate and apply one accepted encoding suggestion to a concept note."""

    content = note_path.read_text(encoding="utf-8")
    fm, body = read_frontmatter(content)
    relations = fm.get("relations", {}) if isinstance(fm.get("relations"), dict) else {}
    validation = validate_encoding_suggestion(
        suggestion,
        title=str(fm.get("title", "")),
        aliases=[str(item) for item in fm.get("aliases", [])] if isinstance(fm.get("aliases"), list) else [],
        body=body,
        relations={str(key): [str(item) for item in value] for key, value in relations.items() if isinstance(value, list)},
    )
    if not validation.ok:
        return validation
    encoding = fm.get("encoding") if isinstance(fm.get("encoding"), dict) else {}
    encoding = {key: str(value) for key, value in encoding.items() if key in ALLOWED_ENCODING_FACETS and str(value).strip()}
    facet = suggestion.facet.strip()
    current = str(encoding.get(facet, "") or "").strip()
    if current and _norm(current) == _norm(suggestion.value):
        return EncodingValidation(False, "redundant with existing encoding")
    encoding[facet] = suggestion.value.strip()
    fm["encoding_version"] = ENCODING_VERSION
    fm["encoding"] = encoding
    provenance = fm.get("encoding_provenance") if isinstance(fm.get("encoding_provenance"), list) else []
    provenance.append(
        {
            "facet": facet,
            "source": suggestion.source_provenance,
            "suggestion_hash": suggestion.suggestion_hash,
            "evidence_hash": evidence_hash(suggestion.evidence_text),
            "accepted_at": utc_now().isoformat(),
        }
    )
    fm["encoding_provenance"] = provenance[-20:]
    note_path.write_text(write_frontmatter(fm, body), encoding="utf-8")
    return EncodingValidation(True)


def suggestion_to_json(suggestion: EncodingSuggestion) -> str:
    return json.dumps(
        {
            "target_concept": suggestion.target_concept,
            "facet": suggestion.facet,
            "value": suggestion.value,
            "source_provenance": suggestion.source_provenance,
            "suggestion_hash": suggestion.suggestion_hash,
        },
        ensure_ascii=True,
        sort_keys=True,
    )
