# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F
"""Atomic concept note writer (Phase 16, D-16-01/D-16-07).

Provides the canonical YAML frontmatter renderer for LearningConcept notes.
No note creation must trigger a concept_confidence write — see D-16-13/L6.

NOTE TYPE CONTRACTS (D-16-02 — schema constraints, not runtime enforcement):
  type: concept     — One concept. One note. Exactly ONE small learnable element.
  type: example     — Child of a concept note. Reusable teaching example.
  type: moc         — Meta-concept / MOC: SHORT, LINK-DENSE, DEPENDENCY-ORIENTED.
                      A dependency map of children, NOT a textbook chapter.
                      Hierarchy: moc -> concept notes -> smaller concepts -> example notes.
                      Rule: if the body exceeds ~800 chars, the note is a candidate for
                      splitting into child concept notes rather than expanding the MOC.

Alias resolution (D-16-09 — Phase 16 delivery = aliases[] YAML field only):
  The aliases[] field records known aliases for this concept (e.g. "OA", "oxidative
  addition step"). Alias resolution to canonical IDs and unresolved-link flagging
  during QC are Phase 17 concerns. Phase 16 only stores the alias strings in YAML.

Graph traversal (D-16-15 — deferred to Phase 17):
  The relations.* typed-link fields establish the directed graph structure. The
  in-memory query layer for parent->children, weak-neighbour, cross-domain traversal
  is built in Phase 17 on top of the YAML schema defined here.

QC agent (D-16-04 — deferred to Phase 17):
  Phase 16 sets qc_status: candidate when body > 1000 chars. The agent decision
  (Reducible -> split; Irreducible -> strip) runs in Phase 17.
"""
from __future__ import annotations

from typing import Any

import structlog

_log = structlog.get_logger(__name__)

QC_BODY_THRESHOLD: int = 1000  # D-16-03/D-16-05: body > 1000 chars = QC candidate


def concept_note_path(domain: str, slug: str) -> str:
    """Return vault-relative path for a concept note (D-16-10).

    Format: knowledge/{domain}/{slug}.md
    The 'knowledge/' folder is the Phase 16 canonical location for concept notes.
    NOT the legacy '20-concepts/' folder from schemas.py.
    """
    return f"knowledge/{domain}/{slug}.md"


def check_body_length(body: str) -> bool:
    """Return True if body is a QC candidate (D-16-03/D-16-05).

    The char count applies to body only (never YAML frontmatter).
    Threshold: body > 1000 chars is a QC candidate for non-atomicity.
    Exactly 1000 chars is NOT a candidate — only strictly greater than.
    """
    return len(body) > QC_BODY_THRESHOLD


def render_concept_frontmatter(
    title: str,
    domain: str,
    slug: str,
    *,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    aliases: list[str] | None = None,
    relations: dict[str, list[str]] | None = None,
    confidence_record: Any = None,
    qc_candidate: bool = False,
) -> str:
    """Render the full D-16-07 YAML frontmatter block for an atomic concept note.

    Args:
        title: Human-readable concept title.
        domain: Domain slug (e.g. "organometallic-catalysis").
        slug: URL-safe concept slug (e.g. "oxidative-addition").
        parents: Parent concept wikilinks (e.g. ["[[Cross-coupling reactions]]"]).
        children: Child concept wikilinks.
        aliases: Alias strings for this concept (D-16-09 field; resolution logic Phase 17).
        relations: Dict of typed-link lists (prerequisites, appears_in, etc.).
        confidence_record: ConceptConfidenceRecord from DB (read-only projection, D-16-07).
                           If None, defaults to none/0.0/1.0 (no evidence yet).
        qc_candidate: If True, adds qc_status: candidate to frontmatter (D-16-03).

    Returns:
        Complete YAML block as string including opening and closing "---" markers.

    CRITICAL (L6): This function NEVER writes to the concept_confidence table.
    Note creation is NOT evidence of knowledge (D-16-13).
    """
    from pb.core.confidence_model import confidence_label

    # Resolve learning.* projection from DB record (read-only)
    if confidence_record is not None:
        score = float(getattr(confidence_record, "confidence_score", 0.0) or 0.0)
        weight = float(getattr(confidence_record, "card_weight", 1.0 - score) or (1.0 - score))
        last_seen = str(getattr(confidence_record, "last_evidence_at", "") or "")
        next_review = str(getattr(confidence_record, "next_review_at", "") or "")
        label = confidence_label(score)
    else:
        score, weight, last_seen, next_review, label = 0.0, 1.0, "", "", "none"

    concept_id = f"concept:{domain}:{slug}"

    # Build YAML manually to maintain exact field order and Obsidian wikilink style
    def _yaml_list(items: list[str] | None) -> str:
        if not items:
            return "[]"
        return "[" + ", ".join(f'"{i}"' for i in items) + "]"

    rel = relations or {}
    rel_keys = [
        "prerequisites", "prerequisite_for", "appears_in", "contrasts_with",
        "usually_precedes", "usually_follows", "examples", "failure_modes",
        "misconceptions", "related",
    ]

    lines = ["---"]
    lines.append(f"type: concept")
    lines.append(f"id: {concept_id}")
    lines.append(f"slug: {slug}")
    lines.append(f"title: {title}")
    lines.append(f"domain: {domain}")
    if qc_candidate:
        lines.append(f"qc_status: candidate")
    lines.append("")
    lines.append(f"parents: {_yaml_list(parents)}")
    lines.append(f"children: {_yaml_list(children)}")
    lines.append(f"aliases: {_yaml_list(aliases)}")
    lines.append("")
    lines.append("relations:")
    for key in rel_keys:
        items = rel.get(key, [])
        lines.append(f"  {key}: {_yaml_list(items)}")
    lines.append("")
    lines.append("learning:")
    lines.append(f"  confidence: {label}")
    lines.append(f"  confidence_score: {score}")
    lines.append(f"  card_weight: {weight}")
    lines.append(f"  last_seen_at: {last_seen}")
    lines.append(f"  next_review_at: {next_review}")
    lines.append("---")

    return "\n".join(lines)


def write_concept_note(
    title: str,
    domain: str,
    body: str,
    *,
    repo: Any,
    vault_root: Any,
    parents: list[str] | None = None,
    children: list[str] | None = None,
    aliases: list[str] | None = None,
    relations: dict[str, list[str]] | None = None,
) -> tuple[str, bool]:
    """Write an atomic concept note to the vault (D-16-01/D-16-07/D-16-10).

    Returns (vault_relative_path, qc_candidate).
    NEVER writes to the concept_confidence table (D-16-13/L6).

    Args:
        title: Concept title.
        domain: Domain name.
        body: Note body (YAML frontmatter excluded — body is everything after ---).
        repo: Repository instance for concept_confidence lookup (read-only).
        vault_root: Path to vault root directory.
        parents, children, aliases, relations: Optional typed-link metadata.

    Returns:
        (vault_relative_path: str, qc_candidate: bool)
    """
    from pathlib import Path
    from pb.core.graph_writer import make_slug

    slug = make_slug(title)
    concept_id = f"concept:{domain}:{slug}"

    # D-16-07: project learning.* from DB (read-only; never write back to DB here)
    records = repo.list_concept_confidence(concept_id) if repo else []
    confidence_record = records[0] if records else None

    # D-16-03/D-16-05: body QC check
    qc_candidate = check_body_length(body)

    # Render full frontmatter
    frontmatter = render_concept_frontmatter(
        title=title,
        domain=domain,
        slug=slug,
        parents=parents,
        children=children,
        aliases=aliases,
        relations=relations,
        confidence_record=confidence_record,
        qc_candidate=qc_candidate,
    )

    full_content = f"{frontmatter}\n\n# {title}\n\n{body}"

    # Write to vault
    rel_path = concept_note_path(domain, slug)
    abs_path = Path(vault_root) / rel_path
    abs_path.parent.mkdir(parents=True, exist_ok=True)
    abs_path.write_text(full_content, encoding="utf-8")

    if qc_candidate:
        _log.warning(
            "concept_note.qc_candidate",
            concept_id=concept_id,
            body_length=len(body),
            message=f"[QC] Note body is {len(body)} chars (>1000). May need splitting.",
        )

    return rel_path, qc_candidate
