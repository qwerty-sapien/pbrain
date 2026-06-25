# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain template registry for evidence notes.

Defines per-domain sub-skill taxonomies and Markdown template file mappings.
Used by EvidenceWriter to resolve the correct template and sub-skill list
for a given learning session domain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

from pb.core.graph_writer import make_slug
from pb.core.learning_metadata import parse_learning_task_metadata
from pb.core.session_blueprints import (
    blueprint_from_payload,
    load_domain_packs,
    resolve_learning_session_blueprint,
)

if TYPE_CHECKING:
    from pb.domain.models import Session, Task


@dataclass
class DomainTemplate:
    """Describes a domain's evidence template and sub-skill taxonomy."""

    name: str
    """Template identifier (e.g., 'math_problem_set')."""

    sub_skill_taxonomy: list[str] = field(default_factory=list)
    """Base sub-skill names for this domain, assessed during each session."""

    markdown_template_file: str = "evidence_generic.md"
    """Filename in the bundled template resources for the Markdown body template."""

    fallback: bool = False
    """True only for the generic fallback template."""


TEMPLATES: dict[str, DomainTemplate] = {
    "math_problem_set": DomainTemplate(
        name="math_problem_set",
        sub_skill_taxonomy=[
            "problem_setup",
            "technique_selection",
            "execution",
            "edge_case_awareness",
            "verification",
        ],
        markdown_template_file="evidence_math.md",
    ),
    "rust_project": DomainTemplate(
        name="rust_project",
        sub_skill_taxonomy=[
            "ownership_borrowing",
            "error_handling",
            "trait_usage",
            "lifetime_annotations",
            "compiler_error_resolution",
            "idiomatic_patterns",
        ],
        markdown_template_file="evidence_rust.md",
    ),
    "german_speaking": DomainTemplate(
        name="german_speaking",
        sub_skill_taxonomy=[
            "conjugation_cases",
            "artikel_gender",
            "vocabulary_recall",
            "sentence_structure",
            "pronunciation_intonation",
            "language_intuition",
        ],
        markdown_template_file="evidence_german.md",
    ),
    "_generic": DomainTemplate(
        name="_generic",
        sub_skill_taxonomy=[
            "session_goal_met",
            "key_concepts_practiced",
            "difficulties_encountered",
            "self_assessment",
        ],
        markdown_template_file="evidence_generic.md",
        fallback=True,
    ),
}


def _template_from_blueprint(pack_id: str, blueprint) -> DomainTemplate | None:
    if blueprint is None:
        return None
    pack = load_domain_packs().get(pack_id)
    template_file = pack.legacy_template_file if pack is not None else "evidence_generic.md"
    fallback = pack_id.startswith("generic.") or pack_id == "universal.fallback"
    return DomainTemplate(
        name=pack_id or "_generic",
        sub_skill_taxonomy=list(blueprint.subskills or []),
        markdown_template_file=template_file,
        fallback=fallback,
    )


def get_template(
    domain: str,
    *,
    branch: str = "study",
    session: "Session | None" = None,
    task: "Task | None" = None,
) -> DomainTemplate:
    """Resolve a DomainTemplate, preferring blueprint-backed domain packs."""

    if domain in TEMPLATES:
        return TEMPLATES[domain]

    task_meta = parse_learning_task_metadata(task) if task is not None else None
    generated = dict(getattr(session, "generated_names", {}) or {}) if session is not None else {}
    pack_id = str(generated.get("domain_pack_id", "") or getattr(task_meta, "domain_pack_id", "") or "").strip()
    blueprint = blueprint_from_payload(
        generated.get("session_blueprint") if isinstance(generated.get("session_blueprint"), dict) else getattr(task_meta, "session_blueprint", None)
    )
    if blueprint is not None:
        template = _template_from_blueprint(pack_id, blueprint)
        if template is not None:
            return template

    resolved = resolve_learning_session_blueprint(
        branch=branch,
        domain=domain,
        topic=getattr(task, "title", "") if task is not None else domain,
    )
    template = _template_from_blueprint(resolved.pack_id, resolved.blueprint)
    if template is not None:
        return template
    return TEMPLATES["_generic"]


def _resolve_domain(session: "Session", task: "Task") -> str:
    """Extract domain string from session/task metadata.

    Resolution order:
    (a) session.subject_scope if non-empty
    (b) task.domain attribute if present and non-empty
    (c) session.evidence_target if set and non-empty
    (d) fallback to "general"

    All values are sanitized via make_slug() for path safety.
    """
    meta = parse_learning_task_metadata(task)
    generated = dict(getattr(session, "generated_names", {}) or {})
    blueprint = blueprint_from_payload(
        generated.get("session_blueprint") if isinstance(generated.get("session_blueprint"), dict) else meta.session_blueprint
    )
    if blueprint is not None and blueprint.domain.strip():
        return make_slug(blueprint.domain)

    # (a) subject_scope
    scope = getattr(session, "subject_scope", None)
    if scope:
        return make_slug(scope)

    # (b) task.domain
    domain = getattr(task, "domain", None)
    if domain:
        return make_slug(domain)

    # (c) evidence_target
    target = getattr(session, "evidence_target", None)
    if target:
        return make_slug(target)

    # (d) fallback
    return "general"
