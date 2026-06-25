# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Domain-pack-backed learning session blueprints."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
import re
from typing import Any

import yaml
from pydantic import BaseModel, Field

from pb.core.enums import EvidenceContract, SessionFeedbackSource, SessionFrame, SkillKind
from pb.core.resources import iter_domain_pack_resources
from pb.llm.drafts import LearningPlanBlockDraft, LearningSessionBlueprintDraft


_MOTOR_HINTS = {
    "cardistry",
    "grip",
    "hold",
    "finger",
    "deck",
    "skate",
    "skateboarding",
    "stance",
    "trick",
    "piano",
    "guitar",
    "violin",
    "drum",
    "dance",
    "card flourish",
}
_PERCEPTUAL_HINTS = {
    "listen",
    "recognition",
    "discrimination",
    "intonation",
    "pronunciation",
    "tone",
    "pitch",
    "sensory",
    "taste",
    "visual",
}
_LANGUAGE_HINTS = {
    "german",
    "language",
    "vocab",
    "vocabulary",
    "speaking",
    "conversation",
    "pronunciation",
    "grammar",
    "listening",
}
_ENGINEERING_HINTS = {
    "debug",
    "bug",
    "compiler",
    "trace",
    "stacktrace",
    "test failure",
    "failing test",
    "rust",
    "c++",
    "cpp",
    "implementation",
    "refactor",
}
_LAB_HINTS = {
    "biochemistry",
    "biochem",
    "enzyme",
    "pathway",
    "mechanism",
    "food science",
    "experiment",
    "assay",
    "fermentation",
    "hypothesis",
    "variable",
}
_CREATIVE_HINTS = {
    "video editing",
    "editing",
    "artifact",
    "render",
    "mix",
    "composition",
    "montage",
    "essay",
    "storyboard",
}
_MATH_PROOF_HINTS = {
    "proof",
    "prove",
    "theorem",
    "lemma",
    "corollary",
}
_MATH_HINTS = {
    "math",
    "mathematics",
    "calculus",
    "algebra",
    "geometry",
    "linear algebra",
    "physics",
    "equation",
    "derivation",
    "tensor",
}


class DomainPackDefinition(BaseModel):
    """Loaded YAML definition for one domain pack."""

    pack_id: str
    aliases: list[str] = Field(default_factory=list)
    nearby_terms: list[str] = Field(default_factory=list)
    skill_kind: SkillKind
    primary_frame: SessionFrame
    secondary_frames: list[SessionFrame] = Field(default_factory=list)
    subskills: list[str] = Field(default_factory=list)
    evidence_contract: list[EvidenceContract] = Field(default_factory=list)
    feedback_sources: list[SessionFeedbackSource] = Field(default_factory=list)
    opening_move: str = ""
    stop_condition: str = ""
    coach_rules: list[str] = Field(default_factory=list)
    safety_notes: list[str] = Field(default_factory=list)
    legacy_template_file: str = "evidence_generic.md"

    def to_blueprint(self, *, domain: str, topic: str) -> LearningSessionBlueprintDraft:
        return LearningSessionBlueprintDraft(
            domain=domain,
            topic=topic,
            skill_kind=self.skill_kind,
            primary_frame=self.primary_frame,
            secondary_frames=list(self.secondary_frames),
            subskills=list(self.subskills),
            evidence_contract=list(self.evidence_contract),
            feedback_sources=list(self.feedback_sources),
            opening_move=self.opening_move.format(topic=topic, domain=domain).strip(),
            stop_condition=self.stop_condition.format(topic=topic, domain=domain).strip(),
            coach_rules=[rule.format(topic=topic, domain=domain).strip() for rule in self.coach_rules if rule.strip()],
            safety_notes=[note.format(topic=topic, domain=domain).strip() for note in self.safety_notes if note.strip()],
        )


@dataclass(frozen=True)
class BlueprintResolution:
    """Resolved blueprint plus how the resolver arrived there."""

    pack_id: str
    blueprint: LearningSessionBlueprintDraft
    source: str
    suggested_pack_ids: list[str] = field(default_factory=list)


def user_pack_dir(vault_path: Path | None) -> Path | None:
    """Return the learner-local pack directory when a vault path exists."""

    if vault_path is None:
        return None
    return Path(vault_path) / "direction" / "blueprints" / "session-packs"


def _normalize_token(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", ".", (text or "").strip().lower()).strip(".")


def _search_blob(*parts: str) -> str:
    return " ".join(part.strip().lower() for part in parts if part and part.strip())


@lru_cache(maxsize=1)
def _load_repo_domain_packs() -> dict[str, DomainPackDefinition]:
    """Load all repo-owned domain packs once."""

    packs: dict[str, DomainPackDefinition] = {}
    for path in iter_domain_pack_resources():
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pack = DomainPackDefinition.model_validate(raw)
        packs[pack.pack_id] = pack
    return packs


@lru_cache(maxsize=32)
def _load_custom_domain_packs(dir_key: str) -> dict[str, DomainPackDefinition]:
    packs: dict[str, DomainPackDefinition] = {}
    custom_dir = Path(dir_key)
    if not custom_dir.exists():
        return packs
    for path in sorted(custom_dir.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        pack = DomainPackDefinition.model_validate(raw)
        packs[pack.pack_id] = pack
    return packs


def load_domain_packs(*, additional_dirs: tuple[Path, ...] = ()) -> dict[str, DomainPackDefinition]:
    """Load repo packs plus any learner-local custom packs."""

    packs = dict(_load_repo_domain_packs())
    for directory in additional_dirs:
        if directory is None:
            continue
        packs.update(_load_custom_domain_packs(str(directory)))
    return packs


def infer_skill_kind(
    *,
    branch: str,
    domain: str = "",
    topic: str = "",
    drill: str = "",
    constraint: str = "",
) -> SkillKind:
    """Infer a compact skill kind from the learning context."""

    blob = _search_blob(branch, domain, topic, drill, constraint)
    if any(token in blob for token in _LANGUAGE_HINTS):
        return SkillKind.LANGUAGE
    if any(token in blob for token in _ENGINEERING_HINTS):
        return SkillKind.ENGINEERING_BUILD_DEBUG
    if any(token in blob for token in _LAB_HINTS):
        return SkillKind.EXPERIMENTAL_LAB
    if any(token in blob for token in _CREATIVE_HINTS):
        return SkillKind.CREATIVE_ARTIFACT
    if any(token in blob for token in _MOTOR_HINTS):
        return SkillKind.PROCEDURAL_MOTOR
    if any(token in blob for token in _PERCEPTUAL_HINTS):
        return SkillKind.PERCEPTUAL
    if any(token in blob for token in _MATH_HINTS):
        return SkillKind.CONCEPTUAL if branch == "study" else SkillKind.PROCEDURAL_COGNITIVE
    if branch == "practise":
        return SkillKind.PROCEDURAL_COGNITIVE
    if branch == "mixed":
        return SkillKind.MIXED
    return SkillKind.CONCEPTUAL


def _targeted_pack_from_context(*, branch: str, domain: str, topic: str, drill: str = "") -> str:
    blob = _search_blob(branch, domain, topic, drill)
    if "cardistry" in blob or ("grip" in blob and "deck" in blob):
        return "cardistry.grips"
    if any(token in blob for token in _ENGINEERING_HINTS):
        if any(token in blob for token in ("debug", "bug", "compiler", "trace", "test failure", "failing test")):
            return "programming.debugging"
        return "programming.implementation"
    if any(token in blob for token in _MATH_PROOF_HINTS):
        return "math.proofs"
    if any(token in blob for token in _MATH_HINTS):
        return "math.problem_solving"
    return ""


def _nearby_pack_from_context(
    *,
    domain: str,
    topic: str,
    branch: str,
    drill: str = "",
) -> list[str]:
    requested = _search_blob(domain, topic, branch, drill)
    packs = load_domain_packs()
    scored: list[tuple[int, str]] = []
    for pack_id, pack in packs.items():
        if pack_id.startswith("generic.") or pack_id == "universal.fallback":
            continue
        score = 0
        for candidate in [pack.pack_id, *pack.aliases, *pack.nearby_terms]:
            token = candidate.strip().lower()
            if not token:
                continue
            if token == requested:
                score += 100
            elif token in requested or requested in token:
                score += max(1, len(token.split(".")))
        if score > 0:
            scored.append((score, pack_id))
    scored.sort(key=lambda item: (-item[0], item[1]))
    ordered: list[str] = []
    for _, pack_id in scored:
        if pack_id not in ordered:
            ordered.append(pack_id)
    return ordered[:3]


def register_custom_blueprint_pack(
    *,
    vault_path: Path,
    branch: str,
    domain: str = "",
    topic: str = "",
    drill: str = "",
    base_pack_id: str = "",
) -> str:
    """Create a learner-local custom pack derived from a generic or chosen base pack."""

    custom_dir = user_pack_dir(vault_path)
    if custom_dir is None:
        raise ValueError("A vault path is required to create a custom blueprint pack.")
    custom_dir.mkdir(parents=True, exist_ok=True)
    additional_dirs = (custom_dir,)
    packs = load_domain_packs(additional_dirs=additional_dirs)
    slug_seed = _normalize_token(topic or domain or drill or "custom-session")
    existing = next(
        (
            pack_id
            for pack_id, pack in packs.items()
            if pack_id.startswith("custom.")
            and any(_normalize_token(alias) == slug_seed for alias in pack.aliases + [pack.pack_id])
        ),
        "",
    )
    if existing:
        return existing

    skill_kind = infer_skill_kind(branch=branch, domain=domain, topic=topic, drill=drill)
    resolved_base = base_pack_id.strip() or f"generic.{skill_kind.value}"
    if resolved_base not in packs:
        resolved_base = "universal.fallback"
    base = packs[resolved_base]
    pack_id = f"custom.{slug_seed or 'session'}"
    payload = base.model_dump(mode="json")
    payload.update(
        {
            "pack_id": pack_id,
            "aliases": list(dict.fromkeys([topic.strip(), domain.strip(), slug_seed])),
            "nearby_terms": list(dict.fromkeys([domain.strip(), topic.strip(), drill.strip()])),
            "coach_rules": list(dict.fromkeys([*base.coach_rules, f"Treat `{topic or domain}` as its own learner-specific blueprint."])),
        }
    )
    path = custom_dir / f"{pack_id}.yaml"
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    _load_custom_domain_packs.cache_clear()
    return pack_id


def resolve_domain_pack_id(
    *,
    branch: str,
    domain: str = "",
    topic: str = "",
    drill: str = "",
    domain_pack_id: str = "",
    additional_dirs: tuple[Path, ...] = (),
) -> str:
    """Resolve the best exact pack id using explicit, custom, or targeted matches."""

    packs = load_domain_packs(additional_dirs=additional_dirs)
    requested = domain_pack_id.strip()
    if requested and requested in packs:
        return requested

    normalized_requested = _normalize_token(requested or topic or domain)
    for pack_id, pack in packs.items():
        if any(_normalize_token(alias) == normalized_requested for alias in [pack.pack_id, *pack.aliases]):
            return pack_id

    targeted = _targeted_pack_from_context(branch=branch, domain=domain, topic=topic, drill=drill)
    if targeted in packs:
        return targeted
    return ""


def resolve_learning_session_blueprint(
    *,
    branch: str,
    domain: str = "",
    topic: str = "",
    drill: str = "",
    domain_pack_id: str = "",
    vault_path: Path | None = None,
    allow_custom_init: bool = False,
) -> BlueprintResolution:
    """Resolve a blueprint, optionally auto-initializing a learner-local custom pack."""

    additional_dirs = (user_pack_dir(vault_path),) if user_pack_dir(vault_path) is not None else ()
    packs = load_domain_packs(additional_dirs=tuple(item for item in additional_dirs if item is not None))
    pack_id = resolve_domain_pack_id(
        branch=branch,
        domain=domain,
        topic=topic,
        drill=drill,
        domain_pack_id=domain_pack_id,
        additional_dirs=tuple(item for item in additional_dirs if item is not None),
    )
    if pack_id:
        pack = packs[pack_id]
        return BlueprintResolution(
            pack_id=pack_id,
            blueprint=pack.to_blueprint(domain=domain.strip() or topic.strip() or "general", topic=topic.strip() or domain.strip() or "general"),
            source="exact",
            suggested_pack_ids=[],
        )

    suggestions = _nearby_pack_from_context(domain=domain, topic=topic, branch=branch, drill=drill)
    if allow_custom_init and vault_path is not None:
        custom_pack_id = register_custom_blueprint_pack(
            vault_path=vault_path,
            branch=branch,
            domain=domain,
            topic=topic,
            drill=drill,
        )
        packs = load_domain_packs(additional_dirs=tuple(item for item in additional_dirs if item is not None))
        custom_pack = packs[custom_pack_id]
        return BlueprintResolution(
            pack_id=custom_pack_id,
            blueprint=custom_pack.to_blueprint(
                domain=domain.strip() or topic.strip() or "general",
                topic=topic.strip() or domain.strip() or "general",
            ),
            source="custom",
            suggested_pack_ids=suggestions,
        )

    skill_kind = infer_skill_kind(branch=branch, domain=domain, topic=topic, drill=drill)
    generic_pack_id = f"generic.{skill_kind.value}" if f"generic.{skill_kind.value}" in packs else "universal.fallback"
    pack = packs[generic_pack_id]
    domain_text = domain.strip() or topic.strip() or "general"
    topic_text = topic.strip() or domain_text
    return BlueprintResolution(
        pack_id=generic_pack_id,
        blueprint=pack.to_blueprint(domain=domain_text, topic=topic_text),
        source="generic" if generic_pack_id.startswith("generic.") else "universal",
        suggested_pack_ids=suggestions,
    )


def hydrate_learning_block_blueprint(
    block: LearningPlanBlockDraft,
    *,
    domain_hint: str = "",
    topic_hint: str = "",
    vault_path: Path | None = None,
) -> LearningPlanBlockDraft:
    """Fill the blueprint fields on a learning block deterministically."""

    hydrated = block.model_copy(deep=True)
    domain = domain_hint.strip() or hydrated.subject_scope.strip() or hydrated.title.strip()
    topic = topic_hint.strip() or hydrated.subject_scope.strip() or hydrated.title.strip() or domain
    resolution = resolve_learning_session_blueprint(
        branch=hydrated.branch,
        domain=domain,
        topic=topic,
        drill=hydrated.drill_type or hydrated.constraint or "",
        domain_pack_id=hydrated.domain_pack_id,
        vault_path=vault_path,
        allow_custom_init=vault_path is not None,
    )
    hydrated.domain_pack_id = resolution.pack_id
    if hydrated.session_blueprint is None:
        hydrated.session_blueprint = resolution.blueprint
    return hydrated


def blueprint_payload(blueprint: LearningSessionBlueprintDraft | dict[str, Any] | None) -> dict[str, Any] | None:
    """Return a JSON-safe blueprint payload."""

    if blueprint is None:
        return None
    if isinstance(blueprint, dict):
        return LearningSessionBlueprintDraft.model_validate(blueprint).model_dump(mode="json")
    return blueprint.model_dump(mode="json")


def blueprint_from_payload(payload: dict[str, Any] | None) -> LearningSessionBlueprintDraft | None:
    """Parse a stored blueprint payload."""

    if not isinstance(payload, dict):
        return None
    try:
        return LearningSessionBlueprintDraft.model_validate(payload)
    except Exception:
        return None


def pack_display_label(pack_id: str) -> str:
    """Return a human-facing label for one pack id."""

    parts = [
        segment.replace("_", " ").strip().title()
        for segment in str(pack_id or "").replace("generic.", "generic ").split(".")
        if segment.strip()
    ]
    return " - ".join(parts)
