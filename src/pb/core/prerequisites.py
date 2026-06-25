# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Prerequisite tracing for learner-control flows."""

from __future__ import annotations

import re
from typing import Optional

from pydantic import BaseModel, Field

from pb.llm.runtime import LLMRuntime


_GENERIC_CHAIN = [
    "Vocabulary and primitives",
    "Representations / notation / tools",
    "Minimal worked examples",
    "Diagnosis of common misconceptions",
    "Guided practice",
    "Independent application",
    "Integrated project",
]

_DIFF_GEOMETRY_CHAIN = [
    "Multivariable calculus in coordinates",
    "Linear algebra: bases, dual spaces, bilinear maps",
    "Index notation and Einstein summation",
    "Tensors as multilinear maps",
    "Manifolds, charts, coordinate changes",
    "Tangent spaces and vector fields",
    "Metric tensor",
    "Christoffel symbols / connection coefficients",
    "Covariant derivative",
    "Riemann curvature tensor",
    "Ricci tensor and scalar curvature",
    "Einstein tensor",
    "Einstein field equations",
    "Einstein-Hilbert action",
    "Palatini variation",
]

_DOMAIN_ALIASES = {
    "general relativity": _DIFF_GEOMETRY_CHAIN,
    "differential geometry": _DIFF_GEOMETRY_CHAIN,
    "semi-riemannian geometry": _DIFF_GEOMETRY_CHAIN,
    "curvature tensor": _DIFF_GEOMETRY_CHAIN,
    "riemann": _DIFF_GEOMETRY_CHAIN,
    "ricci": _DIFF_GEOMETRY_CHAIN,
    "tensor": _DIFF_GEOMETRY_CHAIN,
    "einstein": _DIFF_GEOMETRY_CHAIN,
    "palatini": _DIFF_GEOMETRY_CHAIN,
}


class KnownUnknownInferenceDraft(BaseModel):
    knowns: list[str] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    notes: str = ""


class PrerequisiteChainDraft(BaseModel):
    domain: str = ""
    target: str = ""
    chain: list[str] = Field(default_factory=list)


def _resolve_known_chain(domain: str, target: str = "") -> list[str] | None:
    lowered = " ".join((f"{domain} {target}").lower().split())
    for alias, chain in _DOMAIN_ALIASES.items():
        if alias in lowered:
            return list(chain)
    return None


def infer_knowns_unknowns(
    raw_user_text: str,
    current_artifact: str,
    prior_state,
    *,
    runtime: Optional[LLMRuntime] = None,
) -> tuple[list[str], list[str]]:
    text = " ".join((raw_user_text or "").split())
    knowns: list[str] = []
    unknowns: list[str] = []

    match = re.search(
        r"\bi know (?P<known>.+?) but (?:i )?(?:do not|don't|dont) know (?P<unknown>.+?)(?:$|,|;)",
        text,
        re.IGNORECASE,
    )
    if match:
        knowns.append(match.group("known").strip(" ."))
        unknowns.append(match.group("unknown").strip(" ."))
    for token in re.findall(r"\bwhat is ([^?.,;]+)", text, flags=re.IGNORECASE):
        clean = token.strip(" .")
        if clean:
            unknowns.append(clean)

    if (knowns or unknowns) or runtime is None or not runtime.health().available:
        merged_knowns = list(dict.fromkeys([*(getattr(prior_state, "knowns", []) or []), *knowns]))
        merged_unknowns = list(dict.fromkeys([*(getattr(prior_state, "unknowns", []) or []), *unknowns]))
        return merged_knowns, merged_unknowns

    prompt = (
        "Extract what the learner already knows and what they are missing.\n"
        "Return only concrete concepts or capabilities.\n"
        f"Learner text: {text}\n"
        f"Current artifact excerpt: {current_artifact[:2000]}\n"
        f"Prior knowns: {getattr(prior_state, 'knowns', [])}\n"
        f"Prior unknowns: {getattr(prior_state, 'unknowns', [])}\n"
    )
    draft = runtime.generate_draft(
        KnownUnknownInferenceDraft,
        prompt,
        source_scope="prerequisites:knowns_unknowns",
        model=runtime.config.model_roles.fast_inference or runtime.config.model_roles.default,
        max_output_tokens=4000,
    ).payload
    merged_knowns = list(dict.fromkeys([*(getattr(prior_state, "knowns", []) or []), *draft.knowns]))
    merged_unknowns = list(dict.fromkeys([*(getattr(prior_state, "unknowns", []) or []), *draft.unknowns]))
    return merged_knowns, merged_unknowns


def build_prerequisite_chain(
    domain: str,
    target: str,
    knowns: list[str],
    unknowns: list[str],
    *,
    runtime: Optional[LLMRuntime] = None,
) -> list[str]:
    known_chain = _resolve_known_chain(domain, target)
    if known_chain is not None:
        return known_chain
    if runtime is None or not runtime.health().available:
        raise RuntimeError("Unknown domains require a working LLM for prerequisite tracing.")
    prompt = (
        "Build a prerequisite chain for this learning target.\n"
        "Return 5-10 ordered steps from foundational primitives to the requested target.\n"
        f"Domain: {domain}\n"
        f"Target: {target}\n"
        f"Knowns: {knowns}\n"
        f"Unknowns: {unknowns}\n"
        f"Generic scaffold only as a hint: {_GENERIC_CHAIN}\n"
    )
    draft = runtime.generate_draft(
        PrerequisiteChainDraft,
        prompt,
        source_scope=f"prerequisites:chain:{domain}:{target[:80]}",
        model=runtime.config.model_roles.fast_inference or runtime.config.model_roles.default,
        max_output_tokens=4000,
    ).payload
    return draft.chain or list(_GENERIC_CHAIN)


def choose_foundational_floor(
    chain: list[str],
    repeated_signal_count: int,
    *,
    knowns: list[str] | None = None,
    unknowns: list[str] | None = None,
) -> str:
    if not chain:
        return ""
    knowns = knowns or []
    unknowns = unknowns or []
    unknown_index = None
    for idx, step in enumerate(chain):
        lowered = step.lower()
        if any(item.lower() in lowered or lowered in item.lower() for item in unknowns):
            unknown_index = idx
            break
    if unknown_index is None:
        unknown_index = min(len(chain) - 1, 2)
    if repeated_signal_count <= 1:
        return chain[max(0, unknown_index - 1)]
    if repeated_signal_count == 2:
        return chain[max(0, unknown_index - 3)]
    return chain[0]


def suggest_restart_points(goal, current_node, control_state, *, domain: str = "", target: str = "", runtime: Optional[LLMRuntime] = None) -> list[str]:
    chain = build_prerequisite_chain(
        domain or getattr(goal, "domain", "") or target,
        target or getattr(current_node, "title", "") or getattr(goal, "title", ""),
        getattr(control_state, "knowns", []) or [],
        getattr(control_state, "unknowns", []) or [],
        runtime=runtime,
    )
    if not chain:
        return []
    repeated = max((getattr(control_state, "signal_counts", {}) or {}).values(), default=1)
    floor = choose_foundational_floor(
        chain,
        repeated,
        knowns=getattr(control_state, "knowns", []) or [],
        unknowns=getattr(control_state, "unknowns", []) or [],
    )
    floor_index = next((index for index, item in enumerate(chain) if item == floor), 0)
    anchors = [0, 1, 2, 4, 5, 8, floor_index]
    points: list[str] = []
    for index in anchors:
        if 0 <= index < len(chain):
            value = chain[index]
            if value not in points:
                points.append(value)
    return points[:6]
