# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Strict matching helpers for tasks, notes, and other named entities."""

from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher
import json
import re
from typing import Iterable

from pb.llm.gemini import FLASH_LITE_MODEL, get_client


CONFIDENCE_THRESHOLD = 0.80
GAP_THRESHOLD = 0.10

_TOKEN_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MatchCandidate:
    """Searchable candidate record."""

    key: str
    label: str
    text: str = ""


@dataclass(frozen=True)
class StrictMatchResult:
    """Outcome of a strict matching pass."""

    matched_index: int | None
    confidence: float
    gap: float
    suggestions: list[int]
    reason: str = ""
    source: str = "local"

    @property
    def accepted(self) -> bool:
        return (
            self.matched_index is not None
            and self.confidence >= CONFIDENCE_THRESHOLD
            and self.gap >= GAP_THRESHOLD
        )


def normalize_match_text(text: str) -> str:
    """Lowercase and collapse whitespace for strict matching."""
    return " ".join((text or "").lower().split())


def build_candidate_text(parts: Iterable[str]) -> str:
    """Join non-empty candidate parts into one searchable string."""
    return " | ".join(part.strip() for part in parts if part and part.strip())


def resolve_strict_match(
    query: str,
    candidates: list[MatchCandidate],
    *,
    allow_llm: bool = True,
) -> StrictMatchResult:
    """Resolve a user query to one candidate or refuse to guess."""
    deterministic = _deterministic_match(query, candidates)
    if deterministic is not None:
        return deterministic

    if allow_llm:
        llm_result = _llm_match(query, candidates)
        if llm_result is not None:
            return llm_result

    return _local_match(query, candidates)


def _deterministic_match(query: str, candidates: list[MatchCandidate]) -> StrictMatchResult | None:
    normalized_query = normalize_match_text(query)
    if not normalized_query:
        return StrictMatchResult(None, 0.0, 0.0, [], reason="empty_query")

    exact_matches = [
        index
        for index, candidate in enumerate(candidates)
        if normalized_query in {
            normalize_match_text(candidate.key),
            normalize_match_text(candidate.label),
        }
    ]
    if len(exact_matches) == 1:
        return StrictMatchResult(
            matched_index=exact_matches[0],
            confidence=1.0,
            gap=1.0,
            suggestions=[exact_matches[0]],
            reason="exact_match",
            source="deterministic",
        )

    prefix_matches = [
        index
        for index, candidate in enumerate(candidates)
        if normalize_match_text(candidate.key).startswith(normalized_query)
        or normalize_match_text(candidate.label).startswith(normalized_query)
    ]
    if len(prefix_matches) == 1 and len(normalized_query) >= 4:
        return StrictMatchResult(
            matched_index=prefix_matches[0],
            confidence=0.99,
            gap=0.99,
            suggestions=[prefix_matches[0]],
            reason="unique_prefix_match",
            source="deterministic",
        )
    return None


def _llm_match(query: str, candidates: list[MatchCandidate]) -> StrictMatchResult | None:
    client = get_client()
    if not client.is_available() or not candidates:
        return None

    candidate_lines = []
    for index, candidate in enumerate(candidates, start=1):
        candidate_lines.append(
            f"{index}. label={candidate.label}\n"
            f"   key={candidate.key}\n"
            f"   context={candidate.text[:420]}"
        )

    prompt = (
        "Match the user's request to exactly one candidate only when you are highly confident.\n"
        "If confidence is below 0.80, refuse to guess.\n"
        "Return JSON only with keys:\n"
        'best_index (1-based integer or null), confidence (0-1), '
        'runner_up_index (1-based integer or null), runner_up_confidence (0-1), reason (short string).\n\n'
        f"User query:\n{query.strip()}\n\n"
        f"Candidates:\n{chr(10).join(candidate_lines)}\n"
    )

    try:
        raw = client.generate_with_model(prompt, FLASH_LITE_MODEL)
    except Exception:
        return None
    if not raw:
        return None

    payload = _extract_json_object(raw)
    if not isinstance(payload, dict):
        return None

    best_index = payload.get("best_index")
    runner_up_index = payload.get("runner_up_index")
    confidence = _clamp_float(payload.get("confidence"))
    runner_up_confidence = _clamp_float(payload.get("runner_up_confidence"))
    reason = str(payload.get("reason", "")).strip()

    if best_index is not None:
        try:
            best_index = int(best_index) - 1
        except Exception:
            best_index = None
    if runner_up_index is not None:
        try:
            runner_up_index = int(runner_up_index) - 1
        except Exception:
            runner_up_index = None

    suggestions = [
        idx
        for idx in [best_index, runner_up_index]
        if isinstance(idx, int) and 0 <= idx < len(candidates)
    ]
    if not suggestions:
        suggestions = _top_local_suggestions(query, candidates)

    gap = max(0.0, confidence - runner_up_confidence)
    if best_index is None or not (0 <= best_index < len(candidates)):
        return StrictMatchResult(
            matched_index=None,
            confidence=confidence,
            gap=gap,
            suggestions=suggestions[:3],
            reason=reason or "llm_abstained",
            source="llm",
        )

    return StrictMatchResult(
        matched_index=best_index,
        confidence=confidence,
        gap=gap,
        suggestions=suggestions[:3],
        reason=reason or "llm_match",
        source="llm",
    )


def _local_match(query: str, candidates: list[MatchCandidate]) -> StrictMatchResult:
    normalized_query = normalize_match_text(query)
    if not normalized_query or not candidates:
        return StrictMatchResult(None, 0.0, 0.0, [], reason="empty_or_missing_candidates")

    scores: list[tuple[float, int]] = []
    for index, candidate in enumerate(candidates):
        label = normalize_match_text(candidate.label)
        text = normalize_match_text(candidate.text or candidate.label)
        score = _local_score(normalized_query, label, text)
        scores.append((score, index))

    scores.sort(reverse=True)
    best_score, best_index = scores[0]
    second_score = scores[1][0] if len(scores) > 1 else 0.0
    suggestions = [index for _, index in scores[:3]]
    gap = max(0.0, best_score - second_score)

    if best_score < CONFIDENCE_THRESHOLD or gap < GAP_THRESHOLD:
        return StrictMatchResult(
            matched_index=None,
            confidence=best_score,
            gap=gap,
            suggestions=suggestions,
            reason="low_confidence_local_match",
            source="local",
        )

    return StrictMatchResult(
        matched_index=best_index,
        confidence=best_score,
        gap=gap,
        suggestions=suggestions,
        reason="local_match",
        source="local",
    )


def _local_score(query: str, label: str, text: str) -> float:
    if query == label:
        return 1.0
    if query in {label, text}:
        return 0.96
    if query in label:
        return 0.90

    query_tokens = set(_TOKEN_RE.findall(query))
    label_tokens = set(_TOKEN_RE.findall(label))
    text_tokens = set(_TOKEN_RE.findall(text))
    if not query_tokens:
        return 0.0

    label_overlap = len(query_tokens & label_tokens) / len(query_tokens)
    text_overlap = len(query_tokens & text_tokens) / len(query_tokens)
    token_overlap = max(label_overlap, text_overlap)
    label_ratio = SequenceMatcher(None, query, label).ratio()
    text_ratio = SequenceMatcher(None, query, text).ratio()

    score = (token_overlap * 0.60) + (label_ratio * 0.30) + (text_ratio * 0.10)
    if token_overlap == 1.0 and len(query_tokens) >= 2:
        score = max(score, 0.88)
    elif token_overlap < 0.5:
        score *= 0.65
    return max(0.0, min(0.95, score))


def _top_local_suggestions(query: str, candidates: list[MatchCandidate]) -> list[int]:
    ranked = sorted(
        (
            (_local_score(normalize_match_text(query), normalize_match_text(candidate.label), normalize_match_text(candidate.text or candidate.label)), index)
            for index, candidate in enumerate(candidates)
        ),
        reverse=True,
    )
    return [index for _, index in ranked[:3]]


def _extract_json_object(raw: str) -> dict | None:
    stripped = raw.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except Exception:
            return None
    start = stripped.find("{")
    end = stripped.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    try:
        return json.loads(stripped[start : end + 1])
    except Exception:
        return None


def _clamp_float(value: object) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    return max(0.0, min(1.0, parsed))
