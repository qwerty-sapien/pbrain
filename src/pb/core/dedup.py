"""Fuzzy dedup for tasks, todos, goals, and thoughts on input.

Prevents duplicate entries by checking new input against existing active
items using normalized token overlap and sequence matching.
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher
from typing import Any, Optional

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_SIMILARITY_THRESHOLD = 0.70


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_RE.findall(text.lower()))


def _similarity(a: str, b: str) -> float:
    na, nb = _normalize(a), _normalize(b)
    if na == nb:
        return 1.0

    if na in nb or nb in na:
        shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
        return 0.85 + 0.15 * (len(shorter) / len(longer))

    tokens_a, tokens_b = _token_set(a), _token_set(b)
    if not tokens_a or not tokens_b:
        return 0.0

    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    jaccard = len(intersection) / len(union)
    seq_ratio = SequenceMatcher(None, na, nb).ratio()
    return jaccard * 0.6 + seq_ratio * 0.4


def find_similar_task(
    new_title: str,
    existing: list[Any],
    *,
    threshold: float = _SIMILARITY_THRESHOLD,
) -> Optional[Any]:
    """Return the most similar existing task if above threshold, else None.

    Args:
        new_title: The title of the new task/todo being created.
        existing: List of task-like objects with a `.title` attribute.
        threshold: Minimum similarity score (0.0-1.0) to consider a match.

    Returns:
        The best-matching existing task, or None if no match above threshold.
    """
    if not new_title or not existing:
        return None

    best_score = 0.0
    best_match = None
    for item in existing:
        title = getattr(item, "title", "")
        if not title:
            continue
        score = _similarity(new_title, title)
        if score > best_score:
            best_score = score
            best_match = item

    if best_score >= threshold:
        return best_match
    return None
