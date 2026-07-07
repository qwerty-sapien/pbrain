# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared goal/track/domain scope resolution helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


# Function words and generic request framing. Overlap on these must NOT count as
# a topic match, otherwise a vague request like "about the paper" scores against
# every existing goal on filler words and gets adopted by an unrelated goal.
STOPWORDS = frozenset({
    "a", "an", "the", "of", "and", "or", "to", "in", "on", "for", "with",
    "about", "into", "from", "by", "as", "at", "is", "are", "am", "be",
    "been", "being", "this", "that", "these", "those", "i", "we", "you",
    "it", "he", "she", "they", "me", "us", "my", "our", "your", "their",
    "its", "do", "does", "did", "have", "has", "had", "want", "wants",
    "need", "needs", "like", "would", "could", "should", "can", "will",
    "please", "lets", "let", "learn", "learning", "study", "studying",
    "practise", "practice", "practising", "practicing", "teach", "teaching",
    "understand", "master", "how", "what", "why", "when", "where", "which",
    "who", "paper", "topic", "thing", "stuff", "some", "more", "help",
})


def content_tokens(text: str) -> set[str]:
    """Return meaningful (non-stopword, length>1) tokens from free text."""

    tokens = {token for token in re.split(r"[^a-z0-9]+", (text or "").lower()) if token}
    return {token for token in tokens if token not in STOPWORDS and len(token) > 1}


def _shares_content(needle_content: set[str], *haystacks: str) -> bool:
    """True when the needle shares at least one content token with a haystack."""

    if not needle_content:
        return False
    return any(needle_content & content_tokens(hay) for hay in haystacks if hay)


def match_goal(repo, subject: str, *, allowed_modes: Optional[Iterable[str]] = None):
    needle_content = content_tokens(subject)
    if not needle_content:
        return None
    allowed = {item.lower() for item in allowed_modes} if allowed_modes else None
    for goal in repo.list_goal_arcs(status=None):
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        if allowed is not None and mode not in allowed:
            continue
        if _shares_content(
            needle_content,
            getattr(goal, "title", ""),
            getattr(goal, "domain", ""),
            getattr(goal, "description", ""),
        ):
            return goal
    return None


def match_track(repo, subject: str):
    needle_content = content_tokens(subject)
    if not needle_content:
        return None
    for track in repo.list_tracks(active_only=True):
        if _shares_content(
            needle_content,
            getattr(track, "name", ""),
            getattr(track, "description", ""),
        ):
            return track
    return None


def matching_goals(repo, raw_request: str, *, limit: int = 3) -> list[dict[str, str]]:
    needle_content = content_tokens(raw_request)
    matches: list[dict[str, str]] = []
    if not needle_content:
        return matches
    for goal in repo.list_goal_arcs(status=None):
        if _shares_content(
            needle_content,
            getattr(goal, "title", ""),
            getattr(goal, "domain", ""),
            getattr(goal, "description", ""),
        ):
            matches.append(
                {
                    "title": getattr(goal, "title", ""),
                    "domain": getattr(goal, "domain", ""),
                    "mode": getattr(goal, "execution_mode", "mixed"),
                }
            )
        if len(matches) >= limit:
            break
    return matches


def list_knowledge_domains(vault_path: Optional[Path] = None) -> list[str]:
    if vault_path is None:
        try:
            from pb.vault.config import get_vault_path

            vault_path = get_vault_path()
        except Exception:
            return []
    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return []
    return sorted(
        entry.name
        for entry in knowledge_dir.iterdir()
        if entry.is_dir() and not entry.name.startswith(".") and (entry / "_state.md").exists()
    )


def match_domain_name(subject: str, *, vault_path: Optional[Path] = None, domains: Optional[list[str]] = None) -> str:
    lowered = (subject or "").lower().strip()
    if not lowered:
        return ""
    available = domains or list_knowledge_domains(vault_path)
    normalized_subject = _normalized(lowered)
    exact = next(
        (
            domain
            for domain in available
            if domain.lower() == lowered or _normalized(domain) == normalized_subject
        ),
        None,
    )
    if exact:
        return exact
    partial = next(
        (
            domain
            for domain in available
            if lowered in domain.lower()
            or domain.lower() in lowered
            or normalized_subject in _normalized(domain)
            or _normalized(domain) in normalized_subject
        ),
        None,
    )
    return partial or ""
