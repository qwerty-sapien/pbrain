# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared goal/track/domain scope resolution helpers."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable, Optional


def _normalized(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (value or "").lower())


def match_goal(repo, subject: str, *, allowed_modes: Optional[Iterable[str]] = None):
    needle = (subject or "").lower().strip()
    if not needle:
        return None
    allowed = {item.lower() for item in allowed_modes} if allowed_modes else None
    for goal in repo.list_goal_arcs(status=None):
        mode = (getattr(goal, "execution_mode", "") or "mixed").lower()
        if allowed is not None and mode not in allowed:
            continue
        haystacks = [
            getattr(goal, "title", ""),
            getattr(goal, "domain", ""),
            getattr(goal, "description", ""),
        ]
        lowered = [item.lower() for item in haystacks if item]
        if any(needle in hay or hay in needle for hay in lowered):
            return goal
    return None


def match_track(repo, subject: str):
    needle = (subject or "").lower().strip()
    if not needle:
        return None
    for track in repo.list_tracks(active_only=True):
        haystacks = [getattr(track, "name", ""), getattr(track, "description", "")]
        lowered = [item.lower() for item in haystacks if item]
        if any(needle in hay or hay in needle for hay in lowered):
            return track
    return None


def matching_goals(repo, raw_request: str, *, limit: int = 3) -> list[dict[str, str]]:
    needle = (raw_request or "").strip().lower()
    matches: list[dict[str, str]] = []
    if not needle:
        return matches
    for goal in repo.list_goal_arcs(status=None):
        haystacks = [
            getattr(goal, "title", ""),
            getattr(goal, "domain", ""),
            getattr(goal, "description", ""),
        ]
        lowered = [item.lower() for item in haystacks if item]
        if any(needle in hay or hay in needle for hay in lowered):
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
