# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Visible ref and alias helpers for user-facing entity identifiers."""

from __future__ import annotations

import re
from typing import Any

_UUID_RE = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$",
    re.IGNORECASE,
)


def is_uuid_like(value: str) -> bool:
    """Return True when a value looks like a UUID string."""
    return bool(_UUID_RE.fullmatch((value or "").strip()))


def visible_ref_key(entity_kind: str) -> str:
    """Return the generated-names key storing the visible ref for an entity."""
    return "session_slug" if entity_kind == "session" else "slug"


def _tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def normalize_visible_ref(
    text: str,
    *,
    fallback: str = "item",
    prefix: str = "",
    limit: int = 26,
) -> str:
    """Build a short visible ref with at most one underscore."""
    tokens = _tokens(text)
    fallback_token = (_tokens(fallback) or ["item"])[0]
    prefix_token = (_tokens(prefix) or [""])[0]

    if prefix_token:
        tail_source = tokens or [fallback_token]
        tail = "".join(tail_source)[: max(1, limit - len(prefix_token) - 1)]
        slug = f"{prefix_token}_{tail}".strip("_")
    elif not tokens:
        slug = fallback_token[:limit]
    elif len(tokens) == 1:
        slug = tokens[0][:limit]
    else:
        head = tokens[0]
        tail = "".join(tokens[1:])
        available = max(1, limit - len(head) - 1)
        slug = f"{head}_{tail[:available]}".strip("_")

    slug = re.sub(r"[^a-z0-9_]", "", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    if slug.count("_") > 1:
        first, rest = slug.split("_", 1)
        slug = f"{first}_{rest.replace('_', '')}"
    return slug[:limit].rstrip("_") or fallback_token[:limit] or "item"


def derive_visible_ref(
    entity_kind: str,
    *,
    title: str = "",
    parent_ref: str = "",
    fallback: str = "",
) -> str:
    """Derive a base visible ref for one entity kind."""
    prefix = ""
    base_fallback = fallback or entity_kind
    if entity_kind == "session":
        prefix = "sess"
        if not title and parent_ref:
            title = parent_ref
    elif parent_ref and not title:
        title = parent_ref
    return normalize_visible_ref(title, fallback=base_fallback, prefix=prefix)


def display_ref(entity: Any, entity_kind: str, *, parent_ref: str = "") -> str:
    """Return the stored visible ref or derive a deterministic fallback."""
    generated = dict(getattr(entity, "generated_names", {}) or {})
    key = visible_ref_key(entity_kind)
    stored = str(generated.get(key, "") or "").strip()
    if stored:
        return stored

    title = (
        getattr(entity, "title", "")
        or getattr(entity, "name", "")
        or getattr(entity, "subject_scope", "")
        or getattr(entity, "intended_outcome", "")
    )
    fallback = str(getattr(entity, "id", "") or entity_kind)
    return derive_visible_ref(
        entity_kind,
        title=title,
        parent_ref=parent_ref,
        fallback=fallback,
    )


def dedupe_visible_ref(base: str, existing: set[str], *, limit: int = 26) -> str:
    """Resolve ref collisions deterministically with short numeric suffixes."""
    candidate = base[:limit].rstrip("_")
    if candidate not in existing:
        return candidate
    for suffix in range(2, 1000):
        suffix_text = str(suffix)
        trimmed = candidate[: max(1, limit - len(suffix_text))]
        deduped = f"{trimmed}{suffix_text}"
        if deduped not in existing:
            return deduped
    return candidate[:limit]
