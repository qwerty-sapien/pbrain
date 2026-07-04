# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Deterministic context-lock filtering for retrieval surfaces."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, TypeVar

T = TypeVar("T")

_TOKEN_STOPWORDS = {
    "and",
    "from",
    "general",
    "learning",
    "material",
    "only",
    "pdf",
    "source",
    "the",
    "uploaded",
    "use",
    "vault",
}


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", (text or "").lower())
        if len(token) >= 3 and token not in _TOKEN_STOPWORDS
    }


@dataclass(frozen=True)
class ContextScopeFilter:
    """Small allowlist derived from the active context lock."""

    locked: bool = False
    label: str = ""
    source_bundle_id: str = ""
    source_refs: tuple[str, ...] = ()
    domain_id: str = ""
    scope_boundary: str = ""
    tokens: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def from_repo(cls, repo: Any) -> "ContextScopeFilter":
        try:
            scope = repo.get_locked_context() if repo is not None else None
        except Exception:
            scope = None
        return cls.from_scope(scope)

    @classmethod
    def from_scope(cls, scope: Any) -> "ContextScopeFilter":
        if scope is None or not bool(getattr(scope, "locked", False)):
            return cls()
        source_refs = tuple(str(item) for item in list(getattr(scope, "source_refs", []) or []) if str(item).strip())
        label = str(getattr(scope, "label", "") or "")
        domain_id = str(getattr(scope, "domain_id", "") or "")
        boundary = str(getattr(scope, "scope_boundary", "") or "")
        bundle_id = str(getattr(scope, "source_bundle_id", "") or "")
        token_source = " ".join([label, domain_id, boundary, bundle_id, " ".join(source_refs)])
        return cls(
            locked=True,
            label=label,
            source_bundle_id=bundle_id,
            source_refs=source_refs,
            domain_id=domain_id,
            scope_boundary=boundary,
            tokens=frozenset(_tokens(token_source)),
        )

    def explain(self) -> dict[str, Any]:
        return {
            "locked": self.locked,
            "label": self.label,
            "source_bundle_id": self.source_bundle_id,
            "source_refs": list(self.source_refs),
            "domain_id": self.domain_id,
            "scope_boundary": self.scope_boundary,
            "tokens": sorted(self.tokens),
        }

    def allows(
        self,
        *,
        text: str = "",
        domain: str = "",
        source_ref: str = "",
        source_refs: Iterable[str] | None = None,
        source_bundle_id: str = "",
        concept_id: str = "",
    ) -> bool:
        """Return whether a candidate may be used under the current lock."""

        if not self.locked:
            return True
        candidate_refs = {str(source_ref or "").strip(), *(str(item).strip() for item in (source_refs or []))}
        candidate_refs.discard("")
        if candidate_refs and any(ref in self.source_refs for ref in candidate_refs):
            return True
        if source_bundle_id and self.source_bundle_id and source_bundle_id == self.source_bundle_id:
            return True
        if domain and self.domain_id and domain.strip().lower() == self.domain_id.strip().lower():
            return True
        haystack = " ".join([text or "", domain or "", concept_id or "", source_bundle_id or "", " ".join(candidate_refs)])
        candidate_tokens = _tokens(haystack)
        if not candidate_tokens:
            return False
        return bool(candidate_tokens & self.tokens)

    def filter_items(
        self,
        items: Iterable[T],
        *,
        text: Callable[[T], str] = lambda _item: "",
        domain: Callable[[T], str] = lambda _item: "",
        source_ref: Callable[[T], str] = lambda _item: "",
        source_refs: Callable[[T], Iterable[str]] = lambda _item: (),
        source_bundle_id: Callable[[T], str] = lambda _item: "",
        concept_id: Callable[[T], str] = lambda _item: "",
    ) -> tuple[list[T], list[dict[str, str]]]:
        """Return allowed items plus compact exclusion diagnostics."""

        allowed: list[T] = []
        excluded: list[dict[str, str]] = []
        for item in items:
            ok = self.allows(
                text=text(item),
                domain=domain(item),
                source_ref=source_ref(item),
                source_refs=source_refs(item),
                source_bundle_id=source_bundle_id(item),
                concept_id=concept_id(item),
            )
            if ok:
                allowed.append(item)
            else:
                excluded.append(
                    {
                        "reason": "excluded by context lock",
                        "label": (text(item) or domain(item) or concept_id(item) or source_ref(item))[:160],
                    }
                )
        return allowed, excluded
