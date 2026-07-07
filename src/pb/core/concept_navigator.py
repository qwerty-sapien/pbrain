# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Hierarchical concept navigation over vault concept notes."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pb.core.context_scope import ContextScopeFilter
from pb.vault.lifecycle import read_frontmatter


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9][a-z0-9_+-]*", (text or "").lower())
        if len(token) >= 2
    }


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, tuple):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


@dataclass
class ConceptCandidate:
    id: str
    title: str
    domain: str = ""
    path: str = ""
    note_type: str = ""
    score: float = 0.0
    reason_labels: list[str] = field(default_factory=list)
    parents: list[str] = field(default_factory=list)
    children: list[str] = field(default_factory=list)
    relations: dict[str, list[str]] = field(default_factory=dict)
    aliases: list[str] = field(default_factory=list)
    encoding: dict[str, str] = field(default_factory=dict)

    @property
    def reason(self) -> str:
        return ", ".join(dict.fromkeys(self.reason_labels)) or "matched concept note"


class ConceptNavigator:
    """Build ranked broad/concept/subconcept choices from the vault graph."""

    def __init__(self, *, repo: Any = None, vault_path: Path | None = None, context_filter: ContextScopeFilter | None = None):
        self.repo = repo
        self.vault_path = vault_path
        self.context_filter = context_filter or ContextScopeFilter.from_repo(repo)

    def candidates(self, query: str = "", *, parent: str = "", limit: int = 12) -> list[ConceptCandidate]:
        rows = self._load_concepts()
        scoped, _excluded = self.context_filter.filter_items(
            rows,
            text=lambda item: " ".join(
                [
                    item.title,
                    item.path,
                    " ".join(item.aliases),
                    " ".join(sum(item.relations.values(), [])),
                    " ".join(item.encoding.values()),
                ]
            ),
            domain=lambda item: item.domain,
            concept_id=lambda item: item.id,
        )
        ranked: list[ConceptCandidate] = []
        query_tokens = _tokens(query)
        parent_tokens = _tokens(parent)
        for candidate in scoped:
            scored = self._score_candidate(candidate, query_tokens=query_tokens, parent_tokens=parent_tokens)
            if scored.score > 0 or not query_tokens and not parent_tokens:
                ranked.append(scored)
        ranked.sort(key=lambda item: (item.score, item.title.lower()), reverse=True)
        return ranked[:limit]

    def broad_targets(self, limit: int = 12) -> list[ConceptCandidate]:
        grouped: dict[str, ConceptCandidate] = {}
        for candidate in self.candidates(limit=200):
            label = candidate.domain or "knowledge"
            if label not in grouped:
                grouped[label] = ConceptCandidate(
                    id=f"domain:{label}",
                    title=label,
                    domain=label,
                    note_type="domain",
                    score=0.0,
                    reason_labels=["concept-note domain"],
                )
            grouped[label].score += max(candidate.score, 1.0)
            grouped[label].children.append(candidate.title)
        return sorted(grouped.values(), key=lambda item: (item.score, item.title), reverse=True)[:limit]

    def _score_candidate(
        self,
        candidate: ConceptCandidate,
        *,
        query_tokens: set[str],
        parent_tokens: set[str],
    ) -> ConceptCandidate:
        clone = ConceptCandidate(
            id=candidate.id,
            title=candidate.title,
            domain=candidate.domain,
            path=candidate.path,
            note_type=candidate.note_type,
            score=candidate.score,
            reason_labels=list(candidate.reason_labels),
            parents=list(candidate.parents),
            children=list(candidate.children),
            relations={key: list(value) for key, value in candidate.relations.items()},
            aliases=list(candidate.aliases),
            encoding=dict(candidate.encoding),
        )
        fields = {
            "title": _tokens(candidate.title),
            "alias": _tokens(" ".join(candidate.aliases)),
            "path": _tokens(candidate.path),
            "relations": _tokens(" ".join(sum(candidate.relations.values(), []))),
            "cue": _tokens(candidate.encoding.get("cue", "")),
            "example": _tokens(candidate.encoding.get("example", "")),
            "source_context": _tokens(candidate.encoding.get("source_context", "")),
            "future_use": _tokens(candidate.encoding.get("future_use", "")),
            "failure_mode": _tokens(candidate.encoding.get("failure_mode", "")),
            "parents": _tokens(" ".join(candidate.parents)),
            "children": _tokens(" ".join(candidate.children)),
            "domain": _tokens(candidate.domain),
        }
        if not query_tokens and not parent_tokens:
            clone.score = 1.0
            return clone
        if query_tokens:
            if query_tokens & fields["title"]:
                clone.score += 4.0
                clone.reason_labels.append("matched title")
            if query_tokens & fields["alias"]:
                clone.score += 3.5
                clone.reason_labels.append("matched cue")
            if query_tokens & fields["relations"]:
                clone.score += 2.25
                clone.reason_labels.append("matched graph neighbor")
            for facet, label in (
                ("failure_mode", "matched failure mode"),
                ("future_use", "matched future use"),
                ("source_context", "matched source context"),
                ("example", "matched concrete example"),
                ("cue", "matched cue"),
            ):
                if query_tokens & fields[facet]:
                    clone.score += 2.75
                    clone.reason_labels.append(label)
            if query_tokens & fields["path"]:
                clone.score += 1.2
                clone.reason_labels.append("matched source context")
            if query_tokens & fields["domain"]:
                clone.score += 0.9
                clone.reason_labels.append("matched domain")
        if parent_tokens:
            if parent_tokens & (fields["parents"] | fields["children"] | fields["relations"]):
                clone.score += 3.0
                clone.reason_labels.append("matched graph neighbor")
            if parent_tokens & fields["domain"]:
                clone.score += 1.0
        if self.context_filter.locked and clone.score > 0:
            clone.reason_labels.append("matched locked context")
        return clone

    def _load_concepts(self) -> list[ConceptCandidate]:
        if self.vault_path is None:
            return []
        root = self.vault_path / "knowledge"
        if not root.exists():
            return []
        rows: list[ConceptCandidate] = []
        for path in sorted(root.rglob("*.md")):
            if path.name.startswith("."):
                continue
            try:
                content = path.read_text(encoding="utf-8", errors="replace")
                fm, body = read_frontmatter(content)
            except Exception:
                continue
            note_type = str(fm.get("type", "") or "").strip()
            if note_type not in {"concept", "moc", "example"}:
                continue
            relations_raw = fm.get("relations", {}) if isinstance(fm.get("relations", {}), dict) else {}
            relations = {str(key): _as_list(value) for key, value in relations_raw.items()}
            encoding_raw = fm.get("encoding", {}) if isinstance(fm.get("encoding", {}), dict) else {}
            encoding = {str(key): str(value).strip() for key, value in encoding_raw.items() if str(value).strip()}
            title = str(fm.get("title") or path.stem).strip()
            domain = str(fm.get("domain") or path.parent.name).strip()
            candidate = ConceptCandidate(
                id=str(fm.get("id") or f"concept:{domain}:{path.stem}"),
                title=title,
                domain=domain,
                path=str(path.relative_to(self.vault_path)),
                note_type=note_type,
                parents=_as_list(fm.get("parents")),
                children=_as_list(fm.get("children")),
                relations=relations,
                aliases=_as_list(fm.get("aliases")),
                encoding=encoding,
                reason_labels=["atomic concept note" if note_type == "concept" else f"{note_type} note"],
            )
            if "failure_modes" in relations and relations["failure_modes"]:
                candidate.reason_labels.append("has failure modes")
            if "examples" in relations and relations["examples"]:
                candidate.reason_labels.append("has examples")
            if body:
                # Keep body search light; exact FTS remains the indexer's job.
                candidate.aliases.extend(re.findall(r"`([^`]{3,80})`", body)[:4])
            rows.append(candidate)
        return rows
