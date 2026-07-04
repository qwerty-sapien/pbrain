# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Adaptive recent-interest hierarchy for `pb next`."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Iterable

from pb.core.context_scope import ContextScopeFilter
from pb.core.models import utc_now
from pb.storage.database import get_connection

_STOP_PREFIXES = (
    "study",
    "practise",
    "practice",
    "teach",
    "learn",
    "review",
    "revise",
    "work on",
    "continue",
    "finish",
    "start",
)

_SCAFFOLD_LABELS = {
    "_state",
    "state",
    "general",
    "general learning",
    "learning",
}

_GENERIC_DOMAIN_LABELS = {
    "general",
    "general learning",
    "learning",
    "physical",
}


def _parse_dt(value: object) -> datetime | None:
    if isinstance(value, datetime):
        return value
    text = str(value or "").strip()
    if not text:
        return None
    try:
        return datetime.fromisoformat(text.replace("Z", "+00:00")).replace(tzinfo=None)
    except Exception:
        return None


def _norm(text: str) -> str:
    lowered = re.sub(r"[_/]+", " ", (text or "").strip().lower())
    lowered = re.sub(r"[^a-z0-9+#& ]+", " ", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    for prefix in _STOP_PREFIXES:
        if lowered.startswith(prefix + " "):
            lowered = lowered[len(prefix) + 1 :].strip()
    return lowered


def _title(text: str) -> str:
    aliases = {
        "ai": "AI",
        "ml": "ML",
        "ai ml": "AI & ML",
        "ai and ml": "AI & ML",
        "machine learning": "AI & ML",
        "llm": "LLMs",
        "c": "C",
        "c++": "C++",
        "rust": "Rust",
        "ruby": "Ruby",
        "python": "Python",
    }
    normalized = _norm(text)
    if normalized in aliases:
        return aliases[normalized]
    parts = []
    for token in normalized.split():
        if token in aliases:
            parts.append(aliases[token])
        elif token in {"and", "of", "the", "for"}:
            parts.append(token)
        else:
            parts.append(token[:1].upper() + token[1:])
    return " ".join(parts).strip()


def _is_scaffold_label(text: str) -> bool:
    return _norm(text) in _SCAFFOLD_LABELS


def _is_generic_domain(text: str) -> bool:
    return _norm(text) in _GENERIC_DOMAIN_LABELS


def _filename_label(text: str) -> str:
    stem = Path(text or "").stem
    return re.sub(r"[_-]+", " ", stem).strip()


_PATTERN_TAXONOMY: tuple[tuple[tuple[str, ...], str, str, str], ...] = (
    (("quantum field", " qft ", "casimir", "bosonic string", "field theory"), "Sciences", "Physics", "Quantum Field Theory"),
    (("general relativity", "einstein", "schwarzschild", "ricci flow", "semi riemannian", "curvature tensor"), "Sciences", "Physics and Geometry", "Differential Geometry / GR"),
    (("differential geometry", "manifold", "tangent space", "vector field", "tensor field", "christoffel", "geodesic"), "Mathematics", "Geometry", "Differential Geometry"),
    (("zeta", "riemann", "analytic continuation", "mellin", "jacobi theta", "divergent series", "regularization", "regularisation"), "Mathematics", "Analysis", "Zeta Function Regularization"),
    (("pointer", "heap", "recursion", "computer science", " cs ", "bash", "shell script", "script"), "Technology", "Computer Science", ""),
    (("causal", "causa", "robust agent", "agents learn"), "Technology", "AI & ML", "AI & ML"),
    (("rust", "c++", " c ", "zig", "systems programming", "operating system", "compiler"), "Technology", "System Languages", ""),
    (("python", "ruby", "javascript", "typescript", "java", "rails", "django", "node"), "Technology", "High-Level Programming Languages", ""),
    (("cyber", "security", "cryptography", "reverse engineering", "network security"), "Technology", "Cybersecurity", "Cybersecurity"),
    (("machine learning", "deep learning", " ai ", " ml ", "llm", "transformer", "neural"), "Technology", "AI & ML", "AI & ML"),
    (("finance", "financial", "valuation", "modeling", "modelling", "dcf", "portfolio"), "Finance", "Financial Modeling", "Financial Modeling"),
    (("group theory", "pure maths", "pure math", "algebra", "topology", "analysis", "number theory"), "Mathematics", "Pure Mathematics", ""),
    (("statistics", "probability", "bayes", "regression"), "Mathematics", "Statistics", ""),
    (("music", "harmony", "counterpoint", "piano", "composition", "ear training"), "Arts", "Music Theory", "Music Theory"),
    (("history",), "Humanities", "Humanities", "History"),
    (("literature", "poetry", "novel"), "Humanities", "Humanities", "Literature"),
    (("philosophy", "ethics", "epistemology", "metaphysics"), "Humanities", "Humanities", "Philosophy"),
    (("physics", "chemistry", "biology", "neuroscience"), "Sciences", "Sciences", ""),
)


@dataclass(frozen=True)
class InterestSignal:
    label: str
    source_kind: str
    occurred_at: datetime
    weight: float = 1.0
    source_id: str = ""
    source_ref: str = ""
    domain: str = ""
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class InterestNode:
    ref: str
    label: str
    level: str
    parent_ref: str = ""
    top_label: str = ""
    children: list[str] = field(default_factory=list)
    score: float = 0.0
    signal_count: int = 0
    last_seen_at: datetime | None = None
    archived: bool = False
    reason_labels: list[str] = field(default_factory=list)
    command: str = ""

    @property
    def reason(self) -> str:
        return ", ".join(dict.fromkeys(self.reason_labels)) or "recent learning signal"


@dataclass(frozen=True)
class InterestHierarchyResult:
    nodes: tuple[InterestNode, ...]
    immediate: tuple[Any, ...] = ()
    excluded_by_lock: tuple[dict[str, str], ...] = ()
    archive: bool = False
    within: str = ""


def classify_interest_label(label: str) -> tuple[str, str, str]:
    """Return (top, parent, leaf) for a raw activity label."""

    normalized = f" {_norm(label)} "
    language_leafs = {
        "rust": ("Technology", "System Languages", "Rust"),
        "c++": ("Technology", "System Languages", "C++"),
        " c ": ("Technology", "System Languages", "C"),
        "zig": ("Technology", "System Languages", "Zig"),
        "python": ("Technology", "High-Level Programming Languages", "Python"),
        "ruby": ("Technology", "High-Level Programming Languages", "Ruby"),
        "javascript": ("Technology", "High-Level Programming Languages", "JavaScript"),
        "typescript": ("Technology", "High-Level Programming Languages", "TypeScript"),
    }
    for pattern, result in language_leafs.items():
        if pattern in normalized:
            return result
    if " pointer" in normalized or " pointers " in normalized:
        return "Technology", "Computer Science", "Pointers"
    if " recursion " in normalized:
        return "Technology", "Computer Science", "Recursion"
    if " heap " in normalized:
        return "Technology", "Computer Science", "Heap"
    for patterns, top, parent, fixed_leaf in _PATTERN_TAXONOMY:
        if any(pattern in normalized for pattern in patterns):
            leaf = fixed_leaf or _title(label)
            if top == "Mathematics" and "group theory" in normalized:
                leaf = "Group Theory"
            return top, parent, leaf
    cleaned = _title(label)
    if not cleaned:
        return "Learning", "Other Learning", "Learning"
    return "Learning", "Other Learning", cleaned


def _signal_id(signal: InterestSignal) -> str:
    basis = "|".join(
        [
            signal.source_kind,
            signal.source_id,
            signal.source_ref,
            _norm(signal.label),
            signal.occurred_at.isoformat(),
        ]
    )
    return hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def persist_interest_signals(signals: Iterable[InterestSignal]) -> None:
    """Persist rebuildable interest signals as a cache."""

    now = utc_now().isoformat()
    with get_connection() as conn:
        for signal in signals:
            top, parent, leaf = classify_interest_label(signal.label)
            conn.execute(
                """
                INSERT OR REPLACE INTO interest_signals (
                    id, label, normalized_label, parent_label, top_label,
                    source_kind, source_id, source_ref, weight, occurred_at,
                    evidence_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    _signal_id(signal),
                    leaf,
                    _norm(leaf),
                    parent,
                    top,
                    signal.source_kind,
                    signal.source_id,
                    signal.source_ref,
                    signal.weight,
                    signal.occurred_at.isoformat(),
                    json.dumps(signal.evidence, ensure_ascii=True, default=str),
                    now,
                ),
            )
        conn.commit()


class InterestHierarchyService:
    """Build compact, adaptive interest directions from recent local activity."""

    def __init__(self, repo: Any, *, vault_path: Path | None = None, now: datetime | None = None):
        self.repo = repo
        self.vault_path = vault_path
        self.now = now or utc_now()

    def build(
        self,
        *,
        archive: bool = False,
        within: str = "",
        limit: int = 6,
        immediate: Iterable[Any] = (),
        context_filter: ContextScopeFilter | None = None,
    ) -> InterestHierarchyResult:
        context_filter = context_filter or ContextScopeFilter.from_repo(self.repo)
        raw_signals = [
            signal
            for signal in self._collect_signals()
            if _norm(signal.label) and not _is_scaffold_label(signal.label)
        ]
        allowed, excluded = context_filter.filter_items(
            raw_signals,
            text=lambda item: item.label,
            domain=lambda item: item.domain,
            source_ref=lambda item: item.source_ref,
        )
        persist_interest_signals(allowed)
        signals = self._partition_signals(allowed, archive=archive)
        nodes = self._select_nodes(signals, within=within, limit=limit)
        return InterestHierarchyResult(
            nodes=tuple(nodes),
            immediate=tuple(immediate),
            excluded_by_lock=tuple(excluded),
            archive=archive,
            within=within,
        )

    def _partition_signals(self, signals: list[InterestSignal], *, archive: bool) -> list[InterestSignal]:
        cutoff = self.now - timedelta(days=30)
        grouped: dict[str, list[InterestSignal]] = {}
        for signal in signals:
            _top, _parent, leaf = classify_interest_label(signal.label)
            grouped.setdefault(_norm(leaf), []).append(signal)
        selected: list[InterestSignal] = []
        for group in grouped.values():
            last_seen = max(signal.occurred_at for signal in group)
            is_archived = last_seen < cutoff
            if archive == is_archived:
                selected.extend(group)
        return selected

    def _collect_signals(self) -> list[InterestSignal]:
        signals: list[InterestSignal] = []
        signals.extend(self._session_signals())
        signals.extend(self._goal_signals())
        signals.extend(self._track_signals())
        signals.extend(self._context_source_signals())
        signals.extend(self._concept_confidence_signals())
        signals.extend(self._concept_note_signals())
        return [signal for signal in signals if _norm(signal.label) and not _is_scaffold_label(signal.label)]

    def _session_signals(self) -> list[InterestSignal]:
        rows: list[InterestSignal] = []
        try:
            tasks = self.repo.list_tasks(include_archived=True)
        except TypeError:
            tasks = self.repo.list_tasks()
        except Exception:
            tasks = []
        for task in tasks:
            try:
                sessions = self.repo.list_sessions_for_task(task.id)
            except Exception:
                sessions = []
            for session in sessions:
                label = getattr(session, "subject_scope", "") or getattr(task, "title", "")
                occurred = _parse_dt(getattr(session, "start_at", None)) or self.now
                rows.append(
                    InterestSignal(
                        label=label,
                        source_kind="session",
                        source_id=getattr(session, "id", ""),
                        occurred_at=occurred,
                        weight=1.25 if occurred >= self.now - timedelta(days=7) else 1.0,
                        domain=self._task_domain(task),
                        evidence={"branch": getattr(session, "branch", "study"), "task_title": getattr(task, "title", "")},
                    )
                )
        return rows

    def _goal_signals(self) -> list[InterestSignal]:
        rows: list[InterestSignal] = []
        try:
            goals = self.repo.list_goal_arcs(status="active")
        except Exception:
            goals = []
        for goal in goals:
            domain = str(getattr(goal, "domain", "") or "")
            title = str(getattr(goal, "title", "") or "")
            generated = getattr(goal, "generated_names", {}) or {}
            generated_topic = ""
            if isinstance(generated, dict):
                frontmatter = generated.get("frontmatter") or {}
                if isinstance(frontmatter, dict):
                    generated_topic = str(frontmatter.get("topic") or frontmatter.get("domain") or "")
                generated_topic = generated_topic or str(generated.get("display_title") or generated.get("goal_title") or "")
            label = title if _is_generic_domain(domain) else (generated_topic or domain or title)
            rows.append(
                InterestSignal(
                    label=label,
                    source_kind="goal",
                    source_id=getattr(goal, "id", ""),
                    occurred_at=_parse_dt(getattr(goal, "updated_at", None)) or self.now,
                    weight=1.5,
                    domain=domain if not _is_generic_domain(domain) else "",
                    evidence={"title": getattr(goal, "title", ""), "reason": "active goal"},
                )
            )
        return rows

    def _track_signals(self) -> list[InterestSignal]:
        rows: list[InterestSignal] = []
        try:
            tracks = self.repo.list_tracks(active_only=True)
        except Exception:
            tracks = []
        for track in tracks:
            rows.append(
                InterestSignal(
                    label=getattr(track, "name", ""),
                    source_kind="track",
                    source_id=getattr(track, "id", ""),
                    occurred_at=_parse_dt(getattr(track, "updated_at", None)) or self.now,
                    weight=float(getattr(track, "priority_weight", 1.0) or 1.0),
                    domain=getattr(track, "name", ""),
                    evidence={"reason": "active track"},
                )
            )
        return rows

    def _context_source_signals(self) -> list[InterestSignal]:
        rows: list[InterestSignal] = []
        try:
            sources = self.repo.list_context_sources()
        except Exception:
            sources = []
        for source in sources:
            domain_name = str(source.get("domain_name") or "")
            filename = str(source.get("filename") or "")
            label = _filename_label(filename) if _is_generic_domain(domain_name) else (domain_name or _filename_label(filename))
            rows.append(
                InterestSignal(
                    label=label,
                    source_kind="source",
                    source_id=str(source.get("id", "")),
                    source_ref=str(source.get("source_ref", "")),
                    occurred_at=_parse_dt(source.get("updated_at")) or self.now,
                    weight=1.1,
                    domain=domain_name if not _is_generic_domain(domain_name) else "",
                    evidence={"filename": source.get("filename", ""), "reason": "source bundle"},
                )
            )
        try:
            bundles = self.repo.list_source_bundles()
        except Exception:
            bundles = []
        for bundle in bundles:
            domain_name = str(getattr(bundle, "domain_name", "") or "")
            bundle_name = str(getattr(bundle, "name", "") or "")
            rows.append(
                InterestSignal(
                    label=bundle_name if _is_generic_domain(domain_name) else (domain_name or bundle_name),
                    source_kind="source_bundle",
                    source_id=getattr(bundle, "id", ""),
                    source_ref=" ".join(getattr(bundle, "source_refs", []) or []),
                    occurred_at=_parse_dt(getattr(bundle, "updated_at", None)) or self.now,
                    weight=1.15,
                    domain=domain_name if not _is_generic_domain(domain_name) else "",
                    evidence={"name": getattr(bundle, "name", ""), "reason": "source bundle"},
                )
            )
        return rows

    def _concept_confidence_signals(self) -> list[InterestSignal]:
        rows: list[InterestSignal] = []
        try:
            records = self.repo.list_concept_confidence()
        except Exception:
            records = []
        for record in records:
            concept_id = str(getattr(record, "concept_id", "") or "")
            parts = concept_id.split(":")
            if len(parts) < 3:
                continue
            domain, slug = parts[1], parts[2]
            score = float(getattr(record, "confidence_score", 0.0) or 0.0)
            rows.append(
                InterestSignal(
                    label=f"{domain} {slug.replace('-', ' ')}",
                    source_kind="concept_confidence",
                    source_id=concept_id,
                    occurred_at=_parse_dt(getattr(record, "last_evidence_at", None)) or self.now,
                    weight=1.2 if score < 0.55 else 0.8,
                    domain=domain,
                    evidence={"confidence_score": score, "reason": "weak concept cluster" if score < 0.55 else "concept evidence"},
                )
            )
        return rows

    def _concept_note_signals(self) -> list[InterestSignal]:
        if self.vault_path is None:
            return []
        root = self.vault_path / "knowledge"
        if not root.exists():
            return []
        rows: list[InterestSignal] = []
        for path in sorted(root.rglob("*.md")):
            if path.name.startswith("."):
                continue
            if path.stem == "_state":
                continue
            try:
                stat = path.stat()
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            title = path.stem
            domain = path.parent.name if path.parent != root else ""
            match = re.search(r"^title:\s*(.+)$", text, flags=re.MULTILINE)
            if match:
                title = match.group(1).strip().strip('"')
            rows.append(
                InterestSignal(
                    label=title,
                    source_kind="concept_note",
                    source_id=str(path.relative_to(self.vault_path)),
                    occurred_at=datetime.fromtimestamp(stat.st_mtime),
                    weight=0.7,
                    domain=domain,
                    evidence={"path": str(path.relative_to(self.vault_path))},
                )
            )
        return rows

    def _task_domain(self, task: Any) -> str:
        for goal_id in list(getattr(task, "linked_goal_arc_ids", []) or []):
            try:
                goal = self.repo.get_goal_arc(goal_id)
            except Exception:
                goal = None
            if goal is not None and getattr(goal, "domain", ""):
                return str(goal.domain)
        return ""

    def _select_nodes(self, signals: list[InterestSignal], *, within: str, limit: int) -> list[InterestNode]:
        aggregates = self._aggregate(signals)
        if not aggregates:
            return []
        selected = self._nodes_for_scope(aggregates, within=within)
        selected.sort(key=lambda node: (node.score, node.last_seen_at or datetime.min, node.signal_count), reverse=True)
        for node in selected:
            node.command = self._command_for_node(node)
        return selected[: max(1, limit)]

    def _aggregate(self, signals: list[InterestSignal]) -> dict[str, InterestNode]:
        nodes: dict[str, InterestNode] = {}

        def ensure(ref: str, label: str, level: str, *, parent_ref: str = "", top_label: str = "") -> InterestNode:
            if ref not in nodes:
                nodes[ref] = InterestNode(ref=ref, label=label, level=level, parent_ref=parent_ref, top_label=top_label or label)
            return nodes[ref]

        cutoff7 = self.now - timedelta(days=7)
        cutoff30 = self.now - timedelta(days=30)
        for signal in signals:
            top, parent, leaf = classify_interest_label(signal.label)
            top_ref = self._ref(top)
            parent_ref = self._ref(top, parent)
            leaf_ref = self._ref(top, parent, leaf)
            top_node = ensure(top_ref, top, "top", top_label=top)
            parent_node = ensure(parent_ref, parent, "parent", parent_ref=top_ref, top_label=top)
            leaf_node = ensure(leaf_ref, leaf, "leaf", parent_ref=parent_ref, top_label=top)
            if parent_ref not in top_node.children:
                top_node.children.append(parent_ref)
            if leaf_ref not in parent_node.children:
                parent_node.children.append(leaf_ref)
            age_days = max(0.0, (self.now - signal.occurred_at).total_seconds() / 86400)
            recency = max(0.15, 1.0 - min(age_days, 30) / 36)
            contribution = signal.weight * recency
            for node in (top_node, parent_node, leaf_node):
                node.score += contribution
                node.signal_count += 1
                if node.last_seen_at is None or signal.occurred_at > node.last_seen_at:
                    node.last_seen_at = signal.occurred_at
                if signal.occurred_at >= cutoff7:
                    node.reason_labels.append("recent streak")
                if signal.occurred_at >= cutoff30:
                    node.reason_labels.append("sustained 30-day interest")
                reason = str(signal.evidence.get("reason", "") if isinstance(signal.evidence, dict) else "")
                if reason:
                    node.reason_labels.append(reason)
        for node in nodes.values():
            if node.last_seen_at and node.last_seen_at < cutoff30:
                node.archived = True
                node.reason_labels.append("archived: inactive >30 days")
        return nodes

    def _nodes_for_scope(self, nodes: dict[str, InterestNode], *, within: str) -> list[InterestNode]:
        if within:
            ref = self._resolve_ref(nodes, within)
            if ref and ref in nodes:
                child_refs = list(nodes[ref].children)
                if not child_refs and nodes[ref].level == "leaf":
                    return [nodes[ref]]
                return self._choose_abstraction(nodes, child_refs)
        top_refs = [ref for ref, node in nodes.items() if node.level == "top"]
        return self._choose_abstraction(nodes, top_refs)

    def _choose_abstraction(self, nodes: dict[str, InterestNode], refs: list[str]) -> list[InterestNode]:
        leaf_refs = self._leaf_refs(nodes, refs)
        if 3 <= len(leaf_refs) <= 6:
            return [nodes[ref] for ref in leaf_refs]
        if len(leaf_refs) < 3:
            return [nodes[ref] for ref in leaf_refs or refs]

        chosen: list[InterestNode] = []
        for ref in refs:
            node = nodes[ref]
            child_refs = list(node.children)
            child_leaf_count = len(self._leaf_refs(nodes, child_refs or [ref]))
            if node.level == "leaf":
                chosen.append(node)
            elif child_leaf_count <= 2:
                chosen.extend(nodes[leaf] for leaf in self._leaf_refs(nodes, child_refs or [ref]))
            elif len(child_refs) <= 3 and all(nodes[child].level != "top" for child in child_refs):
                chosen.extend(nodes[child] for child in child_refs)
            else:
                chosen.append(node)

        if len(chosen) <= 6:
            return chosen
        parent_refs = list(dict.fromkeys(node.parent_ref or self._ref(node.top_label) for node in chosen))
        compressed = [nodes[ref] for ref in parent_refs if ref in nodes]
        if 0 < len(compressed) <= 6:
            return compressed
        top_refs = list(dict.fromkeys(self._ref(node.top_label) for node in chosen))
        return [nodes[ref] for ref in top_refs if ref in nodes]

    def _leaf_refs(self, nodes: dict[str, InterestNode], refs: list[str]) -> list[str]:
        leaves: list[str] = []
        for ref in refs:
            node = nodes[ref]
            if node.level == "leaf" or not node.children:
                leaves.append(ref)
                continue
            leaves.extend(self._leaf_refs(nodes, node.children))
        return list(dict.fromkeys(leaves))

    def _resolve_ref(self, nodes: dict[str, InterestNode], value: str) -> str:
        normalized = _norm(value)
        for ref, node in nodes.items():
            if ref == value or _norm(node.label) == normalized or ref.endswith(":" + normalized):
                return ref
        return ""

    @staticmethod
    def _ref(*parts: str) -> str:
        return ":".join(_norm(part).replace(" ", "-") for part in parts if _norm(part))

    @staticmethod
    def _command_for_node(node: InterestNode) -> str:
        if node.level == "leaf":
            return f'study "{node.label}"'
        return f'next --within "{node.ref}"'
