# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Persistent vault graph topology.

Stores the note graph as YAML at {vault}/.pb-graph.yaml.
Updated incrementally on every vault_write; brain.py reads it without scanning.

If the graph is stale (external edits detected via mtime), a full rebuild runs
automatically on next load.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import structlog

from pb.storage.yaml_io import load_yaml_text, load_yaml_with_legacy_json, write_yaml_file

logger = structlog.get_logger()

_GRAPH_FILENAME = ".pb-graph.yaml"
_LEGACY_GRAPH_FILENAME = ".pb-graph.json"
_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_vault_graph(vault_path: Path) -> dict[str, list[str]]:
    """Load the persistent graph, rebuilding if stale or missing.

    Returns dict mapping each note path to its list of outgoing link paths.
    """
    graph_path = vault_path / _GRAPH_FILENAME
    legacy_path = vault_path / _LEGACY_GRAPH_FILENAME
    if graph_path.exists():
        try:
            data = load_yaml_text(graph_path.read_text(), {})
            edges = data.get("edges", {})
            if not _is_stale(vault_path, graph_path):
                logger.debug("vault.graph", source="persistent", nodes=len(edges))
                return edges
        except Exception:
            pass
    elif legacy_path.exists():
        try:
            data = load_yaml_with_legacy_json(graph_path, legacy_path, {})
            edges = data.get("edges", {}) if isinstance(data, dict) else {}
            if edges and not _is_stale(vault_path, legacy_path):
                _save(vault_path, edges)
                logger.debug("vault.graph", source="legacy_migrated", nodes=len(edges))
                return edges
        except Exception:
            pass

    edges = _full_rebuild(vault_path)
    _save(vault_path, edges)
    logger.debug("vault.graph", source="rebuilt", nodes=len(edges))
    return edges


def update_note_in_graph(vault_path: Path, note_rel_path: str, content: str):
    """Incrementally update graph after a single note write.

    Called by vault_write. Resolves [[links]] in the new content and
    updates the note's entry. Also adds the note if it's new.
    """
    graph_path = vault_path / _GRAPH_FILENAME
    try:
        if graph_path.exists():
            edges = load_yaml_text(graph_path.read_text(), {}).get("edges", {})
        else:
            edges = _full_rebuild(vault_path)

        stem_index = _build_stem_index(edges)
        # Add this note to the stem index if new
        stem = Path(note_rel_path).stem
        stem_index[stem] = note_rel_path

        targets = _resolve_links(content, stem_index)
        edges[note_rel_path] = targets

        _save(vault_path, edges)
    except Exception as e:
        logger.debug("vault.graph_update_failed", path=note_rel_path, error=str(e))


def remove_note_from_graph(vault_path: Path, note_rel_path: str):
    """Remove a note from the graph (for future delete support)."""
    graph_path = vault_path / _GRAPH_FILENAME
    try:
        if not graph_path.exists():
            return
        edges = load_yaml_text(graph_path.read_text(), {}).get("edges", {})
        edges.pop(note_rel_path, None)
        _save(vault_path, edges)
    except Exception:
        pass


def graph_to_adjacency_text(edges: dict[str, list[str]]) -> tuple[str, int, int]:
    """Convert edges dict to compact adjacency list text for LLM prompt.

    Returns (text, node_count, edge_count).
    """
    lines: list[str] = []
    total_edges = 0
    for path in sorted(edges):
        targets = edges[path]
        total_edges += len(targets)
        if targets:
            lines.append(f"{path} -> {', '.join(targets)}")
        else:
            lines.append(path)
    return "\n".join(lines), len(edges), total_edges


def get_folder_graph(vault_path: Path, folder_rel: str) -> dict[str, list[str]]:
    """Return folder-scoped subgraph: edges where source or any target starts with folder_rel.

    Per D-15/D-16: this is a read-time view, not a separate data structure.
    Includes both outbound edges (source inside folder) and inbound edges
    (source outside folder but targeting a note inside the folder, filtered to
    only the targets that are inside the folder).
    """
    edges = load_vault_graph(vault_path)
    prefix = folder_rel.rstrip("/") + "/"
    result: dict[str, list[str]] = {}
    for source, targets in edges.items():
        if source.startswith(prefix):
            result[source] = targets
        else:
            folder_targets = [t for t in targets if t.startswith(prefix)]
            if folder_targets:
                result[source] = folder_targets
    return result


def get_backlinks(vault_path: Path) -> dict[str, list[str]]:
    """Load backlink index from .pb-graph.yaml. Returns empty dict if missing."""
    graph_path = vault_path / _GRAPH_FILENAME
    if not graph_path.exists():
        load_vault_graph(vault_path)  # triggers rebuild + _save with backlinks
    try:
        data = load_yaml_text(graph_path.read_text(), {})
        return data.get("backlinks", {})
    except Exception:
        return {}


def check_no_outgoing_links(content: str, note_path: str) -> Optional[str]:
    """Return warning string if note has no outgoing wikilinks, else None (GRPH-03).

    Callers (shell.py, mcp vault tools) should call this after writing a note
    to surface the warning to the user.
    """
    links = _LINK_RE.findall(content)
    if not links:
        return f"[yellow]Warning:[/] {note_path} has no outgoing wikilinks"
    return None


def bulk_vault_write(vault_path: Path, notes: list[tuple[str, str]]) -> int:
    """Write multiple notes then do a single graph rebuild.

    Args:
        vault_path: Root vault path
        notes: List of (rel_path, content) tuples

    Returns:
        Count of notes written
    """
    written = 0
    for rel_path, content in notes:
        target = vault_path / rel_path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content)
        written += 1
    # Single rebuild after all writes (GRPH-04)
    edges = _full_rebuild(vault_path)
    _save(vault_path, edges)
    logger.debug("vault.bulk_write_complete", notes=written, graph_nodes=len(edges))
    return written


# ---------------------------------------------------------------------------
# Internal
# ---------------------------------------------------------------------------


def _is_stale(vault_path: Path, graph_path: Path) -> bool:
    """Check if any .md file is newer than the graph file."""
    graph_mtime = graph_path.stat().st_mtime
    for md in vault_path.rglob("*.md"):
        if any(p.startswith(".") for p in md.relative_to(vault_path).parts):
            continue
        try:
            if md.stat().st_mtime > graph_mtime:
                return True
        except OSError:
            pass
    return False


def _full_rebuild(vault_path: Path) -> dict[str, list[str]]:
    """Scan all vault .md files and build complete edge map."""
    stem_index: dict[str, str] = {}
    paths: list[str] = []

    for md in sorted(vault_path.rglob("*.md")):
        parts = md.relative_to(vault_path).parts
        if any(p.startswith(".") for p in parts):
            continue
        rel = str(md.relative_to(vault_path))
        paths.append(rel)
        stem_index[md.stem] = rel

    edges: dict[str, list[str]] = {}
    for rel in paths:
        try:
            content = (vault_path / rel).read_text()
        except Exception:
            edges[rel] = []
            continue
        edges[rel] = _resolve_links(content, stem_index)

    return edges


def _resolve_links(content: str, stem_index: dict[str, str]) -> list[str]:
    """Extract [[links]] from content and resolve to paths via stem_index."""
    targets = _LINK_RE.findall(content)
    seen: set[str] = set()
    resolved: list[str] = []
    for t in targets:
        p = stem_index.get(t)
        if p and p not in seen:
            seen.add(p)
            resolved.append(p)
    return resolved


def _build_stem_index(edges: dict[str, list[str]]) -> dict[str, str]:
    """Build stem→path index from existing edge keys."""
    return {Path(p).stem: p for p in edges}


def _compute_backlinks(edges: dict[str, list[str]]) -> dict[str, list[str]]:
    """Invert edges dict to produce backlink index. O(N+E)."""
    backlinks: dict[str, list[str]] = {p: [] for p in edges}
    for source, targets in edges.items():
        for target in targets:
            if target in backlinks:
                backlinks[target].append(source)
    return backlinks


def _save(vault_path: Path, edges: dict[str, list[str]]):
    """Write graph + backlinks to disk."""
    backlinks = _compute_backlinks(edges)
    graph_path = vault_path / _GRAPH_FILENAME
    write_yaml_file(graph_path, {"edges": edges, "backlinks": backlinks})


# ---------------------------------------------------------------------------
# Public API — graph connectivity helpers (Phase 17)
# ---------------------------------------------------------------------------


_SKIP_BACKLINK_WRITE: bool = False  # Module-level guard against circular backlink insertion


def enforce_bidirectional_link(vault_path: Path, source_rel: str, target_rel: str) -> None:
    """Insert source_rel into target note's frontmatter backlinks[] if absent (D-09, D-10).

    Guarded by _SKIP_BACKLINK_WRITE to prevent circular loop (Pitfall 1):
    A writes [[B]] -> insert A in B's backlinks -> write B -> skip enforcement.
    """
    global _SKIP_BACKLINK_WRITE
    if _SKIP_BACKLINK_WRITE:
        return
    target_path = vault_path / target_rel
    if not target_path.exists():
        return
    try:
        from pb.vault.lifecycle import read_frontmatter, write_frontmatter

        content = target_path.read_text()
        fm, body = read_frontmatter(content)
        existing = fm.get("backlinks", [])
        if source_rel not in existing:
            existing.append(source_rel)
            fm["backlinks"] = sorted(existing)  # sorted for deterministic output
            _SKIP_BACKLINK_WRITE = True
            try:
                target_path.write_text(write_frontmatter(fm, body))
            finally:
                _SKIP_BACKLINK_WRITE = False
    except Exception as e:
        logger.debug("graph.enforce_backlink_failed", source=source_rel, target=target_rel, error=str(e))


def enforce_backlinks_for_note(vault_path: Path, source_rel: str, content: str) -> None:
    """Enforce bidirectional backlinks for all [[links]] in a note's content."""
    targets = _LINK_RE.findall(content)
    if not targets:
        return
    graph_file = vault_path / _GRAPH_FILENAME
    if graph_file.exists():
        try:
            existing_edges = load_yaml_text(graph_file.read_text(), {}).get("edges", {})
        except Exception:
            existing_edges = {}
    else:
        existing_edges = {}
    stem_index = _build_stem_index(existing_edges)
    for raw_target in targets:
        resolved = stem_index.get(raw_target)
        if resolved and resolved != source_rel:
            enforce_bidirectional_link(vault_path, source_rel, resolved)
