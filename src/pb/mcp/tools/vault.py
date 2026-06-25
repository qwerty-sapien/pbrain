# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault MCP tools for reading, writing, and searching notes.

These tools enable Claude Code to interact with the Obsidian vault.
All paths are relative to the vault root and validated to prevent
path traversal attacks (Pitfall #4 from RESEARCH.md).

Tools:
- vault_search: Search notes by content or filename
- vault_read: Read a specific note
- vault_write: Write or update a note
- vault_link_graph: Get a note's link network
"""

import re
from pathlib import Path
from typing import Optional

from pb.mcp.context import get_mcp_context, get_runtime_context
from pb.mcp.pending import _bypassing, queue_pending, queue_response, register_impl
from pb.mcp.server import mcp
from pb.vault import get_vault_path, scaffold_vault
from pb.vault.graph import get_backlinks, load_vault_graph
from pb.vault.graph_store import get_hop2_neighborhood


class VaultError(Exception):
    """Raised when vault operations fail."""

    pass


def _validate_path(relative_path: str, vault_path: Path) -> Path:
    """Validate and resolve a relative path within the vault.

    Prevents path traversal attacks (Pitfall #4 from RESEARCH.md).

    Args:
        relative_path: User-provided relative path (e.g., "people/alice.md")
        vault_path: Absolute path to vault root

    Returns:
        Resolved absolute path within vault

    Raises:
        VaultError: If path escapes vault or is invalid
    """
    # Normalize the path
    full_path = (vault_path / relative_path).resolve()

    # Verify it's within the vault
    try:
        full_path.relative_to(vault_path.resolve())
    except ValueError:
        raise VaultError(f"Invalid path: '{relative_path}' escapes vault boundary")

    return full_path


def _extract_links(content: str) -> list[str]:
    """Extract wiki-style links from markdown content.

    Matches [[link]] and [[link|alias]] patterns.

    Args:
        content: Markdown content

    Returns:
        List of link targets (without aliases)
    """
    # Match [[target]] or [[target|alias]]
    pattern = r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]"
    return re.findall(pattern, content)


def _resolved_vault_path() -> Path:
    try:
        return get_runtime_context().vault_path
    except Exception:
        return get_vault_path()


def _require_writes() -> None:
    if not get_mcp_context().allow_writes:
        raise VaultError("This MCP server is running in read-only mode. Restart with --allow-writes.")


@mcp.tool()
def vault_search(query: str, folder: str = "") -> dict:
    """Search notes in the vault by content or filename.

    Args:
        query: Search query string (searches filenames and content)
        folder: Optional folder to limit search (e.g., 'people')

    Returns:
        JSON-formatted list of matching notes with paths and snippets
    """
    vault_path = _resolved_vault_path()

    search_root = vault_path
    if folder:
        search_root = _validate_path(folder, vault_path)
        if not search_root.exists():
            return {"matches": [], "error": f"Folder not found: {folder}"}

    matches = []
    query_lower = query.lower()

    for md_file in search_root.rglob("*.md"):
        # Skip hidden files
        if any(part.startswith(".") for part in md_file.parts):
            continue

        relative_path = str(md_file.relative_to(vault_path))

        # Check filename match
        filename_match = query_lower in md_file.name.lower()

        # Check content match
        content_match = False
        snippet = ""
        try:
            content = md_file.read_text()
            content_lower = content.lower()
            if query_lower in content_lower:
                content_match = True
                # Extract snippet around first match
                idx = content_lower.find(query_lower)
                start = max(0, idx - 50)
                end = min(len(content), idx + len(query) + 50)
                snippet = content[start:end].replace("\n", " ").strip()
                if start > 0:
                    snippet = "..." + snippet
                if end < len(content):
                    snippet = snippet + "..."
        except Exception:
            pass  # Skip unreadable files

        if filename_match or content_match:
            matches.append(
                {
                    "path": relative_path,
                    "filename_match": filename_match,
                    "content_match": content_match,
                    "snippet": snippet,
                }
            )

        # Limit results
        if len(matches) >= 50:
            break

    return {"matches": matches, "total": len(matches)}


@mcp.tool()
def vault_read(path: str) -> str:
    """Read a note from the vault.

    Args:
        path: Relative path within vault (e.g., 'people/alice.md')

    Returns:
        Full content of the note
    """
    vault_path = _resolved_vault_path()
    full_path = _validate_path(path, vault_path)

    if not full_path.exists():
        raise VaultError(f"Note not found: {path}")

    if not full_path.is_file():
        raise VaultError(f"Path is not a file: {path}")

    return full_path.read_text()


def _do_vault_write(path: str, content: str, create_folders: bool = True) -> dict:
    """Actual vault write; surfaces (not swallows) index errors so the LLM can react."""
    vault_path = _resolved_vault_path()
    scaffold_vault(vault_path)
    full_path = _validate_path(path, vault_path)
    existed = full_path.exists()

    if create_folders:
        full_path.parent.mkdir(parents=True, exist_ok=True)
    elif not full_path.parent.exists():
        raise VaultError(
            f"Parent folder does not exist: {full_path.parent.relative_to(vault_path)}"
        )

    full_path.write_text(content)

    index_errors: list[str] = []

    try:
        from pb.vault.graph import update_note_in_graph
        update_note_in_graph(vault_path, path, content)
    except Exception as exc:
        index_errors.append(f"graph_update: {type(exc).__name__}: {exc}")

    try:
        from pb.vault.graph import enforce_backlinks_for_note
        enforce_backlinks_for_note(vault_path, path, content)
    except Exception as exc:
        index_errors.append(f"backlinks: {type(exc).__name__}: {exc}")

    try:
        from pb.vault.lifecycle import log_interaction, check_promotion
        log_interaction(note_path=path, event_type="read")
        check_promotion(path, vault_path)
    except Exception as exc:
        index_errors.append(f"lifecycle: {type(exc).__name__}: {exc}")

    try:
        from pb.vault.indexer import update_folder_index
        folder_path = (vault_path / path).parent
        update_folder_index(folder_path, vault_root=vault_path, note_rel_path=path, content=content)
    except Exception as exc:
        index_errors.append(f"folder_index: {type(exc).__name__}: {exc}")

    action = "updated" if existed else "created"
    result: dict = {"action": action, "path": path}
    if index_errors:
        result["index_errors"] = index_errors
        result["warning"] = (
            "Note was written but some index updates failed. State may be partially "
            "inconsistent until the next pb-side refresh."
        )
    return result


register_impl("vault_write", _do_vault_write)


@mcp.tool()
def vault_write(path: str, content: str, create_folders: bool = True) -> dict:
    """Write or update a note in the vault. Tier-2: queues for confirmation.

    Args:
        path: Relative path within vault (e.g., 'people/alice.md')
        content: Full content to write (overwrites existing)
        create_folders: If True, create parent folders if needed (default: True)
    """
    _require_writes()
    args = {"path": path, "content": content, "create_folders": create_folders}
    if _bypassing():
        return _do_vault_write(**args)
    # Truncate the content preview shown to the user in `pb mcp pending`
    preview = content if len(content) <= 240 else content[:240] + " …"
    pending = queue_pending(
        tool_name="vault_write",
        args=args,
        summary=f"Write note: {path} ({len(content)} chars)\n---\n{preview}",
        risk="high",
    )
    return queue_response(pending)


@mcp.tool()
def vault_link_graph(path: str, depth: int = 1) -> dict:
    """Get a note's link network (outgoing and incoming links).

    Args:
        path: Relative path to the note (e.g., 'people/alice.md')
        depth: How many levels of links to traverse (default: 1)

    Returns:
        JSON-formatted link graph with outgoing and incoming links
    """
    vault_path = _resolved_vault_path()
    full_path = _validate_path(path, vault_path)

    if not full_path.exists():
        raise VaultError(f"Note not found: {path}")
    requested_depth = max(1, int(depth or 1))
    effective_depth = min(requested_depth, 2)
    clamped = requested_depth > effective_depth

    note_rel = str(full_path.relative_to(vault_path))
    edges = load_vault_graph(vault_path)
    backlinks = get_backlinks(vault_path)
    outgoing_paths = [str(item) for item in list(edges.get(note_rel, []) or []) if str(item).strip()]
    incoming_paths = [str(item) for item in list(backlinks.get(note_rel, []) or []) if str(item).strip()]
    manual_outgoing, manual_incoming = _scan_manual_neighbors(vault_path, full_path)
    if not outgoing_paths:
        outgoing_paths = [
            str(item.get("resolved_path", "") or "").strip()
            for item in manual_outgoing
            if str(item.get("resolved_path", "") or "").strip()
        ]
    if not incoming_paths:
        incoming_paths = list(manual_incoming)

    outgoing_by_target = {
        str(item.get("target", "")).strip().lower(): item for item in manual_outgoing
    }
    for target in outgoing_paths:
        outgoing_by_target.setdefault(
            Path(target).stem.lower(),
            {
                "target": Path(target).stem,
                "resolved_path": target,
                "exists": (vault_path / target).exists(),
            },
        )
    outgoing = list(outgoing_by_target.values())

    out2_paths: list[str] = []
    in2_paths: list[str] = []
    if effective_depth > 1:
        stem_index: dict[str, list[str]] = {}
        for source_path, targets in edges.items():
            stem_index.setdefault(Path(source_path).stem, []).append(source_path)
            for target in targets:
                stem_index.setdefault(Path(target).stem, []).append(target)
        try:
            hop2 = get_hop2_neighborhood(vault_path, full_path.stem)
        except Exception:
            hop2 = {"out2": [], "in2": []}
        out2_paths = _resolve_neighbor_paths(hop2.get("out2", []), stem_index=stem_index, exclude={note_rel})
        in2_paths = _resolve_neighbor_paths(hop2.get("in2", []), stem_index=stem_index, exclude={note_rel})
        if not out2_paths:
            out2_paths = _dedupe_paths(
                target
                for source in outgoing_paths
                for target in list(edges.get(source, []) or [])
                if target != note_rel
            )
        if not in2_paths:
            in2_paths = _dedupe_paths(
                source
                for target in incoming_paths
                for source in list(backlinks.get(target, []) or [])
                if source != note_rel
            )

    return {
        "note": path,
        "depth": effective_depth,
        "requested_depth": requested_depth,
        "effective_depth": effective_depth,
        "clamped": clamped,
        "outgoing": outgoing,
        "incoming": incoming_paths,
        "outgoing_count": len(outgoing),
        "incoming_count": len(incoming_paths),
        "out1": outgoing,
        "in1": incoming_paths,
        "out2": out2_paths if effective_depth > 1 else [],
        "in2": in2_paths if effective_depth > 1 else [],
    }


def _resolve_neighbor_paths(
    slugs: list[object],
    *,
    stem_index: dict[str, list[str]],
    exclude: set[str],
) -> list[str]:
    resolved: list[str] = []
    for slug in slugs:
        for candidate in stem_index.get(str(slug).strip(), []):
            if candidate and candidate not in exclude:
                resolved.append(candidate)
    return _dedupe_paths(resolved)


def _dedupe_paths(items: list[str] | tuple[str, ...] | object) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        clean = str(item).strip()
        if not clean or clean in seen:
            continue
        seen.add(clean)
        ordered.append(clean)
    return ordered


def _scan_manual_neighbors(vault_path: Path, full_path: Path) -> tuple[list[dict[str, object]], list[str]]:
    content = full_path.read_text()
    outgoing_targets = _extract_links(content)
    outgoing: list[dict[str, object]] = []
    for target in outgoing_targets:
        target_path = _resolve_target_path(vault_path, target)
        outgoing.append(
            {
                "target": target,
                "resolved_path": target_path,
                "exists": target_path is not None,
            }
        )

    note_name = full_path.stem.lower()
    note_rel = str(full_path.relative_to(vault_path))
    incoming: list[str] = []
    for md_file in vault_path.rglob("*.md"):
        if md_file == full_path or any(part.startswith(".") for part in md_file.parts):
            continue
        try:
            file_content = md_file.read_text()
        except Exception:
            continue
        links = [item.strip().lower() for item in _extract_links(file_content)]
        if note_name in links or note_rel.replace(".md", "").lower() in links:
            incoming.append(str(md_file.relative_to(vault_path)))
    return outgoing, _dedupe_paths(incoming)


def _resolve_target_path(vault_path: Path, target: str) -> str | None:
    candidate = vault_path / f"{target}.md"
    if candidate.exists():
        return str(candidate.relative_to(vault_path))
    lowered = (target or "").strip().lower()
    for md_file in vault_path.rglob("*.md"):
        if md_file.stem.lower() == lowered:
            return str(md_file.relative_to(vault_path))
    return None
