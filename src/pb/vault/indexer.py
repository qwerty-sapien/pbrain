# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Per-folder SQLite + FTS5 vault index management.

Each vault folder has its own .pb-index.db with:
- notes table: path, title, mtime, words (metadata)
- notes_fts virtual table: FTS5 full-text search over title + body

Index operations are non-fatal: failures are logged but never propagate to callers.
"""

from __future__ import annotations

import re
import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger()

_INDEX_FILENAME = ".pb-index.db"
_DIRECTORY_FILENAME = ".pb-directory.md"
_H1_RE = re.compile(r"^#\s+(.+)", re.MULTILINE)

FOLDER_INDEX_SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    path  TEXT PRIMARY KEY,
    title TEXT,
    mtime REAL,
    words INTEGER
);
CREATE VIRTUAL TABLE IF NOT EXISTS notes_fts USING fts5(
    path  UNINDEXED,
    title,
    body,
    tokenize="unicode61"
);
"""


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _get_conn(db_path: Path) -> sqlite3.Connection:
    """Open a SQLite connection with WAL mode enabled.

    Caller is responsible for try/finally conn.close().
    """
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def _extract_title(note_path: str, content: str) -> str:
    """Extract title from content H1 heading, or fall back to filename stem."""
    m = _H1_RE.search(content)
    if m:
        return m.group(1).strip()
    return Path(note_path).stem


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def is_folder_index_stale(folder_path: Path) -> bool:
    """Return True if the folder's .pb-index.db is missing or older than any note.

    Only checks single-level .md files (not subfolders — each has its own index).
    Hidden files (starting with '.') are excluded.
    """
    db_path = folder_path / _INDEX_FILENAME
    if not db_path.exists():
        return True

    try:
        idx_mtime = db_path.stat().st_mtime
    except OSError:
        return True

    for md in folder_path.glob("*.md"):
        if md.name.startswith("."):
            continue
        try:
            if md.stat().st_mtime > idx_mtime:
                return True
        except OSError:
            pass

    return False


def _migrate_learning_stage_column(conn: sqlite3.Connection) -> None:
    """Add learning_stage column to notes table if not present (Phase 17, idempotent)."""
    existing = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
    if "learning_stage" not in existing:
        conn.execute("ALTER TABLE notes ADD COLUMN learning_stage TEXT DEFAULT NULL")
    conn.commit()


def rebuild_folder_index(folder_path: Path, vault_root: Path) -> int:
    """Rebuild the folder's .pb-index.db from scratch.

    Scans all non-hidden .md files in folder_path (single-level only).
    Stores paths as vault-relative strings (consistent with graph.py).
    Returns count of indexed notes.
    """
    db_path = folder_path / _INDEX_FILENAME
    conn = _get_conn(db_path)
    count = 0
    try:
        conn.executescript(FOLDER_INDEX_SCHEMA)
        _migrate_learning_stage_column(conn)
        conn.execute("DELETE FROM notes")
        conn.execute("DELETE FROM notes_fts")
        conn.commit()

        for md in sorted(folder_path.glob("*.md")):
            if md.name.startswith("."):
                continue
            try:
                content = md.read_text(encoding="utf-8", errors="replace")
                title = _extract_title(str(md), content)
                mtime = md.stat().st_mtime
                words = len(content.split())
                rel_path = str(md.relative_to(vault_root))
                from pb.vault.lifecycle import read_frontmatter
                fm, _ = read_frontmatter(content)
                learning_stage = fm.get("learning_stage")
                conn.execute(
                    "INSERT OR REPLACE INTO notes (path, title, mtime, words, learning_stage) VALUES (?, ?, ?, ?, ?)",
                    (rel_path, title, mtime, words, learning_stage),
                )
                conn.execute(
                    "INSERT INTO notes_fts (path, title, body) VALUES (?, ?, ?)",
                    (rel_path, title, content),
                )
                count += 1
            except Exception:
                pass

        conn.commit()
    finally:
        conn.close()

    return count


def update_folder_index(
    folder_path: Path,
    vault_root: Path,
    note_rel_path: str,
    content: str,
) -> None:
    """Incrementally update a single note in the folder index.

    If the index does not yet exist, silently returns without creating it.
    Uses DELETE+INSERT for FTS5 (FTS5 does not support UPDATE).
    Non-fatal: all errors are logged at debug level.
    """
    db_path = folder_path / _INDEX_FILENAME
    if not db_path.exists():
        return

    conn = _get_conn(db_path)
    try:
        note_path = vault_root / note_rel_path
        title = _extract_title(note_rel_path, content)
        mtime = note_path.stat().st_mtime
        words = len(content.split())

        conn.execute(
            "INSERT OR REPLACE INTO notes (path, title, mtime, words) VALUES (?, ?, ?, ?)",
            (note_rel_path, title, mtime, words),
        )
        conn.execute("DELETE FROM notes_fts WHERE path = ?", (note_rel_path,))
        conn.execute(
            "INSERT INTO notes_fts (path, title, body) VALUES (?, ?, ?)",
            (note_rel_path, title, content),
        )
        conn.commit()
    except Exception as e:
        logger.debug("vault.index_update_failed", path=note_rel_path, error=str(e))
    finally:
        conn.close()


def search_folder_index(
    folder_path: Path,
    pattern: str,
) -> list[tuple[str, str]] | None:
    """Search the folder's FTS5 index for a pattern.

    Returns list of (vault_relative_path, snippet) tuples on success.
    Returns None on FTS5 OperationalError (bad MATCH syntax) or other failure —
    caller should fall back to filesystem scan.
    Returns empty list [] when no matches found.
    """
    db_path = folder_path / _INDEX_FILENAME
    if not db_path.exists():
        return None

    conn = _get_conn(db_path)
    try:
        cur = conn.execute(
            "SELECT path, snippet(notes_fts, 2, '>', '<', '...', 8) "
            "FROM notes_fts WHERE notes_fts MATCH ? ORDER BY rank",
            (pattern,),
        )
        return list(cur.fetchall())
    except sqlite3.OperationalError:
        # Bad FTS5 MATCH syntax — signal caller to fall back to filesystem scan
        return None
    except Exception:
        return None
    finally:
        conn.close()


def generate_directory_md(folder_path: Path) -> str:
    """Generate a .pb-directory.md summary for the given folder.

    Lists all non-hidden notes and subfolders. Title for each note is
    extracted from its H1 heading or falls back to the filename stem.
    """
    notes = sorted(
        [p for p in folder_path.glob("*.md") if not p.name.startswith(".")],
        key=lambda p: p.name,
    )
    subfolders = sorted(
        [p for p in folder_path.iterdir() if p.is_dir() and not p.name.startswith(".")],
        key=lambda p: p.name,
    )

    lines: list[str] = [
        f"# {folder_path.name}",
        "",
        f"{len(notes)} notes, {len(subfolders)} subfolders",
    ]

    if subfolders:
        lines.append("")
        lines.append("## Subfolders")
        for sf in subfolders:
            lines.append(f"- {sf.name}/")

    if notes:
        lines.append("")
        lines.append("## Notes")
        for note in notes:
            try:
                content = note.read_text(encoding="utf-8", errors="replace")
                title = _extract_title(str(note), content)
            except Exception:
                title = note.stem
            lines.append(f"- {title}")

    lines.append("")
    return "\n".join(lines)
