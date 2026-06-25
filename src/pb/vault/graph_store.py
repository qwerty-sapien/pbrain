# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Two-tier SQLite graph store for vault note links.

Tier 1: <vault_root>/vault.db — central registry of all nodes and links.
Tier 2: <vault_root>/<subfolder>/.2hop.db — per-subfolder materialized
        2-hop neighborhoods, created only when node count exceeds 500.

WAL mode throughout. All queries use parameterized inputs — no f-string SQL.

Public API:
    open_vault_db(vault_path) -> sqlite3.Connection
    ensure_schema(conn) -> None
    upsert_node(conn, slug, subfolder) -> None
    add_link(conn, src_slug, dst_slug) -> None
    remove_link(conn, src_slug, dst_slug) -> None
    get_hop2_neighborhood(vault_path, slug) -> dict
    has_socratic_link_for_note(vault_path, candidate_note_path) -> bool
    migrate_from_json(conn, vault_path) -> int
    GraphStorePool
"""

from __future__ import annotations

import functools
import sqlite3
from pathlib import Path
from typing import Optional

import structlog

from pb.storage.yaml_io import dump_compact_yaml, load_yaml_text, load_yaml_with_legacy_json

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VAULT_DB_FILENAME = "vault.db"
_HOP2_DB_FILENAME = ".2hop.db"
_GRAPH_YAML_FILENAME = ".pb-graph.yaml"
_GRAPH_JSON_FILENAME = ".pb-graph.json"
_THRESHOLD = 500  # On-the-fly below; materialized above

VAULT_DB_PRAGMAS = """
PRAGMA journal_mode = WAL;
PRAGMA synchronous  = NORMAL;
PRAGMA temp_store   = MEMORY;
PRAGMA cache_size   = -8000;
"""

HOP2_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS hop2 (
    slug     TEXT PRIMARY KEY,
    out1     TEXT NOT NULL DEFAULT '[]',
    in1      TEXT NOT NULL DEFAULT '[]',
    out2     TEXT NOT NULL DEFAULT '[]',
    in2      TEXT NOT NULL DEFAULT '[]',
    computed INTEGER NOT NULL DEFAULT (unixepoch())
);
"""

# ---------------------------------------------------------------------------
# Schema (vault.db)
# ---------------------------------------------------------------------------

VAULT_DB_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes (
    slug       TEXT    PRIMARY KEY,
    subfolder  TEXT    NOT NULL,
    dirty_hop2 INTEGER NOT NULL DEFAULT 0,
    updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE TABLE IF NOT EXISTS links (
    src  TEXT NOT NULL REFERENCES nodes(slug) ON DELETE CASCADE,
    dst  TEXT NOT NULL REFERENCES nodes(slug) ON DELETE CASCADE,
    PRIMARY KEY (src, dst)
);

CREATE INDEX IF NOT EXISTS idx_links_dst ON links(dst);

CREATE VIEW  IF NOT EXISTS node_count AS
    SELECT COUNT(*) AS n FROM nodes;
"""


# ---------------------------------------------------------------------------
# Public API — vault.db lifecycle
# ---------------------------------------------------------------------------


def open_vault_db(vault_path: Path) -> sqlite3.Connection:
    """Open vault.db at vault_root, apply PRAGMAs, call ensure_schema.

    Also auto-migrates from .pb-graph.yaml or the legacy .pb-graph.json
    if nodes table is empty and the file exists.

    Returns the open sqlite3.Connection. Caller must close when done.
    """
    db_path = vault_path / _VAULT_DB_FILENAME
    conn = sqlite3.connect(str(db_path))
    conn.executescript(VAULT_DB_PRAGMAS)
    ensure_schema(conn)
    graph_yaml = vault_path / _GRAPH_YAML_FILENAME
    graph_json = vault_path / _GRAPH_JSON_FILENAME
    if graph_yaml.exists() or graph_json.exists():
        empty = conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0] == 0
        if empty:
            try:
                migrate_from_json(conn, vault_path)
            except Exception as e:
                logger.debug("graph_store.migrate_from_json_failed", error=str(e))
    return conn


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create nodes, links tables and node_count view if not present (idempotent)."""
    conn.executescript(VAULT_DB_SCHEMA)
    conn.commit()


# ---------------------------------------------------------------------------
# Public API — node and link CRUD
# ---------------------------------------------------------------------------


def upsert_node(conn: sqlite3.Connection, slug: str, subfolder: str) -> None:
    """Insert new node or replace existing node (updates subfolder and updated_at)."""
    conn.execute(
        "INSERT OR REPLACE INTO nodes (slug, subfolder, dirty_hop2, updated_at) "
        "VALUES (?, ?, COALESCE((SELECT dirty_hop2 FROM nodes WHERE slug = ?), 0), unixepoch())",
        (slug, subfolder, slug),
    )
    conn.commit()


def add_link(conn: sqlite3.Connection, src_slug: str, dst_slug: str) -> None:
    """Add edge (src_slug -> dst_slug) with dirty propagation.

    Runs inside a single BEGIN IMMEDIATE transaction:
    1. INSERT OR IGNORE the link
    2. Dirty-propagate: mark src, dst, and all hop-1 neighbors dirty_hop2=1
    3. Call _schedule_hop2_recompute()
    4. COMMIT
    """
    conn.execute("BEGIN IMMEDIATE")
    conn.execute(
        "INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)", (src_slug, dst_slug)
    )
    # Dirty propagation: src, dst, hop-1 neighbors of both
    conn.execute(
        """
        UPDATE nodes SET dirty_hop2 = 1
        WHERE slug = ? OR slug = ?
          OR slug IN (SELECT dst FROM links WHERE src = ?)
          OR slug IN (SELECT src FROM links WHERE dst = ?)
          OR slug IN (SELECT dst FROM links WHERE src = ?)
          OR slug IN (SELECT src FROM links WHERE dst = ?)
        """,
        (src_slug, dst_slug, src_slug, src_slug, dst_slug, dst_slug),
    )
    _schedule_hop2_recompute(conn)
    conn.commit()
    # Invalidate LRU cache for affected slugs
    _get_hop2_cached.cache_clear()


def remove_link(conn: sqlite3.Connection, src_slug: str, dst_slug: str) -> None:
    """Remove edge (src_slug -> dst_slug) with dirty propagation.

    Same dirty-propagation pattern as add_link, but DELETE FROM links.
    """
    conn.execute("BEGIN IMMEDIATE")
    # Propagate dirty BEFORE deleting so hop-1 neighbors are still resolvable
    conn.execute(
        """
        UPDATE nodes SET dirty_hop2 = 1
        WHERE slug = ? OR slug = ?
          OR slug IN (SELECT dst FROM links WHERE src = ?)
          OR slug IN (SELECT src FROM links WHERE dst = ?)
          OR slug IN (SELECT dst FROM links WHERE src = ?)
          OR slug IN (SELECT src FROM links WHERE dst = ?)
        """,
        (src_slug, dst_slug, src_slug, src_slug, dst_slug, dst_slug),
    )
    conn.execute(
        "DELETE FROM links WHERE src = ? AND dst = ?", (src_slug, dst_slug)
    )
    _schedule_hop2_recompute(conn)
    conn.commit()
    # Invalidate LRU cache
    _get_hop2_cached.cache_clear()


# ---------------------------------------------------------------------------
# Public API — 2-hop neighborhood reads
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=128)
def _get_hop2_cached(vault_path_str: str, slug: str) -> tuple:
    """Internal LRU-cached function keyed by (vault_path_str, slug).

    Returns a tuple (out1_yaml, in1_yaml, out2_yaml, in2_yaml) of compact YAML strings.
    Callers must parse each element to get list[str].
    """
    vault_path = Path(vault_path_str)
    db_path = vault_path / _VAULT_DB_FILENAME
    conn = sqlite3.connect(str(db_path))
    conn.executescript(VAULT_DB_PRAGMAS)
    try:
        n = conn.execute("SELECT n FROM node_count").fetchone()[0]
        if n > _THRESHOLD:
            # Materialized mode: read from .2hop.db
            subfolder_row = conn.execute(
                "SELECT subfolder FROM nodes WHERE slug = ?", (slug,)
            ).fetchone()
            if subfolder_row is None:
                return ("[]", "[]", "[]", "[]")
            subfolder = subfolder_row[0]
            hop2_db = vault_path / subfolder / _HOP2_DB_FILENAME
            if hop2_db.exists():
                hop2_conn = sqlite3.connect(str(hop2_db))
                hop2_conn.executescript(VAULT_DB_PRAGMAS)
                try:
                    row = hop2_conn.execute(
                        "SELECT out1, in1, out2, in2 FROM hop2 WHERE slug = ?", (slug,)
                    ).fetchone()
                    if row is not None:
                        return (row[0], row[1], row[2], row[3])
                finally:
                    hop2_conn.close()
            # .2hop.db doesn't exist or slug not in it: compute on-the-fly
        # On-the-fly computation from vault.db
        out1 = [
            r[0]
            for r in conn.execute(
                "SELECT dst FROM links WHERE src = ?", (slug,)
            ).fetchall()
        ]
        in1 = [
            r[0]
            for r in conn.execute(
                "SELECT src FROM links WHERE dst = ?", (slug,)
            ).fetchall()
        ]
        out2_rows = conn.execute(
            """
            SELECT DISTINCT dst FROM links
            WHERE src IN (SELECT dst FROM links WHERE src = ?)
              AND dst != ?
            """,
            (slug, slug),
        ).fetchall()
        out2 = [r[0] for r in out2_rows]
        in2_rows = conn.execute(
            """
            SELECT DISTINCT src FROM links
            WHERE dst IN (SELECT src FROM links WHERE dst = ?)
              AND src != ?
            """,
            (slug, slug),
        ).fetchall()
        in2 = [r[0] for r in in2_rows]
        return (
            dump_compact_yaml(out1),
            dump_compact_yaml(in1),
            dump_compact_yaml(out2),
            dump_compact_yaml(in2),
        )
    finally:
        conn.close()


def get_hop2_neighborhood(vault_path: Path, slug: str) -> dict:
    """Return {out1, in1, out2, in2} 2-hop neighborhood for slug.

    Uses LRU cache (128 entries). If n>500, triggers dirty recompute before
    reading (materializes .2hop.db). If n<=500 computes on-the-fly from vault.db.
    """
    # For n>500: run dirty recompute so .2hop.db is up to date before the
    # cached read. We open vault.db here to check/run the recompute.
    db_path = vault_path / _VAULT_DB_FILENAME
    if db_path.exists():
        try:
            rconn = sqlite3.connect(str(db_path))
            rconn.executescript(VAULT_DB_PRAGMAS)
            try:
                n = rconn.execute("SELECT n FROM node_count").fetchone()[0]
                if n > _THRESHOLD:
                    dirty_count = rconn.execute(
                        "SELECT COUNT(*) FROM nodes WHERE dirty_hop2 = 1"
                    ).fetchone()[0]
                    if dirty_count > 0:
                        _recompute_dirty_hop2(rconn, vault_path)
                        _get_hop2_cached.cache_clear()
            finally:
                rconn.close()
        except Exception as e:
            logger.debug("graph_store.get_hop2_pre_recompute_failed", error=str(e))

    out1_j, in1_j, out2_j, in2_j = _get_hop2_cached(str(vault_path), slug)
    return {
        "out1": load_yaml_text(out1_j, []),
        "in1": load_yaml_text(in1_j, []),
        "out2": load_yaml_text(out2_j, []),
        "in2": load_yaml_text(in2_j, []),
    }


# ---------------------------------------------------------------------------
# Public API — Socratic link query
# ---------------------------------------------------------------------------


def has_socratic_link_for_note(vault_path: Path, candidate_note_path: str) -> bool:
    """Return True if any note with source:socratic in its frontmatter links to candidate.

    Algorithm:
    1. Derive candidate slug from the note path stem.
    2. Query vault.db for all src slugs that link to this candidate slug.
    3. For each src slug, resolve the note file path and check if frontmatter
       has source: socratic.
    Short-circuits on first match.
    """
    candidate_slug = Path(candidate_note_path).stem
    db_path = vault_path / _VAULT_DB_FILENAME
    if not db_path.exists():
        return False
    conn = sqlite3.connect(str(db_path))
    conn.executescript(VAULT_DB_PRAGMAS)
    try:
        rows = conn.execute(
            "SELECT n.slug, n.subfolder FROM links l "
            "JOIN nodes n ON n.slug = l.src "
            "WHERE l.dst = ?",
            (candidate_slug,),
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return False

    for src_slug, src_subfolder in rows:
        # Resolve the source note file path
        note_file = vault_path / src_subfolder / f"{src_slug}.md"
        if not note_file.exists():
            # Try the direct candidate note path approach
            note_file = vault_path / f"{src_subfolder}/{src_slug}.md"
        if not note_file.exists():
            continue
        try:
            from pb.vault.lifecycle import read_frontmatter

            content = note_file.read_text()
            fm, _ = read_frontmatter(content)
            if fm.get("source") == "socratic":
                return True
        except Exception:
            continue

    return False


# ---------------------------------------------------------------------------
# Public API — migration from .pb-graph.yaml / legacy .pb-graph.json
# ---------------------------------------------------------------------------


def migrate_from_json(conn: sqlite3.Connection, vault_path: Path) -> int:
    """Read .pb-graph.yaml or the legacy .pb-graph.json and insert into vault.db nodes+links.

    For each (src_rel_path, [dst_rel_paths]):
    - upsert_node for src (slug = stem, subfolder = parent dir)
    - For each dst: upsert_node for dst, INSERT OR IGNORE link

    After all edges inserted, marks all nodes dirty_hop2=1 and calls
    _recompute_dirty_hop2 if n>500.

    Returns count of nodes migrated. Returns 0 if neither graph file is found.
    Idempotent: INSERT OR IGNORE prevents duplicates.
    """
    graph_yaml = vault_path / _GRAPH_YAML_FILENAME
    graph_json = vault_path / _GRAPH_JSON_FILENAME
    if not graph_yaml.exists() and not graph_json.exists():
        return 0

    data = load_yaml_with_legacy_json(graph_yaml, graph_json, {})

    if not isinstance(data, dict):
        return 0

    edges = data.get("edges", {})
    if not isinstance(edges, dict):
        return 0

    node_set: set[str] = set()
    link_pairs: list[tuple[str, str]] = []

    for src_rel_path, dst_list in edges.items():
        if not isinstance(src_rel_path, str):
            continue
        src_p = Path(src_rel_path)
        src_slug = src_p.stem
        src_subfolder = str(src_p.parent) if str(src_p.parent) != "." else ""

        node_set.add(src_slug)
        conn.execute(
            "INSERT OR REPLACE INTO nodes (slug, subfolder, dirty_hop2, updated_at) "
            "VALUES (?, ?, 0, unixepoch())",
            (src_slug, src_subfolder),
        )

        if isinstance(dst_list, list):
            for dst_rel_path in dst_list:
                if not isinstance(dst_rel_path, str):
                    continue
                dst_p = Path(dst_rel_path)
                dst_slug = dst_p.stem
                dst_subfolder = str(dst_p.parent) if str(dst_p.parent) != "." else ""

                if dst_slug not in node_set:
                    node_set.add(dst_slug)
                    conn.execute(
                        "INSERT OR REPLACE INTO nodes (slug, subfolder, dirty_hop2, updated_at) "
                        "VALUES (?, ?, 0, unixepoch())",
                        (dst_slug, dst_subfolder),
                    )
                link_pairs.append((src_slug, dst_slug))

    # Insert all links (INSERT OR IGNORE for idempotency)
    for src_slug, dst_slug in link_pairs:
        conn.execute(
            "INSERT OR IGNORE INTO links (src, dst) VALUES (?, ?)",
            (src_slug, dst_slug),
        )

    conn.commit()

    # Mark all nodes dirty after bulk import
    conn.execute("UPDATE nodes SET dirty_hop2 = 1")
    conn.commit()

    # Trigger recompute if over threshold
    n = conn.execute("SELECT n FROM node_count").fetchone()[0]
    if n > _THRESHOLD:
        _recompute_dirty_hop2(conn, vault_path)

    return len(node_set)


# ---------------------------------------------------------------------------
# Internal helpers — dirty recompute and threshold gate
# ---------------------------------------------------------------------------


def _schedule_hop2_recompute(conn: sqlite3.Connection) -> None:
    """Trigger .2hop.db recompute if node count exceeds threshold.

    No-op if n <= 500. Called inside an open transaction (BEGIN IMMEDIATE)
    but commits to vault.db are handled by the caller.
    """
    try:
        n = conn.execute("SELECT n FROM node_count").fetchone()[0]
    except Exception:
        return
    if n > _THRESHOLD:
        # Must commit first since we're inside BEGIN IMMEDIATE
        # _recompute_dirty_hop2 will open its own connection after
        # NOTE: we store vault_path via a workaround — we need it here
        # The caller (add_link/remove_link) already has conn connected to vault.db
        # We call COMMIT in the caller, so here we just set a sentinel that
        # _recompute must run. We do the actual recompute lazily in get_hop2_neighborhood.
        pass  # Recompute deferred to get_hop2_neighborhood call


def _recompute_dirty_hop2(conn: sqlite3.Connection, vault_path: Path) -> None:
    """Rebuild hop2 rows for all dirty nodes, batched by subfolder.

    Opens/creates <vault_root>/<subfolder>/.2hop.db for each subfolder
    that has dirty nodes. Clears dirty_hop2=0 after writing.
    """
    dirty_rows = conn.execute(
        "SELECT slug, subfolder FROM nodes WHERE dirty_hop2 = 1"
    ).fetchall()

    if not dirty_rows:
        return

    # Batch by subfolder
    by_subfolder: dict[str, list[str]] = {}
    for slug, subfolder in dirty_rows:
        by_subfolder.setdefault(subfolder, []).append(slug)

    for subfolder, slugs in by_subfolder.items():
        subfolder_path = vault_path / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)
        hop2_db_path = subfolder_path / _HOP2_DB_FILENAME

        hop2_conn = sqlite3.connect(str(hop2_db_path))
        hop2_conn.executescript(VAULT_DB_PRAGMAS)
        try:
            hop2_conn.executescript(HOP2_DB_SCHEMA)
            hop2_conn.commit()

            for slug in slugs:
                out1 = [
                    r[0]
                    for r in conn.execute(
                        "SELECT dst FROM links WHERE src = ?", (slug,)
                    ).fetchall()
                ]
                in1 = [
                    r[0]
                    for r in conn.execute(
                        "SELECT src FROM links WHERE dst = ?", (slug,)
                    ).fetchall()
                ]
                out2_rows = conn.execute(
                    """
                    SELECT DISTINCT dst FROM links
                    WHERE src IN (SELECT dst FROM links WHERE src = ?)
                      AND dst != ?
                    """,
                    (slug, slug),
                ).fetchall()
                out2 = [r[0] for r in out2_rows]
                in2_rows = conn.execute(
                    """
                    SELECT DISTINCT src FROM links
                    WHERE dst IN (SELECT src FROM links WHERE dst = ?)
                      AND src != ?
                    """,
                    (slug, slug),
                ).fetchall()
                in2 = [r[0] for r in in2_rows]

                hop2_conn.execute(
                    """
                    INSERT OR REPLACE INTO hop2 (slug, out1, in1, out2, in2, computed)
                    VALUES (?, ?, ?, ?, ?, unixepoch())
                    """,
                    (
                        slug,
                        dump_compact_yaml(out1),
                        dump_compact_yaml(in1),
                        dump_compact_yaml(out2),
                        dump_compact_yaml(in2),
                    ),
                )
            hop2_conn.commit()
        finally:
            hop2_conn.close()

        # Clear dirty flag for these slugs in vault.db
        conn.executemany(
            "UPDATE nodes SET dirty_hop2 = 0 WHERE slug = ?",
            [(s,) for s in slugs],
        )
        conn.commit()


# ---------------------------------------------------------------------------
# GraphStorePool — connection pool
# ---------------------------------------------------------------------------


class GraphStorePool:
    """Connection pool: 1 RW vault.db + up to 4 RO .2hop.db connections.

    Usage:
        pool = GraphStorePool(vault_path)
        rw_conn = pool.get_rw()          # single persistent RW conn to vault.db
        ro_conn = pool.get_ro("folder")  # cached RO conn to folder/.2hop.db
    """

    _MAX_RO = 4

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        self._rw_conn: Optional[sqlite3.Connection] = None
        # Ordered dict to support LRU eviction: subfolder -> conn
        self._ro_conns: dict[str, sqlite3.Connection] = {}
        self._ro_order: list[str] = []  # tracks insertion order for LRU eviction

    def get_rw(self) -> sqlite3.Connection:
        """Return the single read/write connection to vault.db, opening if needed."""
        if self._rw_conn is None:
            self._rw_conn = open_vault_db(self._vault_path)
        return self._rw_conn

    def get_ro(self, subfolder: str) -> sqlite3.Connection:
        """Return a cached read-only connection to <subfolder>/.2hop.db.

        Evicts the least-recently-used entry when pool exceeds _MAX_RO.
        """
        if subfolder in self._ro_conns:
            # Move to end (most recently used)
            self._ro_order.remove(subfolder)
            self._ro_order.append(subfolder)
            return self._ro_conns[subfolder]

        # Open new connection
        hop2_db = self._vault_path / subfolder / _HOP2_DB_FILENAME
        conn = sqlite3.connect(str(hop2_db))
        conn.executescript(VAULT_DB_PRAGMAS)

        # Evict LRU if at capacity
        if len(self._ro_conns) >= self._MAX_RO:
            lru_key = self._ro_order.pop(0)
            evicted = self._ro_conns.pop(lru_key, None)
            if evicted is not None:
                try:
                    evicted.close()
                except Exception:
                    pass

        self._ro_conns[subfolder] = conn
        self._ro_order.append(subfolder)
        return conn

    def close(self) -> None:
        """Close all connections in the pool."""
        if self._rw_conn is not None:
            try:
                self._rw_conn.close()
            except Exception:
                pass
            self._rw_conn = None
        for conn in self._ro_conns.values():
            try:
                conn.close()
            except Exception:
                pass
        self._ro_conns.clear()
        self._ro_order.clear()
