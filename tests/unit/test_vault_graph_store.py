"""Unit tests for vault/graph_store.py — Two-tier SQLite graph store.

Tests cover:
- vault.db creation with WAL mode and correct PRAGMAs
- ensure_schema creates nodes, links tables, node_count view
- upsert_node inserts and updates
- add_link dirty propagation (src, dst, hop-1 neighbors)
- remove_link dirty propagation
- get_hop2_neighborhood on-the-fly (n<=500) and LRU cache
- Threshold gate: n<=500 on-the-fly, n>500 materializes .2hop.db
- _recompute_dirty_hop2 creates .2hop.db and clears dirty flag
- has_socratic_link_for_note True/False cases
- migrate_from_json reads .pb-graph.yaml or legacy JSON, inserts nodes+links
- migrate_from_json is idempotent
- GraphStorePool: single RW conn, up to 4 RO .2hop.db conns
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
import yaml

# NOTE: These imports will fail until graph_store.py is created (RED phase).
from pb.vault.graph_store import (
    GraphStorePool,
    add_link,
    ensure_schema,
    get_hop2_neighborhood,
    has_socratic_link_for_note,
    migrate_from_json,
    open_vault_db,
    remove_link,
    upsert_node,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_graph_yaml(vault_path: Path, edges: dict[str, list[str]]) -> None:
    """Write a .pb-graph.yaml file at vault_path for migration tests."""
    graph_path = vault_path / ".pb-graph.yaml"
    graph_path.write_text(yaml.safe_dump({"edges": edges}, sort_keys=False))


def _node_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0]


def _link_count(conn: sqlite3.Connection) -> int:
    return conn.execute("SELECT COUNT(*) FROM links").fetchone()[0]


# ---------------------------------------------------------------------------
# Test: open_vault_db
# ---------------------------------------------------------------------------


def test_open_vault_db_creates_vault_db(tmp_path: Path) -> None:
    """open_vault_db creates vault.db at vault root with WAL mode."""
    conn = open_vault_db(tmp_path)
    try:
        db_path = tmp_path / "vault.db"
        assert db_path.exists(), "vault.db should be created at vault root"
        # Verify WAL mode is active
        row = conn.execute("PRAGMA journal_mode").fetchone()
        assert row[0] == "wal", f"Expected WAL mode, got {row[0]}"
    finally:
        conn.close()


def test_open_vault_db_applies_all_pragmas(tmp_path: Path) -> None:
    """open_vault_db applies all startup PRAGMAs including NORMAL synchronous."""
    conn = open_vault_db(tmp_path)
    try:
        sync_row = conn.execute("PRAGMA synchronous").fetchone()
        # NORMAL = 1 in SQLite integer representation
        assert sync_row[0] == 1, f"Expected synchronous=NORMAL(1), got {sync_row[0]}"
        temp_row = conn.execute("PRAGMA temp_store").fetchone()
        # MEMORY = 2 in SQLite integer representation
        assert temp_row[0] == 2, f"Expected temp_store=MEMORY(2), got {temp_row[0]}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: ensure_schema
# ---------------------------------------------------------------------------


def test_ensure_schema_creates_nodes_table(tmp_path: Path) -> None:
    """ensure_schema creates nodes table with correct columns."""
    conn = open_vault_db(tmp_path)
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(nodes)").fetchall()
        }
        assert "slug" in columns, "nodes table missing 'slug' column"
        assert "subfolder" in columns, "nodes table missing 'subfolder' column"
        assert "dirty_hop2" in columns, "nodes table missing 'dirty_hop2' column"
        assert "updated_at" in columns, "nodes table missing 'updated_at' column"
    finally:
        conn.close()


def test_ensure_schema_creates_links_table(tmp_path: Path) -> None:
    """ensure_schema creates links table with src and dst columns."""
    conn = open_vault_db(tmp_path)
    try:
        columns = {
            row[1] for row in conn.execute("PRAGMA table_info(links)").fetchall()
        }
        assert "src" in columns, "links table missing 'src' column"
        assert "dst" in columns, "links table missing 'dst' column"
    finally:
        conn.close()


def test_ensure_schema_creates_node_count_view(tmp_path: Path) -> None:
    """ensure_schema creates node_count view."""
    conn = open_vault_db(tmp_path)
    try:
        views = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='view'"
            ).fetchall()
        }
        assert "node_count" in views, "node_count view not created"
        n = conn.execute("SELECT n FROM node_count").fetchone()[0]
        assert n == 0, f"Expected 0 nodes initially, got {n}"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: upsert_node
# ---------------------------------------------------------------------------


def test_upsert_node_inserts_new_node(tmp_path: Path) -> None:
    """upsert_node inserts a new node into nodes table."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "my-note", "knowledge/deutsch")
        row = conn.execute(
            "SELECT slug, subfolder FROM nodes WHERE slug = ?", ("my-note",)
        ).fetchone()
        assert row is not None, "Node should be inserted"
        assert row[0] == "my-note"
        assert row[1] == "knowledge/deutsch"
    finally:
        conn.close()


def test_upsert_node_updates_existing_node(tmp_path: Path) -> None:
    """upsert_node updates updated_at when node already exists."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "my-note", "folder-a")
        ts1 = conn.execute(
            "SELECT updated_at FROM nodes WHERE slug = ?", ("my-note",)
        ).fetchone()[0]
        upsert_node(conn, "my-note", "folder-b")
        row = conn.execute(
            "SELECT subfolder, updated_at FROM nodes WHERE slug = ?", ("my-note",)
        ).fetchone()
        assert row is not None
        # upsert should overwrite
        assert row[0] == "folder-b", "subfolder should be updated"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: add_link dirty propagation
# ---------------------------------------------------------------------------


def test_add_link_inserts_edge(tmp_path: Path) -> None:
    """add_link inserts edge into links table."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "a", "folder")
        upsert_node(conn, "b", "folder")
        add_link(conn, "a", "b")
        row = conn.execute(
            "SELECT src, dst FROM links WHERE src=? AND dst=?", ("a", "b")
        ).fetchone()
        assert row is not None, "Link a->b should be inserted"
    finally:
        conn.close()


def test_add_link_marks_src_dst_dirty(tmp_path: Path) -> None:
    """add_link sets dirty_hop2=1 on both src and dst nodes."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "a", "folder")
        upsert_node(conn, "b", "folder")
        add_link(conn, "a", "b")
        rows = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT slug, dirty_hop2 FROM nodes WHERE slug IN (?, ?)", ("a", "b")
            ).fetchall()
        }
        assert rows["a"] == 1, "src 'a' should be marked dirty"
        assert rows["b"] == 1, "dst 'b' should be marked dirty"
    finally:
        conn.close()


def test_add_link_marks_hop1_neighbors_dirty(tmp_path: Path) -> None:
    """add_link marks hop-1 neighbors of src and dst dirty (3-node chain test).

    Chain: A->B, B->C, then add C->D.
    After add C->D: B is hop-1 of C (src=B, dst=C link exists), so B must be dirty.
    """
    conn = open_vault_db(tmp_path)
    try:
        for slug in ["a", "b", "c", "d"]:
            upsert_node(conn, slug, "folder")
        add_link(conn, "a", "b")
        add_link(conn, "b", "c")
        # Clear dirty flags to isolate the next add_link effect
        conn.execute("UPDATE nodes SET dirty_hop2 = 0")
        conn.commit()

        # Now add c->d: b is hop-1 of c (b->c exists), so b must be marked dirty
        add_link(conn, "c", "d")

        dirty = {
            row[0]
            for row in conn.execute(
                "SELECT slug FROM nodes WHERE dirty_hop2 = 1"
            ).fetchall()
        }
        assert "c" in dirty, "c (src) should be dirty"
        assert "d" in dirty, "d (dst) should be dirty"
        assert "b" in dirty, "b (hop-1 of c via b->c link) should be dirty"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: remove_link dirty propagation
# ---------------------------------------------------------------------------


def test_remove_link_deletes_edge_and_marks_dirty(tmp_path: Path) -> None:
    """remove_link deletes edge and marks src+dst dirty."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "a", "folder")
        upsert_node(conn, "b", "folder")
        add_link(conn, "a", "b")
        # Clear dirty flags
        conn.execute("UPDATE nodes SET dirty_hop2 = 0")
        conn.commit()

        remove_link(conn, "a", "b")

        # Edge should be gone
        row = conn.execute(
            "SELECT 1 FROM links WHERE src=? AND dst=?", ("a", "b")
        ).fetchone()
        assert row is None, "Link a->b should be deleted"

        # Both nodes should be dirty
        rows = {
            row[0]: row[1]
            for row in conn.execute(
                "SELECT slug, dirty_hop2 FROM nodes WHERE slug IN (?, ?)", ("a", "b")
            ).fetchall()
        }
        assert rows.get("a") == 1, "src 'a' should be marked dirty after remove"
        assert rows.get("b") == 1, "dst 'b' should be marked dirty after remove"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: get_hop2_neighborhood (on-the-fly mode, n<=500)
# ---------------------------------------------------------------------------


def test_get_hop2_neighborhood_on_the_fly(tmp_path: Path) -> None:
    """get_hop2_neighborhood returns correct out1/in1/out2/in2 for small graph.

    Graph: A->B, A->C, B->D
    For A: out1=[B,C], in1=[], out2=[D] (via B), in2=[]
    """
    conn = open_vault_db(tmp_path)
    try:
        for slug in ["a", "b", "c", "d"]:
            upsert_node(conn, slug, "folder")
        add_link(conn, "a", "b")
        add_link(conn, "a", "c")
        add_link(conn, "b", "d")
    finally:
        conn.close()

    result = get_hop2_neighborhood(tmp_path, "a")
    assert set(result["out1"]) == {"b", "c"}, f"out1 wrong: {result['out1']}"
    assert set(result["in1"]) == set(), f"in1 wrong: {result['in1']}"
    assert set(result["out2"]) == {"d"}, f"out2 wrong: {result['out2']}"
    assert set(result["in2"]) == set(), f"in2 wrong: {result['in2']}"


def test_get_hop2_neighborhood_lru_cache_hit(tmp_path: Path) -> None:
    """get_hop2_neighborhood uses LRU cache on second call (cache hit).

    The public function returns a new dict each call, but the
    underlying _get_hop2_cached function should be called only once.
    We verify cache semantics by checking cache_info and result equality.
    """
    from pb.vault.graph_store import _get_hop2_cached

    # Clear cache before this test
    _get_hop2_cached.cache_clear()

    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "x", "folder")
        upsert_node(conn, "y", "folder")
        add_link(conn, "x", "y")
    finally:
        conn.close()

    # First call populates cache (misses=1)
    r1 = get_hop2_neighborhood(tmp_path, "x")
    info_after_first = _get_hop2_cached.cache_info()

    # Second call should be a cache hit (hits=1)
    r2 = get_hop2_neighborhood(tmp_path, "x")
    info_after_second = _get_hop2_cached.cache_info()

    assert r1 == r2, "Cache hit should return same result"
    assert info_after_second.hits == info_after_first.hits + 1, (
        f"Expected 1 more cache hit; before={info_after_first}, after={info_after_second}"
    )


# ---------------------------------------------------------------------------
# Test: threshold gate (n>500 materializes .2hop.db)
# ---------------------------------------------------------------------------


def test_threshold_gate_small_graph_no_2hop_db(tmp_path: Path) -> None:
    """n<=500: get_hop2_neighborhood computes on-the-fly, .2hop.db not created."""
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "a", "folder")
        upsert_node(conn, "b", "folder")
        add_link(conn, "a", "b")
    finally:
        conn.close()

    get_hop2_neighborhood(tmp_path, "a")
    # .2hop.db should NOT exist in folder/ subdirectory
    hop2_db = tmp_path / "folder" / ".2hop.db"
    assert not hop2_db.exists(), ".2hop.db should not be created for n<=500"


def test_threshold_gate_large_graph_creates_2hop_db(tmp_path: Path) -> None:
    """n>500: _recompute_dirty_hop2 creates .2hop.db in correct subfolder."""
    conn = open_vault_db(tmp_path)
    try:
        # Insert >500 nodes to trigger materialized mode
        subfolder = "big-folder"
        subfolder_path = tmp_path / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)
        for i in range(510):
            upsert_node(conn, f"note-{i}", subfolder)
        # Add a link to trigger dirty propagation
        add_link(conn, "note-0", "note-1")
    finally:
        conn.close()

    # get_hop2_neighborhood for a dirty node should create .2hop.db
    get_hop2_neighborhood(tmp_path, "note-0")
    hop2_db = tmp_path / subfolder / ".2hop.db"
    assert hop2_db.exists(), f".2hop.db should be created at {hop2_db} for n>500"


def test_recompute_dirty_hop2_clears_dirty_flag(tmp_path: Path) -> None:
    """_recompute_dirty_hop2 clears dirty_hop2=0 after recompute."""
    conn = open_vault_db(tmp_path)
    try:
        subfolder = "test-folder"
        subfolder_path = tmp_path / subfolder
        subfolder_path.mkdir(parents=True, exist_ok=True)
        for i in range(510):
            upsert_node(conn, f"slug-{i}", subfolder)
        add_link(conn, "slug-0", "slug-1")
    finally:
        conn.close()

    # Trigger recompute via get_hop2_neighborhood
    get_hop2_neighborhood(tmp_path, "slug-0")

    # dirty_hop2 should be cleared for processed nodes
    conn2 = open_vault_db(tmp_path)
    try:
        dirty_count = conn2.execute(
            "SELECT COUNT(*) FROM nodes WHERE dirty_hop2 = 1 AND subfolder = ?",
            ("test-folder",),
        ).fetchone()[0]
        assert dirty_count == 0, f"Expected 0 dirty nodes after recompute, got {dirty_count}"
    finally:
        conn2.close()


# ---------------------------------------------------------------------------
# Test: has_socratic_link_for_note
# ---------------------------------------------------------------------------


def test_has_socratic_link_for_note_returns_true(tmp_path: Path) -> None:
    """has_socratic_link_for_note returns True when vault.db has socratic-source link."""
    # Create a socratic-source note file
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    socratic_note = knowledge_dir / "socratic-capture-1.md"
    socratic_note.write_text("---\nsource: socratic\nlearning_stage: '#new'\n---\n\nSome content.")

    # Create candidate note
    candidate_note = knowledge_dir / "candidate.md"
    candidate_note.write_text("---\nlearning_stage: '#new'\n---\n\n[[socratic-capture-1]]")

    # Set up vault.db with link from socratic-capture-1 -> candidate
    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "socratic-capture-1", "knowledge")
        upsert_node(conn, "candidate", "knowledge")
        add_link(conn, "socratic-capture-1", "candidate")
    finally:
        conn.close()

    result = has_socratic_link_for_note(tmp_path, "knowledge/candidate.md")
    assert result is True, "Should return True when socratic note links to candidate"


def test_has_socratic_link_for_note_returns_false(tmp_path: Path) -> None:
    """has_socratic_link_for_note returns False when no socratic link exists."""
    knowledge_dir = tmp_path / "knowledge"
    knowledge_dir.mkdir()
    regular_note = knowledge_dir / "regular-note.md"
    regular_note.write_text("---\nsource: manual\n---\n\nSome content.")

    candidate_note = knowledge_dir / "candidate2.md"
    candidate_note.write_text("---\nlearning_stage: '#new'\n---\n\n[[regular-note]]")

    conn = open_vault_db(tmp_path)
    try:
        upsert_node(conn, "regular-note", "knowledge")
        upsert_node(conn, "candidate2", "knowledge")
        add_link(conn, "regular-note", "candidate2")
    finally:
        conn.close()

    result = has_socratic_link_for_note(tmp_path, "knowledge/candidate2.md")
    assert result is False, "Should return False when no socratic source note links to candidate"


# ---------------------------------------------------------------------------
# Test: migrate_from_json
# ---------------------------------------------------------------------------


def test_migrate_from_json_inserts_nodes_and_links(tmp_path: Path) -> None:
    """migrate_from_json reads .pb-graph.yaml edges and inserts into vault.db."""
    edges = {
        "knowledge/deutsch/note-a.md": ["knowledge/deutsch/note-b.md", "knowledge/deutsch/note-c.md"],
        "knowledge/deutsch/note-b.md": ["knowledge/deutsch/note-c.md"],
        "knowledge/deutsch/note-c.md": [],
    }
    _make_graph_yaml(tmp_path, edges)

    conn = open_vault_db(tmp_path)
    try:
        count = migrate_from_json(conn, tmp_path)
        assert count >= 3, f"Expected at least 3 nodes migrated, got {count}"

        # Verify node-a is in nodes
        row = conn.execute(
            "SELECT slug, subfolder FROM nodes WHERE slug = ?", ("note-a",)
        ).fetchone()
        assert row is not None, "note-a should be in nodes after migration"
        assert row[1] == "knowledge/deutsch", f"subfolder wrong: {row[1]}"

        # Verify link exists
        link = conn.execute(
            "SELECT 1 FROM links WHERE src = ? AND dst = ?", ("note-a", "note-b")
        ).fetchone()
        assert link is not None, "link note-a -> note-b should exist after migration"
    finally:
        conn.close()


def test_migrate_from_json_is_idempotent(tmp_path: Path) -> None:
    """migrate_from_json is idempotent — running twice does not duplicate."""
    edges = {
        "folder/note-x.md": ["folder/note-y.md"],
        "folder/note-y.md": [],
    }
    _make_graph_yaml(tmp_path, edges)

    conn = open_vault_db(tmp_path)
    try:
        migrate_from_json(conn, tmp_path)
        count_nodes_1 = _node_count(conn)
        count_links_1 = _link_count(conn)

        # Run again — should not duplicate
        migrate_from_json(conn, tmp_path)
        count_nodes_2 = _node_count(conn)
        count_links_2 = _link_count(conn)

        assert count_nodes_1 == count_nodes_2, (
            f"Node count changed on second migration: {count_nodes_1} -> {count_nodes_2}"
        )
        assert count_links_1 == count_links_2, (
            f"Link count changed on second migration: {count_links_1} -> {count_links_2}"
        )
    finally:
        conn.close()


def test_migrate_from_json_no_graph_file(tmp_path: Path) -> None:
    """migrate_from_json returns 0 and does nothing if no graph file is present."""
    conn = open_vault_db(tmp_path)
    try:
        count = migrate_from_json(conn, tmp_path)
        assert count == 0, f"Expected 0 when no graph json, got {count}"
        assert _node_count(conn) == 0, "No nodes should be inserted without graph json"
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test: GraphStorePool
# ---------------------------------------------------------------------------


def test_graph_store_pool_provides_rw_connection(tmp_path: Path) -> None:
    """GraphStorePool.get_rw returns a single RW connection to vault.db."""
    pool = GraphStorePool(tmp_path)
    conn1 = pool.get_rw()
    conn2 = pool.get_rw()
    assert conn1 is conn2, "GraphStorePool should return the same RW connection on every call"
    # Verify it is a valid connection (can query)
    row = conn1.execute("SELECT n FROM node_count").fetchone()
    assert row is not None


def test_graph_store_pool_ro_connections_for_2hop(tmp_path: Path) -> None:
    """GraphStorePool.get_ro opens .2hop.db read-only per subfolder."""
    subfolder = "test-sub"
    subfolder_path = tmp_path / subfolder
    subfolder_path.mkdir(parents=True, exist_ok=True)

    # Create a .2hop.db in the subfolder with the hop2 schema
    hop2_db_path = subfolder_path / ".2hop.db"
    setup_conn = sqlite3.connect(str(hop2_db_path))
    setup_conn.executescript("""
        CREATE TABLE IF NOT EXISTS hop2 (
            slug TEXT PRIMARY KEY,
            out1 TEXT NOT NULL DEFAULT '[]',
            in1  TEXT NOT NULL DEFAULT '[]',
            out2 TEXT NOT NULL DEFAULT '[]',
            in2  TEXT NOT NULL DEFAULT '[]',
            computed INTEGER NOT NULL DEFAULT (unixepoch())
        );
    """)
    setup_conn.commit()
    setup_conn.close()

    pool = GraphStorePool(tmp_path)
    ro_conn = pool.get_ro(subfolder)
    assert ro_conn is not None, "get_ro should return a connection to .2hop.db"

    # Same subfolder should return the same connection (cached)
    ro_conn2 = pool.get_ro(subfolder)
    assert ro_conn is ro_conn2, "get_ro should return the same cached connection for the same subfolder"
