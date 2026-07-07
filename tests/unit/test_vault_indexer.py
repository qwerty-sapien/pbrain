"""Unit tests for per-folder vault indexer (pb.vault.indexer).

Tests cover:
- is_folder_index_stale: fresh / stale / missing index
- rebuild_folder_index: note count, path format, hidden file exclusion
- update_folder_index: upsert and FTS5 delete+insert
- search_folder_index: FTS5 match, fallback on bad pattern
- generate_directory_md: structure, self-exclusion, empty folder
- get_folder_graph: prefix filtering, inbound edges
"""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# is_folder_index_stale
# ---------------------------------------------------------------------------


class TestIsFolderIndexStale:
    def test_stale_when_no_index(self, tmp_path):
        (tmp_path / "note.md").write_text("# Hello\nSome content")
        from pb.vault.indexer import is_folder_index_stale

        assert is_folder_index_stale(tmp_path) is True

    def test_fresh_after_rebuild(self, tmp_path):
        (tmp_path / "note.md").write_text("# Hello\nSome content")
        from pb.vault.indexer import is_folder_index_stale, rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        assert is_folder_index_stale(tmp_path) is False

    def test_stale_after_new_file(self, tmp_path):
        (tmp_path / "note.md").write_text("# Hello\nSome content")
        from pb.vault.indexer import is_folder_index_stale, rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        time.sleep(0.05)
        (tmp_path / "new.md").write_text("# New Note\nNew content")
        assert is_folder_index_stale(tmp_path) is True

    def test_ignores_hidden_files(self, tmp_path):
        (tmp_path / "note.md").write_text("# Hello\nSome content")
        from pb.vault.indexer import is_folder_index_stale, rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        time.sleep(0.05)
        # Write a hidden file — should NOT trigger staleness
        (tmp_path / ".hidden.md").write_text("hidden content")
        assert is_folder_index_stale(tmp_path) is False


# ---------------------------------------------------------------------------
# rebuild_folder_index
# ---------------------------------------------------------------------------


class TestRebuildFolderIndex:
    def test_indexes_md_files(self, tmp_path):
        (tmp_path / "a.md").write_text("# Alpha\nContent A")
        (tmp_path / "b.md").write_text("# Beta\nContent B")
        (tmp_path / "c.md").write_text("# Gamma\nContent C")
        from pb.vault.indexer import rebuild_folder_index

        count = rebuild_folder_index(tmp_path, vault_root=tmp_path)
        assert count == 3

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            row = conn.execute("SELECT COUNT(*) FROM notes").fetchone()
            assert row[0] == 3
        finally:
            conn.close()

    def test_paths_are_vault_relative(self, tmp_path):
        sub = tmp_path / "30-people"
        sub.mkdir()
        (sub / "alice.md").write_text("# Alice\nHello")
        from pb.vault.indexer import rebuild_folder_index

        count = rebuild_folder_index(sub, vault_root=tmp_path)
        assert count == 1

        conn = sqlite3.connect(str(sub / ".pb-index.db"))
        try:
            row = conn.execute("SELECT path FROM notes").fetchone()
            assert row[0] == "30-people/alice.md"
        finally:
            conn.close()

    def test_skips_hidden_files(self, tmp_path):
        (tmp_path / ".hidden.md").write_text("# Hidden\nShould not appear")
        (tmp_path / "visible.md").write_text("# Visible\nShould appear")
        from pb.vault.indexer import rebuild_folder_index

        count = rebuild_folder_index(tmp_path, vault_root=tmp_path)
        assert count == 1

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            paths = [row[0] for row in conn.execute("SELECT path FROM notes").fetchall()]
            assert all(".hidden" not in p for p in paths)
        finally:
            conn.close()

    def test_empty_folder(self, tmp_path):
        from pb.vault.indexer import rebuild_folder_index

        count = rebuild_folder_index(tmp_path, vault_root=tmp_path)
        assert count == 0

    def test_extracts_h1_title(self, tmp_path):
        (tmp_path / "note.md").write_text("# My Title\nBody content here")
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            row = conn.execute("SELECT title FROM notes WHERE path = 'note.md'").fetchone()
            assert row is not None
            assert row[0] == "My Title"
        finally:
            conn.close()

    def test_falls_back_to_stem(self, tmp_path):
        (tmp_path / "my-note.md").write_text("No heading here, just body text")
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            row = conn.execute("SELECT title FROM notes WHERE path = 'my-note.md'").fetchone()
            assert row is not None
            assert row[0] == "my-note"
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# rebuild_folder_index -- learning_stage column (Phase 17, LIFE-01)
# ---------------------------------------------------------------------------


class TestRebuildFolderIndexLearningStage:
    def test_learning_stage_column_exists_after_rebuild(self, tmp_path):
        """learning_stage column is present in notes table after rebuild."""
        (tmp_path / "note.md").write_text("# Hello\nSome content")
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
            assert "learning_stage" in cols, "learning_stage column must exist"
        finally:
            conn.close()

    def test_learning_stage_populated_from_frontmatter(self, tmp_path):
        """learning_stage is extracted from YAML frontmatter and stored in SQLite."""
        content = "---\nlearning_stage: \"#learning\"\n---\n# My Note\n\nBody text."
        (tmp_path / "my-note.md").write_text(content)
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            row = conn.execute(
                "SELECT learning_stage FROM notes WHERE path = 'my-note.md'"
            ).fetchone()
            assert row is not None
            assert row[0] == "#learning"
        finally:
            conn.close()

    def test_learning_stage_null_when_absent_from_frontmatter(self, tmp_path):
        """learning_stage is NULL when note has no learning_stage in frontmatter."""
        (tmp_path / "plain.md").write_text("# Plain Note\n\nNo frontmatter here.")
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            row = conn.execute(
                "SELECT learning_stage FROM notes WHERE path = 'plain.md'"
            ).fetchone()
            assert row is not None
            assert row[0] is None
        finally:
            conn.close()

    def test_migrate_learning_stage_is_idempotent(self, tmp_path):
        """Calling rebuild twice does not error (migration is idempotent)."""
        (tmp_path / "note.md").write_text("# Note\nContent")
        from pb.vault.indexer import rebuild_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        rebuild_folder_index(tmp_path, vault_root=tmp_path)  # second run

        conn = sqlite3.connect(str(tmp_path / ".pb-index.db"))
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(notes)").fetchall()}
            assert "learning_stage" in cols
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# update_folder_index
# ---------------------------------------------------------------------------


class TestUpdateFolderIndex:
    def test_upserts_note_in_existing_index(self, tmp_path):
        (tmp_path / "note.md").write_text("# Note\nOriginal content")
        from pb.vault.indexer import rebuild_folder_index, update_folder_index, search_folder_index

        # Build the initial index
        rebuild_folder_index(tmp_path, vault_root=tmp_path)

        # Overwrite note with new content and update index
        (tmp_path / "note.md").write_text("# Note\nNew content about zebra")
        update_folder_index(tmp_path, vault_root=tmp_path, note_rel_path="note.md", content="# Note\nNew content about zebra")

        # FTS5 search should find the new content
        results = search_folder_index(tmp_path, "zebra")
        assert results is not None
        assert len(results) >= 1
        assert any("note.md" in r[0] for r in results)

    def test_skips_when_no_index(self, tmp_path):
        # No .pb-index.db exists
        from pb.vault.indexer import update_folder_index

        # Should not raise, should not create the DB
        update_folder_index(tmp_path, vault_root=tmp_path, note_rel_path="note.md", content="content")
        assert not (tmp_path / ".pb-index.db").exists()


# ---------------------------------------------------------------------------
# search_folder_index
# ---------------------------------------------------------------------------


class TestSearchFolderIndex:
    def test_finds_matching_content(self, tmp_path):
        (tmp_path / "note.md").write_text("# Greetings\nhello world content here")
        from pb.vault.indexer import rebuild_folder_index, search_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        results = search_folder_index(tmp_path, "hello")

        assert results is not None
        assert len(results) >= 1
        # First result path should be vault-relative
        path = results[0][0]
        assert path == "note.md"

    def test_returns_none_on_bad_syntax(self, tmp_path):
        (tmp_path / "note.md").write_text("# Note\nSome content")
        from pb.vault.indexer import rebuild_folder_index, search_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        # FTS5-invalid pattern should return None (OperationalError fallback)
        result = search_folder_index(tmp_path, "OR AND NOT")
        assert result is None

    def test_returns_empty_on_no_match(self, tmp_path):
        (tmp_path / "note.md").write_text("# Note\nSome content here")
        from pb.vault.indexer import rebuild_folder_index, search_folder_index

        rebuild_folder_index(tmp_path, vault_root=tmp_path)
        results = search_folder_index(tmp_path, "zzzznonexistent")
        assert results == []


# ---------------------------------------------------------------------------
# generate_directory_md
# ---------------------------------------------------------------------------


class TestGenerateDirectoryMd:
    def test_includes_notes_and_subfolders(self, tmp_path):
        (tmp_path / "note1.md").write_text("# Note One\nContent")
        (tmp_path / "note2.md").write_text("# Note Two\nContent")
        sub = tmp_path / "subfolder"
        sub.mkdir()
        from pb.vault.indexer import generate_directory_md

        output = generate_directory_md(tmp_path)
        assert "2 notes, 1 subfolders" in output
        assert "## Notes" in output
        assert "## Subfolders" in output
        assert "subfolder/" in output

    def test_excludes_hidden_files(self, tmp_path):
        (tmp_path / ".hidden.md").write_text("# Hidden\nContent")
        (tmp_path / "visible.md").write_text("# Visible\nContent")
        from pb.vault.indexer import generate_directory_md

        output = generate_directory_md(tmp_path)
        assert "1 notes" in output
        assert ".hidden" not in output

    def test_empty_folder(self, tmp_path):
        from pb.vault.indexer import generate_directory_md

        output = generate_directory_md(tmp_path)
        assert "0 notes, 0 subfolders" in output


# ---------------------------------------------------------------------------
# get_folder_graph
# ---------------------------------------------------------------------------


class TestGetFolderGraph:
    def test_filters_to_folder_prefix(self, tmp_path):
        # Create vault with notes in two folders; only people/ links
        people = tmp_path / "30-people"
        knowledge = tmp_path / "knowledge"
        people.mkdir()
        knowledge.mkdir()

        (people / "alice.md").write_text("# Alice\nKnows [[bob]]")
        (people / "bob.md").write_text("# Bob\nKnows [[alice]]")
        (knowledge / "ml.md").write_text("# ML\nMachine learning")

        from pb.vault.graph import load_vault_graph, get_folder_graph

        # Build vault graph first so it exists on disk
        load_vault_graph(tmp_path)
        result = get_folder_graph(tmp_path, "30-people")

        # All returned keys should be in the 30-people/ prefix
        for source in result:
            assert source.startswith("30-people/"), f"Unexpected source: {source}"

        # knowledge/ml.md should NOT appear as a source (no links to 30-people/)
        assert "knowledge/ml.md" not in result

    def test_includes_inbound_edges(self, tmp_path):
        # note in knowledge/ links to note in 30-people/
        people = tmp_path / "30-people"
        knowledge = tmp_path / "knowledge"
        people.mkdir()
        knowledge.mkdir()

        (people / "alice.md").write_text("# Alice\nHello")
        (knowledge / "ml.md").write_text("# ML\nSee [[alice]] for context")

        from pb.vault.graph import load_vault_graph, get_folder_graph

        load_vault_graph(tmp_path)
        result = get_folder_graph(tmp_path, "30-people")

        # The knowledge source with an inbound link to people/ should appear
        assert "knowledge/ml.md" in result
        # And its filtered target list should only contain 30-people/ notes
        for t in result["knowledge/ml.md"]:
            assert t.startswith("30-people/")

    def test_empty_folder(self, tmp_path):
        # No edges matching "nonexistent/" prefix
        (tmp_path / "note.md").write_text("# Note\nNo links")
        from pb.vault.graph import load_vault_graph, get_folder_graph

        load_vault_graph(tmp_path)
        result = get_folder_graph(tmp_path, "nonexistent")
        assert result == {}
