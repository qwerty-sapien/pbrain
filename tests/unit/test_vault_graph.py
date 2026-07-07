"""Unit tests for persistent vault graph (pb.vault.graph).

Tests cover:
- _full_rebuild: adjacency list from vault notes with wiki-links
- graph_to_adjacency_text: text serialization for LLM prompt
- update_note_in_graph: incremental updates on vault_write
- load_vault_graph: cache hit / staleness detection / rebuild
- _compute_backlinks: inverted edge map for backlink index
- get_backlinks: public backlink accessor
- bulk_vault_write: multi-note write with single graph rebuild
"""

from __future__ import annotations

import time
from pathlib import Path

import pytest
import yaml


# ---------------------------------------------------------------------------
# _full_rebuild
# ---------------------------------------------------------------------------


class TestFullRebuild:
    def test_builds_edges_from_wiki_links(self, tmp_path):
        (tmp_path / "alice.md").write_text("Links to [[bob]]")
        (tmp_path / "bob.md").write_text("Links back to [[alice]]")
        (tmp_path / "carol.md").write_text("No links")

        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert len(edges) == 3
        assert edges["alice.md"] == ["bob.md"]
        assert edges["bob.md"] == ["alice.md"]
        assert edges["carol.md"] == []

    def test_resolves_across_subdirectories(self, tmp_path):
        (tmp_path / "30-people").mkdir()
        (tmp_path / "knowledge").mkdir()
        (tmp_path / "30-people" / "alice.md").write_text("Knows about [[ml]]")
        (tmp_path / "knowledge" / "ml.md").write_text("Machine learning")

        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert edges["30-people/alice.md"] == ["knowledge/ml.md"]

    def test_skips_hidden_directories(self, tmp_path):
        hidden = tmp_path / ".obsidian"
        hidden.mkdir()
        (hidden / "config.md").write_text("not a note")
        (tmp_path / "real.md").write_text("real note")

        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert len(edges) == 1
        assert "real.md" in edges

    def test_empty_vault(self, tmp_path):
        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert edges == {}

    def test_deduplicates_edges(self, tmp_path):
        (tmp_path / "a.md").write_text("[[b]] and again [[b]] and [[b]]")
        (tmp_path / "b.md").write_text("target")

        from pb.vault.graph import _full_rebuild

        assert len(edges["a.md"]) == 1 if (edges := _full_rebuild(tmp_path)) else False

    def test_handles_aliased_links(self, tmp_path):
        (tmp_path / "a.md").write_text("See [[b|Bob Smith]]")
        (tmp_path / "b.md").write_text("Bob's note")

        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert edges["a.md"] == ["b.md"]

    def test_unresolved_links_produce_no_edges(self, tmp_path):
        (tmp_path / "a.md").write_text("[[nonexistent]]")

        from pb.vault.graph import _full_rebuild

        edges = _full_rebuild(tmp_path)
        assert edges["a.md"] == []


# ---------------------------------------------------------------------------
# graph_to_adjacency_text
# ---------------------------------------------------------------------------


class TestGraphToAdjacencyText:
    def test_formats_edges_and_leaf_nodes(self):
        from pb.vault.graph import graph_to_adjacency_text

        edges = {"a.md": ["b.md"], "b.md": [], "c.md": ["a.md", "b.md"]}
        text, nodes, edge_count = graph_to_adjacency_text(edges)
        assert nodes == 3
        assert edge_count == 3
        assert "a.md -> b.md" in text
        assert "c.md -> a.md, b.md" in text
        # b.md has no outgoing — listed alone
        assert "b.md\n" in text or text.endswith("b.md")

    def test_empty_graph(self):
        from pb.vault.graph import graph_to_adjacency_text

        text, nodes, edges = graph_to_adjacency_text({})
        assert text == ""
        assert nodes == 0
        assert edges == 0


# ---------------------------------------------------------------------------
# update_note_in_graph
# ---------------------------------------------------------------------------


class TestUpdateNoteInGraph:
    def test_adds_new_note_to_graph(self, tmp_path):
        from pb.vault.graph import update_note_in_graph, load_vault_graph

        (tmp_path / "existing.md").write_text("old note")
        # Build initial graph
        edges = load_vault_graph(tmp_path)
        assert "existing.md" in edges

        # Simulate vault_write of a new note
        (tmp_path / "new.md").write_text("links to [[existing]]")
        update_note_in_graph(tmp_path, "new.md", "links to [[existing]]")

        # Read back
        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        assert "new.md" in data["edges"]
        assert data["edges"]["new.md"] == ["existing.md"]

    def test_updates_existing_note_links(self, tmp_path):
        from pb.vault.graph import update_note_in_graph, load_vault_graph

        (tmp_path / "a.md").write_text("[[b]]")
        (tmp_path / "b.md").write_text("target")
        load_vault_graph(tmp_path)  # build initial

        # Update a.md to link to nothing
        update_note_in_graph(tmp_path, "a.md", "no links now")

        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        assert data["edges"]["a.md"] == []

    def test_works_without_existing_graph_file(self, tmp_path):
        from pb.vault.graph import update_note_in_graph

        (tmp_path / "a.md").write_text("content")
        # No .pb-graph.yaml exists — should create it
        update_note_in_graph(tmp_path, "a.md", "content")

        assert (tmp_path / ".pb-graph.yaml").exists()


# ---------------------------------------------------------------------------
# load_vault_graph — staleness detection
# ---------------------------------------------------------------------------


class TestLoadVaultGraph:
    def test_returns_cached_when_fresh(self, tmp_path):
        from pb.vault.graph import load_vault_graph

        (tmp_path / "note.md").write_text("content")
        edges1 = load_vault_graph(tmp_path)
        assert (tmp_path / ".pb-graph.yaml").exists()

        # Second load should use persistent file
        edges2 = load_vault_graph(tmp_path)
        assert edges1 == edges2

    def test_rebuilds_when_stale(self, tmp_path):
        from pb.vault.graph import load_vault_graph

        (tmp_path / "note.md").write_text("content")
        load_vault_graph(tmp_path)

        # Simulate external edit (newer than graph file)
        time.sleep(0.05)
        (tmp_path / "new_note.md").write_text("added outside pb")

        edges = load_vault_graph(tmp_path)
        assert "new_note.md" in edges

    def test_rebuilds_when_missing(self, tmp_path):
        from pb.vault.graph import load_vault_graph

        (tmp_path / "note.md").write_text("content")
        # No graph file — should build from scratch
        edges = load_vault_graph(tmp_path)
        assert "note.md" in edges
        assert (tmp_path / ".pb-graph.yaml").exists()


# ---------------------------------------------------------------------------
# _compute_backlinks
# ---------------------------------------------------------------------------


class TestComputeBacklinks:
    def test_inverts_edges_correctly(self):
        from pb.vault.graph import _compute_backlinks

        edges = {"a.md": ["b.md", "c.md"], "b.md": ["c.md"], "c.md": []}
        bl = _compute_backlinks(edges)

        assert sorted(bl["c.md"]) == ["a.md", "b.md"]
        assert bl["b.md"] == ["a.md"]
        assert bl["a.md"] == []

    def test_empty_edges_returns_empty(self):
        from pb.vault.graph import _compute_backlinks

        bl = _compute_backlinks({})
        assert bl == {}

    def test_external_link_target_silently_skipped(self):
        """Target not in edges keys (external link) should not raise KeyError."""
        from pb.vault.graph import _compute_backlinks

        # "external.md" is referenced but not a key in edges
        edges = {"a.md": ["external.md", "b.md"], "b.md": []}
        bl = _compute_backlinks(edges)
        # external.md should not appear in backlinks (not a known node)
        assert "external.md" not in bl
        assert bl["b.md"] == ["a.md"]
        assert bl["a.md"] == []

    def test_all_keys_present_even_with_no_backlinks(self):
        from pb.vault.graph import _compute_backlinks

        edges = {"a.md": ["b.md"], "b.md": [], "c.md": []}
        bl = _compute_backlinks(edges)
        # All source nodes must be present
        assert set(bl.keys()) == {"a.md", "b.md", "c.md"}


# ---------------------------------------------------------------------------
# _save writes backlinks to YAML
# ---------------------------------------------------------------------------


class TestSaveWritesBacklinks:
    def test_yaml_contains_both_edges_and_backlinks(self, tmp_path):
        from pb.vault.graph import _save

        edges = {"a.md": ["b.md"], "b.md": []}
        _save(tmp_path, edges)

        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        assert "edges" in data
        assert "backlinks" in data

    def test_backlinks_correct_in_saved_json(self, tmp_path):
        from pb.vault.graph import _save

        edges = {"a.md": ["b.md", "c.md"], "b.md": ["c.md"], "c.md": []}
        _save(tmp_path, edges)

        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        assert sorted(data["backlinks"]["c.md"]) == ["a.md", "b.md"]
        assert data["backlinks"]["b.md"] == ["a.md"]
        assert data["backlinks"]["a.md"] == []


# ---------------------------------------------------------------------------
# get_backlinks
# ---------------------------------------------------------------------------


class TestGetBacklinks:
    def test_returns_correct_inverse_after_rebuild(self, tmp_path):
        from pb.vault.graph import load_vault_graph, get_backlinks

        (tmp_path / "a.md").write_text("Links to [[b]]")
        (tmp_path / "b.md").write_text("No links")

        load_vault_graph(tmp_path)  # triggers rebuild + save with backlinks
        bl = get_backlinks(tmp_path)

        assert isinstance(bl, dict)
        assert "a.md" in bl.get("b.md", [])

    def test_returns_dict_when_graph_missing(self, tmp_path):
        """On empty vault dir (no .pb-graph.yaml), get_backlinks should not raise."""
        from pb.vault.graph import get_backlinks

        result = get_backlinks(tmp_path)
        assert isinstance(result, dict)

    def test_returns_dict_on_empty_vault(self, tmp_path):
        from pb.vault.graph import get_backlinks

        # No notes, no graph file
        result = get_backlinks(tmp_path)
        assert result == {}


# ---------------------------------------------------------------------------
# bulk_vault_write
# ---------------------------------------------------------------------------


class TestBulkVaultWrite:
    def test_creates_all_files_on_disk(self, tmp_path):
        from pb.vault.graph import bulk_vault_write

        notes = [
            ("n1.md", "# Note 1"),
            ("n2.md", "# Note 2"),
            ("n3.md", "# Note 3"),
            ("n4.md", "# Note 4"),
            ("n5.md", "# Note 5"),
        ]
        count = bulk_vault_write(tmp_path, notes)

        assert count == 5
        for rel, _ in notes:
            assert (tmp_path / rel).exists(), f"{rel} not written"

    def test_graph_yaml_written_with_edges_and_backlinks(self, tmp_path):
        from pb.vault.graph import bulk_vault_write

        notes = [
            ("a.md", "Links to [[b]]"),
            ("b.md", "Links to [[c]]"),
            ("c.md", "No links"),
        ]
        bulk_vault_write(tmp_path, notes)

        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        assert "edges" in data
        assert "backlinks" in data

    def test_return_count_matches_input(self, tmp_path):
        from pb.vault.graph import bulk_vault_write

        notes = [(f"note{i}.md", f"# Note {i}") for i in range(7)]
        count = bulk_vault_write(tmp_path, notes)
        assert count == 7

    def test_wikilinks_resolved_in_edges(self, tmp_path):
        from pb.vault.graph import bulk_vault_write

        notes = [
            ("folder/note1.md", "# Note 1\n\nLinks to [[note2]]"),
            ("folder/note2.md", "# Note 2\n\nLinks to [[note1]]"),
            ("folder/note3.md", "# Note 3\n\nNo links here"),
        ]
        bulk_vault_write(tmp_path, notes)

        data = yaml.safe_load((tmp_path / ".pb-graph.yaml").read_text())
        edges = data["edges"]
        assert "folder/note1.md" in edges
        assert "folder/note2.md" in edges["folder/note1.md"]

    def test_creates_nested_directories(self, tmp_path):
        from pb.vault.graph import bulk_vault_write

        notes = [
            ("deep/nested/dir/note.md", "# Deep note"),
        ]
        count = bulk_vault_write(tmp_path, notes)
        assert count == 1
        assert (tmp_path / "deep/nested/dir/note.md").exists()


# ---------------------------------------------------------------------------
# load_vault_graph backward compatibility after _save extension
# ---------------------------------------------------------------------------


class TestLoadVaultGraphBackwardCompat:
    def test_returns_edges_only_dict_not_full_structure(self, tmp_path):
        """load_vault_graph still returns plain edges dict, not {edges, backlinks}."""
        from pb.vault.graph import load_vault_graph

        (tmp_path / "a.md").write_text("Links to [[b]]")
        (tmp_path / "b.md").write_text("No links")

        result = load_vault_graph(tmp_path)

        # Should be a dict of str -> list[str], not contain 'edges' or 'backlinks' keys
        assert isinstance(result, dict)
        for key, val in result.items():
            assert isinstance(key, str), f"Key {key!r} is not a string"
            assert isinstance(val, list), f"Value for {key!r} is not a list"
        # Must NOT be the raw persisted graph structure
        assert "edges" not in result
        assert "backlinks" not in result
