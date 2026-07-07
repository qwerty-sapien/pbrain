"""Unit tests for BrainEngine.detect_orphans (GRPH-02).

Tests cover:
- detect_orphans_finds_isolated_notes: note with no links is orphan; linked note is not
- detect_orphans_empty_vault: empty vault returns empty list
- detect_orphans_skips_underscore_files: _state.md and _index.md are skipped
- detect_orphans_includes_metadata: each orphan dict has required keys
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_engine():
    """Create BrainEngine without real GeminiClient."""
    from pb.core.brain import BrainEngine
    from unittest.mock import MagicMock

    engine = BrainEngine.__new__(BrainEngine)
    engine._gemini = MagicMock()
    engine._graph_stats = None
    engine._model_used = None
    return engine


# ---------------------------------------------------------------------------
# detect_orphans
# ---------------------------------------------------------------------------


class TestDetectOrphans:
    def test_detect_orphans_finds_isolated_notes(self, tmp_path):
        """Note A links to B (neither is orphan); note C has no links (orphan)."""
        (tmp_path / "a.md").write_text("# Note A\n\nSee [[b]]")
        (tmp_path / "b.md").write_text("# Note B\n\nLinked from A")
        (tmp_path / "c.md").write_text("# Note C\n\nStanding alone with no links")

        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        paths = [o["path"] for o in orphans]
        assert "c.md" in paths, "c.md should be detected as orphan"
        assert "a.md" not in paths, "a.md has outgoing link, not an orphan"
        assert "b.md" not in paths, "b.md has inbound link, not an orphan"

    def test_detect_orphans_empty_vault(self, tmp_path):
        """Empty vault returns empty list."""
        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        assert orphans == []

    def test_detect_orphans_skips_underscore_files(self, tmp_path):
        """Notes named _state.md or _index.md are skipped even if they are orphans."""
        (tmp_path / "_state.md").write_text("# State\n\nNo links here")
        (tmp_path / "_index.md").write_text("# Index\n\nNo links here either")
        (tmp_path / "regular.md").write_text("# Regular\n\nAlso no links")

        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        paths = [o["path"] for o in orphans]
        assert "_state.md" not in paths, "_state.md must be skipped"
        assert "_index.md" not in paths, "_index.md must be skipped"
        # regular.md is still an orphan
        assert "regular.md" in paths

    def test_detect_orphans_includes_metadata(self, tmp_path):
        """Each orphan dict must contain all required keys."""
        content = "---\nlearning_stage: \"#new\"\ncreated: \"2026-01-01\"\n---\n# Solo Note\n\nNo links."
        (tmp_path / "solo.md").write_text(content)

        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        assert len(orphans) == 1
        o = orphans[0]
        required_keys = {"path", "title", "learning_stage", "created", "words", "folder"}
        assert required_keys.issubset(o.keys()), f"Missing keys: {required_keys - o.keys()}"
        assert o["path"] == "solo.md"
        assert o["title"] == "Solo Note"
        assert o["learning_stage"] == "#new"
        assert o["created"] == "2026-01-01"
        assert isinstance(o["words"], int)
        assert o["folder"] == "."

    def test_detect_orphans_in_subdirectory(self, tmp_path):
        """Orphans in subdirectories are grouped by folder path."""
        sub = tmp_path / "knowledge"
        sub.mkdir()
        (sub / "orphan.md").write_text("# Orphan\n\nNo links")

        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        paths = [o["path"] for o in orphans]
        assert "knowledge/orphan.md" in paths
        o = next(x for x in orphans if x["path"] == "knowledge/orphan.md")
        assert o["folder"] == "knowledge"

    def test_detect_orphans_all_linked(self, tmp_path):
        """When all notes have links, no orphans returned."""
        (tmp_path / "x.md").write_text("# X\n\nSee [[y]]")
        (tmp_path / "y.md").write_text("# Y\n\nSee [[x]]")

        engine = _make_engine()

        with patch("pb.core.brain.get_vault_path", return_value=tmp_path):
            orphans = engine.detect_orphans()

        assert orphans == [], "Bidirectionally linked notes are not orphans"
