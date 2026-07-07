"""Unit tests for INV-3 enforcement and lifecycle.advance_stage().

Tests cover:
- validate_no_learning_without_socratic: raises RuleViolation when no source:socratic link
- validate_no_learning_without_socratic: returns None when socratic link exists
- advance_stage('#learning'): raises RuleViolation when no source:socratic link
- advance_stage('#learning'): writes frontmatter when socratic link exists
- advance_stage('#learnt'): writes frontmatter WITHOUT invoking INV-3 gate
- advance_stage('#new'): writes frontmatter WITHOUT invoking INV-3 gate
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from pb.core.rules import RuleViolation, validate_no_learning_without_socratic
from pb.vault.lifecycle import advance_stage, read_frontmatter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_note(vault: Path, rel: str, fm: dict, body: str = "Body.") -> Path:
    """Write a note with YAML frontmatter to vault."""
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    yaml_text = yaml.dump(fm, default_flow_style=False)
    p.write_text(f"---\n{yaml_text}---\n\n{body}")
    return p


# ---------------------------------------------------------------------------
# Tests for validate_no_learning_without_socratic
# ---------------------------------------------------------------------------


class TestValidateNoLearningWithoutSocratic:
    """INV-3 rule validator tests."""

    def test_raises_rule_violation_when_no_socratic_link(self, tmp_path, monkeypatch):
        """Test 1: raises RuleViolation starting 'INV-3:' when no socratic link found."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#new"})

        # Monkeypatch graph_store to return no socratic link
        import pb.vault.graph_store as gs
        monkeypatch.setattr(gs, "has_socratic_link_for_note", lambda vp, np: False)

        with pytest.raises(RuleViolation) as exc_info:
            validate_no_learning_without_socratic(note, vault)

        assert str(exc_info.value).startswith("INV-3:")

    def test_returns_none_when_socratic_link_exists(self, tmp_path, monkeypatch):
        """Test 2: returns None (no raise) when a linked source:socratic note exists."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#new"})

        import pb.vault.graph_store as gs
        monkeypatch.setattr(gs, "has_socratic_link_for_note", lambda vp, np: True)

        result = validate_no_learning_without_socratic(note, vault)
        assert result is None


# ---------------------------------------------------------------------------
# Tests for lifecycle.advance_stage
# ---------------------------------------------------------------------------


class TestAdvanceStage:
    """advance_stage() centralised stage setter tests."""

    def test_advance_to_learning_raises_when_no_socratic_link(self, tmp_path, monkeypatch):
        """Test 3: advance_stage('#learning') raises RuleViolation when no socratic link."""
        vault = tmp_path / "vault"
        vault.mkdir()
        _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#new"})

        import pb.vault.graph_store as gs
        monkeypatch.setattr(gs, "has_socratic_link_for_note", lambda vp, np: False)

        with pytest.raises(RuleViolation):
            advance_stage("knowledge/piano/scales.md", "#learning", vault)

    def test_advance_to_learning_writes_frontmatter_when_socratic_exists(
        self, tmp_path, monkeypatch
    ):
        """Test 4: advance_stage('#learning') writes learning_stage when socratic link exists."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#new"})

        import pb.vault.graph_store as gs
        monkeypatch.setattr(gs, "has_socratic_link_for_note", lambda vp, np: True)

        advance_stage("knowledge/piano/scales.md", "#learning", vault)

        content = note.read_text()
        fm, _ = read_frontmatter(content)
        assert fm["learning_stage"] == "#learning"
        assert "stage_updated" in fm

    def test_advance_to_learnt_skips_inv3_gate(self, tmp_path, monkeypatch):
        """Test 5: advance_stage('#learnt') writes frontmatter without calling has_socratic_link."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#learning"})

        import pb.vault.graph_store as gs

        def _must_not_be_called(vp, np):
            raise AssertionError("has_socratic_link_for_note must NOT be called for #learnt")

        monkeypatch.setattr(gs, "has_socratic_link_for_note", _must_not_be_called)

        advance_stage("knowledge/piano/scales.md", "#learnt", vault)

        content = note.read_text()
        fm, _ = read_frontmatter(content)
        assert fm["learning_stage"] == "#learnt"

    def test_advance_to_new_skips_inv3_gate(self, tmp_path, monkeypatch):
        """Test 6: advance_stage('#new') writes frontmatter without calling has_socratic_link."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = _make_note(vault, "knowledge/piano/scales.md", {"learning_stage": "#stale"})

        import pb.vault.graph_store as gs

        def _must_not_be_called(vp, np):
            raise AssertionError("has_socratic_link_for_note must NOT be called for #new")

        monkeypatch.setattr(gs, "has_socratic_link_for_note", _must_not_be_called)

        advance_stage("knowledge/piano/scales.md", "#new", vault)

        content = note.read_text()
        fm, _ = read_frontmatter(content)
        assert fm["learning_stage"] == "#new"
