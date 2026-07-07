"""INV-3 integration tests — real SQLite vault.db + real lifecycle code (Plan 24-04, Task 2A).

Tests use no monkeypatching of has_socratic_link_for_note: the real graph code path runs
against a real sqlite vault.db in a tmp directory. This verifies that the assembly of:

  advance_stage -> validate_no_learning_without_socratic -> has_socratic_link_for_note

works correctly with actual database queries.

Requirement coverage:
  SOCR-06 (INV-3: no #new -> #learning without source:socratic link)

Run:
  cd productivity-tool && uv run pytest tests/integration/test_inv3_enforcement.py -x -q
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import yaml

from pb.core.rules import RuleViolation
from pb.vault.lifecycle import advance_stage
from pb.vault.graph_store import open_vault_db, upsert_node, add_link


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_vault_with_note(vault: Path, rel: str, fm_extras: dict | None = None) -> Path:
    """Write a note with frontmatter; return absolute path."""
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    fm: dict = {
        "learning_stage": "#new",
        "stage_updated": datetime.date.today().isoformat(),
        "domain": rel.split("/")[1] if "/" in rel else "misc",
    }
    if fm_extras:
        fm.update(fm_extras)
    yaml_text = yaml.dump(fm, default_flow_style=False)
    p.write_text(f"---\n{yaml_text}---\n\nNote body.\n")
    return p


def _seed_socratic_note(vault: Path, rel: str) -> Path:
    """Write a Socratic source note (source: socratic in frontmatter)."""
    return _seed_vault_with_note(vault, rel, fm_extras={"source": "socratic"})


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


def test_inv3_blocks_learning_without_socratic_link(tmp_path):
    """A #new note with no source:socratic link cannot advance to #learning.

    Scenario: note exists in graph, but no socratic note links to it.
    Expected: advance_stage('#learning') raises RuleViolation("INV-3: ...").
    """
    target_rel = "knowledge/ml/topic.md"
    _seed_vault_with_note(tmp_path, target_rel)

    # Initialise vault.db and register the note node — no socratic link added
    conn = open_vault_db(tmp_path)
    upsert_node(conn, "topic", "knowledge/ml")
    conn.close()

    with pytest.raises(RuleViolation, match="INV-3"):
        advance_stage(target_rel, "#learning", tmp_path)

    # Frontmatter must still show #new (no write occurred)
    content = (tmp_path / target_rel).read_text()
    assert "#new" in content
    assert "#learning" not in content


def test_inv3_allows_learning_with_socratic_link(tmp_path):
    """A #new note with a linked source:socratic note CAN advance to #learning.

    Scenario: a Socratic debrief note exists and is linked to the target note
    in vault.db. advance_stage('#learning') should succeed.
    """
    target_rel = "knowledge/ml/topic.md"
    socratic_rel = "knowledge/ml/topic-debrief.md"

    _seed_vault_with_note(tmp_path, target_rel)
    _seed_socratic_note(tmp_path, socratic_rel)  # source: socratic in frontmatter

    # Register both nodes and add a link: topic-debrief -> topic
    # (debrief note links TO the concept note; has_socratic_link_for_note
    #  queries "which src nodes link to candidate?" then checks their frontmatter)
    conn = open_vault_db(tmp_path)
    upsert_node(conn, "topic", "knowledge/ml")
    upsert_node(conn, "topic-debrief", "knowledge/ml")
    add_link(conn, "topic-debrief", "topic")  # debrief -> topic
    conn.close()

    # Should NOT raise — socratic link exists
    advance_stage(target_rel, "#learning", tmp_path)

    # Frontmatter must have been updated to #learning
    content = (tmp_path / target_rel).read_text()
    assert "#learning" in content
    assert "#new" not in content or "learning_stage: '#learning'" in content


def test_inv3_does_not_fire_for_non_learning_transitions(tmp_path):
    """advance_stage to #learnt, #stale, #archive must not invoke the INV-3 gate.

    Scenario: a note in #learning stage advances to #learnt — no vault.db, no
    socratic link. This must succeed without raising (INV-3 only applies to the
    #new -> #learning transition).
    """
    target_rel = "knowledge/ml/topic.md"
    # Note is already #learning — we are advancing to #learnt
    _seed_vault_with_note(tmp_path, target_rel, fm_extras={"learning_stage": "#learning"})
    # No vault.db created, no graph nodes or links — INV-3 must not fire

    # Must not raise
    advance_stage(target_rel, "#learnt", tmp_path)

    content = (tmp_path / target_rel).read_text()
    assert "#learnt" in content
