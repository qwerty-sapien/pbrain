"""Unit tests for INV-3 enforcement via validate_no_learning_without_socratic and
check_promotion (defence-in-depth) — Plan 24-04, Task 2B.

These tests verify:
1. Direct rule call raises RuleViolation when no socratic link
2. Direct rule call returns None when socratic link exists
3. check_promotion (pre-existing path) blocks promotion when INV-3 fails
4. Both code paths (advance_stage + check_promotion) consistently reject the same scenario

Has_socratic_link_for_note is monkeypatched for isolation (no real SQLite required).

Requirement coverage:
  SOCR-06 (INV-3 defence in depth: both advance_stage and check_promotion enforce the rule)

Run:
  uv run pytest tests/unit/test_check_promotion_inv3.py -x -q
"""
from __future__ import annotations

import datetime
from pathlib import Path

import pytest
import yaml

from pb.core.rules import RuleViolation, validate_no_learning_without_socratic


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _seed_vault_with_note(vault: Path, rel: str, **fm_extras) -> Path:
    """Write a note file with YAML frontmatter; return the absolute path."""
    p = vault / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    fm: dict = {
        "learning_stage": "#new",
        "stage_updated": datetime.date.today().isoformat(),
    }
    fm.update(fm_extras)
    yaml_text = yaml.dump(fm, default_flow_style=False)
    p.write_text(f"---\n{yaml_text}---\n\nBody.\n")
    return p


# ---------------------------------------------------------------------------
# Tests 1-2: validate_no_learning_without_socratic direct calls
# ---------------------------------------------------------------------------


def test_validate_no_learning_without_socratic_raises_when_no_link(tmp_path, monkeypatch):
    """Test 1: Rule raises RuleViolation('INV-3: ...') when graph reports no socratic link."""
    monkeypatch.setattr(
        "pb.vault.graph_store.has_socratic_link_for_note",
        lambda vault_path, note_path: False,
    )
    with pytest.raises(RuleViolation, match="INV-3"):
        validate_no_learning_without_socratic(Path("knowledge/x/topic.md"), tmp_path)


def test_validate_no_learning_without_socratic_passes_when_link_exists(tmp_path, monkeypatch):
    """Test 2: Rule returns None (no raise) when graph reports a socratic link."""
    monkeypatch.setattr(
        "pb.vault.graph_store.has_socratic_link_for_note",
        lambda vault_path, note_path: True,
    )
    result = validate_no_learning_without_socratic(Path("knowledge/x/topic.md"), tmp_path)
    assert result is None


# ---------------------------------------------------------------------------
# Test 3: check_promotion (pre-existing path) blocks INV-3 — defence in depth
# ---------------------------------------------------------------------------


def test_check_promotion_existing_path_still_blocks_inv3(tmp_path, monkeypatch):
    """Test 3: check_promotion (pre-existing function) enforces INV-3 — defence in depth.

    Even when the note has accumulated enough interaction weight to qualify for
    promotion, check_promotion must NOT promote it if INV-3 is violated.

    The function returns a non-promoting message string (not None) when INV-3 blocks,
    and the frontmatter must remain '#new'.
    """
    from pb.vault.lifecycle import check_promotion

    target_rel = "knowledge/ml/topic.md"
    _seed_vault_with_note(tmp_path, target_rel)

    # Make has_socratic_link_for_note return False (no socratic link)
    monkeypatch.setattr(
        "pb.vault.graph_store.has_socratic_link_for_note",
        lambda vault_path, note_path: False,
    )
    # Create a real vault.db file so the vault_db_exists check passes (uses graph path)
    db_path = tmp_path / "vault.db"
    db_path.write_bytes(b"")  # empty file; has_socratic_link_for_note is patched anyway

    # Artificially push total interactions above the promotion threshold
    monkeypatch.setattr("pb.vault.lifecycle.get_weighted_total", lambda np: 999.0)

    msg = check_promotion(target_rel, tmp_path)

    # INV-3 blocks: message must contain "INV-3" signal OR note must remain #new
    # (check_promotion returns a string with the block reason, not None)
    content = (tmp_path / target_rel).read_text()
    assert "#new" in content, "Frontmatter must remain #new when INV-3 blocks"
    assert "#learning" not in content, "Note must NOT be promoted when INV-3 blocks"

    # The return value should be a non-None string describing the block
    # (lifecycle.py L184-188: returns f"... no linked Socratic capture found ...")
    if msg is not None:
        assert "INV-3" in msg or "Socratic" in msg or "socratic" in msg.lower(), (
            f"Expected INV-3 block message, got: {msg!r}"
        )


# ---------------------------------------------------------------------------
# Test 4: Both entry points (advance_stage + check_promotion) reject same scenario
# ---------------------------------------------------------------------------


def test_advance_stage_and_check_promotion_consistent(tmp_path, monkeypatch):
    """Test 4: advance_stage and check_promotion both reject the no-socratic-link scenario.

    Verifies defence-in-depth: whichever code path is used, the same invariant holds.
    """
    from pb.vault.lifecycle import advance_stage, check_promotion

    target_rel = "knowledge/ml/topic.md"
    _seed_vault_with_note(tmp_path, target_rel)

    monkeypatch.setattr(
        "pb.vault.graph_store.has_socratic_link_for_note",
        lambda vault_path, note_path: False,
    )

    # Path A: advance_stage raises RuleViolation
    with pytest.raises(RuleViolation, match="INV-3"):
        advance_stage(target_rel, "#learning", tmp_path)

    # Frontmatter still #new after advance_stage rejection
    content = (tmp_path / target_rel).read_text()
    assert "#new" in content, "Frontmatter must remain #new after advance_stage rejection"

    # Path B: check_promotion does not raise but does not promote either
    # Set up: vault.db exists (so graph path is taken), score above threshold
    db_path = tmp_path / "vault.db"
    db_path.write_bytes(b"")
    monkeypatch.setattr("pb.vault.lifecycle.get_weighted_total", lambda np: 999.0)

    check_promotion(target_rel, tmp_path)

    content = (tmp_path / target_rel).read_text()
    assert "#new" in content, "Frontmatter must remain #new after check_promotion INV-3 block"
    assert "#learning" not in content, "Note must NOT be promoted by check_promotion when INV-3 blocks"
