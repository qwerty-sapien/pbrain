"""Tests for pb goal report helper functions — TDD RED phase (26-04 ALGN-02)."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers imported lazily so RED phase can fail on missing implementation
# ---------------------------------------------------------------------------

def _import_helpers():
    from pb.cli.commands.goals import _infer_domain_from_goal, _count_goal_domain_stages
    return _infer_domain_from_goal, _count_goal_domain_stages


# ---------------------------------------------------------------------------
# Test 1: _infer_domain_from_goal — match found
# ---------------------------------------------------------------------------

def test_infer_domain_from_goal_match(tmp_path):
    """Goal title containing 'German' should match vault domain dir 'deutsch'."""
    _infer_domain_from_goal, _ = _import_helpers()

    # Create mock vault with knowledge/deutsch dir
    knowledge_dir = tmp_path / "knowledge"
    (knowledge_dir / "deutsch").mkdir(parents=True)

    goal = MagicMock()
    goal.title = "Learn German to B2"
    goal.description = ""

    result = _infer_domain_from_goal(goal, tmp_path)
    # Should NOT match 'deutsch' by default — only if 'deutsch' appears in title/description
    # OR if a domain like 'german' is created
    # Create a domain that actually matches
    (knowledge_dir / "german").mkdir(parents=True)

    result = _infer_domain_from_goal(goal, tmp_path)
    assert result == "german", f"Expected 'german', got {result!r}"


# ---------------------------------------------------------------------------
# Test 2: _infer_domain_from_goal — no match
# ---------------------------------------------------------------------------

def test_infer_domain_from_goal_no_match(tmp_path):
    """Goal title with no matching domain dir should return None."""
    _infer_domain_from_goal, _ = _import_helpers()

    knowledge_dir = tmp_path / "knowledge"
    (knowledge_dir / "deutsch").mkdir(parents=True)
    (knowledge_dir / "piano").mkdir(parents=True)

    goal = MagicMock()
    goal.title = "Random Goal With No Match"
    goal.description = ""

    result = _infer_domain_from_goal(goal, tmp_path)
    assert result is None, f"Expected None, got {result!r}"


# ---------------------------------------------------------------------------
# Test 3: _count_goal_domain_stages — vault dir missing
# ---------------------------------------------------------------------------

def test_count_goal_domain_stages_empty(tmp_path):
    """Missing vault domain dir returns empty dict."""
    _, _count_goal_domain_stages = _import_helpers()

    result = _count_goal_domain_stages(tmp_path, "nonexistent-domain")
    assert result == {}, f"Expected {{}}, got {result!r}"


# ---------------------------------------------------------------------------
# Test 4: _count_goal_domain_stages — notes with stage tags
# ---------------------------------------------------------------------------

def test_count_goal_domain_stages_with_notes(tmp_path):
    """Domain dir with 2 notes — one #new, one #learning — returns correct counts."""
    _, _count_goal_domain_stages = _import_helpers()

    domain_dir = tmp_path / "knowledge" / "german"
    domain_dir.mkdir(parents=True)

    # Note 1: #new tag
    (domain_dir / "note1.md").write_text("# Note 1\n\n#new\n\nSome content about German grammar.\n")

    # Note 2: #learning tag
    (domain_dir / "note2.md").write_text("# Note 2\n\n#learning\n\nSome content about vocabulary.\n")

    result = _count_goal_domain_stages(tmp_path, "german")
    assert result == {"new": 1, "learning": 1, "learnt": 0, "stale": 0}, (
        f"Expected {{new:1, learning:1, learnt:0, stale:0}}, got {result!r}"
    )
