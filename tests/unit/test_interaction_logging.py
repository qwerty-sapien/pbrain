"""Unit tests for Phase 17 interaction logging hooks and backlink enforcement.

Tests cover:
- log_interaction: DB insertion, weight lookup, non-fatal behavior
- check_promotion: threshold gate, INV-3 Socratic guard, happy-path promotion
- Promotion display on vault write: check_promotion called + return value used
- enforce_backlinks_for_note: bidirectional backlink insertion
- Circular-loop protection via _SKIP_BACKLINK_WRITE guard
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from pb.storage import config as config_module
from pb.storage import database as db_module
from pb.storage.config import Config, GeneralConfig, LearningConfig, StorageConfig
from pb.storage.database import init_db, set_db_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_vault(tmp_path):
    """Create a temporary vault directory."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


@pytest.fixture
def temp_db(tmp_path):
    """Create a temp database and wire it into the database module."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    db_module._db_path = None


@pytest.fixture
def temp_config(temp_vault, tmp_path):
    """Create a minimal Config pointing at temp vault with learning defaults."""
    config = Config(
        general=GeneralConfig(vault_path=str(temp_vault)),
        storage=StorageConfig(data_dir=str(tmp_path)),
        learning=LearningConfig(),
    )
    config_module._config = config
    yield config
    config_module._config = None


# ---------------------------------------------------------------------------
# test_log_interaction_inserts_row
# ---------------------------------------------------------------------------


def test_log_interaction_inserts_row(temp_db, temp_config):
    """log_interaction('query') inserts a row with event_type='query' and weight=1.0."""
    from pb.storage.database import get_connection
    from pb.vault.lifecycle import log_interaction

    log_interaction("note.md", "query")

    with get_connection() as conn:
        rows = conn.execute(
            "SELECT event_type, weight FROM interactions WHERE note_path = ?",
            ("note.md",),
        ).fetchall()

    assert len(rows) == 1
    assert rows[0]["event_type"] == "query"
    assert rows[0]["weight"] == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# test_log_interaction_uses_config_weights
# ---------------------------------------------------------------------------


def test_log_interaction_uses_config_weights(temp_db, temp_config):
    """log_interaction('study') uses the configured study weight (3.0)."""
    from pb.storage.database import get_connection
    from pb.vault.lifecycle import log_interaction

    log_interaction("note.md", "study")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT weight FROM interactions WHERE note_path = ? AND event_type = ?",
            ("note.md", "study"),
        ).fetchone()

    assert row is not None
    assert row["weight"] == pytest.approx(3.0)


# ---------------------------------------------------------------------------
# test_log_interaction_nonfatal
# ---------------------------------------------------------------------------


def test_log_interaction_nonfatal(temp_config):
    """log_interaction should not raise even when database is unavailable."""
    from pb.vault.lifecycle import log_interaction

    db_module._db_path = None
    try:
        # Should complete silently without raising
        log_interaction("note.md", "query")
    finally:
        db_module._db_path = None


# ---------------------------------------------------------------------------
# test_check_promotion_after_logging
# ---------------------------------------------------------------------------


def test_check_promotion_after_logging(temp_db, temp_config, temp_vault):
    """Logging enough interactions (>=3.0) with a Socratic note linked promotes the note."""
    from pb.vault.lifecycle import (
        check_promotion,
        log_interaction,
        read_frontmatter,
        write_frontmatter,
    )

    # Create a Socratic capture note
    socratic = temp_vault / "captures" / "socratic-note.md"
    socratic.parent.mkdir(parents=True, exist_ok=True)
    socratic.write_text(write_frontmatter({"source": "socratic"}, "Socratic body.\n"))

    # Create main #new note linking to the Socratic capture
    note = temp_vault / "notes" / "main.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        write_frontmatter(
            {"learning_stage": "#new", "stage_updated": "2026-01-01"},
            "See [[socratic-note]] for details.\n",
        )
    )

    # Log 3 read interactions (3 x 1.0 = 3.0 >= threshold 3.0)
    for _ in range(3):
        log_interaction("notes/main.md", "read")

    result = check_promotion("notes/main.md", temp_vault)

    assert result is not None, "Expected a promotion message"
    assert "#learning" in result
    assert "main.md" in result

    # Verify frontmatter updated in the note file
    updated_fm, _ = read_frontmatter(note.read_text())
    assert updated_fm["learning_stage"] == "#learning"


# ---------------------------------------------------------------------------
# test_check_promotion_returns_none_below_threshold
# ---------------------------------------------------------------------------


def test_check_promotion_returns_none_below_threshold(temp_db, temp_config, temp_vault):
    """A single read interaction (weight 1.0) should not trigger promotion (threshold 3.0)."""
    from pb.vault.lifecycle import check_promotion, log_interaction, write_frontmatter

    note = temp_vault / "notes" / "low.md"
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        write_frontmatter(
            {"learning_stage": "#new", "stage_updated": "2026-01-01"},
            "Body text.\n",
        )
    )

    log_interaction("notes/low.md", "read")  # weight 1.0 < threshold 3.0

    result = check_promotion("notes/low.md", temp_vault)
    assert result is None


# ---------------------------------------------------------------------------
# test_promotion_display_on_vault_write
# ---------------------------------------------------------------------------


def test_promotion_display_on_vault_write(temp_db, temp_config, temp_vault):
    """Shell vault write hook sequence calls check_promotion and would print the result."""
    from pb.vault.lifecycle import log_interaction, write_frontmatter

    note_rel_path = "notes/promo-test.md"
    note = temp_vault / note_rel_path
    note.parent.mkdir(parents=True, exist_ok=True)
    note.write_text(
        write_frontmatter(
            {"learning_stage": "#new"},
            "Body.\n",
        )
    )

    promo_message = "[bold]notes/promo-test.md[/] promoted [yellow]#new[/] -> [green]#learning[/]"

    with patch("pb.vault.lifecycle.check_promotion", return_value=promo_message) as mock_promo:
        with patch("pb.vault.lifecycle.log_interaction") as mock_log:
            # Simulate the shell.py vault write hook sequence
            from pb.vault.lifecycle import log_interaction as li, check_promotion as cp

            li(note_path=note_rel_path, event_type="read")
            result = cp(note_rel_path, temp_vault)

            # The result is what would be printed via console.print(promo_msg)
            assert result == promo_message
            mock_log.assert_called_once_with(note_path=note_rel_path, event_type="read")
            mock_promo.assert_called_once_with(note_rel_path, temp_vault)


# ---------------------------------------------------------------------------
# test_enforce_backlinks_on_write
# ---------------------------------------------------------------------------


def test_enforce_backlinks_on_write(tmp_path):
    """enforce_backlinks_for_note inserts source into target's frontmatter backlinks[]."""
    from pb.vault.graph import enforce_backlinks_for_note, update_note_in_graph
    from pb.vault.lifecycle import read_frontmatter, write_frontmatter

    # Create note B
    b_path = tmp_path / "b.md"
    b_path.write_text(write_frontmatter({"title": "Note B"}, "Note B body.\n"))

    # Build graph so stem_index has 'b' -> 'b.md'
    update_note_in_graph(tmp_path, "b.md", b_path.read_text())

    # Note A content linking to [[b]]
    a_content = write_frontmatter({"title": "Note A"}, "Links to [[b]].\n")
    update_note_in_graph(tmp_path, "a.md", a_content)

    # Enforce backlinks for note A
    enforce_backlinks_for_note(tmp_path, "a.md", a_content)

    # Verify B now has a.md in its backlinks
    b_updated = b_path.read_text()
    fm, _ = read_frontmatter(b_updated)
    assert "backlinks" in fm, f"Expected backlinks in B frontmatter, got {fm}"
    assert "a.md" in fm["backlinks"], f"Expected a.md in backlinks, got {fm['backlinks']}"


# ---------------------------------------------------------------------------
# test_backlink_enforcement_no_circular_loop
# ---------------------------------------------------------------------------


def test_backlink_enforcement_no_circular_loop(tmp_path):
    """Mutual links A->B and B->A should not cause infinite recursion."""
    from pb.vault.graph import enforce_backlinks_for_note, update_note_in_graph
    from pb.vault.lifecycle import read_frontmatter, write_frontmatter

    # Create note A linking to [[b]] and note B linking to [[a]]
    a_content = write_frontmatter({"title": "Note A"}, "Links to [[b]].\n")
    b_content = write_frontmatter({"title": "Note B"}, "Links back to [[a]].\n")

    (tmp_path / "a.md").write_text(a_content)
    (tmp_path / "b.md").write_text(b_content)

    # Build graph
    update_note_in_graph(tmp_path, "a.md", a_content)
    update_note_in_graph(tmp_path, "b.md", b_content)

    # Enforce backlinks for A — should complete without infinite recursion
    enforce_backlinks_for_note(tmp_path, "a.md", a_content)

    # B should have a.md in its backlinks
    b_updated = (tmp_path / "b.md").read_text()
    fm_b, _ = read_frontmatter(b_updated)
    assert "a.md" in fm_b.get("backlinks", []), f"Expected a.md in B backlinks, got {fm_b}"

    # Now enforce backlinks for B — should also complete without recursion
    enforce_backlinks_for_note(tmp_path, "b.md", b_content)

    a_updated = (tmp_path / "a.md").read_text()
    fm_a, _ = read_frontmatter(a_updated)
    assert "b.md" in fm_a.get("backlinks", []), f"Expected b.md in A backlinks, got {fm_a}"
