"""Unit tests for pb.vault.lifecycle (Phase 17).

Tests cover:
- read_frontmatter: YAML parsing and fallback
- write_frontmatter: round-trip serialization
- log_interaction: DB insertion
- get_weighted_total: sum calculation
- check_promotion: threshold, INV-3 guard, happy path
- check_staleness: decay detection and frontmatter update
- archive_note: sets #archive stage
"""

from __future__ import annotations

import datetime
from pathlib import Path

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
    """Create a temp database and wire it into database module."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    db_module._db_path = None


@pytest.fixture
def temp_config(temp_vault, tmp_path):
    """Create a minimal Config pointing at temp vault, with learning defaults."""
    config = Config(
        general=GeneralConfig(vault_path=str(temp_vault)),
        storage=StorageConfig(data_dir=str(tmp_path)),
        learning=LearningConfig(),
    )
    config_module._config = config
    yield config
    config_module._config = None


# ---------------------------------------------------------------------------
# read_frontmatter
# ---------------------------------------------------------------------------


class TestReadFrontmatter:
    def test_parses_valid_yaml(self):
        from pb.vault.lifecycle import read_frontmatter

        content = "---\ntitle: Hello\nlearning_stage: '#new'\n---\n\nBody text here."
        fm, body = read_frontmatter(content)
        assert fm["title"] == "Hello"
        assert fm["learning_stage"] == "#new"
        assert body.strip() == "Body text here."

    def test_returns_empty_dict_when_no_frontmatter(self):
        from pb.vault.lifecycle import read_frontmatter

        content = "# Just a heading\n\nSome text."
        fm, body = read_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_returns_empty_dict_for_unclosed_frontmatter(self):
        from pb.vault.lifecycle import read_frontmatter

        content = "---\ntitle: Unclosed\n"
        fm, body = read_frontmatter(content)
        assert fm == {}
        assert body == content

    def test_returns_empty_dict_for_empty_frontmatter(self):
        from pb.vault.lifecycle import read_frontmatter

        content = "---\n---\n\nBody."
        fm, body = read_frontmatter(content)
        assert fm == {}
        assert body.strip() == "Body."


# ---------------------------------------------------------------------------
# write_frontmatter
# ---------------------------------------------------------------------------


class TestWriteFrontmatter:
    def test_round_trips_correctly(self):
        from pb.vault.lifecycle import read_frontmatter, write_frontmatter

        original_fm = {
            "title": "My Note",
            "learning_stage": "#new",
            "backlinks": ["other-note.md"],
            "stage_updated": "2026-05-01",
            "source": "socratic",
        }
        body = "This is the note body.\n"
        rendered = write_frontmatter(original_fm, body)
        recovered_fm, recovered_body = read_frontmatter(rendered)
        assert recovered_fm["title"] == original_fm["title"]
        assert recovered_fm["learning_stage"] == original_fm["learning_stage"]
        assert recovered_fm["backlinks"] == original_fm["backlinks"]
        assert recovered_fm["stage_updated"] == original_fm["stage_updated"]
        assert recovered_fm["source"] == original_fm["source"]
        assert "note body" in recovered_body

    def test_produces_yaml_block(self):
        from pb.vault.lifecycle import write_frontmatter

        fm = {"key": "value"}
        result = write_frontmatter(fm, "body")
        assert result.startswith("---\n")
        assert "---\n\nbody" in result


# ---------------------------------------------------------------------------
# log_interaction + get_weighted_total
# ---------------------------------------------------------------------------


class TestLogInteraction:
    def test_inserts_row_into_interactions_table(self, temp_db, temp_config):
        from pb.storage.database import get_connection
        from pb.vault.lifecycle import log_interaction

        log_interaction("notes/test.md", "read", domain="python")

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM interactions WHERE note_path = ?",
                ("notes/test.md",),
            ).fetchall()
        assert len(rows) == 1
        assert rows[0]["event_type"] == "read"
        assert rows[0]["weight"] == 1.0
        assert rows[0]["domain"] == "python"

    def test_uses_correct_weight_from_config(self, temp_db, temp_config):
        from pb.storage.database import get_connection
        from pb.vault.lifecycle import log_interaction

        log_interaction("notes/test.md", "socratic")

        with get_connection() as conn:
            row = conn.execute(
                "SELECT weight FROM interactions WHERE note_path = ? AND event_type = ?",
                ("notes/test.md", "socratic"),
            ).fetchone()
        assert row["weight"] == 5.0

    def test_uses_default_weight_for_unknown_event(self, temp_db, temp_config):
        from pb.storage.database import get_connection
        from pb.vault.lifecycle import log_interaction

        log_interaction("notes/test.md", "unknown_event_xyz")

        with get_connection() as conn:
            row = conn.execute(
                "SELECT weight FROM interactions WHERE event_type = ?",
                ("unknown_event_xyz",),
            ).fetchone()
        assert row["weight"] == 1.0

    def test_is_non_fatal_when_db_missing(self, temp_config):
        from pb.vault.lifecycle import log_interaction

        # No temp_db fixture — just ensure it doesn't raise
        db_module._db_path = None
        try:
            log_interaction("notes/test.md", "read")  # should not raise
        finally:
            db_module._db_path = None


class TestGetWeightedTotal:
    def test_returns_correct_sum(self, temp_db, temp_config):
        from pb.vault.lifecycle import get_weighted_total, log_interaction

        log_interaction("notes/a.md", "read")    # weight 1.0
        log_interaction("notes/a.md", "study")   # weight 3.0
        log_interaction("notes/a.md", "socratic")  # weight 5.0

        total = get_weighted_total("notes/a.md")
        assert total == pytest.approx(9.0)

    def test_returns_zero_for_unknown_note(self, temp_db, temp_config):
        from pb.vault.lifecycle import get_weighted_total

        total = get_weighted_total("notes/nonexistent.md")
        assert total == 0.0

    def test_returns_zero_on_db_error(self, temp_config):
        from pb.vault.lifecycle import get_weighted_total

        db_module._db_path = None
        try:
            total = get_weighted_total("notes/a.md")
            assert total == 0.0
        finally:
            db_module._db_path = None


# ---------------------------------------------------------------------------
# check_promotion
# ---------------------------------------------------------------------------


class TestCheckPromotion:
    def _make_note(self, vault: Path, rel_path: str, stage: str, extra_fm: dict | None = None) -> Path:
        """Create a note file with given learning_stage."""
        note = vault / rel_path
        note.parent.mkdir(parents=True, exist_ok=True)
        fm = {"learning_stage": stage, "stage_updated": "2026-01-01"}
        if extra_fm:
            fm.update(extra_fm)
        from pb.vault.lifecycle import write_frontmatter
        note.write_text(write_frontmatter(fm, "Note body.\n"))
        return note

    def test_returns_none_when_total_below_threshold(self, temp_db, temp_config, temp_vault):
        from pb.vault.lifecycle import check_promotion, log_interaction

        self._make_note(temp_vault, "notes/low.md", "#new")
        log_interaction("notes/low.md", "read")  # weight 1.0 < threshold 3.0

        result = check_promotion("notes/low.md", temp_vault)
        assert result is None

    def test_returns_inv3_warning_when_no_socratic_link(self, temp_db, temp_config, temp_vault):
        from pb.vault.lifecycle import check_promotion, log_interaction

        self._make_note(temp_vault, "notes/no-socratic.md", "#new")
        # Log enough interactions to meet threshold
        for _ in range(3):
            log_interaction("notes/no-socratic.md", "read")  # 3 x 1.0 = 3.0 >= threshold

        result = check_promotion("notes/no-socratic.md", temp_vault)
        assert result is not None
        assert "Socratic capture" in result
        assert "no-socratic.md" in result

    def test_promotes_when_threshold_met_and_socratic_link_exists(self, temp_db, temp_config, temp_vault):
        from unittest.mock import patch
        from pb.vault.lifecycle import check_promotion, log_interaction, read_frontmatter

        # Create a #new note
        main_note = temp_vault / "notes" / "main.md"
        main_note.parent.mkdir(parents=True, exist_ok=True)
        from pb.vault.lifecycle import write_frontmatter
        fm = {"learning_stage": "#new", "stage_updated": "2026-01-01"}
        main_note.write_text(write_frontmatter(fm, "See [[my-socratic]] for details.\n"))

        # Log enough interactions to meet threshold (3.0)
        for _ in range(3):
            log_interaction("notes/main.md", "read")

        # Phase 18: INV-3 check uses vault.db; mock it to return True (link exists)
        # Create dummy vault.db so the existence guard passes
        (temp_vault / "vault.db").touch()
        with patch("pb.vault.lifecycle.has_socratic_link_for_note", return_value=True):
            result = check_promotion("notes/main.md", temp_vault)

        assert result is not None
        assert "#learning" in result
        assert "#new" in result

        # Verify frontmatter was updated
        updated_content = main_note.read_text()
        updated_fm, _ = read_frontmatter(updated_content)
        assert updated_fm["learning_stage"] == "#learning"
        assert updated_fm["stage_updated"] == datetime.date.today().isoformat()

    def test_returns_none_for_non_new_stage(self, temp_db, temp_config, temp_vault):
        from pb.vault.lifecycle import check_promotion, log_interaction

        self._make_note(temp_vault, "notes/already-learning.md", "#learning")
        for _ in range(5):
            log_interaction("notes/already-learning.md", "study")

        result = check_promotion("notes/already-learning.md", temp_vault)
        assert result is None


# ---------------------------------------------------------------------------
# check_staleness
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def _make_learnt_note(self, vault: Path, rel_path: str, stage_updated: str) -> Path:
        """Create a #learnt note with given stage_updated date."""
        note = vault / rel_path
        note.parent.mkdir(parents=True, exist_ok=True)
        from pb.vault.lifecycle import write_frontmatter
        fm = {"learning_stage": "#learnt", "stage_updated": stage_updated}
        note.write_text(write_frontmatter(fm, "Learnt note body.\n"))
        return note

    def test_flags_notes_past_decay_threshold(self, temp_config, temp_vault):
        from pb.vault.lifecycle import check_staleness, read_frontmatter

        domain_dir = temp_vault / "domain"
        domain_dir.mkdir()

        old_date = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        self._make_learnt_note(temp_vault, "domain/old-note.md", old_date)

        stale = check_staleness(temp_vault, domain_dir, decay_days=7)
        assert len(stale) == 1
        assert "old-note.md" in stale[0]
        assert "#stale" in stale[0]

        # Verify frontmatter was updated
        note_content = (temp_vault / "domain" / "old-note.md").read_text()
        fm, _ = read_frontmatter(note_content)
        assert fm["learning_stage"] == "#stale"
        assert fm["stage_updated"] == datetime.date.today().isoformat()

    def test_does_not_flag_recent_notes(self, temp_config, temp_vault):
        from pb.vault.lifecycle import check_staleness

        domain_dir = temp_vault / "domain"
        domain_dir.mkdir()

        recent_date = (datetime.date.today() - datetime.timedelta(days=2)).isoformat()
        self._make_learnt_note(temp_vault, "domain/recent-note.md", recent_date)

        stale = check_staleness(temp_vault, domain_dir, decay_days=7)
        assert len(stale) == 0

    def test_uses_config_default_when_decay_days_not_provided(self, temp_config, temp_vault):
        from pb.vault.lifecycle import check_staleness

        domain_dir = temp_vault / "domain"
        domain_dir.mkdir()

        # Default decay_days is 7; 8 days old should be stale
        old_date = (datetime.date.today() - datetime.timedelta(days=8)).isoformat()
        self._make_learnt_note(temp_vault, "domain/old.md", old_date)

        stale = check_staleness(temp_vault, domain_dir)  # no decay_days arg
        assert len(stale) == 1

    def test_ignores_non_learnt_stages(self, temp_config, temp_vault):
        from pb.vault.lifecycle import check_staleness, write_frontmatter

        domain_dir = temp_vault / "domain"
        domain_dir.mkdir()

        old_date = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        for stage in ("#new", "#learning", "#archive"):
            note = domain_dir / f"note-{stage.lstrip('#')}.md"
            note.write_text(write_frontmatter({"learning_stage": stage, "stage_updated": old_date}, "Body.\n"))

        stale = check_staleness(temp_vault, domain_dir, decay_days=7)
        assert len(stale) == 0


# ---------------------------------------------------------------------------
# archive_note
# ---------------------------------------------------------------------------


class TestArchiveNote:
    def test_sets_learning_stage_to_archive(self, temp_config, temp_vault):
        from pb.vault.lifecycle import archive_note, read_frontmatter, write_frontmatter

        note = temp_vault / "notes" / "to-archive.md"
        note.parent.mkdir(parents=True, exist_ok=True)
        note.write_text(write_frontmatter({"learning_stage": "#learnt"}, "Body.\n"))

        result = archive_note("notes/to-archive.md", temp_vault)
        assert result is not None
        assert "archive" in result.lower()

        updated_fm, _ = read_frontmatter(note.read_text())
        assert updated_fm["learning_stage"] == "#archive"
        assert updated_fm["stage_updated"] == datetime.date.today().isoformat()

    def test_returns_none_for_missing_note(self, temp_config, temp_vault):
        from pb.vault.lifecycle import archive_note

        result = archive_note("notes/nonexistent.md", temp_vault)
        assert result is None


# ---------------------------------------------------------------------------
# INV-3 vault.db integration in check_promotion (Phase 18, SOCR-06)
# ---------------------------------------------------------------------------


class TestInv3VaultDb:
    """Tests that check_promotion uses vault.db has_socratic_link_for_note (D-15)."""

    def _make_note(self, vault: Path, rel_path: str, stage: str) -> Path:
        """Create a note file with given learning_stage."""
        note = vault / rel_path
        note.parent.mkdir(parents=True, exist_ok=True)
        from pb.vault.lifecycle import write_frontmatter
        fm = {"learning_stage": stage, "stage_updated": "2026-01-01"}
        note.write_text(write_frontmatter(fm, "Note body.\n"))
        return note

    def test_inv3_uses_vault_db_when_available(self, temp_db, temp_config, temp_vault):
        """When vault.db is available and has a socratic link, promotion succeeds."""
        from unittest.mock import patch
        from pb.vault.lifecycle import check_promotion, log_interaction

        self._make_note(temp_vault, "notes/main.md", "#new")
        # Log enough interactions to meet threshold (3.0)
        for _ in range(3):
            log_interaction("notes/main.md", "read")

        # Create dummy vault.db so the existence guard passes
        (temp_vault / "vault.db").touch()
        # Mock has_socratic_link_for_note to return True (vault.db says link exists)
        with patch(
            "pb.vault.lifecycle.has_socratic_link_for_note", return_value=True
        ) as mock_check:
            result = check_promotion("notes/main.md", temp_vault)

        mock_check.assert_called_once_with(temp_vault, "notes/main.md")
        assert result is not None
        assert "#learning" in result
        assert "#new" in result

    def test_inv3_blocks_without_socratic_link_vault_db(self, temp_db, temp_config, temp_vault):
        """When vault.db reports no socratic link, promotion is blocked with warning."""
        from unittest.mock import patch
        from pb.vault.lifecycle import check_promotion, log_interaction

        self._make_note(temp_vault, "notes/no-link.md", "#new")
        for _ in range(3):
            log_interaction("notes/no-link.md", "read")

        # Create dummy vault.db so the existence guard passes
        (temp_vault / "vault.db").touch()
        # Mock has_socratic_link_for_note to return False (no socratic link in vault.db)
        with patch(
            "pb.vault.lifecycle.has_socratic_link_for_note", return_value=False
        ):
            result = check_promotion("notes/no-link.md", temp_vault)

        assert result is not None
        assert "no linked Socratic capture found" in result
        assert "no-link.md" in result

    def test_inv3_falls_back_to_rglob_when_vault_db_raises(self, temp_db, temp_config, temp_vault):
        """When has_socratic_link_for_note raises, fallback rglob is used instead."""
        from unittest.mock import patch
        from pb.vault.lifecycle import check_promotion, log_interaction, write_frontmatter

        # Create a Socratic capture note in vault (for rglob to find)
        socratic = temp_vault / "captures" / "my-socratic.md"
        socratic.parent.mkdir(parents=True, exist_ok=True)
        socratic.write_text(write_frontmatter({"source": "socratic"}, "Body.\n"))

        # Create main note linking to it
        main = temp_vault / "notes" / "main-fallback.md"
        main.parent.mkdir(parents=True, exist_ok=True)
        fm = {"learning_stage": "#new", "stage_updated": "2026-01-01"}
        main.write_text(write_frontmatter(fm, "See [[my-socratic]] here.\n"))

        for _ in range(3):
            log_interaction("notes/main-fallback.md", "read")

        # Create dummy vault.db so the existence guard passes
        (temp_vault / "vault.db").touch()
        # has_socratic_link_for_note raises (simulates vault.db not yet seeded)
        with patch(
            "pb.vault.lifecycle.has_socratic_link_for_note",
            side_effect=Exception("vault.db not ready"),
        ):
            result = check_promotion("notes/main-fallback.md", temp_vault)

        # Rglob fallback should have found the socratic link and promoted
        assert result is not None
        assert "#learning" in result
