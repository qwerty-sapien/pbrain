"""Tests for AnkiService — Phase 26 extraction.

RED phase: tests written before implementation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from pb.storage.database import init_db, set_db_path
import pb.storage.database as db_module


@pytest.fixture(autouse=True)
def isolated_db(tmp_path):
    """Each test gets its own isolated DB path; reset after."""
    db_path = tmp_path / "pb_test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    db_module._db_path = None


@pytest.fixture
def mock_repo():
    return MagicMock()


@pytest.fixture
def svc(tmp_path, mock_repo):
    from pb.vault.anki_service import AnkiService
    return AnkiService(vault_path=tmp_path, repo=mock_repo)


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestAnkiServiceConstruction:
    def test_anki_service_constructs(self, tmp_path, mock_repo):
        """Construct with tmp_path and mock repo; assert no error."""
        from pb.vault.anki_service import AnkiService
        svc = AnkiService(vault_path=tmp_path, repo=mock_repo)
        assert svc.vault_path == tmp_path
        assert svc.repo is mock_repo


# ---------------------------------------------------------------------------
# Format YAML
# ---------------------------------------------------------------------------


class TestFormatJson:
    def test_load_format_json_missing(self, svc):
        """Returns {} for non-existent deck dir."""
        result = svc.load_format_json("NonExistentDeck")
        assert result == {}

    def test_save_and_load_format_json(self, svc):
        """Save dict; load it back; assert core fields survive YAML normalization."""
        data = {"style_instructions": "Keep it brief", "example_front": "Q?", "example_back": "A."}
        svc.save_format_json("German", data)
        loaded = svc.load_format_json("German")
        assert loaded["style_instructions"] == "Keep it brief"
        assert loaded["example_front"] == "Q?"
        assert loaded["example_back"] == "A."
        assert (svc.vault_path / "pb-anki" / "German" / "format.yaml").exists()

    def test_load_format_json_migrates_legacy_json_file(self, svc):
        """Legacy format.json loads and is migrated to format.yaml."""
        legacy_dir = svc.vault_path / "pb-anki" / "German"
        legacy_dir.mkdir(parents=True)
        legacy = legacy_dir / "format.json"
        legacy.write_text('{"style_instructions": "legacy"}', encoding="utf-8")

        loaded = svc.load_format_json("German")

        assert loaded["style_instructions"] == "legacy"
        assert (legacy_dir / "format.yaml").exists()


# ---------------------------------------------------------------------------
# Context MD
# ---------------------------------------------------------------------------


class TestContextMd:
    def test_append_context_md(self, svc):
        """Append twice; load_context_md returns both lines."""
        svc.append_context_md("German", "First line of context")
        svc.append_context_md("German", "Second line of context")
        content = svc.load_context_md("German")
        assert "First line of context" in content
        assert "Second line of context" in content


# ---------------------------------------------------------------------------
# Run log history
# ---------------------------------------------------------------------------


class TestRunLog:
    def test_get_history_empty(self, svc):
        """Returns [] on empty table."""
        result = svc.get_history()
        assert result == []

    def test_insert_and_get_history(self, svc):
        """insert_run_log then get_history; assert returned dict has run_id, source, card_count."""
        svc.insert_run_log("run-001", "my-note", None, 4, "auto")
        history = svc.get_history()
        assert len(history) == 1
        row = history[0]
        assert row["run_id"] == "run-001"
        assert row["source"] == "auto"
        assert row["card_count"] == 4


# ---------------------------------------------------------------------------
# Rollback
# ---------------------------------------------------------------------------


class TestRollback:
    def test_rollback_run_not_found(self, svc):
        """Rollback unknown run_id returns (False, 'Run ...' not found)."""
        success, msg = svc.rollback_run("nonexistent-run")
        assert success is False
        assert "not found" in msg.lower()


# ---------------------------------------------------------------------------
# Sanitize deck name
# ---------------------------------------------------------------------------


class TestSanitizeDeckName:
    def test_sanitize_deck_name(self, svc):
        """'My/Deck..Name' sanitizes correctly: strips '..', '/' -> '_'."""
        result = svc._sanitize_deck_name("My/Deck..Name")
        assert result == "My_DeckName"

    def test_sanitize_backslash(self, svc):
        """Backslash replaced with underscore."""
        result = svc._sanitize_deck_name("My\\Deck")
        assert result == "My_Deck"


# ---------------------------------------------------------------------------
# summarize_review_edits
# ---------------------------------------------------------------------------


class TestSummarizeReviewEdits:
    def test_summarize_review_edits_empty_returns_empty_string(self, svc):
        """edited_cards=[] returns ''."""
        result = svc.summarize_review_edits("German", [])
        assert result == ""

    def test_summarize_review_edits_flash_failure_returns_raw_diff(self, svc):
        """Gemini failure falls back to the raw diff string."""
        edited_cards = [{"front": "Was ist Hund?", "back": "Dog"}]
        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.side_effect = RuntimeError("Flash unavailable")

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            result = svc.summarize_review_edits("German", edited_cards)

        assert result != ""
        assert "Was ist Hund?" in result or "Dog" in result


class TestGenerationSemantics:
    def test_generate_cards_requires_explicit_emulation_flag(self, svc):
        """Saved emulation metadata does not silently turn on deck imitation."""
        svc.save_format_spec(
            "German",
            {
                "note_types": ["Basic"],
                "emulate_existing_deck": True,
                "emulated_samples": [{"Front": "Q", "Back": "A"}],
            },
        )

        with patch("pb.vault.anki_client.generate_auto_cards", return_value=[] ) as mock_generate:
            svc.generate_cards(
                note_slug="test-note",
                note_content="Some content",
                domain="deutsch",
                deck="German",
                source="auto",
            )

        assert mock_generate.call_args.kwargs["emulate_existing_deck"] is False
