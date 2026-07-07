"""Unit tests for pb.vault.anki_client (Phase 18, ANKI-01, ANKI-02).

Tests cover:
- anki_request: ConnectError returns None, success returns result, error field returns None
- is_anki_available: True on success, False on failure
- generate_auto_cards: note_type/card_type handling, front/back from LLM, Cloze detection
- insert_cards_to_db: inserts rows into anki_cards table
- export_cards_to_apkg: creates a real .apkg package through genanki
- export_cards_to_anki: calls addNotes with correct payload structure
- export_cards_to_csv: creates file at vault/pb-anki/export-{date}.csv
- sync_revlog: stores per-deck stats, silently skips when offline
- get_pending_card_count: returns count of status=pending
- get_deck_review_stats: returns dict with deck, cards, reviews keys
"""

from __future__ import annotations

import csv
import json
from unittest.mock import MagicMock, patch

import pytest

from pb.storage.database import init_db, set_db_path
import pb.storage.database as db_module


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def temp_db(tmp_path):
    """Create a temporary database and wire it into database module."""
    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    db_module._db_path = None


# ---------------------------------------------------------------------------
# anki_request
# ---------------------------------------------------------------------------


class TestAnkiRequest:
    def test_returns_none_on_connect_error(self):
        """anki_request returns None when httpx.post raises ConnectError."""
        import httpx
        from pb.vault.anki_client import anki_request

        with patch("pb.vault.anki_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            result = anki_request("deckNames")

        assert result is None

    def test_returns_result_on_success(self):
        """anki_request returns result dict on successful response."""
        from pb.vault.anki_client import anki_request

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": ["Default", "German"], "error": None}

        with patch("pb.vault.anki_client.httpx.post", return_value=mock_response):
            result = anki_request("deckNames")

        assert result == ["Default", "German"]

    def test_returns_none_when_error_field_present(self):
        """anki_request returns None when response has an error field."""
        from pb.vault.anki_client import anki_request

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": None, "error": "Model not found"}

        with patch("pb.vault.anki_client.httpx.post", return_value=mock_response):
            result = anki_request("addNote")

        assert result is None

    def test_returns_none_on_timeout(self):
        """anki_request returns None on timeout exception."""
        import httpx
        from pb.vault.anki_client import anki_request

        with patch("pb.vault.anki_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.TimeoutException("Timeout")
            result = anki_request("deckNames")

        assert result is None


# ---------------------------------------------------------------------------
# is_anki_available
# ---------------------------------------------------------------------------


class TestIsAnkiAvailable:
    def test_returns_true_when_deck_names_succeeds(self):
        """is_anki_available returns True when deckNames succeeds."""
        from pb.vault.anki_client import is_anki_available

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": ["Default"], "error": None}

        with patch("pb.vault.anki_client.httpx.post", return_value=mock_response):
            result = is_anki_available()

        assert result is True

    def test_returns_false_when_anki_offline(self):
        """is_anki_available returns False when Anki is not running."""
        import httpx
        from pb.vault.anki_client import is_anki_available

        with patch("pb.vault.anki_client.httpx.post") as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection refused")
            result = is_anki_available()

        assert result is False


# ---------------------------------------------------------------------------
# generate_auto_cards
# ---------------------------------------------------------------------------


class TestGenerateAutoCards:
    """Tests for generate_auto_cards.

    generate_auto_cards lazily imports get_client from pb.llm.gemini inside
    the function body. We patch the source module attribute so the lazy
    import picks up the mock.
    """

    def test_produces_cards_with_basic_card_type_by_default(self):
        """generate_auto_cards defaults to Basic when note_type is omitted."""
        from pb.vault.anki_client import generate_auto_cards

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = json.dumps([
            {"front": "What is a noun?", "back": "A word for a person, place, or thing.", "sub_deck": "Grammar"}
        ])

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            cards = generate_auto_cards(
                note_slug="german-nouns",
                note_content="## German Nouns\n\nA noun is a word for a person, place, or thing.",
                domain="deutsch",
                deck_base="German",
            )

        assert len(cards) >= 1
        assert cards[0]["card_type"] == "Basic"
        assert cards[0]["front"] == "What is a noun?"
        assert cards[0]["back"] == "A word for a person, place, or thing."

    def test_produces_cards_with_front_and_back_from_llm(self):
        """generate_auto_cards populates front and back from LLM response."""
        from pb.vault.anki_client import generate_auto_cards

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = json.dumps([
            {"front": "Der/Die/Das?", "back": "The definite articles in German.", "sub_deck": "Vocab"}
        ])

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            cards = generate_auto_cards(
                note_slug="german-articles",
                note_content="# Articles\n\nGerman has three genders.",
                domain="deutsch",
            )

        assert cards[0]["front"] == "Der/Die/Das?"
        assert cards[0]["back"] == "The definite articles in German."

    def test_cloze_content_sets_anki_model_cloze(self):
        """generate_auto_cards sets anki_model=Cloze when {{c1::}} cloze syntax detected."""
        from pb.vault.anki_client import generate_auto_cards

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = json.dumps([
            {"front": "Das ist {{c1::ein}} Hund.", "back": "indefinite article for masculine", "sub_deck": "Cloze"}
        ])

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            cards = generate_auto_cards(
                note_slug="german-cloze",
                note_content="# Cloze\n\nUse ein for masculine nouns.",
                domain="deutsch",
                deck_base="German",
            )

        assert cards[0]["anki_model"] == "Cloze"

    def test_discards_cards_outside_selected_note_types(self):
        """Only requested note types survive filtering."""
        from pb.vault.anki_client import generate_auto_cards

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = json.dumps([
            {"note_type": "Basic", "front": "Q1", "back": "A1", "sub_deck": "Concepts"},
            {"note_type": "Cloze", "front": "{{c1::Q2}}", "back": "A2", "sub_deck": "Cloze"},
        ])

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            cards = generate_auto_cards(
                note_slug="german-filter",
                note_content="Some content.",
                domain="deutsch",
                note_types=["Cloze"],
            )

        assert len(cards) == 1
        assert cards[0]["card_type"] == "Cloze"

    def test_returns_empty_list_when_llm_unavailable(self):
        """generate_auto_cards returns [] when LLM is not available."""
        from pb.vault.anki_client import generate_auto_cards

        mock_client = MagicMock()
        mock_client.is_available.return_value = False

        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            cards = generate_auto_cards(
                note_slug="test-note",
                note_content="Some content.",
                domain="test",
            )

        assert cards == []


# ---------------------------------------------------------------------------
# insert_cards_to_db
# ---------------------------------------------------------------------------


class TestInsertCardsToDb:
    def test_inserts_rows_into_anki_cards_table(self, temp_db):
        """insert_cards_to_db inserts card rows into anki_cards table."""
        from pb.vault.anki_client import insert_cards_to_db
        from pb.storage.database import get_connection

        cards = [
            {
                "id": "note-auto-0",
                "note_slug": "test-note",
                "front": "Q: What is Python?",
                "back": "A: A programming language.",
                "card_type": "Basic",
                "status": "pending",
                "deck": "Programming::Concepts",
                "tags": json.dumps(["python", "auto"]),
                "anki_model": "Basic",
                "created_at": "2026-05-04T10:00:00",
                "updated_at": "2026-05-04T10:00:00",
            }
        ]

        count = insert_cards_to_db(cards)
        assert count == 1

        with get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM anki_cards WHERE id = 'note-auto-0'"
            ).fetchone()

        assert row is not None
        assert row["note_slug"] == "test-note"
        assert row["front"] == "Q: What is Python?"
        assert row["card_type"] == "Basic"
        assert row["status"] == "pending"

    def test_returns_zero_for_empty_cards(self, temp_db):
        """insert_cards_to_db returns 0 for empty card list."""
        from pb.vault.anki_client import insert_cards_to_db

        count = insert_cards_to_db([])
        assert count == 0


# ---------------------------------------------------------------------------
# export_cards_to_anki
# ---------------------------------------------------------------------------


class TestExportCardsToAnki:
    def test_calls_add_notes_with_correct_payload(self):
        """export_cards_to_anki calls addNotes with correct model and deck names."""
        from pb.vault.anki_client import export_cards_to_anki

        cards = [
            {
                "id": "note-auto-0",
                "front": "What is X?",
                "back": "X is Y.",
                "deck": "German::Vocab",
                "tags": json.dumps(["deutsch", "auto"]),
                "anki_model": "Basic",
                "status": "pending",
            }
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": [12345], "error": None}

        with patch("pb.vault.anki_client.httpx.post", return_value=mock_response) as mock_post:
            success, msg = export_cards_to_anki(cards)

        assert success is True
        payload = mock_post.call_args.kwargs["json"]
        assert payload["action"] == "addNotes"
        notes = payload["params"]["notes"]
        assert len(notes) == 1
        note = notes[0]
        assert note["deckName"] == "German::Vocab"
        assert note["modelName"] == "Basic"
        assert note["fields"]["Front"] == "What is X?"
        assert note["fields"]["Back"] == "X is Y."
        assert note["options"]["allowDuplicate"] is False

    def test_uses_text_and_extra_fields_for_cloze_model(self):
        """export_cards_to_anki uses Text/Extra fields for Cloze model."""
        from pb.vault.anki_client import export_cards_to_anki

        cards = [
            {
                "id": "cloze-0",
                "front": "Das ist {{c1::ein}} Hund.",
                "back": "indefinite article for masculine",
                "deck": "German::Cloze",
                "tags": json.dumps(["deutsch"]),
                "anki_model": "Cloze",
                "status": "pending",
            }
        ]

        mock_response = MagicMock()
        mock_response.json.return_value = {"result": [99999], "error": None}

        with patch("pb.vault.anki_client.httpx.post", return_value=mock_response) as mock_post:
            success, _ = export_cards_to_anki(cards)

        assert success is True
        payload = mock_post.call_args.kwargs["json"]
        captured_note = payload["params"]["notes"][0]
        assert captured_note["fields"]["Text"] == "Das ist {{c1::ein}} Hund."
        assert captured_note["fields"]["Extra"] == "indefinite article for masculine"

    def test_returns_false_when_anki_unavailable(self):
        """export_cards_to_anki returns (False, message) when AnkiConnect unavailable."""
        from pb.vault.anki_client import export_cards_to_anki

        cards = [
            {
                "id": "note-0",
                "front": "Q?",
                "back": "A.",
                "deck": "Test",
                "tags": "[]",
                "anki_model": "Basic",
                "status": "pending",
            }
        ]

        with patch("pb.vault.anki_client.anki_request", return_value=None):
            success, msg = export_cards_to_anki(cards)

        assert success is False
        assert "unavailable" in msg.lower() or "connect" in msg.lower()


# ---------------------------------------------------------------------------
# export_cards_to_apkg
# ---------------------------------------------------------------------------


class TestExportCardsToApkg:
    def test_creates_apkg_and_marks_cards_exported(self, temp_db, tmp_path):
        """export_cards_to_apkg writes a package and marks rows as exported."""
        pytest.importorskip("genanki")
        from pb.storage.database import get_connection
        from pb.vault.anki_client import export_cards_to_apkg, insert_cards_to_db

        cards = [
            {
                "id": "note-apkg-0",
                "note_slug": "test-note",
                "front": "What is retrieval practice?",
                "back": "Actively recalling knowledge from memory.",
                "card_type": "Basic",
                "status": "pending",
                "deck": "Learning::Concepts",
                "tags": json.dumps(["learning", "recall"]),
                "anki_model": "Basic",
                "created_at": "2026-05-04T10:00:00",
                "updated_at": "2026-05-04T10:00:00",
            }
        ]
        insert_cards_to_db(cards)

        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        success, out_path, msg = export_cards_to_apkg(cards, vault_path)

        assert success is True
        assert out_path is not None
        assert out_path.exists()
        assert out_path.suffix == ".apkg"
        assert "Packaged 1 cards" in msg

        with get_connection() as conn:
            row = conn.execute(
                "SELECT status, exported_at FROM anki_cards WHERE id = ?",
                ("note-apkg-0",),
            ).fetchone()

        assert row is not None
        assert row["status"] == "exported"
        assert row["exported_at"] is not None

    def test_returns_false_when_genanki_is_missing(self, tmp_path):
        """Missing genanki degrades cleanly without raising."""
        from pb.vault.anki_client import export_cards_to_apkg

        cards = [
            {
                "id": "note-apkg-1",
                "front": "Q?",
                "back": "A",
                "deck": "Learning::Concepts",
                "tags": "[]",
                "anki_model": "Basic",
            }
        ]
        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        with patch("pb.vault.anki_client.importlib.import_module", side_effect=ModuleNotFoundError):
            success, out_path, msg = export_cards_to_apkg(cards, vault_path)

        assert success is False
        assert out_path is None
        assert "genanki" in msg


# ---------------------------------------------------------------------------
# export_cards_to_csv
# ---------------------------------------------------------------------------


class TestExportCardsToCsv:
    def test_creates_file_at_vault_pb_anki_export_date_csv(self, tmp_path):
        """export_cards_to_csv creates file at vault/pb-anki/export-{date}.csv."""
        import datetime
        from pb.vault.anki_client import export_cards_to_csv

        cards = [
            {
                "id": "note-0",
                "front": "Question 1",
                "back": "Answer 1",
                "deck": "German::Vocab",
                "tags": json.dumps(["auto"]),
            }
        ]

        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        out_path = export_cards_to_csv(cards, vault_path)

        today = datetime.date.today().isoformat()
        expected_path = vault_path / "pb-anki" / f"export-{today}.csv"
        assert out_path == expected_path
        assert out_path.exists()

    def test_csv_contains_front_back_deck_tags_columns(self, tmp_path):
        """export_cards_to_csv file contains front, back, deck, tags columns."""
        from pb.vault.anki_client import export_cards_to_csv

        cards = [
            {
                "id": "note-0",
                "front": "Was ist das?",
                "back": "What is that?",
                "deck": "German::Vocab",
                "tags": json.dumps(["deutsch", "auto"]),
            }
        ]

        vault_path = tmp_path / "vault"
        vault_path.mkdir()

        out_path = export_cards_to_csv(cards, vault_path)

        with open(out_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert reader.fieldnames is not None
        assert "front" in reader.fieldnames
        assert "back" in reader.fieldnames
        assert "deck" in reader.fieldnames
        assert "tags" in reader.fieldnames
        assert len(rows) == 1
        assert rows[0]["front"] == "Was ist das?"
        assert rows[0]["back"] == "What is that?"
        assert rows[0]["deck"] == "German::Vocab"


# ---------------------------------------------------------------------------
# sync_revlog
# ---------------------------------------------------------------------------


class TestSyncRevlog:
    def test_stores_per_deck_stats_in_anki_revlog(self, temp_db):
        """sync_revlog stores per-deck review stats in anki_revlog table."""
        from pb.vault.anki_client import sync_revlog
        from pb.storage.database import get_connection

        def fake_anki_request(action, **kwargs):
            if action == "deckNames":
                return ["German", "Spanish"]
            if action == "findCards":
                return [1, 2, 3]
            if action == "getReviewsOfCards":
                return {str(k): [{"ease": 2}] for k in kwargs.get("cards", [])}
            return None

        with patch("pb.vault.anki_client.anki_request", side_effect=fake_anki_request):
            sync_revlog()

        with get_connection() as conn:
            rows = conn.execute("SELECT deck FROM anki_revlog").fetchall()
            decks = {row[0] for row in rows}

        assert "German" in decks
        assert "Spanish" in decks

    def test_silently_skips_when_anki_offline(self):
        """sync_revlog silently skips when Anki is offline (no exception raised)."""
        from pb.vault.anki_client import sync_revlog

        with patch("pb.vault.anki_client.anki_request", return_value=None):
            # Must not raise any exception
            sync_revlog()


# ---------------------------------------------------------------------------
# get_pending_card_count
# ---------------------------------------------------------------------------


class TestGetPendingCardCount:
    def test_returns_count_of_pending_cards(self, temp_db):
        """get_pending_card_count returns count of status=pending from anki_cards."""
        from pb.vault.anki_client import insert_cards_to_db, get_pending_card_count

        cards = [
            {
                "id": f"card-{i}",
                "note_slug": "test-note",
                "front": f"Q{i}",
                "back": f"A{i}",
                "card_type": "auto",
                "status": "pending",
                "deck": "Test",
                "tags": "[]",
                "anki_model": "Basic",
                "created_at": "2026-05-04T10:00:00",
                "updated_at": "2026-05-04T10:00:00",
            }
            for i in range(3)
        ]

        insert_cards_to_db(cards)
        count = get_pending_card_count()
        assert count == 3

    def test_returns_zero_when_no_pending_cards(self, temp_db):
        """get_pending_card_count returns 0 when no pending cards exist."""
        from pb.vault.anki_client import get_pending_card_count

        count = get_pending_card_count()
        assert count == 0


# ---------------------------------------------------------------------------
# get_deck_review_stats
# ---------------------------------------------------------------------------


class TestGetDeckReviewStats:
    def test_returns_dict_with_deck_cards_reviews_keys(self):
        """get_deck_review_stats returns dict with deck, cards, reviews keys."""
        from pb.vault.anki_client import get_deck_review_stats

        def fake_anki_request(action, **kwargs):
            if action == "findCards":
                return [1, 2, 3, 4, 5]
            if action == "getReviewsOfCards":
                return {"1": [{"ease": 2}, {"ease": 3}], "2": [{"ease": 4}]}
            return None

        with patch("pb.vault.anki_client.anki_request", side_effect=fake_anki_request):
            stats = get_deck_review_stats("German")

        assert stats is not None
        assert "deck" in stats
        assert "cards" in stats
        assert "reviews" in stats
        assert stats["deck"] == "German"
        assert stats["cards"] == 5
        assert stats["reviews"] == 3  # 2 + 1 reviews

    def test_returns_none_when_anki_offline(self):
        """get_deck_review_stats returns None when Anki is offline."""
        from pb.vault.anki_client import get_deck_review_stats

        with patch("pb.vault.anki_client.anki_request", return_value=None):
            stats = get_deck_review_stats("German")

        assert stats is None
