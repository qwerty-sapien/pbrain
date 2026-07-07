"""Tests for Phase 26 anki_client.py extensions — new columns + _insert_run_log_entry.

RED phase: tests written before implementation.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

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


def _make_card(card_id: str = "test-card-01", **overrides) -> dict:
    """Build a minimal valid card dict."""
    import datetime
    now = datetime.datetime.now().isoformat()
    base = {
        "id": card_id,
        "note_slug": "test-note",
        "front": "What is X?",
        "back": "X is Y.",
        "card_type": "auto",
        "status": "pending",
        "deck": "German::Vocab",
        "tags": json.dumps(["deutsch", "auto"]),
        "anki_model": "Basic",
        "created_at": now,
        "updated_at": now,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# insert_cards_with_new_columns
# ---------------------------------------------------------------------------


def test_insert_cards_with_new_columns(tmp_path):
    """Insert card dict with domain='deutsch', run_id='abc12'; assert both stored in DB."""
    from pb.vault.anki_client import insert_cards_to_db
    from pb.storage.database import get_connection

    card = _make_card("new-cols-card", domain="deutsch", run_id="abc12")
    count = insert_cards_to_db([card])
    assert count == 1

    with get_connection() as conn:
        row = conn.execute(
            "SELECT domain, run_id FROM anki_cards WHERE id = ?", ("new-cols-card",)
        ).fetchone()

    assert row is not None
    assert row["domain"] == "deutsch"
    assert row["run_id"] == "abc12"


def test_insert_cards_without_new_columns(tmp_path):
    """Insert legacy card dict without domain or run_id; no exception; columns are NULL."""
    from pb.vault.anki_client import insert_cards_to_db
    from pb.storage.database import get_connection

    card = _make_card("legacy-card")  # no domain or run_id
    count = insert_cards_to_db([card])
    assert count == 1

    with get_connection() as conn:
        row = conn.execute(
            "SELECT domain, run_id FROM anki_cards WHERE id = ?", ("legacy-card",)
        ).fetchone()

    assert row is not None
    assert row["domain"] is None
    assert row["run_id"] is None


# ---------------------------------------------------------------------------
# _insert_run_log_entry
# ---------------------------------------------------------------------------


def test_insert_run_log_entry(tmp_path):
    """Call _insert_run_log_entry; query generation_run_log; assert run_id row exists."""
    from pb.vault.anki_client import _insert_run_log_entry
    from pb.storage.database import get_connection

    _insert_run_log_entry("r1", "my-note", None, 3, "auto")

    with get_connection() as conn:
        row = conn.execute(
            "SELECT run_id, note_slug, card_count, source FROM generation_run_log WHERE run_id = ?",
            ("r1",),
        ).fetchone()

    assert row is not None
    assert row["run_id"] == "r1"
    assert row["note_slug"] == "my-note"
    assert row["card_count"] == 3
    assert row["source"] == "auto"
