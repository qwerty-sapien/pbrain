"""Tests for Phase 26 DB schema migration — anki_cards new columns and generation_run_log table.

RED phase: tests written before implementation.
"""

from __future__ import annotations

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
    yield db_path
    db_module._db_path = None


def test_anki_cards_phase26_idempotent(tmp_path):
    """Call init_db() twice on a temp DB; assert all 4 Phase 26 columns present after both calls."""
    db_path = tmp_path / "idempotent.db"
    set_db_path(db_path)

    init_db(db_path)  # first call
    init_db(db_path)  # second call — must not raise

    conn = sqlite3.connect(str(db_path))
    cols = {row[1] for row in conn.execute("PRAGMA table_info(anki_cards)").fetchall()}
    conn.close()

    assert "domain" in cols, "domain column missing after Phase 26 migration"
    assert "exported_at" in cols, "exported_at column missing after Phase 26 migration"
    assert "anki_note_id" in cols, "anki_note_id column missing after Phase 26 migration"
    assert "run_id" in cols, "run_id column missing after Phase 26 migration"

    db_module._db_path = None


def test_generation_run_log_created(tmp_path):
    """Call init_db(); assert generation_run_log table exists in sqlite_master."""
    db_path = tmp_path / "run_log.db"
    set_db_path(db_path)

    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='generation_run_log'"
    ).fetchone()
    conn.close()

    assert row is not None, "generation_run_log table was not created by init_db()"

    db_module._db_path = None


def test_generation_run_log_schema(tmp_path):
    """Insert a row with all 6 columns; assert SELECT returns it correctly."""
    db_path = tmp_path / "schema.db"
    set_db_path(db_path)

    init_db(db_path)

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute(
        "INSERT INTO generation_run_log "
        "(run_id, note_slug, term, card_count, source, created_at) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        ("run-abc123", "my-note", "some-term", 5, "auto", "2026-05-08T12:00:00"),
    )
    conn.commit()

    row = conn.execute(
        "SELECT run_id, note_slug, term, card_count, source, created_at "
        "FROM generation_run_log WHERE run_id = 'run-abc123'"
    ).fetchone()
    conn.close()

    assert row is not None, "Inserted row not found"
    assert row["run_id"] == "run-abc123"
    assert row["note_slug"] == "my-note"
    assert row["term"] == "some-term"
    assert row["card_count"] == 5
    assert row["source"] == "auto"
    assert row["created_at"] == "2026-05-08T12:00:00"

    db_module._db_path = None
