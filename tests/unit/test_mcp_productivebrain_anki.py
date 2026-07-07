# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pb.storage.database as db_module
from pb.mcp.tools.productivebrain import (
    anki_candidate_list,
    anki_candidate_status_counts,
    anki_candidate_update,
    anki_export_status,
)
from pb.storage.database import init_db, set_db_path
from pb.vault.anki_client import insert_cards_to_db


def _card(card_id: str, *, status: str, deck: str = "German::Concepts") -> dict:
    return {
        "id": card_id,
        "note_slug": f"note-{card_id}",
        "front": f"front {card_id}",
        "back": f"back {card_id}",
        "card_type": "Basic",
        "status": status,
        "deck": deck,
        "tags": "[]",
        "anki_model": "Basic",
        "domain": "deutsch",
        "created_at": "2026-06-22T00:00:00",
        "updated_at": "2026-06-22T00:00:00",
    }


def setup_function() -> None:
    db_module._db_path = None


def teardown_function() -> None:
    db_module._db_path = None


def test_mcp_anki_candidate_listing_and_counts(tmp_path):
    db_path = tmp_path / "pb_test.db"
    set_db_path(db_path)
    init_db(db_path)
    insert_cards_to_db(
        [
            _card("card-1", status="suggested"),
            _card("card-2", status="accepted"),
            _card("card-3", status="edited"),
        ]
    )

    listed = anki_candidate_list(status="suggested")
    counts = anki_candidate_status_counts()
    export_status = anki_export_status()

    assert listed["count"] == 1
    assert listed["items"][0]["id"] == "card-1"
    assert counts["counts"]["suggested"] == 1
    assert counts["export_ready"] == 2
    assert export_status["count"] == 2


def test_mcp_anki_candidate_update_mutates_status_when_writes_allowed(tmp_path):
    db_path = tmp_path / "pb_test.db"
    set_db_path(db_path)
    init_db(db_path)
    insert_cards_to_db([_card("card-1", status="suggested")])

    with patch("pb.mcp.tools.productivebrain.get_mcp_context", return_value=SimpleNamespace(allow_writes=True)):
        with patch("pb.mcp.tools.productivebrain._bypassing", return_value=True):
            result = anki_candidate_update(["card-1"], "accepted")

    assert result["updated"] == ["card-1"]
    assert anki_candidate_list(status="accepted")["items"][0]["id"] == "card-1"
