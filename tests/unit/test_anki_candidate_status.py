# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from typer.testing import CliRunner

import pb.storage.database as db_module
from pb.cli.commands.anki import app as anki_app
from pb.storage.database import init_db, set_db_path
from pb.vault.anki_client import (
    get_card_by_id,
    get_card_status_counts,
    insert_cards_to_db,
)


def _card(card_id: str, *, status: str, deck: str = "German") -> dict:
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


def test_status_count_helpers_and_pending_cli(tmp_path):
    db_path = tmp_path / "pb_test.db"
    set_db_path(db_path)
    init_db(db_path)
    insert_cards_to_db(
        [
            _card("card-1", status="suggested"),
            _card("card-2", status="accepted"),
            _card("card-3", status="rejected"),
        ]
    )

    counts = get_card_status_counts()
    assert counts["suggested"] == 1
    assert counts["accepted"] == 1
    assert counts["rejected"] == 1
    assert get_card_by_id("card-2")["status"] == "accepted"

    result = CliRunner().invoke(anki_app, ["pending"])

    assert result.exit_code == 0
    assert "Suggested: 1" in result.stdout
    assert "Export ready: 1" in result.stdout


def test_accept_and_reject_commands_update_stored_status(tmp_path):
    db_path = tmp_path / "pb_test.db"
    set_db_path(db_path)
    init_db(db_path)
    insert_cards_to_db([_card("card-1", status="suggested")])

    runner = CliRunner()
    accept = runner.invoke(anki_app, ["accept", "card-1"])
    assert accept.exit_code == 0
    assert get_card_by_id("card-1")["status"] == "accepted"

    reject = runner.invoke(anki_app, ["reject", "card-1"])
    assert reject.exit_code == 0
    assert get_card_by_id("card-1")["status"] == "rejected"
