# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from typer.testing import CliRunner

import pb.storage.config as config_module
import pb.storage.database as db_module
from pb.cli.commands.anki import app as anki_app
from pb.storage.config import Config, GeneralConfig, VaultProfileConfig
from pb.storage.database import init_db, set_db_path
from pb.vault.anki_client import get_card_by_id, insert_cards_to_db


def setup_function() -> None:
    db_module._db_path = None
    config_module._config = None


def teardown_function() -> None:
    db_module._db_path = None
    config_module._config = None


def _card(card_id: str, *, status: str, deck: str = "PB::Concepts") -> dict:
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
        "domain": "learning",
        "created_at": "2026-06-22T00:00:00",
        "updated_at": "2026-06-22T00:00:00",
    }


def _obj(tmp_path: Path) -> dict[str, object]:
    vault = tmp_path / "vault"
    data = tmp_path / "data"
    vault.mkdir()
    data.mkdir()
    db_path = data / "productivebrain.db"
    set_db_path(db_path)
    init_db(db_path)
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path=str(vault)),
        vaults={"main": VaultProfileConfig(path=str(vault), data_dir=str(data))},
    )
    config_module._config = config
    return {
        "runtime": SimpleNamespace(vault_path=vault, data_dir=data, db_path=db_path, config=config),
        "config": config,
        "factory": {"anki_service": lambda: SimpleNamespace()},
        "yes": True,
    }


def test_accept_and_reject_default_to_first_suggested_card(tmp_path: Path) -> None:
    _obj(tmp_path)
    insert_cards_to_db([
        _card("suggested-1", status="suggested"),
        _card("suggested-2", status="suggested"),
    ])

    runner = CliRunner()

    accepted = runner.invoke(anki_app, ["accept"])
    assert accepted.exit_code == 0, accepted.output
    assert get_card_by_id("suggested-1")["status"] == "accepted"

    rejected = runner.invoke(anki_app, ["reject"])
    assert rejected.exit_code == 0, rejected.output
    assert get_card_by_id("suggested-2")["status"] == "rejected"


def test_deck_export_without_deck_exports_local_cards_to_requested_apkg(tmp_path: Path) -> None:
    obj = _obj(tmp_path)
    insert_cards_to_db([_card("accepted-1", status="accepted")])
    generated = tmp_path / "generated.apkg"
    generated.write_bytes(b"package")
    requested = tmp_path / "requested.apkg"

    runner = CliRunner()
    with patch("pb.vault.anki_client.export_cards_to_apkg", return_value=(True, generated, "ok")):
        result = runner.invoke(anki_app, ["deck-export", "--output", str(requested)], obj=obj)

    assert result.exit_code == 0, result.output
    assert requested.read_bytes() == b"package"
    assert "Exported 1 PB cards" in result.stdout
