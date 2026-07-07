# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from typer.testing import CliRunner

import pb.storage.config as config_module
from pb.cli.commands.vault import app as vault_app
from pb.storage.config import Config, GeneralConfig, VaultProfileConfig, get_config, save_config


def setup_function() -> None:
    config_module._config = None


def teardown_function() -> None:
    config_module._config = None


def _obj(tmp_path: Path) -> tuple[dict[str, object], Path]:
    vault = tmp_path / "main"
    data = tmp_path / "data" / "main"
    vault.mkdir(parents=True)
    data.mkdir(parents=True)
    config_path = tmp_path / "config.toml"
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path=str(vault)),
        vaults={"main": VaultProfileConfig(path=str(vault), data_dir=str(data))},
    )
    save_config(config, path=config_path)
    return (
        {
            "runtime": SimpleNamespace(config_path=config_path),
            "config": get_config(config_path, force_reload=True),
            "yes": True,
        },
        config_path,
    )


def test_vault_add_without_path_creates_sibling_profile(tmp_path: Path) -> None:
    obj, config_path = _obj(tmp_path)

    result = CliRunner().invoke(vault_app, ["add", "scratch"], obj=obj)

    assert result.exit_code == 0, result.output
    config = get_config(config_path, force_reload=True)
    assert Path(config.vaults["scratch"].path) == tmp_path / "scratch"
    assert (tmp_path / "scratch").is_dir()


def test_vault_rename_and_remove_report_user_errors_without_traceback(tmp_path: Path) -> None:
    obj, config_path = _obj(tmp_path)
    runner = CliRunner()

    added = runner.invoke(vault_app, ["add", "scratch"], obj=obj)
    assert added.exit_code == 0, added.output

    renamed = runner.invoke(vault_app, ["rename", "scratch", "scratch-renamed"], obj=obj)
    assert renamed.exit_code == 0, renamed.output
    assert "Renamed vault profile" in renamed.stdout

    missing = runner.invoke(vault_app, ["remove", "missing", "--yes"], obj=obj)
    assert missing.exit_code != 0
    assert "Unknown vault profile" in missing.output
    assert "Traceback" not in missing.output

    removed = runner.invoke(vault_app, ["remove", "scratch-renamed", "--yes"], obj=obj)
    assert removed.exit_code == 0, removed.output
    assert "Removed vault profile" in removed.stdout
    assert "scratch-renamed" not in get_config(config_path, force_reload=True).vaults
