from __future__ import annotations

from pb.cli.commands.init import init_command
from pb.core.anki_bootstrap import ANKI_AUTO_OPEN_PREF
from pb.storage import config as config_module
from pb.storage.config import get_config_path, load_config


def test_init_noninteractive_saves_explicit_anki_auto_open_consent(tmp_path, monkeypatch):
    home = tmp_path / "home"
    config_home = tmp_path / "config"
    vault = tmp_path / "vault"
    home.mkdir()
    vault.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(config_home))
    config_module._config = None

    init_command(
        non_interactive=True,
        vault_path=str(vault),
        yes=True,
        allow_anki_auto_open=True,
    )

    cfg = load_config(get_config_path(), force_reload=True)
    assert cfg.preferences[ANKI_AUTO_OPEN_PREF] is True
