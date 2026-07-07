"""Tests for pb.core.dryrun — temp sandbox and stale cleanup."""

from __future__ import annotations

import os
import time
from pathlib import Path

import pytest


@pytest.fixture
def fake_data_dir(tmp_path):
    """Create a fake data_dir with a DB file."""
    data = tmp_path / "data"
    data.mkdir()
    db = data / "productivebrain.db"
    db.write_text("fake-db-content")
    return data


def test_create_dryrun_sandbox_returns_paths(fake_data_dir):
    from pb.core.dryrun import create_dryrun_sandbox
    sandbox = create_dryrun_sandbox(fake_data_dir)
    assert sandbox.root.exists()
    assert sandbox.vault_path.exists()
    assert sandbox.data_dir.exists()
    assert sandbox.db_path.exists()
    assert sandbox.db_path.read_text() == "fake-db-content"


def test_create_dryrun_sandbox_is_under_tmpdir(fake_data_dir):
    from pb.core.dryrun import create_dryrun_sandbox
    import tempfile
    sandbox = create_dryrun_sandbox(fake_data_dir)
    assert str(sandbox.root).startswith(tempfile.gettempdir())


def test_cleanup_stale_dryrun_dirs(tmp_path, monkeypatch):
    from pb.core.dryrun import cleanup_stale_dryrun_dirs, DRYRUN_DIR_PREFIX
    stale = tmp_path / f"{DRYRUN_DIR_PREFIX}old"
    stale.mkdir()
    (stale / "vault").mkdir()
    os.utime(stale, (0, 0))

    fresh = tmp_path / f"{DRYRUN_DIR_PREFIX}fresh"
    fresh.mkdir()
    (fresh / "vault").mkdir()

    monkeypatch.setattr("pb.core.dryrun._get_tmpdir", lambda: tmp_path)
    cleanup_stale_dryrun_dirs()

    assert not stale.exists()
    assert fresh.exists()


def test_find_collect_skips_hidden_dirs(tmp_path):
    """collect_files skips dotfiles/dotdirs in the vault."""
    from pb.cli.commands.find import collect_files

    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "tasks").mkdir()
    (vault / "tasks" / "visible.md").write_text("# Visible")
    hidden = vault / ".obsidian"
    hidden.mkdir()
    (hidden / "workspace.json").write_text("{}")

    data = tmp_path / "data"
    data.mkdir()

    files = collect_files(vault_path=vault, data_dir=data)
    names = {f.name for f in files}
    assert "visible.md" in names
    assert "workspace.json" not in names


def test_delete_refuses_protected_files(tmp_path):
    """delete_files skips pb.db and config.toml."""
    from pb.cli.commands.find import delete_files

    db = tmp_path / "productivebrain.db"
    db.write_text("data")
    note = tmp_path / "note.md"
    note.write_text("content")
    config = tmp_path / "config.toml"
    config.write_text("[x]")

    deleted, skipped = delete_files([db, note, config])
    assert deleted == 1
    assert skipped == 2
    assert not note.exists()
    assert db.exists()
    assert config.exists()
