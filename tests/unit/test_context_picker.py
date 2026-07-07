# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Unit tests for the interactive context-file browser."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from pb.cli import context_picker


@pytest.fixture
def unreadable_dir(tmp_path: Path):
    locked = tmp_path / "locked"
    locked.mkdir()
    (locked / "hidden-from-listing.txt").write_text("x")
    locked.chmod(0o000)
    try:
        yield locked
    finally:
        locked.chmod(0o755)


def test_entries_for_lists_dirs_then_files(tmp_path: Path) -> None:
    (tmp_path / "b.txt").write_text("x")
    (tmp_path / "a").mkdir()
    entries, notice = context_picker._entries_for(tmp_path)
    assert [entry.kind for entry in entries] == ["parent", "dir", "file"]
    assert notice == ""


def test_entries_for_surfaces_permission_denied(unreadable_dir: Path) -> None:
    if os.geteuid() == 0:
        pytest.skip("chmod 000 does not block root")
    entries, notice = context_picker._entries_for(unreadable_dir)
    assert [entry.kind for entry in entries] == ["parent"]
    assert "denied" in notice.lower()
    assert "Privacy & Security" in notice


def test_entries_for_empty_dir_has_no_notice(tmp_path: Path) -> None:
    entries, notice = context_picker._entries_for(tmp_path)
    assert [entry.kind for entry in entries] == ["parent"]
    assert notice == ""


def test_render_shows_notice_line(tmp_path: Path) -> None:
    entries, _ = context_picker._entries_for(tmp_path)
    lines = context_picker._render_context_browser(
        entries,
        current=tmp_path,
        cursor=0,
        selected=set(),
        notice="Access to this folder was denied by macOS.",
    )
    assert any("denied by macOS" in line for line in lines)


def test_render_without_notice_adds_no_notice_line(tmp_path: Path) -> None:
    entries, _ = context_picker._entries_for(tmp_path)
    lines = context_picker._render_context_browser(
        entries,
        current=tmp_path,
        cursor=0,
        selected=set(),
        notice="",
    )
    assert not any("denied" in line for line in lines)
