"""Tests for pb.cli.commands.find — file discovery and display."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

import pytest


@pytest.fixture
def mock_tree(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    tasks = vault / "tasks"
    tasks.mkdir()
    (tasks / "2026-06-05-study-rust.md").write_text("# Study Rust")
    (tasks / "2026-06-02-old-task.md").write_text("# Old task")
    thoughts = vault / "thoughts"
    thoughts.mkdir()
    (thoughts / "idea.md").write_text("# Idea")

    data = tmp_path / "data"
    data.mkdir()
    (data / "pb.db").write_bytes(b"\x00" * 1024)

    log = tmp_path / "log"
    log.mkdir()
    (log / "error.log").write_text("some error")

    config = tmp_path / "config.toml"
    config.write_text("[general]\nname = 'test'\n")

    return {"vault": vault, "data": data, "log": log, "config": config}


def test_collect_all_files(mock_tree):
    from pb.cli.commands.find import collect_files
    files = collect_files(
        vault_path=mock_tree["vault"], data_dir=mock_tree["data"],
        log_dir=mock_tree["log"], config_path=mock_tree["config"],
    )
    names = {f.name for f in files}
    assert "2026-06-05-study-rust.md" in names
    assert "pb.db" in names
    assert "error.log" in names
    assert "config.toml" in names


def test_filter_by_days(mock_tree):
    from pb.cli.commands.find import collect_files, filter_by_days
    old = mock_tree["vault"] / "tasks" / "2026-06-02-old-task.md"
    old_time = (datetime.now() - timedelta(days=10)).timestamp()
    os.utime(old, (old_time, old_time))

    files = collect_files(
        vault_path=mock_tree["vault"], data_dir=mock_tree["data"],
        log_dir=mock_tree["log"], config_path=mock_tree["config"],
    )
    recent = filter_by_days(files, 3)
    names = {f.name for f in recent}
    assert "2026-06-02-old-task.md" not in names
    assert "2026-06-05-study-rust.md" in names


def test_filter_by_since_date(mock_tree):
    from pb.cli.commands.find import collect_files, filter_by_since
    old = mock_tree["vault"] / "tasks" / "2026-06-02-old-task.md"
    old_time = datetime(2026, 5, 1).timestamp()
    os.utime(old, (old_time, old_time))

    files = collect_files(
        vault_path=mock_tree["vault"], data_dir=mock_tree["data"],
        log_dir=mock_tree["log"], config_path=mock_tree["config"],
    )
    since = datetime(2026, 6, 1)
    result = filter_by_since(files, since)
    names = {f.name for f in result}
    assert "2026-06-02-old-task.md" not in names


def test_format_size():
    from pb.cli.commands.find import format_size
    assert format_size(0) == "0B"
    assert format_size(512) == "512B"
    assert format_size(1024) == "1.0K"
    assert format_size(1024 * 1024 * 2.5) == "2.5M"


def test_parse_query_days():
    from pb.cli.commands.find import parse_query
    q = parse_query("7")
    assert q.mode == "days"
    assert q.value == 7


def test_parse_query_date():
    from pb.cli.commands.find import parse_query
    q = parse_query("01-06-26")
    assert q.mode == "since"
    assert q.value.year == 2026
    assert q.value.month == 6
    assert q.value.day == 1


def test_parse_query_string():
    from pb.cli.commands.find import parse_query
    q = parse_query("german")
    assert q.mode == "string"
    assert q.value == "german"


def test_parse_query_none():
    from pb.cli.commands.find import parse_query
    q = parse_query(None)
    assert q.mode == "interactive"
