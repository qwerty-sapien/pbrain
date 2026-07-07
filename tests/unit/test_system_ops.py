# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

import subprocess
from pathlib import Path
from unittest.mock import patch

from pb.cli.commands import system_ops


def _completed(*, stdout: str = "", stderr: str = "", returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args=["git"], returncode=returncode, stdout=stdout, stderr=stderr)


def test_run_update_check_and_dryrun_use_git_state_without_pulling(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()

    calls: list[list[str]] = []

    def fake_run(argv, cwd, text, capture_output, check):
        calls.append(list(argv))
        if argv[:3] == ["git", "status", "--short"]:
            return _completed(stdout="")
        if argv[:3] == ["git", "rev-parse", "HEAD"]:
            return _completed(stdout="abc123\n")
        if argv[:4] == ["git", "fetch", "--tags", "origin"]:
            return _completed(stdout="")
        if argv[:3] == ["git", "rev-parse", "origin/HEAD"]:
            return _completed(stdout="def456\n")
        raise AssertionError(f"Unexpected git call: {argv}")

    with patch("pb.cli.commands.system_ops.subprocess.run", side_effect=fake_run):
        checked = system_ops.run_update(root=repo, check=True)
        dryrun = system_ops.run_update(root=repo, dryrun=True)

    assert checked["current_commit"] == "abc123"
    assert checked["target_commit"] == "def456"
    assert dryrun["ok"] is True
    assert ["git", "pull", "--ff-only"] not in calls


def test_reset_dryrun_and_safety_refusal(tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    (vault / "note.md").write_text("# Note", encoding="utf-8")
    db_path = tmp_path / "productivebrain.db"
    db_path.write_text("sqlite", encoding="utf-8")

    dryrun = system_ops.run_reset(vault_path=vault, db_path=db_path, dryrun=True, repo_root_path=tmp_path / "repo")
    assert dryrun["ok"] is True
    assert vault.joinpath("note.md").exists()
    assert db_path.exists()

    refused = system_ops.run_reset(vault_path=Path.home(), db_path=db_path, dryrun=True, repo_root_path=tmp_path / "repo")
    assert refused["ok"] is False
    assert "suspicious" in refused["message"].lower()
