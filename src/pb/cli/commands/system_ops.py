# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Safe maintenance operations for updating and resetting ProductiveBrain."""

from __future__ import annotations

import os
import shutil
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any

from pb.storage.database import DB_FILENAME, init_db
from pb.vault.scaffold import scaffold_vault


def repo_root() -> Path:
    """Return the repository root that contains the ProductiveBrain checkout."""
    current = Path(__file__).resolve()
    for candidate in (current.parent, *current.parents):
        if (candidate / ".git").exists() and (candidate / "pyproject.toml").exists():
            return candidate
    for candidate in (current.parent, *current.parents):
        if (candidate / "pyproject.toml").exists():
            return candidate
    return Path.cwd()


def inspect_update_state(root: Path | None = None) -> dict[str, Any]:
    """Return git-checkout update metadata without mutating the worktree."""
    root = root or repo_root()
    git_dir = root / ".git"
    if not git_dir.exists():
        return {
            "supported": False,
            "repo_root": str(root),
            "message": "This ProductiveBrain install is not a git checkout. Reinstall or update it through your package manager.",
        }

    dirty = _git_stdout(root, ["git", "status", "--short"]).strip().splitlines()
    current_commit = _git_stdout(root, ["git", "rev-parse", "HEAD"]).strip()
    return {
        "supported": True,
        "repo_root": str(root),
        "dirty": dirty,
        "current_commit": current_commit,
    }


def run_update(
    *,
    root: Path | None = None,
    check: bool = False,
    dryrun: bool = False,
) -> dict[str, Any]:
    """Perform the git-based update flow or return a dry-run/check preview."""
    root = root or repo_root()
    state = inspect_update_state(root)
    if not state.get("supported"):
        return state
    if state["dirty"] and not (check or dryrun):
        return {
            **state,
            "ok": False,
            "message": "Working tree is not clean. Commit or stash changes before running pb update.",
        }

    fetch = _git_run(root, ["git", "fetch", "--tags", "origin"], check=False)
    target_commit = _git_stdout(root, ["git", "rev-parse", "origin/HEAD"], check=False).strip()
    result = {
        **state,
        "ok": fetch.returncode == 0,
        "target_commit": target_commit,
        "fetched": fetch.returncode == 0,
        "fetch_stderr": fetch.stderr.strip(),
    }
    if check or dryrun:
        result["message"] = "Update check complete." if check else "Dry run complete."
        return result

    before_commit = state["current_commit"]
    pull = _git_run(root, ["git", "pull", "--ff-only"], check=False)
    after_commit = _git_stdout(root, ["git", "rev-parse", "HEAD"]).strip()
    lockfiles_changed = _git_stdout(
        root,
        ["git", "diff", "--name-only", before_commit, after_commit, "--", "uv.lock", "pyproject.toml", "requirements.txt"],
        check=False,
    ).strip().splitlines()
    result.update(
        {
            "ok": pull.returncode == 0,
            "before_commit": before_commit,
            "after_commit": after_commit,
            "lockfiles_changed": lockfiles_changed,
            "message": "Updated successfully." if pull.returncode == 0 else pull.stderr.strip() or "git pull failed.",
        }
    )
    return result


def inspect_reset_state(*, vault_path: Path, db_path: Path, repo_root_path: Path | None = None) -> dict[str, Any]:
    """Return reset safety metadata for the configured vault and SQLite paths."""
    repo_root_path = repo_root_path or repo_root()
    suspicious = _is_suspicious_reset_target(vault_path, repo_root_path=repo_root_path)
    file_count = _count_files(vault_path)
    return {
        "vault_path": str(vault_path),
        "db_path": str(db_path),
        "suspicious": suspicious,
        "file_count": file_count,
    }


def run_reset(
    *,
    vault_path: Path,
    db_path: Path,
    dryrun: bool = False,
    backup: bool = False,
    repo_root_path: Path | None = None,
) -> dict[str, Any]:
    """Delete vault contents and reset the SQLite state without following symlinks."""
    repo_root_path = repo_root_path or repo_root()
    state = inspect_reset_state(vault_path=vault_path, db_path=db_path, repo_root_path=repo_root_path)
    if state["suspicious"]:
        return {
            **state,
            "ok": False,
            "message": "Refusing to reset a suspicious vault path.",
        }
    if dryrun:
        return {
            **state,
            "ok": True,
            "message": "Dry run only; no files were deleted.",
        }

    backup_path = ""
    if backup and vault_path.exists():
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        backup_dir = db_path.parent / "backups"
        backup_dir.mkdir(parents=True, exist_ok=True)
        archive_base = backup_dir / f"pb-reset-{timestamp}"
        backup_path = shutil.make_archive(str(archive_base), "zip", root_dir=str(vault_path))

    _clear_directory_contents(vault_path)
    _remove_database_files(db_path)
    vault_path.mkdir(parents=True, exist_ok=True)
    scaffold_vault(vault_path)
    init_db(db_path)
    return {
        **state,
        "ok": True,
        "backup_path": backup_path,
        "message": "Vault contents and SQLite state were reset.",
    }


def _git_stdout(root: Path, argv: list[str], *, check: bool = True) -> str:
    return _git_run(root, argv, check=check).stdout


def _git_run(root: Path, argv: list[str], *, check: bool) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, cwd=root, text=True, capture_output=True, check=check)


def _is_suspicious_reset_target(vault_path: Path, *, repo_root_path: Path) -> bool:
    resolved = vault_path.expanduser().resolve()
    home = Path.home().resolve()
    if str(resolved).strip() in {"", "/"}:
        return True
    if resolved == home:
        return True
    if resolved == repo_root_path.resolve():
        return True
    return False


def _count_files(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for dirpath, dirs, files in os.walk(root, followlinks=False):
        total += len(files)
        base = Path(dirpath)
        total += sum(1 for item in dirs if (base / item).is_symlink())
    return total


def _clear_directory_contents(root: Path) -> None:
    if not root.exists():
        return
    for child in root.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink(missing_ok=True)
            continue
        shutil.rmtree(child)


def _remove_database_files(db_path: Path) -> None:
    for suffix in ("", "-wal", "-shm"):
        target = Path(f"{db_path}{suffix}") if suffix else db_path
        if target.exists():
            target.unlink()


def default_db_path_from_runtime(runtime) -> Path:
    """Derive the active SQLite path from the current runtime context."""
    explicit = getattr(runtime, "db_path", None)
    if explicit:
        return Path(explicit)
    data_dir = Path(getattr(runtime, "data_dir", Path.cwd()))
    return data_dir / DB_FILENAME
