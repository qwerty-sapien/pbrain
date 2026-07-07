"""Dryrun sandbox — redirects all pb writes to a temp directory."""

from __future__ import annotations

import os
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

from pb.storage.database import DB_FILENAME

DRYRUN_DIR_PREFIX = "pb-dryrun-"


def _get_tmpdir() -> Path:
    return Path(tempfile.gettempdir())


@dataclass
class DryrunSandbox:
    root: Path
    vault_path: Path
    data_dir: Path
    db_path: Path


def create_dryrun_sandbox(real_data_dir: Path) -> DryrunSandbox:
    root = Path(tempfile.mkdtemp(prefix=DRYRUN_DIR_PREFIX))
    vault_path = root / "vault"
    vault_path.mkdir()
    data_dir = root / "data"
    data_dir.mkdir()

    real_db = real_data_dir / DB_FILENAME
    sandbox_db = data_dir / DB_FILENAME
    if real_db.exists():
        shutil.copy2(real_db, sandbox_db)
    else:
        sandbox_db.touch()

    return DryrunSandbox(root=root, vault_path=vault_path, data_dir=data_dir, db_path=sandbox_db)


_STALE_THRESHOLD_SECONDS = 3600  # 1 hour


def cleanup_stale_dryrun_dirs() -> int:
    """Remove pb-dryrun-* dirs whose mtime is older than the stale threshold."""
    import time

    tmpdir = _get_tmpdir()
    cutoff = time.time() - _STALE_THRESHOLD_SECONDS
    removed = 0
    for entry in tmpdir.iterdir():
        if not entry.name.startswith(DRYRUN_DIR_PREFIX):
            continue
        if not entry.is_dir():
            continue
        try:
            if entry.stat().st_mtime < cutoff:
                shutil.rmtree(entry)
                removed += 1
        except OSError:
            pass
    return removed
