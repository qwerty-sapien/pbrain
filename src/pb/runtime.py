# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared invocation runtime context for CLI and MCP entrypoints."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from tempfile import gettempdir

from pb.storage.config import Config, get_config, get_data_dir, get_log_dir, get_quarantine_path, get_vault_path
from pb.storage.config import get_active_vault_name
from pb.storage.database import DB_FILENAME


@dataclass
class RuntimeContext:
    """Resolved runtime state for one CLI or MCP invocation."""

    config: Config
    config_path: Path
    vault_name: str
    vault_path: Path
    data_dir: Path
    db_path: Path
    quarantine_path: Path
    yes: bool = False
    config_override: Optional[Path] = None


def build_runtime_context(
    *,
    config_path: Optional[Path] = None,
    vault: Optional[str] = None,
    yes: bool = False,
    force_reload: bool = True,
) -> RuntimeContext:
    """Load config and resolve the active vault/data directories."""
    config = get_config(config_path, vault=vault, force_reload=force_reload)
    resolved_config_path = config_path or Path(os.path.expanduser(str(getattr(config, "__config_path__", ""))))  # pragma: no cover
    # Keep using the public helper for the real path rather than the attribute above.
    from pb.storage.config import get_config_path, get_active_vault_name

    resolved_config_path = get_config_path(config_path)
    vault_name = get_active_vault_name(config, vault=vault)
    vault_path = get_vault_path(config, vault=vault)
    data_dir = get_data_dir(config, vault=vault)
    quarantine_path = get_quarantine_path(config, vault=vault)
    db_path = data_dir / DB_FILENAME
    return RuntimeContext(
        config=config,
        config_path=resolved_config_path,
        vault_name=vault_name,
        vault_path=vault_path,
        data_dir=data_dir,
        db_path=db_path,
        quarantine_path=quarantine_path,
        yes=yes,
        config_override=config_path,
    )


def runtime_from_config(
    config: Config,
    *,
    config_path: Optional[Path] = None,
    vault: Optional[str] = None,
    yes: bool = False,
) -> RuntimeContext:
    """Build a runtime context from an already-loaded config object."""
    if config_path is not None:
        resolved_config_path = config_path
    else:
        env_override = os.environ.get("PRODUCTIVEBRAIN_CONFIG_PATH")
        if env_override:
            resolved_config_path = Path(env_override).expanduser()
        else:
            from pb.storage.config import get_config_path

            resolved_config_path = get_config_path(config_path)
    vault_name = get_active_vault_name(config, vault=vault)
    vault_path = get_vault_path(config, vault=vault)
    data_dir = get_data_dir(config, vault=vault)
    quarantine_path = get_quarantine_path(config, vault=vault)
    db_path = data_dir / DB_FILENAME
    return RuntimeContext(
        config=config,
        config_path=resolved_config_path,
        vault_name=vault_name,
        vault_path=vault_path,
        data_dir=data_dir,
        db_path=db_path,
        quarantine_path=quarantine_path,
        yes=yes,
        config_override=config_path,
    )


def _session_scope_key() -> str:
    try:
        tty = os.ttyname(0)
    except OSError:
        tty = os.environ.get("TERM_SESSION_ID") or os.environ.get("TTY") or "default"
    slug = re.sub(r"[^a-zA-Z0-9_.-]+", "-", tty).strip("-")
    return slug or "default"


def get_session_flag_path(config: Optional[Config] = None, name: str = "auto-yes") -> Path:
    """Return the state-file path for ephemeral session-scoped flags."""
    try:
        state_dir = get_log_dir(config) / "session-flags"
        state_dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        try:
            state_dir = get_data_dir(config) / "session-flags"
            state_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            state_dir = Path(gettempdir()) / "productivebrain-session-flags"
            state_dir.mkdir(parents=True, exist_ok=True)
    return state_dir / f"{_session_scope_key()}-{name}.flag"


def get_session_auto_yes(config: Optional[Config] = None) -> bool:
    """Return whether session-scoped auto-yes is enabled for this terminal."""
    if os.environ.get("PRODUCTIVEBRAIN_AUTO_YES", "").strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return get_session_flag_path(config, "auto-yes").exists()


def set_session_auto_yes(enabled: bool, config: Optional[Config] = None) -> Path:
    """Set or clear session-scoped auto-yes for this terminal."""
    flag_path = get_session_flag_path(config, "auto-yes")
    if enabled:
        flag_path.write_text("on\n")
    else:
        flag_path.unlink(missing_ok=True)
    return flag_path
