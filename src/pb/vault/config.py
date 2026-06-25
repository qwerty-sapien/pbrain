# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault configuration and path resolution.

Reuses vault_path from ProductiveBrain config as the single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from pb.storage.config import Config

# Learning-first default vault schema for v0.2.
# Used by scaffold_vault() for idempotent creation.
VAULT_SCHEMA: list[str] = [
    "00-inbox/captures",
    "direction/aspirations",
    "direction/goals",
    "direction/preferences",
    "knowledge/concepts",
    "knowledge/books",
    "knowledge/research",
    "knowledge/general",
    "30-recall",
    "projects",
    "logs/daily",
    "logs/task-memory",
    "logs/weekly",
    "logs/sessions",
]


def get_vault_path(config: Optional[Config] = None) -> Path:
    """Get the Obsidian vault path from ProductiveBrain config.

    Delegates to storage.config.get_vault_path per D-09.

    Args:
        config: Optional Config object. If None, loads from disk.

    Returns:
        Path to the vault directory.

    Raises:
        FileNotFoundError: If config doesn't exist or vault path is invalid.
    """
    # Import here to avoid circular dependency if needed
    from pb.storage.config import get_vault_path as storage_get_vault_path
    return storage_get_vault_path(config)
