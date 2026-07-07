# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault configuration and scaffold utilities."""
from __future__ import annotations

from pb.vault.config import get_vault_path, VAULT_SCHEMA
from pb.vault.scaffold import scaffold_vault, ensure_vault_folder


__all__ = [
    "get_vault_path",
    "scaffold_vault",
    "ensure_vault_folder",
    "VAULT_SCHEMA",
]
