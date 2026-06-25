# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault configuration, utilities, and intelligence service.

Existing exports (get_vault_path, scaffold_vault, etc.) preserved.
VaultService added via lazy loading for Phase 21+.
"""
from __future__ import annotations
from typing import TYPE_CHECKING

# -- Preserved existing exports (DO NOT REMOVE) --
from pb.vault.config import get_vault_path, VAULT_SCHEMA
from pb.vault.scaffold import scaffold_vault, ensure_vault_folder

if TYPE_CHECKING:
    from pb.vault.service import VaultService


def __getattr__(name: str):
    if name == "VaultService":
        from pb.vault.service import VaultService
        globals()[name] = VaultService
        return VaultService
    raise AttributeError(f"module 'pb.vault' has no attribute {name!r}")


__all__ = [
    "get_vault_path",
    "scaffold_vault",
    "ensure_vault_folder",
    "VAULT_SCHEMA",
    "VaultService",
]
