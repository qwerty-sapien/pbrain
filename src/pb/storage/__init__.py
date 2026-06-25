# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Storage layer public API with lazy imports to avoid package init cycles."""

from __future__ import annotations

from importlib import import_module

_EXPORTS = {
    "load_config": ("pb.storage.config", "load_config"),
    "get_config": ("pb.storage.config", "get_config"),
    "get_config_path": ("pb.storage.config", "get_config_path"),
    "init_db": ("pb.storage.database", "init_db"),
    "get_connection": ("pb.storage.database", "get_connection"),
    "Repository": ("pb.storage.repository", "Repository"),
}


def __getattr__(name: str):
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module attribute behavior
        raise AttributeError(name) from exc
    value = getattr(import_module(module_name), attr_name)
    globals()[name] = value
    return value


__all__ = list(_EXPORTS)
