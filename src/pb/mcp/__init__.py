# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Lazy MCP exports for ProductiveBrain agent integrations."""

from importlib import import_module
from typing import Any

__all__ = ["mcp"]

_EXPORTS = {
    "mcp": ("pb.mcp.server", "mcp"),
}


def __getattr__(name: str) -> Any:
    try:
        module_name, attr_name = _EXPORTS[name]
    except KeyError as exc:  # pragma: no cover - standard module attribute behavior
        raise AttributeError(f"module 'pb.mcp' has no attribute {name!r}") from exc
    module = import_module(module_name)
    return getattr(module, attr_name)
