# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Runtime context for ProductiveBrain MCP server invocations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from pb.runtime import RuntimeContext, build_runtime_context


@dataclass
class MCPServerContext:
    """Process-wide MCP options."""

    vault: Optional[str] = None
    config_path: Optional[Path] = None
    allow_writes: bool = False


_context = MCPServerContext()


def configure_mcp_context(
    *,
    vault: Optional[str] = None,
    config_path: Optional[Path] = None,
    allow_writes: bool = False,
) -> MCPServerContext:
    """Store process-wide MCP options."""
    global _context
    _context = MCPServerContext(vault=vault, config_path=config_path, allow_writes=allow_writes)
    return _context


def get_mcp_context() -> MCPServerContext:
    """Return current MCP process options."""
    return _context


def get_runtime_context() -> RuntimeContext:
    """Build a fresh runtime context for the current MCP request."""
    return build_runtime_context(
        config_path=_context.config_path,
        vault=_context.vault,
        force_reload=True,
    )
