# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""FastMCP server for ProductiveBrain integrations.

Entry points:
- productivebrain-mcp
- pb-mcp

IMPORTANT: Never log to stdout; stdio transport uses stdout for JSON-RPC only.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from mcp.server.fastmcp import FastMCP

from pb.mcp.context import configure_mcp_context


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    stream=sys.stderr,
)

logger = logging.getLogger("productivebrain-mcp")
mcp = FastMCP("productivebrain-mcp")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="productivebrain-mcp", add_help=True)
    parser.add_argument("--vault", default=None, help="Vault profile name to serve")
    parser.add_argument("--config", default=None, help="Explicit ProductiveBrain config.toml path")
    parser.add_argument("--allow-writes", action="store_true", help="Enable write-capable MCP tools")
    return parser.parse_args(argv)


def main() -> None:
    """Start the MCP server with stdio transport."""
    args = _parse_args()
    if args.config:
        os.environ["PRODUCTIVEBRAIN_CONFIG_PATH"] = args.config

    configure_mcp_context(
        vault=args.vault,
        config_path=Path(args.config).expanduser() if args.config else None,
        allow_writes=bool(args.allow_writes),
    )

    try:
        from pb.mcp.tools import vault  # noqa: F401
        from pb.mcp.tools import schema  # noqa: F401
        from pb.mcp.tools import pb_tools  # noqa: F401
        from pb.mcp.tools import productivebrain  # noqa: F401
    except ImportError as exc:
        logger.warning("Some MCP tools were not available: %s", exc)

    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
