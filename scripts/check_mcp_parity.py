#!/usr/bin/env python3
# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Check MCP parity against the visible pbrain CLI surface."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pb.cli.main import app  # noqa: E402
from pb.mcp.server import mcp  # noqa: E402
from pb.mcp.tools import pb_tools, pbrain, schema, vault  # noqa: E402,F401
from pb.mcp.tools.pb_tools import _is_command_allowed, _is_read_only_command  # noqa: E402
from pb.mcp.tools.pbrain import tool_catalog  # noqa: E402


def _visible_cli_paths(typer_app, prefix: tuple[str, ...] = ()) -> list[tuple[str, ...]]:
    paths: list[tuple[str, ...]] = []
    for group in getattr(typer_app, "registered_groups", []) or []:
        if getattr(group, "hidden", False):
            continue
        group_path = (*prefix, group.name)
        paths.append(group_path)
        paths.extend(_visible_cli_paths(group.typer_instance, group_path))
    for command in getattr(typer_app, "registered_commands", []) or []:
        if getattr(command, "hidden", False):
            continue
        name = command.name or command.callback.__name__.replace("_", "-")
        paths.append((*prefix, name))
    return sorted(set(paths))


def _check_read_only_gating() -> list[str]:
    expected_read_only = [
        "context",
        "vault list",
        "context inspect notes.md",
        "next",
        "anki list",
        "notes organise --json",
        "mcp pending",
    ]
    expected_writes = [
        "vault add scratch /tmp/scratch",
        "context add notes.md",
        "next --schedule 10",
        "anki list --suggested all",
        "notes organise --yes",
        "mcp confirm abc123",
    ]
    errors = [
        f"expected read-only: {command}"
        for command in expected_read_only
        if not _is_read_only_command(command)
    ]
    errors.extend(
        f"expected write-gated: {command}"
        for command in expected_writes
        if _is_read_only_command(command)
    )
    return errors


def main() -> int:
    visible_missing = [
        " ".join(path)
        for path in _visible_cli_paths(app)
        if not _is_command_allowed(" ".join(path))
    ]
    registered_tools = {tool.name for tool in asyncio.run(mcp.list_tools())}
    cataloged_tools = {item["name"] for item in tool_catalog()["tools"]}
    missing_catalog = sorted(registered_tools - cataloged_tools)
    read_only_errors = _check_read_only_gating()

    errors: list[str] = []
    errors.extend(f"pb_command does not allow visible CLI command: {item}" for item in visible_missing)
    errors.extend(f"tool_catalog is missing registered MCP tool: {item}" for item in missing_catalog)
    errors.extend(read_only_errors)
    if errors:
        for error in errors:
            print(error, file=sys.stderr)
        return 1

    print(
        "MCP parity OK: "
        f"{len(_visible_cli_paths(app))} visible CLI paths, "
        f"{len(registered_tools)} registered tools."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
