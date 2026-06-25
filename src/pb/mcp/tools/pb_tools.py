# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Advanced CLI escape-hatch MCP tools."""

from __future__ import annotations

import shlex
import shutil
import subprocess
import sys

from pb.mcp.context import get_mcp_context, get_runtime_context
from pb.mcp.server import mcp
from pb.mcp.tools.productivebrain import _require_writes


class PbToolError(Exception):
    """Raised when ProductiveBrain CLI escape-hatch operations fail."""


ALLOWED_SUBCOMMANDS: dict[str, set[str] | None] = {
    "goal": {"list", "add", "refine"},
    "anki": {"generate", "list", "export"},
    "plan": {"day"},
    "model": {"status", "list", "use"},
    "notes": {"inbox", "organise"},
    "review": {"day", "week"},
    "feedback": None,
    "pause": None,
    "resume": None,
    "finish": None,
    "do": None,
    "next": None,
    "study": None,
    "practise": None,
    "practice": None,
    "learn": None,
    "teach": None,
    "doctor": None,
    "init": None,
    "thought": None,
    "todo": None,
    "mcp": {"status", "doctor", "print-config", "pending", "confirm", "reject"},
}

READ_ONLY_PREFIXES = {
    ("anki", "list"),
    ("doctor",),
    ("goal", "list"),
    ("mcp", "status"),
    ("mcp", "doctor"),
    ("mcp", "print-config"),
    ("mcp", "pending"),
    ("model", "status"),
    ("model", "list"),
    ("next",),
    ("plan", "day"),
    ("review", "day"),
    ("review", "week"),
    ("notes", "inbox"),
}

QUERY_COMMANDS = {
    "goals": "goal list",
    "current": "now --json",
}

# Only commands whose stdout is JSON should be parsed; the rest stay as text.
JSON_QUERY_TYPES = {"current"}


def _cli_command_prefix() -> list[str]:
    """Prefer the installed `pb` binary; fall back to module invocation."""
    pb_path = shutil.which("pb")
    if pb_path:
        return [pb_path]
    return [sys.executable, "-m", "pb.cli.main"]


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Empty command.")
    return parts


def _is_command_allowed(command: str) -> bool:
    try:
        parts = _split_command(command)
    except ValueError:
        return False
    if parts[0] not in ALLOWED_SUBCOMMANDS:
        return False
    allowed_subs = ALLOWED_SUBCOMMANDS[parts[0]]
    if allowed_subs is None:
        return True
    if len(parts) < 2:
        return False
    return parts[1] in allowed_subs


def _is_read_only_command(command: str) -> bool:
    try:
        parts = tuple(_split_command(command))
    except ValueError:
        return False
    return any(parts[: len(prefix)] == prefix for prefix in READ_ONLY_PREFIXES)


def _run_pb_command(command: str) -> dict:
    try:
        result = subprocess.run(
            _cli_command_prefix() + _split_command(command),
            capture_output=True,
            text=True,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        return {"command": command, "error": "Command timed out"}
    except FileNotFoundError:
        return {"command": command, "error": "CLI executable not found"}
    except ValueError as exc:
        return {"command": command, "error": f"Invalid command: {exc}"}

    return {
        "command": command,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "success": result.returncode == 0,
    }


@mcp.tool()
def pb_query(query_type: str) -> dict:
    """Query ProductiveBrain state information.

    Returns `data` (parsed JSON) only for known JSON-emitting commands;
    everything else returns `output` (raw text) so the LLM doesn't get
    silently-dropped errors when text is parsed as JSON.
    """
    if query_type not in QUERY_COMMANDS:
        return {
            "error": f"Unknown query type: {query_type}",
            "available_types": list(QUERY_COMMANDS.keys()),
        }

    result = _run_pb_command(QUERY_COMMANDS[query_type])
    if not result.get("success"):
        return {"query": query_type, **result}

    if query_type in JSON_QUERY_TYPES:
        try:
            import json

            data = json.loads(result["stdout"])
            return {"query": query_type, "data": data}
        except json.JSONDecodeError as exc:
            return {
                "query": query_type,
                "output": result["stdout"],
                "parse_error": f"Expected JSON, got: {exc}",
            }
    return {"query": query_type, "output": result["stdout"]}


@mcp.tool()
def pb_command(command: str) -> dict:
    """Execute a whitelisted ProductiveBrain CLI command."""
    if not _is_command_allowed(command):
        return {
            "error": f"Command not allowed: {command}",
            "allowed_commands": sorted(ALLOWED_SUBCOMMANDS.keys()),
        }
    if not get_mcp_context().allow_writes and not _is_read_only_command(command):
        return {
            "command": command,
            "error": "This MCP server is running in read-only mode. Restart with --allow-writes for mutating pb commands.",
            "success": False,
        }
    return _run_pb_command(command)


@mcp.tool()
def pb_respond(session_id: str, select: int | None = None, fill: dict | None = None) -> dict:
    """Advance an active pb dispatch session.

    After pb_command opens a session (status: active), call this tool with the
    session_id to select an option (select=N, 1-indexed) or fill fields (fill={key: value}).
    Returns the next InteractionEnvelope or status: complete when done.

    Args:
        session_id: The active dispatch session ID returned by a previous pb_command('do ...').
        select: 1-based index of the option to select from the envelope's options list.
        fill: Dict of free-text intent or field values. Use {"text": "..."} for raw intent.

    Returns:
        InteractionEnvelope as a dict: {session_id, status, prompt, options, fields}
    """
    if not get_mcp_context().allow_writes:
        return {"error": "pb_respond requires --allow-writes mode"}
    from pb.mcp.protocol import advance_session
    return advance_session(session_id, select=select, fill=fill or {})


@mcp.tool()
def pb_find(query: str = "", days: int = 0) -> dict:
    """Search pb-managed directories (vault + data dir) for files by name or recency.

    Read-only. query: a substring to fuzzy-match against file paths.
    days: if > 0, restrict to files modified within the last N days.
    """
    from pb.cli.commands.find import collect_files, filter_by_days, filter_by_fzf
    runtime = get_runtime_context()
    files = collect_files(runtime.vault_path, runtime.data_dir)
    if days > 0:
        files = filter_by_days(files, days)
    if query:
        files = filter_by_fzf(files, query)
    return {"files": [str(p) for p in files], "count": len(files)}


@mcp.tool()
def pb_ingest(gmail: bool = True, feeds: bool = True, scrapers: bool = True) -> dict:
    """Run the unified ingestion pipeline (gmail / feeds / scrapers). Requires --allow-writes."""
    _require_writes()
    from pb.cli.commands.ingest import run_ingest
    run_ingest(gmail=gmail, feeds=feeds, scrapers=scrapers)
    return {"status": "ok", "ran": {"gmail": gmail, "feeds": feeds, "scrapers": scrapers}}


@mcp.tool()
def pb_agents(action: str = "list", agent_id: str = "") -> dict:
    """Inspect or override agent frecency weights. action: list | pin | suppress | clear.

    action="list" is read-only. pin/suppress/clear require --allow-writes.
    """
    from pb.core.agent_weights import list_agent_weights, set_weight_override
    if action == "list":
        return {"agents": list_agent_weights()}
    _require_writes()
    override = {"pin": "pin", "suppress": "suppress", "clear": None}
    if action not in override:
        return {"error": f"Unknown action: {action}. Use list | pin | suppress | clear."}
    if not agent_id:
        return {"error": "agent_id is required for pin/suppress/clear."}
    set_weight_override(agent_id, override[action])
    return {"agent_id": agent_id, "action": action}
