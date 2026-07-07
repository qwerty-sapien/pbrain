# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of pbrain.
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
from pb.mcp.tools.pbrain import _require_writes


class PbToolError(Exception):
    """Raised when pbrain CLI escape-hatch operations fail."""


_GLOBAL_VALUE_OPTIONS = {"--config", "--vault"}
_GLOBAL_FLAG_OPTIONS = {"--yes", "--verbose", "-v", "--dryrun"}
_HELP_OPTIONS = {"--help", "-h"}
_PROGRAM_NAMES = {"pb", "brain"}

# Free-form prefixes accept command-specific arguments after the prefix.
ALLOWED_COMMAND_PREFIXES: set[tuple[str, ...]] = {
    ("anki", "accept"),
    ("anki", "export"),
    ("anki", "generate"),
    ("anki", "list"),
    ("anki", "pending"),
    ("anki", "reject"),
    ("anki", "review"),
    ("config", "agents"),
    ("config", "session", "auto-yes"),
    ("config", "session", "status"),
    ("config", "set"),
    ("config", "show"),
    ("context", "add"),
    ("context", "ask"),
    ("context", "bundle", "add"),
    ("context", "bundle", "create"),
    ("context", "bundle", "list"),
    ("context", "bundle", "remove"),
    ("context", "bundle", "show"),
    ("context", "doctor"),
    ("context", "infer"),
    ("context", "inspect"),
    ("context", "list"),
    ("context", "lock"),
    ("context", "remove"),
    ("context", "show"),
    ("context", "status"),
    ("context", "unlock"),
    ("do",),
    ("doctor",),
    ("feedback",),
    ("finish",),
    ("goal", "add"),
    ("goal", "delete"),
    ("goal", "list"),
    ("goal", "refine"),
    ("init",),
    ("init", "llm"),
    ("learn",),
    ("mcp", "confirm"),
    ("mcp", "doctor"),
    ("mcp", "pending"),
    ("mcp", "print-config"),
    ("mcp", "reject"),
    ("mcp", "status"),
    ("model", "list"),
    ("model", "status"),
    ("model", "use"),
    ("next",),
    ("notes", "inbox"),
    ("notes", "organise"),
    ("pause",),
    ("plan", "block", "add"),
    ("plan", "block", "edit"),
    ("plan", "block", "list"),
    ("plan", "block", "rm"),
    ("plan", "day"),
    ("plan", "week"),
    ("practice",),
    ("practise",),
    ("reset",),
    ("resume",),
    ("review", "day"),
    ("review", "week"),
    ("set", "language"),
    ("set", "model"),
    ("set", "status"),
    ("study",),
    ("teach",),
    ("thought",),
    ("todo",),
    ("update",),
    ("vault", "add"),
    ("vault", "current"),
    ("vault", "doctor"),
    ("vault", "graph"),
    ("vault", "list"),
    ("vault", "neighbors"),
    ("vault", "orphans"),
    ("vault", "remove"),
    ("vault", "rename"),
    ("vault", "scaffold"),
    ("vault", "use"),
}

# Exact entries are command groups or callbacks that should be callable without
# accidentally opening every hidden subcommand beneath that group.
ALLOWED_EXACT_COMMANDS: set[tuple[str, ...]] = {
    ("anki",),
    ("config",),
    ("config", "session"),
    ("context",),
    ("context", "bundle"),
    ("goal",),
    ("mcp",),
    ("model",),
    ("notes",),
    ("plan",),
    ("plan", "block"),
    ("review",),
    ("set",),
    ("vault",),
}

READ_ONLY_PREFIXES = {
    ("anki", "pending"),
    ("context", "ask"),
    ("context", "bundle", "list"),
    ("context", "bundle", "show"),
    ("context", "doctor"),
    ("context", "infer"),
    ("context", "inspect"),
    ("context", "list"),
    ("context", "show"),
    ("context", "status"),
    ("doctor",),
    ("goal", "list"),
    ("mcp", "doctor"),
    ("mcp", "pending"),
    ("mcp", "print-config"),
    ("mcp", "status"),
    ("model", "list"),
    ("model", "status"),
    ("notes", "inbox"),
    ("review", "day"),
    ("review", "week"),
    ("set", "status"),
    ("vault", "current"),
    ("vault", "doctor"),
    ("vault", "graph"),
    ("vault", "list"),
    ("vault", "neighbors"),
    ("vault", "orphans"),
}

READ_ONLY_EXACT_COMMANDS = {
    ("anki",),
    ("config",),
    ("config", "session"),
    ("context",),
    ("context", "bundle"),
    ("mcp",),
    ("notes",),
    ("plan",),
    ("plan", "block"),
    ("review",),
    ("set",),
    ("vault",),
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


def _mcp_cli_global_options() -> list[str]:
    """Propagate the MCP server's selected vault/config into CLI subprocesses."""
    ctx = get_mcp_context()
    options: list[str] = []
    if ctx.config_path is not None:
        options.extend(["--config", str(ctx.config_path)])
    if ctx.vault:
        options.extend(["--vault", ctx.vault])
    return options


def _split_command(command: str) -> list[str]:
    parts = shlex.split(command)
    if not parts:
        raise ValueError("Empty command.")
    return parts


def _strip_program_name(parts: list[str]) -> list[str]:
    if parts and parts[0] in _PROGRAM_NAMES:
        return parts[1:]
    return parts


def _command_tokens(parts: list[str]) -> tuple[str, ...]:
    """Return command-path tokens after root CLI options and optional `pb`."""
    parts = _strip_program_name(list(parts))
    index = 0
    while index < len(parts):
        token = parts[index]
        if token in _GLOBAL_FLAG_OPTIONS:
            index += 1
            continue
        if token in _GLOBAL_VALUE_OPTIONS:
            index += 2
            continue
        if any(token.startswith(f"{option}=") for option in _GLOBAL_VALUE_OPTIONS):
            index += 1
            continue
        break
    return tuple(parts[index:])


def _matches_prefix(tokens: tuple[str, ...], prefix: tuple[str, ...]) -> bool:
    return len(tokens) >= len(prefix) and tokens[: len(prefix)] == prefix


def _without_trailing_help(tokens: tuple[str, ...]) -> tuple[str, ...]:
    if tokens and tokens[-1] in _HELP_OPTIONS:
        return tokens[:-1]
    return tokens


def _contains_option(tokens: tuple[str, ...], options: set[str]) -> bool:
    for token in tokens:
        if token in options:
            return True
        if any(token.startswith(f"{option}=") for option in options if option.startswith("--")):
            return True
    return False


def _allowed_command_labels() -> list[str]:
    return sorted(
        " ".join(path)
        for path in (ALLOWED_COMMAND_PREFIXES | ALLOWED_EXACT_COMMANDS)
    )


def _is_command_allowed(command: str) -> bool:
    try:
        tokens = _command_tokens(_split_command(command))
    except ValueError:
        return False
    if not tokens:
        return False
    if tokens[0] in _HELP_OPTIONS:
        return True
    comparable = _without_trailing_help(tokens)
    if comparable in ALLOWED_EXACT_COMMANDS:
        return True
    return any(_matches_prefix(comparable, prefix) for prefix in ALLOWED_COMMAND_PREFIXES)


def _is_read_only_command(command: str) -> bool:
    try:
        raw_parts = tuple(_strip_program_name(_split_command(command)))
        tokens = _command_tokens(list(raw_parts))
    except ValueError:
        return False
    if not tokens:
        return False
    if tokens[0] in _HELP_OPTIONS or tokens[-1] in _HELP_OPTIONS:
        return True
    if tokens in READ_ONLY_EXACT_COMMANDS:
        return True
    if _matches_prefix(tokens, ("next",)):
        return not _contains_option(tokens, {"--run", "--schedule", "-s", "--reminder"})
    if _matches_prefix(tokens, ("anki", "list")):
        return not _contains_option(tokens, {"--suggested"})
    if _matches_prefix(tokens, ("notes", "organise")):
        return "--yes" not in raw_parts
    return any(_matches_prefix(tokens, prefix) for prefix in READ_ONLY_PREFIXES)


def _run_pb_command(command: str) -> dict:
    try:
        command_parts = _strip_program_name(_split_command(command))
        result = subprocess.run(
            _cli_command_prefix() + _mcp_cli_global_options() + command_parts,
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
    """Query pbrain state information.

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
    """Execute a whitelisted pbrain CLI command."""
    if not _is_command_allowed(command):
        return {
            "error": f"Command not allowed: {command}",
            "allowed_commands": _allowed_command_labels(),
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
