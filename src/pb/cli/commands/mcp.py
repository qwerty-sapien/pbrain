# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""MCP setup and diagnostics commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from pb.cli.console import get_console


app = typer.Typer(no_args_is_help=True)

_CLIENT_CONFIG_PATHS = {
    "claude-desktop": Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json",
    "codex": Path.home() / ".codex" / "config.toml",
    "cline": Path.home() / ".config" / "Code" / "User" / "globalStorage" / "saoudrizwan.claude-dev" / "settings" / "cline_mcp_settings.json",
    "generic": Path.home() / ".config" / "productivebrain" / "mcp-example.json",
}


def _server_command(vault: str | None = None, *, allow_writes: bool = False, config_path: str | None = None) -> list[str]:
    command = ["productivebrain-mcp"]
    if vault:
        command.extend(["--vault", vault])
    if allow_writes:
        command.append("--allow-writes")
    if config_path:
        command.extend(["--config", config_path])
    return command


def _client_payload(client: str, vault: str | None, allow_writes: bool, config_path: str | None) -> dict[str, object]:
    command = _server_command(vault, allow_writes=allow_writes, config_path=config_path)
    if client == "claude-desktop":
        return {
            "mcpServers": {
                "productivebrain": {
                    "command": command[0],
                    "args": command[1:],
                }
            }
        }
    return {
        "productivebrain": {
            "command": command[0],
            "args": command[1:],
        }
    }


@app.command("print-config")
def mcp_print_config(
    client: str = typer.Option("claude-desktop", "--client", help="claude-desktop | codex | cline | generic"),
    vault: str = typer.Option("", "--vault", help="Optional vault profile name."),
    allow_writes: bool = typer.Option(False, "--allow-writes", help="Include the write-enabled server flag."),
    config_path: str = typer.Option("", "--config", help="Optional explicit ProductiveBrain config path."),
) -> None:
    """Print a ready-to-paste MCP client configuration snippet."""
    client_name = (client or "claude-desktop").strip().lower()
    payload = _client_payload(client_name, vault or None, allow_writes, config_path or None)
    typer.echo(json.dumps(payload, indent=2))


@app.command("status")
def mcp_status() -> None:
    """Show expected client config locations and whether they already exist."""
    console = get_console()
    console.rule("[header]MCP Status[/]")
    server_path = shutil.which("productivebrain-mcp") or shutil.which("pb-mcp")
    console.print(f"Server executable: {server_path or 'not found on PATH'}")
    for client, path in _CLIENT_CONFIG_PATHS.items():
        console.print(f"{client}: {'present' if path.exists() else 'missing'} ({path})")


@app.command("doctor")
def mcp_doctor() -> None:
    """Check MCP server prerequisites."""
    console = get_console()
    server_path = shutil.which("productivebrain-mcp") or shutil.which("pb-mcp")
    console.rule("[header]MCP Doctor[/]")
    console.print(f"Server executable: {server_path or 'missing'}")
    if server_path is None:
        console.print("[error]Install the package entrypoints first so clients can launch the MCP server.[/]")
        raise typer.Exit(code=53)
    console.print("[success]MCP server entrypoint is available.[/]")


@app.command("install", hidden=True)
def mcp_install(
    client: str = typer.Option("claude-desktop", "--client", help="claude-desktop | codex | cline | generic"),
    vault: str = typer.Option("", "--vault", help="Optional vault profile name."),
    allow_writes: bool = typer.Option(False, "--allow-writes", help="Include the write-enabled server flag."),
    config_path: str = typer.Option("", "--config", help="Optional explicit ProductiveBrain config path."),
) -> None:
    """Print the configuration to install for a supported MCP client."""
    path = _CLIENT_CONFIG_PATHS.get((client or "claude-desktop").strip().lower())
    typer.echo(f"Suggested config path: {path}")
    mcp_print_config(client=client, vault=vault, allow_writes=allow_writes, config_path=config_path)


def _ensure_mcp_runtime_for_pending() -> None:
    """Configure the MCP context so pending-queue lookups can find the data_dir."""
    from pb.mcp.context import configure_mcp_context

    configure_mcp_context(allow_writes=True)


@app.command("pending")
def mcp_pending() -> None:
    """List tier-2 MCP actions queued for user confirmation."""
    from pb.mcp.pending import list_pending

    # Make sure the tool implementations are registered.
    import pb.mcp.tools.productivebrain  # noqa: F401
    import pb.mcp.tools.vault  # noqa: F401

    _ensure_mcp_runtime_for_pending()
    console = get_console()
    items = list_pending()
    if not items:
        console.print("[dim]No pending MCP actions.[/]")
        return
    console.rule("[header]Pending MCP actions[/]")
    for action in items:
        console.print(f"[bold]{action.id}[/]  [{action.risk}]  {action.tool_name}")
        console.print(f"  [dim]{action.created_at}[/]")
        console.print(f"  {action.summary}")
        console.print()
    console.print(
        f"[dim]{len(items)} pending. "
        "Run `pb mcp confirm <id>` to execute, `pb mcp reject <id>` to dismiss.[/]"
    )


@app.command("confirm")
def mcp_confirm(action_id: str = typer.Argument(..., help="Pending action id")) -> None:
    """Execute a queued tier-2 MCP action."""
    from pb.mcp.pending import execute_pending

    import pb.mcp.tools.productivebrain  # noqa: F401
    import pb.mcp.tools.vault  # noqa: F401

    _ensure_mcp_runtime_for_pending()
    console = get_console()
    outcome = execute_pending(action_id)
    if not outcome.get("ok"):
        console.print(f"[error]{outcome.get('error', 'Unknown error')}[/]")
        raise typer.Exit(code=1)
    console.print(f"[success]Confirmed {outcome.get('tool')}.[/]")
    console.print(json.dumps(outcome.get("result"), indent=2, default=str))


@app.command("reject")
def mcp_reject(action_id: str = typer.Argument(..., help="Pending action id")) -> None:
    """Dismiss a queued tier-2 MCP action without executing it."""
    from pb.mcp.pending import delete_pending

    _ensure_mcp_runtime_for_pending()
    console = get_console()
    if delete_pending(action_id):
        console.print(f"[success]Rejected {action_id}.[/]")
    else:
        console.print(f"[error]No pending action with id {action_id}.[/]")
        raise typer.Exit(code=1)
