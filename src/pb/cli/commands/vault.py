# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault profile management commands."""

from __future__ import annotations

import json
import os
from pathlib import Path

import typer
import yaml

from pb.cli.helpers import confirm_choice
from pb.runtime import get_session_auto_yes
from pb.storage.config import (
    DEFAULT_QUARANTINE_FOLDER,
    get_active_vault_name,
    get_config,
    get_data_dir,
    get_quarantine_folder,
    get_quarantine_path,
    get_vault_path,
    remove_vault_profile,
    rename_vault_profile,
    set_active_vault,
    upsert_vault_profile,
)
from pb.vault.config import VAULT_SCHEMA


app = typer.Typer(no_args_is_help=True)


def _config_path(ctx: typer.Context) -> Path | None:
    runtime = (ctx.obj or {}).get("runtime")
    return getattr(runtime, "config_path", None)


def _auto_yes(ctx: typer.Context, explicit: bool = False) -> bool:
    if explicit:
        return True
    if ctx.obj and ctx.obj.get("yes"):
        return True
    config = (ctx.obj or {}).get("config")
    return get_session_auto_yes(config)


def _doctor_payload(config, vault_name: str) -> dict[str, object]:
    vault_path = get_vault_path(config, vault=vault_name)
    data_dir = get_data_dir(config, vault=vault_name)
    quarantine_path = get_quarantine_path(config, vault=vault_name)
    markdown_files = list(vault_path.rglob("*.md")) if vault_path.exists() else []
    parse_errors = 0
    for note in markdown_files[:100]:
        try:
            content = note.read_text(errors="replace")
            if content.startswith("---"):
                _, fm, _ = content.split("---", 2)
                yaml.safe_load(fm)
        except Exception:
            parse_errors += 1

    writable = vault_path.exists() and os.access(vault_path, os.W_OK)
    quarantine_ready = quarantine_path.exists() or writable
    return {
        "vault": vault_name,
        "path": str(vault_path),
        "data_dir": str(data_dir),
        "quarantine_folder": get_quarantine_folder(config, vault=vault_name),
        "quarantine_path": str(quarantine_path),
        "path_exists": vault_path.exists(),
        "writable": writable,
        "quarantine_ready": quarantine_ready,
        "db_path": str(data_dir / "productivebrain.db"),
        "markdown_files": len(markdown_files),
        "frontmatter_parse_errors": parse_errors,
        "ok": vault_path.exists() and quarantine_ready and parse_errors == 0,
    }


@app.command("list")
def vault_list(ctx: typer.Context, json_out: bool = typer.Option(False, "--json")) -> None:
    """List configured vault profiles."""
    config = get_config(_config_path(ctx))
    rows = []
    for name, profile in config.vaults.items():
        rows.append(
            {
                "name": name,
                "active": name == config.general.active_vault,
                "path": profile.path,
                "data_dir": profile.data_dir,
                "quarantine_folder": profile.quarantine_folder,
            }
        )
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    for row in rows:
        marker = "*" if row["active"] else " "
        typer.echo(f"{marker} {row['name']}: {row['path']}")
        typer.echo(f"    data: {row['data_dir']}")
        typer.echo(f"    quarantine: {row['quarantine_folder']}")


@app.command("current")
def vault_current(ctx: typer.Context, json_out: bool = typer.Option(False, "--json")) -> None:
    """Show the active vault profile."""
    config = get_config(_config_path(ctx))
    name = get_active_vault_name(config)
    profile = config.vaults[name]
    payload = {
        "name": name,
        "path": profile.path,
        "data_dir": profile.data_dir,
        "quarantine_folder": profile.quarantine_folder,
    }
    typer.echo(json.dumps(payload, indent=2) if json_out else f"{name}: {profile.path}")


@app.command("add")
def vault_add(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Profile name"),
    vault_path: str = typer.Argument("", help="Vault root path; omit to create a sibling profile vault"),
    data_dir: str = typer.Option("", "--data-dir", help="Per-vault data directory"),
    quarantine_folder: str = typer.Option(DEFAULT_QUARANTINE_FOLDER, "--quarantine-folder"),
    use: bool = typer.Option(False, "--use", help="Switch to this vault after adding"),
) -> None:
    """Add a vault profile."""
    if vault_path:
        resolved_vault = Path(vault_path).expanduser()
    else:
        config = get_config(_config_path(ctx), force_reload=True)
        active_path = get_vault_path(config)
        resolved_vault = active_path.parent / name
    resolved_vault.mkdir(parents=True, exist_ok=True)
    updated = upsert_vault_profile(
        name,
        str(resolved_vault),
        data_dir=data_dir or None,
        quarantine_folder=quarantine_folder,
        path=_config_path(ctx),
    )
    Path(updated.vaults[name].data_dir).expanduser().mkdir(parents=True, exist_ok=True)
    if use:
        set_active_vault(name, path=_config_path(ctx))
    typer.echo(f"Added vault profile '{name}' -> {resolved_vault}")


@app.command("use")
def vault_use(ctx: typer.Context, name: str = typer.Argument(..., help="Profile name")) -> None:
    """Switch the active vault profile."""
    set_active_vault(name, path=_config_path(ctx))
    typer.echo(f"Active vault: {name}")


@app.command("rename")
def vault_rename(
    ctx: typer.Context,
    old: str = typer.Argument(..., help="Existing profile"),
    new: str = typer.Argument(..., help="New profile name"),
) -> None:
    """Rename a vault profile."""
    try:
        rename_vault_profile(old, new, path=_config_path(ctx))
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Renamed vault profile '{old}' -> '{new}'")


@app.command("remove")
def vault_remove(
    ctx: typer.Context,
    name: str = typer.Argument(..., help="Profile to remove"),
    yes: bool = typer.Option(False, "--yes", help="Skip confirmation"),
) -> None:
    """Remove a non-active vault profile."""
    if not _auto_yes(ctx, yes) and not confirm_choice(f"Remove vault profile '{name}'?"):
        raise typer.Exit(code=0)
    try:
        remove_vault_profile(name, path=_config_path(ctx))
    except (KeyError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"Removed vault profile '{name}'")


@app.command("doctor")
def vault_doctor(
    ctx: typer.Context,
    name: str = typer.Argument("", help="Optional profile name"),
    json_out: bool = typer.Option(False, "--json"),
    fix: bool = typer.Option(False, "--fix", help="Create missing writable directories"),
) -> None:
    """Validate the selected vault profile."""
    config = get_config(_config_path(ctx), force_reload=True)
    vault_name = name or get_active_vault_name(config)
    payload = _doctor_payload(config, vault_name)
    if fix and payload["path_exists"]:
        Path(payload["data_dir"]).expanduser().mkdir(parents=True, exist_ok=True)
        Path(payload["quarantine_path"]).mkdir(parents=True, exist_ok=True)
        payload = _doctor_payload(get_config(_config_path(ctx), force_reload=True), vault_name)
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return
    typer.echo(f"Vault: {payload['vault']}")
    typer.echo(f"  path exists: {payload['path_exists']}")
    typer.echo(f"  writable: {payload['writable']}")
    typer.echo(f"  quarantine ready: {payload['quarantine_ready']}")
    typer.echo(f"  db path: {payload['db_path']}")
    typer.echo(f"  markdown files: {payload['markdown_files']}")
    typer.echo(f"  frontmatter parse errors: {payload['frontmatter_parse_errors']}")
    if not payload["ok"]:
        raise typer.Exit(code=1)


@app.command("scaffold")
def vault_scaffold(
    ctx: typer.Context,
    name: str = typer.Argument("", help="Optional profile name"),
    yes: bool = typer.Option(False, "--yes", help="Apply immediately"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Preview or create the standard vault directory structure."""
    config = get_config(_config_path(ctx), force_reload=True)
    vault_name = name or get_active_vault_name(config)
    vault_path = get_vault_path(config, vault=vault_name)
    folders = sorted(set([*VAULT_SCHEMA, get_quarantine_folder(config, vault=vault_name)]))
    preview = [str(vault_path / folder) for folder in folders]
    if json_out:
        typer.echo(json.dumps({"vault": vault_name, "folders": preview}, indent=2))
        return
    typer.echo(f"Scaffold preview for {vault_name}:")
    for folder in preview:
        typer.echo(f"  {folder}")
    if not _auto_yes(ctx, yes):
        return
    for folder in folders:
        (vault_path / folder).mkdir(parents=True, exist_ok=True)
    typer.echo(f"Scaffolded {len(folders)} folders in {vault_path}")


@app.command("graph")
def vault_graph(
    note: str = typer.Argument("", help="Note path relative to the vault root; omit for graph overview"),
    depth: int = typer.Option(1, "--depth", min=1, help="Neighborhood depth (currently up to 2 hops)"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of formatted text"),
) -> None:
    """Show the link neighborhood for one note."""
    if not note:
        from pb.core.brain import BrainEngine

        orphan_list = BrainEngine().detect_orphans()
        if json_out:
            typer.echo(json.dumps({"orphan_count": len(orphan_list), "orphans": orphan_list}, indent=2))
            return
        if not orphan_list:
            typer.echo("No graph notes found yet.")
            return
        typer.echo(f"Graph overview: {len(orphan_list)} note(s) without links.")
        for item in orphan_list[:20]:
            typer.echo(f"  - {item['path']} ({item.get('folder', '')})")
        return
    from pb.mcp.tools.vault import VaultError, vault_link_graph

    try:
        payload = vault_link_graph(note, depth=depth)
    except VaultError as exc:
        raise typer.BadParameter(str(exc)) from exc

    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    typer.echo(f"Note: {payload['note']}")
    typer.echo(
        f"Depth: requested {payload['requested_depth']} -> effective {payload['effective_depth']}"
        f"{' (clamped)' if payload['clamped'] else ''}"
    )
    typer.echo(f"Outgoing ({payload['outgoing_count']}):")
    for item in payload["outgoing"]:
        resolved = item.get("resolved_path") or "(missing)"
        typer.echo(f"  - {item.get('target', '')} -> {resolved}")
    typer.echo(f"Incoming ({payload['incoming_count']}):")
    for item in payload["incoming"]:
        typer.echo(f"  - {item}")
    if payload["effective_depth"] > 1:
        typer.echo("2-hop outgoing:")
        for item in payload["out2"]:
            typer.echo(f"  - {item}")
        typer.echo("2-hop incoming:")
        for item in payload["in2"]:
            typer.echo(f"  - {item}")


@app.command("neighbors")
def vault_neighbors(
    note: str = typer.Argument(..., help="Note path relative to the vault root"),
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of formatted text"),
) -> None:
    """Show immediate inbound and outbound neighbors for one note."""
    vault_graph(note=note, depth=1, json_out=json_out)


@app.command("orphans")
def vault_orphans(
    json_out: bool = typer.Option(False, "--json", help="Emit JSON instead of formatted text"),
) -> None:
    """Show notes with no inbound or outbound links."""
    from pb.core.brain import BrainEngine

    orphan_list = BrainEngine().detect_orphans()
    if json_out:
        typer.echo(json.dumps(orphan_list, indent=2))
        return
    if not orphan_list:
        typer.echo("No orphan notes found.")
        return
    for item in orphan_list:
        typer.echo(f"{item['path']} ({item.get('folder', '')})")
