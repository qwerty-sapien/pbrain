# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Quarantine-first notes commands."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import typer
import yaml

from pb.storage.config import get_config, get_quarantine_path, get_vault_path


app = typer.Typer(no_args_is_help=True)


def _config_path(ctx: typer.Context) -> Path | None:
    runtime = (ctx.obj or {}).get("runtime")
    return getattr(runtime, "config_path", None)


def _slugify(text: str) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return normalized or "note"


def _parse_frontmatter(note_path: Path) -> dict[str, object]:
    content = note_path.read_text(errors="replace")
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except Exception:
        return {}


def _target_path(vault_root: Path, quarantine_root: Path, note_path: Path, merge: bool) -> Path:
    frontmatter = _parse_frontmatter(note_path)
    note_type = str(frontmatter.get("type", "") or "").lower()
    title = str(frontmatter.get("title", "") or note_path.stem)
    domain = _slugify(str(frontmatter.get("domain", "") or ""))
    slug = _slugify(title)

    if merge:
        if note_type in {"goal"}:
            return vault_root / "direction" / "goals" / f"{slug}.md"
        if note_type in {"person"}:
            return vault_root / "people" / f"{slug}.md"
        if note_type in {"opportunity", "opp"}:
            return vault_root / "opportunities" / "active" / f"{slug}.md"
        if note_type in {"daily_log"}:
            return vault_root / "logs" / "daily" / note_path.name
        if note_type in {"weekly_log"}:
            return vault_root / "logs" / "weekly" / note_path.name
        if note_type in {"session_summary", "study_summary", "practise_summary"}:
            return vault_root / "logs" / "sessions" / note_path.name
        if domain:
            return vault_root / "knowledge" / domain / f"{slug}.md"
        return vault_root / "knowledge" / "general" / f"{slug}.md"

    bucket = note_type or "notes"
    return quarantine_root / bucket / note_path.name


@dataclass
class OrganiseMove:
    source: Path
    target: Path
    action: str


def _collect_moves(vault_root: Path, quarantine_root: Path, merge: bool) -> list[OrganiseMove]:
    moves: list[OrganiseMove] = []
    for note_path in sorted(quarantine_root.rglob("*.md")):
        if note_path.is_file():
            target = _target_path(vault_root, quarantine_root, note_path, merge)
            action = "skip" if target.exists() and target.resolve() != note_path.resolve() else "move"
            moves.append(OrganiseMove(source=note_path, target=target, action=action))
    return moves


def apply_moves(moves: list[OrganiseMove], *, quarantine_root: Path, flatten: bool = False) -> int:
    """Apply conservative note moves without overwriting existing targets."""
    applied = 0
    for move in moves:
        if move.action != "move":
            continue
        move.target.parent.mkdir(parents=True, exist_ok=True)
        if move.target.resolve() == move.source.resolve():
            continue
        move.source.replace(move.target)
        applied += 1
        if flatten:
            parent = move.source.parent
            while parent != quarantine_root and parent.exists():
                try:
                    parent.rmdir()
                except OSError:
                    break
                parent = parent.parent
    return applied


@app.command("inbox")
def notes_inbox(ctx: typer.Context, json_out: bool = typer.Option(False, "--json")) -> None:
    """List quarantined generated notes."""
    config = get_config(_config_path(ctx))
    quarantine_root = get_quarantine_path(config)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for note_path in sorted(quarantine_root.rglob("*.md")):
        if note_path.is_file():
            rows.append(
                {
                    "path": str(note_path.relative_to(get_vault_path(config))),
                    "title": _parse_frontmatter(note_path).get("title", note_path.stem),
                }
            )
    if json_out:
        typer.echo(json.dumps(rows, indent=2))
        return
    typer.echo(f"Quarantine: {quarantine_root}")
    if not rows:
        typer.echo("  No quarantined notes.")
        return
    for row in rows:
        typer.echo(f"  {row['path']}")


@app.command("organise")
def notes_organise(
    ctx: typer.Context,
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview only"),
    merge: bool = typer.Option(False, "--merge", help="Move notes into the wider vault structure"),
    flatten: bool = typer.Option(False, "--flatten", help="Remove empty packet folders after moving"),
    yes: bool = typer.Option(False, "--yes", help="Apply changes"),
    json_out: bool = typer.Option(False, "--json"),
) -> None:
    """Preview or apply conservative note organization moves."""
    config = get_config(_config_path(ctx), force_reload=True)
    vault_root = get_vault_path(config)
    quarantine_root = get_quarantine_path(config)
    quarantine_root.mkdir(parents=True, exist_ok=True)
    moves = _collect_moves(vault_root, quarantine_root, merge=merge)

    payload = {
        "merge": merge,
        "moves": [
            {
                "source": str(move.source.relative_to(vault_root)),
                "target": str(move.target.relative_to(vault_root)),
                "action": move.action,
            }
            for move in moves
        ],
    }
    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        return

    if not moves:
        typer.echo("No quarantined notes to organize.")
        return

    typer.echo("Notes organize preview:")
    for move in moves:
        rel_source = move.source.relative_to(vault_root)
        rel_target = move.target.relative_to(vault_root)
        typer.echo(f"  {move.action}: {rel_source} -> {rel_target}")

    apply_changes = not dry_run and (yes or (ctx.obj and ctx.obj.get("yes")))
    if not apply_changes:
        return

    applied = apply_moves(moves, quarantine_root=quarantine_root, flatten=flatten)
    typer.echo(f"Applied {applied} note moves.")
