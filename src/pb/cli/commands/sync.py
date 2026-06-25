# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Markdown/SQLite sync commands."""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import typer

from pb.cli.console import get_console
from pb.cli.preview import confirm_preview
from pb.storage.database import get_connection
from pb.storage.repository import Repository
from pb.vault.lifecycle import read_frontmatter


app = typer.Typer(no_args_is_help=False, invoke_without_command=True)


def _iter_markdown_files(vault_root: Path) -> list[Path]:
    return [
        path
        for path in sorted(vault_root.rglob("*.md"))
        if not any(part.startswith(".") for part in path.parts)
    ]


def _note_title(path: Path, frontmatter: dict, body: str) -> str:
    if frontmatter.get("title"):
        return str(frontmatter["title"])
    for line in body.splitlines():
        if line.startswith("#"):
            return line.lstrip("#").strip()
    return path.stem.replace("-", " ")


def _collect_sync_state(vault_root: Path) -> dict[str, object]:
    repo = Repository()
    markdown_notes: list[dict[str, object]] = []
    parse_errors: list[dict[str, str]] = []
    slug_counter: Counter[str] = Counter()
    existing_paths: set[str] = set()
    goal_titles_in_markdown: set[str] = set()

    for note_path in _iter_markdown_files(vault_root):
        rel_path = str(note_path.relative_to(vault_root))
        existing_paths.add(rel_path)
        try:
            content = note_path.read_text(encoding="utf-8")
            frontmatter, body = read_frontmatter(content)
        except Exception as exc:
            parse_errors.append({"path": rel_path, "error": str(exc)})
            continue

        slug = note_path.stem
        slug_counter[slug] += 1
        note_type = str(frontmatter.get("type", "") or "")
        title = _note_title(note_path, frontmatter, body)
        domain = str(frontmatter.get("domain", "") or "")
        if note_type == "goal":
            goal_titles_in_markdown.add(title)
        markdown_notes.append(
            {
                "id": str(frontmatter.get("id") or rel_path),
                "note_type": note_type,
                "slug": slug,
                "path": rel_path,
                "title": title,
                "domain": domain,
                "updated_at": datetime.fromtimestamp(note_path.stat().st_mtime, tz=timezone.utc).isoformat(),
                "source_ref": str(frontmatter.get("source") or frontmatter.get("source_note") or ""),
            }
        )

    duplicates = sorted(slug for slug, count in slug_counter.items() if count > 1)

    with get_connection() as conn:
        indexed_rows = conn.execute(
            "SELECT id, note_type, slug, path, title, domain, updated_at, source_ref FROM vault_notes"
        ).fetchall()

    indexed_by_path = {row["path"]: dict(row) for row in indexed_rows}

    missing_index = sorted(
        note["path"]
        for note in markdown_notes
        if note["path"] not in indexed_by_path
    )
    stale_paths = sorted(path for path in indexed_by_path if path not in existing_paths)

    sqlite_goals_missing_markdown = []
    for goal in repo.list_goal_arcs():
        if goal.title not in goal_titles_in_markdown:
            sqlite_goals_missing_markdown.append(goal.title)

    return {
        "markdown_notes": markdown_notes,
        "missing_index": missing_index,
        "stale_paths": stale_paths,
        "parse_errors": parse_errors,
        "duplicate_slugs": duplicates,
        "sqlite_goals_missing_markdown": sqlite_goals_missing_markdown,
    }


def _apply_sync(markdown_notes: list[dict[str, object]], *, remove_paths: list[str] | None = None) -> dict[str, int]:
    upserted = 0
    removed = 0
    remove_paths = remove_paths or []
    with get_connection() as conn:
        for note in markdown_notes:
            conn.execute(
                """
                INSERT INTO vault_notes (id, note_type, slug, path, title, domain, updated_at, source_ref)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(path) DO UPDATE SET
                    id = excluded.id,
                    note_type = excluded.note_type,
                    slug = excluded.slug,
                    title = excluded.title,
                    domain = excluded.domain,
                    updated_at = excluded.updated_at,
                    source_ref = excluded.source_ref
                """,
                (
                    note["id"],
                    note["note_type"],
                    note["slug"],
                    note["path"],
                    note["title"],
                    note["domain"],
                    note["updated_at"],
                    note["source_ref"],
                ),
            )
            upserted += 1
        for path in remove_paths:
            removed += conn.execute("DELETE FROM vault_notes WHERE path = ?", (path,)).rowcount
        conn.commit()
    return {"upserted": upserted, "removed": removed}


@app.callback(invoke_without_command=True)
def sync_root(
    ctx: typer.Context,
    check: bool = typer.Option(False, "--check", help="Only report drift; do not mutate SQLite."),
    repair: bool = typer.Option(False, "--repair", help="Preview or apply conservative repairs."),
    yes: bool = typer.Option(False, "--yes", help="Apply repairs without prompting."),
    json_out: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    """Mirror canonical Markdown state into SQLite and report drift."""
    console = get_console()
    runtime = ctx.obj.get("runtime") if ctx.obj else None
    if runtime is None:
        raise typer.Exit(code=53)

    state = _collect_sync_state(runtime.vault_path)
    payload = {
        "vault": runtime.vault_name,
        "markdown_count": len(state["markdown_notes"]),
        "missing_index": state["missing_index"],
        "stale_paths": state["stale_paths"],
        "parse_errors": state["parse_errors"],
        "duplicate_slugs": state["duplicate_slugs"],
        "sqlite_goals_missing_markdown": state["sqlite_goals_missing_markdown"],
    }

    if json_out:
        typer.echo(json.dumps(payload, indent=2))
        if payload["parse_errors"] or payload["duplicate_slugs"] or payload["stale_paths"] or payload["missing_index"]:
            raise typer.Exit(code=1)
        return

    console.rule("[header]Sync[/]")
    console.print(f"Vault: {runtime.vault_name} -> {runtime.vault_path}")
    console.print(f"Markdown notes scanned: {payload['markdown_count']}")
    console.print(f"Missing SQLite index rows: {len(payload['missing_index'])}")
    console.print(f"Stale SQLite note paths: {len(payload['stale_paths'])}")
    console.print(f"Frontmatter parse errors: {len(payload['parse_errors'])}")
    console.print(f"Duplicate slugs: {len(payload['duplicate_slugs'])}")
    console.print(f"SQLite goals missing durable Markdown: {len(payload['sqlite_goals_missing_markdown'])}")

    if payload["missing_index"]:
        console.print(f"[dim]Unindexed markdown:[/] {', '.join(payload['missing_index'][:5])}")
    if payload["stale_paths"]:
        console.print(f"[dim]Stale index paths:[/] {', '.join(payload['stale_paths'][:5])}")
    if payload["duplicate_slugs"]:
        console.print(f"[dim]Duplicate slugs:[/] {', '.join(payload['duplicate_slugs'][:5])}")

    if check:
        if payload["parse_errors"] or payload["duplicate_slugs"] or payload["stale_paths"] or payload["missing_index"]:
            raise typer.Exit(code=1)
        return

    if repair:
        accepted = confirm_preview(
            yes=yes or bool(ctx.obj and ctx.obj.get("yes")),
            action_label="Apply the conservative sync repairs",
        )
        if not accepted:
            console.print("[dim]Preview only. No sync repairs were applied.[/]")
            return
        stats = _apply_sync(state["markdown_notes"], remove_paths=state["stale_paths"])
        console.print(f"[success]Repaired sync state.[/] Upserted {stats['upserted']} notes, removed {stats['removed']} stale rows.")
        return

    stats = _apply_sync(state["markdown_notes"])
    console.print(f"[success]Indexed {stats['upserted']} markdown notes into SQLite.[/]")
