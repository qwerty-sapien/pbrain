# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Inbox command -- vault-based inbox with type conversion (D-09 to D-12).

Scans 00-inbox/ vault folders (feeds/, gmail/).
Groups items by source. Supports in-place type conversion via question trees.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
import yaml

from pb.core.naming import stored_display_title
from pb.domain.enums import TaskState
from pb.storage.repository import Repository


app = typer.Typer(no_args_is_help=False)

TYPE_OPTIONS = ["Event", "Opportunity", "Concept", "Task", "Person", "Skip"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content.

    Uses yaml.safe_load to prevent code execution (T-06-20).
    Returns {} on any parse error or missing delimiters.
    """
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


def _scan_vault_inbox(vault_path: Path) -> list[dict]:
    """Scan 00-inbox/feeds/ and 00-inbox/gmail/ for .md files.

    For each file, parses frontmatter and adds:
      _path   -- Path object to the file
      _source -- "feeds" or "gmail"

    Returns items sorted by 'ingested' field descending (newest first).
    Creates directories if they don't exist.

    Only scans specific subdirectories to prevent path traversal (T-06-21).
    """
    inbox_root = vault_path / "00-inbox"
    sources = ["feeds", "gmail"]
    items: list[dict] = []

    for source in sources:
        source_dir = inbox_root / source
        source_dir.mkdir(parents=True, exist_ok=True)

        for md_file in sorted(source_dir.glob("*.md")):
            try:
                content = md_file.read_text()
                fm = _parse_frontmatter(content)
                if not fm:
                    continue
                fm["_path"] = md_file
                fm["_source"] = source
                items.append(fm)
            except Exception:
                continue

    # Sort by ingested field descending (newest first), fallback to filename
    def sort_key(item: dict) -> str:
        return item.get("ingested", item.get("date", "0000-00-00"))

    items.sort(key=sort_key, reverse=True)
    return items


def _display_grouped(vault_items: list[dict]) -> None:
    """Display inbox items grouped by source (D-10).

    Gmail items show subject + sender + date.
    Feed items show title + source + published date.
    """
    gmail_items = [i for i in vault_items if i.get("_source") == "gmail"]
    feed_items = [i for i in vault_items if i.get("_source") == "feeds"]

    item_number = 1

    if gmail_items:
        typer.echo(f"\n--- Gmail ({len(gmail_items)}) ---")
        for item in gmail_items:
            subject = item.get("subject", "untitled")[:50]
            sender = item.get("sender", "unknown")
            date_str = item.get("date", "")
            typer.echo(f"  {item_number}. {subject}  [{sender}]  {date_str}")
            item_number += 1

    if feed_items:
        typer.echo(f"\n--- Feeds ({len(feed_items)}) ---")
        for item in feed_items:
            title = item.get("title", "untitled")[:50]
            source = item.get("source", "unknown")
            date_str = item.get("published", item.get("ingested", ""))
            typer.echo(f"  {item_number}. {title}  [{source}]  {date_str}")
            item_number += 1


def _pick_type() -> Optional[str]:
    """Present type options via interactive picker. Returns type name or None."""
    from pb.cli.helpers import _interactive_pick

    result = _interactive_pick(TYPE_OPTIONS, "Convert to type", multi=False)
    if result is None:
        return None
    return TYPE_OPTIONS[result[0]]


def _launch_conversion(item: dict, target_type: str, vault_path: Path) -> bool:
    """Convert an inbox item to a typed note via question tree (D-11, D-12).

    - "Skip": returns True without converting
    - "Task": creates SQLite task from item data, deletes vault note
    - Others: loads schema, creates QuestionTreeEngine, pre-fills fields,
      runs interactive question tree, deletes vault note on success

    Returns True on success, False on cancel/error.
    """
    item_path: Optional[Path] = item.get("_path")

    if target_type == "Skip":
        return True

    if target_type == "Task":
        # Create a SQLite task from the item
        title = item.get("title") or item.get("subject", "Untitled inbox item")
        repo = Repository()
        from pb.domain.models import Task, generate_slug

        task = Task(
            id=generate_slug(title),
            title=title,
            state=TaskState.ACTIVE,
        )
        repo.create_task(task)
        typer.echo(f"  Created task: {stored_display_title(task) or title}")

        # Delete original vault note (D-12)
        if item_path is not None:
            item_path.unlink(missing_ok=True)
        return True

    # Type conversion via question tree
    try:
        from pb.core.schemas import load_schema, ensure_default_schemas
        from pb.core.question_tree import QuestionTreeEngine

        ensure_default_schemas()
        schema_id = target_type.lower()
        schema = load_schema(schema_id)
        engine = QuestionTreeEngine(schema)

        # Pre-fill fields from inbox item data (D-11)
        # Only pre-fill fields that exist in the schema
        schema_field_names = {f.name for f in schema.fields}

        prefill_map = {
            "title": item.get("title") or item.get("subject", ""),
            "url": item.get("url", ""),
            "source_url": item.get("url", ""),
            "notes": item.get("snippet", ""),
        }

        for field_name, value in prefill_map.items():
            if value and field_name in schema_field_names:
                engine.values[field_name] = value

        typer.echo(f"\n-- Convert to {schema.name} --")
        typer.echo("Commands: /skip (skip optional), /done (finish early)\n")

        # Show pre-filled fields
        for field_name, value in engine.values.items():
            if value:
                label = field_name.replace("_", " ").title()
                typer.echo(f"  Pre-filled {label}: {value[:60]}")

        # Run required fields (skip already pre-filled ones)
        while not engine.is_done:
            field = engine.current_field()
            if field is None:
                break

            # Skip if already pre-filled
            if field.name in engine.values and engine.values[field.name]:
                engine.skip_current_field()
                if engine.current_field() is None:
                    break
                continue

            progress = engine.progress_text()
            prompt_text = f"{progress} {field.prompt}"

            try:
                if field.field_type == "select" and field.options:
                    from pb.cli.helpers import _interactive_pick

                    typer.echo(f"\n{prompt_text}:")
                    result = _interactive_pick(field.options, field.prompt, multi=False)
                    if result is not None:
                        resp = engine.process_input(field.options[result[0]])
                    else:
                        typer.echo("\nCancelled.")
                        return False
                else:
                    raw = typer.prompt(prompt_text, default="", show_default=False)
                    resp = engine.process_input(raw)
                    if resp["action"] == "cancelled":
                        typer.echo("\nCancelled.")
                        return False
                    elif resp["action"] == "error":
                        typer.echo(f"  ! {resp.get('message', '')}")
                        continue
                    elif resp["action"] in ("done", "next_phase"):
                        break
            except (typer.Abort, EOFError, KeyboardInterrupt):
                typer.echo("\nCancelled.")
                return False

            if engine._phase == "optional_select" or engine.is_done:
                break

        # Skip optional fields for inbox conversion -- just write with pre-fills
        if not engine.is_done and engine._phase == "optional_select":
            engine.set_optional_selections([])

        # Generate note content and write to vault
        note_content = engine.generate_note_content()
        vault_rel_path = engine.get_vault_path()
        dest = vault_path / vault_rel_path
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(note_content)

        typer.echo(f"  Note created: {vault_rel_path}")

        # Delete original vault note (D-12)
        if item_path is not None:
            item_path.unlink(missing_ok=True)

        return True

    except FileNotFoundError as e:
        typer.echo(f"  Schema not found: {e}", err=True)
        return False
    except Exception as e:
        typer.echo(f"  Conversion error: {e}", err=True)
        return False


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


@app.callback(invoke_without_command=True)
def list_inbox(
    ctx: typer.Context,
    convert: bool = typer.Option(False, "--convert", "-c", help="Interactive type conversion mode"),
):
    """List vault inbox items from feeds and gmail (D-09)."""
    if ctx.invoked_subcommand is not None:
        return

    from pb.vault import get_vault_path

    vault_path = get_vault_path()
    vault_items = _scan_vault_inbox(vault_path)

    if not vault_items:
        typer.echo("Inbox is empty.")
        return

    _display_grouped(vault_items)

    typer.echo(f"\nTotal: {len(vault_items)} items")

    if convert and vault_items:
        typer.echo("\nConversion mode -- processing vault items:")
        for item in vault_items:
            title = item.get("title") or item.get("subject", "untitled")
            typer.echo(f"\nItem: {title[:60]}")
            selected_type = _pick_type()
            if selected_type is None:
                continue
            if selected_type == "Skip":
                typer.echo("  Skipped.")
                continue
            success = _launch_conversion(item, selected_type, vault_path)
            if success:
                typer.echo(f"  Converted to {selected_type}.")
