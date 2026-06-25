# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""pb anki -- Manage and export Anki cards (ANKI-04)."""
from __future__ import annotations

import sys
import shutil
from pathlib import Path
from typing import Optional

import typer

from pb.cli.console import get_console
from pb.cli.markdown import render_markdown
from pb.storage.yaml_io import load_yaml_file, write_yaml_file

app = typer.Typer(
    name="anki",
    help="Manage and export Anki cards",
    no_args_is_help=True,
)

# ---------------------------------------------------------------------------
# Domain / deck mapping tables
# ---------------------------------------------------------------------------

_DOMAIN_TO_DECK: dict[str, str] = {
    "deutsch": "German",
    "piano": "Piano",
    "ml": "Machine Learning",
    "math": "Mathematics",
    "communication": "Communication",
}
_DECK_TO_DOMAIN: dict[str, str] = {v: k for k, v in _DOMAIN_TO_DECK.items()}
_DEFAULT_NOTE_TYPES = [
    "Basic",
    "Cloze",
    "Fill in the blanks",
    "Basic (and reversed card)",
]


def _domain_to_deck(domain_name: str) -> str:
    """Map domain folder name to Anki parent deck name (D-26)."""
    return _DOMAIN_TO_DECK.get(domain_name, domain_name.title())


def _deck_to_domain(deck_name: str) -> str:
    """Map Anki deck name back to domain folder name."""
    leaf = deck_name.split("::")[-1].strip()
    return _DECK_TO_DOMAIN.get(leaf, leaf.lower())


def _pick_note_types(console, selected: Optional[list[str]] = None) -> list[str]:
    """Prompt for one or more note types, with a non-TTY fallback."""
    if selected:
        return selected

    try:
        from prompt_toolkit.shortcuts import checkboxlist_dialog

        result = checkboxlist_dialog(
            title="Note types",
            text="Choose one or more note types for generation",
            values=[(value, value) for value in _DEFAULT_NOTE_TYPES],
        ).run()
        if result:
            return list(result)
    except Exception:
        pass

    console.print(
        "[dim]Select note types (comma-separated). Defaults: Basic, Cloze, Fill in the blanks[/]"
    )
    import typer
    try:
        raw = typer.prompt(">", default="", show_default=False).strip()
    except (typer.Abort, EOFError, KeyboardInterrupt):
        raw = ""
    choices = [part.strip() for part in raw.split(",") if part.strip()]
    return choices or ["Basic", "Cloze", "Fill in the blanks"]


def _default_yaml_output(deck_name: str, suffix: str) -> Path:
    slug = deck_name.lower().replace(" ", "-").replace("/", "-")
    return Path(f"{slug}-{suffix}.yaml")


def _resolve_deck_and_domain(
    deck_name: Optional[str],
    domain_name: Optional[str],
) -> tuple[str, str]:
    """Fill missing deck/domain values from the shared mapping tables."""
    effective_deck = deck_name or (_domain_to_deck(domain_name) if domain_name else "")
    effective_domain = domain_name or (_deck_to_domain(deck_name) if deck_name else "")
    return effective_deck, effective_domain


def _load_yaml_rows(path: Path) -> tuple[list[dict], dict]:
    """Load deck import rows from either a bare YAML list or a metadata wrapper."""
    data = load_yaml_file(path, [])
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)], {}
    if isinstance(data, dict):
        rows = data.get("rows", [])
        if isinstance(rows, list):
            return [row for row in rows if isinstance(row, dict)], data
    return [], {}


def _first_card_with_status(status: str) -> dict | None:
    """Return the oldest card in a logical status bucket."""
    from pb.vault.anki_client import get_cards_by_status

    cards = get_cards_by_status(status)
    return cards[0] if cards else None


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------

@app.command("generate")
def generate_cards(
    ctx: typer.Context,
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain to generate for"),
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck name (defaults from --domain)"),
    note_type: Optional[list[str]] = typer.Option(None, "--note-type", "-t", help="Repeatable note type filter"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Gemini model or alias: flash-lite, flash, pro"),
    emul: bool = typer.Option(False, "--emul", "-e", help="Emulate an existing deck style using current deck samples"),
    term: Optional[list[str]] = typer.Argument(None, help="Optional specific term (no quotes needed)"),
):
    """Generate Anki cards from vault notes, or a specific term (ANKI-02, D-04)."""
    from pb.llm.gemini import FLASH_MODEL, resolve_model

    console = get_console()

    term_str = " ".join(term) if term else None
    deck, effective_domain = _resolve_deck_and_domain(deck, domain)
    selected_note_types = list(note_type or [])
    model_label = resolve_model(model, fallback=FLASH_MODEL) if model else FLASH_MODEL

    # Tier-2 confirmation (D-06 tier-2 rules)
    from pb.cli.pickers import pick_boolean
    target_desc = f'"{term_str}"' if term_str else f"all notes in {effective_domain or 'vault'}"
    if not pick_boolean(title=f"Generate cards for {target_desc}?"):
        console.print("[dim]Cancelled.[/]")
        return

    note_type_suffix = f" | note types: {', '.join(selected_note_types)}" if selected_note_types else ""
    console.print(
        f"[dim]Generating... ({model_label})"
        f"{' | emulating deck style' if emul else ''}"
        f"{note_type_suffix}[/]"
    )
    try:
        anki_service = ctx.obj['factory']['anki_service']()
        # For term-targeted generation, use term as note_slug
        note_slug = term_str or effective_domain or "vault"
        note_content = term_str or ""
        result = anki_service.generate_cards(
            note_slug=note_slug,
            note_content=note_content,
            domain=effective_domain,
            deck=deck,
            term=term_str,
            source="term" if term_str else "auto",
            note_types=selected_note_types or None,
            model=model,
            emulate_existing_deck=emul,
        )
        cards = result.get("cards", [])
        run_id = result.get("run_id", "")
        count = result.get("count", 0)
        console.print(f"\n[success]Generated {count} cards (run-id: {run_id})[/success]\n")
        for i, c in enumerate(cards, 1):
            card_type = c.get("card_type", "Basic")
            front_preview = c.get("front", "")[:60]
            console.print(f"  [value.low]o[/] [dim]{i}[/] {front_preview}  [dim][{card_type}][/]")
        if cards:
            console.print("\n[dim]Review with 'pb anki list --suggested' or export with 'pb anki export'[/]")
    except Exception as e:
        console.print(f"[error]Card generation failed: {e}[/error]")
        console.print("[dim]Check GEMINI_API_KEY and try again.[/]")


@app.command("list")
def list_cards(
    ctx: typer.Context,
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain"),
    status: Optional[str] = typer.Option(
        None,
        "--status",
        "-s",
        help="Filter by status: suggested/accepted/edited/exportable/exported/rejected",
    ),
    suggested: Optional[str] = typer.Option(
        None, "--suggested",
        help="Suggested review: omit value (30 random), N (count), or 'all'",
    ),
):
    """List Anki cards with interactive TUI (ANKI-04, D-25). Use --suggested for card-by-card review."""
    console = get_console()

    if suggested is not None:
        # Treat bare --suggested flag (no value given) as empty string = default 30
        _run_suggested_review(ctx, suggested_arg=suggested or "")
        return

    # Default: existing TUI checkboxlist
    from pb.vault.anki_client import get_cards_by_status

    effective_status = status or "suggested"
    cards = get_cards_by_status(effective_status, domain)
    if not cards:
        console.print(f"No cards with status '{effective_status}'. Run 'pb anki generate' to create cards.")
        return

    domain_label = domain or "all"
    console.rule(f"[header]Anki Cards -- {domain_label} ({len(cards)} {effective_status})[/]")

    if not sys.stdin.isatty():
        # Non-interactive fallback: print table
        for card in cards:
            console.print(f"  {card['note_slug']}: {card['front'][:50]}... [{card['card_type']}]")
        return

    # Interactive TUI with arrow navigation (D-25, Pattern C)
    _run_card_tui(cards, console, domain)


@app.command("pending")
def pending_cards(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain"),
) -> None:
    """Show suggested candidates and export-ready counts."""
    console = get_console()
    from pb.vault.anki_client import get_card_status_counts, get_cards_by_status, get_pending_card_count

    suggested = get_cards_by_status("suggested", domain)
    counts = get_card_status_counts(domain)
    export_ready = get_pending_card_count(domain)

    console.rule("[header]Anki Pending[/]")
    console.print(f"Suggested: {len(suggested)}")
    console.print(f"Export ready: {export_ready}")
    if counts:
        summary = ", ".join(f"{status}={count}" for status, count in sorted(counts.items()))
        console.print(f"[dim]{summary}[/]")
    for card in suggested[:10]:
        console.print(f"  {card['id']}: {card['front'][:70]}")
    if len(suggested) > 10:
        console.print(f"[dim]... and {len(suggested) - 10} more suggested cards[/]")


@app.command("review")
def review_cards(
    ctx: typer.Context,
    suggested: str = typer.Option("", "--suggested", help="Review count or 'all'"),
) -> None:
    """Run the suggested-card review flow."""
    _run_suggested_review(ctx, suggested_arg=suggested or "")


@app.command("accept")
def accept_card(card_id: str = typer.Argument("", help="Candidate card id; omit to accept the first suggested card")) -> None:
    """Accept one suggested candidate without opening the review TUI."""
    console = get_console()
    from pb.vault.anki_client import ACCEPTED_STATUS, get_card_by_id, update_card_status

    card = get_card_by_id(card_id) if card_id else _first_card_with_status("suggested")
    if card is None:
        console.print(f"[error]Card not found: {card_id or 'first suggested card'}[/error]")
        console.print("[dim]Run `pb anki pending` to see available candidates.[/]")
        raise typer.Exit(code=1)
    card_id = str(card["id"])
    update_card_status(card_id, ACCEPTED_STATUS)
    console.print(f"[success]Accepted {card_id}.[/success]")


@app.command("reject")
def reject_card(card_id: str = typer.Argument("", help="Candidate card id; omit to reject the first suggested card")) -> None:
    """Reject one suggested candidate without opening the review TUI."""
    console = get_console()
    from pb.vault.anki_client import REJECTED_STATUS, get_card_by_id, update_card_status

    card = get_card_by_id(card_id) if card_id else _first_card_with_status("suggested")
    if card is None:
        console.print(f"[error]Card not found: {card_id or 'first suggested card'}[/error]")
        console.print("[dim]Run `pb anki pending` to see available candidates.[/]")
        raise typer.Exit(code=1)
    card_id = str(card["id"])
    update_card_status(card_id, REJECTED_STATUS)
    console.print(f"[success]Rejected {card_id}.[/success]")


@app.command("export")
def export_selected(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Export cards for domain"),
    deck: Optional[str] = typer.Option(None, "--deck", help="Limit export to one exact deck name"),
    csv_only: bool = typer.Option(False, "--csv", help="Force CSV export (skip AnkiConnect)"),
):
    """Export accepted or edited cards to .apkg, optionally sync to Anki, or force CSV."""
    console = get_console()

    from pb.vault.anki_client import get_cards_by_status

    cards = get_cards_by_status("exportable", domain)
    if deck:
        cards = [card for card in cards if str(card.get("deck", "") or "") == deck]
    if not cards:
        console.print("No accepted or edited cards to export.")
        return

    # ANKI-04 review loop: confirm card count before export (CLAUDE.md explicit review loop)
    from pb.cli.pickers import pick_boolean
    console.print(
        f"[value.high]{len(cards)}[/] accepted/edited cards to export"
        f"{f' (domain: {domain})' if domain else ''}:"
    )
    for card in cards[:5]:
        console.print(f"  {card['note_slug'][:40]}  [dim]{card['card_type']}  {card.get('deck', '')}[/]")
    if len(cards) > 5:
        console.print(f"  [dim]... and {len(cards) - 5} more[/]")
    if not pick_boolean(title=f"Export {len(cards)} cards?"):
        console.print("[dim]Export cancelled.[/]")
        return

    _do_export(cards, console, domain, csv_only=csv_only)


@app.command("revlog", hidden=True)
def revlog(
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Filter by domain"),
):
    """Sync and display Anki review log (ANKI-04)."""
    console = get_console()

    try:
        from pb.vault.anki_client import sync_revlog, is_anki_available
        if not is_anki_available():
            console.print("[warn]AnkiConnect not running. Cannot sync revlog.[/warn]")
            console.print("[dim]Start Anki with AnkiConnect add-on enabled and retry.[/]")
            return
        entries = sync_revlog()
        if not entries:
            console.print("[dim]No revlog entries found.[/]")
            return
        console.rule("[header]Anki Review Log[/]")
        for entry in entries[:20]:
            console.print(f"  [dim]{entry.get('card_id', '')}[/]  {entry.get('note_slug', '')}  [dim]{entry.get('reviewed_at', '')}[/]")
        if len(entries) > 20:
            console.print(f"  [dim]... and {len(entries) - 20} more[/]")
    except Exception as e:
        console.print(f"[error]Revlog sync failed: {e}[/error]")


@app.command("history", hidden=True)
def anki_history(
    ctx: typer.Context,
    limit: int = typer.Option(20, "--limit", "-n", help="Max rows to show"),
):
    """Show Anki card generation history (D-04, D-08)."""
    from rich.table import Table
    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()
    rows = anki_service.get_history(limit=limit)
    console.rule("[header]Anki Generation History[/]")
    if not rows:
        console.print("No generation runs found.")
        console.print("[dim]Run 'pb anki generate' to create your first batch.[/]")
        return
    table = Table(show_header=True, show_edge=False, pad_edge=False, box=None)
    table.add_column("RUN-ID", style="dim", no_wrap=True)
    table.add_column("TIMESTAMP", style="dim", no_wrap=True)
    table.add_column("NOTE/TERM", max_width=22)
    table.add_column("COUNT", justify="right")
    table.add_column("SOURCE", style="dim")
    for r in rows:
        run_id_short = str(r.get("run_id", ""))[:6]
        ts = str(r.get("created_at", ""))[:19]
        note_term = str(r.get("term") or r.get("note_slug") or "")[:22]
        if r.get("source") == "socratic":
            note_term = "[socratic debrief]"
        count_str = str(r.get("card_count", 0))
        source = str(r.get("source", "auto"))
        table.add_row(run_id_short, ts, note_term, count_str, source)
    console.print()
    console.print(table)
    console.print()
    console.print("[dim]Run 'pb anki rollback <run-id>' to undo a generation run.[/]")


@app.command("rollback", hidden=True)
def anki_rollback(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run ID to undo (from 'pb anki history')"),
):
    """Undo a generation run: delete cards from pb.db and AnkiConnect (D-04, D-08)."""
    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()

    # Lookup run info before confirming
    history = anki_service.get_history(limit=100)
    run_info = next((r for r in history if str(r.get("run_id", "")).startswith(run_id)), None)
    if not run_info:
        console.print(f"[error]Run '{run_id}' not found in generation history.[/error]")
        console.print("[dim]Run 'pb anki history' to see available run-ids.[/]")
        return

    console.rule(f"[header]Rollback: run {run_id}[/]")
    console.print()
    console.print(
        f"[warn]This will permanently delete {run_info.get('card_count', 0)} cards from pb.db AND call AnkiConnect deleteNotes.[/warn]"
    )
    console.print(f"[dim]  Run:    {run_id}[/]")
    console.print(f"[dim]  Source: {run_info.get('source', 'auto')} ({run_info.get('note_slug', '')})[/]")
    console.print(f"[dim]  Cards:  {run_info.get('card_count', 0)}[/]")
    console.print(f"[dim]  Date:   {str(run_info.get('created_at', ''))[:19]}[/]")
    console.print()
    console.print("[value.high]This cannot be undone. CSV exports are NOT reversed.[/value.high]")
    console.print()
    from pb.cli.pickers import pick_boolean
    if not pick_boolean(title="Confirm rollback?"):
        console.print("[dim]Cancelled.[/]")
        return

    ok, msg = anki_service.rollback_run(run_id)
    if ok:
        console.print(f"[success]{msg}[/success]")
    else:
        console.print(f"[error]{msg}[/error]")


@app.command("init-format", hidden=True)
def anki_init_format(
    ctx: typer.Context,
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck name"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain name (maps to deck automatically)"),
    note_type: Optional[list[str]] = typer.Option(None, "--note-type", "-t", help="Repeatable note type selection"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Gemini model or alias: flash-lite, flash, pro"),
    emul: bool = typer.Option(False, "--emul", "-e", help="Capture live deck samples for later emulation"),
    copy_from: Optional[str] = typer.Option(None, "--copy-from", help="Copy another deck's local format.yaml first"),
    view_context: bool = typer.Option(False, "--view-context", help="Show deck context.md"),
):
    """Create or refresh a deck format spec in YAML."""
    from pb.llm.gemini import FLASH_MODEL, resolve_model

    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()
    deck, domain = _resolve_deck_and_domain(deck, domain)

    # Deck name prompt
    console.rule("[header]Deck Format Setup[/]")
    console.print("[dim]Creates vault/pb-anki/<DeckName>/format.yaml[/]")

    if not deck:
        from pb.cli.pickers import pick_deck
        deck = pick_deck(title="Deck Name")
        if not deck:
            console.print("\n[dim]Cancelled.[/]")
            return
    if not deck:
        console.print("[error]Deck name is required.[/error]")
        return
    if not domain:
        domain = _deck_to_domain(deck)

    # View context flag
    if view_context:
        ctx_text = anki_service.load_context_md(deck)
        console.rule(f"[header]Deck Context: {deck}[/]")
        if ctx_text.strip():
            render_markdown(ctx_text)
        else:
            console.print("[dim]No context yet.[/]")
        console.print(f"[dim]Edit: vault/pb-anki/{deck}/context.md[/]")
        return

    existing = anki_service.load_format_spec(deck)
    selected_note_types = _pick_note_types(
        console,
        note_type or existing.get("note_types"),
    )

    if copy_from:
        source = anki_service.load_format_spec(copy_from)
        if not source:
            console.print(f"[error]No local format found for '{copy_from}'.[/error]")
            return
        data = dict(source)
        data["deck"] = deck
        data["domain"] = domain
        data["note_types"] = selected_note_types
        if list(source.get("note_types", [])) != selected_note_types:
            data.pop("field_map", None)
        if emul:
            data["emulate_existing_deck"] = True
            data["emulated_samples"] = anki_service.gather_emulation_samples(
                deck,
                selected_note_types,
            )
        else:
            data["emulate_existing_deck"] = False
            data.pop("emulated_samples", None)
        data["llm_model"] = resolve_model(model or data.get("llm_model"), fallback=FLASH_MODEL)
        path = anki_service.save_format_spec(deck, data)
        console.print(f"[success]Copied format from {copy_from} -> {deck}[/success]")
        console.print(f"[dim]{path}[/]")
        return

    sample_rows = (
        anki_service.gather_emulation_samples(deck, selected_note_types)
        if emul
        else []
    )
    data = anki_service.draft_format_spec(
        deck,
        domain,
        selected_note_types,
        model=model,
        emulate_existing_deck=emul,
        sample_rows=sample_rows,
    )
    path = anki_service.save_format_spec(deck, data)
    console.print(f"[success]format.yaml written for {deck}[/success]")
    console.print(f"[dim]{path}[/]")
    console.print(f"[dim]Note types: {', '.join(selected_note_types)}[/]")
    console.print(
        f"[dim]Model: {resolve_model(model or data.get('llm_model'), fallback=FLASH_MODEL)}"
        f"{' | emulation samples captured' if sample_rows else ''}[/]"
    )
    if emul and not sample_rows:
        console.print("[warn]No existing deck samples were found, so emulation will stay sample-free until export/import data exists.[/warn]")


@app.command("diagnostic", hidden=True)
def anki_diagnostic(
    ctx: typer.Context,
    topic: Optional[list[str]] = typer.Argument(None, help="Optional topic or concept to probe"),
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck name"),
    domain: Optional[str] = typer.Option(None, "--domain", "-d", help="Domain name (maps to deck automatically)"),
    note_type: Optional[list[str]] = typer.Option(None, "--note-type", "-t", help="Repeatable note type selection"),
    model: Optional[str] = typer.Option(None, "--model", "-m", help="Gemini model or alias: flash-lite, flash, pro"),
    limit: Optional[str] = typer.Option(None, "--limit", "-l", help="Soft difficulty ceiling, e.g. Undergrad or First year PhD"),
    start: Optional[str] = typer.Option(None, "--start", "-s", help="Initial difficulty, e.g. foundational basics"),
    time_limit: int = typer.Option(10, "--time-limit", min=3, max=15, help="Hard time limit in minutes (3-15)"),
    max_rounds: int = typer.Option(30, "--max-rounds", help="Hard ceiling on exchanges"),
    soft_cap: int = typer.Option(24, "--soft-cap", help="Soft target before the interviewer starts closing gaps"),
):
    """Run a strict adaptive Socratic diagnostic and save downstream YAML gaps."""
    from pb.llm.gemini import FLASH_MODEL, resolve_model

    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()
    socratic_service = ctx.obj['factory']['socratic_service']()
    deck, domain = _resolve_deck_and_domain(deck, domain)
    topic_str = " ".join(topic) if topic else ""

    if not deck:
        from pb.cli.pickers import pick_deck
        deck = pick_deck(title="Deck Name")
        if not deck:
            console.print("\n[dim]Cancelled.[/]")
            return
    if not deck:
        console.print("[error]Deck name is required.[/error]")
        return
    if not domain:
        domain = _deck_to_domain(deck)

    existing = anki_service.load_format_spec(deck)
    selected_note_types = _pick_note_types(
        console,
        note_type or existing.get("note_types"),
    )
    model_label = resolve_model(model, fallback=FLASH_MODEL)
    console.rule(f"[header]Anki Diagnostic -- {deck}[/]")
    console.print(f"[dim]Domain:[/] {domain}")
    console.print(f"[dim]Note types:[/] {', '.join(selected_note_types)}")
    console.print(f"[dim]Model:[/] {model_label}")
    console.print(f"[dim]Start difficulty:[/] {start or 'foundational basics'}")
    console.print(f"[dim]Difficulty limit:[/] {limit or 'unbounded'}")
    console.print(f"[dim]Time limit:[/] {time_limit} min")
    if topic_str:
        console.print(f"[dim]Topic:[/] {topic_str}")
    console.print()

    qa_pairs = socratic_service.run_adaptive_diagnostic(
        domain=domain,
        console=console,
        topic=topic_str,
        difficulty_start=start or "foundational basics",
        difficulty_limit=limit or "",
        max_rounds=max_rounds,
        soft_cap_rounds=soft_cap,
        model=model,
        time_limit_minutes=time_limit,
    )
    if not qa_pairs:
        console.print("[warn]No diagnostic transcript captured.[/]")
        return

    report = socratic_service.build_diagnostic_report(
        qa_pairs,
        domain,
        topic=topic_str,
        difficulty_start=start or "foundational basics",
        difficulty_limit=limit or "",
        note_types=selected_note_types,
        model=model,
    )
    report["deck"] = deck
    report["domain"] = domain
    report["note_types"] = selected_note_types
    report["rounds"] = len(qa_pairs)
    path = anki_service.save_diagnostic_report(deck, report)
    transcript_note = socratic_service.cache_diagnostic_transcript(
        qa_pairs=qa_pairs,
        domain=domain,
        topic=topic_str,
    )
    gap_count = len(report.get("knowledge_gaps") or [])
    console.print()
    console.print(f"[success]Diagnostic saved to {path}[/success]")
    if transcript_note:
        console.print(f"[dim]Transcript cached in vault:[/] {transcript_note}")
    console.print(f"[dim]Knowledge gaps captured: {gap_count}[/]")
    if report.get("summary"):
        console.print(f"[dim]{report['summary']}[/]")


@app.command("deck-export", hidden=True)
def anki_deck_export(
    ctx: typer.Context,
    deck: Optional[str] = typer.Option(None, "--deck", help="Deck name to export"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Raw Anki query instead of --deck"),
    note_type: Optional[str] = typer.Option(None, "--note-type", "-t", help="Restrict export to one note type"),
    limit: Optional[int] = typer.Option(None, "--limit", "-n", help="Max notes to export"),
    output: Optional[Path] = typer.Option(None, "--output", "-o", help="YAML output path"),
):
    """Export existing Anki notes or local PB export-ready cards."""
    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()

    if not deck and not query:
        from pb.vault import get_vault_path
        from pb.vault.anki_client import export_cards_to_apkg, export_cards_to_csv, get_cards_by_status

        cards = get_cards_by_status("exportable")
        if not cards:
            console.print("[error]No accepted or edited PB cards to export.[/error]")
            console.print("[dim]Run `pb anki pending`, then `pb anki accept <card-id>` before exporting.[/]")
            raise typer.Exit(code=1)
        output_path = output or _default_yaml_output("pb-cards", "export")
        if output_path.suffix.lower() == ".apkg":
            packaged, package_path, package_msg = export_cards_to_apkg(cards, get_vault_path())
            if not packaged or package_path is None:
                console.print(f"[error]{package_msg}[/error]")
                raise typer.Exit(code=1)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if package_path.resolve() != output_path.resolve():
                shutil.copy2(package_path, output_path)
            console.print(f"[success]Exported {len(cards)} PB cards to {output_path}[/success]")
            return
        csv_path = export_cards_to_csv(cards, get_vault_path())
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if csv_path.resolve() != output_path.resolve():
            shutil.copy2(csv_path, output_path)
        console.print(f"[success]Exported {len(cards)} PB cards to {output_path}[/success]")
        return

    output_path = output or _default_yaml_output(deck or "anki-query", "export")
    try:
        rows, inferred_note_type = anki_service.export_existing_notes(
            deck or "",
            note_type=note_type,
            query=query,
            limit=limit,
        )
    except Exception as exc:
        console.print(f"[error]Deck export failed: {exc}[/error]")
        return

    payload = {
        "deck": deck,
        "query": query,
        "note_type": note_type or inferred_note_type,
        "row_count": len(rows),
        "rows": rows,
    }
    write_yaml_file(output_path, payload)
    console.print(f"[success]Exported {len(rows)} notes to {output_path}[/success]")


@app.command("deck-import", hidden=True)
def anki_deck_import(
    ctx: typer.Context,
    input_path: Path = typer.Argument(..., exists=True, readable=True, help="YAML file from deck-export or a compatible row list"),
    deck: str = typer.Option("", "--deck", help="Deck name to import into"),
    note_type: Optional[str] = typer.Option(None, "--note-type", "-t", help="Anki note type to use"),
    key_field: Optional[str] = typer.Option(None, "--key-field", "-k", help="Field used to match existing notes"),
):
    """Import or update Anki notes from YAML rows."""
    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()
    if input_path.suffix.lower() == ".apkg":
        console.print(f"[success]APKG package is ready for Anki import: {input_path}[/success]")
        console.print("[dim]Open it with Anki, or use the normal `pb anki export` flow for local PB cards.[/]")
        return
    if not deck:
        console.print("[error]Pass --deck when importing YAML rows.[/error]")
        raise typer.Exit(code=1)

    rows, meta = _load_yaml_rows(input_path)
    if not rows:
        console.print("[error]No rows found in the YAML input.[/error]")
        return

    try:
        result = anki_service.import_existing_notes(
            rows,
            deck,
            note_type=note_type or meta.get("note_type"),
            key_field=key_field or meta.get("key_field"),
        )
    except Exception as exc:
        console.print(f"[error]Deck import failed: {exc}[/error]")
        return

    console.print(
        f"[success]Imported into {result.get('deck', deck)}[/success]\n"
        f"[dim]Added: {result.get('added', 0)} | Updated: {result.get('updated', 0)} | "
        f"Note type: {result.get('note_type', note_type or '')}[/]"
    )


# ---------------------------------------------------------------------------
# Private helpers — card review TUI
# ---------------------------------------------------------------------------

def _run_card_tui(cards: list[dict], console, domain: Optional[str]) -> None:
    """Interactive card review TUI (D-25, 18-UI-SPEC Pattern C)."""
    console.print("[dim]up/down navigate  Space to toggle  Enter confirm  x export selected  q quit[/]")
    console.print()

    # Use prompt_toolkit checkboxlist if available
    try:
        from prompt_toolkit.shortcuts import checkboxlist_dialog
        from prompt_toolkit.key_binding import KeyBindings

        bindings = KeyBindings()

        @bindings.add("x")
        def _export(event):
            event.app.exit(result="export")

        @bindings.add("q")
        def _quit(event):
            event.app.exit(result="quit")

        values = [
            (
                c["id"],
                f"{c['note_slug'][:40]}  [{c['card_type']}  {c['deck'].split('::')[-1] if '::' in c.get('deck', '') else c.get('deck', '')}]",
            )
            for c in cards
        ]
        result = checkboxlist_dialog(
            title=f"Anki Cards ({len(cards)} {cards[0]['status'] if cards else 'suggested'})",
            text="Space to toggle, Enter to confirm selection, x to export",
            values=values,
            key_bindings=bindings,
        ).run()

        if result == "quit" or result is None:
            return
        if result == "export" or (isinstance(result, list) and result):
            selected = result if isinstance(result, list) else []
            if selected:
                selected_cards = [c for c in cards if c["id"] in selected]
                _do_export(selected_cards, console, domain)
    except (ImportError, Exception):
        # Fallback to numbered list
        _run_card_tui_fallback(cards, console, domain)


def _run_card_tui_fallback(cards: list[dict], console, domain: Optional[str]) -> None:
    """Numbered list fallback for non-prompt_toolkit environments."""
    for i, card in enumerate(cards, 1):
        console.print(f"  [value.low]o[/] [dim]{i}[/] {card['note_slug'][:40]}  [dim]{card['card_type']}  {card.get('deck', '')}[/]")

    # Show preview of first card
    if cards:
        console.print()
        console.print("[subheader]Preview[/]")
        console.print(f"Front: {cards[0]['front'][:80]}")
        console.print(f"Back:  {cards[0]['back'][:80]}")
        console.print(f"Deck:  {cards[0].get('deck', '')}")

    console.print()
    console.print("[dim]Enter card numbers (comma-separated) to select, x to export all, q to quit[/]")
    import typer
    try:
        selection = typer.prompt(">", default="", show_default=False).strip()
        if selection.lower() == "q":
            return
        if selection.lower() == "x":
            _do_export(cards, console, domain)
            return
        indices = [int(x.strip()) - 1 for x in selection.split(",") if x.strip().isdigit()]
        selected = [cards[i] for i in indices if 0 <= i < len(cards)]
        if selected:
            _do_export(selected, console, domain)
    except (typer.Abort, EOFError, KeyboardInterrupt):
        pass


def _run_suggested_review(ctx: typer.Context, suggested_arg: str = "") -> None:
    """Card-by-card suggested review flow: a/e/s/q keys. D-06, Screen 2 of UI-SPEC."""
    console = get_console()
    anki_service = ctx.obj['factory']['anki_service']()
    from pb.vault.anki_client import update_card, update_card_status

    # Determine batch size
    if suggested_arg.lower() == "all":
        batch_size = 9999
    elif suggested_arg.isdigit():
        batch_size = int(suggested_arg)
    else:
        batch_size = 30  # D-06 default

    cards = anki_service.get_suggested_cards(batch_size=batch_size)
    # D-05/D-06: thin swappable mastery adapter — reorder so weak-skill cards surface first.
    from pb.core.anki_mastery_adapter import reorder_by_mastery
    cards = reorder_by_mastery(cards, repo=ctx.obj["repo"])
    if not cards:
        console.print("No suggested cards. Run 'pb anki generate' to create cards.")
        return

    total = len(cards)
    # Get run_id for header
    run_id_short = str(cards[0].get("run_id", ""))[:6] if cards else ""
    console.rule(f"[header]Suggested Review — {total} cards (run {run_id_short})[/]")

    accepted: list[dict] = []
    edited: list[dict] = []
    skipped: list[dict] = []

    for i, card in enumerate(cards, 1):
        console.print(f"\n[dim]Card {i} of {total}[/dim]")
        console.print()
        console.print("[subheader]FRONT[/subheader]")
        render_markdown(card.get("front", ""))
        console.print()
        console.print("[subheader]BACK[/subheader]")
        render_markdown(card.get("back", ""))
        console.print()
        console.print(f"[dim]Deck:[/dim]  {card.get('deck', '')}")
        console.print(f"[dim]Note:[/dim]  {card.get('note_slug', '')}")
        console.print(f"[dim]Type:[/dim]  {card.get('card_type', 'Basic')}")
        console.print()
        
        from pb.cli.pickers import pick_single_choice
        options = [
            ("a", "Accept"),
            ("e", "Edit inline"),
            ("s", "Skip"),
            ("q", "Quit")
        ]
        key = pick_single_choice(options, title="Action") or "q"

        if key == "a":
            update_card_status(card["id"], "accepted")
            console.print("[success]Accepted.[/]")
            accepted.append(card)
        elif key == "e":
            # Inline edit flow (UI-SPEC Screen 2)
            console.print()
            console.print("[subheader]Edit FRONT[/subheader] [dim](Enter to keep current)[/dim]:")
            console.print(card.get("front", ""))
            import typer
            try:
                new_front = typer.prompt(">", default="", show_default=False).strip()
            except (typer.Abort, EOFError, KeyboardInterrupt):
                new_front = ""
            console.print("[subheader]Edit BACK[/subheader] [dim](Enter to keep current)[/dim]:")
            console.print(card.get("back", ""))
            try:
                new_back = typer.prompt(">", default="", show_default=False).strip()
            except (typer.Abort, EOFError, KeyboardInterrupt):
                new_back = ""
            if new_front:
                card["front"] = new_front
            if new_back:
                card["back"] = new_back
            if new_front or new_back:
                update_card(card["id"], card["front"], card["back"])
                update_card_status(card["id"], "edited")
                console.print("[success]Card updated.[/]")
            else:
                update_card_status(card["id"], "accepted")
            edited.append(card)
        elif key == "s":
            update_card_status(card["id"], "rejected")
            console.print("[dim]Skipped.[/]")
            skipped.append(card)
        elif key == "q":
            break
        else:
            # Unknown key — treat as skip
            skipped.append(card)

    # Post-review summary
    console.print()
    console.print(
        f"[dim]Review complete: {len(accepted)} accepted, {len(edited)} edited, {len(skipped)} skipped.[/dim]"
    )

    # Regeneration offer if any cards were edited (UI-SPEC Screen 2)
    if edited:
        deck = cards[0].get("deck", "").split("::")[0] if cards else ""
        domain = cards[0].get("domain", "") if cards else ""
        remaining_count = total - len(accepted) - len(edited)
        console.print()
        console.print()
        from pb.cli.pickers import pick_boolean
        if pick_boolean(
            title="Regenerate remaining batch?",
            text=f"{len(edited)} cards were edited. Regenerate remaining batch with improved prompt?"
        ):
            if pick_boolean(title="Tier-2: Regenerate remaining cards?"):
                # Flash-summarize what was edited; append to context.md
                # D-07: Flash-summarize edits before appending to context.md
                edit_summary = anki_service.summarize_review_edits(deck, edited)
                anki_service.append_context_md(deck, f"\n## Review session edit notes\n{edit_summary}\n")
                console.print(f"[dim]Regenerating {remaining_count} cards...[/]")
                # Re-run generation using latest note/deck context
                note_slug = cards[0].get("note_slug", "") if cards else ""
                result = anki_service.generate_cards(
                    note_slug=note_slug, note_content="",
                    domain=domain, deck=deck, source="auto",
                )
                console.print(f"[success]Done. {result.get('count', 0)} new cards generated.[/success]")
        else:
            console.print("[dim]Session saved to deck context.[/dim]")
    else:
        console.print("[dim]Session complete.[/dim]")


def _do_export(cards: list[dict], console, domain: Optional[str], csv_only: bool = False) -> None:
    """Export selected cards with .apkg first, then optional sync, then CSV fallback."""
    from pb.vault.anki_client import (
        export_cards_to_anki,
        export_cards_to_apkg,
        export_cards_to_csv,
        is_anki_available,
    )
    from pb.vault import get_vault_path

    vault_path = get_vault_path()
    exported_ok = False

    if csv_only:
        csv_path = export_cards_to_csv(cards, vault_path)
        console.print(f"[success]CSV export written to {csv_path}[/]")
        console.print("[dim]Import manually via Anki File > Import.[/]")
        exported_ok = True
    else:
        packaged, package_path, package_msg = export_cards_to_apkg(cards, vault_path)
        if packaged:
            console.print(f"[success]{package_msg}[/]")
            console.print(f"[dim]Package: {package_path}[/]")
            exported_ok = True
        else:
            console.print(f"[warn]{package_msg}[/]")

    if not csv_only and is_anki_available():
        success, msg = export_cards_to_anki(cards)
        if success:
            console.print(f"[success]{msg}[/]")
            exported_ok = True
        else:
            console.print(f"[warn]{msg}[/]")
    if not exported_ok:
        csv_path = export_cards_to_csv(cards, vault_path)
        if csv_only:
            console.print(f"[success]CSV export written to {csv_path}[/]")
        elif is_anki_available():
            console.print(f"[warn]Falling back to CSV: {csv_path}[/]")
        else:
            console.print(f"[warn]Anki not running. Exported to {csv_path}[/]")
        console.print("[dim]Import manually via Anki File > Import.[/]")
        exported_ok = True

    # ANKI-04: Update domain _state.md after successful export
    if exported_ok:
        try:
            from pb.core.graph_writer import GraphWriter
            gw = GraphWriter(vault_path)
            knowledge_dir = vault_path / "knowledge"
            # Collect unique domains from exported cards
            exported_domains: set[str] = set()
            for card in cards:
                card_domain = card.get("domain")
                if not card_domain and card.get("deck"):
                    card_domain = card["deck"].split("::")[0]
                if card_domain:
                    exported_domains.add(card_domain)
            if domain:
                exported_domains.add(domain)
            for d in exported_domains:
                try:
                    domain_path = knowledge_dir / d
                    if domain_path.is_dir():
                        gw.update_state_md(domain_path, f"Anki export: {len(cards)} cards", vault_path)
                except Exception:
                    pass
        except Exception:
            pass  # Non-fatal: _state.md update is best-effort
