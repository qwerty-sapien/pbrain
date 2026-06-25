# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Brain commands -- cross-domain vault intelligence (D-01) + orphan detection (GRPH-02)."""
from __future__ import annotations

import structlog
import typer

from pb.domain.exceptions import ExitCode

logger = structlog.get_logger()

brain_app = typer.Typer(no_args_is_help=False)


@brain_app.callback(invoke_without_command=True)
def brain_command(
    ctx: typer.Context,
    question: list[str] = typer.Argument(None, help="Question to ask (no quotes needed)"),
    show_prompt: bool = typer.Option(False, "--show-prompt", help="Display the LLM prompt"),
    auto: bool = typer.Option(False, "--auto", help="Flash Lite with adaptive escalation to Flash/Pro"),
    flash: bool = typer.Option(False, "--flash", help="Use Flash model directly"),
    pro: bool = typer.Option(False, "--pro", help="Use Pro model directly"),
    verbose: bool = typer.Option(False, "--verbose", help="Show per-note signal breakdown (D-06)"),
):
    """Query your vault with graph-aware AI search.

    Sends your vault's graph topology to the LLM, which reads
    relevant notes on demand via function calling. Pre-ranks candidates
    via CompositeScorer before querying (Phase 25 D-04).

    Default: Flash Lite, no escalation.
    --auto:    Flash Lite first; model self-selects escalation if needed.
    --flash:   Flash directly.
    --pro:     Pro directly (slowest, highest quality).
    --verbose: Show per-note signal breakdown and constellation tree after response.

    Gap detection: questions starting with 'gaps in' route to detect_gaps (D-09).

    Examples:
        pb brain what did I work on last week
        pb brain --auto who knows about machine learning
        pb brain --pro summarize my career trajectory
        pb brain --verbose neural networks
        pb brain gaps in machine-learning
    """
    if ctx.invoked_subcommand is not None:
        return

    question_text = " ".join(question) if question else ""
    if not question_text.strip():
        # D-01: No question provided, launch interactive REPL instead of erroring
        from pb.core.chat import ChatEngine
        import asyncio
        
        engine = ChatEngine(use_pro=pro, use_flash=flash, auto_escalate=auto)
        if not engine.is_available():
            typer.echo("LLM unavailable -- set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT to enable chat.", err=True)
            raise typer.Exit(code=ExitCode.CONFIG_ERROR)
            
        typer.echo("pb brain (interactive) -- type /exit to quit")
        
        while True:
            try:
                user_input = typer.prompt(">", prompt_suffix=" ")
            except (EOFError, KeyboardInterrupt):
                typer.echo("")
                break
                
            stripped = user_input.strip()
            if stripped in ("/exit", "/quit", "exit", "quit"):
                break
            if not stripped:
                continue
                
            try:
                asyncio.run(engine._async_chat_turn(stripped))
            except Exception as e:
                typer.echo(f"Chat error: {e}", err=True)
        
        return

    from pb.vault.embeddings import EmbeddingUnavailableError

    scoring_svc = ctx.obj['factory']['scoring_service']()

    try:
        # Gap detection routing: prefix "gaps in " detected here (D-10)
        if question_text.lower().startswith("gaps in "):
            domain = question_text[len("gaps in "):].strip()
            result = scoring_svc.detect_gaps(domain)
            _render_gap_table(result)
            return

        result = scoring_svc.rank_and_query(
            question_text,
            use_flash=flash,
            use_pro=pro,
            auto_escalate=auto,
            show_prompt=show_prompt,
            verbose=verbose,
        )
    except EmbeddingUnavailableError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1)

    if result.get("context_display"):
        typer.echo(f"  [{result['context_display']}]", err=True)

    typer.echo(result["answer"])

    if verbose and result.get("signal_data"):
        _render_signal_table(result["signal_data"])

    if verbose and result.get("top_slug") and result.get("constellation"):
        # Get stage_map for constellation node labels
        from pb.vault import get_vault_path
        from pb.vault.lifecycle import read_frontmatter
        vault_path = get_vault_path()
        stage_map: dict = {}
        try:
            for md_file in (vault_path / "knowledge").rglob("*.md"):
                content = md_file.read_text(errors="replace")
                fm, _ = read_frontmatter(content)
                stage_map[md_file.stem] = fm.get("learning_stage", "#new").lstrip("#")
        except Exception:
            pass
        _render_constellation(result["constellation"], result["top_slug"], stage_map)


def _render_signal_table(signal_data: list) -> None:
    """Render per-note signal breakdown table (--verbose, existing behavior)."""
    from pb.cli.console import get_console
    from rich.table import Table
    console = get_console()
    console.print()
    t = Table(
        title="Signal Breakdown",
        show_header=True,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        box=None,
    )
    t.add_column("NOTE", style="cyan")
    t.add_column("SCORE", justify="right")
    t.add_column("SEM", justify="right")
    t.add_column("LINK", justify="right")
    t.add_column("BACK", justify="right")
    t.add_column("TAG", justify="right")
    t.add_column("REC", justify="right")
    t.add_column("USE", justify="right")
    t.add_column("RED", justify="right")
    t.add_column("NOV", justify="right")
    for slug, score, signals in signal_data[:10]:
        t.add_row(
            slug[:30],
            f"{score:.3f}",
            f"{signals.semantic_similarity:.2f}",
            f"{signals.link_strength:.2f}",
            f"{signals.backlink_strength:.2f}",
            f"{signals.tag_affinity:.2f}",
            f"{signals.recency:.2f}",
            f"{signals.usage:.2f}",
            f"{signals.redundancy_penalty:.2f}",
            f"{signals.novelty_boost:.2f}",
        )
    console.print(t)
    console.print()


def _render_constellation(neighborhood: dict, top_slug: str, stage_map: dict) -> None:
    """Render 2-hop constellation as Rich tree (D-15). Hard cap: 2 hops (D-16)."""
    from pb.cli.console import get_console
    from rich.tree import Tree
    console = get_console()
    console.print()

    tree = Tree(f"Constellation: [bold]{top_slug}[/bold]")

    # Pitfall 5: deduplicate out1+in1 before iterating (dict.fromkeys preserves order)
    hop1 = list(dict.fromkeys(neighborhood.get("out1", []) + neighborhood.get("in1", [])))

    hop2_all = list(dict.fromkeys(
        neighborhood.get("out2", []) + neighborhood.get("in2", [])
    ))

    hop1_set = set(hop1)
    for slug in hop1:
        stage = stage_map.get(slug, "?")
        branch = tree.add(f"{slug} ([dim]#{stage}[/dim])")
        for hop2_slug in hop2_all:
            # Don't use shared `seen` for hop2 — avoids first-branch monopoly (WR-01)
            if hop2_slug not in hop1_set and hop2_slug != top_slug:
                hop2_stage = stage_map.get(hop2_slug, "?")
                branch.add(f"{hop2_slug} ([dim]#{hop2_stage}[/dim]) [[dim]2-hop[/dim]]")

    console.print(tree)
    console.print()


def _render_gap_table(gap_result: dict) -> None:
    """Render gap detection results as two-column Rich table (D-12)."""
    from pb.cli.console import get_console
    from rich.table import Table
    from rich.markup import escape
    console = get_console()

    domain = gap_result["domain"]
    have = gap_result["have"]      # [(slug, stage), ...]
    missing = gap_result["missing"]  # [concept_name, ...]

    console.print(f"\nGaps in: [bold]{escape(domain)}[/bold]\n")

    t = Table(
        show_header=True,
        header_style="bold",
        show_edge=False,
        pad_edge=False,
        box=None,
    )
    t.add_column(f"Have ({len(have)})", style="green", min_width=30)
    t.add_column(f"Missing ({len(missing)})", style="red", min_width=30)

    have_rows = [
        f"[green]✓[/green] {escape(slug)}  [dim]#{escape(stage)}[/dim]"
        for slug, stage in have
    ]
    missing_rows = [
        f"[red]✗[/red] {escape(concept)}\n  [dim]pb note {escape(concept)}[/dim]"
        for concept in missing
    ]

    for i in range(max(len(have_rows), len(missing_rows), 1)):
        t.add_row(
            have_rows[i] if i < len(have_rows) else "",
            missing_rows[i] if i < len(missing_rows) else "",
        )
    console.print(t)
    console.print()


@brain_app.command("orphans")
def orphans_command():
    """Show vault notes with no inbound or outbound links, grouped by folder (D-08)."""
    from pb.core.brain import BrainEngine
    from pb.cli.console import get_console
    from rich.table import Table

    engine = BrainEngine()
    orphan_list = engine.detect_orphans()
    console = get_console()

    if not orphan_list:
        console.print("[dim]No orphan notes found.[/]")
        return

    # Group by folder (D-08)
    by_folder: dict[str, list] = {}
    for o in orphan_list:
        by_folder.setdefault(o["folder"], []).append(o)

    total = len(orphan_list)
    console.print(f"\n[bold]Orphan Notes[/] ({total} total)\n")

    for folder, notes in sorted(by_folder.items()):
        console.rule(f"[bold]{folder}[/]")
        t = Table(
            show_header=True,
            header_style="bold",
            show_edge=False,
            show_lines=False,
            pad_edge=False,
            box=None,
        )
        t.add_column("Path", style="cyan")
        t.add_column("Title")
        t.add_column("Stage", style="yellow")
        t.add_column("Created")
        t.add_column("Words", justify="right")
        for note in notes:
            t.add_row(
                note["path"],
                note["title"],
                note.get("learning_stage") or "---",
                note.get("created", "---"),
                str(note.get("words", "---")),
            )
        console.print(t)
        console.print()
