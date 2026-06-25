# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Ingest commands -- unified ingestion pipeline: Gmail + RSS + scrapers (D-19).

Commands: (bare) run all, with --no-gmail / --no-feeds / --no-scrapers flags.
Subsumes pb scrape for scheduled runs (D-20).
"""

from __future__ import annotations

import typer

app = typer.Typer(no_args_is_help=False)


@app.callback(invoke_without_command=True)
def ingest_default(
    ctx: typer.Context,
    gmail: bool = typer.Option(True, "--gmail/--no-gmail", help="Include Gmail"),
    feeds: bool = typer.Option(True, "--feeds/--no-feeds", help="Include RSS feeds"),
    scrapers: bool = typer.Option(
        True, "--scrapers/--no-scrapers", help="Include scrapers"
    ),
):
    """Run the unified ingestion pipeline.

    Bare 'pb ingest' runs all sources. Use flags to skip specific sources.
    """
    if ctx.invoked_subcommand is not None:
        return
    run_ingest(gmail=gmail, feeds=feeds, scrapers=scrapers)


def run_ingest(
    *, gmail: bool = True, feeds: bool = True, scrapers: bool = True
) -> None:
    """Execute the ingestion pipeline (D-19)."""
    from pb.core.ingestion import IngestionOrchestrator

    orch = IngestionOrchestrator()
    typer.echo("Running ingestion pipeline...")
    results = orch.run_all(run_gmail=gmail, run_feeds=feeds, run_scrapers=scrapers)

    if not results:
        typer.echo("No sources to ingest.")
        return

    # Print summary table (pattern from scrape.py)
    typer.echo(
        f"\n  {'Source':20s}  {'Fetched':8s}  {'Filtered':9s}  "
        f"{'Written':8s}  {'Queued':7s}  {'Status'}"
    )
    typer.echo("  " + "-" * 72)
    for r in results:
        if r.skipped:
            status = "skipped"
        elif r.errors:
            status = "errors"
        else:
            status = "ok"
        typer.echo(
            f"  {r.source:20s}  {r.fetched:8d}  {r.filtered:9d}  "
            f"{r.written:8d}  {r.queued:7d}  {status}"
        )
        for err in r.errors:
            typer.echo(f"    [ERR] {err}")

    # Gmail non-tracked count (D-04)
    for r in results:
        if r.source == "gmail" and r.other_count > 0:
            typer.echo(f"\n  {r.other_count} other emails since last check.")

    # Queue summary
    total_queued = sum(r.queued for r in results)
    if total_queued > 0:
        typer.echo(
            f"\n  {total_queued} items queued (LLM unavailable). "
            "Will retry on next run."
        )

    total_filtered = sum(r.filtered for r in results)
    if total_filtered > 0:
        typer.echo(f"  Filtered: {total_filtered} low-relevance items.")
