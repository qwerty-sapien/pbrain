# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Ingestion orchestrator -- unified pipeline: Gmail -> RSS -> scrapers (D-19).

Coordinates all external data sources into the vault inbox.
Gmail items are NOT relevance-filtered (D-15: tracked senders are user-curated).
RSS and scraped items pass through RelevanceFilter (D-13).

Each source runs in its own try/except so one failure does not block others
(T-06-17 mitigation).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import structlog
import yaml

logger = structlog.get_logger()


@dataclass
class IngestResult:
    """Summary of one source's ingestion run."""

    source: str
    fetched: int = 0
    filtered: int = 0
    written: int = 0
    queued: int = 0
    other_count: int = 0  # Gmail non-tracked count (D-04)
    errors: list[str] = field(default_factory=list)
    skipped: bool = False


def _load_goal_titles(vault_path: Path) -> list[str]:
    """Read goal titles from direction/goals/*.md frontmatter.

    Returns a list of title strings for use as additional relevance keywords.
    Falls back to empty list on any error (non-critical helper).
    """
    titles: list[str] = []
    goals_dir = vault_path / "direction" / "goals"
    if not goals_dir.is_dir():
        return titles

    for md_file in goals_dir.glob("*.md"):
        try:
            text = md_file.read_text()
            if not text.startswith("---"):
                continue
            # Extract YAML frontmatter between first two ---
            parts = text.split("---", 2)
            if len(parts) < 3:
                continue
            fm = yaml.safe_load(parts[1])
            if isinstance(fm, dict) and fm.get("title"):
                titles.append(str(fm["title"]))
        except Exception:
            continue

    return titles


class IngestionOrchestrator:
    """Coordinates Gmail, RSS feeds, and scrapers into a unified pipeline.

    Call run_all() with source flags to execute the full ingestion.
    Config and paths are loaded lazily inside run_all (no init state needed).
    """

    def run_all(
        self,
        *,
        run_gmail: bool = True,
        run_feeds: bool = True,
        run_scrapers: bool = True,
    ) -> list[IngestResult]:
        """Execute ingestion pipeline: Gmail -> RSS -> scrapers.

        Each source is isolated -- failures in one do not affect others
        (T-06-17). Gmail items are NOT relevance-filtered (D-15).

        Args:
            run_gmail: Include Gmail source.
            run_feeds: Include RSS feeds.
            run_scrapers: Include scrapers.

        Returns:
            List of IngestResult, one per source (or per feed/scraper).
        """
        from pb.storage.config import get_config, get_data_dir, get_vault_path

        config = get_config()
        data_dir = get_data_dir(config)
        vault_path = get_vault_path(config)
        results: list[IngestResult] = []

        if run_gmail:
            try:
                results.append(self._run_gmail(config, vault_path))
            except Exception as exc:
                logger.warning("ingestion.gmail_error", error=str(exc))
                results.append(
                    IngestResult(
                        source="gmail",
                        errors=[f"Unexpected error: {exc}"],
                        skipped=True,
                    )
                )

        if run_feeds:
            try:
                results.extend(self._run_feeds(config, data_dir, vault_path))
            except Exception as exc:
                logger.warning("ingestion.feeds_error", error=str(exc))
                results.append(
                    IngestResult(
                        source="feeds",
                        errors=[f"Unexpected error: {exc}"],
                        skipped=True,
                    )
                )

        return results

    # -- Gmail (D-15: NOT relevance-filtered) --------------------------------

    def _run_gmail(self, config, vault_path: Path) -> IngestResult:
        """Ingest tracked Gmail senders. No relevance filtering (D-15)."""
        from pb.core.gmail import GmailClient

        result = IngestResult(source="gmail")
        client = GmailClient()

        if not client.is_authenticated():
            result.errors.append("Gmail is not authenticated. Reconnect the Gmail integration and try again.")
            result.skipped = True
            return result

        senders = config.gmail.senders
        if not senders:
            result.errors.append(
                "No tracked Gmail senders configured. Add at least one sender in your ProductiveBrain config."
            )
            result.skipped = True
            return result

        since_date = (datetime.now() - timedelta(days=7)).strftime("%Y/%m/%d")

        try:
            gmail_result = client.fetch_tracked(senders, since_date, vault_path)
            result.fetched = gmail_result.fetched
            result.written = gmail_result.written
            result.errors.extend(gmail_result.errors)
        except Exception as exc:
            logger.warning("ingestion.gmail_fetch_error", error=str(exc))
            result.errors.append(f"Gmail fetch failed: {exc}")

        # Non-tracked email count (D-04) -- non-critical
        try:
            result.other_count = client.count_other_emails(senders, since_date)
        except Exception:
            pass

        return result

    # -- RSS Feeds (D-13: relevance-filtered) --------------------------------

    def _run_feeds(
        self, config, data_dir: Path, vault_path: Path
    ) -> list[IngestResult]:
        """Ingest RSS feeds with relevance filtering (D-13, T-06-18)."""
        from pb.core.feed_reader import FeedReader
        from pb.core.relevance import RelevanceFilter
        from pb.mcp.tools.feeds import _load_feeds

        feeds_data = _load_feeds(vault_path)
        feeds = [
            f for f in feeds_data.get("feeds", []) if f.get("enabled", True)
        ]
        if not feeds:
            return [
                IngestResult(
                    source="feeds",
                    skipped=True,
                    errors=["No feeds configured"],
                )
            ]

        reader = FeedReader(data_dir)
        relevance_filter = RelevanceFilter(data_dir)

        # Build interest string from config keywords + goal titles
        keywords = config.scrape.filters.keywords or []
        goal_titles = _load_goal_titles(vault_path)
        all_interests = keywords + goal_titles
        interests = ", ".join(all_interests) if all_interests else ""

        threshold = config.ingest.relevance.threshold

        feed_results = reader.fetch_all(
            feeds,
            vault_path,
            relevance_filter=relevance_filter if interests else None,
            interests=interests or None,
            threshold=threshold,
        )

        # Convert FeedResult -> IngestResult
        ingest_results: list[IngestResult] = []
        for fr in feed_results:
            ingest_results.append(
                IngestResult(
                    source=fr.source,
                    fetched=fr.fetched,
                    filtered=fr.filtered,
                    written=fr.written,
                    queued=fr.queued,
                    errors=fr.errors,
                    skipped=fr.skipped,
                )
            )

        return ingest_results

