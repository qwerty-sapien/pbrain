# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Relevance filter for ingestion pipeline (D-13 to D-18).

Batch-scores feed items via Gemini Flash Lite and filters by threshold.
Items that cannot be scored (LLM unavailable) are queued to a YAML file
for retry on the next daemon run. Structured YAML/JSON model output is handled
gracefully.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog

from pb.llm.gemini import get_client
from pb.storage.yaml_io import dump_yaml, extract_structured_yaml, load_yaml_file, write_yaml_file

logger = structlog.get_logger()

QUEUE_FILENAME = "pending-relevance.yaml"

RELEVANCE_PROMPT_TEMPLATE = """\
You are a relevance filter for a personal knowledge system.

User interests: {interests}

Rate each item's relevance to the user's interests on a scale 0.0 to 1.0.
Return YAML only, one item per input row, in the same order:
- id: 0
  score: 0.8
- id: 1
  score: 0.1

Items to score:
{items_yaml}"""


class RelevanceFilter:
    """LLM-powered batch relevance scoring with offline queue.

    Args:
        state_dir: Directory for pending-relevance.yaml persistence.
    """

    def __init__(self, state_dir: Path) -> None:
        self.state_dir = state_dir

    # -- Scoring ---------------------------------------------------------------

    def score_batch(self, items: list[dict], interests: str) -> list[float]:
        """Score a batch of items via Gemini Flash Lite.

        Returns list of float scores (0.0-1.0), one per item, in same order.
        Returns [] if LLM is unavailable or returns None.
        """
        client = get_client()
        if not client.is_available():
            return []

        # Build compact item list for prompt
        items_for_prompt = [
            {"id": i, "title": item.get("title", ""), "snippet": item.get("snippet", "")}
            for i, item in enumerate(items)
        ]

        # Split into batches based on configured batch_size
        batch_size = self._get_batch_size()
        all_scores: list[float] = []

        for start in range(0, len(items_for_prompt), batch_size):
            chunk = items_for_prompt[start : start + batch_size]
            prompt = RELEVANCE_PROMPT_TEMPLATE.format(
                interests=interests,
                items_yaml=dump_yaml(chunk).strip(),
            )
            result = client.generate(prompt)
            if result is None:
                return []

            scores = self._parse_scores(result, len(chunk))
            if not scores:
                return []

            all_scores.extend(scores)

        return all_scores

    def _get_batch_size(self) -> int:
        """Read batch_size from config, defaulting to 100."""
        try:
            from pb.storage.config import get_config

            return get_config().ingest.relevance.batch_size
        except Exception:
            return 100

    def _parse_scores(self, result: str, expected: int) -> list[float]:
        """Parse LLM response into a list of float scores.

        Handles markdown fences and both YAML/JSON payloads. Returns [] on any
        parse failure (missing keys, wrong types).
        """
        try:
            data = extract_structured_yaml(result, [])
            if not isinstance(data, list):
                logger.debug("relevance.parse_scores", error="not_a_list")
                return []

            # Sort by id to preserve order, extract score values
            sorted_items = sorted(data, key=lambda x: x["id"])
            scores = [float(item["score"]) for item in sorted_items]

            return scores

        except (KeyError, TypeError, ValueError) as exc:
            logger.debug("relevance.parse_scores", error=str(exc))
            return []

    # -- Filtering -------------------------------------------------------------

    def filter_items(
        self,
        items: list[dict],
        interests: str,
        threshold: float = 0.3,
    ) -> tuple[list[dict], list[dict], int]:
        """Score and filter items by relevance threshold.

        Returns:
            (passed_items, queued_items, filtered_count)
            - passed: items scoring >= threshold
            - queued: items when LLM unavailable (all of them)
            - filtered_count: items scoring below threshold (discarded)
        """
        scores = self.score_batch(items, interests)

        if not scores:
            # LLM unavailable — queue everything, filter nothing
            return [], list(items), 0

        passed: list[dict] = []
        filtered_count = 0

        for item, score in zip(items, scores):
            if score >= threshold:
                passed.append(item)
            else:
                filtered_count += 1

        return passed, [], filtered_count

    # -- Queue persistence -----------------------------------------------------

    def _queue_path(self) -> Path:
        """Return path to pending-relevance.yaml."""
        return self.state_dir / QUEUE_FILENAME

    def enqueue_items(self, items: list[dict]) -> None:
        """Append items to the offline queue. Atomic write."""
        existing = self.load_queue()
        existing.extend(items)
        tmp = self._queue_path().with_suffix(".tmp")
        write_yaml_file(tmp, existing)
        tmp.replace(self._queue_path())

    def load_queue(self) -> list[dict]:
        """Read queued items from disk. Returns [] if missing or corrupt."""
        legacy_path = self.state_dir / "pending-relevance.json"
        data = load_yaml_file(self._queue_path(), None)
        if data is None:
            data = load_yaml_file(legacy_path, [])
        return data if isinstance(data, list) else []

    def clear_queue(self) -> None:
        """Remove the queue file."""
        self._queue_path().unlink(missing_ok=True)

    def process_queue(
        self,
        interests: str,
        threshold: float = 0.3,
    ) -> tuple[list[dict], int]:
        """Re-score queued items and return results.

        Returns:
            (passed_items, filtered_count)
        Clears the queue on successful scoring.
        """
        items = self.load_queue()
        if not items:
            return [], 0

        passed, queued, filtered_count = self.filter_items(items, interests, threshold)

        # Only clear if scoring succeeded (queued empty means LLM responded)
        if not queued:
            self.clear_queue()

        return passed, filtered_count
