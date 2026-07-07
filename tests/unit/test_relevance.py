"""Unit tests for RelevanceFilter — LLM-powered batch relevance scoring.

Tests cover: batch scoring, threshold filtering, offline queue persistence,
malformed structured-output recovery, and markdown-fence stripping.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml

from pb.core.relevance import RelevanceFilter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def state_dir(tmp_path: Path) -> Path:
    """Provide a temporary state directory for queue persistence."""
    d = tmp_path / "state"
    d.mkdir()
    return d


@pytest.fixture
def rf(state_dir: Path) -> RelevanceFilter:
    """Provide a RelevanceFilter wired to the temp state dir."""
    return RelevanceFilter(state_dir=state_dir)


@pytest.fixture
def sample_items() -> list[dict]:
    """Three sample feed items for scoring."""
    return [
        {"title": "New Python release", "snippet": "Python 3.14 released today"},
        {"title": "Football scores", "snippet": "Premier League results"},
        {"title": "AI conference", "snippet": "NeurIPS call for papers"},
    ]


@pytest.fixture
def mock_client_available():
    """Mock get_client returning an available GeminiClient."""
    with patch("pb.core.relevance.get_client") as mock_gc:
        client = MagicMock()
        client.is_available.return_value = True
        mock_gc.return_value = client
        yield client


@pytest.fixture
def mock_client_unavailable():
    """Mock get_client returning an unavailable GeminiClient."""
    with patch("pb.core.relevance.get_client") as mock_gc:
        client = MagicMock()
        client.is_available.return_value = False
        mock_gc.return_value = client
        yield client


# ---------------------------------------------------------------------------
# Test 1: score_batch returns list of floats when LLM returns valid YAML
# ---------------------------------------------------------------------------


class TestScoreBatch:
    def test_returns_floats_on_valid_yaml(
        self, rf: RelevanceFilter, sample_items: list[dict], mock_client_available
    ):
        """score_batch returns list of floats when LLM returns valid YAML."""
        mock_client_available.generate.return_value = yaml.safe_dump(
            [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.1}, {"id": 2, "score": 0.7}]
        )
        scores = rf.score_batch(sample_items, "python, AI")
        assert scores == [0.9, 0.1, 0.7]
        assert all(isinstance(s, float) for s in scores)

    # Test 2: score_batch returns [] when LLM is unavailable
    def test_returns_empty_when_unavailable(
        self, rf: RelevanceFilter, sample_items: list[dict], mock_client_unavailable
    ):
        """score_batch returns [] when LLM is unavailable."""
        scores = rf.score_batch(sample_items, "python, AI")
        assert scores == []

    # Test 3: score_batch returns [] when LLM returns None
    def test_returns_empty_when_generate_none(
        self, rf: RelevanceFilter, sample_items: list[dict], mock_client_available
    ):
        """score_batch returns [] when LLM returns None."""
        mock_client_available.generate.return_value = None
        scores = rf.score_batch(sample_items, "python, AI")
        assert scores == []


# ---------------------------------------------------------------------------
# Test 4-6: _parse_scores edge cases
# ---------------------------------------------------------------------------


class TestParseScores:
    def test_handles_markdown_fenced_json(self, rf: RelevanceFilter):
        """_parse_scores handles markdown-fenced YAML."""
        fenced = "```yaml\n- id: 0\n  score: 0.5\n- id: 1\n  score: 0.8\n```"
        scores = rf._parse_scores(fenced, 2)
        assert scores == [0.5, 0.8]

    def test_returns_empty_on_malformed_yaml(self, rf: RelevanceFilter):
        """_parse_scores returns [] on malformed YAML."""
        scores = rf._parse_scores("not valid json at all!", 3)
        assert scores == []

    def test_returns_empty_on_missing_score_key(self, rf: RelevanceFilter):
        """_parse_scores returns [] when 'score' key is missing."""
        no_score = yaml.safe_dump([{"id": 0, "relevance": 0.5}])
        scores = rf._parse_scores(no_score, 1)
        assert scores == []


# ---------------------------------------------------------------------------
# Test 7-8: filter_items
# ---------------------------------------------------------------------------


class TestFilterItems:
    def test_separates_above_below_threshold(
        self, rf: RelevanceFilter, sample_items: list[dict], mock_client_available
    ):
        """filter_items separates items above/below threshold."""
        mock_client_available.generate.return_value = yaml.safe_dump(
            [{"id": 0, "score": 0.9}, {"id": 1, "score": 0.1}, {"id": 2, "score": 0.5}]
        )
        passed, queued, filtered_count = rf.filter_items(
            sample_items, "python, AI", threshold=0.3
        )
        assert len(passed) == 2  # items 0 and 2
        assert len(queued) == 0
        assert filtered_count == 1  # item 1 below 0.3

    def test_queues_all_when_llm_unavailable(
        self, rf: RelevanceFilter, sample_items: list[dict], mock_client_unavailable
    ):
        """filter_items queues all items when LLM unavailable."""
        passed, queued, filtered_count = rf.filter_items(
            sample_items, "python, AI", threshold=0.3
        )
        assert len(passed) == 0
        assert len(queued) == 3  # all items queued
        assert filtered_count == 0


# ---------------------------------------------------------------------------
# Test 9-11: queue persistence
# ---------------------------------------------------------------------------


class TestQueuePersistence:
    def test_enqueue_persists_and_appends(
        self, rf: RelevanceFilter, state_dir: Path
    ):
        """enqueue_items persists to YAML file and appends to existing queue."""
        items_a = [{"title": "A"}]
        items_b = [{"title": "B"}, {"title": "C"}]
        rf.enqueue_items(items_a)
        rf.enqueue_items(items_b)
        queue = rf.load_queue()
        assert len(queue) == 3
        assert queue[0]["title"] == "A"
        assert queue[2]["title"] == "C"
        # Verify file exists
        assert (state_dir / "pending-relevance.yaml").exists()

    def test_load_queue_returns_empty_when_missing(
        self, rf: RelevanceFilter
    ):
        """load_queue returns [] when file missing."""
        queue = rf.load_queue()
        assert queue == []

    def test_clear_queue_removes_file(
        self, rf: RelevanceFilter, state_dir: Path
    ):
        """clear_queue removes the file."""
        rf.enqueue_items([{"title": "X"}])
        assert (state_dir / "pending-relevance.yaml").exists()
        rf.clear_queue()
        assert not (state_dir / "pending-relevance.yaml").exists()


# ---------------------------------------------------------------------------
# Test 12: process_queue re-scores queued items
# ---------------------------------------------------------------------------


class TestProcessQueue:
    def test_rescores_queued_items_and_clears(
        self, rf: RelevanceFilter, state_dir: Path, mock_client_available
    ):
        """process_queue re-scores queued items and clears queue on success."""
        queued_items = [
            {"title": "Queued A", "snippet": "..."},
            {"title": "Queued B", "snippet": "..."},
        ]
        rf.enqueue_items(queued_items)
        mock_client_available.generate.return_value = yaml.safe_dump(
            [{"id": 0, "score": 0.8}, {"id": 1, "score": 0.05}]
        )
        passed, filtered_count = rf.process_queue("python, AI", threshold=0.3)
        assert len(passed) == 1  # only Queued A passes
        assert filtered_count == 1
        # Queue should be cleared after success
        assert not (state_dir / "pending-relevance.yaml").exists()
