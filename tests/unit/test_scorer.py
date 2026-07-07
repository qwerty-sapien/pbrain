"""Unit tests for CompositeScorer -- 7-signal composite scoring engine.

TDD RED phase: these tests are written before scorer.py exists.
All 18 behaviors from the plan are covered here.
"""
from __future__ import annotations

import datetime
import sqlite3
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest


# ---------------------------------------------------------------------------
# Helpers to build in-memory vault.db for tests
# ---------------------------------------------------------------------------

def _make_vault_db(tmp_path: Path, nodes: list[dict], links: list[tuple[str, str]]) -> None:
    """Create a minimal vault.db with nodes and links for testing."""
    db_path = tmp_path / "vault.db"
    conn = sqlite3.connect(str(db_path))
    conn.executescript("""
        PRAGMA journal_mode = WAL;
        CREATE TABLE IF NOT EXISTS nodes (
            slug       TEXT PRIMARY KEY,
            subfolder  TEXT NOT NULL,
            dirty_hop2 INTEGER NOT NULL DEFAULT 0,
            updated_at INTEGER NOT NULL DEFAULT (unixepoch())
        );
        CREATE TABLE IF NOT EXISTS links (
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            PRIMARY KEY (src, dst)
        );
        CREATE VIEW IF NOT EXISTS node_count AS SELECT COUNT(*) AS n FROM nodes;
    """)
    for node in nodes:
        conn.execute(
            "INSERT INTO nodes (slug, subfolder) VALUES (?, ?)",
            (node["slug"], node.get("subfolder", "knowledge/test")),
        )
    for src, dst in links:
        conn.execute("INSERT INTO links (src, dst) VALUES (?, ?)", (src, dst))
    conn.commit()
    conn.close()


def _make_mock_config(top_n: int = 20) -> MagicMock:
    """Return a mock config with default scoring weights."""
    cfg = MagicMock()
    cfg.learning.scoring_weights = {
        "semantic": 0.3,
        "link": 0.15,
        "backlink": 0.1,
        "tag_affinity": 0.1,
        "recency": 0.15,
        "usage": 0.1,
        "redundancy": -0.1,
    }
    cfg.learning.top_n_candidates = top_n
    return cfg


# ---------------------------------------------------------------------------
# Tests for composite_score() pure function
# ---------------------------------------------------------------------------

class TestCompositeScoreFunction:
    """Tests 1-2: composite_score() weighted sum formula."""

    def test_all_zero_signals_returns_zero(self):
        """Test 1: composite_score() with all zero signals returns 0.0."""
        from pb.core.scorer import ScoreSignals, composite_score

        signals = ScoreSignals()
        weights = {
            "semantic": 0.3, "link": 0.15, "backlink": 0.1,
            "tag_affinity": 0.1, "recency": 0.15, "usage": 0.1, "redundancy": -0.1,
        }
        result = composite_score(signals, weights)
        assert result == 0.0

    def test_known_signal_values_return_correct_weighted_sum(self):
        """Test 2: composite_score() with known signal values returns correct weighted sum."""
        from pb.core.scorer import ScoreSignals, composite_score

        signals = ScoreSignals(
            semantic_similarity=0.5,
            link_strength=0.8,
            backlink_strength=0.6,
            tag_affinity=1.0,
            recency=0.7,
            usage=0.4,
            redundancy_penalty=0.0,
            novelty_boost=0.0,
        )
        weights = {
            "semantic": 0.3, "link": 0.15, "backlink": 0.1,
            "tag_affinity": 0.1, "recency": 0.15, "usage": 0.1, "redundancy": -0.1,
        }
        expected = (
            0.3 * 0.5     # semantic
            + 0.15 * 0.8  # link
            + 0.1 * 0.6   # backlink
            + 0.1 * 1.0   # tag_affinity
            + 0.15 * 0.7  # recency
            + 0.1 * 0.4   # usage
            + (-0.1) * 0.0  # redundancy
            + 0.0           # novelty_boost (additive)
        )
        result = composite_score(signals, weights)
        assert abs(result - expected) < 1e-9

    def test_novelty_boost_is_additive_not_weighted(self):
        """Test 6: novelty_boost is additive (not weighted) -- verify formula adds it directly."""
        from pb.core.scorer import ScoreSignals, composite_score

        weights = {
            "semantic": 0.3, "link": 0.15, "backlink": 0.1,
            "tag_affinity": 0.1, "recency": 0.15, "usage": 0.1, "redundancy": -0.1,
        }
        # With novelty_boost = 0.5, score should be 0.0 (weighted) + 0.5 (additive)
        signals_with_boost = ScoreSignals(novelty_boost=0.5)
        signals_without_boost = ScoreSignals(novelty_boost=0.0)

        score_with = composite_score(signals_with_boost, weights)
        score_without = composite_score(signals_without_boost, weights)

        # The difference should be exactly 0.5 (additive, not weighted)
        assert abs(score_with - score_without - 0.5) < 1e-9


# ---------------------------------------------------------------------------
# Tests for _compute_novelty_boost()
# ---------------------------------------------------------------------------

class TestNoveltyBoost:
    """Tests 3-5: novelty boost linear decay."""

    def test_novelty_boost_today_is_max(self):
        """Test 3: novelty_boost for note created today = 1.0 (max(0, (7-0)/7))."""
        from pb.core.scorer import _compute_novelty_boost

        today = datetime.date.today().isoformat()
        result = _compute_novelty_boost(today)
        assert abs(result - 1.0) < 1e-9

    def test_novelty_boost_3_days_ago(self):
        """Test 4: novelty_boost for note created 3 days ago = ~0.571 (max(0, (7-3)/7))."""
        from pb.core.scorer import _compute_novelty_boost

        three_days_ago = (datetime.date.today() - datetime.timedelta(days=3)).isoformat()
        result = _compute_novelty_boost(three_days_ago)
        expected = (7 - 3) / 7
        assert abs(result - expected) < 1e-9

    def test_novelty_boost_7_plus_days_ago_is_zero(self):
        """Test 5: novelty_boost for note created 7+ days ago = 0.0."""
        from pb.core.scorer import _compute_novelty_boost

        seven_days_ago = (datetime.date.today() - datetime.timedelta(days=7)).isoformat()
        result_7 = _compute_novelty_boost(seven_days_ago)
        assert result_7 == 0.0

        ten_days_ago = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        result_10 = _compute_novelty_boost(ten_days_ago)
        assert result_10 == 0.0

    def test_novelty_boost_none_returns_zero(self):
        """novelty_boost with None created_date returns 0.0."""
        from pb.core.scorer import _compute_novelty_boost

        assert _compute_novelty_boost(None) == 0.0

    def test_novelty_boost_invalid_date_returns_zero(self):
        """novelty_boost with invalid date string returns 0.0."""
        from pb.core.scorer import _compute_novelty_boost

        assert _compute_novelty_boost("not-a-date") == 0.0


# ---------------------------------------------------------------------------
# Tests for _compute_tag_affinity()
# ---------------------------------------------------------------------------

class TestTagAffinity:
    """Test 9: tag_affinity signal."""

    def test_full_match_returns_one(self):
        """Full overlap between note tags and query tags returns 1.0."""
        from pb.core.scorer import _compute_tag_affinity

        result = _compute_tag_affinity(["python", "ml", "stats"], ["python", "ml"])
        assert result == 1.0

    def test_no_match_returns_zero(self):
        """No overlap returns 0.0."""
        from pb.core.scorer import _compute_tag_affinity

        result = _compute_tag_affinity(["python", "ml"], ["history", "art"])
        assert result == 0.0

    def test_partial_match_is_proportional(self):
        """Partial match returns proportion of query tags found."""
        from pb.core.scorer import _compute_tag_affinity

        result = _compute_tag_affinity(["python", "art"], ["python", "ml", "art"])
        # 2 of 3 query tags match
        expected = 2 / 3
        assert abs(result - expected) < 1e-9

    def test_empty_query_tags_returns_zero(self):
        """Empty query_tags returns 0.0."""
        from pb.core.scorer import _compute_tag_affinity

        result = _compute_tag_affinity(["python", "ml"], [])
        assert result == 0.0

    def test_hash_prefix_stripped(self):
        """Tags with # prefix are normalized."""
        from pb.core.scorer import _compute_tag_affinity

        result = _compute_tag_affinity(["#python", "#ml"], ["python", "ml"])
        assert result == 1.0


# ---------------------------------------------------------------------------
# Tests for link/backlink signals (normalized)
# ---------------------------------------------------------------------------

class TestLinkStrength:
    """Tests 7-8: link_strength and backlink_strength normalized 0-1."""

    def test_link_strength_normalized(self, tmp_path):
        """Test 7: link_strength = outgoing_links / max_outgoing across candidates."""
        _make_vault_db(
            tmp_path,
            nodes=[
                {"slug": "note-a"},
                {"slug": "note-b"},
                {"slug": "note-c"},
                {"slug": "note-d"},
            ],
            links=[
                ("note-a", "note-b"),
                ("note-a", "note-c"),
                ("note-a", "note-d"),  # 3 outgoing for note-a
                ("note-b", "note-c"),  # 1 outgoing for note-b
            ],
        )

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            conn = sqlite3.connect(str(tmp_path / "vault.db"))
            candidates = {"note-a": {}, "note-b": {}, "note-c": {}, "note-d": {}}
            link_map, backlink_map = scorer._compute_link_signals(conn, candidates)
            conn.close()

        # note-a has max outgoing (3), should be 1.0
        assert link_map["note-a"] == 1.0
        # note-b has 1 outgoing (1/3)
        assert abs(link_map["note-b"] - 1 / 3) < 1e-9
        # note-c and note-d have 0 outgoing
        assert link_map["note-c"] == 0.0
        assert link_map["note-d"] == 0.0

    def test_backlink_strength_normalized(self, tmp_path):
        """Test 8: backlink_strength = incoming_links / max_incoming across candidates."""
        _make_vault_db(
            tmp_path,
            nodes=[
                {"slug": "note-a"},
                {"slug": "note-b"},
                {"slug": "note-c"},
            ],
            links=[
                ("note-a", "note-c"),
                ("note-b", "note-c"),  # note-c has 2 incoming
                ("note-a", "note-b"),  # note-b has 1 incoming
            ],
        )

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            conn = sqlite3.connect(str(tmp_path / "vault.db"))
            candidates = {"note-a": {}, "note-b": {}, "note-c": {}}
            _, backlink_map = scorer._compute_link_signals(conn, candidates)
            conn.close()

        assert backlink_map["note-c"] == 1.0
        assert abs(backlink_map["note-b"] - 0.5) < 1e-9
        assert backlink_map["note-a"] == 0.0


# ---------------------------------------------------------------------------
# Tests for recency signal
# ---------------------------------------------------------------------------

class TestRecencySignal:
    """Test 10: recency decays from 1.0 today toward 0.0 for old notes."""

    def test_recency_today_is_one(self):
        from pb.core.scorer import _compute_recency

        today = datetime.date.today().isoformat()
        result = _compute_recency(today)
        assert abs(result - 1.0) < 1e-9

    def test_recency_old_note_is_less_than_one(self):
        from pb.core.scorer import _compute_recency

        old_date = (datetime.date.today() - datetime.timedelta(days=30)).isoformat()
        result = _compute_recency(old_date)
        assert 0.0 < result < 1.0

    def test_recency_none_returns_zero(self):
        from pb.core.scorer import _compute_recency

        assert _compute_recency(None) == 0.0


# ---------------------------------------------------------------------------
# Tests for usage signal normalization (Test 11)
# ---------------------------------------------------------------------------

class TestUsageSignal:
    """Test 11: usage signal = get_weighted_total() / max_usage across candidates."""

    def test_usage_normalized(self, tmp_path):
        """Usage signal is normalized: highest usage note gets 1.0."""
        _make_vault_db(
            tmp_path,
            nodes=[{"slug": "note-a"}, {"slug": "note-b"}, {"slug": "note-c"}],
            links=[],
        )

        usage_map = {"note-a": 10.0, "note-b": 5.0, "note-c": 0.0}

        def fake_weighted_total(slug):
            return usage_map.get(slug, 0.0)

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", side_effect=fake_weighted_total):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            from pb.core.scorer import ScoreSignals
            scored = [
                ("note-a", ScoreSignals(), {"slug": "note-a"}),
                ("note-b", ScoreSignals(), {"slug": "note-b"}),
                ("note-c", ScoreSignals(), {"slug": "note-c"}),
            ]
            scorer._normalize_usage(scored)

        assert scored[0][1].usage == 1.0       # note-a has max usage
        assert abs(scored[1][1].usage - 0.5) < 1e-9  # note-b is half
        assert scored[2][1].usage == 0.0       # note-c has no usage


# ---------------------------------------------------------------------------
# Test redundancy penalty (Test 12)
# ---------------------------------------------------------------------------

class TestRedundancyPenalty:
    """Test 12: redundancy_penalty = -1.0 for notes in redundant pairs."""

    def test_redundancy_penalty_applied(self, tmp_path):
        """Notes in redundant pairs get redundancy_penalty = 1.0."""
        _make_vault_db(
            tmp_path,
            nodes=[{"slug": "note-a"}, {"slug": "note-b"}, {"slug": "note-c"}],
            links=[],
        )

        mock_store = MagicMock()
        mock_store.find_redundant_pairs.return_value = [("note-a", "note-b", 0.92)]

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            from pb.core.scorer import CompositeScorer, ScoreSignals
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = mock_store

            scored = [
                ("note-a", ScoreSignals(usage=0.5), {}),  # lower score
                ("note-b", ScoreSignals(usage=1.0), {}),  # higher score
                ("note-c", ScoreSignals(usage=0.3), {}),
            ]
            scorer._apply_redundancy(scored)

        # note-a has lower usage score, should get redundancy penalty
        # note-b has higher usage score, no penalty
        penalties = {slug: sig.redundancy_penalty for slug, sig, _ in scored}
        # One of the pair should have penalty 1.0 (the lower scorer)
        assert penalties["note-a"] == 1.0 or penalties["note-b"] == 1.0
        assert not (penalties["note-a"] == 1.0 and penalties["note-b"] == 1.0)
        assert penalties["note-c"] == 0.0


# ---------------------------------------------------------------------------
# Tests for rank_notes() (Tests 13-16)
# ---------------------------------------------------------------------------

class TestRankNotes:
    """Tests 13-16: rank_notes() behavior."""

    def test_rank_notes_returns_sorted_descending_limited_to_top_n(self, tmp_path):
        """Test 13: rank_notes() returns list sorted by score descending, limited to top_n."""
        _make_vault_db(
            tmp_path,
            nodes=[
                {"slug": "note-a", "subfolder": "knowledge/test"},
                {"slug": "note-b", "subfolder": "knowledge/test"},
                {"slug": "note-c", "subfolder": "knowledge/test"},
            ],
            links=[],
        )
        # Create markdown files for the nodes (needed for frontmatter reading)
        domain_path = tmp_path / "knowledge" / "test"
        domain_path.mkdir(parents=True, exist_ok=True)
        for note in ["note-a", "note-b", "note-c"]:
            (domain_path / f"{note}.md").write_text(
                "---\nlearning_stage: '#new'\n---\n\nContent."
            )

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config(top_n=2)), \
             patch("pb.core.scorer.get_weighted_total", return_value=1.0):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            cfg = _make_mock_config(top_n=2)
            scorer._global_weights = cfg.learning.scoring_weights
            scorer._top_n = 2
            scorer._embedding_store = None

            results = scorer.rank_notes()

        assert len(results) <= 2
        if len(results) > 1:
            assert results[0][1] >= results[1][1]  # sorted descending

    def test_rank_notes_with_domain_filter(self, tmp_path):
        """Test 14: rank_notes() with domain filter only includes notes from that subfolder."""
        _make_vault_db(
            tmp_path,
            nodes=[
                {"slug": "note-a", "subfolder": "knowledge/python"},
                {"slug": "note-b", "subfolder": "knowledge/history"},
            ],
            links=[],
        )
        python_path = tmp_path / "knowledge" / "python"
        python_path.mkdir(parents=True, exist_ok=True)
        (python_path / "note-a.md").write_text("---\nlearning_stage: '#new'\n---\n\nContent.")

        history_path = tmp_path / "knowledge" / "history"
        history_path.mkdir(parents=True, exist_ok=True)
        (history_path / "note-b.md").write_text("---\nlearning_stage: '#new'\n---\n\nContent.")

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            cfg = _make_mock_config()
            scorer._global_weights = cfg.learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            results = scorer.rank_notes(domain="python")

        slugs = [r[0] for r in results]
        assert "note-a" in slugs
        assert "note-b" not in slugs

    def test_rank_notes_empty_vault_returns_empty_list(self, tmp_path):
        """Test 15: rank_notes() with empty vault returns empty list."""
        _make_vault_db(tmp_path, nodes=[], links=[])

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            cfg = _make_mock_config()
            scorer._global_weights = cfg.learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            results = scorer.rank_notes()

        assert results == []

    def test_embedding_unavailable_propagates_from_scorer(self, tmp_path):
        """Test 16: CompositeScorer._init_embedding_store propagates EmbeddingUnavailableError (D-06/Pitfall 4)."""
        import builtins
        import unittest.mock as mock
        from pb.vault.embeddings import EmbeddingUnavailableError

        real_import = builtins.__import__

        def mock_import(name, *args, **kwargs):
            if name == "sqlite_vec":
                raise ImportError("mocked")
            return real_import(name, *args, **kwargs)

        with mock.patch("builtins.__import__", side_effect=mock_import), \
             patch("pb.core.scorer.get_config", return_value=_make_mock_config()):
            from pb.core.scorer import CompositeScorer
            with pytest.raises(EmbeddingUnavailableError):
                CompositeScorer(tmp_path)


# ---------------------------------------------------------------------------
# Test stage-aware filtering (Test 17)
# ---------------------------------------------------------------------------

class TestStageAwareFiltering:
    """Test 17: stage-aware filtering for explore/consolidate/exploit/re-engage modes."""

    def _make_staged_vault(self, tmp_path: Path) -> None:
        """Create vault with notes at different learning stages."""
        _make_vault_db(
            tmp_path,
            nodes=[
                {"slug": "new-note", "subfolder": "knowledge/test"},
                {"slug": "learning-note", "subfolder": "knowledge/test"},
                {"slug": "learnt-note", "subfolder": "knowledge/test"},
                {"slug": "stale-note", "subfolder": "knowledge/test"},
            ],
            links=[],
        )
        domain_path = tmp_path / "knowledge" / "test"
        domain_path.mkdir(parents=True, exist_ok=True)
        (domain_path / "new-note.md").write_text("---\nlearning_stage: '#new'\n---\n\nContent.")
        (domain_path / "learning-note.md").write_text("---\nlearning_stage: '#learning'\n---\n\nContent.")
        (domain_path / "learnt-note.md").write_text("---\nlearning_stage: '#learnt'\n---\n\nContent.")
        (domain_path / "stale-note.md").write_text("---\nlearning_stage: '#stale'\n---\n\nContent.")

    def _make_scorer(self, tmp_path: Path):
        from pb.core.scorer import CompositeScorer
        scorer = CompositeScorer.__new__(CompositeScorer)
        scorer._vault_path = tmp_path
        scorer._global_weights = _make_mock_config().learning.scoring_weights
        scorer._top_n = 20
        scorer._embedding_store = None
        return scorer

    def test_explore_mode_returns_only_new_notes(self, tmp_path):
        """Test 17a: mode='explore' returns only #new notes."""
        self._make_staged_vault(tmp_path)
        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            scorer = self._make_scorer(tmp_path)
            results = scorer.rank_notes(stage_mode="explore")

        slugs = [r[0] for r in results]
        assert "new-note" in slugs
        assert "learning-note" not in slugs
        assert "learnt-note" not in slugs
        assert "stale-note" not in slugs

    def test_consolidate_mode_returns_only_learning_notes(self, tmp_path):
        """Test 17b: mode='consolidate' returns only #learning notes."""
        self._make_staged_vault(tmp_path)
        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            scorer = self._make_scorer(tmp_path)
            results = scorer.rank_notes(stage_mode="consolidate")

        slugs = [r[0] for r in results]
        assert "learning-note" in slugs
        assert "new-note" not in slugs
        assert "learnt-note" not in slugs

    def test_exploit_mode_returns_only_learnt_notes(self, tmp_path):
        """Test 17c: mode='exploit' returns only #learnt notes."""
        self._make_staged_vault(tmp_path)
        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            scorer = self._make_scorer(tmp_path)
            results = scorer.rank_notes(stage_mode="exploit")

        slugs = [r[0] for r in results]
        assert "learnt-note" in slugs
        assert "new-note" not in slugs
        assert "learning-note" not in slugs

    def test_reengage_mode_returns_only_stale_notes(self, tmp_path):
        """Test 17d: mode='re-engage' returns only #stale notes."""
        self._make_staged_vault(tmp_path)
        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()), \
             patch("pb.core.scorer.get_weighted_total", return_value=0.0):
            scorer = self._make_scorer(tmp_path)
            results = scorer.rank_notes(stage_mode="re-engage")

        slugs = [r[0] for r in results]
        assert "stale-note" in slugs
        assert "new-note" not in slugs
        assert "learning-note" not in slugs
        assert "learnt-note" not in slugs


# ---------------------------------------------------------------------------
# Test domain weight overrides (Test 18)
# ---------------------------------------------------------------------------

class TestDomainWeightOverrides:
    """Test 18: _load_domain_weight_overrides reads scoring_weights from _state.md."""

    def test_load_domain_weight_overrides_reads_state_md(self, tmp_path):
        """Test 18a: _load_domain_weight_overrides returns scoring_weights from _state.md."""
        from pb.core.scorer import _load_domain_weight_overrides

        domain_path = tmp_path / "knowledge" / "test-domain"
        domain_path.mkdir(parents=True, exist_ok=True)
        state_md = domain_path / "_state.md"
        state_md.write_text(
            "---\nscoring_weights:\n  semantic: 0.5\n  recency: 0.05\n---\n\nBody text."
        )

        result = _load_domain_weight_overrides(tmp_path, "test-domain")
        assert result == {"semantic": 0.5, "recency": 0.05}

    def test_load_domain_weight_overrides_no_state_md_returns_empty(self, tmp_path):
        """Test 18b: no _state.md returns empty dict."""
        from pb.core.scorer import _load_domain_weight_overrides

        result = _load_domain_weight_overrides(tmp_path, "nonexistent-domain")
        assert result == {}

    def test_resolve_weights_with_domain_merges_overrides(self, tmp_path):
        """Test 18c: _resolve_weights('domain') returns global defaults + domain overrides merged."""
        domain_path = tmp_path / "knowledge" / "test-domain"
        domain_path.mkdir(parents=True, exist_ok=True)
        (domain_path / "_state.md").write_text(
            "---\nscoring_weights:\n  semantic: 0.5\n  recency: 0.05\n---\n\nBody."
        )

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            merged = scorer._resolve_weights("test-domain")

        # Overridden values from _state.md
        assert merged["semantic"] == 0.5
        assert merged["recency"] == 0.05
        # Global defaults preserved for keys not in _state.md
        assert merged["link"] == 0.15

    def test_resolve_weights_no_domain_returns_global_defaults(self, tmp_path):
        """Test 18d: _resolve_weights(None) returns global defaults unchanged."""
        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            result = scorer._resolve_weights(None)

        assert result["semantic"] == 0.3
        assert result["link"] == 0.15
        assert result["recency"] == 0.15

    def test_resolve_weights_state_md_without_scoring_weights_preserves_defaults(self, tmp_path):
        """Test 18e: _state.md without scoring_weights keeps global defaults."""
        domain_path = tmp_path / "knowledge" / "empty-domain"
        domain_path.mkdir(parents=True, exist_ok=True)
        (domain_path / "_state.md").write_text("---\ntitle: Empty domain\n---\n\nBody.")

        with patch("pb.core.scorer.get_config", return_value=_make_mock_config()):
            from pb.core.scorer import CompositeScorer
            scorer = CompositeScorer.__new__(CompositeScorer)
            scorer._vault_path = tmp_path
            scorer._global_weights = _make_mock_config().learning.scoring_weights
            scorer._top_n = 20
            scorer._embedding_store = None

            result = scorer._resolve_weights("empty-domain")

        assert result["semantic"] == 0.3  # global default
        assert result["link"] == 0.15


# ---------------------------------------------------------------------------
# Test STAGE_MODES and MODE_STAGES constants
# ---------------------------------------------------------------------------

class TestStageConstants:
    """Verify STAGE_MODES and MODE_STAGES dicts are present and correct."""

    def test_stage_modes_dict_exists(self):
        from pb.core.scorer import STAGE_MODES
        assert "#new" in STAGE_MODES
        assert STAGE_MODES["#new"] == "explore"
        assert STAGE_MODES["#learning"] == "consolidate"
        assert STAGE_MODES["#learnt"] == "exploit"
        assert STAGE_MODES["#stale"] == "re-engage"

    def test_mode_stages_dict_exists(self):
        from pb.core.scorer import MODE_STAGES
        assert "explore" in MODE_STAGES
        assert "#new" in MODE_STAGES["explore"]
        assert "#learning" in MODE_STAGES["consolidate"]
        assert "#learnt" in MODE_STAGES["exploit"]
        assert "#stale" in MODE_STAGES["re-engage"]


# ---------------------------------------------------------------------------
# Test module-level exports
# ---------------------------------------------------------------------------

class TestModuleExports:
    """Verify key exports and class structure."""

    def test_imports_ok(self):
        from pb.core.scorer import CompositeScorer, ScoreSignals, composite_score
        assert CompositeScorer is not None
        assert ScoreSignals is not None
        assert composite_score is not None

    def test_score_signals_is_dataclass_with_all_fields(self):
        from pb.core.scorer import ScoreSignals
        import dataclasses
        assert dataclasses.is_dataclass(ScoreSignals)
        fields = {f.name for f in dataclasses.fields(ScoreSignals)}
        expected = {
            "semantic_similarity", "link_strength", "backlink_strength",
            "tag_affinity", "recency", "usage", "redundancy_penalty", "novelty_boost",
        }
        assert expected == fields

    def test_no_llm_in_scorer_module(self):
        """Verify scorer.py does not import LLM clients directly."""
        import inspect
        import pb.core.scorer as scorer_module
        source = inspect.getsource(scorer_module)
        # Should not directly import or call LLM generation functions
        assert "generate(" not in source
        assert "get_client()" not in source
