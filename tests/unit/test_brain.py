"""Unit tests for BrainEngine -- graph-aware vault intelligence.

Tests cover:
- is_available: delegates to GeminiClient
- query: model selection, auto-escalation, LLM fallback
- query with pre_ranked_candidates (Phase 19 D-04)
- query with learning_stage (Phase 19 D-07 stage-aware prompts)
- query_constellation: 1-hop + 2-hop neighborhood challenge (Phase 19 D-08)
- WEAK_AREA_SENTINEL constant (Phase 19 D-10)
- get_context_display: graph stats + model info
- detect_orphans: orphan note detection
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch, call

import pytest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_brain_engine(available: bool = True):
    """Create a BrainEngine with mocked GeminiClient."""
    from pb.core.brain import BrainEngine

    engine = BrainEngine.__new__(BrainEngine)
    engine._gemini = MagicMock()
    engine._gemini.is_available.return_value = available
    engine._graph_stats = None
    engine._model_used = None
    return engine


# Fake graph edges for patching load_vault_graph
_FAKE_EDGES = {
    "note-a.md": ["note-b.md"],
    "note-b.md": [],
}

# Fake adjacency text output for patching graph_to_adjacency_text
_FAKE_GRAPH_TEXT = ("  note-a -> note-b", 2, 1)


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_returns_true_when_gemini_available(self):
        """BrainEngine.is_available returns True when GeminiClient is available."""
        engine = _make_brain_engine(available=True)
        assert engine.is_available() is True

    def test_returns_false_when_gemini_unavailable(self):
        """BrainEngine.is_available returns False when GeminiClient is not available."""
        engine = _make_brain_engine(available=False)
        assert engine.is_available() is False


# ---------------------------------------------------------------------------
# query -- basic model selection
# ---------------------------------------------------------------------------


class TestQuery:
    def test_returns_unavailable_message_when_no_llm(self):
        """query returns 'LLM unavailable' message when Gemini not available."""
        engine = _make_brain_engine(available=False)
        result = engine.query("any question")
        assert "LLM unavailable" in result

    def test_returns_empty_vault_message_when_no_nodes(self):
        """query returns 'Vault is empty' when graph has zero nodes."""
        engine = _make_brain_engine(available=True)

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value={}):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text", return_value=("", 0, 0)
                ):
                    result = engine.query("some question")

        assert "empty" in result.lower()

    def test_default_model_is_flash_lite(self):
        """query uses Flash Lite model when no model flags are set."""
        from pb.core.brain import FLASH_LITE_MODEL

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer from Flash Lite"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("simple question")

        assert result == "Answer from Flash Lite"
        call_args = engine._gemini.generate_with_tools.call_args
        assert call_args[0][1] == FLASH_LITE_MODEL

    def test_use_flash_selects_flash_model(self):
        """query uses Flash model when use_flash=True."""
        from pb.core.brain import FLASH_MODEL

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer from Flash"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question", use_flash=True)

        assert result == "Answer from Flash"
        call_args = engine._gemini.generate_with_tools.call_args
        assert call_args[0][1] == FLASH_MODEL

    def test_use_pro_selects_pro_model(self):
        """query uses Pro model when use_pro=True."""
        from pb.core.brain import PRO_MODEL

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer from Pro"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question", use_pro=True)

        assert result == "Answer from Pro"
        call_args = engine._gemini.generate_with_tools.call_args
        assert call_args[0][1] == PRO_MODEL

    def test_returns_failure_message_when_generate_returns_none(self):
        """query returns error message when generate_with_tools returns None."""
        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = None

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question")

        assert "failed" in result.lower()


# ---------------------------------------------------------------------------
# query with pre_ranked_candidates (Phase 19 D-04)
# ---------------------------------------------------------------------------


class TestQueryPreRankedCandidates:
    def test_pre_ranked_candidates_builds_filtered_graph(self):
        """When pre_ranked_candidates is provided, a filtered graph is built via vault.db query."""
        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Filtered answer"

        fake_candidates = ["note-a", "note-b", "note-c"]
        fake_conn = MagicMock()
        fake_conn.execute.return_value.fetchall.return_value = [
            ("note-a", "note-b"),
        ]

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.open_vault_db", fake_conn, create=True):
                with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                    # Patch the import inside query()
                    import pb.vault.graph_store as gs
                    with patch.object(gs, "open_vault_db", return_value=fake_conn):
                        with patch(
                            "pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES
                        ):
                            with patch(
                                "pb.core.brain.graph_to_adjacency_text",
                                return_value=_FAKE_GRAPH_TEXT,
                            ):
                                result = engine.query(
                                    "question", pre_ranked_candidates=fake_candidates
                                )

        # Should call generate_with_tools — the result path worked
        assert engine._gemini.generate_with_tools.called

    def test_pre_ranked_candidates_accepted_in_signature(self):
        """query() signature accepts pre_ranked_candidates parameter (backward compat check)."""
        import inspect
        from pb.core.brain import BrainEngine

        sig = inspect.signature(BrainEngine.query)
        assert "pre_ranked_candidates" in sig.parameters
        param = sig.parameters["pre_ranked_candidates"]
        assert param.default is None

    def test_none_pre_ranked_candidates_falls_back_to_full_graph(self):
        """When pre_ranked_candidates=None, full graph is loaded normally."""
        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Full graph answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch(
                "pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES
            ) as mock_load:
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question", pre_ranked_candidates=None)

        # load_vault_graph must be called when no pre_ranked_candidates
        assert mock_load.called
        assert result == "Full graph answer"


# ---------------------------------------------------------------------------
# query with learning_stage (Phase 19 D-07)
# ---------------------------------------------------------------------------


class TestQueryLearningStage:
    def test_learning_stage_new_includes_explore_prompt(self):
        """When learning_stage='#new', STAGE_PROMPTS['explore'] is prepended to prompt."""
        from pb.core.brain import STAGE_PROMPTS

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Stage answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("question", learning_stage="#new")

        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        assert STAGE_PROMPTS["explore"] in prompt_used

    def test_learning_stage_learning_includes_consolidate_prompt(self):
        """When learning_stage='#learning', STAGE_PROMPTS['consolidate'] is prepended."""
        from pb.core.brain import STAGE_PROMPTS

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("question", learning_stage="#learning")

        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        assert STAGE_PROMPTS["consolidate"] in prompt_used

    def test_learning_stage_learnt_includes_exploit_prompt(self):
        """When learning_stage='#learnt', STAGE_PROMPTS['exploit'] is prepended."""
        from pb.core.brain import STAGE_PROMPTS

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("question", learning_stage="#learnt")

        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        assert STAGE_PROMPTS["exploit"] in prompt_used

    def test_no_learning_stage_uses_base_prompt_only(self):
        """When learning_stage=None, no stage prompt is prepended."""
        from pb.core.brain import STAGE_PROMPTS

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("question", learning_stage=None)

        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        # None of the stage prompts should be present
        for stage_text in STAGE_PROMPTS.values():
            assert stage_text not in prompt_used

    def test_learning_stage_also_includes_pushback_suffix(self):
        """When learning_stage is set, PUSHBACK_SUFFIX is included in the prompt."""
        from pb.core.brain import PUSHBACK_SUFFIX

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("question", learning_stage="#new")

        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        assert PUSHBACK_SUFFIX in prompt_used


# ---------------------------------------------------------------------------
# WEAK_AREA_SENTINEL constant (Phase 19 D-10)
# ---------------------------------------------------------------------------


class TestWeakAreaSentinel:
    def test_sentinel_equals_exact_string(self):
        """WEAK_AREA_SENTINEL must equal exactly 'This needs more study' for D-10 CLI detection."""
        from pb.core.brain import WEAK_AREA_SENTINEL

        assert WEAK_AREA_SENTINEL == "This needs more study"

    def test_sentinel_is_importable(self):
        """WEAK_AREA_SENTINEL is importable from pb.core.brain."""
        from pb.core.brain import WEAK_AREA_SENTINEL  # noqa: F401 — import test


# ---------------------------------------------------------------------------
# STAGE_PROMPTS constants (Phase 19 D-07)
# ---------------------------------------------------------------------------


class TestStagePrompts:
    def test_stage_prompts_has_all_four_keys(self):
        """STAGE_PROMPTS contains exactly the four required mode keys."""
        from pb.core.brain import STAGE_PROMPTS

        assert set(STAGE_PROMPTS.keys()) == {"explore", "consolidate", "exploit", "re-engage"}

    def test_all_stage_prompts_are_non_empty_strings(self):
        """All STAGE_PROMPTS values are non-empty strings."""
        from pb.core.brain import STAGE_PROMPTS

        for key, value in STAGE_PROMPTS.items():
            assert isinstance(value, str), f"STAGE_PROMPTS[{key!r}] is not a string"
            assert len(value) > 20, f"STAGE_PROMPTS[{key!r}] is suspiciously short"


# ---------------------------------------------------------------------------
# query_constellation (Phase 19 D-08)
# ---------------------------------------------------------------------------


class TestQueryConstellation:
    def test_query_constellation_exists(self):
        """BrainEngine has a query_constellation(slug) method."""
        from pb.core.brain import BrainEngine

        assert hasattr(BrainEngine, "query_constellation")

    def test_returns_unavailable_when_no_llm(self):
        """query_constellation returns 'LLM unavailable' when Gemini not available."""
        engine = _make_brain_engine(available=False)
        result = engine.query_constellation("some-note")
        assert "LLM unavailable" in result

    def test_returns_no_connections_when_neighborhood_empty(self):
        """query_constellation returns 'No connections found' when neighborhood is empty."""
        engine = _make_brain_engine(available=True)

        empty_neighborhood = {"out1": [], "in1": [], "out2": [], "in2": []}
        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch(
                "pb.vault.graph_store.get_hop2_neighborhood",
                return_value=empty_neighborhood,
            ):
                result = engine.query_constellation("some-note")

        assert "No connections found" in result

    def test_uses_hop2_neighborhood_silently(self):
        """query_constellation calls get_hop2_neighborhood (1-hop + 2-hop) without exposing graph to user."""
        from pb.core.brain import CONSTELLATION_PROMPT

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Challenge question"

        fake_neighborhood = {
            "out1": ["note-b"],
            "in1": ["note-c"],
            "out2": ["note-d"],
            "in2": [],
        }

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                with patch(
                    "pb.vault.graph_store.get_hop2_neighborhood",
                    return_value=fake_neighborhood,
                ) as mock_hop2:
                    result = engine.query_constellation("center-note")

        # hop2 was called
        mock_hop2.assert_called_once()
        # CONSTELLATION_PROMPT instructs model NOT to show graph to user
        prompt_used = engine._gemini.generate_with_tools.call_args[0][0]
        assert "Do NOT show the graph to the user" in prompt_used
        assert result == "Challenge question"

    def test_constellation_prompt_instructs_no_graph_reveal(self):
        """CONSTELLATION_PROMPT explicitly tells the LLM never to show the graph."""
        from pb.core.brain import CONSTELLATION_PROMPT

        assert "Do NOT show the graph to the user" in CONSTELLATION_PROMPT


# ---------------------------------------------------------------------------
# get_context_display
# ---------------------------------------------------------------------------


class TestGetContextDisplay:
    def test_returns_none_before_any_query(self):
        """get_context_display returns None when no query has been executed."""
        engine = _make_brain_engine()
        assert engine.get_context_display() is None

    def test_returns_graph_stats_after_query(self):
        """get_context_display returns graph stats after a successful query."""
        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        engine.query("test question")

        display = engine.get_context_display()
        assert display is not None
        assert "Graph:" in display
        assert "Model:" in display


# ---------------------------------------------------------------------------
# auto-escalation
# ---------------------------------------------------------------------------


class TestQueryAutoEscalation:
    def test_auto_escalate_uses_flash_lite_first(self):
        """auto_escalate mode sends first request via Flash Lite."""
        from pb.core.brain import FLASH_LITE_MODEL

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.return_value = "Non-escalation answer"

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question", auto_escalate=True)

        first_call = engine._gemini.generate_with_tools.call_args_list[0]
        assert first_call[0][1] == FLASH_LITE_MODEL
        assert result == "Non-escalation answer"

    def test_auto_escalate_escalates_to_flash_when_requested(self):
        """auto_escalate mode escalates to Flash when LLM responds with ESCALATE: flash."""
        from pb.core.brain import FLASH_MODEL, FLASH_LITE_MODEL

        engine = _make_brain_engine(available=True)
        engine._gemini.generate_with_tools.side_effect = [
            "ESCALATE: flash",
            "Full Flash answer",
        ]

        with patch("pb.core.brain.get_vault_path", return_value=Path("/fake")):
            with patch("pb.core.brain.load_vault_graph", return_value=_FAKE_EDGES):
                with patch(
                    "pb.core.brain.graph_to_adjacency_text",
                    return_value=_FAKE_GRAPH_TEXT,
                ):
                    with patch("pb.core.brain._make_read_fn", return_value=MagicMock()):
                        result = engine.query("question", auto_escalate=True)

        calls = engine._gemini.generate_with_tools.call_args_list
        assert calls[0][0][1] == FLASH_LITE_MODEL
        assert calls[1][0][1] == FLASH_MODEL
        assert result == "Full Flash answer"
