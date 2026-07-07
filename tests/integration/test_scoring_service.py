"""Integration tests for ScoringService — verifies service contracts used by brain.py CLI.

Tests mock external dependencies (BrainEngine, GeminiClient, sqlite-vec) so they run
without network access or installed extensions.
"""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


def _make_vault(tmp_path: Path) -> Path:
    """Create minimal vault structure with one domain and one note."""
    domain_dir = tmp_path / "knowledge" / "piano"
    domain_dir.mkdir(parents=True)
    (domain_dir / "_state.md").write_text(
        "---\nlast_activity: 2026-01-01\n---\n"
    )
    (domain_dir / "circle-of-fifths.md").write_text(
        "---\nlearning_stage: '#learning'\n---\n\nContent about circle of fifths."
    )
    # Create vault.db stub (tests don't need real DB, but ScoringService path-checks)
    (tmp_path / "vault.db").touch()
    return tmp_path


class TestScoringService:

    def test_rank_and_query_returns_answer_string(self, tmp_path):
        """rank_and_query delegates to BrainEngine and returns answer string."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()
        svc._brain = None

        mock_brain = MagicMock()
        mock_brain.query.return_value = "test answer about piano"
        mock_brain.get_context_display.return_value = "flash-lite"
        svc._brain = mock_brain

        # Mock scorer to avoid embedding dependency (Exception is caught internally, not fatal)
        with patch.object(ScoringService, '_get_scorer', side_effect=Exception("no scorer")):
            result = svc.rank_and_query("what is piano")

        assert result["answer"] == "test answer about piano"
        assert result["context_display"] == "flash-lite"
        assert isinstance(result["signal_data"], list)
        assert result["constellation"] is None  # not verbose

    def test_rank_and_query_returns_signal_data_list(self, tmp_path):
        """rank_and_query returns signal_data as a list."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()

        mock_brain = MagicMock()
        mock_brain.query.return_value = "an answer"
        mock_brain.get_context_display.return_value = ""
        svc._brain = mock_brain

        with patch.object(ScoringService, '_get_scorer', side_effect=Exception("no scorer")):
            result = svc.rank_and_query("neural networks")

        assert isinstance(result["signal_data"], list)

    def test_rank_and_query_result_has_required_keys(self, tmp_path):
        """rank_and_query result dict has all required keys."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()

        mock_brain = MagicMock()
        mock_brain.query.return_value = "answer"
        mock_brain.get_context_display.return_value = "context"
        svc._brain = mock_brain

        with patch.object(ScoringService, '_get_scorer', side_effect=Exception("no scorer")):
            result = svc.rank_and_query("piano scales")

        required_keys = {"answer", "signal_data", "top_slug", "context_display", "constellation"}
        assert required_keys.issubset(result.keys()), (
            f"Missing keys: {required_keys - result.keys()}"
        )

    def test_rank_and_query_verbose_includes_constellation_keys(self, tmp_path):
        """rank_and_query with verbose=True returns result with 'constellation' key present."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()

        mock_brain = MagicMock()
        mock_brain.query.return_value = "answer"
        mock_brain.get_context_display.return_value = ""
        svc._brain = mock_brain

        fake_constellation = {"out1": ["note-b"], "in1": [], "out2": ["note-c"], "in2": []}

        with patch.object(ScoringService, '_get_scorer', side_effect=Exception("no scorer")), \
             patch.object(ScoringService, 'get_constellation', return_value=fake_constellation):
            result = svc.rank_and_query("piano", verbose=True)

        # The constellation key must always be present in the result dict
        assert "constellation" in result
        assert "answer" in result

    def test_detect_gaps_structure_and_flash_call(self, tmp_path):
        """detect_gaps calls Flash one-shot and returns have/missing structure."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()
        svc._brain = None

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = '["voice-leading", "counterpoint", "sight-reading"]'

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = [("circle-of-fifths",)]

        with patch("pb.vault.graph_store.open_vault_db", return_value=mock_conn), \
             patch("pb.llm.gemini.get_client", return_value=mock_client):
            result = svc.detect_gaps("piano")

        assert result["domain"] == "piano"
        assert isinstance(result["have"], list)
        assert isinstance(result["missing"], list)
        assert "voice-leading" in result["missing"]
        assert "counterpoint" in result["missing"]

    def test_detect_gaps_handles_non_json_flash_response(self, tmp_path):
        """detect_gaps returns empty missing list when Flash returns prose (Pitfall 6)."""
        vault = _make_vault(tmp_path)

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()
        svc._brain = None

        mock_client = MagicMock()
        mock_client.is_available.return_value = True
        mock_client.generate_with_model.return_value = (
            "Here are some missing concepts: voice leading and counterpoint."
        )

        mock_conn = MagicMock()
        mock_conn.execute.return_value.fetchall.return_value = []

        with patch("pb.vault.graph_store.open_vault_db", return_value=mock_conn), \
             patch("pb.llm.gemini.get_client", return_value=mock_client):
            result = svc.detect_gaps("piano")

        # Non-JSON prose falls back to empty (re.search found no array)
        assert result["missing"] == []
        assert result["domain"] == "piano"

    def test_get_constellation_returns_hop2_dict(self, tmp_path):
        """get_constellation delegates to get_hop2_neighborhood and returns correct keys."""
        vault = _make_vault(tmp_path)

        fake_hood = {"out1": ["note-b"], "in1": ["note-c"], "out2": [], "in2": []}

        from pb.vault.scoring_service import ScoringService
        svc = ScoringService.__new__(ScoringService)
        svc.vault_path = vault
        import structlog
        svc._log = structlog.get_logger()
        svc._brain = None

        with patch("pb.vault.graph_store.get_hop2_neighborhood", return_value=fake_hood):
            result = svc.get_constellation("circle-of-fifths")

        assert result == fake_hood
        assert all(k in result for k in ("out1", "in1", "out2", "in2"))

    def test_embedding_unavailable_error_importable(self):
        """EmbeddingUnavailableError is importable from expected module path."""
        from pb.vault.embeddings import EmbeddingUnavailableError
        assert issubclass(EmbeddingUnavailableError, Exception)

    def test_embedding_store_raises_on_missing_sqlite_vec(self, tmp_path):
        """EmbeddingStore raises EmbeddingUnavailableError when sqlite_vec not available."""
        import sys
        from pb.vault.embeddings import EmbeddingUnavailableError
        with patch.dict(sys.modules, {"sqlite_vec": None}):
            import importlib
            import pb.vault.embeddings as emb_mod
            importlib.reload(emb_mod)
            with pytest.raises(emb_mod.EmbeddingUnavailableError):
                emb_mod.EmbeddingStore(tmp_path)
