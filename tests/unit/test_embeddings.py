"""Unit tests for EmbeddingStore — sqlite-vec wrapper with hard-fail on missing extension.

Tests 1-5: pure unit tests using mocks (no sqlite-vec required).
Tests 6-8: integration tests skipped when sqlite-vec not installed.
"""
from __future__ import annotations

import builtins
import struct
import sys
import tempfile
import unittest.mock as mock
from pathlib import Path
from types import ModuleType
from unittest.mock import MagicMock, patch

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

sqlite_vec_available = False
try:
    import sqlite3 as _sqlite3
    import sqlite_vec  # noqa: F401
    _conn = _sqlite3.connect(":memory:")
    _conn.enable_load_extension(True)
    _conn.close()
    sqlite_vec_available = True
except (ImportError, AttributeError):
    pass

requires_sqlite_vec = pytest.mark.skipif(
    not sqlite_vec_available,
    reason="sqlite-vec not installed or enable_load_extension not available",
)


def _make_store(vault_path: Path):
    """Create EmbeddingStore bypassing constructor (avoids sqlite_vec check in unit tests)."""
    from pb.vault.embeddings import EmbeddingStore
    store = EmbeddingStore.__new__(EmbeddingStore)
    store._vault_path = vault_path
    store._available = True
    store._dimensions = 768
    return store


def _fake_embedding(dim: int = 768) -> list[float]:
    """Return a normalised unit vector of length dim."""
    val = 1.0 / (dim ** 0.5)
    return [val] * dim


def _mock_import_blocking_sqlite_vec(name, *args, **kwargs):
    """Side-effect for builtins.__import__ that raises ImportError for sqlite_vec."""
    if name == "sqlite_vec":
        raise ImportError("mocked: sqlite_vec not installed")
    return builtins.__import__(name, *args, **kwargs)


# ---------------------------------------------------------------------------
# Test 1: EmbeddingStore raises EmbeddingUnavailableError when sqlite_vec missing
# ---------------------------------------------------------------------------


def test_raises_when_sqlite_vec_missing(tmp_path):
    """EmbeddingStore raises EmbeddingUnavailableError when sqlite_vec cannot be imported (D-06)."""
    from pb.vault.embeddings import EmbeddingStore, EmbeddingUnavailableError

    with mock.patch("builtins.__import__", side_effect=_mock_import_blocking_sqlite_vec):
        with pytest.raises(EmbeddingUnavailableError) as exc_info:
            EmbeddingStore(tmp_path)
    assert "pip install sqlite-vec" in str(exc_info.value)
    assert "pb init --embeddings" in str(exc_info.value)


# ---------------------------------------------------------------------------
# Test 2: EmbeddingUnavailableError on construction — no _available=False state
# ---------------------------------------------------------------------------


def test_store_embedding_raises_when_called_without_sqlite_vec(tmp_path):
    """Constructing EmbeddingStore raises EmbeddingUnavailableError — no no-op state (D-06)."""
    from pb.vault.embeddings import EmbeddingStore, EmbeddingUnavailableError

    with mock.patch("builtins.__import__", side_effect=_mock_import_blocking_sqlite_vec):
        with pytest.raises(EmbeddingUnavailableError):
            EmbeddingStore(tmp_path)


# ---------------------------------------------------------------------------
# Test 3: EmbeddingUnavailableError on construction — query_similarity path
# ---------------------------------------------------------------------------


def test_query_similarity_raises_when_called_without_sqlite_vec(tmp_path):
    """Constructing EmbeddingStore raises EmbeddingUnavailableError — no empty-list fallback (D-06)."""
    from pb.vault.embeddings import EmbeddingStore, EmbeddingUnavailableError

    with mock.patch("builtins.__import__", side_effect=_mock_import_blocking_sqlite_vec):
        with pytest.raises(EmbeddingUnavailableError):
            EmbeddingStore(tmp_path)


# ---------------------------------------------------------------------------
# Test 4: _get_embedding returns None when Gemini client is unavailable
# ---------------------------------------------------------------------------


def test_get_embedding_returns_none_when_gemini_unavailable(tmp_path):
    """_get_embedding() returns None when GeminiClient.is_available() is False."""
    store = _make_store(tmp_path)

    mock_client = MagicMock()
    mock_client.is_available.return_value = False

    with patch("pb.vault.embeddings.EmbeddingStore._get_embedding", wraps=store._get_embedding):
        with patch("pb.llm.gemini.get_client", return_value=mock_client):
            result = store._get_embedding("hello world")
    assert result is None


# ---------------------------------------------------------------------------
# Test 5: _get_embedding returns list[float] of length 768 when Gemini responds
# ---------------------------------------------------------------------------


def test_get_embedding_returns_vector_when_gemini_available(tmp_path):
    """_get_embedding() returns a float list of length 768 from mocked Gemini."""
    store = _make_store(tmp_path)

    fake_vec = _fake_embedding(768)

    # Build mock response chain: response.embeddings[0].values
    mock_embedding_obj = MagicMock()
    mock_embedding_obj.values = fake_vec

    mock_response = MagicMock()
    mock_response.embeddings = [mock_embedding_obj]

    mock_inner_client = MagicMock()
    mock_inner_client.models.embed_content.return_value = mock_response

    mock_gemini_client = MagicMock()
    mock_gemini_client.is_available.return_value = True
    mock_gemini_client._client = mock_inner_client

    # _get_embedding uses lazy `from pb.llm.gemini import get_client` at call time,
    # so patch at the definition site in pb.llm.gemini.
    with patch("pb.llm.gemini.get_client", return_value=mock_gemini_client):
        result = store._get_embedding("hello world")

    assert result is not None
    assert isinstance(result, list)
    assert len(result) == 768
    assert all(isinstance(v, float) for v in result)
    # Verify embed_content was called with correct model
    mock_inner_client.models.embed_content.assert_called_once()
    call_kwargs = mock_inner_client.models.embed_content.call_args
    assert call_kwargs.kwargs["model"] == "gemini-embedding-001"


# ---------------------------------------------------------------------------
# Test 6: ensure_schema creates vec_note_embeddings virtual table
# (requires sqlite-vec installed)
# ---------------------------------------------------------------------------


@requires_sqlite_vec
def test_ensure_schema_creates_virtual_table(tmp_path):
    """ensure_schema() creates the vec_note_embeddings virtual table."""
    from pb.vault.embeddings import EmbeddingStore
    store = EmbeddingStore(tmp_path)
    if not store._available:
        pytest.skip("sqlite-vec not available at runtime")

    store.ensure_schema()

    conn = store._open_conn()
    try:
        tables = {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' OR type='shadow'"
        ).fetchall()}
        # vec_note_embeddings should appear (possibly with shadow tables)
        assert any("vec_note_embeddings" in t for t in tables), (
            f"vec_note_embeddings not found in {tables}"
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test 7: store_embedding + query_similarity round-trip
# (requires sqlite-vec installed)
# ---------------------------------------------------------------------------


@requires_sqlite_vec
def test_store_and_query_round_trip(tmp_path):
    """store_embedding + query_similarity returns the stored slug with high similarity."""
    from pb.vault.embeddings import EmbeddingStore
    store = EmbeddingStore(tmp_path)
    if not store._available:
        pytest.skip("sqlite-vec not available at runtime")

    store.ensure_schema()

    slug = "my-note-slug"
    fake_vec = _fake_embedding(768)

    # Patch _get_embedding to return known vector
    with patch.object(store, "_get_embedding", return_value=fake_vec):
        store.store_embedding(slug, "test content")

    with patch.object(store, "_get_embedding", return_value=fake_vec):
        results = store.query_similarity("test content", k=5)

    assert len(results) > 0
    slugs = [r[0] for r in results]
    assert slug in slugs
    # Similarity to itself should be very high
    slug_sim = next(sim for s, sim in results if s == slug)
    assert slug_sim > 0.9


# ---------------------------------------------------------------------------
# Test 8: find_redundant_pairs returns pairs above threshold
# (requires sqlite-vec installed)
# ---------------------------------------------------------------------------


@requires_sqlite_vec
def test_find_redundant_pairs_returns_high_similarity_pairs(tmp_path):
    """find_redundant_pairs() returns pairs with cosine similarity > threshold."""
    from pb.vault.embeddings import EmbeddingStore
    store = EmbeddingStore(tmp_path)
    if not store._available:
        pytest.skip("sqlite-vec not available at runtime")

    store.ensure_schema()

    # Two nearly identical vectors should be flagged as redundant
    vec_a = _fake_embedding(768)
    vec_b = _fake_embedding(768)  # identical direction

    with patch.object(store, "_get_embedding", return_value=vec_a):
        store.store_embedding("note-a", "content a")

    with patch.object(store, "_get_embedding", return_value=vec_b):
        store.store_embedding("note-b", "content b")

    pairs = store.find_redundant_pairs(["note-a", "note-b"], threshold=0.5)

    assert len(pairs) > 0
    pair_slugs = {(p[0], p[1]) for p in pairs}
    assert ("note-a", "note-b") in pair_slugs or ("note-b", "note-a") in pair_slugs
    # Similarity should be above threshold for identical vectors
    for slug_a, slug_b, sim in pairs:
        assert sim > 0.5
