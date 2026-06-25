# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Embedding store — sqlite-vec wrapper for semantic similarity in vault.db.

Hard-fail on missing sqlite-vec: raises EmbeddingUnavailableError on construction
so the CLI can surface D-07 setup instructions to the user.

Threat mitigations (T-19-01 through T-19-03):
  T-19-01: Input truncated to 2048 chars before sending to Gemini API.
  T-19-02: try/except around every embed call; silent no-op on failure; never retry in loop.
  T-19-03: enable_load_extension(True) immediately disabled after loading extension.
"""
from __future__ import annotations

import sqlite3
import struct
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class EmbeddingUnavailableError(Exception):
    """Raised when sqlite-vec extension is not installed."""
    pass


_VIRTUAL_TABLE_DDL = """
CREATE VIRTUAL TABLE IF NOT EXISTS vec_note_embeddings
USING vec0(
    note_slug TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);

CREATE VIRTUAL TABLE IF NOT EXISTS vec_thought_embeddings
USING vec0(
    thought_id TEXT PRIMARY KEY,
    embedding float[768] distance_metric=cosine
);
"""


def _serialize_f32(vector: list[float]) -> bytes:
    """Pack float list into bytes for sqlite-vec."""
    return struct.pack(f"{len(vector)}f", *vector)


def _deserialize_f32(raw: bytes | memoryview) -> list[float]:
    """Unpack sqlite-vec bytes into a Python float list."""
    payload = bytes(raw)
    n = len(payload) // 4
    return list(struct.unpack(f"{n}f", payload))


class EmbeddingStore:
    """Thin wrapper around sqlite-vec virtual table in vault.db.

    Opens a SEPARATE connection to vault.db with extension loading enabled.
    Does NOT use open_vault_db() from graph_store (no extension support there).
    """

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        self._check_available()   # raises EmbeddingUnavailableError if not installed
        self._available = True    # always True past this point
        self._dimensions: int = 768  # default; overridden by config if needed

    def _check_available(self) -> None:
        """Raises EmbeddingUnavailableError if sqlite-vec is not installed (D-06/D-07)."""
        try:
            import sqlite_vec  # noqa: F401
        except ImportError:
            raise EmbeddingUnavailableError(
                "Embeddings unavailable. Run:\n"
                "  pip install sqlite-vec\n"
                "  pb init --embeddings"
            )

    @property
    def available(self) -> bool:
        return self._available

    def _open_conn(self) -> sqlite3.Connection:
        """Open vault.db connection with sqlite-vec extension loaded.

        T-19-03: enable_load_extension(True) is set only for the duration
        of the extension load, then immediately disabled.
        """
        import sqlite_vec
        db_path = self._vault_path / "vault.db"
        conn = sqlite3.connect(str(db_path))
        conn.enable_load_extension(True)
        sqlite_vec.load(conn)
        conn.enable_load_extension(False)
        # Apply same PRAGMAs as vault.db
        conn.executescript("""
            PRAGMA journal_mode = WAL;
            PRAGMA synchronous  = NORMAL;
            PRAGMA temp_store   = MEMORY;
            PRAGMA cache_size   = -8000;
        """)
        return conn

    def ensure_schema(self) -> None:
        """Create vec_note_embeddings virtual table if not exists."""
        try:
            conn = self._open_conn()
            try:
                conn.executescript(_VIRTUAL_TABLE_DDL)
                conn.commit()
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.ensure_schema_failed", error=str(exc))
            self._available = False

    def _get_embedding(self, text: str) -> Optional[list[float]]:
        """Call gemini-embedding-001 and return float vector. Returns None on failure.

        T-19-01: Input truncated to 2048 chars before sending to Gemini API.
        T-19-02: Wrapped in try/except; never retries.
        """
        try:
            from pb.llm.gemini import get_client
            client = get_client()
            if not client.is_available():
                return None
            from google.genai import types
            response = client._client.models.embed_content(
                model="gemini-embedding-001",
                contents=text[:2048],  # T-19-01: truncate at 2048 chars
                config=types.EmbedContentConfig(
                    task_type="SEMANTIC_SIMILARITY",
                    output_dimensionality=self._dimensions,
                ),
            )
            return response.embeddings[0].values
        except Exception as exc:
            logger.debug("embeddings.get_embedding_failed", error=str(exc))
            return None

    def store_embedding(self, note_slug: str, text: str) -> None:
        """Generate and store embedding for a note. Silent no-op on any failure.

        T-19-02: All exceptions are caught; no retry loop.
        """
        try:
            embedding = self._get_embedding(text)
            if embedding is None:
                return
            self.ensure_schema()   # ensure schema before opening the write connection
            conn = self._open_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_note_embeddings(note_slug, embedding) VALUES (?, ?)",
                    [note_slug, _serialize_f32(embedding)],
                )
                conn.commit()
                logger.info("embeddings.stored", slug=note_slug)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.store_failed", slug=note_slug, error=str(exc))

    def store_thought_embedding(self, thought_id: str, text: str) -> None:
        """Generate and store embedding for an enriched thought."""
        try:
            embedding = self._get_embedding(text)
            if embedding is None:
                return
            self.ensure_schema()
            conn = self._open_conn()
            try:
                conn.execute(
                    "INSERT OR REPLACE INTO vec_thought_embeddings(thought_id, embedding) VALUES (?, ?)",
                    [thought_id, _serialize_f32(embedding)],
                )
                conn.commit()
                logger.info("embeddings.thought_stored", thought_id=thought_id)
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.thought_store_failed", thought_id=thought_id, error=str(exc))

    def query_similarity(self, query_text: str, k: int = 20) -> list[tuple[str, float]]:
        """Return [(slug, cosine_similarity), ...] sorted by similarity desc.

        Returns empty list if sqlite-vec unavailable or query fails.
        cosine distance -> similarity: similarity = 1 - distance
        """
        try:
            query_embedding = self._get_embedding(query_text)
            if query_embedding is None:
                return []
            conn = self._open_conn()
            try:
                self.ensure_schema()
                rows = conn.execute(
                    """SELECT note_slug, distance
                       FROM vec_note_embeddings
                       WHERE embedding MATCH ?
                       ORDER BY distance LIMIT ?""",
                    [_serialize_f32(query_embedding), k],
                ).fetchall()
                # cosine distance -> similarity: similarity = 1 - distance
                return [(slug, 1.0 - dist) for slug, dist in rows]
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.query_failed", error=str(exc))
            return []

    def get_thought_embedding(self, thought_id: str) -> Optional[list[float]]:
        """Return a stored thought embedding, or None when unavailable."""
        try:
            self.ensure_schema()
            conn = self._open_conn()
            try:
                row = conn.execute(
                    "SELECT embedding FROM vec_thought_embeddings WHERE thought_id = ?",
                    [thought_id],
                ).fetchone()
                if row is None:
                    return None
                return _deserialize_f32(row[0])
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.get_thought_embedding_failed", thought_id=thought_id, error=str(exc))
            return None

    def find_similar_thought_pairs(
        self,
        thought_ids: list[str],
        threshold: float = 0.72,
    ) -> list[tuple[str, str, float]]:
        """Find thought pairs with cosine similarity greater than threshold."""
        if len(thought_ids) < 2:
            return []
        try:
            conn = self._open_conn()
            try:
                self.ensure_schema()
                pairs: list[tuple[str, str, float]] = []
                seen: set[tuple[str, str]] = set()
                known = set(thought_ids)
                for thought_id in thought_ids:
                    row = conn.execute(
                        "SELECT embedding FROM vec_thought_embeddings WHERE thought_id = ?",
                        [thought_id],
                    ).fetchone()
                    if row is None:
                        continue
                    vec = _deserialize_f32(row[0])
                    results = conn.execute(
                        """SELECT thought_id, distance
                           FROM vec_thought_embeddings
                           WHERE embedding MATCH ? AND k = ?
                           ORDER BY distance""",
                        [_serialize_f32(vec), len(thought_ids)],
                    ).fetchall()
                    for other_id, dist in results:
                        if other_id == thought_id or other_id not in known:
                            continue
                        sim = 1.0 - dist
                        if sim <= threshold:
                            continue
                        key = tuple(sorted((thought_id, other_id)))
                        if key in seen:
                            continue
                        seen.add(key)
                        pairs.append((key[0], key[1], sim))
                return pairs
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.find_similar_thought_pairs_failed", error=str(exc))
            return []

    def find_redundant_pairs(
        self, slugs: list[str], threshold: float = 0.85
    ) -> list[tuple[str, str, float]]:
        """Find pairs of notes with cosine similarity > threshold.

        Used for redundancy penalty in composite scorer (D-03).
        Returns [(slug_a, slug_b, similarity), ...].
        """
        if len(slugs) < 2:
            return []
        try:
            conn = self._open_conn()
            try:
                self.ensure_schema()
                pairs: list[tuple[str, str, float]] = []
                seen: set[tuple[str, str]] = set()
                for slug_a in slugs:
                    row_a = conn.execute(
                        "SELECT embedding FROM vec_note_embeddings WHERE note_slug = ?",
                        [slug_a],
                    ).fetchone()
                    if row_a is None:
                        continue
                    vec = _deserialize_f32(row_a[0])
                    results = conn.execute(
                        """SELECT note_slug, distance
                           FROM vec_note_embeddings
                           WHERE embedding MATCH ? AND k = ?
                           ORDER BY distance""",
                        [_serialize_f32(vec), len(slugs)],
                    ).fetchall()
                    results = [(s, d) for s, d in results if s != slug_a]
                    for slug_b, dist in results:
                        sim = 1.0 - dist
                        if sim > threshold and slug_b in slugs:
                            key = tuple(sorted([slug_a, slug_b]))
                            if key not in seen:
                                seen.add(key)
                                pairs.append((key[0], key[1], sim))
                return pairs
            finally:
                conn.close()
        except Exception as exc:
            logger.debug("embeddings.redundancy_check_failed", error=str(exc))
            return []
