# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Composite scoring engine -- deterministic 7-signal formula. No LLM in this path.

Per D-04: Pre-rank then LLM reads top-N. CompositeScorer ranks all vault notes.
Top 15-20 passed to BrainEngine's LLM context.
"""
from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import structlog

from pb.storage.config import get_config
from pb.vault.lifecycle import get_weighted_total, read_frontmatter

logger = structlog.get_logger()


# Stage-aware retrieval modes per D-07
STAGE_MODES = {
    "#new": "explore",
    "#learning": "consolidate",
    "#learnt": "exploit",
    "#stale": "re-engage",
}

MODE_STAGES = {
    "explore": ["#new"],
    "consolidate": ["#learning"],
    "exploit": ["#learnt"],
    "re-engage": ["#stale"],
}


@dataclass
class ScoreSignals:
    """Individual signal values for a scored note."""

    semantic_similarity: float = 0.0
    link_strength: float = 0.0
    backlink_strength: float = 0.0
    tag_affinity: float = 0.0
    recency: float = 0.0
    usage: float = 0.0
    redundancy_penalty: float = 0.0
    novelty_boost: float = 0.0


def composite_score(signals: ScoreSignals, weights: dict[str, float]) -> float:
    """Compute weighted sum of signals. Novelty boost is additive, not weighted (D-05)."""
    return (
        weights.get("semantic", 0.3) * signals.semantic_similarity
        + weights.get("link", 0.15) * signals.link_strength
        + weights.get("backlink", 0.1) * signals.backlink_strength
        + weights.get("tag_affinity", 0.1) * signals.tag_affinity
        + weights.get("recency", 0.15) * signals.recency
        + weights.get("usage", 0.1) * signals.usage
        + weights.get("redundancy", -0.1) * signals.redundancy_penalty
        + signals.novelty_boost  # additive per D-05
    )


def _compute_novelty_boost(created_date: Optional[str]) -> float:
    """Linear decay from 1.0 at day 0 to 0.0 at day 7 (D-05)."""
    if not created_date:
        return 0.0
    try:
        created = datetime.date.fromisoformat(str(created_date))
        days_old = (datetime.date.today() - created).days
        return max(0.0, (7 - days_old) / 7)
    except (ValueError, TypeError):
        return 0.0


def _compute_recency(last_interaction_iso: Optional[str]) -> float:
    """Decay from 1.0 (today) toward 0.0. Uses 30-day half-life."""
    if not last_interaction_iso:
        return 0.0
    try:
        last = datetime.date.fromisoformat(str(last_interaction_iso)[:10])
        days_ago = (datetime.date.today() - last).days
        return math.exp(-0.693 * days_ago / 30)  # half-life = 30 days
    except (ValueError, TypeError):
        return 0.0


def _compute_tag_affinity(note_tags: list[str], query_tags: list[str]) -> float:
    """Proportion of query_tags that appear in note_tags."""
    if not query_tags:
        return 0.0
    note_set = set(t.lower().strip("#") for t in note_tags)
    query_set = set(t.lower().strip("#") for t in query_tags)
    if not query_set:
        return 0.0
    return len(note_set & query_set) / len(query_set)


def _load_domain_weight_overrides(vault_path: Path, domain: str) -> dict[str, float]:
    """Load scoring_weights overrides from domain _state.md frontmatter (per D-02).

    Domain _state.md can include a `scoring_weights` key in its YAML frontmatter
    to override specific global defaults. Example _state.md frontmatter:
        ---
        scoring_weights:
          semantic: 0.5
          recency: 0.05
        ---
    Only keys present in _state.md override; missing keys keep global defaults.

    Returns partial dict of overrides (may be empty).
    """
    try:
        state_path = vault_path / "knowledge" / domain / "_state.md"
        if not state_path.exists():
            return {}
        content = state_path.read_text()
        fm, _ = read_frontmatter(content)
        overrides = fm.get("scoring_weights", {})
        if isinstance(overrides, dict):
            return {k: float(v) for k, v in overrides.items() if isinstance(v, (int, float))}
        return {}
    except Exception:
        logger.debug("scorer.domain_weight_override_failed", domain=domain)
        return {}


class CompositeScorer:
    """Deterministic 7-signal composite scorer. No LLM in this path."""

    def __init__(self, vault_path: Path) -> None:
        self._vault_path = vault_path
        config = get_config()
        self._global_weights = config.learning.scoring_weights
        self._top_n = config.learning.top_n_candidates
        self._embedding_store = self._init_embedding_store()

    def _init_embedding_store(self):
        from pb.vault.embeddings import EmbeddingStore, EmbeddingUnavailableError
        try:
            store = EmbeddingStore(self._vault_path)
            store.ensure_schema()
            return store
        except EmbeddingUnavailableError:
            raise  # propagate so CLI can show D-07 message
        except Exception:
            pass   # non-embedding failures degrade gracefully
        return None

    def _resolve_weights(self, domain: Optional[str]) -> dict[str, float]:
        """Resolve effective scoring weights: global defaults merged with domain _state.md overrides (D-02).

        When domain filter is active, reads _state.md scoring_weights and merges
        (domain overrides take precedence over global defaults).
        When no domain filter, returns global defaults unchanged.
        """
        if not domain:
            return self._global_weights
        overrides = _load_domain_weight_overrides(self._vault_path, domain)
        if not overrides:
            return self._global_weights
        # Merge: global defaults + domain overrides (overrides win)
        merged = dict(self._global_weights)
        merged.update(overrides)
        return merged

    def rank_notes(
        self,
        query_text: Optional[str] = None,
        domain: Optional[str] = None,
        stage_mode: Optional[str] = None,
        query_tags: Optional[list[str]] = None,
    ) -> list[tuple[str, float, ScoreSignals]]:
        """Return top-N notes ranked by composite score.

        Args:
            query_text: User query for semantic similarity signal.
            domain: Filter to notes in this subfolder only. Also activates
                    domain-specific weight overrides from _state.md per D-02.
            stage_mode: One of "explore"/"consolidate"/"exploit"/"re-engage".
            query_tags: Tags for tag_affinity signal.

        Returns:
            [(note_slug, score, signals), ...] sorted desc, limited to top_n.
        """
        # Resolve weights: global + domain _state.md overrides per D-02
        weights = self._resolve_weights(domain)

        try:
            from pb.vault.graph_store import open_vault_db
            conn = open_vault_db(self._vault_path)
        except Exception:
            logger.debug("scorer.vault_db_unavailable")
            return []

        try:
            # 1. Load all candidate nodes
            candidates = self._load_candidates(conn, domain, stage_mode)
            if not candidates:
                return []

            # 2. Compute semantic similarity signal
            semantic_map = self._compute_semantic_map(query_text, candidates)

            # 3. Compute link/backlink signals
            link_map, backlink_map = self._compute_link_signals(conn, candidates)

            # 4. Compute per-note signals and score
            scored: list[tuple[str, ScoreSignals, dict]] = []
            for slug, note_info in candidates.items():
                signals = ScoreSignals(
                    semantic_similarity=semantic_map.get(slug, 0.0),
                    link_strength=link_map.get(slug, 0.0),
                    backlink_strength=backlink_map.get(slug, 0.0),
                    tag_affinity=_compute_tag_affinity(
                        note_info.get("tags", []), query_tags or []
                    ),
                    recency=_compute_recency(note_info.get("last_interaction")),
                    usage=0.0,  # normalized below
                    redundancy_penalty=0.0,  # computed below
                    novelty_boost=_compute_novelty_boost(note_info.get("created")),
                )
                scored.append((slug, signals, note_info))

            # 5. Normalize usage signal across candidates
            self._normalize_usage(scored)

            # 6. Apply redundancy penalty (D-03)
            self._apply_redundancy(scored)

            # 7. Compute final scores using resolved weights (global + domain overrides per D-02)
            results: list[tuple[str, float, ScoreSignals]] = []
            for slug, signals, _ in scored:
                score = composite_score(signals, weights)
                results.append((slug, score, signals))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[: self._top_n]

        finally:
            conn.close()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_candidates(
        self,
        conn,
        domain: Optional[str],
        stage_mode: Optional[str],
    ) -> dict[str, dict]:
        """Load candidate notes from vault.db, optionally filtered by domain and stage.

        Returns dict mapping slug -> note_info dict with keys:
            subfolder, created, last_interaction, tags, learning_stage
        """
        if domain:
            # Filter to notes in the specified domain subfolder
            subfolder_pattern = f"knowledge/{domain}"
            rows = conn.execute(
                "SELECT slug, subfolder FROM nodes WHERE subfolder = ?",
                (subfolder_pattern,),
            ).fetchall()
        else:
            rows = conn.execute("SELECT slug, subfolder FROM nodes").fetchall()

        candidates: dict[str, dict] = {}
        allowed_stages: Optional[list[str]] = None
        if stage_mode and stage_mode in MODE_STAGES:
            allowed_stages = MODE_STAGES[stage_mode]

        for slug, subfolder in rows:
            note_info = self._read_note_info(slug, subfolder)

            # Stage-mode filter: check learning_stage in frontmatter
            if allowed_stages is not None:
                note_stage = note_info.get("learning_stage", "")
                if note_stage not in allowed_stages:
                    continue

            candidates[slug] = note_info

        return candidates

    def _read_note_info(self, slug: str, subfolder: str) -> dict:
        """Read frontmatter from note file to extract created, tags, learning_stage."""
        info: dict = {
            "subfolder": subfolder,
            "learning_stage": "",
            "tags": [],
            "created": None,
            "last_interaction": None,
        }
        try:
            note_file = self._vault_path / subfolder / f"{slug}.md"
            if not note_file.exists():
                return info
            content = note_file.read_text()
            fm, _ = read_frontmatter(content)
            info["learning_stage"] = fm.get("learning_stage", "")
            info["tags"] = fm.get("tags", []) or []
            info["created"] = fm.get("created") or fm.get("date")
            info["last_interaction"] = fm.get("stage_updated")
        except Exception:
            pass
        return info

    def _compute_semantic_map(
        self, query_text: Optional[str], candidates: dict[str, dict]
    ) -> dict[str, float]:
        """Compute semantic similarity for all candidates.

        Returns {slug: similarity} dict. Returns empty dict if no embedding store
        or no query_text (graceful degradation).
        """
        if not self._embedding_store or not query_text:
            return {}
        try:
            results = self._embedding_store.query_similarity(query_text, k=len(candidates) + 20)
            # Filter to candidates only
            return {
                slug: sim
                for slug, sim in results
                if slug in candidates
            }
        except Exception:
            return {}

    def _compute_link_signals(
        self, conn, candidates: dict[str, dict]
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Compute normalized link_strength and backlink_strength for all candidates.

        link_strength   = outgoing_links / max_outgoing across candidates
        backlink_strength = incoming_links / max_incoming across candidates

        Returns (link_map, backlink_map) dicts mapping slug -> float [0, 1].
        """
        slugs = list(candidates.keys())
        if not slugs:
            return {}, {}

        # Count outgoing links (only where dst is also a candidate)
        out_counts: dict[str, int] = {s: 0 for s in slugs}
        in_counts: dict[str, int] = {s: 0 for s in slugs}

        slug_set = set(slugs)
        for slug in slugs:
            # Outgoing: links where src = slug and dst in candidates
            out_rows = conn.execute(
                "SELECT COUNT(*) FROM links WHERE src = ?", (slug,)
            ).fetchone()
            out_counts[slug] = out_rows[0] if out_rows else 0

            # Incoming: links where dst = slug and src in candidates
            in_rows = conn.execute(
                "SELECT COUNT(*) FROM links WHERE dst = ?", (slug,)
            ).fetchone()
            in_counts[slug] = in_rows[0] if in_rows else 0

        max_out = max(out_counts.values(), default=0)
        max_in = max(in_counts.values(), default=0)

        link_map = {
            slug: (out_counts[slug] / max_out if max_out > 0 else 0.0)
            for slug in slugs
        }
        backlink_map = {
            slug: (in_counts[slug] / max_in if max_in > 0 else 0.0)
            for slug in slugs
        }

        return link_map, backlink_map

    def _normalize_usage(self, scored: list[tuple[str, ScoreSignals, dict]]) -> None:
        """Normalize usage signal in-place: divide by max usage across candidates."""
        usages = []
        for slug, signals, _ in scored:
            total = get_weighted_total(slug)
            usages.append(total)

        max_usage = max(usages, default=0.0)
        for i, (slug, signals, info) in enumerate(scored):
            if max_usage > 0:
                signals.usage = usages[i] / max_usage
            else:
                signals.usage = 0.0

    def _apply_redundancy(self, scored: list[tuple[str, ScoreSignals, dict]]) -> None:
        """Apply redundancy_penalty = 1.0 to the lower-scoring note in each redundant pair.

        Uses embedding store find_redundant_pairs(threshold=0.85).
        For each redundant pair, the note with lower current partial score gets penalized.
        No-op if no embedding store available.
        """
        if not self._embedding_store:
            return

        slugs = [slug for slug, _, _ in scored]
        try:
            pairs = self._embedding_store.find_redundant_pairs(slugs, threshold=0.85)
        except Exception:
            return

        if not pairs:
            return

        # Build a quick lookup: slug -> (index, signals)
        slug_index: dict[str, int] = {slug: i for i, (slug, _, _) in enumerate(scored)}

        for slug_a, slug_b, _ in pairs:
            idx_a = slug_index.get(slug_a)
            idx_b = slug_index.get(slug_b)
            if idx_a is None or idx_b is None:
                continue

            # Compute partial scores to determine which is lower
            _, sig_a, _ = scored[idx_a]
            _, sig_b, _ = scored[idx_b]
            score_a = sig_a.usage + sig_a.recency + sig_a.link_strength
            score_b = sig_b.usage + sig_b.recency + sig_b.link_strength

            # Penalize the lower-scoring note
            if score_a <= score_b:
                sig_a.redundancy_penalty = 1.0
            else:
                sig_b.redundancy_penalty = 1.0
