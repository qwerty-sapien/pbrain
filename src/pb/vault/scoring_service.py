# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""ScoringService — wraps CompositeScorer, BrainEngine, gap detection, and constellation view.

INV-4: this module never imports rich or typer. All Rich rendering is handled by CLI callers.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import structlog

from pb.core.base import BaseService, LoggableMixin
from pb.storage.yaml_io import extract_structured_yaml


class ScoringService(BaseService, LoggableMixin):
    """Service-layer orchestrator for composite scoring, gap detection, and constellation.

    All public methods return plain Python dicts/lists — no Rich or typer in this module.
    """

    def __init__(self, vault_path: Path) -> None:
        super().__init__()
        self.vault_path = vault_path
        self._log = structlog.get_logger()
        self._brain: Optional[Any] = None  # lazy init — avoids eager Gemini API key check

    # --- Private helpers ---

    def _get_brain(self) -> Any:
        """Lazy BrainEngine init (A3 from RESEARCH.md: avoids eager API key check)."""
        if self._brain is None:
            from pb.core.brain import BrainEngine
            self._brain = BrainEngine()
        return self._brain

    def _get_scorer(self):
        """Return a fresh CompositeScorer for this vault path."""
        from pb.core.scorer import CompositeScorer
        return CompositeScorer(self.vault_path)

    def _read_stage_map(self, domain: str, slugs: list[str]) -> dict[str, str]:
        """Return {slug: learning_stage} for a set of slugs in a domain."""
        from pb.vault.lifecycle import read_frontmatter
        domain_path = self.vault_path / "knowledge" / domain
        stage_map: dict[str, str] = {}
        for slug in slugs:
            for ext in (".md",):
                note_path = domain_path / f"{slug}{ext}"
                if note_path.exists():
                    try:
                        content = note_path.read_text()
                        fm, _ = read_frontmatter(content)
                        stage_map[slug] = fm.get("learning_stage", "#new").lstrip("#")
                    except Exception:
                        stage_map[slug] = "new"
                    break
        return stage_map

    # --- Public API ---

    def rank_and_query(
        self,
        question: str,
        *,
        use_flash: bool = False,
        use_pro: bool = False,
        auto_escalate: bool = False,
        show_prompt: bool = False,
        verbose: bool = False,
    ) -> dict:
        """Rank vault notes by composite score and query BrainEngine.

        Returns dict with keys:
          answer: str — LLM answer text
          signal_data: list[tuple[str, float, ScoreSignals]] — top-10 ranked notes with signals
          top_slug: str | None — slug of the highest-scored note
          context_display: str — model/context info string from BrainEngine
          constellation: dict | None — hop2 neighborhood if verbose=True, else None
        """
        self._log.info("scoring.rank_and_query", question=question[:80])

        pre_ranked: Optional[list[str]] = None
        signal_data = []
        top_slug: Optional[str] = None
        learning_stage: Optional[str] = None

        # 1. Pre-rank via CompositeScorer (EmbeddingUnavailableError propagates to CLI)
        from pb.vault.embeddings import EmbeddingUnavailableError
        try:
            scorer = self._get_scorer()
            ranked = scorer.rank_notes(query_text=question)
            if ranked:
                pre_ranked = [slug for slug, _score, _signals in ranked]
                signal_data = ranked
                top_slug = pre_ranked[0]
                # Determine dominant learning stage from top note
                try:
                    from pb.vault.lifecycle import read_frontmatter
                    knowledge_path = self.vault_path / "knowledge"
                    for md_file in knowledge_path.rglob("*.md"):
                        if md_file.stem == top_slug or top_slug in str(md_file):
                            content = md_file.read_text()
                            fm, _ = read_frontmatter(content)
                            learning_stage = fm.get("learning_stage")
                            break
                except Exception:
                    pass
        except EmbeddingUnavailableError:
            raise  # propagate to CLI for D-07 message
        except Exception:
            pass  # Non-fatal: scorer failure degrades to full-graph path

        # 2. Query BrainEngine
        brain = self._get_brain()
        answer = brain.query(
            question,
            show_prompt=show_prompt,
            use_pro=use_pro,
            use_flash=use_flash,
            auto_escalate=auto_escalate,
            pre_ranked_candidates=pre_ranked,
            learning_stage=learning_stage,
            verbose=verbose,
        )
        context_display = brain.get_context_display()

        # 3. Constellation (only if verbose)
        constellation: Optional[dict] = None
        if verbose and top_slug:
            try:
                constellation = self.get_constellation(top_slug)
            except Exception as exc:
                self._log.warning("scoring.constellation_failed", error=str(exc))

        return {
            "answer": answer,
            "signal_data": signal_data,
            "top_slug": top_slug,
            "context_display": context_display,
            "constellation": constellation,
        }

    def get_constellation(self, slug: str) -> dict:
        """Return 2-hop neighborhood dict for a note slug.

        Returns: {"out1": [...], "in1": [...], "out2": [...], "in2": [...]}
        Uses graph_store.get_hop2_neighborhood — hard cap at 2 hops (SCOR-04/D-16).
        """
        from pb.vault.graph_store import get_hop2_neighborhood
        return get_hop2_neighborhood(self.vault_path, slug)

    def detect_gaps(self, domain: str) -> dict:
        """Compare existing domain notes against Flash-suggested prerequisites (SCOR-06).

        One-shot Flash call (not Lite — domain reasoning per D-11). Returns:
          {"domain": str, "have": [(slug, stage), ...], "missing": [concept_name, ...]}
        """
        self._log.info("scoring.detect_gaps", domain=domain)

        # 1. Collect existing slugs in the domain
        from pb.vault.graph_store import open_vault_db
        existing_slugs: list[str] = []
        try:
            conn = open_vault_db(self.vault_path)
            try:
                rows = conn.execute(
                    "SELECT slug FROM nodes WHERE subfolder = ?",
                    (f"knowledge/{domain}",),
                ).fetchall()
                existing_slugs = [r[0] for r in rows]
            finally:
                conn.close()
        except Exception as exc:
            self._log.warning("scoring.detect_gaps_db_failed", error=str(exc))

        # 2. Read stages for display
        stage_map = self._read_stage_map(domain, existing_slugs)

        # 3. Flash one-shot: expected prerequisites (D-11/D-13)
        from pb.llm.gemini import get_client, FLASH_MODEL
        client = get_client()
        expected_concepts: list[str] = []
        if client.is_available():
            prompt = (
                f"Domain: {domain}\n"
                f"Existing notes: {', '.join(existing_slugs)}\n\n"
                f"List the prerequisite concept notes that SHOULD exist for complete "
                f"understanding of '{domain}' but are NOT in the existing list above.\n"
                "Return ONLY a YAML list of concept name strings, no explanation.\n"
                "Example:\n"
                "- voice-leading\n"
                "- counterpoint\n"
                "- sight-reading"
            )
            result = client.generate_with_model(prompt, FLASH_MODEL, timeout=15)
            parsed = extract_structured_yaml(result or "", [])
            if isinstance(parsed, list):
                expected_concepts = [str(item) for item in parsed if isinstance(item, str)]
            else:
                self._log.warning("scoring.gap_parse_failed", raw=str(result)[:200])

        # 4. Deterministic diff — normalize concept names to slug format for comparison
        existing_set = set(existing_slugs)
        missing = [
            c for c in expected_concepts
            if c.lower().replace(" ", "-") not in existing_set
        ]

        have = [(slug, stage_map.get(slug, "new")) for slug in existing_slugs]
        return {"domain": domain, "have": have, "missing": missing}
