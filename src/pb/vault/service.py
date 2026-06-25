# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Vault intelligence service: knowledge graph, scoring, packets.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.brain, pb.core.relevance,
pb.core.scorer, pb.core.packet_engine in later phases.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional
from pb.core.models import Packet
from pb.core.base import BaseService, AIMixin


class VaultService(BaseService, AIMixin):
    """Manages vault knowledge operations: query, scoring, packets, graph.

    Constructor takes vault_path and optional AI service reference.
    """

    def __init__(self, vault_path: Path, ai: Optional[object] = None):
        super().__init__()
        self.vault_path = vault_path
        self.ai = ai

    def query(self, query: str, model: str = "flash_lite") -> str:
        raise NotImplementedError

    def write_note(self, path: str, content: str) -> Path:
        raise NotImplementedError

    def read_note(self, path: str) -> str:
        raise NotImplementedError

    def score_note(self, path: str) -> float:
        raise NotImplementedError

    def generate_packet(self, project_id: str) -> Packet:
        raise NotImplementedError

    def update_graph(self, note_path: Path) -> None:
        raise NotImplementedError

    def get_relevant_notes(self, query: str, top_k: int = 5) -> list[str]:
        raise NotImplementedError

    def rebuild_graph(self) -> None:
        raise NotImplementedError

    def get_orphan_notes(self) -> list[str]:
        raise NotImplementedError
