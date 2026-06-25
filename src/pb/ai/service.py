# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AI integration service: chat, probing, prompt management.

Stub service for Phase 21 -- method bodies raise NotImplementedError.
Real implementations migrated from pb.core.chat, pb.core.probing,
pb.core.prompts, pb.core.question_tree in later phases.

NOTE: tier2_confirm from pb.core.suggestions is NOT included here --
it uses typer.echo (INV-4 violation) and belongs in pb.cli layer.
"""
from __future__ import annotations
from typing import Optional, Callable
from pb.core.base import BaseService, AIMixin


class AIService(BaseService, AIMixin):
    """Manages LLM interactions: chat, probing, and session summarization.

    Constructor takes model name and optional vault service reference
    for context-aware operations.
    """

    def __init__(self, model: str = "flash_lite",
                 vault_service: Optional[object] = None):
        super().__init__()
        self.model = model
        self.vault_service = vault_service

    def chat(self, query: str, context: str | None = None) -> str:
        raise NotImplementedError

    def chat_streaming(self, query: str,
                       on_chunk: Callable[[str], None]) -> None:
        raise NotImplementedError

    def probe(self, note_path: str) -> list[str]:
        raise NotImplementedError

    def suggest_next_actions(self, task_id: str) -> list[str]:
        raise NotImplementedError

    def generate_session_summary(self, session_id: str) -> str:
        raise NotImplementedError

    def get_prompts(self, category: str | None = None) -> list[str]:
        raise NotImplementedError

    def build_question_tree(self, domain: str, depth: int = 3) -> dict:
        raise NotImplementedError
