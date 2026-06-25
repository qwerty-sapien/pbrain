# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Chat engine for retrieval-augmented query and multi-turn conversation.

Implements:
- D-01: Flash Lite as default model for chat and query
- D-02: Auto-escalation to Flash when query requires extended processing
- D-05: pb query returns plain text; --json handled at CLI layer
- D-06: Transparent retrieval context display
- D-07: Full preview before vault write (handled at CLI layer)

Phase 13 additions:
- CACH-01: Stable prefix assembled from system instruction + .pb-directory.md + active task
- CACH-02: Prefix rebuilt on invalidate_prefix() -- cd, /new, task transition
- CACH-03: get_prefix_status() returns formatted one-liner
- CACH-04: Prefix assembly falls back silently to bare system instruction on any error
- CLUX-05: Async streaming path delegates to GeminiClient.generate_streaming_async()

Security mitigations:
- T-02-10: MAX_VAULT_RESULTS=5 caps context size (Pitfall #4)
- T-02-11: API key read from env var only; never logged; never included in prompts
- T-13-04: PREFIX_TOKEN_THRESHOLD=1500 caps prefix size (D-04)
"""

from __future__ import annotations

import asyncio
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, Optional

import structlog

from pb.llm.gemini import (
    _create_genai_client,
    get_client as get_gemini_client,
    _CREDS_HINT,
    FLASH_LITE_MODEL,
    FLASH_MODEL,
    PRO_MODEL,
)
from pb.storage.repository import Repository

logger = structlog.get_logger()

# ---------------------------------------------------------------------------
# Module-level constants
# ---------------------------------------------------------------------------

CONTEXT_WINDOW_DAYS = 7
MAX_VAULT_RESULTS = 5

PREFIX_TOKEN_THRESHOLD = 1500
SYSTEM_INSTRUCTION = (
    "You are a personal productivity assistant with access to the user's "
    "task history, session logs, and Obsidian vault notes. "
    "Answer questions based on provided context. Be concise and actionable."
)


def vault_search(query: str, folder: str = "") -> str:
    """Thin wrapper around pb.mcp.tools.vault.vault_search for testability.

    Keeping this at module level means tests can patch pb.core.chat.vault_search
    without needing the MCP server dependency available.
    """
    from pb.mcp.tools.vault import vault_search as _vault_search
    return _vault_search(query, folder)


class ChatEngine:
    """Retrieval-augmented chat and query engine.

    Pulls context from Obsidian vault (via vault_search) and pb database
    (via Repository), then uses Gemini for answering.

    Phase 13 additions:
    - Stable prefix assembly (system instruction + .pb-directory.md + active task)
    - Prefix reuse across turns for Gemini implicit caching
    - Async streaming path (CLUX-05, D-06)
    - --fast flag for sync mode (D-07, D-08)
    """

    def __init__(
        self,
        repo: Optional[Repository] = None,
        use_pro: bool = False,
        use_flash: bool = False,
        auto_escalate: bool = False,
        fast: bool = False,
        get_vault_cwd: Optional[Callable[[], Path]] = None,
    ):
        self._repo = repo or Repository()
        self._gemini = get_gemini_client()
        self._chat = None  # Lazy-initialized on first sync chat_turn
        self._async_chat = None  # Lazy-initialized on first async chat turn
        self._raw_client = None  # google.genai.Client for chats API
        self._auto_escalate = auto_escalate
        self._fast = fast
        self._get_vault_cwd = get_vault_cwd  # Callable[[], Path] or None
        if use_pro:
            self._current_model = PRO_MODEL
        elif use_flash:
            self._current_model = FLASH_MODEL
        else:
            self._current_model = FLASH_LITE_MODEL
        self._base_model = self._current_model
        self._context_display: Optional[str] = None  # Last retrieval context for display

        # Prefix state (CACH-01 through CACH-04)
        self._prefix_text: Optional[str] = None
        self._prefix_token_estimate: int = 0
        self._prefix_task_id: Optional[str] = None  # Task ID used when prefix was built

        # Build prefix immediately on construction
        self._build_prefix()

    # -----------------------------------------------------------------------
    # Prefix management (CACH-01 through CACH-04)
    # -----------------------------------------------------------------------

    def _build_prefix(self) -> None:
        """Assemble stable prefix from system instruction + folder summary + active task.

        Rebuilds self._prefix_text and self._prefix_token_estimate.
        Silent fallback if any component is unavailable (D-12, D-13, CACH-04).
        Token cap applied at PREFIX_TOKEN_THRESHOLD (D-04, T-13-04).
        """
        parts = [SYSTEM_INSTRUCTION]
        folder_summary_idx = None

        # Component 2: folder summary from .pb-directory.md
        try:
            if self._get_vault_cwd is not None:
                vault_cwd = self._get_vault_cwd()
                dir_md = vault_cwd / ".pb-directory.md"
                if dir_md.exists():
                    folder_summary = dir_md.read_text(encoding="utf-8", errors="replace")
                    folder_summary_idx = len(parts)
                    parts.append(folder_summary)
        except Exception:
            pass  # Silent fallback per D-12

        # Component 3: active task
        try:
            task = self._repo.get_active_task()
            if task:
                self._prefix_task_id = task.id
                task_ctx = f"Active task: {task.title}"
                if getattr(task, "description", None):
                    task_ctx += f"\nNotes: {task.description}"
                parts.append(task_ctx)
            else:
                self._prefix_task_id = None
        except Exception:
            self._prefix_task_id = None  # Silent fallback

        prefix = "\n\n".join(parts)
        estimated_tokens = len(prefix) // 4

        # Compact if over threshold (D-04): trim folder summary first, keep system + task intact
        if estimated_tokens > PREFIX_TOKEN_THRESHOLD and folder_summary_idx is not None:
            budget_chars = PREFIX_TOKEN_THRESHOLD * 4
            other_len = sum(len(p) for i, p in enumerate(parts) if i != folder_summary_idx)
            available = budget_chars - other_len
            if available > 200:
                parts[folder_summary_idx] = parts[folder_summary_idx][:available] + "\n[summary truncated]"
            else:
                parts.pop(folder_summary_idx)
            prefix = "\n\n".join(parts)
            estimated_tokens = len(prefix) // 4

        self._prefix_text = prefix
        self._prefix_token_estimate = estimated_tokens
        logger.debug("chat.prefix_built", tokens=estimated_tokens, parts=len(parts))

    def invalidate_prefix(self, new_vault_cwd: Optional[Path] = None) -> None:
        """Trigger prefix rebuild. Called on cd, /new, or task transition detection.

        Clears existing prefix state and async/sync chat sessions, then rebuilds
        the prefix immediately (D-03, CACH-02).
        """
        self._prefix_text = None
        self._prefix_task_id = None
        self._async_chat = None
        self._chat = None
        self._build_prefix()

    def get_prefix_status(self) -> str:
        """Return status one-liner for /status command (CACH-03, D-10, D-11).

        Returns:
            Formatted string like "  flash-lite · streaming · prefix 842tok"
            or "  flash-lite · streaming · no context"
        """
        mode = "sync" if self._fast else "streaming"
        if self._prefix_token_estimate > 0 and len(self._prefix_text or "") > len(SYSTEM_INSTRUCTION):
            return f"  {self._current_model} · {mode} · prefix {self._prefix_token_estimate}tok"
        return f"  {self._current_model} · {mode} · no context"

    def _check_task_transition(self) -> None:
        """Poll for active task changes and rebuild prefix if task changed (Pitfall #4).

        Called at the start of each sync chat_turn to detect when the user has
        switched active tasks outside the chat session (e.g., via `pb start`).
        Cost: one SQLite read per turn (~0.1ms), deterministic behavior.
        """
        try:
            task = self._repo.get_active_task()
            current_id = task.id if task else None
            if current_id != self._prefix_task_id:
                self.invalidate_prefix()
        except Exception:
            pass  # Silent fallback

    # -----------------------------------------------------------------------
    # Async streaming path (CLUX-05, D-06, D-09)
    # -----------------------------------------------------------------------

    async def _async_chat_turn(self, user_input: str) -> str:
        """Streaming turn for default async mode (CLUX-05, D-06).

        Creates the AsyncChat session lazily, then delegates streaming to
        GeminiClient.generate_streaming_async() which handles the chunk loop.
        Returns the full response text after streaming completes.

        Per anti-pattern note in RESEARCH.md: do NOT duplicate the streaming
        loop inline -- delegation to GeminiClient keeps the loop in one place.
        """
        try:
            if self._async_chat is None:
                if self._raw_client is None:
                    self._raw_client = _create_genai_client()
                    if self._raw_client is None:
                        return "LLM unavailable — set GEMINI_API_KEY or GOOGLE_CLOUD_PROJECT"
                # aio.chats.create() is synchronous — do NOT await it (Pitfall #1)
                self._async_chat = self._raw_client.aio.chats.create(
                    model=self._current_model,
                    config={"system_instruction": self._prefix_text or SYSTEM_INSTRUCTION},
                )
                logger.debug("chat.async_init", model=self._current_model)

            per_turn_ctx = self._retrieve_context(user_input)
            full_msg = f"{per_turn_ctx}\n\n{user_input}"

            # Delegate streaming to GeminiClient (single streaming loop location)
            response = await self._gemini.generate_streaming_async(self._async_chat, full_msg)
            return response
        except Exception as e:
            logger.debug("chat.async_turn_error", error=str(e))
            if self._auto_escalate and self._current_model == FLASH_LITE_MODEL:
                return self._escalate_to_flash(user_input)
            return f"Chat error: {e}"

    # -----------------------------------------------------------------------
    # Existing public API (unchanged semantics)
    # -----------------------------------------------------------------------

    def is_available(self) -> bool:
        """Check if the LLM backend is available."""
        return self._gemini.is_available()

    def _retrieve_context(self, query: str) -> str:
        """Pull top-5 vault matches + recent sessions as context.

        Per T-02-10 / Pitfall #4: limits to MAX_VAULT_RESULTS with snippets only
        (not full note content) to avoid context token overflow.
        """
        context_parts = []

        # Vault search -- calls module-level vault_search for testability
        try:
            raw = vault_search(query)
            results = json.loads(raw)
            matches = results.get("matches", [])[:MAX_VAULT_RESULTS]
            for m in matches:
                context_parts.append(f"Note: {m['path']}\n{m.get('snippet', '')}")
            logger.debug("chat.vault_search", query=query, matches=len(matches))
        except Exception as e:
            logger.debug("chat.vault_search_failed", error=str(e))

        # Recent sessions from pb DB
        try:
            end = datetime.utcnow()
            start = end - timedelta(days=CONTEXT_WINDOW_DAYS)
            sessions = self._repo.list_sessions_in_range(start, end)
            for s in sessions[-5:]:  # Last 5 sessions
                task = self._repo.get_task(s.task_id)
                if task:
                    date_str = s.start_at.strftime("%Y-%m-%d %H:%M") if s.start_at else "unknown"
                    outcome = s.actual_outcome or "in progress"
                    context_parts.append(
                        f"Session: {task.title} ({date_str}) - {outcome}"
                    )
            logger.debug("chat.db_sessions", count=min(len(sessions), 5))
        except Exception as e:
            logger.debug("chat.db_sessions_failed", error=str(e))

        return "\n---\n".join(context_parts) if context_parts else "No relevant context found."

    def _format_context_display(self, query: str) -> str:
        """Format retrieval context for transparent display (D-06).

        Returns: "Found N related notes: [path1, path2, ...]. Based on these..."
        """
        try:
            raw = vault_search(query)
            results = json.loads(raw)
            matches = results.get("matches", [])[:MAX_VAULT_RESULTS]
            if matches:
                paths = [m["path"] for m in matches]
                return f"Found {len(matches)} related notes: {', '.join(paths)}. Based on these..."
            return "No vault notes matched this query."
        except Exception:
            return "Vault search unavailable."

    def query(self, question: str) -> str:
        """One-shot NL query (D-05). Returns plain text answer.

        Uses the model selected at construction (Flash Lite by default).
        """
        if not self.is_available():
            return f"LLM unavailable -- {_CREDS_HINT} to enable queries."

        context = self._retrieve_context(question)
        self._context_display = self._format_context_display(question)

        prompt = (
            f"You are a personal productivity assistant. Answer based on the context provided.\n\n"
            f"Context:\n{context}\n\n"
            f"Question: {question}\n\n"
            f"Answer concisely and factually based on the context. "
            f"If the context doesn't contain relevant information, say so."
        )

        result = self._gemini.generate_with_model(prompt, self._current_model)
        if result is None:
            return "Failed to generate response. Try again."

        logger.debug("chat.query", model=self._current_model, question=question[:50], response_len=len(result))
        return result

    def get_context_display(self) -> Optional[str]:
        """Get the last retrieval context display string (D-06)."""
        return self._context_display

    def chat_turn(self, user_input: str) -> str:
        """One turn in a REPL session (sync path, used for --fast mode).

        Checks for task transitions before each turn (Pitfall #4).
        Lazy-initializes the chat on first call. Uses google-genai chats API
        for automatic history management (D-01).
        """
        # Check for task transition — rebuild prefix if active task changed
        self._check_task_transition()

        if not self.is_available():
            return f"LLM unavailable -- {_CREDS_HINT} to enable chat."

        try:
            if self._chat is None:
                response = self._init_chat(user_input)
                if response:
                    return response
                return "Chat initialization failed."

            response = self._chat.send_message(user_input)
            return response.text

        except Exception as e:
            logger.debug("chat.turn_error", error=str(e))
            if self._auto_escalate and self._current_model == FLASH_LITE_MODEL:
                return self._escalate_to_flash(user_input)
            return f"Chat error: {e}"

    def _init_chat(self, first_query: str) -> Optional[str]:
        """Initialize the multi-turn chat session (sync path).

        Uses self._prefix_text as system_instruction (assembled by _build_prefix).
        Returns the LLM response text for the first query, or None on failure.
        T-02-11: credentials read from env vars only; never logged or embedded in prompts.
        """
        try:
            self._raw_client = _create_genai_client()
            if self._raw_client is None:
                return None

            # Retrieve context for first turn
            context = self._retrieve_context(first_query)
            self._context_display = self._format_context_display(first_query)

            # Use assembled prefix instead of hard-coded system instruction (CACH-01)
            self._chat = self._raw_client.chats.create(
                model=self._current_model,
                config={"system_instruction": self._prefix_text or SYSTEM_INSTRUCTION},
            )
            # Send context + first query as priming message and return the response
            response = self._chat.send_message(
                f"Here is relevant context for this conversation:\n\n{context}\n\n"
                f"The user's first question is: {first_query}"
            )
            logger.debug("chat.init", model=self._current_model)
            return response.text
        except Exception as e:
            logger.debug("chat.init_error", error=str(e))
            self._chat = None
            return None

    def _escalate_to_flash(self, user_input: str) -> str:
        """Escalate to Flash model when Flash Lite cannot handle the query (D-02)."""
        logger.debug("chat.escalate", from_model=FLASH_LITE_MODEL, to_model=FLASH_MODEL)
        self._current_model = FLASH_MODEL
        self._chat = None  # Reset chat to reinitialize with new model
        try:
            response = self._init_chat(user_input)
            if response:
                return response
        except Exception as e:
            logger.debug("chat.escalate_error", error=str(e))
        return "Escalation to Flash failed. Try again later."

    def reset(self) -> None:
        """Reset chat session and prefix (for /new command in REPL).

        Clears all session state including prefix, async chat, and sync chat.
        Immediately rebuilds the prefix with fresh state (D-03, CACH-02).
        """
        self._chat = None
        self._async_chat = None
        self._raw_client = None
        self._current_model = self._base_model
        self._context_display = None
        self._prefix_text = None
        self._prefix_token_estimate = 0
        self._prefix_task_id = None
        self._build_prefix()  # Rebuild immediately with fresh state
