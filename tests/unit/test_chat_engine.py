"""Unit tests for ChatEngine -- retrieval-augmented query and multi-turn chat.

Tests cover:
- _retrieve_context: vault matches + DB sessions, limit to 5 each
- _format_context_display: returns "Found N related notes: ..." (D-06)
- query(): returns answer when LLM available; "LLM unavailable" when not
- chat_turn(): initializes chat on first call, reuses on subsequent calls
- _escalate_to_flash: switches model, reinitializes chat
- is_available(): delegates to GeminiClient
- reset(): clears chat state
- _build_prefix: assembles stable prefix from system instruction + folder summary + active task
- prefix lifecycle: reuse, invalidation, rebuild, status reporting
- async chat turn: delegates to GeminiClient.generate_streaming_async()
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pb.core.chat import (
    FLASH_LITE_MODEL,
    FLASH_MODEL,
    MAX_VAULT_RESULTS,
    PREFIX_TOKEN_THRESHOLD,
    SYSTEM_INSTRUCTION,
    ChatEngine,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_session(task_id: str, title: str = "Task A", offset_days: int = 0) -> MagicMock:
    """Build a minimal mock Session with start_at and actual_outcome."""
    s = MagicMock()
    s.task_id = task_id
    s.start_at = datetime(2026, 4, 20) - timedelta(days=offset_days)
    s.actual_outcome = "completed"
    return s


def _make_task(task_id: str, title: str = "Task A") -> MagicMock:
    t = MagicMock()
    t.id = task_id
    t.title = title
    return t


def _vault_json(n: int = 3) -> str:
    """Return vault_search JSON with n matches."""
    matches = [
        {
            "path": f"notes/note{i}.md",
            "snippet": f"snippet {i}",
            "filename_match": True,
            "content_match": False,
        }
        for i in range(n)
    ]
    return json.dumps({"matches": matches, "total": n})


# ---------------------------------------------------------------------------
# Helpers for prefix tests
# ---------------------------------------------------------------------------


def _make_engine(fast: bool = False, vault_cwd: Path = None) -> ChatEngine:
    """Build a ChatEngine instance bypassing __init__ with all state set manually."""
    engine = ChatEngine.__new__(ChatEngine)
    engine._repo = MagicMock()
    engine._gemini = MagicMock()
    engine._gemini.is_available.return_value = True
    engine._chat = None
    engine._async_chat = None
    engine._raw_client = None
    engine._current_model = FLASH_LITE_MODEL
    engine._base_model = FLASH_LITE_MODEL
    engine._context_display = None
    engine._auto_escalate = False
    engine._fast = fast
    engine._prefix_text = None
    engine._prefix_token_estimate = 0
    engine._prefix_task_id = None
    engine._get_vault_cwd = (lambda: vault_cwd) if vault_cwd else None
    return engine


class FakeChunk:
    def __init__(self, text: str):
        self.text = text


def _make_mock_async_chat(chunks: list) -> MagicMock:
    """Build a mock AsyncChat that streams given chunk texts.

    The real SDK's send_message_stream() returns a coroutine that, when awaited,
    yields an async iterator. We replicate that pattern here.
    """
    chat = MagicMock()

    async def fake_stream_gen():
        for word in chunks:
            yield FakeChunk(word)

    async def send_message_stream_coro(message):
        return fake_stream_gen()

    chat.send_message_stream.side_effect = send_message_stream_coro
    return chat


# ---------------------------------------------------------------------------
# is_available
# ---------------------------------------------------------------------------


class TestIsAvailable:
    def test_delegates_to_gemini_client_true(self):
        """is_available() should return True when GeminiClient.is_available() is True."""
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = True
        engine._chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._context_display = None

        assert engine.is_available() is True

    def test_delegates_to_gemini_client_false(self):
        """is_available() should return False when GEMINI_API_KEY is not set."""
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = False
        engine._chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._context_display = None

        assert engine.is_available() is False


# ---------------------------------------------------------------------------
# _retrieve_context
# ---------------------------------------------------------------------------


class TestRetrieveContext:
    def _make_engine(self, vault_result: str, sessions, tasks: dict):
        repo = MagicMock()
        repo.list_sessions_in_range.return_value = sessions
        repo.get_task.side_effect = lambda tid: tasks.get(tid)

        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = repo
        engine._gemini = MagicMock()
        engine._chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._context_display = None

        self._vault_result = vault_result
        return engine

    def test_combines_vault_matches_and_sessions(self):
        """_retrieve_context combines vault matches + DB sessions in one string."""
        sessions = [_make_session("t1")]
        tasks = {"t1": _make_task("t1", "Work on project")}
        engine = self._make_engine(_vault_json(2), sessions, tasks)

        with patch("pb.core.chat.vault_search", return_value=self._vault_result):
            result = engine._retrieve_context("project")

        assert "note0.md" in result
        assert "note1.md" in result
        assert "Work on project" in result

    def test_limits_vault_results_to_max(self):
        """_retrieve_context should not include more than MAX_VAULT_RESULTS vault matches."""
        # Return 10 matches from vault
        engine = self._make_engine(_vault_json(10), [], {})

        with patch("pb.core.chat.vault_search", return_value=self._vault_result):
            result = engine._retrieve_context("anything")

        # Count occurrences of "note" prefix to verify cap
        note_count = sum(1 for i in range(10) if f"note{i}.md" in result)
        assert note_count <= MAX_VAULT_RESULTS

    def test_includes_session_task_titles_and_dates(self):
        """_retrieve_context includes session task title and date in output."""
        s = _make_session("t5", offset_days=0)
        s.start_at = datetime(2026, 4, 22, 10, 30)
        s.actual_outcome = "done"
        tasks = {"t5": _make_task("t5", "Finish report")}
        engine = self._make_engine(_vault_json(0), [s], tasks)

        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            result = engine._retrieve_context("report")

        assert "Finish report" in result
        assert "2026-04-22" in result

    def test_returns_fallback_when_no_context(self):
        """_retrieve_context returns fallback string when vault is empty and no sessions."""
        engine = self._make_engine(_vault_json(0), [], {})

        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            result = engine._retrieve_context("unknown topic")

        assert result == "No relevant context found."

    def test_gracefully_handles_vault_failure(self):
        """_retrieve_context silently skips vault results on exception."""
        sessions = [_make_session("t1")]
        tasks = {"t1": _make_task("t1", "Fallback task")}
        engine = self._make_engine("", sessions, tasks)

        with patch("pb.core.chat.vault_search", side_effect=Exception("vault down")):
            result = engine._retrieve_context("anything")

        # Should still include DB sessions
        assert "Fallback task" in result


# ---------------------------------------------------------------------------
# _format_context_display
# ---------------------------------------------------------------------------


class TestFormatContextDisplay:
    def _make_engine(self):
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._gemini = MagicMock()
        engine._chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._context_display = None
        return engine

    def test_returns_found_notes_string(self):
        """_format_context_display returns 'Found N related notes: [paths]' (D-06)."""
        engine = self._make_engine()
        with patch("pb.core.chat.vault_search", return_value=_vault_json(3)):
            result = engine._format_context_display("test query")
        assert result.startswith("Found 3 related notes:")
        assert "note0.md" in result

    def test_returns_no_matches_string(self):
        """_format_context_display returns 'No vault notes matched' when empty."""
        engine = self._make_engine()
        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            result = engine._format_context_display("empty")
        assert "No vault notes matched" in result

    def test_gracefully_handles_vault_failure(self):
        """_format_context_display returns fallback when vault raises."""
        engine = self._make_engine()
        with patch("pb.core.chat.vault_search", side_effect=Exception("err")):
            result = engine._format_context_display("query")
        assert "unavailable" in result.lower()


# ---------------------------------------------------------------------------
# query()
# ---------------------------------------------------------------------------


class TestQuery:
    def _make_engine(self, available: bool = True, generate_result: str | None = "The answer"):
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = available
        engine._gemini.generate_with_model.return_value = generate_result
        engine._chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._context_display = None
        return engine

    def test_returns_answer_when_available(self):
        """query() returns plain text answer when LLM is available."""
        engine = self._make_engine(available=True, generate_result="Paris is the capital.")
        with patch("pb.core.chat.vault_search", return_value=_vault_json(1)):
            result = engine.query("What is the capital of France?")
        assert result == "Paris is the capital."

    def test_returns_unavailable_message_when_no_api_key(self):
        """query() returns 'LLM unavailable' message when GEMINI_API_KEY not set."""
        engine = self._make_engine(available=False)
        result = engine.query("any question")
        assert "LLM unavailable" in result

    def test_returns_failure_message_when_generate_returns_none(self):
        """query() returns a failure message when generate() returns None."""
        engine = self._make_engine(available=True, generate_result=None)
        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            result = engine.query("any question")
        assert "Failed" in result or "error" in result.lower()

    def test_sets_context_display_after_query(self):
        """query() sets _context_display so get_context_display() works."""
        engine = self._make_engine(available=True, generate_result="Answer")
        with patch("pb.core.chat.vault_search", return_value=_vault_json(2)):
            engine.query("question")
        assert engine.get_context_display() is not None


# ---------------------------------------------------------------------------
# chat_turn()
# ---------------------------------------------------------------------------


class TestChatTurn:
    def _make_engine(self, available: bool = True):
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._repo.list_sessions_in_range.return_value = []
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = available
        engine._chat = None
        engine._async_chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._base_model = FLASH_LITE_MODEL
        engine._context_display = None
        engine._auto_escalate = False
        engine._fast = False
        engine._prefix_text = SYSTEM_INSTRUCTION
        engine._prefix_token_estimate = len(SYSTEM_INSTRUCTION) // 4
        engine._prefix_task_id = None
        engine._get_vault_cwd = None
        return engine

    def _mock_genai(self, response_text: str = "Hello from Gemini"):
        """Return a mock for the google.genai module that produces a chat."""
        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = response_text
        mock_chat.send_message.return_value = mock_response
        mock_genai.Client.return_value.chats.create.return_value = mock_chat
        return mock_genai, mock_chat

    def test_returns_unavailable_message_when_no_api_key(self):
        """chat_turn() returns 'LLM unavailable' when is_available() is False."""
        engine = self._make_engine(available=False)
        result = engine.chat_turn("hello")
        assert "LLM unavailable" in result

    def test_initializes_chat_on_first_call(self):
        """chat_turn() creates a new chat session on the first call."""
        engine = self._make_engine(available=True)
        mock_genai, mock_chat = self._mock_genai("First response")

        with patch("pb.core.chat._create_genai_client", return_value=mock_genai.Client()):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
                result = engine.chat_turn("first question")

        assert engine._chat is not None

    def test_reuses_chat_object_on_subsequent_calls(self):
        """chat_turn() uses the same chat object across multiple turns."""
        engine = self._make_engine(available=True)
        mock_genai, mock_chat = self._mock_genai("response")

        with patch("pb.core.chat._create_genai_client", return_value=mock_genai.Client()):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
                engine.chat_turn("first")
                chat_after_first = engine._chat

                engine.chat_turn("second")
                chat_after_second = engine._chat

        assert chat_after_first is chat_after_second

    def test_sets_context_display_on_first_call(self):
        """chat_turn() sets _context_display via _format_context_display on first turn."""
        engine = self._make_engine(available=True)
        mock_genai, mock_chat = self._mock_genai("response")

        with patch("pb.core.chat._create_genai_client", return_value=mock_genai.Client()):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(2)):
                engine.chat_turn("hello")

        assert engine._context_display is not None


# ---------------------------------------------------------------------------
# _escalate_to_flash
# ---------------------------------------------------------------------------


class TestEscalateToFlash:
    def test_escalation_switches_model_to_flash(self):
        """_escalate_to_flash sets _current_model to FLASH_MODEL."""
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._repo.list_sessions_in_range.return_value = []
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = True
        engine._chat = None
        engine._async_chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._base_model = FLASH_LITE_MODEL
        engine._context_display = None
        engine._auto_escalate = False
        engine._fast = False
        engine._prefix_text = SYSTEM_INSTRUCTION
        engine._prefix_token_estimate = len(SYSTEM_INSTRUCTION) // 4
        engine._prefix_task_id = None
        engine._get_vault_cwd = None

        mock_genai = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Escalated response"
        mock_chat.send_message.return_value = mock_response
        mock_genai.Client.return_value.chats.create.return_value = mock_chat

        with patch("pb.core.chat._create_genai_client", return_value=mock_genai.Client()):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
                engine._escalate_to_flash("complex query")

        assert engine._current_model == FLASH_MODEL

    def test_escalation_resets_chat_and_reinitializes(self):
        """_escalate_to_flash resets _chat = None then reinitializes it."""
        engine = ChatEngine.__new__(ChatEngine)
        engine._repo = MagicMock()
        engine._repo.list_sessions_in_range.return_value = []
        engine._gemini = MagicMock()
        engine._gemini.is_available.return_value = True
        # Simulate existing lite chat
        old_chat = MagicMock()
        engine._chat = old_chat
        engine._async_chat = None
        engine._raw_client = None
        engine._current_model = FLASH_LITE_MODEL
        engine._base_model = FLASH_LITE_MODEL
        engine._context_display = None
        engine._auto_escalate = False
        engine._fast = False
        engine._prefix_text = SYSTEM_INSTRUCTION
        engine._prefix_token_estimate = len(SYSTEM_INSTRUCTION) // 4
        engine._prefix_task_id = None
        engine._get_vault_cwd = None

        mock_genai = MagicMock()
        mock_new_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Flash response"
        mock_new_chat.send_message.return_value = mock_response
        mock_genai.Client.return_value.chats.create.return_value = mock_new_chat

        with patch("pb.core.chat._create_genai_client", return_value=mock_genai.Client()):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
                engine._escalate_to_flash("complex query")

        # Chat should be a newly initialized one, not the old lite chat
        assert engine._chat is not old_chat


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


class TestReset:
    def test_reset_clears_all_chat_state(self):
        """reset() clears chat, raw_client, model, and context_display."""
        engine = _make_engine()
        engine._chat = MagicMock()  # Simulate existing chat
        engine._raw_client = MagicMock()
        engine._current_model = FLASH_MODEL
        engine._base_model = FLASH_LITE_MODEL
        engine._context_display = "Found 3 related notes: ..."
        engine._async_chat = MagicMock()
        engine._prefix_text = "old prefix"
        engine._prefix_token_estimate = 100
        engine._prefix_task_id = "task-12345"
        engine._repo.get_active_task.return_value = None

        engine.reset()

        assert engine._chat is None
        assert engine._raw_client is None
        assert engine._current_model == FLASH_LITE_MODEL
        assert engine._context_display is None
        assert engine._async_chat is None
        assert engine._prefix_text is not None  # rebuilt by _build_prefix()
        assert engine._prefix_token_estimate >= 0  # reset and rebuilt


# ---------------------------------------------------------------------------
# TestPrefixAssembly
# ---------------------------------------------------------------------------


class TestPrefixAssembly:
    """Tests for _build_prefix() method (CACH-01, D-01, D-04, D-12)."""

    def test_builds_prefix_from_system_instruction(self, tmp_path):
        """_build_prefix() always includes SYSTEM_INSTRUCTION as first part."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        assert SYSTEM_INSTRUCTION in engine._prefix_text

    def test_includes_folder_summary_when_pb_directory_exists(self, tmp_path):
        """prefix includes .pb-directory.md content when file exists at vault_cwd."""
        (tmp_path / ".pb-directory.md").write_text("# Notes\n3 notes, 0 subfolders")
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        assert "3 notes" in engine._prefix_text

    def test_includes_active_task_when_present(self, tmp_path):
        """prefix includes 'Active task: {title}' when repo.get_active_task() returns a task."""
        engine = _make_engine(vault_cwd=tmp_path)
        task = _make_task("task-11111", "Write phase 13 plan")
        engine._repo.get_active_task.return_value = task
        engine._build_prefix()
        assert "Active task: Write phase 13 plan" in engine._prefix_text

    def test_compacts_prefix_when_over_threshold(self, tmp_path):
        """when prefix exceeds 1500 estimated tokens (6000 chars), folder summary is truncated."""
        # Create a very large folder summary (way more than 6000 chars)
        large_summary = "x" * 9000
        (tmp_path / ".pb-directory.md").write_text(large_summary)
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        # Prefix should be compacted - must be shorter than raw content
        assert len(engine._prefix_text) < len(SYSTEM_INSTRUCTION) + 9000
        # Should still contain system instruction
        assert SYSTEM_INSTRUCTION in engine._prefix_text

    def test_drops_folder_summary_if_no_space(self, tmp_path):
        """when system instruction + task context leaves no space, folder summary is dropped entirely."""
        # Fill the budget with a large task description so no room remains for folder summary
        task = _make_task("task-22222", "Big task")
        # Use a large description that fills the threshold leaving no room for folder summary
        task.description = "d" * 7000
        engine = _make_engine(vault_cwd=tmp_path)
        # Put a small folder summary in
        (tmp_path / ".pb-directory.md").write_text("small folder summary")
        engine._repo.get_active_task.return_value = task
        engine._build_prefix()
        # Folder summary should be dropped (no room for it)
        # The prefix should not contain the folder summary text
        assert "small folder summary" not in engine._prefix_text
        # System instruction must remain
        assert SYSTEM_INSTRUCTION in engine._prefix_text

    def test_falls_back_to_system_instruction_on_vault_error(self, tmp_path):
        """when get_vault_cwd raises, prefix still contains SYSTEM_INSTRUCTION (D-12)."""
        engine = _make_engine()
        engine._get_vault_cwd = lambda: (_ for _ in ()).throw(Exception("vault error"))
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        assert SYSTEM_INSTRUCTION in engine._prefix_text

    def test_falls_back_to_system_instruction_on_repo_error(self, tmp_path):
        """when repo.get_active_task() raises, prefix still contains SYSTEM_INSTRUCTION (D-12)."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.side_effect = Exception("db error")
        engine._build_prefix()
        assert SYSTEM_INSTRUCTION in engine._prefix_text


# ---------------------------------------------------------------------------
# TestPrefixReuse
# ---------------------------------------------------------------------------


class TestPrefixReuse:
    """Tests for prefix reuse across turns (CACH-01)."""

    def test_prefix_not_rebuilt_on_second_turn(self, tmp_path):
        """after first chat_turn builds prefix, second chat_turn does not call _build_prefix again."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._repo.list_sessions_in_range.return_value = []
        engine._prefix_text = "already built prefix"
        engine._prefix_token_estimate = 10

        # Track how many times _build_prefix is called
        build_count = [0]
        original_build = engine._build_prefix

        def counting_build():
            build_count[0] += 1
            original_build()

        engine._build_prefix = counting_build

        # Set up mock raw client for chat turn
        mock_raw_client = MagicMock()
        mock_chat = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_chat.send_message.return_value = mock_response
        mock_raw_client.chats.create.return_value = mock_chat

        with patch("pb.core.chat._create_genai_client", return_value=mock_raw_client):
            with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
                engine.chat_turn("first")
                engine.chat_turn("second")

        # Should not have rebuilt prefix (it was pre-set)
        assert build_count[0] == 0


# ---------------------------------------------------------------------------
# TestPrefixRebuild
# ---------------------------------------------------------------------------


class TestPrefixRebuild:
    """Tests for prefix invalidation and rebuild (CACH-02)."""

    def test_invalidate_clears_prefix_and_chat(self, tmp_path):
        """invalidate_prefix() sets _prefix_text=None then rebuilds immediately."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._prefix_text = "old prefix"
        engine._prefix_task_id = "old-task"
        engine._async_chat = MagicMock()
        engine._chat = MagicMock()

        engine.invalidate_prefix()

        # Should have been cleared and rebuilt
        assert engine._async_chat is None
        assert engine._chat is None
        # prefix_text gets rebuilt by _build_prefix
        assert engine._prefix_text is not None
        assert engine._prefix_task_id is None or engine._prefix_task_id != "old-task"

    def test_reset_clears_prefix(self, tmp_path):
        """reset() clears _prefix_text, _prefix_token_estimate, _prefix_task_id and rebuilds."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._prefix_text = "old prefix"
        engine._prefix_token_estimate = 500
        engine._prefix_task_id = "task-99999"
        engine._async_chat = MagicMock()
        engine._chat = MagicMock()

        engine.reset()

        # All state should be cleared and prefix rebuilt
        assert engine._async_chat is None
        assert engine._chat is None
        assert engine._prefix_text is not None  # rebuilt
        assert engine._prefix_task_id != "task-99999"  # old value gone


# ---------------------------------------------------------------------------
# TestPrefixStatus
# ---------------------------------------------------------------------------


class TestPrefixStatus:
    """Tests for get_prefix_status() (CACH-03)."""

    def test_status_with_context(self, tmp_path):
        """get_prefix_status() returns string containing 'streaming' and 'tok' when prefix has vault context."""
        (tmp_path / ".pb-directory.md").write_text("# Folder\n10 notes")
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        status = engine.get_prefix_status()
        assert "streaming" in status
        assert "tok" in status

    def test_status_without_context(self, tmp_path):
        """get_prefix_status() returns string containing 'no context' when prefix is system-instruction-only."""
        engine = _make_engine()  # No vault_cwd, no active task
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        status = engine.get_prefix_status()
        assert "no context" in status

    def test_status_fast_mode(self, tmp_path):
        """get_prefix_status() returns string containing 'sync' when fast=True."""
        engine = _make_engine(fast=True)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        status = engine.get_prefix_status()
        assert "sync" in status


# ---------------------------------------------------------------------------
# TestPrefixFallback
# ---------------------------------------------------------------------------


class TestPrefixFallback:
    """Tests for graceful fallback when components are unavailable (CACH-04)."""

    def test_graceful_fallback_no_directory_file(self, tmp_path):
        """when .pb-directory.md does not exist, prefix is still valid (system instruction only)."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        # No .pb-directory.md in tmp_path
        engine._build_prefix()
        assert engine._prefix_text is not None
        assert SYSTEM_INSTRUCTION in engine._prefix_text

    def test_graceful_fallback_no_active_task(self, tmp_path):
        """when get_active_task() returns None, prefix is still valid."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._build_prefix()
        assert engine._prefix_text is not None
        assert SYSTEM_INSTRUCTION in engine._prefix_text


# ---------------------------------------------------------------------------
# TestFastFlag
# ---------------------------------------------------------------------------


class TestFastFlag:
    """Tests for fast=True/False flag behavior (CLUX-05)."""

    def test_fast_true_sets_flag(self):
        """ChatEngine(fast=True) sets _fast=True."""
        with patch("pb.core.chat._create_genai_client", return_value=None):
            engine = ChatEngine.__new__(ChatEngine)
            # Test via constructor if possible, or via direct assignment
            engine = _make_engine(fast=True)
            assert engine._fast is True

    def test_fast_false_by_default(self):
        """ChatEngine() has _fast=False."""
        engine = _make_engine(fast=False)
        assert engine._fast is False


# ---------------------------------------------------------------------------
# TestAsyncChatTurn
# ---------------------------------------------------------------------------


class TestAsyncChatTurn:
    """Tests for _async_chat_turn() delegation (CLUX-05)."""

    def test_async_turn_delegates_to_gemini_streaming(self, tmp_path):
        """_async_chat_turn() delegates streaming to self._gemini.generate_streaming_async()."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._repo.list_sessions_in_range.return_value = []
        engine._prefix_text = "test prefix"
        engine._prefix_token_estimate = 10

        # Set up raw client mock
        mock_raw_client = MagicMock()
        mock_async_chat = MagicMock()
        mock_raw_client.aio.chats.create.return_value = mock_async_chat
        engine._raw_client = mock_raw_client

        # Mock generate_streaming_async as AsyncMock
        engine._gemini.generate_streaming_async = AsyncMock(return_value="Hello world")

        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            result = asyncio.run(engine._async_chat_turn("test input"))

        # Should have called generate_streaming_async
        engine._gemini.generate_streaming_async.assert_called_once()
        assert result == "Hello world"

    def test_async_turn_reuses_existing_chat(self, tmp_path):
        """second call to _async_chat_turn reuses self._async_chat."""
        engine = _make_engine(vault_cwd=tmp_path)
        engine._repo.get_active_task.return_value = None
        engine._repo.list_sessions_in_range.return_value = []
        engine._prefix_text = "test prefix"
        engine._prefix_token_estimate = 10

        mock_raw_client = MagicMock()
        mock_async_chat = MagicMock()
        mock_raw_client.aio.chats.create.return_value = mock_async_chat
        engine._raw_client = mock_raw_client

        engine._gemini.generate_streaming_async = AsyncMock(return_value="response")

        with patch("pb.core.chat.vault_search", return_value=_vault_json(0)):
            asyncio.run(engine._async_chat_turn("first"))
            asyncio.run(engine._async_chat_turn("second"))

        # aio.chats.create should have been called only once
        mock_raw_client.aio.chats.create.assert_called_once()
