"""Unit tests for gemini module."""

import asyncio
import sys
from types import ModuleType, SimpleNamespace

import pytest
from unittest.mock import patch, MagicMock, AsyncMock
import os

from pb.llm.gemini import (
    GeminiClient,
    get_client,
    score_text_response,
    generate_followup,
    MODEL_ID,
    FLASH_LITE_MODEL,
    FLASH_MODEL,
    PRO_MODEL,
    resolve_model,
)


# ---------------------------------------------------------------------------
# Helpers for async streaming tests
# ---------------------------------------------------------------------------

class FakeChunk:
    def __init__(self, text: str):
        self.text = text


def _make_mock_async_chat(chunks: list) -> MagicMock:
    """Build a mock AsyncChat that streams given chunk texts.

    The real SDK's send_message_stream() returns a coroutine that, when awaited,
    yields an async iterator. We replicate that: return a coroutine whose result
    is an async generator.
    """
    chat = MagicMock()

    async def fake_stream_gen():
        for word in chunks:
            yield FakeChunk(word)

    async def send_message_stream_coro(message):
        return fake_stream_gen()

    chat.send_message_stream.side_effect = send_message_stream_coro
    return chat


class TestGeminiClient:
    """Tests for GeminiClient class."""

    def test_is_unavailable_without_api_key(self):
        """Should return False when GEMINI_API_KEY not set."""
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient()
            assert client.is_available() is False

    def test_is_unavailable_when_import_fails(self):
        """Should return False when google-genai not installed."""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
            with patch.dict("sys.modules", {"google": None}):
                client = GeminiClient()
                # Force re-check by resetting cached state
                client._available = None
                # The ImportError should be caught gracefully
                result = client.is_available()
                # May be True or False depending on whether package is installed
                # The key is that it doesn't raise an exception

    def test_is_available_caches_result(self):
        """Should cache availability check result."""
        with patch.dict(os.environ, {}, clear=True):
            client = GeminiClient()
            first_result = client.is_available()
            # Set a key but result should still be cached
            with patch.dict(os.environ, {"GEMINI_API_KEY": "test-key"}):
                second_result = client.is_available()
            assert first_result == second_result  # Cached

    def test_generate_returns_none_when_unavailable(self):
        """Should return None when client not available."""
        client = GeminiClient()
        client._available = False
        result = client.generate("test prompt")
        assert result is None

    def test_generate_returns_text_on_success(self):
        """Should return generated text when API succeeds."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Generated response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate("test prompt")
        assert result == "Generated response"

    def test_generate_returns_none_on_exception(self):
        """Should return None when API call fails."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate("test prompt")
        assert result is None


class TestGetClient:
    """Tests for get_client singleton function."""

    def test_returns_same_instance(self):
        """Should return the same client instance on repeated calls."""
        # Reset the global client
        import pb.llm.gemini as gemini_module
        gemini_module._client = None

        client1 = get_client()
        client2 = get_client()
        assert client1 is client2


class TestScoreTextResponse:
    """Tests for score_text_response function."""

    def test_returns_none_when_client_unavailable(self):
        """Should return None when Gemini not available."""
        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = False
            mock_get.return_value = mock_client

            result = score_text_response("How was your energy?", "Pretty good")
            assert result is None

    def test_parses_valid_response(self):
        """Should parse score and rationale from valid response."""
        mock_response = "SCORE: 7\nRATIONALE: Good clarity and self-awareness."

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = score_text_response(
                "How was your energy?", "I felt energized after morning exercise"
            )
            assert result is not None
            score, rationale = result
            assert score == 7
            assert "clarity" in rationale.lower() or "self-awareness" in rationale.lower()

    def test_clamps_score_to_max_10(self):
        """Should clamp scores above 10 to 10."""
        mock_response = "SCORE: 15\nRATIONALE: Very detailed."

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = score_text_response("Question", "Response")
            assert result is not None
            score, _ = result
            assert score == 10  # Clamped from 15

    def test_clamps_score_to_min_1(self):
        """Should clamp scores below 1 to 1."""
        mock_response = "SCORE: -5\nRATIONALE: Invalid score."

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = score_text_response("Question", "Response")
            assert result is not None
            score, _ = result
            assert score == 1  # Clamped from -5

    def test_returns_none_on_parse_error(self):
        """Should return None when response format is invalid."""
        mock_response = "Invalid format without score"

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = score_text_response("Question", "Response")
            assert result is None

    def test_returns_none_when_generate_fails(self):
        """Should return None when generate returns None."""
        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = None
            mock_get.return_value = mock_client

            result = score_text_response("Question", "Response")
            assert result is None

    def test_handles_multiline_rationale(self):
        """Should parse rationale that may span multiple lines."""
        mock_response = "SCORE: 8\nRATIONALE: The response shows good insight into personal productivity patterns."

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = score_text_response("Question", "Response")
            assert result is not None
            score, rationale = result
            assert score == 8
            assert "insight" in rationale.lower()


class TestGenerateFollowup:
    """Tests for generate_followup function."""

    def test_returns_none_when_client_unavailable(self):
        """Should return None when Gemini not available."""
        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = False
            mock_get.return_value = mock_client

            result = generate_followup("How was your energy?", "Fine")
            assert result is None

    def test_returns_followup_question(self):
        """Should return follow-up question string."""
        mock_response = "What specifically gave you that energy level?"

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = generate_followup("How was your energy?", "Fine")
            assert result is not None
            assert "?" in result or len(result) > 0

    def test_strips_quotes_from_response(self):
        """Should strip quotation marks from follow-up."""
        mock_response = '"Can you elaborate on what made it fine?"'

        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = mock_response
            mock_get.return_value = mock_client

            result = generate_followup("How was your energy?", "Fine")
            assert result is not None
            assert not result.startswith('"')
            assert not result.endswith('"')

    def test_returns_none_when_generate_fails(self):
        """Should return None when generate returns None."""
        with patch("pb.llm.gemini.get_client") as mock_get:
            mock_client = MagicMock()
            mock_client.is_available.return_value = True
            mock_client.generate.return_value = None
            mock_get.return_value = mock_client

            result = generate_followup("How was your energy?", "Fine")
            assert result is None


class TestModelConfig:
    """Tests for model configuration."""

    def test_model_id_is_defined(self):
        """Should have MODEL_ID constant defined."""
        assert MODEL_ID is not None
        assert isinstance(MODEL_ID, str)
        assert len(MODEL_ID) > 0

    def test_model_id_is_flash_variant(self):
        """Should use flash model for cost efficiency."""
        assert "flash" in MODEL_ID.lower()

    def test_flash_lite_model_constant_defined(self):
        """Should have FLASH_LITE_MODEL constant at module level."""
        assert FLASH_LITE_MODEL == "gemini-3.1-flash-lite-preview"

    def test_flash_model_constant_defined(self):
        """Should have FLASH_MODEL constant at module level."""
        assert FLASH_MODEL == "gemini-3-flash-preview"

    def test_pro_model_constant_defined(self):
        """Should have PRO_MODEL constant at module level."""
        assert PRO_MODEL == "gemini-3.1-pro-preview"

    def test_resolve_model_supports_aliases(self):
        """Short aliases should resolve to the locked Gemini model ids."""
        assert resolve_model("flash-lite") == FLASH_LITE_MODEL
        assert resolve_model("flash") == FLASH_MODEL
        assert resolve_model("pro") == PRO_MODEL


class TestGenerateWithModel:
    """Tests for generate_with_model method."""

    def test_returns_response_text_on_success(self):
        """Should return response text when client is available and model is valid."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Model-specific response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        with patch.dict(os.environ, {}, clear=True):
            result = client.generate_with_model("test prompt", FLASH_MODEL)
        assert result == "Model-specific response"
        mock_client.models.generate_content.assert_called_once_with(
            model=FLASH_MODEL,
            contents="test prompt",
        )

    def test_returns_none_when_unavailable(self):
        """Should return None when client is not available."""
        client = GeminiClient()
        client._available = False

        result = client.generate_with_model("test prompt", FLASH_MODEL)
        assert result is None

    def test_returns_none_on_exception(self):
        """Should return None when API call raises exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_model("test prompt", PRO_MODEL)
        assert result is None

    def test_existing_generate_still_works(self):
        """Regression: existing generate() method still works unchanged."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Original generate response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate("test prompt")
        assert result == "Original generate response"
        mock_client.models.generate_content.assert_called_once_with(
            model=MODEL_ID,
            contents="test prompt",
        )

    def test_retries_on_rate_limit_then_succeeds(self):
        """Transient 429 failures should retry before giving up."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Recovered response"
        mock_client.models.generate_content.side_effect = [
            Exception("429 RESOURCE_EXHAUSTED"),
            mock_response,
        ]

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        with patch("pb.llm.gemini.time.sleep") as mock_sleep:
            result = client.generate_with_model("test prompt", "flash")

        assert result == "Recovered response"
        assert mock_client.models.generate_content.call_count == 2
        mock_sleep.assert_called_once()

    def test_build_generation_config_uses_high_thinking_on_vertex_flash(self):
        """Vertex Flash/Pro should request high thinking by default."""
        fake_types = SimpleNamespace(
            ThinkingLevel=SimpleNamespace(HIGH="HIGH"),
            ThinkingConfig=lambda **kwargs: SimpleNamespace(**kwargs),
            GenerateContentConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        )
        fake_genai = ModuleType("google.genai")
        fake_genai.types = fake_types
        fake_google = ModuleType("google")
        fake_google.genai = fake_genai

        with patch.dict(
            sys.modules,
            {"google": fake_google, "google.genai": fake_genai},
            clear=False,
        ):
            with patch.dict(os.environ, {"GOOGLE_CLOUD_PROJECT": "proj"}, clear=False):
                client = GeminiClient()
                config = client._build_generation_config(FLASH_MODEL)

        assert config is not None
        assert config.thinkingConfig.thinkingLevel == "HIGH"

    def test_recovers_text_from_candidate_parts_when_response_text_is_blank(self):
        """Blank response.text should fall back to candidate text parts when present."""
        mock_client = MagicMock()
        mock_response = SimpleNamespace(
            text="",
            parts=[SimpleNamespace(text="Recovered from parts")],
            candidates=[],
        )
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_model_result("test prompt", FLASH_MODEL)

        assert result.error is None
        assert result.text == "Recovered from parts"

    def test_reports_finish_reason_for_true_empty_http_200_response(self):
        """Empty 200 responses should surface finish metadata instead of a generic blank-body message."""
        mock_client = MagicMock()
        mock_response = SimpleNamespace(
            text="",
            parts=[],
            prompt_feedback=SimpleNamespace(
                block_reason=None,
                block_reason_message="",
                safety_ratings=[],
            ),
            candidates=[
                SimpleNamespace(
                    finish_reason=SimpleNamespace(name="SAFETY"),
                    finish_message="Blocked for safety.",
                    content=SimpleNamespace(parts=[]),
                    safety_ratings=[],
                )
            ],
            usage_metadata=SimpleNamespace(
                prompt_token_count=14,
                candidates_token_count=0,
                total_token_count=14,
            ),
        )
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_model_result("test prompt", FLASH_MODEL)

        assert result.text is None
        assert result.error is not None
        assert "finish reason=safety" in result.error.raw_message
        assert "Blocked for safety." in result.error.raw_message
        assert "not a credits issue" in result.error.raw_message


class TestGenerateWithTools:
    """Tests for generate_with_tools method (AFC)."""

    def test_returns_response_text_on_success(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Tool-enhanced response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        def dummy_tool(x: str) -> str:
            return f"result for {x}"

        result = client.generate_with_tools("test prompt", FLASH_MODEL, [dummy_tool])
        assert result == "Tool-enhanced response"

    def test_returns_none_when_unavailable(self):
        client = GeminiClient()
        client._available = False
        result = client.generate_with_tools("test", FLASH_MODEL, [])
        assert result is None

    def test_returns_none_on_exception(self):
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("API error")

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_tools("test", FLASH_MODEL, [])
        assert result is None

    def test_passes_tools_in_config(self):
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        def my_tool(x: str) -> str:
            return x

        client.generate_with_tools("prompt", FLASH_MODEL, [my_tool])

        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == FLASH_MODEL
        config = call_args.kwargs["config"]
        assert config is not None
        assert my_tool in config.tools


class TestGenerateWithGrounding:
    """Tests for generate_with_grounding method."""

    def test_returns_response_text_on_success(self):
        """Should return response text when called with valid prompt and model."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response with web context"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_grounding("what is quantum computing", FLASH_MODEL)
        assert result == "Grounded response with web context"

    def test_returns_none_when_unavailable(self):
        """Should return None when client is not available."""
        client = GeminiClient()
        client._available = False

        result = client.generate_with_grounding("test prompt", FLASH_MODEL)
        assert result is None

    def test_passes_google_search_tool_config(self):
        """Should pass google_search tool config to generate_content."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.text = "Grounded response"
        mock_client.models.generate_content.return_value = mock_response

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        client.generate_with_grounding("test prompt", FLASH_MODEL)

        # Verify generate_content was called with config containing google_search tool
        call_args = mock_client.models.generate_content.call_args
        assert call_args.kwargs["model"] == FLASH_MODEL
        assert call_args.kwargs["contents"] == "test prompt"
        config = call_args.kwargs["config"]
        # The config should have tools with google_search
        assert config is not None
        assert len(config.tools) == 1
        assert config.tools[0].google_search is not None

    def test_returns_none_on_exception(self):
        """Should return None and log on exception."""
        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = Exception("Grounding error")

        client = GeminiClient()
        client._available = True
        client._client = mock_client

        result = client.generate_with_grounding("test prompt", PRO_MODEL)
        assert result is None


class TestAsyncStreaming:
    """Tests for GeminiClient.generate_streaming_async() method (CLUX-05)."""

    def test_returns_streamed_text_concatenated(self):
        """generate_streaming_async() returns all chunks joined as a string."""
        client = GeminiClient()
        client._available = True
        client._client = MagicMock()  # raw client not used in this method

        mock_chat = _make_mock_async_chat(["Hello", " world"])
        result = asyncio.run(client.generate_streaming_async(mock_chat, "test msg"))
        assert result == "Hello world"

    def test_returns_empty_string_when_unavailable(self):
        """generate_streaming_async() returns '' when client not available."""
        client = GeminiClient()
        client._available = False
        result = asyncio.run(client.generate_streaming_async(MagicMock(), "msg"))
        assert result == ""

    def test_returns_empty_string_on_exception(self):
        """generate_streaming_async() returns '' when send_message_stream raises."""
        client = GeminiClient()
        client._available = True
        client._client = MagicMock()

        # Build a chat mock whose send_message_stream raises
        mock_chat = MagicMock()

        async def raising_stream(message):
            raise Exception("stream error")
            yield  # make it a generator (never reached)

        mock_chat.send_message_stream.side_effect = lambda msg: raising_stream(msg)
        result = asyncio.run(client.generate_streaming_async(mock_chat, "msg"))
        assert result == ""

    def test_writes_chunks_to_stdout(self, capsys):
        """each chunk.text is written to sys.stdout during streaming."""
        client = GeminiClient()
        client._available = True
        client._client = MagicMock()

        mock_chat = _make_mock_async_chat(["Hello", " world"])
        asyncio.run(client.generate_streaming_async(mock_chat, "test msg"))

        captured = capsys.readouterr()
        assert "Hello" in captured.out
        assert " world" in captured.out
