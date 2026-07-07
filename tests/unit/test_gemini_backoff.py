"""Unit tests for the 429/RESOURCE_EXHAUSTED retry backoff.

Contract: exponential backoff with jitter (equal-jitter variant), so concurrent
clients hitting the same quota do not retry in lockstep. base=1.5s, doubling per
attempt, capped, with a positive floor (cap/2 of the current band).
"""

import random

from unittest.mock import MagicMock

from pb.llm import gemini
from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL


class TestRetryBackoffSeconds:
    def test_attempt_one_within_equal_jitter_band(self):
        # base=1.5, temp=min(cap, 1.5*2**0)=1.5 -> equal jitter band [0.75, 1.5]
        for _ in range(200):
            s = gemini._retry_backoff_seconds(1)
            assert 0.75 <= s <= 1.5

    def test_attempt_two_band(self):
        # temp = 1.5*2**1 = 3.0 -> [1.5, 3.0]
        for _ in range(200):
            s = gemini._retry_backoff_seconds(2)
            assert 1.5 <= s <= 3.0

    def test_attempt_three_band(self):
        # temp = 1.5*2**2 = 6.0 -> [3.0, 6.0]
        for _ in range(200):
            s = gemini._retry_backoff_seconds(3)
            assert 3.0 <= s <= 6.0

    def test_is_exponential_not_linear(self):
        # attempt-3 lower bound (3.0) strictly exceeds attempt-1 upper bound (1.5),
        # so growth is exponential regardless of which jitter value is drawn.
        lo3 = min(gemini._retry_backoff_seconds(3) for _ in range(200))
        hi1 = max(gemini._retry_backoff_seconds(1) for _ in range(200))
        assert lo3 > hi1

    def test_cap_respected(self):
        for _ in range(200):
            s = gemini._retry_backoff_seconds(10, cap=8.0)
            assert s <= 8.0
            assert s >= 4.0  # equal-jitter floor = cap/2 once saturated

    def test_jitter_varies(self):
        vals = {gemini._retry_backoff_seconds(2) for _ in range(100)}
        assert len(vals) > 1  # jitter must introduce variation

    def test_deterministic_with_seeded_rng(self):
        a = gemini._retry_backoff_seconds(2, rng=random.Random(123))
        b = gemini._retry_backoff_seconds(2, rng=random.Random(123))
        assert a == b

    def test_floor_is_positive(self):
        # never sleeps zero — avoids hammering the endpoint on a 429
        for _ in range(200):
            assert gemini._retry_backoff_seconds(1) > 0


class TestRateLimitStableFallback:
    """On a 429/RESOURCE_EXHAUSTED for a gemini PREVIEW model, the client must
    fall back to the verified stable GA model (real/separate quota) instead of
    blacking out. Gemini-specific (AI Studio + Vertex share these IDs). IDs
    verified 2026-06-03; Pro target (gemini-2.5-pro) user-confirmed.
    """

    def _client(self, behavior):
        client = gemini.GeminiClient()
        client._available = True
        mock = MagicMock()
        mock.models.generate_content.side_effect = behavior
        client._client = mock
        return client

    def test_stable_fallback_map_targets_verified_ga_ids(self):
        assert gemini._RATE_LIMIT_STABLE_FALLBACK[FLASH_MODEL] == "gemini-3.5-flash"
        assert gemini._RATE_LIMIT_STABLE_FALLBACK[FLASH_LITE_MODEL] == "gemini-3.1-flash-lite"
        assert gemini._RATE_LIMIT_STABLE_FALLBACK[PRO_MODEL] == "gemini-2.5-pro"

    def test_no_fallback_for_already_stable_or_non_gemini(self):
        assert gemini._stable_fallback_for("gemini-3.5-flash") is None
        assert gemini._stable_fallback_for("gemini-2.5-pro") is None
        assert gemini._stable_fallback_for("claude-sonnet-4-6") is None
        assert gemini._stable_fallback_for("") is None

    def test_429_on_preview_flash_falls_back_to_stable(self, monkeypatch):
        monkeypatch.setattr("pb.llm.gemini.time.sleep", lambda *_a, **_k: None)
        calls = []

        def behavior(**kwargs):
            model = kwargs.get("model")
            calls.append(model)
            if model == FLASH_MODEL:
                raise Exception("429 RESOURCE_EXHAUSTED: Quota exceeded for model")
            resp = MagicMock()
            resp.text = "ok-from-stable"
            return resp

        result = self._client(behavior).generate_with_model_result("hi", FLASH_MODEL, timeout=5)
        assert result.error is None, f"expected stable success, got {result.error}"
        assert result.text == "ok-from-stable"
        assert "gemini-3.5-flash" in calls, f"never switched to stable: {calls}"

    def test_429_on_preview_pro_falls_back_to_2_5_pro(self, monkeypatch):
        monkeypatch.setattr("pb.llm.gemini.time.sleep", lambda *_a, **_k: None)
        calls = []

        def behavior(**kwargs):
            model = kwargs.get("model")
            calls.append(model)
            if model == PRO_MODEL:
                raise Exception("RESOURCE_EXHAUSTED: rate limit / quota")
            resp = MagicMock()
            resp.text = "pro-stable-ok"
            return resp

        result = self._client(behavior).generate_with_model_result("hi", PRO_MODEL, timeout=5)
        assert result.text == "pro-stable-ok"
        assert "gemini-2.5-pro" in calls

    def test_non_rate_limit_error_does_not_fall_back(self, monkeypatch):
        monkeypatch.setattr("pb.llm.gemini.time.sleep", lambda *_a, **_k: None)
        calls = []

        def behavior(**kwargs):
            calls.append(kwargs.get("model"))
            raise Exception("400 INVALID_ARGUMENT: bad request")

        result = self._client(behavior).generate_with_model_result("hi", FLASH_MODEL, timeout=5)
        assert result.error is not None
        assert result.error.category == "config"
        # A non-retryable error must never trigger the stable swap.
        assert "gemini-3.5-flash" not in calls, f"should not fall back on config error: {calls}"
