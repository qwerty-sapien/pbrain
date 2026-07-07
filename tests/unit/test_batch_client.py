"""Unit tests for pb.vault.batch_client — VertexBatchClient transport layer.

Tests cover:
- build_batch_line: structure and config override
- VertexBatchClient.__init__: lazy validation
- VertexBatchClient.submit: early rejection when config empty
- TERMINAL_STATES module attribute
"""

from __future__ import annotations

import pytest

from pb.vault.batch_client import (
    TERMINAL_STATES,
    VertexBatchClient,
    build_batch_line,
)


class TestBuildBatchLine:
    """Tests for build_batch_line helper."""

    def test_default_shape(self):
        """Test 1: build_batch_line returns expected structure with defaults."""
        result = build_batch_line(key="k1", prompt="P", system_instruction="SI")
        assert result["key"] == "k1"
        req = result["request"]
        assert req["contents"] == [{"role": "user", "parts": [{"text": "P"}]}]
        assert req["system_instruction"] == {"parts": [{"text": "SI"}]}
        assert req["generationConfig"]["temperature"] == 0.4
        assert req["generationConfig"]["maxOutputTokens"] == 4000

    def test_override_values(self):
        """Test 5: override temperature and max_output_tokens propagate correctly."""
        result = build_batch_line(
            key="k",
            prompt="P",
            system_instruction="SI",
            temperature=0.7,
            max_output_tokens=2048,
        )
        assert result["request"]["generationConfig"]["temperature"] == 0.7
        assert result["request"]["generationConfig"]["maxOutputTokens"] == 2048


class TestVertexBatchClientInit:
    """Tests for VertexBatchClient instantiation."""

    def test_instantiates_without_error_on_empty_config(self):
        """Test 2: empty gcp_project/gcs_bucket instantiates fine (validation deferred to submit)."""
        client = VertexBatchClient(gcp_project="", gcs_bucket="", location="us-central1")
        assert client.gcp_project == ""
        assert client.gcs_bucket == ""
        assert client.location == "us-central1"


class TestVertexBatchClientSubmit:
    """Tests for VertexBatchClient.submit()."""

    def test_submit_raises_when_gcp_project_empty(self):
        """Test 3: raises ValueError containing 'GCS not configured' when gcp_project empty."""
        client = VertexBatchClient(gcp_project="", gcs_bucket="my-bucket")
        with pytest.raises(ValueError, match="GCS not configured"):
            client.submit(requests=[], model="any", run_id="r1")

    def test_submit_raises_when_gcs_bucket_empty(self):
        """Test 3 variant: raises ValueError when gcs_bucket is empty."""
        client = VertexBatchClient(gcp_project="my-project", gcs_bucket="")
        with pytest.raises(ValueError, match="GCS not configured"):
            client.submit(requests=[], model="any", run_id="r1")


class TestTerminalStates:
    """Tests for TERMINAL_STATES module attribute."""

    def test_terminal_states_set(self):
        """Test 4: TERMINAL_STATES equals expected set."""
        assert TERMINAL_STATES == {
            "JOB_STATE_SUCCEEDED",
            "JOB_STATE_FAILED",
            "JOB_STATE_CANCELLED",
            "JOB_STATE_EXPIRED",
        }
