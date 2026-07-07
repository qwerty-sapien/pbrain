"""Tests for swarm/swarm/interfaces/http.py — HTTPInterface.

Verifies:
- HTTPInterface starts CaptureServer bound to an ISOLATED vault (no singleton leak)
- probe_capture returns {"status": "ok"} when server is running
- A .md capture file appears in the isolated vault (proving vault isolation)
- Port conflict returns error dict, never raises
- probe_capture against stopped/unreachable server returns error dict, never raises
- run_parity_probes returns a list (one result per capture-type action), never raises
"""

import json
import socket
import time
import tempfile
from pathlib import Path

import pytest

from swarm.swarm.interfaces.http import HTTPInterface


def _find_free_port() -> int:
    """Bind port 0 to find a free ephemeral port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def isolated_vault(tmp_path):
    """Return a temporary vault root for testing isolation."""
    vault = tmp_path / "vault"
    vault.mkdir()
    return vault


class TestHTTPInterfaceStart:
    """start() starts the CaptureServer bound to the isolated vault."""

    def test_start_returns_none_on_success(self, isolated_vault):
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        result = iface.start()
        try:
            assert result is None, f"Expected None on successful start, got {result!r}"
        finally:
            iface.stop()

    def test_start_port_conflict_returns_error_dict(self, isolated_vault):
        """Binding an already-occupied port returns error dict, never raises."""
        port = _find_free_port()
        # Occupy the port with a raw socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind(("127.0.0.1", port))
            blocker.listen(1)

            iface = HTTPInterface(vault_path=isolated_vault, port=port)
            result = iface.start()

        assert isinstance(result, dict), f"Expected dict on port conflict, got {result!r}"
        assert result.get("error") == "port_conflict"
        assert result.get("port") == port

    def test_start_port_conflict_does_not_raise(self, isolated_vault):
        """Absolutely must not propagate OSError."""
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind(("127.0.0.1", port))
            blocker.listen(1)

            iface = HTTPInterface(vault_path=isolated_vault, port=port)
            try:
                iface.start()  # must not raise
            except Exception as exc:
                pytest.fail(f"start() raised on port conflict: {exc!r}")


class TestHTTPInterfaceProbeCapture:
    """probe_capture sends POST /capture and returns the server response."""

    def test_probe_capture_ok(self, isolated_vault):
        """Happy path: probe returns {"status": "ok"}."""
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        # Give the daemon thread a moment to bind
        time.sleep(0.15)
        try:
            result = iface.probe_capture(
                url="https://example.com",
                title="Test Page",
                selection="some text",
                tags=["test"],
            )
            assert result.get("status") == "ok", f"Unexpected result: {result!r}"
        finally:
            iface.stop()

    def test_probe_capture_writes_to_isolated_vault(self, isolated_vault):
        """The captured .md file must appear inside isolated_vault, NOT the user's vault."""
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        time.sleep(0.15)
        try:
            result = iface.probe_capture(
                url="https://test-isolation.example",
                title="Isolation Check",
            )
            assert result.get("status") == "ok"
            # Find capture files under the isolated vault
            captures = list(isolated_vault.rglob("*.md"))
            assert len(captures) >= 1, (
                f"No .md file found in isolated vault {isolated_vault}; "
                "CaptureServer may be writing to the singleton vault."
            )
        finally:
            iface.stop()

    def test_probe_capture_after_stop_returns_error_dict(self, isolated_vault):
        """probe_capture against a stopped server returns error dict, never raises."""
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        time.sleep(0.15)
        iface.stop()
        time.sleep(0.1)

        result = iface.probe_capture(url="https://example.com", title="Dead server")
        assert isinstance(result, dict)
        assert "error" in result, f"Expected error key, got {result!r}"

    def test_probe_capture_against_unreachable_port_returns_error_dict(self, isolated_vault):
        """probe_capture against a port with no server never raises."""
        # Use an ephemeral port that nothing is listening on
        port = _find_free_port()
        # Do NOT start the server
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        try:
            result = iface.probe_capture(url="https://example.com", title="No server")
        except Exception as exc:
            pytest.fail(f"probe_capture raised against unreachable port: {exc!r}")

        assert isinstance(result, dict)
        assert "error" in result


class TestHTTPInterfaceRunParityProbes:
    """run_parity_probes returns a list and never raises."""

    def test_empty_actions_returns_empty_list(self, isolated_vault):
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        time.sleep(0.15)
        try:
            results = iface.run_parity_probes([])
            assert isinstance(results, list)
            assert len(results) == 0
        finally:
            iface.stop()

    def test_capture_action_produces_one_result(self, isolated_vault):
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        time.sleep(0.15)
        try:
            semantic_actions = [
                {"intent": "capture", "url": "https://example.com", "title": "Parity test"},
            ]
            results = iface.run_parity_probes(semantic_actions)
            assert isinstance(results, list)
            assert len(results) == 1
            entry = results[0]
            assert "action" in entry or "http_result" in entry or "error" in entry
        finally:
            iface.stop()

    def test_non_capture_actions_return_empty_list(self, isolated_vault):
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        iface.start()
        time.sleep(0.15)
        try:
            semantic_actions = [
                {"intent": "plan", "content": "some planning action"},
                {"intent": "review", "content": "weekly review"},
            ]
            results = iface.run_parity_probes(semantic_actions)
            assert isinstance(results, list)
            assert len(results) == 0
        finally:
            iface.stop()

    def test_port_conflict_parity_probes_returns_finding_list(self, isolated_vault):
        """Port conflict finding propagates through run_parity_probes as a list entry."""
        port = _find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as blocker:
            blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            blocker.bind(("127.0.0.1", port))
            blocker.listen(1)

            iface = HTTPInterface(vault_path=isolated_vault, port=port)
            iface.start()  # will return port_conflict dict, stored internally

            semantic_actions = [
                {"intent": "capture", "url": "https://example.com", "title": "Conflict test"},
            ]
            try:
                results = iface.run_parity_probes(semantic_actions)
            except Exception as exc:
                pytest.fail(f"run_parity_probes raised on port conflict: {exc!r}")

        assert isinstance(results, list)
        assert len(results) >= 1

    def test_run_parity_probes_never_raises(self, isolated_vault):
        """Even with a broken/stopped server, run_parity_probes must not raise."""
        port = _find_free_port()
        iface = HTTPInterface(vault_path=isolated_vault, port=port)
        # Do not start — probe against a non-running server
        semantic_actions = [
            {"intent": "capture", "url": "https://example.com", "title": "No server"},
        ]
        try:
            results = iface.run_parity_probes(semantic_actions)
        except Exception as exc:
            pytest.fail(f"run_parity_probes raised: {exc!r}")
        assert isinstance(results, list)
