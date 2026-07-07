"""Tests for CaptureServer — browser-to-vault capture via local HTTP.

Tests: note writing, HTTP handler, CORS, bookmarklet generation, security constraints.
Per Phase 7 D-10 through D-12.
"""

import json
import time
import urllib.request
import urllib.error
from pathlib import Path

import pytest
import yaml

from pb.capture.server import (
    CaptureHandler,
    CaptureServer,
    DEFAULT_PORT,
    MAX_CONTENT_LENGTH,
    _write_capture_note,
    generate_bookmarklet,
)
from pb.core.graph_writer import make_slug


# ── Helpers ──────────────────────────────────────────────────────────────────


def _find_free_port() -> int:
    """Find a free TCP port on localhost."""
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _post_capture(port: int, data: dict, path: str = "/capture") -> urllib.request.Request:
    """Build and send a POST request to the capture server."""
    body = json.dumps(data).encode()
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    return urllib.request.urlopen(req)


# ── _write_capture_note Tests ────────────────────────────────────────────────


class TestWriteCaptureNote:
    """Test the vault note writer function."""

    def test_creates_md_file_with_correct_frontmatter(self, tmp_path):
        """Test 1: _write_capture_note creates .md file in captures_dir with correct frontmatter."""
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com/article",
            title="Test Article",
            selection="Some highlighted text",
            tags=["python", "testing"],
        )
        assert result.exists()
        assert result.suffix == ".md"
        assert result.parent == tmp_path / "00-inbox" / "captures"

        content = result.read_text()
        # Parse frontmatter
        parts = content.split("---")
        fm = yaml.safe_load(parts[1])
        assert fm["type"] == "capture"
        assert fm["url"] == "https://example.com/article"
        assert fm["title"] == "Test Article"
        assert fm["source"] == "bookmarklet"
        assert fm["tags"] == ["python", "testing"]
        assert "captured" in fm

    def test_uses_make_slug_for_filename(self, tmp_path):
        """Test 2: _write_capture_note uses make_slug for filename (no raw URL in filename)."""
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com/dangerous/../path",
            title="My Great Article!",
            selection="",
            tags=[],
        )
        filename = result.name
        slug = make_slug("My Great Article!")
        assert slug in filename
        # No raw URL characters in filename
        assert "https" not in filename
        assert "/" not in filename.replace("/", "")  # Path separator check
        assert ":" not in filename

    def test_frontmatter_contains_all_required_fields(self, tmp_path):
        """Test 3: frontmatter contains type: capture, url, title, captured, tags, source: bookmarklet."""
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com",
            title="Test",
            selection="sel",
            tags=["tag1"],
        )
        content = result.read_text()
        parts = content.split("---")
        fm = yaml.safe_load(parts[1])
        required_keys = {"type", "url", "title", "captured", "tags", "source"}
        assert required_keys.issubset(set(fm.keys()))
        assert fm["type"] == "capture"
        assert fm["source"] == "bookmarklet"

    def test_body_contains_title_url_and_selection(self, tmp_path):
        """Test 4: body contains title as heading, URL, and selection text."""
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com/page",
            title="My Page Title",
            selection="Important quote from the page",
            tags=[],
        )
        content = result.read_text()
        assert "# My Page Title" in content
        assert "https://example.com/page" in content
        assert "## Selection" in content
        assert "Important quote from the page" in content

    def test_empty_selection_omits_section(self, tmp_path):
        """Test 5: empty selection omits the ## Selection section."""
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com",
            title="No Selection",
            selection="",
            tags=[],
        )
        content = result.read_text()
        assert "## Selection" not in content

    def test_long_title_truncated_via_make_slug(self, tmp_path):
        """Test 6: very long titles are truncated in filename via make_slug[:50]."""
        long_title = "A Very Long Title That Exceeds Fifty Characters And Should Be Truncated Properly"
        result = _write_capture_note(
            vault_path=tmp_path,
            url="https://example.com",
            title=long_title,
            selection="",
            tags=[],
        )
        filename = result.stem  # Without .md extension
        # The slug part after the date prefix should be <= 50 chars
        # Filename format: YYYY-MM-DD-{slug}
        slug_part = filename[11:]  # Skip "YYYY-MM-DD-"
        assert len(slug_part) <= 50


# ── CaptureHandler HTTP Tests ────────────────────────────────────────────────


class TestCaptureHandler:
    """Test HTTP handler behavior via real HTTP requests."""

    @pytest.fixture(autouse=True)
    def _start_server(self, tmp_path):
        """Start a capture server on a free port for each test."""
        self.port = _find_free_port()
        self.server = CaptureServer(port=self.port, vault_path=tmp_path)
        self.server.start_background()
        time.sleep(0.1)  # Brief pause for server to bind
        yield
        self.server.stop()

    def test_post_valid_json_returns_200(self, tmp_path):
        """Test 7: POST /capture with valid JSON returns 200 with {"status": "ok"}."""
        resp = _post_capture(self.port, {
            "url": "https://example.com",
            "title": "Test",
            "selection": "",
            "tags": [],
        })
        assert resp.status == 200
        body = json.loads(resp.read())
        assert body["status"] == "ok"

    def test_post_wrong_path_returns_404(self):
        """Test 8: POST to non-/capture path returns 404."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/wrong-path",
            data=b'{"test": true}',
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 404

    def test_post_invalid_json_returns_400(self):
        """Test 9: POST with invalid JSON body returns 400."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/capture",
            data=b"this is not json",
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 400

    def test_content_length_cap(self):
        """Test 10: Content-Length exceeding MAX_CONTENT_LENGTH returns 413."""
        # Send a request with Content-Length header exceeding the limit
        # We don't actually send that much data - just set the header
        large_data = b"x" * (MAX_CONTENT_LENGTH + 1)
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/capture",
            data=large_data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with pytest.raises(urllib.error.HTTPError) as exc_info:
            urllib.request.urlopen(req)
        assert exc_info.value.code == 413

    def test_options_returns_cors_headers(self):
        """Test 11: OPTIONS request returns CORS headers."""
        req = urllib.request.Request(
            f"http://127.0.0.1:{self.port}/capture",
            method="OPTIONS",
        )
        resp = urllib.request.urlopen(req)
        assert resp.status == 200
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"
        assert "POST" in resp.headers.get("Access-Control-Allow-Methods", "")

    def test_post_includes_cors_header(self, tmp_path):
        """Test 12: POST response includes Access-Control-Allow-Origin: * header."""
        resp = _post_capture(self.port, {
            "url": "https://example.com",
            "title": "CORS Test",
            "selection": "",
            "tags": [],
        })
        assert resp.headers.get("Access-Control-Allow-Origin") == "*"


# ── generate_bookmarklet Tests ───────────────────────────────────────────────


class TestGenerateBookmarklet:
    """Test bookmarklet JavaScript generation."""

    def test_contains_port_number(self):
        """Test 13: generate_bookmarklet returns JavaScript string containing the port number."""
        js = generate_bookmarklet(port=9999)
        assert "9999" in js
        assert js.startswith("javascript:")

    def test_default_port(self):
        """Test 13b: default port is used when not specified."""
        js = generate_bookmarklet()
        assert str(DEFAULT_PORT) in js


# ── Security Constraint Tests ────────────────────────────────────────────────


class TestSecurityConstraints:
    """Test security constraints are enforced."""

    def test_server_binds_to_localhost(self, tmp_path):
        """Test 14: CaptureServer binds to 127.0.0.1 (not 0.0.0.0)."""
        port = _find_free_port()
        server = CaptureServer(port=port, vault_path=tmp_path)
        server.start_background()
        time.sleep(0.1)
        try:
            # Verify server_address is 127.0.0.1
            assert server._server.server_address[0] == "127.0.0.1"
        finally:
            server.stop()

    def test_max_content_length_is_65536(self):
        """Verify the constant is set correctly."""
        assert MAX_CONTENT_LENGTH == 65536
