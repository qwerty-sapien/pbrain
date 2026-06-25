# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Browser capture server -- local HTTP endpoint for bookmarklet capture (D-10, D-11, D-12).

Security mitigations:
- T-07-09: Binds to 127.0.0.1 only (never 0.0.0.0)
- T-07-10: Content-Length capped at 64KB
- T-07-11: make_slug for filenames (no path traversal via URL/title)
- T-07-12: yaml.dump for frontmatter (no YAML injection)
"""

from __future__ import annotations

import json
import threading
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from typing import Optional

import structlog
import yaml

from pb.core.graph_writer import make_slug

logger = structlog.get_logger()

DEFAULT_PORT = 9421
MAX_CONTENT_LENGTH = 65536  # 64KB cap


class CaptureHandler(BaseHTTPRequestHandler):
    """HTTP handler for bookmarklet capture POST requests."""

    vault_path: Path  # Set by CaptureServer before serving

    def do_POST(self):
        if self.path != "/capture":
            self.send_response(404)
            self.end_headers()
            return

        # Security: cap content length
        length = int(self.headers.get("Content-Length", 0))
        if length > MAX_CONTENT_LENGTH:
            self.send_response(413)
            self.end_headers()
            return

        body = self.rfile.read(length)
        try:
            data = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"error": "invalid JSON"}')
            return

        url = str(data.get("url", ""))
        title = str(data.get("title", "Untitled"))
        selection = str(data.get("selection", ""))
        tags = data.get("tags", [])
        if not isinstance(tags, list):
            tags = []
        tags = [str(t) for t in tags]

        try:
            _write_capture_note(self.vault_path, url, title, selection, tags)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(b'{"status": "ok"}')
            logger.debug("capture.saved", url=url[:80], title=title[:50])
        except Exception as e:
            logger.warning("capture.write_failed", error=str(e))
            self.send_response(500)
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()

    def do_OPTIONS(self):
        """CORS preflight handler."""
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def log_message(self, format, *args):
        """Suppress default HTTP server logging (use structlog instead)."""
        pass


def _write_capture_note(
    vault_path: Path, url: str, title: str, selection: str, tags: list[str]
) -> Path:
    """Write a capture note to 00-inbox/captures/.

    Returns the path to the created file.
    """
    captures_dir = vault_path / "00-inbox" / "captures"
    captures_dir.mkdir(parents=True, exist_ok=True)

    now = datetime.utcnow()
    date_str = now.strftime("%Y-%m-%d")
    slug = make_slug(title) if title else "untitled"
    filename = f"{date_str}-{slug}.md"
    filepath = captures_dir / filename

    # Avoid overwriting: append counter if exists
    counter = 1
    while filepath.exists():
        filename = f"{date_str}-{slug}-{counter}.md"
        filepath = captures_dir / filename
        counter += 1

    frontmatter = {
        "type": "capture",
        "url": url,
        "title": title,
        "captured": now.isoformat(),
        "tags": tags,
        "source": "bookmarklet",
    }

    parts = ["---"]
    parts.append(yaml.dump(frontmatter, default_flow_style=False, sort_keys=False).strip())
    parts.append("---")
    parts.append("")
    parts.append(f"# {title}")
    parts.append("")
    parts.append(url)

    if selection:
        parts.append("")
        parts.append("## Selection")
        parts.append("")
        parts.append(selection)

    filepath.write_text("\n".join(parts) + "\n")
    return filepath


class CaptureServer:
    """Local HTTP server for bookmarklet capture.

    Runs in a daemon thread. Binds to 127.0.0.1 only (never 0.0.0.0).
    """

    def __init__(self, port: int = DEFAULT_PORT, vault_path: Optional[Path] = None):
        self.port = port
        if vault_path is not None:
            self.vault_path = vault_path
        else:
            from pb.storage.config import get_vault_path
            self.vault_path = get_vault_path()
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        """Start the capture server (blocking in foreground)."""
        CaptureHandler.vault_path = self.vault_path
        self._server = HTTPServer(("127.0.0.1", self.port), CaptureHandler)
        logger.info("capture.server_start", port=self.port, address="127.0.0.1")
        try:
            self._server.serve_forever()
        except KeyboardInterrupt:
            self.stop()

    def start_background(self) -> None:
        """Start in a daemon thread (for testing)."""
        CaptureHandler.vault_path = self.vault_path
        self._server = HTTPServer(("127.0.0.1", self.port), CaptureHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the server."""
        if self._server:
            self._server.shutdown()
            logger.info("capture.server_stop")


def generate_bookmarklet(port: int = DEFAULT_PORT) -> str:
    """Generate bookmarklet JavaScript for browser bookmark bar.

    The bookmarklet injects a modal overlay showing pre-filled URL, title,
    and selection, with a tags input. User clicks Save to POST to localhost.
    """
    js = (
        "javascript:(function(){"
        "var d=document,url=d.URL,title=d.title,"
        "sel=d.getSelection?d.getSelection().toString():'';"
        "var m=d.createElement('div');"
        "m.style.cssText='position:fixed;top:20px;right:20px;z-index:99999;"
        "background:#fff;border:2px solid #333;padding:16px;width:360px;"
        "font-family:sans-serif;box-shadow:0 4px 12px rgba(0,0,0,.3);border-radius:8px';"
        "m.innerHTML=\"<b>Capture to pb</b><br><br>"
        "<label>URL:<br><input id='pb-url' style='width:100%;box-sizing:border-box' value='\"+url.replace(/'/g,\"\\\\'\").replace(/\"/g,'&quot;')+\"'></label><br><br>"
        "<label>Title:<br><input id='pb-title' style='width:100%;box-sizing:border-box' value='\"+title.replace(/'/g,\"\\\\'\").replace(/\"/g,'&quot;')+\"'></label><br><br>"
        "<label>Tags:<br><input id='pb-tags' style='width:100%;box-sizing:border-box' placeholder='tag1 tag2'></label><br><br>"
        "<button id='pb-save' style='padding:6px 16px;cursor:pointer'>Save</button> "
        "<button id='pb-cancel' style='padding:6px 16px;cursor:pointer'>Cancel</button>\";"
        "d.body.appendChild(m);"
        "d.getElementById('pb-cancel').onclick=function(){m.remove()};"
        "d.getElementById('pb-save').onclick=function(){"
        "fetch('http://localhost:" + str(port) + "/capture',"
        "{method:'POST',headers:{'Content-Type':'application/json'},"
        "body:JSON.stringify({url:d.getElementById('pb-url').value,"
        "title:d.getElementById('pb-title').value,selection:sel,"
        "tags:d.getElementById('pb-tags').value.split(' ').filter(Boolean)})})"
        ".then(function(){m.innerHTML=\"<span style='color:green'>Saved!</span>\";"
        "setTimeout(function(){m.remove()},1200)})"
        ".catch(function(){m.innerHTML=\"<span style='color:red'>Failed -- is pb capture server running?</span>\";"
        "setTimeout(function(){m.remove()},3000)});};"
        "})()"
    )
    return js
