from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from pb.cli.markdown import (
    render_markdown,
    render_markdown_for_terminal,
    resolve_glow_binary,
    resolve_glow_style_path,
)


def test_resolve_glow_binary_prefers_pb_glow_path(tmp_path, monkeypatch):
    glow = tmp_path / "glow"
    glow.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    glow.chmod(0o755)

    monkeypatch.setenv("PB_GLOW_PATH", str(glow))
    monkeypatch.setattr("pb.cli.markdown.shutil.which", lambda _: None)

    assert resolve_glow_binary() == str(glow)


def test_resolve_glow_binary_uses_path_lookup(monkeypatch):
    monkeypatch.delenv("PB_GLOW_PATH", raising=False)
    monkeypatch.setattr("pb.cli.markdown.shutil.which", lambda _: "/usr/local/bin/glow")
    monkeypatch.setattr(Path, "is_file", lambda self: str(self) == "/usr/local/bin/glow")
    monkeypatch.setattr("pb.cli.markdown.os.access", lambda path, mode: str(path) == "/usr/local/bin/glow")

    assert resolve_glow_binary() == "/usr/local/bin/glow"


def test_resolve_glow_style_path_uses_bundled_style():
    style_path = resolve_glow_style_path()

    assert style_path is not None
    assert style_path.endswith("glow_style.json")


def test_render_markdown_falls_back_to_rich_markdown(monkeypatch):
    monkeypatch.setattr("pb.cli.markdown.resolve_glow_binary", lambda: None)
    fake_console = SimpleNamespace(print=lambda *args, **kwargs: None)

    with patch("pb.cli.markdown.get_console", return_value=fake_console) as mock_console:
        rendered = render_markdown("# hello")

    assert rendered is True
    mock_console.assert_called_once()


def test_render_markdown_uses_glow_when_available(monkeypatch):
    monkeypatch.setattr("pb.cli.markdown.resolve_glow_binary", lambda: "/usr/local/bin/glow")
    monkeypatch.setattr("pb.cli.markdown.resolve_glow_style_path", lambda: "/tmp/glow-style.json")
    monkeypatch.setattr("pb.cli.markdown.resolve_render_width", lambda: 72)

    with patch("pb.cli.markdown.subprocess.run") as mock_run:
        rendered = render_markdown("# hello")

    assert rendered is True
    args, kwargs = mock_run.call_args
    assert args[0] == ["/usr/local/bin/glow", "-s", "/tmp/glow-style.json", "-w", "72", "-"]
    assert kwargs["input"] == "# hello"
    assert kwargs["text"] is True
    assert kwargs["check"] is False
    assert "GLOW_CONFIG_HOME" in kwargs["env"]


def test_render_markdown_preprocesses_latex_for_glow(monkeypatch):
    monkeypatch.setattr("pb.cli.markdown.resolve_glow_binary", lambda: "/usr/local/bin/glow")
    monkeypatch.setattr("pb.cli.markdown.resolve_glow_style_path", lambda: None)
    monkeypatch.setattr("pb.cli.markdown.resolve_render_width", lambda: 72)

    with patch("pb.cli.markdown.subprocess.run") as mock_run:
        rendered = render_markdown(r"Open sets in \(\mathbb{R}^n\).")

    assert rendered is True
    _, kwargs = mock_run.call_args
    assert "ℝⁿ" in kwargs["input"]
    assert r"\mathbb" not in kwargs["input"]


def test_render_markdown_for_terminal_preserves_fenced_code():
    rendered = render_markdown_for_terminal(
        "Outside \\(\\mathbb{R}^n\\)\n"
        "```text\n"
        "Inside \\(\\mathbb{R}^n\\)\n"
        "```\n"
    )

    assert "Outside ℝⁿ" in rendered
    assert r"Inside \(\mathbb{R}^n\)" in rendered
