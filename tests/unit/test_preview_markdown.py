from __future__ import annotations

from unittest.mock import patch

from pb.cli.preview import markdown_step_lines, render_markdown_preview


def test_render_markdown_preview_renders_rich_panel():
    with patch("pb.cli.preview.get_console") as mock_console:
        render_markdown_preview(
            title="Study Block Draft",
            rows=[("Scope", "Linear algebra"), ("Bloom target", "apply")],
            sections=[("Steps", ["1. **Recall**", "   *Define the core term*"])],
        )

    mock_console().print.assert_called_once()


def test_render_markdown_preview_allows_empty_section_title():
    with patch("pb.cli.preview.get_console") as mock_console:
        render_markdown_preview(
            title="Goal Draft",
            sections=[("", ["# Linear Algebra", "", "### mathematics"])],
        )

    mock_console().print.assert_called_once()


def test_markdown_step_lines_use_spaced_numbered_format():
    lines = markdown_step_lines(
        [
            {
                "title": "Recall",
                "instruction": "Define the core term.",
                "success_check": "You can say it cleanly.",
            }
        ]
    )

    assert lines == [
        "1. **Recall**",
        "   - Do: Define the core term.",
        "   - Check: You can say it cleanly.",
    ]
