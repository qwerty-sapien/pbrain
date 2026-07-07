# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

import pytest
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from pb.cli.console import LatexAwareConsole
from pb.cli.helpers import _render_picker
from pb.cli.markdown import render_markdown_for_terminal, render_markdown_to_rich_blocks, resolve_glow_binary
from pb.core.renderables import renderable_cli_text


def test_renderable_cli_text_handles_common_latex_tokens_globally() -> None:
    rendered = renderable_cli_text(
        r"Use $\mathbb{R}^2$ with $x_n + y^{2}-z_1$, \alpha, and \frac{a+b}{c-d}."
    )

    assert "$" not in rendered
    assert r"\mathbb" not in rendered
    assert "{" not in rendered
    assert "}" not in rendered
    assert "ℝ²" in rendered
    assert "𝑥ₙ" in rendered
    assert "𝑦²" in rendered
    assert "𝑧₁" in rendered
    assert "+" in rendered
    assert "-" in rendered
    assert "α" in rendered
    assert "𝑎+𝑏" in rendered
    assert "─" in rendered
    assert "𝑐-𝑑" in rendered


def test_render_markdown_for_terminal_handles_multiline_display_math_and_preserves_fences() -> None:
    rendered = render_markdown_for_terminal(
        'Before\n'
        '$$\n'
        r'\frac{x_1 + y^2}{\sqrt{n}}'
        '\n$$\n'
        '```python\n'
        'literal = "$x_n$"\n'
        '```\n'
        'After $x_n$'
    )

    assert "$$" not in rendered
    assert r"\frac" not in rendered
    assert "𝑥₁ + 𝑦²" in rendered
    assert "─" in rendered
    assert "√(𝑛)" in rendered
    assert 'literal = "$x_n$"' in rendered
    assert "After 𝑥ₙ" in rendered


def test_picker_rendering_applies_latex_to_headers_labels_and_details() -> None:
    lines = _render_picker(
        [r"[ Identify open/closed sets in R^2 ]", r"Verify $x_n + y^2$"],
        0,
        set(),
        r"Practice in $\mathbb{R}^2$ or inspect $x_n$?",
        False,
        width=140,
        details=[r"Detail: $$\frac{x_1}{y^2}$$"],
        preview_open=True,
    )
    rendered = "\n".join(lines)

    assert r"\mathbb" not in rendered
    assert "$" not in rendered
    assert "Practice in ℝ² or inspect 𝑥ₙ?" in rendered
    assert "[ Identify open/closed sets in ℝ² ]" in rendered
    assert "Verify 𝑥ₙ + 𝑦²" in rendered
    assert "𝑥₁" in rendered
    assert "─" in rendered
    assert "𝑦²" in rendered


def test_latex_aware_console_renders_plain_string_output() -> None:
    console = LatexAwareConsole(record=True, width=80, color_system=None)

    console.print(r"Direct output: $x_n$ in $\mathbb{R}^2$")

    assert "Direct output: 𝑥ₙ in ℝ²" in console.export_text()


def test_renderable_cli_text_settles_nested_and_already_rendered_scripts() -> None:
    assert renderable_cli_text(r"$B_\epsilon(x)$") == "𝐵ₑ(𝑥)"
    assert renderable_cli_text("𝐵_ε") == "𝐵ₑ"


def test_renderable_cli_text_spaces_rendered_latex_from_adjacent_prose() -> None:
    assert renderable_cli_text(r"not$\subseteq$ S") == "not ⊆ S"
    assert renderable_cli_text(r"not\subseteq S") == "not ⊆ S"
    assert renderable_cli_text(r"A$\subseteq$B") == "A ⊆ B"
    assert renderable_cli_text(r"word$x$word") == "word 𝑥 word"
    assert renderable_cli_text(r"$B_\epsilon(p)$ not$\subseteq$ S") == "𝐵ₑ(𝑝) not ⊆ S"


def test_renderable_cli_text_renders_negated_subset_commands_as_single_symbol() -> None:
    assert renderable_cli_text(r"$\not\subseteq$") == "⊈"
    assert renderable_cli_text(r"\(A \not\subseteq B\)") == "𝐴 ⊈ 𝐵"
    assert renderable_cli_text(r"\(A \nsubseteq B\)") == "𝐴 ⊈ 𝐵"


def test_renderable_cli_text_renders_text_macros_as_plain_prose() -> None:
    rendered = renderable_cli_text(r"$(x, y) \in \mathbb{R}^2 \mid x > 0 \text{ and } y < 3$")

    assert rendered == "(𝑥, 𝑦) ∈ ℝ² | 𝑥 > 0 and 𝑦 < 3"
    assert "text" not in rendered


def test_latex_aware_console_recurses_into_rich_renderables() -> None:
    header = Text()
    header.append(r"Open Sets in $\mathbb{R}^2$", style="bold")
    table = Table(title=r"Table $x_n$")
    table.add_column(r"Column $\mathbb{R}^2$")
    table.add_row(r"Cell $B_\epsilon(x)$")
    console = LatexAwareConsole(record=True, width=100, color_system=None)

    console.print(
        Panel(
            Group(header, table),
            title=r"[bold]Header $\mathbb{R}^2$[/]",
        )
    )
    console.rule(r"[header]Rule $x_n$[/]")

    rendered = console.export_text()
    assert r"\mathbb" not in rendered
    assert "$" not in rendered
    assert "Header ℝ²" in rendered
    assert "Open Sets in ℝ²" in rendered
    assert "Table 𝑥ₙ" in rendered
    assert "Column ℝ²" in rendered
    assert "Cell 𝐵ₑ(𝑥)" in rendered
    assert "Rule 𝑥ₙ" in rendered


@pytest.mark.skipif(resolve_glow_binary() is None, reason="Glow is not installed")
def test_markdown_to_rich_blocks_uses_glow_for_basic_markdown_and_preserves_fenced_code() -> None:
    blocks = render_markdown_to_rich_blocks(
        "# Heading $\\mathbb{R}^2$\n\n"
        "Paragraph with **bold**, *italic*, and ***both*** plus $B_\\epsilon(x)$.\n\n"
        "> Quote **bold**\n\n"
        "1. First\n"
        "2. Second\n\n"
        "- Bullet\n\n"
        "`code`\n\n"
        "```python\n"
        'literal = "$x_n$"\n'
        "```\n\n"
        "---\n\n"
        "[link](https://example.com)\n\n"
        "![alt](image.png)\n\n"
        "1968\\. escaped period\n\n"
        "Inline <br> HTML\n",
        width=100,
    )

    rendered = "\n".join(getattr(block, "plain", str(block)) for block in blocks)
    assert "Heading ℝ²" in rendered
    assert "bold" in rendered
    assert "italic" in rendered
    assert "both" in rendered
    assert "Quote bold" in rendered
    assert "First" in rendered and "Second" in rendered
    assert "Bullet" in rendered
    assert "code" in rendered
    assert 'literal = "$x_n$"' in rendered
    assert "𝐵ₑ(𝑥)" in rendered
    assert "link" in rendered and "https://example.com" in rendered
    assert "alt" in rendered and "image.png" in rendered
    assert "1968. escaped period" in rendered
    assert "Inline" in rendered and "HTML" in rendered
    assert "**" not in rendered
    assert "***" not in rendered
    assert "```" not in rendered
    assert r"\mathbb" not in rendered
