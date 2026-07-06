# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared helpers for learner-facing text that may contain explicit LaTeX."""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, model_validator


class RenderableText(BaseModel):
    """Text payload with explicit LaTeX opt-in."""

    text: str = ""
    is_latex: bool = False

    @model_validator(mode="before")
    @classmethod
    def _coerce(cls, value: Any) -> Any:
        if isinstance(value, cls):
            return value.model_dump(mode="python")
        if value is None:
            return {"text": "", "is_latex": False}
        if isinstance(value, str):
            return {"text": value, "is_latex": False}
        if isinstance(value, dict):
            if "text" not in value:
                raise ValueError("RenderableText objects must include a text field.")
            return {
                "text": str(value.get("text", "")),
                "is_latex": bool(value.get("is_latex", False)),
            }
        raise TypeError(f"Unsupported renderable value: {type(value)!r}")


def ensure_renderable_text(value: RenderableText | str | dict[str, Any] | None) -> RenderableText:
    """Coerce a value into RenderableText."""
    if isinstance(value, RenderableText):
        return value
    return RenderableText.model_validate(value)


def renderable_plain_text(value: RenderableText | str | dict[str, Any] | None) -> str:
    """Return the literal stored text without implicit math handling."""
    return ensure_renderable_text(value).text


def renderable_cli_text(value: RenderableText | str | dict[str, Any] | None) -> str:
    """Return text for terminal previews.

    Explicit LaTeX is rendered into a terminal-friendly approximation so the
    shell preview can distinguish math from plain prose without relying on a
    browser-based MathJax renderer.
    """
    item = ensure_renderable_text(value)
    if item.is_latex:
        core, _ = _unwrap_latex(item.text)
        return _latex_to_terminal(core)
    result = _rewrite_inline_latex(item.text, mode="cli")
    return _apply_bare_math_until_stable(result)


def renderable_markdown_text(value: RenderableText | str | dict[str, Any] | None) -> str:
    """Return Markdown-friendly text for durable learner artifacts."""
    item = ensure_renderable_text(value)
    if item.is_latex:
        core, display = _unwrap_latex(item.text)
        if display:
            return f"$$\n{core}\n$$"
        return f"${core}$"
    return _rewrite_inline_latex(item.text, mode="markdown")


def renderable_anki_text(value: RenderableText | str | dict[str, Any] | None) -> str:
    """Return Anki-safe MathJax delimiters for explicit LaTeX."""
    item = ensure_renderable_text(value)
    if item.is_latex:
        core, display = _unwrap_latex(item.text)
        if display:
            return f"\\[{core}\\]"
        return f"\\({core}\\)"
    return _rewrite_inline_latex(item.text, mode="anki")


def renderable_payload(value: RenderableText | str | dict[str, Any] | None) -> dict[str, Any]:
    """Return a JSON-safe payload that preserves explicit LaTeX typing."""
    item = ensure_renderable_text(value)
    return {"text": item.text, "is_latex": item.is_latex}


def _unwrap_latex(text: str) -> tuple[str, bool]:
    """Strip common math delimiters and infer inline vs display mode."""
    stripped = (text or "").strip()
    wrappers = (
        ("\\[", "\\]", True),
        ("$$", "$$", True),
        ("[$$]", "[/$$]", True),
        ("\\(", "\\)", False),
        ("[$]", "[/$]", False),
        ("$", "$", False),
    )
    for prefix, suffix, display in wrappers:
        if stripped.startswith(prefix) and stripped.endswith(suffix):
            core = stripped[len(prefix) : len(stripped) - len(suffix)].strip()
            return core, display or _looks_like_display_math(core)
    return stripped, _looks_like_display_math(stripped)


def _looks_like_display_math(text: str) -> bool:
    stripped = (text or "").strip()
    return (
        "\n" in stripped
        or "\\begin{displaymath}" in stripped
        or "\\begin{equation" in stripped
        or "\\begin{align" in stripped
    )


_SYMBOL_RENDER = {
    "alpha": "α",
    "beta": "β",
    "gamma": "γ",
    "delta": "δ",
    "epsilon": "ε",
    "varepsilon": "ε",
    "zeta": "ζ",
    "eta": "η",
    "theta": "θ",
    "vartheta": "θ",
    "iota": "ι",
    "kappa": "κ",
    "lambda": "λ",
    "mu": "μ",
    "nu": "ν",
    "xi": "ξ",
    "pi": "π",
    "varpi": "π",
    "rho": "ρ",
    "varrho": "ρ",
    "sigma": "σ",
    "varsigma": "ς",
    "tau": "τ",
    "upsilon": "υ",
    "phi": "φ",
    "varphi": "φ",
    "chi": "χ",
    "psi": "ψ",
    "omega": "ω",
    "Gamma": "Γ",
    "Delta": "Δ",
    "Theta": "Θ",
    "Lambda": "Λ",
    "Xi": "Ξ",
    "Pi": "Π",
    "Sigma": "Σ",
    "Phi": "Φ",
    "Psi": "Ψ",
    "Omega": "Ω",
    "nabla": "∇",
    "partial": "∂",
    "infty": "∞",
    "cdot": "·",
    "times": "×",
    "div": "÷",
    "pm": "±",
    "mp": "∓",
    "leq": "≤",
    "le": "≤",
    "geq": "≥",
    "ge": "≥",
    "neq": "≠",
    "ne": "≠",
    "approx": "≈",
    "cong": "≅",
    "sim": "~",
    "to": "→",
    "rightarrow": "→",
    "leftarrow": "←",
    "Rightarrow": "⟹",
    "Leftarrow": "⟸",
    "Leftrightarrow": "⟺",
    "mapsto": "↦",
    "langle": "⟨",
    "rangle": "⟩",
    "int": "∫",
    "oint": "∮",
    "wedge": "∧",
    "in": "∈",
    "notin": "∉",
    "nsubset": "⊄",
    "nsubseteq": "⊈",
    "nsupset": "⊅",
    "nsupseteq": "⊉",
    "nleq": "≰",
    "nle": "≰",
    "ngeq": "≱",
    "nge": "≱",
    "mid": "|",
    "vert": "|",
    "lvert": "|",
    "rvert": "|",
    "forall": "∀",
    "exists": "∃",
    "nexists": "∄",
    "subset": "⊂",
    "subseteq": "⊆",
    "supset": "⊃",
    "supseteq": "⊇",
    "cup": "∪",
    "cap": "∩",
    "emptyset": "∅",
    "varnothing": "∅",
    "ldots": "…",
    "cdots": "⋯",
    "therefore": "∴",
    "because": "∵",
    "ell": "ℓ",
    "hbar": "ℏ",
    "oplus": "⊕",
    "otimes": "⊗",
    "lfloor": "⌊",
    "rfloor": "⌋",
    "lceil": "⌈",
    "rceil": "⌉",
    "perp": "⊥",
    "parallel": "∥",
}

_NEGATED_SYMBOL_RENDER = (
    ("subseteq", "⊈"),
    ("subset", "⊄"),
    ("supseteq", "⊉"),
    ("supset", "⊅"),
    ("in", "∉"),
    ("leq", "≰"),
    ("le", "≰"),
    ("geq", "≱"),
    ("ge", "≱"),
    ("equal", "≠"),
)

_TEXT_BRACE_MACROS = (
    "textnormal",
    "operatorname",
    "textrm",
    "textbf",
    "textit",
    "mathrm",
    "mbox",
    "text",
)

_MATHBB_MAP = {
    "R": "ℝ",  # U+211D real numbers
    "N": "ℕ",  # U+2115 natural numbers
    "Z": "ℤ",  # U+2124 integers
    "Q": "ℚ",  # U+211A rationals
    "C": "ℂ",  # U+2102 complex numbers
}

_SUBSCRIPT_MAP = {
    "0": "₀",
    "1": "₁",
    "2": "₂",
    "3": "₃",
    "4": "₄",
    "5": "₅",
    "6": "₆",
    "7": "₇",
    "8": "₈",
    "9": "₉",
    "+": "₊",
    "-": "₋",
    "=": "₌",
    "(": "₍",
    ")": "₎",
    "a": "ₐ",
    "e": "ₑ",
    "h": "ₕ",
    "i": "ᵢ",
    "j": "ⱼ",
    "k": "ₖ",
    "l": "ₗ",
    "m": "ₘ",
    "n": "ₙ",
    "o": "ₒ",
    "p": "ₚ",
    "r": "ᵣ",
    "s": "ₛ",
    "t": "ₜ",
    "u": "ᵤ",
    "v": "ᵥ",
    "x": "ₓ",
    "alpha": "ₐ",
    "beta": "ᵦ",
    "epsilon": "ₑ",
    "varepsilon": "ₑ",
    "gamma": "ᵧ",
    "rho": "ᵨ",
    "phi": "ᵩ",
    "varphi": "ᵩ",
    "chi": "ᵪ",
    "α": "ₐ",
    "β": "ᵦ",
    "γ": "ᵧ",
    "ε": "ₑ",
    "ϵ": "ₑ",
    "ρ": "ᵨ",
    "φ": "ᵩ",
    "χ": "ᵪ",
}

_SUPERSCRIPT_MAP = {
    "0": "⁰",
    "1": "¹",
    "2": "²",
    "3": "³",
    "4": "⁴",
    "5": "⁵",
    "6": "⁶",
    "7": "⁷",
    "8": "⁸",
    "9": "⁹",
    "+": "⁺",
    "-": "⁻",
    "=": "⁼",
    "(": "⁽",
    ")": "⁾",
    "a": "ᵃ",
    "b": "ᵇ",
    "c": "ᶜ",
    "d": "ᵈ",
    "e": "ᵉ",
    "f": "ᶠ",
    "g": "ᵍ",
    "h": "ʰ",
    "i": "ⁱ",
    "j": "ʲ",
    "k": "ᵏ",
    "l": "ˡ",
    "m": "ᵐ",
    "n": "ⁿ",
    "o": "ᵒ",
    "p": "ᵖ",
    "r": "ʳ",
    "s": "ˢ",
    "t": "ᵗ",
    "u": "ᵘ",
    "v": "ᵛ",
    "w": "ʷ",
    "x": "ˣ",
    "y": "ʸ",
    "z": "ᶻ",
    "alpha": "ᵅ",
    "beta": "ᵝ",
    "gamma": "ᵞ",
    "delta": "ᵟ",
    "epsilon": "ᵋ",
    "varepsilon": "ᵋ",
    "theta": "ᶿ",
    "phi": "ᵠ",
    "varphi": "ᵠ",
    "chi": "ᵡ",
    "α": "ᵅ",
    "β": "ᵝ",
    "γ": "ᵞ",
    "δ": "ᵟ",
    "ε": "ᵋ",
    "ϵ": "ᵋ",
    "θ": "ᶿ",
    "φ": "ᵠ",
    "χ": "ᵡ",
}

_MATH_ITALIC_MAP = {
    "a": "𝑎",
    "b": "𝑏",
    "c": "𝑐",
    "d": "𝑑",
    "e": "𝑒",
    "f": "𝑓",
    "g": "𝑔",
    "h": "ℎ",
    "i": "𝑖",
    "j": "𝑗",
    "k": "𝑘",
    "l": "𝑙",
    "m": "𝑚",
    "n": "𝑛",
    "o": "𝑜",
    "p": "𝑝",
    "q": "𝑞",
    "r": "𝑟",
    "s": "𝑠",
    "t": "𝑡",
    "u": "𝑢",
    "v": "𝑣",
    "w": "𝑤",
    "x": "𝑥",
    "y": "𝑦",
    "z": "𝑧",
    "A": "𝐴",
    "B": "𝐵",
    "C": "𝐶",
    "D": "𝐷",
    "E": "𝐸",
    "F": "𝐹",
    "G": "𝐺",
    "H": "𝐻",
    "I": "𝐼",
    "J": "𝐽",
    "K": "𝐾",
    "L": "𝐿",
    "M": "𝑀",
    "N": "𝑁",
    "O": "𝑂",
    "P": "𝑃",
    "Q": "𝑄",
    "R": "𝑅",
    "S": "𝑆",
    "T": "𝑇",
    "U": "𝑈",
    "V": "𝑉",
    "W": "𝑊",
    "X": "𝑋",
    "Y": "𝑌",
    "Z": "𝑍"
}


def _latex_to_terminal(text: str) -> str:
    rendered = (text or "").strip()
    if not rendered:
        return ""

    rendered = rendered.replace("\\left", "").replace("\\right", "")
    rendered = rendered.replace("\\|", "||")
    rendered = rendered.replace("\\,", " ")
    rendered = rendered.replace("\\;", " ")
    rendered = rendered.replace("\\:", " ")
    rendered = rendered.replace("\\!", "")
    rendered = rendered.replace("\\\\", "\n")
    rendered = rendered.replace("\\{", "{").replace("\\}", "}")

    rendered = _replace_nested_macro(rendered, "frac", lambda a, b: f"({a})/({b})")
    rendered = _replace_single_brace_macro(rendered, "sqrt", lambda inner: f"√({inner})")
    rendered = _apply_text_latex_macros(rendered)
    rendered = _apply_negated_latex_symbols(rendered)

    rendered = re.sub(
        r"\\mathbb\{([A-Z])\}",
        lambda m: _MATHBB_MAP.get(m.group(1), m.group(1)),
        rendered,
    )

    for name, symbol in _SYMBOL_RENDER.items():
        rendered = re.sub(rf"\\{name}(?=[^A-Za-z]|$)", symbol, rendered)

    rendered = _apply_script_markup(rendered)
    rendered = rendered.replace("{", "").replace("}", "")
    rendered = re.sub(r"\\([A-Za-z]+)", r"\1", rendered)
    rendered = re.sub(r"[ \t]+", " ", rendered)
    rendered = re.sub(r" *\n *", "\n", rendered)
    return _space_relation_boundaries(_apply_bare_math_until_stable(_italicize_math_variables(rendered.strip())))


def _rewrite_inline_latex(text: str, *, mode: str) -> str:
    source = text or ""
    if "$" not in source and "\\(" not in source and "\\[" not in source:
        return source

    pieces: list[str] = []
    cursor = 0
    converted_segments = 0
    length = len(source)

    while cursor < length:
        if source.startswith("\\[", cursor):
            close_at = _find_verbatim_delimiter(source, cursor + 2, "\\]")
            if close_at >= 0:
                raw = source[cursor + 2 : close_at]
                replacement = _replace_math_text(raw, mode=mode, display=True)
                if replacement != f"\\[{raw}\\]":
                    converted_segments += 1
                pieces.append(_space_cli_math_replacement(source, cursor, close_at + 2, replacement, mode=mode))
                cursor = close_at + 2
                continue
        if source.startswith("\\(", cursor):
            close_at = _find_verbatim_delimiter(source, cursor + 2, "\\)")
            if close_at >= 0:
                raw = source[cursor + 2 : close_at]
                replacement = _replace_math_text(raw, mode=mode, display=False)
                if replacement != f"\\({raw}\\)":
                    converted_segments += 1
                pieces.append(_space_cli_math_replacement(source, cursor, close_at + 2, replacement, mode=mode))
                cursor = close_at + 2
                continue
        if source.startswith("$$", cursor):
            close_at = _find_math_closing_delimiter(source, cursor + 2, "$$")
            if close_at >= 0:
                raw = source[cursor + 2 : close_at]
                replacement = _replace_math_text(raw, mode=mode, display=True)
                if replacement != f"$${raw}$$":
                    converted_segments += 1
                pieces.append(_space_cli_math_replacement(source, cursor, close_at + 2, replacement, mode=mode))
                cursor = close_at + 2
                continue
        if source[cursor] == "$":
            close_at = _find_math_closing_delimiter(source, cursor + 1, "$")
            if close_at >= 0:
                raw = source[cursor + 1 : close_at]
                replacement = _replace_math_text(raw, mode=mode, display=False)
                if replacement != f"${raw}$":
                    converted_segments += 1
                pieces.append(_space_cli_math_replacement(source, cursor, close_at + 1, replacement, mode=mode))
                cursor = close_at + 1
                continue
        pieces.append(source[cursor])
        cursor += 1

    rendered = "".join(pieces)
    if mode == "cli" and "$" in rendered and (converted_segments > 0 or _looks_like_inline_latex_segment(source)):
        rendered = re.sub(r"(?<!\\)\$", "", rendered)
    return rendered


def _replace_math_segment(match: re.Match[str], *, mode: str, display: bool) -> str:
    raw = match.group(1)
    return _replace_math_text(raw, mode=mode, display=display)


def _replace_math_text(raw: str, *, mode: str, display: bool) -> str:
    if not _looks_like_inline_latex_segment(raw):
        return f"$${raw}$$" if display else f"${raw}$"
    core, inferred_display = _unwrap_latex(raw)
    display = display or inferred_display
    if mode == "cli":
        return _latex_to_terminal(core)
    if mode == "anki":
        return f"\\[{core}\\]" if display else f"\\({core}\\)"
    return f"$$\n{core}\n$$" if display else f"${core}$"


def _find_math_closing_delimiter(text: str, start: int, delimiter: str) -> int:
    cursor = start
    width = len(delimiter)
    while cursor < len(text):
        if text[cursor] == "\\":
            cursor += 2
            continue
        if text.startswith(delimiter, cursor):
            return cursor
        cursor += 1
    return -1


def _find_verbatim_delimiter(text: str, start: int, delimiter: str) -> int:
    """Find a verbatim closing delimiter such as ``\\)`` or ``\\]``."""
    return text.find(delimiter, start)


def _looks_like_inline_latex_segment(text: str) -> bool:
    candidate = (text or "").strip()
    if not candidate:
        return False
    if any(token in candidate for token in ("\\", "{", "}", "(", ")", ",", "/", "+", "-", "=", "*", "^", "_")):
        return True
    if " " not in candidate:
        if len(candidate) <= 8:
            return True
        return any(char.isdigit() for char in candidate)
    return False


def _space_cli_math_replacement(source: str, start: int, end: int, replacement: str, *, mode: str) -> str:
    """Separate rendered inline math from adjacent prose when delimiters were tight."""
    if mode != "cli" or not replacement:
        return replacement
    rendered = replacement
    previous = source[start - 1] if start > 0 else ""
    following = source[end] if end < len(source) else ""
    if previous and not previous.isspace() and _is_boundary_word_char(previous):
        rendered = " " + rendered
    if following and not following.isspace() and _is_boundary_word_char(following):
        rendered = rendered + " "
    return rendered


def _replace_nested_macro(text: str, macro: str, formatter) -> str:
    pattern = f"\\{macro}"
    rendered = text
    while pattern in rendered:
        start = rendered.find(pattern)
        if start < 0:
            break
        first = _read_braced_group(rendered, start + len(pattern))
        if first is None:
            break
        second = _read_braced_group(rendered, first[1])
        if second is None:
            break
        replacement = formatter(first[0], second[0])
        rendered = rendered[:start] + replacement + rendered[second[1] :]
    return rendered


def _replace_single_brace_macro(text: str, macro: str, formatter) -> str:
    pattern = f"\\{macro}"
    rendered = text
    while pattern in rendered:
        start = rendered.find(pattern)
        if start < 0:
            break
        group = _read_braced_group(rendered, start + len(pattern))
        if group is None:
            break
        replacement = formatter(group[0])
        rendered = rendered[:start] + replacement + rendered[group[1] :]
    return rendered


def _apply_text_latex_macros(text: str) -> str:
    """Render LaTeX text macros as plain prose instead of leaking command names."""
    rendered = text
    for macro in _TEXT_BRACE_MACROS:
        pattern = f"\\{macro}"
        cursor = 0
        while True:
            start = rendered.find(pattern, cursor)
            if start < 0:
                break
            after_macro = start + len(pattern)
            if after_macro < len(rendered) and rendered[after_macro].isalpha():
                cursor = after_macro
                continue
            group = _read_braced_group(rendered, after_macro)
            if group is None:
                cursor = after_macro
                continue
            replacement = _format_latex_text_macro(group[0])
            rendered = rendered[:start] + replacement + rendered[group[1] :]
            cursor = start + len(replacement)
    return rendered


def _format_latex_text_macro(content: str) -> str:
    plain = re.sub(r"\s+", " ", (content or "").strip())
    return f" {plain} " if plain else " "


def _read_braced_group(text: str, index: int) -> tuple[str, int] | None:
    while index < len(text) and text[index].isspace():
        index += 1
    if index >= len(text) or text[index] != "{":
        return None
    depth = 0
    start = index + 1
    for cursor in range(index, len(text)):
        char = text[cursor]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:cursor], cursor + 1
    return None


def _apply_negated_latex_symbols(text: str) -> str:
    """Render LaTeX's prefix negation forms before plain command replacement."""
    result = text
    for command, symbol in _NEGATED_SYMBOL_RENDER:
        result = re.sub(
            rf"\\not\s*\\{command}(?=[^A-Za-z]|$)",
            symbol,
            result,
        )
    result = re.sub(r"\\not\s*=", "≠", result)
    return result


def _apply_script_markup(text: str) -> str:
    rendered = text
    rendered = re.sub(r"_\{([^{}]+)\}", lambda match: _script_to_unicode(match.group(1), _SUBSCRIPT_MAP, "_"), rendered)
    rendered = re.sub(r"\^\{([^{}]+)\}", lambda match: _script_to_unicode(match.group(1), _SUPERSCRIPT_MAP, "^"), rendered)
    rendered = re.sub(rf"_(?!\{{)({_SCRIPT_TOKEN_MULTI_PATTERN})", lambda match: _script_to_unicode(match.group(1), _SUBSCRIPT_MAP, "_"), rendered)
    rendered = re.sub(rf"\^(?!\{{)({_SCRIPT_TOKEN_MULTI_PATTERN})", lambda match: _script_to_unicode(match.group(1), _SUPERSCRIPT_MAP, "^"), rendered)
    rendered = re.sub(rf"_({_SCRIPT_TOKEN_PATTERN})", lambda match: _script_to_unicode(match.group(1), _SUBSCRIPT_MAP, "_"), rendered)
    rendered = re.sub(rf"\^({_SCRIPT_TOKEN_PATTERN})", lambda match: _script_to_unicode(match.group(1), _SUPERSCRIPT_MAP, "^"), rendered)
    return rendered


_SCRIPT_TOKEN_CHARS = (
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    "+-="
    "αβγδεϵζηθικλμνξοπρσςτυφχψω"
    "ΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ"
)
_SCRIPT_TOKEN_CLASS = rf"[{re.escape(_SCRIPT_TOKEN_CHARS)}]"
_SCRIPT_TOKEN_PATTERN = rf"{_SCRIPT_TOKEN_CLASS}+"
_SCRIPT_TOKEN_MULTI_PATTERN = rf"{_SCRIPT_TOKEN_CLASS}{{2,}}"
_BARE_BASE_CHARS = "".join(
    sorted(
        set(
            "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"
            + "".join(_MATH_ITALIC_MAP.values())
            + "".join(_MATHBB_MAP.values())
            + "".join(_SYMBOL_RENDER.values())
        )
    )
)
_BARE_MATH_RE = re.compile(
    rf"(\\(?:[A-Za-z]+)\b|(?<![A-Za-z0-9])(?:[{re.escape(_BARE_BASE_CHARS)}])(?:\^{{?{_SCRIPT_TOKEN_PATTERN}}}?|_{{?{_SCRIPT_TOKEN_PATTERN}}}?))"
)
_SCRIPT_SUFFIX_CHARS = "".join(
    sorted(set(_SUBSCRIPT_MAP.values()) | set(_SUPERSCRIPT_MAP.values()))
)
_SCRIPTED_VARIABLE_RE = re.compile(rf"(?<![A-Za-z])([A-Za-z])(?=[{re.escape(_SCRIPT_SUFFIX_CHARS)}])")
_RELATION_BOUNDARY_SYMBOLS = frozenset(
    "⊂⊆⊄⊈⊃⊇⊅⊉∈∉≤≥≰≱=≠≈≅<>"
)
_BOUNDARY_WORD_CHARS = frozenset(
    "".join(_MATH_ITALIC_MAP.values())
    + "".join(_MATHBB_MAP.values())
    + "".join(_SYMBOL_RENDER.values())
    + "".join(_SUBSCRIPT_MAP.values())
    + "".join(_SUPERSCRIPT_MAP.values())
)


def _contains_bare_math_notation(text: str) -> bool:
    """Return True when plain text likely contains terminal-friendly math notation."""
    return bool(_BARE_MATH_RE.search(text or ""))


def _apply_bare_math_notation(text: str) -> str:
    """Apply script markup to text without dollar-sign delimiters.

    Only triggers when ^ or _ patterns match standard math notation
    (e.g. R^3, x_n) to avoid false positives on non-math text.
    Handles \\mathbb in bare text as well.
    """
    result = text
    result = result.replace("\\left", "").replace("\\right", "")
    result = result.replace("\\,", " ")
    result = result.replace("\\;", " ")
    result = result.replace("\\:", " ")
    result = result.replace("\\!", "")
    result = result.replace("\\{", "{").replace("\\}", "}")
    result = _replace_nested_macro(
        result,
        "frac",
        lambda numerator, denominator: f"({_latex_to_terminal(numerator)})/({_latex_to_terminal(denominator)})",
    )
    result = _replace_single_brace_macro(
        result,
        "sqrt",
        lambda inner: f"√({_latex_to_terminal(inner)})",
    )
    result = _apply_text_latex_macros(result)
    result = _apply_negated_latex_symbols(result)
    # Handle \mathbb in bare text
    result = re.sub(
        r"\\mathbb\{([A-Z])\}",
        lambda m: _MATHBB_MAP.get(m.group(1), m.group(1)),
        result,
    )
    # Handle bare blackboard-bold shorthand such as R^3 -> ℝ³.
    result = re.sub(
        r"(?<![A-Za-z0-9\\])([RNCQZ])(?=(?:\^\{?[A-Za-z0-9+\-=]+\}?|_\{?[A-Za-z0-9+\-=]+\}?))",
        lambda m: _MATHBB_MAP.get(m.group(1), m.group(1)),
        result,
    )
    # Handle \symbol patterns (nabla, alpha, etc.)
    for name, symbol in _SYMBOL_RENDER.items():
        result = re.sub(rf"\\{name}(?=[^A-Za-z]|$)", symbol, result)
    # Apply superscript/subscript
    result = _apply_script_markup(result)
    result = _SCRIPTED_VARIABLE_RE.sub(lambda match: _MATH_ITALIC_MAP.get(match.group(1), match.group(1)), result)
    result = result.replace("{", "").replace("}", "")
    result = re.sub(r"\\([A-Za-z]+)", r"\1", result)
    result = re.sub(r"[ \t]+", " ", result)
    return _space_relation_boundaries(result)


def _apply_bare_math_until_stable(text: str) -> str:
    """Apply bare math cleanup until nested/generated script notation settles."""
    result = text or ""
    applied = False
    for _ in range(4):
        if not _contains_bare_math_notation(result):
            break
        next_result = _apply_bare_math_notation(result)
        applied = True
        if next_result == result:
            break
        result = next_result
    return _space_relation_boundaries(result) if applied else result


def _space_relation_boundaries(text: str) -> str:
    """Ensure rendered relation symbols do not fuse with adjacent text."""
    if not text:
        return ""
    pieces: list[str] = []
    length = len(text)
    for index, char in enumerate(text):
        previous_output = pieces[-1] if pieces else ""
        if (
            char in _RELATION_BOUNDARY_SYMBOLS
            and previous_output
            and not previous_output.isspace()
            and _is_boundary_word_char(previous_output)
        ):
            pieces.append(" ")
        pieces.append(char)
        following = text[index + 1] if index + 1 < length else ""
        if (
            char in _RELATION_BOUNDARY_SYMBOLS
            and following
            and not following.isspace()
            and _is_boundary_word_char(following)
        ):
            pieces.append(" ")
    return "".join(pieces)


def _is_boundary_word_char(char: str) -> bool:
    return bool(char and (char.isalnum() or char in _BOUNDARY_WORD_CHARS))


def _script_to_unicode(content: str, alphabet: dict[str, str], fallback_prefix: str) -> str:
    lowered = content.strip()
    if not lowered:
        return ""
    mapped: list[str] = []
    for char in lowered:
        substitute = alphabet.get(char) or alphabet.get(char.lower())
        if substitute is None:
            return f"{fallback_prefix}({lowered})"
        mapped.append(substitute)
    return "".join(mapped)


def _italicize_math_variables(text: str) -> str:
    return re.sub(
        r"(?<![A-Za-z])([A-Za-z])(?![A-Za-z])",
        lambda match: _MATH_ITALIC_MAP.get(match.group(1), match.group(1)),
        text,
    )
