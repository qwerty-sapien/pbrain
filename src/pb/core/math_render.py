# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Terminal-safe LaTeX → Unicode best-effort renderer.

Handles common inline patterns that appear in math/science lesson content.
Unknown commands are left unchanged. Never raises — always returns a string.
"""
from __future__ import annotations

import re

_GREEK: dict[str, str] = {
    "alpha": "α", "beta": "β", "gamma": "γ", "delta": "δ",
    "epsilon": "ε", "varepsilon": "ε", "zeta": "ζ", "eta": "η",
    "theta": "θ", "vartheta": "θ", "iota": "ι", "kappa": "κ",
    "lambda": "λ", "mu": "μ", "nu": "ν", "xi": "ξ", "pi": "π",
    "varpi": "π", "rho": "ρ", "varrho": "ρ", "sigma": "σ",
    "varsigma": "ς", "tau": "τ", "upsilon": "υ", "phi": "φ",
    "varphi": "φ", "chi": "χ", "psi": "ψ", "omega": "ω",
    "Alpha": "Α", "Beta": "Β", "Gamma": "Γ", "Delta": "Δ",
    "Epsilon": "Ε", "Zeta": "Ζ", "Eta": "Η", "Theta": "Θ",
    "Lambda": "Λ", "Mu": "Μ", "Nu": "Ν", "Xi": "Ξ", "Pi": "Π",
    "Sigma": "Σ", "Tau": "Τ", "Phi": "Φ", "Chi": "Χ",
    "Psi": "Ψ", "Omega": "Ω",
}

_SYMBOLS: dict[str, str] = {
    "infty": "∞",
    "cdot": "·", "times": "×", "div": "÷",
    "leq": "≤", "le": "≤", "geq": "≥", "ge": "≥",
    "neq": "≠", "ne": "≠", "approx": "≈", "cong": "≅", "sim": "~",
    "pm": "±", "mp": "∓",
    "to": "→", "rightarrow": "→", "leftarrow": "←",
    "Rightarrow": "⟹", "Leftarrow": "⟸", "Leftrightarrow": "⟺",
    "mapsto": "↦",
    "partial": "∂", "nabla": "∇",
    "in": "∈", "notin": "∉",
    "subset": "⊂", "subseteq": "⊆", "supset": "⊃", "supseteq": "⊇",
    "cup": "∪", "cap": "∩",
    "forall": "∀", "exists": "∃", "nexists": "∄",
    "ldots": "…", "cdots": "⋯",
    "therefore": "∴", "because": "∵",
    "ell": "ℓ", "hbar": "ℏ",
    "oplus": "⊕", "otimes": "⊗",
    "emptyset": "∅", "varnothing": "∅",
    "langle": "⟨", "rangle": "⟩",
    "lfloor": "⌊", "rfloor": "⌋", "lceil": "⌈", "rceil": "⌉",
    "perp": "⊥", "parallel": "∥",
}

_BIG_OPS: dict[str, str] = {
    "sum": "Σ", "prod": "Π", "int": "∫", "oint": "∮",
    "bigcup": "⋃", "bigcap": "⋂",
}

# Commands that take a following delimiter but contribute no rendered glyph
_SKIP_CMDS = frozenset({
    "left", "right", "bigl", "bigr", "big", "Big", "bigg", "Bigg",
})
# Commands whose brace argument should be rendered verbatim (no markup)
_TEXT_CMDS = frozenset({
    "text", "mathrm", "mathit", "mathbf", "mathsf",
    "textbf", "textit", "mbox", "operatorname",
})

_DIGIT_SUP = str.maketrans("0123456789+-n", "⁰¹²³⁴⁵⁶⁷⁸⁹⁺⁻ⁿ")
_DIGIT_SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
_LETTER_SUB = str.maketrans("aeiouv", "ₐₑᵢₒᵤᵥ")


def _sym(cmd: str) -> str:
    return _GREEK.get(cmd) or _SYMBOLS.get(cmd) or f"\\{cmd}"


def _to_sup(s: str) -> str:
    clean = s.strip()
    if not clean:
        return ""
    # Pure digits / signs / common super letters → unicode
    if all(c in "0123456789+-n" for c in clean) and len(clean) <= 4:
        return clean.translate(_DIGIT_SUP)
    # Single Greek or symbol
    resolved = _GREEK.get(clean) or _SYMBOLS.get(clean)
    if resolved:
        return resolved
    # Try inner render for complex content
    inner = render_math(clean)
    return f"^({inner})" if len(inner) > 1 else f"^{inner}"


def _to_sub(s: str) -> str:
    clean = s.strip()
    if not clean:
        return ""
    if all(c in "0123456789" for c in clean) and len(clean) <= 3:
        return clean.translate(_DIGIT_SUB)
    if len(clean) == 1 and clean in "aeiouv":
        return clean.translate(_LETTER_SUB)
    inner = render_math(clean)
    return f"({inner})" if len(inner) > 1 else inner


# ---------------------------------------------------------------------------
# Compiled regex passes (applied in order — most specific first)
# ---------------------------------------------------------------------------

# \frac{num}{den}  (single-level braces; no nesting)
_FRAC_RE = re.compile(r"\\frac\{([^{}]*)\}\{([^{}]*)\}")

# \sqrt[n]{x} or \sqrt{x}
_SQRT_DEGREE_RE = re.compile(r"\\sqrt\[([^\]]*)\]\{([^{}]*)\}")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]*)\}")

# Big operators: handle up to two limit groups in either order
# Groups:  1=op  2=lower_a  3=upper_a  (_{a}^{b})
#          1=op  4=upper_b  5=lower_b  (^{b}_{a})
#          1=op  6=lower_c  7=bare_upper  (_{a}^\word)
#          1=op  8=lower_d  (_{a} only)
_BIGOP_RE = re.compile(
    r"\\(sum|prod|int|oint|bigcup|bigcap)"
    r"(?:"
    r"\s*_\{([^{}]*)\}\s*\^\{([^{}]*)\}"    # _{a}^{b}
    r"|\s*\^\{([^{}]*)\}\s*_\{([^{}]*)\}"   # ^{b}_{a}
    r"|\s*_\{([^{}]*)\}\s*\^\\([a-zA-Z]+)"  # _{a}^\cmd
    r"|\s*_\{([^{}]*)\}"                     # _{a} only
    r")?"
)

# Subscripts / superscripts with braces
_SUB_BRACE_RE = re.compile(r"_\{([^{}]*)\}")
_SUP_BRACE_RE = re.compile(r"\^\{([^{}]*)\}")
# Bare single-char subscripts/superscripts
_SUB_CHAR_RE = re.compile(r"_([a-zA-Z0-9])")
_SUP_CHAR_RE = re.compile(r"\^([0-9n+\-])")

# Remaining \command tokens
_CMD_RE = re.compile(r"\\([a-zA-Z]+)")

# Bare grouping braces left over after substitutions
_BARE_BRACE_RE = re.compile(r"\{([^{}]*)\}")


def render_math(text: str) -> str:
    """Convert common inline LaTeX patterns to Unicode approximations.

    Safe to call on any string — returns the input unchanged if no LaTeX found.
    """
    if not text:
        return text
    if "\\" not in text and "_" not in text and "^" not in text:
        return text

    out = text

    # 1. Fractions
    out = _FRAC_RE.sub(
        lambda m: f"({render_math(m.group(1))}/{render_math(m.group(2))})",
        out,
    )

    # 2. Square roots
    out = _SQRT_DEGREE_RE.sub(
        lambda m: f"{_to_sup(m.group(1))}√({render_math(m.group(2))})",
        out,
    )
    out = _SQRT_RE.sub(
        lambda m: f"√({render_math(m.group(1))})",
        out,
    )

    # 3. Big operators with optional limits
    def _bigop(m: re.Match) -> str:
        sym = _BIG_OPS.get(m.group(1), m.group(1))
        # Determine lower / upper from whichever alternation matched
        if m.group(2) is not None:
            lower, upper = render_math(m.group(2)), render_math(m.group(3))
        elif m.group(4) is not None:
            lower, upper = render_math(m.group(5)), render_math(m.group(4))
        elif m.group(6) is not None:
            lower = render_math(m.group(6))
            upper = _sym(m.group(7))
        elif m.group(8) is not None:
            lower, upper = render_math(m.group(8)), ""
        else:
            return sym
        if lower and upper:
            return f"{sym}({lower} to {upper})"
        if lower:
            return f"{sym}({lower} …)"
        return sym

    out = _BIGOP_RE.sub(_bigop, out)

    # 4. Subscripts and superscripts (braced, then bare char)
    out = _SUB_BRACE_RE.sub(lambda m: _to_sub(m.group(1)), out)
    out = _SUP_BRACE_RE.sub(lambda m: _to_sup(m.group(1)), out)
    out = _SUB_CHAR_RE.sub(lambda m: _to_sub(m.group(1)), out)
    out = _SUP_CHAR_RE.sub(lambda m: _to_sup(m.group(1)), out)

    # 5. Named commands
    def _cmd(m: re.Match) -> str:
        cmd = m.group(1)
        if cmd in _SKIP_CMDS:
            return ""
        if cmd in _TEXT_CMDS:
            return ""  # brace arg handled by _BARE_BRACE_RE below
        return _sym(cmd)

    out = _CMD_RE.sub(_cmd, out)

    # 6. Strip residual bare braces (grouping artifacts)
    out = _BARE_BRACE_RE.sub(lambda m: m.group(1), out)

    return out


# ---------------------------------------------------------------------------
# Block / inline split
# ---------------------------------------------------------------------------

# Display math: $$...$$ or \[...\]
_BLOCK_RE = re.compile(r'\$\$(.*?)\$\$|\\\[(.*?)\\\]', re.DOTALL)

# Inline math: $...$ (no nested $, no newlines) or \(...\)
# Negative lookahead/lookbehind avoids matching $$ as two singles.
_INLINE_DELIM_RE = re.compile(r'(?<!\$)\$([^$\n]+?)\$(?!\$)|\\\((.+?)\\\)', re.DOTALL)


def _render_inline_segments(text: str) -> list[tuple[str, bool]]:
    """Strip $...$ / \\(..\\) delimiters and render LaTeX within inline text."""
    chunks: list[tuple[str, bool]] = []
    last = 0
    for m in _INLINE_DELIM_RE.finditer(text):
        before = text[last:m.start()]
        if before:
            chunks.append((render_math(before), False))
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if inner is not None:
            chunks.append((render_math(inner.strip()), False))
        last = m.end()
    tail = text[last:]
    if tail:
        chunks.append((render_math(tail), False))
    return chunks or [(render_math(text), False)]


def extract_and_render(text: str) -> list[tuple[str, bool]]:
    """Split *text* into ``(rendered_content, is_block)`` chunks.

    ``is_block=True`` for display math ($$…$$ or \\[…\\]).
    ``is_block=False`` for normal prose and inline math ($…$ or \\(…\\)).
    Callers can render block chunks centred on their own line for visual weight.
    Never raises — returns ``[(text, False)]`` on empty input.
    """
    if not text:
        return [(text, False)]
    chunks: list[tuple[str, bool]] = []
    last = 0
    for m in _BLOCK_RE.finditer(text):
        before = text[last:m.start()]
        if before:
            chunks.extend(_render_inline_segments(before))
        inner = m.group(1) if m.group(1) is not None else m.group(2)
        if inner is not None:
            chunks.append((render_math(inner.strip()), True))
        last = m.end()
    tail = text[last:]
    if tail:
        chunks.extend(_render_inline_segments(tail))
    return chunks or [(text, False)]
