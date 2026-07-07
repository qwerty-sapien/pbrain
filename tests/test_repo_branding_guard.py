"""Guardrails that prevent legacy two-letter branding from creeping back in."""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCAN_ROOTS = [
    ROOT / "src",
    ROOT / "tests",
    ROOT / "docs",
    ROOT / "README.md",
    ROOT / "pyproject.toml",
    ROOT / ".claude" / "settings.local.json",
    ROOT / "AGENTS.md",
]
THIS_FILE = Path(__file__).resolve()
LEGACY = "p" + "t"
FORBIDDEN_PATTERNS = [
    re.compile(rf"\bfrom\s+{LEGACY}(?=[\s.])"),
    re.compile(rf"\bimport\s+{LEGACY}(?=[\s.])"),
    re.compile(rf"\b{LEGACY}_command\b"),
    re.compile(rf"\b{LEGACY}_query\b"),
    re.compile(rf"\b_get_{LEGACY}_command\b"),
    re.compile(rf"\b{LEGACY.upper()}_SHELL_VAULT_CWD\b"),
    re.compile(rf"\b{LEGACY.upper()}_CONFIG_PATH\b"),
    re.compile(rf"src/{LEGACY}\b"),
    re.compile(rf"\.{LEGACY}-(?:index|graph)"),
    re.compile(rf"shutil\.which\([\"']{LEGACY}[\"']\)"),
]


def _iter_files():
    for root in SCAN_ROOTS:
        if root.is_file():
            yield root
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "__pycache__" in path.parts or path.suffix == ".pyc":
                continue
            if any(part.startswith(".git") for part in path.parts):
                continue
            yield path


def test_repo_has_no_forbidden_legacy_branding_markers():
    offenders: list[str] = []
    for path in _iter_files():
        if path.resolve() == THIS_FILE:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for pattern in FORBIDDEN_PATTERNS:
            if pattern.search(text):
                offenders.append(f"{path.relative_to(ROOT)} -> {pattern.pattern}")
                break

    assert offenders == [], "Forbidden legacy branding markers remain:\n" + "\n".join(offenders)
