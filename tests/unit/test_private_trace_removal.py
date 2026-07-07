"""Verify that no private test-mode or ADC-gating traces remain in the repo."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIRS = [REPO_ROOT / "pb"]
PRIVATE_PROJECT_ID = "productivity-494409"


def _source_files():
    for root in SRC_DIRS:
        for path in root.rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            yield path


def test_no_private_project_id_in_source():
    """No source file may contain the private GCP project ID."""
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if PRIVATE_PROJECT_ID in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"Private project ID found in: {offenders}"


def test_no_adc_gating_logic_in_source():
    """No source file may gate features on a specific GOOGLE_CLOUD_PROJECT value."""
    offenders = []
    for path in _source_files():
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "is_owner_machine" in text or "_OWNER_PROJECT" in text:
            offenders.append(str(path.relative_to(REPO_ROOT)))
    assert offenders == [], f"ADC-gating logic found in: {offenders}"


def test_no_test_mode_module():
    """The private test_mode module must not exist."""
    assert not (REPO_ROOT / "pb" / "llm" / "test_mode.py").exists()


def test_help_does_not_mention_test_flag():
    """pb --help must not show --test."""
    result = subprocess.run(
        [sys.executable, "-m", "pb", "--help"],
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        timeout=30,
    )
    assert "--test" not in result.stdout, f"--test found in help output:\n{result.stdout}"
