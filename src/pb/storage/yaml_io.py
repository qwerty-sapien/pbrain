# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared YAML helpers for repo-owned state and structured LLM payloads.

These helpers favor YAML for local persistence while retaining JSON fallback
for legacy files and external payloads that may still contain JSON text.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def dump_yaml(data: Any, *, flow: bool = False) -> str:
    """Serialize data to YAML with stable, human-readable defaults."""
    return yaml.safe_dump(
        data,
        allow_unicode=True,
        default_flow_style=flow,
        sort_keys=False,
    )


def dump_compact_yaml(data: Any) -> str:
    """Serialize data to single-line flow-style YAML."""
    return dump_yaml(data, flow=True).strip()


def load_yaml_text(text: str, default: Any) -> Any:
    """Parse YAML text, returning default on empty or invalid input.

    JSON is a YAML subset, so JSON text also parses here. A final JSON
    fallback is kept for legacy robustness when YAML parsing fails.
    """
    stripped = (text or "").strip()
    if not stripped:
        return default
    try:
        loaded = yaml.safe_load(stripped)
        return default if loaded is None else loaded
    except yaml.YAMLError:
        try:
            loaded = json.loads(stripped)
            return default if loaded is None else loaded
        except Exception:
            return default


def load_yaml_file(path: Path, default: Any) -> Any:
    """Parse YAML from disk, returning default on missing or invalid files."""
    try:
        if not path.exists():
            return default
        return load_yaml_text(path.read_text(encoding="utf-8"), default)
    except OSError:
        return default


def load_yaml_with_legacy_json(path: Path, legacy_path: Path, default: Any) -> Any:
    """Load YAML from `path`, falling back to a legacy JSON file."""
    if path.exists():
        return load_yaml_file(path, default)
    return load_yaml_file(legacy_path, default)


def write_yaml_file(path: Path, data: Any, *, flow: bool = False) -> None:
    """Write YAML to disk, creating parent directories when needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dump_yaml(data, flow=flow), encoding="utf-8")


def extract_structured_yaml(text: str, default: Any) -> Any:
    """Best-effort parse of YAML/JSON embedded in plain text or code fences."""
    stripped = (text or "").strip()
    if not stripped:
        return default
    if stripped.startswith("```"):
        lines = [line for line in stripped.splitlines() if not line.strip().startswith("```")]
        stripped = "\n".join(lines).strip()
    return load_yaml_text(stripped, default)
