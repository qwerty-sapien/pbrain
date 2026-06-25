# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Goal reader for vault-based goal notes.

Scans direction/goals/*.md, filters by active status and 3/6-month horizon,
generates compact goal banner for pb plan day output.

Per Phase 3 decisions:
- D-08: Read vault goals on demand (no caching)
- D-09: Filter by status: active + horizon in [3_month, 6_month, quarter]
- D-10: Compact banner at top of pb plan day output

Security:
- T-03-07: Uses yaml.safe_load() to prevent YAML injection
- Malformed YAML returns empty dict (non-fatal per I-09)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import structlog
import yaml

from pb.vault.config import get_vault_path

logger = structlog.get_logger()


def _parse_frontmatter(content: str) -> dict:
    """Extract YAML frontmatter from markdown content.

    Uses yaml.safe_load (never yaml.load) to prevent YAML injection.
    Returns empty dict on missing or malformed frontmatter.
    """
    if not content.startswith("---"):
        return {}
    parts = content.split("---", 2)
    if len(parts) < 3:
        return {}
    try:
        return yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        return {}


class GoalReader:
    """Read active goal notes from vault for daily planning context.

    Scans direction/goals/*.md, filters by status and horizon.
    Per D-08: reads on demand, no caching.
    Per D-09: only 3-month and 6-month horizons for daily planning.
    """

    DAILY_HORIZONS = {
        "3_month", "three_month", "3month", "3-month",
        "6_month", "six_month", "6month", "6-month",
        "quarter",  # quarter maps to ~3 months
    }

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path or get_vault_path()
        self.goals_dir = self.vault_path / "direction" / "goals"

    def read_active_goals(self) -> list[dict]:
        """Return active goals with 3-month or 6-month horizons.

        Returns empty list if goals directory does not exist.
        Skips files with malformed frontmatter (non-fatal).
        """
        if not self.goals_dir.exists():
            return []
        goals = []
        for md_file in self.goals_dir.glob("*.md"):
            try:
                content = md_file.read_text()
                fm = _parse_frontmatter(content)
                if (fm.get("status") == "active"
                        and self._horizon_matches(fm.get("horizon", ""))):
                    goals.append(fm)
            except Exception:
                logger.warning(
                    "goal_reader.skip_file",
                    path=str(md_file),
                )
        return goals

    def _horizon_matches(self, horizon: str) -> bool:
        """Check if horizon string matches daily planning window.

        Normalizes by lowercasing and replacing hyphens/spaces with underscores.
        """
        normalized = str(horizon).lower().replace("-", "_").replace(" ", "_")
        return normalized in self.DAILY_HORIZONS


def generate_goal_banner(goals: list[dict]) -> str:
    """Generate compact goal context banner for pb plan day output (D-10).

    Returns empty string if no goals qualify.
    """
    if not goals:
        return ""
    lines = ["## Active Goals", ""]
    for g in goals:
        horizon_label = g.get("horizon", "")
        title = g.get("title", "Untitled")
        lines.append(f"  [{horizon_label}] {title}")
    lines.append("")
    return "\n".join(lines)
