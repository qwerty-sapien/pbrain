# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""StudyService — time-boxed interleaved study plan generation.

INV-4: this module never imports rich or typer.

Extracts DomainStatus, StudyBlock, allocate_blocks, get_domain_statuses,
_resolve_threshold from cli/commands/study.py (D-03).
"""
from __future__ import annotations

import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import structlog

from pb.core.base import BaseService, LoggableMixin


@dataclass
class DomainStatus:
    """Status of a knowledge domain for study planning."""
    name: str
    path: Path
    stage_new: int = 0
    stage_learning: int = 0
    stage_learnt: int = 0
    stage_stale: int = 0
    decay_pressure: float = 0.0  # days past threshold; 0 = not stale
    is_stale: bool = False
    top_notes: list[str] = field(default_factory=list)  # slugs for pb commands


@dataclass
class StudyBlock:
    """A single time block in the study plan."""
    domain: str
    minutes: int
    mode: str           # "re-engage" | "consolidate" | "explore" | "review"
    stage: str          # "#stale" | "#learning" | "#new" | "#learnt"
    pb_command: str     # The exact pb command the user should run
    focus_note: str = ""  # Optional: specific note to focus on


def _resolve_threshold(domain_name: str, thresholds: dict[str, int]) -> int:
    """Resolve decay threshold for a domain. Exact match -> prefix match -> _default.

    Examples: "deutsch" -> fallback to _default (no match for "deutsch").
              "piano" matches "piano" exactly.
              "ml-notes" matches "ml" via prefix.
    """
    lower = domain_name.lower()
    # Exact match
    if lower in thresholds:
        return thresholds[lower]
    # Prefix match (e.g., "ml-notes" matches "ml")
    for key, val in thresholds.items():
        if key != "_default" and lower.startswith(key):
            return val
    return thresholds.get("_default", 5)


def _get_pb_command(mode: str, domain: str, focus_note: str = "") -> str:
    """Generate the specific pb command for a study block.

    Per STDY-04: Only active engagement. Never passive reading.
    Commands returned here must be valid on the current CLI surface.
    """
    if mode == "review":
        return f"pb study recall {domain}"
    if mode in {"re-engage", "consolidate", "explore"}:
        return f"pb study {domain}"
    return f"pb study {domain}"


def get_domain_statuses(vault_path: Path, thresholds: dict[str, int]) -> list[DomainStatus]:
    """Scan knowledge domains and compute per-domain status.

    T-19-12 mitigation: Only iterates immediate children of knowledge/
    (no recursive descent). Skips non-dirs and dotfiles.
    """
    from pb.vault.lifecycle import read_frontmatter, check_staleness

    knowledge_dir = vault_path / "knowledge"
    if not knowledge_dir.exists():
        return []

    statuses = []
    # T-19-12: iterate only immediate children (no recursive descent)
    for domain_dir in sorted(knowledge_dir.iterdir()):
        if not domain_dir.is_dir() or domain_dir.name.startswith("."):
            continue
        if not (domain_dir / "_state.md").exists():
            continue

        status = DomainStatus(name=domain_dir.name, path=domain_dir)
        threshold = _resolve_threshold(domain_dir.name, thresholds)

        # Count notes by stage (immediate children only, not recursive)
        for md_file in domain_dir.glob("*.md"):
            if md_file.name.startswith("_"):
                continue
            try:
                content = md_file.read_text()
                fm, _ = read_frontmatter(content)
                stage = fm.get("learning_stage", "#new")
                if stage == "#new":
                    status.stage_new += 1
                elif stage == "#learning":
                    status.stage_learning += 1
                elif stage == "#learnt":
                    status.stage_learnt += 1
                elif stage == "#stale":
                    status.stage_stale += 1
                    status.top_notes.append(md_file.stem)
                elif stage == "#archive":
                    continue  # never surfaced
            except Exception:
                continue

        # Check staleness via lifecycle
        try:
            stale_msgs = check_staleness(vault_path, domain_dir, decay_days=threshold)
            if stale_msgs or status.stage_stale > 0:
                status.is_stale = True
                # Compute decay pressure: how many days past threshold
                state_file = domain_dir / "_state.md"
                try:
                    sfm, _ = read_frontmatter(state_file.read_text())
                    last_activity = sfm.get("last_activity", sfm.get("last_session"))
                    if last_activity:
                        last_date = datetime.date.fromisoformat(str(last_activity)[:10])
                        days_inactive = (datetime.date.today() - last_date).days
                        status.decay_pressure = max(0, days_inactive - threshold)
                except Exception:
                    status.decay_pressure = float(threshold)  # assume max stale
        except Exception:
            pass

        statuses.append(status)

    return statuses


def allocate_blocks(
    total_minutes: int,
    domains: list[DomainStatus],
    domain_filter: Optional[str] = None,
) -> list[StudyBlock]:
    """Allocate study time blocks across domains.

    Per D-12: Stale domains first, then 40% #learning / 35% #new / 25% #learnt.
    Per STDY-04: Only active engagement scheduled.
    """
    if domain_filter:
        domains = [d for d in domains if d.name.lower() == domain_filter.lower()]

    if not domains:
        return []

    blocks: list[StudyBlock] = []
    remaining = total_minutes

    # 1. Stale domains first (sorted by decay pressure descending)
    stale_domains = sorted(
        [d for d in domains if d.is_stale],
        key=lambda d: d.decay_pressure,
        reverse=True,
    )
    for domain in stale_domains:
        if remaining <= 0:
            break
        alloc = min(remaining, 15)  # 15-min stale refresh blocks
        blocks.append(StudyBlock(
            domain=domain.name,
            minutes=alloc,
            mode="re-engage",
            stage="#stale",
            pb_command=_get_pb_command("re-engage", domain.name),
        ))
        remaining -= alloc

    if remaining <= 0:
        return blocks

    # 2. Remaining: 40/35/25 across #learning/#new/#learnt
    learning_budget = int(remaining * 0.40)
    new_budget = int(remaining * 0.35)
    learnt_budget = remaining - learning_budget - new_budget  # gets rounding remainder

    # Collect domains with notes in each stage
    learning_domains = sorted(
        [d for d in domains if d.stage_learning > 0],
        key=lambda d: d.decay_pressure,
        reverse=True,
    )
    new_domains = sorted(
        [d for d in domains if d.stage_new > 0],
        key=lambda d: d.stage_new,
        reverse=True,
    )
    learnt_domains = sorted(
        [d for d in domains if d.stage_learnt > 0],
        key=lambda d: d.decay_pressure,
        reverse=True,
    )

    # Allocate learning blocks (consolidate)
    if learning_domains and learning_budget > 0:
        per_domain = max(10, learning_budget // len(learning_domains))
        for domain in learning_domains:
            if learning_budget <= 0:
                break
            alloc = min(learning_budget, per_domain)
            blocks.append(StudyBlock(
                domain=domain.name,
                minutes=alloc,
                mode="consolidate",
                stage="#learning",
                pb_command=_get_pb_command("consolidate", domain.name),
            ))
            learning_budget -= alloc

    # Allocate new blocks (explore)
    if new_domains and new_budget > 0:
        per_domain = max(10, new_budget // len(new_domains))
        for domain in new_domains:
            if new_budget <= 0:
                break
            alloc = min(new_budget, per_domain)
            blocks.append(StudyBlock(
                domain=domain.name,
                minutes=alloc,
                mode="explore",
                stage="#new",
                pb_command=_get_pb_command("explore", domain.name),
            ))
            new_budget -= alloc

    # Allocate learnt blocks (review via Anki)
    if learnt_domains and learnt_budget > 0:
        per_domain = max(10, learnt_budget // len(learnt_domains))
        for domain in learnt_domains:
            if learnt_budget <= 0:
                break
            alloc = min(learnt_budget, per_domain)
            blocks.append(StudyBlock(
                domain=domain.name,
                minutes=alloc,
                mode="review",
                stage="#learnt",
                pb_command=_get_pb_command("review", domain.name),
            ))
            learnt_budget -= alloc

    return blocks


class StudyService(BaseService, LoggableMixin):
    """Service-layer orchestrator for time-boxed study plan generation.

    INV-4: never imports rich or typer.
    Extracts DomainStatus, StudyBlock, allocate_blocks, get_domain_statuses,
    _resolve_threshold from cli/commands/study.py (D-03).
    """

    def __init__(self, vault_path: Path, config: Any) -> None:
        super().__init__()
        self.vault_path = vault_path
        self._thresholds = config.learning.decay_thresholds
        self._log = structlog.get_logger()

    def generate_plan(
        self,
        total_minutes: int = 60,
        domain_filter: Optional[str] = None,
    ) -> list[StudyBlock]:
        """Generate interleaved study plan. Returns list[StudyBlock]. No Rich/typer."""
        statuses = get_domain_statuses(self.vault_path, self._thresholds)
        return allocate_blocks(total_minutes, statuses, domain_filter)

    def get_domain_statuses(self) -> list[DomainStatus]:
        """Return list[DomainStatus] for all knowledge domains."""
        return get_domain_statuses(self.vault_path, self._thresholds)
