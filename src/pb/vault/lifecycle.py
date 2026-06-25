# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Learning lifecycle module for vault notes.

Provides frontmatter read/write, interaction logging, promotion checks,
staleness detection, and archival support for Phase 17 learning lifecycle.
"""

from __future__ import annotations

import datetime
import re
from pathlib import Path
from typing import Optional

import yaml
import structlog

from pb.storage.database import get_connection
from pb.storage.config import get_config

logger = structlog.get_logger()

# Phase 18: vault.db-based INV-3 check (D-15, SOCR-06).
# Imported at module level so tests can patch it via "pb.vault.lifecycle.has_socratic_link_for_note".
# Gracefully set to None when graph_store not yet available (migration not run).
try:
    from pb.vault.graph_store import has_socratic_link_for_note
except Exception:
    has_socratic_link_for_note = None  # type: ignore[assignment]

_LINK_RE = re.compile(r"\[\[([^\]|]+)(?:\|[^\]]+)?\]\]")

# Valid learning stage values
STAGES = ("#new", "#learning", "#learnt", "#stale", "#archive")


# ---------------------------------------------------------------------------
# Frontmatter read/write
# ---------------------------------------------------------------------------


def read_frontmatter(content: str) -> tuple[dict, str]:
    """Parse YAML frontmatter between --- delimiters.

    Returns (frontmatter_dict, body_text). If no frontmatter is found,
    returns ({}, content) unchanged. Uses yaml.safe_load() exclusively
    (T-17-01 mitigation).
    """
    if not content.startswith("---"):
        return {}, content

    # Find closing ---
    end_idx = content.find("\n---", 3)
    if end_idx == -1:
        return {}, content

    yaml_text = content[3:end_idx].strip()
    body = content[end_idx + 4:].lstrip("\n")

    try:
        fm = yaml.safe_load(yaml_text) or {}
        if not isinstance(fm, dict):
            fm = {}
    except yaml.YAMLError:
        fm = {}

    return fm, body


def write_frontmatter(fm: dict, body: str) -> str:
    """Serialize frontmatter dict back into a markdown string.

    Returns: '---\\n{yaml}---\\n\\n{body}'
    """
    yaml_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
    return f"---\n{yaml_text}---\n\n{body}"


# ---------------------------------------------------------------------------
# Interaction logging
# ---------------------------------------------------------------------------


def log_interaction(note_path: str, event_type: str, domain: str = "") -> None:
    """Log a learning interaction for a note (non-fatal, T-17-02 mitigation).

    Inserts a row into the interactions table. Silently swallows all
    exceptions so interaction logging never breaks the calling command.
    """
    try:
        weight = get_config().learning.weights.get(event_type, 1.0)
        ts = datetime.datetime.utcnow().isoformat()
        with get_connection() as conn:
            conn.execute(
                "INSERT INTO interactions (note_path, event_type, weight, ts, domain) "
                "VALUES (?, ?, ?, ?, ?)",
                (note_path, event_type, weight, ts, domain),
            )
            conn.commit()
        logger.debug("lifecycle.log_interaction", note_path=note_path, event_type=event_type, weight=weight)
    except Exception:
        pass  # Non-fatal: interaction logging must never break a command


def get_weighted_total(note_path: str) -> float:
    """Return the sum of interaction weights for a note.

    Returns 0.0 on any error (including missing table or note).
    """
    try:
        with get_connection() as conn:
            row = conn.execute(
                "SELECT SUM(weight) FROM interactions WHERE note_path = ?",
                (note_path,),
            ).fetchone()
            if row and row[0] is not None:
                return float(row[0])
            return 0.0
    except Exception:
        return 0.0


# ---------------------------------------------------------------------------
# Promotion check (#new -> #learning)
# ---------------------------------------------------------------------------


def _check_socratic_link_rglob(content: str, vault_path: Path) -> bool:
    """Fallback INV-3 check using rglob scan (used when vault.db not available).

    Kept as a private migration-period fallback. Once vault.db is seeded,
    check_promotion uses has_socratic_link_for_note instead.
    """
    linked_stems = _LINK_RE.findall(content)
    for stem in linked_stems:
        for candidate in vault_path.rglob(f"{stem}.md"):
            try:
                linked_content = candidate.read_text()
                linked_fm, _ = read_frontmatter(linked_content)
                if linked_fm.get("source") == "socratic":
                    return True
            except Exception:
                continue
    return False


def check_promotion(note_path: str, vault_path: Path) -> Optional[str]:
    """Check if a note qualifies for #new -> #learning promotion (LIFE-03, D-12).

    Returns a Rich-formatted message string if the note was promoted or
    if promotion is blocked by INV-3 (no linked Socratic capture).
    Returns None if the note does not yet meet the interaction threshold,
    or on any error.
    """
    try:
        total = get_weighted_total(note_path)
        threshold = get_config().learning.promotion_threshold

        if total < threshold:
            return None

        # Read note from vault
        note_abs = vault_path / note_path
        if not note_abs.exists():
            return None

        content = note_abs.read_text()
        fm, body = read_frontmatter(content)

        if fm.get("learning_stage") != "#new":
            return None

        # INV-3: require at least one linked Socratic capture note (D-15, SOCR-06)
        # Phase 18: use vault.db graph query when vault.db exists, rglob fallback otherwise.
        vault_db_exists = (vault_path / "vault.db").exists()
        if has_socratic_link_for_note is not None and vault_db_exists:
            try:
                has_socratic = has_socratic_link_for_note(vault_path, note_path)
            except Exception:
                has_socratic = _check_socratic_link_rglob(content, vault_path)
        else:
            # graph_store not yet available (migration not run) — use rglob fallback
            has_socratic = _check_socratic_link_rglob(content, vault_path)

        if not has_socratic:
            return (
                f"[dim]{note_path}[/]: {total:.1f} interactions reached, "
                f"but no linked Socratic capture found. Add one to advance."
            )

        # Promote: update frontmatter
        fm["learning_stage"] = "#learning"
        fm["stage_updated"] = datetime.date.today().isoformat()
        note_abs.write_text(write_frontmatter(fm, body))
        logger.info("lifecycle.promoted", note_path=note_path, total=total)
        return f"[bold]{note_path}[/] promoted [yellow]#new[/] -> [green]#learning[/]"

    except Exception as exc:
        logger.debug("lifecycle.check_promotion_error", note_path=note_path, error=str(exc))
        return None


# ---------------------------------------------------------------------------
# Staleness check (#learnt -> #stale)
# ---------------------------------------------------------------------------


def check_staleness(
    vault_path: Path,
    domain_path: Path,
    decay_days: Optional[int] = None,
) -> list[str]:
    """Flag #learnt notes in domain_path that haven't been updated recently (LIFE-05, D-11).

    Updates their frontmatter to #stale and returns a list of Rich-formatted
    strings for each stale note.
    """
    if decay_days is None:
        decay_days = get_config().learning.decay_days_default

    cutoff = datetime.date.today() - datetime.timedelta(days=decay_days)
    stale_notes: list[str] = []

    try:
        for md_file in domain_path.rglob("*.md"):
            try:
                content = md_file.read_text()
                fm, body = read_frontmatter(content)

                if fm.get("learning_stage") != "#learnt":
                    continue

                stage_updated_str = fm.get("stage_updated", "")
                if not stage_updated_str:
                    # No date recorded — treat as stale
                    is_stale = True
                else:
                    try:
                        stage_updated = datetime.date.fromisoformat(str(stage_updated_str))
                        is_stale = stage_updated < cutoff
                    except ValueError:
                        is_stale = True

                if is_stale:
                    fm["learning_stage"] = "#stale"
                    fm["stage_updated"] = datetime.date.today().isoformat()
                    md_file.write_text(write_frontmatter(fm, body))

                    rel_path = md_file.relative_to(vault_path)
                    stale_notes.append(
                        f"[yellow]{rel_path}[/] flagged [red]#stale[/]"
                    )
                    logger.info("lifecycle.flagged_stale", path=str(rel_path))

            except Exception as exc:
                logger.debug("lifecycle.staleness_check_file_error", path=str(md_file), error=str(exc))
                continue

    except Exception as exc:
        logger.debug("lifecycle.check_staleness_error", domain_path=str(domain_path), error=str(exc))

    return stale_notes


# ---------------------------------------------------------------------------
# Archive note (#archive)
# ---------------------------------------------------------------------------


def advance_stage(note_path: str, new_stage: str, vault_path: Path) -> None:
    """Centralised stage setter -- all stage updates must route here (INV-3 gate).

    If new_stage == '#learning', validates INV-3 before writing.
    Raises RuleViolation if INV-3 violated.

    Args:
        note_path: Vault-relative path of the note (e.g. 'knowledge/piano/scales.md').
        new_stage: One of '#new', '#learning', '#learnt', '#stale', '#archive'.
        vault_path: Vault root path.
    """
    from pb.core.rules import validate_no_learning_without_socratic

    if new_stage == "#learning":
        validate_no_learning_without_socratic(Path(note_path), vault_path)

    note_abs = vault_path / note_path
    content = note_abs.read_text()
    fm, body = read_frontmatter(content)
    fm["learning_stage"] = new_stage
    fm["stage_updated"] = datetime.date.today().isoformat()
    note_abs.write_text(write_frontmatter(fm, body))
    logger.info("lifecycle.stage_advanced", note_path=note_path, new_stage=new_stage)


def archive_note(note_path: str, vault_path: Path) -> Optional[str]:
    """Set a note's learning_stage to #archive (LIFE-06).

    Returns a confirmation string or None on error.
    """
    try:
        note_abs = vault_path / note_path
        if not note_abs.exists():
            return None

        content = note_abs.read_text()
        fm, body = read_frontmatter(content)

        fm["learning_stage"] = "#archive"
        fm["stage_updated"] = datetime.date.today().isoformat()
        note_abs.write_text(write_frontmatter(fm, body))

        logger.info("lifecycle.archived", note_path=note_path)
        return f"[dim]{note_path}[/] archived [red]#archive[/]"

    except Exception as exc:
        logger.debug("lifecycle.archive_note_error", note_path=note_path, error=str(exc))
        return None
