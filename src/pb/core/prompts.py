# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Proactive prompts engine -- deterministic relationship intelligence (D-17, D-18).

Scans all active person notes and generates prompts for:
- overdue_commitment: past cadence window or due date
- birthday: within 14 days
- gift_reminder: idea exists + occasion approaching
- decay_warning: no interaction in 90+ days or 2x cadence

All computation is date math only -- no LLM dependency (D-18).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Literal, Optional

import structlog

try:
    from pb.core.people import (
        PeopleManager,
        _parse_frontmatter,
        extract_section,
        last_contact_date,
        CADENCE_WINDOWS,
    )
except ImportError:
    # pb.core.people quarantined to remove_review/ (Phase 1 D-08)
    PeopleManager = None  # type: ignore[assignment, misc]
    _parse_frontmatter = None  # type: ignore[assignment]
    extract_section = None  # type: ignore[assignment]
    last_contact_date = None  # type: ignore[assignment]
    CADENCE_WINDOWS = {}  # type: ignore[assignment]

logger = structlog.get_logger()


# -- Prompt dataclass -------------------------------------------------------


@dataclass
class Prompt:
    """A single actionable prompt surfaced to the user.

    prompt_type: category of prompt
    person_name: who it concerns
    message: human-readable description
    urgency: positive int -- higher = more urgent (days overdue or urgency score)
    """

    prompt_type: Literal[
        "overdue_commitment", "birthday", "gift_reminder", "decay_warning",
        "goal_deadline", "event_prep", "skill_gap", "stale_inbox",
    ]
    person_name: str
    message: str
    urgency: int


# -- Commitment parsing helpers ---------------------------------------------

# Extracts cadence and last-done date: (cadence: monthly, last: 2026-03-15)
CADENCE_RE = re.compile(r"\(cadence:\s*(\w+),\s*last:\s*(\d{4}-\d{2}-\d{2})\)")

# Extracts due date: (due: 2026-05-01)
DUE_RE = re.compile(r"\(due:\s*(\d{4}-\d{2}-\d{2})\)")


def _commitment_prompts(name: str, commitments_section: str, today: date) -> list[Prompt]:
    """Generate overdue_commitment prompts from ## Commitments section.

    Only processes open items (lines starting with ``- [ ]``).
    Cadence-based: overdue if days_since > CADENCE_WINDOWS[cadence].
    Due-date: overdue if today > due date.
    Malformed dates skipped with warning (T-04-07).
    """
    prompts: list[Prompt] = []
    if not commitments_section:
        return prompts

    for line in commitments_section.splitlines():
        line = line.strip()
        # Only open items (Pitfall 5: skip completed [x])
        if not line.startswith("- [ ]"):
            continue

        # Try cadence-based
        cadence_match = CADENCE_RE.search(line)
        if cadence_match:
            cadence_str = cadence_match.group(1)
            last_str = cadence_match.group(2)
            try:
                last_date = date.fromisoformat(last_str)
                window = CADENCE_WINDOWS.get(cadence_str)
                if window is not None:
                    days_since = (today - last_date).days
                    if days_since > window:
                        urgency = days_since - window
                        # Extract description (between "- [ ] " and "(")
                        desc = line[6:].split("(")[0].strip()
                        prompts.append(Prompt(
                            prompt_type="overdue_commitment",
                            person_name=name,
                            message=f"Overdue: {desc} ({days_since - window}d past {cadence_str} window)",
                            urgency=urgency,
                        ))
            except (ValueError, TypeError):
                logger.warning("prompts.malformed_commitment_date", line=line)
            continue

        # Try due-date based
        due_match = DUE_RE.search(line)
        if due_match:
            due_str = due_match.group(1)
            try:
                due_date = date.fromisoformat(due_str)
                if today > due_date:
                    days_overdue = (today - due_date).days
                    desc = line[6:].split("(")[0].strip()
                    prompts.append(Prompt(
                        prompt_type="overdue_commitment",
                        person_name=name,
                        message=f"Overdue: {desc} ({days_overdue}d past due)",
                        urgency=days_overdue,
                    ))
            except (ValueError, TypeError):
                logger.warning("prompts.malformed_due_date", line=line)

    return prompts


# -- Birthday prompt helper -------------------------------------------------


def _birthday_prompts(frontmatter: dict, name: str, today: date) -> list[Prompt]:
    """Generate birthday prompts for occasions within 14 days (D-17).

    Birthday format in frontmatter: ``birthday: "1990-05-15"`` (YYYY-MM-DD).
    If birthday already passed this year, no prompt.
    urgency = 14 - days_until (higher urgency = sooner).
    Malformed dates skipped (T-04-07).
    """
    birthday_str = frontmatter.get("birthday")
    if not birthday_str:
        return []

    try:
        # Handle both string and date objects from YAML
        if isinstance(birthday_str, date):
            bday = birthday_str
        else:
            bday = date.fromisoformat(str(birthday_str))

        # This year's occurrence
        this_year_bday = bday.replace(year=today.year)

        # If already passed this year, no prompt
        if this_year_bday < today:
            return []

        days_until = (this_year_bday - today).days
        if 0 <= days_until <= 14:
            if days_until == 0:
                msg = "Birthday is today!"
            else:
                msg = f"Birthday in {days_until} days"
            return [Prompt(
                prompt_type="birthday",
                person_name=name,
                message=msg,
                urgency=14 - days_until,
            )]
    except (ValueError, TypeError, OverflowError):
        logger.warning("prompts.malformed_birthday", name=name)

    return []


# -- Gift reminder helper ---------------------------------------------------

# Extracts occasion type and YYYY-MM: (occasion: birthday 2026-05)
GIFT_OCCASION_RE = re.compile(r"\(occasion:\s*(\w+)\s+(\d{4}-\d{2})\)")


def _gift_prompts(name: str, gifts_section: str, frontmatter: dict, today: date) -> list[Prompt]:
    """Generate gift_reminder prompts for idea-stage gifts with approaching occasions.

    Only processes idea lines (starting with ``- \U0001f4a1``).
    Purchased/given items don't need reminders.
    Occasion month within 30 days of today triggers reminder.
    """
    prompts: list[Prompt] = []
    if not gifts_section:
        return prompts

    for line in gifts_section.splitlines():
        line = line.strip()
        # Only idea stage (D-08: 💡 = idea, 🛒 = purchased, ✅ = given)
        if not line.startswith("- \U0001f4a1"):
            continue

        occasion_match = GIFT_OCCASION_RE.search(line)
        if not occasion_match:
            continue

        occasion_type = occasion_match.group(1)
        occasion_ym = occasion_match.group(2)

        try:
            # Parse occasion as first day of month
            occasion_date = date.fromisoformat(f"{occasion_ym}-01")
            days_until = (occasion_date - today).days

            if 0 <= days_until <= 30:
                # Extract item name: between "💡 " and " ("
                item_part = line.split("\U0001f4a1", 1)[1].strip()
                item_name = item_part.split("(")[0].strip()
                prompts.append(Prompt(
                    prompt_type="gift_reminder",
                    person_name=name,
                    message=f"Gift idea: {item_name} for {occasion_type} ({occasion_ym})",
                    urgency=max(1, 30 - days_until),
                ))
        except (ValueError, TypeError):
            logger.warning("prompts.malformed_gift_occasion", line=line)

    return prompts


# -- Decay warning helper ---------------------------------------------------


def _decay_prompts(name: str, interaction_log: str, cadence: str, today: date) -> list[Prompt]:
    """Generate decay_warning prompts for relationship decay (D-17).

    - No contact ever: decay_warning with urgency=999 ("Never contacted")
    - cadence != "none": threshold = window * 2; warn if days_since > threshold
    - cadence == "none": warn if days_since > 90
    """
    last = last_contact_date(interaction_log)

    if last is None:
        return [Prompt(
            prompt_type="decay_warning",
            person_name=name,
            message="Never contacted",
            urgency=999,
        )]

    days_since = (today - last).days
    window = CADENCE_WINDOWS.get(cadence)

    if window is not None:
        # Cadence-based: warn at 2x cadence window
        threshold = window * 2
        if days_since > threshold:
            return [Prompt(
                prompt_type="decay_warning",
                person_name=name,
                message=f"No contact in {days_since}d (2x {cadence} window exceeded)",
                urgency=days_since,
            )]
    else:
        # No cadence: warn at 90+ days (D-17)
        if days_since > 90:
            return [Prompt(
                prompt_type="decay_warning",
                person_name=name,
                message=f"No contact in {days_since}d (90d threshold exceeded)",
                urgency=days_since,
            )]

    return []


# -- Cross-domain prompt helpers (D-13) ------------------------------------

STALE_INBOX_THRESHOLD_DAYS = 7


def _goal_deadline_prompts(vault_path: Path, today: date) -> list[Prompt]:
    """Goals with target_date within 14 days and status != complete (D-13)."""
    goals_dir = vault_path / "direction" / "goals"
    prompts: list[Prompt] = []
    if not goals_dir.exists():
        return prompts
    for md_file in goals_dir.glob("*.md"):
        try:
            fm = _parse_frontmatter(md_file.read_text())
            if fm.get("status") == "complete":
                continue
            target = fm.get("target_date")
            if not target:
                continue
            target_date = date.fromisoformat(str(target))
            days = (target_date - today).days
            if 0 <= days <= 14:
                urgency = 14 - days  # closer = more urgent
                title = fm.get("title", md_file.stem)
                prompts.append(Prompt(
                    prompt_type="goal_deadline",
                    person_name=title,
                    message=f"Goal deadline in {days}d: {title}",
                    urgency=urgency,
                ))
        except Exception:
            logger.warning("prompts.goal_scan_failed", file=str(md_file))
    return prompts


def _event_prep_prompts(vault_path: Path, today: date) -> list[Prompt]:
    """Events within 48 hours that need preparation (D-13)."""
    events_dir = vault_path / "events"
    prompts: list[Prompt] = []
    if not events_dir.exists():
        return prompts
    for md_file in events_dir.glob("*.md"):
        try:
            fm = _parse_frontmatter(md_file.read_text())
            if fm.get("type") != "event":
                continue
            event_date_str = fm.get("date")
            if not event_date_str:
                continue
            event_date = date.fromisoformat(str(event_date_str)[:10])
            days = (event_date - today).days
            if 0 <= days <= 2:  # within 48h
                title = fm.get("title", md_file.stem)
                linked = fm.get("linked_people", [])
                people_note = f" (with {', '.join(linked)})" if linked else ""
                hours = days * 24
                msg = (
                    f"Event in {hours}h: {title}{people_note}"
                    if days > 0
                    else f"Event today: {title}{people_note}"
                )
                prompts.append(Prompt(
                    prompt_type="event_prep",
                    person_name=title,
                    message=msg,
                    urgency=3 - days,  # today=3, tomorrow=2, day after=1
                ))
        except Exception:
            logger.warning("prompts.event_scan_failed", file=str(md_file))
    return prompts


def _skill_gap_prompts(vault_path: Path, today: date) -> list[Prompt]:
    """Opportunities with weak skill match and deadline within 30 days (D-13)."""
    opps_dir = vault_path / "opportunities"
    prompts: list[Prompt] = []
    if not opps_dir.exists():
        return prompts
    for md_file in opps_dir.glob("*.md"):
        try:
            fm = _parse_frontmatter(md_file.read_text())
            if fm.get("status") == "archived":
                continue
            deadline_str = fm.get("deadline")
            if not deadline_str:
                continue
            deadline = date.fromisoformat(str(deadline_str)[:10])
            days = (deadline - today).days
            if days < 0 or days > 30:
                continue
            score = fm.get("skill_match_score", 1.0)
            try:
                score = float(score)
            except (ValueError, TypeError):
                score = 1.0
            if score >= 0.7:  # strong match, no nudge needed
                continue
            title = fm.get("title", md_file.stem)
            prompts.append(Prompt(
                prompt_type="skill_gap",
                person_name=title,
                message=f"Skill gap: {title} (match {score:.0%}, deadline in {days}d)",
                urgency=30 - days,
            ))
        except Exception:
            logger.warning("prompts.opp_scan_failed", file=str(md_file))
    return prompts


def _stale_inbox_prompts(vault_path: Path, today: date) -> list[Prompt]:
    """Inbox items older than 7 days that haven't been triaged (D-13)."""
    inbox_dir = vault_path / "00-inbox"
    prompts: list[Prompt] = []
    if not inbox_dir.exists():
        return prompts
    # Scan .md files in inbox root (not subdirectories to avoid feeds/gmail/captures)
    for md_file in inbox_dir.glob("*.md"):
        try:
            fm = _parse_frontmatter(md_file.read_text())
            created_str = fm.get("created")
            if not created_str:
                continue
            created_date = date.fromisoformat(str(created_str)[:10])
            days_old = (today - created_date).days
            if days_old >= STALE_INBOX_THRESHOLD_DAYS:
                title = fm.get("title", md_file.stem)
                prompts.append(Prompt(
                    prompt_type="stale_inbox",
                    person_name=title,
                    message=f"Stale inbox ({days_old}d): {title}",
                    urgency=days_old,
                ))
        except Exception:
            logger.warning("prompts.inbox_scan_failed", file=str(md_file))
    return prompts


# -- ProactivePromptsEngine -------------------------------------------------


class ProactivePromptsEngine:
    """Scan all active person notes and generate deterministic prompts.

    Prompts are sorted by urgency descending (most urgent first).
    Each person is processed in try/except -- malformed notes are skipped (non-fatal).
    No LLM calls: all prompt computation is date math (D-18).
    """

    def __init__(self, vault_path: Optional[Path] = None):
        self.people_mgr = PeopleManager(vault_path)
        self.vault_path = vault_path or self.people_mgr.vault_path

    def get_prompts(self, today: Optional[date] = None) -> list[Prompt]:
        """Scan all active person notes and cross-domain vault folders, return sorted list of prompts."""
        today = today or date.today()
        prompts: list[Prompt] = []

        # People-based prompts (D-17)
        if self.people_mgr.people_dir.exists():
            for md_file in sorted(self.people_mgr.people_dir.glob("*.md")):
                try:
                    content = md_file.read_text()
                    fm = _parse_frontmatter(content)

                    # Filter non-person notes and archived (D-13, Pitfall 4)
                    if fm.get("type") != "person" or fm.get("status") == "archived":
                        continue

                    name = fm.get("name", md_file.stem)
                    prompts.extend(self._prompts_for_person(fm, content, name, today))
                except Exception:
                    logger.warning("prompts.scan_failed", file=str(md_file))

        # Cross-domain prompts (D-13)
        vault_path = self.vault_path
        prompts.extend(_goal_deadline_prompts(vault_path, today))
        prompts.extend(_event_prep_prompts(vault_path, today))
        prompts.extend(_skill_gap_prompts(vault_path, today))
        prompts.extend(_stale_inbox_prompts(vault_path, today))

        return sorted(prompts, key=lambda p: -p.urgency)

    def _prompts_for_person(
        self, fm: dict, content: str, name: str, today: date
    ) -> list[Prompt]:
        """Generate all prompt types for a single person."""
        results: list[Prompt] = []

        # Birthday prompts
        results.extend(_birthday_prompts(fm, name, today))

        # Commitment prompts
        commitments = extract_section(content, "Commitments")
        results.extend(_commitment_prompts(name, commitments, today))

        # Gift prompts
        gifts = extract_section(content, "Gifts")
        results.extend(_gift_prompts(name, gifts, fm, today))

        # Decay prompts
        interaction_log = extract_section(content, "Interaction Log")
        cadence = fm.get("contact_cadence", "none")
        results.extend(_decay_prompts(name, interaction_log, cadence, today))

        return results
