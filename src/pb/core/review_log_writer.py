# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Review log writer for daily and weekly quarantine notes.

Writes review output to the ProductiveBrain quarantine area:
- vault/Learning/Inbox/pb/reviews/daily/
- vault/Learning/Inbox/pb/reviews/weekly/

Per Phase 3 D-02, D-03, D-04, D-06, D-07.
All writes are non-fatal (I-09): log warning and return None on failure.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Optional

import structlog

logger = structlog.get_logger()


class ReviewLogWriter:
    """Writes daily and weekly review logs to the quarantine inbox.

    Daily logs: vault/Learning/Inbox/pb/reviews/daily/{YYYY-MM-DD}-daily.md
    Weekly logs: vault/Learning/Inbox/pb/reviews/weekly/{YYYY}-W{WW}-weekly.md

    Uses ISO 8601 week numbering via isocalendar() for weekly logs.
    Never uses strftime("%W") which is Sunday-based.

    Vault write failures are non-fatal: logs a warning and returns None.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        if vault_path is None:
            from pb.storage.config import get_quarantine_path

            quarantine_root = get_quarantine_path()
        else:
            quarantine_root = vault_path / "Learning" / "Inbox" / "pb"
        self.quarantine_root = quarantine_root
        self.daily_dir = self.quarantine_root / "reviews" / "daily"
        self.weekly_dir = self.quarantine_root / "reviews" / "weekly"

    def _unique_path(self, base_path: Path) -> Path:
        """Return base_path if it does not exist, else append -2, -3, etc.

        Replicates GraphWriter._unique_path() collision handling (D-07).
        """
        if not base_path.exists():
            return base_path
        stem = base_path.stem
        suffix = base_path.suffix
        parent = base_path.parent
        counter = 2
        while True:
            candidate = parent / f"{stem}-{counter}{suffix}"
            if not candidate.exists():
                return candidate
            counter += 1

    def write_daily_log(self, content: str, dt: date) -> Optional[Path]:
        """Write daily review log to vault. Returns path or None on failure.

        Creates file at vault/Learning/Inbox/pb/reviews/daily/{YYYY-MM-DD}-daily.md
        with YAML frontmatter containing type and date fields.

        Args:
            content: The review output string to write as the note body.
            dt: The date of the review.

        Returns:
            Path to the written file, or None if the write failed.
        """
        try:
            self.daily_dir.mkdir(parents=True, exist_ok=True)
            date_str = dt.strftime("%Y-%m-%d")
            frontmatter = (
                f"---\n"
                f"type: daily_log\n"
                f"date: {date_str}\n"
                f"---\n\n"
            )
            full_content = frontmatter + content
            base = self.daily_dir / f"{date_str}-daily.md"
            path = self._unique_path(base)
            path.write_text(full_content)
            logger.info("review_log_writer.daily_written", path=str(path))
            return path
        except Exception as e:
            logger.warning("review_log_writer.daily_failed", error=str(e))
            return None

    def write_weekly_log(self, content: str, dt: date) -> Optional[Path]:
        """Write weekly review log to vault. Returns path or None on failure.

        Creates file at vault/Learning/Inbox/pb/reviews/weekly/{YYYY}-W{WW}-weekly.md
        Uses ISO 8601 week numbering via dt.isocalendar() -- never strftime("%W").

        Args:
            content: The review output string to write as the note body.
            dt: The date of the review (used to derive ISO week).

        Returns:
            Path to the written file, or None if the write failed.
        """
        try:
            self.weekly_dir.mkdir(parents=True, exist_ok=True)
            iso = dt.isocalendar()
            week_str = f"{iso[0]}-W{iso[1]:02d}"
            frontmatter = (
                f"---\n"
                f"type: weekly_log\n"
                f"week: {week_str}\n"
                f"date: {dt.strftime('%Y-%m-%d')}\n"
                f"---\n\n"
            )
            full_content = frontmatter + content
            base = self.weekly_dir / f"{week_str}-weekly.md"
            path = self._unique_path(base)
            path.write_text(full_content)
            logger.info("review_log_writer.weekly_written", path=str(path))
            return path
        except Exception as e:
            logger.warning("review_log_writer.weekly_failed", error=str(e))
            return None
