# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Retry queue writer for the learning evidence system.

Provides CRUD operations for the retry_queue SQLite table.
Resurfaces weak sub-skills and incomplete session items.

Per Phase 2 decisions:
  D-13: NOT spaced repetition. Simple resurface list with priority and cooldown.
  D-14: Assessment-identified weaknesses are enqueued with priority=1.
  D-15: cooldown_until prevents showing same item twice in one day.

Priority levels:
  1 = assessment-identified weakness (highest priority)
  2 = incomplete session item
  3 = manual/optional (lowest priority)

Security (T-02-02): all queries use ? parameterized placeholders -- never f-string SQL.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from typing import Optional

import structlog
from pydantic import BaseModel

logger = structlog.get_logger()


class RetryItem(BaseModel):
    """A single retry queue item."""

    id: str
    domain: str
    item_text: str
    source: str = "manual"          # "assessment" | "incomplete" | "manual"
    priority: int = 1               # 1=weakness, 2=incomplete, 3=optional (lower=higher priority)
    status: str = "pending"         # "pending" | "resolved"
    cooldown_until: Optional[str] = None  # ISO date string (YYYY-MM-DD)
    evidence_id: Optional[str] = None
    created_at: str = ""
    updated_at: str = ""


class RetryQueueWriter:
    """CRUD operations for the retry_queue SQLite table.

    Per D-13: NOT spaced repetition. Simple resurface list with priority and cooldown.
    Per D-15: cooldown_until prevents showing same item twice in one day.
    Priority: 1=assessment-identified weakness, 2=incomplete session, 3=manual/optional.

    All SQL queries use ? parameterized placeholders (T-02-02).
    """

    def enqueue(
        self,
        domain: str,
        item_text: str,
        source: str = "manual",
        priority: int = 3,
        evidence_id: Optional[str] = None,
    ) -> Optional[str]:
        """Add an item to the retry queue. Returns item ID or None on failure."""
        try:
            from pb.storage.database import get_connection
            item_id = str(uuid.uuid4())
            now = datetime.utcnow().isoformat()
            with get_connection() as conn:
                conn.execute(
                    """INSERT INTO retry_queue
                       (id, domain, item_text, source, priority, status, cooldown_until, evidence_id, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, 'pending', NULL, ?, ?, ?)""",
                    (item_id, domain, item_text, source, priority, evidence_id, now, now),
                )
                conn.commit()
            logger.info("retry_queue.enqueued", item_id=item_id, domain=domain, source=source)
            return item_id
        except Exception as e:
            logger.warning("retry_queue.enqueue_failed", error=str(e))
            return None

    def enqueue_from_assessment(
        self,
        domain: str,
        retry_items: list[str],
        evidence_id: str,
    ) -> list[str]:
        """Enqueue multiple items from an assessment result (per D-10, D-14).

        Assessment-identified weaknesses get priority=1.
        Returns list of created item IDs.
        """
        ids = []
        for item_text in retry_items:
            item_id = self.enqueue(
                domain=domain,
                item_text=item_text,
                source="assessment",
                priority=1,
                evidence_id=evidence_id,
            )
            if item_id:
                ids.append(item_id)
        return ids

    def resolve(self, item_id: str) -> bool:
        """Mark a retry item as resolved. Returns True on success."""
        try:
            from pb.storage.database import get_connection
            now = datetime.utcnow().isoformat()
            with get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE retry_queue SET status = 'resolved', updated_at = ? WHERE id = ?",
                    (now, item_id),
                )
                conn.commit()
                if cursor.rowcount == 0:
                    logger.warning("retry_queue.resolve_not_found", item_id=item_id)
                    return False
            logger.info("retry_queue.resolved", item_id=item_id)
            return True
        except Exception as e:
            logger.warning("retry_queue.resolve_failed", error=str(e))
            return False

    def reschedule(self, item_id: str, cooldown_date: Optional[str] = None) -> bool:
        """Reschedule a retry item by setting its cooldown date.

        If cooldown_date is None, defaults to tomorrow (per D-15: 1-day cooldown).
        cooldown_date format: "YYYY-MM-DD".
        """
        try:
            from pb.storage.database import get_connection
            if cooldown_date is None:
                cooldown_date = (datetime.utcnow() + timedelta(days=1)).strftime("%Y-%m-%d")
            now = datetime.utcnow().isoformat()
            with get_connection() as conn:
                cursor = conn.execute(
                    "UPDATE retry_queue SET cooldown_until = ?, updated_at = ? WHERE id = ?",
                    (cooldown_date, now, item_id),
                )
                conn.commit()
                if cursor.rowcount == 0:
                    logger.warning("retry_queue.reschedule_not_found", item_id=item_id)
                    return False
            logger.info("retry_queue.rescheduled", item_id=item_id, cooldown_until=cooldown_date)
            return True
        except Exception as e:
            logger.warning("retry_queue.reschedule_failed", error=str(e))
            return False

    def list_pending(
        self,
        domain: str = "",
        today: Optional[str] = None,
        limit: int = 10,
    ) -> list[RetryItem]:
        """List pending retry items past their cooldown, ordered by priority.

        Per D-15: items where cooldown_until > today are suppressed.
        All SQL uses ? parameterized placeholders (T-02-02).
        """
        try:
            from pb.storage.database import get_connection
            today = today or datetime.utcnow().strftime("%Y-%m-%d")
            if domain:
                query = """
                    SELECT * FROM retry_queue
                    WHERE status = 'pending'
                      AND (cooldown_until IS NULL OR cooldown_until <= ?)
                      AND domain = ?
                    ORDER BY priority ASC, created_at ASC
                    LIMIT ?
                """
                params = (today, domain, limit)
            else:
                query = """
                    SELECT * FROM retry_queue
                    WHERE status = 'pending'
                      AND (cooldown_until IS NULL OR cooldown_until <= ?)
                    ORDER BY priority ASC, created_at ASC
                    LIMIT ?
                """
                params = (today, limit)

            with get_connection() as conn:
                rows = conn.execute(query, params).fetchall()
                return [RetryItem(**dict(row)) for row in rows]
        except Exception as e:
            logger.warning("retry_queue.list_failed", error=str(e))
            return []
