# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Session log writer for the quarantine inbox.

Writes a session log note to vault/Learning/Inbox/pb/sessions/ on every finish.
Per Phase 3 D-01, D-04, D-05, D-06, D-07 decisions.

Non-fatal: vault write failures log a warning and return None.
"""

from __future__ import annotations
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional

import structlog

from pb.core.durations import elapsed_minutes_and_label
from pb.core.graph_writer import make_slug
from pb.core.resources import read_template_text
from pb.domain.models import Project, Session, Task

logger = structlog.get_logger()


class SessionLogWriter:
    """Writes session log notes to vault/Learning/Inbox/pb/sessions/.

    Each `pb finish` call produces one note:
      {vault}/Learning/Inbox/pb/sessions/{YYYY-MM-DD}-{task-slug}.md

    Collision handling (D-07): if the file already exists, appends -2, -3, etc.
    Vault failures are non-fatal: log warning and return None.
    """

    def __init__(self, vault_path: Optional[Path] = None):
        if vault_path is None:
            from pb.storage.config import get_quarantine_path

            quarantine_root = get_quarantine_path()
        else:
            quarantine_root = vault_path / "Learning" / "Inbox" / "pb"
        self.quarantine_root = quarantine_root
        self.sessions_dir = self.quarantine_root / "sessions"

    def _unique_path(self, base_path: Path) -> Path:
        """Return base_path if it does not exist, else append -2, -3, etc."""
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

    def write_session_log(
        self,
        session: Session,
        task: Task,
        project: Optional[Project],
        next_steps: list,
    ) -> Optional[Path]:
        """Write session log note to vault. Returns path or None on failure.

        Path: {vault}/Learning/Inbox/pb/sessions/{YYYY-MM-DD}-{task-slug}.md

        Args:
            session: The completed session.
            task: The task associated with this session.
            project: Optional project the task belongs to.
            next_steps: List of next step strings (rendered as wikilinks).

        Returns:
            Path to the written file, or None if write failed.
        """
        try:
            self.sessions_dir.mkdir(parents=True, exist_ok=True)

            # Date from session end (or utcnow if missing)
            date_str = (
                session.end_at.strftime("%Y-%m-%d")
                if session.end_at
                else datetime.utcnow().strftime("%Y-%m-%d")
            )

            # Slug from task title -- path safety via make_slug
            slug = make_slug(task.title)

            # Collision-safe path (D-07)
            base = self.sessions_dir / f"{date_str}-{slug}.md"
            path = self._unique_path(base)

            # Compute elapsed duration separately from any planned timer.
            duration_min, duration_display = elapsed_minutes_and_label(session.start_at, session.end_at)

            # Time strings
            start_time = session.start_at.strftime("%H:%M") if session.start_at else ""
            end_time = session.end_at.strftime("%H:%M") if session.end_at else ""

            # Task note link (wikilink slug)
            task_note_link = slug

            # Project name
            project_name = project.name if project else "_unassigned"

            # Tags: always include "session", plus linked track IDs
            tag_items = ["session"]
            if task.linked_track_ids:
                tag_items.extend(task.linked_track_ids)
            tags = "[" + ", ".join(tag_items) + "]"

            # Next steps section
            if next_steps:
                next_steps_section = "\n".join(f"- [[{s}]]" for s in next_steps)
            else:
                next_steps_section = "_None_"

            # Actual outcome
            actual_outcome = session.actual_outcome or ""

            # Completion and distraction
            completion_pct = (
                str(session.completion_pct)
                if session.completion_pct is not None
                else ""
            )
            distraction = (
                str(session.distraction) if session.distraction is not None else ""
            )

            # Render template
            template = Template(read_template_text("session_log.md"))
            content = template.safe_substitute(
                date=date_str,
                task_note_link=task_note_link,
                task_title=task.title,
                project_name=project_name,
                duration_min=str(duration_min),
                duration_display=duration_display,
                planned_duration_min=str(getattr(session, "duration_minutes", "") or ""),
                start_time=start_time,
                end_time=end_time,
                completion_pct=completion_pct,
                distraction=distraction,
                tags=tags,
                actual_outcome=actual_outcome,
                next_steps_section=next_steps_section,
            )

            path.write_text(content)
            logger.info("session_log_writer.written", path=str(path))
            return path

        except Exception as e:
            logger.warning("session_log_writer.failed", error=str(e))
            return None
