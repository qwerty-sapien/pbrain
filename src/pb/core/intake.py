# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared intake helpers for quick todo capture and task creation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time

from pb.core.dedup import find_similar_task
from pb.core.enums import Horizon, TaskState
from pb.core.exceptions import ValidationError
from pb.core.models import Task, generate_slug

_BRACKET_DUE_RE = re.compile(r"\s+@\[(\d{4}-\d{2}-\d{2})\]\s*$")
_SLASH_DUE_RE = re.compile(r"\s+/due\s+(\d{4}-\d{2}-\d{2})\s*$", re.IGNORECASE)
_BRACKET_DUE_ANY_RE = re.compile(r"@\[(\d{4}-\d{2}-\d{2})\]")
_SLASH_DUE_ANY_RE = re.compile(r"/due\s+(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


@dataclass
class ParsedTodoEntry:
    """Normalized todo capture payload."""

    raw_text: str
    title: str
    description: str
    due_date: datetime | None
    horizon: Horizon


def _require_text(text: str, *, label: str) -> str:
    normalized = " ".join((text or "").split()).strip()
    if not normalized:
        raise ValidationError(f"{label} text is required.")
    return normalized


def _derive_horizon(due_day: date | None, *, today: date | None = None) -> Horizon:
    if due_day is None:
        return Horizon.WEEK
    baseline = today or date.today()
    delta_days = (due_day - baseline).days
    if delta_days <= 0:
        return Horizon.TODAY
    if delta_days <= 7:
        return Horizon.WEEK
    return Horizon.MONTH


def parse_todo_entry(text: str, *, today: date | None = None) -> ParsedTodoEntry:
    """Parse one todo entry with optional trailing due-date syntax.

    Supported due syntaxes:
    - `@[YYYY-MM-DD]`
    - `/due YYYY-MM-DD`
    """

    raw_text = _require_text(text, label="Todo")
    bracket_any = _BRACKET_DUE_ANY_RE.search(raw_text)
    slash_any = _SLASH_DUE_ANY_RE.search(raw_text)
    if bracket_any and slash_any:
        raise ValidationError("Use only one due-date syntax: either `@[YYYY-MM-DD]` or `/due YYYY-MM-DD`.")
    bracket_match = _BRACKET_DUE_RE.search(raw_text)
    slash_match = _SLASH_DUE_RE.search(raw_text)

    title = raw_text
    parsed_due: datetime | None = None
    active_match = bracket_match or slash_match
    if active_match:
        due_raw = active_match.group(1)
        try:
            due_day = date.fromisoformat(due_raw)
        except ValueError:
            due_day = None
        else:
            parsed_due = datetime.combine(due_day, time.min)
            title = raw_text[: active_match.start()].strip()

    normalized_title = _require_text(title, label="Todo")
    description = raw_text if normalized_title != raw_text else ""
    horizon = _derive_horizon(parsed_due.date() if parsed_due is not None else None, today=today)
    return ParsedTodoEntry(
        raw_text=raw_text,
        title=normalized_title,
        description=description,
        due_date=parsed_due,
        horizon=horizon,
    )


def create_task(repo, text: str, *, due_date: datetime | None = None, today: date | None = None) -> Task:
    """Create and persist a todo task from free text.

    Checks for fuzzy duplicates among active tasks. If a near-duplicate
    exists, updates that task's completion to 0% (re-activates) instead
    of creating a new row.
    """
    if due_date is None:
        parsed = parse_todo_entry(text, today=today)
    else:
        normalized = _require_text(text, label="Todo")
        parsed = ParsedTodoEntry(
            raw_text=normalized,
            title=normalized,
            description="",
            due_date=due_date,
            horizon=_derive_horizon(due_date.date(), today=today),
        )

    # Default deadline: 1 week from today if none specified
    effective_today = today or date.today()
    if parsed.due_date is None:
        from datetime import timedelta
        default_due = effective_today + timedelta(weeks=1)
        parsed = ParsedTodoEntry(
            raw_text=parsed.raw_text,
            title=parsed.title,
            description=parsed.description,
            due_date=datetime.combine(default_due, time.min),
            horizon=Horizon.WEEK,
        )

    # Fuzzy dedup: check active tasks for near-duplicates
    try:
        active_tasks = repo.list_tasks(state=TaskState.ACTIVE)
        similar = find_similar_task(parsed.title, active_tasks)
        if similar is not None:
            if parsed.due_date and (not getattr(similar, "due_date", None) or parsed.due_date > similar.due_date):
                similar.due_date = parsed.due_date
            similar.completion = 0
            repo.update_task(similar)
            return similar
    except Exception:
        pass  # Dedup is best-effort; never blocks creation

    task = Task(
        id=generate_slug(parsed.title),
        title=parsed.title,
        description=parsed.description,
        horizon=parsed.horizon,
        state=TaskState.ACTIVE,
        due_date=parsed.due_date,
        work_type="todo",
    )
    repo.create_task(task)
    return task
