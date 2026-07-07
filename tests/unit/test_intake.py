"""Tests for shared quick-capture intake helpers."""

from __future__ import annotations

from datetime import date

import pytest

from pb.core.enums import Horizon
from pb.core.exceptions import ValidationError
from pb.core.intake import parse_todo_entry


def test_parse_todo_entry_accepts_bracket_due_syntax():
    entry = parse_todo_entry("Draft chapter @[2026-05-20]", today=date(2026, 5, 18))

    assert entry.title == "Draft chapter"
    assert entry.due_date is not None
    assert entry.due_date.strftime("%Y-%m-%d") == "2026-05-20"
    assert entry.horizon == Horizon.WEEK


def test_parse_todo_entry_accepts_slash_due_syntax():
    entry = parse_todo_entry("Submit summary /due 2026-05-18", today=date(2026, 5, 18))

    assert entry.title == "Submit summary"
    assert entry.due_date is not None
    assert entry.horizon == Horizon.TODAY


def test_parse_todo_entry_leaves_invalid_date_like_text_untouched():
    entry = parse_todo_entry("Research idea @[2026-99-99]", today=date(2026, 5, 18))

    assert entry.title == "Research idea @[2026-99-99]"
    assert entry.due_date is None
    assert entry.description == ""


def test_parse_todo_entry_rejects_mixed_due_syntaxes():
    with pytest.raises(ValidationError):
        parse_todo_entry("Outline talk @[2026-05-20] /due 2026-05-21")


def test_parse_todo_entry_defaults_to_week_without_due_date():
    entry = parse_todo_entry("Follow up with advisor", today=date(2026, 5, 18))

    assert entry.due_date is None
    assert entry.horizon == Horizon.WEEK


def test_parse_todo_entry_no_due_still_gets_week_horizon():
    """No explicit due -> horizon defaults to WEEK (deadline assigned at creation)."""
    entry = parse_todo_entry("Clean desk", today=date(2026, 6, 6))
    assert entry.horizon == Horizon.WEEK
    assert entry.due_date is None  # parse doesn't assign default; create_todo_task does
