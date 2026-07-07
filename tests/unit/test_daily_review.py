"""Unit tests for daily review debrief system (D-30 to D-33).

Tests 5-section compact debrief: completion, friction, energy, learning, tomorrow.
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Optional
from unittest.mock import MagicMock, patch

import pytest

from pb.domain.models import DailyDebrief
from pb.storage.database import init_db, set_db_path


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_db(tmp_path):
    """Provide a temporary database for tests."""
    db_path = tmp_path / "test_debrief.db"
    set_db_path(db_path)
    init_db(db_path)
    yield db_path
    set_db_path(None)


@pytest.fixture
def repo(tmp_db):
    """Provide a repository backed by the temp database."""
    from pb.storage.repository import Repository
    return Repository()


@pytest.fixture
def engine(repo):
    """Provide a ReviewEngine backed by the temp repository."""
    from pb.core.review_engine import ReviewEngine
    return ReviewEngine(repo)


@pytest.fixture
def full_debrief():
    """Provide a fully populated DailyDebrief for testing."""
    return DailyDebrief(
        review_date="2026-04-25",
        top1_completed="yes",
        top3_completed=["Write review tests", "Fix login bug"],
        what_shipped="Review engine overhaul committed",
        biggest_blocker="unclear_next_action",
        blocker_note="Wasn't sure which test to write first",
        energy_morning=4,
        energy_midday=3,
        energy_evening=2,
        energy_task_match="yes",
        learning_question="What should I repeat tomorrow?",
        learning_answer="Writing tests first made the implementation obvious",
        learning_score=8,
        learning_rationale="Specific and actionable answer",
        tomorrow_top1="Deploy review engine changes",
        tomorrow_next_action="Run full test suite and verify",
    )


# ---------------------------------------------------------------------------
# DailyDebrief model tests
# ---------------------------------------------------------------------------


class TestDailyDebriefModel:
    """Tests for the DailyDebrief domain model."""

    def test_daily_debrief_has_id(self):
        """DailyDebrief has auto-generated UUID4 id."""
        d = DailyDebrief(review_date="2026-04-25")
        assert d.id is not None
        assert len(d.id) == 36  # UUID4 with dashes

    def test_daily_debrief_requires_review_date(self):
        """DailyDebrief requires review_date field."""
        d = DailyDebrief(review_date="2026-04-25")
        assert d.review_date == "2026-04-25"

    def test_daily_debrief_has_completion_fields(self):
        """DailyDebrief has section A (completion) fields."""
        d = DailyDebrief(review_date="2026-04-25")
        assert hasattr(d, "top1_completed")
        assert hasattr(d, "top3_completed")
        assert hasattr(d, "what_shipped")

    def test_daily_debrief_has_friction_fields(self):
        """DailyDebrief has section B (friction) fields."""
        d = DailyDebrief(review_date="2026-04-25")
        assert hasattr(d, "biggest_blocker")
        assert hasattr(d, "blocker_note")

    def test_daily_debrief_has_energy_fields(self):
        """DailyDebrief has section C (energy) fields."""
        d = DailyDebrief(review_date="2026-04-25")
        assert hasattr(d, "energy_morning")
        assert hasattr(d, "energy_midday")
        assert hasattr(d, "energy_evening")
        assert hasattr(d, "energy_task_match")

    def test_daily_debrief_has_learning_fields(self):
        """DailyDebrief has section D (learning) fields."""
        d = DailyDebrief(review_date="2026-04-25")
        assert hasattr(d, "learning_question")
        assert hasattr(d, "learning_answer")
        assert hasattr(d, "learning_score")
        assert hasattr(d, "learning_rationale")

    def test_daily_debrief_has_tomorrow_fields(self):
        """DailyDebrief has section E (tomorrow) fields."""
        d = DailyDebrief(review_date="2026-04-25")
        assert hasattr(d, "tomorrow_top1")
        assert hasattr(d, "tomorrow_next_action")

    def test_daily_debrief_top3_defaults_to_empty_list(self):
        """DailyDebrief top3_completed defaults to empty list."""
        d = DailyDebrief(review_date="2026-04-25")
        assert d.top3_completed == []

    def test_daily_debrief_optional_fields_default_none(self):
        """DailyDebrief optional fields default to None."""
        d = DailyDebrief(review_date="2026-04-25")
        assert d.top1_completed is None
        assert d.biggest_blocker is None
        assert d.energy_morning is None
        assert d.learning_question is None
        assert d.tomorrow_top1 is None

    def test_daily_debrief_has_created_at(self):
        """DailyDebrief has created_at datetime."""
        d = DailyDebrief(review_date="2026-04-25")
        assert d.created_at is not None
        assert isinstance(d.created_at, datetime)


# ---------------------------------------------------------------------------
# LEARNING_QUESTIONS tests
# ---------------------------------------------------------------------------


class TestLearningQuestions:
    """Tests for the rotating learning questions constant."""

    def test_learning_questions_has_exactly_7(self):
        """LEARNING_QUESTIONS has exactly 7 questions."""
        from pb.core.review_engine import LEARNING_QUESTIONS
        assert len(LEARNING_QUESTIONS) == 7

    def test_all_learning_questions_are_strings(self):
        """All LEARNING_QUESTIONS are non-empty strings."""
        from pb.core.review_engine import LEARNING_QUESTIONS
        for q in LEARNING_QUESTIONS:
            assert isinstance(q, str)
            assert len(q) > 0

    def test_learning_questions_are_unique(self):
        """All LEARNING_QUESTIONS are distinct."""
        from pb.core.review_engine import LEARNING_QUESTIONS
        assert len(set(LEARNING_QUESTIONS)) == len(LEARNING_QUESTIONS)


# ---------------------------------------------------------------------------
# DEBRIEF_BLOCKERS tests
# ---------------------------------------------------------------------------


class TestDebriefBlockers:
    """Tests for the predefined blocker list."""

    def test_debrief_blockers_has_at_least_8(self):
        """DEBRIEF_BLOCKERS has at least 8 options."""
        from pb.core.review_engine import DEBRIEF_BLOCKERS
        assert len(DEBRIEF_BLOCKERS) >= 8

    def test_debrief_blockers_includes_unclear_next_action(self):
        """DEBRIEF_BLOCKERS includes 'unclear_next_action'."""
        from pb.core.review_engine import DEBRIEF_BLOCKERS
        assert "unclear_next_action" in DEBRIEF_BLOCKERS

    def test_debrief_blockers_includes_other(self):
        """DEBRIEF_BLOCKERS includes 'other' as fallback."""
        from pb.core.review_engine import DEBRIEF_BLOCKERS
        assert "other" in DEBRIEF_BLOCKERS

    def test_debrief_blockers_are_strings(self):
        """All DEBRIEF_BLOCKERS are non-empty strings."""
        from pb.core.review_engine import DEBRIEF_BLOCKERS
        for b in DEBRIEF_BLOCKERS:
            assert isinstance(b, str)
            assert len(b) > 0


# ---------------------------------------------------------------------------
# get_rotating_question tests
# ---------------------------------------------------------------------------


class TestGetRotatingQuestion:
    """Tests for the rotating question logic."""

    def test_rotating_question_returns_string(self, engine):
        """get_rotating_question returns a non-empty string."""
        q = engine.get_rotating_question()
        assert isinstance(q, str)
        assert len(q) > 0

    def test_rotating_question_cycles_through_7(self, engine):
        """get_rotating_question returns different questions for different days."""
        from pb.core.review_engine import LEARNING_QUESTIONS
        results = set()
        # Iterate over 7 sequential days
        for offset in range(7):
            dt = datetime(2026, 1, offset + 1)  # Jan 1-7 cover different day_of_year values
            results.add(engine.get_rotating_question(dt))
        # Expect multiple distinct questions
        assert len(results) > 1

    def test_rotating_question_deterministic(self, engine):
        """Same date always returns same question."""
        dt = datetime(2026, 4, 25)
        q1 = engine.get_rotating_question(dt)
        q2 = engine.get_rotating_question(dt)
        assert q1 == q2

    def test_rotating_question_uses_7_mod(self, engine):
        """Questions cycle exactly every 7 days."""
        from pb.core.review_engine import LEARNING_QUESTIONS
        dt1 = datetime(2026, 1, 1)
        dt2 = datetime(2026, 1, 8)  # 7 days later
        assert engine.get_rotating_question(dt1) == engine.get_rotating_question(dt2)


# ---------------------------------------------------------------------------
# generate_daily_debrief tests
# ---------------------------------------------------------------------------


class TestGenerateDailyDebrief:
    """Tests for the 5-section debrief output generator."""

    def test_debrief_output_has_header(self, engine, full_debrief):
        """generate_daily_debrief output has a date header."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "Daily Debrief" in output

    def test_debrief_output_has_section_a(self, engine, full_debrief):
        """generate_daily_debrief output has ## A. Completion."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "## A. Completion" in output

    def test_debrief_output_has_section_b(self, engine, full_debrief):
        """generate_daily_debrief output has ## B. Friction."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "## B. Friction" in output

    def test_debrief_output_has_section_c(self, engine, full_debrief):
        """generate_daily_debrief output has ## C. Energy."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "## C. Energy" in output

    def test_debrief_output_has_section_d(self, engine, full_debrief):
        """generate_daily_debrief output has ## D. Learning."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "## D. Learning" in output

    def test_debrief_output_has_section_e(self, engine, full_debrief):
        """generate_daily_debrief output has ## E. Tomorrow."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "## E. Tomorrow" in output

    def test_debrief_output_shows_top1_status(self, engine, full_debrief):
        """generate_daily_debrief shows top1_completed value."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "yes" in output

    def test_debrief_output_shows_blocker(self, engine, full_debrief):
        """generate_daily_debrief shows biggest_blocker in readable form."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "unclear next action" in output

    def test_debrief_output_shows_energy(self, engine, full_debrief):
        """generate_daily_debrief shows energy values."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "4/5" in output  # energy_morning = 4

    def test_debrief_output_shows_learning_answer(self, engine, full_debrief):
        """generate_daily_debrief shows learning answer."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "Writing tests first" in output

    def test_debrief_output_shows_tomorrow_top1(self, engine, full_debrief):
        """generate_daily_debrief shows tomorrow's Top 1."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "Deploy review engine" in output

    def test_debrief_output_includes_completion_rate(self, engine, full_debrief):
        """generate_daily_debrief includes completion rate percentage."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "Completion:" in output

    def test_debrief_output_is_string(self, engine, full_debrief):
        """generate_daily_debrief returns a string."""
        output = engine.generate_daily_debrief(full_debrief)
        assert isinstance(output, str)

    def test_debrief_with_minimal_data(self, engine):
        """generate_daily_debrief works with only review_date set."""
        d = DailyDebrief(review_date="2026-04-25")
        output = engine.generate_daily_debrief(d)
        assert "## A. Completion" in output
        assert "## E. Tomorrow" in output

    def test_debrief_output_shows_learning_score(self, engine, full_debrief):
        """generate_daily_debrief shows LLM learning score."""
        output = engine.generate_daily_debrief(full_debrief)
        assert "8/10" in output


# ---------------------------------------------------------------------------
# Repository persistence tests
# ---------------------------------------------------------------------------


class TestDailyDebriefRepository:
    """Tests for DailyDebrief storage in repository."""

    def test_create_daily_debrief(self, repo, full_debrief):
        """create_daily_debrief persists a debrief."""
        saved = repo.create_daily_debrief(full_debrief)
        assert saved.id == full_debrief.id

    def test_get_daily_debrief_by_date(self, repo, full_debrief):
        """get_daily_debrief retrieves by review_date."""
        repo.create_daily_debrief(full_debrief)
        retrieved = repo.get_daily_debrief("2026-04-25")
        assert retrieved is not None
        assert retrieved.review_date == "2026-04-25"

    def test_get_daily_debrief_returns_none_when_missing(self, repo):
        """get_daily_debrief returns None for unknown date."""
        result = repo.get_daily_debrief("2000-01-01")
        assert result is None

    def test_create_daily_debrief_roundtrip_fields(self, repo, full_debrief):
        """Saved and retrieved debrief have identical fields."""
        repo.create_daily_debrief(full_debrief)
        retrieved = repo.get_daily_debrief("2026-04-25")
        assert retrieved.top1_completed == full_debrief.top1_completed
        assert retrieved.biggest_blocker == full_debrief.biggest_blocker
        assert retrieved.energy_morning == full_debrief.energy_morning
        assert retrieved.learning_score == full_debrief.learning_score
        assert retrieved.tomorrow_top1 == full_debrief.tomorrow_top1

    def test_create_daily_debrief_preserves_top3_list(self, repo, full_debrief):
        """Saved debrief preserves top3_completed list."""
        repo.create_daily_debrief(full_debrief)
        retrieved = repo.get_daily_debrief("2026-04-25")
        assert retrieved.top3_completed == full_debrief.top3_completed

    def test_create_daily_debrief_upserts_on_same_date(self, repo):
        """create_daily_debrief replaces existing entry for same date."""
        d1 = DailyDebrief(review_date="2026-04-25", top1_completed="no")
        d2 = DailyDebrief(review_date="2026-04-25", top1_completed="yes")
        repo.create_daily_debrief(d1)
        repo.create_daily_debrief(d2)
        retrieved = repo.get_daily_debrief("2026-04-25")
        assert retrieved.top1_completed == "yes"

    def test_list_daily_debriefs_returns_recent(self, repo):
        """list_daily_debriefs returns recent debriefs."""
        d1 = DailyDebrief(review_date="2026-04-24")
        d2 = DailyDebrief(review_date="2026-04-25")
        repo.create_daily_debrief(d1)
        repo.create_daily_debrief(d2)
        results = repo.list_daily_debriefs(days=7)
        assert len(results) == 2
