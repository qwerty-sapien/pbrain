"""Unit tests for review response repository methods."""

import pytest
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

from pb.domain.models import DailyReviewResponse
from pb.storage.repository import Repository
from pb.storage import database


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as td:
        db_path = Path(td) / "test.db"
        database.set_db_path(db_path)
        database.init_db()
        yield db_path
        database._db_path = None  # Reset


@pytest.fixture
def repo(temp_db):
    """Create a repository with temp database."""
    return Repository()


class TestCreateReviewResponse:
    """Tests for create_review_response method."""

    def test_creates_new_response(self, repo):
        """Should create a new review response."""
        response = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=7,
        )

        result = repo.create_review_response(response)

        assert result.id == response.id
        assert result.numeric_score == 7

    def test_upserts_on_conflict(self, repo):
        """Should update existing response for same date+question."""
        response1 = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=5,
        )
        response2 = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=8,
        )

        repo.create_review_response(response1)
        repo.create_review_response(response2)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 1
        assert responses[0].numeric_score == 8  # Updated value

    def test_stores_text_response_and_rationale(self, repo):
        """Should store optional text response and LLM rationale."""
        response = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="blockers",
            numeric_score=6,
            text_response="Too many meetings",
            llm_rationale="Clear identification of blocker",
        )

        repo.create_review_response(response)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 1
        assert responses[0].text_response == "Too many meetings"
        assert responses[0].llm_rationale == "Clear identification of blocker"

    def test_allows_multiple_questions_same_date(self, repo):
        """Should allow different questions for the same date."""
        response1 = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=7,
        )
        response2 = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="presence",
            numeric_score=8,
        )

        repo.create_review_response(response1)
        repo.create_review_response(response2)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 2

    def test_preserves_null_text_fields(self, repo):
        """Should preserve null text_response and llm_rationale."""
        response = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="alignment",
            numeric_score=9,
            text_response=None,
            llm_rationale=None,
        )

        repo.create_review_response(response)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 1
        assert responses[0].text_response is None
        assert responses[0].llm_rationale is None


class TestGetReviewResponsesForDate:
    """Tests for get_review_responses_for_date method."""

    def test_returns_empty_list_when_no_responses(self, repo):
        """Should return empty list when no responses for date."""
        responses = repo.get_review_responses_for_date("2026-04-24")
        assert responses == []

    def test_returns_all_responses_for_date(self, repo):
        """Should return all responses for the given date."""
        for q_id in ["energy", "presence", "blockers"]:
            response = DailyReviewResponse(
                review_date="2026-04-24",
                question_id=q_id,
                numeric_score=7,
            )
            repo.create_review_response(response)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 3

    def test_does_not_return_other_dates(self, repo):
        """Should not return responses from other dates."""
        response1 = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=7,
        )
        response2 = DailyReviewResponse(
            review_date="2026-04-25",
            question_id="energy",
            numeric_score=8,
        )

        repo.create_review_response(response1)
        repo.create_review_response(response2)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 1
        assert responses[0].numeric_score == 7

    def test_orders_by_created_at(self, repo):
        """Should return responses ordered by created_at."""
        # Create in reverse order to test sorting
        for i, q_id in enumerate(["energy", "presence", "blockers"]):
            response = DailyReviewResponse(
                review_date="2026-04-24",
                question_id=q_id,
                numeric_score=i + 5,
            )
            repo.create_review_response(response)

        responses = repo.get_review_responses_for_date("2026-04-24")
        assert len(responses) == 3
        # Verify all question types are present
        question_ids = [r.question_id for r in responses]
        assert "energy" in question_ids
        assert "presence" in question_ids
        assert "blockers" in question_ids


class TestGetYesterdayResponse:
    """Tests for get_yesterday_response method."""

    def test_returns_none_when_no_yesterday_data(self, repo):
        """Should return None when no response from yesterday."""
        result = repo.get_yesterday_response("energy", "2026-04-24")
        assert result is None

    def test_returns_yesterday_response(self, repo):
        """Should return yesterday's response for trend calculation."""
        yesterday_response = DailyReviewResponse(
            review_date="2026-04-23",
            question_id="blockers",
            numeric_score=5,
        )
        repo.create_review_response(yesterday_response)

        result = repo.get_yesterday_response("blockers", "2026-04-24")
        assert result is not None
        assert result.numeric_score == 5

    def test_returns_none_for_wrong_question(self, repo):
        """Should return None when question ID doesn't match."""
        yesterday_response = DailyReviewResponse(
            review_date="2026-04-23",
            question_id="energy",
            numeric_score=7,
        )
        repo.create_review_response(yesterday_response)

        result = repo.get_yesterday_response("blockers", "2026-04-24")
        assert result is None

    def test_returns_none_for_two_days_ago(self, repo):
        """Should not return response from two days ago."""
        two_days_ago = DailyReviewResponse(
            review_date="2026-04-22",
            question_id="energy",
            numeric_score=6,
        )
        repo.create_review_response(two_days_ago)

        result = repo.get_yesterday_response("energy", "2026-04-24")
        assert result is None

    def test_handles_date_format_correctly(self, repo):
        """Should handle ISO date format correctly."""
        yesterday_response = DailyReviewResponse(
            review_date="2026-04-23",
            question_id="presence",
            numeric_score=8,
            text_response="Good focus",
        )
        repo.create_review_response(yesterday_response)

        result = repo.get_yesterday_response("presence", "2026-04-24")
        assert result is not None
        assert result.text_response == "Good focus"


class TestDailyReviewResponseModel:
    """Tests for DailyReviewResponse model."""

    def test_generates_uuid4_for_id(self):
        """Should generate UUID4 for id field."""
        response = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=7,
        )
        assert response.id is not None
        assert len(response.id) == 36  # UUID4 length

    def test_sets_created_at_to_now(self):
        """Should set created_at to current time."""
        before = datetime.utcnow()
        response = DailyReviewResponse(
            review_date="2026-04-24",
            question_id="energy",
            numeric_score=7,
        )
        after = datetime.utcnow()

        assert before <= response.created_at <= after

    def test_accepts_all_question_ids(self):
        """Should accept all standard question IDs."""
        for q_id in ["energy", "presence", "best_window", "blockers", "alignment"]:
            response = DailyReviewResponse(
                review_date="2026-04-24",
                question_id=q_id,
                numeric_score=5,
            )
            assert response.question_id == q_id

    def test_accepts_score_range(self):
        """Should accept scores in 1-10 range."""
        for score in [1, 5, 10]:
            response = DailyReviewResponse(
                review_date="2026-04-24",
                question_id="energy",
                numeric_score=score,
            )
            assert response.numeric_score == score
