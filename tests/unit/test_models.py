"""Tests for domain models."""

import re
from datetime import datetime

import pytest

from pb.domain.enums import (
    EnergyType,
    Horizon,
    PacketType,
    ProjectStatus,
    ProjectType,
    SessionMode,
    TaskState,
)
from pb.domain.models import (
    Clip,
    GoalArc,
    Packet,
    Project,
    Session,
    Task,
    TimeBlock,
    Track,
    generate_slug,
    generate_internal_id,
    utc_now,
)


class TestSlugGeneration:
    """Test slug ID generation (D-09)."""

    def test_generate_slug_format(self):
        slug = generate_slug("Fix Login Bug")
        assert re.match(r"^fix_login_bug_\d{5}$", slug), f"Bad slug: {slug}"

    def test_generate_slug_strips_special_chars(self):
        slug = generate_slug("  Hello World!!! ")
        parts = slug.rsplit("_", 1)
        assert parts[0] == "hello_world"

    def test_generate_slug_empty_fallback(self):
        slug = generate_slug("")
        assert slug.startswith("task_")

    def test_generate_slug_truncates_long_titles(self):
        slug = generate_slug("a" * 60)
        title_part = slug.rsplit("_", 1)[0]
        assert len(title_part) <= 40

    def test_generate_slug_unique(self):
        slugs = {generate_slug("Same Title") for _ in range(50)}
        assert len(slugs) > 1  # Random suffixes should differ

    def test_generate_slug_no_leading_trailing_underscores(self):
        slug = generate_slug("___test___")
        title_part = slug.rsplit("_", 1)[0]
        assert not title_part.startswith("_")
        assert not title_part.endswith("_")


class TestInternalId:
    """Test internal ID generation (D-11)."""

    def test_generate_internal_id_format(self):
        uid = generate_internal_id()
        assert len(uid) == 36  # UUID4 with dashes
        assert uid.count("-") == 4

    def test_generate_internal_id_unique(self):
        ids = {generate_internal_id() for _ in range(100)}
        assert len(ids) == 100


class TestModelIdDefaults:
    """Test that models use correct ID generators (D-09, D-11)."""

    def test_task_uses_slug_id(self):
        task = Task(title="Test Task")
        # Slug format: word_word_NNNNN
        assert "_" in task.id
        assert task.id[-5:].isdigit() or len(task.id) == 36  # slug or backward-compat

    def test_session_uses_uuid4(self):
        session = Session(task_id="test_task_12345")
        assert len(session.id) == 36
        assert session.id.count("-") == 4

    def test_goalark_uses_uuid4(self):
        goal = GoalArc(title="Learn Rust")
        assert len(goal.id) == 36

    def test_project_uses_slug_id(self):
        project = Project(name="My Project", packet_path="/tmp/p")
        assert "_" in project.id


class TestDualFormatLookup:
    """Test that prefix matching works for both ULID and slug IDs (D-10)."""

    def test_prefix_match_slug_format(self):
        """Slug-format IDs can be found by prefix."""
        task = Task(title="Fix Login Bug", id="fix_login_bug_83721")
        tasks = [task]
        matches = [t for t in tasks if t.id.startswith("fix_login")]
        assert len(matches) == 1
        assert matches[0] is task

    def test_prefix_match_ulid_format(self):
        """Existing ULID-format IDs can still be found by prefix."""
        task = Task(title="Old Task", id="01HGX5M3N4QKJR2P6YVWT8ABCD")
        tasks = [task]
        matches = [t for t in tasks if t.id.startswith("01HGX")]
        assert len(matches) == 1
        assert matches[0] is task

    def test_mixed_format_lookup(self):
        """Prefix matching works correctly in a mixed-format list."""
        slug_task = Task(title="New Task", id="new_task_12345")
        ulid_task = Task(title="Old Task", id="01HGX5M3N4QKJR2P6YVWT8ABCD")
        tasks = [slug_task, ulid_task]

        slug_matches = [t for t in tasks if t.id.startswith("new_task")]
        assert len(slug_matches) == 1
        assert slug_matches[0] is slug_task

        ulid_matches = [t for t in tasks if t.id.startswith("01HGX")]
        assert len(ulid_matches) == 1
        assert ulid_matches[0] is ulid_task


class TestTask:
    """Test Task model."""

    def test_task_creation_defaults(self):
        task = Task(title="Test")
        assert task.title == "Test"
        assert task.state == TaskState.ACTIVE
        assert task.horizon == Horizon.TODAY
        assert task.energy_type == EnergyType.DEEP
        assert task.interruption_count == 0
        assert task.id is not None
        assert task.created_at is not None

    def test_task_all_fields(self):
        task = Task(
            title="Full task",
            description="Description",
            project_id="proj123",
            horizon=Horizon.WEEK,
            state=TaskState.ACTIVE,
            estimate_minutes=60,
            energy_type=EnergyType.SHALLOW,
            linked_goal_arc_ids=["g1", "g2"],
            linked_track_ids=["t1"],
        )
        assert task.title == "Full task"
        assert task.project_id == "proj123"
        assert task.estimate_minutes == 60
        assert len(task.linked_goal_arc_ids) == 2


class TestProject:
    """Test Project model."""

    def test_project_requires_packet_path(self):
        project = Project(
            name="Test",
            packet_path="/path/to/packet.md",
        )
        assert project.packet_path == "/path/to/packet.md"

    def test_project_defaults(self):
        project = Project(name="Test", packet_path="/path.md")
        assert project.project_type == ProjectType.BUILD
        assert project.status == ProjectStatus.READY
        assert project.tags == []


class TestSession:
    """Test Session model."""

    def test_session_defaults(self):
        session = Session(task_id="task123")
        assert session.task_id == "task123"
        assert session.mode == SessionMode.FOCUS
        assert session.end_at is None
        assert session.interruption_count == 0
        assert session.llm_summary_used is False

    # Phase 8 field tests
    def test_session_phase8_fields_default_to_none(self):
        """New Phase 8 fields must default to None (RED: will fail before model update)."""
        session = Session(task_id="task123")
        assert session.expectation is None
        assert session.completion_pct is None
        assert session.distraction is None

    def test_session_phase8_fields_accept_values(self):
        """Phase 8 fields accept string and integer values."""
        session = Session(
            task_id="task123",
            expectation="finish graph writer",
            completion_pct=75,
            distraction=2,
        )
        assert session.expectation == "finish graph writer"
        assert session.completion_pct == 75
        assert session.distraction == 2


class TestTrack:
    """Test Track model."""

    def test_track_defaults(self):
        track = Track(name="German")
        assert track.name == "German"
        assert track.active is True
        assert track.priority_weight == 1.0


class TestGoalArc:
    """Test GoalArc model."""

    def test_goal_arc_defaults(self):
        goal = GoalArc(title="Learn German")
        assert goal.title == "Learn German"
        assert goal.horizon == Horizon.SIX_MONTH
        assert goal.status == "active"
