"""Tests for EvidenceWriter and index_evidence_note -- evidence note generation.

Per Phase 2 Plan 02 TEST-01.
Pattern follows tests/unit/test_session_log_writer.py.
All tests are deterministic: no LLM calls, no network, tmp_path only.
"""

from datetime import datetime, timedelta


# -- Helpers -----------------------------------------------------------------


def _make_session(
    id="test-session-001",
    task_id="test-task-001",
    start_at=None,
    end_at=None,
    branch="study",
    subject_scope="math",
    actual_outcome="done",
    goal_id=None,
):
    from pb.core.models import Session

    now = datetime.utcnow()
    return Session(
        id=id,
        task_id=task_id,
        start_at=start_at or (now - timedelta(minutes=45)),
        end_at=end_at or now,
        branch=branch,
        subject_scope=subject_scope,
        actual_outcome=actual_outcome,
        goal_id=goal_id,
    )


def _make_task(id="test-task-001", title="Linear Algebra Practice"):
    from pb.core.models import Task

    return Task(id=id, title=title, completion=100)


def _make_assessment(sub_skills=None, critique="Good work", retry_items=None):
    """Create a mock assessment result object."""

    class MockSubSkill:
        def __init__(self, name, score, is_weak, notes=""):
            self.name = name
            self.score = score
            self.is_weak = is_weak
            self.notes = notes

    class MockAssessment:
        def __init__(self, sub_skill_scores, critique, retry_items):
            self.sub_skill_scores = sub_skill_scores
            self.critique = critique
            self.retry_items = retry_items

    if sub_skills is None:
        sub_skills = [
            MockSubSkill("problem_setup", 4, False),
            MockSubSkill("execution", 2, True),
        ]
    return MockAssessment(sub_skills, critique, retry_items or [])


def _init_db(tmp_path):
    """Initialize a fresh SQLite DB in tmp_path and return the path."""
    from pb.storage.database import init_db, set_db_path

    db_path = tmp_path / "test.db"
    set_db_path(db_path)
    init_db(db_path)
    return db_path


# -- Tests -------------------------------------------------------------------


class TestEvidenceWriterCreatesFile:
    """Evidence note file creation at correct path."""

    def test_creates_evidence_in_correct_domain_directory(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        assert path.exists()
        domain_dir = tmp_path / "evidence" / "math"
        assert path.parent == domain_dir

    def test_filename_starts_with_date_string(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        now = datetime.utcnow()
        end_at = now
        start_at = now - timedelta(minutes=30)
        session = _make_session(start_at=start_at, end_at=end_at)
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        date_str = end_at.strftime("%Y-%m-%d")
        assert path is not None
        assert path.name.startswith(date_str)

    def test_creates_evidence_with_assessment(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        assessment = _make_assessment()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=assessment, domain="math")

        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "sub_skills_assessed" in content

    def test_bare_evidence_has_assessment_skipped_true(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        content = path.read_text()
        parts = content.split("---")
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["assessment_skipped"] is True

    def test_assessed_evidence_has_assessment_skipped_false(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        assessment = _make_assessment()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=assessment, domain="math")

        assert path is not None
        content = path.read_text()
        parts = content.split("---")
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["assessment_skipped"] is False

    def test_short_positive_session_displays_less_than_one_minute(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        start = datetime.utcnow()
        session = _make_session(start_at=start, end_at=start + timedelta(seconds=20))
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        content = path.read_text()
        frontmatter = yaml.safe_load(content.split("---")[1])
        assert frontmatter["duration_min"] == 1
        assert "**Duration:** <1 min" in content


class TestEvidenceWriterCollision:
    """Collision-safe path generation: appends -2, -3 suffixes."""

    def test_collision_appends_suffix(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path1 = writer.write_evidence(session, task, assessment=None, domain="math")
        path2 = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path1 is not None
        assert path2 is not None
        assert path1 != path2
        assert path2.stem.endswith("-2")

    def test_three_collisions(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        writer.write_evidence(session, task, assessment=None, domain="math")
        writer.write_evidence(session, task, assessment=None, domain="math")
        path3 = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path3 is not None
        assert path3.stem.endswith("-3")


class TestEvidenceWriterFrontmatter:
    """YAML frontmatter contains required fields and correct values."""

    def test_frontmatter_contains_required_fields(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task()
        assessment = _make_assessment()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=assessment, domain="math")

        assert path is not None
        content = path.read_text()
        parts = content.split("---")
        frontmatter = yaml.safe_load(parts[1])

        required_keys = [
            "type",
            "domain",
            "date",
            "session_id",
            "duration_min",
            "outcome",
            "template",
            "assessment_skipped",
            "sub_skills_assessed",
            "retry_items_generated",
        ]
        for key in required_keys:
            assert key in frontmatter, f"Missing frontmatter key: {key}"

    def test_frontmatter_domain_matches_input(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session(subject_scope="")
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="german_speaking")

        assert path is not None
        content = path.read_text()
        parts = content.split("---")
        frontmatter = yaml.safe_load(parts[1])
        assert frontmatter["domain"] == "german_speaking"

    def test_frontmatter_sub_skills_populated(self, tmp_path):
        import yaml

        from pb.core.evidence_writer import EvidenceWriter

        class MockSubSkill:
            def __init__(self, name, score, is_weak):
                self.name = name
                self.score = score
                self.is_weak = is_weak

        assessment = _make_assessment(
            sub_skills=[
                MockSubSkill("problem_setup", 5, False),
                MockSubSkill("execution", 2, True),
            ]
        )
        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=assessment, domain="math")

        assert path is not None
        content = path.read_text()
        parts = content.split("---")
        frontmatter = yaml.safe_load(parts[1])
        sub_skills = frontmatter["sub_skills_assessed"]
        assert len(sub_skills) == 2
        for ss in sub_skills:
            assert "name" in ss
            assert "score" in ss
            assert "weak" in ss


class TestEvidenceWriterSessionSignal:
    """Captured finish signal should be rendered into evidence bodies."""

    def test_generic_body_uses_session_fields_and_finish_checkin(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session(subject_scope="swimming", actual_outcome="Breathing is steadier but still not natural.")
        session.expectation = "Keep the stroke rhythm while breathing."
        session.observed_errors = "I still stand up to inhale after a few strokes."
        session.next_adjustment = "Drill the pull-breathe timing without kicking."
        session.generated_names = {
            "finish_checkin_qa": [
                {
                    "field": "actual_outcome",
                    "question": "What improved in this session?",
                    "answer": "I can stay horizontal slightly longer now.",
                },
                {
                    "field": "observed_errors",
                    "question": "What still breaks down?",
                    "answer": "My breathing still interrupts momentum.",
                },
            ]
        }
        task = _make_task(title="Learn swimming")
        writer = EvidenceWriter(vault_path=tmp_path)

        path = writer.write_evidence(session, task, assessment=None, domain="general")

        assert path is not None
        content = path.read_text()
        assert "Keep the stroke rhythm while breathing." in content
        assert "swimming" in content
        assert "I still stand up to inhale after a few strokes." in content
        assert "Breathing is steadier but still not natural." in content
        assert "## What Was Practised" in content
        assert "## What You Learnt" in content
        assert "## Assessment of Proficiency" in content
        assert "Drill the pull-breathe timing without kicking." in content
        assert "## Finish Check-In" in content
        assert "What improved in this session?" in content
        assert "I can stay horizontal slightly longer now." in content

    def test_domain_specific_body_uses_compact_finish_sections(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session(subject_scope="rust_project", actual_outcome="Handled the cancellation path end to end.")
        session.observed_errors = "Join errors still need a dedicated regression test."
        session.next_adjustment = "Write one failing integration test for join error handling."
        task = _make_task(title="Rust async cancellation")
        writer = EvidenceWriter(vault_path=tmp_path)

        path = writer.write_evidence(session, task, assessment=None, domain="rust_project")

        assert path is not None
        content = path.read_text()
        assert "## Session Summary" not in content
        assert "## Session Blueprint" not in content
        assert "## Evidence Observed" not in content
        assert "## What Was Practised" in content
        assert "Proved proficient in: Handled the cancellation path end to end." in content
        assert "Mistake corrected: Join errors still need a dedicated regression test." in content
        assert "## What You Learnt" in content
        assert "Write one failing integration test for join error handling." in content

    def test_compact_finish_body_has_no_placeholder_subscripts(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session(subject_scope="topology", actual_outcome="done")
        task = _make_task(title="Open sets in R^n")
        writer = EvidenceWriter(vault_path=tmp_path)

        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        content = path.read_text()
        assert "## What You Learnt" in content
        assert "- Stabilised the current work on topology." in content
        assert "_None_" not in content
        assert "_No" not in content
        assert "_none" not in content
        assert "_no" not in content


class TestEvidenceWriterEdgeCases:
    """Edge cases: unwritable paths, special characters."""

    def test_unwritable_vault_returns_none(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        # Create a file where the directory should be -- causes mkdir to fail
        blocker = tmp_path / "evidence"
        blocker.write_text("I am a file, not a directory")

        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        result = writer.write_evidence(session, task, assessment=None, domain="math")

        assert result is None  # must not raise

    def test_special_chars_in_task_title(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter

        session = _make_session()
        task = _make_task(title="Café conversation / 'test'")
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        assert path.exists()
        # make_slug should have sanitized the title -- no slashes or quotes
        assert "/" not in path.name
        assert "'" not in path.name


class TestIndexEvidenceNote:
    """SQLite write-through indexing via index_evidence_note()."""

    def test_indexes_evidence_in_sqlite(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter, index_evidence_note
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        session = _make_session()
        task = _make_task()
        assessment = _make_assessment()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=assessment, domain="math")

        assert path is not None
        index_evidence_note(session, task, assessment, path, "math")

        with get_connection() as conn:
            rows = conn.execute(
                "SELECT * FROM evidence_notes WHERE id = ?", (session.id,)
            ).fetchall()
        assert len(rows) == 1
        row = dict(rows[0])
        assert row["domain"] == "math"

    def test_index_handles_duplicate_gracefully(self, tmp_path):
        from pb.core.evidence_writer import EvidenceWriter, index_evidence_note
        from pb.storage.database import get_connection

        _init_db(tmp_path)
        session = _make_session()
        task = _make_task()
        writer = EvidenceWriter(vault_path=tmp_path)
        path = writer.write_evidence(session, task, assessment=None, domain="math")

        assert path is not None
        # Call twice with same session ID -- should use INSERT OR REPLACE, no error
        index_evidence_note(session, task, None, path, "math")
        index_evidence_note(session, task, None, path, "math")

        with get_connection() as conn:
            count = conn.execute(
                "SELECT COUNT(*) FROM evidence_notes WHERE id = ?", (session.id,)
            ).fetchone()[0]
        assert count == 1  # only one row despite two inserts
