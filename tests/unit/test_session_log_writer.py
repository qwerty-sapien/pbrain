"""Tests for SessionLogWriter -- session log vault note generation.

Per Phase 3 D-01, D-04, D-05, D-06, D-07 decisions.
Model: tests/unit/test_graph_writer.py
"""

from datetime import datetime, timedelta

from pb.domain.enums import ProjectStatus, ProjectType, SessionMode
from pb.domain.models import Project, Session, Task


# -- Helpers -----------------------------------------------------------------

def _make_session(
    task_id="test-task-id",
    start=None,
    end=None,
    actual_outcome="Completed the feature",
    completion_pct=80,
    distraction=2,
) -> Session:
    if start is None:
        start = datetime(2026, 4, 25, 9, 0, 0)
    if end is None:
        end = start + timedelta(minutes=45)
    return Session(
        task_id=task_id,
        mode=SessionMode.FOCUS,
        start_at=start,
        end_at=end,
        actual_outcome=actual_outcome,
        completion_pct=completion_pct,
        distraction=distraction,
    )


def _make_task(title="Fix auth bug", project_id=None, linked_track_ids=None) -> Task:
    return Task(
        title=title,
        project_id=project_id,
        linked_track_ids=linked_track_ids or [],
    )


def _make_project(name="Test Project") -> Project:
    return Project(
        name=name,
        project_type=ProjectType.BUILD,
        packet_path="/unused/path.md",
        status=ProjectStatus.ACTIVE,
    )


# -- Tests -------------------------------------------------------------------

class TestSessionLogWriterCreatesFile:
    """I-01: Session log file created at correct vault path."""

    def test_creates_session_log_in_correct_directory(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        assert path is not None
        assert "Learning/Inbox/pb/sessions" in str(path)
        assert path.exists()

    def test_filename_uses_date_and_slug(self, temp_dir):
        """D-06: filename matches {date}-{slug}.md"""
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(title="Fix auth bug")
        session = _make_session(
            task_id=task.id,
            start=datetime(2026, 4, 25, 9, 0, 0),
            end=datetime(2026, 4, 25, 9, 45, 0),
        )
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        assert path.name == "2026-04-25-fix-auth-bug.md"


class TestSessionLogCollision:
    """I-02, D-07: Collision handling appends -2, -3, etc."""

    def test_second_write_same_task_same_date_gets_suffix(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(title="Fix auth bug")
        session = _make_session(
            task_id=task.id,
            start=datetime(2026, 4, 25, 9, 0, 0),
            end=datetime(2026, 4, 25, 9, 45, 0),
        )
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path1 = writer.write_session_log(session, task, project, [])
        path2 = writer.write_session_log(session, task, project, [])

        assert path1 != path2
        assert path2.name == "2026-04-25-fix-auth-bug-2.md"


class TestSessionLogFrontmatter:
    """I-03, D-04, D-05: Frontmatter contains required fields."""

    def test_frontmatter_contains_type_session_log(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "type: session_log" in content

    def test_frontmatter_contains_task_title(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(title="Build API")
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert 'task_title: "Build API"' in content

    def test_frontmatter_contains_duration_and_times(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(
            task_id=task.id,
            start=datetime(2026, 4, 25, 9, 0, 0),
            end=datetime(2026, 4, 25, 9, 45, 0),
            completion_pct=75,
            distraction=3,
        )
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "duration_min: 45" in content
        assert "completion_pct: 75" in content
        assert "distraction: 3" in content

    def test_short_positive_session_displays_less_than_one_minute(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        start = datetime(2026, 4, 25, 9, 0, 0)
        session = _make_session(
            task_id=task.id,
            start=start,
            end=start + timedelta(seconds=20),
        )
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "duration_min: 1" in content
        assert "**Duration:** <1 min" in content

    def test_frontmatter_contains_tags_from_linked_tracks(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(linked_track_ids=["python-track"])
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "session" in content
        assert "python-track" in content


class TestSessionLogWikilinks:
    """I-04, D-04: Body contains wikilinks."""

    def test_body_contains_wikilink_to_task(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(title="Fix auth bug")
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "[[" in content and "]]" in content

    def test_body_contains_actual_outcome(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id, actual_outcome="Refactored the auth module")
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        content = path.read_text()
        assert "Refactored the auth module" in content

    def test_body_contains_next_steps_as_wikilinks(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, ["add tests", "deploy"])

        content = path.read_text()
        assert "[[add tests]]" in content
        assert "[[deploy]]" in content


class TestSessionLogEdgeCases:
    """Edge cases: missing data, special characters, vault failures."""

    def test_vault_directory_auto_created(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        # vault doesn't exist yet; writer should create it
        assert not vault.exists()

        task = _make_task()
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        assert path is not None
        assert path.exists()

    def test_unwritable_vault_returns_none(self, tmp_path):
        """I-09: vault write failure returns None without raising."""
        from pb.core.session_log_writer import SessionLogWriter

        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")

        task = _make_task()
        session = _make_session(task_id=task.id)
        project = _make_project()

        writer = SessionLogWriter(vault_path=blocker)
        result = writer.write_session_log(session, task, project, [])

        assert result is None  # must not raise

    def test_session_with_no_end_at_uses_utcnow(self, temp_dir):
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id)
        session.end_at = None  # simulate missing end_at

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, _make_project(), [])

        assert path is not None
        assert path.exists()
        # File should have today's date in the name
        today_str = datetime.utcnow().strftime("%Y-%m-%d")
        assert today_str in path.name

    def test_special_chars_in_title_produce_clean_slug(self, temp_dir):
        """Task title 'Fix: auth & bug!' produces slug 'fix-auth-bug'."""
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task(title="Fix: auth & bug!")
        session = _make_session(
            task_id=task.id,
            start=datetime(2026, 4, 25, 9, 0, 0),
            end=datetime(2026, 4, 25, 9, 45, 0),
        )
        project = _make_project()

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, project, [])

        assert "fix-auth-bug" in path.name
        # No special characters in filename
        assert ":" not in path.name
        assert "&" not in path.name
        assert "!" not in path.name

    def test_no_project_still_writes_log(self, temp_dir):
        """Session log works when project is None."""
        from pb.core.session_log_writer import SessionLogWriter

        vault = temp_dir / "vault"
        task = _make_task()
        session = _make_session(task_id=task.id)

        writer = SessionLogWriter(vault_path=vault)
        path = writer.write_session_log(session, task, None, [])

        assert path is not None
        assert path.exists()
        content = path.read_text()
        assert "type: session_log" in content
