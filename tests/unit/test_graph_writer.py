"""Tests for GraphWriter — Obsidian graph note generation.

Model: tests/unit/test_packet_engine.py
Per Phase 8 D-01 through D-10.
"""

import pytest

from pb.core.graph_writer import GraphWriter, get_templates_dir, make_slug
from pb.domain.enums import ProjectStatus, ProjectType, SessionMode
from pb.domain.models import Project, Session, Task


# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_project(name="Test Project") -> Project:
    return Project(
        name=name,
        project_type=ProjectType.BUILD,
        packet_path="/unused/path.md",
        status=ProjectStatus.ACTIVE,
    )


def _make_task(title="Test Task", project_id=None) -> Task:
    return Task(title=title, project_id=project_id)


def _make_session(task_id, expectation="done", completion_pct=80, distraction=2) -> Session:
    from datetime import datetime, timedelta
    start = datetime(2026, 4, 25, 9, 0, 0)
    end = start + timedelta(minutes=45)
    return Session(
        task_id=task_id,
        mode=SessionMode.FOCUS,
        expectation=expectation,
        completion_pct=completion_pct,
        distraction=distraction,
        start_at=start,
        end_at=end,
    )


# ── Tests ─────────────────────────────────────────────────────────────────────

class TestTemplatesExist:
    """Verify that required template files exist."""

    def test_templates_directory_exists(self):
        assert get_templates_dir().is_dir()

    def test_task_note_template_exists(self):
        assert get_templates_dir().joinpath("task_note.md").is_file()

    def test_project_note_template_exists(self):
        assert get_templates_dir().joinpath("project_note.md").is_file()


class TestMakeSlug:
    """Test slug generation from task titles."""

    def test_lowercase_and_hyphens(self):
        assert make_slug("Hello World") == "hello-world"

    def test_special_chars_become_hyphens(self):
        slug = make_slug("Task: Special & Chars!")
        assert slug == "task-special-chars"

    def test_truncated_at_50(self):
        long_title = "A" * 100
        assert len(make_slug(long_title)) <= 50

    def test_no_leading_trailing_hyphens(self):
        slug = make_slug("  leading trailing  ")
        assert not slug.startswith("-")
        assert not slug.endswith("-")

    def test_empty_string(self):
        slug = make_slug("")
        assert slug == ""


class TestTaskNoteWriting:
    """Test write_task_note() creates correct files."""

    def test_task_note_file_created(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Write Tests", project_id=project.id)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, [])

        assert path is not None
        assert path.exists()

    def test_task_note_contains_type_frontmatter(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Test Task", project_id=project.id)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, [])

        content = path.read_text()
        assert "type: task" in content

    def test_task_note_contains_completion_pct(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Test Task", project_id=project.id)
        session = _make_session(task.id, completion_pct=75)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, [])

        content = path.read_text()
        assert "75" in content

    def test_task_note_contains_distraction(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Test Task", project_id=project.id)
        session = _make_session(task.id, distraction=3)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, [])

        content = path.read_text()
        assert "3" in content

    def test_task_note_next_steps_as_wikilinks(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Test Task", project_id=project.id)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, ["step one", "step two"])

        content = path.read_text()
        assert "[[step one]]" in content
        assert "[[step two]]" in content

    def test_task_note_next_steps_section_present(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Test Task", project_id=project.id)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, project, [])

        assert "## Next steps" in path.read_text()

    def test_duplicate_filename_gets_counter_suffix(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project()
        task = _make_task("Write Tests", project_id=project.id)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path1 = writer.write_task_note(session, task, project, [])
        path2 = writer.write_task_note(session, task, project, [])

        assert path1 is not None
        assert path2 is not None
        assert path1 != path2
        assert "-2" in path2.name

    def test_task_note_no_project_writes_to_tasks_dir(self, temp_dir):
        vault = temp_dir / "vault"
        task = _make_task("Orphan Task", project_id=None)
        session = _make_session(task.id)

        writer = GraphWriter(vault_path=vault)
        path = writer.write_task_note(session, task, None, [])

        assert path is not None
        assert path.exists()
        # Should be under vault/tasks/, not vault/projects/
        assert "tasks" in str(path)


class TestProjectNoteUpsert:
    """Test upsert_project_note() create/append behavior."""

    def test_project_note_created_on_first_call(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project("My Project")

        writer = GraphWriter(vault_path=vault)
        path = writer.upsert_project_note(project, "test-task", "2026-04-25")

        assert path is not None
        assert path.exists()

    def test_project_note_contains_type_frontmatter(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project("My Project")

        writer = GraphWriter(vault_path=vault)
        path = writer.upsert_project_note(project, "test-task", "2026-04-25")

        assert "type: project" in path.read_text()

    def test_project_note_contains_completed_tasks_section(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project("My Project")

        writer = GraphWriter(vault_path=vault)
        path = writer.upsert_project_note(project, "test-task", "2026-04-25")

        assert "## Completed tasks" in path.read_text()

    def test_project_note_contains_first_task_link(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project("My Project")

        writer = GraphWriter(vault_path=vault)
        path = writer.upsert_project_note(project, "first-task-slug", "2026-04-25")

        assert "[[2026-04-25-first-task-slug]]" in path.read_text()

    def test_second_call_appends_link(self, temp_dir):
        vault = temp_dir / "vault"
        project = _make_project("My Project")
        writer = GraphWriter(vault_path=vault)

        writer.upsert_project_note(project, "first-task", "2026-04-25")
        path = writer.upsert_project_note(project, "second-task", "2026-04-26")

        content = path.read_text()
        assert "[[2026-04-25-first-task]]" in content
        assert "[[2026-04-26-second-task]]" in content

    def test_append_adds_newline_before_link(self, temp_dir):
        """Pitfall 6: ensure no malformed Markdown when appending."""
        vault = temp_dir / "vault"
        project = _make_project("My Project")
        writer = GraphWriter(vault_path=vault)

        writer.upsert_project_note(project, "task-one", "2026-04-25")
        path = writer.upsert_project_note(project, "task-two", "2026-04-26")

        content = path.read_text()
        # Link must be on its own line
        assert "\n- [[2026-04-26-task-two]]" in content


class TestGracefulDegradation:
    """Vault failures must not raise exceptions."""

    def test_write_task_note_missing_vault_returns_none(self, tmp_path):
        # Point to a path that cannot be created (file in place of dir)
        blocker = tmp_path / "blocker"
        blocker.write_text("I am a file, not a directory")
        # vault/projects/ cannot be created because vault is a file
        writer = GraphWriter(vault_path=blocker)
        project = _make_project()
        task = _make_task("Test", project_id=project.id)
        session = _make_session(task.id)

        result = writer.write_task_note(session, task, project, [])

        assert result is None  # Must not raise
