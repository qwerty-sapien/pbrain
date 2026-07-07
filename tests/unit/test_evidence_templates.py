"""Tests for domain_templates module -- template registry, rendering, domain resolution.

Per Phase 2 Plan 02 TEST-05.
Pattern follows tests/unit/test_session_log_writer.py.
All tests are deterministic: no LLM calls, no network.
"""

from string import Template


# -- Helpers -----------------------------------------------------------------


def _make_session(subject_scope="math", evidence_target=None):
    from pb.core.models import Session

    return Session(
        task_id="test-task-001",
        subject_scope=subject_scope,
        evidence_target=evidence_target,
    )


def _make_task(title="Test Task", domain=None):
    from pb.core.models import Task

    task = Task(title=title)
    if domain:
        task.__dict__["domain"] = domain
    return task


# -- Tests -------------------------------------------------------------------


class TestDomainTemplateRegistry:
    """TEMPLATES dict and get_template() cover all four domains."""

    def test_math_template_has_five_sub_skills(self):
        from pb.core.domain_templates import get_template

        t = get_template("math_problem_set")
        assert len(t.sub_skill_taxonomy) == 5
        assert "problem_setup" in t.sub_skill_taxonomy
        assert "verification" in t.sub_skill_taxonomy

    def test_rust_template_has_six_sub_skills(self):
        from pb.core.domain_templates import get_template

        t = get_template("rust_project")
        assert len(t.sub_skill_taxonomy) == 6
        assert "ownership_borrowing" in t.sub_skill_taxonomy

    def test_german_template_has_six_sub_skills(self):
        from pb.core.domain_templates import get_template

        t = get_template("german_speaking")
        assert len(t.sub_skill_taxonomy) == 6
        assert "conjugation_cases" in t.sub_skill_taxonomy

    def test_unknown_domain_returns_skill_kind_aware_generic_fallback(self):
        from pb.core.domain_templates import get_template

        t = get_template("unknown_domain_xyz")
        assert t.fallback is True
        assert t.name == "generic.conceptual"

    def test_all_registered_templates_have_markdown_template_file(self):
        from pb.core.domain_templates import TEMPLATES

        for name, tmpl in TEMPLATES.items():
            assert tmpl.markdown_template_file.endswith(".md"), (
                f"Template {name} missing .md extension"
            )

    def test_generic_template_is_in_templates_dict(self):
        from pb.core.domain_templates import TEMPLATES

        assert "_generic" in TEMPLATES
        assert TEMPLATES["_generic"].fallback is True


class TestMathTemplateRendering:
    """Math evidence template renders correctly."""

    def test_math_template_renders_with_safe_substitute(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        template_path = templates_dir / "evidence_math.md"
        assert template_path.exists(), f"Template not found: {template_path}"

        tmpl = Template(template_path.read_text())
        output = tmpl.safe_substitute(
            title="Test",
            date="2026-01-01",
            duration_min="45",
            sub_skills_section="- problem_setup (score: 5/5)",
            mistakes_log="None",
            critique="Good",
            retry_items_section="None",
            problem_set="Problems 1-5",
        )

        assert "## Mistakes Log" in output
        assert "## Sub-Skills Assessed" in output
        assert "math_problem_set" in output

    def test_math_template_domain_line_present(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        raw = (templates_dir / "evidence_math.md").read_text()
        assert "math_problem_set" in raw


class TestRustTemplateRendering:
    """Rust evidence template renders correctly."""

    def test_rust_template_renders_correctly(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        template_path = templates_dir / "evidence_rust.md"
        assert template_path.exists(), f"Template not found: {template_path}"

        tmpl = Template(template_path.read_text())
        output = tmpl.safe_substitute(
            title="Test",
            date="2026-01-01",
            duration_min="30",
            sub_skills_section="- ownership_borrowing (score: 4/5)",
            compiler_errors="None",
            critique="Good",
            retry_items_section="None",
            concepts_applied="Lifetimes",
        )

        assert "## Compiler Errors" in output
        assert "rust_project" in output

    def test_rust_template_domain_line_present(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        raw = (templates_dir / "evidence_rust.md").read_text()
        assert "rust_project" in raw


class TestGermanTemplateRendering:
    """German evidence template renders correctly."""

    def test_german_template_renders_correctly(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        template_path = templates_dir / "evidence_german.md"
        assert template_path.exists(), f"Template not found: {template_path}"

        tmpl = Template(template_path.read_text())
        output = tmpl.safe_substitute(
            title="Test",
            date="2026-01-01",
            duration_min="60",
            sub_skills_section="- conjugation_cases (score: 3/5)",
            phrases_attempted="Guten Morgen",
            corrections="None",
            critique="Good",
            retry_items_section="None",
        )

        assert "## Phrases Attempted" in output
        assert "german_speaking" in output

    def test_german_template_domain_line_present(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        raw = (templates_dir / "evidence_german.md").read_text()
        assert "german_speaking" in raw


class TestGenericTemplateRendering:
    """Generic evidence template renders correctly."""

    def test_generic_template_renders_correctly(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        template_path = templates_dir / "evidence_generic.md"
        assert template_path.exists(), f"Template not found: {template_path}"

        tmpl = Template(template_path.read_text())
        output = tmpl.safe_substitute(
            title="Test",
            date="2026-01-01",
            domain="general",
            duration_min="45",
            session_goal="Learn basics",
            what_practiced="Concepts A and B",
            difficulties="None",
            self_assessment="Good",
            sub_skills_section="- key_concepts_practiced (score: 4/5)",
            critique="Well done",
            retry_items_section="None",
        )

        assert "## Session Goal" in output
        assert "## What Was Practiced" in output

    def test_generic_template_has_sub_skills_section(self):
        from pathlib import Path

        templates_dir = (
            Path(__file__).parent.parent.parent / "pb" / "templates"
        )
        raw = (templates_dir / "evidence_generic.md").read_text()
        assert "sub_skills_section" in raw


class TestResolveDomain:
    """_resolve_domain() correctly extracts domain from session/task metadata."""

    def test_resolve_domain_from_subject_scope(self):
        from pb.core.domain_templates import _resolve_domain

        session = _make_session(subject_scope="calculus")
        task = _make_task()
        result = _resolve_domain(session, task)
        assert result  # non-empty
        assert isinstance(result, str)
        assert len(result) > 0

    def test_resolve_domain_falls_back_to_general(self):
        from pb.core.domain_templates import _resolve_domain

        session = _make_session(subject_scope="")
        task = _make_task(title="Some Task")  # no domain attribute
        result = _resolve_domain(session, task)
        assert result == "general"

    def test_resolve_domain_sanitizes_with_make_slug(self):
        from pb.core.domain_templates import _resolve_domain

        session = _make_session(subject_scope="German / Speaking")
        task = _make_task()
        result = _resolve_domain(session, task)
        assert "/" not in result, "Slash not sanitized from domain"
        assert " " not in result, "Space not sanitized from domain"
