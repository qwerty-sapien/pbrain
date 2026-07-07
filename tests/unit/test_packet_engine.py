"""Tests for packet engine.

Tests deterministic packet generation (INV-8).
"""

import pytest

from pb.core.packet_engine import PacketEngine, get_templates_dir
from pb.domain.enums import ProjectStatus, ProjectType
from pb.domain.models import Project, Task


class TestTemplatesExist:
    """Verify template files exist."""

    def test_templates_directory_exists(self):
        templates_dir = get_templates_dir()
        assert templates_dir.is_dir()

    def test_project_packet_template_exists(self):
        templates_dir = get_templates_dir()
        assert templates_dir.joinpath("project_packet.md").is_file()

    def test_review_packet_template_exists(self):
        templates_dir = get_templates_dir()
        assert templates_dir.joinpath("review_packet.md").is_file()


class TestProjectPacketRendering:
    """Test project packet generation."""

    def test_render_project_packet(self, temp_config):
        engine = PacketEngine()
        project = Project(
            name="Test Project",
            project_type=ProjectType.BUILD,
            packet_path="/test/packet.md",
            status=ProjectStatus.ACTIVE,
        )

        content = engine.render_project_packet(project)

        assert "# Test Project" in content
        assert "Status: active" in content
        assert "## Objective" in content
        assert "## Next action" in content

    def test_project_packet_is_valid_markdown(self, temp_config):
        engine = PacketEngine()
        project = Project(
            name="Markdown Test",
            packet_path="/test.md",
        )

        content = engine.render_project_packet(project)

        assert content.startswith("#")
        assert "\n## " in content


# TestHandoffPacketRendering removed in Phase 8 (D-14): render_handoff_packet and write_handoff_packet removed from PacketEngine


class TestReviewPacketRendering:
    """Test review packet generation."""

    def test_render_review_packet(self, temp_config):
        engine = PacketEngine()

        content = engine.render_review_packet("2024-01-15")

        assert "# Review: 2024-01-15" in content
        assert "## Planned" in content
        assert "## Actual" in content
        assert "## Wins" in content
        assert "## Slippage" in content

    def test_review_with_custom_content(self, temp_config):
        engine = PacketEngine()

        content = engine.render_review_packet(
            "Week 3",
            planned="- Complete auth\n- Start API",
            actual="- Auth done\n- API started",
            wins="- Auth shipped",
        )

        assert "Complete auth" in content
        assert "Auth done" in content
        assert "Auth shipped" in content


class TestPacketWriting:
    """Test writing packets to disk."""

    def test_write_project_packet(self, temp_config, temp_dir):
        engine = PacketEngine(vault_path=temp_dir)
        project = Project(
            name="Write Test",
            packet_path=str(temp_dir / "project.md"),
        )

        path = engine.write_project_packet(project)

        assert path.exists()
        assert "Write Test" in path.read_text()

    def test_write_review_packet(self, temp_config, temp_dir):
        engine = PacketEngine(vault_path=temp_dir)

        path = engine.write_review_packet("2024-01-15")

        assert path.exists()
        assert "2024-01-15" in path.read_text()
