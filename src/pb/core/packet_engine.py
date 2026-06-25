# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Packet generation engine.

Renders Markdown packets from templates per OBSIDIAN_PACKET_SCHEMA.md.
Supports deterministic templating without LLM (INV-8).
"""

from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional

from pb.domain.models import Project, Session, Task
from pb.core.resources import read_template_text, resource
from pb.storage.config import get_vault_path


def get_templates_dir():
    """Return the installed template resource directory."""
    return resource("templates")


class PacketEngine:
    """Renders Markdown packets for projects, tasks, and reviews."""

    def __init__(self, vault_path: Optional[Path] = None):
        self.vault_path = vault_path or get_vault_path()
        self.packets_dir = self.vault_path / "pb-packets"

    def ensure_packets_dir(self) -> None:
        """Ensure the packets directory exists in the vault."""
        self.packets_dir.mkdir(parents=True, exist_ok=True)

    def render_project_packet(self, project: Project) -> str:
        """
        Render a project context packet.

        Args:
            project: Project to render

        Returns:
            Rendered Markdown content
        """
        template = Template(read_template_text("project_packet.md"))

        return template.safe_substitute(
            project_name=project.name,
            objective="[Describe the project objective]",
            track=project.track_id or "None",
            goal_arc="[Link to goal arc]",
            why_now="[Why this matters now]",
            status=project.status.value,
            last_touched=project.updated_at.strftime("%Y-%m-%d"),
            confidence="[High/Medium/Low]",
            main_risk="[Primary risk or friction point]",
            next_action="[Single concrete next action]",
            blockers="- None",
            open_loops="- None",
            repo=project.repo_path or "N/A",
            branch="main",
            files="[Key files]",
            urls="[Relevant URLs]",
            recent_decisions="- " + datetime.utcnow().strftime("%Y-%m-%d") + ": Project created",
            session_notes="- No sessions yet",
        )

    def render_review_packet(
        self,
        period: str,
        planned: str = "",
        actual: str = "",
        wins: str = "",
        slippage: str = "",
        root_causes: str = "",
        changes: str = "",
    ) -> str:
        """
        Render a review packet.

        Args:
            period: Review period (e.g., "2024-01-15" or "Week 3")
            planned: What was planned
            actual: What actually happened
            wins: Wins/accomplishments
            slippage: What slipped
            root_causes: Root causes of slippage
            changes: Changes for next period

        Returns:
            Rendered Markdown content
        """
        template = Template(read_template_text("review_packet.md"))

        return template.safe_substitute(
            period=period,
            planned=planned or "- [What was planned]",
            actual=actual or "- [What actually happened]",
            wins=wins or "- [Wins]",
            slippage=slippage or "- [What slipped]",
            root_causes=root_causes or "- [Root causes]",
            changes=changes or "- [Changes for next period]",
        )

    def write_project_packet(self, project: Project) -> Path:
        """
        Write a project packet to the vault.

        Args:
            project: Project to write packet for

        Returns:
            Path to the written packet
        """
        self.ensure_packets_dir()
        content = self.render_project_packet(project)
        safe_name = project.name.replace(" ", "_").replace("/", "_")
        path = self.packets_dir / f"project_{safe_name}.md"
        path.write_text(content)
        return path

    def write_review_packet(self, period: str, **kwargs) -> Path:
        """
        Write a review packet to the vault.

        Args:
            period: Review period
            **kwargs: Review content

        Returns:
            Path to the written packet
        """
        self.ensure_packets_dir()
        content = self.render_review_packet(period, **kwargs)
        safe_period = period.replace(" ", "_").replace("/", "_")
        path = self.packets_dir / f"review_{safe_period}.md"
        path.write_text(content)
        return path

    def get_packet_path(self, project: Project) -> Path:
        """Get the expected packet path for a project."""
        safe_name = project.name.replace(" ", "_").replace("/", "_")
        return self.packets_dir / f"project_{safe_name}.md"
