"""Tests for business rules and invariants.

Tests locked invariants:
- INV-2: Project packet required
"""

import pytest

from pb.domain.models import Project
from pb.domain.rules import (
    RuleViolation,
    validate_project_has_packet,
)


class TestProjectPacketRequired:
    """INV-2: Every project must have a packet_path."""

    def test_valid_packet_path(self):
        project = Project(name="Test", packet_path="/path/to/packet.md")
        validate_project_has_packet(project)

    def test_empty_packet_path_rejected(self):
        project = Project(name="Test", packet_path="")
        with pytest.raises(RuleViolation) as exc:
            validate_project_has_packet(project)
        assert "requires a packet_path" in str(exc.value)

    def test_whitespace_packet_path_rejected(self):
        project = Project(name="Test", packet_path="   ")
        with pytest.raises(RuleViolation):
            validate_project_has_packet(project)
