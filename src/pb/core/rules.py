# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Business rules and invariant validation.

Implements locked invariants from the spec:
- INV-1: Single active task (enforced via active session check)
- INV-2: Project packet required
- INV-3: No note advances #new -> #learning without a linked source:socratic note
"""

from __future__ import annotations

from pathlib import Path

from pb.core.exceptions import ValidationError
from pb.core.models import Project


class RuleViolation(ValidationError):
    """Raised when a business rule is violated."""

    pass


def validate_project_has_packet(project: Project) -> None:
    """
    INV-2: Every project must have a packet_path.

    Args:
        project: Project to validate

    Raises:
        RuleViolation: If packet_path is empty
    """
    if not project.packet_path or not project.packet_path.strip():
        raise RuleViolation(
            f"Project '{project.name}' requires a packet_path. "
            "Every project must have an associated context packet."
        )


def validate_no_learning_without_socratic(note_path: Path, vault_path: Path) -> None:
    """INV-3: A note cannot transition #new -> #learning without a linked source:socratic note.

    Args:
        note_path: Vault-relative or absolute path of the note being promoted.
        vault_path: Vault root path.

    Raises:
        RuleViolation: If no linked source:socratic note is found in the graph.
    """
    try:
        from pb.vault.graph_store import has_socratic_link_for_note
    except ImportError:
        # Graph store unavailable -- skip enforcement gracefully (e.g. test envs without sqlite-vec)
        return
    try:
        has_socratic = has_socratic_link_for_note(vault_path, str(note_path))
    except Exception:
        # On any graph error, do not block (graceful fallback per existing check_promotion behaviour)
        return
    if not has_socratic:
        raise RuleViolation(
            f"INV-3: '{note_path}' cannot advance to #learning -- "
            "no linked source:socratic note found. "
            "Capture a Socratic note first with `pb note`."
        )


def validate_single_active_task(active_count: int) -> None:
    """INV-1: Only one active human focus task at a time.

    Pure function: accepts the count of currently active sessions.
    Raises RuleViolation if another session is already active.

    Args:
        active_count: Number of currently active (un-finished) sessions.

    Raises:
        RuleViolation: If active_count > 0 (a session is already running).
    """
    if active_count > 0:
        raise RuleViolation(
            f"INV-1 violated: {active_count} active session(s) already running. "
            "Only one focus task can be active at a time. "
            "Finish or pause the current session first."
        )
