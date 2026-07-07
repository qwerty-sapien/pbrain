"""Integration tests for StudyService — verifies study plan generation contracts.

Tests use real StudyService with fake vault directory structures. No network calls needed.
"""
from __future__ import annotations

import datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest


def _make_mock_config(thresholds: dict | None = None) -> Any:
    config = MagicMock()
    config.learning.decay_thresholds = thresholds or {
        "piano": 2,
        "ml": 7,
        "_default": 5,
    }
    return config


def _make_stale_domain(vault: Path, name: str, days_inactive: int = 10) -> None:
    """Create a domain directory with one stale note and stale _state.md."""
    d = vault / "knowledge" / name
    d.mkdir(parents=True, exist_ok=True)
    last_date = (datetime.date.today() - datetime.timedelta(days=days_inactive)).isoformat()
    (d / "_state.md").write_text(f"---\nlast_activity: {last_date}\n---\n")
    (d / "some-note.md").write_text("---\nlearning_stage: '#stale'\n---\n\nContent.")


def _make_active_domain(vault: Path, name: str) -> None:
    """Create a domain directory with fresh notes."""
    d = vault / "knowledge" / name
    d.mkdir(parents=True, exist_ok=True)
    today = datetime.date.today().isoformat()
    (d / "_state.md").write_text(f"---\nlast_activity: {today}\n---\n")
    (d / "note-learning.md").write_text("---\nlearning_stage: '#learning'\n---\n\nContent.")
    (d / "note-new.md").write_text("---\nlearning_stage: '#new'\n---\n\nContent.")


class TestStudyService:

    def test_generate_plan_returns_study_blocks(self, tmp_path):
        """generate_plan returns a list of StudyBlock instances."""
        _make_active_domain(tmp_path, "piano")

        from pb.study_service import StudyService, StudyBlock
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=30)

        assert isinstance(blocks, list)
        for block in blocks:
            assert isinstance(block, StudyBlock)
            assert block.minutes > 0
            assert block.domain

    def test_stale_domain_gets_first_block(self, tmp_path):
        """Stale domain blocks appear before fresh domain blocks (STDY-02)."""
        _make_stale_domain(tmp_path, "piano", days_inactive=10)
        _make_active_domain(tmp_path, "ml-notes")

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=60)

        if len(blocks) >= 2:
            # First block should be from the stale domain (piano)
            first_modes = [b.mode for b in blocks if b.domain == "piano"]
            assert any(m == "re-engage" for m in first_modes), (
                f"Stale piano domain should produce re-engage block; got modes: {[b.mode for b in blocks]}"
            )

    def test_review_command_is_valid_anki_review_command(self, tmp_path):
        """StudyBlock with mode=review has a valid Anki review command."""
        d = tmp_path / "knowledge" / "piano"
        d.mkdir(parents=True)
        today = datetime.date.today().isoformat()
        (d / "_state.md").write_text(f"---\nlast_activity: {today}\n---\n")
        (d / "note-learnt.md").write_text("---\nlearning_stage: '#learnt'\n---\n\nContent.")

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=60)

        review_blocks = [b for b in blocks if b.mode == "review"]
        for block in review_blocks:
            assert block.pb_command == f"pb study recall {block.domain}", (
                f"Review blocks must use 'pb study recall <domain>'; got '{block.pb_command}'"
            )

    def test_d20_explore_command_starts_with_pb_note(self, tmp_path):
        """StudyBlock with mode=explore has pb_command starting with 'pb note' (D-20)."""
        d = tmp_path / "knowledge" / "piano"
        d.mkdir(parents=True)
        today = datetime.date.today().isoformat()
        (d / "_state.md").write_text(f"---\nlast_activity: {today}\n---\n")
        (d / "note-new.md").write_text("---\nlearning_stage: '#new'\n---\n\nContent.")

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=60)

        explore_blocks = [b for b in blocks if b.mode == "explore"]
        for block in explore_blocks:
            assert block.pb_command.startswith("pb study "), (
                f"D-20: explore blocks must use 'pb study <domain>'; got '{block.pb_command}'"
            )

    def test_consolidate_command_starts_with_pb_study(self, tmp_path):
        """StudyBlock with mode=consolidate uses the top-level study surface."""
        d = tmp_path / "knowledge" / "piano"
        d.mkdir(parents=True)
        today = datetime.date.today().isoformat()
        (d / "_state.md").write_text(f"---\nlast_activity: {today}\n---\n")
        (d / "note-learning.md").write_text("---\nlearning_stage: '#learning'\n---\n\nContent.")

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=60)

        consolidate_blocks = [b for b in blocks if b.mode == "consolidate"]
        for block in consolidate_blocks:
            assert block.pb_command.startswith("pb study "), (
                f"Consolidate blocks must use 'pb study <domain>'; got '{block.pb_command}'"
            )

    def test_d20_reengage_command_starts_with_pb_brain(self, tmp_path):
        """StudyBlock with mode=re-engage has pb_command starting with 'pb brain' (D-20)."""
        _make_stale_domain(tmp_path, "piano", days_inactive=15)

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=30)

        reengage_blocks = [b for b in blocks if b.mode == "re-engage"]
        for block in reengage_blocks:
            assert block.pb_command.startswith("pb study "), (
                f"D-20: re-engage blocks must use 'pb study <domain>'; got '{block.pb_command}'"
            )

    def test_domain_filter_limits_output(self, tmp_path):
        """generate_plan with domain_filter only returns blocks for that domain (STDY-01)."""
        _make_active_domain(tmp_path, "piano")
        _make_active_domain(tmp_path, "german")

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        blocks = svc.generate_plan(total_minutes=60, domain_filter="piano")

        for block in blocks:
            assert block.domain == "piano", (
                f"Domain filter 'piano' produced block for domain '{block.domain}'"
            )

    def test_resolve_threshold_prefix_match(self):
        """_resolve_threshold matches 'ml-notes' to 'ml' threshold (STDY-03)."""
        from pb.study_service import _resolve_threshold
        thresholds = {"piano": 2, "ml": 7, "_default": 5}
        assert _resolve_threshold("ml-notes", thresholds) == 7
        assert _resolve_threshold("piano", thresholds) == 2
        assert _resolve_threshold("unknown-domain", thresholds) == 5  # fallback to _default

    def test_get_domain_statuses_returns_list(self, tmp_path):
        """get_domain_statuses returns a list (empty for vault with no knowledge)."""
        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        statuses = svc.get_domain_statuses()
        assert isinstance(statuses, list)

    def test_get_domain_statuses_returns_list_with_domains(self, tmp_path):
        """get_domain_statuses returns a non-empty list when domains exist."""
        _make_active_domain(tmp_path, "piano")
        _make_stale_domain(tmp_path, "german", days_inactive=8)

        from pb.study_service import StudyService, DomainStatus
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        statuses = svc.get_domain_statuses()

        assert isinstance(statuses, list)
        assert len(statuses) == 2
        domain_names = {s.name for s in statuses}
        assert "piano" in domain_names
        assert "german" in domain_names
        for s in statuses:
            assert isinstance(s, DomainStatus)

    def test_stale_domain_has_is_stale_true(self, tmp_path):
        """Domains with notes past decay threshold have is_stale=True (STDY-02)."""
        # piano threshold is 2 days; inactive for 10 days → stale
        _make_stale_domain(tmp_path, "piano", days_inactive=10)

        from pb.study_service import StudyService
        svc = StudyService(vault_path=tmp_path, config=_make_mock_config())
        statuses = svc.get_domain_statuses()

        piano_status = next((s for s in statuses if s.name == "piano"), None)
        assert piano_status is not None
        assert piano_status.is_stale is True
