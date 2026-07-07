"""Unit tests for _state.md update and staleness checks (Phase 17 plan 05).

Tests cover:
- test_update_state_md_creates_from_stub: basic write with session summary and stage counts
- test_update_state_md_rotates_summaries: only last 3 summaries kept (D-05)
- test_update_state_md_30_line_cap: output is <= 30 lines
- test_check_staleness_flags_old_learnt: #learnt + old stage_updated -> #stale
- test_check_staleness_ignores_fresh_learnt: #learnt + today -> not flagged
- test_suggest_learnt_promotions_finds_candidates: #learning + weight >= 10 -> candidate
- test_suggest_learnt_promotions_ignores_below_threshold: #learning + weight < 10 -> not returned
"""

from __future__ import annotations

import datetime
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_writer(vault_path: Path):
    """Create GraphWriter with a given vault_path (avoids get_vault_path() call)."""
    from pb.core.graph_writer import GraphWriter
    return GraphWriter(vault_path=vault_path)


def _make_domain(tmp_path: Path, name: str = "german") -> Path:
    """Create a minimal knowledge domain directory with a _state.md stub."""
    knowledge = tmp_path / "knowledge"
    knowledge.mkdir(parents=True, exist_ok=True)
    domain = knowledge / name
    domain.mkdir(exist_ok=True)
    (domain / "_state.md").write_text(
        "---\ntype: domain_state\nupdated: 2026-05-01\nstage_counts:\n  new: 0\n"
        "session_summaries: []\n---\n\n# german\n"
    )
    return domain


def _write_note(domain: Path, name: str, stage: str, stage_updated: str = "") -> Path:
    """Write a note with the given learning_stage to a domain dir."""
    fm: dict = {"learning_stage": stage}
    if stage_updated:
        fm["stage_updated"] = stage_updated
    yaml_text = yaml.dump(fm, default_flow_style=False, allow_unicode=True)
    p = domain / name
    p.write_text(f"---\n{yaml_text}---\n\n# Note\n")
    return p


# ---------------------------------------------------------------------------
# _update_state_md tests
# ---------------------------------------------------------------------------


class TestUpdateStateMd:
    def test_update_state_md_creates_from_stub(self, tmp_path):
        """_update_state_md writes session summary and stage counts to _state.md."""
        domain = _make_domain(tmp_path)
        _write_note(domain, "vocab.md", "#new")

        writer = _make_writer(tmp_path)
        result = writer.update_state_md(domain, "Practiced vocab for 30 min")

        assert result is not None, "Expected path back"
        assert result.exists(), "_state.md should exist"

        content = result.read_text()
        assert "---" in content
        parts = content.split("---", 2)
        fm = yaml.safe_load(parts[1])
        body = parts[2] if len(parts) > 2 else ""

        assert fm["type"] == "domain_state"
        assert "stage_counts" in fm
        assert fm["stage_counts"]["new"] == 1
        assert "session_summaries" in fm
        assert "Practiced vocab for 30 min" in fm["session_summaries"]
        assert "german" in content

    def test_update_state_md_rotates_summaries(self, tmp_path):
        """After 5 calls, only the last 3 session summaries are retained (D-05)."""
        domain = _make_domain(tmp_path)
        writer = _make_writer(tmp_path)

        summaries = [f"Session {i}" for i in range(1, 6)]
        for s in summaries:
            writer.update_state_md(domain, s)

        state_path = domain / "_state.md"
        content = state_path.read_text()
        parts = content.split("---", 2)
        fm = yaml.safe_load(parts[1])

        assert len(fm["session_summaries"]) == 3, "Only last 3 summaries should be kept"
        assert fm["session_summaries"] == ["Session 3", "Session 4", "Session 5"]

    def test_update_state_md_30_line_cap(self, tmp_path):
        """_state.md output never exceeds 30 lines (D-05)."""
        domain = _make_domain(tmp_path)
        # Add several notes to inflate stage counts
        for i in range(5):
            _write_note(domain, f"note{i}.md", "#new")

        writer = _make_writer(tmp_path)
        # Write 3 max-length summaries
        for i in range(3):
            long_summary = f"Session {i}: " + "x" * 60
            writer.update_state_md(domain, long_summary[:80])

        state_path = domain / "_state.md"
        lines = state_path.read_text().splitlines()
        assert len(lines) <= 30, f"_state.md has {len(lines)} lines, expected <= 30"


# ---------------------------------------------------------------------------
# check_staleness tests
# ---------------------------------------------------------------------------


class TestCheckStaleness:
    def test_check_staleness_flags_old_learnt(self, tmp_path, temp_config):
        """#learnt note with stage_updated 10 days ago is promoted to #stale."""
        domain = _make_domain(tmp_path)
        # Use the vault root as vault_path
        vault_path = temp_config.general.vault_path
        vault = Path(vault_path)
        knowledge = vault / "knowledge"
        knowledge.mkdir(parents=True, exist_ok=True)
        test_domain = knowledge / "german"
        test_domain.mkdir(exist_ok=True)

        old_date = (datetime.date.today() - datetime.timedelta(days=10)).isoformat()
        note = _write_note(test_domain, "vocab.md", "#learnt", stage_updated=old_date)

        from pb.vault.lifecycle import check_staleness
        stale = check_staleness(vault, test_domain, decay_days=7)

        assert len(stale) == 1, f"Expected 1 stale note, got {stale}"
        assert "stale" in stale[0].lower() or "#stale" in stale[0]

        # Verify frontmatter was updated
        import yaml as _yaml
        content = note.read_text()
        parts = content.split("---", 2)
        fm = _yaml.safe_load(parts[1])
        assert fm["learning_stage"] == "#stale"

    def test_check_staleness_ignores_fresh_learnt(self, tmp_path, temp_config):
        """#learnt note with stage_updated today is NOT flagged stale."""
        vault_path = temp_config.general.vault_path
        vault = Path(vault_path)
        knowledge = vault / "knowledge"
        knowledge.mkdir(parents=True, exist_ok=True)
        test_domain = knowledge / "french"
        test_domain.mkdir(exist_ok=True)

        today = datetime.date.today().isoformat()
        _write_note(test_domain, "verbs.md", "#learnt", stage_updated=today)

        from pb.vault.lifecycle import check_staleness
        stale = check_staleness(vault, test_domain, decay_days=7)

        assert stale == [], f"Fresh #learnt note should not be stale, got {stale}"


# ---------------------------------------------------------------------------
# suggest_learnt_promotions tests
# ---------------------------------------------------------------------------


class TestSuggestLearntPromotions:
    def test_suggest_learnt_promotions_finds_candidates(self, tmp_path, temp_db):
        """#learning note with weighted total >= threshold is returned as candidate."""
        vault = tmp_path / "vault"
        vault.mkdir()
        domain = vault / "knowledge" / "python"
        domain.mkdir(parents=True)

        note = _write_note(domain, "decorators.md", "#learning")
        rel_path = str(note.relative_to(vault))

        # Insert enough interactions to exceed the default threshold (10.0)
        from pb.vault.lifecycle import log_interaction
        # "study" weight = 3.0 by default; need >= 10.0 total
        for _ in range(4):
            log_interaction(rel_path, "study")  # 4 * 3.0 = 12.0

        from pb.core.graph_writer import suggest_learnt_promotions
        candidates = suggest_learnt_promotions(vault)

        assert rel_path in candidates, f"Expected {rel_path} in candidates, got {candidates}"

    def test_suggest_learnt_promotions_ignores_below_threshold(self, tmp_path, temp_db):
        """#learning note with weighted total < threshold is NOT returned."""
        vault = tmp_path / "vault"
        vault.mkdir()
        domain = vault / "knowledge" / "rust"
        domain.mkdir(parents=True)

        note = _write_note(domain, "ownership.md", "#learning")
        rel_path = str(note.relative_to(vault))

        # Only 1 interaction — well below threshold of 10.0
        from pb.vault.lifecycle import log_interaction
        log_interaction(rel_path, "read")  # weight 1.0

        from pb.core.graph_writer import suggest_learnt_promotions
        candidates = suggest_learnt_promotions(vault)

        assert rel_path not in candidates, f"{rel_path} should not be a candidate"
