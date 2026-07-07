"""Tests for GoalReader -- vault goal note scanning and banner generation.

Per Phase 3 D-08, D-09, D-10: GoalReader scans direction/goals/*.md,
filters by status: active and horizon in [3_month, 6_month, quarter],
generates compact goal banner for pb plan day.
"""

import pytest

from pb.core.goal_reader import GoalReader, generate_goal_banner


# -- Helpers ------------------------------------------------------------------


def _write_goal(goals_dir, filename, status="active", horizon="3_month", title="Goal"):
    """Write a minimal goal note with YAML frontmatter."""
    goals_dir.mkdir(parents=True, exist_ok=True)
    content = f"---\nstatus: {status}\nhorizon: {horizon}\ntitle: {title}\n---\n\n# {title}\n"
    (goals_dir / filename).write_text(content)


# -- GoalReader filtering tests -----------------------------------------------


class TestGoalReaderFiltering:
    """Test read_active_goals() filtering by status and horizon."""

    def test_returns_active_3month_goals(self, temp_dir):
        """I-05, D-09: active + 3_month horizon included."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "ship-feature.md", status="active", horizon="3_month", title="Ship Feature")

        reader = GoalReader(vault_path=vault)
        goals = reader.read_active_goals()
        assert len(goals) == 1
        assert goals[0]["title"] == "Ship Feature"

    def test_returns_active_6month_goals(self, temp_dir):
        """I-05, D-09: active + 6_month horizon included."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "long-goal.md", status="active", horizon="6_month", title="Long Goal")

        reader = GoalReader(vault_path=vault)
        goals = reader.read_active_goals()
        assert len(goals) == 1
        assert goals[0]["title"] == "Long Goal"

    def test_returns_active_six_month_goals(self, temp_dir):
        """Horizon alias: six_month should also match."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "alias-goal.md", status="active", horizon="six_month", title="Alias Goal")

        reader = GoalReader(vault_path=vault)
        goals = reader.read_active_goals()
        assert len(goals) == 1
        assert goals[0]["title"] == "Alias Goal"

    def test_excludes_inactive_goals(self, temp_dir):
        """I-05: inactive goals not shown."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "old-goal.md", status="inactive", horizon="3_month", title="Old Goal")

        reader = GoalReader(vault_path=vault)
        assert reader.read_active_goals() == []

    def test_excludes_annual_horizon_goals(self, temp_dir):
        """I-05, D-09: annual horizon out of daily planning window."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "annual.md", status="active", horizon="annual", title="Annual")

        reader = GoalReader(vault_path=vault)
        assert reader.read_active_goals() == []

    def test_excludes_month_horizon_goals(self, temp_dir):
        """D-09: month horizon too short for daily goal banner (only 3/6 month)."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "short.md", status="active", horizon="month", title="Short")

        reader = GoalReader(vault_path=vault)
        assert reader.read_active_goals() == []


class TestHorizonNormalization:
    """Test horizon string normalization accepts common variants."""

    @pytest.mark.parametrize("horizon", [
        "three_month", "3-month", "3_month", "3month",
    ])
    def test_3month_variants_accepted(self, temp_dir, horizon):
        """Pitfall 2: various 3-month formats all accepted."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "goal.md", status="active", horizon=horizon, title="Goal")

        reader = GoalReader(vault_path=vault)
        assert len(reader.read_active_goals()) == 1

    @pytest.mark.parametrize("horizon", [
        "six_month", "6-month", "6_month", "6month",
    ])
    def test_6month_variants_accepted(self, temp_dir, horizon):
        """Pitfall 2: various 6-month formats all accepted."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "goal.md", status="active", horizon=horizon, title="Goal")

        reader = GoalReader(vault_path=vault)
        assert len(reader.read_active_goals()) == 1

    def test_quarter_horizon_accepted(self, temp_dir):
        """Quarter maps to ~3 months, should be in daily horizons."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        _write_goal(goals_dir, "quarterly.md", status="active", horizon="quarter", title="Quarterly")

        reader = GoalReader(vault_path=vault)
        assert len(reader.read_active_goals()) == 1


class TestGoalReaderEdgeCases:
    """Test graceful handling of missing dirs and malformed data."""

    def test_missing_goals_dir_returns_empty(self, temp_dir):
        """D-08: graceful when goals folder does not exist."""
        reader = GoalReader(vault_path=temp_dir / "empty-vault")
        assert reader.read_active_goals() == []

    def test_malformed_frontmatter_skipped(self, temp_dir):
        """Resilience: bad YAML in one file does not crash reader; valid files still returned."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        goals_dir.mkdir(parents=True)
        (goals_dir / "bad.md").write_text("---\n: invalid: yaml: {{{\n---\n")
        _write_goal(goals_dir, "good.md", status="active", horizon="3_month", title="Good")

        reader = GoalReader(vault_path=vault)
        goals = reader.read_active_goals()
        assert len(goals) == 1
        assert goals[0]["title"] == "Good"

    def test_missing_horizon_field_excluded(self, temp_dir):
        """Goal without horizon field is excluded (not matching)."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        goals_dir.mkdir(parents=True)
        (goals_dir / "nohorizon.md").write_text(
            "---\nstatus: active\ntitle: No Horizon\n---\n"
        )

        reader = GoalReader(vault_path=vault)
        assert reader.read_active_goals() == []

    def test_file_without_frontmatter_skipped(self, temp_dir):
        """File with no --- frontmatter delimiter returns empty dict, excluded."""
        vault = temp_dir / "vault"
        goals_dir = vault / "direction" / "goals"
        goals_dir.mkdir(parents=True)
        (goals_dir / "plain.md").write_text("Just a plain markdown file.\n")

        reader = GoalReader(vault_path=vault)
        assert reader.read_active_goals() == []


# -- Banner generation tests --------------------------------------------------


class TestGoalBanner:
    """Test generate_goal_banner() output formatting."""

    def test_banner_with_goals_has_header(self, temp_dir):
        """D-10: banner starts with ## Active Goals."""
        goals = [
            {"title": "Ship MVP", "horizon": "3_month"},
            {"title": "Learn Rust", "horizon": "6_month"},
        ]
        banner = generate_goal_banner(goals)
        assert "## Active Goals" in banner

    def test_banner_with_goals_shows_titles(self, temp_dir):
        """D-10: each goal title listed."""
        goals = [
            {"title": "Ship MVP", "horizon": "3_month"},
            {"title": "Learn Rust", "horizon": "6_month"},
        ]
        banner = generate_goal_banner(goals)
        assert "Ship MVP" in banner
        assert "Learn Rust" in banner

    def test_banner_shows_horizon_labels(self, temp_dir):
        """D-10: horizon label shown in brackets for each goal."""
        goals = [
            {"title": "Ship MVP", "horizon": "3_month"},
        ]
        banner = generate_goal_banner(goals)
        assert "[3_month]" in banner

    def test_banner_empty_list_returns_empty_string(self, temp_dir):
        """D-10: empty goals list returns empty string, not header."""
        banner = generate_goal_banner([])
        assert banner == ""

    def test_banner_handles_missing_title(self, temp_dir):
        """Edge: goal dict without title shows Untitled."""
        goals = [{"horizon": "3_month"}]
        banner = generate_goal_banner(goals)
        assert "Untitled" in banner

    def test_banner_handles_missing_horizon(self, temp_dir):
        """Edge: goal dict without horizon shows empty bracket."""
        goals = [{"title": "NoHorizon"}]
        banner = generate_goal_banner(goals)
        assert "NoHorizon" in banner
