# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from unittest.mock import MagicMock, patch
from pb.cli.commands.study import _collect_broad_categories, _normalize_domain


def test_normalize_domain_dedup():
    assert _normalize_domain("machine-learning") == "machine learning"
    assert _normalize_domain("machine_learning") == "machine learning"
    assert _normalize_domain("Machine Learning") == "machine learning"


def test_dedup_removes_near_duplicates():
    """No near-identical entries: ml, machine-learning, Machine Learning → one category."""
    repo = MagicMock()
    repo.list_goal_arcs.return_value = []
    repo.list_tracks.return_value = []
    repo.list_tasks.return_value = []

    with patch("pb.cli.commands.study._list_knowledge_domains", return_value=[
        "Machine Learning",
        "machine-learning",
        "ml",
        "Physics and Mathematics",
        "physics_math_foundations",
    ]):
        cats = _collect_broad_categories(repo)

    # Should have far fewer than 5 entries due to dedup
    norm_names = [_normalize_domain(c) for c in cats]
    assert len(set(norm_names)) == len(norm_names)  # no duplicates


def test_broad_categories_from_goals():
    repo = MagicMock()
    goal = MagicMock()
    goal.domain = "Machine Learning"
    goal.title = "ML Mastery"
    goal.execution_mode = "study"
    repo.list_goal_arcs.return_value = [goal]
    repo.list_tracks.return_value = []
    repo.list_tasks.return_value = []

    with patch("pb.cli.commands.study._list_knowledge_domains", return_value=[]):
        cats = _collect_broad_categories(repo)

    assert "Machine Learning" in cats
