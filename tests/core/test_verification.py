# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from pb.core.verification import compute_spillover_weight, estimate_quiz_duration


def test_no_spillover_above_95():
    assert compute_spillover_weight(96, 97) == 0.0


def test_spillover_below_95():
    weight = compute_spillover_weight(80, 70)
    assert weight > 0.0
    assert weight <= 1.0


def test_quiz_duration_scales():
    from unittest.mock import MagicMock
    repo = MagicMock()
    assert estimate_quiz_duration(repo, ["t1"]) == 3
    assert estimate_quiz_duration(repo, ["t1", "t2", "t3"]) == 7
    assert estimate_quiz_duration(repo, ["t1", "t2", "t3", "t4", "t5"]) == 15
