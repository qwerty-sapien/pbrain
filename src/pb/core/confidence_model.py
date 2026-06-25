# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F
"""Confidence model for concept learning state (Phase 16, D-16-17/D-16-18/D-16-27).

No imports of any pb modules at module level — pure Python stdlib only.
"""
from __future__ import annotations

from dataclasses import dataclass

# ── Thresholds (D-16-18) ────────────────────────────────────────────────────
THRESHOLD_NONE: float = 0.3   # confidence_score < THRESHOLD_NONE → "none"
THRESHOLD_FULL: float = 0.7   # confidence_score > THRESHOLD_FULL → "full"

# ── Delta constants (Claude's Discretion — tunable without migration) ────────
DELTA_DIAGNOSTIC_CORRECT: float = 0.08   # per probe question correct (max +0.24 for 3/3)
DELTA_TEACH_FULL_COVERAGE: float = 0.15  # teach session with all gaps closed
DELTA_PRACTICE_CORRECT: float = 0.10     # practice session correct answer
DELTA_WRONG: float = -0.05               # wrong answer penalty (floor 0.0)

# ── Drill-burst constant (D-16-27) ─────────────────────────────────────────
BURST_N: int = 3   # consecutive correct answers required to recover from drill-burst


@dataclass
class ConceptConfidenceRecord:
    """Typed representation of one concept_confidence table row (D-16-17)."""

    concept_id: str
    confidence_score: float = 0.0
    card_weight: float = 1.0
    next_review_at: str = ""
    last_evidence_at: str = ""
    created_at: str = ""
    updated_at: str = ""
    burst_active: int = 0    # 1 if drill-burst recovery is active (D-16-27)
    burst_streak: int = 0    # consecutive correct answers in current burst


def confidence_label(score: float) -> str:
    """Map float confidence score to visible label (D-16-18).

    Thresholds:
      score < 0.3  → "none"
      0.3 ≤ score ≤ 0.7 → "partial"
      score > 0.7  → "full"
    """
    if score < THRESHOLD_NONE:
        return "none"
    if score <= THRESHOLD_FULL:
        return "partial"
    return "full"


def clamp_score(score: float) -> float:
    """Clamp confidence_score to [0.0, 1.0]."""
    return max(0.0, min(1.0, score))
