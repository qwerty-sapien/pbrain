"""Tests for swarm/swarm/scoring/schema.py — ScoreRow dataclass + write_score_tsv.

Verifies:
- ScoreRow carries exactly the D-13 locked columns in order
- write_score_tsv() emits tab-delimited TSV (not JSON) that round-trips via csv.DictReader
- selection and execution are separate string columns (EVAL-05)
- severity matches P[0-3] and repro contains '#' (EVAL-04)
- TSV header matches the locked D-13 column order
"""

import csv
import re
from pathlib import Path

import pytest

from swarm.swarm.scoring.schema import (
    CONFIDENCE_VALUES,
    EXECUTION_VALUES,
    PILLARS,
    SELECTION_VALUES,
    SEVERITY_VALUES,
    TSV_COLUMNS,
    ScoreRow,
    write_score_tsv,
)

# ---------------------------------------------------------------------------
# Illustrative rows from CONTEXT.md <specifics> (LOCKED)
# ---------------------------------------------------------------------------

ROW_A = ScoreRow(
    "do/action-selector",
    "prioritise-across-commitments",
    "study-session",
    "wrong",
    "pass",
    "high",
    "Productivity",
    "P1",
    "overwhelmed-planner#a3-a6",
)

ROW_B = ScoreRow(
    "next/action-select",
    "next-action-selection",
    "next-action",
    "right",
    "partial",
    "med",
    "Goal-alignment",
    "P2",
    "what-now-user#a2",
)

LOCKED_HEADER = "input_pattern\tintent\tmode_chosen\tselection\texecution\tconfidence\tpillar\tseverity\trepro"


# ---------------------------------------------------------------------------
# TestScoreRowConstruction
# ---------------------------------------------------------------------------

class TestScoreRowConstruction:
    def test_tsv_columns_locked_order(self):
        """TSV_COLUMNS must equal the D-13 locked column list in order."""
        assert TSV_COLUMNS == [
            "input_pattern",
            "intent",
            "mode_chosen",
            "selection",
            "execution",
            "confidence",
            "pillar",
            "severity",
            "repro",
        ], TSV_COLUMNS

    def test_scorerow_has_nine_fields(self):
        """ScoreRow must expose exactly 9 fields."""
        from dataclasses import fields
        assert len(fields(ScoreRow)) == 9

    def test_all_fields_are_str(self):
        """Every ScoreRow field must be typed str (no int/float)."""
        from dataclasses import fields
        for f in fields(ScoreRow):
            assert f.type == "str", f"Field {f.name} has unexpected type {f.type}"

    def test_pillars_count_and_contents(self):
        """PILLARS must contain exactly 7 locked names."""
        assert len(PILLARS) == 7
        assert "Productivity" in PILLARS
        assert "Goal-alignment" in PILLARS
        assert "Agentic-routing" in PILLARS
        assert PILLARS[0] == "Productivity"

    def test_bounded_vocabulary_constants(self):
        """Bounded-vocabulary frozensets match spec."""
        assert SELECTION_VALUES == {"right", "wrong"}
        assert EXECUTION_VALUES == {"fail", "partial", "pass"}
        assert CONFIDENCE_VALUES == {"high", "med", "low"}
        assert SEVERITY_VALUES == {"P0", "P1", "P2", "P3"}


# ---------------------------------------------------------------------------
# TestWriteScoreTsv
# ---------------------------------------------------------------------------

class TestWriteScoreTsv:
    def test_header_is_tab_delimited_in_locked_order(self, tmp_path: Path):
        """First line of score.tsv must equal the locked tab-delimited header."""
        path = tmp_path / "score.tsv"
        write_score_tsv([ROW_A, ROW_B], path)
        first_line = path.read_text().splitlines()[0]
        assert first_line == LOCKED_HEADER

    def test_round_trip_preserves_values(self, tmp_path: Path):
        """All 9 field values must survive a write→DictReader round-trip."""
        path = tmp_path / "score.tsv"
        write_score_tsv([ROW_A, ROW_B], path)
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        assert len(rows) == 2
        assert rows[0] == ROW_A.__dict__
        assert rows[1] == ROW_B.__dict__

    def test_is_tsv_not_json(self, tmp_path: Path):
        """score.tsv must contain tabs and must not contain JSON braces."""
        path = tmp_path / "score.tsv"
        write_score_tsv([ROW_A, ROW_B], path)
        text = path.read_text()
        assert "{" not in text and "}" not in text, "JSON braces found in TSV output"
        assert "\t" in text, "No tab characters found — not TSV"

    def test_selection_and_execution_are_separate_columns(self, tmp_path: Path):
        """selection and execution must be independent columns (EVAL-05 guard)."""
        path = tmp_path / "score.tsv"
        write_score_tsv([ROW_A, ROW_B], path)
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        row_a = rows[0]
        # Values are independent
        assert row_a["selection"] == "wrong"
        assert row_a["execution"] == "pass"
        # No combined column exists
        for col in row_a.keys():
            assert not ("selection" in col and "execution" in col), (
                f"Combined selection+execution column found: {col}"
            )

    def test_severity_and_repro_present(self, tmp_path: Path):
        """Each row must have severity matching P[0-3] and repro containing '#' (EVAL-04)."""
        path = tmp_path / "score.tsv"
        write_score_tsv([ROW_A, ROW_B], path)
        with open(path, newline="") as fh:
            rows = list(csv.DictReader(fh, delimiter="\t"))
        for row in rows:
            assert row["severity"], "severity must be non-empty"
            assert re.match(r"^P[0-3]$", row["severity"]), (
                f"severity '{row['severity']}' does not match P[0-3]"
            )
            assert "#" in row["repro"], f"repro '{row['repro']}' missing '#'"
