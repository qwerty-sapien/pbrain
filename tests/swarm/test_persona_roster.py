"""Roster validation tests for the SWARM-02 non-learning-weighted persona library.

Asserts:
- The roster has exactly 10 personas (D-13).
- Non-learning personas >= 7 and learning personas <= total/3 (SWARM-02 ratio).
- All 7 required non-learning archetype IDs are present (D-13 roster).
- Every persona file loads without error and has non-empty win_condition + anti_goal.
"""

from pathlib import Path

import pytest

from swarm.swarm.personas import list_personas, load_persona

_PERSONAS_DIR = Path(__file__).resolve().parent.parent.parent / "personas"

_REQUIRED_NON_LEARNING_IDS = {
    "overwhelmed-planner",
    "accountability-avoider",
    "ambiguous-milestone-founder",
    "thought-dumper",
    "what-now-user",
    "chronic-drifter",
    "review-driven",
}


@pytest.fixture(scope="module")
def all_personas():
    """Load every persona from the personas/ directory."""
    paths = list_personas(_PERSONAS_DIR)
    return [load_persona(p) for p in paths]


def test_roster_ratio(all_personas):
    """SWARM-02: non_learning >= 7, learning <= total/3, total == 10."""
    total = len(all_personas)
    learning = [p for p in all_personas if p["domain_tag"] == "learning"]
    non_learning = [p for p in all_personas if p["domain_tag"] != "learning"]

    assert total == 10, (
        f"Expected exactly 10 personas (D-13), found {total}: "
        f"{[p['id'] for p in all_personas]}"
    )
    assert len(non_learning) >= 7, (
        f"SWARM-02 violation: need >= 7 non-learning personas, got {len(non_learning)}"
    )
    assert len(learning) <= total / 3, (
        f"SWARM-02 violation: learning personas ({len(learning)}) must be <= 1/3 of "
        f"total ({total}), i.e. <= {total / 3}"
    )


def test_all_personas_load(all_personas):
    """Every persona loads without error and has non-empty win_condition + anti_goal."""
    for persona in all_personas:
        pid = persona.get("id", "<missing id>")
        assert persona.get("win_condition"), (
            f"Persona '{pid}' has empty or missing win_condition"
        )
        assert persona.get("anti_goal"), (
            f"Persona '{pid}' has empty or missing anti_goal"
        )


def test_required_archetypes_present(all_personas):
    """The 7 named non-learning archetype IDs are all present in the roster."""
    loaded_ids = {p["id"] for p in all_personas}
    missing = _REQUIRED_NON_LEARNING_IDS - loaded_ids
    assert not missing, (
        f"Required non-learning archetype(s) missing from roster: {sorted(missing)}"
    )
