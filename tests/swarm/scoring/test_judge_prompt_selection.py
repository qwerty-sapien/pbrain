"""Fix 2 — the judge prompt must define when selection is WRONG.

The original OUTPUT_CONTRACT declared selection=right|wrong but gave the model no
criteria for 'wrong', and the scaffold leans explicitly non-adversarial — so the
judge defaulted to 'right' even on a total routing collapse. This pins the
menu-aware wrong-selection criteria into the assembled prompt.
"""

from __future__ import annotations

from swarm.swarm.scoring.judge import build_judge_prompt


def _prompt() -> str:
    return build_judge_prompt(
        transcript="## Turn [0]\n**Brain said:** what should I do?\n```\nDo\n1. Study x\n```",
        persona_profile={
            "id": "what-now-user",
            "primary_intent": "next-action-selection",
            "win_condition": "prioritised view",
            "anti_goal": "routes into study",
        },
        facts={"total_actions": 3},
        band_anchors={"Productivity": {1: "a", 9: "b"}},
        mode_taxonomy={"study-session", "next-action-selection"},
    )


class TestSelectionCalibration:
    def test_prompt_defines_wrong_selection_criteria(self) -> None:
        low = _prompt().lower()
        assert "selection=wrong" in low
        # the four concrete wrong-selection triggers
        assert "echoes the user" in low
        assert "prioritisation" in low or "what should i do" in low
        assert "explicitly rejects" in low
        # a clean execution of the wrong mode is still a wrong selection
        assert "still selection=wrong" in low

    def test_prompt_still_contains_output_contract_and_transcript(self) -> None:
        prompt = _prompt()
        assert "Output Instructions" in prompt
        assert "<transcript>" in prompt
