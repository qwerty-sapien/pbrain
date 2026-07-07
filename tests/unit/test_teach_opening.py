# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.core.learning_partner import LearningPartnerSession


def test_teach_first_move_asks_learner_to_explain_before_diagnosis() -> None:
    partner = object.__new__(LearningPartnerSession)
    partner.branch = "teach"
    partner.topic = "PBFT consensus"
    partner.transcript = []
    partner._sync_session_metadata = lambda: None
    appended: list[str] = []
    partner._append_assistant_turn = appended.append

    turn = partner.open_with_first_move()

    assert turn.question_type == "free_text"
    assert "Please explain PBFT consensus in your own words" in turn.reply
    assert "choose" not in turn.reply.lower()
    assert "subtopic" not in turn.reply.lower()
    assert appended == [turn.reply]
