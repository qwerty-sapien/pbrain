"""pb do must never echo a conversational request back as a 'Study <request>' task.

The canonical swarm run showed pb echoing the user's own words into a Study/Practise
menu title for 9/10 personas (e.g. "Study I only have five minutes, just give me the
single most important one."). The fallback that routed *any* unmatched free-text into
route_learning_intent is the culprit. Fix: route prioritisation/next-action requests to
the real next-action surface; only treat a CLEAN learning topic as study; otherwise
degrade gracefully (no-capable-mode) — never echo the raw sentence.

A clean topic ("communication - how to speak with charisma") must still route to study
(regression guard — see test_action_routing.py).
"""

from __future__ import annotations

from unittest.mock import patch

from pb.core.action_routing import suggest_commands_for_intent


def _haystack(cands) -> str:
    return " || ".join((c.backing_command + " " + c.human_label).lower() for c in cands)


def _studyish_commands(cands) -> list[str]:
    """Backing commands that route to a study/practise menu item (the A-07 surface)."""
    return [
        c.backing_command
        for c in cands
        if c.backing_command.lower().startswith(("study ", "practise ", "practice "))
    ]


def _identity_rerank(intent, candidates):  # keep tests offline + deterministic
    return list(candidates)


class TestDoNoEcho:
    def test_prioritisation_request_routes_to_next_not_study_echo(self, repo):
        cands = suggest_commands_for_intent(
            repo, "just give me the single most important one", limit=5
        )
        h = _haystack(cands)
        assert "single most important one" not in h, f"echo leaked into candidates: {h}"
        assert not any(
            c.backing_command.lower().startswith(("study just", "study give", "study i ", "practise just"))
            for c in cands
        )

    def test_conversational_ramble_not_echoed_as_study(self, repo):
        intent = "stop stalling and repeating my prompt back to me, I don't need a summary"
        cands = suggest_commands_for_intent(repo, intent, limit=5)
        assert "stop stalling and repeating" not in _haystack(cands), (
            "pb must not echo a conversational complaint as a Study/Practise task"
        )

    def test_my_todos_request_routes_to_next_not_echo(self, repo):
        cands = suggest_commands_for_intent(repo, "tell me which of my todos to start", limit=5)
        assert cands, "should surface next actions rather than nothing"
        assert "which of my todos" not in _haystack(cands)
        assert not any(
            c.backing_command.lower().startswith(("study tell", "practise tell")) for c in cands
        )

    def test_clean_topic_still_routes_to_study(self, repo):
        # Regression guard: a genuine learning topic must still become a study suggestion.
        cands = suggest_commands_for_intent(
            repo, "communication - how to speak with charisma", limit=5
        )
        assert any(
            c.backing_command.lower().startswith("study") and "charisma" in c.backing_command.lower()
            for c in cands
        ), f"clean topic must still route to study. Got: {[c.backing_command for c in cands]}"


class TestKeywordBranchNoEcho:
    """The residual 3/10 echoes (german-learner, review-driven, rust-developer) were
    NOT a _looks_like_topic gap in the fallback path — they entered through the
    _STUDY_KEYWORDS / _PRACTISE_KEYWORDS branches, which set topic = the whole intent
    and echoed it as 'study <rant>' / 'practise <rant>' WITHOUT any cleanliness guard.
    These are the real failing inputs, lightly trimmed, from run 20260603_120946.
    """

    @patch("pb.core.action_routing.rerank_candidates_with_gemini", _identity_rerank)
    def test_rust_developer_learning_rant_not_echoed_as_study(self, repo):
        # Triggers the _STUDY_KEYWORDS branch via "learn" inside "learning plan".
        intent = (
            "Cut the meta-talk. I don't need a learning plan, I need a compiler error to "
            "solve. Give me the buffer pool scenario where ownership actually matters."
        )
        cmds = _studyish_commands(suggest_commands_for_intent(repo, intent, limit=5))
        for cmd in cmds:
            low = cmd.lower()
            assert "compiler error" not in low, f"echoed rant into study/practise: {cmd}"
            assert "buffer pool" not in low, f"echoed rant into study/practise: {cmd}"
            assert "meta-talk" not in low, f"echoed rant into study/practise: {cmd}"

    @patch("pb.core.action_routing.rerank_candidates_with_gemini", _identity_rerank)
    def test_german_drill_request_not_echoed(self, repo):
        # Triggers _PRACTISE_KEYWORDS ("drill") AND _STUDY_KEYWORDS ("theory").
        intent = (
            "Hallo Sofia! We have 7 weeks until your B1 exam. To help you stop freezing "
            "during the 'Sprechen' part, let's drill Akkusativ and Dativ through "
            "production. No theory, just sentences."
        )
        cmds = _studyish_commands(suggest_commands_for_intent(repo, intent, limit=5))
        for cmd in cmds:
            low = cmd.lower()
            assert "7 weeks" not in low, f"echoed rant into study/practise: {cmd}"
            assert "sprechen" not in low, f"echoed rant into study/practise: {cmd}"

    @patch("pb.core.action_routing.rerank_candidates_with_gemini", _identity_rerank)
    def test_review_complaint_not_echoed_as_practise(self, repo):
        # Triggers the _PRACTISE_KEYWORDS branch via "session".
        intent = (
            "This review is completely inaccurate. I finished a 45-minute session on "
            "'Product spec — section 3' today. Why are you reporting zero minutes?"
        )
        cmds = _studyish_commands(suggest_commands_for_intent(repo, intent, limit=5))
        for cmd in cmds:
            low = cmd.lower()
            assert "completely inaccurate" not in low, f"echoed rant into study/practise: {cmd}"
            assert "45-minute session" not in low, f"echoed rant into study/practise: {cmd}"

    @patch("pb.core.action_routing.rerank_candidates_with_gemini", _identity_rerank)
    def test_clean_keyword_topic_still_routes_to_study(self, repo):
        # Regression: a CLEAN topic that contains a study keyword must still route to study.
        cands = suggest_commands_for_intent(repo, "learn German grammar", limit=5)
        assert any(
            c.backing_command.lower().startswith("study") and "grammar" in c.backing_command.lower()
            for c in cands
        ), f"clean keyword topic must still route to study: {[c.backing_command for c in cands]}"

    @patch("pb.core.action_routing.rerank_candidates_with_gemini", _identity_rerank)
    def test_clean_practise_topic_still_routes_to_practise(self, repo):
        # Regression: a CLEAN skill topic containing a practise keyword still routes to practise.
        cands = suggest_commands_for_intent(repo, "tennis serve technique", limit=5)
        assert any(
            c.backing_command.lower().startswith("practise") and "tennis" in c.backing_command.lower()
            for c in cands
        ), f"clean keyword topic must still route to practise: {[c.backing_command for c in cands]}"
