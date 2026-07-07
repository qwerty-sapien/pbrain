from __future__ import annotations

from pb.cli.helpers import interpret_confirmation, prompt_confirmation
from pb.core.matching import MatchCandidate, resolve_strict_match


def test_preview_confirmation_treats_free_text_as_modify():
    decision = interpret_confirmation("let's do half the time per session", default=True, mode="preview")

    assert decision.kind == "modify"
    assert "half the time" in decision.text


def test_preview_confirmation_reserves_backtick_and_q():
    assert interpret_confirmation("`", default=True, mode="preview").kind == "accept"
    assert interpret_confirmation("q", default=True, mode="preview").kind == "cancel"


def test_preview_prompt_accepts_backtick_without_enter(monkeypatch, capsys):
    keys = iter(["backtick"])
    monkeypatch.setattr("pb.cli.helpers._is_real_tty", lambda: True)
    monkeypatch.setattr("pb.cli.helpers._read_key", lambda: next(keys))

    decision = prompt_confirmation("Create this goal?", default=True, mode="preview")

    out = capsys.readouterr().out
    assert decision.kind == "accept"
    assert "accept (`) | reject (q)" in out
    assert "Enter" not in out
    assert "\\`" not in out


def test_preview_refine_prompt_accepts_typed_refinement(monkeypatch, capsys):
    keys = iter(["m", "o", "r", "e", "space", "d", "e", "t", "a", "i", "l", "enter"])
    monkeypatch.setattr("pb.cli.helpers._is_real_tty", lambda: True)
    monkeypatch.setattr("pb.cli.helpers._read_key", lambda: next(keys))

    decision = prompt_confirmation("Create this plan?", default=True, mode="preview_refine")

    out = capsys.readouterr().out
    assert decision.kind == "modify"
    assert decision.text == "more detail"
    assert "accept (`) | reject (q) | refine (<type>)" in out
    assert "type refinement" not in out


def test_standard_confirmation_treats_unclear_text_as_safe_cancel():
    decision = interpret_confirmation("maybe later after I think about it", default=False, mode="standard")

    assert decision.kind == "cancel"


def test_strict_match_abstains_when_confidence_is_low():
    candidates = [
        MatchCandidate(key="task-a", label="Machine learning validation", text="scikit-learn model implementation"),
        MatchCandidate(key="task-b", label="Present tense regular verbs", text="spanish conjugation drills"),
    ]

    result = resolve_strict_match("ricci flow", candidates, allow_llm=False)

    assert result.accepted is False
    assert result.matched_index is None
    assert result.suggestions


def test_strict_match_accepts_unique_prefix():
    candidates = [
        MatchCandidate(key="task-abc123", label="Ricci flow foundations", text="geometry prerequisites"),
        MatchCandidate(key="task-def456", label="Machine learning validation", text="scikit-learn model implementation"),
    ]

    result = resolve_strict_match("ricci flow foun", candidates, allow_llm=False)

    assert result.accepted is True
    assert result.matched_index == 0
