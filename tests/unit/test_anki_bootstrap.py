from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from pb.core.anki_bootstrap import (
    ANKI_AUTO_OPEN_PREF,
    ANKI_RECENT_REVIEW_SIGNAL_PREF,
    bootstrap_anki_if_approved,
    record_anki_review_signal,
)
from pb.core.learner_memory import build_global_learner_profile, learner_profile_prompt
from pb.storage.config import Config, GeneralConfig


class FakeConsole:
    def __init__(self):
        self.messages: list[str] = []

    def print(self, message):
        self.messages.append(str(message))


def _config(tmp_path, *, approved: bool = False) -> Config:
    cfg = Config(general=GeneralConfig(vault_path=str(tmp_path)))
    cfg.preferences = {ANKI_AUTO_OPEN_PREF: approved}
    return cfg


def test_anki_bootstrap_skips_when_consent_not_saved(tmp_path):
    cfg = _config(tmp_path, approved=False)

    with patch("pb.vault.anki_client.is_anki_available") as available_mock:
        result = bootstrap_anki_if_approved(cfg, interactive=True, console=FakeConsole())

    available_mock.assert_not_called()
    assert result.approved is False
    assert result.available is False


def test_anki_bootstrap_opens_when_offline_then_syncs_reviews(tmp_path):
    cfg = _config(tmp_path, approved=True)
    cfg.preferences["anki_last_review_total"] = 5

    with patch("pb.vault.anki_client.is_anki_available", side_effect=[False, True]), \
         patch("pb.core.anki_bootstrap._attempt_open_anki", return_value=(True, "")) as open_mock, \
         patch("pb.vault.anki_client.sync_revlog", return_value=[{"deck": "Math", "cards": 20, "reviews": 16}]), \
         patch("pb.core.anki_bootstrap.save_config") as save_mock:
        result = bootstrap_anki_if_approved(
            cfg,
            interactive=False,
            console=FakeConsole(),
            retry_delay_seconds=0,
        )

    open_mock.assert_called_once()
    save_mock.assert_called_once()
    assert result.available is True
    assert result.opened_attempted is True
    assert result.reviews_since_last_check == 11
    assert result.personalized is True


def test_anki_bootstrap_unavailable_message_is_user_friendly(tmp_path):
    cfg = _config(tmp_path, approved=True)
    console = FakeConsole()

    with patch("pb.vault.anki_client.is_anki_available", side_effect=[False, False]), \
         patch("pb.core.anki_bootstrap._attempt_open_anki", return_value=(False, "macOS could not open Anki.")):
        result = bootstrap_anki_if_approved(
            cfg,
            interactive=True,
            console=console,
            retry_delay_seconds=0,
        )

    assert result.available is False
    assert "Anki is not reachable yet" in result.message
    assert any("AnkiConnect" in message for message in console.messages)
    assert not any("Traceback" in message for message in console.messages)


def test_anki_bootstrap_sync_failure_is_nonfatal(tmp_path):
    cfg = _config(tmp_path, approved=True)
    console = FakeConsole()

    with patch("pb.vault.anki_client.is_anki_available", return_value=True), \
         patch("pb.vault.anki_client.sync_revlog", side_effect=RuntimeError("boom")):
        result = bootstrap_anki_if_approved(
            cfg,
            interactive=True,
            console=console,
            retry_delay_seconds=0,
        )

    assert result.approved is True
    assert result.message
    assert any("skipped" in message for message in console.messages)


def test_record_anki_review_signal_uses_since_last_check_threshold(tmp_path):
    cfg = _config(tmp_path, approved=True)
    cfg.preferences["anki_last_review_total"] = 30

    signal = record_anki_review_signal(
        cfg,
        [{"deck": "Math", "cards": 12, "reviews": 39}],
        save=False,
    )

    assert signal["reviews_since_last_check"] == 9
    assert signal["eligible"] is False
    assert cfg.preferences[ANKI_RECENT_REVIEW_SIGNAL_PREF]["reviews_since_last_check"] == 9


class FakeRepo:
    def list_feedback_events(self, scope_key: str, limit: int):
        return []


def test_learner_profile_hides_anki_signal_below_ten_reviews(tmp_path):
    cfg = _config(tmp_path, approved=True)
    cfg.preferences[ANKI_RECENT_REVIEW_SIGNAL_PREF] = {
        "reviews_since_last_check": 9,
        "review_threshold": 10,
    }
    profile = build_global_learner_profile(
        FakeRepo(),
        SimpleNamespace(vault_path=tmp_path, config=cfg),
    )

    assert profile["anki_review_signal"] == {}


def test_learner_profile_includes_anki_signal_at_ten_reviews(tmp_path):
    cfg = _config(tmp_path, approved=True)
    cfg.preferences[ANKI_RECENT_REVIEW_SIGNAL_PREF] = {
        "reviews_since_last_check": 10,
        "review_threshold": 10,
        "decks": 1,
    }
    profile = build_global_learner_profile(
        FakeRepo(),
        SimpleNamespace(vault_path=tmp_path, config=cfg),
    )

    assert profile["anki_review_signal"]["reviews_since_last_check"] == 10
    assert "Qualifying Anki review signal" in learner_profile_prompt(profile)
