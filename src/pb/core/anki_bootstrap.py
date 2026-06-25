# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Non-fatal Anki startup and review-signal sync for CLI bootstrap."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import platform
import subprocess
import time
from typing import Any

import structlog

from pb.storage.config import Config, save_config


ANKI_AUTO_OPEN_PREF = "anki_auto_open_approved"
ANKI_LAST_REVIEW_TOTAL_PREF = "anki_last_review_total"
ANKI_LAST_CHECK_AT_PREF = "anki_last_check_at"
ANKI_RECENT_REVIEW_SIGNAL_PREF = "anki_recent_review_signal"
ANKI_REVIEW_THRESHOLD = 10

logger = structlog.get_logger()


@dataclass
class AnkiBootstrapResult:
    """Outcome of one best-effort Anki bootstrap check."""

    approved: bool
    available: bool = False
    opened_attempted: bool = False
    synced: bool = False
    reviews_since_last_check: int = 0
    review_threshold: int = ANKI_REVIEW_THRESHOLD
    personalized: bool = False
    message: str = ""
    warnings: list[str] = field(default_factory=list)


def anki_auto_open_approved(config: Config) -> bool:
    """Return whether the user approved automatic Anki launch."""

    prefs = getattr(config, "preferences", {}) or {}
    return bool(prefs.get(ANKI_AUTO_OPEN_PREF))


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _summarize_synced_rows(rows: list[dict[str, Any]]) -> tuple[int, int, int]:
    cards_total = 0
    reviews_total = 0
    for row in rows:
        cards_total += _safe_int(row.get("cards", row.get("cards_total")))
        reviews_total += _safe_int(row.get("reviews", row.get("reviews_total")))
    return len(rows), cards_total, reviews_total


def record_anki_review_signal(
    config: Config,
    synced_rows: list[dict[str, Any]],
    *,
    config_path: Path | None = None,
    save: bool = True,
) -> dict[str, Any]:
    """Record a thresholded Anki review signal in config preferences."""

    deck_count, cards_total, reviews_total = _summarize_synced_rows(synced_rows)
    prefs = dict(getattr(config, "preferences", {}) or {})
    previous_raw = prefs.get(ANKI_LAST_REVIEW_TOTAL_PREF)
    previous_total = _safe_int(previous_raw) if previous_raw is not None else None
    baseline_initialized = previous_total is None
    reviews_since_last_check = 0 if baseline_initialized else max(0, reviews_total - previous_total)
    checked_at = datetime.utcnow().isoformat()
    signal = {
        "checked_at": checked_at,
        "decks": deck_count,
        "cards_total": cards_total,
        "reviews_total": reviews_total,
        "reviews_since_last_check": reviews_since_last_check,
        "review_threshold": ANKI_REVIEW_THRESHOLD,
        "eligible": reviews_since_last_check >= ANKI_REVIEW_THRESHOLD,
        "baseline_initialized": baseline_initialized,
    }

    prefs[ANKI_LAST_REVIEW_TOTAL_PREF] = reviews_total
    prefs[ANKI_LAST_CHECK_AT_PREF] = checked_at
    prefs[ANKI_RECENT_REVIEW_SIGNAL_PREF] = signal
    config.preferences = prefs
    if save:
        try:
            save_config(config, path=config_path)
        except Exception as exc:  # pragma: no cover - defensive, non-fatal
            logger.debug("anki.bootstrap_save_failed", error=str(exc))
    return signal


def _attempt_open_anki(*, system_name: str, timeout: float = 5.0) -> tuple[bool, str]:
    if system_name != "Darwin":
        return False, "Automatic Anki launch is only supported on macOS."
    try:
        completed = subprocess.run(
            ["open", "-gja", "Anki"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
    except Exception as exc:
        return False, f"Could not ask macOS to open Anki ({exc})."
    if completed.returncode != 0:
        detail = (completed.stderr or "").strip()
        suffix = f" ({detail})" if detail else ""
        return False, f"macOS could not open Anki{suffix}."
    return True, ""


def bootstrap_anki_if_approved(
    config: Config,
    *,
    config_path: Path | None = None,
    console: Any = None,
    interactive: bool = False,
    system_name: str | None = None,
    retry_delay_seconds: float = 1.0,
) -> AnkiBootstrapResult:
    """Ensure Anki is reachable and sync review stats when the user approved it."""

    result = AnkiBootstrapResult(approved=anki_auto_open_approved(config))
    if not result.approved:
        return result

    try:
        from pb.vault.anki_client import is_anki_available, sync_revlog

        available = is_anki_available()
        if not available:
            result.opened_attempted = True
            ok, warning = _attempt_open_anki(system_name=system_name or platform.system())
            if warning:
                result.warnings.append(warning)
            if ok and retry_delay_seconds > 0:
                time.sleep(retry_delay_seconds)
            available = is_anki_available()

        if not available:
            result.available = False
            result.message = (
                "Anki is not reachable yet. Open Anki and make sure AnkiConnect is enabled; "
                "pb will continue without review personalization."
            )
            if interactive and console is not None:
                for warning in result.warnings:
                    console.print(f"[warn]{warning}[/]")
                console.print(f"[warn]{result.message}[/]")
            return result

        result.available = True
        rows = sync_revlog()
        result.synced = True
        signal = record_anki_review_signal(config, rows, config_path=config_path)
        result.reviews_since_last_check = _safe_int(signal.get("reviews_since_last_check"))
        result.review_threshold = _safe_int(signal.get("review_threshold"), ANKI_REVIEW_THRESHOLD)
        result.personalized = bool(signal.get("eligible"))
        if interactive and console is not None and not result.personalized:
            remaining = max(0, result.review_threshold - result.reviews_since_last_check)
            if signal.get("baseline_initialized"):
                console.print(
                    "[dim]Anki connected; review baseline captured. "
                    f"Review {result.review_threshold} cards to unlock better personalization.[/]"
                )
            else:
                console.print(
                    "[dim]"
                    f"Reviewed {result.reviews_since_last_check} Anki card(s) since last check; "
                    f"{remaining} more unlocks better personalization.[/]"
                )
        return result
    except Exception as exc:
        result.message = (
            "Anki review sync was skipped because pb could not complete the AnkiConnect check."
        )
        result.warnings.append(str(exc))
        logger.debug("anki.bootstrap_failed", error=str(exc))
        if interactive and console is not None:
            console.print(f"[warn]{result.message}[/]")
        return result
