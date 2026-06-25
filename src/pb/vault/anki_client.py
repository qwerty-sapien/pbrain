# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AnkiConnect HTTP client and card management for pb Anki pipeline (Phase 18).

Provides:
- Thin HTTP wrapper around AnkiConnect (localhost:8765) with offline degradation
- Auto card generation via Flash/Flash Lite (ANKI-02, D-21)
- CRUD operations on pb.db anki_cards table (D-23)
- AnkiConnect bulk export with CSV fallback (D-24)
- Revlog pull with silent skip when offline (D-27)

Security:
- All SQL queries use parameterized ? placeholders (T-18-12)
- httpx timeout=5.0 on all requests (T-18-11)
- ConnectError and TimeoutException caught, returns None gracefully
"""

from __future__ import annotations

import csv
import datetime
import hashlib
import importlib
import re
from pathlib import Path
from typing import Any, Optional

import httpx
import structlog

from pb.core.renderables import ensure_renderable_text, renderable_anki_text
from pb.storage.yaml_io import dump_compact_yaml, extract_structured_yaml, load_yaml_text

logger = structlog.get_logger()

ANKI_DIRNAME = "pb-anki"
LEGACY_ANKI_DIRNAME = "pb-anki"

ANKI_URL = "http://localhost:8765"
ANKI_VERSION = 6
SUGGESTED_STATUS = "suggested"
ACCEPTED_STATUS = "accepted"
EDITED_STATUS = "edited"
EXPORTED_STATUS = "exported"
REJECTED_STATUS = "rejected"
EXPORT_READY_STATUSES = (ACCEPTED_STATUS, EDITED_STATUS)
LEGACY_EXPORT_READY_STATUSES = ("pending",)


# ---------------------------------------------------------------------------
# AnkiConnect HTTP client
# ---------------------------------------------------------------------------


def anki_request(action: str, **params) -> Optional[Any]:
    """Send AnkiConnect request. Returns result or None on error/offline.

    Uses httpx with 5s timeout (T-18-11). Catches all connection/timeout
    errors and returns None so callers can degrade gracefully (D-24, D-27).
    No f-string SQL — this is pure HTTP, not DB (T-18-12 applies to DB layer).
    """
    try:
        payload: dict[str, Any] = {"action": action, "version": ANKI_VERSION}
        if params:
            payload["params"] = params
        response = httpx.post(ANKI_URL, json=payload, timeout=5.0)
        data = response.json()
        if data.get("error"):
            logger.debug("anki.request_error", action=action, error=data["error"])
            return None
        return data.get("result")
    except (httpx.ConnectError, httpx.TimeoutException, Exception) as e:
        logger.debug("anki.connect_failed", action=action, error=str(e))
        return None


def is_anki_available() -> bool:
    """Check if AnkiConnect is reachable by calling deckNames."""
    result = anki_request("deckNames")
    return result is not None


def anki_quote(value: str) -> str:
    """Quote a search token for Anki query syntax."""
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def list_deck_names() -> list[str]:
    """Return all Anki deck names or [] when unavailable."""
    result = anki_request("deckNames")
    return list(result) if isinstance(result, list) else []


def find_notes(query: str) -> list[int]:
    """Return note IDs matching an Anki search query."""
    result = anki_request("findNotes", query=query)
    return [int(note_id) for note_id in result] if isinstance(result, list) else []


def notes_info(note_ids: list[int]) -> list[dict]:
    """Return note metadata rows from AnkiConnect notesInfo."""
    if not note_ids:
        return []
    result = anki_request("notesInfo", notes=note_ids)
    return list(result) if isinstance(result, list) else []


def model_field_names(model_name: str) -> list[str]:
    """Return field names for an Anki note type."""
    result = anki_request("modelFieldNames", modelName=model_name)
    return [str(name) for name in result] if isinstance(result, list) else []


def get_deck_note_types(deck_name: str, limit: int = 200) -> list[str]:
    """Return unique note types found in a deck."""
    note_ids = find_notes(f"deck:{anki_quote(deck_name)}")
    if limit:
        note_ids = note_ids[:limit]
    models = sorted(
        {
            str(info.get("modelName", "")).strip()
            for info in notes_info(note_ids)
            if info.get("modelName")
        }
    )
    return models


def infer_note_type_for_deck(deck_name: str) -> Optional[str]:
    """Infer a single note type for a deck, returning None when mixed or empty."""
    note_types = get_deck_note_types(deck_name)
    if len(note_types) == 1:
        return note_types[0]
    return None


def add_notes(notes: list[dict]) -> list[Optional[int]]:
    """Create notes in Anki and return the resulting note IDs."""
    if not notes:
        return []
    result = anki_request("addNotes", notes=notes)
    return list(result) if isinstance(result, list) else []


def _status_aliases(status: Optional[str]) -> tuple[str, ...]:
    """Resolve user-facing and legacy status names to stored DB values."""
    normalized = (status or SUGGESTED_STATUS).strip().lower()
    if normalized == "all":
        return ()
    if normalized == "exportable":
        return EXPORT_READY_STATUSES + LEGACY_EXPORT_READY_STATUSES
    if normalized == "pending":
        return EXPORT_READY_STATUSES + LEGACY_EXPORT_READY_STATUSES
    if normalized == ACCEPTED_STATUS:
        return (ACCEPTED_STATUS,)
    if normalized == EDITED_STATUS:
        return (EDITED_STATUS,)
    if normalized == EXPORTED_STATUS:
        return (EXPORTED_STATUS,)
    if normalized == REJECTED_STATUS:
        return (REJECTED_STATUS,)
    if normalized == SUGGESTED_STATUS:
        return (SUGGESTED_STATUS,)
    return (normalized,)


# ---------------------------------------------------------------------------
# Auto card generation (D-21, D-22, D-26)
# ---------------------------------------------------------------------------


def generate_auto_cards(
    note_slug: str,
    note_content: str,
    domain: str,
    deck_base: str = "",
    *,
    note_types: Optional[list[str]] = None,
    model: Optional[str] = None,
    emulate_existing_deck: bool = False,
) -> list[dict]:
    """Generate Anki cards from a vault note via Gemini (ANKI-02, D-21).

    Calls LLM on-demand (not automatic on note creation per D-21).
    Uses YAML for the model response we control, with JSON/YAML fallback parsing.
    Returns list of card dicts ready for insert_cards_to_db.
    Returns [] if LLM unavailable or parsing fails — never raises.
    """
    try:
        from pb.llm.gemini import get_client, FLASH_MODEL, resolve_model

        client = get_client()
        if not client.is_available():
            return []

        from pb.vault.lifecycle import read_frontmatter
        fm, body = read_frontmatter(note_content)
        selected_note_types = note_types or ["Basic"]
        resolved_model = resolve_model(model, fallback=FLASH_MODEL)

        prompt = (
            "Generate Anki flashcards from this note content.\n"
            f"Domain: {domain}\n"
            f"Note: {note_slug}\n\n"
            f"Content:\n{body[:3000]}\n\n"
            "Rules:\n"
            "- Generate 1-5 cards per note\n"
            "- Each card: one atomic fact or concept\n"
            f"- Allowed note types: {', '.join(selected_note_types)}\n"
            "- Front: clear question or prompt\n"
            "- Back: concise answer\n"
            "- If using Cloze or Fill in the blanks, include the blank/cloze syntax directly in front or back\n"
            f"- {'Match the tone and pacing of the existing deck examples embedded in the context' if emulate_existing_deck else 'Prefer the repo-native deck style over emulating existing Anki cards'}\n"
            "- Return ONLY a YAML list of objects with keys: note_type, front, back, sub_deck\n"
            "- `front` and `back` may be either plain strings or objects shaped like {text: ..., is_latex: true|false}\n"
            "- If a front or back should be treated as mathematical TeX/LaTeX, you MUST use the object form and set is_latex: true\n"
            "- Plain strings are always treated as plain text, even if they contain $...$\n"
            "- sub_deck should be one of: Vocab, Grammar, Cloze, Concepts, Practice\n"
            "- If you choose a note_type not in the allowed list, that card will be discarded"
        )
        result = client.generate_with_model(
            prompt,
            resolved_model,
            timeout=30,
            max_output_tokens=4000,
        )
        if not result:
            return []

        raw_cards = extract_structured_yaml(result.strip(), [])
        if not isinstance(raw_cards, list):
            # Backward-compatible fallback: extract a fenced/bare JSON array.
            match = re.search(r"\[.*\]", result, re.DOTALL)
            raw_cards = extract_structured_yaml(match.group() if match else "", [])

        if not isinstance(raw_cards, list):
            return []

        now = datetime.datetime.now().isoformat()
        cards = []
        for i, raw in enumerate(raw_cards):
            if not isinstance(raw, dict) or "front" not in raw or "back" not in raw:
                continue
            note_type_name = str(raw.get("note_type", raw.get("card_type", "Basic"))).strip() or "Basic"
            if note_type_name not in selected_note_types:
                continue
            front = renderable_anki_text(ensure_renderable_text(raw.get("front", "")))
            back = renderable_anki_text(ensure_renderable_text(raw.get("back", "")))
            sub_deck = raw.get("sub_deck", "Concepts")
            deck = f"{deck_base}::{sub_deck}" if deck_base else sub_deck
            # Detect cloze pattern for anki_model (D-26)
            has_cloze = "{{c1::" in front or "{{c1::" in back
            anki_model = "Cloze" if ("cloze" in note_type_name.lower() or has_cloze) else "Basic"
            card_id = f"{note_slug}-auto-{i}"
            cards.append({
                "id": card_id,
                "note_slug": note_slug,
                "front": front,
                "back": back,
                "card_type": note_type_name,
                "status": SUGGESTED_STATUS,
                "deck": deck,
                "tags": dump_compact_yaml([domain, "auto", note_type_name]),
                "anki_model": anki_model,
                "created_at": now,
                "updated_at": now,
            })
        return cards
    except Exception as e:
        logger.debug("anki.generate_auto_failed", note=note_slug, error=str(e))
        return []


# ---------------------------------------------------------------------------
# Card storage (D-23)
# ---------------------------------------------------------------------------


def insert_cards_to_db(cards: list[dict]) -> int:
    """Insert cards into anki_cards table. Returns count inserted.

    Uses INSERT OR REPLACE to handle re-generation of existing cards.
    All values use parameterized ? placeholders (T-18-12).
    Extended in Phase 26 to store domain, run_id, anki_note_id (None-safe for legacy callers).
    """
    if not cards:
        return 0
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            for card in cards:
                conn.execute(
                    "INSERT OR REPLACE INTO anki_cards "
                    "(id, note_slug, front, back, card_type, status, deck, tags, anki_model, "
                    "domain, run_id, anki_note_id, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        card["id"],
                        card["note_slug"],
                        card["front"],
                        card["back"],
                        card["card_type"],
                        card["status"],
                        card["deck"],
                        card["tags"],
                        card.get("anki_model", "Basic"),
                        card.get("domain"),         # NEW — None if caller omits
                        card.get("run_id"),         # NEW — None if caller omits
                        card.get("anki_note_id"),   # NEW — None at insert time; set on export
                        card["created_at"],
                        card["updated_at"],
                    ),
                )
            conn.commit()
        return len(cards)
    except Exception as e:
        logger.debug("anki.insert_cards_failed", error=str(e))
        return 0


def get_pending_card_count(domain: Optional[str] = None) -> int:
    """Count accepted/edited cards awaiting export.

    Returns 0 on any error — never raises.
    Parameterized queries only (T-18-12).
    """
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            placeholders = ", ".join("?" for _ in EXPORT_READY_STATUSES + LEGACY_EXPORT_READY_STATUSES)
            params: list[str] = list(EXPORT_READY_STATUSES + LEGACY_EXPORT_READY_STATUSES)
            if domain:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM anki_cards WHERE status IN ({placeholders}) AND deck LIKE ?",
                    (*params, f"%{domain}%"),
                ).fetchone()
            else:
                row = conn.execute(
                    f"SELECT COUNT(*) FROM anki_cards WHERE status IN ({placeholders})",
                    params,
                ).fetchone()
            return row[0] if row else 0
    except Exception:
        return 0


def get_cards_by_status(status: str = SUGGESTED_STATUS, domain: Optional[str] = None) -> list[dict]:
    """Get cards from anki_cards filtered by one logical status and optionally domain.

    Parameterized queries only (T-18-12).
    """
    try:
        from pb.storage.database import get_connection
        statuses = _status_aliases(status)
        with get_connection() as conn:
            if not statuses:
                base_sql = (
                    "SELECT id, note_slug, front, back, card_type, status, deck, tags, anki_model, domain "
                    "FROM anki_cards"
                )
                params: tuple[object, ...]
                if domain:
                    base_sql += " WHERE deck LIKE ?"
                    params = (f"%{domain}%",)
                else:
                    params = ()
                rows = conn.execute(f"{base_sql} ORDER BY created_at", params).fetchall()
            elif domain:
                placeholders = ", ".join("?" for _ in statuses)
                rows = conn.execute(
                    "SELECT id, note_slug, front, back, card_type, status, deck, tags, anki_model, domain "
                    f"FROM anki_cards WHERE status IN ({placeholders}) AND deck LIKE ? ORDER BY created_at",
                    (*statuses, f"%{domain}%"),
                ).fetchall()
            else:
                placeholders = ", ".join("?" for _ in statuses)
                rows = conn.execute(
                    "SELECT id, note_slug, front, back, card_type, status, deck, tags, anki_model, domain "
                    f"FROM anki_cards WHERE status IN ({placeholders}) ORDER BY created_at",
                    statuses,
                ).fetchall()
            return [
                {
                    "id": r[0],
                    "note_slug": r[1],
                    "front": r[2],
                    "back": r[3],
                    "card_type": r[4],
                    "status": r[5],
                    "deck": r[6],
                    "tags": r[7],
                    "anki_model": r[8],
                    "domain": r[9],
                }
                for r in rows
            ]
    except Exception:
        return []


def get_card_by_id(card_id: str) -> Optional[dict]:
    """Return one stored Anki candidate card by id, or None if missing."""
    try:
        from pb.storage.database import get_connection

        with get_connection() as conn:
            row = conn.execute(
                "SELECT id, note_slug, front, back, card_type, status, deck, tags, anki_model, domain "
                "FROM anki_cards WHERE id = ?",
                (card_id,),
            ).fetchone()
        if row is None:
            return None
        return {
            "id": row[0],
            "note_slug": row[1],
            "front": row[2],
            "back": row[3],
            "card_type": row[4],
            "status": row[5],
            "deck": row[6],
            "tags": row[7],
            "anki_model": row[8],
            "domain": row[9],
        }
    except Exception:
        return None


def get_card_status_counts(domain: Optional[str] = None) -> dict[str, int]:
    """Return candidate counts grouped by stored status."""
    try:
        from pb.storage.database import get_connection

        with get_connection() as conn:
            if domain:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM anki_cards WHERE deck LIKE ? GROUP BY status",
                    (f"%{domain}%",),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT status, COUNT(*) FROM anki_cards GROUP BY status"
                ).fetchall()
        return {str(status): int(count) for status, count in rows}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Export backends
# ---------------------------------------------------------------------------


def _stable_anki_id(seed: str) -> int:
    """Create a stable positive numeric identifier for genanki decks/models."""
    digest = hashlib.sha1(seed.encode("utf-8")).hexdigest()[:15]
    return int(digest, 16)


def _normalise_tags(tags_raw: Any) -> list[str]:
    """Coerce stored YAML/JSON-ish tags into a clean tag list."""
    tags = load_yaml_text(tags_raw, []) if isinstance(tags_raw, str) else tags_raw
    if not isinstance(tags, list):
        return []
    return [str(tag).strip().replace(" ", "_") for tag in tags if str(tag).strip()]


def _resolve_export_model(card: dict) -> str:
    """Map stored card metadata to a supported genanki model key."""
    note_type = str(card.get("card_type", "")).lower()
    anki_model = str(card.get("anki_model", "")).lower()

    if "reversed" in note_type:
        return "basic_reversed"
    if anki_model == "cloze" or "{{c1::" in card.get("front", "") or "{{c1::" in card.get("back", ""):
        return "cloze"
    return "basic"


def _build_genanki_model(genanki_module: Any, model_key: str):
    """Create a deterministic genanki model for the supported card families."""
    if model_key == "cloze":
        return genanki_module.Model(
            _stable_anki_id("pb:model:cloze"),
            "pb Cloze",
            fields=[{"name": "Text"}, {"name": "Extra"}],
            templates=[
                {
                    "name": "Cloze",
                    "qfmt": "{{cloze:Text}}",
                    "afmt": "{{cloze:Text}}<hr id=\"answer\">{{Extra}}",
                }
            ],
            css="""
.card {
  font-family: Arial, sans-serif;
  font-size: 20px;
  text-align: left;
  color: black;
  background-color: white;
}
""".strip(),
            model_type=genanki_module.Model.CLOZE,
        )

    if model_key == "basic_reversed":
        return genanki_module.Model(
            _stable_anki_id("pb:model:basic_reversed"),
            "pb Basic (and reversed card)",
            fields=[{"name": "Front"}, {"name": "Back"}],
            templates=[
                {
                    "name": "Card 1",
                    "qfmt": "{{Front}}",
                    "afmt": "{{FrontSide}}<hr id=\"answer\">{{Back}}",
                },
                {
                    "name": "Card 2",
                    "qfmt": "{{Back}}",
                    "afmt": "{{FrontSide}}<hr id=\"answer\">{{Front}}",
                },
            ],
        )

    return genanki_module.Model(
        _stable_anki_id("pb:model:basic"),
        "pb Basic",
        fields=[{"name": "Front"}, {"name": "Back"}],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{Front}}",
                "afmt": "{{FrontSide}}<hr id=\"answer\">{{Back}}",
            }
        ],
    )


def export_cards_to_apkg(cards: list[dict], vault_path: Path) -> tuple[bool, Optional[Path], str]:
    """Package cards into an .apkg file using genanki."""
    if not cards:
        return False, None, "No cards to export"

    try:
        genanki = importlib.import_module("genanki")
    except ModuleNotFoundError:
        return False, None, "genanki is not installed; falling back to non-package export"
    except Exception as exc:
        return False, None, f"genanki could not be loaded ({exc})"

    out_dir = _anki_export_dir(vault_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    out_path = out_dir / f"export-{timestamp}.apkg"

    decks: dict[str, Any] = {}
    models: dict[str, Any] = {}

    for card in cards:
        deck_name = card.get("deck") or "PB"
        deck = decks.get(deck_name)
        if deck is None:
            deck = genanki.Deck(_stable_anki_id(f"pb:deck:{deck_name}"), deck_name)
            decks[deck_name] = deck

        model_key = _resolve_export_model(card)
        model = models.get(model_key)
        if model is None:
            model = _build_genanki_model(genanki, model_key)
            models[model_key] = model

        if model_key == "cloze":
            fields = [card.get("front", ""), card.get("back", "")]
        else:
            fields = [card.get("front", ""), card.get("back", "")]

        note = genanki.Note(
            model=model,
            fields=fields,
            tags=_normalise_tags(card.get("tags", [])),
            guid=str(card.get("id", "")) or None,
        )
        deck.add_note(note)

    try:
        genanki.Package(list(decks.values())).write_to_file(str(out_path))
    except Exception as exc:
        return False, None, f"Failed to write .apkg ({exc})"

    _mark_cards_exported([card["id"] for card in cards])
    return True, out_path, f"Packaged {len(cards)} cards into {out_path.name}"


def export_cards_to_anki(cards: list[dict]) -> tuple[bool, str]:
    """Bulk export cards via AnkiConnect addNotes. Returns (success, message).

    Falls back to CSV if AnkiConnect unavailable (D-24).
    Uses allowDuplicate=False to avoid re-importing existing cards.
    Cloze model uses Text/Extra fields; Basic uses Front/Back.

    Unlike anki_request(), makes the HTTP call inline so the actual
    AnkiConnect error string is captured and returned to the caller.
    """
    if not cards:
        return False, "No cards to export"

    # D-24 Extension: Ensure decks exist before adding notes
    decks_to_create = set(card["deck"] for card in cards if card.get("deck"))
    for deck in decks_to_create:
        try:
            payload: dict[str, Any] = {
                "action": "createDeck",
                "version": ANKI_VERSION,
                "params": {"deck": deck},
            }
            httpx.post(ANKI_URL, json=payload, timeout=5.0)
        except Exception as e:
            logger.debug("anki.create_deck_failed", deck=deck, error=str(e))

    notes = []
    for card in cards:
        model = card.get("anki_model", "Basic")
        if model == "Cloze":
            fields = {"Text": card["front"], "Extra": card["back"]}
        else:
            fields = {"Front": card["front"], "Back": card["back"]}
        tags_raw = card.get("tags", "[]")
        tags = load_yaml_text(tags_raw, []) if isinstance(tags_raw, str) else tags_raw
        notes.append({
            "deckName": card["deck"],
            "modelName": model,
            "fields": fields,
            "tags": tags if isinstance(tags, list) else [],
            "options": {"allowDuplicate": False},
        })

    try:
        payload: dict[str, Any] = {
            "action": "addNotes",
            "version": ANKI_VERSION,
            "params": {"notes": notes},
        }
        response = httpx.post(ANKI_URL, json=payload, timeout=5.0)
        data = response.json()
    except (httpx.ConnectError, httpx.TimeoutException) as e:
        return False, f"Cannot reach AnkiConnect ({e})"
    except Exception as e:
        return False, f"Export request failed: {e}"

    if data.get("error"):
        return False, f"AnkiConnect: {data['error']}"

    result = data.get("result")
    if result is None:
        return False, "AnkiConnect returned empty result"

    exported_note_ids = {
        card["id"]: note_id
        for card, note_id in zip(cards, result)
        if note_id is not None
    }
    exported_count = len(exported_note_ids)
    _mark_cards_exported(list(exported_note_ids), exported_note_ids)

    if exported_count == 0:
        return False, f"AnkiConnect rejected all {len(cards)} notes (likely: duplicate or model name mismatch)"
    return True, f"Exported {exported_count} of {len(cards)} cards to Anki"


def export_cards_to_csv(cards: list[dict], vault_path: Path) -> Path:
    """Export cards to CSV as fallback when AnkiConnect offline (D-24).

    Creates vault/pb-anki/export-{date}.csv with front, back, deck, tags columns.
    Directory is created if it doesn't exist.
    """
    out_dir = _anki_export_dir(vault_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    date_str = datetime.date.today().isoformat()
    out_path = out_dir / f"export-{date_str}.csv"
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["front", "back", "deck", "tags"])
        writer.writeheader()
        for card in cards:
            tags_raw = card.get("tags", "[]")
            tags = load_yaml_text(tags_raw, []) if isinstance(tags_raw, str) else tags_raw
            writer.writerow({
                "front": card["front"],
                "back": card["back"],
                "deck": card.get("deck", ""),
                "tags": " ".join(tags) if isinstance(tags, list) else str(tags),
            })
    _mark_cards_exported([c["id"] for c in cards])
    return out_path


def _anki_export_dir(vault_path: Path) -> Path:
    preferred = vault_path / ANKI_DIRNAME
    legacy = vault_path / LEGACY_ANKI_DIRNAME
    if preferred.exists() or not legacy.exists():
        return preferred
    return legacy


def update_card_status(card_id: str, new_status: str) -> None:
    """Update a single card status in anki_cards. Non-fatal."""
    update_cards_status([card_id], new_status)


def update_cards_status(card_ids: list[str], new_status: str) -> None:
    """Update status of cards in anki_cards table. Non-fatal — silently swallows errors.

    Parameterized queries only (T-18-12).
    """
    try:
        from pb.storage.database import get_connection
        now = datetime.datetime.now().isoformat()
        with get_connection() as conn:
            for cid in card_ids:
                conn.execute(
                    "UPDATE anki_cards SET status = ?, updated_at = ? WHERE id = ?",
                    (new_status, now, cid),
                )
            conn.commit()
    except Exception:
        pass


def _mark_cards_exported(
    card_ids: list[str],
    note_ids: Optional[dict[str, int]] = None,
) -> None:
    """Mark cards as exported and persist optional Anki note IDs."""
    try:
        from pb.storage.database import get_connection

        now = datetime.datetime.now().isoformat()
        with get_connection() as conn:
            for card_id in card_ids:
                note_id = note_ids.get(card_id) if note_ids else None
                if note_id is None:
                    conn.execute(
                        "UPDATE anki_cards "
                        "SET status = 'exported', exported_at = ?, updated_at = ? "
                        "WHERE id = ?",
                        (now, now, card_id),
                    )
                else:
                    conn.execute(
                        "UPDATE anki_cards "
                        "SET status = 'exported', exported_at = ?, updated_at = ?, anki_note_id = ? "
                        "WHERE id = ?",
                        (now, now, note_id, card_id),
                    )
            conn.commit()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Revlog pull (D-27)
# ---------------------------------------------------------------------------


def sync_revlog() -> list[dict[str, Any]]:
    """Pull Anki revlog stats via AnkiConnect. Silently skips if offline (D-27).

    Stores per-deck review counts in pb.db anki_revlog table.
    Called by pb plan day and pb study — non-fatal, nice-to-have.
    """
    try:
        decks = anki_request("deckNames")
        if not decks:
            return []

        from pb.storage.database import get_connection
        now = datetime.datetime.now().isoformat()
        synced: list[dict[str, Any]] = []

        with get_connection() as conn:
            for deck_name in decks:
                stats = get_deck_review_stats(deck_name)
                if stats:
                    row = {
                        "deck": deck_name,
                        "cards": int(stats.get("cards", 0) or 0),
                        "reviews": int(stats.get("reviews", 0) or 0),
                        "pulled_at": now,
                    }
                    synced.append(row)
                    conn.execute(
                        "INSERT INTO anki_revlog (deck, cards_total, reviews_total, pulled_at) "
                        "VALUES (?, ?, ?, ?)",
                        (row["deck"], row["cards"], row["reviews"], row["pulled_at"]),
                    )
            conn.commit()
        return synced
    except Exception as e:
        logger.debug("anki.sync_revlog_failed", error=str(e))
        return []


def get_deck_review_stats(deck_name: str) -> Optional[dict]:
    """Pull review stats for a single deck via AnkiConnect.

    Returns dict with deck, cards, reviews keys, or None if Anki offline.
    Uses findCards + getReviewsOfCards AnkiConnect actions.
    """
    card_ids = anki_request("findCards", query=f"deck:{deck_name}")
    if not card_ids:
        return None
    reviews = anki_request("getReviewsOfCards", cards=card_ids)
    if not reviews:
        return {"deck": deck_name, "cards": len(card_ids), "reviews": 0}
    total_reviews = sum(len(v) for v in reviews.values() if isinstance(v, list))
    return {"deck": deck_name, "cards": len(card_ids), "reviews": total_reviews}


# ---------------------------------------------------------------------------
# Card CRUD helpers
# ---------------------------------------------------------------------------


def delete_card(card_id: str) -> bool:
    """Delete a card from anki_cards table. Returns True on success.

    Parameterized query (T-18-12).
    """
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            conn.execute("DELETE FROM anki_cards WHERE id = ?", (card_id,))
            conn.commit()
        return True
    except Exception:
        return False


def update_card(
    card_id: str,
    front: Optional[str] = None,
    back: Optional[str] = None,
    deck: Optional[str] = None,
) -> bool:
    """Update card fields in anki_cards table. Returns True on success.

    Only updates provided fields. Parameterized queries (T-18-12).
    """
    try:
        from pb.storage.database import get_connection
        now = datetime.datetime.now().isoformat()
        with get_connection() as conn:
            if front is not None:
                conn.execute(
                    "UPDATE anki_cards SET front = ?, updated_at = ? WHERE id = ?",
                    (front, now, card_id),
                )
            if back is not None:
                conn.execute(
                    "UPDATE anki_cards SET back = ?, updated_at = ? WHERE id = ?",
                    (back, now, card_id),
                )
            if deck is not None:
                conn.execute(
                    "UPDATE anki_cards SET deck = ?, updated_at = ? WHERE id = ?",
                    (deck, now, card_id),
                )
            conn.commit()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Run log helper (Phase 26 — D-08)
# ---------------------------------------------------------------------------


def _insert_run_log_entry(
    run_id: str,
    note_slug: str,
    term: Optional[str],
    card_count: int,
    source: str,
) -> None:
    """Insert a generation_run_log row. Called by Socratic hook and AnkiService. D-08."""
    import datetime as _dt_mod
    try:
        from pb.storage.database import get_connection
        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO generation_run_log "
                "(run_id, note_slug, term, card_count, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, note_slug, term, card_count, source,
                 _dt_mod.datetime.now().isoformat()),
            )
            conn.commit()
    except Exception as exc:
        logger.debug("anki.insert_run_log_failed", error=str(exc))
