# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""AnkiService — orchestration for generation, review feedback, and deck YAML IO.

INV-4: this module never imports rich or typer. All rendering stays in CLI callers.
"""

from __future__ import annotations

import datetime as dt
import random
import uuid
from pathlib import Path
from typing import Any, Optional

import structlog

from pb.core.base import BaseService, LoggableMixin
from pb.core.feedback_profile import feedback_prompt_suffix
from pb.llm.gemini import FLASH_MODEL, resolve_model
from pb.storage.yaml_io import (
    dump_yaml,
    extract_structured_yaml,
    load_yaml_with_legacy_json,
    write_yaml_file,
)

ANKI_DIRNAME = "pb-anki"
LEGACY_ANKI_DIRNAME = "pb-anki"


class AnkiService(BaseService, LoggableMixin):
    """Service-layer orchestrator for Anki card generation and YAML-backed deck state."""

    def __init__(self, vault_path: Path, repo: Any) -> None:
        super().__init__()
        self.vault_path = vault_path
        self.repo = repo
        self._log = structlog.get_logger()

    # --- Private helpers ---

    def _sanitize_deck_name(self, name: str) -> str:
        """Strip '..' and replace '/' and '\\' with '_' (T-26-04)."""
        return name.replace("..", "").replace("/", "_").replace("\\", "_")

    def _anki_root(self) -> Path:
        preferred = self.vault_path / ANKI_DIRNAME
        legacy = self.vault_path / LEGACY_ANKI_DIRNAME
        if preferred.exists() or not legacy.exists():
            return preferred
        return legacy

    def _deck_dir(self, deck_name: str) -> Path:
        safe = self._sanitize_deck_name(deck_name)
        return self._anki_root() / safe

    def _format_yaml_path(self, deck_name: str) -> Path:
        return self._deck_dir(deck_name) / "format.yaml"

    def _legacy_format_json_path(self, deck_name: str) -> Path:
        return self._deck_dir(deck_name) / "format.json"

    def _diagnostic_yaml_path(self, deck_name: str) -> Path:
        return self._deck_dir(deck_name) / "diagnostic.yaml"

    def _default_field_map(self, note_types: list[str]) -> dict[str, dict[str, str]]:
        mapping: dict[str, dict[str, str]] = {}
        for note_type in note_types:
            lower = note_type.lower()
            if "cloze" in lower:
                mapping[note_type] = {"front": "Text", "back": "Extra"}
            else:
                mapping[note_type] = {"front": "Front", "back": "Back"}
        return mapping

    # --- Format YAML ---

    def load_format_spec(self, deck_name: str) -> dict:
        """Read vault/pb-anki/<deck>/format.yaml with legacy fallback."""
        path = self._format_yaml_path(deck_name)
        legacy = self._legacy_format_json_path(deck_name)
        data = load_yaml_with_legacy_json(path, legacy, {})
        if not isinstance(data, dict):
            return {}
        if not path.exists() and legacy.exists() and data:
            try:
                self.save_format_spec(deck_name, data)
            except Exception as exc:
                self._log.warning("anki.migrate_legacy_format_failed", error=str(exc))
        return data

    def save_format_spec(self, deck_name: str, data: dict) -> Path:
        """Write vault/pb-anki/<deck>/format.yaml and keep metadata normalized."""
        path = self._format_yaml_path(deck_name)
        now = dt.datetime.now().isoformat()
        payload = dict(data)
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        if payload.get("note_types") and "field_map" not in payload:
            payload["field_map"] = self._default_field_map(list(payload["note_types"]))
        write_yaml_file(path, payload)
        return path

    # Backward-compatible wrappers kept for older callers/tests.
    def load_format_json(self, deck_name: str) -> dict:
        return self.load_format_spec(deck_name)

    def save_format_json(self, deck_name: str, data: dict) -> None:
        self.save_format_spec(deck_name, data)

    def draft_format_spec(
        self,
        deck_name: str,
        domain: str,
        note_types: list[str],
        *,
        model: Optional[str] = None,
        emulate_existing_deck: bool = False,
        sample_rows: Optional[list[dict]] = None,
    ) -> dict:
        """Draft a YAML-backed deck format using local context and an optional LLM call."""
        context_bundle = self.build_context_bundle(domain, term="")
        defaults = {
            "deck": deck_name,
            "domain": domain,
            "llm_model": resolve_model(model, fallback=FLASH_MODEL),
            "note_types": note_types,
            "field_map": self._default_field_map(note_types),
            "style_instructions": (
                "One atomic idea per note, concise wording, explicit contrasts, "
                "and no ornamental fluff."
            ),
            "front_template": "Prompt the learner to retrieve the exact concept.",
            "back_template": "State the answer succinctly, then add one clarifying line if needed.",
            "examples": [
                {
                    "note_type": note_types[0] if note_types else "Basic",
                    "front": "What principle is being tested here?",
                    "back": "Name the principle, then anchor it with one concrete cue.",
                }
            ],
            "emulate_existing_deck": emulate_existing_deck,
        }
        if sample_rows:
            defaults["emulated_samples"] = sample_rows[:8]

        try:
            from pb.llm.gemini import get_client

            client = get_client()
            if not client.is_available():
                return defaults

            sample_block = dump_yaml(sample_rows or [], flow=False).strip() or "[]"
            prompt = (
                "Design a compact Anki deck format for a CLI-first study system.\n"
                f"Deck: {deck_name}\n"
                f"Domain: {domain or 'general'}\n"
                f"Selected note types: {', '.join(note_types) if note_types else 'Basic'}\n"
                f"Emulate existing deck: {'yes' if emulate_existing_deck else 'no'}\n\n"
                "Existing note context:\n"
                f"{context_bundle.get('context_text', '')[:3000]}\n\n"
                "Existing deck samples:\n"
                f"{sample_block[:2500]}\n\n"
                "Return YAML only with keys:\n"
                "style_instructions: string\n"
                "front_template: string\n"
                "back_template: string\n"
                "examples:\n"
                "  - note_type: string\n"
                "    front: string\n"
                "    back: string\n"
                "Keep it practical and deterministic."
            )
            result = client.generate_with_model(
                prompt,
                resolve_model(model, fallback=FLASH_MODEL),
                timeout=30,
                max_output_tokens=4500,
            )
            structured = extract_structured_yaml(result or "", {})
            if isinstance(structured, dict):
                defaults.update({k: v for k, v in structured.items() if v is not None})
        except Exception as exc:
            self._log.warning("anki.draft_format_spec_failed", error=str(exc))

        defaults["note_types"] = note_types
        defaults["field_map"] = self._default_field_map(note_types)
        defaults["emulate_existing_deck"] = emulate_existing_deck
        defaults["llm_model"] = resolve_model(model, fallback=FLASH_MODEL)
        if sample_rows:
            defaults["emulated_samples"] = sample_rows[:8]
        return defaults

    # --- Diagnostic YAML ---

    def load_diagnostic_report(self, deck_name: str) -> dict:
        """Read vault/pb-anki/<deck>/diagnostic.yaml."""
        from pb.storage.yaml_io import load_yaml_file

        data = load_yaml_file(self._diagnostic_yaml_path(deck_name), {})
        return data if isinstance(data, dict) else {}

    def save_diagnostic_report(self, deck_name: str, data: dict) -> Path:
        """Write the downstream-ready Socratic diagnostic YAML report."""
        path = self._diagnostic_yaml_path(deck_name)
        write_yaml_file(path, data)
        return path

    # --- Context MD ---

    def load_context_md(self, deck_name: str) -> str:
        """Read vault/pb-anki/<deck>/context.md; return '' if absent."""
        path = self._deck_dir(deck_name) / "context.md"
        try:
            return path.read_text(encoding="utf-8") if path.exists() else ""
        except Exception as exc:
            self._log.warning("anki.load_context_md_failed", error=str(exc))
            return ""

    def append_context_md(self, deck_name: str, text: str) -> None:
        """Append text + newline to vault/pb-anki/<deck>/context.md."""
        path = self._deck_dir(deck_name) / "context.md"
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as fh:
            fh.write(text.rstrip("\n") + "\n")

    def build_context_bundle(self, domain: str, term: str = "", max_notes: int = 5) -> dict:
        """Collect local note context from the vault to feed downstream generation."""
        from pb.vault.lifecycle import read_frontmatter

        domain_dir = self.vault_path / "knowledge" / domain if domain else None
        sections: list[str] = []
        sources: list[str] = []

        if domain_dir and (domain_dir / "_state.md").exists():
            state_path = domain_dir / "_state.md"
            try:
                sections.append(f"## State\n{state_path.read_text(encoding='utf-8')[:1400].strip()}")
                sources.append(str(state_path.relative_to(self.vault_path)))
            except OSError:
                pass

        if not domain_dir or not domain_dir.exists():
            return {"context_text": "\n\n".join(sections), "source_notes": sources}

        needle = term.strip().lower()
        matched = 0
        for md_path in sorted(domain_dir.glob("*.md")):
            if md_path.name.startswith("_"):
                continue
            try:
                raw = md_path.read_text(encoding="utf-8")
                _, body = read_frontmatter(raw)
            except OSError:
                continue
            searchable = f"{md_path.stem}\n{body}".lower()
            if needle and needle not in searchable:
                continue
            rel = str(md_path.relative_to(self.vault_path))
            snippet = body.strip()[:900]
            if snippet:
                sections.append(f"## Source: {rel}\n{snippet}")
                sources.append(rel)
                matched += 1
            if matched >= max_notes:
                break

        return {"context_text": "\n\n".join(sections), "source_notes": sources}

    def gather_emulation_samples(
        self,
        deck_name: str,
        note_types: Optional[list[str]] = None,
        *,
        per_type_limit: int = 3,
    ) -> list[dict]:
        """Best-effort export of existing deck samples for style emulation."""
        if not deck_name:
            return []

        samples: list[dict] = []
        requested_types = [nt for nt in (note_types or []) if nt]
        for note_type in requested_types[:4]:
            try:
                rows, _ = self.export_existing_notes(
                    deck_name,
                    note_type=note_type,
                    limit=per_type_limit,
                )
            except Exception as exc:
                self._log.debug(
                    "anki.gather_emulation_samples_failed",
                    deck=deck_name,
                    note_type=note_type,
                    error=str(exc),
                )
                continue
            samples.extend(rows)

        if samples:
            return samples[:10]

        try:
            rows, _ = self.export_existing_notes(deck_name, limit=per_type_limit)
        except Exception as exc:
            self._log.debug(
                "anki.gather_emulation_samples_fallback_failed",
                deck=deck_name,
                error=str(exc),
            )
            return []
        return rows[:10]

    # --- Run log ---

    def insert_run_log(
        self,
        run_id: str,
        note_slug: str,
        term: Optional[str],
        card_count: int,
        source: str,
    ) -> None:
        """Insert a generation run entry. source: 'auto' | 'socratic' | 'term'."""
        from pb.storage.database import get_connection

        with get_connection() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO generation_run_log "
                "(run_id, note_slug, term, card_count, source, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    note_slug,
                    term,
                    card_count,
                    source,
                    dt.datetime.now().isoformat(),
                ),
            )
            conn.commit()

    def get_history(self, limit: int = 20) -> list[dict]:
        """Return list[dict] from generation_run_log ORDER BY created_at DESC LIMIT limit."""
        from pb.storage.database import get_connection

        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT run_id, note_slug, term, card_count, source, created_at "
                    "FROM generation_run_log ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]
        except Exception as exc:
            self._log.warning("anki.get_history_failed", error=str(exc))
            return []

    # --- Rollback ---

    def rollback_run(self, run_id: str) -> tuple[bool, str]:
        """Delete cards for run_id from pb.db; call AnkiConnect deleteNotes if IDs stored. D-08."""
        from pb.storage.database import get_connection
        from pb.vault.anki_client import anki_request

        try:
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT id, anki_note_id FROM anki_cards WHERE run_id = ?",
                    (run_id,),
                ).fetchall()
            if not rows:
                return False, f"Run '{run_id}' not found"
            anki_ids = [r["anki_note_id"] for r in rows if r["anki_note_id"] is not None]
            with get_connection() as conn:
                conn.execute("DELETE FROM anki_cards WHERE run_id = ?", (run_id,))
                conn.execute("DELETE FROM generation_run_log WHERE run_id = ?", (run_id,))
                conn.commit()
            if anki_ids:
                anki_request("deleteNotes", notes=anki_ids)
            return True, f"Rollback complete: {len(rows)} cards deleted"
        except Exception as exc:
            self._log.warning("anki.rollback_failed", error=str(exc))
            return False, f"Rollback failed: {exc}"

    # --- Suggested cards ---

    def get_suggested_cards(self, batch_size: int = 30) -> list[dict]:
        """Return a random sample of suggested cards from the latest run."""
        from pb.storage.database import get_connection

        try:
            with get_connection() as conn:
                row = conn.execute(
                    "SELECT run_id FROM generation_run_log ORDER BY created_at DESC LIMIT 1"
                ).fetchone()
            if not row:
                return []
            latest_run_id = row["run_id"]
            with get_connection() as conn:
                rows = conn.execute(
                    "SELECT * FROM anki_cards WHERE run_id = ? AND status IN ('suggested', 'pending')",
                    (latest_run_id,),
                ).fetchall()
            cards = [dict(r) for r in rows]
            if len(cards) <= batch_size:
                return cards
            return random.sample(cards, batch_size)
        except Exception as exc:
            self._log.warning("anki.get_suggested_cards_failed", error=str(exc))
            return []

    # --- Deck export/import workflow ---

    def export_existing_notes(
        self,
        deck_name: str,
        *,
        note_type: Optional[str] = None,
        query: Optional[str] = None,
        limit: Optional[int] = None,
    ) -> tuple[list[dict], str]:
        """Export an existing Anki deck/query to YAML-ready row dicts."""
        from pb.vault.anki_client import (
            anki_quote,
            find_notes,
            model_field_names,
            notes_info,
        )

        query_text = query or f"deck:{anki_quote(deck_name)}"
        if note_type:
            query_text += f" note:{anki_quote(note_type)}"

        note_ids = find_notes(query_text)
        if limit is not None:
            note_ids = note_ids[:limit]
        info_rows = notes_info(note_ids)
        if not info_rows:
            return [], note_type or ""

        if not note_type:
            note_types = sorted(
                {
                    str(info.get("modelName", "")).strip()
                    for info in info_rows
                    if info.get("modelName")
                }
            )
            if len(note_types) != 1:
                raise ValueError(
                    "Deck export spans multiple note types. Re-run with --note-type."
                )
            note_type = note_types[0]

        field_names = model_field_names(note_type)
        rows: list[dict] = []
        for info in info_rows:
            if note_type and info.get("modelName") != note_type:
                continue
            row = {"noteId": info.get("noteId"), "noteType": info.get("modelName", note_type)}
            fields = info.get("fields", {}) or {}
            for field_name in field_names:
                field_info = fields.get(field_name, {}) or {}
                row[field_name] = field_info.get("value", "")
            rows.append(row)

        return rows, note_type or ""

    def import_existing_notes(
        self,
        rows: list[dict],
        deck_name: str,
        *,
        note_type: Optional[str] = None,
        key_field: Optional[str] = None,
    ) -> dict:
        """Import or update YAML rows against an Anki deck, mirroring anki-llm semantics."""
        from pb.vault.anki_client import (
            add_notes,
            anki_quote,
            anki_request,
            find_notes,
            infer_note_type_for_deck,
            model_field_names,
            notes_info,
        )

        model_name = note_type or infer_note_type_for_deck(deck_name)
        if not model_name:
            raise ValueError(
                "Could not infer note type from deck. Pass --note-type explicitly."
            )

        field_names = model_field_names(model_name)
        if not field_names:
            raise ValueError(f"Note type '{model_name}' has no fields.")

        effective_key = key_field or ("noteId" if any("noteId" in row for row in rows) else field_names[0])
        if effective_key != "noteId" and effective_key not in field_names:
            raise ValueError(
                f"Key field '{effective_key}' is not present on note type '{model_name}'."
            )

        existing_ids = find_notes(
            f"deck:{anki_quote(deck_name)} note:{anki_quote(model_name)}"
        )
        existing_notes = notes_info(existing_ids)
        existing_by_key: dict[str, int] = {}
        for info in existing_notes:
            if effective_key == "noteId":
                existing_key = str(info.get("noteId", "")).strip()
            else:
                fields = info.get("fields", {}) or {}
                existing_key = str((fields.get(effective_key, {}) or {}).get("value", "")).strip()
            if existing_key:
                existing_by_key[existing_key] = int(info.get("noteId"))

        notes_to_add: list[dict] = []
        updates: list[tuple[int, dict[str, str]]] = []
        for row in rows:
            if not isinstance(row, dict):
                continue
            key_value = str(row.get(effective_key, "")).strip()
            if not key_value:
                continue
            mapped_fields = {
                field_name: str(row.get(field_name, "") or "")
                for field_name in field_names
                if field_name in row
            }
            if key_value in existing_by_key:
                updates.append((existing_by_key[key_value], mapped_fields))
            else:
                if effective_key != "noteId" and effective_key not in mapped_fields:
                    mapped_fields[effective_key] = key_value
                notes_to_add.append(
                    {
                        "deckName": deck_name,
                        "modelName": model_name,
                        "fields": mapped_fields,
                        "tags": ["pb-import"],
                        "options": {"allowDuplicate": False},
                    }
                )

        add_results = add_notes(notes_to_add) if notes_to_add else []
        if updates:
            actions = [
                {
                    "action": "updateNoteFields",
                    "params": {"note": {"id": note_id, "fields": fields}},
                }
                for note_id, fields in updates
            ]
            anki_request("multi", actions=actions)

        return {
            "deck": deck_name,
            "note_type": model_name,
            "added": sum(1 for result in add_results if result is not None),
            "updated": len(updates),
            "requested_adds": len(notes_to_add),
        }

    # --- Card generation ---

    def generate_cards(
        self,
        note_slug: str,
        note_content: str,
        domain: str,
        deck: str,
        *,
        term: Optional[str] = None,
        source: str = "auto",
        note_types: Optional[list[str]] = None,
        model: Optional[str] = None,
        emulate_existing_deck: bool = False,
    ) -> dict:
        """Generate cards from a note (or specific term). Returns {cards, run_id, count}."""
        from pb.vault.anki_client import generate_auto_cards, insert_cards_to_db

        run_id = str(uuid.uuid4())[:8]
        fmt = self.load_format_spec(deck)
        ctx_text = self.load_context_md(deck)
        diagnostic = self.load_diagnostic_report(deck)
        source_bundle = self.build_context_bundle(domain, term or note_slug)
        selected_note_types = note_types or list(fmt.get("note_types", []) or []) or ["Basic"]

        prompt_parts: list[str] = []
        if fmt:
            if fmt.get("style_instructions"):
                prompt_parts.append(f"Style instructions:\n{fmt['style_instructions']}")
            if fmt.get("front_template"):
                prompt_parts.append(f"Front template:\n{fmt['front_template']}")
            if fmt.get("back_template"):
                prompt_parts.append(f"Back template:\n{fmt['back_template']}")
            if fmt.get("examples"):
                prompt_parts.append("Deck examples:\n" + dump_yaml(fmt["examples"]).strip())
        if emulate_existing_deck:
            samples = list(fmt.get("emulated_samples", []) or [])
            if not samples:
                samples = self.gather_emulation_samples(deck, selected_note_types)
            if samples:
                prompt_parts.append(
                    "Emulate the deck style shown in these samples:\n"
                    + dump_yaml(samples[:8]).strip()
                )
        prompt_parts.append(f"Allowed note types: {', '.join(selected_note_types)}")

        if diagnostic:
            gaps = diagnostic.get("knowledge_gaps") or []
            if gaps:
                prompt_parts.append("Prioritize these diagnosed gaps:\n" + dump_yaml(gaps).strip())
        if ctx_text.strip():
            prompt_parts.append(f"Deck feedback context:\n{ctx_text.strip()}")
        feedback_guidance = feedback_prompt_suffix(self.vault_path, "anki").strip()
        if feedback_guidance:
            prompt_parts.append(feedback_guidance)
        if source_bundle.get("context_text"):
            prompt_parts.append(f"Knowledge source context:\n{source_bundle['context_text']}")

        if source == "socratic":
            prompt_parts.append(
                "Weight 70-80% of cards on omissions and misconceptions identified in the Q&A exchange. "
                "Weight 10-20% on concepts the user partially understands. Preserve the user's exact words."
            )

        prompt_prefix = "\n\n".join(part for part in prompt_parts if part)
        enriched_content = (
            (prompt_prefix + "\n\n" + note_content.strip()).strip()
            if prompt_prefix
            else note_content
        )

        effective_slug = term or note_slug
        cards = generate_auto_cards(
            note_slug=effective_slug,
            note_content=enriched_content,
            domain=domain,
            deck_base=deck,
            note_types=selected_note_types,
            model=resolve_model(model or fmt.get("llm_model"), fallback=FLASH_MODEL),
            emulate_existing_deck=emulate_existing_deck,
        )
        for card in cards:
            card["run_id"] = run_id
            card["domain"] = domain
        count = insert_cards_to_db(cards)
        self.insert_run_log(run_id, note_slug, term, count, source)
        return {"cards": cards, "run_id": run_id, "count": count}

    # --- Review edits summary ---

    def summarize_review_edits(self, deck: str, edited_cards: list[dict]) -> str:
        """Flash-summarize what the user edited/deleted during suggested review.

        Returns a summary string. Fallback to raw diff string if Flash fails. D-07.
        """
        if not edited_cards:
            return ""
        raw_diff = "\n".join(
            f"- Front changed to: {c.get('front', '')!r} / Back changed to: {c.get('back', '')!r}"
            for c in edited_cards
        )
        try:
            from pb.llm.gemini import get_client

            client = get_client()
            if not client.is_available():
                return raw_diff
            prompt = (
                f"You are summarizing card edits made during an Anki review session for deck '{deck}'.\n"
                f"The user made the following changes:\n{raw_diff}\n\n"
                "Write a concise 2-4 sentence note capturing what topics needed correction "
                "and what the user preferred (tone, phrasing, depth). "
                "This note will be appended to context.md to guide future card generation."
            )
            summary = client.generate_with_model(prompt, FLASH_MODEL, timeout=20, max_output_tokens=4000)
            if summary and summary.strip():
                return summary.strip()
            return raw_diff
        except Exception as exc:
            self._log.warning("anki.summarize_review_edits_failed", error=str(exc))
            return raw_diff
