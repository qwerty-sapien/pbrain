# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Question tree engine for structured note creation (D-41 to D-44).

Drives an interactive flow based on YAML schemas:
- Required fields presented one-at-a-time with progress indicator [N/M]
- Optional fields offered as multi-select checklist
- /chat activates Flash Lite with answered fields + schema context (D-42)
- /skip skips current optional field; refuses to skip required fields (D-43)
- /done finishes early, writing only filled fields (D-43)
- Confirmation shows all fields before vault write (D-44)

Filename sanitization is applied at get_vault_path() to prevent path
traversal via user-supplied field values (threat T-02-07-01).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import structlog

from pb.core.schemas import NoteSchema, SchemaField

logger = structlog.get_logger()


class QuestionTreeEngine:
    """Drives structured note creation from a NoteSchema."""

    def __init__(self, schema: NoteSchema):
        self.schema = schema
        self.values: dict[str, Optional[str]] = {}
        self._current_index = 0
        self._active_fields: list[SchemaField] = []
        self._phase = "required"  # "required", "optional_select", "optional_fill", "done"
        self._chat_active = False
        self._done = False

        # Separate required and optional fields from schema
        self._required_fields: list[SchemaField] = [f for f in schema.fields if f.required]
        self._optional_fields: list[SchemaField] = [f for f in schema.fields if not f.required]
        self._active_fields = list(self._required_fields)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_done(self) -> bool:
        return self._done

    @property
    def chat_active(self) -> bool:
        return self._chat_active

    # ------------------------------------------------------------------
    # Field accessors
    # ------------------------------------------------------------------

    def get_required_fields(self) -> list[SchemaField]:
        """Return a list of required fields from the schema."""
        return list(self._required_fields)

    def get_optional_fields(self) -> list[SchemaField]:
        """Return a list of optional fields from the schema."""
        return list(self._optional_fields)

    def current_field(self) -> Optional[SchemaField]:
        """Get the field currently being prompted, or None if flow is complete."""
        if self._done or self._phase in ("done", "optional_select"):
            return None
        if self._phase in ("required", "optional_fill"):
            if self._current_index < len(self._active_fields):
                return self._active_fields[self._current_index]
        return None

    # ------------------------------------------------------------------
    # Progress
    # ------------------------------------------------------------------

    def progress_text(self) -> str:
        """Return progress indicator like [2/5] or [optional 1/3]."""
        if self._phase == "required":
            total = len(self._required_fields)
            pos = min(self._current_index + 1, total)
            return f"[{pos}/{total}]"
        elif self._phase == "optional_fill":
            total = len(self._active_fields)
            pos = min(self._current_index + 1, total)
            return f"[optional {pos}/{total}]"
        return ""

    # ------------------------------------------------------------------
    # Field advancement
    # ------------------------------------------------------------------

    def skip_current_field(self) -> None:
        """Advance past the current field without storing a value.

        If the index overshoots the active list during the "required" phase,
        transition to "optional_select" (mirrors the inline logic previously
        used by inbox conversion).
        """
        self._current_index += 1
        if self._current_index >= len(self._active_fields):
            if self._phase == "required":
                self._phase = "optional_select"

    # ------------------------------------------------------------------
    # Input processing
    # ------------------------------------------------------------------

    def process_input(self, raw: str) -> dict:
        """Process user input and return a status dict.

        Returns dict with keys:
            action   : "stored" | "skipped" | "done" | "chat_activated" | "next_phase" | "error"
            field    : current SchemaField or None
            value    : the stored value or None
            chat_active : bool
            done     : bool
            message  : (only on "error") explanation string
            phase    : (only on "next_phase") next phase name
        """
        raw_stripped = raw.strip()

        # ---- Global commands ----------------------------------------
        if raw_stripped.lower() in ("quit", "q", "cancel", "/quit", "/cancel"):
            self._done = True
            return {
                "action": "cancelled",
                "field": None,
                "value": None,
                "chat_active": False,
                "done": True,
            }

        if raw_stripped.lower() == "/done":
            self._done = True
            return {
                "action": "done",
                "field": None,
                "value": None,
                "chat_active": False,
                "done": True,
            }

        if raw_stripped.lower() == "/skip":
            field = self.current_field()
            if field is not None and field.required:
                return {
                    "action": "error",
                    "field": field,
                    "value": None,
                    "chat_active": False,
                    "done": False,
                    "message": f"Cannot skip required field: {field.name}",
                }
            if field is not None:
                self.values[field.name] = None
                self._current_index += 1
            return self._advance()

        if raw_stripped.lower() == "/chat":
            self._chat_active = True
            return {
                "action": "chat_activated",
                "field": self.current_field(),
                "value": None,
                "chat_active": True,
                "done": False,
            }

        # ---- Normal value storage -----------------------------------
        field = self.current_field()
        if field is None:
            # No current field — treat as /done
            self._done = True
            return {
                "action": "done",
                "field": None,
                "value": None,
                "chat_active": False,
                "done": True,
            }

        # Validate select type
        if field.field_type == "select" and field.options and raw_stripped not in field.options:
            return {
                "action": "error",
                "field": field,
                "value": raw_stripped,
                "chat_active": False,
                "done": False,
                "message": f"Must be one of: {', '.join(field.options)}",
            }

        # Validate number type
        if field.field_type == "number":
            try:
                float(raw_stripped)
            except ValueError:
                return {
                    "action": "error",
                    "field": field,
                    "value": raw_stripped,
                    "chat_active": False,
                    "done": False,
                    "message": "Must be a number",
                }

        self.values[field.name] = raw_stripped
        self._current_index += 1
        return self._advance()

    def _advance(self) -> dict:
        """Advance to next field or transition to next phase."""
        if self._phase == "required":
            if self._current_index >= len(self._required_fields):
                # All required fields answered — move to optional selection
                self._phase = "optional_select"
                return {
                    "action": "next_phase",
                    "field": None,
                    "value": None,
                    "chat_active": False,
                    "done": False,
                    "phase": "optional_select",
                }
            return {
                "action": "stored",
                "field": self.current_field(),
                "value": None,
                "chat_active": False,
                "done": False,
            }

        elif self._phase == "optional_fill":
            if self._current_index >= len(self._active_fields):
                self._done = True
                return {
                    "action": "done",
                    "field": None,
                    "value": None,
                    "chat_active": False,
                    "done": True,
                }
            return {
                "action": "stored",
                "field": self.current_field(),
                "value": None,
                "chat_active": False,
                "done": False,
            }

        # Fallback
        return {
            "action": "stored",
            "field": self.current_field(),
            "value": None,
            "chat_active": False,
            "done": False,
        }

    # ------------------------------------------------------------------
    # Optional field selection
    # ------------------------------------------------------------------

    def set_optional_selections(self, selected_indices: list) -> None:
        """Set which optional fields the user wants to fill.

        Args:
            selected_indices: 0-based indices into get_optional_fields()
        """
        self._active_fields = [
            self._optional_fields[i]
            for i in selected_indices
            if i < len(self._optional_fields)
        ]
        self._current_index = 0
        if self._active_fields:
            self._phase = "optional_fill"
        else:
            self._phase = "done"
            self._done = True

    # ------------------------------------------------------------------
    # Chat mode
    # ------------------------------------------------------------------

    def deactivate_chat(self) -> None:
        """Return from chat mode to normal field entry."""
        self._chat_active = False

    def get_chat_context(self) -> str:
        """Build context string for /chat: schema name + filled fields + remaining fields.

        Sent to Flash Lite to give the LLM awareness of what has been answered (D-42).
        """
        parts = [f"Schema: {self.schema.name}"]
        parts.append(f"Vault folder: {self.schema.vault_folder}")
        parts.append("\nFilled fields:")
        for f in self.schema.fields:
            val = self.values.get(f.name)
            if val is not None:
                parts.append(f"  {f.name}: {val}")

        remaining = [
            f
            for f in self.schema.fields
            if f.name not in self.values or self.values[f.name] is None
        ]
        if remaining:
            parts.append("\nRemaining fields:")
            for f in remaining:
                parts.append(f"  {f.name}: ({f.prompt})")

        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Confirmation and note generation
    # ------------------------------------------------------------------

    def format_confirmation(self) -> str:
        """Format all filled fields for confirmation display before vault write (D-44)."""
        lines = [f"== {self.schema.name} Note ==", ""]
        for f in self.schema.fields:
            val = self.values.get(f.name)
            if val is not None:
                label = f.name.replace("_", " ").title()
                if "\n" in val:
                    lines.append(f"{label}:")
                    for line in val.split("\n"):
                        lines.append(f"  {line}")
                else:
                    lines.append(f"{label}: {val}")
        return "\n".join(lines)

    def generate_note_content(self) -> str:
        """Generate markdown note with YAML frontmatter for vault write.

        - Simple fields (text, select, number, date, tags) go in frontmatter
        - Multiline fields go in ## sections in the body
        """
        fm_lines = ["---"]
        fm_lines.append(f"type: {self.schema.name.lower()}")
        fm_lines.append(f"created: {datetime.utcnow().strftime('%Y-%m-%d')}")

        for f in self.schema.fields:
            val = self.values.get(f.name)
            if val is None:
                continue
            if f.field_type == "tags":
                tags = [t.strip() for t in val.split(",") if t.strip()]
                fm_lines.append(f"{f.name}: [{', '.join(tags)}]")
            elif f.field_type in ("text", "select", "number", "date"):
                fm_lines.append(f"{f.name}: {val}")
            # multiline fields go in body

        fm_lines.append("---")
        fm_lines.append("")

        # Title line
        title_val = self.values.get(self.schema.filename_field) or self.schema.name
        fm_lines.append(f"# {title_val}")
        fm_lines.append("")

        # Body: multiline fields as ## sections
        for f in self.schema.fields:
            val = self.values.get(f.name)
            if val is not None and f.field_type == "multiline":
                heading = f.name.replace("_", " ").title()
                fm_lines.append(f"## {heading}")
                fm_lines.append("")
                fm_lines.append(val)
                fm_lines.append("")

        return "\n".join(fm_lines)

    def get_vault_path(self) -> str:
        """Get the vault-relative path for this note.

        Sanitizes the filename field value to prevent path traversal
        (threat T-02-07-01): spaces become hyphens, only alphanumeric
        and - _ characters are kept, result is lowercased.
        """
        filename_val = self.values.get(self.schema.filename_field) or "untitled"
        # Sanitize: lowercase, spaces -> hyphens, keep only safe chars
        safe_name = filename_val.lower().replace(" ", "-")
        safe_name = "".join(c for c in safe_name if c.isalnum() or c in "-_")
        if not safe_name:
            safe_name = "untitled"
        return f"{self.schema.vault_folder}/{safe_name}.md"
