# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""YAML schema system for structured note types (D-45, D-46).

Schemas are stored in ~/.config/pb/schemas/*.yaml.
Each schema defines fields with name, type, required flag, and prompt text.

Security: yaml.safe_load is used to prevent code execution from schema files
(threat T-02-07-03).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml

SCHEMAS_DIR = Path.home() / ".config" / "pb" / "schemas"


@dataclass
class SchemaField:
    """A single field definition within a note schema."""

    name: str
    field_type: str  # text, multiline, tags, number, date, select
    required: bool = False
    prompt: str = ""
    description: str = ""
    options: list = field(default_factory=list)  # for select type
    default: Optional[str] = None


@dataclass
class NoteSchema:
    """A complete note type schema."""

    name: str
    vault_folder: str  # e.g., "20-concepts", "people"
    filename_field: str  # which field becomes the filename
    fields: list = field(default_factory=list)  # list[SchemaField]


# V1 default schemas per D-46
DEFAULT_SCHEMAS = {
    "concept": {
        "name": "Concept",
        "vault_folder": "20-concepts",
        "filename_field": "title",
        "fields": [
            {"name": "title", "type": "text", "required": True, "prompt": "Concept name"},
            {"name": "domain", "type": "text", "required": True, "prompt": "Domain (e.g., math, CS, physics)"},
            {"name": "definition", "type": "multiline", "required": True, "prompt": "Definition (your understanding, not Wikipedia)"},
            {"name": "related_concepts", "type": "tags", "required": False, "prompt": "Related concepts (comma-separated)"},
            {"name": "prerequisites", "type": "tags", "required": False, "prompt": "Prerequisites (comma-separated)"},
            {"name": "source", "type": "text", "required": False, "prompt": "Where did you learn this?"},
            {"name": "intuition", "type": "multiline", "required": False, "prompt": "Intuitive explanation or analogy"},
            {"name": "examples", "type": "multiline", "required": False, "prompt": "Key examples"},
            {"name": "open_questions", "type": "multiline", "required": False, "prompt": "What's still unclear?"},
        ],
    },
    "person": {
        "name": "Person",
        "vault_folder": "people",
        "filename_field": "name",
        "fields": [
            {"name": "name", "type": "text", "required": True, "prompt": "Full name"},
            {"name": "context", "type": "text", "required": True, "prompt": "How do you know them?"},
            {"name": "relationship_type", "type": "select", "required": True,
             "prompt": "Relationship type",
             "options": ["friend", "acquaintance", "family", "colleague", "mentor",
                         "mentee", "partner", "ex-partner", "community",
                         "network-node", "confidant", "overseas-contacts"]},
            {"name": "birthday", "type": "date", "required": False, "prompt": "Birthday (YYYY-MM-DD)"},
            {"name": "contact_cadence", "type": "select", "required": False,
             "prompt": "Contact cadence",
             "options": ["weekly", "biweekly", "monthly", "quarterly", "none"]},
            {"name": "organization", "type": "text", "required": False, "prompt": "Organization / company"},
            {"name": "role", "type": "text", "required": False, "prompt": "Their role / what they do"},
            {"name": "location", "type": "text", "required": False, "prompt": "Location"},
            {"name": "email", "type": "text", "required": False, "prompt": "Email"},
            {"name": "phone", "type": "text", "required": False, "prompt": "Phone"},
            {"name": "notes", "type": "multiline", "required": False, "prompt": "Notes about this person"},
            {"name": "tags", "type": "tags", "required": False, "prompt": "Tags (comma-separated)"},
        ],
    },
    "book": {
        "name": "Book",
        "vault_folder": "20-concepts/books",
        "filename_field": "title",
        "fields": [
            {"name": "title", "type": "text", "required": True, "prompt": "Book title"},
            {"name": "author", "type": "text", "required": True, "prompt": "Author"},
            {"name": "status", "type": "select", "required": True, "prompt": "Reading status", "options": ["to-read", "reading", "finished", "abandoned"]},
            {"name": "rating", "type": "number", "required": False, "prompt": "Rating (1-5)"},
            {"name": "key_ideas", "type": "multiline", "required": False, "prompt": "Key ideas / takeaways"},
            {"name": "quotes", "type": "multiline", "required": False, "prompt": "Notable quotes"},
            {"name": "tags", "type": "tags", "required": False, "prompt": "Tags (comma-separated)"},
        ],
    },
    "event": {
        "name": "Event",
        "vault_folder": "events/upcoming",
        "filename_field": "title",
        "fields": [
            {"name": "title", "type": "text", "required": True, "prompt": "Event title"},
            {"name": "date", "type": "date", "required": True, "prompt": "Event date (YYYY-MM-DD)"},
            {"name": "end_date", "type": "date", "required": False, "prompt": "End date (YYYY-MM-DD)"},
            {"name": "location", "type": "text", "required": True, "prompt": "Location"},
            {"name": "category", "type": "select", "required": True, "prompt": "Category",
             "options": ["data-science", "engineering", "business", "social", "fitness",
                         "arts", "community", "career", "learning", "other"]},
            {"name": "subcategory", "type": "text", "required": False, "prompt": "Subcategory"},
            {"name": "price", "type": "text", "required": False, "prompt": "Price (e.g., free, $20)"},
            {"name": "organizer", "type": "text", "required": False, "prompt": "Organizer"},
            {"name": "capacity", "type": "number", "required": False, "prompt": "Capacity"},
            {"name": "source_url", "type": "text", "required": False, "prompt": "Event URL"},
            {"name": "tags", "type": "tags", "required": False, "prompt": "Tags (comma-separated)"},
            {"name": "people", "type": "tags", "required": False, "prompt": "People attending (comma-separated names)"},
            {"name": "details", "type": "multiline", "required": False, "prompt": "Event details"},
        ],
    },
    "routine": {
        "name": "Routine",
        "vault_folder": "events/routines",
        "filename_field": "title",
        "fields": [
            {"name": "title", "type": "text", "required": True, "prompt": "Routine name"},
            {"name": "cadence", "type": "select", "required": True, "prompt": "Cadence",
             "options": ["weekly", "biweekly", "monthly"]},
            {"name": "day", "type": "select", "required": True, "prompt": "Day",
             "options": ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]},
            {"name": "time", "type": "text", "required": False, "prompt": "Time (e.g., 18:00)"},
            {"name": "location", "type": "text", "required": False, "prompt": "Location"},
            {"name": "category", "type": "select", "required": True, "prompt": "Category",
             "options": ["data-science", "engineering", "business", "social", "fitness",
                         "arts", "community", "career", "learning", "other"]},
            {"name": "organizer", "type": "text", "required": False, "prompt": "Organizer"},
            {"name": "tags", "type": "tags", "required": False, "prompt": "Tags (comma-separated)"},
            {"name": "people", "type": "tags", "required": False, "prompt": "Regular attendees (comma-separated)"},
            {"name": "details", "type": "multiline", "required": False, "prompt": "Routine details"},
        ],
    },
    "opportunity": {
        "name": "Opportunity",
        "vault_folder": "opportunities/active",
        "filename_field": "title",
        "fields": [
            {"name": "title", "type": "text", "required": True, "prompt": "Opportunity title"},
            {"name": "opp_type", "type": "select", "required": True, "prompt": "Opportunity type",
             "options": ["competition", "learning", "promo"]},
            {"name": "deadline", "type": "date", "required": False, "prompt": "Deadline (YYYY-MM-DD)"},
            {"name": "platform", "type": "text", "required": False, "prompt": "Platform (e.g., Kaggle, Coursera)"},
            {"name": "source_url", "type": "text", "required": False, "prompt": "URL"},
            {"name": "skills_required", "type": "tags", "required": False, "prompt": "Skills required (comma-separated)"},
            {"name": "prize", "type": "text", "required": False, "prompt": "Prize / reward"},
            {"name": "company", "type": "text", "required": False, "prompt": "Company / organization"},
            {"name": "description", "type": "multiline", "required": False, "prompt": "Description"},
            {"name": "requirements", "type": "multiline", "required": False, "prompt": "Requirements"},
            {"name": "tags", "type": "tags", "required": False, "prompt": "Tags (comma-separated)"},
        ],
    },
}


def ensure_default_schemas(schemas_dir: Optional[Path] = None) -> Path:
    """Create default schema YAML files if they don't exist. Returns schemas dir.

    Uses yaml.dump (safe by default) to prevent writing executable YAML.
    Does not overwrite existing files to preserve user customisations.
    """
    schemas_dir = schemas_dir or SCHEMAS_DIR
    schemas_dir.mkdir(parents=True, exist_ok=True)

    for schema_id, schema_data in DEFAULT_SCHEMAS.items():
        schema_path = schemas_dir / f"{schema_id}.yaml"
        if not schema_path.exists():
            yaml_content = {
                "name": schema_data["name"],
                "vault_folder": schema_data["vault_folder"],
                "filename_field": schema_data["filename_field"],
                "fields": schema_data["fields"],
            }
            with open(schema_path, "w") as f:
                yaml.dump(yaml_content, f, default_flow_style=False, sort_keys=False)

    return schemas_dir


def load_schema(schema_id: str, schemas_dir: Optional[Path] = None) -> NoteSchema:
    """Load a schema from YAML file. Raises FileNotFoundError if not found.

    Uses yaml.safe_load to prevent code execution from schema YAML
    (threat T-02-07-03).
    """
    schemas_dir = schemas_dir or SCHEMAS_DIR
    schema_path = schemas_dir / f"{schema_id}.yaml"

    if not schema_path.exists():
        raise FileNotFoundError(f"Schema not found: {schema_path}")

    with open(schema_path) as f:
        data = yaml.safe_load(f)

    fields = []
    for fd in data.get("fields", []):
        fields.append(
            SchemaField(
                name=fd["name"],
                field_type=fd.get("type", "text"),
                required=fd.get("required", False),
                prompt=fd.get("prompt", fd["name"]),
                description=fd.get("description", ""),
                options=fd.get("options", []),
                default=fd.get("default"),
            )
        )

    return NoteSchema(
        name=data["name"],
        vault_folder=data["vault_folder"],
        filename_field=data["filename_field"],
        fields=fields,
    )


def list_schemas(schemas_dir: Optional[Path] = None) -> list:
    """List available schema IDs."""
    schemas_dir = schemas_dir or SCHEMAS_DIR
    if not schemas_dir.exists():
        return []
    return [p.stem for p in sorted(schemas_dir.glob("*.yaml"))]


# Phase 4 fields for person schema extension (D-02, D-06, D-19)
_PERSON_PHASE4_FIELDS = [
    {
        "name": "relationship_type",
        "type": "select",
        "required": True,
        "prompt": "Relationship type",
        "options": [
            "friend", "acquaintance", "family", "colleague", "mentor",
            "mentee", "partner", "ex-partner", "community",
            "network-node", "confidant", "overseas-contacts",
        ],
    },
    {
        "name": "birthday",
        "type": "date",
        "required": False,
        "prompt": "Birthday (YYYY-MM-DD)",
    },
    {
        "name": "contact_cadence",
        "type": "select",
        "required": False,
        "prompt": "Contact cadence",
        "options": ["weekly", "biweekly", "monthly", "quarterly", "none"],
    },
]


def ensure_person_schema_fields(schemas_dir: Optional[Path] = None) -> None:
    """Merge Phase 4 fields into existing person.yaml without overwriting (Pitfall 7).

    Reads existing YAML, checks for missing fields, inserts after 'context' field.
    Uses yaml.safe_load for read and yaml.dump for write (T-04-04).
    Idempotent: calling multiple times does not duplicate fields.
    """
    schemas_dir = schemas_dir or SCHEMAS_DIR
    person_path = schemas_dir / "person.yaml"
    if not person_path.exists():
        return  # ensure_default_schemas() handles creation

    data = yaml.safe_load(person_path.read_text()) or {}
    existing_names = {f["name"] for f in data.get("fields", [])}

    fields = data.get("fields", [])
    # Find insertion point: after "context" field
    insert_idx = next(
        (i + 1 for i, f in enumerate(fields) if f["name"] == "context"),
        len(fields),
    )

    for nf in reversed(_PERSON_PHASE4_FIELDS):
        if nf["name"] not in existing_names:
            fields.insert(insert_idx, nf)

    data["fields"] = fields
    with open(person_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)
