# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Schema MCP tools for managing question tree definitions.

Question trees define the guided note creation flow for each note type.
Schemas are stored as YAML files in the vault's schemas/ folder.

Tools:
- schema_get: Get question tree for a note type
- schema_update: Update question tree for a note type
"""

import json
from pathlib import Path
from typing import Optional

import yaml

from pb.mcp.context import get_mcp_context
from pb.mcp.server import mcp
from pb.vault import get_vault_path, ensure_vault_folder


class SchemaError(Exception):
    """Raised when schema operations fail."""
    pass


def _require_writes() -> None:
    if not get_mcp_context().allow_writes:
        raise SchemaError("This MCP server is running in read-only mode. Restart with --allow-writes.")


# Default schemas for note types (from ROADMAP.md Phase 2)
# These are used when no custom schema exists
DEFAULT_SCHEMAS = {
    "person": {
        "name": "Person",
        "fields": [
            {"name": "name", "required": True, "prompt": "Name?"},
            {"name": "relationship", "required": True, "prompt": "Relationship?"},
            {"name": "met_context", "required": False, "prompt": "How did you meet?"},
            {"name": "work_role", "required": False, "prompt": "Work/role?"},
            {"name": "interests", "required": False, "prompt": "Interests?"},
            {"name": "contact_cadence", "required": False, "prompt": "Contact cadence?"},
            {"name": "gift_preferences", "required": False, "prompt": "Gift preferences?"},
        ],
    },
    "event": {
        "name": "Event",
        "fields": [
            {"name": "name", "required": True, "prompt": "Event name?"},
            {"name": "date", "required": True, "prompt": "Date?"},
            {"name": "category", "required": False, "prompt": "Category?"},
            {"name": "venue", "required": False, "prompt": "Venue?"},
            {"name": "notes", "required": False, "prompt": "Notes?"},
        ],
    },
    "concept": {
        "name": "Concept",
        "fields": [
            {"name": "title", "required": True, "prompt": "Concept title?"},
            {"name": "domain", "required": False, "prompt": "Domain?"},
            {"name": "definition", "required": True, "prompt": "One-line definition (<30 words)?"},
            {"name": "source", "required": False, "prompt": "Source?"},
            {"name": "related_concepts", "required": False, "prompt": "Related concepts?"},
        ],
    },
    "goal": {
        "name": "Goal",
        "fields": [
            {"name": "title", "required": True, "prompt": "Goal title?"},
            {"name": "aspiration", "required": True, "prompt": "Parent aspiration?"},
            {"name": "target", "required": True, "prompt": "Measurable target?"},
            {"name": "deadline", "required": False, "prompt": "Deadline?"},
            {"name": "track", "required": False, "prompt": "Track?"},
        ],
    },
}


def _get_schemas_dir(vault_path: Optional[Path] = None, *, create: bool = False) -> Path:
    """Get the schemas directory path."""
    if vault_path is None:
        vault_path = get_vault_path()
    schemas_dir = vault_path / "schemas"
    if create:
        schemas_dir.mkdir(parents=True, exist_ok=True)
    return schemas_dir


def _get_schema_path(note_type: str, vault_path: Optional[Path] = None) -> Path:
    """Get the path to a schema file."""
    schemas_dir = _get_schemas_dir(vault_path)
    return schemas_dir / f"{note_type}.yaml"


@mcp.tool()
def schema_get(note_type: str) -> str:
    """Get the question tree schema for a note type.

    Args:
        note_type: Type of note (e.g., 'person', 'event', 'concept', 'goal')

    Returns:
        JSON-formatted schema with fields and prompts
    """
    vault_path = get_vault_path()
    schema_path = _get_schema_path(note_type, vault_path)

    # Try to load custom schema
    if schema_path.exists():
        try:
            with open(schema_path) as f:
                schema = yaml.safe_load(f)
            return json.dumps({"note_type": note_type, "schema": schema, "source": "custom"})
        except yaml.YAMLError as e:
            raise SchemaError(f"Invalid YAML in schema file: {e}")

    # Fall back to default schema
    if note_type in DEFAULT_SCHEMAS:
        return json.dumps({
            "note_type": note_type,
            "schema": DEFAULT_SCHEMAS[note_type],
            "source": "default",
        })

    # No schema found
    return json.dumps({
        "note_type": note_type,
        "schema": None,
        "source": "none",
        "available_types": list(DEFAULT_SCHEMAS.keys()),
    })


@mcp.tool()
def schema_update(note_type: str, schema_yaml: str) -> str:
    """Update the question tree schema for a note type.

    Args:
        note_type: Type of note (e.g., 'person', 'event', 'concept')
        schema_yaml: YAML-formatted schema definition

    Returns:
        Confirmation message
    """
    # Validate YAML
    try:
        schema = yaml.safe_load(schema_yaml)
    except yaml.YAMLError as e:
        raise SchemaError(f"Invalid YAML: {e}")

    # Validate schema structure
    if not isinstance(schema, dict):
        raise SchemaError("Schema must be a YAML object/dict")

    if "name" not in schema:
        raise SchemaError("Schema must have a 'name' field")

    if "fields" not in schema or not isinstance(schema["fields"], list):
        raise SchemaError("Schema must have a 'fields' list")

    for i, field in enumerate(schema["fields"]):
        if not isinstance(field, dict):
            raise SchemaError(f"Field {i} must be an object")
        if "name" not in field:
            raise SchemaError(f"Field {i} missing 'name'")
        if "prompt" not in field:
            raise SchemaError(f"Field {i} missing 'prompt'")

    _require_writes()

    # Write schema file
    vault_path = get_vault_path()
    schema_path = _get_schema_path(note_type, vault_path)
    schema_path.parent.mkdir(parents=True, exist_ok=True)

    with open(schema_path, "w") as f:
        yaml.dump(schema, f, default_flow_style=False, allow_unicode=True)

    return f"Updated schema for '{note_type}' at schemas/{note_type}.yaml"
