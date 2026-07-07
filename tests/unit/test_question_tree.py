"""Unit tests for schema system and question tree engine (Plan 02-07).

Tests cover:
- Schema loading (load_schema, ensure_default_schemas)
- SchemaField and NoteSchema dataclasses
- QuestionTreeEngine flow: required fields, optional fields
- /skip, /done, /chat commands
- Confirmation and note generation
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from pb.core.schemas import (
    DEFAULT_SCHEMAS,
    NoteSchema,
    SchemaField,
    ensure_default_schemas,
    list_schemas,
    load_schema,
)
from pb.core.question_tree import QuestionTreeEngine


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tmp_schemas_dir(tmp_path):
    """Provide a temporary schemas directory with all defaults installed."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    return schemas_dir


@pytest.fixture
def concept_engine(tmp_schemas_dir):
    """Return a QuestionTreeEngine loaded with the concept schema."""
    schema = load_schema("concept", tmp_schemas_dir)
    return QuestionTreeEngine(schema)


@pytest.fixture
def person_engine(tmp_schemas_dir):
    """Return a QuestionTreeEngine loaded with the person schema."""
    schema = load_schema("person", tmp_schemas_dir)
    return QuestionTreeEngine(schema)


# ---------------------------------------------------------------------------
# Schema tests
# ---------------------------------------------------------------------------

def test_schema_field_required_flag():
    """SchemaField with required=True is required; required=False is optional."""
    required_field = SchemaField(name="title", field_type="text", required=True, prompt="Title")
    optional_field = SchemaField(name="tags", field_type="tags", required=False, prompt="Tags")
    assert required_field.required is True
    assert optional_field.required is False


def test_load_schema_concept(tmp_schemas_dir):
    """load_schema('concept') returns a NoteSchema with field list."""
    schema = load_schema("concept", tmp_schemas_dir)
    assert isinstance(schema, NoteSchema)
    assert schema.name == "Concept"
    assert len(schema.fields) > 0
    assert schema.vault_folder == "20-concepts"
    assert schema.filename_field == "title"


def test_load_schema_all_defaults(tmp_schemas_dir):
    """All 4 default schemas load successfully."""
    for schema_id in ("concept", "person", "book", "opportunity"):
        schema = load_schema(schema_id, tmp_schemas_dir)
        assert isinstance(schema, NoteSchema)
        assert len(schema.fields) > 0


def test_load_schema_nonexistent_raises(tmp_schemas_dir):
    """load_schema('nonexistent') raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_schema("nonexistent", tmp_schemas_dir)


def test_ensure_default_schemas_creates_yaml_files(tmp_path):
    """ensure_default_schemas creates 4 YAML files in the given directory."""
    schemas_dir = tmp_path / "schemas"
    result_dir = ensure_default_schemas(schemas_dir)
    assert result_dir == schemas_dir
    for schema_id in ("concept", "person", "book", "opportunity"):
        assert (schemas_dir / f"{schema_id}.yaml").exists()


def test_ensure_default_schemas_idempotent(tmp_schemas_dir):
    """Calling ensure_default_schemas again does not error or overwrite existing files."""
    before_mtime = {
        p.name: p.stat().st_mtime for p in tmp_schemas_dir.glob("*.yaml")
    }
    ensure_default_schemas(tmp_schemas_dir)
    after_mtime = {
        p.name: p.stat().st_mtime for p in tmp_schemas_dir.glob("*.yaml")
    }
    # Files should not have been modified (or may be same; at minimum no error)
    for name in before_mtime:
        assert before_mtime[name] == after_mtime.get(name), f"{name} was overwritten"


def test_list_schemas_returns_all_ids(tmp_schemas_dir):
    """list_schemas returns at least 4 schema IDs."""
    ids = list_schemas(tmp_schemas_dir)
    assert set(["concept", "person", "book", "opportunity"]).issubset(set(ids))


def test_schema_uses_safe_load(tmp_schemas_dir):
    """Schema YAML files use yaml.safe_load (threat T-02-07-03)."""
    # Verify no !!python tags were written (safe_dump only)
    for schema_file in tmp_schemas_dir.glob("*.yaml"):
        content = schema_file.read_text()
        assert "!!python" not in content, f"{schema_file.name} contains unsafe YAML tag"


def test_default_schemas_dict_has_required_keys():
    """DEFAULT_SCHEMAS has all expected keys."""
    assert set(DEFAULT_SCHEMAS.keys()) == {"concept", "person", "book", "opportunity", "event", "routine"}


# ---------------------------------------------------------------------------
# QuestionTreeEngine: required / optional field separation
# ---------------------------------------------------------------------------

def test_get_required_fields(concept_engine):
    """get_required_fields returns only fields where required=True."""
    required = concept_engine.get_required_fields()
    assert all(f.required for f in required)
    assert len(required) > 0


def test_get_optional_fields(concept_engine):
    """get_optional_fields returns only fields where required=False."""
    optional = concept_engine.get_optional_fields()
    assert all(not f.required for f in optional)
    assert len(optional) > 0


def test_required_and_optional_are_disjoint(concept_engine):
    """Required and optional field sets have no overlap."""
    required_names = {f.name for f in concept_engine.get_required_fields()}
    optional_names = {f.name for f in concept_engine.get_optional_fields()}
    assert required_names.isdisjoint(optional_names)


# ---------------------------------------------------------------------------
# QuestionTreeEngine: process_input normal flow
# ---------------------------------------------------------------------------

def test_process_input_stores_value(concept_engine):
    """process_input with text stores the value for the current field."""
    field = concept_engine.current_field()
    assert field is not None
    result = concept_engine.process_input("Recursion")
    assert result["action"] in ("stored", "next_phase", "done")
    assert concept_engine.values.get(field.name) == "Recursion"


def test_process_input_advances_field(concept_engine):
    """After storing a value, current_field advances to next field."""
    first_field = concept_engine.current_field()
    concept_engine.process_input("Some value")
    second_field = concept_engine.current_field()
    assert first_field is not None
    # Either advanced to next or moved to next phase
    if second_field is not None:
        assert second_field.name != first_field.name


# ---------------------------------------------------------------------------
# QuestionTreeEngine: /skip command
# ---------------------------------------------------------------------------

def test_process_input_skip_required_returns_error(concept_engine):
    """/skip on a required field returns action='error' with informative message."""
    field = concept_engine.current_field()
    assert field is not None and field.required
    result = concept_engine.process_input("/skip")
    assert result["action"] == "error"
    assert "Cannot skip required" in result.get("message", "")


def test_process_input_skip_optional(tmp_schemas_dir):
    """/skip on an optional field advances past it, setting value to None."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)

    # Fill all required fields so we reach optional phase
    for f in engine.get_required_fields():
        if f.field_type == "select":
            engine.process_input(f.options[0])
        else:
            engine.process_input(f"value for {f.name}")

    # Now select all optional fields for filling
    optional = engine.get_optional_fields()
    if optional:
        engine.set_optional_selections(list(range(len(optional))))
        first_optional = engine.current_field()
        if first_optional is not None:
            result = engine.process_input("/skip")
            # /skip on optional sets value to None and advances
            assert result["action"] != "error"
            assert engine.values.get(first_optional.name) is None


# ---------------------------------------------------------------------------
# QuestionTreeEngine: /done command
# ---------------------------------------------------------------------------

def test_process_input_done_sets_is_done(concept_engine):
    """/done sets engine.is_done to True."""
    assert concept_engine.is_done is False
    result = concept_engine.process_input("/done")
    assert result["action"] == "done"
    assert result["done"] is True
    assert concept_engine.is_done is True


# ---------------------------------------------------------------------------
# QuestionTreeEngine: /chat command
# ---------------------------------------------------------------------------

def test_process_input_chat_activates(concept_engine):
    """/chat sets chat_active=True and returns chat_activated action."""
    result = concept_engine.process_input("/chat")
    assert result["action"] == "chat_activated"
    assert result["chat_active"] is True
    assert concept_engine.chat_active is True


def test_deactivate_chat(concept_engine):
    """deactivate_chat() turns off chat mode."""
    concept_engine.process_input("/chat")
    assert concept_engine.chat_active is True
    concept_engine.deactivate_chat()
    assert concept_engine.chat_active is False


# ---------------------------------------------------------------------------
# QuestionTreeEngine: confirmation and note generation
# ---------------------------------------------------------------------------

def test_format_confirmation_shows_filled_fields(tmp_schemas_dir):
    """format_confirmation() returns text containing all filled field names/values."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)

    # Fill a couple of required fields manually
    engine.values["title"] = "Recursion"
    engine.values["domain"] = "CS"

    confirmation = engine.format_confirmation()
    assert "Recursion" in confirmation
    assert "CS" in confirmation


def test_format_confirmation_excludes_empty_fields(tmp_schemas_dir):
    """format_confirmation() does not show fields with None values."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Recursion"
    # Leave domain as None (not set)
    confirmation = engine.format_confirmation()
    assert "Recursion" in confirmation
    # domain should not appear with a None label
    assert "domain: None" not in confirmation.lower()
    assert "Domain: None" not in confirmation


def test_generate_note_content_has_frontmatter(tmp_schemas_dir):
    """generate_note_content() returns content starting with --- frontmatter."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Recursion"
    engine.values["domain"] = "CS"
    engine.values["definition"] = "A function that calls itself."

    content = engine.generate_note_content()
    assert content.startswith("---")
    assert "---" in content[3:]  # closing ---
    assert "type: concept" in content
    assert "# Recursion" in content


def test_generate_note_content_multiline_in_body(tmp_schemas_dir):
    """Multiline fields appear in body sections, not frontmatter."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Recursion"
    engine.values["domain"] = "CS"
    engine.values["definition"] = "A function calls itself.\nBase case stops it."

    content = engine.generate_note_content()
    # definition is multiline, should be in body as ## section
    assert "## Definition" in content
    assert "A function calls itself." in content


def test_get_vault_path_sanitizes_filename(tmp_schemas_dir):
    """get_vault_path() sanitizes spaces and special chars in filename."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Big O Notation!"
    path = engine.get_vault_path()
    # Must be within vault_folder
    assert path.startswith("20-concepts/")
    # Must not contain raw spaces or exclamation marks
    assert " " not in path
    assert "!" not in path


def test_get_chat_context_includes_filled_fields(tmp_schemas_dir):
    """get_chat_context() returns context with schema name and filled fields."""
    schema = load_schema("concept", tmp_schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Recursion"

    context = engine.get_chat_context()
    assert "Concept" in context
    assert "Recursion" in context


# ---------------------------------------------------------------------------
# Progress text
# ---------------------------------------------------------------------------

def test_progress_text_during_required_phase(concept_engine):
    """progress_text returns [N/M] format during required field phase."""
    text = concept_engine.progress_text()
    assert text.startswith("[")
    assert "/" in text
    assert text.endswith("]")
