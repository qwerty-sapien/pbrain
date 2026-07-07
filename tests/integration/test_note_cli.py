"""Integration tests for note CLI commands (Plan 02-07).

Tests that:
- 'pb note --help' shows concept/person/book/opp subcommands
- Default schemas are created on disk
- All 4 default schemas load cleanly
- QuestionTreeEngine completes a full required-field flow
- Skipping all optional fields terminates cleanly
"""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from pb.cli.main import app
from pb.core.schemas import ensure_default_schemas, load_schema, NoteSchema
from pb.core.question_tree import QuestionTreeEngine


runner = CliRunner()


# ---------------------------------------------------------------------------
# CLI registration tests
# ---------------------------------------------------------------------------

def test_note_help():
    """'pb note --help' lists subcommands."""
    result = runner.invoke(app, ["note", "--help"])
    assert result.exit_code == 0, result.output
    output = result.output.lower()
    assert "concept" in output
    assert "person" in output
    assert "opp" in output


def test_note_concept_registered():
    """'pb note concept' subcommand is registered (variadic topic arg means --help goes through callback)."""
    from unittest.mock import patch
    with patch("pb.cli.commands.note._is_interactive", return_value=True), \
         patch("pb.cli.commands.note._run_question_tree"):
        result = runner.invoke(app, ["note", "--concept"])
    assert result.exit_code == 0, result.output


def test_note_person_registered():
    """'pb note --person' flag routes correctly."""
    from unittest.mock import patch
    with patch("pb.cli.commands.note._is_interactive", return_value=True), \
         patch("pb.cli.commands.note._run_question_tree"):
        result = runner.invoke(app, ["note", "--person"])
    assert result.exit_code == 0, result.output


def test_note_opp_registered():
    """'pb note --opp' flag routes correctly."""
    from unittest.mock import patch
    with patch("pb.cli.commands.note._is_interactive", return_value=True), \
         patch("pb.cli.commands.note._run_question_tree"):
        result = runner.invoke(app, ["note", "--opp"])
    assert result.exit_code == 0, result.output


# ---------------------------------------------------------------------------
# Schema creation tests
# ---------------------------------------------------------------------------

def test_concept_schema_created(tmp_path):
    """After ensure_default_schemas(tmp_path), concept.yaml exists."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    assert (schemas_dir / "concept.yaml").exists()


def test_all_four_default_schemas_created(tmp_path):
    """All 4 default schemas are created in the given directory."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    for name in ("concept", "person", "book", "opportunity"):
        assert (schemas_dir / f"{name}.yaml").exists(), f"Missing {name}.yaml"


# ---------------------------------------------------------------------------
# Schema loading tests
# ---------------------------------------------------------------------------

def test_load_all_default_schemas(tmp_path):
    """All 4 default schemas load and return NoteSchema instances."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    for schema_id in ("concept", "person", "book", "opportunity"):
        schema = load_schema(schema_id, schemas_dir)
        assert isinstance(schema, NoteSchema)
        assert len(schema.fields) > 0, f"{schema_id} has no fields"


def test_concept_schema_has_required_fields(tmp_path):
    """Concept schema has at least 3 required fields (title, domain, definition)."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("concept", schemas_dir)
    required_names = {f.name for f in schema.fields if f.required}
    assert "title" in required_names
    assert "domain" in required_names
    assert "definition" in required_names


def test_book_schema_status_field_has_options(tmp_path):
    """Book schema status field has select options."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("book", schemas_dir)
    status_fields = [f for f in schema.fields if f.name == "status"]
    assert status_fields, "Book schema missing 'status' field"
    assert len(status_fields[0].options) > 0


# ---------------------------------------------------------------------------
# QuestionTreeEngine full-flow tests
# ---------------------------------------------------------------------------

def test_question_tree_required_field_flow(tmp_path):
    """Feed all required concept fields and verify values stored correctly."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("concept", schemas_dir)
    engine = QuestionTreeEngine(schema)

    required = engine.get_required_fields()
    assert len(required) == 3  # title, domain, definition

    test_values = {
        "title": "Recursion",
        "domain": "Computer Science",
        "definition": "A function that calls itself with a base case.",
    }

    for f in required:
        result = engine.process_input(test_values[f.name])
        assert result["action"] in ("stored", "next_phase", "done"), f"Unexpected: {result}"

    assert engine.values.get("title") == "Recursion"
    assert engine.values.get("domain") == "Computer Science"
    assert engine.values.get("definition") == "A function that calls itself with a base case."


def test_question_tree_skip_all_optional(tmp_path):
    """After required fields, skipping all optional marks engine done."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("concept", schemas_dir)
    engine = QuestionTreeEngine(schema)

    # Fill required fields
    for f in engine.get_required_fields():
        engine.process_input(f"value for {f.name}")

    # Skip all optional
    engine.set_optional_selections([])
    assert engine.is_done is True


def test_question_tree_done_command_anywhere(tmp_path):
    """Sending /done at any point sets engine.is_done."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("concept", schemas_dir)
    engine = QuestionTreeEngine(schema)

    # /done immediately, even before filling required fields
    result = engine.process_input("/done")
    assert result["action"] == "done"
    assert engine.is_done is True


def test_question_tree_confirmation_format(tmp_path):
    """format_confirmation returns non-empty text after filling fields."""
    schemas_dir = tmp_path / "schemas"
    ensure_default_schemas(schemas_dir)
    schema = load_schema("concept", schemas_dir)
    engine = QuestionTreeEngine(schema)
    engine.values["title"] = "Memoization"
    engine.values["domain"] = "CS"

    confirmation = engine.format_confirmation()
    assert "Memoization" in confirmation
    assert "CS" in confirmation
    assert len(confirmation) > 10
