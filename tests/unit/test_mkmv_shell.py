"""Unit tests for mkmv gap closure fixes in shell.py.

Tests cover:
- _handle_mkmv shows explicit API-unavailable message when Gemini is down (AISG-04)
- _mkmv_collection writes .pb-directory.md after creating collection note (AISG-06)
- _mkmv_collection places note in top-level vault folder via rank_folder or manual picker (D-06, D-10)
- _mkmv_collection falls back to _pick_numbered when rank_folder returns None
- _mkmv_collection aborts when _pick_numbered returns None
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch, MagicMock, call

import pytest

from pb.cli.shell import _handle_mkmv, _mkmv_collection


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(available: bool = True) -> MagicMock:
    """Return a mock GeminiClient with configurable is_available()."""
    client = MagicMock()
    client.is_available.return_value = available
    return client


# ---------------------------------------------------------------------------
# _handle_mkmv tests
# ---------------------------------------------------------------------------


def test_handle_mkmv_prints_api_warning_when_unavailable(tmp_path, capsys):
    """When get_client().is_available() is False, _handle_mkmv prints the API-unavailable message."""
    # Create a vault with a matching note
    projects = tmp_path / "projects"
    projects.mkdir()
    (projects / "note.md").write_text("# Test Note\n", encoding="utf-8")

    mock_mkmv = MagicMock()
    mock_mkmv.find_matching_notes.return_value = []  # empty so we exit early

    with patch("pb.cli.shell.get_client", return_value=_make_client(available=False)), \
         patch("pb.cli.shell.MkMvEngine", return_value=mock_mkmv):
        _handle_mkmv(["test", "topic"], tmp_path, [tmp_path])

    captured = capsys.readouterr()
    assert "AI features unavailable" in captured.out


def test_handle_mkmv_proceeds_to_search_when_api_down(tmp_path, capsys):
    """Even when API is unavailable, _handle_mkmv still calls find_matching_notes (filesystem search)."""
    projects = tmp_path / "projects"
    projects.mkdir()
    note = projects / "note.md"
    note.write_text("# Test Note\n", encoding="utf-8")

    mock_mkmv = MagicMock()
    mock_mkmv.find_matching_notes.return_value = [(note, "snippet")]
    mock_mkmv.ai_filter_notes.return_value = [note]

    with patch("pb.cli.shell.get_client", return_value=_make_client(available=False)), \
         patch("pb.cli.shell.MkMvEngine", return_value=mock_mkmv), \
         patch("pb.cli.shell.tier2_confirm", return_value=False):
        _handle_mkmv(["test", "topic"], tmp_path, [tmp_path])

    # find_matching_notes must have been called (filesystem search runs offline)
    mock_mkmv.find_matching_notes.assert_called_once()


def test_handle_mkmv_no_api_warning_when_available(tmp_path, capsys):
    """When API is available, the API-unavailable message should NOT appear."""
    mock_mkmv = MagicMock()
    mock_mkmv.find_matching_notes.return_value = []

    with patch("pb.cli.shell.get_client", return_value=_make_client(available=True)), \
         patch("pb.cli.shell.MkMvEngine", return_value=mock_mkmv):
        _handle_mkmv(["test"], tmp_path, [tmp_path])

    captured = capsys.readouterr()
    assert "AI features unavailable" not in captured.out


# ---------------------------------------------------------------------------
# _mkmv_collection tests
# ---------------------------------------------------------------------------


def test_mkmv_collection_writes_directory_md(tmp_path):
    """After _mkmv_collection writes the collection note, .pb-directory.md is created in the target folder."""
    projects = tmp_path / "projects"
    projects.mkdir()

    # Create a dummy note to include in the collection
    source_note = projects / "existing-note.md"
    source_note.write_text("# Existing Note\n", encoding="utf-8")

    with patch("pb.cli.shell.tier2_confirm", return_value=True), \
         patch("pb.cli.shell._list_dirs", return_value=[projects]), \
         patch("pb.cli.shell.MkMvEngine") as mock_engine_cls, \
         patch("pb.cli.shell.update_note_in_graph"), \
         patch("pb.cli.shell.rebuild_folder_index"), \
         patch("pb.cli.shell.generate_directory_md", return_value="# Projects directory") as mock_gen_dir:

        mock_engine = mock_engine_cls.return_value
        mock_engine.rank_folder.return_value = "projects"

        _mkmv_collection([source_note], "test topic", tmp_path, [tmp_path])

    # The .pb-directory.md file must exist in the target folder (projects/)
    dir_md = projects / ".pb-directory.md"
    assert dir_md.exists(), ".pb-directory.md should be written to the target folder"
    assert dir_md.read_text(encoding="utf-8") == "# Projects directory"


def test_mkmv_collection_uses_ranked_folder_not_cwd(tmp_path):
    """_mkmv_collection places the collection note in the AI-ranked folder, not _cwd_ref[0]."""
    projects = tmp_path / "projects"
    projects.mkdir()
    archive = tmp_path / "archive"
    archive.mkdir()

    source_note = projects / "note.md"
    source_note.write_text("# Note\n", encoding="utf-8")

    # User's cwd is the archive subfolder -- collection note must NOT go there
    cwd_ref = [archive]

    with patch("pb.cli.shell.tier2_confirm", return_value=True), \
         patch("pb.cli.shell._list_dirs", return_value=[projects, archive]), \
         patch("pb.cli.shell.MkMvEngine") as mock_engine_cls, \
         patch("pb.cli.shell.update_note_in_graph"), \
         patch("pb.cli.shell.rebuild_folder_index"), \
         patch("pb.cli.shell.generate_directory_md", return_value="# dir"):

        mock_engine = mock_engine_cls.return_value
        mock_engine.rank_folder.return_value = "projects"

        _mkmv_collection([source_note], "test topic", tmp_path, cwd_ref)

    # Collection note must be in projects/, not archive/
    archive_files = list(archive.glob("*.md"))
    assert archive_files == [], f"No .md files should exist in archive/; found: {archive_files}"

    projects_files = [f for f in projects.glob("*.md") if f.name != "note.md"]
    assert len(projects_files) >= 1, "Collection note should be in projects/"


def test_mkmv_collection_falls_back_to_picker_when_no_rank(tmp_path):
    """When rank_folder returns None, _mkmv_collection uses _pick_numbered as fallback."""
    projects = tmp_path / "projects"
    projects.mkdir()

    source_note = projects / "note.md"
    source_note.write_text("# Note\n", encoding="utf-8")

    with patch("pb.cli.shell.tier2_confirm", return_value=True), \
         patch("pb.cli.shell._list_dirs", return_value=[projects]), \
         patch("pb.cli.shell.MkMvEngine") as mock_engine_cls, \
         patch("pb.cli.shell._pick_numbered", return_value=projects) as mock_picker, \
         patch("pb.cli.shell.update_note_in_graph"), \
         patch("pb.cli.shell.rebuild_folder_index"), \
         patch("pb.cli.shell.generate_directory_md", return_value="# dir"):

        mock_engine = mock_engine_cls.return_value
        mock_engine.rank_folder.return_value = None

        _mkmv_collection([source_note], "test topic", tmp_path, [tmp_path])

    # _pick_numbered must have been called (manual fallback when AI rank unavailable)
    mock_picker.assert_called_once()

    # Collection note should exist in projects/
    projects_files = [f for f in projects.glob("*.md") if f.name != "note.md"]
    assert len(projects_files) >= 1, "Collection note should be in picker-selected folder"


def test_mkmv_collection_aborts_when_picker_returns_none(tmp_path):
    """When _pick_numbered returns None (user cancelled), no file should be written."""
    projects = tmp_path / "projects"
    projects.mkdir()

    source_note = projects / "note.md"
    source_note.write_text("# Note\n", encoding="utf-8")

    with patch("pb.cli.shell.tier2_confirm", return_value=True), \
         patch("pb.cli.shell._list_dirs", return_value=[projects]), \
         patch("pb.cli.shell.MkMvEngine") as mock_engine_cls, \
         patch("pb.cli.shell._pick_numbered", return_value=None):

        mock_engine = mock_engine_cls.return_value
        mock_engine.rank_folder.return_value = None

        _mkmv_collection([source_note], "test topic", tmp_path, [tmp_path])

    # No additional .md files should be created -- only the pre-existing note.md
    all_md = list(tmp_path.rglob("*.md"))
    assert all_md == [source_note], f"Expected only source_note to exist; found: {all_md}"
