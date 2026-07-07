"""Unit tests for SuggestionEngine, MkMvEngine, tier2_confirm, and get_recent_commands.

Tests cover:
- SuggestionEngine.suggest: returns (command, explanation) tuple or None
- SuggestionEngine._build_context: assembles context from active task, vault cwd, usage log
- get_recent_commands: reads usage_log ordered by timestamp DESC
- tier2_confirm: Y/n inline confirmation
- MkMvEngine.find_matching_notes: searches vault for notes matching a topic
- MkMvEngine.ai_filter_notes: Flash Lite filters relevant notes from candidates
- MkMvEngine.rank_folder: folder ranking with depth guard and validation
- AVAILABLE_COMMANDS: static list
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mock_client(available: bool = True, generate_result=None):
    """Create a mock GeminiClient."""
    client = MagicMock()
    client.is_available.return_value = available
    client.generate_with_model.return_value = generate_result
    return client


def _make_suggestion_engine(client=None, active_task=None, vault_cwd=None, vault_root=None):
    """Create a SuggestionEngine with mocked dependencies."""
    from pb.core.suggestions import SuggestionEngine

    repo = MagicMock()
    repo.get_active_task.return_value = active_task

    get_vault_cwd = (lambda: vault_cwd) if vault_cwd is not None else None

    with patch("pb.core.suggestions.get_client", return_value=client or _make_mock_client()):
        engine = SuggestionEngine(
            repo=repo,
            get_vault_cwd=get_vault_cwd,
            vault_root=vault_root,
        )
    return engine


# ---------------------------------------------------------------------------
# SuggestionEngine.suggest tests
# ---------------------------------------------------------------------------


def test_suggest_returns_tuple_when_client_available():
    """suggest returns (command, explanation) tuple when client is available and model responds."""
    client = _make_mock_client(available=True, generate_result="pb start my-task\nStart working on your active task")
    engine = _make_suggestion_engine(client=client)

    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.suggest("start my task")

    assert result is not None
    assert isinstance(result, tuple)
    assert len(result) == 2
    command, explanation = result
    assert command == "pb start my-task"
    assert "Start working" in explanation


def test_suggest_returns_none_when_client_unavailable():
    """suggest returns None when client is not available."""
    client = _make_mock_client(available=False)
    engine = _make_suggestion_engine(client=client)

    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.suggest("start my task")

    assert result is None


def test_suggest_returns_none_when_generate_returns_none():
    """suggest returns None when generate_with_model returns None."""
    client = _make_mock_client(available=True, generate_result=None)
    engine = _make_suggestion_engine(client=client)

    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.suggest("start my task")

    assert result is None


def test_suggest_returns_none_when_generate_returns_empty_string():
    """suggest returns None when generate_with_model returns empty string."""
    client = _make_mock_client(available=True, generate_result="")
    engine = _make_suggestion_engine(client=client)

    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.suggest("start my task")

    assert result is None


# ---------------------------------------------------------------------------
# SuggestionEngine._build_context tests
# ---------------------------------------------------------------------------


def test_build_context_includes_active_task():
    """_build_context includes 'Active task:' when repo returns a task."""
    task = MagicMock()
    task.title = "My Task"

    from pb.core.suggestions import SuggestionEngine

    repo = MagicMock()
    repo.get_active_task.return_value = task
    engine = SuggestionEngine(repo=repo)

    context = engine._build_context()
    assert "Active task:" in context
    assert "My Task" in context


def test_build_context_includes_vault_location():
    """_build_context includes 'Vault location:' when get_vault_cwd returns a Path."""
    from pb.core.suggestions import SuggestionEngine

    vault_root = Path("/vault")
    cwd = Path("/vault/projects")

    repo = MagicMock()
    repo.get_active_task.return_value = None
    engine = SuggestionEngine(
        repo=repo,
        get_vault_cwd=lambda: cwd,
        vault_root=vault_root,
    )

    context = engine._build_context()
    assert "Vault location:" in context
    assert "projects" in context


def test_build_context_includes_recent_commands():
    """_build_context includes 'Recent commands:' when get_recent_commands returns entries."""
    from pb.core.suggestions import SuggestionEngine

    repo = MagicMock()
    repo.get_active_task.return_value = None
    engine = SuggestionEngine(repo=repo)

    with patch("pb.core.suggestions.get_recent_commands", return_value=["chat", "start"]):
        context = engine._build_context()

    assert "Recent commands:" in context
    assert "chat" in context
    assert "start" in context


def test_build_context_returns_no_context_when_all_raise():
    """_build_context returns 'No context available.' when all sources raise exceptions."""
    from pb.core.suggestions import SuggestionEngine

    repo = MagicMock()
    repo.get_active_task.side_effect = Exception("db error")
    engine = SuggestionEngine(repo=repo)

    with patch("pb.core.suggestions.get_recent_commands", side_effect=Exception("error")):
        context = engine._build_context()

    assert context == "No context available."


# ---------------------------------------------------------------------------
# get_recent_commands tests
# ---------------------------------------------------------------------------


def test_get_recent_commands_returns_command_strings():
    """get_recent_commands returns list of command strings ordered by timestamp DESC."""
    from pb.core.suggestions import get_recent_commands

    row1 = MagicMock()
    row1.__getitem__ = lambda self, key: "chat" if key == "command" else None
    row2 = MagicMock()
    row2.__getitem__ = lambda self, key: "start" if key == "command" else None

    mock_conn = MagicMock()
    mock_conn.execute.return_value.fetchall.return_value = [row1, row2]

    with patch("pb.core.suggestions.get_connection") as mock_get_conn:
        mock_get_conn.return_value.__enter__ = MagicMock(return_value=mock_conn)
        mock_get_conn.return_value.__exit__ = MagicMock(return_value=False)
        result = get_recent_commands(limit=20)

    assert isinstance(result, list)
    # Should have called with the correct query
    call_args = mock_conn.execute.call_args
    assert "usage_log ORDER BY timestamp DESC LIMIT" in call_args[0][0]


def test_get_recent_commands_returns_empty_on_exception():
    """get_recent_commands returns [] when database raises exception."""
    from pb.core.suggestions import get_recent_commands

    with patch("pb.core.suggestions.get_connection", side_effect=Exception("db error")):
        result = get_recent_commands()

    assert result == []


# ---------------------------------------------------------------------------
# tier2_confirm tests
# ---------------------------------------------------------------------------


def test_tier2_confirm_returns_true_on_empty_input():
    """tier2_confirm returns True when input() returns empty string."""
    from pb.core.suggestions import tier2_confirm

    with patch("builtins.input", return_value=""):
        result = tier2_confirm("pb start foo", "Start the task")

    assert result is True


def test_tier2_confirm_returns_true_on_y():
    """tier2_confirm returns True when input() returns 'y'."""
    from pb.core.suggestions import tier2_confirm

    with patch("builtins.input", return_value="y"):
        result = tier2_confirm("pb start foo", "")

    assert result is True


def test_tier2_confirm_returns_true_on_yes():
    """tier2_confirm returns True when input() returns 'yes'."""
    from pb.core.suggestions import tier2_confirm

    with patch("builtins.input", return_value="yes"):
        result = tier2_confirm("pb start foo", "")

    assert result is True


def test_tier2_confirm_returns_false_on_n():
    """tier2_confirm returns False when input() returns 'n'."""
    from pb.core.suggestions import tier2_confirm

    with patch("builtins.input", return_value="n"):
        result = tier2_confirm("pb start foo", "")

    assert result is False


def test_tier2_confirm_returns_false_on_keyboard_interrupt():
    """tier2_confirm returns False when input() raises KeyboardInterrupt."""
    from pb.core.suggestions import tier2_confirm

    with patch("builtins.input", side_effect=KeyboardInterrupt):
        result = tier2_confirm("pb start foo", "")

    assert result is False


# ---------------------------------------------------------------------------
# MkMvEngine.find_matching_notes tests
# ---------------------------------------------------------------------------


def test_find_matching_notes_finds_by_filename(tmp_path):
    """find_matching_notes finds notes whose filenames match keywords."""
    from pb.core.suggestions import MkMvEngine

    folder = tmp_path / "projects"
    folder.mkdir()
    (folder / "meeting-notes.md").write_text("# Meeting Notes\nSome content")
    (folder / "unrelated.md").write_text("# Unrelated\nOther stuff")

    engine = MkMvEngine(vault_root=tmp_path)
    with patch("pb.vault.indexer.search_folder_index", return_value=None):
        results = engine.find_matching_notes("meeting")

    paths = [p for p, _ in results]
    assert folder / "meeting-notes.md" in paths
    assert folder / "unrelated.md" not in paths


def test_find_matching_notes_uses_fts_index(tmp_path):
    """find_matching_notes uses FTS index results when available."""
    from pb.core.suggestions import MkMvEngine

    folder = tmp_path / "projects"
    folder.mkdir()
    note = folder / "deep-work.md"
    note.write_text("# Deep Work\nFocus strategies")

    engine = MkMvEngine(vault_root=tmp_path)
    fts_results = [("projects/deep-work.md", "Focus strategies")]
    with patch("pb.vault.indexer.search_folder_index", return_value=fts_results):
        results = engine.find_matching_notes("focus")

    assert len(results) == 1
    assert results[0][0] == note


def test_find_matching_notes_returns_empty_when_no_match(tmp_path):
    """find_matching_notes returns empty list when nothing matches."""
    from pb.core.suggestions import MkMvEngine

    folder = tmp_path / "projects"
    folder.mkdir()
    (folder / "unrelated.md").write_text("# Unrelated")

    engine = MkMvEngine(vault_root=tmp_path)
    with patch("pb.vault.indexer.search_folder_index", return_value=None):
        results = engine.find_matching_notes("quantum physics")

    assert results == []


# ---------------------------------------------------------------------------
# MkMvEngine.ai_filter_notes tests
# ---------------------------------------------------------------------------


def test_ai_filter_notes_parses_numbered_response():
    """ai_filter_notes parses Flash Lite's numbered selection."""
    from pb.core.suggestions import MkMvEngine

    client = _make_mock_client(available=True, generate_result="1,3")
    vault_root = Path("/vault")
    candidates = [
        (Path("/vault/a.md"), "Note A"),
        (Path("/vault/b.md"), "Note B"),
        (Path("/vault/c.md"), "Note C"),
    ]

    engine = MkMvEngine(vault_root=vault_root)
    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.ai_filter_notes("topic", candidates)

    assert result == [Path("/vault/a.md"), Path("/vault/c.md")]


def test_ai_filter_notes_returns_all_on_all_response():
    """ai_filter_notes returns all candidates when Flash Lite says 'all'."""
    from pb.core.suggestions import MkMvEngine

    client = _make_mock_client(available=True, generate_result="all")
    vault_root = Path("/vault")
    candidates = [
        (Path("/vault/a.md"), "Note A"),
        (Path("/vault/b.md"), "Note B"),
    ]

    engine = MkMvEngine(vault_root=vault_root)
    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.ai_filter_notes("topic", candidates)

    assert result == [Path("/vault/a.md"), Path("/vault/b.md")]


def test_ai_filter_notes_returns_all_when_client_unavailable():
    """ai_filter_notes returns all candidates when Flash Lite unavailable."""
    from pb.core.suggestions import MkMvEngine

    client = _make_mock_client(available=False)
    vault_root = Path("/vault")
    candidates = [
        (Path("/vault/a.md"), "Note A"),
        (Path("/vault/b.md"), "Note B"),
    ]

    engine = MkMvEngine(vault_root=vault_root)
    with patch("pb.core.suggestions.get_client", return_value=client):
        result = engine.ai_filter_notes("topic", candidates)

    assert result == [Path("/vault/a.md"), Path("/vault/b.md")]


# ---------------------------------------------------------------------------
# MkMvEngine.rank_folder tests
# ---------------------------------------------------------------------------


def test_rank_folder_returns_folder_from_list():
    """rank_folder returns a folder name from the provided list."""
    from pb.core.suggestions import MkMvEngine

    folders = ["projects", "reference", "daily"]
    client = _make_mock_client(available=True, generate_result="reference")
    vault_root = Path("/vault")

    with patch("pb.core.suggestions.get_client", return_value=client):
        engine = MkMvEngine(vault_root=vault_root)
        result = engine.rank_folder("meeting notes", folders)

    assert result in folders
    assert result == "reference"


def test_rank_folder_depth_guard_rejects_slash_path():
    """rank_folder returns first folder name when Flash Lite returns a name containing '/'."""
    from pb.core.suggestions import MkMvEngine

    folders = ["projects", "reference", "daily"]
    client = _make_mock_client(available=True, generate_result="projects/subdir")
    vault_root = Path("/vault")

    with patch("pb.core.suggestions.get_client", return_value=client):
        engine = MkMvEngine(vault_root=vault_root)
        result = engine.rank_folder("meeting notes", folders)

    assert result == folders[0]


def test_rank_folder_falls_back_when_name_not_in_list():
    """rank_folder returns first folder name when Flash Lite returns a name not in the list."""
    from pb.core.suggestions import MkMvEngine

    folders = ["projects", "reference", "daily"]
    client = _make_mock_client(available=True, generate_result="unknown-folder")
    vault_root = Path("/vault")

    with patch("pb.core.suggestions.get_client", return_value=client):
        engine = MkMvEngine(vault_root=vault_root)
        result = engine.rank_folder("meeting notes", folders)

    assert result == folders[0]


# ---------------------------------------------------------------------------
# AVAILABLE_COMMANDS tests
# ---------------------------------------------------------------------------


def test_available_commands_contains_required_commands():
    """AVAILABLE_COMMANDS list contains at least the required commands."""
    from pb.core.suggestions import AVAILABLE_COMMANDS

    required = ["goal", "plan", "study", "practise", "start", "finish", "next", "do"]
    for cmd in required:
        assert cmd in AVAILABLE_COMMANDS, f"'{cmd}' missing from AVAILABLE_COMMANDS"


def test_available_commands_is_list():
    """AVAILABLE_COMMANDS is a list."""
    from pb.core.suggestions import AVAILABLE_COMMANDS

    assert isinstance(AVAILABLE_COMMANDS, list)
    assert len(AVAILABLE_COMMANDS) >= 8
