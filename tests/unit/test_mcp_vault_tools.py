"""Unit tests for vault MCP tools."""

from types import SimpleNamespace
import pytest
from pathlib import Path
from unittest.mock import patch

from pb.mcp.tools.vault import (
    vault_search,
    vault_read,
    vault_write,
    vault_link_graph,
    _validate_path,
    _extract_links,
    VaultError,
)


class TestValidatePath:
    """Tests for path validation (Pitfall #4 prevention)."""

    def test_valid_relative_path(self, tmp_path: Path) -> None:
        """_validate_path accepts valid relative paths."""
        vault = tmp_path / "vault"
        vault.mkdir()

        result = _validate_path("30-people/alice.md", vault)

        assert result == vault / "30-people" / "alice.md"

    def test_rejects_path_traversal(self, tmp_path: Path) -> None:
        """_validate_path rejects paths that escape vault."""
        vault = tmp_path / "vault"
        vault.mkdir()

        with pytest.raises(VaultError, match="escapes vault boundary"):
            _validate_path("../../../etc/passwd", vault)

    def test_rejects_absolute_path_outside_vault(self, tmp_path: Path) -> None:
        """_validate_path rejects absolute paths outside vault."""
        vault = tmp_path / "vault"
        vault.mkdir()

        with pytest.raises(VaultError, match="escapes vault boundary"):
            _validate_path("/etc/passwd", vault)

    def test_normalizes_path(self, tmp_path: Path) -> None:
        """_validate_path normalizes paths with ./ and extra slashes."""
        vault = tmp_path / "vault"
        vault.mkdir()

        result = _validate_path("./30-people//alice.md", vault)

        assert result == vault / "30-people" / "alice.md"


class TestExtractLinks:
    """Tests for wiki link extraction."""

    def test_extracts_simple_links(self) -> None:
        """_extract_links finds [[link]] patterns."""
        content = "See [[Alice]] and [[Bob]] for details."

        links = _extract_links(content)

        assert links == ["Alice", "Bob"]

    def test_extracts_aliased_links(self) -> None:
        """_extract_links handles [[link|alias]] patterns."""
        content = "See [[Alice|my friend]] for details."

        links = _extract_links(content)

        assert links == ["Alice"]

    def test_handles_no_links(self) -> None:
        """_extract_links returns empty list when no links."""
        content = "No links here."

        links = _extract_links(content)

        assert links == []


class TestVaultRead:
    """Tests for vault_read tool."""

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_reads_existing_note(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_read returns content of existing note."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "30-people" / "alice.md"
        note.parent.mkdir(parents=True)
        note.write_text("# Alice\nA friend")
        mock_vault.return_value = vault

        result = vault_read("30-people/alice.md")

        assert result == "# Alice\nA friend"

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_raises_for_missing_note(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_read raises VaultError for non-existent note."""
        vault = tmp_path / "vault"
        vault.mkdir()
        mock_vault.return_value = vault

        with pytest.raises(VaultError, match="Note not found"):
            vault_read("30-people/nobody.md")

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_rejects_path_traversal(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_read rejects path traversal attempts."""
        vault = tmp_path / "vault"
        vault.mkdir()
        mock_vault.return_value = vault

        with pytest.raises(VaultError, match="escapes vault boundary"):
            vault_read("../../../etc/passwd")


class TestVaultWrite:
    """Tests for vault_write tool."""

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    @patch("pb.mcp.tools.vault._bypassing", return_value=True)
    @patch("pb.mcp.tools.vault.get_mcp_context", return_value=SimpleNamespace(allow_writes=True))
    def test_creates_new_note(
        self, mock_ctx: object, mock_bypass: object, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_write creates a new note."""
        vault = tmp_path / "vault"
        vault.mkdir()
        mock_vault.return_value = vault

        result = vault_write("30-people/bob.md", "# Bob\nNew friend")

        assert result["path"] == "30-people/bob.md"
        assert result["action"] == "created"
        assert (vault / "30-people" / "bob.md").read_text() == "# Bob\nNew friend"

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    @patch("pb.mcp.tools.vault._bypassing", return_value=True)
    @patch("pb.mcp.tools.vault.get_mcp_context", return_value=SimpleNamespace(allow_writes=True))
    def test_updates_existing_note(
        self, mock_ctx: object, mock_bypass: object, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_write overwrites existing note."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "30-people" / "alice.md"
        note.parent.mkdir(parents=True)
        note.write_text("# Alice\nOld content")
        mock_vault.return_value = vault

        vault_write("30-people/alice.md", "# Alice\nNew content")

        assert note.read_text() == "# Alice\nNew content"

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    @patch("pb.mcp.tools.vault._bypassing", return_value=True)
    @patch("pb.mcp.tools.vault.get_mcp_context", return_value=SimpleNamespace(allow_writes=True))
    def test_creates_parent_folders(
        self, mock_ctx: object, mock_bypass: object, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_write creates parent folders when create_folders=True."""
        vault = tmp_path / "vault"
        vault.mkdir()
        mock_vault.return_value = vault

        vault_write("new-folder/nested/note.md", "content", create_folders=True)

        assert (vault / "new-folder" / "nested" / "note.md").exists()

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    @patch("pb.mcp.tools.vault._bypassing", return_value=True)
    @patch("pb.mcp.tools.vault.get_mcp_context", return_value=SimpleNamespace(allow_writes=True))
    def test_rejects_missing_parent_when_disabled(
        self, mock_ctx: object, mock_bypass: object, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_write raises when create_folders=False and parent missing."""
        vault = tmp_path / "vault"
        vault.mkdir()
        mock_vault.return_value = vault

        with pytest.raises(VaultError, match="Parent folder does not exist"):
            vault_write("missing/note.md", "content", create_folders=False)


class TestVaultSearch:
    """Tests for vault_search tool."""

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    def test_finds_by_filename(
        self, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_search matches filenames."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "30-people" / "alice.md"
        note.parent.mkdir(parents=True)
        note.write_text("# Alice")
        mock_vault.return_value = vault

        result = vault_search("alice")

        assert len(result["matches"]) == 1
        assert result["matches"][0]["path"] == "30-people/alice.md"
        assert result["matches"][0]["filename_match"] is True

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    def test_finds_by_content(
        self, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_search matches content."""
        vault = tmp_path / "vault"
        vault.mkdir()
        note = vault / "30-people" / "bob.md"
        note.parent.mkdir(parents=True)
        note.write_text("# Bob\nLikes playing tennis")
        mock_vault.return_value = vault

        result = vault_search("tennis")

        assert len(result["matches"]) == 1
        assert result["matches"][0]["content_match"] is True
        assert "tennis" in result["matches"][0]["snippet"]

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    @patch("pb.mcp.tools.vault.scaffold_vault")
    def test_limits_to_folder(
        self, mock_scaffold: None, mock_vault: Path, tmp_path: Path
    ) -> None:
        """vault_search respects folder filter."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "30-people").mkdir()
        (vault / "30-people" / "alice.md").write_text("# Alice")
        (vault / "knowledge").mkdir()
        (vault / "knowledge" / "alice-concept.md").write_text("# Alice concept")
        mock_vault.return_value = vault

        result = vault_search("alice", folder="30-people")

        assert len(result["matches"]) == 1
        assert "30-people" in result["matches"][0]["path"]


class TestVaultLinkGraph:
    """Tests for vault_link_graph tool."""

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_finds_outgoing_links(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_link_graph extracts outgoing links."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "30-people").mkdir()
        (vault / "30-people" / "alice.md").write_text("Friends with [[Bob]] and [[Carol]]")
        (vault / "30-people" / "bob.md").write_text("# Bob")
        mock_vault.return_value = vault

        result = vault_link_graph("30-people/alice.md")

        assert result["outgoing_count"] == 2
        targets = [link["target"] for link in result["outgoing"]]
        assert "Bob" in targets
        assert "Carol" in targets

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_finds_incoming_links(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_link_graph finds notes linking to this one."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "30-people").mkdir()
        (vault / "30-people" / "alice.md").write_text("# Alice")
        (vault / "30-people" / "bob.md").write_text("Best friend is [[alice]]")
        mock_vault.return_value = vault

        result = vault_link_graph("30-people/alice.md")

        assert result["incoming_count"] == 1
        assert "30-people/bob.md" in result["incoming"]

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_depth_two_returns_two_hop_neighborhood(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_link_graph honors depth=2 with real second-hop links."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "30-people").mkdir()
        (vault / "30-people" / "alice.md").write_text("See [[bob]]")
        (vault / "30-people" / "bob.md").write_text("See [[carol]]")
        (vault / "30-people" / "carol.md").write_text("# Carol")
        (vault / "30-people" / "dave.md").write_text("See [[alice]]")
        (vault / "30-people" / "erin.md").write_text("See [[dave]]")
        mock_vault.return_value = vault

        result = vault_link_graph("30-people/alice.md", depth=2)

        assert result["requested_depth"] == 2
        assert result["effective_depth"] == 2
        assert result["clamped"] is False
        assert "30-people/carol.md" in result["out2"]
        assert "30-people/erin.md" in result["in2"]

    @patch("pb.mcp.tools.vault._resolved_vault_path")
    def test_depth_is_clamped_to_supported_max(self, mock_vault: Path, tmp_path: Path) -> None:
        """vault_link_graph reports depth clamping above the supported max."""
        vault = tmp_path / "vault"
        vault.mkdir()
        (vault / "30-people").mkdir()
        (vault / "30-people" / "alice.md").write_text("# Alice")
        mock_vault.return_value = vault

        result = vault_link_graph("30-people/alice.md", depth=5)

        assert result["requested_depth"] == 5
        assert result["effective_depth"] == 2
        assert result["clamped"] is True
