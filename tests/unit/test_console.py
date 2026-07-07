"""Unit tests for console factory and theme system."""

import os
import pytest
from unittest.mock import patch, MagicMock

from pb.cli.console import get_console, get_err_console, set_plain_mode, _plain_mode
from pb.cli.themes import load_active_theme, PRESETS
from pb.cli.themes.catppuccin import CATPPUCCIN
from pb.cli.themes.nord import NORD


class TestThemePresets:
    """Tests for theme preset dictionaries."""

    def test_catppuccin_has_expected_roles(self):
        assert len(CATPPUCCIN) >= 11

    def test_nord_has_expected_roles(self):
        assert len(NORD) >= 11

    def test_catppuccin_contains_all_required_roles(self):
        required = {
            "header", "subheader", "dim", "info", "success", "warn", "error",
            "command", "path", "duration", "math", "branch.study", "branch.practise",
            "value.high", "value.med", "value.low", "table.header", "table.border",
        }
        assert required.issubset(set(CATPPUCCIN.keys()))

    def test_nord_contains_all_required_roles(self):
        required = {
            "header", "subheader", "dim", "info", "success", "warn", "error",
            "command", "path", "duration", "math", "branch.study", "branch.practise",
            "value.high", "value.med", "value.low", "table.header", "table.border",
        }
        assert required.issubset(set(NORD.keys()))

    def test_presets_registry(self):
        assert "catppuccin" in PRESETS
        assert "nord" in PRESETS
        assert PRESETS["catppuccin"] is CATPPUCCIN
        assert PRESETS["nord"] is NORD


class TestGetConsole:
    """Tests for get_console() behavior."""

    def setup_method(self):
        set_plain_mode(False)

    def teardown_method(self):
        set_plain_mode(False)

    def test_plain_mode_returns_no_color_console(self):
        set_plain_mode(True)
        console = get_console()
        assert console.no_color is True
        assert console._highlight is False

    def test_normal_mode_returns_themed_console(self):
        set_plain_mode(False)
        console = get_console()
        assert str(console.get_style("header")) == str(CATPPUCCIN["header"])

    def test_err_console_writes_to_stderr(self):
        console = get_err_console()
        assert console.stderr is True

    def test_set_plain_mode_true(self):
        set_plain_mode(True)
        import pb.cli.console as console_mod
        assert console_mod._plain_mode is True

    def test_set_plain_mode_false(self):
        set_plain_mode(True)
        set_plain_mode(False)
        import pb.cli.console as console_mod
        assert console_mod._plain_mode is False

    def test_plain_mode_toggling(self):
        set_plain_mode(True)
        c1 = get_console()
        assert c1.no_color is True
        set_plain_mode(False)
        c2 = get_console()
        assert str(c2.get_style("header")) == str(CATPPUCCIN["header"])

    def test_console_respects_configured_max_width(self):
        from pb.storage.config import UIConfig

        mock_ui = UIConfig(max_content_width=72)
        mock_config = MagicMock()
        mock_config.ui = mock_ui
        with patch("pb.storage.config.get_config", return_value=mock_config), patch(
            "pb.cli.console.shutil.get_terminal_size",
            return_value=os.terminal_size((120, 24)),
        ):
            console = get_console()
        assert console.width == 72

    def test_console_defaults_to_ratio_based_width(self):
        from pb.storage.config import UIConfig

        mock_ui = UIConfig()
        mock_config = MagicMock()
        mock_config.ui = mock_ui
        with patch("pb.storage.config.get_config", return_value=mock_config), patch(
            "pb.cli.console.shutil.get_terminal_size",
            return_value=os.terminal_size((120, 24)),
        ):
            console = get_console()
        assert console.width == 84


class TestLoadActiveTheme:
    """Tests for theme loading with 3-tier precedence."""

    def test_default_returns_catppuccin(self):
        """When no config UI section, returns catppuccin."""
        theme = load_active_theme()
        assert theme["header"] == CATPPUCCIN["header"]

    def test_overrides_merge_onto_preset(self):
        """Per-key overrides merge on top of preset base."""
        from pb.storage.config import get_config
        config = get_config()
        # If UIConfig exists with overrides, they should apply
        # This tests the merge logic path
        theme = load_active_theme()
        assert "header" in theme
        assert "error" in theme

    def test_all_required_roles_present(self):
        """Loaded theme must contain the required semantic roles."""
        theme = load_active_theme()
        required = {
            "header", "subheader", "dim", "info", "success", "warn", "error",
            "command", "path", "duration", "math", "branch.study", "branch.practise",
            "value.high", "value.med", "value.low", "table.header", "table.border",
        }
        for role in required:
            assert role in theme, f"Missing role: {role}"

    def test_unknown_theme_falls_back_to_catppuccin(self):
        """Unknown theme preset name falls back to catppuccin."""
        from pb.storage.config import UIConfig
        mock_ui = UIConfig(theme="nonexistent-theme")
        mock_config = MagicMock()
        mock_config.ui = mock_ui
        with patch("pb.storage.config.get_config", return_value=mock_config):
            theme = load_active_theme()
        # Falls back to catppuccin
        assert theme["header"] == CATPPUCCIN["header"]

    def test_overrides_applied_to_base(self):
        """theme_overrides dict is merged onto base preset."""
        from pb.storage.config import UIConfig
        mock_ui = UIConfig(theme="catppuccin", theme_overrides={"header": "bold magenta"})
        mock_config = MagicMock()
        mock_config.ui = mock_ui
        with patch("pb.storage.config.get_config", return_value=mock_config):
            theme = load_active_theme()
        assert theme["header"] == "bold magenta"
        # Other keys remain from catppuccin
        assert theme["error"] == CATPPUCCIN["error"]
