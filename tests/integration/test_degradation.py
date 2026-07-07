"""Integration tests for CLI degradation via CliRunner (D-01, D-02, D-04, D-11, D-12).

Uses CliRunner to exercise the full CLI callback path for bare `pb` invocations.
"""
from __future__ import annotations

from typer.testing import CliRunner

from pb.cli import main as main_mod

runner = CliRunner()


def test_no_config_exits_zero_and_shows_guidance(tmp_path, monkeypatch):
    """D-01/D-04/D-12: No config -> exit 0, shows setup guidance."""
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(tmp_path / "no_config.toml"))
    monkeypatch.setattr("pb.cli.commands.init.init_command", lambda **kw: None)
    result = runner.invoke(main_mod.app, [])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    assert "ProductiveBrain" in result.output or "setup" in result.output.lower(), (
        f"Expected welcome/setup text in output, got: {result.output!r}"
    )


def test_vault_missing_exits_zero_and_shows_vault_guidance(tmp_path, monkeypatch):
    """D-02/D-04/D-12: Config present, vault missing -> exit 0, vault message."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[general]\nvault_path = "/nonexistent/vault/path"\n\n[vaults.main]\npath = "/nonexistent/vault/path"\n')
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(config_path))
    result = runner.invoke(main_mod.app, [])
    assert result.exit_code == 0, f"Expected exit 0, got {result.exit_code}. Output: {result.output}"
    assert "vault" in result.output.lower() or "pb init" in result.output, (
        f"Expected vault guidance in output, got: {result.output!r}"
    )


def test_subcommand_with_no_config_does_not_trigger_setup(tmp_path, monkeypatch):
    """Pitfall 2: Subcommands should NOT get the setup wizard."""
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(tmp_path / "no_config.toml"))
    result = runner.invoke(main_mod.app, ["goal", "list"])
    # Should NOT contain welcome/setup text — subcommands get the normal error path
    assert "ProductiveBrain" not in (result.output or ""), (
        f"Setup wizard should not appear for subcommands; got: {result.output!r}"
    )
    # Should not exit 0 (no config is an error for subcommands)
    assert result.exit_code != 0, (
        f"Subcommand with no config should not exit 0; got {result.exit_code}"
    )


def test_no_config_welcome_text_not_leaked_from_exception(tmp_path, monkeypatch):
    """T-01-01 integration: Vault-missing path does not print raw exception text."""
    config_path = tmp_path / "config.toml"
    config_path.write_text('[general]\nvault_path = "/nonexistent/vault/path"\n\n[vaults.main]\npath = "/nonexistent/vault/path"\n')
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(config_path))
    result = runner.invoke(main_mod.app, [])
    # The raw exception string contains the path — it should not appear in output
    assert "/nonexistent/vault/path" not in result.output, (
        f"Filesystem path should not be leaked in output; got: {result.output!r}"
    )
