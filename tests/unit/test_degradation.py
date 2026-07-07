"""Unit tests for CLI degradation handlers (D-01, D-02).

Tests exercise _handle_no_config_degradation and _handle_vault_missing_degradation
in isolation via monkeypatching.
"""
from __future__ import annotations

import io
from unittest.mock import MagicMock

import pytest
from rich.console import Console


def _make_console(buf: io.StringIO) -> Console:
    """Return a Rich Console that writes to buf."""
    return Console(file=buf, markup=True, highlight=False)


def test_no_config_handler_calls_init_command(monkeypatch):
    """D-01: No config -> welcome + auto-launch pb init."""
    from pb.cli import main as main_mod

    called = []
    monkeypatch.setattr("pb.cli.commands.init.init_command", lambda **kw: called.append("init"))
    # Patch get_console so output goes to a buffer (not stderr/stdout)
    buf = io.StringIO()
    monkeypatch.setattr("pb.cli.console.get_console", lambda: _make_console(buf))

    ctx = MagicMock()
    main_mod._handle_no_config_degradation(ctx)
    assert "init" in called, "init_command should have been invoked"


def test_no_config_handler_prints_welcome(monkeypatch):
    """D-01: Welcome message mentions ProductiveBrain."""
    from pb.cli import main as main_mod

    monkeypatch.setattr("pb.cli.commands.init.init_command", lambda **kw: None)
    buf = io.StringIO()
    monkeypatch.setattr("pb.cli.console.get_console", lambda: _make_console(buf))

    ctx = MagicMock()
    main_mod._handle_no_config_degradation(ctx)
    output = buf.getvalue()
    assert "ProductiveBrain" in output, f"Expected 'ProductiveBrain' in output, got: {output!r}"


def test_vault_missing_handler_does_not_call_init(monkeypatch):
    """D-02: Vault missing -> message only, no auto-launch of init_command."""
    from pb.cli import main as main_mod

    called = []
    monkeypatch.setattr("pb.cli.commands.init.init_command", lambda **kw: called.append("init"))
    buf = io.StringIO()
    monkeypatch.setattr("pb.cli.console.get_console", lambda: _make_console(buf))

    ctx = MagicMock()
    exc = FileNotFoundError("Vault not found at /some/path")
    main_mod._handle_vault_missing_degradation(ctx, exc)
    assert called == [], "init_command should NOT have been invoked for vault-missing scenario"


def test_vault_missing_handler_mentions_vault_and_init(monkeypatch):
    """D-02: Message mentions vault problem and pb init."""
    from pb.cli import main as main_mod

    buf = io.StringIO()
    monkeypatch.setattr("pb.cli.console.get_console", lambda: _make_console(buf))

    ctx = MagicMock()
    exc = FileNotFoundError("Vault not found at /some/path")
    main_mod._handle_vault_missing_degradation(ctx, exc)
    output = buf.getvalue().lower()
    assert "vault" in output, f"Expected 'vault' in output, got: {output!r}"
    assert "pb init" in output or "init" in output, f"Expected init guidance in output, got: {output!r}"


def test_vault_missing_handler_does_not_leak_exc_message(monkeypatch):
    """T-01-01: Vault-missing handler must NOT print the raw exc message (filesystem path leak)."""
    from pb.cli import main as main_mod

    buf = io.StringIO()
    monkeypatch.setattr("pb.cli.console.get_console", lambda: _make_console(buf))

    ctx = MagicMock()
    sensitive_path = "/home/user/.secret/vault"
    exc = FileNotFoundError(f"Vault not found at {sensitive_path}")
    main_mod._handle_vault_missing_degradation(ctx, exc)
    output = buf.getvalue()
    assert sensitive_path not in output, (
        f"Handler must not leak filesystem path from exception; got: {output!r}"
    )
