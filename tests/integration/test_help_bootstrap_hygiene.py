"""Help rendering must not bootstrap runtime, config, or the database."""

from __future__ import annotations

from typer.testing import CliRunner

from pb.cli import main as main_mod


runner = CliRunner()


def test_stable_help_surfaces_do_not_bootstrap_runtime(monkeypatch):
    def explode(**kwargs):
        raise AssertionError("build_runtime_context should not run for help")

    monkeypatch.setattr(main_mod, "build_runtime_context", explode)

    commands = [
        ["--help"],
        ["goal", "--help"],
        ["plan", "--help"],
        ["review", "--help"],
        ["notes", "--help"],
        ["anki", "--help"],
        ["mcp", "--help"],
        ["feedback", "--help"],
        ["study", "--help"],
        ["practise", "--help"],
        ["model", "--help"],
        ["doctor", "--help"],
        ["init", "--help"],
    ]

    for argv in commands:
        result = runner.invoke(main_mod.app, argv)
        assert result.exit_code == 0, f"{argv}: {result.output}"
        assert "Usage:" in result.output
