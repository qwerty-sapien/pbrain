#!/usr/bin/env python3
"""Deterministic CLI smoke runner for ProductiveBrain.

This script keeps the surface small and reliable:
- boot a temporary config + vault
- run a handful of public `pb` commands
- fail fast with readable output

It intentionally avoids live LLM flows unless the caller extends it.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


@dataclass
class CommandResult:
    command: list[str]
    returncode: int
    stdout: str
    stderr: str


class CLISmokeSuite:
    """Run a deterministic ProductiveBrain smoke suite."""

    def __init__(self) -> None:
        self._tmpdir = Path(tempfile.mkdtemp(prefix="productivebrain-smoke-"))
        self.vault_path = self._tmpdir / "vault"
        self.config_path = self._tmpdir / "config.toml"
        self.home_path = self._tmpdir / "home"

    def env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["PRODUCTIVEBRAIN_CONFIG_PATH"] = str(self.config_path)
        env["HOME"] = str(self.home_path)
        env["VIRTUAL_ENV_PROMPT"] = ""
        return env

    def run(self, *args: str, timeout: int = 60) -> CommandResult:
        command = ["uv", "run", "pb", *args]
        print(f"$ {' '.join(command)}")
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=self.env(),
        )
        if result.stdout.strip():
            print(result.stdout.rstrip())
        if result.stderr.strip():
            print(result.stderr.rstrip())
        if result.returncode != 0:
            raise RuntimeError(f"Command failed: {' '.join(command)}")
        return CommandResult(command, result.returncode, result.stdout, result.stderr)

    def setup(self) -> None:
        self.vault_path.mkdir(parents=True, exist_ok=True)
        self.home_path.mkdir(parents=True, exist_ok=True)
        self.run(
            "init",
            "--non-interactive",
            "--vault-name",
            "main",
            "--vault-path",
            str(self.vault_path),
            "--provider",
            "gemini",
            "--model",
            "gemini-3-flash-preview",
            "--yes",
        )

    def smoke(self) -> None:
        self.setup()
        self.run("--help")
        self.run("doctor", "--json")
        self.run("vault", "current", "--json")
        self.run("notes", "inbox")
        self.run("mcp", "print-config", "--client", "generic", "--vault", "main")

    def cleanup(self) -> None:
        shutil.rmtree(self._tmpdir, ignore_errors=True)


def main() -> None:
    suite = CLISmokeSuite()
    try:
        suite.smoke()
        print("\nSmoke suite passed.")
    finally:
        suite.cleanup()


if __name__ == "__main__":
    main()
