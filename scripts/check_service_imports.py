#!/usr/bin/env python3
"""CI gate: verify service modules do not import typer or rich (ARCH-08, INV-4).

Run: python3 scripts/check_service_imports.py
Exit 0 = clean. Exit 1 = violations found (with file:line report).
Uses stdlib ast.walk() to catch function-level imports (not just top-level).

Service directories checked:
  src/pb/tasks, src/pb/sessions, src/pb/vault,
  src/pb/ai, src/pb/plan, src/pb/review, src/pb/goals
"""
from __future__ import annotations

import ast
import sys
from pathlib import Path

FORBIDDEN = {"typer", "rich"}
SERVICE_DIRS = [
    "src/pb/tasks",
    "src/pb/sessions",
    "src/pb/vault",
    "src/pb/ai",
    "src/pb/plan",
    "src/pb/review",
    "src/pb/goals",
]

# Files to exclude from checking (e.g., existing vault files that predate INV-4)
EXCLUDE_FILES = {
    "src/pb/vault/config.py",
    "src/pb/vault/scaffold.py",
    "src/pb/vault/graph.py",
    "src/pb/vault/graph_store.py",
    "src/pb/vault/indexer.py",
    "src/pb/vault/lifecycle.py",
    "src/pb/vault/embeddings.py",
    "src/pb/vault/anki_client.py",
    "src/pb/vault/socratic.py",
}


def check_file(path: Path, root: Path) -> list[str]:
    """Check a single Python file for forbidden imports using AST."""
    rel = str(path.relative_to(root))
    if rel in EXCLUDE_FILES:
        return []

    violations = []
    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return [f"{rel}: SyntaxError (cannot parse)"]

    for node in ast.walk(tree):
        # Check: import typer / import rich / import rich.console
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_module = alias.name.split(".")[0]
                if top_module in FORBIDDEN:
                    violations.append(
                        f"{rel}:{node.lineno}: import {alias.name}"
                    )
        # Check: from typer import ... / from rich.console import ...
        elif isinstance(node, ast.ImportFrom) and node.module:
            top_module = node.module.split(".")[0]
            if top_module in FORBIDDEN:
                violations.append(
                    f"{rel}:{node.lineno}: from {node.module} import ..."
                )
        # Check: TYPE_CHECKING guard — allow typer/rich ONLY inside TYPE_CHECKING blocks
        # The AST-walk approach catches all imports including those in TYPE_CHECKING.
        # We need to EXCLUDE imports inside `if TYPE_CHECKING:` blocks.

    # Re-check: filter out imports inside TYPE_CHECKING blocks
    violations = _filter_type_checking_imports(path, root, violations)
    return violations


def _filter_type_checking_imports(
    path: Path, root: Path, violations: list[str]
) -> list[str]:
    """Remove violations that occur inside TYPE_CHECKING blocks."""
    if not violations:
        return violations

    try:
        tree = ast.parse(path.read_text())
    except SyntaxError:
        return violations

    # Find line ranges inside TYPE_CHECKING blocks
    tc_lines: set[int] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.If):
            test = node.test
            # Match: if TYPE_CHECKING:
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        tc_lines.add(child.lineno)
            # Match: if typing.TYPE_CHECKING:
            elif (
                isinstance(test, ast.Attribute)
                and test.attr == "TYPE_CHECKING"
            ):
                for child in ast.walk(node):
                    if hasattr(child, "lineno"):
                        tc_lines.add(child.lineno)

    # Filter: keep only violations NOT inside TYPE_CHECKING
    filtered = []
    for v in violations:
        # Extract line number from "rel:LINE: import ..."
        parts = v.split(":")
        if len(parts) >= 2:
            try:
                lineno = int(parts[1])
                if lineno not in tc_lines:
                    filtered.append(v)
            except ValueError:
                filtered.append(v)
        else:
            filtered.append(v)
    return filtered


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    all_violations: list[str] = []

    for sdir in SERVICE_DIRS:
        service_path = root / sdir
        if not service_path.exists():
            continue
        for f in service_path.rglob("*.py"):
            all_violations.extend(check_file(f, root))

    if all_violations:
        print("ARCH-08 VIOLATION: Service modules must not import typer or rich")
        for v in all_violations:
            print(f"  {v}")
        sys.exit(1)
    else:
        dirs_checked = sum(
            1 for d in SERVICE_DIRS if (root / d).exists()
        )
        files_checked = sum(
            len(list((root / d).rglob("*.py")))
            for d in SERVICE_DIRS
            if (root / d).exists()
        )
        print(
            f"ARCH-08 OK: No forbidden imports in {files_checked} files "
            f"across {dirs_checked} service directories"
        )


if __name__ == "__main__":
    main()
