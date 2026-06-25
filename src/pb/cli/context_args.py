# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Shared argv pre-parser for variable-length `--context` learning inputs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class ParsedContextArgv:
    """Parsed free-form learning argv with detached context-file paths."""

    topic_tokens: tuple[str, ...] = ()
    context_tokens: tuple[str, ...] = ()
    ignored_flags: tuple[str, ...] = ()


def parse_context_argv(tokens: Iterable[str]) -> ParsedContextArgv:
    """Extract `--context` values until the next flag or end-of-input."""

    topic_tokens: list[str] = []
    context_tokens: list[str] = []
    ignored_flags: list[str] = []
    argv = [str(token) for token in tokens if str(token).strip()]
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--context":
            index += 1
            while index < len(argv):
                next_token = argv[index]
                if next_token.startswith("-"):
                    break
                context_tokens.append(next_token)
                index += 1
            continue
        if token.startswith("--context="):
            value = token.split("=", 1)[1].strip()
            if value:
                context_tokens.append(value)
            index += 1
            continue
        if token.startswith("-"):
            ignored_flags.append(token)
            index += 1
            continue
        topic_tokens.append(token)
        index += 1
    return ParsedContextArgv(
        topic_tokens=tuple(topic_tokens),
        context_tokens=tuple(context_tokens),
        ignored_flags=tuple(ignored_flags),
    )


def resolve_context_paths(tokens: Iterable[str]) -> list[Path]:
    """Expand parsed context tokens into concrete paths."""

    return [Path(token).expanduser() for token in parse_context_argv(tokens).context_tokens]
