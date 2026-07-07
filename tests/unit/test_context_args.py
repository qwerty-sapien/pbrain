# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.cli.context_args import parse_context_argv


def test_parse_context_argv_collects_multiple_files_until_next_flag() -> None:
    parsed = parse_context_argv(
        ["stokes", "theorem", "--context", "A.pdf", "B.pdf", "C.pdf", "--dryrun"]
    )

    assert parsed.topic_tokens == ("stokes", "theorem")
    assert parsed.context_tokens == ("A.pdf", "B.pdf", "C.pdf")
    assert parsed.ignored_flags == ("--dryrun",)


def test_parse_context_argv_supports_repeated_context_flags() -> None:
    parsed = parse_context_argv(
        ["charts", "--context", "syllabus.pdf", "--context", "notes with spaces.md"]
    )

    assert parsed.topic_tokens == ("charts",)
    assert parsed.context_tokens == ("syllabus.pdf", "notes with spaces.md")
