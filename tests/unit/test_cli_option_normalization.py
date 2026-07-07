# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/productivity
# Compliance fingerprint: PB-2026-A17F

from __future__ import annotations

from pb.cli.main import _normalize_global_option_order, _normalize_topic_command_option_order


def test_global_yes_can_follow_nested_command() -> None:
    argv = ["pb", "context", "add", "source.md", "--yes"]

    assert _normalize_global_option_order(argv) == ["pb", "--yes", "context", "add", "source.md"]


def test_global_config_and_vault_can_follow_subcommand_arguments() -> None:
    argv = [
        "pb",
        "goal",
        "add",
        "learn kernels",
        "--config",
        "/tmp/pb.toml",
        "--vault=research",
    ]

    assert _normalize_global_option_order(argv) == [
        "pb",
        "--config",
        "/tmp/pb.toml",
        "--vault=research",
        "goal",
        "add",
        "learn kernels",
    ]


def test_double_dash_stops_global_option_normalization() -> None:
    argv = ["pb", "do", "--", "literal", "--yes"]

    assert _normalize_global_option_order(argv) == argv


def test_study_options_after_topic_move_before_topic() -> None:
    argv = ["pb", "--yes", "study", "Rust", "async", "--duration", "10m", "--understand", "--steps"]

    assert _normalize_topic_command_option_order(argv) == [
        "pb",
        "--yes",
        "study",
        "--duration",
        "10m",
        "--understand",
        "--steps",
        "Rust",
        "async",
    ]


def test_practise_options_after_topic_move_before_topic() -> None:
    argv = [
        "pb",
        "practise",
        "Bayes",
        "word",
        "problems",
        "--duration",
        "5m",
        "--drill",
        "translate story",
        "--cues",
        "name prior",
    ]

    assert _normalize_topic_command_option_order(argv) == [
        "pb",
        "practise",
        "--duration",
        "5m",
        "--drill",
        "translate story",
        "--cues",
        "name prior",
        "Bayes",
        "word",
        "problems",
    ]


def test_topic_option_normalization_leaves_subcommands_alone() -> None:
    argv = ["pb", "study", "plan", "--time", "30"]

    assert _normalize_topic_command_option_order(argv) == argv
