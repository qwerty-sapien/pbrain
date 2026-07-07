# SPDX-License-Identifier: AGPL-3.0-or-later
"""Unit tests for pb.cli.normalize."""
import pytest

from pb.cli.normalize import join_words_safe


class TestJoinWordsSafe:
    def test_normal_words(self):
        assert join_words_safe(["integration", "by", "parts"]) == "integration by parts"

    def test_strips_double_hyphen_flags(self):
        assert join_words_safe(["integration", "by", "parts", "--study"]) == "integration by parts"

    def test_strips_multiple_flags(self):
        assert join_words_safe(["rust", "--steps", "--yes"]) == "rust"

    def test_preserves_hyphenated_words(self):
        assert join_words_safe(["step-by-step", "guide"]) == "step-by-step guide"

    def test_preserves_single_hyphen(self):
        assert join_words_safe(["-v", "topic"]) == "-v topic"

    def test_none_returns_empty(self):
        assert join_words_safe(None) == ""

    def test_empty_list_returns_empty(self):
        assert join_words_safe([]) == ""

    def test_only_flags_returns_empty(self):
        assert join_words_safe(["--yes"]) == ""
