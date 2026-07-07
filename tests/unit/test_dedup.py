"""Tests for fuzzy dedup on task/todo/thought input."""
from pb.core.dedup import find_similar_task


class FakeTask:
    def __init__(self, title, state="active"):
        self.title = title
        self.state = state


def test_exact_duplicate_detected():
    existing = [FakeTask("Buy Milk")]
    match = find_similar_task("Buy Milk", existing)
    assert match is not None
    assert match.title == "Buy Milk"


def test_case_insensitive_match():
    existing = [FakeTask("buy milk")]
    match = find_similar_task("Buy Milk", existing)
    assert match is not None


def test_minor_variation_detected():
    existing = [FakeTask("Buy Milk")]
    match = find_similar_task("buy some milk", existing)
    assert match is not None


def test_different_task_not_matched():
    existing = [FakeTask("Buy Milk")]
    match = find_similar_task("Study calculus", existing)
    assert match is None


def test_empty_existing_returns_none():
    match = find_similar_task("Buy Milk", [])
    assert match is None


def test_short_input_requires_higher_similarity():
    existing = [FakeTask("Go")]
    match = find_similar_task("Go", existing)
    assert match is not None


def test_substring_match_detected():
    existing = [FakeTask("Buy Milk from the store")]
    match = find_similar_task("Buy Milk", existing)
    assert match is not None
