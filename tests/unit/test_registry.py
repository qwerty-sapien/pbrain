"""Unit tests for CommandHandler and CommandRegistry.

Tests cover:
- CommandRegistry.register: adds handler accessible by name
- CommandRegistry.dispatch: returns True when handled, False when not
- CommandRegistry.dispatch: splits args correctly
- CommandRegistry.dispatch: case-insensitive command lookup
- CommandHandler.aliases: alias dispatch
- CommandRegistry.help_lines: formatted help, no duplicates, registration order
- Edge cases: empty line, multiple handlers coexisting
"""

from __future__ import annotations

from unittest.mock import MagicMock


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_registry():
    """Create a fresh CommandRegistry instance."""
    from pb.core.registry import CommandRegistry

    return CommandRegistry()


def _make_handler(name: str, help_text: str = "Help text", aliases=None, handler=None):
    """Create a CommandHandler with optional mock handler."""
    from pb.core.registry import CommandHandler

    if handler is None:
        handler = MagicMock()
    return CommandHandler(
        name=name,
        help_text=help_text,
        handler=handler,
        aliases=aliases or [],
    )


# ---------------------------------------------------------------------------
# register tests
# ---------------------------------------------------------------------------


def test_register_adds_handler_accessible_by_name():
    """register() adds handler accessible by name."""
    registry = _make_registry()
    handler_fn = MagicMock()
    h = _make_handler("/help", handler=handler_fn)
    registry.register(h)

    # Dispatch to verify handler is accessible
    result = registry.dispatch("/help", {})
    assert result is True
    handler_fn.assert_called_once()


# ---------------------------------------------------------------------------
# dispatch tests
# ---------------------------------------------------------------------------


def test_dispatch_returns_true_for_registered_command():
    """dispatch('/help', ctx) calls handler and returns True."""
    registry = _make_registry()
    handler_fn = MagicMock()
    registry.register(_make_handler("/help", handler=handler_fn))

    result = registry.dispatch("/help", {"key": "value"})

    assert result is True
    handler_fn.assert_called_once_with("", {"key": "value"})


def test_dispatch_returns_false_for_unknown_command():
    """dispatch('unknown', ctx) returns False without calling any handler."""
    registry = _make_registry()
    handler_fn = MagicMock()
    registry.register(_make_handler("/help", handler=handler_fn))

    result = registry.dispatch("unknown", {})

    assert result is False
    handler_fn.assert_not_called()


def test_dispatch_passes_args_to_handler():
    """dispatch('/help extra args', ctx) passes 'extra args' as args to handler."""
    registry = _make_registry()
    handler_fn = MagicMock()
    registry.register(_make_handler("/help", handler=handler_fn))

    registry.dispatch("/help extra args", {})

    handler_fn.assert_called_once_with("extra args", {})


def test_dispatch_is_case_insensitive():
    """dispatch dispatches '/Help' to '/help' handler (case-insensitive)."""
    registry = _make_registry()
    handler_fn = MagicMock()
    registry.register(_make_handler("/help", handler=handler_fn))

    result = registry.dispatch("/Help", {})

    assert result is True
    handler_fn.assert_called_once()


def test_dispatch_returns_false_for_empty_line():
    """dispatch returns False for empty line gracefully."""
    registry = _make_registry()
    result = registry.dispatch("", {})
    assert result is False


def test_dispatch_returns_false_for_whitespace_only():
    """dispatch returns False for whitespace-only line."""
    registry = _make_registry()
    result = registry.dispatch("   ", {})
    assert result is False


# ---------------------------------------------------------------------------
# alias tests
# ---------------------------------------------------------------------------


def test_alias_allows_dispatch_via_alias():
    """CommandHandler with aliases=['/h'] allows dispatch via '/h'."""
    registry = _make_registry()
    handler_fn = MagicMock()
    h = _make_handler("/help", aliases=["/h"], handler=handler_fn)
    registry.register(h)

    result = registry.dispatch("/h", {})

    assert result is True
    handler_fn.assert_called_once()


def test_resolve_unique_prefix_dispatches_without_exact_match():
    """Unique command prefixes resolve before execution."""
    registry = _make_registry()
    handler_fn = MagicMock()
    registry.register(_make_handler("/hint", handler=handler_fn))

    resolution = registry.resolve("/hin")

    assert resolution.status == "unique_prefix"
    assert resolution.command == "/hint"


def test_resolve_ambiguous_prefix_reports_matches():
    """Ambiguous prefixes surface the canonical matching commands."""
    registry = _make_registry()
    registry.register(_make_handler("/hint"))
    registry.register(_make_handler("/harder"))
    registry.register(_make_handler("/help"))

    resolution = registry.resolve("/h")

    assert resolution.status == "ambiguous_prefix"
    assert resolution.matches == ("/harder", "/help", "/hint")


# ---------------------------------------------------------------------------
# help_lines tests
# ---------------------------------------------------------------------------


def test_help_lines_returns_formatted_help_strings():
    """help_lines() returns formatted help strings, one per unique handler."""
    registry = _make_registry()
    registry.register(_make_handler("/help", help_text="Show help"))
    registry.register(_make_handler("/new", help_text="Start new session"))

    lines = registry.help_lines()

    assert len(lines) == 2
    assert any("/help" in line and "Show help" in line for line in lines)
    assert any("/new" in line and "Start new session" in line for line in lines)


def test_help_lines_no_duplicates_from_aliases():
    """help_lines() returns one line per unique handler (no duplicates from aliases)."""
    registry = _make_registry()
    h = _make_handler("/help", help_text="Show help", aliases=["/h", "/?"])
    registry.register(h)

    lines = registry.help_lines()

    # Should be exactly 1 line — aliases don't create duplicate entries
    assert len(lines) == 1
    assert "/help" in lines[0]


def test_help_lines_preserves_registration_order():
    """help_lines() preserves registration order."""
    registry = _make_registry()
    registry.register(_make_handler("/first", help_text="First"))
    registry.register(_make_handler("/second", help_text="Second"))
    registry.register(_make_handler("/third", help_text="Third"))

    lines = registry.help_lines()

    assert len(lines) == 3
    assert "/first" in lines[0]
    assert "/second" in lines[1]
    assert "/third" in lines[2]


# ---------------------------------------------------------------------------
# coexistence tests
# ---------------------------------------------------------------------------


def test_multiple_handlers_coexist_without_interference():
    """Multiple handlers can coexist without interfering with each other."""
    registry = _make_registry()
    handler_a = MagicMock()
    handler_b = MagicMock()

    registry.register(_make_handler("/cmd-a", handler=handler_a))
    registry.register(_make_handler("/cmd-b", handler=handler_b))

    registry.dispatch("/cmd-a", {})

    handler_a.assert_called_once()
    handler_b.assert_not_called()

    registry.dispatch("/cmd-b", {})

    handler_b.assert_called_once()
