def test_readline_imported_by_helpers():
    """helpers.py must import readline so text prompts handle arrow keys."""
    import pb.cli.helpers  # noqa: F401
    import sys
    assert "readline" in sys.modules or "gnureadline" in sys.modules
