"""Taskwarrior and Timewarrior hook integration tests.

All adapter integration tests removed per D-21 through D-25 (Phase 9):
SessionManager no longer accepts taskwarrior/timewarrior parameters.
Adapter files remain in pb/adapters/ for future import/export use (D-23).

Functional session lifecycle tests are in:
- tests/unit/test_sessions.py   — core session lifecycle
- tests/unit/test_sessions_timer.py — timer integration
"""
