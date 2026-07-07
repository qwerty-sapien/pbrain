"""Tests for domain enums."""

import pytest

from pb.domain.enums import (
    EnergyType,
    Horizon,
    PacketType,
    ProjectStatus,
    ProjectType,
    SessionMode,
    TaskOutcome,
    TaskState,
)


class TestTaskState:
    """Test TaskState enum."""

    def test_all_states_exist(self):
        states = [s.value for s in TaskState]
        assert "active" in states
        assert "paused" in states
        assert "done" in states

    def test_state_count(self):
        assert len(TaskState) == 3


class TestSessionMode:
    """Test SessionMode enum."""

    def test_all_modes_exist(self):
        modes = [m.value for m in SessionMode]
        assert "focus" in modes
        assert "supervisory" in modes
        assert "review" in modes
        assert "practice" in modes


class TestEnergyType:
    """Test EnergyType enum."""

    def test_all_types_exist(self):
        types = [t.value for t in EnergyType]
        assert "deep" in types
        assert "shallow" in types
        assert "supervisory" in types
        assert "admin" in types
        assert "practice" in types


class TestHorizon:
    """Test Horizon enum."""

    def test_all_horizons_exist(self):
        horizons = [h.value for h in Horizon]
        assert "today" in horizons
        assert "week" in horizons
        assert "month" in horizons
        assert "quarter" in horizons
        assert "six_month" in horizons


class TestProjectType:
    """Test ProjectType enum."""

    def test_all_types_exist(self):
        types = [t.value for t in ProjectType]
        assert "build" in types
        assert "study" in types
        assert "practice" in types
        assert "admin" in types
        assert "research" in types


class TestTaskOutcome:
    """Test TaskOutcome enum."""

    def test_all_outcomes_exist(self):
        outcomes = [o.value for o in TaskOutcome]
        assert "done" in outcomes
        assert "partial" in outcomes
        assert "blocked" in outcomes
        assert "abandoned" in outcomes
