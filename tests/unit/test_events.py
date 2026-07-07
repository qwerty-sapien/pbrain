"""Unit tests for pb.events — EventBus, sinks, and secret scrubbing."""

from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from pb.events import (
    EventBus,
    JsonlFileSink,
    NoOpSink,
    _scrub_secrets,
    emit,
    get_event_bus,
    reset_event_bus,
)


class TestNoOpSink:
    def test_emit_does_nothing(self):
        sink = NoOpSink()
        sink.emit({"event": "test"})
        sink.close()


class TestJsonlFileSink:
    def test_writes_jsonl_to_file(self, tmp_path: Path):
        log_file = tmp_path / "events.jsonl"
        sink = JsonlFileSink(path=log_file)
        sink.emit({"event": "app.started", "ts": "2026-01-01T00:00:00Z"})
        sink.emit({"event": "app.exited", "ts": "2026-01-01T00:01:00Z"})
        sink.close()

        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 2
        first = json.loads(lines[0])
        assert first["event"] == "app.started"
        second = json.loads(lines[1])
        assert second["event"] == "app.exited"

    def test_creates_parent_directories(self, tmp_path: Path):
        log_file = tmp_path / "deep" / "nested" / "events.jsonl"
        sink = JsonlFileSink(path=log_file)
        sink.emit({"event": "test"})
        assert log_file.exists()

    def test_rotation_at_size_limit(self, tmp_path: Path):
        log_file = tmp_path / "events.jsonl"
        log_file.write_text("x" * (50 * 1024 * 1024 + 1))
        sink = JsonlFileSink(path=log_file)
        sink.emit({"event": "post.rotate"})

        rotated = tmp_path / "events.jsonl.1"
        assert rotated.exists()
        lines = log_file.read_text().strip().splitlines()
        assert len(lines) == 1
        assert json.loads(lines[0])["event"] == "post.rotate"


class TestEventBus:
    def test_emit_adds_event_type_and_timestamp(self, tmp_path: Path):
        log_file = tmp_path / "events.jsonl"
        bus = EventBus(sink=JsonlFileSink(path=log_file))
        bus.emit("command.started", command="goal")
        bus.close()

        record = json.loads(log_file.read_text().strip())
        assert record["event"] == "command.started"
        assert "ts" in record
        assert "mono" in record
        assert record["command"] == "goal"

    def test_noop_default(self):
        bus = EventBus()
        assert isinstance(bus.sink, NoOpSink)
        bus.emit("test.event")


class TestSecretScrubbing:
    def test_redacts_known_secret_keys(self):
        payload = {
            "GEMINI_API_KEY": "sk-1234",
            "model": "flash",
            "OPENAI_API_KEY": "sk-abcd",
        }
        clean = _scrub_secrets(payload)
        assert clean["GEMINI_API_KEY"] == "<redacted>"
        assert clean["OPENAI_API_KEY"] == "<redacted>"
        assert clean["model"] == "flash"

    def test_redacts_suffix_pattern_keys(self):
        payload = {"my_api_key": "secret123", "auth_token": "tok-999", "name": "safe"}
        clean = _scrub_secrets(payload)
        assert clean["my_api_key"] == "<redacted>"
        assert clean["auth_token"] == "<redacted>"
        assert clean["name"] == "safe"

    def test_scrubs_nested_dicts(self):
        payload = {"outer": {"GEMINI_API_KEY": "secret"}, "ok": 1}
        clean = _scrub_secrets(payload)
        assert clean["outer"]["GEMINI_API_KEY"] == "<redacted>"
        assert clean["ok"] == 1

    def test_event_log_does_not_contain_secret_env_vars(self, tmp_path: Path):
        log_file = tmp_path / "events.jsonl"
        bus = EventBus(sink=JsonlFileSink(path=log_file))
        bus.emit("llm.request.started", model="flash", GEMINI_API_KEY="real-secret-key")
        bus.close()

        content = log_file.read_text()
        assert "real-secret-key" not in content
        record = json.loads(content.strip())
        assert record["GEMINI_API_KEY"] == "<redacted>"


class TestGlobalBus:
    def setup_method(self):
        reset_event_bus(None)

    def teardown_method(self):
        reset_event_bus(None)

    def test_disabled_by_default(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("PB_EVENT_LOG", None)
            reset_event_bus(None)
            bus = get_event_bus()
            assert isinstance(bus.sink, NoOpSink)

    def test_enabled_via_env(self, tmp_path: Path):
        log_file = tmp_path / "events.jsonl"
        with patch.dict(os.environ, {"PB_EVENT_LOG": "1", "PB_EVENT_LOG_PATH": str(log_file)}):
            reset_event_bus(None)
            bus = get_event_bus()
            assert isinstance(bus.sink, JsonlFileSink)
            emit("test.global")
            assert log_file.exists()
            content = log_file.read_text()
            assert "test.global" in content

    def test_disabled_produces_no_file(self, tmp_path: Path):
        log_file = tmp_path / "should_not_exist.jsonl"
        with patch.dict(os.environ, {"PB_EVENT_LOG": "0", "PB_EVENT_LOG_PATH": str(log_file)}):
            reset_event_bus(None)
            emit("test.noop")
            assert not log_file.exists()
