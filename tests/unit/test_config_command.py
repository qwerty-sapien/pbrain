from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, call, patch

import pytest
import typer.main

from pb.cli.commands import config as config_cmd
from pb.cli.commands import model as model_cmd
from pb.cli.input_router import PbCommandResolver, classify_interactive_input
from pb.cli.main import app as pb_app
from pb.cli.pickers import PickerResult
from pb.core.learning_block_flow import collect_revision_feedback
from pb.core.product_control import AdaptiveOption, ControlDecision, ControlState
from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL, PRO_MODEL
from pb.storage.config import (
    ANTHROPIC_FAST_MODEL,
    Config,
    GeneralConfig,
    LLMConfig,
    ModelRolesConfig,
    OPENAI_FAST_MODEL,
    ProviderConfig,
    VaultProfileConfig,
    create_default_config,
    load_config,
    set_config_value,
    set_default_model_binding,
)


@pytest.fixture
def temp_config(monkeypatch, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path=str(vault)),
        vaults={
            "main": VaultProfileConfig(
                path=str(vault),
                data_dir=str(tmp_path / "data"),
            )
        },
    )
    monkeypatch.setattr(config_cmd, "get_config", lambda *args, **kwargs: config)
    return config


def test_interactive_config_updates_collect_selected_values(temp_config, monkeypatch):
    monkeypatch.setattr(
        config_cmd,
        "pick_many_choices",
        lambda *args, **kwargs: ["llm.default_model", "ui.content_width_ratio"],
    )
    answers = iter(["gemini-3.1-pro-preview", "0.8"])
    monkeypatch.setattr(config_cmd, "prompt_text", lambda *args, **kwargs: next(answers))

    updates = config_cmd._interactive_config_updates()

    assert updates == [
        ("llm.default_model", "gemini-3.1-pro-preview"),
        ("ui.content_width_ratio", 0.8),
    ]


def test_interactive_config_updates_hide_developer_keys_by_default(temp_config, monkeypatch):
    captured: dict[str, list[str]] = {}

    def fake_pick(options, *args, **kwargs):
        captured["keys"] = [key for key, _label in options]
        return []

    monkeypatch.setattr(config_cmd, "pick_many_choices", fake_pick)

    updates = config_cmd._interactive_config_updates(developer=False)

    assert updates == []
    assert "providers.gemini.base_url" not in captured["keys"]
    assert "llm.default_model" in captured["keys"]


def test_config_set_without_key_uses_interactive_multiselect(monkeypatch):
    monkeypatch.setattr(config_cmd.sys.stdin, "isatty", lambda: True)
    monkeypatch.setattr(
        config_cmd,
        "_interactive_config_updates",
        lambda **kwargs: [("ui.plain_mode", True), ("llm.default_model", "gemini-3.1-pro-preview")],
    )

    with patch("pb.cli.commands.config.set_config_value") as mock_set:
        config_cmd.config_set(None, None)

    mock_set.assert_has_calls(
        [
            call("ui", "plain_mode", True),
            call("llm", "default_model", "gemini-3.1-pro-preview"),
        ]
    )


def test_config_agents_pin_and_clear_override(temp_db, capsys):
    from pb.core.agent_weights import get_weight_override

    config_cmd.config_agents("pin", "review", json_out=False)
    assert get_weight_override("review") == "pin"
    assert "review: pin" in capsys.readouterr().out

    config_cmd.config_agents("clear", "review", json_out=False)
    assert get_weight_override("review") == ""
    assert "review: cleared" in capsys.readouterr().out


def test_config_agents_apply_and_revert_instruction_patch(temp_db, capsys):
    from pb.core.agent_instruction_judge import (
        active_agent_instruction_patch,
        create_agent_instruction_patch,
    )

    record = create_agent_instruction_patch(
        agent_id="review",
        summary="Make review direct.",
        instruction_patch="Keep review output direct and evidence-linked.",
    )

    config_cmd.config_agents("apply", record.id, json_out=False)
    assert active_agent_instruction_patch("review") == (
        "Keep review output direct and evidence-linked."
    )
    assert f"{record.id}: applied for review" in capsys.readouterr().out

    config_cmd.config_agents("revert", record.id, json_out=False)
    assert active_agent_instruction_patch("review") == ""
    assert f"{record.id}: reverted for review" in capsys.readouterr().out


def test_gemini_fast_roles_repair_legacy_flash_bindings():
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp/vault"),
        vaults={"main": VaultProfileConfig(path="/tmp/vault")},
        providers={"gemini": ProviderConfig(api_key_env="GEMINI_API_KEY", default_model=FLASH_MODEL)},
        llm=LLMConfig(provider="gemini", default_model=FLASH_MODEL),
        model_roles=ModelRolesConfig(
            default=f"gemini:{FLASH_MODEL}",
            planner=f"gemini:{FLASH_MODEL}",
            reviewer=f"gemini:{FLASH_MODEL}",
            recall=f"gemini:{FLASH_MODEL}",
            fast=f"gemini:{FLASH_MODEL}",
            fast_inference=f"gemini:{FLASH_MODEL}",
            namer=f"gemini:{FLASH_MODEL}",
        ),
    )

    assert config.model_roles.default == f"gemini:{FLASH_MODEL}"
    assert config.model_roles.fast == f"gemini:{FLASH_LITE_MODEL}"
    assert config.model_roles.fast_inference == f"gemini:{FLASH_LITE_MODEL}"


def test_provider_fast_roles_default_to_lightweight_models():
    openai_config = Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp/vault"),
        vaults={"main": VaultProfileConfig(path="/tmp/vault")},
        providers={
            "openai": ProviderConfig(
                api_key_env="OPENAI_API_KEY",
                default_model="gpt-5.4",
            )
        },
        llm=LLMConfig(provider="openai", default_model="gpt-5.4"),
        model_roles=ModelRolesConfig(default="openai:gpt-5.4"),
    )
    anthropic_config = Config(
        general=GeneralConfig(active_vault="main", vault_path="/tmp/vault"),
        vaults={"main": VaultProfileConfig(path="/tmp/vault")},
        providers={
            "anthropic": ProviderConfig(
                api_key_env="ANTHROPIC_API_KEY",
                default_model="claude-sonnet-4-5",
            )
        },
        llm=LLMConfig(provider="anthropic", default_model="claude-sonnet-4-5"),
        model_roles=ModelRolesConfig(default="anthropic:claude-sonnet-4-5"),
    )

    assert openai_config.model_roles.fast == f"openai:{OPENAI_FAST_MODEL}"
    assert openai_config.model_roles.fast_inference == f"openai:{OPENAI_FAST_MODEL}"
    assert anthropic_config.model_roles.fast == f"anthropic:{ANTHROPIC_FAST_MODEL}"
    assert anthropic_config.model_roles.fast_inference == f"anthropic:{ANTHROPIC_FAST_MODEL}"


def test_default_model_updates_preserve_gemini_fast_roles(tmp_path, monkeypatch):
    vault = tmp_path / "vault"
    vault.mkdir()
    config_path = tmp_path / "config.toml"
    monkeypatch.setenv("PRODUCTIVEBRAIN_CONFIG_PATH", str(config_path))
    config_path.write_text(create_default_config(str(vault), provider="gemini", model=FLASH_MODEL), encoding="utf-8")

    cfg = set_default_model_binding(f"gemini:{PRO_MODEL}", path=config_path)
    assert cfg.model_roles.default == f"gemini:{PRO_MODEL}"
    assert cfg.model_roles.fast == f"gemini:{FLASH_LITE_MODEL}"
    assert cfg.model_roles.fast_inference == f"gemini:{FLASH_LITE_MODEL}"

    set_config_value("llm", "default_model", FLASH_MODEL, path=config_path)
    reloaded = load_config(config_path, force_reload=True)
    assert reloaded.model_roles.default == f"gemini:{FLASH_MODEL}"
    assert reloaded.model_roles.fast == f"gemini:{FLASH_LITE_MODEL}"
    assert reloaded.model_roles.fast_inference == f"gemini:{FLASH_LITE_MODEL}"


def test_interactive_model_selection_accepts_typed_model_id(monkeypatch, tmp_path):
    vault = tmp_path / "vault"
    vault.mkdir()
    config = Config(
        general=GeneralConfig(active_vault="main", vault_path=str(vault)),
        vaults={"main": VaultProfileConfig(path=str(vault))},
        providers={"gemini": ProviderConfig(api_key_env="GEMINI_API_KEY", default_model=FLASH_MODEL)},
        llm=LLMConfig(provider="gemini", default_model=FLASH_MODEL),
        model_roles=ModelRolesConfig(default=f"gemini:{FLASH_MODEL}"),
    )
    ctx = SimpleNamespace(obj={"config": config})
    selections = iter(["gemini", "gemini-custom-preview"])
    captured: dict[str, str] = {}

    monkeypatch.setattr(model_cmd, "pick_single_choice", lambda *args, **kwargs: next(selections))

    def _fake_set_default(binding: str):
        captured["binding"] = binding
        return SimpleNamespace(
            model_roles=SimpleNamespace(
                fast=f"gemini:{FLASH_LITE_MODEL}",
                fast_inference=f"gemini:{FLASH_LITE_MODEL}",
            )
        )

    monkeypatch.setattr(model_cmd, "set_default_model_binding", _fake_set_default)

    model_cmd._interactive_model_selection(ctx)

    assert captured["binding"] == "gemini:gemini-custom-preview"


def test_model_slash_alias_routes_to_pb_model_command():
    resolver = PbCommandResolver(typer.main.get_command(pb_app))

    decision = classify_interactive_input(
        "/model",
        pb_command_resolver=resolver,
        slash_registry=None,
        active_learning=False,
        allow_shell_commands=True,
        allow_nl_dispatch=True,
    )

    assert decision.kind == "pb_command"
    assert decision.command == "model"


class _FlowRepoStub:
    def list_feedback_events(self, **kwargs):
        return []


def _flow_engine_stub():
    state = ControlState(scope="artifact")
    decision = ControlDecision(action="local_refine", reason="ok", instruction="Refine it.")
    event = SimpleNamespace(kind="custom_revision")
    engine = SimpleNamespace()
    engine.load_state = lambda **kwargs: ("artifact:test", state)
    engine.default_options = lambda **kwargs: [
        AdaptiveOption(key="needs_prerequisite", label="Start earlier", description="Rebase first.", control_signal="needs_prerequisite"),
        AdaptiveOption(key="wrong_scope", label="Wrong scope", description="Shift the topic.", control_signal="wrong_scope"),
    ]
    engine.record_feedback = lambda **kwargs: (event, state, decision)
    return engine


def test_collect_revision_feedback_uses_inline_custom_without_legacy_prompts(monkeypatch, tmp_path):
    prompts: list[str] = []
    monkeypatch.setattr(
        "pb.core.learning_block_flow.pick_single_choice",
        lambda *args, **kwargs: PickerResult(kind="inline_text", value="Focus this on transfer examples."),
    )
    monkeypatch.setattr(
        "pb.core.learning_block_flow.prompt_text",
        lambda label, **kwargs: prompts.append(label) or "",
    )

    result = collect_revision_feedback(
        engine=_flow_engine_stub(),
        repo=_FlowRepoStub(),
        runtime_ctx=SimpleNamespace(vault_path=Path(tmp_path)),
        mode="study",
        artifact_kind="study_block",
        artifact_id="bayes-rule",
        current_artifact="{}",
        domain="probability",
        target="Bayes rule",
    )

    assert result is not None
    assert result.free_text == "Focus this on transfer examples."
    assert prompts == []


def test_collect_revision_feedback_discuss_path_uses_discuss_prompt(monkeypatch, tmp_path):
    prompts: list[str] = []
    monkeypatch.setattr(
        "pb.core.learning_block_flow.pick_single_choice",
        lambda *args, **kwargs: PickerResult(kind="selection", value="chat"),
    )
    monkeypatch.setattr(
        "pb.core.learning_block_flow.prompt_text",
        lambda label, **kwargs: prompts.append(label) or "Make this more conversational and concrete.",
    )

    result = collect_revision_feedback(
        engine=_flow_engine_stub(),
        repo=_FlowRepoStub(),
        runtime_ctx=SimpleNamespace(vault_path=Path(tmp_path)),
        mode="teach",
        artifact_kind="teach_block",
        artifact_id="konjunktiv-ii",
        current_artifact="{}",
        domain="german",
        target="Konjunktiv II",
    )

    assert result is not None
    assert result.free_text == "Make this more conversational and concrete."
    assert prompts == ["Discuss"]
