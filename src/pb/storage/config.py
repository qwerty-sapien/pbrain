# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Configuration loading and management."""

from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

import tomli
from pydantic import BaseModel, Field, model_validator

from pb.llm.gemini import FLASH_LITE_MODEL, FLASH_MODEL

DEFAULT_PRODUCT_NAME = "ProductiveBrain"
DEFAULT_QUARANTINE_FOLDER = "Learning/Inbox/pb"
DEFAULT_DATA_HOME = "~/.local/share/productivebrain"
DEFAULT_STATE_HOME = "~/.local/state/productivebrain"

DEFAULT_PROVIDER_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
}
DEFAULT_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "openrouter": "https://openrouter.ai/api/v1",
}


class VaultProfileConfig(BaseModel):
    """One named Obsidian vault profile."""

    path: str
    data_dir: str = ""
    quarantine_folder: str = DEFAULT_QUARANTINE_FOLDER


class GeneralConfig(BaseModel):
    """General configuration settings."""

    active_vault: str = "main"
    verbose: bool = False
    vault_path: Optional[str] = None


class StorageConfig(BaseModel):
    """Storage location configuration."""

    data_dir: str = DEFAULT_DATA_HOME
    log_dir: str = DEFAULT_STATE_HOME


class NotesConfig(BaseModel):
    """Durable markdown defaults."""

    default_write_mode: str = "quarantine"
    link_style: str = "wikilink"


class InteractionConfig(BaseModel):
    """CLI interaction defaults."""

    mode: str = "guided"
    question_batch_size: int = 3
    confirmation_style: str = "preview"


class CommitPolicyConfig(BaseModel):
    """AI-content commit defaults."""

    session_summaries: str = "auto_to_quarantine"
    daily_reviews: str = "auto_to_quarantine"
    weekly_reviews: str = "auto_to_quarantine"
    daily_plans: str = "preview"
    goals: str = "preview"
    anki_candidates: str = "preview"
    vault_merge: str = "confirm"
    destructive_actions: str = "confirm"


class ProviderConfig(BaseModel):
    """One configured LLM provider."""

    api_key_env: str = ""
    default_model: str = ""
    base_url: str = ""


class ModelRolesConfig(BaseModel):
    """Model-role bindings using provider:model strings."""

    default: str = ""
    planner: str = ""
    reviewer: str = ""
    recall: str = ""
    fast: str = ""
    fast_inference: str = ""
    namer: str = ""


class AdaptersConfig(BaseModel):
    """External adapter configuration."""

    taskwarrior_enabled: bool = False
    timewarrior_enabled: bool = False


class ScrapeSourceConfig(BaseModel):
    """Per-source scraper configuration."""

    name: str
    enabled: bool = True
    url: str = ""
    api_key: str = ""


class ScrapeFiltersConfig(BaseModel):
    """Relevance filter configuration for scraped events."""

    city: str = ""
    radius_km: int = 50
    keywords: list[str] = Field(default_factory=list)
    categories: list[str] = Field(default_factory=list)


class CrawlFieldConfig(BaseModel):
    """One field in a crawl extraction schema."""

    name: str
    selector: str = ""
    type: str = "text"
    attribute: str = ""
    default: str = ""


class CrawlSourceConfig(BaseModel):
    """Config for one crawl4ai JS-rendered source."""

    name: str
    enabled: bool = True
    url: str
    base_selector: str
    fields: list[CrawlFieldConfig] = Field(default_factory=list)
    required_fields: list[str] = Field(default_factory=lambda: ["title", "url"])


class ScrapeConfig(BaseModel):
    """Scraping infrastructure configuration."""

    sources: list[ScrapeSourceConfig] = Field(default_factory=list)
    crawl_sources: list[CrawlSourceConfig] = Field(default_factory=list)
    filters: ScrapeFiltersConfig = Field(default_factory=ScrapeFiltersConfig)
    event_categories: list[str] = Field(
        default_factory=lambda: [
            "data-science",
            "engineering",
            "business",
            "social",
            "fitness",
            "arts",
            "community",
            "career",
            "learning",
            "other",
        ]
    )


class GmailConfig(BaseModel):
    """Gmail integration configuration (legacy compatibility)."""

    client_secrets_path: str = ""
    senders: list[str] = Field(default_factory=list)


class IngestRelevanceConfig(BaseModel):
    """LLM relevance filtering configuration."""

    threshold: float = 0.3
    batch_size: int = 100


class IngestConfig(BaseModel):
    """Unified ingestion configuration."""

    relevance: IngestRelevanceConfig = Field(default_factory=IngestRelevanceConfig)


class UIConfig(BaseModel):
    """UI and display configuration."""

    theme: str = "catppuccin"
    theme_overrides: dict[str, Any] = Field(default_factory=dict)
    theme_file: str = ""
    plain_mode: bool = False
    content_width_ratio: float = Field(default=0.70, ge=0.40, le=1.0)
    max_content_width: int = 0
    language: str = "auto"
    model_max_tokens: dict[str, int] = Field(default_factory=dict)


class LearningConfig(BaseModel):
    """Learning lifecycle configuration."""

    weights: dict[str, float] = Field(default_factory=lambda: {
        "read": 1.0,
        "query": 1.0,
        "study": 3.0,
        "socratic": 5.0,
    })
    promotion_threshold: float = 3.0
    decay_days_default: int = 7
    learnt_suggestion_threshold: float = 10.0
    scoring_weights: dict[str, float] = Field(default_factory=lambda: {
        "semantic": 0.3,
        "link": 0.15,
        "backlink": 0.1,
        "tag_affinity": 0.1,
        "recency": 0.15,
        "usage": 0.1,
        "redundancy": -0.1,
    })
    decay_thresholds: dict[str, int] = Field(default_factory=lambda: {
        "languages": 3,
        "piano": 2,
        "ml": 7,
        "math": 7,
        "communication": 5,
        "_default": 5,
    })
    embedding_dimensions: int = 768
    top_n_candidates: int = 20
    model_policy: "LearningModelPolicyConfig" = Field(default_factory=lambda: LearningModelPolicyConfig())


class LearningModelPolicyConfig(BaseModel):
    """Role bindings for learning-specific model-selection decisions."""

    routing: str = "fast_inference"
    command_repair: str = "fast_inference"
    mcq_or_cloze_repair: str = "fast_inference"
    answer_check: str = "fast_inference"
    small_retry: str = "fast_inference"
    recall_inline: str = "fast_inference"
    session_explain: str = "default"
    lesson_hint_intuitive: str = "default"
    drill_generation: str = "default"
    complex_free_response_eval: str = "default"
    lesson_planning: str = "planner"
    scoped_recall_generation: str = "recall"


class LLMConfig(BaseModel):
    """Legacy compatibility section for runtime defaults."""

    provider: str = "gemini"
    backend: str = "auto"
    default_model: str = FLASH_MODEL
    prompt_template_version: str = "v3"
    require_llm_for_core_workflows: bool = True
    long_model_timeout_seconds: int = 90
    auto_pro_fallback: bool = False


class ScaffoldConfig(BaseModel):
    """Scaffold generator configuration."""

    gcs_bucket: str = ""
    gcp_project: str = ""
    location: str = "us-central1"
    poll_interval_seconds: int = 90
    similarity_threshold: float = 0.6


def _default_profile_data_dir(name: str) -> str:
    return f"{DEFAULT_DATA_HOME}/vaults/{name}"


def _parse_role_binding(binding: str) -> tuple[str, str]:
    raw = (binding or "").strip()
    if ":" not in raw:
        return "gemini", raw or FLASH_MODEL
    provider, model = raw.split(":", 1)
    return provider.strip().lower(), model.strip()


def _gemini_flash_binding() -> str:
    return f"gemini:{FLASH_MODEL}"


def _gemini_flash_lite_binding() -> str:
    return f"gemini:{FLASH_LITE_MODEL}"


def _should_repair_gemini_fast_role(binding: str, default_binding: str) -> bool:
    clean = (binding or "").strip()
    if not clean:
        return True
    return clean == (default_binding or "").strip() == _gemini_flash_binding()


def _default_role_bindings(provider_name: str, default_model: str) -> dict[str, str]:
    provider = (provider_name or "gemini").strip().lower()
    default_binding = f"{provider}:{default_model}"
    fast_binding = _gemini_flash_lite_binding() if provider == "gemini" else default_binding
    return {
        "default": default_binding,
        "planner": default_binding,
        "reviewer": default_binding,
        "recall": default_binding,
        "fast": fast_binding,
        "fast_inference": fast_binding,
        "namer": default_binding,
    }


class Config(BaseModel):
    """Root configuration model."""

    general: GeneralConfig
    vaults: dict[str, VaultProfileConfig] = Field(default_factory=dict)
    notes: NotesConfig = Field(default_factory=NotesConfig)
    interaction: InteractionConfig = Field(default_factory=InteractionConfig)
    commit_policy: CommitPolicyConfig = Field(default_factory=CommitPolicyConfig)
    providers: dict[str, ProviderConfig] = Field(default_factory=dict)
    model_roles: ModelRolesConfig = Field(default_factory=ModelRolesConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    adapters: AdaptersConfig = Field(default_factory=AdaptersConfig)
    scrape: ScrapeConfig = Field(default_factory=ScrapeConfig)
    gmail: GmailConfig = Field(default_factory=GmailConfig)
    ingest: IngestConfig = Field(default_factory=IngestConfig)
    ui: UIConfig = Field(default_factory=UIConfig)
    learning: LearningConfig = Field(default_factory=LearningConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    scaffold: ScaffoldConfig = Field(default_factory=ScaffoldConfig)
    preferences: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _normalize(self) -> "Config":
        if not self.vaults and self.general.vault_path:
            migrated_data_dir = self.storage.data_dir or _default_profile_data_dir("main")
            self.vaults = {
                "main": VaultProfileConfig(
                    path=self.general.vault_path,
                    data_dir=migrated_data_dir,
                    quarantine_folder=DEFAULT_QUARANTINE_FOLDER,
                )
            }
            self.general.active_vault = "main"

        if not self.vaults:
            self.vaults = {
                "main": VaultProfileConfig(
                    path="",
                    data_dir=_default_profile_data_dir("main"),
                    quarantine_folder=DEFAULT_QUARANTINE_FOLDER,
                )
            }

        if self.general.active_vault not in self.vaults:
            self.general.active_vault = next(iter(self.vaults))

        for name, profile in self.vaults.items():
            if not profile.data_dir:
                profile.data_dir = _default_profile_data_dir(name)
            if not profile.quarantine_folder:
                profile.quarantine_folder = DEFAULT_QUARANTINE_FOLDER

        active_profile = self.vaults[self.general.active_vault]
        self.general.vault_path = active_profile.path

        if not self.providers:
            provider_name = (self.llm.provider or "gemini").strip().lower()
            self.providers[provider_name] = ProviderConfig(
                api_key_env=DEFAULT_PROVIDER_ENVS.get(provider_name, ""),
                default_model=self.llm.default_model,
                base_url=DEFAULT_PROVIDER_BASE_URLS.get(provider_name, ""),
            )

        for name, provider in self.providers.items():
            if not provider.api_key_env:
                provider.api_key_env = DEFAULT_PROVIDER_ENVS.get(name, "")
            if not provider.base_url:
                provider.base_url = DEFAULT_PROVIDER_BASE_URLS.get(name, "")
            if not provider.default_model and name == self.llm.provider:
                provider.default_model = self.llm.default_model

        if not self.model_roles.default:
            provider_name = (self.llm.provider or next(iter(self.providers))).strip().lower()
            default_model = self.providers[provider_name].default_model or self.llm.default_model
            self.model_roles.default = f"{provider_name}:{default_model}"

        provider_name = (self.llm.provider or next(iter(self.providers))).strip().lower()
        if self.ui.max_content_width == 80 and self.ui.content_width_ratio > 0:
            self.ui.max_content_width = 0

        if provider_name == "gemini":
            default_binding = (self.model_roles.default or "").strip()
            fast_binding = _gemini_flash_lite_binding()
            if _should_repair_gemini_fast_role(self.model_roles.fast, default_binding):
                self.model_roles.fast = fast_binding
            if _should_repair_gemini_fast_role(self.model_roles.fast_inference, default_binding):
                self.model_roles.fast_inference = fast_binding

        if not self.model_roles.fast_inference and self.model_roles.fast:
            self.model_roles.fast_inference = self.model_roles.fast
        if not self.model_roles.fast and self.model_roles.fast_inference:
            self.model_roles.fast = self.model_roles.fast_inference

        for field_name in ("planner", "reviewer", "recall", "fast", "fast_inference", "namer"):
            if not getattr(self.model_roles, field_name):
                setattr(self.model_roles, field_name, self.model_roles.default)

        provider_name, model_name = _parse_role_binding(self.model_roles.default)
        self.llm.provider = provider_name
        self.llm.default_model = model_name or FLASH_MODEL
        return self


_config: Optional[Config] = None
_config_cache_key: tuple[str, str] | None = None
_config_cache_mtime_ns: int | None = None


def get_config_path(path: Optional[Path] = None) -> Path:
    """Get the configuration file path."""
    if path is not None:
        return path

    env_override = os.environ.get("PRODUCTIVEBRAIN_CONFIG_PATH")
    if env_override:
        return Path(os.path.expanduser(env_override))

    xdg_config = os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config"))
    return Path(xdg_config) / "productivebrain" / "config.toml"


def _config_cache_identity(path: Path, vault: Optional[str]) -> tuple[str, str]:
    return (str(path.resolve()), vault or "")


def _expand_path(raw: str) -> Path:
    return Path(os.path.expanduser(raw)).expanduser()


def get_active_vault_name(config: Optional[Config] = None, vault: Optional[str] = None) -> str:
    """Return the selected active vault name."""
    cfg = config or get_config()
    selected = (vault or cfg.general.active_vault or "main").strip()
    if selected not in cfg.vaults:
        raise KeyError(f"Unknown vault profile: {selected}")
    return selected


def get_vault_profile(config: Optional[Config] = None, vault: Optional[str] = None) -> VaultProfileConfig:
    """Return the resolved vault profile."""
    cfg = config or get_config()
    return cfg.vaults[get_active_vault_name(cfg, vault=vault)]


def get_data_dir(config: Optional[Config] = None, vault: Optional[str] = None) -> Path:
    """Get the active vault data directory, creating it if needed."""
    profile = get_vault_profile(config, vault=vault)
    data_dir = _expand_path(profile.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


def get_log_dir(config: Optional[Config] = None) -> Path:
    """Get the log/state directory."""
    cfg = config or get_config()
    log_dir = _expand_path(cfg.storage.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def get_vault_path(config: Optional[Config] = None, vault: Optional[str] = None) -> Path:
    """Get the selected Obsidian vault path."""
    profile = get_vault_profile(config, vault=vault)
    return _expand_path(profile.path)


def get_quarantine_folder(config: Optional[Config] = None, vault: Optional[str] = None) -> str:
    """Return the quarantine folder path relative to the vault root."""
    profile = get_vault_profile(config, vault=vault)
    return profile.quarantine_folder or DEFAULT_QUARANTINE_FOLDER


def get_quarantine_path(config: Optional[Config] = None, vault: Optional[str] = None) -> Path:
    """Return the absolute quarantine directory inside the selected vault."""
    return get_vault_path(config, vault=vault) / get_quarantine_folder(config, vault=vault)


def _normalize_loaded_data(data: dict[str, Any]) -> dict[str, Any]:
    """Upgrade legacy single-vault config dictionaries in memory."""
    upgraded = deepcopy(data)
    general = upgraded.setdefault("general", {})
    vaults = upgraded.setdefault("vaults", {})
    storage = upgraded.setdefault("storage", {})
    storage.setdefault("data_dir", DEFAULT_DATA_HOME)
    storage.setdefault("log_dir", DEFAULT_STATE_HOME)

    legacy_vault_path = general.get("vault_path")
    if legacy_vault_path and not vaults:
        vaults["main"] = {
            "path": legacy_vault_path,
            "data_dir": _default_profile_data_dir("main"),
            "quarantine_folder": DEFAULT_QUARANTINE_FOLDER,
        }
        general["active_vault"] = "main"

    if not general.get("active_vault"):
        general["active_vault"] = next(iter(vaults), "main")

    upgraded.setdefault("notes", {})
    upgraded.setdefault("interaction", {})
    upgraded.setdefault("commit_policy", {})
    upgraded.setdefault("providers", {})
    upgraded.setdefault("model_roles", {})
    upgraded.setdefault("preferences", {})
    return upgraded


def load_config(
    path: Optional[Path] = None,
    *,
    vault: Optional[str] = None,
    force_reload: bool = False,
) -> Config:
    """Load configuration from TOML file."""
    global _config, _config_cache_key, _config_cache_mtime_ns

    path = get_config_path(path)
    identity = _config_cache_identity(path, vault)
    mtime_ns = path.stat().st_mtime_ns if path.exists() else None

    if (
        not force_reload
        and _config is not None
        and _config_cache_key == identity
        and _config_cache_mtime_ns == mtime_ns
    ):
        return _config

    if not path.exists():
        raise FileNotFoundError(
            f"Config file not found: {path}\n"
            "Create it with at least:\n\n"
            "[general]\n"
            'active_vault = "main"\n\n'
            "[vaults.main]\n"
            'path = "/path/to/obsidian/vault"\n'
        )

    with open(path, "rb") as handle:
        data = tomli.load(handle)

    normalized = _normalize_loaded_data(data)
    cfg = Config(**normalized)

    if vault:
        cfg.general.active_vault = get_active_vault_name(cfg, vault=vault)
        cfg.general.vault_path = cfg.vaults[cfg.general.active_vault].path

    vault_path = get_vault_path(cfg)
    if not cfg.vaults[cfg.general.active_vault].path:
        raise FileNotFoundError(
            f"Vault path is not configured for active profile '{cfg.general.active_vault}' in {path}"
        )
    if vault_path and str(vault_path) not in {"", "."} and not vault_path.exists():
        raise FileNotFoundError(
            f"Vault not found at {vault_path}. "
            f"Update the vault profile in {path}"
        )

    _config = cfg
    _config_cache_key = identity
    _config_cache_mtime_ns = mtime_ns
    return cfg


def get_config(
    path: Optional[Path] = None,
    *,
    vault: Optional[str] = None,
    force_reload: bool = False,
) -> Config:
    """Get the current configuration, loading if necessary."""
    if _config is None or force_reload:
        return load_config(path, vault=vault, force_reload=force_reload)
    if path is not None or vault is not None:
        return load_config(path, vault=vault, force_reload=force_reload)
    return _config


def _toml_escape(value: str) -> str:
    return value.replace("\\", "\\\\").replace('"', '\\"')


def _toml_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return f'"{_toml_escape(value)}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_literal(item) for item in value) + "]"
    raise TypeError(f"Unsupported TOML literal: {value!r}")


def _dump_table(name: Optional[str], payload: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    scalars: list[tuple[str, Any]] = []
    nested: list[tuple[str, dict[str, Any]]] = []
    arrays: list[tuple[str, list[dict[str, Any]]]] = []

    for key, value in payload.items():
        if value is None:
            continue
        if isinstance(value, dict):
            nested.append((key, value))
        elif isinstance(value, list) and value and all(isinstance(item, dict) for item in value):
            arrays.append((key, value))
        else:
            scalars.append((key, value))

    if name is not None:
        lines.append(f"[{name}]")
    for key, value in scalars:
        lines.append(f"{key} = {_toml_literal(value)}")
    if scalars and (nested or arrays):
        lines.append("")

    for index, (key, value) in enumerate(nested):
        child_name = key if name is None else f"{name}.{key}"
        child_lines = _dump_table(child_name, value)
        lines.extend(child_lines)
        if index != len(nested) - 1 or arrays:
            lines.append("")

    for array_index, (key, items) in enumerate(arrays):
        array_name = key if name is None else f"{name}.{key}"
        for item_index, item in enumerate(items):
            lines.append(f"[[{array_name}]]")
            nested_lines = _dump_table(None, item)
            lines.extend(line for line in nested_lines if line)
            if item_index != len(items) - 1:
                lines.append("")
        if array_index != len(arrays) - 1:
            lines.append("")

    while lines and lines[-1] == "":
        lines.pop()
    return lines


def save_config(config: Config, path: Optional[Path] = None) -> Path:
    """Write the canonical config back to TOML."""
    path = get_config_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = config.model_dump(mode="python", exclude_none=True)
    lines = _dump_table(None, payload)
    path.write_text("\n".join(lines) + "\n")
    load_config(path, force_reload=True)
    return path


def create_default_config(
    vault_path: str,
    *,
    vault_name: str = "main",
    provider: str = "gemini",
    model: Optional[str] = None,
    interaction_mode: str = "guided",
) -> str:
    """Generate default config TOML content."""
    provider_name = (provider or "gemini").strip().lower()
    default_model = model or FLASH_MODEL
    role_bindings = _default_role_bindings(provider_name, default_model)
    cfg = Config(
        general=GeneralConfig(active_vault=vault_name, vault_path=vault_path),
        vaults={
            vault_name: VaultProfileConfig(
                path=vault_path,
                data_dir=_default_profile_data_dir(vault_name),
                quarantine_folder=DEFAULT_QUARANTINE_FOLDER,
            )
        },
        interaction=InteractionConfig(mode=interaction_mode),
        providers={
            provider_name: ProviderConfig(
                api_key_env=DEFAULT_PROVIDER_ENVS.get(provider_name, ""),
                default_model=default_model,
                base_url=DEFAULT_PROVIDER_BASE_URLS.get(provider_name, ""),
            )
        },
        model_roles=ModelRolesConfig(
            **role_bindings,
        ),
        llm=LLMConfig(
            provider=provider_name,
            backend="auto",
            default_model=default_model,
            prompt_template_version="v3",
            require_llm_for_core_workflows=True,
        ),
    )
    return "\n".join(_dump_table(None, cfg.model_dump(mode="python", exclude_none=True))) + "\n"


def ensure_config_dir(path: Optional[Path] = None) -> Path:
    """Ensure config directory exists and return its path."""
    config_path = get_config_path(path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    return config_path.parent


def _set_nested_value(target: dict[str, Any], path_parts: list[str], value: Any) -> None:
    cursor = target
    for part in path_parts[:-1]:
        next_value = cursor.get(part)
        if not isinstance(next_value, dict):
            next_value = {}
            cursor[part] = next_value
        cursor = next_value
    cursor[path_parts[-1]] = value


def _propagate_default_model(payload: dict[str, Any], new_model: str) -> None:
    """When llm.default_model changes, propagate to model_roles and provider."""
    provider = payload.get("llm", {}).get("provider", "gemini")
    providers = payload.get("providers", {})
    if provider in providers:
        providers[provider]["default_model"] = new_model
    roles = payload.get("model_roles", {})
    old_default = roles.get("default", "")
    desired = _default_role_bindings(str(provider), str(new_model))
    for role_name in list(roles):
        if roles[role_name] == old_default:
            roles[role_name] = desired.get(role_name, desired["default"])


def set_config_value(section: str, key: str, value: Any, *, path: Optional[Path] = None) -> None:
    """Set a single config key and write back."""
    cfg = get_config(path, force_reload=True)
    payload = cfg.model_dump(mode="python", exclude_none=True)
    path_parts = [part for part in [section, *key.split(".")] if part]
    _set_nested_value(payload, path_parts, value)

    if path_parts == ["llm", "default_model"]:
        _propagate_default_model(payload, value)

    updated = Config(**payload)

    # Verify key wasn't silently dropped by Pydantic (catches typos)
    result = updated.model_dump(mode="python")
    cursor: Any = result
    for part in path_parts:
        if isinstance(cursor, dict) and part in cursor:
            cursor = cursor[part]
        else:
            raise ValueError(f"Unknown config key: {section}.{key}")

    save_config(updated, path=path)


def upsert_vault_profile(
    name: str,
    vault_path: str,
    *,
    data_dir: Optional[str] = None,
    quarantine_folder: str = DEFAULT_QUARANTINE_FOLDER,
    path: Optional[Path] = None,
) -> Config:
    """Create or update a vault profile and persist it."""
    cfg = get_config(path, force_reload=True)
    cfg.vaults[name] = VaultProfileConfig(
        path=vault_path,
        data_dir=data_dir or _default_profile_data_dir(name),
        quarantine_folder=quarantine_folder,
    )
    if not cfg.general.active_vault:
        cfg.general.active_vault = name
    cfg.general.vault_path = cfg.vaults[cfg.general.active_vault].path
    save_config(cfg, path=path)
    return cfg


def set_active_vault(name: str, *, path: Optional[Path] = None) -> Config:
    """Persist the active vault selection."""
    cfg = get_config(path, force_reload=True)
    cfg.general.active_vault = get_active_vault_name(cfg, vault=name)
    cfg.general.vault_path = cfg.vaults[cfg.general.active_vault].path
    save_config(cfg, path=path)
    return cfg


def remove_vault_profile(name: str, *, path: Optional[Path] = None) -> Config:
    """Remove a vault profile from config."""
    cfg = get_config(path, force_reload=True)
    if name not in cfg.vaults:
        raise KeyError(f"Unknown vault profile: {name}")
    if cfg.general.active_vault == name:
        raise ValueError("Cannot remove the active vault profile.")
    del cfg.vaults[name]
    save_config(cfg, path=path)
    return cfg


def rename_vault_profile(old: str, new: str, *, path: Optional[Path] = None) -> Config:
    """Rename a vault profile and preserve its data."""
    cfg = get_config(path, force_reload=True)
    if old not in cfg.vaults:
        raise KeyError(f"Unknown vault profile: {old}")
    if new in cfg.vaults:
        raise ValueError(f"Vault profile already exists: {new}")
    profile = cfg.vaults.pop(old)
    if profile.data_dir == _default_profile_data_dir(old):
        profile.data_dir = _default_profile_data_dir(new)
    cfg.vaults[new] = profile
    if cfg.general.active_vault == old:
        cfg.general.active_vault = new
    cfg.general.vault_path = cfg.vaults[cfg.general.active_vault].path
    save_config(cfg, path=path)
    return cfg


def upsert_provider(
    name: str,
    *,
    api_key_env: Optional[str] = None,
    default_model: str,
    base_url: Optional[str] = None,
    path: Optional[Path] = None,
) -> Config:
    """Create or update a provider entry."""
    cfg = get_config(path, force_reload=True)
    provider_name = name.strip().lower()
    cfg.providers[provider_name] = ProviderConfig(
        api_key_env=api_key_env or DEFAULT_PROVIDER_ENVS.get(provider_name, ""),
        default_model=default_model,
        base_url=base_url or DEFAULT_PROVIDER_BASE_URLS.get(provider_name, ""),
    )
    if not cfg.model_roles.default:
        cfg.model_roles.default = f"{provider_name}:{default_model}"
    save_config(cfg, path=path)
    return cfg


def set_default_model_binding(binding: str, *, path: Optional[Path] = None) -> Config:
    """Persist the preferred default binding while preserving Gemini fast tiers."""
    cfg = get_config(path, force_reload=True)
    provider_name, model_name = _parse_role_binding(binding)
    if provider_name not in cfg.providers:
        cfg.providers[provider_name] = ProviderConfig(
            api_key_env=DEFAULT_PROVIDER_ENVS.get(provider_name, ""),
            default_model=model_name,
            base_url=DEFAULT_PROVIDER_BASE_URLS.get(provider_name, ""),
        )
    else:
        cfg.providers[provider_name].default_model = model_name

    cfg.llm.provider = provider_name
    cfg.llm.default_model = model_name
    desired = _default_role_bindings(provider_name, model_name)
    old_default = cfg.model_roles.default
    for role_name in ("default", "planner", "reviewer", "recall", "namer"):
        current = getattr(cfg.model_roles, role_name)
        if role_name == "default" or not current or current == old_default:
            setattr(cfg.model_roles, role_name, desired[role_name])

    for role_name in ("fast", "fast_inference"):
        current = getattr(cfg.model_roles, role_name)
        if not current or current == old_default or _should_repair_gemini_fast_role(current, old_default):
            setattr(cfg.model_roles, role_name, desired[role_name])

    save_config(cfg, path=path)
    return cfg


def set_model_role(role: str, binding: str, *, path: Optional[Path] = None) -> Config:
    """Persist a model role binding."""
    cfg = get_config(path, force_reload=True)
    if not hasattr(cfg.model_roles, role):
        raise AttributeError(f"Unknown model role: {role}")
    setattr(cfg.model_roles, role, binding)
    if role == "default":
        for field_name in ("planner", "reviewer", "recall", "fast", "fast_inference", "namer"):
            if not getattr(cfg.model_roles, field_name):
                setattr(cfg.model_roles, field_name, binding)
    if role == "fast" and not cfg.model_roles.fast_inference:
        cfg.model_roles.fast_inference = binding
    if role == "fast_inference" and not cfg.model_roles.fast:
        cfg.model_roles.fast = binding
    save_config(cfg, path=path)
    return cfg


def set_ui_language(lang: str, *, path: Optional[Path] = None) -> Config:
    """Persist the UI response language preference."""
    cfg = get_config(path, force_reload=True)
    cfg.ui.language = lang.strip().lower() if lang.strip() else "auto"
    save_config(cfg, path=path)
    return cfg


def set_model_max_tokens(tier: str, tokens: int, *, path: Optional[Path] = None) -> Config:
    """Persist a preferred max output token count for a model tier (fast/balanced/pro)."""
    cfg = get_config(path, force_reload=True)
    cfg.ui.model_max_tokens[tier.strip().lower()] = max(1, tokens)
    save_config(cfg, path=path)
    return cfg


def update_preferences(patches: dict[str, Any], *, path: Optional[Path] = None) -> Config:
    """Merge user-approved preference patches into config."""
    cfg = get_config(path, force_reload=True)
    cfg.preferences.update(patches)
    save_config(cfg, path=path)
    return cfg


def reset_config_cache() -> None:
    """Clear the in-process config cache."""
    global _config, _config_cache_key, _config_cache_mtime_ns
    _config = None
    _config_cache_key = None
    _config_cache_mtime_ns = None
