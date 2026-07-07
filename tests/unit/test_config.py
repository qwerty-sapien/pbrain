"""Tests for configuration loading."""

import tempfile
from pathlib import Path

import pytest

from pb.storage.config import (
    Config,
    GeneralConfig,
    StorageConfig,
    create_default_config,
    load_config,
)


class TestConfigLoading:
    """Test configuration file loading."""

    def test_load_valid_config(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'''
[general]
vault_path = "{vault}"
verbose = true

[storage]
data_dir = "/test/data"
''')
        config = load_config(config_path)

        assert config.general.vault_path == str(vault)
        assert config.general.verbose is True
        assert config.storage.data_dir == "/test/data"

    def test_load_minimal_config(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'''
[general]
vault_path = "{vault}"
''')
        config = load_config(config_path)

        assert config.general.vault_path == str(vault)
        assert config.general.verbose is False
        assert config.storage.data_dir == "~/.local/share/productivebrain"

    def test_missing_config_file(self, tmp_path):
        config_path = tmp_path / "nonexistent.toml"

        with pytest.raises(FileNotFoundError) as exc:
            load_config(config_path)

        assert "Config file not found" in str(exc.value)

    def test_vault_missing(self, tmp_path):
        """ONBD-02: load_config raises when vault_path doesn't exist on disk."""
        config_path = tmp_path / "config.toml"
        config_path.write_text('''
[general]
vault_path = "/nonexistent/vault/path/that/does/not/exist"
''')
        with pytest.raises(FileNotFoundError) as exc:
            load_config(config_path)
        assert "Vault not found at" in str(exc.value)

    def test_load_config_with_valid_vault(self, tmp_path):
        """ONBD-02 positive path: load_config succeeds when vault exists."""
        vault = tmp_path / "my_vault"
        vault.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'''
[general]
vault_path = "{vault}"
''')
        config = load_config(config_path)
        assert config.general.vault_path == str(vault)

    def test_missing_vault_path(self, tmp_path):
        config_path = tmp_path / "config.toml"
        config_path.write_text('''
[general]
verbose = true
''')
        with pytest.raises(FileNotFoundError, match="Vault path is not configured"):
            load_config(config_path)


class TestDefaultConfig:
    """Test default config generation."""

    def test_create_default_config(self):
        content = create_default_config("/my/vault")

        assert 'vault_path = "/my/vault"' in content
        assert "verbose = false" in content
        assert "data_dir" in content

    def test_legacy_single_vault_config_migrates_to_profile(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'''
[general]
vault_path = "{vault}"
''')

        config = load_config(config_path)

        assert config.general.active_vault == "main"
        assert "main" in config.vaults
        assert config.vaults["main"].path == str(vault)

    def test_per_vault_default_data_dir_uses_productivebrain_home(self, tmp_path):
        vault = tmp_path / "vault"
        vault.mkdir()
        config_path = tmp_path / "config.toml"
        config_path.write_text(f'''
[general]
active_vault = "main"

[vaults.main]
path = "{vault}"
''')

        config = load_config(config_path)

        assert config.vaults["main"].data_dir.endswith("/productivebrain/vaults/main")


class TestConfigModel:
    """Test Config model validation."""

    def test_config_model_creation(self):
        config = Config(
            general=GeneralConfig(vault_path="/test"),
            storage=StorageConfig(),
        )

        assert config.general.vault_path == "/test"
        assert config.adapters.taskwarrior_enabled is False


class TestGmailConfig:
    """Test GmailConfig model defaults and loading."""

    def test_gmail_config_defaults(self):
        """GmailConfig has correct defaults (D-01, D-03)."""
        from pb.storage.config import GmailConfig

        g = GmailConfig()
        assert g.client_secrets_path == ""
        assert g.senders == []

    def test_gmail_config_with_senders(self):
        """GmailConfig loads senders list."""
        from pb.storage.config import GmailConfig

        g = GmailConfig(senders=["alice@example.com", "bob@example.com"])
        assert len(g.senders) == 2
        assert "alice@example.com" in g.senders


class TestIngestConfig:
    """Test IngestConfig and IngestRelevanceConfig models."""

    def test_ingest_config_defaults(self):
        """IngestConfig has correct defaults (D-18)."""
        from pb.storage.config import IngestConfig

        i = IngestConfig()
        assert i.relevance.threshold == 0.3
        assert i.relevance.batch_size == 100

    def test_ingest_config_custom_threshold(self):
        """IngestConfig loads custom threshold."""
        from pb.storage.config import IngestConfig, IngestRelevanceConfig

        i = IngestConfig(relevance=IngestRelevanceConfig(threshold=0.5, batch_size=50))
        assert i.relevance.threshold == 0.5
        assert i.relevance.batch_size == 50

    def test_config_includes_gmail_and_ingest(self):
        """Root Config model includes gmail and ingest fields."""
        from pb.storage.config import Config, GeneralConfig

        c = Config(general=GeneralConfig(vault_path="/tmp/test-vault"))
        assert hasattr(c, "gmail")
        assert hasattr(c, "ingest")
        assert c.gmail.senders == []
        assert c.ingest.relevance.threshold == 0.3


class TestVaultSchemaInbox:
    """Test vault schema includes inbox captures."""

    def test_vault_schema_includes_inbox_captures(self):
        """VAULT_SCHEMA includes 00-inbox/captures."""
        from pb.vault.config import VAULT_SCHEMA

        assert "00-inbox/captures" in VAULT_SCHEMA
