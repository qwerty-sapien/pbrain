"""Unit tests for MCP encryption module.

Note: These tests use mocks for macOS Keychain to avoid polluting
the real Keychain during testing.
"""

import pytest
from unittest.mock import patch, MagicMock

from pb.mcp.encryption import (
    encrypt_content,
    decrypt_content,
    get_or_create_key,
    has_encryption_key,
    delete_encryption_key,
    EncryptionError,
    KeychainError,
    SERVICE_NAME,
    KEY_NAME,
)


class TestKeyManagement:
    """Tests for key generation and storage."""

    def test_service_name_is_descriptive(self) -> None:
        """SERVICE_NAME identifies the application."""
        assert SERVICE_NAME == "pb-brain-encryption"

    def test_key_name_is_descriptive(self) -> None:
        """KEY_NAME identifies the key purpose."""
        assert KEY_NAME == "fernet-key"

    @patch("pb.mcp.encryption.keyring")
    def test_get_or_create_key_returns_existing(self, mock_keyring: MagicMock) -> None:
        """get_or_create_key returns existing key from Keychain."""
        # Simulate existing key
        existing_key = b"existing-key-32-bytes-for-fernet"
        mock_keyring.get_password.return_value = existing_key.decode()

        result = get_or_create_key()

        assert result == existing_key
        mock_keyring.get_password.assert_called_once_with(SERVICE_NAME, KEY_NAME)
        mock_keyring.set_password.assert_not_called()

    @patch("pb.mcp.encryption.keyring")
    @patch("pb.mcp.encryption.Fernet")
    def test_get_or_create_key_generates_new(
        self, mock_fernet: MagicMock, mock_keyring: MagicMock
    ) -> None:
        """get_or_create_key generates new key when none exists (D-06)."""
        mock_keyring.get_password.return_value = None
        new_key = b"new-generated-key-32-bytes-long!"
        mock_fernet.generate_key.return_value = new_key

        result = get_or_create_key()

        assert result == new_key
        mock_fernet.generate_key.assert_called_once()
        mock_keyring.set_password.assert_called_once_with(
            SERVICE_NAME, KEY_NAME, new_key.decode()
        )

    @patch("pb.mcp.encryption.keyring")
    def test_get_or_create_key_raises_on_locked_keychain(
        self, mock_keyring: MagicMock
    ) -> None:
        """get_or_create_key raises KeychainError when Keychain is locked."""
        from keyring.errors import KeyringLocked

        mock_keyring.get_password.side_effect = KeyringLocked()

        with pytest.raises(KeychainError, match="Keychain is locked"):
            get_or_create_key()


class TestEncryptDecrypt:
    """Tests for encryption and decryption functions."""

    @patch("pb.mcp.encryption.get_or_create_key")
    def test_encrypt_decrypt_roundtrip(self, mock_get_key: MagicMock) -> None:
        """encrypt_content and decrypt_content are inverse operations."""
        from cryptography.fernet import Fernet

        # Use a real Fernet key for roundtrip test
        real_key = Fernet.generate_key()
        mock_get_key.return_value = real_key

        original = "Secret people note content"
        encrypted = encrypt_content(original)
        decrypted = decrypt_content(encrypted)

        assert decrypted == original
        assert encrypted != original.encode()  # Actually encrypted

    @patch("pb.mcp.encryption.get_or_create_key")
    def test_encrypt_returns_bytes(self, mock_get_key: MagicMock) -> None:
        """encrypt_content returns bytes, not string."""
        from cryptography.fernet import Fernet

        mock_get_key.return_value = Fernet.generate_key()

        result = encrypt_content("test content")

        assert isinstance(result, bytes)

    @patch("pb.mcp.encryption.get_or_create_key")
    def test_decrypt_returns_string(self, mock_get_key: MagicMock) -> None:
        """decrypt_content returns string, not bytes."""
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        mock_get_key.return_value = key
        f = Fernet(key)
        ciphertext = f.encrypt(b"test content")

        result = decrypt_content(ciphertext)

        assert isinstance(result, str)
        assert result == "test content"

    @patch("pb.mcp.encryption.get_or_create_key")
    def test_decrypt_with_wrong_key_raises(self, mock_get_key: MagicMock) -> None:
        """decrypt_content raises EncryptionError with wrong key."""
        from cryptography.fernet import Fernet

        # Encrypt with one key
        key1 = Fernet.generate_key()
        f1 = Fernet(key1)
        ciphertext = f1.encrypt(b"secret")

        # Try to decrypt with different key
        key2 = Fernet.generate_key()
        mock_get_key.return_value = key2

        with pytest.raises(EncryptionError, match="invalid token"):
            decrypt_content(ciphertext)

    @patch("pb.mcp.encryption.get_or_create_key")
    def test_decrypt_corrupted_data_raises(self, mock_get_key: MagicMock) -> None:
        """decrypt_content raises EncryptionError with corrupted data."""
        from cryptography.fernet import Fernet

        mock_get_key.return_value = Fernet.generate_key()

        with pytest.raises(EncryptionError, match="invalid token"):
            decrypt_content(b"not-valid-ciphertext")


class TestHasEncryptionKey:
    """Tests for has_encryption_key function."""

    @patch("pb.mcp.encryption.keyring")
    def test_returns_true_when_key_exists(self, mock_keyring: MagicMock) -> None:
        """has_encryption_key returns True when key is in Keychain."""
        mock_keyring.get_password.return_value = "some-key"

        assert has_encryption_key() is True

    @patch("pb.mcp.encryption.keyring")
    def test_returns_false_when_no_key(self, mock_keyring: MagicMock) -> None:
        """has_encryption_key returns False when no key in Keychain."""
        mock_keyring.get_password.return_value = None

        assert has_encryption_key() is False

    @patch("pb.mcp.encryption.keyring")
    def test_returns_false_on_keyring_error(self, mock_keyring: MagicMock) -> None:
        """has_encryption_key returns False on Keychain errors."""
        from keyring.errors import KeyringLocked

        mock_keyring.get_password.side_effect = KeyringLocked()

        assert has_encryption_key() is False


class TestDeleteEncryptionKey:
    """Tests for delete_encryption_key function."""

    @patch("pb.mcp.encryption.has_encryption_key")
    @patch("pb.mcp.encryption.keyring")
    def test_deletes_existing_key(
        self, mock_keyring: MagicMock, mock_has_key: MagicMock
    ) -> None:
        """delete_encryption_key removes key from Keychain."""
        mock_has_key.return_value = True

        result = delete_encryption_key()

        assert result is True
        mock_keyring.delete_password.assert_called_once_with(SERVICE_NAME, KEY_NAME)

    @patch("pb.mcp.encryption.has_encryption_key")
    def test_returns_false_when_no_key(self, mock_has_key: MagicMock) -> None:
        """delete_encryption_key returns False when no key exists."""
        mock_has_key.return_value = False

        result = delete_encryption_key()

        assert result is False
