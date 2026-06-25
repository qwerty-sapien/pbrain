# SPDX-License-Identifier: AGPL-3.0-or-later
# This file is part of ProductiveBrain.
# Canonical source: https://github.com/qwerty-sapien/pbrain
# Compliance fingerprint: PB-2026-A17F

"""Fernet encryption with macOS Keychain key storage.

Encrypts sensitive content (people notes) before LLM API calls (D-04).
Key is auto-generated on first use (D-06) and stored in Keychain (D-05).

Uses Fernet symmetric encryption with AES-256-GCM (D-03).
"""

import keyring
from keyring.errors import KeyringError, KeyringLocked
from cryptography.fernet import Fernet, InvalidToken

# Keychain identifiers
SERVICE_NAME = "pb-brain-encryption"
KEY_NAME = "fernet-key"


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


class KeychainError(Exception):
    """Raised when Keychain operations fail."""

    pass


def get_or_create_key() -> bytes:
    """Retrieve encryption key from Keychain, creating if needed.

    Implements D-05 (Keychain storage) and D-06 (auto-generate on first use).

    Returns:
        Fernet key as bytes.

    Raises:
        KeychainError: If Keychain is locked or inaccessible.
    """
    try:
        stored_key = keyring.get_password(SERVICE_NAME, KEY_NAME)
        if stored_key:
            return stored_key.encode()

        # Auto-generate on first use (D-06)
        new_key = Fernet.generate_key()
        keyring.set_password(SERVICE_NAME, KEY_NAME, new_key.decode())
        return new_key

    except KeyringLocked:
        raise KeychainError(
            "macOS Keychain is locked. Unlock it in Keychain Access and try again."
        )
    except KeyringError as e:
        raise KeychainError(f"Keychain error: {e}")


def encrypt_content(plaintext: str) -> bytes:
    """Encrypt content for LLM API transmission.

    Used before sending people notes to external APIs (D-04).

    Args:
        plaintext: Content to encrypt.

    Returns:
        Encrypted content as bytes.

    Raises:
        KeychainError: If key retrieval fails.
        EncryptionError: If encryption fails.
    """
    try:
        key = get_or_create_key()
        f = Fernet(key)
        return f.encrypt(plaintext.encode())
    except (KeychainError, KeyringError):
        raise
    except Exception as e:
        raise EncryptionError(f"Encryption failed: {e}")


def decrypt_content(ciphertext: bytes) -> str:
    """Decrypt content received from LLM API.

    Local-only decryption, mechanical (Success Criteria #5).

    Args:
        ciphertext: Encrypted content as bytes.

    Returns:
        Decrypted plaintext string.

    Raises:
        KeychainError: If key retrieval fails.
        EncryptionError: If decryption fails (wrong key or corrupted data).
    """
    try:
        key = get_or_create_key()
        f = Fernet(key)
        return f.decrypt(ciphertext).decode()
    except InvalidToken:
        raise EncryptionError(
            "Decryption failed: invalid token. "
            "Key may have changed or data is corrupted."
        )
    except (KeychainError, KeyringError):
        raise
    except Exception as e:
        raise EncryptionError(f"Decryption failed: {e}")


def has_encryption_key() -> bool:
    """Check if an encryption key exists in Keychain.

    Returns:
        True if key exists, False otherwise.
    """
    try:
        stored_key = keyring.get_password(SERVICE_NAME, KEY_NAME)
        return stored_key is not None
    except (KeyringLocked, KeyringError):
        return False


def delete_encryption_key() -> bool:
    """Delete the encryption key from Keychain.

    WARNING: This will make all encrypted content unrecoverable.

    Returns:
        True if key was deleted, False if it didn't exist.

    Raises:
        KeychainError: If deletion fails.
    """
    try:
        if not has_encryption_key():
            return False
        keyring.delete_password(SERVICE_NAME, KEY_NAME)
        return True
    except KeyringLocked:
        raise KeychainError(
            "macOS Keychain is locked. Unlock it in Keychain Access and try again."
        )
    except KeyringError as e:
        raise KeychainError(f"Failed to delete key: {e}")
