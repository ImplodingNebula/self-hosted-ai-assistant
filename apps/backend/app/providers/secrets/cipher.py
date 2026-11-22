"""AES-256-GCM authenticated encryption for credentials at rest.

The store binds each row's identity - ``tenant:owner:key_version:name``, built
in ``SecretStore._aad`` - into the ciphertext as Associated Data (AAD). GCM
verifies that AAD on decrypt, so a blob decrypts ONLY under the exact row it was
sealed for. If a query bug ever hands one request another row's bytes,
decryption raises loudly rather than returning a credential to the wrong caller:
the wrong secret is not merely unlikely, it is cryptographically unreachable.
"""

from __future__ import annotations

import base64
import binascii
import os

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from app.providers.secrets.base import SecretConfigError

# GCM's standard 96-bit nonce, prepended to each ciphertext. Random per message
# (never a counter): a fresh 12-byte nonce per encryption keeps nonce reuse -
# the one catastrophic GCM misuse - astronomically unlikely without any shared
# state to coordinate.
_NONCE_BYTES = 12
_VALID_KEY_BYTES = (16, 24, 32)


class SecretCipher:
    """Seals and opens one secret at a time under a single master key."""

    def __init__(self, key: bytes) -> None:
        """Build a cipher from raw key bytes (16, 24 or 32 for AES-128/192/256)."""
        if len(key) not in _VALID_KEY_BYTES:
            raise SecretConfigError(f"Master key must be {_VALID_KEY_BYTES} bytes; got {len(key)}.")
        self._aead = AESGCM(key)

    def encrypt(self, plaintext: str, *, aad: bytes) -> bytes:
        """Return ``nonce || ciphertext+tag`` for ``plaintext`` bound to ``aad``."""
        nonce = os.urandom(_NONCE_BYTES)
        sealed = self._aead.encrypt(nonce, plaintext.encode("utf-8"), aad)
        return nonce + sealed

    def decrypt(self, blob: bytes, *, aad: bytes) -> str:
        """Recover the plaintext, or raise if ``aad`` or the ciphertext is wrong.

        Raises:
            SecretConfigError: The blob is too short to hold a nonce - a stored
                value this cipher never produced (corruption, or a schema/type
                mix-up), worth failing on distinctly from a tampered tag.
            cryptography.exceptions.InvalidTag: The tag or the AAD does not
                verify - a tampered ciphertext, the wrong master key, or a blob
                whose row identity does not match the one presented in ``aad``.
        """
        if len(blob) <= _NONCE_BYTES:
            raise SecretConfigError("Ciphertext is too short to contain a nonce.")
        nonce, sealed = blob[:_NONCE_BYTES], blob[_NONCE_BYTES:]
        opened = self._aead.decrypt(nonce, sealed, aad)  # raises InvalidTag on mismatch
        return opened.decode("utf-8")


# The deployment requires AES-256, so the master key must decode to exactly 32
# bytes. Pinning the length is also what catches the likeliest misconfiguration:
# a raw pass-phrase pasted where a base64 KEY belongs. Many such strings are
# themselves valid base64 (a 32-char pass-phrase decodes to 24 bytes -> a silent,
# unintended AES-192 key), so "is it valid base64" cannot catch it - but "does it
# decode to 32 bytes" does, and turns a silent wrong-key into a loud startup error
# instead of InvalidTag on every reveal later.
_REQUIRED_KEY_BYTES = 32


def build_cipher(master_key: str | None) -> SecretCipher | None:
    """Build the AES-256 cipher from the base64 master key, or None when unset.

    Unset (None or blank) returns None - the caller treats that as "the secret
    store is unavailable" and answers 503; it never invents a key. A key that is
    SET but malformed raises: that is an operator mistake to surface, not to
    paper over.

    Raises:
        SecretConfigError: The key is set but not valid base64, or does not
            decode to exactly 32 bytes (AES-256).
    """
    if not master_key or not master_key.strip():
        return None
    try:
        key = base64.b64decode(master_key.strip(), validate=True)
    except (binascii.Error, ValueError) as exc:
        raise SecretConfigError("SECRET_MASTER_KEY is not valid base64.") from exc
    if len(key) != _REQUIRED_KEY_BYTES:
        raise SecretConfigError(
            f"SECRET_MASTER_KEY must decode to {_REQUIRED_KEY_BYTES} bytes (AES-256); "
            f"got {len(key)}. A pass-phrase is not a key - generate one with "
            "base64.b64encode(os.urandom(32))."
        )
    return SecretCipher(key)


# Re-exported so callers can catch the decrypt failure without importing from the
# cryptography package directly.
__all__ = ["InvalidTag", "SecretCipher", "build_cipher"]
