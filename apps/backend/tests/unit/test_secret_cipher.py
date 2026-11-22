"""AES-GCM cipher: round-trip, AAD binding, tamper detection, key handling."""

from __future__ import annotations

import base64
import os

import pytest
from cryptography.exceptions import InvalidTag

from app.providers.secrets import SecretConfigError, build_cipher
from app.providers.secrets.cipher import SecretCipher

KEY = base64.b64encode(b"k" * 32).decode()


def _cipher() -> SecretCipher:
    cipher = build_cipher(KEY)
    assert cipher is not None
    return cipher


def test_round_trips_a_secret_under_matching_aad() -> None:
    cipher = _cipher()
    blob = cipher.encrypt("hunter2", aad=b"tenant:user:1:github")
    assert cipher.decrypt(blob, aad=b"tenant:user:1:github") == "hunter2"


def test_a_fresh_nonce_makes_two_seals_of_one_value_differ() -> None:
    cipher = _cipher()
    a = cipher.encrypt("same", aad=b"x")
    b = cipher.encrypt("same", aad=b"x")
    assert a != b  # random nonce per message, so no ciphertext-equality oracle
    assert cipher.decrypt(a, aad=b"x") == cipher.decrypt(b, aad=b"x") == "same"


def test_decrypt_under_a_different_aad_is_refused() -> None:
    # The leak tripwire: a blob sealed for one row's identity will not open under
    # another's, so a misrouted ciphertext raises instead of leaking.
    cipher = _cipher()
    blob = cipher.encrypt("secret", aad=b"tenant:alice:1:github")
    with pytest.raises(InvalidTag):
        cipher.decrypt(blob, aad=b"tenant:bob:1:github")


def test_a_tampered_ciphertext_is_refused() -> None:
    cipher = _cipher()
    blob = bytearray(cipher.encrypt("secret", aad=b"x"))
    blob[-1] ^= 0x01  # flip one tag bit
    with pytest.raises(InvalidTag):
        cipher.decrypt(bytes(blob), aad=b"x")


def test_the_wrong_master_key_cannot_decrypt() -> None:
    sealed = _cipher().encrypt("secret", aad=b"x")
    other = build_cipher(base64.b64encode(b"j" * 32).decode())
    assert other is not None
    with pytest.raises(InvalidTag):
        other.decrypt(sealed, aad=b"x")


def test_a_truncated_blob_is_a_config_error_not_a_tag_failure() -> None:
    cipher = _cipher()
    with pytest.raises(SecretConfigError, match="too short"):
        cipher.decrypt(b"short", aad=b"x")


@pytest.mark.parametrize("value", [None, "", "   "])
def test_build_cipher_returns_none_when_the_key_is_unset(value) -> None:
    # Unset is a first-class state - the store is "unavailable", not defaulted.
    assert build_cipher(value) is None


def test_build_cipher_rejects_non_base64() -> None:
    with pytest.raises(SecretConfigError, match="base64"):
        build_cipher("not valid base64 !!!")


@pytest.mark.parametrize("n", [8, 16, 24, 48])
def test_build_cipher_requires_exactly_32_bytes(n) -> None:
    # AES-256 only: pinning the length is what catches a raw pass-phrase pasted
    # where a base64 key belongs (a 32-char pass-phrase is valid base64 that
    # decodes to 24 bytes - a silently-wrong AES-192 key without this check).
    with pytest.raises(SecretConfigError, match="32 bytes"):
        build_cipher(base64.b64encode(os.urandom(n)).decode())


def test_build_cipher_accepts_a_32_byte_key() -> None:
    cipher = build_cipher(base64.b64encode(os.urandom(32)).decode())
    assert cipher is not None
    assert cipher.decrypt(cipher.encrypt("v", aad=b"a"), aad=b"a") == "v"


@pytest.mark.parametrize("n", [16, 24, 32])
def test_the_cipher_primitive_itself_accepts_any_aes_key_length(n) -> None:
    # The cipher is a general AES-GCM primitive; the 32-byte policy lives in
    # build_cipher, not here.
    cipher = SecretCipher(os.urandom(n))
    assert cipher.decrypt(cipher.encrypt("v", aad=b"a"), aad=b"a") == "v"
