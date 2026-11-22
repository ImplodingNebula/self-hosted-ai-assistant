"""Credential secrecy: the value types, the cipher, and their errors.

The DB-backed store that composes these lives at
``app.services.secret_store`` (it needs a session and a tenant context); this
package holds only the pieces with no database dependency, so the cipher can be
unit-tested and reviewed on its own.
"""

from app.providers.secrets.base import (
    SecretConfigError,
    SecretError,
    SecretMetadata,
    SecretNotFoundError,
    SecretScope,
    SecretStoreUnavailableError,
)
from app.providers.secrets.cipher import SecretCipher, build_cipher

__all__ = [
    "SecretCipher",
    "SecretConfigError",
    "SecretError",
    "SecretMetadata",
    "SecretNotFoundError",
    "SecretScope",
    "SecretStoreUnavailableError",
    "build_cipher",
]
