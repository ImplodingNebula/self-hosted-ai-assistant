"""Value types and errors for the SecretStore seam - no database, no crypto here."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict


class SecretScope(StrEnum):
    """Who a secret belongs to.

    ``WORKSPACE`` secrets are shared by every member of a tenant; ``USER``
    secrets are private to the acting user. The scope is chosen by the caller;
    the *identity* it resolves to (which tenant, which user) is never a caller
    argument - it comes from the bound request context - so one member cannot
    address another's secret.
    """

    WORKSPACE = "workspace"
    USER = "user"


class SecretMetadata(BaseModel):
    """The only shape a read of a secret ever returns: no ciphertext, no plaintext.

    ``last4`` is the sole readable hint - enough to recognise which token is
    stored ("...ab12") without revealing it.
    """

    model_config = ConfigDict(frozen=True)

    name: str
    scope: SecretScope
    last4: str
    key_version: int
    created_at: datetime
    updated_at: datetime


class SecretError(Exception):
    """Base class for every SecretStore failure."""


class SecretConfigError(SecretError):
    """The configured master key is present but unusable (wrong length/encoding).

    A misconfiguration to fix loudly, distinct from the key simply being unset.
    """


class SecretStoreUnavailableError(SecretError):
    """No master key is configured, so credentials cannot be sealed or revealed.

    The deliberate, first-class "off" state: the integration surface answers
    503 rather than falling back to plaintext or minting a throwaway key.
    """


class SecretNotFoundError(SecretError):
    """No secret of that name exists in the requested scope for this principal."""
