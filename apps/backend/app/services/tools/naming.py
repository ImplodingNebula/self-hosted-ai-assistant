"""Secret-name derivation shared by the credential and OAuth layers.

Lives below both ``services.tools.mcp_sources`` and ``services.oauth`` so
either can import it without a cycle (the OAuth bundle derives its names from
the same stable identity this defines).
"""

from __future__ import annotations

import hashlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from uuid import UUID


def credential_secret_name(server_id: UUID, base_url: str) -> str:
    """The SecretStore name a server's per-user (or workspace) credential is keyed under.

    Bound to the server's STABLE id AND its base_url, NOT its mutable display
    name. A credential a user connected to one server at one URL can then never be
    revealed for a DIFFERENT server (a delete + recreate under the same name gets
    a fresh id) or a RE-POINTED one (a base_url change changes the key) - either of
    which, keyed on the name, would forward a member's private token to an endpoint
    they never consented to. The url is digested so the secret name stays bounded
    and does not itself carry the URL into the secrets table. Storing and revealing
    both derive the name from the same row, so they agree until the server is
    re-pointed, at which point the old credential is orphaned and the user
    reconnects - re-consent for the new endpoint, which is the safe behaviour for a
    forwarded credential.
    """
    digest = hashlib.sha256(base_url.encode()).hexdigest()[:16]
    return f"mcp-server:{server_id}:{digest}"
