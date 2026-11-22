"""Tenant identity for the request lifecycle.

Every data access is scoped to a :class:`TenantContext`. Where no OIDC issuer is
configured, ``get_current_tenant()`` (in ``app.api.deps``) returns the seeded
default tenant/user below and the platform behaves as it did before login
existed; configure an issuer and the same dependency populates the context from
a verified session instead - one dependency, no schema change, and every
consumer untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID

# Well-known ids for the seeded default tenant and user. Fixed (not random) so
# every environment - dev DB, test sqlite, a fresh deploy - agrees on them and
# rows created before auth lands stay owned by the same principal after it.
DEFAULT_TENANT_ID = UUID("00000000-0000-0000-0000-000000000001")
DEFAULT_USER_ID = UUID("00000000-0000-0000-0000-000000000002")


@dataclass(frozen=True, slots=True)
class TenantContext:
    """The authenticated principal a request acts as.

    Attributes:
        tenant_id: Tenant every query is filtered by. Never client-supplied.
        user_id: Acting user within the tenant.
        role: Coarse role ("admin" or "member"); enforced once OIDC lands.
    """

    tenant_id: UUID
    user_id: UUID
    role: str = "member"
