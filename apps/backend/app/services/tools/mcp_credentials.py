"""Per-user credential resolution for MCP tool calls (Phase 5 slice 6b).

The acting user's stored credential for a server is fetched from the
:class:`~app.services.secret_store.SecretStore` and returned as the
Authorization header to inject into THAT call's client - the per-call injection
the one-client-per-call rule (see :class:`~app.providers.mcp.source.McpTool`)
exists for. A resolver holds only process-wide collaborators; the identity it
acts as comes solely from the :class:`~app.providers.tools.base.ToolContext`
the executor filled from verified request state, so it - like the SecretStore
it composes - never accepts a ``user_id`` argument.

Lives in the service layer (not beside the provider) deliberately: it composes
the SecretStore, a service, so a provider importing it would invert the
layering. :class:`~app.providers.mcp.source.McpToolSource` takes it as a plain
async callable and never imports this module.
"""

from __future__ import annotations

import logging
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.tenancy import TenantContext
from app.providers.secrets import SecretCipher, SecretError, SecretNotFoundError, SecretScope
from app.providers.tools.base import ToolContext
from app.services.secret_store import SecretStore

logger = logging.getLogger(__name__)


class MissingUserCredentialError(Exception):
    """A user-scoped server had no usable per-user credential for this caller.

    Raised only by a resolver bound ``required=True`` (a ``credential_scope=user``
    server from the DB registry): the acting user has connected no credential, the
    master key is unset, or the reveal otherwise failed, and a user-scoped server
    must FAIL CLOSED rather than fall back to the shared workspace header. Carries
    only the secret/server name - never a URL or the secret - so it is safe to
    log; :class:`~app.providers.mcp.source.McpTool` wraps it into a URL-free
    ``ToolError`` naming the server.
    """

    def __init__(self, secret_name: str) -> None:
        """Record the server/secret name the credential was missing for."""
        super().__init__(f"no per-user credential for {secret_name!r}")
        self.secret_name = secret_name


class PerUserCredentialResolver:
    """Resolves one server's per-user bearer header, once per tool call.

    Bound to the process-wide sessionmaker, the process cipher (``None`` when
    ``SECRET_MASTER_KEY`` is unset), and one server's secret name. Called with a
    turn's :class:`ToolContext`, it opens a short-lived session, reveals the
    acting user's USER-scoped secret for that name, and returns
    ``{"Authorization": "Bearer <secret>"}``.

    Two modes, chosen by ``required``:

    - ``required=False`` (the default, and the env-seeded PR1 path): it NEVER
      raises. A user who has stored no credential, an unset master key, a decrypt
      failure, or a database hiccup all resolve to ``{}``, so the call falls back
      to the server's static workspace header. Additive - a server that used the
      shared header keeps working whether or not a user has connected one.
    - ``required=True`` (a ``credential_scope=user`` server from the DB registry):
      it FAILS CLOSED. Every non-success outcome raises
      :class:`MissingUserCredentialError` instead of returning ``{}``, so a
      user-scoped server never silently falls back to the shared workspace
      credential - the whole point of per-user scoping.
    """

    def __init__(
        self,
        *,
        sessionmaker: async_sessionmaker[AsyncSession],
        cipher: SecretCipher | None,
        secret_name: str,
        scope: SecretScope = SecretScope.USER,
        required: bool = False,
    ) -> None:
        """Bind the resolver to its collaborators and the secret it reveals.

        ``required`` selects fail-closed (a user-scoped server) over the additive
        fallback (the shared-header default). See the class docstring.
        """
        self._sessionmaker = sessionmaker
        self._cipher = cipher
        self._secret_name = secret_name
        self._scope = scope
        self._required = required

    async def __call__(self, context: ToolContext) -> dict[str, str]:
        """Return the acting user's Authorization header for this call.

        Returns ``{}`` (fall back to the static header) on any non-success when
        ``required`` is False; raises :class:`MissingUserCredentialError` on any
        non-success when ``required`` is True (fail closed).

        The identity is rebuilt from ``context`` alone (never an argument to the
        SecretStore), so a USER-scoped reveal can only ever reach the caller's
        own row; the store's AAD tripwire is the backstop if a query somehow did
        not.

        Raises:
            MissingUserCredentialError: ``required`` is True and no usable
                per-user credential could be revealed for this caller.
        """
        tenant = TenantContext(
            tenant_id=UUID(context.tenant_id),
            user_id=UUID(context.user_id),
        )
        try:
            async with self._sessionmaker() as session:
                store = SecretStore(session, tenant, self._cipher)
                secret = await store.reveal(self._secret_name, scope=self._scope)
        except SecretNotFoundError as exc:
            # The user has not connected this server: expected, not an error.
            return self._deny(exc)
        except SecretError as exc:
            # No master key, or another secret-layer problem: degrade to the
            # static header (or fail closed). No traceback - a configuration state.
            logger.info(
                "per-user MCP credential unavailable for %r (%s); %s",
                self._secret_name,
                type(exc).__name__,
                "failing closed" if self._required else "using the static header",
            )
            return self._deny(exc)
        except Exception as exc:
            # A decrypt/tamper failure or a database hiccup must not fall back
            # silently on a user-scoped server; log it for audit either way.
            logger.warning(
                "per-user MCP credential resolution failed for %r; %s",
                self._secret_name,
                "failing closed" if self._required else "using the static header",
                exc_info=True,
            )
            return self._deny(exc)
        return {"Authorization": f"Bearer {secret}"}

    def _deny(self, exc: Exception) -> dict[str, str]:
        """Fail closed (raise) when required, else fall back to the static header."""
        if self._required:
            raise MissingUserCredentialError(self._secret_name) from exc
        return {}
