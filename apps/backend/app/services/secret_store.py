"""The SecretStore seam - write-only credential storage, scoped and sealed.

Composes the AES-GCM cipher (``app.providers.secrets.cipher``) with the
``secrets`` table, and is the ONLY way credentials are written or read. The
rules below make a cross-user or cross-scope leak unrepresentable rather than
merely unlikely:

- **Every method takes a** :class:`SecretScope`, **never a ``user_id``.** The
  identity a scope resolves to comes from the bound :class:`TenantContext` and
  nowhere else, so one member cannot address another's secret by passing an id.
- **``reveal`` is named ``reveal``, not ``get``** - the one call that returns
  plaintext reads as dangerous at every call site.
- **No by-name lookup spans scopes.** A workspace name and a user name are
  different rows found by different queries; there is no method that searches
  both at once.
- **The AAD is built from the EXPECTED identity, not the fetched row.** On a
  reveal the tenant and user woven into the associated data come from the
  context, so if a query ever returned the wrong row's ciphertext, GCM
  verification would fail instead of handing back someone else's credential.
"""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql.elements import ColumnElement

from app.core.tenancy import TenantContext
from app.models import Secret, utcnow
from app.providers.secrets import (
    SecretCipher,
    SecretMetadata,
    SecretNotFoundError,
    SecretScope,
    SecretStoreUnavailableError,
)

# The key version new writes are sealed under. Rotation (a second key + a
# re-seal sweep) bumps this later; today every row is version 1.
_CURRENT_KEY_VERSION = 1
# The AAD token standing in for "no owner" on a workspace secret. A UUID never
# renders as this word, so a workspace AAD can never collide with a user's.
_WORKSPACE_OWNER_TOKEN = "workspace"
# The shortest secret whose last four characters may be surfaced as the hint.
# Below this, ``last4`` would be most or all of the value, so it is hidden
# entirely - the write-only contract must not spring a leak for a short PIN.
_LAST4_MIN_LEN = 8


class SecretStore:
    """Seal, reveal, list and delete a principal's credentials.

    Bound to one request: a database session, the acting :class:`TenantContext`,
    and the process cipher (``None`` when ``SECRET_MASTER_KEY`` is unset, in
    which case sealing and revealing raise :class:`SecretStoreUnavailableError`
    and the surface above answers 503).
    """

    def __init__(
        self,
        session: AsyncSession,
        tenant: TenantContext,
        cipher: SecretCipher | None,
    ) -> None:
        self.session = session
        self.tenant = tenant
        self._cipher = cipher

    async def store(self, name: str, value: str, *, scope: SecretScope) -> SecretMetadata:
        """Seal ``value`` under ``name`` in ``scope``, replacing any existing one.

        An upsert on the natural key (tenant, owner, name), so re-connecting an
        account overwrites the old token rather than colliding with it. The
        read-then-write is not atomic on its own, so a concurrent first-write of
        the same name is caught on the unique constraint and folded into an
        update rather than surfacing as a 500 - the constraint is the backstop.

        Raises:
            SecretStoreUnavailableError: No master key is configured.
        """
        cipher = self._require_cipher()
        aad = self._aad(scope=scope, name=name, key_version=_CURRENT_KEY_VERSION)
        ciphertext = cipher.encrypt(value, aad=aad)
        last4 = value[-4:] if len(value) >= _LAST4_MIN_LEN else ""
        existing = await self._find(name=name, scope=scope)
        if existing is None:
            row = Secret(
                tenant_id=self.tenant.tenant_id,
                user_id=self._owner(scope),
                name=name,
                ciphertext=ciphertext,
                last4=last4,
                key_version=_CURRENT_KEY_VERSION,
            )
            try:
                # A savepoint, so a lost insert race rolls back just this INSERT -
                # not the caller's whole transaction - leaving the session usable
                # to update the row that won.
                async with self.session.begin_nested():
                    self.session.add(row)
                    await self.session.flush()
            except IntegrityError:
                existing = await self._find(name=name, scope=scope)
                if existing is None:
                    raise
            else:
                return self._to_metadata(row, scope)
        existing.ciphertext = ciphertext
        existing.last4 = last4
        existing.key_version = _CURRENT_KEY_VERSION
        existing.updated_at = utcnow()
        self.session.add(existing)
        await self.session.flush()
        return self._to_metadata(existing, scope)

    async def store_existing_only(self, name: str, value: str, *, scope: SecretScope) -> bool:
        """Re-seal ``value`` under an EXISTING ``name``; return False if the row is gone.

        The update-only half of :meth:`store`, for a writer that is renewing
        something it read earlier rather than connecting it: a rotated OAuth
        token must never RESURRECT a grant an admin disconnected while the
        renewal was in flight. ``store`` is an upsert, so it would happily
        re-insert the deleted row with tokens that are newer - and still live -
        than the ones the disconnect revoked.

        This is the guarantee that holds across worker processes, where an
        in-memory lock does not: the row's absence IS the signal.

        Raises:
            SecretStoreUnavailableError: No master key is configured.
        """
        cipher = self._require_cipher()
        existing = await self._find(name=name, scope=scope)
        if existing is None:
            return False
        aad = self._aad(scope=scope, name=name, key_version=_CURRENT_KEY_VERSION)
        existing.ciphertext = cipher.encrypt(value, aad=aad)
        existing.last4 = value[-4:] if len(value) >= _LAST4_MIN_LEN else ""
        existing.key_version = _CURRENT_KEY_VERSION
        existing.updated_at = utcnow()
        self.session.add(existing)
        await self.session.flush()
        return True

    async def get(self, name: str, *, scope: SecretScope) -> SecretMetadata | None:
        """Return a secret's metadata (name, last4, timestamps), or None. No plaintext."""
        row = await self._find(name=name, scope=scope)
        return None if row is None else self._to_metadata(row, scope)

    async def reveal(self, name: str, *, scope: SecretScope) -> str:
        """Decrypt and return the plaintext - the one call that exposes a secret.

        Raises:
            SecretStoreUnavailableError: No master key is configured.
            SecretNotFoundError: No such secret in this scope for this principal.
            cryptography.exceptions.InvalidTag: The stored bytes do not verify
                under this row's identity - tampering, the wrong master key, or a
                blob that does not belong to the row it was fetched as.
        """
        cipher = self._require_cipher()
        row = await self._find(name=name, scope=scope)
        if row is None:
            raise SecretNotFoundError(f"No {scope.value} secret named {name!r}.")
        aad = self._aad(scope=scope, name=name, key_version=row.key_version)
        return cipher.decrypt(row.ciphertext, aad=aad)

    async def list(self, *, scope: SecretScope) -> list[SecretMetadata]:
        """Return metadata for every secret in ``scope`` for this principal, name order."""
        stmt = (
            select(Secret)
            .where(Secret.tenant_id == self.tenant.tenant_id, self._scope_predicate(scope))
            .order_by(Secret.name)
        )
        rows = (await self.session.execute(stmt)).scalars().all()
        return [self._to_metadata(row, scope) for row in rows]

    async def delete(self, name: str, *, scope: SecretScope) -> bool:
        """Delete a secret; return False when it did not exist."""
        row = await self._find(name=name, scope=scope)
        if row is None:
            return False
        await self.session.delete(row)
        await self.session.flush()
        return True

    def _require_cipher(self) -> SecretCipher:
        if self._cipher is None:
            raise SecretStoreUnavailableError(
                "SECRET_MASTER_KEY is not configured; credentials cannot be sealed or revealed."
            )
        return self._cipher

    def _owner(self, scope: SecretScope):
        """The ``user_id`` a scope writes: NULL for workspace, the bound user otherwise."""
        return None if scope is SecretScope.WORKSPACE else self.tenant.user_id

    def _scope_predicate(self, scope: SecretScope) -> ColumnElement[bool]:
        """The owner filter for a scope - NULL owner for workspace, the bound user else.

        The bound user is read from the context, never an argument, so a USER
        query can only ever match the caller's own rows.
        """
        if scope is SecretScope.WORKSPACE:
            return Secret.user_id.is_(None)
        return Secret.user_id == self.tenant.user_id

    async def _find(self, *, name: str, scope: SecretScope) -> Secret | None:
        stmt = select(Secret).where(
            Secret.tenant_id == self.tenant.tenant_id,
            Secret.name == name,
            self._scope_predicate(scope),
        )
        return (await self.session.execute(stmt)).scalar_one_or_none()

    def _aad(self, *, scope: SecretScope, name: str, key_version: int) -> bytes:
        """Associated data binding a ciphertext to its row identity.

        Built from the EXPECTED principal (the bound tenant/user), not a fetched
        row, so it doubles as a leak tripwire: a ciphertext sealed for one
        (tenant, owner, name) will not verify when opened as another. ``name`` is
        last, after the fixed-format UUID/int fields, so no name can forge the
        prefix of a different row's AAD.
        """
        owner = (
            _WORKSPACE_OWNER_TOKEN if scope is SecretScope.WORKSPACE else str(self.tenant.user_id)
        )
        return f"{self.tenant.tenant_id}:{owner}:{key_version}:{name}".encode()

    def _to_metadata(self, row: Secret, scope: SecretScope) -> SecretMetadata:
        return SecretMetadata(
            name=row.name,
            scope=scope,
            last4=row.last4,
            key_version=row.key_version,
            created_at=row.created_at,
            updated_at=row.updated_at,
        )
