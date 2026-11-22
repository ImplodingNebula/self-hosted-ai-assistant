"""Access to the tenant's workspace-wide defaults.

``workspace_settings`` is keyed by ``tenant_id`` itself (one row per tenant,
no ``id`` column), so the generic ``BaseRepository.get`` - which selects on
``id`` - does not apply; this repository reads by primary key directly. It is
tenant-scoped like the MCP server registry: a workspace resource, shared by
every user of the tenant.
"""

from __future__ import annotations

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.tenancy import TenantContext
from app.models import WorkspaceSettings, utcnow


class WorkspaceSettingsRepository:
    """Reads and merges the bound tenant's workspace defaults."""

    def __init__(self, session: AsyncSession, tenant: TenantContext) -> None:
        self.session = session
        self.tenant = tenant

    async def load(self) -> dict:
        """Return the stored defaults blob, or an empty dict.

        Robust to an absent row (a deployment that skipped seeding, a bare
        test database): defaults simply read as unset rather than crashing
        the turn that asked.
        """
        row = await self.session.get(WorkspaceSettings, self.tenant.tenant_id)
        if row is None or not row.settings:
            return {}
        return dict(row.settings)

    async def load_for_update(self) -> dict:
        """Return the stored defaults holding a row lock for a read-merge-write.

        The settings PATCH merges the stored JSON with the request and writes
        it back; without the lock two overlapping single-field saves could both
        read the same base and the later commit would silently drop the
        earlier one's field - the same hazard the conversation PATCH takes
        ``FOR UPDATE`` for, and a no-op on sqlite's single-writer model.

        ``populate_existing`` is belt-and-braces, and the honest version of why
        it is here: a locking ``get()`` already re-emits SQL rather than serving
        the identity map, so a cross-session probe cannot tell this flag's
        presence from its absence - it was tried, and it could not. What it
        pins is the INTENT, which the lock alone does not state: the caller
        wants this row's current values, not whichever copy the session loaded
        earlier. That matters because a caller may well have read the defaults
        already in the same session - the delete flow does, to decide whether
        the model it is removing is the default - and merging over a stale blob
        under a lock is a lost update wearing a lock's clothing.

        An absent row takes no lock, because there is no row to lock. Two
        PATCHes racing against a tenant whose row was never seeded therefore
        both INSERT and the loser gets an integrity error (a 500, and its save
        is lost). Accepted rather than handled: the row is seeded at boot
        beside its tenant, so reaching this needs a deployment that skipped
        seeding AND two simultaneous admin saves - and the honest failure is a
        visibly refused save, not a silent one.

        Returns:
            The stored defaults blob, or an empty dict.
        """
        row = await self.session.get(
            WorkspaceSettings,
            self.tenant.tenant_id,
            with_for_update=True,
            populate_existing=True,
        )
        if row is None or not row.settings:
            return {}
        return dict(row.settings)

    async def replace_defaults(self, values: dict[str, object]) -> None:
        """Store ``values`` as the whole defaults blob, creating the row if absent.

        The write half of a read-merge-write: unlike :meth:`merge_defaults` it
        can REMOVE a key, which is what an explicit ``null`` in the PATCH
        means. Pass a dict that already carries everything worth keeping -
        anything missing from it is gone.

        Staged and flushed, not committed, like every other repository here.

        Args:
            values: The complete settings blob to store.
        """
        await self._write(dict(values))

    async def merge_defaults(self, values: dict[str, object]) -> None:
        """Merge ``values`` over the stored defaults, creating the row if absent.

        Only the keys in ``values`` change; anything else already stored -
        including keys written by another code version - rides along
        untouched, the same per-field merge the conversation PATCH does.

        The merged dict is REASSIGNED to the column rather than updated in
        place. The JSON column has no mutable wrapper, so only a new object
        marks the row dirty: an in-place update reads back correctly within
        the session and then silently never persists (the same trap documented
        on :meth:`app.schemas.settings._SettingsPatch.merged_into`).

        Staged and flushed, not committed: the session owner decides the
        transaction boundary, like every other repository here.

        Args:
            values: The keys to write over the stored defaults.
        """
        row = await self.session.get(WorkspaceSettings, self.tenant.tenant_id)
        stored = dict(row.settings) if row is not None and isinstance(row.settings, dict) else {}
        stored.update(values)
        await self._write(stored, row=row)

    async def _write(
        self, values: dict[str, object], *, row: WorkspaceSettings | None = None
    ) -> None:
        """Assign ``values`` to the tenant's row, creating it when absent.

        The reassignment is load-bearing: the JSON column has no mutable
        wrapper, so only a new object marks the row dirty.

        Args:
            values: The complete blob to store.
            row: The already-loaded row, when the caller has one.
        """
        if row is None:
            row = await self.session.get(WorkspaceSettings, self.tenant.tenant_id)
        if row is None:
            # A deployment that skipped seeding still gets a working default;
            # the row is this tenant's by construction (primary key = tenant).
            row = WorkspaceSettings(tenant_id=self.tenant.tenant_id, settings={})
            self.session.add(row)
        row.settings = values
        row.updated_at = utcnow()
        await self.session.flush()
