"""Workspace endpoints: the tenant-wide defaults every turn resolves through.

The writer for ``workspace_settings``, which read-only callers have consulted
since the resolution chain landed. Routers stay HTTP-only: the merge, the row
lock and the name validation all live in
:mod:`app.services.workspace_defaults`, so the settings page, a manager row
action and anything else that changes a default share one implementation
rather than three.

Reads are open to any member - the defaults describe how the workspace's own
chat behaves, and a member who cannot see them cannot understand their own
turns. Writes are admin-gated, matching the model and integration registries:
one PATCH changes what everybody's next turn runs on.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.api.deps import get_current_tenant, get_session, get_settings, require_role
from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.schemas.settings import WorkspaceDefaults, WorkspaceDefaultsPatch
from app.schemas.workspace import WorkspaceDefaultsOut
from app.services.workspace_defaults import (
    apply_defaults,
    load_defaults,
    reranker_availability,
)

router = APIRouter(tags=["workspace"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]
AdminDep = Annotated[TenantContext, Depends(require_role("admin"))]
SettingsDep = Annotated[Settings, Depends(get_settings)]


async def _dress(defaults: WorkspaceDefaults, settings: Settings) -> WorkspaceDefaultsOut:
    """Add this deployment's facts to a set of stored defaults already in hand."""
    return WorkspaceDefaultsOut.build(defaults, reranker_availability(settings))


@router.get("/workspace/defaults", response_model=WorkspaceDefaultsOut)
async def read_workspace_defaults(
    session: SessionDep,
    tenant: TenantDep,
    settings: SettingsDep,
) -> WorkspaceDefaultsOut:
    """Return the workspace's stored defaults and what this deployment can honour.

    Every value may be null, which means UNSET: the next tier of the
    resolution chain decides that axis. The reranker status says whether a
    sidecar exists for the toggle to govern at all, and the ceiling bounds and
    the system-prompt length limit are what a write will accept - all facts
    about the deployment, so a page never has to infer them from a failed save.
    """
    return await _dress(await load_defaults(session, tenant), settings)


@router.patch("/workspace/defaults", response_model=WorkspaceDefaultsOut)
async def update_workspace_defaults(
    payload: WorkspaceDefaultsPatch,
    session: SessionDep,
    tenant: AdminDep,
    settings: SettingsDep,
) -> WorkspaceDefaultsOut:
    """Change some of the workspace defaults, leaving the rest exactly as they were.

    A per-field merge, never a whole-blob write: a field absent from the body
    is untouched, an explicit ``null`` clears it, and a value overwrites. That
    is what lets two controls save at once without either dropping the
    other's field - and the read-merge-write holds a row lock so the two
    genuinely serialize. Unknown keys are refused (422) rather than stored
    where nothing would ever read them.

    Committed before responding, so the page's next read - and the very next
    chat turn - see what was just saved. The response is built from the merge's
    own result rather than by re-reading the row: they are the same answer, and
    a second read is a query whose only job is to agree with the first.
    """
    merged = await apply_defaults(session, tenant, payload)
    await session.commit()
    return await _dress(merged, settings)
