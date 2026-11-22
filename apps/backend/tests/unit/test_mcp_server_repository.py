"""McpServerRepository: the per-tenant MCP-server registry read path (6b PR2).

Servers are a WORKSPACE resource, so the repository scopes by tenant only. These
prove the read the source builder depends on (enabled + name-ordered + tenant
isolated) and the bootstrap seed's two load-bearing properties: idempotent and
NON-DESTRUCTIVE (an operator's edit to the seeded row is never clobbered).
"""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.core.errors import ConflictError
from app.core.tenancy import DEFAULT_TENANT_ID, TenantContext
from app.models import McpCredentialScope, McpServer
from app.repositories.mcp_servers import McpServerRepository

pytestmark = pytest.mark.anyio

# Two tenants; only the tenant id differs. TENANT_A is the seeded default.
TENANT_A = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=uuid4(), role="admin")
TENANT_B = TenantContext(tenant_id=uuid4(), user_id=uuid4(), role="admin")


async def _add(db_sessionmaker, tenant: TenantContext, **kwargs) -> McpServer:
    """Insert a server row for ``tenant`` and return it."""
    async with db_sessionmaker() as session:
        repo = McpServerRepository(session, tenant)
        row = repo.add(McpServer(tenant_id=tenant.tenant_id, base_url="http://x/mcp", **kwargs))
        await session.commit()
        await session.refresh(row)
        return row


async def test_list_enabled_returns_only_enabled_rows_name_ordered(db_sessionmaker) -> None:
    await _add(db_sessionmaker, TENANT_A, name="beta", enabled=True)
    await _add(db_sessionmaker, TENANT_A, name="alpha", enabled=True)
    await _add(db_sessionmaker, TENANT_A, name="gamma", enabled=False)

    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_enabled()

    # gamma is disabled and excluded; the rest come back name-ordered so which
    # source wins a tool-name collision is stable across turns.
    assert [row.name for row in rows] == ["alpha", "beta"]


async def test_list_enabled_is_tenant_scoped(db_sessionmaker) -> None:
    await _add(db_sessionmaker, TENANT_A, name="a-server", enabled=True)
    await _add(db_sessionmaker, TENANT_B, name="b-server", enabled=True)

    async with db_sessionmaker() as session:
        a_rows = await McpServerRepository(session, TENANT_A).list_enabled()
        b_rows = await McpServerRepository(session, TENANT_B).list_enabled()

    # Each tenant sees only its own server - the registry is per-tenant.
    assert [row.name for row in a_rows] == ["a-server"]
    assert [row.name for row in b_rows] == ["b-server"]


async def test_credential_scope_defaults_to_workspace(db_sessionmaker) -> None:
    row = await _add(db_sessionmaker, TENANT_A, name="plain")
    # A row inserted without a scope keeps today's shared-header behaviour.
    assert row.credential_scope == McpCredentialScope.WORKSPACE.value


async def test_ensure_seed_inserts_once_and_is_idempotent(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        first = await McpServerRepository(session, TENANT_A).ensure_seed(
            name="mcp-chassis",
            base_url="http://chassis/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
            config={"seeded_from_env": True},
        )
        await session.commit()
        first_id = first.id

    async with db_sessionmaker() as session:
        second = await McpServerRepository(session, TENANT_A).ensure_seed(
            name="mcp-chassis",
            base_url="http://chassis/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
        )
        await session.commit()

    # Second seed does not duplicate the row.
    assert second.id == first_id
    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_enabled()
    assert [row.name for row in rows] == ["mcp-chassis"]


async def test_ensure_seed_never_overwrites_an_operator_edit(db_sessionmaker) -> None:
    # Seed the row, then simulate an operator editing it in the DB (disabled,
    # re-pointed, rescoped). A later re-seed with the ORIGINAL env values must
    # leave every one of those edits intact - the env value has receded to a
    # first-boot seed and the DB row is authoritative.
    async with db_sessionmaker() as session:
        seeded = await McpServerRepository(session, TENANT_A).ensure_seed(
            name="mcp-chassis",
            base_url="http://original/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
        )
        seeded.enabled = False
        seeded.base_url = "http://operator-edited/mcp"
        seeded.credential_scope = McpCredentialScope.USER.value
        session.add(seeded)
        await session.commit()
        edited_id = seeded.id

    async with db_sessionmaker() as session:
        reseeded = await McpServerRepository(session, TENANT_A).ensure_seed(
            name="mcp-chassis",
            base_url="http://original/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
        )
        await session.commit()

    assert reseeded.id == edited_id
    assert reseeded.enabled is False
    assert reseeded.base_url == "http://operator-edited/mcp"
    assert reseeded.credential_scope == McpCredentialScope.USER.value


async def test_create_inserts_a_tenant_stamped_row(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        server = await McpServerRepository(session, TENANT_A).create(
            name="new",
            base_url="http://x/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
        )
        await session.commit()
    # tenant_id is stamped from the bound context, not the caller.
    assert server.tenant_id == TENANT_A.tenant_id
    assert server.name == "new"
    assert server.credential_scope == McpCredentialScope.WORKSPACE.value


async def test_create_with_a_duplicate_name_raises_conflict(db_sessionmaker) -> None:
    async with db_sessionmaker() as session:
        await McpServerRepository(session, TENANT_A).create(
            name="dup",
            base_url="http://x/mcp",
            credential_scope=McpCredentialScope.WORKSPACE.value,
        )
        await session.commit()

    async with db_sessionmaker() as session:
        with pytest.raises(ConflictError):
            await McpServerRepository(session, TENANT_A).create(
                name="dup",
                base_url="http://y/mcp",
                credential_scope=McpCredentialScope.WORKSPACE.value,
            )


async def test_list_all_returns_enabled_and_disabled_tenant_scoped(db_sessionmaker) -> None:
    await _add(db_sessionmaker, TENANT_A, name="beta", enabled=True)
    await _add(db_sessionmaker, TENANT_A, name="alpha", enabled=False)
    await _add(db_sessionmaker, TENANT_B, name="other", enabled=True)

    async with db_sessionmaker() as session:
        rows = await McpServerRepository(session, TENANT_A).list_all()

    # Includes the disabled "alpha", name-ordered, and never TENANT_B's "other".
    assert [row.name for row in rows] == ["alpha", "beta"]
