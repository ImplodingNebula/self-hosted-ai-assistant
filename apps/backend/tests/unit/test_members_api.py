"""The members API - the roster and the role change, with both of its guards.

Drives the real endpoints over the in-process app (ASGITransport + the sqlite
sessionmaker), overriding only the auth seam (``get_current_tenant``) and
``get_settings``. That override is the whole point: pre-sign-in every caller is
the seeded admin, so an admin gate and a tenant boundary can only be proven by
handing the app a different principal - which is exactly what real login will do.

The two refusals are what this surface is for, so each has a test that fails if
the guard is removed:

* the last administrator cannot be demoted (nobody would be left to promote
  anyone);
* an account the deployment lists as an administrator cannot be demoted here
  (the environment re-floors it at the next sign-in, so the change would appear
  to work and silently revert).
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from unittest import mock
from uuid import uuid4

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import update
from sqlalchemy.dialects import postgresql

from app.api.deps import get_current_tenant, get_settings
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models import Tenant, User
from app.repositories.users import UserRepository

pytestmark = pytest.mark.anyio

MEMBERS = "/api/v1/members"

ADMIN = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")
OTHER_TENANT_ID = uuid4()


class _NoOidcSettings(Settings):
    """Settings as they are on a build that predates the sign-in work.

    ``oidc_admin_emails`` is owned by that work and does not exist on this
    branch, so the env-floor read has to survive its absence - and the way to
    prove it survives is a settings object that genuinely lacks the attribute,
    not one that has it set to empty.
    """

    def __getattribute__(self, name: str):  # noqa: ANN401 - mirrors object.__getattribute__
        if name == "oidc_admin_emails":
            raise AttributeError(name)
        return super().__getattribute__(name)


def _settings(admin_emails: object = None, **overrides) -> Settings:
    """Settings with a stand-in for the sign-in work's admin-floor list.

    The field arrives with that work; until it does, the attribute is attached
    here in the shapes the environment can produce (a list, or the raw
    comma-separated string) so the defensive read is tested against both.
    """
    settings = Settings(_env_file=None, model_provider="fake", **overrides)
    if admin_emails is not None:
        object.__setattr__(settings, "oidc_admin_emails", admin_emails)
    return settings


@asynccontextmanager
async def members_client(db_sessionmaker, *, tenant=ADMIN, settings=None):
    """An app-wired client acting as ``tenant``."""
    app = create_app()
    app.state.db_sessionmaker = db_sessionmaker
    app.dependency_overrides[get_current_tenant] = lambda: tenant
    app.dependency_overrides[get_settings] = lambda: settings or _settings()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
    app.dependency_overrides.clear()


async def add_user(
    db_sessionmaker,
    *,
    display_name: str,
    email: str | None = None,
    role: str = "member",
    tenant_id=DEFAULT_TENANT_ID,
) -> User:
    """Insert one user row (the shape real sign-in will create)."""
    async with db_sessionmaker() as session:
        if tenant_id != DEFAULT_TENANT_ID and await session.get(Tenant, tenant_id) is None:
            session.add(Tenant(id=tenant_id, name=str(tenant_id)))
            await session.commit()
        user = User(tenant_id=tenant_id, display_name=display_name, email=email, role=role)
        session.add(user)
        await session.commit()
        await session.refresh(user)
        return user


async def roles_of(client: AsyncClient) -> dict[str, str]:
    """The roster as {display_name: role}."""
    response = await client.get(MEMBERS)
    assert response.status_code == 200, response.text
    return {row["display_name"]: row["role"] for row in response.json()["items"]}


# --- the roster ------------------------------------------------------------


async def test_list_returns_the_workspace_roster_oldest_first(db_sessionmaker) -> None:
    await add_user(db_sessionmaker, display_name="Ada", email="ada@example.com")
    async with members_client(db_sessionmaker) as client:
        response = await client.get(MEMBERS)

    assert response.status_code == 200
    items = response.json()["items"]
    # The seeded principal joined first, so it leads the list.
    assert [row["display_name"] for row in items] == ["Default User", "Ada"]
    assert items[0]["role"] == "admin"
    assert items[1]["email"] == "ada@example.com"
    # tenant_id is a server-side fact and never crosses the wire.
    assert all("tenant_id" not in row for row in items)


async def test_list_never_shows_another_tenants_people(db_sessionmaker) -> None:
    await add_user(
        db_sessionmaker, display_name="Outsider", role="admin", tenant_id=OTHER_TENANT_ID
    )
    async with members_client(db_sessionmaker) as client:
        assert "Outsider" not in await roles_of(client)


async def test_a_member_cannot_read_the_roster(db_sessionmaker) -> None:
    # The list names every colleague's email address; that is not something a
    # member needs in order to use the assistant.
    async with members_client(db_sessionmaker, tenant=MEMBER) as client:
        response = await client.get(MEMBERS)

    assert response.status_code == 403
    assert response.headers["content-type"].startswith("application/problem+json")


async def test_a_member_cannot_change_a_role(db_sessionmaker) -> None:
    target = await add_user(db_sessionmaker, display_name="Ada")
    async with members_client(db_sessionmaker, tenant=MEMBER) as client:
        response = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "admin"})
    assert response.status_code == 403

    async with members_client(db_sessionmaker) as client:
        assert (await roles_of(client))["Ada"] == "member"


# --- the role change -------------------------------------------------------


async def test_promote_and_demote_round_trip(db_sessionmaker) -> None:
    target = await add_user(db_sessionmaker, display_name="Ada")
    async with members_client(db_sessionmaker) as client:
        promoted = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "admin"})
        assert promoted.status_code == 200
        assert promoted.json()["role"] == "admin"
        assert (await roles_of(client))["Ada"] == "admin"

        demoted = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "member"})
        assert demoted.status_code == 200
        assert (await roles_of(client))["Ada"] == "member"


async def test_an_unknown_or_malformed_id_is_a_404(db_sessionmaker) -> None:
    async with members_client(db_sessionmaker) as client:
        missing = await client.patch(f"{MEMBERS}/{uuid4()}", json={"role": "admin"})
        malformed = await client.patch(f"{MEMBERS}/not-a-uuid", json={"role": "admin"})

    assert missing.status_code == 404
    assert malformed.status_code == 404


async def test_another_tenants_member_is_indistinguishable_from_a_missing_one(
    db_sessionmaker,
) -> None:
    outsider = await add_user(db_sessionmaker, display_name="Outsider", tenant_id=OTHER_TENANT_ID)
    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{outsider.id}", json={"role": "admin"})

    assert response.status_code == 404
    async with db_sessionmaker() as session:
        assert (await session.get(User, outsider.id)).role == "member"


async def test_an_unknown_role_is_rejected_before_anything_is_written(db_sessionmaker) -> None:
    target = await add_user(db_sessionmaker, display_name="Ada")
    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "owner"})
        assert response.status_code == 422
        assert (await roles_of(client))["Ada"] == "member"


# --- guard: the last administrator -----------------------------------------


def test_the_admin_count_is_read_under_a_row_lock() -> None:
    # MUTATION GUARD: drop .with_for_update() from lock_admins and this fails.
    #
    # The guard reads the admin set and then writes, which is only safe if the
    # read is serialized: two demotions of DIFFERENT admins overlapping in a
    # two-admin workspace would otherwise each see the other still in place,
    # both pass, and leave the workspace with no administrator. That race cannot
    # be staged on the sqlite test database - it takes one writer at a time, so
    # the interleaving does not exist there - so what is asserted is the SQL the
    # repository actually emits against the database this ships on.
    session = mock.Mock()
    repo = UserRepository(session, ADMIN)
    statement = repo.locked_admins_statement()
    compiled = str(statement.compile(dialect=postgresql.dialect()))

    assert "FOR UPDATE" in compiled
    assert "role" in compiled
    # A deterministic lock order, so two of these queue instead of deadlocking.
    assert "ORDER BY" in compiled


async def test_a_demotion_that_lost_the_race_writes_nothing(db_sessionmaker, monkeypatch) -> None:
    # The other half of the lock: the request that WAITED must decide on the
    # world it wakes up in. Ada is demoted out of band in the window between the
    # target being loaded and the admin set being locked, which is precisely
    # what a granted lock reveals on a real database - and the guard must then
    # treat the change as already done rather than re-applying it or refusing.
    ada = await add_user(db_sessionmaker, display_name="Ada", role="admin")
    original = UserRepository.lock_admins

    async def demote_first_then_lock(self):
        monkeypatch.undo()
        # synchronize_session=False on purpose: a demotion committed by ANOTHER
        # request never reaches into this session's identity map, so the row
        # loaded before the wait stays stale here exactly as it would in
        # production. Letting SQLAlchemy synchronize it would hand the guard a
        # fresh object for free and leave the re-read untested.
        await self.session.execute(
            update(User)
            .where(User.id == ada.id)
            .values(role="member")
            .execution_options(synchronize_session=False)
        )
        return await original(self)

    monkeypatch.setattr(UserRepository, "lock_admins", demote_first_then_lock)

    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{ada.id}", json={"role": "member"})

        assert response.status_code == 200
        # The body describes the row as it now IS, not as it was loaded before
        # the wait: the locking query refreshes only the rows it returns, and
        # this one is absent from them exactly because it was demoted.
        assert response.json()["role"] == "member"
        # The seeded admin is untouched: nothing tried to "finish" a change that
        # had already happened, and no second demotion slipped past the count.
        assert (await roles_of(client))["Default User"] == "admin"


async def test_the_last_admin_cannot_be_demoted(db_sessionmaker) -> None:
    # MUTATION GUARD: drop the admin-count check in set_member_role and the
    # seeded admin becomes a member here, leaving nobody able to promote anyone.
    await add_user(db_sessionmaker, display_name="Ada")
    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{DEFAULT_USER_ID}", json={"role": "member"})

        assert response.status_code == 409
        assert response.headers["content-type"].startswith("application/problem+json")
        assert "only administrator" in response.json()["detail"]
        assert (await roles_of(client))["Default User"] == "admin"


async def test_an_admin_may_step_down_once_somebody_else_can_administer(
    db_sessionmaker,
) -> None:
    # Self-demotion is allowed - it is only the LAST admin that is refused, and
    # a guard that refused every self-demotion would be a different rule.
    await add_user(db_sessionmaker, display_name="Ada", role="admin")
    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{DEFAULT_USER_ID}", json={"role": "member"})

        assert response.status_code == 200
        assert (await roles_of(client))["Default User"] == "member"


async def test_demoting_the_last_admin_to_admin_is_a_no_op_not_a_refusal(
    db_sessionmaker,
) -> None:
    # Setting the role it already holds changes nothing, so the last-admin guard
    # has nothing to protect against and must not fire.
    async with members_client(db_sessionmaker) as client:
        response = await client.patch(f"{MEMBERS}/{DEFAULT_USER_ID}", json={"role": "admin"})

    assert response.status_code == 200
    assert response.json()["role"] == "admin"


# --- guard: the deployment's admin floor -----------------------------------


@pytest.mark.parametrize(
    "configured",
    [["ada@example.com"], "ada@example.com, bob@example.com", ["ADA@EXAMPLE.COM"]],
    ids=["list", "comma-separated-string", "different-case"],
)
async def test_a_deployment_listed_admin_cannot_be_demoted(db_sessionmaker, configured) -> None:
    # MUTATION GUARD: drop the is_env_admin check and this demotion succeeds -
    # and then silently reverts at Ada's next sign-in, which is the failure the
    # refusal exists to prevent. Parametrized over every shape the setting can
    # arrive in, because the read is defensive by necessity.
    target = await add_user(
        db_sessionmaker, display_name="Ada", email="Ada@example.com", role="admin"
    )
    settings = _settings(admin_emails=configured)
    async with members_client(db_sessionmaker, settings=settings) as client:
        response = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "member"})

        assert response.status_code == 409
        assert "OIDC_ADMIN_EMAILS" in response.json()["detail"]
        assert (await roles_of(client))["Ada"] == "admin"


async def test_the_roster_flags_deployment_listed_admins(db_sessionmaker) -> None:
    await add_user(db_sessionmaker, display_name="Ada", email="ada@example.com", role="admin")
    settings = _settings(admin_emails=["ada@example.com"])
    async with members_client(db_sessionmaker, settings=settings) as client:
        response = await client.get(MEMBERS)

    flags = {row["display_name"]: row["is_env_admin"] for row in response.json()["items"]}
    assert flags == {"Ada": True, "Default User": False}


async def test_an_unlisted_admin_is_demotable_while_a_listed_one_stands(
    db_sessionmaker,
) -> None:
    # The floor is per ACCOUNT, not a blanket freeze on the admin role.
    target = await add_user(
        db_sessionmaker, display_name="Bob", email="bob@example.com", role="admin"
    )
    settings = _settings(admin_emails=["ada@example.com"])
    async with members_client(db_sessionmaker, settings=settings) as client:
        response = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "member"})

        assert response.status_code == 200
        assert (await roles_of(client))["Bob"] == "member"


async def test_the_floor_is_absent_when_the_setting_is(db_sessionmaker) -> None:
    # This branch runs against a Settings object that genuinely has no
    # oidc_admin_emails attribute - the state of every build before the sign-in
    # work merges. Nothing is floored, which is the honest pre-login answer.
    target = await add_user(
        db_sessionmaker, display_name="Ada", email="ada@example.com", role="admin"
    )
    settings = _NoOidcSettings(_env_file=None, model_provider="fake")
    async with members_client(db_sessionmaker, settings=settings) as client:
        listed = await client.get(MEMBERS)
        assert all(row["is_env_admin"] is False for row in listed.json()["items"])

        response = await client.patch(f"{MEMBERS}/{target.id}", json={"role": "member"})
        assert response.status_code == 200
