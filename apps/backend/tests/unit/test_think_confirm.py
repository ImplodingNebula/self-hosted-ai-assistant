"""Confirming a measured think mode: the first writer the policy store has.

Every test drives the real service (and, for the refusals a client sees, the
real route), because the value of this endpoint is entirely in what it refuses
and in the SHAPE of the row it appends.

Two of the three store rules it inherits - drop the read cache, write one axis
- are pinned here by a test that fails when the rule is removed rather than by
a comment. The third (validate the stored value) is satisfied by construction
rather than by a branch: the only value this writer can store is one the probe
reader already vouched for, so the guard is tested where it lives
(``test_an_unknown_stored_mode_is_dropped`` in test_earned.py) and the outcome
is pinned here.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider
from app.core.config import Settings, get_settings
from app.core.errors import NotFoundError, UnprocessableEntityError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import ModelInfo
from app.providers.fake_provider import FakeProvider
from app.repositories.model_policy import ModelPolicyRepository
from app.services import think_check
from app.services.earned import clear_earned_cache
from app.services.model_policy import clear_policy_cache, policy_for, resolve_policy
from app.services.tool_check import harness_version

CHAT = "chat-tag"
DIGEST = "sha256:chatweights"
OTHER = "sha256:otherweights"


@pytest.fixture(autouse=True)
def _fresh_caches() -> Iterator[None]:
    clear_policy_cache()
    clear_earned_cache()
    yield
    clear_policy_cache()
    clear_earned_cache()


def tenant() -> TenantContext:
    # Admin, because the verbs that ACT are gated - the seam that starts
    # meaning something once OIDC populates a verified role.
    return TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


def provider(*, name: str = CHAT, digest: str | None = DIGEST) -> FakeProvider:
    return FakeProvider(models=[ModelInfo(name=name, digest=digest)])


async def seed_probe(
    db_sessionmaker,
    *,
    model: str = CHAT,
    digest: str = DIGEST,
    think_mode: object = "always",
    harness: str | None = None,
) -> str:
    """Append one probe row carrying a measured think mode."""
    row = ModelProbe(
        provider="fake",
        model=model,
        digest=digest,
        harness_version=harness if harness is not None else harness_version(),
        depth="fast",
        earned_agency="assisted",
        thinking_with_tools=False,
        cases={"measured_think_mode": think_mode},
    )
    async with db_sessionmaker() as session:
        session.add(row)
        await session.commit()
        return str(row.id)


async def confirm(db_sessionmaker, *, model: str = CHAT, prov: FakeProvider | None = None):
    return await think_check.confirm_think_mode(
        provider=prov if prov is not None else provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=tenant(),
        model=model,
    )


async def stored_rows(db_sessionmaker, *, model: str = CHAT) -> list[ModelPolicy]:
    async with db_sessionmaker() as session:
        return await ModelPolicyRepository(session).recent("fake", model)


# --- what gets written --------------------------------------------------------


async def test_a_confirmation_stores_the_measured_mode_bound_to_the_weights(
    db_sessionmaker,
) -> None:
    probe_id = await seed_probe(db_sessionmaker, think_mode="always")
    stored = await confirm(db_sessionmaker)

    assert (stored.model, stored.think_mode, stored.digest) == (CHAT, "always", DIGEST)
    assert stored.probe_id == probe_id
    rows = await stored_rows(db_sessionmaker)
    assert len(rows) == 1
    assert (rows[0].think_mode, rows[0].digest) == ("always", DIGEST)
    # Audit columns, constant today and recorded anyway: the trail becomes
    # meaningful the moment real identity populates it.
    assert rows[0].declared_by_tenant_id == DEFAULT_TENANT_ID
    assert rows[0].declared_by_user_id == DEFAULT_USER_ID
    assert str(rows[0].probe_id) == probe_id


async def test_the_row_declares_think_mode_and_nothing_else(db_sessionmaker) -> None:
    # Store rule 2, and the mutation guard for it. The obvious "merge on write"
    # - copying the previous row's values so the newest row is always complete -
    # would carry an agency ceiling declared against one set of weights forward
    # as a side effect of somebody confirming a think mode, silently re-arming
    # a grant a re-pull is supposed to retire. Copy any axis into the new row
    # and this goes red.
    async with db_sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(provider="fake", model=CHAT, digest=OTHER, ceiling="agentic", num_ctx=16384)
        )
        await session.commit()
    await seed_probe(db_sessionmaker)
    await confirm(db_sessionmaker)

    newest = (await stored_rows(db_sessionmaker))[0]
    assert newest.think_mode == "always"
    assert newest.ceiling is None
    assert newest.thinking_with_tools is None
    assert newest.num_ctx is None
    assert newest.thinking_model is None
    # And the fold still retires the old ceiling, because it stayed on the row
    # that declared it - against weights nobody holds any more.
    view = await policy_for(
        provider_name="fake", model=CHAT, digests={CHAT: DIGEST}, sessionmaker=db_sessionmaker
    )
    assert view is not None
    assert (view.think_mode, view.ceiling) == ("always", None)


async def test_the_declaration_is_visible_at_once(db_sessionmaker) -> None:
    # Store rule 1, and the mutation guard for it. Declarations are cached for
    # minutes; without the drop, a confirmation is a button that appears to do
    # nothing until a TTL lapses. The read below WARMS the cache first, so
    # removing clear_policy_cache() from the writer turns this red.
    digests = {CHAT: DIGEST}
    assert (
        await policy_for(
            provider_name="fake", model=CHAT, digests=digests, sessionmaker=db_sessionmaker
        )
        is None
    )
    await seed_probe(db_sessionmaker)
    await confirm(db_sessionmaker)

    view = await policy_for(
        provider_name="fake", model=CHAT, digests=digests, sessionmaker=db_sessionmaker
    )
    assert view is not None and view.think_mode == "always"


async def test_a_confirmation_survives_a_stale_installed_listing(db_sessionmaker) -> None:
    # Store rule 1's other half, and the scenario this writer's fresh listing
    # exists for. Weights are re-pulled OUTSIDE the app - nothing in-process
    # clears a cache for that - so the shared installed-digest cache still
    # holds the OLD digest. The confirmation binds to the digest the runtime
    # actually reports NOW, and must drop the identity cache with the policy
    # one: otherwise every reader gates the fresh row out against the stale
    # digest and the confirmation sits inert for up to five minutes.
    fresh = provider(digest=DIGEST)
    await resolve_policy(
        provider=provider(digest="sha256:oldweights"),
        provider_name="fake",
        model=CHAT,
        sessionmaker=db_sessionmaker,
    )  # warms the shared identity cache with the OLD digest

    await seed_probe(db_sessionmaker, digest=DIGEST, think_mode="always")
    stored = await confirm(db_sessionmaker, prov=fresh)
    assert stored.digest == DIGEST

    # The read a TURN makes, which resolves the digest through that shared
    # cache rather than carrying a listing of its own. Without the identity-cache
    # drop it still resolves the OLD digest, the fold gates the fresh row out,
    # and the confirmation is inert for as long as the cache holds.
    view = await resolve_policy(
        provider=fresh, provider_name="fake", model=CHAT, sessionmaker=db_sessionmaker
    )
    assert view is not None and view.think_mode == "always"


async def test_the_row_is_written_under_the_installed_spelling(db_sessionmaker) -> None:
    # One matcher answers both sides: a row written under the CONFIGURED
    # spelling of a tag the runtime lists differently is a declaration nobody
    # ever finds again.
    await seed_probe(db_sessionmaker, model="mistral", digest="d")
    stored = await confirm(
        db_sessionmaker,
        model="mistral",
        prov=provider(name="mistral:latest", digest="d"),
    )
    assert stored.model == "mistral:latest"
    assert await stored_rows(db_sessionmaker, model="mistral:latest") != []


# --- what gets refused --------------------------------------------------------


async def test_a_model_nothing_measured_is_refused(db_sessionmaker) -> None:
    # The rule this endpoint owns, and the mutation guard for it: confirming
    # without evidence is not a thing. Drop the `found is None` refusal and
    # this goes red - along with the whole meaning of the row.
    with pytest.raises(UnprocessableEntityError) as exc:
        await confirm(db_sessionmaker)
    assert "Run a check first" in str(exc.value)
    assert await stored_rows(db_sessionmaker) == []


async def test_a_measurement_on_other_weights_is_not_evidence(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, digest=OTHER)
    with pytest.raises(UnprocessableEntityError):
        await confirm(db_sessionmaker)
    assert await stored_rows(db_sessionmaker) == []


async def test_a_measurement_from_an_older_harness_is_not_evidence(db_sessionmaker) -> None:
    await seed_probe(db_sessionmaker, harness="h-old")
    with pytest.raises(UnprocessableEntityError):
        await confirm(db_sessionmaker)


async def test_a_stored_mode_this_build_does_not_know_never_becomes_a_declaration(
    db_sessionmaker,
) -> None:
    # Store rule 3, pinned as an OUTCOME rather than as a branch here. The
    # value comes out of a machine-written JSON column, and the guard that
    # drops an unrecognized one lives at the READ
    # (app.services.earned._think_mode_of, exercised by its own test), so this
    # writer satisfies the rule by construction: the mode it sees is one the
    # reader already vouched for. A second check here would be a branch no test
    # could reach - which is why the refusal below is the no-evidence one.
    await seed_probe(db_sessionmaker, think_mode="sometimes")
    with pytest.raises(UnprocessableEntityError) as exc:
        await confirm(db_sessionmaker)
    assert "Run a check first" in str(exc.value)
    assert await stored_rows(db_sessionmaker) == []


async def test_a_model_that_is_not_installed_is_a_404(db_sessionmaker) -> None:
    # A different sentence from "nothing has measured it", and worth the fresh
    # listing it costs: one sends a reader to run a check, the other to pull.
    await seed_probe(db_sessionmaker)
    with pytest.raises(NotFoundError):
        await confirm(db_sessionmaker, model="never-pulled")


# --- through the route --------------------------------------------------------


def make_settings(tmp_path: Path) -> Settings:
    return Settings(
        _env_file=None,
        model_provider="fake",
        models_yaml_path=str(tmp_path / "models.yaml"),
    )


@asynccontextmanager
async def client(prov: FakeProvider, tmp_path: Path, db_sessionmaker) -> AsyncIterator[AsyncClient]:
    settings = make_settings(tmp_path)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_provider] = lambda: prov
    app.dependency_overrides[get_current_tenant] = tenant
    app.state.db_sessionmaker = db_sessionmaker
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=30.0
    ) as http:
        yield http
    app.dependency_overrides.clear()


async def test_the_route_answers_with_what_it_stored(tmp_path, db_sessionmaker) -> None:
    probe_id = await seed_probe(db_sessionmaker, think_mode="hybrid")
    async with client(provider(), tmp_path, db_sessionmaker) as http:
        resp = await http.post("/api/v1/models/think-mode", json={"name": CHAT})

    assert resp.status_code == 200
    body = resp.json()
    assert (body["model"], body["think_mode"], body["digest"]) == (CHAT, "hybrid", DIGEST)
    assert body["probe_id"] == probe_id


async def test_the_route_refuses_an_unmeasured_model_as_problem_json(
    tmp_path, db_sessionmaker
) -> None:
    async with client(provider(), tmp_path, db_sessionmaker) as http:
        resp = await http.post("/api/v1/models/think-mode", json={"name": CHAT})

    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert "Run a check first" in resp.json()["detail"]


async def test_the_route_refuses_an_unusable_name(tmp_path, db_sessionmaker) -> None:
    async with client(provider(), tmp_path, db_sessionmaker) as http:
        resp = await http.post("/api/v1/models/think-mode", json={"name": "../etc/passwd"})

    assert resp.status_code == 422


async def test_a_member_cannot_confirm(tmp_path, db_sessionmaker) -> None:
    # The admin gate on the first write path into a store that changes routing.
    # Pre-OIDC every caller is the seeded admin, so this is the seam rather than
    # a live restriction - but the seam is what identity will make mean
    # something, and an ungated writer would be the one place it silently did
    # not. Refused before any work: nothing is written.
    member = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")
    await seed_probe(db_sessionmaker)
    settings = make_settings(tmp_path)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    app.dependency_overrides[get_model_provider] = lambda: provider()
    app.dependency_overrides[get_current_tenant] = lambda: member
    app.state.db_sessionmaker = db_sessionmaker
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test", timeout=30.0
    ) as http:
        resp = await http.post("/api/v1/models/think-mode", json={"name": CHAT})
    app.dependency_overrides.clear()

    assert resp.status_code == 403
    assert await stored_rows(db_sessionmaker) == []
