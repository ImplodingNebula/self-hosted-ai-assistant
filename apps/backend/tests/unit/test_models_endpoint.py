"""Tests for the /api/v1/models endpoints (listing, residency, and warmup).

The listing is an INVENTORY now: one entry per tag the model server actually
holds, described by what the runtime reports about it. There is no second list
of models a deployment might have downloaded but did not - a menu of things a
picker can only refuse - so what these tests set up is a SERVER, not a file,
and what they assert is that the rows describe it honestly.
"""

import asyncio
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_model_provider
from app.api.v1.models import _MAX_CONCURRENT_THINK_READS
from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.models.entities import ModelPolicy, ModelProbe
from app.providers.base import (
    ModelCapabilities,
    ModelInfo,
    ModelProvider,
    ModelResidency,
    ProviderUnavailableError,
)
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.providers.ollama_provider import OllamaProvider
from app.providers.search.searxng import FakeSearchProvider, SearXNGSearchProvider
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.schemas.settings import WorkspaceDefaultsPatch
from app.services.earned import clear_earned_cache
from app.services.model_facts import clear_facts_cache
from app.services.model_manager import build_manager_view
from app.services.model_policy import clear_policy_cache
from app.services.prompts import DEFAULT_SYSTEM_PROMPT
from app.services.tool_check import harness_version
from app.services.workspace_defaults import apply_defaults
from tests.conftest import make_memory_store

# The seeded single-tenant principal, for tests that stage workspace settings.
TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")

# The MODEL_PROVIDER value every store keys on. `models_client` leaves the
# setting at its default while overriding the provider OBJECT, so a seeded row
# has to key on this string or it describes a backend nothing is asking about.
PROVIDER_NAME = "ollama"

# The window a warm asks for when nothing else sizes the tag. Deliberately not
# the shipped default, so a request carrying the deployment's own affordance is
# distinguishable from one carrying a constant.
WARM_WINDOW = 12288

# The pair the warm tests configure. Both tags are in the code-shipped quirk
# seed - the instruct checkpoint never reasons, its sibling always does - which
# is what makes the think flag on a warm assertable without declaring anything.
WARM_CHAT = "qwen3:4b-instruct"
WARM_THINK = "qwen3:4b"


class _UnlistableProvider(FakeProvider):
    """A fake whose inventory cannot be read, but which generates fine.

    The shape a warm has to degrade gracefully against: the model server is
    answering generations while its listing endpoint is unreachable (a proxy
    that fronts only the generate route, a listing that timed out).

    What tells "the resolver was skipped" apart from "the resolver ran and
    found nothing" is the NAME that reaches the provider, not this fake's
    inventory - a raise leaves no listing to match against either way. A test
    therefore asserts the typed name arrived verbatim; resolving against an
    empty listing would 404 instead.
    """

    async def list_models(self) -> list[ModelInfo]:
        """Fail the way an unreachable listing endpoint does."""
        raise ProviderUnavailableError("Cannot reach the model server.")


@pytest.fixture(autouse=True)
def _fresh_caches():
    # Runtime capability reports are cached per (provider name, tag) for five
    # minutes, and every test here reuses a handful of tag names against a
    # differently-scripted provider: one leaked report would answer the next
    # test's probe.
    clear_facts_cache()
    # The declaration and probe reads are cached per model too, and both now
    # answer this endpoint - a row seeded by one test would describe the next
    # test's identically-named tag.
    clear_policy_cache()
    clear_earned_cache()
    yield
    clear_facts_cache()
    clear_policy_cache()
    clear_earned_cache()


@asynccontextmanager
async def models_client(
    provider: ModelProvider,
    db_sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    settings_overrides: dict[str, object] | None = None,
):
    """App-wired client over a scripted model server.

    Endpoints that read the database (the warm endpoint loads the workspace
    default) need ``db_sessionmaker`` wired onto ``app.state``, or the app
    falls back to a real postgres connection. The vector store is wired for
    the same reason: the lazy fallback builds a client against the configured
    Qdrant URL, which a unit test has no business pointing at.

    No embedder and no search provider are wired here, so the warm endpoint
    has no tools to advertise and primes the persona alone - which is what
    every test below is about. The prefix these collaborators produce is the
    subject of tests/unit/test_warm_prefix.py instead.
    """
    settings = Settings(_env_file=None, **(settings_overrides or {}))
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.vector_store = make_memory_store()
    if db_sessionmaker is not None:
        app.state.db_sessionmaker = db_sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


def warm_client(provider: ModelProvider, db_sessionmaker, **overrides):
    """A client configured with the warm tests' chat pair and window."""
    return models_client(
        provider,
        db_sessionmaker=db_sessionmaker,
        settings_overrides={
            "default_chat_model": WARM_CHAT,
            "thinking_chat_model": WARM_THINK,
            "default_context_window": WARM_WINDOW,
            **overrides,
        },
    )


# --- GET /api/v1/models ---


async def test_list_models_happy_path(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    # The "models" key shape is a frontend contract - it must stay untouched.
    assert data["models"]
    assert data["models"][0]["name"] == "fake"
    assert data["models"][0]["capabilities"]["tools"] is True
    # And one described row per held tag rides alongside it.
    assert [entry["model"] for entry in data["entries"]] == ["fake"]


# What a remote pod holds, and what the runtime says about each tag: two chat
# models, as the motivating remote inventory actually looked.
POD_INSTALLED = [
    ModelInfo(name="llama3.1:8b"),
    ModelInfo(name="llama3.2:3b"),
]
POD_REPORTS = {
    "llama3.1:8b": ModelCapabilities(completion=True, tools=True, context_length=131072),
    "llama3.2:3b": ModelCapabilities(completion=True, tools=True, context_length=131072),
}


class ProbeFailingProvider(FakeProvider):
    """A fake whose capability probe fails for named tags, and answers otherwise."""

    def __init__(self, *, failing: set[str], **kwargs) -> None:
        super().__init__(**kwargs)
        self._failing = failing

    async def capabilities(self, model: str) -> ModelCapabilities:
        """Fail for a scripted tag; a probe that cannot be made is not an answer."""
        if model in self._failing:
            raise ProviderUnavailableError("the model server dropped the probe")
        return await super().capabilities(model)


async def test_the_entries_are_one_row_per_held_tag_in_provider_order() -> None:
    # The motivating deployment: the model server holds whatever it holds, and
    # nothing anywhere describes it. Before the inventory became the list, the
    # picker offered models that were not there beside a server full of models
    # it never mentioned.
    #
    # Provider ORDER, and every tag: a pair's thinking sibling is a row like any
    # other now. It used to be withheld so a conversation could not pin it, and
    # withholding it is what made a remote server's inventory unpickable.
    provider = FakeProvider(models=POD_INSTALLED, capabilities_by_model=POD_REPORTS)
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    entries = resp.json()["entries"]
    assert [(e["model"], e["role"]) for e in entries] == [
        ("llama3.1:8b", "chat"),
        ("llama3.2:3b", "chat"),
    ]
    # Exactly once each: there is one source now, so a duplicate could only be
    # a row this endpoint invented.
    assert len({e["model"] for e in entries}) == len(entries)


async def test_a_row_carries_the_mechanical_report_and_no_policy() -> None:
    # The line the fill stops at. A MECHANICAL fact is something the serving
    # stack knows about itself and is entitled to report; POLICY - how much a
    # model may drive - is a human decision a runtime cannot make on anyone's
    # behalf. So no `agency` key is ever written here, and a tag nobody has
    # declared is described honestly while still running at the safe floor.
    provider = FakeProvider(models=POD_INSTALLED, capabilities_by_model=POD_REPORTS)
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    row = resp.json()["entries"][-1]
    # ``thinking: False`` is the code-shipped seed's: the llama3.2 family
    # carries no think plumbing in Ollama's templates, which is a measured
    # family fact rather than anything this deployment declared.
    assert row["capabilities"] == {"tools": "native", "vision": False, "thinking": False}
    assert "agency" not in row["capabilities"]
    assert row["provider"] == PROVIDER_NAME


async def test_an_installed_tag_that_only_embeds_is_rowed_as_an_embedder() -> None:
    # Role from the report, so an embedder never lands in a chat picker.
    provider = FakeProvider(
        models=[ModelInfo(name="nomic-embed-text:v1.5")],
        capabilities_by_model={
            "nomic-embed-text:v1.5": ModelCapabilities(embedding=True),
        },
    )
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    row = resp.json()["entries"][-1]
    assert (row["model"], row["role"]) == ("nomic-embed-text:v1.5", "embedding")


async def test_a_tag_whose_probe_fails_is_left_out_of_the_entries() -> None:
    # A failed probe is unknown, never a report of no capabilities: rowing it
    # role-less would present an unreachable moment as a fact about the
    # weights. It stays in the live list, and the next fetch tries again.
    provider = ProbeFailingProvider(
        failing={"llama3.2:3b"}, models=POD_INSTALLED, capabilities_by_model=POD_REPORTS
    )
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    data = resp.json()
    assert [e["model"] for e in data["entries"]] == ["llama3.1:8b"]
    assert "llama3.2:3b" in {m["name"] for m in data["models"]}


async def test_a_dual_capability_report_rows_as_chat() -> None:
    # No runtime reports completion and embedding together today, but the
    # tie-break decides whether such a tag would reach the chat picker under a
    # future provider - so the precedence is a recorded choice, not an
    # accident of expression order.
    provider = FakeProvider(
        models=[ModelInfo(name="dual:7b")],
        capabilities_by_model={"dual:7b": ModelCapabilities(completion=True, embedding=True)},
    )
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    row = resp.json()["entries"][-1]
    assert (row["model"], row["role"]) == ("dual:7b", "chat")


async def test_the_capability_probes_run_concurrently() -> None:
    # The whole cold-cache latency story rests on the probes overlapping: a
    # dozen held tags must cost one round trip of wall time, not twelve. A
    # serialized gather keeps every other assertion green, so the overlap
    # itself is pinned - by observed concurrency, not wall clocks.
    class OverlapCountingProvider(FakeProvider):
        def __init__(self, **kwargs) -> None:
            super().__init__(**kwargs)
            self.in_flight = 0
            self.peak = 0

        async def capabilities(self, model: str) -> ModelCapabilities:
            self.in_flight += 1
            self.peak = max(self.peak, self.in_flight)
            try:
                await asyncio.sleep(0.02)
                return await super().capabilities(model)
            finally:
                self.in_flight -= 1

    provider = OverlapCountingProvider(models=POD_INSTALLED, capabilities_by_model=POD_REPORTS)
    async with models_client(provider) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    assert provider.peak > 1


async def test_list_models_provider_down_returns_503() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models")
    app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["status"] == 503
    assert body["code"] == "service_unavailable"
    assert "model server" in body["detail"].lower()
    # This endpoint IS the inventory: no partial payload on 503, or an empty
    # list would read as a server holding nothing.
    assert "entries" not in body


@pytest.mark.parametrize(
    ("url", "headers", "remote", "auth_configured", "host"),
    [
        ("http://ollama:11434", {}, False, False, "ollama"),
        ("http://localhost:11434", {"Authorization": "Bearer changeme"}, False, True, "localhost"),
        ("https://pod-11434.proxy.example.net", {}, True, False, "pod-11434.proxy.example.net"),
        (
            "https://pod-11434.proxy.example.net/",
            {"Authorization": "Bearer changeme"},
            True,
            True,
            "pod-11434.proxy.example.net",
        ),
    ],
)
async def test_models_payload_says_where_the_model_server_is(
    url: str,
    headers: dict[str, str],
    remote: bool,
    auth_configured: bool,
    host: str,
) -> None:
    # Read from configuration alone, so it is as true under this fake provider
    # as under a live one - and it is what lets a client warn that a model
    # server outside the deployment is answering without a credential.
    settings = Settings(ollama_url=url, ollama_auth_headers=headers, _env_file=None)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_settings] = lambda: settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["server"] == {
        "remote": remote,
        "auth_configured": auth_configured,
        "host": host,
    }
    # The credential itself is never disclosed - only that one is set.
    assert "changeme" not in resp.text


async def test_models_payload_never_discloses_a_credential_in_the_url() -> None:
    # A URL can carry a password in its userinfo, and this payload is served
    # to a browser: the hostname alone survives the parse.
    settings = Settings(ollama_url="https://user:hunter2@gpu.example.net:11434", _env_file=None)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider()
    app.dependency_overrides[get_settings] = lambda: settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json()["server"]["host"] == "gpu.example.net"
    assert "hunter2" not in resp.text


async def test_list_models_unauthorized_names_the_setting_that_fixes_it() -> None:
    # A refused credential is still a 503, but the detail must carry the fix
    # rather than the generic "cannot reach" line, which would send somebody to
    # check a model server that is running perfectly.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(401, json={"error": "unauthorized"})

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models")
    app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert "OLLAMA_AUTH_HEADERS" in body["detail"]


async def test_list_models_redirect_names_the_address_to_configure() -> None:
    # The everyday failure of a model server behind a proxy that upgrades
    # plain HTTP: the address it points at is the whole fix.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            301, headers={"location": "https://pod-11434.proxy.example.net/api/tags"}
        )

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models")
    app.dependency_overrides.clear()

    assert resp.status_code == 503
    detail = resp.json()["detail"]
    assert "https://pod-11434.proxy.example.net/api/tags" in detail
    assert "OLLAMA_URL" in detail


# --- GET /api/v1/models/manager ---


async def test_manager_view_happy_path(client: AsyncClient) -> None:
    resp = await client.get("/api/v1/models/manager")
    assert resp.status_code == 200
    data = resp.json()
    # The provider block reports which verbs exist at all, so a UI never
    # renders a button the backend could not honour.
    assert set(data["provider"]) >= {"name", "can_pull", "can_delete", "can_list_installed"}
    assert isinstance(data["groups"], list)
    assert isinstance(data["role_gaps"], list)
    for group in data["groups"]:
        assert group["management"] in {"manageable", "locked", "status_only"}
        for row in group["rows"]:
            assert row["model"]
            # Every row is a tag the server holds: there is no second list of
            # models nobody downloaded, and no row that could say otherwise.
            assert row["installed"] is True


async def test_manager_view_provider_down_returns_503() -> None:
    # The view IS the inventory: a page rendered entirely "not installed" would
    # invite re-pulling models that are already here.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models/manager")
    app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["code"] == "service_unavailable"
    assert "groups" not in body


async def test_manager_view_leaves_the_models_payload_untouched(client: AsyncClient) -> None:
    # /models is a frozen frontend contract; the management view is a sibling
    # endpoint, never an extension of it. ("server" is the model server's own
    # address and credential state, which belongs to the selector payload -
    # it is not the management view leaking into it.)
    data = (await client.get("/api/v1/models")).json()
    assert set(data) == {"models", "entries", "server"}


# --- GET /api/v1/models/residency ---


async def test_residency_reports_what_the_runtime_holds() -> None:
    provider = FakeProvider(
        resident_models=[
            ModelResidency(model="fake", loaded=True, total_bytes=4_000, vram_bytes=1_600),
            ModelResidency(model="stray:1b", loaded=True, total_bytes=1_000, vram_bytes=0),
        ]
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models/residency")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    data = resp.json()
    assert data["reported"] is True
    assert [m["model"] for m in data["models"]] == ["fake", "stray:1b"]
    assert data["models"][0]["gpu_fraction"] == 0.4
    # A full CPU spill is a real reading of zero, so the fraction is 0.0 and
    # never null - the two mean different things to whatever renders them.
    assert data["models"][1]["gpu_fraction"] == 0.0


async def test_residency_unreported_is_not_the_same_as_nothing_loaded(
    client: AsyncClient,
) -> None:
    # The default fake reports no residency at all, which is what a runtime
    # that loads one model at process start looks like. The flag says so; the
    # empty list must not be read as an answer on its own.
    resp = await client.get("/api/v1/models/residency")
    assert resp.status_code == 200
    assert resp.json() == {"reported": False, "models": []}


async def test_residency_holding_nothing_is_a_reported_answer() -> None:
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: FakeProvider(resident_models=[])
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.get("/api/v1/models/residency")
    app.dependency_overrides.clear()

    assert resp.status_code == 200
    assert resp.json() == {"reported": True, "models": []}


# --- POST /api/v1/models/warm ---


async def test_warm_defaults_to_configured_chat_model(db_sessionmaker) -> None:
    # The server holds the configured default, so warm primes exactly it.
    provider = FakeProvider(models=[ModelInfo(name=WARM_CHAT)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    data = resp.json()
    # The warm default mirrors the chat path's resolution (conversation pick >
    # workspace default > DEFAULT_CHAT_MODEL), not a raw settings field.
    assert data["model"] == WARM_CHAT
    assert isinstance(data["took_ms"], int)
    # The warm turn mirrors a real one: system prompt first, then a user turn,
    # capped to a single generated token.
    assert provider.last_messages[0].role == "system"
    assert provider.last_messages[0].content == DEFAULT_SYSTEM_PROMPT
    assert provider.last_messages[-1].role == "user"
    assert provider.last_options.max_tokens == 1
    # Think normalization mirrors the chat path: the quirk seed knows this
    # instruct checkpoint never reasons, so the flag is omitted - the primed
    # template prefix must match what real turns send.
    assert provider.last_options.think is None


async def test_warm_prefers_the_stored_workspace_default(db_sessionmaker) -> None:
    # Two installed tags, neither of them the configured default: substitution
    # alone would land on the FIRST in listing order, and ignoring the store
    # would land on the environment's tag - only the stored-chain read warms
    # "beta". The pick is honoured through the provider listing, the
    # remote-server shape where the workspace default names a tag the
    # deployment's own configuration never mentions.
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults({"model": "beta:1b"})
        await session.commit()
    provider = FakeProvider(models=[ModelInfo(name="alpha:1b"), ModelInfo(name="beta:1b")])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    assert resp.json()["model"] == "beta:1b"
    assert provider.last_model == "beta:1b"


async def test_warm_blank_model_is_treated_as_unset(db_sessionmaker) -> None:
    # A blank body model is not an explicit pick: it must fall through to the
    # stored chain exactly like an omitted one (every sibling path treats
    # blank as unset), never reach the provider verbatim as a model tag.
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults({"model": "beta:1b"})
        await session.commit()
    provider = FakeProvider(models=[ModelInfo(name="alpha:1b"), ModelInfo(name="beta:1b")])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "  "})
    assert resp.status_code == 200
    assert resp.json()["model"] == "beta:1b"
    assert provider.last_model == "beta:1b"


async def test_warm_routes_around_a_missing_default_like_a_turn(db_sessionmaker) -> None:
    # The remote-inventory case: the configured tag is not on the server, so
    # warm primes what a default turn would actually run - the substitution's
    # own choice - instead of refusing on a tag no turn would use.
    provider = FakeProvider(models=POD_INSTALLED, capabilities_by_model=POD_REPORTS)
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    assert resp.json()["model"] == "llama3.1:8b"
    assert provider.last_model == "llama3.1:8b"


async def test_warm_omits_the_flag_for_a_never_thinking_template(db_sessionmaker) -> None:
    # The warm's whole job is to prime the prefix a real turn will send, so its
    # think normalization has to read the same source the turn's does. The turn
    # resolves an undescribed tag's think mode from the runtime's TEMPLATE
    # report; a warm that could not reach it would call the tag "unknown", send
    # `think: false`, and buy a 400 plus a provider retry on every warm - for a
    # model the turn sends no flag to at all.
    provider = FakeProvider(
        models=[ModelInfo(name="stray:7b")],
        capabilities_by_model={
            "stray:7b": ModelCapabilities(completion=True, template_thinking=False)
        },
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    assert provider.last_model == "stray:7b"
    assert provider.last_options.think is None


async def test_warm_chatless_server_maps_to_404(db_sessionmaker) -> None:
    # An inventory that is KNOWN and holds nothing that can chat surfaces the
    # turn's typed refusal as a 404 carrying the same message.
    provider = FakeProvider(
        models=[ModelInfo(name="embed-only:1b")],
        capabilities_by_model={"embed-only:1b": ModelCapabilities(embedding=True)},
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert "no model this deployment can chat with" in resp.json()["detail"]


async def test_warm_always_thinking_model_gets_think_true(db_sessionmaker) -> None:
    # Warming the thinking pair member must prime the thinking-enabled
    # template state, exactly like a real thinking turn. The quirk seed is what
    # says these weights always reason.
    provider = FakeProvider(models=[ModelInfo(name=WARM_THINK)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": WARM_THINK})
    assert resp.status_code == 200
    assert provider.last_options.think is True


async def test_warm_accepts_missing_body(client: AsyncClient) -> None:
    resp = await client.post("/api/v1/models/warm")
    assert resp.status_code == 200


async def test_warm_num_ctx_matches_chat_turns(db_sessionmaker) -> None:
    # Parity is load-bearing: the model server restarts the runner when the
    # context size changes, which would void the cache this warm primed. These
    # weights report no native window, so what a turn would ask for is what
    # this deployment affords - and the value here is not the shipped default,
    # so a warm carrying a constant would show up.
    provider = FakeProvider(models=[ModelInfo(name=WARM_CHAT)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm")
    assert resp.status_code == 200
    assert provider.last_options.num_ctx == WARM_WINDOW


async def test_warm_thinking_pair_member_shares_num_ctx(db_sessionmaker) -> None:
    provider = FakeProvider(models=[ModelInfo(name=WARM_THINK)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": WARM_THINK})
    assert resp.status_code == 200
    assert provider.last_options.num_ctx == WARM_WINDOW


async def test_a_native_window_smaller_than_the_affordance_is_what_warms(db_sessionmaker) -> None:
    # The other half of the same rule, and the one only the runtime can supply:
    # nothing may ask a model for more window than its weights were trained
    # for, so the smaller of the two is what a turn requests and what a warm
    # primes.
    provider = FakeProvider(
        models=[ModelInfo(name=WARM_CHAT)],
        capabilities_by_model={WARM_CHAT: ModelCapabilities(completion=True, context_length=4096)},
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm")
    assert resp.status_code == 200
    assert provider.last_options.num_ctx == 4096


async def test_warm_passes_model_and_keep_alive_through(db_sessionmaker) -> None:
    provider = FakeProvider(models=[ModelInfo(name="custom:1b")])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post(
            "/api/v1/models/warm", json={"model": "custom:1b", "keep_alive": "5m"}
        )
    assert resp.status_code == 200
    assert resp.json()["model"] == "custom:1b"
    assert provider.last_model == "custom:1b"
    assert provider.last_options.keep_alive == "5m"


async def test_warm_coerces_numeric_keep_alive_string(
    client: AsyncClient, fake_provider: FakeProvider
) -> None:
    # Shell callers (make warm) can only send quoted values, but Ollama's
    # duration parser rejects a unit-less string ("-1") while accepting the
    # number -1 as "keep loaded indefinitely" - the boundary must coerce.
    resp = await client.post("/api/v1/models/warm", json={"keep_alive": "-1"})
    assert resp.status_code == 200
    assert fake_provider.last_options.keep_alive == -1


async def test_warm_resolves_a_partial_name_to_the_one_installed_match(db_sessionmaker) -> None:
    # `make warm CHAT_MODEL=instruct` warms the obvious model instead of
    # bouncing off a 404 for a spelling, and the response names what it warmed
    # so nothing about the substitution is silent.
    provider = FakeProvider(
        models=[ModelInfo(name=WARM_CHAT), ModelInfo(name="qwen3-embedding:0.6b")]
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "instruct"})
    assert resp.status_code == 200
    assert resp.json()["model"] == WARM_CHAT
    assert provider.last_model == WARM_CHAT


async def test_warm_exact_name_wins_over_the_longer_tags_containing_it(db_sessionmaker) -> None:
    # "qwen3:4b" is a substring of "qwen3:4b-instruct" as well as being
    # installed in its own right. Reading it as a fragment would make naming a
    # model exactly the one way to get an ambiguity refusal.
    provider = FakeProvider(models=[ModelInfo(name=WARM_CHAT), ModelInfo(name=WARM_THINK)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": WARM_THINK})
    assert resp.status_code == 200
    assert resp.json()["model"] == WARM_THINK


async def test_warm_ambiguous_name_refuses_with_the_candidates(db_sessionmaker) -> None:
    # Two tags fit "qwen3", and every rule for picking one of them (shortest,
    # first alphabetically, first the server listed) is arbitrary. The refusal
    # has to carry the candidates or the person cannot act on it.
    provider = FakeProvider(models=[ModelInfo(name=WARM_CHAT), ModelInfo(name="qwen3:8b")])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "qwen3"})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/problem+json")
    body = resp.json()
    assert body["code"] == "bad_request"
    assert "'qwen3'" in body["detail"]
    assert WARM_CHAT in body["detail"]
    assert "qwen3:8b" in body["detail"]
    # Never guessed: an ambiguous name reaches no generation at all.
    assert provider.last_model is None


async def test_warm_unknown_name_is_a_404_naming_the_input(db_sessionmaker) -> None:
    provider = FakeProvider(models=[ModelInfo(name=WARM_CHAT)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "llama"})
    assert resp.status_code == 404
    assert resp.json()["code"] == "not_found"
    assert "'llama'" in resp.json()["detail"]
    assert provider.last_model is None


async def test_warm_unlisted_inventory_passes_the_name_through(db_sessionmaker) -> None:
    # The degradation rule: a model server that cannot be listed knows nothing
    # about what is installed, so resolution is skipped entirely and the name
    # goes through as it always did. Turning a valid explicit warm into a
    # resolution error because the inventory endpoint is down would blame the
    # name for the server's problem.
    provider = _UnlistableProvider(models=[ModelInfo(name=WARM_CHAT)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "qwen"})
    assert resp.status_code == 200
    assert resp.json()["model"] == "qwen"
    assert provider.last_model == "qwen"


async def test_warm_fragment_landing_on_a_non_chat_model_refuses(db_sessionmaker) -> None:
    # `make warm CHAT_MODEL=0.6b` fits exactly one installed tag - the
    # embedder. Resolving it and generating anyway would send a chat turn to a
    # model that cannot answer one, and on a card that fits a single model it
    # would evict the chat model the person was using to do it. The refusal
    # names BOTH sides so the mis-inference is visible, not just the input.
    provider = FakeProvider(
        models=[ModelInfo(name=WARM_CHAT), ModelInfo(name="qwen3-embedding:0.6b")],
        capabilities_by_model={"qwen3-embedding:0.6b": ModelCapabilities(embedding=True)},
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "0.6b"})
    assert resp.status_code == 400
    assert resp.headers["content-type"].startswith("application/problem+json")
    detail = resp.json()["detail"]
    assert "'0.6b'" in detail
    assert "qwen3-embedding:0.6b" in detail
    # The whole point: the embedder is never loaded to discover this.
    assert provider.last_model is None


async def test_warm_exact_non_chat_tag_is_still_honoured(db_sessionmaker) -> None:
    # The deliberate asymmetry. Spelling the embedder's tag out in full is an
    # explicit request, not something this platform inferred, so it goes
    # through and the model server gives its own verdict. Guarding it here
    # would overrule a person who meant exactly what they typed.
    provider = FakeProvider(
        models=[ModelInfo(name=WARM_CHAT), ModelInfo(name="qwen3-embedding:0.6b")],
        capabilities_by_model={"qwen3-embedding:0.6b": ModelCapabilities(embedding=True)},
    )
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "qwen3-embedding:0.6b"})
    assert resp.status_code == 200
    assert provider.last_model == "qwen3-embedding:0.6b"


async def test_warm_fragment_is_allowed_when_the_capability_probe_fails(db_sessionmaker) -> None:
    # A probe that did not ANSWER is unknown, never evidence - the rule the
    # configured-routing rescue applies to the same question. Refusing here
    # would turn a flaky describe endpoint into "that model cannot chat",
    # which sends someone off to fix the one thing that was fine.
    class _UnprobeableProvider(FakeProvider):
        async def capabilities(self, model: str) -> ModelCapabilities:
            raise ProviderUnavailableError("Cannot reach the model server.")

    provider = _UnprobeableProvider(models=[ModelInfo(name=WARM_CHAT)])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "instruct"})
    assert resp.status_code == 200
    assert resp.json()["model"] == WARM_CHAT
    assert provider.last_model == WARM_CHAT


async def test_warm_stored_default_is_never_fuzzy_resolved(db_sessionmaker) -> None:
    # The stored default is CONFIGURATION - it governs every turn, not just
    # this call - so it keeps its own exact-match-then-substitute chain. If the
    # typed-name resolver reached it, "beta" would resolve to "beta-large:9b"
    # here while every real turn kept running the substitution's pick, and the
    # warm would prime weights no turn uses.
    async with db_sessionmaker() as session:
        await WorkspaceSettingsRepository(session, TENANT).merge_defaults({"model": "beta"})
        await session.commit()
    provider = FakeProvider(models=[ModelInfo(name="alpha:1b"), ModelInfo(name="beta-large:9b")])
    async with warm_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    assert resp.status_code == 200
    # The substitution's own choice (first in listing order), not the tag whose
    # name happens to contain the stored one.
    assert resp.json()["model"] == "alpha:1b"
    assert provider.last_model == "alpha:1b"


async def test_warm_missing_model_maps_to_404(db_sessionmaker) -> None:
    # The PROVIDER's refusal, not the name resolver's: the tag is listed as
    # installed and vanishes before the generation, which is what losing a race
    # with a delete looks like. Seeding the listing is what keeps this test
    # driving the error-chunk path - without it the resolver answers 404 first
    # and the mapping under test never runs.
    provider = FakeProvider(
        script=[],
        models=[ModelInfo(name="ghost:1b")],
        error="Model 'ghost:1b' is not available.",
        error_code="model_not_found",
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.state.db_sessionmaker = db_sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={"model": "ghost:1b"})
    app.dependency_overrides.clear()

    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "not_found"


async def test_warm_provider_rejection_maps_to_400(db_sessionmaker) -> None:
    provider = FakeProvider(
        script=[], error="The model server rejected the request.", error_code="bad_request"
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.state.db_sessionmaker = db_sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    app.dependency_overrides.clear()

    assert resp.status_code == 400
    assert resp.json()["code"] == "bad_request"


async def test_warm_provider_down_maps_to_503(db_sessionmaker) -> None:
    provider = FakeProvider(
        script=[], error="Cannot reach the model server.", error_code="provider_unavailable"
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.state.db_sessionmaker = db_sessionmaker
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm", json={})
    app.dependency_overrides.clear()

    assert resp.status_code == 503
    assert resp.json()["code"] == "service_unavailable"


# --- POST /api/v1/models/warm-embed ---

EMBED_TAG = "test-embed:1b"


def embed_server(**kwargs) -> FakeProvider:
    """A server holding exactly one tag the runtime says can embed.

    Exactly one, because that is what the resolution chain's inventory rung
    will settle on: with several it refuses to guess, and the choice becomes a
    workspace setting.
    """
    kwargs.setdefault("models", [ModelInfo(name=EMBED_TAG)])
    kwargs["capabilities_by_model"] = {
        EMBED_TAG: ModelCapabilities(embedding=True),
        **(kwargs.get("capabilities_by_model") or {}),
    }
    return FakeProvider(**kwargs)


async def test_warm_embed_uses_the_resolved_model_and_device(db_sessionmaker) -> None:
    # The embedding model comes from the resolution chain (it defines the
    # vector space), never a caller override; its placement comes from
    # EMBEDDING_DEVICE; keep_alive rides through to the provider.
    provider = embed_server()
    async with models_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed", json={"keep_alive": "5m"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["model"] == EMBED_TAG
    assert isinstance(data["took_ms"], int)
    # An embedder has no prompt-prefix cache, so nothing is reported here.
    assert data["prompt_tokens"] is None
    assert provider.last_embed_options is not None
    assert provider.last_embed_options.device == "cpu"
    assert provider.last_embed_options.keep_alive == "5m"


async def test_warm_embed_warms_the_workspace_choice(db_sessionmaker) -> None:
    # What gets primed has to be what the next retrieval will run, or the warm
    # loads one model and the first query loads another - on a card that fits
    # one, that is the cold load this endpoint exists to avoid, paid twice. The
    # stored choice is the TOP rung, so it wins over the inventory below it.
    async with db_sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model="chosen:2b"))
        await session.commit()
    async with models_client(embed_server(), db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed")
    assert resp.status_code == 200
    assert resp.json()["model"] == "chosen:2b"


async def test_warm_embed_honours_the_embedding_device_setting(db_sessionmaker) -> None:
    # Placement is deployment configuration, so a box with card to spare gets
    # "auto" whatever else is true of the model.
    provider = embed_server()
    async with models_client(provider, db_sessionmaker, {"embedding_device": "auto"}) as client:
        resp = await client.post("/api/v1/models/warm-embed")
    assert resp.status_code == 200
    assert provider.last_embed_options is not None
    assert provider.last_embed_options.device == "auto"


async def test_warm_embed_ignores_body_model(db_sessionmaker) -> None:
    # `model` in the body must NOT redirect the vector space: resolution wins.
    async with models_client(embed_server(), db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed", json={"model": "other:9b"})
    assert resp.status_code == 200
    assert resp.json()["model"] == EMBED_TAG


async def test_warm_embed_coerces_numeric_keep_alive(db_sessionmaker) -> None:
    provider = embed_server()
    async with models_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed", json={"keep_alive": "-1"})
    assert resp.status_code == 200
    assert provider.last_embed_options is not None
    assert provider.last_embed_options.keep_alive == -1


async def test_warm_embed_accepts_missing_body(db_sessionmaker) -> None:
    async with models_client(embed_server(), db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed")
    assert resp.status_code == 200


async def test_warm_embed_no_embedding_resolvable_maps_to_404(db_sessionmaker) -> None:
    # Nothing stored, nothing in the corpus, and no tag the runtime says can
    # embed - so there is no vector space to warm.
    provider = FakeProvider(
        models=[ModelInfo(name="chat-only:1b")],
        capabilities_by_model={"chat-only:1b": ModelCapabilities(completion=True)},
    )
    async with models_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed")
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "not_found"
    # The refusal points at the surface that can fix it, not at a tier or a
    # file - neither of which is where the answer lives.
    detail = resp.json()["detail"]
    assert "workspace defaults" in detail
    assert "MODEL_TIER" not in detail


async def test_warm_embed_provider_down_maps_to_503(db_sessionmaker) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = OllamaProvider(
        base_url="http://mock:11434",
        default_model="m",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    async with db_sessionmaker() as session:
        await apply_defaults(session, TENANT, WorkspaceDefaultsPatch(embedding_model=EMBED_TAG))
        await session.commit()
    async with models_client(provider, db_sessionmaker) as client:
        resp = await client.post("/api/v1/models/warm-embed")
    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "service_unavailable"


# --- POST /api/v1/models/warm-search ---


async def test_warm_search_primes_backend() -> None:
    app = create_app()
    # Injected provider (tests) is used exactly as web-search chat turns do.
    app.state.search_provider = FakeSearchProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm-search")
    assert resp.status_code == 200
    data = resp.json()
    assert isinstance(data["took_ms"], int)
    # The fake synthesizes one result for the single-result warm query.
    assert data["results"] == 1


async def test_warm_search_unconfigured_maps_to_404() -> None:
    # No injected provider and SEARXNG_URL forced unset -> web search is off, so
    # there is nothing to prime. Settings are overridden so the test does not
    # depend on the ambient env (the live stack sets SEARXNG_URL).
    settings = Settings(searxng_url=None, _env_file=None)
    app = create_app()
    app.dependency_overrides[get_settings] = lambda: settings
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm-search")
    app.dependency_overrides.clear()
    assert resp.status_code == 404
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "not_found"


async def test_warm_search_backend_down_maps_to_503() -> None:
    # A transport failure survives the provider's one retry and surfaces as 503.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    provider = SearXNGSearchProvider(
        "http://searxng:8080",
        retry_backoff=0.0,
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    app = create_app()
    app.state.search_provider = provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        resp = await http_client.post("/api/v1/models/warm-search")
    assert resp.status_code == 503
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "service_unavailable"


# --- the composer list and the management card must agree ---------------------

# One installed chat tag whose template carries no think plumbing, so the
# resolution without any stored or measured answer is a definite "never" - and
# a stored or measured one has something to visibly disagree with.
AGREEMENT_TAG = "chat:4b"
AGREEMENT_INSTALLED = [ModelInfo(name=AGREEMENT_TAG, size=10, digest="sha256:chatweights")]
AGREEMENT_REPORTS = {
    AGREEMENT_TAG: ModelCapabilities(completion=True, tools=True, template_thinking=False)
}


def agreement_provider(**kwargs) -> FakeProvider:
    """The one-chat-model server both surfaces are asked to describe."""
    kwargs.setdefault("models", list(AGREEMENT_INSTALLED))
    kwargs["capabilities_by_model"] = {
        **AGREEMENT_REPORTS,
        **(kwargs.get("capabilities_by_model") or {}),
    }
    return FakeProvider(**kwargs)


def _entry_thinking(payload: dict, model: str) -> object:
    """The ``thinking`` capability the composer list carries for one tag."""
    row = next(entry for entry in payload["entries"] if entry["model"] == model)
    return row["capabilities"].get("thinking")


async def _card_think(provider, db_sessionmaker) -> str:
    """The think mode the MANAGEMENT view resolves for the same tag."""
    view = await build_manager_view(
        provider=provider,
        provider_name=PROVIDER_NAME,
        settings=Settings(_env_file=None),
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        store=None,
    )
    return next(
        row.think for group in view.groups for row in group.rows if row.model == AGREEMENT_TAG
    )


async def test_the_two_surfaces_agree_before_anything_is_declared(db_sessionmaker) -> None:
    # The control that gives the two tests below something to move: with
    # nothing stored and nothing measured, both surfaces answer from the
    # runtime's template report alone, and they answer the same thing.
    provider = agreement_provider()
    async with models_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    assert _entry_thinking(resp.json(), AGREEMENT_TAG) is False
    assert await _card_think(provider, db_sessionmaker) == "never"


async def test_a_measured_mode_reaches_the_composer_list(db_sessionmaker) -> None:
    # The whole reason this endpoint gained a database read. The composer picks
    # a model from THIS list and the management card describes the same model
    # from the full chain; a list that answered from the report alone would say
    # "never" beside a card saying "always" - one tag, two screens.
    provider = agreement_provider()
    async with db_sessionmaker() as session:
        session.add(
            ModelProbe(
                provider=PROVIDER_NAME,
                model=AGREEMENT_TAG,
                digest="sha256:chatweights",
                harness_version=harness_version(),
                depth="fast",
                earned_agency="assisted",
                cases={"measured_think_mode": "always"},
            )
        )
        await session.commit()

    async with models_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    assert _entry_thinking(resp.json(), AGREEMENT_TAG) == "always"
    assert await _card_think(provider, db_sessionmaker) == "always"


async def test_a_stored_think_mode_reaches_the_composer_list(db_sessionmaker) -> None:
    # The same agreement for the other source: a confirmed declaration.
    provider = agreement_provider()
    async with db_sessionmaker() as session:
        session.add(
            ModelPolicy(
                provider=PROVIDER_NAME,
                model=AGREEMENT_TAG,
                digest="sha256:chatweights",
                think_mode="hybrid",
            )
        )
        await session.commit()

    async with models_client(provider, db_sessionmaker) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    assert _entry_thinking(resp.json(), AGREEMENT_TAG) == "hybrid"
    assert await _card_think(provider, db_sessionmaker) == "hybrid"


async def test_a_store_that_is_down_is_asked_once_not_once_per_row() -> None:
    # The failure that made this endpoint unusable: every read swallows its own
    # exception, so without a shared record a dozen rows paid a dozen connect
    # timeouts on EVERY render - an endpoint that needed no database at all
    # before the declarations reached it. One attempt, then the report answers.
    attempts = 0

    def failing_sessionmaker():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database down")

    provider = FakeProvider(models=POD_INSTALLED, capabilities_by_model=POD_REPORTS)
    async with models_client(provider, failing_sessionmaker) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200  # degraded to the report, never a 500
    assert attempts <= 1


async def test_an_inventory_larger_than_the_bound_stops_asking() -> None:
    # The half the failure cache cannot cover. Rows are resolved concurrently
    # under a bound, so the first batch is already in flight before any of them
    # knows the store is down - nothing can spare those. What the shared record
    # spares is every row BEHIND the bound, which on a pod holding dozens of
    # tags is most of them. Without it each one waits its turn and pays its own
    # connect timeout, which is the shape that measured in seconds per render.
    #
    # The sessionmaker AWAITS before failing, as a real connect timeout does:
    # one that raised synchronously would let each task run to completion
    # before the next started, and the concurrency this is about would never
    # happen.
    attempts = 0
    rows = _MAX_CONCURRENT_THINK_READS * 3

    @asynccontextmanager
    async def failing_session():
        nonlocal attempts
        attempts += 1
        await asyncio.sleep(0)  # the await a real connection attempt has
        raise RuntimeError("database down")
        yield  # pragma: no cover - unreachable, keeps this a generator

    installed_many = [
        ModelInfo(name=f"stray:{index}b", size=10, digest=f"sha256:stray{index}")
        for index in range(rows)
    ]
    reports = {info.name: ModelCapabilities(completion=True, tools=True) for info in installed_many}
    provider = FakeProvider(models=installed_many, capabilities_by_model=reports)
    async with models_client(provider, failing_session) as http_client:
        resp = await http_client.get("/api/v1/models")

    assert resp.status_code == 200
    # Everything in the first batch is unavoidable; nothing behind it is.
    assert attempts <= _MAX_CONCURRENT_THINK_READS
    assert attempts < rows


async def test_a_remembered_failure_is_not_retried_on_the_next_render() -> None:
    # The other half: the short-circuit is per REQUEST, so without failure
    # caching the next render pays the timeout again. Mirrors the runtime-facts
    # cache's own precedent for a failed probe.
    attempts = 0

    def failing_sessionmaker():
        nonlocal attempts
        attempts += 1
        raise RuntimeError("database down")

    async with models_client(agreement_provider(), failing_sessionmaker) as http_client:
        assert (await http_client.get("/api/v1/models")).status_code == 200
        first = attempts
        assert (await http_client.get("/api/v1/models")).status_code == 200

    assert first >= 1  # the first render really did try
    assert attempts == first  # and the second one did not


# --- the console's declaration writers, end to end --------------------------

# The server these tests declare against: a chat model, a reasoning partner
# whose template carries think plumbing, and a chat alternate. Every tag has a
# digest, because a declaration binds to weights.
DECLARE_REPORTS = {
    "fake": ModelCapabilities(completion=True, tools=True, template_thinking=False),
    "fake-think": ModelCapabilities(completion=True, tools=True, template_thinking=True),
    "fake-alt": ModelCapabilities(completion=True, tools=True, template_thinking=True),
    # A model that reasons and cannot hold a conversation - the shape that made
    # "can it reason" the wrong question to ask of a thinking partner.
    "seer": ModelCapabilities(vision=True, template_thinking=True),
    "embed:1b": ModelCapabilities(embedding=True),
}


def declaring_provider(*names: str) -> FakeProvider:
    """A server holding ``names``, each with a digest a declaration can bind to."""
    return FakeProvider(
        models=[ModelInfo(name=name, digest=f"sha256:{name}") for name in names],
        capabilities_by_model=DECLARE_REPORTS,
    )


async def test_declaring_a_ceiling_reports_what_it_actually_changed(db_sessionmaker) -> None:
    """The response says what the declaration DID, not just what was written.

    Declaring `agentic` on a model nothing has deep-checked stores the
    permission and moves nothing yet - `min(CEILING, EARNED)` still holds it at
    the grace floor. A response echoing only the stored value would let the
    page announce an escalation that did not happen.

    MUTATION: return `ceiling` in place of the resolved `effective` and this
    reads back "agentic" for a model no check has certified.
    """
    async with models_client(
        declaring_provider("fake", "fake-think"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/ceiling", json={"name": "fake", "ceiling": "agentic"}
        )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["ceiling"] == "agentic"
    assert body["digest"] == "sha256:fake"
    # Stored, and not yet in force: no deep certificate binds these weights.
    assert body["earned"] == "assisted"
    assert body["effective"] == "assisted"


async def test_a_down_clamp_takes_effect_immediately(db_sessionmaker) -> None:
    # The other direction, and the one that needs no evidence at all: min()
    # lowers without proof, so a console must never make this hard to reach.
    async with models_client(
        declaring_provider("fake", "fake-think"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/ceiling", json={"name": "fake", "ceiling": "deterministic"}
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["effective"] == "deterministic"
        # And the management view agrees, through its own independent read.
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    fake = next(row for row in rows if row["model"] == "fake")
    assert fake["agency"]["effective"] == "deterministic"
    assert fake["ceiling_declared"] is True


async def test_an_unknown_ceiling_is_rejected_by_the_schema(db_sessionmaker) -> None:
    async with models_client(declaring_provider("fake"), db_sessionmaker) as http_client:
        resp = await http_client.post(
            "/api/v1/models/ceiling", json={"name": "fake", "ceiling": "autonomous"}
        )
    assert resp.status_code == 422


async def test_granting_reasoning_with_tools_reports_that_nothing_changed_yet(
    db_sessionmaker,
) -> None:
    """The test-first gate, end to end: the grant is stored and inert.

    The console's whole promise on this axis rests on the response being
    honest. Granting it on a pair nothing has deep-checked stores a durable
    permission and moves nothing - and this is the axis where an over-claiming
    page is dangerous rather than merely wrong, because the capability it would
    announce fails as an EMPTY answer instead of an error.

    MUTATION: return `stored.thinking_with_tools` in place of the resolved
    `explanation.thinking_with_tools` and `effective` reads back true for a
    model no check has certified.
    """
    async with models_client(
        declaring_provider("fake", "fake-think"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-with-tools",
            json={"name": "fake", "thinking_with_tools": True},
        )
        assert resp.status_code == 200, resp.text
        body = resp.json()
        assert body["thinking_with_tools"] is True
        assert body["digest"] == "sha256:fake"
        assert body["effective"] is False
        # ...and the management view agrees, through its own independent read:
        # the claim is shown, the fold is shown, and they differ.
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    fake = next(row for row in rows if row["model"] == "fake")
    assert fake["thinking_with_tools"] is True
    assert fake["thinking_with_tools_declared"] is True
    assert fake["thinking_with_tools_effective"] is False


async def test_the_wire_carries_the_reasoning_verdict_apart_from_the_claim(
    db_sessionmaker,
) -> None:
    """End to end: a passing check is visible BEFORE anyone declares.

    The console words its declare dialog off this field, so it has to survive
    the whole trip. Without it the page can only see "not in force" and would
    tell an operator whose deep check just passed that granting it changes
    nothing - the opposite of what happens.

    The environment pairs the two tags deliberately: the reasoning verdict
    binds to the PARTNER's weights, so a card that resolved no partner would
    compare the row's ``thinking_digest`` against nothing and report the check
    as unestablished.
    """
    async with models_client(
        declaring_provider("fake", "fake-think"),
        db_sessionmaker,
        {"thinking_chat_model": "fake-think"},
    ) as http_client:
        async with db_sessionmaker() as session:
            session.add(
                ModelProbe(
                    # The app's own selector, not the fake's class name: rows
                    # key on MODEL_PROVIDER, which these settings leave at its
                    # default while overriding the provider object.
                    provider=PROVIDER_NAME,
                    model="fake",
                    digest="sha256:fake",
                    harness_version=harness_version(),
                    depth="deep",
                    earned_agency="agentic",
                    thinking_with_tools=True,
                    serving_endpoint=FAKE_SERVING_ENDPOINT,
                    cases={
                        "representative": True,
                        "thinking_digest": "sha256:fake-think",
                    },
                )
            )
            await session.commit()
        clear_earned_cache()
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    fake = next(row for row in rows if row["model"] == "fake")
    assert fake["thinking_with_tools"] is False  # nothing declares it
    assert fake["thinking_with_tools_effective"] is False  # so no turn acts on it
    assert fake["measured"]["thinking_with_tools_passed"] is True  # but it IS established


async def test_withdrawing_reasoning_with_tools_needs_no_evidence(db_sessionmaker) -> None:
    """The safe direction applies at once, exactly as a ceiling down-clamp does.

    The store is append-only, so revoking is a newer row rather than a delete -
    which is what lets it say "withheld" at all, distinct from "nobody ever
    declared". Both the declared claim and the fold have to follow it.
    """
    async with models_client(
        declaring_provider("fake", "fake-think"), db_sessionmaker
    ) as http_client:
        granted = await http_client.post(
            "/api/v1/models/thinking-with-tools",
            json={"name": "fake", "thinking_with_tools": True},
        )
        assert granted.status_code == 200, granted.text
        resp = await http_client.post(
            "/api/v1/models/thinking-with-tools",
            json={"name": "fake", "thinking_with_tools": False},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json()["effective"] is False
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    fake = next(row for row in rows if row["model"] == "fake")
    assert fake["thinking_with_tools"] is False
    assert fake["thinking_with_tools_declared"] is True  # a decision, not an absence
    assert fake["thinking_with_tools_effective"] is False


async def test_a_reasoning_grant_for_absent_weights_is_a_404(db_sessionmaker) -> None:
    async with models_client(declaring_provider("fake"), db_sessionmaker) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-with-tools",
            json={"name": "fake-absent", "thinking_with_tools": True},
        )
    assert resp.status_code == 404


async def test_declaring_a_thinking_pair_names_the_partner_on_the_card(db_sessionmaker) -> None:
    """The pair reaches the management view through the same read a turn uses.

    MUTATION: resolve the card's sibling from anything but
    `thinking_sibling_for` and the card names no partner while turns route to
    `fake-alt`.
    """
    async with models_client(
        declaring_provider("fake", "fake-think", "fake-alt"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-pair",
            json={"name": "fake", "thinking_model": "fake-alt"},
        )
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"model": "fake", "thinking_model": "fake-alt"}
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    paired = next(row for row in rows if row["model"] == "fake")
    assert paired["thinking_model"] == "fake-alt"
    assert paired["thinking_model_declared"] is True


async def test_a_vision_partner_is_refused_end_to_end(db_sessionmaker) -> None:
    """The role check has to be armed on the ENDPOINT, through the real resolver.

    The service takes `partner_serves_chat` as an argument, so a unit test can
    only prove the branch. This proves the endpoint computes it - that a tag the
    runtime reports as seeing but not completing resolves to "does not serve
    chat" rather than the argument silently defaulting true. The partner's think
    mode is a genuine `hybrid` here, so the reasoning check it would otherwise
    trip on passes and this refusal is the one under test.

    MUTATION: pass a literal `True` for `partner_serves_chat` in the endpoint
    and this stores a pairing onto a model that cannot answer.
    """
    async with models_client(declaring_provider("fake", "seer"), db_sessionmaker) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-pair",
            json={"name": "fake", "thinking_model": "seer"},
        )
    assert resp.status_code == 422
    assert "does not serve chat turns" in resp.json()["detail"]


async def test_the_listing_carries_a_declared_pair_to_the_composer(db_sessionmaker) -> None:
    """The composer decides from this key whether its Thinking control means anything.

    A listing reporting less than the whole chain would grey the toggle out on
    exactly the models the pair feature exists to rescue - and the message it
    greys out with tells the reader to go and declare the pairing they have
    already declared. The console and the composer must not disagree about a
    state the console just wrote.

    MUTATION: drop the `thinking_model` write from `list_models` and the entry
    reports nothing while turns route to `fake-alt`.
    """
    async with models_client(
        declaring_provider("fake", "fake-think", "fake-alt"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-pair",
            json={"name": "fake", "thinking_model": "fake-alt"},
        )
        assert resp.status_code == 200, resp.text
        listing = await http_client.get("/api/v1/models")
    entry = next(e for e in listing.json()["entries"] if e["model"] == "fake")
    assert entry["capabilities"]["thinking_model"] == "fake-alt"


async def test_an_environment_override_reaches_the_listing_too(db_sessionmaker) -> None:
    # The kill switch has to be visible where the control is drawn: with pair
    # routing disabled deployment-wide, a listing still naming a sibling would
    # leave the composer offering reasoning no turn will do.
    async with models_client(
        declaring_provider("fake", "fake-think"),
        db_sessionmaker,
        {"thinking_chat_model": "none"},
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-pair",
            json={"name": "fake", "thinking_model": "fake-think"},
        )
        assert resp.status_code == 200, resp.text
        listing = await http_client.get("/api/v1/models")
    entry = next(e for e in listing.json()["entries"] if e["model"] == "fake")
    assert "thinking_model" not in entry["capabilities"]


async def test_a_partner_that_answers_directly_is_refused(db_sessionmaker) -> None:
    """The candidate rule, enforced at the writer and not only in the menu.

    The runtime reports `fake`'s template with no think plumbing, so pairing it
    as somebody's reasoning partner would produce a toggle that changes nothing.

    MUTATION: pass a hard-coded "always" as `partner_think_mode` in the
    endpoint and this stores a pair that cannot reason.
    """
    async with models_client(
        declaring_provider("fake", "fake-alt"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post(
            "/api/v1/models/thinking-pair",
            json={"name": "fake-alt", "thinking_model": "fake"},
        )
    assert resp.status_code == 422
    assert "can reason" in resp.json()["detail"]


async def test_setting_the_workspace_default_is_readable_at_once(db_sessionmaker) -> None:
    """The default is stored, validated, and visible to the very next read.

    Deliberately NOT claiming to cover the endpoint's explicit commit. Under
    ASGITransport the whole app call finishes before `post()` returns, and the
    session dependency's own success-path commit runs on teardown either way -
    so removing the explicit one would leave this green. The explicit commit is
    still right in production, where teardown can lose the race with a real
    client's refetch; it just cannot be proven from here, and a MUTATION note
    promising otherwise would be worse than no note.
    """
    async with models_client(
        declaring_provider("fake", "fake-alt"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post("/api/v1/models/default", json={"name": "fake-alt"})
        assert resp.status_code == 200, resp.text
        assert resp.json() == {"model": "fake-alt"}
        view = await http_client.get("/api/v1/models/manager")
    rows = [row for group in view.json()["groups"] for row in group["rows"]]
    assert next(row for row in rows if row["model"] == "fake-alt")["workspace_default"] is True


async def test_a_default_the_server_does_not_hold_is_refused(db_sessionmaker) -> None:
    # The worst kind of stored setting: accepted silently, then surfacing one
    # turn later on somebody else's question as a substitution note.
    async with models_client(declaring_provider("fake"), db_sessionmaker) as http_client:
        resp = await http_client.post("/api/v1/models/default", json={"name": "fake-absent"})
    assert resp.status_code == 404


async def test_a_model_that_cannot_chat_cannot_become_the_workspace_default(
    db_sessionmaker,
) -> None:
    # An embedder defines a vector space, not an answer, and the RUNTIME is
    # what says so. The successor rule a delete enforces and this one have to
    # agree, or the console would offer a default the delete path refuses as a
    # successor.
    async with models_client(
        declaring_provider("fake", "embed:1b"), db_sessionmaker
    ) as http_client:
        resp = await http_client.post("/api/v1/models/default", json={"name": "embed:1b"})
    assert resp.status_code == 409
    assert "cannot drive a chat turn" in resp.json()["detail"]
