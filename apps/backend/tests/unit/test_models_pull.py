"""Tests for the download preflight and the streamed pull endpoint."""

from __future__ import annotations

from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager

import httpx
import pytest
from httpx import ASGITransport, AsyncClient

from app.api.deps import get_current_tenant, get_model_provider
from app.core.config import Settings, get_settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.providers.base import ManagementCapabilities, ModelProvider, ProviderUnavailableError
from app.providers.fake_provider import FakeProvider
from app.providers.model_registry import OllamaRegistryClient
from app.schemas.models import PullErrorEvent, PullProgressEvent
from app.services import model_pull

MANIFEST = {
    "config": {"digest": "sha256:cfg", "size": 500},
    "layers": [{"digest": "sha256:a", "size": 2_000_000}],
}

# A non-admin of the same tenant, for the role gate.
MEMBER = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="member")


@pytest.fixture(autouse=True)
def _fresh_state() -> Iterator[None]:
    model_pull.clear_pulls_in_flight()
    yield
    model_pull.clear_pulls_in_flight()


def registry_client(handler) -> OllamaRegistryClient:
    return OllamaRegistryClient(
        "https://registry.example",
        client=httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )


def manifest_handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=MANIFEST)


@asynccontextmanager
async def management_client(
    provider: ModelProvider,
    *,
    registry=manifest_handler,
    tenant: TenantContext | None = None,
) -> AsyncIterator[AsyncClient]:
    """App-wired client over a mock-transport registry."""
    settings = Settings(
        default_chat_model="",
        thinking_chat_model=None,
        sse_heartbeat_seconds=0,
        _env_file=None,
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    if tenant is not None:
        app.dependency_overrides[get_current_tenant] = lambda: tenant
    app.state.model_registry_client = registry_client(registry)
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


def frames(body: str) -> list[tuple[str, str]]:
    """Parse an SSE body into (event, data) pairs, dropping keep-alive comments."""
    parsed: list[tuple[str, str]] = []
    for block in body.split("\n\n"):
        lines = [line for line in block.splitlines() if line and not line.startswith(":")]
        if not lines:
            continue
        event = next((line[len("event: ") :] for line in lines if line.startswith("event: ")), "")
        data = next((line[len("data: ") :] for line in lines if line.startswith("data: ")), "")
        parsed.append((event, data))
    return parsed


# --- POST /api/v1/models/preflight ---


async def test_preflight_reports_the_exact_download_size() -> None:
    async with management_client(FakeProvider()) as client:
        resp = await client.post("/api/v1/models/preflight", json={"name": "qwen3:4b"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "available"
    # Summed from the manifest, never estimated from a similarly-named tag.
    assert body["download_size_bytes"] == 2_000_500
    assert body["kind"] == "library"
    assert body["warning"] is None


async def test_preflight_reports_a_missing_tag_as_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": []})

    async with management_client(FakeProvider(), registry=handler) as client:
        resp = await client.post("/api/v1/models/preflight", json={"name": "ghost:1b"})
    body = resp.json()
    assert body["status"] == "not_found"
    assert body["download_size_bytes"] is None
    assert "spelling" in body["detail"]


async def test_an_unreachable_registry_is_unknown_never_missing() -> None:
    # The whole discipline of this endpoint: a network blip reported as a
    # missing model sends someone off to fix a spelling that was already right.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    async with management_client(FakeProvider(), registry=handler) as client:
        resp = await client.post("/api/v1/models/preflight", json={"name": "qwen3:4b"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "unknown"
    assert body["download_size_bytes"] is None


async def test_preflight_of_a_huggingface_name_warns_instead_of_guessing() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("an hf.co name must not be looked up in this registry")

    async with management_client(FakeProvider(), registry=handler) as client:
        resp = await client.post("/api/v1/models/preflight", json={"name": "hf.co/user/repo"})
    body = resp.json()
    assert body["kind"] == "huggingface"
    assert body["status"] == "unknown"
    assert body["download_size_bytes"] is None
    assert body["warning"]


async def test_preflight_on_a_backend_that_cannot_pull_is_refused() -> None:
    # "available, 2.0 GB" about a tag this backend can never fetch is simply
    # false - and this is the one management endpoint whose work is an outbound
    # request to a public host, so a backend with no use for it should not make
    # the request at all.
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a backend that cannot pull must not reach the registry")

    provider = FakeProvider(management=ManagementCapabilities(can_list_installed=True))
    async with management_client(provider, registry=handler) as client:
        resp = await client.post("/api/v1/models/preflight", json={"name": "qwen3:4b"})
    assert resp.status_code == 400
    assert "redeployment" in resp.json()["detail"]


async def test_preflight_refuses_an_unsafe_name_before_any_request() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("a rejected name must never reach the network")

    async with management_client(FakeProvider(), registry=handler) as client:
        resp = await client.post(
            "/api/v1/models/preflight", json={"name": "https://evil.example/x"}
        )
    assert resp.status_code == 422
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "validation_error"


# --- POST /api/v1/models/pull ---


async def test_pull_streams_progress_then_a_terminal_done() -> None:
    provider = FakeProvider(
        pull_script=[
            {"status": "pulling manifest"},
            {"status": "pulling sha256:a", "digest": "sha256:a", "total": 200, "completed": 50},
            {"status": "success"},
        ]
    )
    # verify off: this is a test about the DOWNLOAD's frames. The hand-off into
    # the automatic tool check is exercised in test_models_verify.py, where
    # there is a measurement log for it to write to.
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b", "verify": False})
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/event-stream")
    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "progress", "progress", "done"]
    middle = PullProgressEvent.model_validate_json(events[1][1])
    assert (middle.completed, middle.total, middle.percent) == (50, 200, 25.0)
    assert middle.digest == "sha256:a"
    # A stage with no transfer behind it reports no progress rather than a
    # number a bar would then have to un-draw.
    assert PullProgressEvent.model_validate_json(events[0][1]).percent is None
    assert provider.pulled == ["qwen3:4b"]


async def test_a_failure_mid_download_becomes_an_error_frame() -> None:
    # Never a dead connection: a stream that just stops leaves a progress bar
    # frozen with no way to tell a finished download from a broken one.
    provider = FakeProvider(
        pull_script=[{"status": "pulling manifest"}],
        pull_error=ProviderUnavailableError("Lost the connection to the model server."),
    )
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "error"]
    error = PullErrorEvent.model_validate_json(events[-1][1])
    assert error.code == "provider_unavailable"


async def test_an_in_stream_error_ends_the_download() -> None:
    # The model server reports a missing tag as prose inside a 200 stream: the
    # download had already started by the time the manifest lookup failed.
    provider = FakeProvider(
        pull_script=[
            {"status": "pulling manifest"},
            {"error": "pull model manifest: file does not exist"},
            {"status": "success"},
        ]
    )
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "ghost:1b"})
    events = frames(resp.text)
    assert [name for name, _ in events] == ["progress", "error"]
    assert PullErrorEvent.model_validate_json(events[-1][1]).code == "model_not_found"


async def test_an_unexpected_failure_never_leaks_its_exception() -> None:
    provider = FakeProvider(pull_script=[], pull_error=RuntimeError("secret internal detail"))
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    events = frames(resp.text)
    assert [name for name, _ in events] == ["error"]
    error = PullErrorEvent.model_validate_json(events[0][1])
    assert error.code == "internal_error"
    assert "secret internal detail" not in error.message


async def test_a_second_pull_of_the_same_tag_is_refused() -> None:
    # An impatient second click must not start a second download competing with
    # the first for the same blobs.
    model_pull.claim_pull("qwen3:4b")
    async with management_client(FakeProvider()) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    assert resp.status_code == 409
    assert resp.headers["content-type"].startswith("application/problem+json")
    assert resp.json()["code"] == "conflict"


async def test_a_different_tag_pulls_while_another_is_in_flight() -> None:
    # Only the SAME tag twice is the mistake; two models at once is slow but
    # perfectly valid, and refusing it would make the guard a queue.
    model_pull.claim_pull("other:1b")
    async with management_client(FakeProvider()) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b", "verify": False})
    assert resp.status_code == 200


async def test_a_finished_download_releases_its_claim() -> None:
    body = {"name": "qwen3:4b", "verify": False}
    async with management_client(FakeProvider()) as client:
        await client.post("/api/v1/models/pull", json=body)
        assert model_pull.pulls_in_flight() == frozenset()
        # ...so the same tag can be pulled again, e.g. to repair a bad download.
        again = await client.post("/api/v1/models/pull", json=body)
    assert again.status_code == 200


async def test_a_failed_download_releases_its_claim() -> None:
    provider = FakeProvider(pull_script=[], pull_error=RuntimeError("boom"))
    async with management_client(provider) as client:
        await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    assert model_pull.pulls_in_flight() == frozenset()


async def test_a_refused_download_leaves_no_claim_behind() -> None:
    # A wedged claim is unrecoverable without a restart, so every path out of
    # the request has to drop it - including the ones that never stream at all.
    provider = FakeProvider(management=ManagementCapabilities(can_list_installed=True))
    async with management_client(provider) as client:
        await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    assert model_pull.pulls_in_flight() == frozenset()


async def test_the_stream_itself_never_releases_the_claim() -> None:
    # The release belongs to the REQUEST scope, not to this generator, and that
    # is load-bearing: a client that hangs up before the body is iterated leaves
    # the generator unstarted, and an unstarted generator's finally never runs -
    # so a generator-owned release would wedge the tag until a restart.
    # Draining the stream must therefore leave the claim exactly as it found it.
    #
    # Driven directly rather than through the ASGI client on purpose: that
    # transport runs the app to completion and buffers the whole body before
    # returning, so a "client hangs up early" case cannot be staged through it -
    # a test written that way passes whichever scope owns the release.
    model_pull.claim_pull("qwen3:4b")
    provider = FakeProvider(pull_script=[{"status": "success"}])
    events = [event async for event in model_pull.stream_pull(provider, tag="qwen3:4b")]
    assert [type(event).__name__ for event in events] == ["PullProgressEvent", "PullDoneEvent"]
    assert model_pull.pulls_in_flight() == frozenset({model_pull.pull_claim_key("qwen3:4b")})


@pytest.mark.parametrize(
    ("claimed", "requested"),
    [
        ("qwen3", "qwen3:latest"),
        ("qwen3:latest", "qwen3"),
        ("qwen3", "library/qwen3"),
        ("library/qwen3:latest", "qwen3"),
        # The model server lowercases every part of a name while resolving it,
        # so these differ only in how they were typed - one set of local
        # weights, one download. (The tag grammar itself is case-SENSITIVE, and
        # the library ships uppercase tags like q4_K_M; the fold is a fact about
        # this runtime, not about the registry.)
        ("qwen3:4b", "QWEN3:4B"),
        ("qwen3:q4_k_m", "qwen3:Q4_K_M"),
    ],
)
async def test_a_duplicate_cannot_slip_through_by_respelling_the_name(
    claimed: str, requested: str
) -> None:
    # The guard is only as good as its key: keyed on what somebody typed, a
    # second click gets through simply by naming the model differently, and two
    # downloads then race each other onto the same blobs.
    model_pull.claim_pull(claimed)
    async with management_client(FakeProvider()) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": requested})
    assert resp.status_code == 409


def test_a_hugging_face_name_keeps_its_case() -> None:
    # The opposite of the rule above: a Hugging Face repository name IS
    # case-sensitive, so folding it would refuse a genuinely different download
    # as a duplicate.
    assert model_pull.pull_claim_key("hf.co/User/Repo") != model_pull.pull_claim_key(
        "hf.co/user/repo"
    )


async def test_a_backend_that_cannot_pull_says_so() -> None:
    # Never a silent no-op: a process-per-model runtime changes by redeployment,
    # and a 200 with no download would look exactly like a working button.
    provider = FakeProvider(management=ManagementCapabilities(can_list_installed=True))
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "qwen3:4b"})
    assert resp.status_code == 400
    assert "redeployment" in resp.json()["detail"]
    assert provider.pulled == []


# --- the admin gate ---


@pytest.mark.parametrize(
    ("method", "path", "body"),
    [
        ("POST", "/api/v1/models/pull", {"name": "qwen3:4b"}),
        ("DELETE", "/api/v1/models/qwen3:4b", None),
        # Preflight rides with the verb it precedes: it is the endpoint that
        # makes the backend issue an outbound request, and being able to shop
        # the registry without being able to download would be a strange line.
        ("POST", "/api/v1/models/preflight", {"name": "qwen3:4b"}),
    ],
)
async def test_the_destructive_verbs_are_admin_gated(
    method: str, path: str, body: dict | None
) -> None:
    # The check passes for everyone today (the default principal is an admin),
    # so this pins that the SEAM is wired: the day OIDC populates a real role,
    # these two verbs start refusing without anybody having to remember them.
    provider = FakeProvider()
    async with management_client(provider, tenant=MEMBER) as client:
        resp = await client.request(method, path, json=body)
    assert resp.status_code == 403
    assert resp.json()["code"] == "forbidden"
    assert provider.pulled == []
    assert provider.deleted == []


async def test_pull_refuses_an_unsafe_name() -> None:
    provider = FakeProvider()
    async with management_client(provider) as client:
        resp = await client.post("/api/v1/models/pull", json={"name": "../../etc/passwd"})
    assert resp.status_code == 422
    assert provider.pulled == []


@pytest.mark.parametrize("path", ["/api/v1/models/pull", "/api/v1/models/preflight"])
async def test_an_absurd_name_is_rejected_while_parsing(path: str) -> None:
    # The cap is declared on the schema, not only in the validator: nothing else
    # bounds a JSON body field, so without it a multi-megabyte "name" is
    # materialised in full and only then measured.
    async with management_client(FakeProvider()) as client:
        resp = await client.post(path, json={"name": "a" * 5_000})
    assert resp.status_code == 422
    assert resp.json()["code"] == "validation_error"


# --- the pure progress normalizer ---


@pytest.mark.parametrize(
    "raw",
    [
        {},
        {"status": ""},
        {"nothing": "useful"},
        "not a dict",
        None,
    ],
)
def test_events_with_nothing_to_say_are_dropped(raw: object) -> None:
    assert model_pull.normalize_pull_event(raw) is None


@pytest.mark.parametrize(
    ("raw", "percent"),
    [
        ({"status": "x", "completed": 5, "total": 10}, 50.0),
        # A blob's last frame can overshoot its declared total; a bar that
        # reads 100.4% looks broken.
        ({"status": "x", "completed": 11, "total": 10}, 100.0),
        ({"status": "x", "completed": 5}, None),
        ({"status": "x", "total": 10}, None),
        ({"status": "x", "completed": 5, "total": 0}, None),
        ({"status": "x", "completed": "5", "total": 10}, None),
    ],
)
def test_percent_is_derived_only_from_real_byte_counts(raw: dict, percent: float | None) -> None:
    event = model_pull.normalize_pull_event(raw)
    assert isinstance(event, PullProgressEvent)
    assert event.percent == percent


@pytest.mark.parametrize(
    ("message", "code"),
    [
        ("pull model manifest: file does not exist", "model_not_found"),
        ("model not found", "model_not_found"),
        ("repository not found", "model_not_found"),
        ("manifest unknown", "model_not_found"),
        # A DNS, disk, or template failure is the model server's problem, not
        # the name's - telling someone to check their spelling is wrong advice,
        # which is why the markers name what is missing rather than matching a
        # bare "not found" anywhere in the sentence.
        ("dial tcp: no such host", "provider_unavailable"),
        ("write /root/.ollama: no space left on device", "provider_unavailable"),
        ("template not found", "provider_unavailable"),
    ],
)
def test_in_stream_errors_are_classified(message: str, code: str) -> None:
    event = model_pull.normalize_pull_event({"error": message})
    assert isinstance(event, PullErrorEvent)
    assert event.code == code
