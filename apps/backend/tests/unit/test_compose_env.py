"""Guard against inert environment variables in the compose stack.

Compose passes only the variables each service explicitly lists (the stack has
no ``env_file``), so a documented, config-read variable that is missing from a
service's ``environment:`` block is silently ignored under ``make up`` - the
code default wins and the ``.env`` value never arrives. There is nothing to see
when this happens: the stack boots, the tests pass, and the setting simply has
no effect, which is why it has bitten this project more than once.

Two layers guard it. Named allowlists pin the keys whose SERVICE matters (a
request-path knob must reach the backend specifically, a parsing knob the
worker), and one exhaustive test asserts that every field of the Settings model
reaches somewhere at all. The exhaustive one exists because the allowlists are
hand-maintained and had quietly fallen a whole phase behind.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from app.core.config import Settings


def _find_compose() -> Path | None:
    """Walk up for infra/docker-compose.yml, rather than counting directories.

    A fixed number of parents only holds when the tests run from the repo. Inside
    the backend image the backend IS the root, so counting four levels up walked
    off the top of the filesystem and took the whole collection down with it.
    """
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "infra" / "docker-compose.yml"
        if candidate.is_file():
            return candidate
    return None


_COMPOSE = _find_compose()
_PROD_COMPOSE = _COMPOSE.with_name("docker-compose.prod.yml") if _COMPOSE else None

# What `ports: !reset []` parses to here. `!reset` is a Compose tag rather than
# anything YAML knows, so the safe loader refuses the file outright without this
# - and the distinction it carries is the whole point of the tests below: a
# plain `ports: []` parses as an ordinary empty list, which is exactly the
# mistake that leaves the base file's published ports in place.
_RESET = object()


class _ComposeLoader(yaml.SafeLoader):
    """Safe loader that understands Compose's `!reset` override tag."""


_ComposeLoader.add_constructor("!reset", lambda loader, node: _RESET)

# The compose file is not shipped inside the backend image, and these tests are
# only ever about the repo's own configuration - so where it is absent there is
# nothing to check, not a failure.
pytestmark = pytest.mark.skipif(
    _COMPOSE is None, reason="no infra/docker-compose.yml above this checkout"
)


def _env_keys(service: dict) -> set[str]:
    """Return the bare variable names a compose service exposes.

    Handles both entry shapes compose allows: ``- NAME`` (pass through from the
    host env) and ``- NAME=value`` (explicit value).
    """
    entries = service.get("environment", []) or []
    keys: set[str] = set()
    for entry in entries:
        name = entry.split("=", 1)[0].strip()
        if name:
            keys.add(name)
    return keys


@pytest.fixture(scope="module")
def compose() -> dict:
    assert _COMPOSE is not None  # guarded by the module-level skip
    return yaml.safe_load(_COMPOSE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prod_compose() -> dict:
    assert _PROD_COMPOSE is not None  # sits beside the base file
    return yaml.load(_PROD_COMPOSE.read_text(encoding="utf-8"), Loader=_ComposeLoader)


@pytest.fixture(scope="module")
def backend_env(compose: dict) -> set[str]:
    return _env_keys(compose["services"]["backend"])


@pytest.fixture(scope="module")
def worker_env(compose: dict) -> set[str]:
    return _env_keys(compose["services"]["worker"])


# Read on the chat REQUEST path (chat_service / rag_service), so they must reach
# the backend or a tuned .env value is inert.
_BACKEND_REQUEST_KEYS = {
    "ATTACHMENT_MAX_BYTES",
    "ATTACHMENT_WAIT_SECONDS",
    "ATTACHMENT_POLL_SECONDS",
    "RAG_ATTACHMENT_TOP_K",
    "RAG_ATTACHMENT_TOKEN_BUDGET",
    "VISION_MAX_IMAGES",
    "VISION_IMAGE_BUDGET_CELLS",
    "VISION_MAX_SOURCE_BYTES",
    "WEB_FETCH_MAX_PAGES",
    "WEB_RESULT_CACHE_MINUTES",
    # How much window a turn on an unsized tag gets. Inert here would mean a
    # deployment that raised it to its hardware still running at the 8192 floor.
    "DEFAULT_CONTEXT_WINDOW",
    # Where retrieval's query embedding runs. Inert here would silently put it
    # back on the card the pin exists to keep it off.
    "EMBEDDING_DEVICE",
    # The agency policy and the tool loop. AGENCY_MODE is the documented kill
    # switch ("one env line apart"), so it being inert made the switch a lie.
    "AGENCY_MODE",
    "AGENCY_REQUIRE_VERIFICATION",
    "TOOL_MAX_ITERATIONS",
    "TOOL_MAX_CALLS_PER_TURN",
    "TOOL_TIMEOUT_SECONDS",
    "TOOL_RESULT_MAX_TOKENS",
    "TOOL_RESULT_TOTAL_MAX_TOKENS",
    "TOOL_RESULT_MIN_TOKENS",
    "TOOL_SELECTION_THRESHOLD",
    "TOOL_SELECTION_TOP_K",
    # The caps on files a tool hands back; read where the tool loop runs.
    "TOOL_ARTIFACT_MAX_BYTES",
    "TOOL_ARTIFACT_MAX_PER_TURN",
    # The MCP tool sources, built once per process from these.
    "MCP_CHASSIS_URL",
    "MCP_CHASSIS_API_KEY",
    "MCP_PER_USER_CREDENTIALS",
    "MCP_CHASSIS_CREDENTIAL_NAME",
    # The document-generation sidecar, seeded into the registry the same way.
    "DOCGEN_MCP_URL",
    "DOCGEN_API_KEY",
    "DOCGEN_CREDENTIAL_NAME",
    "MCP_TIMEOUT_SECONDS",
    "MCP_LIST_TIMEOUT_SECONDS",
    "MCP_LIST_CACHE_SECONDS",
    "MCP_LIST_CACHE_ERROR_SECONDS",
    # Unset means the secret store is unavailable; it must be able to arrive.
    "SECRET_MASTER_KEY",
    # Rate limiting runs as middleware on every request. TRUSTED_PROXY_IPS in
    # particular decides WHO a limit is counted against, so a value that never
    # arrives means a proxied deployment shares one budget across all callers.
    "RATE_LIMIT_ENABLED",
    "RATE_LIMIT_PER_MINUTE",
    "RATE_LIMIT_HEAVY_PER_MINUTE",
    "RATE_LIMIT_LIGHT_PER_MINUTE",
    "TRUSTED_PROXY_IPS",
    # Selects the SearchProvider the request path builds; unlisted, setting it
    # to "none" would not turn web search off, which is its whole job.
    "SEARCH_PROVIDER",
}

# Settings fields that are deliberately NOT deployment configuration, and so are
# not expected in any service's environment. Keep this EMPTY if at all possible:
# every entry is a key an operator can set in .env and watch do nothing. It
# exists as a pressure valve with a visible cost, not as a place to file things.
_NOT_DEPLOYMENT_CONFIG: set[str] = set()

# Read only in the WORKER (parsing and garbage collection), never on a request.
_WORKER_ONLY_KEYS = {
    "ATTACHMENT_GC_INTERVAL_MINUTES",
    "ATTACHMENT_ORPHAN_GRACE_MINUTES",
    "ATTACHMENT_STALL_MINUTES",
    # The retention windows, swept from the GC cron. Inert here would mean a
    # deployment that turned generated-file expiry off still expiring files.
    "GENERATED_FILE_RETENTION_DAYS",
    "ATTACHMENT_RETENTION_DAYS",
    "OCR_ENABLED",
    "OCR_LANGUAGE",
    "OCR_DPI",
    "OCR_MIN_CHARS",
    "OCR_MAX_PAGES",
    "OCR_TIMEOUT_SECONDS",
}


@pytest.mark.parametrize("key", sorted(_BACKEND_REQUEST_KEYS))
def test_request_path_knob_reaches_the_backend(backend_env: set[str], key: str) -> None:
    assert key in backend_env, (
        f"{key} is read on the chat request path but is missing from the backend "
        "compose environment; a value set in .env would be silently inert."
    )


@pytest.mark.parametrize("key", sorted(_WORKER_ONLY_KEYS))
def test_worker_knob_reaches_the_worker(worker_env: set[str], key: str) -> None:
    assert key in worker_env, (
        f"{key} is read by the worker but is missing from the worker compose "
        "environment; a value set in .env would be silently inert."
    )


def test_every_setting_reaches_some_service(backend_env: set[str], worker_env: set[str]) -> None:
    """Every Settings field must be able to arrive somewhere.

    The two allowlists above are hand-maintained, which is exactly how this bug
    class survives: they were written for the attachment knobs and simply never
    grew, so an entire phase's worth of keys (the whole agency and tool-loop
    block, the secret master key, two web-search knobs) sat unlisted and inert
    while these tests stayed green. An allowlist can only catch what somebody
    remembered to add to it.

    So this one is exhaustive and needs no memory: it reads the Settings model
    itself, and a new field fails here on the day it is added unless it reaches
    a service or is consciously exempted.
    """
    declared = {name.upper() for name in Settings.model_fields}
    reachable = backend_env | worker_env | _NOT_DEPLOYMENT_CONFIG
    unreachable = sorted(declared - reachable)
    assert not unreachable, (
        "These Settings fields reach no service's compose environment, so setting "
        "them in .env does nothing at all: "
        f"{', '.join(unreachable)}. Add each to the service that reads it, or - "
        "only if it is genuinely not deployment configuration - to "
        "_NOT_DEPLOYMENT_CONFIG with a reason."
    )


def test_the_default_context_window_reaches_both_callers(
    backend_env: set[str], worker_env: set[str]
) -> None:
    """Both services size a generation, and they must size it the same.

    The backend resolves the window for a chat turn and the worker resolves it
    for the title job, through the same rule precisely so one tag is never
    asked for at two different windows - the model server reloads the model
    between them, which on a card that fits one model is a reload for six words
    of title. A key that reached only one service would reintroduce exactly
    that disagreement, silently, the moment anyone tuned it.
    """
    for service, keys in (("backend", backend_env), ("worker", worker_env)):
        assert "DEFAULT_CONTEXT_WINDOW" in keys, (
            f"DEFAULT_CONTEXT_WINDOW is missing from the {service} compose environment; "
            "the chat turn and the title job would size the same model differently."
        )


def test_the_embedding_device_reaches_both_callers(
    backend_env: set[str], worker_env: set[str]
) -> None:
    """Both services embed, and they must place that work on the same device.

    The backend embeds retrieval queries; the worker embeds every uploaded
    file. A key that reached only one of them would put ingestion on the GPU
    while queries stayed off it - and on a card that fits one model, that is an
    eviction of the warmed chat model per upload, which is precisely what
    pinning the embedder to the CPU exists to prevent. Silent, too: everything
    still works, just many times slower.
    """
    for service, keys in (("backend", backend_env), ("worker", worker_env)):
        assert "EMBEDDING_DEVICE" in keys, (
            f"EMBEDDING_DEVICE is missing from the {service} compose environment; "
            "queries and ingestion would place embedding work differently."
        )


def test_model_server_credentials_reach_both_callers(
    backend_env: set[str], worker_env: set[str]
) -> None:
    """The credential must arrive wherever the model server is called.

    Both services call it, and each fails differently without it: the backend
    could not answer a single chat turn, and the worker could not embed an
    uploaded file - so pinning it to one service would leave the other broken
    with nothing to point at. The exhaustive test above only asks that a key
    reaches SOMEWHERE, which is exactly the gap this closes.
    """
    for service, keys in (("backend", backend_env), ("worker", worker_env)):
        assert "OLLAMA_AUTH_HEADERS" in keys, (
            f"OLLAMA_AUTH_HEADERS is missing from the {service} compose environment; "
            "a protected model server would refuse every call it makes."
        )


@pytest.mark.parametrize(
    "key",
    [
        "STORAGE_BACKEND",
        "STORAGE_LOCAL_DIR",
        "S3_ENDPOINT_URL",
        "S3_BUCKET",
        "S3_ACCESS_KEY_ID",
        "S3_SECRET_ACCESS_KEY",
        "S3_REGION",
    ],
)
def test_storage_configuration_reaches_both_services(
    backend_env: set[str], worker_env: set[str], key: str
) -> None:
    """The API and the worker must address the SAME store, key for key.

    The API writes uploaded bytes; the worker reads them to parse, deletes them
    on a purge, and walks the whole store on the orphan sweep. Half a
    configuration is worse than none: a worker left on the local-fs default
    while the API writes to S3 would find an empty store, parse nothing, and -
    because the sweep deletes what no row claims - would be one changed fence
    away from reaping live objects. The exhaustive test above only asks that a
    key reaches SOMEWHERE, which is exactly the gap this closes.
    """
    for service, keys in (("backend", backend_env), ("worker", worker_env)):
        assert key in keys, (
            f"{key} is missing from the {service} compose environment; the two "
            "services would resolve different object stores."
        )


def test_every_service_restarts_itself(compose: dict) -> None:
    """A crashed container must come back without a human.

    Same class of bug as an inert variable - something configured out of the
    stack that only shows up as strange behaviour much later. The worker was
    killed by the kernel once and simply stayed dead, and since the worker is
    the only thing that moves an upload off `pending`, every file attached after
    that sat on a spinner indefinitely.
    """
    missing = sorted(
        name for name, service in compose["services"].items() if not (service or {}).get("restart")
    )
    assert not missing, (
        f"no restart policy on: {', '.join(missing)}. A crash in one of these is "
        "permanent until somebody notices."
    )


# The proxy is the deployment's single public entrance, so it is the one service
# a production stack publishes. Named here rather than derived, because the whole
# assertion is that this list has exactly one entry.
_PUBLIC_SERVICE = "caddy"


def test_production_publishes_only_the_proxy(compose: dict, prod_compose: dict) -> None:
    """A deployment must expose the proxy and nothing else.

    This is a security property, and its failure is entirely silent: the stack
    boots, the app works, and an unauthenticated API (or Postgres, or the model
    server) is answering on a public interface with nobody the wiser. There is
    no login yet, so anyone who reaches the API port IS the workspace's default
    user - which is why "did we actually unpublish it" cannot be left to a
    reading of the file.

    The check is derived from the base file rather than from a list kept here,
    so a service that starts publishing a port later is covered on the day it
    does, with no test to remember to update.
    """
    base_publishers = {
        name for name, service in compose["services"].items() if (service or {}).get("ports")
    }
    prod_services = prod_compose["services"]
    still_published = sorted(
        name
        for name in base_publishers
        if (prod_services.get(name) or {}).get("ports") is not _RESET
    )
    assert not still_published, (
        "these services still publish host ports in the production overlay: "
        f"{', '.join(still_published)}. Each needs `ports: !reset []` - and the "
        "`!reset` tag specifically, because Compose MERGES port sequences across "
        "files, so a plain `ports: []` leaves the base file's ports published."
    )

    unexpected = sorted(
        name
        for name, service in prod_services.items()
        if name != _PUBLIC_SERVICE and (service or {}).get("ports") not in (None, _RESET)
    )
    assert not unexpected, (
        f"only {_PUBLIC_SERVICE} may publish ports in production; these also do: "
        f"{', '.join(unexpected)}."
    )
    assert prod_services[_PUBLIC_SERVICE].get("ports"), (
        f"{_PUBLIC_SERVICE} publishes no ports, so a production deployment is "
        "unreachable from outside the machine."
    )


def test_every_production_service_restarts_itself(prod_compose: dict) -> None:
    """Same rule as the base file, applied to what the overlay adds.

    A service declared only in the overlay is invisible to the base-file check
    above, and the proxy is precisely the container whose death takes the whole
    deployment offline with it.
    """
    missing = sorted(
        name
        for name, service in prod_compose["services"].items()
        if not (service or {}).get("restart")
    )
    assert not missing, (
        f"no restart policy on: {', '.join(missing)}. On a server this means the "
        "service stays down after a crash or a host reboot."
    )


def test_every_service_is_hardened_for_production(compose: dict, prod_compose: dict) -> None:
    """A service added to the base file must be raised to `always` in the overlay.

    The base file restarts everything `on-failure`, which recovers a crash but
    deliberately does NOT bring the stack up when the Docker daemon starts - so
    on a laptop, launching Docker does not boot a GPU-resident model nobody
    asked for. On a server that trade is backwards: after a host reboot, an
    `on-failure` service simply never comes back, and the first anyone hears of
    it is a feature that has quietly stopped working.

    Every base service is raised to `always` in the overlay today. The check
    exists because that is currently a habit rather than a rule: the test above
    walks the overlay, so a new service that never got an entry is invisible to
    it, and nothing else would notice until a reboot.
    """
    hardened = set(prod_compose["services"])
    unhardened = sorted(name for name in compose["services"] if name not in hardened)
    assert not unhardened, (
        "these services have no production overlay entry, so they keep the base "
        f"file's `restart: on-failure` on a server: {', '.join(unhardened)}. Add "
        "`restart: always` for each - on-failure does not restore a service after "
        "a host reboot."
    )
