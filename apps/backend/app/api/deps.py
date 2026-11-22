"""FastAPI dependency-injection providers."""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator, Callable
from typing import Annotated

from fastapi import Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.concurrency import StreamLimiter
from app.core.config import Settings, get_settings
from app.core.errors import ForbiddenError, ServiceUnavailableError
from app.core.factories import build_model_provider, build_search_provider, build_storage_backend
from app.core.tenancy import TenantContext
from app.db.engine import build_engine, build_sessionmaker
from app.parsing.registry import ParserRegistry, default_registry
from app.providers.base import ModelProvider
from app.providers.model_registry import OllamaRegistryClient
from app.providers.rag.base import Embedder, Reranker, VectorStore
from app.providers.rag.embedder import ProviderEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.providers.rag.reranker import InfinityReranker
from app.providers.search.base import SearchProvider
from app.providers.search.fetcher import HttpPageFetcher, PageFetcher
from app.providers.secrets import SecretCipher, SecretConfigError, build_cipher
from app.providers.tools.base import ToolSource
from app.services.auth.principal import resolve_principal, seeded_principal
from app.services.auth.store import KeyValueStore
from app.services.embedding_resolution import resolve_embedding
from app.services.secret_store import SecretStore
from app.services.tools.mcp_sources import McpSourceProvider
from app.services.tools.selection import ToolVectorCache
from app.services.workspace_defaults import load_defaults, reranker_availability
from app.storage.base import StorageBackend
from app.workers.queue import ArqJobQueue, JobQueue

logger = logging.getLogger(__name__)


def get_model_provider(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> ModelProvider:
    """Return the process-wide provider built at startup.

    The instance lives on ``app.state`` (created by the lifespan handler) so all
    requests share one HTTP connection pool. When the lifespan has not run -
    e.g. unit tests driving the ASGI app directly - a fallback instance is
    built on first use.
    """
    provider = getattr(request.app.state, "model_provider", None)
    if provider is None:
        provider = build_model_provider(settings)
        request.app.state.model_provider = provider
    return provider


def get_sessionmaker(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> async_sessionmaker[AsyncSession]:
    """Return the process-wide session factory built at startup.

    Same lazy-fallback pattern as ``get_model_provider``: tests that set
    ``app.state.db_sessionmaker`` directly (sqlite) bypass the fallback, and a
    bare ASGI app without lifespan still works against configured postgres.
    """
    maker = getattr(request.app.state, "db_sessionmaker", None)
    if maker is None:
        engine = build_engine(settings)
        maker = build_sessionmaker(engine)
        request.app.state.db_engine = engine
        request.app.state.db_sessionmaker = maker
    return maker


async def get_session(
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
) -> AsyncIterator[AsyncSession]:
    """Yield a request-scoped database session (one session = one request).

    Rolls back when the handler raises, so a failing handler leaves no
    partial writes behind. The success-path commit here is only a SAFETY NET:
    teardown runs after the response has been sent, so mutating endpoints
    must commit explicitly via ``repo.commit()`` before returning or an
    immediate follow-up read can observe pre-commit state. Streaming
    endpoints manage their own sessions via ``get_sessionmaker`` instead.
    """
    async with sessionmaker() as session:
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        else:
            await session.commit()


def get_session_store(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> KeyValueStore:
    """Return the expiring key-value store login sessions live in.

    The process's one Redis pool, borrowed from the job queue - the same move
    the rate limiter makes for its counter, and for the same reason: a second
    pool for a handful of small reads per request buys nothing.

    Unlike the limiter, a store that cannot serve the protocol is a hard failure
    rather than a shrug. The limiter treats a queue that cannot count as "no
    counter" and lets traffic through; an authenticator that cannot read
    sessions must refuse, or "we could not check" becomes "come in".

    Raises:
        ServiceUnavailableError: The wired queue does not offer the store verbs.
    """
    queue = get_job_queue(request, settings)
    if not isinstance(queue, KeyValueStore):
        raise ServiceUnavailableError("Sessions cannot be checked right now.")
    return queue


async def get_current_tenant(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> TenantContext:
    """Return the acting tenant/user - the auth seam.

    With no ``OIDC_ISSUER`` configured this returns the seeded default tenant as
    an admin, exactly as it did before login existed: same value, and no store
    or database read on the way to it. That is the no-regression clamp, and it
    is the first branch here so it stays cheap as well as identical.

    With login configured the context comes from the session cookie - see
    :func:`app.services.auth.principal.resolve_principal` for the full order,
    including the in-container operator path the deployment's own make targets
    take. The role is read from the users row on EVERY request, so a change to
    it applies immediately rather than at the next sign-in.

    The store and the session factory are resolved INSIDE rather than declared
    as dependencies so that a deployment with no login pays nothing for them:
    every route in the app depends on this one, and half of them touch neither.
    One consequence worth knowing before writing a test against this:
    ``app.dependency_overrides[get_sessionmaker]`` does NOT reach the auth path,
    because nothing here is resolved through the override table. Set
    ``app.state.db_sessionmaker`` (and ``app.state.job_queue``) instead, as the
    route tests do - an override would leave a test passing for a reason that
    has nothing to do with the code under it.
    """
    if not settings.oidc_enabled:
        return seeded_principal()
    return await resolve_principal(
        request,
        settings=settings,
        store=get_session_store(request, settings),
        sessionmaker=get_sessionmaker(request, settings),
    )


def require_role(role: str) -> Callable[..., TenantContext]:
    """Build a dependency that admits only a caller holding ``role``.

    With login configured, ``get_current_tenant`` carries the role from the
    signed-in person's row, so this genuinely refuses a member (403). With no
    issuer configured every caller is the seeded admin and the gate passes -
    the pre-login behaviour, kept deliberately. A member-accessible endpoint
    depends on ``get_current_tenant`` directly instead of this.
    """

    def dependency(
        tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    ) -> TenantContext:
        if tenant.role != role:
            raise ForbiddenError(f"This action requires the {role!r} role.")
        return tenant

    return dependency


def get_stream_limiter(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StreamLimiter:
    """Return the process-wide per-user stream limiter (lazy fallback for tests)."""
    limiter = getattr(request.app.state, "stream_limiter", None)
    if limiter is None:
        limiter = StreamLimiter(settings.max_streams_per_tenant)
        request.app.state.stream_limiter = limiter
    return limiter


def get_vector_store(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> VectorStore:
    """Return the process-wide vector store (lazy fallback for tests).

    Construction performs no network I/O - connections happen on first use -
    so the fallback is safe even in offline contexts. Tests inject an
    in-memory store by setting ``app.state.vector_store``.
    """
    store = getattr(request.app.state, "vector_store", None)
    if store is None:
        store = QdrantVectorStore(url=settings.qdrant_url)
        request.app.state.vector_store = store
    return store


async def get_embedder(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
) -> Embedder | None:
    """Return the workspace's embedder, or None when nothing resolves one.

    The embedder instance is the single source of truth for the vector space
    (its model tag) and dimension, and which model that is now comes from the
    resolution chain rather than from one written-down answer - see
    :func:`app.services.embedding_resolution.resolve_embedding`. Built per
    request; construction is trivially cheap and never loads the model. Tests
    inject a ``FakeEmbedder`` by setting ``app.state.embedder``.

    The workspace rung costs one indexed single-row read per request, in a
    session opened and closed HERE rather than held. A dependency's session
    would live as long as the response, and this dependency feeds
    ``/chat/stream`` - a pooled connection held open for the minutes an answer
    streams, to answer a question settled before the first token. The evidence
    rung adds nothing on a warm process: it reads the stranded-vectors alarm's
    own cached enumeration.
    """
    override = getattr(request.app.state, "embedder", None)
    if override is not None:
        return override
    async with sessionmaker() as session:
        stored = await load_defaults(session, tenant)
    choice = await resolve_embedding(
        settings=settings,
        stored=stored,
        store=store,
        provider=provider,
        provider_name=settings.model_provider,
    )
    if choice.config is None:
        return None
    return ProviderEmbedder(provider, choice.config, batch_size=settings.rag_embed_batch)


async def get_reranker(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> Reranker | None:
    """Return the reranker this DEPLOYMENT wired, else None (dense-only).

    ``RERANKER_URL`` is the whole gate: setting it is how an operator says a
    sidecar is running, and retrieval must never depend on that sidecar being
    up. Nothing written down votes any more - a second source used to have to
    AGREE, which is how the shipped configuration ended up with a URL beside a
    declaration that disabled reranking, looking wired and doing nothing.

    Deployment-scoped only: the workspace toggle that can withhold a wired
    reranker is applied downstream, by the callers that already hold the
    tenant's defaults (see
    :func:`app.services.workspace_defaults.apply_reranker_default`). Reading
    the row here instead would put a database session on every SSE stream for
    its whole lifetime, which is a pooled connection held open for minutes to
    answer a question the turn has already answered.
    """
    reranker = getattr(request.app.state, "reranker", None)
    if reranker is not None:
        return reranker
    if not settings.reranker_url:
        # Cheap answer first: a deployment with no sidecar reaches here on
        # EVERY turn (nothing is ever cached above), so any work beyond what
        # this line already knows would be paid per request.
        return None
    # The same verdict the settings page renders, read from the same function -
    # so a page that says "not configured" and a turn that reranks anyway is
    # not expressible. There is nothing left to refuse on: a URL is the gate,
    # and the model name behind it cannot be blank (a blank one reads as unset
    # and falls to the shipped default, see
    # app.core.config.Settings._blank_reranker_model_as_default).
    availability = reranker_availability(settings)
    reranker = InfinityReranker(settings.reranker_url, availability.model or "")
    request.app.state.reranker = reranker
    return reranker


def get_storage_backend(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> StorageBackend:
    """Return the object-storage backend, building it once per process.

    Never None, unlike :func:`get_reranker` and :func:`get_search_provider`:
    those front optional sidecars a turn can do without, whereas an attachment
    whose bytes cannot be reached is simply an attachment the app cannot answer
    about. Tests inject a ``FakeStorageBackend`` by setting ``app.state.storage``.
    """
    backend = getattr(request.app.state, "storage", None)
    if backend is not None:
        return backend
    backend = build_storage_backend(settings)
    request.app.state.storage = backend
    return backend


def get_parser_registry(request: Request) -> ParserRegistry:
    """Return the process-wide parser registry, building it once per process.

    The registry is a small stateless table of parsers, so one instance serves
    every request. Tests can inject a narrower registry by setting
    ``app.state.parser_registry``.
    """
    registry = getattr(request.app.state, "parser_registry", None)
    if registry is not None:
        return registry
    registry = default_registry()
    request.app.state.parser_registry = registry
    return registry


async def get_mcp_sources(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
) -> list[ToolSource]:
    """Return the acting tenant's MCP tool sources, read from the DB registry.

    The registry is now per-tenant: an :class:`McpSourceProvider` (built once and
    held on ``app.state``) reads this tenant's enabled ``mcp_servers`` rows,
    seeds the env server on first use, and returns one listing-cached source per
    row. The provider MEMOIZES each source per ``(tenant, server)``, so the remote
    listing cache still survives across turns - the DB read here is only a small
    indexed lookup, and construction opens no connection.

    Building a source per user-scoped row wires a per-call credential resolver; the
    provider builds the secret cipher lazily and only then, so a malformed
    ``SECRET_MASTER_KEY`` never breaks the (default) shared-header path. Empty when
    the tenant has no enabled server. Tests (and a fixed-list deployment) inject a
    static source list by setting ``app.state.mcp_tool_sources``, which short-
    circuits the DB path entirely.
    """
    override = getattr(request.app.state, "mcp_tool_sources", None)
    if override is not None:
        return override
    provider = getattr(request.app.state, "mcp_source_provider", None)
    if provider is None:
        provider = McpSourceProvider(settings, sessionmaker=sessionmaker)
        request.app.state.mcp_source_provider = provider
    return await provider.sources_for(tenant)


def get_mcp_source_provider(request: Request) -> McpSourceProvider | None:
    """Return the process MCP source provider if one exists, else None.

    Deliberately does NOT build one: the provider is created lazily by
    :func:`get_mcp_sources` on the first tool-capable turn. The Integrations write
    endpoints use this only to INVALIDATE cached sources after a change; when no
    turn has built a provider yet (or a test injected a static
    ``app.state.mcp_tool_sources`` override), there is nothing to invalidate and
    this returns None.
    """
    return getattr(request.app.state, "mcp_source_provider", None)


def get_secret_cipher(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SecretCipher | None:
    """Return the process secret cipher, or None when it is unavailable.

    Built once and cached on ``app.state`` (None is a valid cached value). An
    unset ``SECRET_MASTER_KEY`` yields None - the credential endpoints then answer
    503. A key that is SET but malformed is logged once and also treated as
    unavailable, so a bad key degrades the credential surface to 503 rather than
    raising on every request.
    """
    if hasattr(request.app.state, "secret_cipher"):
        return request.app.state.secret_cipher
    try:
        cipher = build_cipher(settings.secret_master_key)
    except SecretConfigError:
        logger.warning(
            "SECRET_MASTER_KEY is set but unusable; credential endpoints will answer 503"
        )
        cipher = None
    request.app.state.secret_cipher = cipher
    return cipher


def get_secret_store(
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    cipher: Annotated[SecretCipher | None, Depends(get_secret_cipher)],
) -> SecretStore:
    """Return a SecretStore bound to this request's session, tenant, and cipher.

    A None cipher does not fail here: the store constructs fine and raises
    ``SecretStoreUnavailableError`` only when a write or reveal is attempted -
    which the credential endpoints map to 503. Metadata reads still work on a None
    cipher (they touch no plaintext).
    """
    return SecretStore(session, tenant, cipher)


def get_tool_vector_cache(request: Request) -> ToolVectorCache:
    """Return the process-wide tool-description vector cache.

    Process-wide because the whole point is to embed a tool's description once
    and reuse it across turns; a per-request cache would embed every description
    on every turn that ranks tools, which is the cost this exists to avoid. It is
    keyed by content hash, so a server changing a description re-embeds only that
    tool. Holds no per-caller data.
    """
    cache = getattr(request.app.state, "tool_vector_cache", None)
    if cache is not None:
        return cache
    cache = ToolVectorCache()
    request.app.state.tool_vector_cache = cache
    return cache


def get_job_queue(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> JobQueue:
    """Return the process-wide job queue, building it once per process.

    Never None: work that must happen in the background has no synchronous
    fallback, so an unreachable Redis is a 503 at enqueue time rather than a
    silently skipped job. Construction opens no connection - the pool is built
    on the first enqueue. Tests inject a ``FakeJobQueue`` by setting
    ``app.state.job_queue``.
    """
    queue = getattr(request.app.state, "job_queue", None)
    if queue is not None:
        return queue
    queue = ArqJobQueue(settings.redis_url)
    request.app.state.job_queue = queue
    return queue


def get_search_provider(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchProvider | None:
    """Return the web-search provider, or None when web search is unconfigured.

    None whenever ``SEARCH_PROVIDER`` is ``none`` or ``SEARXNG_URL`` is unset -
    the turn stays fully local, the same graceful-degrade contract as the
    reranker (never depend on the sidecar being up). Construction performs no
    network I/O. Tests inject a ``FakeSearchProvider`` by setting
    ``app.state.search_provider``.
    """
    provider = getattr(request.app.state, "search_provider", None)
    if provider is not None:
        return provider
    provider = build_search_provider(settings)
    if provider is not None:
        request.app.state.search_provider = provider
    return provider


def get_model_registry_client(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> OllamaRegistryClient:
    """Return the process-wide registry client for download preflights.

    Shared like the other HTTP clients so preflights reuse one connection pool
    rather than paying a TLS handshake per keystroke-triggered check.
    Construction performs no network I/O, and nothing calls it unless somebody
    asks about a model - a deployment that never opens the management page
    never contacts a registry at all. Tests inject a mock-transport client by
    setting ``app.state.model_registry_client``.
    """
    client = getattr(request.app.state, "model_registry_client", None)
    if client is None:
        client = OllamaRegistryClient(settings.ollama_registry_url)
        request.app.state.model_registry_client = client
    return client


def get_page_fetcher(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
) -> PageFetcher | None:
    """Return the web-page fetcher, or None when page fetching is off/unused.

    None when ``web_fetch_pages`` is disabled or web search is unconfigured -
    the fetcher only ever runs inside a web-search turn, so there is nothing to
    fetch without a search backend. It asks the same question the factory does,
    ``SEARCH_PROVIDER`` included, rather than re-deriving "is search on" from
    the URL alone. Construction does no network I/O. Tests inject a
    ``FakePageFetcher`` by setting ``app.state.page_fetcher``.
    """
    fetcher = getattr(request.app.state, "page_fetcher", None)
    if fetcher is not None:
        return fetcher
    if not settings.web_fetch_pages or build_search_provider(settings) is None:
        return None
    fetcher = HttpPageFetcher(
        timeout=settings.web_fetch_timeout_seconds,
        max_bytes=settings.web_fetch_max_bytes,
        max_chars=settings.web_fetch_max_chars,
    )
    request.app.state.page_fetcher = fetcher
    return fetcher
