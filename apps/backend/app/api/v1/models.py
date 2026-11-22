"""Models endpoints: the UI model listing, the warmup hook, and management."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import aclosing
from dataclasses import dataclass
from typing import Annotated

from fastapi import APIRouter, Depends, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import (
    get_current_tenant,
    get_embedder,
    get_mcp_sources,
    get_model_provider,
    get_model_registry_client,
    get_page_fetcher,
    get_reranker,
    get_search_provider,
    get_session,
    get_sessionmaker,
    get_tool_vector_cache,
    get_vector_store,
    require_role,
)
from app.core.config import Settings, get_settings, ollama_url_host, ollama_url_is_remote
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.core.tenancy import TenantContext
from app.providers.base import (
    ModelCapabilities,
    ModelNotFoundError,
    ModelProvider,
    ProviderUnavailableError,
)
from app.providers.model_registry import OllamaRegistryClient
from app.providers.rag.base import Embedder, Reranker, VectorStore
from app.providers.search.base import SearchProvider
from app.providers.search.fetcher import PageFetcher
from app.providers.tools.base import ToolSource
from app.schemas.models import (
    EmbedderCheckRequest,
    EmbedderCheckResponse,
    ModelCeilingRequest,
    ModelCeilingResponse,
    ModelDefaultResponse,
    ModelDeleteResponse,
    ModelEntry,
    ModelManagerResponse,
    ModelNameRequest,
    ModelPreflightResponse,
    ModelPullRequest,
    ModelResidencyResponse,
    ModelServerStatus,
    ModelsResponse,
    ModelThinkingPairRequest,
    ModelThinkingPairResponse,
    ModelThinkingWithToolsRequest,
    ModelThinkingWithToolsResponse,
    ModelThinkModeResponse,
    ModelVerifyRequest,
    PullDoneEvent,
    PullErrorEvent,
    PullProgressEvent,
    ResidentModel,
    VerifyDoneEvent,
    VerifyErrorEvent,
    VerifyProgressEvent,
    WarmRequest,
    WarmResponse,
    WarmSearchResponse,
)
from app.services import (
    embedder_verify,
    model_declare,
    model_delete,
    model_pull,
    model_verify,
    think_check,
    workspace_defaults,
)
from app.services.agency import parse_agency_mode, resolve_agency
from app.services.earned import explain_earned
from app.services.embedding_resolution import resolve_embedding
from app.services.model_facts import (
    effective_agency_ceiling,
    effective_think_mode,
    effective_tools_mode,
    effective_vision,
    report_for,
    runtime_capabilities_many,
)
from app.services.model_identity import digest_of
from app.services.model_manager import build_manager_view, derive_role, serves_chat_role
from app.services.model_names import MAX_NAME_LENGTH, require_model_name
from app.services.model_policy import ReadHealth, policy_for
from app.services.model_resolution import (
    ThinkMode,
    thinking_sibling_for,
)
from app.services.tools.offer import ToolWiring
from app.services.tools.selection import ToolVectorCache
from app.services.warmup import WarmToolPrefix, warm_embedder, warm_model, warm_search
from app.services.workspace_defaults import apply_reranker_default, load_defaults
from app.sse import sse_event, with_heartbeat

logger = logging.getLogger(__name__)

router = APIRouter(tags=["models"])

# The verbs that ACT are admin-gated, matching the integration registry: they
# change what the whole workspace runs on. So are the two management READS, for
# a different reason - each discloses deployment-wide operator information (a
# cross-tenant space inventory, and what the shared model server is holding) -
# and both were declared as TODOs while the role check could not yet refuse
# anybody. A configured login makes it refuse, so they are wired now.
#
# Everything else on this router takes the plain principal. It is not decoration:
# where login is on, a route with NO principal dependency is a route that
# answers an unauthenticated caller, and a guard test walks the live surface to
# prove none is left.
AdminDep = Annotated[TenantContext, Depends(require_role("admin"))]
TenantDep = Annotated[TenantContext, Depends(get_current_tenant)]

# How many think-mode resolutions the model listing runs at once. Each opens
# its own session, so the bound is about not asking the connection pool for one
# per row when a server holds a large inventory - the same reasoning as the
# capability probes' bound, and the same number.
_MAX_CONCURRENT_THINK_READS = 8


@router.get("/models")
async def list_models(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: TenantDep,
) -> ModelsResponse:
    """Return the models this server holds, and what is known about each.

    The ``models`` key is the provider's live list (the UI selector depends on
    its shape); ``entries`` describes each held tag - the role the runtime's
    report derives for it, and the capabilities that report plus the stored
    declarations resolve. The chat ones are what a conversation can be pinned
    to, so a ``role`` repeats across the list: filter on it, never key on it.

    Every row is a model the server actually holds. A menu that also listed
    models nobody had downloaded was a menu of things a picker could only
    refuse, and against a remote host it described a machine that does not
    exist - so the inventory is the list.

    ``server`` describes where the model server is and whether it is reached
    with credentials, read from configuration alone so it is as true under a
    stub provider as a live one. It is what lets a client warn that a model
    server outside this deployment is unprotected - the server authenticates
    nothing itself.

    The ``thinking`` capability on every row resolves through the SAME chain a
    turn does, stored declarations and unconfirmed measurements included. This
    list feeds the composer, and the composer sits beside a management card
    that resolves the full chain - two answers about one model on two screens
    is exactly the disagreement worth paying for.

    What it costs, stated plainly because it is not free: each row pays one
    capability probe (concurrent, bounded, and cached for minutes) plus, on a
    cold cache, one declaration read and up to four probe-log reads. The rows
    are resolved CONCURRENTLY behind the same caches the chat path fills, and
    the first read that cannot reach the store stops the rest - so a database
    that is down costs this endpoint one timeout, not one per row.
    """
    try:
        models = await provider.list_models()
    except ProviderUnavailableError as exc:
        # An unreachable provider must surface as 503: this endpoint IS the
        # inventory, and an empty list would read as a server holding nothing
        # rather than as a server that did not answer. The provider's own
        # message is carried through - a redirect or a refused credential names
        # the setting that fixes it, and a fixed string here would throw that
        # away. The AppError handler renders the problem+json body.
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    digests = {m.name: m.digest for m in models if m.digest}
    tags = [m.name for m in models]

    # One record for the whole request. The reads degrade to "nothing declared"
    # individually, which is right - but a store that is down would otherwise
    # be asked once per row, and a connect timeout per row is the difference
    # between a slow page and an unusable one. The first read that cannot reach
    # it stops the rest.
    health = ReadHealth()
    resolutions = asyncio.Semaphore(_MAX_CONCURRENT_THINK_READS)

    async def resolved_think(tag: str, caps: ModelCapabilities | None) -> ThinkMode:
        """The full think resolution for one tag, through the one service."""
        async with resolutions:
            if health.failed:
                # Somebody already found the store unreachable. The report is
                # the answer a failed read degrades to anyway, so this returns
                # the same thing without the wait.
                return effective_think_mode(tag, caps)
            return await think_check.resolve_think_mode(
                provider_name=settings.model_provider,
                model=tag,
                report=caps,
                digests=digests,
                sessionmaker=sessionmaker,
                health=health,
            )

    async def resolved_sibling(tag: str) -> str | None:
        """The thinking partner for one tag, through the chain a turn applies.

        The composer reads this to decide whether its Thinking control can do
        anything, so a listing that reported only half the chain would grey the
        toggle out for exactly the models the pair feature exists to rescue -
        and tell the reader to go and declare the pairing they had already
        declared.

        Through the SHARED resolution rather than a local one, so the tag named
        here is the tag the console's grid names, the verify job drives, and a
        reasoning verdict binds against. Its self-exclusion matters most on the
        surface a person actually looks at: a deployment whose environment
        points the pair override at the chat tag itself would otherwise be told
        the model is its own partner.
        """
        if digest_of(tag, digests) is None:
            return thinking_sibling_for(settings, tag, declared=None)
        async with resolutions:
            declared: str | None = None
            if not health.failed:
                policy = await policy_for(
                    provider_name=settings.model_provider,
                    model=tag,
                    digests=digests,
                    sessionmaker=sessionmaker,
                    health=health,
                )
                declared = None if policy is None else policy.thinking_model
            return thinking_sibling_for(settings, tag, declared=declared)

    # Concurrently but BOUNDED, and cached for a short TTL by the facts
    # service: a server holding a dozen tags would otherwise pay a dozen serial
    # metadata round trips on every render of the model picker, while an
    # unbounded fan-out would open one connection per tag at once against a
    # server that may hold far more than a dozen.
    reports = await runtime_capabilities_many(provider, settings.model_provider, tags)
    described = [(tag, caps) for tag, caps in zip(tags, reports, strict=True) if caps is not None]
    skipped = [tag for tag, caps in zip(tags, reports, strict=True) if caps is None]
    # Gathered rather than resolved in the loop, for the same reason the probes
    # above are: these reach the declaration store, and one round trip per tag
    # chained behind the last is what made this endpoint slow.
    thinks = await asyncio.gather(*(resolved_think(tag, caps) for tag, caps in described))
    siblings = await asyncio.gather(*(resolved_sibling(tag) for tag, _caps in described))
    entries: list[ModelEntry] = []
    for (tag, caps), think, sibling in zip(described, thinks, siblings, strict=True):
        capabilities = _reported_capabilities(tag, caps, think)
        if sibling is not None:
            capabilities["thinking_model"] = sibling
        entries.append(
            ModelEntry(
                role=derive_role(caps),
                provider=settings.model_provider,
                model=tag,
                capabilities=capabilities,
            )
        )
    if skipped:
        # Deliberately not surfaced to the client: the response is a healthy
        # 200 whose entries simply lack these rows, so this line is the one
        # place a live diagnosis can see WHICH tags a half-answering server
        # (tags endpoint up, describe endpoint refused) kept out of the picker.
        logger.warning(
            "Capability probes failed for %d installed tag(s) %s; they are listed "
            "but not described this fetch",
            len(skipped),
            skipped,
        )
    return ModelsResponse(
        models=models,
        entries=entries,
        server=ModelServerStatus(
            remote=ollama_url_is_remote(settings.ollama_url),
            auth_configured=bool(settings.ollama_auth_headers),
            host=ollama_url_host(settings.ollama_url),
        ),
    )


def _reported_capabilities(
    tag: str, caps: ModelCapabilities, think: ThinkMode
) -> dict[str, object]:
    """Project a runtime report onto the capability vocabulary a client reads.

    The line the fill stops at is the one that matters: a MECHANICAL fact is
    something the serving stack knows about itself and is entitled to report,
    while POLICY (how much a model may drive, whether it may reason and call
    tools at once) is a human decision a runtime cannot make on anyone's
    behalf. So no ``agency`` key is written here, ever - a tag nobody has
    declared is described honestly and still runs at the safe floor.

    ``thinking`` is written only when SOMETHING answered - a stored
    confirmation, a measurement, the quirk seed
    (:mod:`app.services.known_models`), or the runtime's template. Absent means
    unknown, and a key claiming a model answers directly on no evidence would
    be a worse lie than saying nothing.

    Args:
        tag: The installed tag being described.
        caps: The runtime's report for this tag.
        think: The caller's already-resolved think mode - taken as an argument
            rather than resolved here because resolving it reaches the database
            and this function is pure.

    Returns:
        The capability map for this row.
    """
    reported: dict[str, object] = {
        "tools": effective_tools_mode(caps),
        "vision": effective_vision(tag, caps),
    }
    if think != "unknown":
        reported["thinking"] = False if think == "never" else think
    return reported


@router.get("/models/manager")
async def model_manager(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    tenant: AdminDep,
) -> ModelManagerResponse:
    """Return the management view: the inventory joined to measured and declared state.

    Read-only. Every row carries what the runtime REPORTS about the weights, what
    provider currently holds the weights, what a tool-check actually MEASURED
    on them, and which management verbs apply - assembled per request, with
    nothing stored and nothing predicted. Each row also says what is still set
    to use that model: the deployment's own chat pair withdraws Delete (the
    one refusal a request cannot resolve), while a pin count and the
    workspace-default flag ride along so the confirm dialog can disclose what
    a delete will move - pins cleared, the default handed to a successor
    named in the same request. The payload carries
    whether the vectors already stored are still reachable by the configured
    embedding model too, which is the one fact about the locked embedding role
    that the lock itself cannot guarantee.

    A provider that cannot be reached is a 503 rather than a page with
    every row marked "not installed": those flags are the point of the view,
    and inverting them all would invite re-pulling models that are already
    here. An unreachable VECTOR store is not a 503, though - it makes the
    stranded report say "unknown" and leaves the rest of the page useful, and
    it is bounded by its own short timeout so a black-holed store cannot hold
    the page either. The AppError handler renders the problem+json body.

    ADMIN-ONLY, and now genuinely so. The response carries a DEPLOYMENT-WIDE
    space inventory - every collection, its point count, and which embedding
    models wrote it, summed across every tenant - which is operator information
    rather than anything a member's own work needs. The gate was declared but
    unenforceable while every caller was the single seeded user; a configured
    login makes the role real, so it is wired here alongside the three verbs
    that ACT (pull, delete, and the preflight before a pull).
    """
    try:
        return await build_manager_view(
            provider=provider,
            provider_name=settings.model_provider,
            settings=settings,
            sessionmaker=sessionmaker,
            store=store,
            tenant=tenant,
        )
    except ProviderUnavailableError as exc:
        # Verbatim where the provider explained itself (a redirect names the
        # address to configure, a refused credential names the setting), and
        # the generic line only where it did not.
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc


@router.get("/models/residency")
async def model_residency(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    tenant: AdminDep,
) -> ModelResidencyResponse:
    """Return what the serving runtime is holding in memory right now.

    The live counterpart to the management view's declared and measured facts:
    that one says what a model IS, this one says what is loaded this second.
    Read straight off the seam's residency view - the same one the verify job
    asserts GPU residency with - and reported exactly as the runtime gave it,
    including a model nothing here describes.

    A runtime that does not report residency answers ``reported: false`` with an
    empty list rather than an error: not knowing is a normal state for a backend
    that loads one model at process start, and it is emphatically not the same
    answer as "nothing is loaded". An unreachable runtime lands there too, since
    the seam's residency hooks are best-effort by design and a page that fails
    over one small widget would be a worse page.

    ADMIN-ONLY, gated with the management view above and for the same reason:
    what the shared model server is holding right now is operator information
    about the deployment, not about the caller's own work.
    """
    readings = await provider.resident_models()
    if readings is None:
        return ModelResidencyResponse(reported=False, models=[])
    return ModelResidencyResponse(
        reported=True,
        models=[
            ResidentModel(
                model=reading.model,
                total_bytes=reading.total_bytes,
                vram_bytes=reading.vram_bytes,
                gpu_fraction=(
                    None if reading.gpu_fraction is None else round(reading.gpu_fraction, 3)
                ),
            )
            for reading in readings
        ],
    )


@router.post("/models/preflight")
async def preflight_model(
    req: ModelNameRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    registry: Annotated[OllamaRegistryClient, Depends(get_model_registry_client)],
    tenant: AdminDep,
) -> ModelPreflightResponse:
    """Ask the registry whether a name exists and what it would cost to download.

    On demand only - nothing about rendering a list calls this. Admin-gated with
    the verb it precedes: it is the endpoint that makes the backend issue an
    outbound request, and it would be odd for a caller to be able to shop the
    registry but not to download. A malformed or unsafe name is a 422 before any
    request leaves the process, and a backend that cannot download models at all
    is a 400 rather than an answer about a tag it could never fetch; every other
    outcome, including an unreachable registry, is a 200 whose ``status`` says
    honestly how much is known.
    """
    return await model_pull.preflight(name=req.name, provider=provider, client=registry)


@dataclass(frozen=True)
class PullClaim:
    """A claimed download: the tag, and what to do once it lands."""

    tag: str
    verify: bool


@dataclass(frozen=True)
class VerifyClaim:
    """A claimed probe slot: the tag to measure, and how thoroughly."""

    tag: str
    deep: bool


# One event type, one frame name, on every stream that carries it. The
# verification frames keep their own names rather than reusing progress/done/
# error, because a download can continue into a check on ONE connection and two
# differently-shaped ``done`` frames there would be unreadable.
_FRAME_NAMES: dict[type, str] = {
    PullProgressEvent: "progress",
    PullDoneEvent: "done",
    PullErrorEvent: "error",
    VerifyProgressEvent: "verify_progress",
    VerifyDoneEvent: "verify_done",
    VerifyErrorEvent: "verify_error",
}


async def _frames(events: AsyncIterator[BaseModel]) -> AsyncIterator[str]:
    """Map a service's typed events onto SSE frames, terminal event last.

    ``aclosing`` so an early client hang-up closes the service generator - and
    with it the connection to the model server, and the probe's re-warm -
    promptly, rather than leaving them to garbage collection.
    """
    async with aclosing(events) as source:
        async for event in source:
            yield sse_event(_FRAME_NAMES[type(event)], event)


async def hold_pull_claim(
    req: ModelPullRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    tenant: AdminDep,
) -> AsyncIterator[PullClaim]:
    """Claim the tag for the request's full lifetime, and yield it.

    A yield dependency releases in the request exit stack, which unwinds only
    after the response body finishes or aborts - the one hook that also covers
    a client disconnecting before the response even starts. Releasing inside
    the SSE generator instead would wedge the tag on such a disconnect: the
    body is never iterated, and a never-started generator skips its ``finally``
    forever, so that model could not be downloaded again until a restart. Same
    reasoning, and the same shape, as the chat route's stream slot.

    Raises:
        ForbiddenError: The caller does not hold the admin role.
        UnprocessableEntityError: The name is not a usable model name (422).
        BadRequestError: The backend cannot download models at all (400).
        ConflictError: The same model is already downloading (409).
    """
    tag = model_pull.begin_pull(provider, name=req.name)
    # Pre-OIDC the actor is a constant, and recording it anyway is the point:
    # the trail becomes meaningful the moment real identity populates it.
    logger.info(
        "model pull started by tenant=%s user=%s: model=%r verify=%s",
        tenant.tenant_id,
        tenant.user_id,
        tag,
        req.verify,
    )
    try:
        yield PullClaim(tag=tag, verify=req.verify)
    finally:
        model_pull.release_pull(tag)


async def hold_verify_claim(
    req: ModelVerifyRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    tenant: AdminDep,
) -> AsyncIterator[VerifyClaim]:
    """Claim the single probe slot for the request's lifetime, and yield it.

    Same shape and the same reasoning as the download's claim, and the stakes
    are higher: a wedged probe slot would refuse every later check until a
    restart, and this one is a single slot for the whole process rather than
    one per tag.

    Raises:
        ForbiddenError: The caller does not hold the admin role.
        UnprocessableEntityError: The name is not a usable model name (422).
        BadRequestError: The runtime reports this tag cannot chat (400).
        NotFoundError: The backend does not hold these weights (404).
        ConflictError: A probe is already running (409).
        ServiceUnavailableError: The model server cannot be reached (503).
    """
    try:
        tag = await model_verify.begin_verification(
            provider, provider_name=settings.model_provider, name=req.name
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    logger.info(
        "model verify started by tenant=%s user=%s: model=%r depth=%s",
        tenant.tenant_id,
        tenant.user_id,
        tag,
        req.depth,
    )
    try:
        yield VerifyClaim(tag=tag, deep=req.depth == "deep")
    finally:
        model_verify.release_verification(tag)


@router.post("/models/pull")
async def pull_model(
    claim: Annotated[PullClaim, Depends(hold_pull_claim)],
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> StreamingResponse:
    """Download a model, streaming its progress as SSE (progress, done, error).

    Everything a download can be refused for is decided before any work, so a
    refusal arrives as problem+json rather than as an event on a stream a
    progress bar is already attached to: a malformed name (422), a backend with
    no download verb (400), and a second download of a tag already being
    fetched (409). Failures after that are in-band ``error`` events, because by
    then the connection IS a progress stream.

    A successful download continues into the fast tool check unless the body
    turns it off, on the same stream - see :mod:`app.schemas.models` for how
    the ``done`` frame keeps the two outcomes separate.
    """
    events = model_verify.stream_pull_with_check(
        provider=provider,
        settings=settings,
        sessionmaker=sessionmaker,
        tenant=tenant,
        tag=claim.tag,
        verify=claim.verify,
    )
    return StreamingResponse(
        with_heartbeat(_frames(events), settings.sse_heartbeat_seconds),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/models/verify")
async def verify_model(
    claim: Annotated[VerifyClaim, Depends(hold_verify_claim)],
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> StreamingResponse:
    """Measure a model's tool-calling, streaming each finished sample as SSE.

    Streamed rather than answered, because a deep check runs twenty-six
    exchanges over several minutes and a silent POST for that long is
    indistinguishable from a hang. Refusals are decided before the response
    begins and arrive as problem+json: 422 (unusable name), 404 (the backend
    does not hold those weights), 409 (a probe is already running), 403 (not an
    admin), 503 (the model server is unreachable). Everything after that is an
    in-band ``verify_error`` frame.

    The run evicts the chat model and the embedder to measure a fresh load, and
    re-warms both afterwards whatever its outcome - see
    :mod:`app.services.model_verify` for the full order and why it is that one.
    """
    events = model_verify.stream_verification(
        provider=provider,
        settings=settings,
        sessionmaker=sessionmaker,
        tenant=tenant,
        model=claim.tag,
        deep=claim.deep,
    )
    return StreamingResponse(
        with_heartbeat(_frames(events), settings.sse_heartbeat_seconds),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@dataclass(frozen=True)
class EmbedderCheckClaim:
    """A claimed embedder check: the tag to measure."""

    tag: str


async def hold_embedder_check_claim(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    tenant: AdminDep,
    req: EmbedderCheckRequest | None = None,
) -> AsyncIterator[EmbedderCheckClaim]:
    """Claim the probe slots for the request's lifetime, and yield the target.

    Same shape and the same reasoning as the download's and the tool check's
    claims: a yield dependency releases in the request exit stack, which
    unwinds after the response finishes or aborts, so a client that hangs up
    cannot wedge a slot until a restart.

    The ACTIVE embedder is resolved here rather than inside the service,
    because the chain that answers it needs a workspace and a vector store and
    the service is meant to hold neither. It decides two things: which tag a
    bare request measures, and whether the run afterwards has to put the
    deployment's own embedder back.

    Raises:
        ForbiddenError: The caller does not hold the admin role.
        UnprocessableEntityError: The name is not a usable model name (422).
        BadRequestError: The model server reports this tag cannot embed, or
            nothing was named and nothing resolves an embedder (400).
        NotFoundError: The backend does not hold those weights (404).
        ConflictError: A check is already running, or a tool check is (409).
        ServiceUnavailableError: The model server cannot be reached (503).
    """
    body = req or EmbedderCheckRequest()
    try:
        active = await resolve_embedding(
            settings=settings,
            stored=await load_defaults(session, tenant),
            store=store,
            provider=provider,
            provider_name=settings.model_provider,
        )
        tag = await embedder_verify.begin_embedder_check(
            provider,
            provider_name=settings.model_provider,
            active=active.model,
            name=body.name,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    logger.info(
        "embedder check started by tenant=%s user=%s: model=%r",
        tenant.tenant_id,
        tenant.user_id,
        tag,
    )
    try:
        yield EmbedderCheckClaim(tag=tag)
    finally:
        embedder_verify.release_probe_slots(tag)


@router.post("/models/embedder-check")
async def embedder_check(
    claim: Annotated[EmbedderCheckClaim, Depends(hold_embedder_check_claim)],
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> EmbedderCheckResponse:
    """Measure an embedder's relevance bands and judge the retrieval floors.

    The retrieval-side sibling of the tool check, and a plain JSON answer
    rather than a stream: it embeds a fixed corpus of query/answer/distractor
    triplets, both with and without the query prefix under test, and reports
    the two score bands that came out. Seconds, not minutes - which is why
    there is nothing to stream.

    The answer either VALIDATES the floors already declared for this embedder
    or PROPOSES a pair derived from the measured bands, with the derivation
    spelled out. Nothing is applied: ``RAG_MIN_SCORE`` and ``RAG_REL_MARGIN``
    decide what every answer in the deployment may cite, so a number that
    changes them is a person's decision, not a probe's.

    Errors map to problem+json: 403 (not an admin), 422 (unusable name), 400
    (the tag is a declared non-embedding model, or no embedder is configured
    and none was named), 404 (the backend does not hold those weights), 409 (a
    check or a tool check is already running), 503 (the model server is
    unreachable, or it answered with vectors no score could be matched to).
    """
    try:
        outcome = await embedder_verify.run_embedder_check_job(
            provider=provider,
            settings=settings,
            sessionmaker=sessionmaker,
            tenant=tenant,
            model=claim.tag,
        )
    except ProviderUnavailableError as exc:
        # The embedding backend collapses "model would not load" and "server
        # unreachable" into one error whose message already names the fix, so
        # it is carried through verbatim rather than replaced.
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    except ValueError as exc:
        # The harness raises this when the backend answers with a vector count
        # or a dimension no passage could be matched to. That is the model
        # server misbehaving rather than a bad request, and it must not reach
        # the client as a bare 500 the endpoint never documented.
        raise ServiceUnavailableError(str(exc)) from exc
    return embedder_verify.build_report(outcome, active=outcome.active)


@router.post("/models/think-mode")
async def confirm_think_mode(
    req: ModelNameRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> ModelThinkModeResponse:
    """Store what a tool check measured about this model's think flag.

    The confirmation step of the think check: a check watches how the weights
    treat the flag, and this is where an operator agrees that the measurement
    is how the deployment should route the model from now on. The stored value
    is the MEASURED one - the body names a model and nothing else, so a caller
    cannot declare a mode nothing observed.

    Admin-gated with the other verbs that act. Errors map to problem+json: 403
    (not an admin), 422 (the name is not a usable model name, or no check has
    measured the installed weights - confirming without evidence is refused),
    404 (the backend does not hold these weights), 503 (the model server is
    unreachable).
    """
    tag = require_model_name(req.name).text
    try:
        stored = await think_check.confirm_think_mode(
            provider=provider,
            provider_name=settings.model_provider,
            sessionmaker=sessionmaker,
            tenant=tenant,
            model=tag,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    return ModelThinkModeResponse(
        model=stored.model,
        think_mode=stored.think_mode,
        digest=stored.digest,
        probe_id=stored.probe_id,
    )


@router.post("/models/ceiling")
async def declare_ceiling(
    req: ModelCeilingRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> ModelCeilingResponse:
    """Declare how much this model may be allowed to drive.

    The console's writer for the ceiling axis. A declaration is a CEILING and
    only a ceiling: raising it withdraws the human objection and nothing more,
    because ``min(CEILING, EARNED)`` still holds the model at whatever a
    digest-bound deep check has proven on the server being served from. So an
    ``agentic`` declaration on an unproven model is STORED and takes effect the
    moment a deep check passes - which is why the response reports ``earned``
    and ``effective`` beside the value written, rather than echoing the write
    and letting a page claim a change that did not happen.

    A DOWN-clamp is the opposite: ``min()`` needs no evidence to lower
    something, so it applies to the next turn.

    Admin-gated with the other verbs that act. Errors map to problem+json: 403
    (not an admin), 422 (the name is not a usable model name, the mode is not
    one this build knows, or the runtime exposes no weights digest to bind a
    claim about weights to), 404 (the backend does not hold these weights), 503
    (the model server is unreachable).
    """
    try:
        listing = await provider.list_models()
        digests = {info.name: info.digest for info in listing if info.digest}
        report = await report_for(provider, settings.model_provider, req.name)
        # Read BEFORE the write, because this is provenance: the measurement
        # the operator was looking at when they decided. Reading it afterwards
        # would record whatever the log holds once the decision is already made,
        # which is a different (and less useful) claim.
        # Resolved once and handed to the read below: it carries the ceiling AND
        # the stored pairing the reasoning binding resolves against, and reading
        # it twice would be two chances for one model to get two answers.
        prior_policy = await policy_for(
            provider_name=settings.model_provider,
            model=req.name,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        prior = await explain_earned(
            provider_name=settings.model_provider,
            model=req.name,
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=sessionmaker,
            digests=digests,
            serving_endpoint=provider.serving_endpoint(),
            effective_ceiling=effective_agency_ceiling(report, policy=prior_policy),
            policy=prior_policy,
        )
        # The deepest verdict bound to these weights, which is the row an
        # agentic decision actually rests on; the fast one where that is all
        # there is. None is legitimate - a down-clamp needs no evidence.
        relied_on = next(
            (view.probe_id for view in (prior.deep, prior.fast) if view is not None), None
        )
        stored = await model_declare.declare_ceiling(
            provider=provider,
            provider_name=settings.model_provider,
            sessionmaker=sessionmaker,
            tenant=tenant,
            model=req.name,
            ceiling=req.ceiling,
            probe=relied_on,
        )
        # Re-resolved from the store rather than predicted, and through the
        # SAME functions the management view uses: a response that computed the
        # effect its own way would be a second answer about one model.
        policy = await policy_for(
            provider_name=settings.model_provider,
            model=stored.model,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        ceiling = effective_agency_ceiling(report, policy=policy)
        explanation = await explain_earned(
            provider_name=settings.model_provider,
            model=stored.model,
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=sessionmaker,
            digests=digests,
            serving_endpoint=provider.serving_endpoint(),
            effective_ceiling=ceiling,
            policy=policy,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    return ModelCeilingResponse(
        model=stored.model,
        ceiling=stored.ceiling,
        digest=stored.digest,
        earned=explanation.earned,
        effective=resolve_agency(
            ceiling=ceiling,
            earned=explanation.earned,
            env_mode=parse_agency_mode(settings.agency_mode),
            runtime_floors=(() if effective_tools_mode(report) == "native" else ("deterministic",)),
        ),
    )


@router.post("/models/thinking-with-tools")
async def declare_thinking_with_tools(
    req: ModelThinkingWithToolsRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> ModelThinkingWithToolsResponse:
    """Declare whether this model may reason and call tools in the same turn.

    The console's writer for the capability axis, gated exactly as the ceiling's
    ``agentic`` rung is and by the same mechanism - the fold, never a refusal
    here. A GRANT is stored whatever the log holds and reaches a turn only once
    a deep check has passed the reasoning-with-tools case on these weights, at
    the running harness version and on the host being served from; a re-pull, a
    harness edit or a move to another model server retires it through that same
    fold. So the response reports ``effective`` beside what was written, rather
    than echoing the write and letting a page claim a capability the model was
    not given.

    A REVOCATION is the opposite and applies to the next turn: the failure this
    axis guards is silent - reasoning plus tools on an incapable pair returns an
    EMPTY answer, not an error - so turning it off must never wait for evidence.

    Admin-gated with the other verbs that act. Errors map to problem+json: 403
    (not an admin), 422 (the name is not a usable model name, or the runtime
    exposes no weights digest to bind a claim about weights to), 404 (the
    backend does not hold these weights), 503 (the model server is
    unreachable).
    """
    try:
        listing = await provider.list_models()
        digests = {info.name: info.digest for info in listing if info.digest}
        report = await report_for(provider, settings.model_provider, req.name)
        # Read BEFORE the write, on the ceiling endpoint's own rule: this is
        # provenance - the measurement the operator was reading when they
        # decided - and reading it afterwards would record whatever the log
        # holds once the decision is already made.
        # Read once for both rungs it feeds, exactly as the ceiling endpoint
        # reads it: the ceiling handed to the explanation, and the stored
        # pairing the reasoning binding resolves against.
        prior_policy = await policy_for(
            provider_name=settings.model_provider,
            model=req.name,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        prior = await explain_earned(
            provider_name=settings.model_provider,
            model=req.name,
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=sessionmaker,
            digests=digests,
            serving_endpoint=provider.serving_endpoint(),
            effective_ceiling=effective_agency_ceiling(report, policy=prior_policy),
            policy=prior_policy,
        )
        # Only the DEEP row is provenance here: it is the sole depth that runs
        # the reasoning-with-tools case at all, so naming a fast row would
        # record a measurement that decided nothing about this axis.
        stored = await model_declare.declare_thinking_with_tools(
            provider=provider,
            provider_name=settings.model_provider,
            sessionmaker=sessionmaker,
            tenant=tenant,
            model=req.name,
            thinking_with_tools=req.thinking_with_tools,
            probe=prior.deep.probe_id if prior.deep is not None else None,
        )
        # Re-resolved from the store rather than predicted, and through the SAME
        # function the management view uses: a response that computed the effect
        # its own way would be a second answer about one model.
        policy = await policy_for(
            provider_name=settings.model_provider,
            model=stored.model,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        explanation = await explain_earned(
            provider_name=settings.model_provider,
            model=stored.model,
            # Carries the environment rung of the sibling resolution; the stored
            # pairing rung rides in on ``policy`` below. Between them the
            # binding lands on the tag a TURN routes to, which is the same tag
            # the verify job drove - resolve it any other way here and this
            # response would report a capability the chat path withholds.
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=sessionmaker,
            digests=digests,
            serving_endpoint=provider.serving_endpoint(),
            effective_ceiling=effective_agency_ceiling(report, policy=policy),
            policy=policy,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    return ModelThinkingWithToolsResponse(
        model=stored.model,
        thinking_with_tools=stored.thinking_with_tools,
        digest=stored.digest,
        effective=explanation.thinking_with_tools,
    )


@router.post("/models/thinking-pair")
async def declare_thinking_pair(
    req: ModelThinkingPairRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: AdminDep,
) -> ModelThinkingPairResponse:
    """Pair a direct-answer model with the partner that serves its thinking turns.

    A PREFERENCE axis: it says how this deployment wants two tags used, not
    what either checkpoint is, so it follows the tags across a re-pull rather
    than retiring with the weights.

    The partner is refused unless its think mode already RESOLVES to ``always``
    or ``hybrid`` - through the one chain a turn and the management card both
    read, never through a declaration invented here. Pairing two direct-answer
    models would produce a thinking toggle that quietly changed nothing, and an
    unmeasured partner is that same offer made on no evidence.

    Admin-gated with the other verbs that act. Errors map to problem+json: 403
    (not an admin), 422 (either name is unusable, the pair names one tag twice,
    or nothing establishes that the partner can reason), 404 (the backend does
    not hold one of the two), 503 (the model server is unreachable).
    """
    try:
        listing = await provider.list_models()
        digests = {info.name: info.digest for info in listing if info.digest}
        partner_tag = require_model_name(req.thinking_model).text
        partner_report = await report_for(provider, settings.model_provider, partner_tag)
        partner_think = await think_check.resolve_think_mode(
            provider_name=settings.model_provider,
            model=partner_tag,
            report=partner_report,
            digests=digests,
            sessionmaker=sessionmaker,
        )
        stored = await model_declare.declare_thinking_pair(
            provider=provider,
            provider_name=settings.model_provider,
            sessionmaker=sessionmaker,
            tenant=tenant,
            model=req.name,
            thinking_model=req.thinking_model,
            partner_think_mode=partner_think,
            # Whether the partner can ANSWER, which the think mode says nothing
            # about. Resolved through the same rule the management grid groups
            # rows by, so the console's candidate list and this refusal cannot
            # disagree about what "serves chat" means.
            partner_serves_chat=serves_chat_role(partner_report),
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    return ModelThinkingPairResponse(model=stored.model, thinking_model=stored.thinking_model)


@router.post("/models/default")
async def set_default_model(
    req: ModelNameRequest,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant: AdminDep,
) -> ModelDefaultResponse:
    """Make this model the workspace default every unpinned conversation follows.

    The console's counterpart to the successor a delete already writes, and it
    validates the same way: the tag has to be installed and has to serve a role
    that can drive a chat turn. A default nothing can route to is the worst
    kind of stored setting - it is accepted silently and surfaces one turn
    later, on somebody else's question, as a substitution note.

    Admin-gated with the other verbs that act. Errors map to problem+json: 403
    (not an admin), 422 (the name is not a usable model name), 404 (the backend
    does not hold these weights), 409 (the tag serves a locked role), 503 (the
    model server is unreachable).
    """
    try:
        stored = await workspace_defaults.set_validated_default_model(
            provider=provider,
            provider_name=settings.model_provider,
            session=session,
            tenant=tenant,
            name=req.name,
        )
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    # Committed before the response leaves, not left to the dependency's safety
    # net: the page refetches the manager view immediately, and that read has to
    # see the new default.
    await session.commit()
    return ModelDefaultResponse(model=stored)


@router.delete("/models/{name:path}")
async def delete_model(
    name: str,
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    store: Annotated[VectorStore, Depends(get_vector_store)],
    tenant: AdminDep,
    replacement: Annotated[str | None, Query(max_length=MAX_NAME_LENGTH)] = None,
) -> ModelDeleteResponse:
    """Remove a downloaded model, moving the stored state that named it.

    Conversation pins on the tag are cleared in the same transaction (those
    threads follow the workspace default), and deleting the workspace default
    itself requires ``replacement`` - the validated successor written before
    the weights go. The response says what changed; every refusal explains
    itself.

    Errors map to problem+json: 403 (not an admin), 422 (the name or the
    replacement is not a usable model name), 400 (the backend cannot remove
    models at all), 404 (it does not hold this one), 409 (the role is locked,
    an embedding model is still active or referenced by stored passages, the
    tag is the deployment's own chat pair, or it is the workspace default
    without a usable successor), 503 (the model server is unreachable).
    """
    try:
        outcome = await model_delete.delete_model(
            provider=provider,
            settings=settings,
            session=session,
            sessionmaker=sessionmaker,
            tenant=tenant,
            store=store,
            name=name,
            replacement=replacement,
        )
    except ModelNotFoundError as exc:
        # Lost a race with another caller: the listing said it was here.
        raise NotFoundError(str(exc)) from exc
    except ProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc) or "Cannot reach the model server.") from exc
    # Committed before the response leaves, not left to the dependency's
    # safety net: the page refetches the manager view immediately, and that
    # read must see the cleared pins and the successor default.
    await session.commit()
    return ModelDeleteResponse(
        removed=outcome.removed,
        unpinned_conversations=outcome.unpinned_conversations,
        new_default=outcome.new_default,
    )


@router.post("/models/warm")
async def warm(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    tenant: Annotated[TenantContext, Depends(get_current_tenant)],
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    embedder: Annotated[Embedder | None, Depends(get_embedder)],
    reranker: Annotated[Reranker | None, Depends(get_reranker)],
    search_provider: Annotated[SearchProvider | None, Depends(get_search_provider)],
    page_fetcher: Annotated[PageFetcher | None, Depends(get_page_fetcher)],
    mcp_sources: Annotated[list[ToolSource], Depends(get_mcp_sources)],
    tool_vector_cache: Annotated[ToolVectorCache, Depends(get_tool_vector_cache)],
    req: WarmRequest | None = None,
) -> WarmResponse:
    """Load a chat model and prime its prompt cache with the system prompt.

    Runs a one-token generation through the real chat prompt so the FIRST
    real turn does not pay the model load plus the system-prompt evaluation
    (tens of seconds on small, partially-offloaded GPUs). With no ``model``
    in the body, the warmed default resolves through the same stored chain a
    chat turn uses - workspace default, then env override, then profile
    binding - including the missing-model substitution, so what gets primed
    is what a default turn would actually run (a remote server rarely holds
    the local profile's tags).

    "The real chat prompt" includes the TOOL BLOCK a tools-offered turn opens
    with. A chat template renders the offered tool list into its system region,
    so on a deployment with tools registered the block is most of the standing
    prefix - and a prompt cache is prefix-matched, so a warm that omitted it
    primed a head no real turn ever sends. The same collaborators a chat turn
    is handed are therefore resolved here, and the reported ``prompt_tokens``
    is the size of the prefix that was actually primed.

    "The same stored chain" now covers the THINKING default as well, and not
    as a refinement: a workspace that reasons by default routes its turns to a
    different tag on a paired profile, so warming the pair's default member
    would prime weights no turn runs, and the first question would evict them
    to load the ones it needs. The window ceiling rides along for the
    mechanically similar reason - a differently sized request restarts the
    runner. Blocks until the warm generation finishes; `make warm` is the
    intended caller.

    A ``model`` that IS named does not have to be the exact tag. It is
    resolved against what the server holds: an exact tag wins (the
    ``:latest`` and publisher-namespace spellings are equivalent, and case is
    not identity), otherwise a fragment matching exactly one installed tag
    resolves to it, and the response names whichever model was warmed. A
    fragment matching several is refused with the candidates rather than
    guessed through, and one reached by fragment is also refused when it
    cannot serve a chat turn - an exact tag is honoured either way, since only
    the inferred half can be wrong about which model was meant. A server whose
    inventory cannot be read resolves nothing and takes the name as given.

    Errors map to problem+json: 404 (model not pulled, no installed model
    matches the name, or nothing on the server can chat), 400 (the name
    matched several installed models or resolved to one that cannot chat -
    both decided before the model server is called - or the model server
    rejected the request), 503 (model server unreachable).
    """
    body = req or WarmRequest()
    # Read unconditionally, unlike the model tier it used to serve alone: the
    # context ceiling shapes the window an EXPLICITLY named model is warmed
    # with too, and a warm that primed a different window than the turn asks
    # for makes the runtime reload the model - the one thing a warm exists to
    # avoid. The stored model still only participates when nothing was named.
    defaults = await load_defaults(session, tenant)
    workspace_model = None if (body.model or "").strip() else defaults.model
    model, took_ms, prompt_tokens = await warm_model(
        provider,
        settings,
        model=body.model,
        workspace_model=workspace_model,
        # Unconditional like the ceiling: which member of a routed pair a turn
        # runs on, and which think flag it carries, follow the stored default
        # whether or not this request named a model.
        thinking=defaults.thinking,
        # And the web default with them: it decides whether the web_search
        # schema is in the block a turn sends, which is prefix, not behaviour.
        web_search=defaults.web_search,
        # The persona this workspace's turns actually open with. Unconditional
        # too: it is prefix whatever model was named.
        workspace_prompt=defaults.system_prompt,
        context_ceiling=defaults.context_ceiling,
        keep_alive=body.keep_alive,
        # So the warm normalizes the think flag through the same stored
        # declarations a turn reads; a warm that primed a different flag would
        # prime a prefix the turn never reuses.
        sessionmaker=sessionmaker,
        tool_prefix=WarmToolPrefix(
            tenant=tenant,
            wiring=ToolWiring(
                settings=settings,
                search_provider=search_provider,
                page_fetcher=page_fetcher,
                embedder=embedder,
                vector_store=vector_store,
                # Inert for the PREFIX - both local tools' specs are static
                # literals and a warm never runs one, so no reranker choice can
                # move a byte of what is primed. Carried, with the workspace
                # toggle applied, so this wiring IS the turn's rather than a
                # near-copy of it: the day a tool's advertised spec depends on
                # its collaborators, a warm assembled from a different set
                # would drift silently, which is the whole failure this change
                # exists to end.
                reranker=apply_reranker_default(reranker, defaults),
                mcp_sources=tuple(mcp_sources),
            ),
            vector_cache=tool_vector_cache,
        ),
    )
    return WarmResponse(model=model, took_ms=took_ms, prompt_tokens=prompt_tokens)


@router.post("/models/warm-embed")
async def warm_embed(
    provider: Annotated[ModelProvider, Depends(get_model_provider)],
    settings: Annotated[Settings, Depends(get_settings)],
    session: Annotated[AsyncSession, Depends(get_session)],
    tenant: TenantDep,
    vector_store: Annotated[VectorStore, Depends(get_vector_store)],
    req: WarmRequest | None = None,
) -> WarmResponse:
    """Load this workspace's embedding model into the model server.

    Primes the embedding model so the first ingest or retrieval of a session
    does not pay a multi-second cold load (the stall a first `/rag/ingest`
    otherwise shows). Which model that is is not a caller's choice - it defines
    the vector space - so ``model`` in the body is ignored here and the same
    resolution a retrieval runs through decides; ``keep_alive`` is honored.
    Errors map to problem+json: 404 (nothing resolves an embedding model), 503
    (model not pulled or the model server is unreachable). ``prompt_tokens`` is
    always null - an embedder has no prompt-prefix cache to report.
    """
    body = req or WarmRequest()
    choice = await resolve_embedding(
        settings=settings,
        stored=await load_defaults(session, tenant),
        store=vector_store,
    )
    model, took_ms = await warm_embedder(
        provider, settings, embedding=choice.config, keep_alive=body.keep_alive
    )
    return WarmResponse(model=model, took_ms=took_ms)


@router.post("/models/warm-search")
async def warm_search_endpoint(
    provider: Annotated[SearchProvider | None, Depends(get_search_provider)],
    tenant: TenantDep,
) -> WarmSearchResponse:
    """Prime the web-search backend so the first real search of a session is fast.

    Issues one throwaway query to SearXNG so the DNS cache, connection pool, and
    lazy-loaded workers are warm before a user's first web-search turn - the
    turn that otherwise times out on a cold fan-out ("first search fails, second
    works"). Errors map to problem+json: 404 when web search is not configured
    (``SEARCH_PROVIDER=none``, or ``SEARXNG_URL`` unset), 503 when the search
    backend is unreachable. Called by `make warm`; also fired in the background
    at startup.
    """
    if provider is None:
        # Two ways to have no provider, and naming the wrong one sends an
        # operator to edit a URL that was never the reason.
        raise NotFoundError(
            "Web search is not configured; set SEARCH_PROVIDER=searxng and SEARXNG_URL "
            "to enable it."
        )
    took_ms, results = await warm_search(provider)
    return WarmSearchResponse(took_ms=took_ms, results=results)
