"""Model warmup: load a chat model AND prime its prompt-prefix cache.

Loading weights is not enough to make the first reply fast. The standing
system prompt runs to a thousand tokens or more once a deployment's tools are
advertised in it, and on a partially-offloaded model (small GPUs) evaluating
it can take tens of seconds - a cost the model server pays once and then
serves from its prompt cache. A bare load-only request (e.g. Ollama's empty
/api/generate) skips that evaluation entirely, so the FIRST real turn still
stalls. Warming therefore runs a real one-token chat turn through the exact
prompt a normal turn opens with: the shared prefix lands in the server's cache
and later turns only evaluate their own user message.

"The exact prompt" is literal, and it is why this module reaches as far into
the turn's own machinery as it does. A prompt cache is prefix-MATCHED, so the
warm's head has to be byte-identical to the turn's or it buys nothing at all -
and a chat template renders the offered TOOL LIST into its system region, so
the tool block is part of that head. The warm therefore plans the turn with
:func:`~app.services.turn_plan.plan_turn` and builds its tool set with
:func:`~app.services.tools.offer.build_offered_tools`, the same two callers
``chat_service`` uses, rather than composing a lookalike of its own.
"""

from __future__ import annotations

import time
from contextlib import aclosing
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import BadRequestError, NotFoundError, ServiceUnavailableError
from app.core.tenancy import TenantContext
from app.providers.base import (
    ChatMessage,
    EmbedOptions,
    GenerationOptions,
    ModelCapabilities,
    ModelInfo,
    ModelProvider,
    ProviderUnavailableError,
)
from app.providers.search.base import SearchProvider, SearchProviderUnavailableError
from app.providers.tools.base import Tool, ToolContext
from app.schemas.chat import ChatRequest
from app.schemas.settings import SettingMode, TurnDefaults
from app.services.earned import (
    EffectiveVerdicts,
    declared_thinking_with_tools,
    effective_verdicts,
    measured_think_mode,
)
from app.services.embedding_resolution import UNCONFIGURED_DETAIL, resolve_embedding
from app.services.model_facts import (
    resolve_num_ctx,
    runtime_capabilities,
)
from app.services.model_manager import serves_chat_role
from app.services.model_policy import PolicyView, resolve_policy
from app.services.model_resolution import (
    AgencyMode,
    ChatModelPair,
    EmbeddingConfig,
    ThinkMode,
    resolve_chat_models,
)
from app.services.model_routing import (
    NoChatModelConfiguredError,
    match_installed_name,
    route_around_missing,
)
from app.services.prompts import append_tool_guidance, compose_system_prompt
from app.services.tools.offer import (
    ToolWiring,
    build_offered_tools,
    guidance_shape,
    tool_result_room,
    tools_to_wire,
)
from app.services.tools.selection import ToolVectorCache
from app.services.turn_plan import TurnPlan, plan_turn

# Any short user turn works: the cache key is the shared token prefix BEFORE
# the user content, which is identical for every turn of the same model.
_WARM_USER_MESSAGE = "Hi"

# The warm prime advertises itself as confirmation-capable, because the client
# whose first turn this exists to speed up is the web UI, which does. The flag
# decides whether SIDE-EFFECTING tools are offered at all, so a warm that said
# "no" would prime a shorter tool block than the browser's turns carry and miss
# their cache by exactly the width of those schemas. A client that genuinely
# cannot confirm (a bare curl turn) gets its own, smaller prefix and pays one
# prefill for it - the right way round, since it is not the client anyone warms
# for.
_WARM_SUPPORTS_CONFIRMATION = True


@dataclass(frozen=True)
class WarmToolPrefix:
    """What a warm needs to compose the tool half of a turn's prompt prefix.

    Optional as a whole: a caller holding none of this (the tool-check
    re-warm, a test) warms exactly as warming worked before the tool block
    existed - a shorter prefix than a tools-offered turn opens with, so the
    first such turn pays its own prefill.

    Attributes:
        tenant: The principal the tool sources are listed for. A tool source
            may offer different tools per tenant, so there is no tenant-free
            way to ask what a turn would be handed.
        wiring: The live collaborators the tools are built from.
        vector_cache: The process-wide tool-description vector cache, when the
            caller holds one. Unused below the selection threshold, which is
            every deployment that has not registered a crowd of MCP servers.
    """

    tenant: TenantContext
    wiring: ToolWiring
    vector_cache: ToolVectorCache | None = None


# A trivial input is enough to pull the embedding weights into the model server;
# unlike chat there is no prompt-prefix cache to prime, so the content is moot.
_WARM_EMBED_INPUT = "warm"

# A throwaway search query: only the round-trip matters (it primes SearXNG's DNS
# cache, connection pool, and lazy-loaded workers), so the text is irrelevant.
_WARM_SEARCH_QUERY = "warm"


async def _plan_warm_turn(
    provider: ModelProvider,
    settings: Settings,
    *,
    model: str | None,
    pair: ChatModelPair,
    listing: list[ModelInfo] | None,
    thinking: SettingMode | None,
    web_search: SettingMode | None,
    context_ceiling: int | None,
    sessionmaker: async_sessionmaker[AsyncSession] | None,
    tool_prefix: WarmToolPrefix | None,
) -> TurnPlan:
    """Plan the turn this warm stands in for, through the chat path's planner.

    The request is the plainest one a deployment can produce: no per-request
    switches, the tenant's stored defaults underneath, and a model named only
    when somebody named one - which is what an unprompted next turn looks like.

    Model ROUTING is left to the planner rather than decided before it, and
    that is load-bearing rather than tidy. The planner routes on the ARBITRATED
    think decision, not the raw heuristic, because an agentic turn that yielded
    its reasoning to keep its tools must run on the direct-answer member of a
    pair. A warm that picked the tag from the heuristic and then took its think
    flag from the plan would be deciding one turn's weights and another turn's
    flag - agreeing today only because the warm's own message provokes no
    reasoning, and loading the wrong model outright the day that changes.

    Args:
        provider: The active model provider.
        settings: The active application settings.
        model: The tag somebody explicitly named, already resolved against the
            inventory; None warms the deployment's own default, which the
            planner then routes exactly as a turn's is routed.
        pair: The tier's (default, thinking) tags.
        listing: The provider inventory this warm already took, or None when it
            could not be read. Handed to the router so the warm lists once.
        thinking: The tenant's stored thinking default, or None.
        web_search: The tenant's stored web-search default, or None.
        context_ceiling: The tenant's stored window ceiling, or None.
        sessionmaker: Session factory for the stored declarations and verdicts,
            or None when the caller has no database.
        tool_prefix: The tool wiring, or None when this warm carries no tools.

    Returns:
        The :class:`~app.services.turn_plan.TurnPlan` the prime is composed from.

    Raises:
        NoChatModelConfiguredError: The inventory is known to hold nothing that
            can chat. Raised by the router, exactly as it is on a turn.
    """
    wiring = tool_prefix.wiring if tool_prefix is not None else None

    async def _fetch_verdicts(tag: str, ceiling: AgencyMode) -> EffectiveVerdicts:
        if sessionmaker is None:
            # Defensive rather than live: the planner asks for verdicts only
            # when a tool could reach the model, and a caller with no database
            # is a caller with no tool wiring either, so it floors to
            # deterministic before reaching here. Kept because the seam takes
            # a fetcher, not an optional one, and answering with the declared
            # values is the honest reading of "nothing was measured" - the
            # same pair the planner would have stood in had it never asked.
            return EffectiveVerdicts(
                earned=ceiling,
                thinking_with_tools=declared_thinking_with_tools(None),
            )
        return await effective_verdicts(
            provider=provider,
            provider_name=settings.model_provider,
            model=tag,
            settings=settings,
            require_verification=settings.agency_require_verification,
            sessionmaker=sessionmaker,
            effective_ceiling=ceiling,
            policy=await _fetch_policy(tag),
        )

    async def _route(tag: str, think: bool) -> tuple[str, str | None]:
        if listing is None:
            # The server could not be listed, so routing has nothing to judge
            # by: keep the configured tag and let the generation call surface
            # the provider's own typed failure. The same short-circuit the chat
            # path applies, and for the same reason - so the warm lists once.
            return tag, None
        return await route_around_missing(
            provider,
            settings.model_provider,
            pair,
            tag,
            think,
            listing=listing,
        )

    async def _resolve_ctx(tag: str) -> int:
        return await resolve_num_ctx(
            provider,
            settings.model_provider,
            tag,
            default_window=settings.default_context_window,
            ceiling=context_ceiling,
        )

    async def _fetch_report(tag: str) -> ModelCapabilities | None:
        return await runtime_capabilities(provider, settings.model_provider, tag)

    async def _fetch_policy(tag: str) -> PolicyView | None:
        if sessionmaker is None:
            return None
        return await resolve_policy(
            provider=provider,
            provider_name=settings.model_provider,
            model=tag,
            sessionmaker=sessionmaker,
        )

    async def _fetch_measured_think(tag: str) -> ThinkMode | None:
        if sessionmaker is None:
            return None
        found = await measured_think_mode(
            provider=provider,
            provider_name=settings.model_provider,
            model=tag,
            sessionmaker=sessionmaker,
        )
        return None if found is None else found.mode

    return await plan_turn(
        req=ChatRequest(
            message=_WARM_USER_MESSAGE,
            model=model,
            supports_confirmation=_WARM_SUPPORTS_CONFIRMATION,
        ),
        history=(),
        settings=settings,
        chat_pair=pair,
        # A warm with no wiring can offer nothing, and saying so here rather
        # than downstream keeps the planner's answer honest: it reports the
        # turn this warm actually sends, not the one a wired caller would.
        search_wired=wiring is not None and wiring.search_provider is not None,
        rag_wired=(
            wiring is not None and wiring.embedder is not None and wiring.vector_store is not None
        ),
        fetch_verdicts=_fetch_verdicts,
        route=_route,
        resolve_ctx=_resolve_ctx,
        fetch_report=_fetch_report,
        fetch_policy=_fetch_policy,
        fetch_measured_think=_fetch_measured_think,
        defaults=TurnDefaults(thinking=thinking, web_search=web_search),
    )


async def _warm_tools(
    *,
    plan: TurnPlan,
    settings: Settings,
    messages: list[ChatMessage],
    system_prompt: str,
    tool_prefix: WarmToolPrefix | None,
) -> list[Tool]:
    """Build the tool set the primed prefix advertises.

    The chat path's own sequence, in its order: build the offer, then price it
    against the window and drop it whole if a result could not fit. Only the
    RANKING step differs - see the ``query`` note below - and it is inert on
    every deployment whose tool count is under the selection threshold.

    Args:
        plan: The planned turn; its offer decides whether tools ride at all.
        settings: The active application settings.
        messages: The prime's messages, for the affordability floor.
        system_prompt: The persona, empty when system prompts are disabled.
        tool_prefix: The tenant and wiring, or None for a persona-only warm.

    Returns:
        The tools to advertise, empty when this warm carries none.
    """
    if tool_prefix is None or not plan.offer_tools:
        return []
    offered = await build_offered_tools(
        tool_prefix.wiring,
        # No conversation and no session: a warm belongs to no thread, and a
        # listing is identity-independent within a tenant (see CachingToolSource),
        # so leaving both unset asks the same question a turn's context does
        # without inventing a conversation id no row backs.
        ToolContext(
            tenant_id=str(tool_prefix.tenant.tenant_id),
            user_id=str(tool_prefix.tenant.user_id),
        ),
        offer=plan.offer,
        # Deliberately EMPTY, and this is where prefix identity ends. Selection
        # ranks tool descriptions against the turn's words, so above the
        # selection threshold the offered set is a function of the question and
        # no single prefix can match every turn. An empty query takes the
        # stable head instead - registry order, no embed, no IO - which IS the
        # turn's own answer below the threshold (where selection is inert) and
        # the best available guess above it.
        query="",
        supports_confirmation=_WARM_SUPPORTS_CONFIRMATION,
        cache=tool_prefix.vector_cache,
    )
    # The same floor a turn applies before generation, on the same numbers: a
    # window too small to hold a tool RESULT means the turn runs deterministic,
    # and priming a tool block for it would prime a head no turn sends. The
    # prime's prompt is the shortest one a conversation can start with (no
    # history, no grounding), so this agrees with the first turn's answer and
    # can only be more permissive than a later, fuller one's.
    room = tool_result_room(
        messages=messages,
        tools=offered,
        settings=settings,
        num_ctx=plan.num_ctx,
        reserve_tokens=plan.reserve_tokens,
        guidance_pending=bool(system_prompt),
    )
    if room < settings.tool_result_min_tokens:
        return []
    return offered


async def warm_model(
    provider: ModelProvider,
    settings: Settings,
    *,
    model: str | None = None,
    workspace_model: str | None = None,
    thinking: SettingMode | None = None,
    web_search: SettingMode | None = None,
    workspace_prompt: str | None = None,
    context_ceiling: int | None = None,
    keep_alive: str | int | None = None,
    sessionmaker: async_sessionmaker[AsyncSession] | None = None,
    tool_prefix: WarmToolPrefix | None = None,
) -> tuple[str, int, int | None]:
    """Run a one-token generation so the model is loaded and its cache primed.

    The message list mirrors a real chat turn (system prompt first, when
    configured; the tool-guidance layer on top of it when the turn would carry
    tools) and the think flag, the window and the offered tool list are all
    resolved by the SAME planner and builder a chat turn runs through, so the
    primed prefix is the one real turns will reuse. Nothing is persisted - the
    provider is called directly, no conversation is created.

    Args:
        provider: The active model provider.
        settings: The active application settings.
        model: Model to warm. A non-blank value is an explicit pick, resolved
            against what the server holds (:func:`match_installed_name`): an
            exact tag wins, a fragment fitting exactly one installed tag
            resolves to it, and a fragment fitting several or none is refused.
            A server whose inventory cannot be READ resolves nothing and the
            name goes through verbatim, so an unreachable listing endpoint
            cannot fail a warm that would have worked.
            A tag reached by FRAGMENT is additionally refused when it cannot
            serve a chat turn; an EXACT tag is not, because that is what the
            caller asked for and only the inferred half can be wrong about
            which model was meant.
            Resolving is not SUBSTITUTING - it only ever lands on the model the
            name already named, so the chat path's rule that an explicit pick is
            never swapped for another model still holds. None or blank warms the
            resolved default chat model.
        workspace_model: The tenant's stored workspace default, when the
            caller holds one; joins the default's resolution exactly as it
            joins a turn's.
        thinking: The tenant's stored thinking default, when the caller holds
            one. Decides which member of a routed pair is warmed and which
            think flag the warm request carries - both of which a turn decides
            the same way, and a warm that decided them differently primes
            weights or a prefix no turn reuses. On a model that cannot reason
            and call tools at once it also decides whether the prefix carries a
            tool block AT ALL: an explicitly-thinking turn keeps its reasoning
            and is offered nothing, so warming it with tools would prime a head
            no such turn ever sends.
        web_search: The tenant's stored web-search default, when the caller
            holds one. It gates the ``web_search`` tool - "off" is locked to
            mean fully local - so a workspace that turned the web off warms a
            prefix without that schema in it, exactly as its turns send.
        workspace_prompt: The tenant's stored persona, when the caller holds
            one - named like ``workspace_model`` because the composed result
            takes the plain name. The most literal prefix there is: it is the
            FIRST thing in the system message, so a warm that skipped it primes
            a head no turn on this workspace ever sends, and every first turn
            pays the full prefill anyway.
        context_ceiling: The tenant's stored context ceiling, when the caller
            holds one. Load-bearing for the same reason the think flag is: the
            model server restarts its runner when the requested window
            changes, so a warm that primed the unclamped window and a turn
            that then asks for the clamped one throw away exactly the cache
            this exists to build.
        keep_alive: Optional residency hint passed through to the provider.
        sessionmaker: Session factory for the stored declarations, the earned
            tool-check verdicts and the think measurement behind them. None
            resolves the pure half only, which is what a caller with no
            database gets - and what every warm did before those existed.
        tool_prefix: The tenant and collaborators the tool block is composed
            from. None warms the persona alone, which is a SHORTER prefix than
            a tools-offered turn opens with.

    Returns:
        A ``(model, took_ms, prompt_tokens)`` tuple; ``prompt_tokens`` is the
        evaluated prefix size when the provider reported usage - the visible
        proof of what was primed, since a tool block roughly quadruples it.

    Raises:
        NotFoundError: The model is not installed on the provider, or the
            server's inventory holds nothing that can chat.
        BadRequestError: The named model fits more than one installed tag, or
            the provider rejected the warm request.
        ServiceUnavailableError: The provider cannot be reached or failed.
    """
    # Blank is unset, the rule every sibling path applies to req.model (the
    # estimate path strips too): only a real tag is an explicit pick.
    resolved = (model or "").strip() or None
    # Listed ONCE, above the branch, because both halves need the same answer:
    # the default resolution routes around what is missing, and an explicit name
    # is resolved against what is present. A listing that cannot be fetched is
    # not a failure here - it is simply no information, and both halves fall
    # back to the behaviour they had before the inventory was consulted at all.
    try:
        listing = await provider.list_models()
    except ProviderUnavailableError:
        listing = None
    # The tier's (default, thinking) tags, resolved the way a TURN resolves
    # them: the workspace default participates and a stored tag counts when the
    # server actually holds it. Resolved for BOTH branches, because an
    # explicitly named model still needs the pair - it is what tells the
    # earned-verdict lookup which sibling a paired verdict was recorded against.
    #
    # Which MEMBER of it this warm runs on is decided by the planner below,
    # not here: the pair's thinking tag serves thinking turns, and which turns
    # those are is an arbitration, not a heuristic (see _plan_warm_turn).
    pair = resolve_chat_models(
        settings,
        workspace_model=workspace_model,
        installed={info.name for info in listing} if listing is not None else None,
    )
    if resolved is not None and listing is not None:
        # A name somebody typed is resolved against the inventory instead of
        # being handed to the provider verbatim: `make warm CHAT_MODEL=qwen`
        # should warm the obvious match rather than bounce off a 404 for a
        # spelling. Only the TYPED name is treated this way. The stored default
        # above keeps its exact-match-or-substitute chain, because it is
        # configuration that governs every turn - quietly re-pointing it at a
        # model whose tag merely CONTAINS it would change what the workspace
        # runs, not just what this one call primes.
        #
        # A listing that could not be fetched skips this entirely: a model
        # server that is down must not turn a perfectly valid explicit warm
        # into a resolution error, so the name goes through as it always did
        # and the provider path reports what is actually wrong.
        match = match_installed_name(resolved, [info.name for info in listing])
        if match.candidates:
            raise BadRequestError(
                f"'{resolved}' matches more than one installed model: "
                f"{', '.join(match.candidates)}. Name one of them exactly."
            )
        if match.model is None:
            raise NotFoundError(
                f"No installed model matches '{resolved}'. Pull it - the Model "
                "Manager can - or name one the model server already holds."
            )
        if match.by_fragment:
            # Guarded on ONE side only, deliberately. An exact tag is honoured
            # verbatim even if it cannot chat: that is an explicit request, and
            # the model server's own refusal is a better answer than a guess
            # made here. A fragment win is this platform's INFERENCE about
            # which model was meant, and an inference that lands a chat warm on
            # an embedder is wrong in a way the person never asked for - on a
            # card that fits one model it also evicts the model they were
            # using. So the inferred half, and only it, has to be right.
            #
            # The verdict comes from serves_chat_role, the predicate the
            # console and the pair writer already share (its docstring says
            # plainly that a second definition is the failure mode), fed by the
            # same runtime-facts seam route_around_missing reads. Metadata
            # only: the report comes from the runtime's describe endpoint, so
            # nothing is loaded to answer it.
            report = await runtime_capabilities(provider, settings.model_provider, match.model)
            # A probe that did not ANSWER is unknown, never evidence - the rule
            # route_around_missing applies to the same question. Refusing on an
            # unreachable metadata endpoint would replace the provider's own
            # precise error with a guess about the model.
            if report is not None and not serves_chat_role(report):
                raise BadRequestError(
                    f"'{resolved}' matched the installed model '{match.model}', which "
                    "cannot serve a chat turn. Name a chat model, or spell that tag "
                    "out in full to warm it deliberately."
                )
        resolved = match.model
    # Every remaining decision - which member of the pair runs, the think flag,
    # the window, and whether this turn would carry tools at all - comes from
    # the turn's own planner, driven with the request a plain warm stands for:
    # the explicitly named model if there was one, no switches of its own, and
    # the workspace's stored defaults underneath. Reimplementing any of it here
    # is the failure this module is built to avoid; a warm that decided one
    # axis differently primes a prefix no turn reuses, silently and while
    # looking perfectly healthy.
    try:
        plan = await _plan_warm_turn(
            provider,
            settings,
            model=resolved,
            pair=pair,
            listing=listing,
            thinking=thinking,
            web_search=web_search,
            context_ceiling=context_ceiling,
            sessionmaker=sessionmaker,
            tool_prefix=tool_prefix,
        )
    except NoChatModelConfiguredError as exc:
        # The turn's typed refusal, mapped for this REST endpoint: an inventory
        # that is known to hold nothing chat-capable is a 404 carrying the same
        # honest message, not an SSE code.
        raise NotFoundError(str(exc)) from exc
    # What routing landed on, which is what the generation call must name and
    # what the response reports - a substitution the warm hid would leave the
    # caller believing it primed a tag it never touched.
    resolved = plan.model
    messages: list[ChatMessage] = []
    system_prompt = compose_system_prompt(settings, workspace_prompt=workspace_prompt)
    if system_prompt:
        messages.append(ChatMessage(role="system", content=system_prompt))
    messages.append(ChatMessage(role="user", content=_WARM_USER_MESSAGE))
    offered_tools = await _warm_tools(
        plan=plan,
        settings=settings,
        messages=messages,
        system_prompt=system_prompt,
        tool_prefix=tool_prefix,
    )
    # The guidance layer lands on the persona exactly as the turn lands it, and
    # only for an offer that survived the floor - the same order, because it is
    # the ORDER that makes the two strings equal.
    if offered_tools and system_prompt:
        web_on, knowledge_on, other_on = guidance_shape(offered_tools)
        messages[0] = ChatMessage(
            role="system",
            content=append_tool_guidance(
                system_prompt, web=web_on, knowledge=knowledge_on, other=other_on
            ),
        )
    # num_ctx must MATCH what chat turns send (the planner resolved it through
    # the same helper): the model server restarts the runner when the context
    # size changes, which would throw away exactly the cache this warm primed.
    # That is also why a tag cannot simply be left unsized here -
    # the turn would size it, and the two requests would not agree.
    options = GenerationOptions(
        think=plan.think_flag,
        num_ctx=plan.num_ctx,
        max_tokens=1,
        keep_alive=keep_alive,
    )

    started = time.monotonic()
    prompt_tokens: int | None = None
    # aclosing: raising out of an async-for does NOT close the generator; the
    # provider's streaming HTTP response must be released deterministically,
    # not left to garbage collection (same rule as the SSE layer).
    async with aclosing(
        provider.chat_stream(
            messages,
            model=resolved,
            # The tool schemas are rendered into the template's SYSTEM region,
            # so they are part of the prefix being primed - passing them is the
            # whole point of this change, not a detail of it.
            tools=tools_to_wire(offered_tools),
            options=options,
        )
    ) as stream:
        async for chunk in stream:
            if chunk.type == "error":
                detail = chunk.error or "The model server failed the warm request."
                if chunk.error_code == "model_not_found":
                    raise NotFoundError(detail)
                if chunk.error_code == "bad_request":
                    raise BadRequestError(detail)
                raise ServiceUnavailableError(detail)
            if chunk.type == "done" and chunk.usage is not None:
                prompt_tokens = chunk.usage.prompt_tokens
    took_ms = int((time.monotonic() - started) * 1000)
    return resolved, took_ms, prompt_tokens


async def warm_embedder(
    provider: ModelProvider,
    settings: Settings,
    *,
    embedding: EmbeddingConfig | None = None,
    keep_alive: str | int | None = None,
) -> tuple[str, int]:
    """Load the active embedding model so the first retrieval is fast.

    Unlike chat warming there is no prompt-prefix cache to prime - an embedder
    has no standing system turn - so a single trivial embed call is enough to
    pull the weights into the model server with the requested residency. This
    removes the multi-second cold-load stall on the first ingest or query of a
    session. Which model that is is never a caller's free choice - it defines
    the vector space - and the device is honored so a CPU-pinned embedder warms
    on the CPU exactly as retrieval uses it.

    Args:
        provider: The active model provider.
        settings: The active application settings.
        embedding: The resolved embedding configuration, from
            :func:`app.services.embedding_resolution.resolve_embedding`. Passed
            by every caller that holds a workspace, so what gets warmed is what
            that workspace's next retrieval will actually run. Omitted, the
            deployment's own answer is resolved here - the rungs a caller with
            no tenant and no vector store could reach anyway.
        keep_alive: Optional residency hint passed through to the provider.

    Returns:
        A ``(model, took_ms)`` tuple naming the warmed embedding model.

    Raises:
        NotFoundError: Nothing resolves an embedding model.
        ServiceUnavailableError: The embedding model is not pulled or the model
            server cannot be reached (the provider's message names the fix).
    """
    config = embedding
    if config is None:
        config = (
            await resolve_embedding(
                settings=settings, provider=provider, provider_name=settings.model_provider
            )
        ).config
    if config is None:
        raise NotFoundError(UNCONFIGURED_DETAIL)
    options = EmbedOptions(device=config.device, keep_alive=keep_alive)
    started = time.monotonic()
    try:
        await provider.embed([_WARM_EMBED_INPUT], model=config.model, options=options)
    except ProviderUnavailableError as exc:
        # The provider collapses "model not pulled" and "server unreachable"
        # into one error whose message already names the fix (pull it / bring
        # the server up), so surface it verbatim as a 503 - mirrors how the
        # /rag ingest and query endpoints treat the same failure.
        raise ServiceUnavailableError(str(exc)) from exc
    took_ms = int((time.monotonic() - started) * 1000)
    return config.model, took_ms


async def warm_search(provider: SearchProvider) -> tuple[int, int]:
    """Issue one throwaway search so the first real web-search turn is fast.

    A cold SearXNG pays DNS resolution, an empty connection pool, and a
    lazy-loaded worker on its FIRST outbound call, so that first search is the
    slow one the client can time out on - the "first search fails, second works"
    symptom. Firing a throwaway query at startup (and from `make warm`) moves
    that cost off the first user turn. The provider's own retry-once already
    absorbs a transient cold-connection fault, so this usually succeeds on the
    first attempt; the query text is irrelevant.

    Args:
        provider: The active search provider.

    Returns:
        A ``(took_ms, result_count)`` tuple; ``result_count`` is informational
        (a warm query can legitimately return zero results).

    Raises:
        ServiceUnavailableError: The search backend is unavailable.
    """
    started = time.monotonic()
    try:
        results = await provider.search(_WARM_SEARCH_QUERY, limit=1)
    except SearchProviderUnavailableError as exc:
        raise ServiceUnavailableError(str(exc)) from exc
    took_ms = int((time.monotonic() - started) * 1000)
    return took_ms, len(results)
