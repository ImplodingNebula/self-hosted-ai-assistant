"""Building the tool set a turn offers, and pricing it into the context window.

Everything a turn advertises to the model - which in-process tools it builds,
which remote sources join them, which of those survive the confirmation gate
and per-turn selection, how many tokens the resulting schemas cost - is decided
here, once. Three callers need exactly this answer and must not compute it
three ways:

- the chat turn, assembling the offer it sends;
- a confirmation RESUME, rebuilding the runnable tools a checkpoint recorded;
- the warm prime, which exists to put the turn's own prompt prefix in the model
  server's cache and is worthless if it composes a different one.

That last caller is the reason this module exists rather than staying inline in
``chat_service``. A chat template renders the offered tool list into its SYSTEM
region, so the tool block is part of the shared prefix every turn opens with -
and a prompt cache is prefix-MATCHED. A warm that assembled "the same" prefix
from a second implementation would drift from the turn's the first time either
side changed, and the failure is silent: everything answers correctly, the
first real turn after each warm just pays a full prefill nobody can see.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field

from app.core.config import Settings
from app.providers.base import ChatMessage
from app.providers.rag.base import Embedder, Reranker, VectorStore
from app.providers.search.base import SearchProvider
from app.providers.search.fetcher import PageFetcher
from app.providers.tools.base import Tool, ToolContext, ToolSource
from app.services.agency import ToolOffer
from app.services.context_budget import approx_tokens
from app.services.prompts import tool_guidance
from app.services.tools.local import (
    SEARCH_KNOWLEDGE_TOOL,
    WEB_SEARCH_TOOL,
    SearchKnowledgeTool,
    WebSearchTool,
)
from app.services.tools.registry import ToolRegistry
from app.services.tools.selection import CORE_TOOL_NAMES, ToolVectorCache, select_tools
from app.services.tools.sources import StaticToolSource

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ToolWiring:
    """The live collaborators a turn's tools are constructed from.

    Grouped into one value because they travel together everywhere and are
    never meaningful apart: a caller either holds the request's collaborators
    or it holds none of them. Nothing here is a policy decision - whether a
    tool may be OFFERED is :class:`~app.services.agency.ToolOffer`'s answer,
    and this only says what could be built at all.

    Attributes:
        settings: The active application settings.
        search_provider: The web-search backend, or None when unwired.
        page_fetcher: The page fetcher web search reads results with.
        embedder: The embedder retrieval runs on, or None.
        vector_store: The vector store retrieval reads, or None.
        reranker: The reranker retrieval re-scores with, or None.
        mcp_sources: The tenant's MCP tool sources, already resolved.
    """

    settings: Settings
    search_provider: SearchProvider | None = None
    page_fetcher: PageFetcher | None = None
    embedder: Embedder | None = None
    vector_store: VectorStore | None = None
    reranker: Reranker | None = None
    mcp_sources: Sequence[ToolSource] = field(default_factory=tuple)


def guidance_shape(tools: Sequence[Tool]) -> tuple[bool, bool, int]:
    """Return ``(web, knowledge, other)`` - the guidance layer's shape for an offered set.

    One function, so the affordability floor and the layer itself can never
    disagree about what the prompt will contain: both read the SAME offered set.
    They used to derive it differently - the floor from the policy's booleans,
    the layer from the built tool names - which diverged the moment a
    collaborator was unwired and the floor counted a bullet the prompt never got.

    Args:
        tools: The tools the turn actually offers.

    Returns:
        Whether ``web_search`` is offered, whether ``search_knowledge`` is, and
        how many offered tools the guidance layer has no bullet for.
    """
    names = {tool.spec.name for tool in tools}
    return (
        WEB_SEARCH_TOOL in names,
        SEARCH_KNOWLEDGE_TOOL in names,
        sum(1 for tool in tools if tool.spec.name not in CORE_TOOL_NAMES),
    )


def tools_to_wire(tools: Sequence[Tool]) -> list[dict]:
    """Convert offered tools to the OpenAI-function schema the provider seam takes.

    ``ModelProvider.chat_stream`` accepts ``tools`` as a list of these dicts - the
    tool schema every current runtime (Ollama, and later vLLM's OpenAI-compatible
    API) understands - so the ToolSpec -> wire conversion lives above the seam,
    keeping the model seam independent of the tool seam.

    Args:
        tools: The tools the turn offers, in the order they should be advertised.

    Returns:
        One function-schema dict per tool, in the same order. Order is part of
        the prompt PREFIX (the template renders this list verbatim), so callers
        that want a cache hit must preserve it.
    """
    return [
        {
            "type": "function",
            "function": {
                "name": tool.spec.name,
                "description": tool.spec.description,
                "parameters": tool.spec.parameters or {"type": "object", "properties": {}},
            },
        }
        for tool in tools
    ]


async def list_candidate_tools(
    wiring: ToolWiring,
    context: ToolContext,
    *,
    offer_web: bool = True,
    offer_knowledge: bool = True,
) -> list[Tool]:
    """List every tool a turn could offer ``context``: in-process plus remote.

    The local source goes FIRST because the registry gives a contested name to
    the earliest source: a remote server must not be able to shadow a built-in
    by calling itself ``web_search``. One catalog, listed ONCE - a listing can
    cross the network, so a fan-out per call would put a round trip on every
    tool the model asks for.

    Args:
        wiring: The live collaborators to build the in-process tools from.
        context: The identity every source is asked what it offers.
        offer_web: Whether the turn's policy admits ``web_search`` at all.
        offer_knowledge: Whether it admits ``search_knowledge``.

    Returns:
        The composed catalog, name-unique, in registry order.
    """
    local_tools: list[Tool] = []
    if offer_web and wiring.search_provider is not None:
        local_tools.append(
            WebSearchTool(
                provider=wiring.search_provider,
                fetcher=wiring.page_fetcher,
                settings=wiring.settings,
                # The RAG collaborators route results through the session
                # ingest pipeline so they come back as citable sources;
                # absent, the tool degrades to fetch-and-quote.
                embedder=wiring.embedder,
                store=wiring.vector_store,
                reranker=wiring.reranker,
            )
        )
    if offer_knowledge and wiring.embedder is not None and wiring.vector_store is not None:
        local_tools.append(
            SearchKnowledgeTool(
                embedder=wiring.embedder,
                store=wiring.vector_store,
                reranker=wiring.reranker,
                settings=wiring.settings,
            )
        )
    registry = ToolRegistry([StaticToolSource(local_tools), *wiring.mcp_sources])
    return await registry.list_tools(context)


async def build_offered_tools(
    wiring: ToolWiring,
    context: ToolContext,
    *,
    offer: ToolOffer,
    query: str,
    supports_confirmation: bool,
    cache: ToolVectorCache | None = None,
) -> list[Tool]:
    """Build the tool set one turn advertises, from policy to final order.

    Args:
        wiring: The live collaborators the tools are built from.
        context: The identity the sources are listed for.
        offer: The arbitrated per-turn offer; its two booleans gate the local
            tools and nothing here can widen them.
        query: The turn's words, which selection ranks tool descriptions
            against. An empty string ranks nothing - see :func:`select_tools`.
        supports_confirmation: Whether the client can resume a suspended turn.
            A side-effecting tool is offered ONLY to a client that can confirm
            it, because a call that suspends a turn the client cannot continue
            is worse than useless. ``ToolSpec`` defaults ``side_effecting``
            True and a remote tool is read-only only when its server said so,
            so an unannotated one is treated as dangerous.
        cache: The process-wide description-vector cache, when the caller holds
            one. None still selects correctly and simply re-embeds - harmless
            below the selection threshold, where nothing is embedded at all.

    Returns:
        The tools to offer, in the order they will be advertised.
    """
    candidates = await list_candidate_tools(
        wiring,
        context,
        offer_web=offer.offer_web,
        offer_knowledge=offer.offer_knowledge,
    )
    if supports_confirmation:
        offerable = list(candidates)
    else:
        offerable = [tool for tool in candidates if not tool.spec.side_effecting]
        withheld = len(candidates) - len(offerable)
        if withheld:
            logger.info(
                "Withholding %d side-effecting tool(s) from a client that cannot confirm them",
                withheld,
            )
    return await select_tools(
        offerable,
        query=query,
        embedder=wiring.embedder,
        threshold=wiring.settings.tool_selection_threshold,
        top_k=wiring.settings.tool_selection_top_k,
        cache=cache if cache is not None else ToolVectorCache(),
    )


def tool_result_room(
    *,
    messages: Sequence[ChatMessage],
    tools: Sequence[Tool],
    settings: Settings,
    num_ctx: int,
    reserve_tokens: int,
    guidance_pending: bool,
    extra_tokens: int = 0,
) -> int:
    """Return how many tokens are left for tool RESULTS once the prompt is paid for.

    A tool result is RESERVED room, not merely capped: ``rag_budget_tokens``
    deliberately fills ``num_ctx``, so a result appended on top would evict the
    prompt HEAD - the system prompt, taking the tool guidance and the injection
    defense with it, invisibly to every correctness test. Every caller that
    decides whether tools are affordable computes the room HERE, so the
    pre-generation floor, the assembled floor inside the loop, and the warm
    prime cannot price the same prompt three different ways.

    The tool schemas ride in the prompt too - the runtime renders them from the
    wire payload - so they are counted. The JSON is an approximation of the
    rendered template, which is the closest honest measure available above the
    provider seam; an UNCOUNTED schema overstates the room in the unsafe
    direction, and a remote server's description alone may run to 1024
    characters.

    Args:
        messages: The prompt as assembled so far.
        tools: The tools the turn would offer.
        settings: The active settings (the turn-wide result ceiling).
        num_ctx: The context window this turn requests.
        reserve_tokens: The answer reserve matching the turn's think flag.
        guidance_pending: True when the tool-guidance layer is NOT yet inside
            ``messages`` and will be appended once the offer survives - the
            pre-generation case. False once it is already there.
        extra_tokens: Prompt cost the messages do not carry, today the attached
            images (their size is unknown until the attachment wait resolves,
            which is why the pre-generation floor passes zero).

    Returns:
        The room for tool results, in tokens. A value below
        ``settings.tool_result_min_tokens`` means the turn cannot afford tools.
    """
    guidance_tokens = 0
    if guidance_pending:
        web_on, knowledge_on, other_on = guidance_shape(tools)
        guidance_tokens = approx_tokens(
            tool_guidance(web=web_on, knowledge=knowledge_on, other=other_on)
        )
    prompt_tokens = (
        sum(approx_tokens(message.content) for message in messages)
        + extra_tokens
        + guidance_tokens
        + approx_tokens(json.dumps(tools_to_wire(tools)))
    )
    return min(settings.tool_result_total_max_tokens, num_ctx - reserve_tokens - prompt_tokens)
