"""The in-process tools: web search and knowledge search.

These are ordinary Python behind the :class:`~app.providers.tools.base.Tool`
seam - no transport, no sidecar. "MCP is a source of tools, not the
architecture": an MCP tool and one of these join the same registry and the
generation loop cannot tell them apart.

Each tool takes exactly ONE parameter, ``query``. That is a deliberate design
rule for the whole tool surface, not a shortcut: a small local model asked to
fill several structured arguments (``max_results``, ``recency``, ``site``) fills
them wrong, so the tools expose the one argument the model can reliably produce
and keep every other knob in configuration.

Both are read-only lookups (``side_effecting=False``). Their collaborators (the
search provider, the embedder, the vector store) are injected at construction;
the per-call :class:`~app.providers.tools.base.ToolContext` supplies only the
caller's identity, and the tenant/user a call acts for is taken from there and
nowhere else.

Results come back as :class:`~app.providers.tools.base.ToolResultSource` blocks the
generation loop numbers, fences as untrusted data, and offers to the citation
flow - a tool-grounded answer cites [n] exactly like a retrieval-grounded one
(slice 3c). ``web_search`` routes through ``search_and_ingest``, so it shares
the deterministic pre-fetch path's session result cache, URL canonicalization
and session-scoped ingestion: a repeated call within the cache window costs one
search instead of the full fetch-and-embed pipeline, and the fetched pages stay
retrievable for the rest of the conversation.

When the RAG collaborators are absent (no embedder or vector store wired),
``web_search`` degrades to its original self-contained behaviour - fetch the
pages and quote them back uncited - so a retrieval-less deployment keeps a
working, honest web tool.
"""

from __future__ import annotations

from uuid import UUID

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.providers.base import ProviderUnavailableError
from app.providers.rag.base import (
    Embedder,
    Reranker,
    ScoredChunk,
    VectorStore,
    VectorStoreUnavailableError,
)
from app.providers.search.base import SearchProvider, SearchProviderUnavailableError, SearchResult
from app.providers.search.fetcher import PageFetcher
from app.providers.tools.base import (
    Tool,
    ToolContext,
    ToolError,
    ToolResult,
    ToolResultSource,
    ToolSpec,
)
from app.services import rag_service, web_search

WEB_SEARCH_TOOL = "web_search"
SEARCH_KNOWLEDGE_TOOL = "search_knowledge"

# Per-result text cap for the retrieval-less fallback digest. The loop caps
# the whole fenced message again (tool_result_max_tokens), so this only keeps
# any single result from crowding out the others before that ceiling applies.
# The cited path's cap lives with its renderer (prompts.tool_source_digest).
_WEB_RESULT_CHARS = 800
# A result title/label is attacker-influenceable (a search engine returns it),
# and the digest is line-oriented ("[n] title\nurl\nbody"), so a title carrying
# newlines could forge extra numbered blocks or a spoofed URL line. Collapse all
# whitespace to single spaces and cap the length before it goes near the model.
_LABEL_CHARS = 120

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "What to look up, as a natural-language search query.",
        }
    },
    "required": ["query"],
}


def _query_of(arguments: dict) -> str:
    """Pull the single ``query`` argument out, trimmed; "" for anything non-string.

    A JSON null (``{"query": null}``, a common small-model malformation) or a
    non-string value becomes "", so it trips the empty-query guard instead of
    running a live search for the literal text "None".
    """
    value = arguments.get("query")
    return value.strip() if isinstance(value, str) else ""


def _safe_label(text: str) -> str:
    """Collapse whitespace and cap a title/label so it cannot forge digest structure."""
    return " ".join(text.split())[:_LABEL_CHARS]


def _tenant_of(context: ToolContext) -> TenantContext:
    """Rebuild the acting tenant/user from the call context - the only identity source."""
    return TenantContext(tenant_id=UUID(context.tenant_id), user_id=UUID(context.user_id))


def _source_of(chunk: ScoredChunk) -> ToolResultSource:
    """Convert a retrieval hit into a citable tool source.

    The URL is sanitized HERE, at the seam boundary, because the digest
    renderer trusts a ToolResultSource url to already be a plain http(s) link -
    a chunk's stored url is attacker-influenceable (it came off the web).
    """
    url = web_search.safe_web_url(chunk.url) if chunk.url else None
    return ToolResultSource(
        id=chunk.id,
        doc_id=chunk.doc_id,
        text=chunk.text,
        title=chunk.title,
        url=url,
        source_type=chunk.source_type,
        score=chunk.score,
    )


class WebSearchTool(Tool):
    """Search the web and hand the top results back as citable sources.

    With the RAG collaborators wired (the normal case), results go through
    ``search_and_ingest`` - session cache, canonical URLs, session-scoped
    vectors - and come back as :class:`ToolResultSource` blocks the loop can cite.
    Without them, the tool degrades to the original fetch-and-quote behaviour.
    """

    def __init__(
        self,
        *,
        provider: SearchProvider,
        fetcher: PageFetcher | None,
        settings: Settings,
        embedder: Embedder | None = None,
        store: VectorStore | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self._provider = provider
        self._fetcher = fetcher
        self._settings = settings
        self._embedder = embedder
        self._store = store
        self._reranker = reranker

    @property
    def spec(self) -> ToolSpec:
        """The advertised web-search tool: one ``query`` argument, read-only."""
        return ToolSpec(
            name=WEB_SEARCH_TOOL,
            description=(
                "Search the web for current or external information and read the top "
                "results. Use it for facts that may have changed or that are not in the "
                "conversation or the knowledge base."
            ),
            parameters=_QUERY_SCHEMA,
            side_effecting=False,
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        """Search, ingest the leading pages, and return them as citable sources.

        Raises:
            ToolError: The search backend is unavailable - a failure to feed
                back, not an empty result. No hits is a normal, non-error result.
        """
        query = _query_of(arguments)
        if not query:
            return ToolResult(content="No search query was provided.", is_error=True)
        session_id = context.session_id or context.conversation_id or None
        if self._embedder is None or self._store is None or session_id is None:
            return await self._run_unattributed(query)
        try:
            ingested = await web_search.search_and_ingest(
                query=query,
                provider=self._provider,
                tenant=_tenant_of(context),
                embedder=self._embedder,
                store=self._store,
                settings=self._settings,
                session_id=session_id,
                fetcher=self._fetcher,
            )
        except SearchProviderUnavailableError as exc:
            raise ToolError(f"the web search backend is unavailable: {exc}") from exc
        except (ProviderUnavailableError, VectorStoreUnavailableError) as exc:
            raise ToolError(f"the knowledge store is unavailable: {exc}") from exc
        # Read the pages back through retrieval (session scope only), so the
        # result blocks are the SAME chunks a later turn retrieves and cites -
        # one identity from tool result to citation to reload. Run it even when
        # THIS search ingested nothing: a transiently empty engine response
        # (the documented SearXNG failure mode) must not blind the tool to
        # relevant pages an earlier search this session already fetched - the
        # deterministic pre-fetch path grounds on those in the same situation.
        try:
            chunks = await rag_service.retrieve(
                query=query,
                tenant=_tenant_of(context),
                embedder=self._embedder,
                store=self._store,
                settings=self._settings,
                reranker=self._reranker,
                session_id=session_id,
                include_kb=False,
            )
        except (ProviderUnavailableError, VectorStoreUnavailableError) as exc:
            raise ToolError(f"the knowledge store is unavailable: {exc}") from exc
        web_chunks = rag_service.dedupe_chunks_by_source(
            [chunk for chunk in chunks if chunk.source_type == "web"]
        )
        if not web_chunks:
            if ingested == 0:
                return ToolResult(content=f"No web results were found for {query!r}.")
            return ToolResult(
                content=f"The pages found for {query!r} held no passage close enough to use."
            )
        return ToolResult(content="", sources=[_source_of(chunk) for chunk in web_chunks])

    async def _run_unattributed(self, query: str) -> ToolResult:
        """The retrieval-less fallback: fetch and quote, no ingestion, no citations.

        Raises:
            ToolError: The search backend is unavailable.
        """
        try:
            results = await self._provider.search(query, limit=self._settings.web_search_top_k)
        except SearchProviderUnavailableError as exc:
            raise ToolError(f"the web search backend is unavailable: {exc}") from exc
        unique = _dedupe_by_url(results)
        if not unique:
            return ToolResult(content=f"No web results were found for {query!r}.")
        texts = await web_search.fetch_page_texts(unique, self._fetcher, self._settings)
        return ToolResult(content=_format_web_results(unique, texts), count=len(unique))


class SearchKnowledgeTool(Tool):
    """Search the tenant's stored records and this conversation's scratch, cited-style."""

    def __init__(
        self,
        *,
        embedder: Embedder,
        store: VectorStore,
        reranker: Reranker | None,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._store = store
        self._reranker = reranker
        self._settings = settings

    @property
    def spec(self) -> ToolSpec:
        """The advertised knowledge-search tool: one ``query`` argument, read-only.

        The description names WHAT lives in the store, not just that a store
        exists: described abstractly ("search the knowledge base"), a small
        model never connects an organization-specific question to it and
        answers "I don't have access" instead. Measured on qwen3:4b-instruct
        (k=3 per question, 2026-07-16): the abstract wording drew 0/6 calls on
        questions whose answers sat in the store; this wording plus the
        guidance layer (see ``prompts.tool_guidance``) drew 5/9.
        """
        return ToolSpec(
            name=SEARCH_KNOWLEDGE_TOOL,
            description=(
                "Search this workspace's own stored records and files: its policies, "
                "products, pricing, support and account details, orders, documentation, "
                "and anything uploaded earlier in this conversation. Questions about "
                '"the" company, product, subscription, or policy refer to THIS '
                "workspace's records - always search before saying such information is "
                "unavailable."
            ),
            parameters=_QUERY_SCHEMA,
            side_effecting=False,
        )

    async def run(self, arguments: dict, context: ToolContext) -> ToolResult:
        """Retrieve the best-matching passages and return them as readable text.

        Raises:
            ToolError: The embedder or vector store is unavailable.
        """
        query = _query_of(arguments)
        if not query:
            return ToolResult(content="No search query was provided.", is_error=True)
        tenant = _tenant_of(context)
        # None, not "": rag_service.retrieve treats any non-None session_id as a
        # real scope, so an empty string would add a wasted, always-empty session
        # query on a latency-sensitive turn.
        session_id = context.session_id or context.conversation_id or None
        try:
            chunks = await rag_service.retrieve(
                query=query,
                tenant=tenant,
                embedder=self._embedder,
                store=self._store,
                settings=self._settings,
                reranker=self._reranker,
                session_id=session_id,
                # Honour the operator's RAG kill switch, exactly as the chat
                # service does: RAG_ENABLED=false keeps the stored knowledge base
                # out and leaves the tool searching only this conversation's scope.
                include_kb=self._settings.rag_enabled,
            )
        except (ProviderUnavailableError, VectorStoreUnavailableError) as exc:
            raise ToolError(f"the knowledge store is unavailable: {exc}") from exc
        chunks = rag_service.dedupe_chunks_by_source(chunks)
        if not chunks:
            return ToolResult(content=f"No stored knowledge matched {query!r}.")
        return ToolResult(content="", sources=[_source_of(chunk) for chunk in chunks])


def _dedupe_by_url(results: list[SearchResult]) -> list[SearchResult]:
    """Drop duplicate URLs (a page several engines return is one source)."""
    seen: set[str] = set()
    unique: list[SearchResult] = []
    for result in results:
        if result.url in seen:
            continue
        seen.add(result.url)
        unique.append(result)
    return unique


def _format_web_results(results: list[SearchResult], texts: list[str | None]) -> str:
    """Render results as numbered blocks: title, URL, and page text or snippet.

    The title and URL are attacker-influenceable, so the title is whitespace-
    collapsed and capped and the URL is dropped unless it is a plain http(s)
    link - neither can inject newlines that forge extra numbered blocks.
    """
    blocks: list[str] = []
    for index, (result, text) in enumerate(zip(results, texts, strict=False), start=1):
        body = (text or result.snippet or "").strip()[:_WEB_RESULT_CHARS].rstrip()
        block = f"[{index}] {_safe_label(result.title)}"
        url = web_search.safe_web_url(result.url)
        if url:
            block = f"{block}\n{url}"
        if body:
            block = f"{block}\n{body}"
        blocks.append(block)
    return "\n\n".join(blocks)
