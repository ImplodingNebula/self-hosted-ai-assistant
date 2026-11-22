"""Web-search policy, query building, and session ingestion of results.

Three deterministic v1 responsibilities, mirroring how ``thinking.py`` owns the
reasoning decision: :func:`resolve_web_search` decides whether a turn consults
the web (the auto/on/off freshness heuristic, history-aware for follow-ups);
:func:`build_search_query` turns the message into a standalone query, folding
recent context into a subject-less follow-up so the small-model / non-tool path
does not search a bare fragment;
and :func:`search_and_ingest` runs the search and lands the results in the RAG
session scope so the existing retrieval + citation path grounds the answer on
them. No new retrieval or citation code is needed - web results ride the Phase 3
machinery as ``source_type="web"`` chunks. Tool-capable models formulate their
own query in Phase 5; these heuristics remain the graceful small-model fallback.
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import time
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Literal
from urllib.parse import urlsplit

from app.core.config import Settings
from app.core.tenancy import TenantContext
from app.providers.base import ChatMessage
from app.providers.rag.base import Embedder, VectorStore
from app.providers.search.base import SearchProvider, SearchResult
from app.providers.search.fetcher import PageFetcher
from app.schemas.rag import RagIngestItem
from app.services import rag_service

logger = logging.getLogger(__name__)

# Recency cues that usually mean the answer needs information fresher than the
# model's training data. Conservative on purpose: a false positive spends a
# network round-trip and latency on a turn that a local answer would have
# served. Bias is toward answering locally (mirrors the thinking heuristic's
# bias toward the fast path). Deliberately narrow on words: broad tokens like
# "version", "release", or a bare "current" over-fired on conceptual questions
# ("version control", "current directory"), so they are left out. Years are
# handled separately (see below) because a bare four-digit match cannot tell a
# future year from a historical one.
_FRESHNESS_HINT = re.compile(
    r"\b(latest|newest|recent(?:ly)?|currently|right now|today|tonight|tomorrow"
    r"|yesterday|upcoming|(?:this|next) (?:year|month|week)"
    # "next" + an event noun (future fixtures, elections, releases) without the
    # over-firing of a bare "next" ("next step", "next line of code").
    r"|next (?:games?|match(?:es)?|fixtures?|races?|seasons?|episodes?"
    r"|elections?|releases?|world cup|olympics)"
    r"|so far|up[ -]?to[ -]?date|as of|breaking|news|headline|price|stock"
    r"|weather"
    # "current" + a mutable-fact noun (title holders, versions, standings).
    # Bare "current" stays out on purpose: its technical determiner use
    # ("current working directory", "current loop") is common in standalone
    # local questions and would over-fire (see the generic-tech-terms test).
    # "current" + a mutable PUBLIC fact only. Nouns with a common technical
    # determiner reading (values, rates, scores, records - "the current values
    # of these variables") stay out, same bias as excluding bare "current".
    r"|current (?:versions?(?! control)|prices?|standings?"
    r"|rankings?|champions?|winners?|leaders?|ceo|president"
    r"|prime minister)"
    # The colloquial current-affairs idiom ("what's going on with iran now?")
    # carries no recency WORD, so without this it answered stale and silent.
    # The lookahead keeps the code-comprehension shape local ("what is
    # happening in this loop", "when I run it", "here"): demonstratives and
    # possessives after the idiom point at something in front of the asker,
    # not at the news.
    r"|what(?:'s| is) (?:going on|happening)"
    r"(?! (?:in|with|to|inside) (?:this|that|these|those|my|our|your)\b"
    r"| when\b| here\b))\b",
    re.IGNORECASE,
)
# Any four-digit 20xx year in the message, checked against the current year in
# code: a year at or after now is a freshness cue (post-training-cutoff), a past
# year is historical and stays local. This replaces a bare ``20\d\d`` match,
# which fired on long-past years like 2010.
_YEAR = re.compile(r"\b(20\d\d)\b")
# Explicit asks to consult the web, which force a search in auto mode.
_EXPLICIT_SEARCH = re.compile(
    r"\b(search (?:the web|online|for)|look (?:it|this|that|them) up|google it"
    r"|web search)\b",
    re.IGNORECASE,
)


def _names_current_or_future_year(message: str, current_year: int) -> bool:
    """True when ``message`` names a year at or after ``current_year``.

    A year the model could not have covered in training (this year or later) is
    a strong freshness cue - the reason a bare year match is worth keeping once
    it is gated this way.
    """
    return any(int(year) >= current_year for year in _YEAR.findall(message))


def _message_triggers_search(message: str, current_year: int) -> bool:
    """True when a message on its own warrants a web search.

    The freshness core shared by the auto decision and the history-aware
    follow-up check below: an explicit ask to search, a recency word, or a year
    at or after the current one.
    """
    return bool(
        _EXPLICIT_SEARCH.search(message)
        or _FRESHNESS_HINT.search(message)
        or _names_current_or_future_year(message, current_year)
    )


def resolve_web_search(
    mode: Literal["auto", "on", "off"],
    message: str,
    history: Sequence[ChatMessage] = (),
    *,
    current_year: int | None = None,
) -> bool:
    """Decide whether this turn should consult the web.

    The v1 heuristic, replaced by the orchestrator in a later phase. ``on`` and
    ``off`` are explicit user choices and always win. ``auto`` is local-first:
    the network is consulted only when the message shows a recency word, names a
    year at or after the current one, explicitly asks to search, or is a
    context-dependent follow-up in a thread whose recent turn already warranted
    one - the bias is toward answering locally.

    Args:
        mode: The client-requested web-search mode.
        message: The new user message for this turn.
        history: Prior turns. A subject-less follow-up ("how would it be for a
            java-backend stack") carries no cue of its own, so the recent user
            turns are checked: if one warranted a search, the follow-up inherits
            it rather than silently answering from cached or stale context.
        current_year: The year a named year is compared against; defaults to the
            current UTC year (injectable so the year gate is deterministic in
            tests).

    Returns:
        True when the turn should run a web search.
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    if current_year is None:
        current_year = datetime.now(tz=UTC).year
    if _message_triggers_search(message, current_year):
        return True
    # Follow-up inheritance: only a context-dependent message (one leaning on an
    # earlier turn) can borrow freshness, and only from the recent user turns - so
    # a fresh-info thread keeps searching while an unrelated aside does not.
    if _is_context_dependent(message):
        recent_user = [m.content for m in history if m.role == "user"]
        return any(
            _message_triggers_search(turn, current_year)
            for turn in recent_user[-_CONTEXT_MAX_TURNS:]
        )
    return False


# Words that signal a message leans on an earlier turn for its subject. Kept to
# unambiguous personal pronouns plus a few leading continuations; broad
# demonstratives ("this", "that") are left out because their determiner use
# ("this week", "that year") is common in standalone fresh queries and would
# wrongly fold context into them. The bias is conservative: a false negative
# just searches the message verbatim (today's behavior), while a false positive
# muddies a good query - so misses are cheaper than over-eager folding.
_ANAPHORA = re.compile(
    r"\b(they|them|their|theirs|he|him|his|she|her|hers)\b",
    re.IGNORECASE,
)
# "it"/"its" are matched case-sensitively (lower- and Title-case, never the
# all-caps acronym) so "IT" - information technology, as in "latest IT news" -
# is not misread as the pronoun and folded into an unrelated query.
_ANAPHORA_IT = re.compile(r"\b(it|It|its|Its)\b")
# Leading continuations. "also"/"plus" are deliberately excluded: they commonly
# begin standalone queries ("also known as ...", "plus size ...") rather than a
# continuation of the prior turn.
_LEADING_CONTINUATION = re.compile(
    r"^\s*(and|so|but|then)\b|\b(what|how)\s+about\b",
    re.IGNORECASE,
)
# How much recent context a follow-up folds in: the last few USER turns (the
# topic anchor), capped in length so the query gains the subject without
# ballooning into the whole conversation. The tail is kept when truncating, so
# the most recent - most relevant - context survives.
_CONTEXT_MAX_TURNS = 2
_CONTEXT_MAX_CHARS = 300


def _is_context_dependent(message: str) -> bool:
    """True when a message leans on earlier turns (anaphora or a continuation)."""
    return bool(
        _ANAPHORA.search(message)
        or _ANAPHORA_IT.search(message)
        or _LEADING_CONTINUATION.search(message)
    )


def build_search_query(message: str, history: Sequence[ChatMessage] = ()) -> str:
    """Build a standalone web-search query, folding context into a follow-up.

    The deterministic v1 query builder - the small-model / non-tool path, and the
    graceful fallback for models that cannot formulate their own query (Phase 5
    gives tool-capable models that ability). A self-contained message is searched
    verbatim; a context-dependent follow-up ("and how is it currently") leads the
    query with the recent user turns trailing as the topic anchor, so the search
    engine and the retrieval ranking that scores the fetched pages both weight the
    follow-up's own subject up front rather than a subject-less fragment buried
    behind the prior turn. A subject introduced only by the assistant's reply, or
    older than the last few user turns, is not resolved - an accepted v1 limit
    that degrades to today's verbatim search, never to something worse.

    Args:
        message: The new user message for this turn.
        history: Prior turns; only user turns are folded in (assistant text is
            long and would swamp the query).

    Returns:
        The message itself, or recent user context followed by the message.
    """
    message = message.strip()
    if not message or not _is_context_dependent(message):
        return message
    prior = [m.content.strip() for m in history if m.role == "user" and m.content.strip()]
    if not prior:
        return message
    context = " ".join(prior[-_CONTEXT_MAX_TURNS:])[-_CONTEXT_MAX_CHARS:]
    # Lead with the follow-up so the engine weights its own subject; the context
    # trails as the anchor a bare fragment ("how is it now") still needs.
    return f"{message} {context}" if context else message


# Field caps mirroring RagIngestItem (schemas/rag.py). Search-engine output is
# untrusted and unbounded, so every field is clamped to these before
# construction: an over-long title/text/url would otherwise raise a pydantic
# ValidationError that escapes the best-effort ingest and aborts the chat turn.
_MAX_TITLE_LEN = 500
_MAX_TEXT_LEN = 50_000
_MAX_URL_LEN = 2000


def safe_web_url(url: str) -> str | None:
    """Return ``url`` only if it is a length-bounded http(s) link, else None.

    Web results are attacker-influenceable and the citation ``url`` renders as a
    clickable anchor, so a link is kept only when it is a normal browsable
    http/https URL within the ingest schema's cap. Anything else (a ``javascript:``
    or ``data:`` scheme, a magnet/ftp link, or an over-long tracking URL) drops to
    None: the result is still ingested and cited, just not clickable.

    Whitespace and control characters reject the URL outright, BEFORE parsing:
    ``urlsplit`` strips tab/CR/LF itself and would validate a string this
    function then returns VERBATIM - and downstream renderers put a trusted URL
    on its own digest line, where an embedded newline forges extra numbered
    blocks (the exact injection the label sanitizers exist to stop).
    """
    if len(url) > _MAX_URL_LEN:
        return None
    if any(ch.isspace() or ord(ch) < 0x20 or ord(ch) == 0x7F for ch in url):
        return None
    return url if urlsplit(url).scheme.lower() in ("http", "https") else None


def _web_doc_id(url: str) -> str:
    """Return a stable, length-bounded session-doc id for a result URL.

    ``ingest_structured`` is delete-then-upsert per doc id, so a stable id means
    re-searching the same URL in a conversation rewrites its chunk instead of
    duplicating it. Hashed because ``RagIngestItem.id`` caps at 256 characters
    and URLs can exceed that.
    """
    return "web-" + hashlib.sha256(url.encode()).hexdigest()[:32]


def _snippet_fingerprint(result: SearchResult, settings: Settings) -> str:
    """Fingerprint of the snippet-only item this result would ingest as.

    Mirrors the exact fallback item construction in :func:`search_and_ingest`,
    so a stored ``content_hash`` equal to it identifies an ingest that was
    grounded on the snippet alone (its page fetch failed or was skipped).
    Empty snippet+title returns "" - never equal to a stored sha256 hex.
    """
    text = (result.snippet or result.title).strip()
    if not text:
        return ""
    return rag_service.content_fingerprint(
        RagIngestItem(
            id=_web_doc_id(result.url),
            title=result.title[:_MAX_TITLE_LEN],
            text=text[:_MAX_TEXT_LEN],
            url=safe_web_url(result.url),
        ),
        settings,
    )


async def fetch_page_texts(
    results: Sequence[SearchResult],
    fetcher: PageFetcher | None,
    settings: Settings,
) -> list[str | None]:
    """Fetch page text for the TOP results concurrently; None where unavailable.

    Best-effort and aligned with ``results``: fetching is disabled (or the
    fetcher unwired) yields all None, and any per-page failure is None so the
    caller falls back to that result's snippet. Only the first
    ``web_fetch_max_pages`` results get a full-page fetch - the tail grounds
    on snippets, bounding how many chunks the turn must embed before the
    first token.
    """
    if fetcher is None or not settings.web_fetch_pages or not results:
        return [None] * len(results)
    fetch_count = max(0, settings.web_fetch_max_pages)
    targets = results[:fetch_count]
    semaphore = asyncio.Semaphore(max(1, settings.web_fetch_concurrency))

    async def fetch_one(url: str) -> str | None:
        async with semaphore:
            try:
                return await fetcher.fetch_text(url)
            except Exception as exc:  # noqa: BLE001 - never break ingest on a fetch
                logger.debug("Web page fetch raised for %s: %s", url, exc)
                return None

    fetched = list(await asyncio.gather(*(fetch_one(result.url) for result in targets)))
    return fetched + [None] * (len(results) - len(targets))


async def search_and_ingest(
    *,
    query: str,
    provider: SearchProvider,
    tenant: TenantContext,
    embedder: Embedder,
    store: VectorStore,
    settings: Settings,
    session_id: str,
    fetcher: PageFetcher | None = None,
) -> int:
    """Search the web and ingest the results into the RAG session scope.

    Reuses :func:`app.services.rag_service.ingest_structured` so web chunks are
    retrieved and cited by the same pipeline as any other source. Deduplicates
    by URL so a page several engines return is fetched and ingested once, and
    reuses pages ingested into this session within the
    ``web_result_cache_minutes`` window as-is (no re-fetch, no re-embed) - a
    repeat or follow-up search then costs one SearXNG call instead of the full
    pipeline. When a ``fetcher`` is supplied and page fetching is enabled, the
    top results' pages are fetched and their extracted text grounds the answer;
    otherwise (or on any per-page fetch failure) the search snippet is used.

    Args:
        query: The search query (the user's message for this turn).
        provider: The active search provider.
        tenant: The acting tenant/user (scopes the ingested vectors).
        embedder: The active embedder.
        store: The active vector store.
        settings: Application settings (top-k, session TTL).
        session_id: The conversation id, so retrieval sees these this turn.
        fetcher: Optional page fetcher; None keeps grounding snippet-only.

    Returns:
        The number of result documents now retrievable in the session scope -
        freshly ingested plus reused-from-cache (0 when nothing was found).

    Raises:
        SearchProviderUnavailableError: The search backend is unavailable.
        ProviderUnavailableError: The embedding backend is unavailable.
        VectorStoreUnavailableError: The vector store is unavailable.
    """
    results = await provider.search(query, limit=settings.web_search_top_k)
    # Dedup by stable doc id first (a page several engines return is one doc), so
    # fetching and ingestion each happen once per unique page.
    unique: list[SearchResult] = []
    seen: set[str] = set()
    for result in results:
        doc_id = _web_doc_id(result.url)
        if doc_id in seen:
            continue
        seen.add(doc_id)
        unique.append(result)
    # Session reuse window: a page ingested into this conversation recently is
    # already retrievable - drop it from the fetch/embed pipeline entirely.
    cached_docs = 0
    if unique and settings.web_result_cache_minutes > 0:
        fingerprints = await store.doc_fingerprints(
            "session",
            embedder.model,
            str(tenant.tenant_id),
            [_web_doc_id(result.url) for result in unique],
            session_id=session_id,
        )
        horizon = time.time() - settings.web_result_cache_minutes * 60
        fetch_active = fetcher is not None and settings.web_fetch_pages
        reusable: set[str] = set()
        for position, result in enumerate(unique):
            fingerprint = fingerprints.get(_web_doc_id(result.url))
            if fingerprint is None or fingerprint.created_at < horizon:
                continue
            if (
                fetch_active
                and position < settings.web_fetch_max_pages
                and fingerprint.content_hash == _snippet_fingerprint(result, settings)
            ):
                # The stored ingest was grounded on the snippet alone (a
                # failed or skipped fetch last time), and this turn WOULD
                # fetch the full page - retry the fetch rather than pinning
                # the thin snippet for the whole window.
                continue
            reusable.add(fingerprint.doc_id)
        if reusable:
            unique = [r for r in unique if _web_doc_id(r.url) not in reusable]
            cached_docs = len(reusable)
            # Reused docs are in active use: extend their session TTL exactly
            # as a fresh ingest would have stamped it.
            await store.refresh_doc_ttl(
                "session",
                embedder.model,
                str(tenant.tenant_id),
                sorted(reusable),
                ttl_at=time.time() + settings.rag_session_ttl_minutes * 60,
                session_id=session_id,
            )
    page_texts = await fetch_page_texts(unique, fetcher, settings)
    items: list[RagIngestItem] = []
    for result, page_text in zip(unique, page_texts, strict=True):
        # Fetched page text is richer than the snippet; fall back to snippet then
        # title when the fetch is unavailable or empty.
        text = (page_text or result.snippet or result.title).strip()
        if not text:
            continue
        # Clamp every untrusted field to the ingest caps so one oversized or
        # oddly-schemed result can never raise ValidationError and abort the turn.
        items.append(
            RagIngestItem(
                id=_web_doc_id(result.url),
                title=result.title[:_MAX_TITLE_LEN],
                text=text[:_MAX_TEXT_LEN],
                url=safe_web_url(result.url),
            )
        )
    if not items:
        return cached_docs
    documents, _chunks = await rag_service.ingest_structured(
        source_type="web",
        tenant=tenant,
        items=items,
        embedder=embedder,
        store=store,
        settings=settings,
        session_id=session_id,
    )
    return documents + cached_docs
