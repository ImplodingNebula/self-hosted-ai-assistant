"""Chat-path web-search tests: policy gating, grounding, citations, degradation.

Web results reuse the Phase 3 retrieval + citation pipeline, so these assert the
same citation shape as stored records plus the web-only ``url`` field, and that
a search failure degrades to a local answer instead of breaking the stream.
"""

import json
from contextlib import asynccontextmanager

from httpx import ASGITransport, AsyncClient

from app.api.deps import get_model_provider, get_settings
from app.core.concurrency import StreamLimiter
from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.main import create_app
from app.providers.fake_provider import FakeProvider
from app.providers.rag.embedder import FakeEmbedder
from app.providers.search.base import (
    SearchProvider,
    SearchProviderUnavailableError,
    SearchResult,
)
from app.providers.search.fetcher import FakePageFetcher
from app.providers.search.searxng import FakeSearchProvider
from app.schemas.rag import RagIngestItem
from app.services import rag_service
from app.workers.queue import FakeJobQueue
from tests.conftest import make_memory_store

CHAT_URL = "/api/v1/chat/stream"

# A freshness-cue message so web_search="auto" fires; the results' snippets are
# set equal to it so the hash embedder retrieves them (cosine 1.0).
FRESH_MESSAGE = "What is the latest Python version today?"
LOCAL_MESSAGE = "Explain how a hash map works."


def web_results(snippet: str) -> list[SearchResult]:
    return [
        SearchResult(title="Python releases", url="https://python.org/downloads", snippet=snippet),
    ]


class DownSearchProvider(SearchProvider):
    """A provider whose search always fails, to exercise graceful degradation."""

    async def search(self, query: str, *, limit: int) -> list[SearchResult]:
        raise SearchProviderUnavailableError("search backend down")


def make_settings(**overrides) -> Settings:
    # These tests exercise the deterministic web pre-fetch path (search + ingest
    # before the turn, then citations). The shipped default is
    # AGENCY_MODE=agentic, where web_search is a model-driven TOOL and there is
    # no pre-fetch, so pin the mode the pre-fetch logic under test runs in. A
    # test that wants a different mode overrides it.
    overrides.setdefault("agency_mode", "deterministic")
    return Settings(_env_file=None, model_provider="fake", **overrides)


def citing_provider() -> FakeProvider:
    """A provider whose scripted answer references passage [1] inline.

    Citations are usage-gated on the answer's own [n] markers, so tests that
    expect a citation script an answer that actually cites.
    """
    return FakeProvider(script=["Fresh", " answer", " [1]."])


@asynccontextmanager
async def web_client(provider, db_sessionmaker, search_provider, settings=None):
    """App-wired client with the RAG collaborators and an optional web provider.

    Defaults to the deterministic pre-fetch settings these tests exercise (see
    ``make_settings``); pass ``settings`` to run a different mode.
    """
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings or make_settings()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    if search_provider is not None:
        app.state.search_provider = search_provider
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as http_client:
        yield http_client
    app.dependency_overrides.clear()


async def collect_events(client: AsyncClient, payload: dict) -> list[tuple[str, dict]]:
    events: list[tuple[str, dict]] = []
    current = ""
    async with client.stream("POST", CHAT_URL, json=payload) as resp:
        assert resp.status_code == 200
        async for line in resp.aiter_lines():
            if line.startswith("event: "):
                current = line.removeprefix("event: ")
            elif line.startswith("data: "):
                events.append((current, json.loads(line.removeprefix("data: "))))
    return events


async def test_web_search_grounds_and_cites_with_url(db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(citing_provider(), db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    names = [name for name, _ in events]
    assert fake.last_query == FRESH_MESSAGE
    # A searching_web status precedes the tokens; the usage-gated citations
    # follow the answer they back.
    statuses = [d for n, d in events if n == "status"]
    assert any(s["code"] == "searching_web" for s in statuses)
    assert names.index("status") < names.index("token") < names.index("citation")
    citation = next(d for n, d in events if n == "citation")
    assert citation["source_type"] == "web"
    assert citation["url"] == "https://python.org/downloads"
    assert citation["source"] == "Python releases"
    # A search that actually grounded the answer carries no "web contributed
    # nothing" notice.
    codes = [d.get("code") for n, d in events if n == "status"]
    assert "web_search_unavailable" not in codes
    assert "web_search_no_results" not in codes
    assert names[-1] == "done"


async def test_web_search_fetches_pages_when_fetcher_wired(db_sessionmaker) -> None:
    # A wired page fetcher runs through the full chat path, and the answer
    # grounds on the fetched page text rather than the search snippet. The hash
    # embedder retrieves only identical text, so the page body equals the query
    # while the snippet is unrelated: a citation appearing proves the page text
    # (not the snippet) grounded the turn.
    provider = citing_provider()
    fetcher = FakePageFetcher(default=FRESH_MESSAGE)
    fake = FakeSearchProvider(
        results=[
            SearchResult(
                title="Python releases",
                url="https://python.org/downloads",
                snippet="unrelated site boilerplate",
            )
        ]
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = make_memory_store()
    app.state.embedder = FakeEmbedder(dim=32)
    app.state.search_provider = fake
    app.state.page_fetcher = fetcher
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    app.dependency_overrides.clear()
    # The fetcher was invoked for the result URL through the whole chat path.
    assert fetcher.fetched == ["https://python.org/downloads"]
    citation = next(d for n, d in events if n == "citation")
    assert citation["source_type"] == "web"
    # The page body (== the query) grounded the turn; the unrelated snippet would
    # not have retrieved, so the citation content is the fetched page text.
    assert "latest Python version" in citation["snippet"]


async def test_web_search_off_is_fully_local(fake_provider, db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "off"})
    # The provider was never called and nothing about the web appears.
    assert fake.last_query is None
    assert not any(d.get("code") == "searching_web" for n, d in events if n == "status")
    assert "citation" not in [name for name, _ in events]


async def test_web_search_auto_fires_on_freshness_cue(fake_provider, db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "auto"})
    assert fake.last_query == FRESH_MESSAGE
    assert any(d.get("code") == "searching_web" for n, d in events if n == "status")


async def test_web_search_auto_stays_local_without_cue(fake_provider, db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(LOCAL_MESSAGE))
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": LOCAL_MESSAGE, "web_search": "auto"})
    assert fake.last_query is None
    assert not any(d.get("code") == "searching_web" for n, d in events if n == "status")


async def test_web_search_down_degrades_to_local_answer(fake_provider, db_sessionmaker) -> None:
    async with web_client(fake_provider, db_sessionmaker, DownSearchProvider()) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    names = [name for name, _ in events]
    codes = [d.get("code") for n, d in events if n == "status"]
    # We announced the search, it failed, and the turn still answered locally -
    # and told the user the web was unavailable rather than looking web-informed.
    assert "searching_web" in codes
    assert "web_search_unavailable" in codes
    assert "web_search_no_results" not in codes
    assert "error" not in names
    assert "citation" not in names
    assert names[-1] == "done"


async def test_web_search_empty_notes_no_relevant_sources(fake_provider, db_sessionmaker) -> None:
    # The backend is reachable but returns nothing citeable: the announced search
    # contributed no sources, so the turn says it fell back to local knowledge -
    # distinct from the "unavailable" note, since the search itself did run.
    fake = FakeSearchProvider(results=[])
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    codes = [d.get("code") for n, d in events if n == "status"]
    names = [name for name, _ in events]
    assert fake.last_query == FRESH_MESSAGE
    assert "searching_web" in codes
    assert "web_search_no_results" in codes
    assert "web_search_unavailable" not in codes
    assert "citation" not in names
    assert names[-1] == "done"


async def test_web_unavailable_notice_coexists_with_local_citation(db_sessionmaker) -> None:
    # Web search fails, but a stored record still grounds the answer: the turn
    # cites the local record AND tells the user the web itself contributed
    # nothing. The notice keys on "no WEB source", not "no source at all".
    provider = citing_provider()
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    tenant = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
    await rag_service.ingest_structured(
        source_type="records",
        tenant=tenant,
        items=[RagIngestItem(id="kb-1", title="LOCAL_RECORD", text=FRESH_MESSAGE)],
        embedder=embedder,
        store=store,
        settings=make_settings(),
        collection_id="col-1",
    )
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: make_settings()
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = store
    app.state.embedder = embedder
    app.state.search_provider = DownSearchProvider()
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    app.dependency_overrides.clear()
    citations = [d for n, d in events if n == "citation"]
    codes = [d.get("code") for n, d in events if n == "status"]
    # The local record grounded and was cited (not a web source)...
    assert citations
    assert all(c["source_type"] != "web" for c in citations)
    assert any(c["source"] == "LOCAL_RECORD" for c in citations)
    # ...and the failed web search is still surfaced honestly.
    assert "web_search_unavailable" in codes


async def test_web_citation_url_persists_on_reload(db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(citing_provider(), db_sessionmaker, fake) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
        conversation_id = dict(events)["done"]["conversation_id"]
        detail = await client.get(f"/api/v1/conversations/{conversation_id}")
    assistant = detail.json()["messages"][-1]
    assert assistant["role"] == "assistant"
    assert assistant["citations"]
    assert assistant["citations"][0]["url"] == "https://python.org/downloads"
    assert assistant["citations"][0]["source_type"] == "web"


async def test_no_provider_configured_stays_local(fake_provider, db_sessionmaker) -> None:
    # No search provider injected and SEARXNG_URL unset: web_search="on" is a
    # no-op (get_search_provider returns None), the turn is fully local.
    async with web_client(fake_provider, db_sessionmaker, None) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    assert not any(d.get("code") == "searching_web" for n, d in events if n == "status")
    assert "citation" not in [name for name, _ in events]


async def test_web_search_disabled_setting_stays_local(fake_provider, db_sessionmaker) -> None:
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    settings = make_settings(web_search_enabled=False)
    async with web_client(fake_provider, db_sessionmaker, fake, settings) as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    assert fake.last_query is None
    assert not any(d.get("code") == "searching_web" for n, d in events if n == "status")


async def test_web_search_contextualizes_followup_query(db_sessionmaker) -> None:
    # A subject-less follow-up in an ongoing conversation searches - and grounds -
    # on the topic from earlier turns, not the bare fragment. The result text
    # equals the CONTEXT-FOLDED query, so a citation appears only if BOTH the
    # search and the retrieval ranking used that folded query (the hash embedder
    # retrieves identical text only; the bare follow-up would not match it).
    topic = "the political status of Iran"
    followup = "and how is it currently"
    # Message-first: the follow-up leads, the topic trails as the anchor.
    folded = f"{followup} {topic}"
    fake = FakeSearchProvider(
        results=[
            SearchResult(title="Iran politics", url="https://example.org/iran", snippet=folded)
        ]
    )
    async with web_client(citing_provider(), db_sessionmaker, fake) as client:
        # Turn 1 establishes the topic locally (no freshness cue, so no search).
        first = await collect_events(client, {"message": topic, "web_search": "off"})
        conversation_id = dict(first)["done"]["conversation_id"]
        # Turn 2 is the follow-up, with web search on.
        events = await collect_events(
            client,
            {"message": followup, "conversation_id": conversation_id, "web_search": "on"},
        )
    # The search engine received the context-folded query, not the bare follow-up.
    assert fake.last_query == folded
    # And the turn grounded on it: a web citation appears, which requires the
    # retrieval ranking to have used the folded query too.
    citations = [d for n, d in events if n == "citation"]
    assert citations
    assert citations[0]["source_type"] == "web"
    assert citations[0]["source"] == "Iran politics"


async def test_web_search_auto_fires_on_followup_in_fresh_thread(db_sessionmaker) -> None:
    # A cue-less follow-up auto-searches when it continues a thread whose earlier
    # turn warranted a search (2b-i). Turn 1 is a fresh-info question (auto
    # searches); turn 2 carries no cue of its own yet must still search. The
    # result text equals turn 2's folded query, so the citation also proves the
    # turn grounded on the fresh search rather than stale cached context.
    topic = "what is the best tech stack today in 2026"
    followup = "and how would it be for a java backend"
    folded = f"{followup} {topic}"
    fake = FakeSearchProvider(
        results=[SearchResult(title="Java backend", url="https://example.org/java", snippet=folded)]
    )
    async with web_client(citing_provider(), db_sessionmaker, fake) as client:
        first = await collect_events(client, {"message": topic, "web_search": "auto"})
        conversation_id = dict(first)["done"]["conversation_id"]
        events = await collect_events(
            client,
            {"message": followup, "conversation_id": conversation_id, "web_search": "auto"},
        )
    # last_query is turn 2's folded query, proving the cue-less follow-up searched
    # (auto mode would not have fired on the follow-up's own words alone).
    assert fake.last_query == folded
    citations = [d for n, d in events if n == "citation"]
    assert citations
    assert citations[0]["source_type"] == "web"


async def test_web_search_auto_followup_stays_local_without_fresh_thread(
    fake_provider, db_sessionmaker
) -> None:
    # The mirror case: a cue-less follow-up in a thread with no freshness anywhere
    # does NOT auto-search - inheritance is gated on a genuinely fresh thread.
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        first = await collect_events(
            client, {"message": "explain how a hash map works", "web_search": "auto"}
        )
        conversation_id = dict(first)["done"]["conversation_id"]
        await collect_events(
            client,
            {
                "message": "and how would it be for a java backend",
                "conversation_id": conversation_id,
                "web_search": "auto",
            },
        )
    assert fake.last_query is None


async def test_web_search_does_not_contextualize_standalone_followup(
    fake_provider, db_sessionmaker
) -> None:
    # A self-contained second turn (no anaphora / continuation) is searched
    # verbatim despite history - guarding against over-eager context folding.
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    async with web_client(fake_provider, db_sessionmaker, fake) as client:
        first = await collect_events(
            client, {"message": "tell me about databases", "web_search": "off"}
        )
        conversation_id = dict(first)["done"]["conversation_id"]
        await collect_events(
            client,
            {"message": FRESH_MESSAGE, "conversation_id": conversation_id, "web_search": "on"},
        )
    assert fake.last_query == FRESH_MESSAGE


async def test_web_search_does_not_reopen_kb_when_rag_disabled(db_sessionmaker) -> None:
    # RAG_ENABLED=false disables stored-record grounding. A web-search turn must
    # ground ONLY on the freshly-ingested web results, never re-query the
    # operator-disabled knowledge base - even when a stored record would match.
    provider = citing_provider()
    store = make_memory_store()
    embedder = FakeEmbedder(dim=32)
    tenant = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")
    # Seed a kb record whose text equals the query, so it WOULD score 1.0 and be
    # cited if the kb were queried - making a leak unmistakable.
    await rag_service.ingest_structured(
        source_type="records",
        tenant=tenant,
        items=[RagIngestItem(id="kb-secret", title="STORED_KB_SECRET", text=FRESH_MESSAGE)],
        embedder=embedder,
        store=store,
        settings=make_settings(),
        collection_id="col-1",
    )
    fake = FakeSearchProvider(results=web_results(FRESH_MESSAGE))
    settings = make_settings(rag_enabled=False)
    app = create_app()
    app.dependency_overrides[get_model_provider] = lambda: provider
    app.dependency_overrides[get_settings] = lambda: settings
    app.state.db_sessionmaker = db_sessionmaker
    app.state.stream_limiter = StreamLimiter(2)
    # Faked like storage: the lazy fallback opens a real Redis pool, and a first
    # turn now queues the background title job.
    app.state.job_queue = FakeJobQueue()
    app.state.vector_store = store
    app.state.embedder = embedder
    app.state.search_provider = fake
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        events = await collect_events(client, {"message": FRESH_MESSAGE, "web_search": "on"})
    app.dependency_overrides.clear()
    citations = [d for n, d in events if n == "citation"]
    # The web result grounds the turn; the disabled kb record never surfaces.
    assert citations
    assert all(c["source_type"] == "web" for c in citations)
    assert all(c["source"] != "STORED_KB_SECRET" for c in citations)
