"""Tests for the page fetcher: stdlib text extraction and defensive HTTP fetch.

The fetcher turns a search-result URL into readable text for grounding. The
extractor is a pure stdlib parser (no network); the HTTP fetcher is exercised
through an httpx MockTransport so no real requests are made, and its SSRF host
guard is checked against IP literals (numeric hosts resolve without DNS).
"""

import asyncio
import socket

import httpx

from app.providers.search import fetcher as fetcher_mod
from app.providers.search.fetcher import (
    FakePageFetcher,
    HttpPageFetcher,
    extract_text,
    host_is_public,
)


async def _always_public(host: str) -> bool:
    """Stand-in for the SSRF host guard so MockTransport fetches reach the client."""
    return True


def _client(handler) -> httpx.AsyncClient:
    """An httpx client backed by a MockTransport handler (no real network)."""
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


# --- extract_text (pure stdlib parser) --------------------------------------


def test_extract_drops_script_and_style() -> None:
    html = (
        "<html><head><style>body{color:red}</style></head>"
        "<body><p>Visible copy</p><script>steal()</script></body></html>"
    )
    text = extract_text(html, max_chars=1000)
    assert "Visible copy" in text
    assert "color" not in text
    assert "steal" not in text


def test_extract_breaks_block_tags_into_lines() -> None:
    text = extract_text("<ul><li>First</li><li>Second</li></ul>", max_chars=1000)
    assert "First" in text and "Second" in text
    # Block tags keep the two items apart; they must not run together.
    assert "FirstSecond" not in text


def test_extract_decodes_entities() -> None:
    text = extract_text("<p>Fish &amp; Chips &lt;tasty&gt;</p>", max_chars=1000)
    assert "Fish & Chips" in text


def test_extract_respects_max_chars() -> None:
    html = "<p>" + "x" * 500 + "</p>"
    assert len(extract_text(html, max_chars=100)) == 100


def test_extract_drops_chrome() -> None:
    # nav/header/footer/aside/form are page chrome - navigation, banners, forms -
    # and must not pollute the grounding text.
    html = (
        "<body><nav>Home About Contact</nav><header>Site Name Login</header>"
        "<p>The actual body content.</p>"
        "<aside>Related links widget</aside><footer>Copyright 2026</footer></body>"
    )
    text = extract_text(html, max_chars=1000)
    assert "The actual body content." in text
    for chrome in ("Home About Contact", "Site Name Login", "Related links widget", "Copyright"):
        assert chrome not in text


def test_extract_prefers_main_region() -> None:
    # When a page marks a <main> region, only its text is kept - the surrounding
    # menus and chrome are dropped even if they were not chrome tags.
    html = (
        "<body><div>Menu One Menu Two Menu Three</div>"
        "<main><h1>Real Headline</h1><p>The article body text.</p></main>"
        "<div>Sidebar promo</div></body>"
    )
    text = extract_text(html, max_chars=1000)
    assert "Real Headline" in text and "The article body text." in text
    assert "Menu One" not in text
    assert "Sidebar promo" not in text


def test_extract_drops_chrome_inside_main() -> None:
    # Chrome tags nested inside <main> are still dropped (skip beats main-keep),
    # while the real content survives.
    html = "<body><main><nav>MENU LINKS</nav><p>Body text here.</p></main></body>"
    text = extract_text(html, max_chars=1000)
    assert "Body text here." in text
    assert "MENU LINKS" not in text


def test_extract_article_does_not_eclipse_body() -> None:
    # <article> commonly marks teaser/promo cards; it must NOT trigger the
    # readability pass, or a small promo would eclipse a real body in a bare div.
    # (Regression for the confirmed review finding.)
    body = "The real article body paragraph with the actual news. " * 6
    html = f"<body><div class='story'><h1>Main Headline</h1><p>{body}</p></div><article>You might also like other stories</article></body>"
    text = extract_text(html, max_chars=2000)
    assert "The real article body paragraph" in text  # primary content survives


def test_extract_falls_back_when_main_is_empty() -> None:
    # A page that opens <main> but leaves it empty must still yield the body.
    html = "<body><main></main><p>Fallback body text.</p></body>"
    text = extract_text(html, max_chars=1000)
    assert "Fallback body text." in text


def test_extract_falls_back_when_main_is_whitespace_only() -> None:
    # A <main> holding only whitespace (data.strip() is falsy) must not flip
    # main-mode on and return an empty extract - it falls back to the body.
    html = "<body><main>   </main><p>Real body text.</p></body>"
    text = extract_text(html, max_chars=1000)
    assert "Real body text." in text


def test_extract_main_with_only_chrome_falls_back() -> None:
    # A <main> whose only content is skipped chrome holds no real text, so
    # extraction falls back to the surrounding body rather than returning nothing.
    html = "<body><main><nav>only menu</nav></main><p>Outside body.</p></body>"
    text = extract_text(html, max_chars=1000)
    assert "Outside body." in text
    assert "only menu" not in text


def test_extract_survives_stray_close_tags() -> None:
    # Orphan close tags (common in live HTML) must not underflow the depth
    # counters and corrupt tracking for the rest of the document.
    html = "</main></nav></article><p>After stray closers.</p>"
    text = extract_text(html, max_chars=1000)
    assert "After stray closers." in text


def test_extract_handles_nested_main_blocks() -> None:
    # The depth counter (not a boolean flag) keeps main-mode correct through
    # nested block structure inside <main>.
    html = "<body><main><section><p>Inner story text.</p></section></main></body>"
    text = extract_text(html, max_chars=1000)
    assert "Inner story text." in text


def test_extract_salvages_malformed_markup() -> None:
    # A lenient parse never raises; whatever was collected comes back.
    text = extract_text("<div><p>Salvaged text", max_chars=1000)
    assert "Salvaged text" in text


def test_extract_handles_omitted_head_close() -> None:
    # HTML5 makes </head> optional and minifiers routinely drop it; the stdlib
    # parser does no implied-end-tag construction, so the body must still be
    # extracted (regression: a <head> skip region would otherwise never close
    # and swallow the whole document).
    html = (
        "<html><head><title>IGNOREDTITLE</title><meta charset=utf-8>"
        "<body><p>Real article content</p></body></html>"
    )
    text = extract_text(html, max_chars=1000)
    assert "Real article content" in text
    assert "IGNOREDTITLE" not in text  # title is skipped by name


def test_extract_skips_title_without_explicit_head() -> None:
    # A <title> outside an explicit <head> must not leak into (or fuse with) the
    # body text - it is skipped by tag name, not via a head region.
    text = extract_text("<title>SiteName</title><body>Hello world content</body>", max_chars=1000)
    assert "SiteName" not in text
    assert "Hello world content" in text


# --- HttpPageFetcher (via MockTransport) -------------------------------------


async def test_fetch_extracts_html(monkeypatch) -> None:
    monkeypatch.setattr(fetcher_mod, "host_is_public", _always_public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/html; charset=utf-8"},
            content=b"<html><body><p>Grounded content here</p></body></html>",
        )

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    text = await fetcher.fetch_text("http://news.test/story")
    assert text is not None and "Grounded content here" in text
    await client.aclose()


async def test_fetch_rejects_non_html_content_type(monkeypatch) -> None:
    monkeypatch.setattr(fetcher_mod, "host_is_public", _always_public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "application/json"}, content=b'{"x":1}')

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    assert await fetcher.fetch_text("http://api.test/data") is None
    await client.aclose()


async def test_fetch_returns_none_on_http_error(monkeypatch) -> None:
    monkeypatch.setattr(fetcher_mod, "host_is_public", _always_public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    assert await fetcher.fetch_text("http://down.test/") is None
    await client.aclose()


async def test_fetch_returns_none_on_empty_extraction(monkeypatch) -> None:
    # A page whose only content is script/style extracts to nothing -> None, so
    # the caller falls back to the snippet rather than ingesting empty text.
    monkeypatch.setattr(fetcher_mod, "host_is_public", _always_public)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<script>x()</script>"
        )

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    assert await fetcher.fetch_text("http://empty.test/") is None
    await client.aclose()


async def test_fetch_caps_bytes_read(monkeypatch) -> None:
    # The read stops at max_bytes even when the server streams far more, so a
    # huge page cannot exhaust memory.
    monkeypatch.setattr(fetcher_mod, "host_is_public", _always_public)
    chunk = b"<p>" + b"A" * 1000 + b"</p>"

    async def body():
        for _ in range(50):  # ~50 KB total if fully read
            yield chunk

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, headers={"content-type": "text/html"}, content=body())

    client = _client(handler)
    fetcher = HttpPageFetcher(max_bytes=2000, client=client)  # cap after ~2 chunks
    text = await fetcher.fetch_text("http://big.test/")
    assert text is not None
    # Far fewer than the 50000 'A's a full read would yield: the cap truncated.
    assert text.count("A") < 5000
    await client.aclose()


async def test_fetch_follows_redirect_to_public_host() -> None:
    # A redirect to another public host is chased and its page extracted. Both
    # hops use a public IP literal so the real host guard passes without DNS.
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/start":
            return httpx.Response(302, headers={"location": "http://93.184.216.34/final"})
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<p>Final page body</p>"
        )

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    text = await fetcher.fetch_text("http://93.184.216.34/start")
    assert text is not None and "Final page body" in text
    await client.aclose()


async def test_fetch_redirect_to_private_host_is_blocked() -> None:
    # SSRF guard must re-run per hop: a public page redirecting to loopback (or
    # cloud metadata) is refused, and the internal request is never issued.
    requested: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requested.append(str(request.url))
        return httpx.Response(302, headers={"location": "http://169.254.169.254/latest/meta-data/"})

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    assert await fetcher.fetch_text("http://93.184.216.34/page") is None
    # Only the public origin was ever contacted; the metadata host was not.
    assert requested == ["http://93.184.216.34/page"]
    await client.aclose()


async def test_fetch_redirect_loop_hits_cap() -> None:
    # An endless redirect chain terminates at the hop cap and returns None.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://93.184.216.34/next"})

    client = _client(handler)
    fetcher = HttpPageFetcher(client=client)
    assert await fetcher.fetch_text("http://93.184.216.34/start") is None
    await client.aclose()


async def test_fetch_rejects_non_http_scheme() -> None:
    # Non-browsable schemes never reach the network (checked before any fetch).
    fetcher = HttpPageFetcher()
    assert await fetcher.fetch_text("ftp://x.test/file") is None
    assert await fetcher.fetch_text("file:///etc/passwd") is None


async def test_fetch_rejects_loopback_ssrf() -> None:
    # SSRF guard: a result pointing at loopback/link-local is refused with no
    # request. IP literals resolve without DNS, so this needs no network.
    fetcher = HttpPageFetcher()
    assert await fetcher.fetch_text("http://127.0.0.1/admin") is None
    assert await fetcher.fetch_text("http://169.254.169.254/latest/meta-data/") is None


# --- host_is_public (SSRF guard) --------------------------------------------


async def test_host_guard_rejects_private_and_special() -> None:
    assert await host_is_public("127.0.0.1") is False
    assert await host_is_public("10.0.0.1") is False
    assert await host_is_public("192.168.1.1") is False
    assert await host_is_public("169.254.169.254") is False  # cloud metadata
    assert await host_is_public("100.64.0.1") is False  # shared / carrier-grade NAT


async def test_host_guard_accepts_public_ip() -> None:
    # A public IP literal resolves to itself without a DNS lookup.
    assert await host_is_public("93.184.216.34") is True


async def test_host_guard_fails_closed_on_resolve_error(monkeypatch) -> None:
    loop = asyncio.get_running_loop()

    async def boom(*args, **kwargs):
        raise socket.gaierror("name does not resolve")

    monkeypatch.setattr(loop, "getaddrinfo", boom)
    assert await host_is_public("nonexistent.invalid") is False


# --- FakePageFetcher ---------------------------------------------------------


async def test_fake_page_fetcher_scripts_and_records() -> None:
    fake = FakePageFetcher(pages={"http://a.test": "AAA"}, default="DEF")
    assert await fake.fetch_text("http://a.test") == "AAA"
    assert await fake.fetch_text("http://b.test") == "DEF"
    assert fake.fetched == ["http://a.test", "http://b.test"]


async def test_fake_page_fetcher_default_is_none() -> None:
    fake = FakePageFetcher()
    assert await fake.fetch_text("http://x.test") is None
