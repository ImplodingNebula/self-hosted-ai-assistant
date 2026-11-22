"""Registry-client tests over httpx.MockTransport - no public registry needed."""

from __future__ import annotations

import httpx
import pytest

from app.providers.model_registry import (
    OllamaRegistryClient,
    RegistryManifestNotFoundError,
    RegistryUnavailableError,
    manifest_size,
)

# A realistic config digest, not a stub: the client matches the digest grammar
# before it will fetch a blob, so a short placeholder would make every facts
# test pass for the wrong reason (no blob requested at all).
CONFIG_DIGEST = "sha256:" + "ab12cd34" * 8

MANIFEST = {
    "schemaVersion": 2,
    "config": {"digest": CONFIG_DIGEST, "size": 500},
    "layers": [
        {"digest": "sha256:a", "size": 2_000_000},
        {"digest": "sha256:b", "size": 30_000},
    ],
}


def build_client(handler) -> OllamaRegistryClient:
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return OllamaRegistryClient("https://registry.example", client=client)


async def test_happy_manifest_sums_every_blob() -> None:
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, json=MANIFEST)

    facts = await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")
    assert facts.download_size_bytes == 2_030_500
    # The path is built from the validated parts, in the registry's own shape.
    assert seen[0].url.path == "/v2/library/qwen3/manifests/4b"
    assert "manifest" in seen[0].headers["accept"]


async def test_unknown_tag_is_not_found_not_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(404, json={"errors": [{"code": "MANIFEST_UNKNOWN"}]})

    with pytest.raises(RegistryManifestNotFoundError):
        await build_client(handler).tag_facts(namespace="library", repo="ghost", tag="1b")


async def test_transport_failure_is_unavailable_not_not_found() -> None:
    # The distinction is the point of the whole module: reporting a network
    # blip as a missing model sends someone to fix a correct spelling.
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    with pytest.raises(RegistryUnavailableError):
        await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")


async def test_server_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, text="upstream down")

    with pytest.raises(RegistryUnavailableError):
        await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")


async def test_a_redirect_is_not_followed() -> None:
    # Manifests are served directly; a 3xx means something is intercepting the
    # request, and chasing it would be following a host nobody configured.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"Location": "https://elsewhere.example/x"})

    with pytest.raises(RegistryUnavailableError):
        await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")


async def test_unreadable_body_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text="<html>a captive portal</html>")

    with pytest.raises(RegistryUnavailableError):
        await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")


async def test_a_manifest_without_layers_is_unavailable_not_zero_bytes() -> None:
    # Reporting "0 bytes to download" for a manifest this client could not read
    # would be a fabricated size, which is exactly what preflight must not do.
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"schemaVersion": 2, "manifests": []})

    with pytest.raises(RegistryUnavailableError):
        await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="4b")


# --- the facts that make a size mean something ---


def facts_handler(config: object, *, blob_status: int = 200):
    """Serve the manifest at its path and ``config`` at the blob path."""

    def handler(request: httpx.Request) -> httpx.Response:
        if "/blobs/" in request.url.path:
            return httpx.Response(blob_status, json=config)
        return httpx.Response(200, json=MANIFEST)

    return handler


async def test_tag_facts_reports_quantization_and_parameter_count() -> None:
    """The two fields that turn a byte count into a decision.

    Read under names that say what they are: the registry calls them
    `file_type` and `model_type`, neither of which reads as "quantization" or
    "parameter count" to anybody who has not looked it up.

    MUTATION: swap the two keys in `_config_facts` and the assertion catches it,
    because the values are distinguishable rather than both plausible strings.
    """
    handler = facts_handler({"file_type": "Q4_K_M", "model_type": "8.2B"})
    facts = await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="8b")
    assert facts.download_size_bytes == 2_030_500
    assert (facts.quantization, facts.parameter_size) == ("Q4_K_M", "8.2B")


@pytest.mark.parametrize(
    "config",
    [
        # A registry that publishes no config fields at all.
        {},
        # Present but empty, which is not a fact.
        {"file_type": "  ", "model_type": ""},
        # A shape this client does not recognise.
        "not a config",
        [1, 2, 3],
    ],
)
async def test_an_unreadable_config_still_answers_the_size(config: object) -> None:
    """The decoration must never cost the answer.

    MUTATION: raise instead of returning `(None, None)` from `_config_facts`
    and a registry that serves manifests but publishes no config turns a
    perfectly good "4.7 GB to download" into "the registry could not be
    reached" - wrong, and the more alarming of the two.
    """
    facts = await build_client(facts_handler(config)).tag_facts(
        namespace="library", repo="qwen3", tag="8b"
    )
    assert facts.download_size_bytes == 2_030_500
    assert (facts.quantization, facts.parameter_size) == (None, None)


async def test_a_refused_blob_still_answers_the_size() -> None:
    # A registry that serves manifests and refuses blobs is an ordinary
    # deployment, not a failure of the lookup.
    handler = facts_handler({"file_type": "Q4_K_M"}, blob_status=403)
    facts = await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="8b")
    assert (facts.download_size_bytes, facts.quantization) == (2_030_500, None)


async def test_a_config_digest_that_could_escape_the_blob_path_is_refused() -> None:
    """The digest is interpolated into a URL and the manifest is remote input.

    MUTATION: relax `_DIGEST` to anything non-empty and this client requests a
    path the caller never named.
    """
    asked: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        if "/blobs/" in request.url.path:
            return httpx.Response(200, json={"file_type": "Q4_K_M"})
        return httpx.Response(
            200,
            json={
                "schemaVersion": 2,
                "config": {"digest": "../../../etc/passwd", "size": 5},
                "layers": [{"digest": "sha256:a", "size": 10}],
            },
        )

    facts = await build_client(handler).tag_facts(namespace="library", repo="qwen3", tag="8b")
    assert facts.quantization is None
    # Exactly ONE request: the manifest. Asserting only that no path contains
    # "/blobs/" would be hollow - httpx resolves the dot segments away before
    # the request is made, so the escaped URL never contains that substring
    # either, and the assertion would hold with the guard removed.
    assert asked == ["/v2/library/qwen3/manifests/8b"]


# --- the pure size reader ---


@pytest.mark.parametrize(
    ("document", "expected"),
    [
        (MANIFEST, 2_030_500),
        # A manifest without a config blob is still fully described.
        ({"layers": [{"size": 10}]}, 10),
        # All or nothing: skipping an unreadable layer would report a partial
        # sum as an EXACT size, off by orders of magnitude - the confidently
        # wrong number this whole path exists to avoid.
        ({"layers": [{"size": 10}, {"digest": "x"}]}, None),
        ({"layers": [{"size": 10}, {"size": "big"}]}, None),
        ({"layers": [{"size": True}]}, None),
        ({"layers": [{"size": -1}]}, None),
        # An unreadable CONFIG is tolerated: the layers still describe the
        # download, and the config blob is small enough not to move the answer.
        ({"layers": [{"size": 10}], "config": "junk"}, 10),
        ({"layers": []}, None),
        ({}, None),
        ("not a manifest", None),
        (None, None),
    ],
)
def test_manifest_size_reads_only_real_numbers(document: object, expected: int | None) -> None:
    assert manifest_size(document) == expected
