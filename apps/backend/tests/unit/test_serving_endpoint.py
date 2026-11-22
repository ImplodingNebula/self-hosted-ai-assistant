"""The serving-endpoint identity: what a probe row records about WHERE it ran.

Two things are under test. The sanitizer, which has to reduce a configured
model-server URL to a host identity that is safe to keep in a database row, an
API payload and a management page forever - so anything credential-shaped must
never survive it. And the seam wiring, which has to derive that identity from
the URL the provider actually talks to rather than from a second copy of the
setting.
"""

from __future__ import annotations

import pytest

from app.providers.base import serving_endpoint_of
from app.providers.fake_provider import FAKE_SERVING_ENDPOINT, FakeProvider
from app.providers.ollama_provider import OllamaProvider


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("http://ollama:11434", "http://ollama:11434"),
        # A trailing path is dropped: it names a route, not a machine, and a
        # hosted proxy is free to put a routing token in one.
        ("https://pod-abc.proxy.runpod.net/api/", "https://pod-abc.proxy.runpod.net"),
        ("https://pod-abc.proxy.runpod.net?key=shhh", "https://pod-abc.proxy.runpod.net"),
        ("https://host#frag", "https://host"),
        # Case is not identity: two spellings of one host must compare equal,
        # or a verdict would stop certifying because somebody retyped the URL.
        ("HTTP://OLLAMA:11434", "http://ollama:11434"),
        ("http://127.0.0.1:11434/", "http://127.0.0.1:11434"),
        # An IPv6 literal keeps its brackets, without which host and port run
        # together into a string that names neither.
        ("http://[::1]:11434", "http://[::1]:11434"),
        ("https://ollama.example.com", "https://ollama.example.com"),
    ],
)
def test_endpoint_keeps_scheme_host_and_port_only(url: str, expected: str) -> None:
    assert serving_endpoint_of(url) == expected


@pytest.mark.parametrize(
    ("written_one_way", "written_another"),
    [
        # A port that merely restates the scheme's default names no different
        # address, and an operator who adds one has changed nothing.
        ("https://pod.example.com", "https://pod.example.com:443"),
        ("http://ollama", "http://ollama:80"),
        # A trailing dot is the DNS root label.
        ("https://pod.example.com", "https://pod.example.com."),
        # And the folding composes with the rest of the normalization.
        ("https://pod.example.com", "HTTPS://Pod.Example.COM.:443/api?x=1"),
    ],
)
def test_two_spellings_of_one_address_compare_equal(
    written_one_way: str, written_another: str
) -> None:
    """Every unfolded spelling is a certificate voided by an edit that changed nothing.

    Comparison downstream is textual, so an operator who retypes OLLAMA_URL
    with an explicit ``:443`` would otherwise silently un-certify every verdict
    on the machine, with no diagnosis anywhere.
    """
    assert serving_endpoint_of(written_one_way) == serving_endpoint_of(written_another)


def test_a_non_default_port_is_never_folded_away() -> None:
    # The safe direction. Folding two DIFFERENT addresses together is the
    # mistake that hands a machine a certificate it never earned, so only the
    # scheme's own default is ever dropped.
    assert serving_endpoint_of("https://pod.example.com:8443") == "https://pod.example.com:8443"
    assert serving_endpoint_of("http://ollama:11434") != serving_endpoint_of("http://ollama")


@pytest.mark.parametrize(
    "url",
    [
        "https://someone:sup3r-s3cret@ollama.example.com:11434/api",
        "https://token123@ollama.example.com",
        "http://admin:hunter2@127.0.0.1:11434",
    ],
)
def test_endpoint_never_carries_userinfo(url: str) -> None:
    """The adversarial case, and the reason this is a function and not f-strings.

    ``user:token@host`` is a valid address and a common way to reach a hosted
    proxy. This value is stored, returned by an API and rendered on a page -
    every one of which outlives the URL - so a credential that survives here
    is a credential leaked to all three at once.
    """
    endpoint = serving_endpoint_of(url)
    assert endpoint is not None
    assert "@" not in endpoint
    for secret in ("sup3r-s3cret", "token123", "hunter2", "someone", "admin"):
        assert secret not in endpoint
    assert endpoint.endswith(("ollama.example.com", "ollama.example.com:11434", "127.0.0.1:11434"))


@pytest.mark.parametrize(
    "url",
    [
        None,
        "",
        "   ",
        # No authority to read: "ollama:11434" parses with the HOST as the
        # scheme, which would record a machine name as a protocol.
        "ollama:11434",
        "/api/chat",
        # A port that is not a number identifies nothing.
        "http://ollama:not-a-port",
    ],
)
def test_endpoint_is_none_when_nothing_names_a_host(url: str | None) -> None:
    """None is "this deployment did not say" - and it certifies nothing."""
    assert serving_endpoint_of(url) is None


def test_ollama_provider_reports_its_configured_host() -> None:
    # No client is injected: the provider builds one lazily and this call makes
    # no request, which is itself the property under test - reading the host
    # must never cost a round trip on the chat path.
    provider = OllamaProvider(
        base_url="https://someone:secret@pod-abc.proxy.runpod.net/",
        default_model="qwen3:4b-instruct",
    )
    assert provider.serving_endpoint() == "https://pod-abc.proxy.runpod.net"


def test_fake_provider_reports_a_host_and_can_be_scripted_to_another() -> None:
    assert FakeProvider().serving_endpoint() == FAKE_SERVING_ENDPOINT
    assert FakeProvider(serving_endpoint="http://elsewhere:11434").serving_endpoint() == (
        "http://elsewhere:11434"
    )
    assert FakeProvider(serving_endpoint=None).serving_endpoint() is None
