"""Tests for the vision gate: the runtime's report is the whole answer.

Nothing written down admits pixels any more - a tag is mutable, so a
declaration could never have been enough on its own and the measurement always
had to agree. What is left to pin is the FRESHNESS contract around that one
report, which is where the gate differs from every other reader of the same
cache: a stale ``vision=True`` keeps sending pixels to weights a re-pull may
have swapped, so this caller accepts a much younger answer than a card does.
"""

from __future__ import annotations

from app.providers.base import ModelCapabilities, ModelProvider
from app.services import model_facts
from app.services.model_facts import clear_facts_cache, runtime_capabilities
from app.services.vision import can_see


class RecordingProvider(ModelProvider):
    """Counts probes and returns a scripted answer (or raises)."""

    def __init__(self, *, vision: bool = True, raises: Exception | None = None) -> None:
        self.vision = vision
        self.raises = raises
        self.probes = 0

    async def capabilities(self, model: str) -> ModelCapabilities:
        self.probes += 1
        if self.raises is not None:
            raise self.raises
        return ModelCapabilities(vision=self.vision)

    async def chat_stream(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    async def embed(self, *args, **kwargs):  # pragma: no cover - unused
        raise NotImplementedError

    async def list_models(self):  # pragma: no cover - unused
        raise NotImplementedError

    def pull_model(self, name: str):  # pragma: no cover - unused
        raise NotImplementedError

    async def delete_model(self, name: str) -> None:  # pragma: no cover - unused
        raise NotImplementedError


async def resolve(provider: ModelProvider, model: str = "multi:4b") -> bool:
    """Ask the gate about one tag, with the provider name every test uses.

    Args:
        provider: The scripted provider to probe.
        model: The tag to ask about.

    Returns:
        Whether the gate admits pixels.
    """
    return await can_see(provider=provider, provider_name="ollama", model=model)


async def test_a_runtime_that_reports_vision_sends_pixels() -> None:
    provider = RecordingProvider(vision=True)
    assert await resolve(provider) is True
    assert provider.probes == 1


async def test_a_runtime_that_calls_the_model_blind_gets_no_pixels() -> None:
    # The one axis that decides it says no, so the pixels do not go - and a
    # multimodal-looking tag name is not a second opinion.
    provider = RecordingProvider(vision=False)
    assert await resolve(provider) is False


async def test_a_blank_model_is_refused_without_a_probe() -> None:
    # A deployment that has not chosen a chat model yet resolves to the empty
    # tag; probing it would ask the runtime about nothing.
    provider = RecordingProvider(vision=True)
    assert await resolve(provider, "") is False
    assert provider.probes == 0


async def test_probe_failure_resolves_to_no_vision() -> None:
    # "The runtime would not tell us" reads as "the pixels do not go".
    provider = RecordingProvider(raises=TimeoutError("wedged"))
    assert await resolve(provider) is False


async def test_probe_failure_is_cached_only_briefly(monkeypatch) -> None:
    # A transient outage must not pin no-vision for the whole TTL. It IS held
    # for the short failure window the shared cache keeps (the turn path needs
    # that bound, and this gate reads the same cache), and a stale negative is
    # safe in exactly the direction this gate already fails: an unanswered
    # probe resolves to "no images" anyway, so the negative can only withhold
    # the pixels for half a minute - never send them.
    now = 0.0
    monkeypatch.setattr(model_facts.time, "monotonic", lambda: now)
    provider = RecordingProvider(raises=ConnectionError("down"))
    assert await resolve(provider) is False
    provider.raises = None
    assert await resolve(provider) is False
    assert provider.probes == 1

    now = model_facts._FAILURE_TTL_SECONDS
    assert await resolve(provider) is True
    assert provider.probes == 2


async def test_successful_probe_is_cached_per_model() -> None:
    provider = RecordingProvider(vision=True)
    assert await resolve(provider, "multi:4b") is True
    assert await resolve(provider, "multi:4b") is True
    assert provider.probes == 1, "second call must hit the cache"
    # A different tag is a different fact and gets its own probe.
    assert await resolve(provider, "other-vl:2b") is True
    assert provider.probes == 2


async def test_the_gate_reprobes_sooner_than_the_shared_ttl(monkeypatch) -> None:
    # The gate accepts a much fresher answer than a card does, and it is the
    # same cache: the failure directions are opposite. A stale capability read
    # is harmless on a card, while a stale vision=True keeps sending pixels to
    # weights a re-pull may have swapped - which is real on a SHARED remote
    # server, where no in-app invalidation can fire at all.
    now = 0.0
    monkeypatch.setattr(model_facts.time, "monotonic", lambda: now)
    provider = RecordingProvider(vision=True)
    assert await resolve(provider) is True
    assert provider.probes == 1

    now = 90.0  # past the gate's minute, well inside the shared 300s TTL
    assert await resolve(provider) is True
    assert provider.probes == 2, "the gate must not read a 90-second-old answer"

    # The manager path, on the same freshly-cached entry, is content with it.
    assert await runtime_capabilities(provider, "ollama", "multi:4b") is not None
    assert provider.probes == 2


async def test_the_gate_still_fails_closed_when_the_fresh_probe_fails(monkeypatch) -> None:
    now = 0.0
    monkeypatch.setattr(model_facts.time, "monotonic", lambda: now)
    provider = RecordingProvider(vision=True)
    assert await resolve(provider) is True

    now = 90.0
    provider.raises = TimeoutError("wedged")
    # A cached answer exists and is inside the shared TTL, but this caller
    # would not accept it and the re-probe failed - so no pixels, rather than
    # falling back to the stale yes the freshness limit exists to reject.
    assert await resolve(provider) is False


async def test_clearing_the_cache_forces_a_reprobe() -> None:
    provider = RecordingProvider(vision=True)
    await resolve(provider)
    clear_facts_cache()
    await resolve(provider)
    assert provider.probes == 2
