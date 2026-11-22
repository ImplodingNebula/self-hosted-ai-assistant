"""Acquiring a model: the question asked first, and the download itself.

Two operations that a UI runs back to back but that are deliberately separate
here, because they fail in different ways and one must never imply the other.

**Preflight** asks the remote registry whether a name exists and what it would
cost. Its whole discipline is that a failure to ASK is not an answer: an
unreachable registry reports ``unknown``, never ``not_found``, because being
told a correctly-spelled name does not exist sends someone off to fix the one
thing that was already right.

**Pull** streams the download. It normalizes the backend's own progress dicts
onto the typed wire contract in :mod:`app.schemas.models`, so a client never
learns which runtime is serving it, and it holds an in-process claim on the tag
being fetched - the guard that turns an impatient second click into a refusal
instead of a second download competing with the first for the same disk.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator

from app.core.errors import BadRequestError, ConflictError
from app.providers.base import ModelNotFoundError, ModelProvider, ProviderUnavailableError
from app.providers.model_registry import (
    OllamaRegistryClient,
    RegistryManifestNotFoundError,
    RegistryUnavailableError,
)
from app.schemas.models import (
    ModelPreflightResponse,
    PullDoneEvent,
    PullErrorEvent,
    PullProgressEvent,
)
from app.services.model_names import ModelNameRejection, parse_model_name, require_model_name

logger = logging.getLogger(__name__)

PullEvent = PullProgressEvent | PullDoneEvent | PullErrorEvent

# Downloads currently running in this process, keyed by CANONICAL tag (see
# pull_claim_key) rather than by what somebody typed - otherwise "qwen3",
# "qwen3:latest" and "QWEN3" are three claims on one download and the guard is
# defeated by a spelling.
#
# In-process is the honest scope and the useful one: the case this exists for is
# one person clicking twice, or two tabs of the same page, both of which land on
# the same worker. A cross-process claim would need a lock in Redis to defend
# against a race the deployment shape does not currently produce, and would then
# have to answer what happens when a worker dies holding one.
_in_flight: set[str] = set()

# Substrings the model server uses when a tag simply is not in the registry.
# Matched on because the failure arrives as prose inside a 200 response rather
# than as a status code - the download stream had already started successfully
# by the time the manifest lookup failed.
_NOT_FOUND_MARKERS = (
    "file does not exist",
    "model not found",
    "manifest unknown",
    "repository not found",
)

_NO_PULL_VERB = (
    "This deployment's model backend cannot download models on request - it "
    "serves what it was started with, so its model set changes by redeployment."
)

_REGISTRY_UNREACHABLE_DETAIL = (
    "The model registry could not be reached, so whether this tag exists and "
    "how large it is are both unknown. The download can still be started - it "
    "will report the same problem if the registry is genuinely down."
)

_NOT_FOUND_DETAIL = (
    "The registry has no such tag. Check the spelling, including the part after "
    "the ':' - a model's name and its size variants are separate tags."
)

_HUGGINGFACE_DETAIL = (
    "Hugging Face repositories are not served by this registry, so nothing can "
    "be checked before the download starts."
)


def require_pull_verb(provider: ModelProvider) -> None:
    """Refuse outright when the backend cannot download models at all.

    Never a silent success: a 200 with no download looks exactly like a working
    button, and the model set would simply never change.

    Args:
        provider: The model seam that would perform the download.

    Raises:
        BadRequestError: The backend has no pull verb.
    """
    if not provider.management_capabilities().can_pull:
        raise BadRequestError(_NO_PULL_VERB)


async def preflight(
    *,
    name: str,
    provider: ModelProvider,
    client: OllamaRegistryClient,
) -> ModelPreflightResponse:
    """Ask the registry what a name would cost, without downloading anything.

    Gated on the same capability as the download itself, and for two reasons.
    An answer of "available, 2.0 GB" about a tag this backend can never fetch is
    simply false; and this is the one management endpoint whose work is an
    outbound request to a public host, so a backend that has no use for it
    should not be making it.

    Args:
        name: The requested model name, as typed.
        provider: The model seam, for the capability gate.
        client: The registry client to ask.

    Returns:
        The assembled :class:`ModelPreflightResponse`.

    Raises:
        UnprocessableEntityError: The name failed validation.
        BadRequestError: The backend cannot download models at all.
    """
    parsed = require_model_name(name)
    require_pull_verb(provider)
    common = {
        "name": parsed.text,
        "kind": parsed.kind,
        "warning": parsed.warning,
    }
    if parsed.kind == "huggingface":
        return ModelPreflightResponse(status="unknown", detail=_HUGGINGFACE_DETAIL, **common)
    try:
        facts = await client.tag_facts(namespace=parsed.namespace, repo=parsed.repo, tag=parsed.tag)
    except RegistryManifestNotFoundError:
        return ModelPreflightResponse(status="not_found", detail=_NOT_FOUND_DETAIL, **common)
    except RegistryUnavailableError:
        # Info, not error: an offline laptop reaching a public registry is a
        # perfectly ordinary state, and the answer already says so honestly.
        logger.info("Registry preflight for %s could not be completed.", parsed.text)
        return ModelPreflightResponse(
            status="unknown", detail=_REGISTRY_UNREACHABLE_DETAIL, **common
        )
    return ModelPreflightResponse(
        status="available",
        download_size_bytes=facts.download_size_bytes,
        quantization=facts.quantization,
        parameter_size=facts.parameter_size,
        **common,
    )


def begin_pull(provider: ModelProvider, *, name: str) -> str:
    """Decide everything a download can be refused for, then claim the tag.

    Split out of :func:`stream_pull` so that all of it happens BEFORE the
    response begins: a refusal reaches the client as problem+json rather than
    as an event on a stream a progress bar is already attached to.

    Args:
        provider: The model seam that would perform the download.
        name: The requested model name, as received.

    Returns:
        The normalized tag, now claimed - the caller MUST pair this with
        :func:`release_pull` in a scope that unwinds on every outcome.

    Raises:
        UnprocessableEntityError: The name failed validation.
        BadRequestError: The backend cannot download models at all. Never a
            silent success: a 200 with no download looks exactly like a working
            button, and the model set would simply never change.
        ConflictError: The same tag is already downloading.
    """
    tag = require_model_name(name).text
    require_pull_verb(provider)
    claim_pull(tag)
    return tag


def pull_claim_key(name: str) -> str:
    """Return the one key every spelling of ``name`` is claimed under.

    The duplicate guard is only as good as its key. ``qwen3``, ``qwen3:latest``
    and ``library/qwen3`` all name one download, so keying on what somebody
    typed would let a second click through simply by spelling the model
    differently - and two downloads of the same blobs then race each other onto
    the same disk.

    Case is folded for LIBRARY names only, and on a RUNTIME fact rather than a
    spec one: Ollama lowercases every part of a name while resolving it, so
    ``qwen3:Q4_K_M`` and ``qwen3:q4_k_m`` are one set of local weights and one
    download. (The OCI grammar is no help here - it requires lowercase
    repositories but declares TAGS case-sensitive, and the model library ships
    uppercase tags routinely: ``q4_K_M``, ``Q8_0``.) A Hugging Face repository
    name is genuinely case-sensitive, so folding it would refuse a legitimately
    different download as a duplicate.

    Args:
        name: A model name that has already passed validation.

    Returns:
        The canonical claim key. A name this function cannot parse is keyed on
        its trimmed self, which is never less specific than the input.
    """
    parsed = parse_model_name(name)
    if isinstance(parsed, ModelNameRejection):
        return name.strip()  # unreachable via begin_pull, which validates first
    if parsed.kind == "huggingface":
        return f"hf.co/{parsed.namespace}/{parsed.repo}:{parsed.tag}"
    return f"{parsed.namespace}/{parsed.repo}:{parsed.tag}".lower()


def claim_pull(tag: str) -> None:
    """Claim ``tag`` as downloading in this process, or refuse the duplicate.

    Args:
        tag: The model tag about to be downloaded, in any spelling.

    Raises:
        ConflictError: A download of the same model is already running.
            Different models never conflict - fetching two at once is slow but
            perfectly valid, and only the SAME one twice is the mistake.
    """
    key = pull_claim_key(tag)
    if key in _in_flight:
        raise ConflictError(
            f"'{tag}' is already downloading. Watch the download that is already "
            "running rather than starting a second one."
        )
    _in_flight.add(key)


def release_pull(tag: str) -> None:
    """Release a claim taken by :func:`claim_pull`; safe to call twice."""
    _in_flight.discard(pull_claim_key(tag))


def pulls_in_flight() -> frozenset[str]:
    """Return the CANONICAL keys currently downloading (see :func:`pull_claim_key`)."""
    return frozenset(_in_flight)


def clear_pulls_in_flight() -> None:
    """Drop every claim (test isolation only)."""
    _in_flight.clear()


def normalize_pull_event(raw: object) -> PullEvent | None:
    """Map one backend progress object onto the typed wire contract.

    Pure, so the shapes a real download produces can be pinned without one.
    Objects that carry neither an error nor a status are dropped rather than
    forwarded as an empty frame - a runtime is free to emit bookkeeping lines,
    and a client should not have to filter them.

    Args:
        raw: One decoded progress object from the provider.

    Returns:
        The event to send, or None when there is nothing to say.
    """
    if not isinstance(raw, dict):
        return None
    error = raw.get("error")
    if error:
        message = str(error)
        lowered = message.lower()
        code = (
            "model_not_found"
            if any(marker in lowered for marker in _NOT_FOUND_MARKERS)
            else "provider_unavailable"
        )
        return PullErrorEvent(message=message, code=code)
    status = raw.get("status")
    if not status:
        return None
    completed = _byte_count(raw.get("completed"))
    total = _byte_count(raw.get("total"))
    percent: float | None = None
    if completed is not None and total:
        # Clamped: the last frame of a blob can report a completed count a few
        # bytes past its declared total, and a bar that reads 100.4% looks broken.
        percent = round(min(completed / total, 1.0) * 100, 1)
    digest = raw.get("digest")
    return PullProgressEvent(
        status=str(status),
        completed=completed,
        total=total,
        percent=percent,
        digest=str(digest) if isinstance(digest, str) and digest else None,
    )


def _byte_count(value: object) -> int | None:
    """Read a byte counter, ignoring anything that is not a plain integer."""
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        return None
    return value


async def stream_pull(provider: ModelProvider, *, tag: str) -> AsyncIterator[PullEvent]:
    """Run a download, yielding typed events; never raises past its own stream.

    Every failure becomes a terminal :class:`PullErrorEvent` rather than an
    exception: by the time this runs the response has begun, and a stream that
    simply stops leaves a progress bar frozen at whatever it last showed with
    no way to tell a finished download from a dead one.

    Deliberately does NOT release the tag's claim. The claim outlives this
    generator on purpose - a client that hangs up before the body is ever
    iterated leaves this generator unstarted, and an unstarted generator's
    ``finally`` never runs - so the release belongs to the request scope that
    took it (see :func:`begin_pull`), not to the stream.

    Args:
        provider: The model seam performing the download.
        tag: The normalized tag being downloaded.

    Yields:
        Progress events, then exactly one terminal ``done`` or ``error`` event.
    """
    chunks: AsyncIterator[dict] | None = None
    status = ""
    try:
        chunks = provider.pull_model(tag)
        async for raw in chunks:
            event = normalize_pull_event(raw)
            if event is None:
                continue
            yield event
            if isinstance(event, PullErrorEvent):
                return
            status = event.status
        yield PullDoneEvent(model=tag, status=status or "success")
    except ModelNotFoundError as exc:
        yield PullErrorEvent(message=str(exc), code="model_not_found")
    except ProviderUnavailableError as exc:
        yield PullErrorEvent(message=str(exc), code="provider_unavailable")
    except Exception:
        # Log the real failure server-side; never leak exception text.
        logger.exception("model pull failed for %s", tag)
        yield PullErrorEvent(message="The download failed unexpectedly.", code="internal_error")
    finally:
        # Close the provider's stream promptly on an early exit, so the HTTP
        # connection to the model server is released now rather than at garbage
        # collection. None when pull_model itself raised, which some conforming
        # implementations may do before returning an iterator at all.
        aclose = getattr(chunks, "aclose", None)
        if aclose is not None:
            await aclose()
