"""Which installed weights a configured tag names, answered by ONE matcher.

Two questions travel together everywhere a stored fact is looked up by model:
what the runtime calls these weights, and which digest they currently are.
They have to be answered by the same rule, because rows about a model are
written under the name the PROVIDER listed while the read side arrives holding
whatever spelling the deployment CONFIGURED. A deployment running ``mistral``
against an installed ``mistral:latest`` that resolved the digest one way and
the row key another would find the weights and then find nothing stored about
them - a stored declaration or a failed verdict that clamps a management card
and not the turn, which is exactly the direction no invalidation may point.

So the matcher lives here, once, and every store that keys on (provider,
model) reads through it: the tool-check verdict log
(:mod:`app.services.earned`) and the policy declaration store
(:mod:`app.services.model_policy`). Membership itself goes through the seam's
:func:`~app.providers.base.installed_tag_matches`, so no backend naming
convention leaks into a service.

The listing is cached in-process per provider with a short TTL. What is
installed changes only by explicit operator action (a pull, a delete), and the
readers sit on the chat path, which must not pay a listing round trip per
turn. The cache is dropped whenever a caller knows the inventory moved.
"""

from __future__ import annotations

import time

from app.providers.base import ModelProvider, installed_tag_matches

# The listing is re-read at most this often. Within the window a pull or a
# delete is not yet visible, which is acceptable for facts that change only by
# operator action - and is what keeps the chat path free of a provider round
# trip per turn.
_TTL_SECONDS = 300.0

# provider_name -> (fetched_at, installed name -> digest)
_digest_cache: dict[str, tuple[float, dict[str, str]]] = {}


def clear_identity_cache() -> None:
    """Drop the cached installed listings (test isolation, forced refresh)."""
    _digest_cache.clear()


async def installed_digests(provider: ModelProvider, provider_name: str) -> dict[str, str]:
    """Return the installed name -> digest map, cached per provider.

    Tags without a digest are omitted rather than mapped to None: a runtime
    that exposes no digest for some weights has given nothing a stored fact
    could bind to, and an entry saying so would only invite a caller to bind
    to the absence.

    Args:
        provider: The model seam to list.
        provider_name: Its configured name (``settings.model_provider``), the
            cache key.

    Returns:
        Installed model names mapped to their digests.
    """
    now = time.monotonic()
    hit = _digest_cache.get(provider_name)
    if hit is not None and now - hit[0] < _TTL_SECONDS:
        return hit[1]
    digests = {info.name: info.digest for info in await provider.list_models() if info.digest}
    _digest_cache[provider_name] = (now, digests)
    return digests


def installed_name_of(tag: str, digests: dict[str, str]) -> str | None:
    """Resolve a configured tag to the name the provider actually lists it under.

    Membership goes through :func:`~app.providers.base.installed_tag_matches`
    (the seam rule: no backend naming convention leaks into services); this
    only reports WHICH installed name matched, so the one matcher answers both
    questions asked of it - what digest is installed, and what the runtime
    calls these weights. See the module docstring for why the second one is
    load-bearing.

    Args:
        tag: The tag as configured, stored, or requested.
        digests: The installed listing from :func:`installed_digests`.

    Returns:
        The provider's own spelling of the tag, or None when it holds no such
        weights.
    """
    if not installed_tag_matches(tag, set(digests)):
        return None
    return tag if tag in digests else f"{tag}:latest"


def digest_of(tag: str, digests: dict[str, str]) -> str | None:
    """Resolve the installed digest a configured tag currently points at.

    Args:
        tag: The tag as configured, stored, or requested.
        digests: The installed listing from :func:`installed_digests`.

    Returns:
        The digest, or None when the tag is not installed or the runtime
        exposes no digest for it. Callers must treat None as "nothing to bind
        a stored fact to", never as a digest value in its own right.
    """
    name = installed_name_of(tag, digests)
    return digests.get(name) if name is not None else None
