"""The provider's signing keys, cached, with key rotation handled.

An id_token names the key that signed it by ``kid``. Providers rotate those
keys on their own schedule and publish the new one only shortly before they
start using it, so a cache that never refetches breaks every login the moment a
rotation lands - and a cache that refetches on every token throws away the
point of caching and hands anyone who can reach the callback a way to make this
process hammer the provider.

The rule here is the middle one: serve from cache, and refetch ONLY on a kid
the cache does not know, at most once per :data:`MIN_REFETCH_SECONDS`. A real
rotation is picked up within that window; a flood of tokens bearing invented
kids costs one fetch a minute and is otherwise refused from memory.

The document itself is fetched under the same byte cap and no-redirect rule as
every other remote document in this codebase - it is the one input that decides
whether a signature is believed.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from jwt import PyJWK, PyJWKSet
from jwt.exceptions import PyJWKError, PyJWKSetError

from app.services.auth.errors import LoginConfigurationError, LoginProtocolError
from app.services.oauth.fetch import OversizedResponseError, read_json_capped

if TYPE_CHECKING:
    import httpx

# How long a known key set is served without asking again. Rotation is handled
# by the unknown-kid path below, so this is only a floor on background staleness.
CACHE_TTL_SECONDS = 3600.0

# The throttle that makes "refetch on an unknown kid" safe to expose to
# unauthenticated callers: at most one fetch per JWKS URI per minute, however
# many forged kids arrive.
MIN_REFETCH_SECONDS = 60.0

# Signature algorithms a login will verify with. Every one is asymmetric.
#
# HS* is absent DELIBERATELY and is the whole reason this list exists: with a
# symmetric algorithm the verification key is the signing key, so a token forged
# with the provider's PUBLIC key as an HMAC secret verifies perfectly. "none" is
# absent for the obvious reason. PyJWT is given this list rather than the
# token's own header algorithm, so a header claiming HS256 is refused before any
# key is loaded.
ALLOWED_ALGORITHMS = frozenset(
    {
        "RS256",
        "RS384",
        "RS512",
        "PS256",
        "PS384",
        "PS512",
        "ES256",
        "ES384",
        "ES512",
        "EdDSA",
    }
)

# A published key set is a handful of keys. A document with thousands is either
# broken or hostile, and each one costs a public-key parse.
_MAX_KEYS = 32


class _Entry:
    """One JWKS URI's cached keys and the clock that governs refetching."""

    __slots__ = ("expires_at", "keys", "last_fetch")

    def __init__(self, keys: dict[str, PyJWK], *, now: float) -> None:
        self.keys = keys
        self.last_fetch = now
        self.expires_at = now + CACHE_TTL_SECONDS


class JwksCache:
    """Key sets by JWKS URI, refetched on an unknown kid and rate-limited."""

    def __init__(self) -> None:
        self._entries: dict[str, _Entry] = {}

    def clear(self) -> None:
        """Forget every cached key set (tests, and a provider change)."""
        self._entries.clear()

    async def key_for(
        self,
        kid: str,
        *,
        jwks_uri: str,
        client: httpx.AsyncClient,
        now: float | None = None,
    ) -> PyJWK:
        """Return the signing key named by ``kid``, fetching if it is new.

        Args:
            kid: The ``kid`` from the token header. Required - a token that
                names no key gives this cache nothing to rotate towards.
            jwks_uri: The provider's key-set URL, from discovery.
            client: The httpx client to fetch with.
            now: Monotonic clock, injectable for tests.

        Returns:
            The parsed key.

        Raises:
            LoginProtocolError: The kid is not in the key set, and either the
                refetch throttle is closed or a fresh fetch did not produce it.
                This is the forged-token path, so it is a protocol failure
                rather than a configuration one.
            LoginConfigurationError: The key set could not be fetched or did
                not parse at all - the provider or the network, not the token.
        """
        clock = time.monotonic() if now is None else now
        entry = self._entries.get(jwks_uri)
        if entry is not None and clock < entry.expires_at and kid in entry.keys:
            return entry.keys[kid]

        # A kid the cache does not know is either a rotation or a forgery, and
        # the two are indistinguishable from here - which is exactly why the
        # refetch is throttled rather than skipped. A STALE entry falls straight
        # through to a refetch: that is the background refresh, not this path.
        if (
            entry is not None
            and clock < entry.expires_at
            and kid not in entry.keys
            and clock - entry.last_fetch < MIN_REFETCH_SECONDS
        ):
            raise LoginProtocolError("the token was signed by an unrecognised key")

        keys = await self._fetch(jwks_uri, client=client)
        self._entries[jwks_uri] = _Entry(keys, now=clock)
        key = keys.get(kid)
        if key is None:
            raise LoginProtocolError("the token was signed by an unrecognised key")
        return key

    async def _fetch(self, jwks_uri: str, *, client: httpx.AsyncClient) -> dict[str, PyJWK]:
        """Fetch and parse a key set into a kid -> key map."""
        try:
            # Streamed under the byte cap and never following a redirect, for
            # the same reason every other remote document here is: the size and
            # the destination are the remote side's choice, and this process
            # also serves every live chat stream.
            async with client.stream("GET", jwks_uri, follow_redirects=False) as response:
                if response.status_code != 200:
                    raise LoginConfigurationError(
                        f"the provider's key set answered HTTP {response.status_code}"
                    )
                raw = await read_json_capped(response, what="JWKS document")
        except OversizedResponseError as exc:
            raise LoginConfigurationError("the provider's key set was oversized") from exc
        except LoginConfigurationError:
            raise
        except Exception as exc:  # noqa: BLE001 - transport and parse failures read alike here
            raise LoginConfigurationError("the provider's key set could not be fetched") from exc

        try:
            key_set = PyJWKSet.from_dict(raw)
        except (PyJWKSetError, PyJWKError, AttributeError, TypeError, ValueError) as exc:
            raise LoginConfigurationError("the provider's key set did not parse") from exc

        keys: dict[str, PyJWK] = {}
        for key in key_set.keys:
            if key.key_id is None:
                # A key nothing can NAME cannot be selected by a token's kid,
                # so keeping it would only widen what a forged header can reach.
                continue
            if key.algorithm_name is not None and key.algorithm_name not in ALLOWED_ALGORITHMS:
                continue
            keys[key.key_id] = key
            if len(keys) >= _MAX_KEYS:
                break
        if not keys:
            raise LoginConfigurationError("the provider's key set carried no usable signing key")
        return keys


# One cache per process: the whole point is that a login does not re-fetch keys
# another login already has.
_default_cache = JwksCache()


def default_jwks_cache() -> JwksCache:
    """Return the process-wide key cache."""
    return _default_cache


def clear_jwks_cache() -> None:
    """Empty the process-wide key cache (tests, and a provider change)."""
    _default_cache.clear()
