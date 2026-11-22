"""A fake OpenID Connect provider, in process, with a real signature.

Real crypto, no network. A locally-generated RSA key signs the id_tokens and is
published through a JWKS document served by an ``httpx.MockTransport``, so the
verification path under test runs its ACTUAL signature check against an actual
key rather than against a patched-out stub. That distinction is the whole point:
a fake that returned "valid" from a mock would leave the one check that matters
untested, and the mutation tests could not tell a working verifier from a
missing one.

The issuer is on loopback so nothing here resolves a name or touches DNS -
``host_is_public`` answers instantly for ``127.0.0.1``, and ``validate_endpoint``
permits plain http there, which is exactly the local-development case it was
written for.

Key generation is the one slow part (an RSA keypair), so the default key is
generated once per session and shared; a test that needs a DIFFERENT key - to
forge a signature - asks for one of its own.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from functools import lru_cache
from typing import Any

import httpx
import jwt
from cryptography.hazmat.primitives.asymmetric import rsa

DEFAULT_ISSUER = "http://127.0.0.1:8765"
DEFAULT_CLIENT_ID = "client-under-test"
DEFAULT_KID = "test-key-1"


@lru_cache(maxsize=4)
def _rsa_key(label: str) -> rsa.RSAPrivateKey:
    """A cached RSA keypair per label - generation is the expensive part."""
    return rsa.generate_private_key(public_exponent=65537, key_size=2048)


class FakeIdp:
    """An OpenID provider that answers discovery, JWKS and token requests.

    Attributes:
        issuer: The issuer this provider declares and is configured as.
        client_id: The audience its tokens are minted for.
        token_requests: Every form body posted to its token endpoint, in order,
            so a test can assert what the flow actually sent (the PKCE verifier
            above all).
    """

    def __init__(
        self,
        *,
        issuer: str = DEFAULT_ISSUER,
        client_id: str = DEFAULT_CLIENT_ID,
        kid: str = DEFAULT_KID,
        key_label: str = "default",
    ) -> None:
        self.issuer = issuer
        # Where the documents are actually SERVED, which is not always the
        # issuer string: a provider whose canonical issuer ends in a slash
        # (Auth0's does) still serves /.well-known/... at the single-slash path,
        # and still puts the slashed form in `iss`. Keeping the two apart is
        # what lets a test drive that shape.
        self._base = issuer.rstrip("/")
        self.client_id = client_id
        self.kid = kid
        self.private_key = _rsa_key(key_label)
        self.token_requests: list[dict[str, str]] = []
        self.jwks_fetches = 0
        self.discovery_fetches = 0
        # What the next token exchange answers with. A test that wants a
        # refusal, a missing id_token, or a specific token replaces this.
        self.token_response: httpx.Response | None = None
        self.next_id_token: str | None = None

    # ---- documents ----------------------------------------------------

    @property
    def authorization_endpoint(self) -> str:
        """Where the consent redirect points."""
        return f"{self._base}/authorize"

    @property
    def token_endpoint(self) -> str:
        """Where the code is exchanged."""
        return f"{self._base}/token"

    @property
    def jwks_uri(self) -> str:
        """Where the signing keys are published."""
        return f"{self._base}/jwks"

    def discovery_document(self, **overrides: Any) -> dict[str, Any]:
        """The openid-configuration document, with optional field overrides."""
        document = {
            "issuer": self.issuer,
            "authorization_endpoint": self.authorization_endpoint,
            "token_endpoint": self.token_endpoint,
            "jwks_uri": self.jwks_uri,
        }
        document.update(overrides)
        return document

    def jwks_document(self) -> dict[str, Any]:
        """The published key set, holding this provider's one public key."""
        public = jwt.algorithms.RSAAlgorithm.to_jwk(self.private_key.public_key(), as_dict=True)
        public.update({"kid": self.kid, "use": "sig", "alg": "RS256"})
        return {"keys": [public]}

    # ---- tokens -------------------------------------------------------

    def id_token(
        self,
        *,
        nonce: str,
        subject: str = "sub-123",
        email: str = "person@example.com",
        email_verified: Any = True,
        name: str | None = "A Person",
        audience: str | None = None,
        issuer: str | None = None,
        expires_in: timedelta = timedelta(minutes=5),
        issued_at: datetime | None = None,
        algorithm: str = "RS256",
        kid: str | None = None,
        signing_key: Any = None,
        omit: tuple[str, ...] = (),
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """Mint an id_token, with every field a test might need to corrupt.

        Args:
            nonce: The nonce claim; the flow requires it to match its own.
            subject: The ``sub`` claim - the identity key.
            email: The ``email`` claim.
            email_verified: Whatever the provider should send; deliberately
                untyped so a test can send the string form or omit truth.
            name: The ``name`` claim, or None to leave it out.
            audience: Override the audience (a token for another client).
            issuer: Override the issuer (a token from another provider).
            expires_in: How far ahead ``exp`` sits; negative for an expired one.
            issued_at: Override ``iat``.
            algorithm: The signing algorithm to declare and use.
            kid: The key id in the header, for the unknown-key path.
            signing_key: Sign with a different key, for the bad-signature path.
            omit: Claim names to leave out entirely.
            extra_claims: Claims to add or override (``azp``, say).

        Returns:
            The encoded token.
        """
        now = issued_at or datetime.now(UTC)
        claims: dict[str, Any] = {
            "iss": issuer if issuer is not None else self.issuer,
            "aud": audience if audience is not None else self.client_id,
            "sub": subject,
            "iat": int(now.timestamp()),
            "exp": int((now + expires_in).timestamp()),
            "nonce": nonce,
            "email": email,
            "email_verified": email_verified,
        }
        if name is not None:
            claims["name"] = name
        claims.update(extra_claims or {})
        for field in omit:
            claims.pop(field, None)
        return jwt.encode(
            claims,
            signing_key if signing_key is not None else self.private_key,
            algorithm=algorithm,
            headers={"kid": kid or self.kid},
        )

    # ---- transport ----------------------------------------------------

    def transport(self) -> httpx.MockTransport:
        """An httpx transport that serves this provider and nothing else.

        Any URL it does not recognise answers 404, so a flow that reaches for an
        address this provider never published fails visibly rather than hanging.
        """

        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if url == f"{self._base}/.well-known/openid-configuration":
                self.discovery_fetches += 1
                return httpx.Response(200, json=self.discovery_document())
            if url == self.jwks_uri:
                self.jwks_fetches += 1
                return httpx.Response(200, json=self.jwks_document())
            if url == self.token_endpoint and request.method == "POST":
                form = dict(httpx.QueryParams(request.content.decode()))
                self.token_requests.append(form)
                if self.token_response is not None:
                    return self.token_response
                return httpx.Response(
                    200,
                    json={
                        "access_token": "an-access-token-nobody-keeps",
                        "token_type": "Bearer",
                        "id_token": self.next_id_token or "",
                    },
                )
            # Includes the RFC 8414 well-known root, which a real OIDC provider
            # does not serve either - so the candidate fallback order is
            # genuinely exercised rather than assumed.
            return httpx.Response(404, json={"error": "not found"})

        return httpx.MockTransport(handler)

    def client(self) -> httpx.AsyncClient:
        """An httpx client wired to this provider."""
        return httpx.AsyncClient(transport=self.transport())

    def json_response(self, status: int, payload: dict[str, Any]) -> httpx.Response:
        """Build a canned token-endpoint response (assign to ``token_response``)."""
        return httpx.Response(
            status, content=json.dumps(payload), headers={"content-type": "application/json"}
        )
