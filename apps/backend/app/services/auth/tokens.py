"""Verifying the id_token - the single point where an identity is believed.

Everything downstream (admission, the users row, the session) trusts whatever
comes out of here, so every check the OpenID Connect core spec calls for is
made, and each one is made explicitly rather than left to a library default
that a future upgrade could relax:

- **Signature**, against the provider's published key for the token's ``kid``,
  restricted to asymmetric algorithms (see :data:`~.jwks.ALLOWED_ALGORITHMS` -
  an HS256 header is the classic forgery and is refused before a key is loaded).
- **``iss``** equals the configured issuer. **``aud``** contains this client id.
  Without both, a token minted for a different application at the same provider
  signs in here perfectly.
- **``exp``** and **``iat``** are present and current.
- **``nonce``** equals the one this login generated. This is what binds the
  token to THIS browser's login attempt; without it a token captured elsewhere
  can be replayed into somebody else's callback.
- **``email_verified``** is true, and an email is present. A provider that lets
  a person type any address without proving it would otherwise let anyone claim
  an allowed domain.

The token is never stored and never logged. Nothing above this module sees a
raw token, and no error raised here quotes a claim value - the id_token is a
bearer credential, and a rejection message that echoes it puts it in the log.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import jwt
from jwt.exceptions import InvalidTokenError, PyJWTError

from app.services.auth.errors import LoginProtocolError
from app.services.auth.jwks import ALLOWED_ALGORITHMS

if TYPE_CHECKING:
    import httpx

    from app.services.auth.jwks import JwksCache

# Tolerance for clock skew between this host and the provider, in seconds. Small
# on purpose: it exists so a host a few seconds fast does not refuse a
# just-issued token, not to extend a token's life.
CLOCK_SKEW_SECONDS = 30


@dataclass(frozen=True, slots=True)
class VerifiedIdentity:
    """Who the provider says this is, after every check above has passed.

    Attributes:
        issuer: The issuer that signed the token - half of the identity key.
        subject: The provider's stable, opaque id for this person - the other
            half. Never the email: an email can be reassigned, a ``sub`` is the
            claim the spec promises is stable for the life of the account.
        email: The verified address, lowercased so admission, the stored row,
            and the members surface all agree with a hand-typed .env entry -
            ``lower``, matching every other email comparison in the feature, so
            the login floor and the members demote-guard cannot disagree on who
            is a protected admin.
        display_name: The ``name`` claim, or the email when the provider sends
            none - a person with no display name should still be nameable.
    """

    issuer: str
    subject: str
    email: str
    display_name: str


def _is_verified_email(value: Any) -> bool:
    """Read ``email_verified``, accepting the string form some providers send.

    OpenID Connect declares this a boolean, and several real providers send
    ``"true"`` instead; the spec's own implementer notes call for accepting
    both. Nothing else counts - absent, false, ``"1"``, or any other value is
    NOT verified, so the leniency is exactly one well-known spelling wide.
    """
    if value is True:
        return True
    return isinstance(value, str) and value.strip().casefold() == "true"


async def verify_id_token(
    token: str,
    *,
    issuer: str,
    client_id: str,
    nonce: str,
    jwks_uri: str,
    jwks: JwksCache,
    client: httpx.AsyncClient,
) -> VerifiedIdentity:
    """Verify an id_token and return the identity inside it.

    Args:
        token: The raw id_token from the token response.
        issuer: The configured issuer the token must declare.
        client_id: This deployment's client id, which the audience must cover.
        nonce: The nonce this login generated and stashed with its state.
        jwks_uri: The provider's key-set URL, from discovery.
        jwks: The process key cache (handles rotation).
        client: The httpx client used if a key fetch is needed.

    Returns:
        The verified identity.

    Raises:
        LoginProtocolError: Any check failed. The message is deliberately
            coarse - a caller probing the callback learns that the token was
            refused, never which check refused it.
        LoginConfigurationError: The key set itself could not be fetched or
            parsed, which is the provider or the network rather than the token.
    """
    try:
        header = jwt.get_unverified_header(token)
    except PyJWTError as exc:
        raise LoginProtocolError("the identity token could not be read") from exc

    algorithm = header.get("alg")
    if algorithm not in ALLOWED_ALGORITHMS:
        # Refused before a key is loaded: an "alg": "none" or "HS256" header is
        # a forgery attempt, not a provider this deployment should adapt to.
        raise LoginProtocolError("the identity token names an unacceptable signature algorithm")
    kid = header.get("kid")
    if not isinstance(kid, str) or not kid:
        raise LoginProtocolError("the identity token names no signing key")

    key = await jwks.key_for(kid, jwks_uri=jwks_uri, client=client)

    try:
        claims = jwt.decode(
            token,
            key,
            # The ALLOWLIST, never the header's own choice: handing PyJWT the
            # token's algorithm would let the token decide how it is verified.
            algorithms=sorted(ALLOWED_ALGORITHMS),
            audience=client_id,
            issuer=issuer,
            leeway=CLOCK_SKEW_SECONDS,
            options={"require": ["exp", "iat", "iss", "aud", "sub"]},
        )
    except InvalidTokenError as exc:
        raise LoginProtocolError("the identity token failed verification") from exc

    # OIDC core: when the audience names more than one party, ``azp`` says which
    # of them the token was actually issued TO, and it must be this client. PyJWT
    # is satisfied by an ``aud`` list that merely CONTAINS the client id, so
    # without this a provider that mints multi-audience tokens would let a token
    # issued for a sibling application sign somebody in here.
    audience = claims.get("aud")
    if isinstance(audience, list) and len(audience) > 1 and claims.get("azp") != client_id:
        raise LoginProtocolError("the identity token was issued for another application")

    # Constant-time-ish is not the concern here (the nonce is single-use and
    # already consumed); what matters is that a token carrying no nonce at all
    # cannot pass by comparing None to None.
    token_nonce = claims.get("nonce")
    if not isinstance(token_nonce, str) or token_nonce != nonce:
        raise LoginProtocolError("the identity token does not belong to this sign-in")

    if not _is_verified_email(claims.get("email_verified")):
        raise LoginProtocolError("the provider has not verified this address")
    email = claims.get("email")
    if not isinstance(email, str) or not email.strip():
        raise LoginProtocolError("the provider sent no email address")

    subject = claims.get("sub")
    if not isinstance(subject, str) or not subject.strip():
        raise LoginProtocolError("the identity token carries no subject")

    name = claims.get("name")
    display_name = name.strip() if isinstance(name, str) and name.strip() else email.strip()

    return VerifiedIdentity(
        # Verbatim, NOT rstripped: this is half the stored identity key, and the
        # caller passes the provider's own declared spelling. Normalizing it here
        # would make the key depend on this function's idea of canonical rather
        # than on what the provider says it is called.
        issuer=issuer,
        subject=subject.strip(),
        email=email.strip().lower(),
        display_name=display_name,
    )
