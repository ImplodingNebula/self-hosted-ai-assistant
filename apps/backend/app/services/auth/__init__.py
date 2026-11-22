"""Login: signing a person in against an OpenID Connect provider.

A SIBLING of :mod:`app.services.oauth`, not an extension of it. That package
answers "may this deployment act on a user's behalf against somebody else's
API" - it stores grants, refreshes them, and hands access tokens to MCP tool
calls. This one answers "who is making this request", holds no third-party
tokens at all, and throws the provider's tokens away the moment the identity
inside them has been verified.

They share the low-level pieces where the protocol is genuinely the same - PKCE
and state generation, the metadata walk, byte-capped fetches, endpoint
validation - and those are IMPORTED from the oauth package rather than copied,
so a fix to either lands in both.

The pipeline, in the order a login runs:

1. :mod:`.discovery` - the issuer's own metadata (authorize, token, JWKS).
2. :mod:`.login` - build the consent URL, stash the pending state, and later
   exchange the code.
3. :mod:`.tokens` - verify the returned id_token (signature, iss, aud, exp,
   nonce, email_verified) against :mod:`.jwks`.
4. :mod:`.admission` - decide whether this verified person may be here at all.
5. :mod:`.users` - resolve or create the row keyed on (issuer, sub).
6. :mod:`.store` - mint the server-side session.
7. :mod:`.principal` - and, on every later request, turn the session cookie
   back into the acting principal.
"""

from __future__ import annotations
