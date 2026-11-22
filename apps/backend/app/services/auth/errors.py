"""What can go wrong during a login, as three distinguishable outcomes.

Three, not one, because the caller answers them differently and a person needs
to be told different things:

- :class:`LoginConfigurationError` is the operator's problem (the issuer will
  not answer, the client secret is wrong). The person signing in can do nothing
  about it, so the page says so and names no internals.
- :class:`LoginProtocolError` is a callback that does not add up - an expired
  or replayed ``state``, a token that fails verification, a nonce that does not
  match. Suspicious by nature, so the message stays generic: an attacker
  probing the callback must not be told WHICH check refused them.
- :class:`AdmissionDeniedError` is the one honest refusal. The sign-in worked
  perfectly and this person is simply not allowed in, which is worth saying
  plainly - anything vaguer sends them to look for a fault that is not there.

None of them ever carries a claim value, a token, or a secret: every one of
these is rendered to a browser and written to a log.
"""

from __future__ import annotations


class LoginError(Exception):
    """Base class for a login that could not complete."""


class LoginConfigurationError(LoginError):
    """The deployment's own login configuration or provider is not usable."""


class LoginProtocolError(LoginError):
    """The callback did not satisfy the protocol's checks."""


class AdmissionDeniedError(LoginError):
    """The identity verified, but this deployment does not admit it."""
