"""Who may sign in - one deny-by-default rule, in one place.

Verifying an identity says who somebody is; it says nothing about whether they
belong here. On a public issuer those are wildly different questions: a
perfectly valid Google id_token can be minted by anyone with a Google account,
so a deployment that admits every verified identity has published its workspace
to the internet.

Three allowlists, any of which admits: the bootstrap admins, an explicitly
listed address, or a listed domain. Everything else is refused. The default -
all three empty - therefore admits NOBODY, which is why
:meth:`~app.core.config.Settings._oidc_configuration_is_complete` insists on at
least one admin address before a deployment with login can boot.

The role decision lives here too, because it is the same question asked once
more: an address in ``OIDC_ADMIN_EMAILS`` is an admin at EVERY login, not only
the first. That standing floor is what makes the key a safe bootstrap - a
workspace cannot lock itself out by demoting its last admin - and it is why the
members surface must refuse to demote a listed address and say where to change
it instead.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from app.core.config import Settings

ROLE_ADMIN = "admin"
ROLE_MEMBER = "member"


def _domain_of(email: str) -> str:
    """The part after the last ``@``, or "" when the address has none."""
    _, separator, domain = email.rpartition("@")
    return domain if separator else ""


def is_admitted(email: str, *, settings: Settings) -> bool:
    """Whether this verified address may sign in to this deployment.

    Args:
        email: The verified address, already lowercased by token verification.
        settings: The deployment's configuration.

    Returns:
        True when the address is listed as an admin, listed explicitly, or its
        domain is listed. False otherwise - there is no other way in.
    """
    candidate = email.strip().lower()
    if not candidate:
        return False
    if candidate in settings.oidc_admin_emails or candidate in settings.oidc_allowed_emails:
        return True
    domain = _domain_of(candidate)
    return bool(domain) and domain in settings.oidc_allowed_domains


def is_env_admin(email: str, *, settings: Settings) -> bool:
    """Whether this address holds the standing admin floor from the environment.

    Separate from :func:`role_for` so the members surface can ask the question
    directly: it needs to mark a row as "administered from .env" and to refuse a
    demotion that would be silently undone at the next login. Both this login
    side and that members side lowercase with ``str.lower`` - they MUST use the
    same operation, or one lane floors an address the other refuses to demote.
    """
    return email.strip().lower() in settings.oidc_admin_emails


def role_for(email: str, *, stored_role: str | None, settings: Settings) -> str:
    """The role this login should end with - the environment floor, then storage.

    A listed address is an admin whatever the row says (the floor). Anyone else
    keeps the role already stored for them, and a first-time login with nothing
    stored is a member. The floor only ever raises: it can promote a stored
    member, and it never demotes a stored admin.

    Args:
        email: The verified address.
        stored_role: The role on the existing users row, or None on first login.
        settings: The deployment's configuration.

    Returns:
        ``"admin"`` or ``"member"``.
    """
    if is_env_admin(email, settings=settings):
        return ROLE_ADMIN
    return ROLE_ADMIN if stored_role == ROLE_ADMIN else ROLE_MEMBER
