"""Mint a signed-in session against a running development stack.

Configure an identity provider and every `/api/v1` route wants a session
cookie, which only a completed consent at that provider produces. An automated
browser cannot complete one - there is a human decision in the middle, by
design - so the whole signed-in surface stops being reachable by anything that
is not a person clicking. That is the gap this closes.

It closes it without inventing anything, because of a decision the session
store already made: sessions are SERVER-SIDE (`app/services/auth/store.py`) -
an opaque id naming a record in Redis. The record IS the credential, so
writing one and handing over its cookie leaves exactly the state a completed
sign-in leaves: same key, same value, read back by the same code on the next
request. This calls the app's own `create_session` against the app's own
store. There is no second implementation of a session here, and nothing on the
login path is relaxed to make it work.

Run it with `make dev-session`, which pipes this file into the backend
container where the app package and its Redis are both reachable.
`EMAIL=<address>` mints for a member who has signed in at least once; `TTL=`
sets the window in minutes.

**Security posture.** Reaching this at all requires a shell on the host running
the stack, and whoever holds that can already read the database and write a
session record by hand - so it grants nothing that access did not already
carry. What it must never be is a tool that runs somewhere nobody meant it to,
which is why it REFUSES under the production compose layout, and refuses again
when the layout is not stated at all.
"""

from __future__ import annotations

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.db.engine import build_engine, build_sessionmaker
from app.models import User
from app.repositories.users import UserRepository
from app.services.auth.principal import SESSION_COOKIE_NAME, seeded_principal
from app.services.auth.store import create_session
from app.workers.queue import ArqJobQueue, JobQueueUnavailableError

# WHICH COMPOSE LAYOUT this stack runs, read straight from the environment.
#
# `DEPLOY_ENV` is a Makefile and compose switch, and the backend's settings do
# not read it - nothing in `app.core.config` names it - so there is no
# `settings.deploy_env` to consult and this reads the variable itself. The make
# target passes the value make resolved from `.env`, the same way the other
# terminal diagnostics pass their own flags.
DEPLOY_ENV = os.environ.get("DEPLOY_ENV", "").strip().lower()
DEV_LAYOUT = "dev"
PROD_LAYOUT = "prod"

# Whose session, and for how long. Read at import, like the sibling
# diagnostics: both arrive per invocation from the make target.
EMAIL = os.environ.get("DEV_SESSION_EMAIL", "").strip()
TTL_MINUTES = os.environ.get("DEV_SESSION_TTL", "").strip()

# Two hours by default: long enough to work through a signed-in surface, short
# enough that a value left in a terminal's scrollback has stopped being a key by
# the end of the afternoon. It is a floor rather than a promise - see
# `_expiry_note` for what the first request does to it.
DEFAULT_TTL_MINUTES = 120

# Two distinguishable failures. REFUSED means the guard declined to run here at
# all; FAILED means it tried and could not - a bad TTL, an unreachable store, a
# name nobody in the workspace answers to.
REFUSED = 2
FAILED = 1


def layout_refusal(deploy_env: str) -> str | None:
    """Say why this must not run here, or None when the development layout is confirmed.

    FAILS CLOSED, and the second half of that is the part worth stating: a
    production layout is refused for the obvious reason, and an UNSTATED layout
    is refused too. The make target always passes the value, so a run without
    one is a hand-assembled command - and the machine somebody assembled it on
    is exactly what is in question.

    Args:
        deploy_env: The `DEPLOY_ENV` value, already stripped and lowercased.

    Returns:
        A message naming the refusal, or None when minting may proceed.
    """
    if deploy_env == DEV_LAYOUT:
        return None
    if deploy_env == PROD_LAYOUT:
        return (
            "dev-session: refused - this stack runs the production layout "
            "(DEPLOY_ENV=prod).\n"
            "Minting a session walks around the identity provider, which is a "
            "development affordance and never a production one. Sign in through the "
            "browser instead."
        )
    stated = f'"{deploy_env}"' if deploy_env else "unset"
    return (
        f"dev-session: refused - DEPLOY_ENV is {stated}, so the compose layout cannot "
        "be established.\n"
        "This writes a credential and runs only where the layout says development, so "
        "an unknown layout is refused rather than assumed. `make dev-session` passes "
        "the value from .env; a hand-assembled command has to state it."
    )


def ttl_seconds(raw: str) -> int:
    """Turn the requested minutes into seconds, defaulting when nothing was asked for.

    Args:
        raw: The `DEV_SESSION_TTL` value, possibly empty.

    Returns:
        How long the minted session lives, in seconds.

    Raises:
        ValueError: The value is not a positive whole number of minutes.
    """
    if not raw:
        return DEFAULT_TTL_MINUTES * 60
    try:
        minutes = int(raw)
    except ValueError:
        raise ValueError(f"TTL must be a whole number of minutes, not {raw!r}") from None
    if minutes <= 0:
        raise ValueError(f"TTL must be a positive number of minutes, not {minutes}")
    return minutes * 60


def normalize_email(raw: str) -> str:
    """Trim and lowercase an address, matching what a login stores.

    Token verification lowercases with `str.lower` before the row is written
    (`app/services/auth/tokens.py`), and admission and the members surface
    compare the same way. A lookup that normalized differently would miss a row
    that is plainly on the roster.

    Args:
        raw: An address as typed.

    Returns:
        The address in the form the users table holds.
    """
    return raw.strip().lower()


def build_store(settings: Settings) -> ArqJobQueue:
    """Build the store sessions live in - the same object a request resolves.

    `app.api.deps.get_session_store` hands routes the process's job queue,
    which doubles as the expiring key-value store a session needs. It is built
    directly here because there is no request to resolve a dependency against:
    same class, same Redis, same keys.

    Args:
        settings: The deployment's configuration (the Redis URL).

    Returns:
        The store, which the caller closes.
    """
    return ArqJobQueue(settings.redis_url)


async def find_member(sessionmaker: async_sessionmaker[AsyncSession], email: str) -> User | None:
    """Find the workspace member carrying this address.

    Reads through the tenant-scoped repository the members surface uses rather
    than a query of its own, so it sees exactly the roster that surface lists -
    which is what the not-found message points at. The workspace is the seeded
    one: this deployment is single-workspace, and every sign-in lands in it.

    Args:
        sessionmaker: A session factory over the deployment's database.
        email: An address already through :func:`normalize_email`.

    Returns:
        The matching users row, or None when the roster holds no such address.
    """
    async with sessionmaker() as session:
        repository = UserRepository(session, seeded_principal())
        for member in await repository.list_all():
            if normalize_email(member.email or "") == email:
                return member
    return None


def unknown_email_message(email: str) -> str:
    """Explain a miss, and name the surface that lists what would not be one."""
    return (
        f'dev-session: no member of this workspace uses the address "{email}".\n'
        "A users row is written by a first sign-in, so the address has to have signed "
        "in here at least once - there is no session to mint for somebody the "
        "deployment has never met.\n"
        "The roster is the console's Members page (GET /api/v1/members); the addresses "
        "listed there are the ones this can mint for. With no EMAIL at all it mints "
        "for the seeded default user, which needs no sign-in."
    )


def _duration(seconds: int) -> str:
    """Read a second count back as the unit it was probably written in."""
    if seconds % 86_400 == 0:
        days = seconds // 86_400
        return f"{days} day" if days == 1 else f"{days} days"
    if seconds % 3_600 == 0:
        hours = seconds // 3_600
        return f"{hours} hour" if hours == 1 else f"{hours} hours"
    minutes = max(1, seconds // 60)
    return f"{minutes} minute" if minutes == 1 else f"{minutes} minutes"


def _expiry_note(settings: Settings) -> str:
    """Say what the requested window actually promises, which is less than it looks.

    A short mint is a short leash only until the session is used: reading a
    session re-arms it with the deployment's own sliding window, so the first
    authenticated request stretches this record to exactly the lifetime any
    other session gets. Worth printing, because "expires in two hours" read on
    its own is a claim this tool cannot keep.
    """
    window = _duration(settings.session_ttl_seconds)
    return (
        "That window is how long an UNUSED session lasts. Reading a session re-arms\n"
        f"it with the deployment's own sliding window ({window}), so the first request\n"
        "carrying this cookie extends it exactly as it extends anybody else's. To end\n"
        "it sooner, sign out - or let it sit."
    )


def _sign_in_off_note() -> str:
    """Warn that a minted session changes nothing while sign-in is off."""
    return (
        "NOTE: sign-in is OFF on this deployment (no OIDC_ISSUER configured), so every\n"
        "request is answered as the seeded default administrator and session cookies are\n"
        "not read at all. The session below is real and was written, but it changes\n"
        "nothing until an issuer is configured - testing a signed-in surface against this\n"
        "stack proves only that the surface works with no sign-in in front of it."
    )


def report(
    *,
    session_id: str,
    ttl: int,
    subject: str,
    user_id: UUID,
    tenant_id: UUID,
    settings: Settings,
) -> None:
    """Print the cookie, what it is for, and the two forms it is usually pasted into.

    Args:
        session_id: The minted session id - the cookie's value, and the credential.
        ttl: The window it was minted with, in seconds.
        subject: Who it acts as, in words.
        user_id: The principal's user row.
        tenant_id: The workspace that principal belongs to.
        settings: The deployment's configuration.
    """
    expires = datetime.now(UTC) + timedelta(seconds=ttl)
    if not settings.oidc_enabled:
        print(_sign_in_off_note())
        print()
    print(f"Session minted for {subject}.")
    print()
    print(f"  user id    {user_id}")
    print(f"  tenant id  {tenant_id}")
    print(f"  cookie     {SESSION_COOKIE_NAME}")
    print(f"  value      {session_id}")
    print(f"  expires    {expires.isoformat(timespec='seconds')} ({_duration(ttl)} from now)")
    print()
    print(_expiry_note(settings))
    print()
    # The two addresses are the layout's DEFAULTS (FRONTEND_PORT / BACKEND_PORT
    # move them, and neither reaches this container), so they are offered as
    # defaults rather than asserted as this deployment's addresses.
    print("Set it on the origin the app is served from, whose relay carries the cookie")
    print("through to the API - by default http://localhost:3000:")
    print()
    print(f'  document.cookie = "{SESSION_COOKIE_NAME}={session_id}; path=/; max-age={ttl}"')
    print()
    print("Or send it straight to the API, on whatever port the backend publishes:")
    print()
    print(f'  curl -H "Cookie: {SESSION_COOKIE_NAME}={session_id}" \\')
    print("    http://localhost:8000/api/v1/auth/me")
    print()
    print("A cookie written from a console is not HttpOnly, which a real sign-in's is:")
    print("script on that origin can read this one back. That is a property of how it")
    print("was set rather than of the session, and one more reason to keep the window")
    print("short.")


async def _resolve_principal(settings: Settings, email: str) -> tuple[UUID, UUID, str] | None:
    """Decide whose session to mint: a named member, or the seeded default user.

    Args:
        settings: The deployment's configuration.
        email: A normalized address, or "" to mint for the seeded default user.

    Returns:
        The user id, the tenant id, and a description - or None when the
        address named nobody, or the roster could not be read (either of which
        has already been reported).
    """
    if not email:
        principal = seeded_principal()
        return (
            principal.user_id,
            principal.tenant_id,
            "the seeded default user (the principal every request acts as before sign-in)",
        )

    engine = None
    try:
        engine = build_engine(settings)
        member = await find_member(build_sessionmaker(engine), email)
    except Exception as exc:  # noqa: BLE001 - any read failure reports the same way
        print(f"dev-session: could not read the workspace roster ({exc}).", file=sys.stderr)
        return None
    finally:
        # Built inside the try, so a URL the engine refuses is a message rather
        # than a traceback - and there is nothing to dispose of when it was.
        if engine is not None:
            await engine.dispose()

    if member is None:
        print(unknown_email_message(email), file=sys.stderr)
        return None
    label = member.email or member.display_name
    return member.id, member.tenant_id, f"{label} ({member.role})"


async def mint(settings: Settings, ttl: int) -> int:
    """Resolve the principal, write the session, and print it.

    Args:
        settings: The deployment's configuration.
        ttl: The session's window in seconds.

    Returns:
        The process exit code.
    """
    resolved = await _resolve_principal(settings, normalize_email(EMAIL))
    if resolved is None:
        return FAILED
    user_id, tenant_id, subject = resolved

    store = build_store(settings)
    try:
        session_id = await create_session(
            store, user_id=user_id, tenant_id=tenant_id, ttl_seconds=ttl
        )
    except JobQueueUnavailableError as exc:
        print(
            f"dev-session: the session store is unreachable ({exc}); is the stack up?",
            file=sys.stderr,
        )
        return FAILED
    finally:
        await store.aclose()

    report(
        session_id=session_id,
        ttl=ttl,
        subject=subject,
        user_id=user_id,
        tenant_id=tenant_id,
        settings=settings,
    )
    return 0


def main() -> int:
    """Guard the layout, then mint - see the module docstring for the whole story."""
    refusal = layout_refusal(DEPLOY_ENV)
    if refusal is not None:
        print(refusal, file=sys.stderr)
        return REFUSED

    try:
        ttl = ttl_seconds(TTL_MINUTES)
    except ValueError as exc:
        print(f"dev-session: {exc}", file=sys.stderr)
        return FAILED

    return asyncio.run(mint(Settings(), ttl))


if __name__ == "__main__":
    sys.exit(main())
