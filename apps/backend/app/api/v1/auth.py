"""Login endpoints: start a sign-in, finish it, end it, and say who is here.

These four are the ONLY ``/api/v1`` routes that do not resolve a principal, and
they cannot: three of them run before there is one, and the fourth exists to
report that there is not. Everything else in the API depends on
``get_current_tenant`` - a guard test walks the live route surface and proves
it, so a new route cannot quietly join this list.

The router stays HTTP-only, as every router here does: it reads the request,
calls :mod:`app.services.auth.login`, and turns the result into a redirect, a
cookie, or a document. No verification, no storage, no provider calls.

**Why the callback answers HTML and not problem+json.** The rest of the API is
called by code, so RFC 7807 is right for it. The callback is opened by a
BROWSER, arriving from somebody else's site, and a person who was refused needs
to read what happened - a raw JSON document is not an answer to them. So the
three refusals here render a small page and say plainly which of the three
things went wrong, without echoing a claim, a token, or an internal detail.
"""

from __future__ import annotations

import logging
from hmac import compare_digest
from html import escape
from typing import Annotated

from fastapi import APIRouter, Depends, Query, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.api.deps import get_session_store, get_sessionmaker, get_settings
from app.core.config import Settings
from app.core.errors import NotFoundError, ServiceUnavailableError
from app.models import User
from app.schemas.auth import AuthMeResponse
from app.services.auth.errors import (
    AdmissionDeniedError,
    LoginConfigurationError,
    LoginProtocolError,
)
from app.services.auth.login import complete_login, start_login
from app.services.auth.principal import SESSION_COOKIE_NAME, resolve_principal
from app.services.auth.store import (
    PENDING_TTL_SECONDS,
    KeyValueStore,
    create_session,
    delete_session,
)
from app.workers.queue import JobQueueUnavailableError

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

# 302 rather than 307: the browser is being sent to a different resource with a
# GET, which is exactly what 302 means, and 307's method preservation is
# meaningless for a redirect the address bar follows.
_REDIRECT_STATUS = 302

# Binds a callback to the BROWSER that started the sign-in.
#
# Single-use ``state`` alone stops a callback being REPLAYED; it does not stop
# it being planted. An attacker can run a real sign-in as themselves up to the
# consent screen, keep the resulting code and state, and then get a victim to
# open that callback URL - at which point the victim's browser is holding a
# session for the ATTACKER'S account, and everything they do next lands
# somewhere the attacker can read. That is login CSRF, and the state value
# cannot detect it because the attacker has the genuine one.
#
# So the state is also written to a short-lived cookie at /auth/login and must
# come back with the callback. The attacker cannot set that cookie on somebody
# else's browser, so a planted callback simply has nothing to match.
LOGIN_STATE_COOKIE_NAME = "login_state"


def _cookie_is_secure(settings: Settings) -> bool:
    """Whether the session cookie may only travel over HTTPS.

    Keyed on the deployment's public origin rather than on a switch of its own:
    a production deployment is served over https by Caddy and gets a Secure
    cookie automatically, while a local http stack gets one that a browser will
    actually send. A separate flag would be one more thing to set wrongly, and
    the wrong setting here is silent - the cookie is simply never sent, and
    every sign-in appears to succeed and then not have happened.
    """
    return settings.oauth_redirect_base_url.strip().lower().startswith("https://")


def _set_session_cookie(response: Response, session_id: str, *, settings: Settings) -> None:
    """Attach the session cookie, with the flags a session cookie must carry.

    ``httponly`` keeps script from reading it, which is what stops one injected
    line of JavaScript from exfiltrating a login. ``samesite=lax`` stops another
    site's form or image from spending it, while still allowing the top-level
    navigation the provider's own redirect performs - ``strict`` would drop the
    cookie on exactly that hop and land every fresh sign-in back on the login
    page.
    """
    response.set_cookie(
        SESSION_COOKIE_NAME,
        session_id,
        max_age=settings.session_ttl_seconds,
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(settings),
        path="/",
    )


def _set_login_state_cookie(response: Response, state: str, *, settings: Settings) -> None:
    """Attach the browser-binding cookie for one sign-in attempt.

    Same flags as the session cookie and the same lifetime as the pending
    record, so it disappears on its own whether the attempt is finished,
    abandoned, or never was one.
    """
    response.set_cookie(
        LOGIN_STATE_COOKIE_NAME,
        state,
        max_age=PENDING_TTL_SECONDS,
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(settings),
        path="/",
    )


def _clear_login_state_cookie(response: Response, *, settings: Settings) -> None:
    """Drop the browser-binding cookie once its attempt is over, either way."""
    response.delete_cookie(
        LOGIN_STATE_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(settings),
    )


def _spent(response: Response, settings: Settings) -> Response:
    """Mark a sign-in attempt finished by dropping its browser-binding cookie.

    Applied on EVERY exit from the callback, success or refusal. A cookie left
    behind would still match the state it was issued for, which is precisely the
    single use this is meant to have.
    """
    _clear_login_state_cookie(response, settings=settings)
    return response


def _clear_session_cookie(response: Response, *, settings: Settings) -> None:
    """Remove the session cookie, matching the attributes it was set with.

    Path and Secure have to match or the browser keeps the original cookie and
    quietly signs the person back in on the next request.
    """
    response.delete_cookie(
        SESSION_COOKIE_NAME,
        path="/",
        httponly=True,
        samesite="lax",
        secure=_cookie_is_secure(settings),
    )


def _refusal_page(title: str, message: str, *, status: int) -> HTMLResponse:
    """A minimal, self-contained refusal page for the callback.

    Deliberately plain HTML with no styling, no asset, and no link back into the
    app: it is rendered at the end of a redirect chain from another origin,
    where a half-loaded themed page reads worse than a sentence that is simply
    legible. Both strings are escaped - the message is written by this module,
    but escaping at the boundary means a future message carrying a value cannot
    become the first injection here.
    """
    body = (
        "<!doctype html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        f"<title>{escape(title)}</title></head>"
        "<body style='font-family:system-ui,sans-serif;margin:3rem auto;max-width:32rem;"
        "line-height:1.5'>"
        f"<h1 style='font-size:1.25rem'>{escape(title)}</h1>"
        f"<p>{escape(message)}</p>"
        "</body></html>"
    )
    return HTMLResponse(content=body, status_code=status)


@router.get("/login")
async def login(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[KeyValueStore, Depends(get_session_store)],
    next: Annotated[str | None, Query(description="Relative path to land on afterwards.")] = None,
) -> RedirectResponse:
    """Start a sign-in: redirect the browser to the identity provider.

    The pending attempt (its nonce, its PKCE verifier, and where to land) is
    stashed under a single-use ``state`` for ten minutes; the callback consumes
    it or it expires. The same ``state`` goes to the browser as a short-lived
    cookie, which is what ties the eventual callback to THIS browser - see
    :data:`LOGIN_STATE_COOKIE_NAME`.

    ``next`` must be a RELATIVE path. An absolute or scheme-bearing value is
    discarded rather than followed - a sign-in endpoint that will redirect
    anywhere is an open redirector wearing the most trustworthy hat on the site.

    Answers 404 where no issuer is configured: there is no sign-in on that
    deployment, so the route genuinely names nothing rather than being a thing
    that exists and refuses.
    """
    if not settings.oidc_enabled:
        raise NotFoundError("This deployment has no sign-in; it is not configured for login.")
    try:
        started = await start_login(settings=settings, store=store, next_path=next)
    except JobQueueUnavailableError as exc:
        raise ServiceUnavailableError(
            "Sign-in cannot be started right now; try again shortly."
        ) from exc
    except LoginConfigurationError as exc:
        logger.warning("auth.login_unavailable reason=%s", exc)
        raise ServiceUnavailableError(
            "The identity provider could not be reached; try again shortly."
        ) from exc
    response = RedirectResponse(started.authorization_url, status_code=_REDIRECT_STATUS)
    _set_login_state_cookie(response, started.state, settings=settings)
    return response


@router.get("/callback")
async def callback(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[KeyValueStore, Depends(get_session_store)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    code: str | None = None,
    state: str | None = None,
    error: str | None = None,
) -> Response:
    """Finish a sign-in: verify what came back, mint a session, and land.

    Unauthenticated by necessity - the provider's redirect arrives here with no
    session - which makes it the most exposed route in this router. Every check
    the flow depends on happens before anything is written: the ``state`` must
    match this browser's own cookie and is then consumed atomically, the code is
    exchanged with the PKCE verifier, and the id_token's signature, issuer,
    audience, expiry and nonce are all verified before an identity is admitted
    or a row is touched.

    ``error`` is the provider's own refusal (the person pressed Cancel, most
    often). It is reported as a refusal rather than as a fault: nothing is
    broken and there is nothing to retry differently.
    """
    if not settings.oidc_enabled:
        raise NotFoundError("This deployment has no sign-in; it is not configured for login.")
    if error:
        # The provider's error CODE is never rendered back: it is attacker-
        # influencable text on a page, and it means nothing to the person.
        return _refusal_page(
            "Sign-in cancelled",
            "The identity provider did not complete the sign-in. Try again.",
            status=400,
        )
    if not code or not state:
        return _refusal_page(
            "Sign-in could not be completed",
            "This sign-in link is incomplete. Start again from the sign-in page.",
            status=400,
        )

    # Before anything is spent on this callback: is it even this browser's?
    # A planted callback carries a genuine state and no matching cookie, and
    # refusing here means it never reaches the token exchange either.
    #
    # Compared as BYTES, not as text. `hmac.compare_digest` raises TypeError on
    # a str holding any non-ASCII character, and `state` is raw query text from
    # an unauthenticated request - so comparing the strings hands anybody a 500
    # with a traceback on the authentication boundary, which is both a log-flood
    # primitive and a hole in this module's "every refusal reads the same" rule.
    expected = request.cookies.get(LOGIN_STATE_COOKIE_NAME)
    if not expected or not compare_digest(expected.encode("utf-8"), state.encode("utf-8")):
        logger.info("auth.callback_unbound")
        # The cookie is deliberately LEFT ALONE here, unlike on every other exit.
        # It belongs to a sign-in this browser may still be in the middle of, and
        # a planted callback must not get to cancel somebody else's attempt as a
        # consolation prize. It expires on its own within ten minutes.
        return _refusal_page(
            "Sign-in could not be completed",
            "This sign-in did not start in this browser. Start again from the sign-in page.",
            status=400,
        )

    try:
        async with sessionmaker() as session:
            completed = await complete_login(
                code=code, state=state, settings=settings, store=store, session=session
            )
            landing = completed.next_path
            user_id, tenant_id, role = (
                completed.user.id,
                completed.user.tenant_id,
                completed.user.role,
            )
            # Committed BEFORE the session is minted: a cookie naming a user row
            # that was never written is a session that fails on its first use.
            await session.commit()
    except JobQueueUnavailableError:
        raise ServiceUnavailableError(
            "Sign-in cannot be completed right now; try again shortly."
        ) from None
    except AdmissionDeniedError as exc:
        return _spent(_refusal_page("Not permitted", str(exc), status=403), settings)
    except LoginProtocolError:
        # Coarse on purpose: a caller probing this endpoint must not learn which
        # check refused them. The real reason is in the log, not on the page.
        logger.info("auth.callback_refused")
        return _spent(
            _refusal_page(
                "Sign-in could not be completed",
                "This sign-in link has expired or was already used. Start again "
                "from the sign-in page.",
                status=400,
            ),
            settings,
        )
    except LoginConfigurationError as exc:
        logger.warning("auth.callback_unavailable reason=%s", exc)
        return _spent(
            _refusal_page(
                "Sign-in is unavailable",
                "The identity provider could not be reached. Try again shortly.",
                status=503,
            ),
            settings,
        )

    session_id = await create_session(
        store,
        user_id=user_id,
        tenant_id=tenant_id,
        ttl_seconds=settings.session_ttl_seconds,
    )
    # One line per mint, with ids only - never the address, which would turn the
    # log into a roster of everyone who has ever signed in.
    logger.info("auth.session_minted user_id=%s tenant_id=%s role=%s", user_id, tenant_id, role)
    response = RedirectResponse(landing, status_code=_REDIRECT_STATUS)
    _set_session_cookie(response, session_id, settings=settings)
    return _spent(response, settings)


@router.post("/logout", status_code=204)
async def logout(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    store: Annotated[KeyValueStore, Depends(get_session_store)],
) -> Response:
    """End the current session, if there is one.

    Deletes the server-side record, so the cookie is worthless the moment this
    returns even if a copy of it survives somewhere - which is the whole reason
    sessions are server-side rather than a signed token.

    Answers 204 whether or not a session was found, and whether or not login is
    configured. Signing out is idempotent by nature, and a sign-out that reports
    "there was nothing to sign out of" tells a caller about somebody else's
    session state for the price of a guessed cookie.
    """
    response = Response(status_code=204)
    session_id = request.cookies.get(SESSION_COOKIE_NAME)
    if session_id:
        try:
            await delete_session(store, session_id)
        except JobQueueUnavailableError:
            raise ServiceUnavailableError(
                "The session could not be ended right now; try again shortly."
            ) from None
    _clear_session_cookie(response, settings=settings)
    return response


@router.get("/me")
async def me(
    request: Request,
    settings: Annotated[Settings, Depends(get_settings)],
    sessionmaker: Annotated[async_sessionmaker[AsyncSession], Depends(get_sessionmaker)],
    store: Annotated[KeyValueStore, Depends(get_session_store)],
) -> AuthMeResponse:
    """Report who the caller is, and whether this deployment has login at all.

    The frontend's first call. It resolves the principal through the SAME path
    every other route does, so what it reports is what a real request would act
    as - not a second, parallel reading of the cookie that could disagree with
    the first.

    401 ``auth_required`` where login is on and there is no valid session, which
    is the response the client turns into a hop to the sign-in page.
    """
    tenant = await resolve_principal(
        request, settings=settings, store=store, sessionmaker=sessionmaker
    )
    # The stored row in BOTH modes, rather than a literal for the no-login case:
    # the seeded principal is a real row with a real name, and inventing a
    # second, hardcoded description of it is how the two drift apart.
    async with sessionmaker() as session:
        user = await session.get(User, tenant.user_id)
    return AuthMeResponse(
        auth="oidc" if settings.oidc_enabled else "none",
        user_id=tenant.user_id,
        email=user.email if user else None,
        display_name=user.display_name if user else "Operator",
        role=tenant.role,
    )
