"""The login service: discovery, token verification, admission, sessions.

Every check here is one an attacker would want removed, so each test is written
to FAIL if its guard is deleted rather than merely to pass while the guard
happens to exist. The signature is verified against a real locally-generated RSA
key (see :mod:`tests.idp`), not a patched-out stub, so "the token verifies" is a
statement about the shipped code path.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from urllib.parse import parse_qs, urlsplit

import pytest
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import ValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID
from app.models import User
from app.services.auth.admission import is_admitted, is_env_admin, role_for
from app.services.auth.discovery import clear_discovery_cache, discover_provider
from app.services.auth.errors import (
    AdmissionDeniedError,
    LoginConfigurationError,
    LoginProtocolError,
)
from app.services.auth.jwks import JwksCache, clear_jwks_cache
from app.services.auth.login import (
    LOGIN_SCOPES,
    complete_login,
    redirect_uri,
    safe_next_path,
    start_login,
)
from app.services.auth.store import (
    PendingLogin,
    create_session,
    delete_session,
    pending_key,
    read_session,
    session_key,
    take_pending,
)
from app.services.auth.tokens import verify_id_token
from app.services.auth.users import resolve_or_create
from app.workers.queue import FakeJobQueue
from tests.idp import DEFAULT_CLIENT_ID, FakeIdp


@pytest.fixture(autouse=True)
def _fresh_auth_caches() -> None:
    """Isolate the process-wide discovery and key caches between tests."""
    clear_discovery_cache()
    clear_jwks_cache()


def make_settings(**overrides: object) -> Settings:
    """Settings with login configured against the fake provider."""
    base: dict[str, object] = {
        "oidc_issuer": FakeIdp().issuer,
        "oidc_client_id": DEFAULT_CLIENT_ID,
        "oidc_client_secret": "a-client-secret",
        "oidc_admin_emails": ["boss@example.com"],
        "oidc_allowed_domains": ["example.com"],
        "oidc_allowed_emails": [],
        "oauth_redirect_base_url": "http://localhost:3000",
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


# ---------------------------------------------------------------- discovery


async def test_discovery_falls_back_to_the_openid_configuration_location() -> None:
    idp = FakeIdp()

    async with idp.client() as client:
        metadata = await discover_provider(idp.issuer, client=client)

    # The RFC 8414 root 404s on this provider, exactly as a real OIDC-only one
    # does, so reaching the endpoints at all proves the fallback order runs.
    assert metadata.token_endpoint == idp.token_endpoint
    assert metadata.jwks_uri == idp.jwks_uri


async def test_discovery_refuses_metadata_declaring_another_issuer() -> None:
    # RFC 8414's mix-up defense. Without it a look-alike host answers for the
    # configured issuer and supplies its own signing keys - which is a complete
    # bypass of every check downstream.
    idp = FakeIdp()
    original = idp.discovery_document

    idp.discovery_document = lambda **kw: {  # type: ignore[method-assign]
        **original(**kw),
        "issuer": "http://127.0.0.1:9999",
    }

    async with idp.client() as client:
        with pytest.raises(LoginConfigurationError, match="declares issuer"):
            await discover_provider(idp.issuer, client=client)


async def test_discovery_refuses_a_document_missing_an_endpoint() -> None:
    idp = FakeIdp()
    original = idp.discovery_document

    idp.discovery_document = lambda **kw: {  # type: ignore[method-assign]
        k: v for k, v in original(**kw).items() if k != "jwks_uri"
    }

    async with idp.client() as client:
        with pytest.raises(LoginConfigurationError, match="missing an endpoint"):
            await discover_provider(idp.issuer, client=client)


async def test_discovery_is_cached_so_a_second_login_pays_nothing() -> None:
    idp = FakeIdp()

    async with idp.client() as client:
        await discover_provider(idp.issuer, client=client)
        await discover_provider(idp.issuer, client=client)

    assert idp.discovery_fetches == 1


async def test_a_public_issuer_may_not_name_a_private_endpoint() -> None:
    # The SSRF anchor. A metadata document is fetched over the network, and one
    # naming an internal address as the token endpoint would have this process
    # POST the client secret and the authorization code straight at an internal
    # service. The issuer here is an ADDRESS LITERAL on purpose: it resolves
    # instantly and globally, so the test needs no DNS and no patched guard.
    idp = FakeIdp(issuer="https://93.184.216.34")
    original = idp.discovery_document

    idp.discovery_document = lambda **kw: {  # type: ignore[method-assign]
        **original(**kw),
        "token_endpoint": "https://10.0.0.5/collect",
    }

    async with idp.client() as client:
        with pytest.raises(LoginConfigurationError, match="token endpoint"):
            await discover_provider(idp.issuer, client=client)


async def test_a_private_issuer_may_name_private_endpoints() -> None:
    # The other half of the same rule, and the reason it is not simply "refuse
    # private": a self-hosted provider on the compose network or on loopback is
    # a legitimate deployment, and the operator pointed the app there in .env.
    idp = FakeIdp()

    async with idp.client() as client:
        metadata = await discover_provider(idp.issuer, client=client)

    assert metadata.token_endpoint.startswith("http://127.0.0.1")


# ------------------------------------------------------------ verification


async def _verified(idp: FakeIdp, token: str, *, nonce: str = "the-nonce"):
    async with idp.client() as client:
        return await verify_id_token(
            token,
            issuer=idp.issuer,
            client_id=idp.client_id,
            nonce=nonce,
            jwks_uri=idp.jwks_uri,
            jwks=JwksCache(),
            client=client,
        )


async def test_a_well_formed_token_verifies_and_yields_the_identity() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", email="Person@Example.COM")

    identity = await _verified(idp, token)

    assert identity.subject == "sub-123"
    # Casefolded, so admission and the stored row agree with a hand-typed
    # .env entry rather than depending on how the provider capitalises.
    assert identity.email == "person@example.com"
    assert identity.display_name == "A Person"


async def test_a_token_signed_by_another_key_is_refused() -> None:
    idp = FakeIdp()
    impostor = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    # Same kid, so the published key IS selected - and must then fail to verify.
    token = idp.id_token(nonce="the-nonce", signing_key=impostor)

    with pytest.raises(LoginProtocolError, match="failed verification"):
        await _verified(idp, token)


async def test_an_hmac_token_is_refused_before_a_key_is_loaded() -> None:
    # The algorithm-confusion forgery: sign with the provider's PUBLIC key as an
    # HMAC secret and a verifier that trusts the header's own algorithm accepts
    # it. Refused on the allowlist, so it never reaches a key at all.
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", algorithm="HS256", signing_key="not-a-secret")

    with pytest.raises(LoginProtocolError, match="unacceptable signature algorithm"):
        await _verified(idp, token)


async def test_a_token_for_another_client_is_refused() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", audience="some-other-app")

    with pytest.raises(LoginProtocolError, match="failed verification"):
        await _verified(idp, token)


async def test_a_token_from_another_issuer_is_refused() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", issuer="http://127.0.0.1:9999")

    with pytest.raises(LoginProtocolError, match="failed verification"):
        await _verified(idp, token)


async def test_an_expired_token_is_refused() -> None:
    idp = FakeIdp()
    token = idp.id_token(
        nonce="the-nonce",
        issued_at=datetime.now(UTC) - timedelta(hours=2),
        expires_in=timedelta(minutes=5),
    )

    with pytest.raises(LoginProtocolError, match="failed verification"):
        await _verified(idp, token)


async def test_a_token_with_the_wrong_nonce_is_refused() -> None:
    # Replay protection: a token obtained through somebody else's login cannot
    # be injected into this callback, because it carries their nonce.
    idp = FakeIdp()
    token = idp.id_token(nonce="a-different-attempt")

    with pytest.raises(LoginProtocolError, match="does not belong to this sign-in"):
        await _verified(idp, token)


async def test_a_token_with_no_nonce_at_all_is_refused() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", omit=("nonce",))

    with pytest.raises(LoginProtocolError, match="does not belong to this sign-in"):
        await _verified(idp, token)


@pytest.mark.parametrize("verified", [False, "false", None, "yes", 1])
async def test_an_unverified_address_is_refused(verified: object) -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", email_verified=verified)

    with pytest.raises(LoginProtocolError, match="not verified this address"):
        await _verified(idp, token)


async def test_the_string_form_of_email_verified_is_accepted() -> None:
    # OpenID Connect declares this a boolean and several real providers send
    # the string; the leniency is exactly one spelling wide (see the case above).
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", email_verified="true")

    assert (await _verified(idp, token)).email == "person@example.com"


async def test_a_token_with_no_email_is_refused() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", omit=("email",))

    with pytest.raises(LoginProtocolError, match="no email address"):
        await _verified(idp, token)


async def test_the_display_name_falls_back_to_the_address() -> None:
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", name=None)

    assert (await _verified(idp, token)).display_name == "person@example.com"


# -------------------------------------------------------------------- jwks


async def test_an_unknown_key_id_triggers_exactly_one_refetch() -> None:
    # Rotation handling: a kid the cache has never seen is worth one fetch,
    # because it is how a real rotation arrives.
    idp = FakeIdp()
    cache = JwksCache()

    async with idp.client() as client:
        await cache.key_for(idp.kid, jwks_uri=idp.jwks_uri, client=client, now=0.0)
        assert idp.jwks_fetches == 1
        with pytest.raises(LoginProtocolError):
            await cache.key_for("rotated-in", jwks_uri=idp.jwks_uri, client=client, now=200.0)

    assert idp.jwks_fetches == 2


async def test_forged_key_ids_cannot_make_the_provider_be_hammered() -> None:
    # ... and the same path throttled, so a flood of invented kids costs one
    # fetch a minute rather than one fetch per token.
    idp = FakeIdp()
    cache = JwksCache()

    async with idp.client() as client:
        await cache.key_for(idp.kid, jwks_uri=idp.jwks_uri, client=client, now=0.0)
        for tick in range(20):
            with pytest.raises(LoginProtocolError, match="unrecognised key"):
                await cache.key_for(
                    f"forged-{tick}", jwks_uri=idp.jwks_uri, client=client, now=1.0 + tick
                )

    assert idp.jwks_fetches == 1


async def test_a_rotated_key_is_picked_up_once_the_throttle_opens() -> None:
    idp = FakeIdp()
    cache = JwksCache()

    async with idp.client() as client:
        await cache.key_for(idp.kid, jwks_uri=idp.jwks_uri, client=client, now=0.0)
        idp.kid = "the-new-key"
        key = await cache.key_for("the-new-key", jwks_uri=idp.jwks_uri, client=client, now=120.0)

    assert key.key_id == "the-new-key"


# --------------------------------------------------------------- admission


@pytest.mark.parametrize(
    ("email", "admitted"),
    [
        ("boss@example.com", True),
        ("someone@example.com", True),
        ("named@other.test", True),
        ("stranger@other.test", False),
        ("stranger@evil.example", False),
        ("", False),
        ("no-at-sign", False),
    ],
)
def test_admission_is_deny_by_default(email: str, admitted: bool) -> None:
    settings = make_settings(oidc_allowed_emails=["named@other.test"])

    assert is_admitted(email, settings=settings) is admitted


def test_a_deployment_with_no_lists_admits_only_its_bootstrap_admins() -> None:
    settings = make_settings(oidc_allowed_domains=[], oidc_allowed_emails=[])

    assert is_admitted("boss@example.com", settings=settings) is True
    assert is_admitted("anyone@example.com", settings=settings) is False


def test_a_domain_entry_written_with_an_at_sign_still_matches() -> None:
    settings = Settings(
        oidc_issuer="http://127.0.0.1:8765",
        oidc_client_id="c",
        oidc_client_secret="s",
        oidc_admin_emails="boss@example.com",
        oidc_allowed_domains="@Example.COM",
    )

    assert is_admitted("person@example.com", settings=settings) is True


def test_the_env_admin_floor_promotes_and_never_demotes() -> None:
    settings = make_settings()

    assert role_for("boss@example.com", stored_role="member", settings=settings) == "admin"
    assert role_for("other@example.com", stored_role="admin", settings=settings) == "admin"
    assert role_for("other@example.com", stored_role="member", settings=settings) == "member"
    assert role_for("other@example.com", stored_role=None, settings=settings) == "member"
    assert is_env_admin("BOSS@example.com", settings=settings) is True


def test_the_admin_floor_normalizes_with_lower_to_match_the_members_lane() -> None:
    # The members surface (lane C) decides who is a protected env-admin by
    # lowercasing both sides with str.lower. The login floor MUST use the same
    # operation, or an .env entry floors nobody at sign-in while the members API
    # still refuses to demote it - a split-brain over who is a protected admin.
    #
    # str.lower and str.casefold agree on ASCII and DIVERGE on a handful of code
    # points, so the assertions below use the German sharp-s (U+00DF), where
    # lower keeps the character and casefold expands it to "ss". Swapping either
    # the list parse or the comparison to casefold changes what these observe -
    # which is what makes this pin the choice rather than merely describe it.
    sharp_s = "ß"  # the address is "stra<sharp-s>e@example.com"
    entry = f"Stra{sharp_s}e@example.com"
    lowered = f"stra{sharp_s}e@example.com"

    parsed = Settings(
        oidc_issuer="http://127.0.0.1:8765",
        oidc_client_id="c",
        oidc_client_secret="s",
        oidc_admin_emails=entry,
    ).oidc_admin_emails
    # lower keeps the sharp-s; casefold would have stored "strasse@..." here.
    assert parsed == [lowered]

    settings = make_settings(oidc_admin_emails=entry)
    # The same address from the provider is floored under lower; under casefold
    # the input becomes "strasse@..." and misses the sharp-s list entry.
    assert is_env_admin(entry, settings=settings) is True
    assert role_for(lowered, stored_role="member", settings=settings) == "admin"


async def test_a_verified_email_is_lowercased_the_same_way_the_floor_expects(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The token-side normalization has to be the same str.lower, or a provider
    # sending a sharp-s address would be stored (and admission-checked) as
    # "strasse@..." while the floor list holds "stra<sharp-s>e@..." - the same
    # split, one layer down.
    sharp_s = "ß"
    entry = f"Stra{sharp_s}e@example.com"
    idp = FakeIdp()
    settings = make_settings(oidc_admin_emails=entry, oidc_allowed_domains=["example.com"])

    completed, _store, _started = await _run_login(
        idp, settings, db_sessionmaker, subject="sub-sharp", email=entry
    )

    assert completed.user.email == f"stra{sharp_s}e@example.com"
    assert completed.user.role == "admin"


# ------------------------------------------------------------------- store


async def test_a_pending_login_is_consumed_exactly_once() -> None:
    # Single-use state is what shuts the authorization-code-injection door; a
    # second callback bearing the same state must find nothing.
    store = FakeJobQueue()
    pending = PendingLogin(nonce="n", code_verifier="v", next_path="/")
    await store.kv_set(pending_key("the-state"), pending.dumps(), 600)

    first = await take_pending(store, "the-state")
    second = await take_pending(store, "the-state")

    assert first is not None
    assert second is None


async def test_reading_a_session_slides_its_expiry() -> None:
    store = FakeJobQueue()
    expiries: list[int] = []
    original = store.kv_expire

    async def _record(key: str, ttl_seconds: int) -> None:
        expiries.append(ttl_seconds)
        await original(key, ttl_seconds)

    store.kv_expire = _record  # type: ignore[method-assign]
    session_id = await create_session(
        store, user_id=DEFAULT_USER_ID, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )

    record = await read_session(store, session_id, ttl_seconds=99)

    assert record is not None
    assert record.user_id == DEFAULT_USER_ID
    assert expiries == [99]


async def test_a_deleted_session_is_gone_immediately() -> None:
    store = FakeJobQueue()
    session_id = await create_session(
        store, user_id=DEFAULT_USER_ID, tenant_id=DEFAULT_TENANT_ID, ttl_seconds=99
    )

    await delete_session(store, session_id)

    assert await read_session(store, session_id, ttl_seconds=99) is None


async def test_an_unparseable_session_value_is_dropped_not_trusted() -> None:
    store = FakeJobQueue()
    await store.kv_set(session_key("hand-written"), "{not json", 99)

    assert await read_session(store, "hand-written", ttl_seconds=99) is None
    assert session_key("hand-written") not in store.values


# ------------------------------------------------------------- next paths


@pytest.mark.parametrize(
    "value",
    [
        "https://evil.example/steal",
        "//evil.example/steal",
        "/\\evil.example",
        "http://127.0.0.1/admin",
        "javascript:alert(1)",
        "evil.example",
        None,
        "",
    ],
)
def test_only_a_relative_path_survives_the_round_trip(value: str | None) -> None:
    # A sign-in endpoint that will redirect anywhere is an open redirector
    # wearing the most trustworthy hat on the site.
    assert safe_next_path(value) == "/"


@pytest.mark.parametrize("value", ["/", "/console/models", "/chat?id=7"])
def test_a_relative_path_is_kept_verbatim(value: str) -> None:
    assert safe_next_path(value) == value


# ---------------------------------------------------------------- the flow


async def test_start_login_builds_a_consent_url_and_stashes_the_attempt() -> None:
    idp = FakeIdp()
    settings = make_settings()
    store = FakeJobQueue()

    async with idp.client() as client:
        started = await start_login(
            settings=settings, store=store, next_path="/console/models", http_client=client
        )

    query = parse_qs(urlsplit(started.authorization_url).query)
    assert urlsplit(started.authorization_url).path == "/authorize"
    assert query["client_id"] == [DEFAULT_CLIENT_ID]
    assert query["redirect_uri"] == [redirect_uri(settings)]
    assert query["scope"] == [" ".join(LOGIN_SCOPES)]
    assert query["code_challenge_method"] == ["S256"]
    assert query["state"] == [started.state]
    # The nonce goes out with the request and must be the one stashed, or the
    # token that comes back can never match it.
    pending = await take_pending(store, started.state)
    assert pending is not None
    assert query["nonce"] == [pending.nonce]
    assert pending.next_path == "/console/models"


async def test_start_login_refuses_when_no_issuer_is_configured() -> None:
    settings = Settings(oidc_issuer=None)

    with pytest.raises(LoginConfigurationError, match="not configured"):
        await start_login(settings=settings, store=FakeJobQueue())


async def _run_login(
    idp: FakeIdp,
    settings: Settings,
    sessionmaker: async_sessionmaker[AsyncSession],
    *,
    store: FakeJobQueue | None = None,
    **token_kwargs: object,
):
    """Drive a full start-to-callback login against the fake provider."""
    store = store or FakeJobQueue()
    async with idp.client() as client:
        started = await start_login(settings=settings, store=store, http_client=client)
        raw = await store.kv_get(pending_key(started.state))
        assert raw is not None
        nonce = PendingLogin.loads(raw).nonce  # type: ignore[union-attr]
        idp.next_id_token = idp.id_token(nonce=nonce, **token_kwargs)  # type: ignore[arg-type]
        async with sessionmaker() as session:
            completed = await complete_login(
                code="the-code",
                state=started.state,
                settings=settings,
                store=store,
                session=session,
                http_client=client,
            )
            await session.commit()
    return completed, store, started


async def test_a_full_login_creates_the_user_and_sends_the_pkce_verifier(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()

    completed, _store, _started = await _run_login(idp, settings, db_sessionmaker)

    assert completed.user.email == "person@example.com"
    assert completed.user.oidc_sub == "sub-123"
    assert completed.user.tenant_id == DEFAULT_TENANT_ID
    assert completed.user.role == "member"
    # A verifier that never reached the provider would leave the exchange
    # unbound to the process that started it.
    assert idp.token_requests[0]["code_verifier"]
    assert idp.token_requests[0]["client_secret"] == "a-client-secret"


async def test_a_login_never_lands_on_the_seeded_pre_auth_user(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The seeded user owns everything created before login existed. Resolving to
    # it would hand the first person who signs in somebody else's history and
    # their stored integration credentials.
    idp = FakeIdp()
    completed, _store, _started = await _run_login(idp, make_settings(), db_sessionmaker)

    assert completed.user.id != DEFAULT_USER_ID


async def test_a_second_login_reuses_the_row_and_refreshes_its_claims(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()

    first, _s, _t = await _run_login(idp, settings, db_sessionmaker)
    second, _s2, _t2 = await _run_login(
        idp, settings, db_sessionmaker, email="person@example.com", name="Renamed Person"
    )

    assert second.user.id == first.user.id
    assert second.user.display_name == "Renamed Person"


async def test_a_listed_admin_is_promoted_at_every_login(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The standing floor, written through to the row - not an overlay - so the
    # members surface and every later request read the same role.
    idp = FakeIdp()
    settings = make_settings()

    async with db_sessionmaker() as session:
        demoted = User(
            tenant_id=DEFAULT_TENANT_ID,
            email="boss@example.com",
            display_name="Boss",
            role="member",
            oidc_issuer=idp.issuer,
            oidc_sub="sub-boss",
        )
        session.add(demoted)
        await session.commit()
        demoted_id = demoted.id

    completed, _s, _t = await _run_login(
        idp, settings, db_sessionmaker, subject="sub-boss", email="boss@example.com"
    )

    assert completed.user.id == demoted_id
    assert completed.user.role == "admin"
    async with db_sessionmaker() as session:
        stored = await session.get(User, demoted_id)
        assert stored is not None
        assert stored.role == "admin"


async def test_an_unadmitted_identity_is_refused_and_writes_no_row(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()

    with pytest.raises(AdmissionDeniedError):
        await _run_login(idp, settings, db_sessionmaker, email="stranger@evil.example")

    async with db_sessionmaker() as session:
        rows = (await session.execute(select(User).where(User.oidc_sub.is_not(None)))).all()
    assert rows == []


async def test_the_same_state_cannot_be_used_twice(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()
    store = FakeJobQueue()

    _completed, _store, started = await _run_login(idp, settings, db_sessionmaker, store=store)

    async with idp.client() as client, db_sessionmaker() as session:
        with pytest.raises(LoginProtocolError, match="expired or was already used"):
            await complete_login(
                code="the-code",
                state=started.state,
                settings=settings,
                store=store,
                session=session,
                http_client=client,
            )


async def test_an_unknown_state_is_refused_before_any_network_call(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()

    async with idp.client() as client, db_sessionmaker() as session:
        with pytest.raises(LoginProtocolError):
            await complete_login(
                code="the-code",
                state="never-issued",
                settings=settings,
                store=FakeJobQueue(),
                session=session,
                http_client=client,
            )

    assert idp.token_requests == []


async def test_a_refused_exchange_is_reported_without_echoing_the_request(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()
    store = FakeJobQueue()
    idp.token_response = idp.json_response(400, {"error": "invalid_grant", "hint": "secret-ish"})

    async with idp.client() as client:
        started = await start_login(settings=settings, store=store, http_client=client)
        async with db_sessionmaker() as session:
            with pytest.raises(LoginProtocolError) as caught:
                await complete_login(
                    code="the-code",
                    state=started.state,
                    settings=settings,
                    store=store,
                    session=session,
                    http_client=client,
                )

    assert "secret-ish" not in str(caught.value)


async def test_a_token_response_with_no_id_token_is_a_configuration_failure(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    idp = FakeIdp()
    settings = make_settings()
    store = FakeJobQueue()
    idp.token_response = idp.json_response(200, {"access_token": "only-this"})

    async with idp.client() as client:
        started = await start_login(settings=settings, store=store, http_client=client)
        async with db_sessionmaker() as session:
            with pytest.raises(LoginConfigurationError, match="no id_token"):
                await complete_login(
                    code="the-code",
                    state=started.state,
                    settings=settings,
                    store=store,
                    session=session,
                    http_client=client,
                )


async def test_the_provider_access_token_is_never_stored(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # This flow wants to know WHO signed in, not to act at the provider. A token
    # nothing calls is a credential held for no reason.
    idp = FakeIdp()
    _completed, store, _started = await _run_login(idp, make_settings(), db_sessionmaker)

    assert not any("an-access-token-nobody-keeps" in value for value in store.values.values())


# ------------------------------------------------------------------- users


async def test_resolve_or_create_keys_on_the_subject_not_the_email(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # An address can be released and reassigned; a sub cannot. Keying on the
    # address would eventually hand one person another person's history.
    from app.services.auth.tokens import VerifiedIdentity

    settings = make_settings()
    async with db_sessionmaker() as session:
        first = await resolve_or_create(
            session,
            VerifiedIdentity(
                issuer="http://127.0.0.1:8765",
                subject="sub-A",
                email="shared@example.com",
                display_name="First",
            ),
            settings=settings,
        )
        second = await resolve_or_create(
            session,
            VerifiedIdentity(
                issuer="http://127.0.0.1:8765",
                subject="sub-B",
                email="shared@example.com",
                display_name="Second",
            ),
            settings=settings,
        )
        await session.commit()

    assert first.id != second.id


# ------------------------------------------------------------- the index


async def test_two_identityless_users_coexist_but_one_identity_is_unique(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # The index is PARTIAL for a reason: the seeded pre-login user carries NULLs
    # and must stay insertable beside every real account, while no two accounts
    # may claim the same provider subject. A plain unique index would enforce
    # the second; only the partial one leaves the first alone, and only a real
    # insert proves the declaration reached the database.
    from sqlalchemy.exc import IntegrityError

    async with db_sessionmaker() as session:
        session.add(User(tenant_id=DEFAULT_TENANT_ID, display_name="No identity A"))
        session.add(User(tenant_id=DEFAULT_TENANT_ID, display_name="No identity B"))
        await session.commit()

    async with db_sessionmaker() as session:
        session.add(
            User(
                tenant_id=DEFAULT_TENANT_ID,
                display_name="First",
                oidc_issuer="http://127.0.0.1:8765",
                oidc_sub="sub-same",
            )
        )
        await session.commit()

    async with db_sessionmaker() as session:
        session.add(
            User(
                tenant_id=DEFAULT_TENANT_ID,
                display_name="Impostor",
                oidc_issuer="http://127.0.0.1:8765",
                oidc_sub="sub-same",
            )
        )
        with pytest.raises(IntegrityError):
            await session.commit()


# ------------------------------------------------- review-round hardening


@pytest.mark.parametrize(
    "value",
    [
        "/\t/evil.example",
        "/\n/evil.example",
        "/\r/evil.example",
        "/\x00/evil.example",
        "/\x1f/evil.example",
        "/\x7f/evil.example",
        r"/\evil.example",
        r"/ok\..\evil",
        "/ok\rSet-Cookie: x=y",
    ],
)
def test_a_control_character_or_backslash_is_refused_outright(value: str) -> None:
    # A BROWSER STRIPS tab, CR and LF from a URL before parsing it, so
    # "/<tab>/evil.example" inspects server-side as an ordinary relative path
    # and arrives at the browser as "//evil.example" - protocol-relative, i.e.
    # another origin. Any check that reasons about the string as-written is
    # walked around by that, so these characters are refused rather than
    # inspected. NUL and the rest of C0 are here because they are the same
    # class of trick and, unlike tab/CR/LF, no URL parser quietly drops them.
    assert safe_next_path(value) == "/"


async def test_a_trailing_slash_issuer_still_signs_people_in(
    db_sessionmaker: async_sessionmaker[AsyncSession],
) -> None:
    # Auth0's canonical issuer ends in a slash, and its tokens carry that exact
    # string in `iss`. Verification compares `iss` EXACTLY, so a deployment that
    # verified against the operator-typed value could never accept a real token
    # from such a provider - and the coarse refusal page would give nobody a
    # clue why. The provider's own declared spelling is what must be used.
    idp = FakeIdp(issuer="http://127.0.0.1:8765/")
    settings = make_settings(oidc_issuer="http://127.0.0.1:8765/")

    completed, _store, _started = await _run_login(idp, settings, db_sessionmaker)

    assert completed.user.oidc_issuer == "http://127.0.0.1:8765/"


async def test_a_multi_audience_token_must_name_this_client_as_the_party() -> None:
    # PyJWT is satisfied by an `aud` list that merely CONTAINS the client id, so
    # without the azp check a token minted for a sibling application at the same
    # provider signs somebody in here.
    idp = FakeIdp()
    token = idp.id_token(nonce="the-nonce", audience=[idp.client_id, "the-other-app"])

    with pytest.raises(LoginProtocolError, match="another application"):
        await _verified(idp, token)


async def test_a_multi_audience_token_naming_this_client_is_accepted() -> None:
    idp = FakeIdp()
    token = idp.id_token(
        nonce="the-nonce",
        audience=[idp.client_id, "the-other-app"],
        extra_claims={"azp": idp.client_id},
    )

    assert (await _verified(idp, token)).subject == "sub-123"


@pytest.mark.parametrize(
    "issuer", ["http://idp.example", "ftp://idp.example", "https://user:pw@idp.example"]
)
def test_an_issuer_that_is_not_https_refuses_to_boot(issuer: str) -> None:
    # The issuer's metadata document names the signing keys, so fetching it in
    # the clear lets anyone on the path substitute keys of their own and forge
    # accepted sign-ins. Every endpoint the document names is already held to
    # this rule; the thing that names them cannot be the exception.
    with pytest.raises(ValidationError, match="OIDC_ISSUER"):
        Settings(
            oidc_issuer=issuer,
            oidc_client_id="an-id",
            oidc_client_secret="a-secret",
            oidc_admin_emails="boss@example.com",
        )


def test_an_issuer_on_loopback_may_still_be_plain_http() -> None:
    # The local-development provider the http exception was written for.
    assert make_settings(oidc_issuer="http://127.0.0.1:8765").oidc_enabled is True


async def test_two_simultaneous_first_logins_resolve_to_one_row(
    db_sessionmaker: async_sessionmaker[AsyncSession], monkeypatch: pytest.MonkeyPatch
) -> None:
    # Two first logins for one account at once: both look, both find nothing,
    # both insert, and the partial unique index refuses one. The loser must
    # re-read rather than escape as a 500 - an IntegrityError is not one of the
    # login errors the route knows how to render, so it would surface as an
    # unhandled fault on the authentication boundary.
    #
    # The race cannot be run for real here: the test database is one sqlite
    # connection pinned by StaticPool, so two sessions serialize and the second
    # would simply SEE the first's row. What is driven instead is the exact
    # sequence the loser experiences - a lookup that misses while the row is
    # already there - by making the FIRST lookup answer None. Everything after
    # that is the shipped code path, including the real IntegrityError from the
    # real index.
    from app.services.auth import users as users_mod
    from app.services.auth.tokens import VerifiedIdentity

    settings = make_settings()
    identity = VerifiedIdentity(
        issuer="http://127.0.0.1:8765",
        subject="sub-race",
        email="racer@example.com",
        display_name="Racer",
    )
    async with db_sessionmaker() as setup:
        winner = await resolve_or_create(setup, identity, settings=settings)
        await setup.commit()
        winner_id = winner.id

    real_find = users_mod._find
    calls = {"n": 0}

    async def _miss_once(session, ident):  # noqa: ANN001, ANN202
        calls["n"] += 1
        if calls["n"] == 1:
            return None
        return await real_find(session, ident)

    monkeypatch.setattr(users_mod, "_find", _miss_once)

    async with db_sessionmaker() as loser_session:
        loser = await resolve_or_create(loser_session, identity, settings=settings)
        await loser_session.commit()

    assert calls["n"] >= 2, "the insert path was never taken, so no race was driven"
    assert loser.id == winner_id
