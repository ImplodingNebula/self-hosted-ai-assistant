"""The console's declaration writers: two trust axes, and a thinking pair.

Every test drives the REAL service the endpoint calls, against a fake provider
holding real digests - the digest binding is the whole reason a ceiling is a
trust axis, so a test that stubbed it would prove nothing about the axis it
claims to cover.
"""

from __future__ import annotations

import uuid

import pytest

from app.core.errors import NotFoundError, UnprocessableEntityError
from app.core.tenancy import DEFAULT_TENANT_ID, DEFAULT_USER_ID, TenantContext
from app.providers.base import ModelCapabilities, ModelInfo
from app.providers.fake_provider import FakeProvider
from app.services import model_declare
from app.services.model_facts import effective_agency_ceiling
from app.services.model_policy import PolicyView, clear_policy_cache, policy_for, resolve_policy

CHAT = "chat:4b"
THINK = "chat:4b-thinking"
CHAT_DIGEST = "sha256:chat-weights"
THINK_DIGEST = "sha256:think-weights"

TENANT = TenantContext(tenant_id=DEFAULT_TENANT_ID, user_id=DEFAULT_USER_ID, role="admin")


@pytest.fixture(autouse=True)
def _fresh_caches():
    clear_policy_cache()
    yield
    clear_policy_cache()


def provider(*, digests: bool = True) -> FakeProvider:
    """A server holding the dev-style pair, with or without weights digests."""
    return FakeProvider(
        models=[
            ModelInfo(name=CHAT, digest=CHAT_DIGEST if digests else None),
            ModelInfo(name=THINK, digest=THINK_DIGEST if digests else None),
        ]
    )


async def view(db_sessionmaker, model: str, prov: FakeProvider | None = None):
    return await resolve_policy(
        provider=prov or provider(),
        provider_name="fake",
        model=model,
        sessionmaker=db_sessionmaker,
    )


# --- the ceiling ------------------------------------------------------------


async def test_a_declared_ceiling_is_readable_at_once(db_sessionmaker) -> None:
    """The writer drops the read caches, so the click is not a no-op for minutes.

    MUTATION: delete the `clear_policy_cache()` call in `declare_ceiling` and
    this passes only because nothing warmed the cache first - which the read
    below does deliberately, so it fails.
    """
    # Warm the cache with the pre-write state, exactly as rendering the page
    # would have done a moment before the operator clicked.
    assert await view(db_sessionmaker, CHAT) is None
    await model_declare.declare_ceiling(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="agentic",
    )
    resolved = await view(db_sessionmaker, CHAT)
    assert resolved is not None and resolved.ceiling == "agentic"


async def test_a_declaration_survives_a_stale_installed_listing(db_sessionmaker) -> None:
    """Rule 1's SECOND half: the identity cache goes too, or the row is invisible.

    A ceiling is honoured only while the digest it names is what the runtime
    holds - and "what the runtime holds" is itself cached. A pull made outside
    the app leaves that cache holding the OLD digest, so a declaration written
    against the new one is gated out by every reader for the length of the TTL:
    the click appears to do nothing, and then starts working minutes later for
    no visible reason.

    The writer avoids it by taking a FRESH listing and dropping BOTH caches.
    This drives exactly that sequence: warm the identity cache from the old
    weights, then declare against a provider reporting new ones.

    MUTATION: delete `clear_identity_cache()` from `declare_ceiling` and the
    declaration reads back as absent.
    """
    from app.services.model_identity import clear_identity_cache, installed_digests

    clear_identity_cache()
    stale = FakeProvider(models=[ModelInfo(name=CHAT, digest="sha256:old-weights")])
    # Warm it, exactly as any read on the page would have a moment earlier.
    assert await installed_digests(stale, "fake") == {CHAT: "sha256:old-weights"}
    repulled = FakeProvider(models=[ModelInfo(name=CHAT, digest=CHAT_DIGEST)])
    await model_declare.declare_ceiling(
        provider=repulled,
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="deterministic",
    )
    resolved = await view(db_sessionmaker, CHAT, repulled)
    assert resolved is not None and resolved.ceiling == "deterministic"
    clear_identity_cache()


async def test_a_ceiling_row_carries_no_other_axis(db_sessionmaker) -> None:
    """Rule 2: one action writes one row, and merge-on-write must not exist.

    The trap this rules out is specific: a ceiling declared against one
    checkpoint copied forward onto a later row would survive a re-pull as a
    side effect of somebody setting something else, which is exactly the silent
    re-arm the per-axis fold exists to prevent.

    MUTATION: copy the folded `think_mode` into the ModelPolicy row that
    `declare_ceiling` builds and the second assertion fails.
    """
    await model_declare.declare_thinking_pair(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_model=THINK,
        partner_think_mode="always",
        partner_serves_chat=True,
    )
    await model_declare.declare_ceiling(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="assisted",
    )
    from app.repositories.model_policy import ModelPolicyRepository

    async with db_sessionmaker() as session:
        rows = await ModelPolicyRepository(session).recent("fake", CHAT)
    newest = rows[0]
    assert newest.ceiling == "assisted"
    assert newest.thinking_model is None


async def test_a_ceiling_binds_to_the_installed_weights(db_sessionmaker) -> None:
    """The row names the digest that is installed RIGHT NOW, and retires with it.

    Both halves matter and only together: a row carrying no digest would read
    as retired forever (a declaration nobody can ever act on), and a row
    carrying the wrong one would survive a re-pull it was supposed to retire.

    MUTATION: write `digest=None` in `declare_ceiling`'s row and the first
    assertion fails; hard-code a constant digest and the first fails too.
    """
    stored = await model_declare.declare_ceiling(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="agentic",
    )
    assert stored.digest == CHAT_DIGEST
    from app.repositories.model_policy import ModelPolicyRepository

    async with db_sessionmaker() as session:
        rows = await ModelPolicyRepository(session).recent("fake", CHAT)
    assert rows[0].digest == CHAT_DIGEST
    # And the consequence of that binding: weights the tag no longer points at
    # take the declaration with them, with no code doing anything.
    clear_policy_cache()
    repulled = FakeProvider(models=[ModelInfo(name=CHAT, digest="sha256:different")])
    resolved = await view(db_sessionmaker, CHAT, repulled)
    assert resolved is None or resolved.ceiling is None


async def test_a_clamp_on_one_tag_leaves_its_pair_sibling_alone(db_sessionmaker) -> None:
    """The two-gate shape, stated for the axis with teeth.

    A dev tier routes between two tags, and they are two different sets of
    weights. Clamping the direct-answer half to deterministic must not clamp
    the thinking half - a declaration is about the model it names, and a
    writer keying on anything looser (a family, a shared prefix, the pair)
    would silently disarm a model nobody touched.

    MUTATION: key the row on the pair's default tag rather than on the model
    argument and the second assertion fails.
    """
    await model_declare.declare_ceiling(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="deterministic",
    )
    clamped = await view(db_sessionmaker, CHAT)
    sibling = await view(db_sessionmaker, THINK)
    assert clamped is not None and clamped.ceiling == "deterministic"
    assert sibling is None or sibling.ceiling is None


def test_a_pair_default_clamp_reaches_the_thinking_sibling() -> None:
    """A per-tag clamp has to reach the tag a thinking turn actually runs on.

    Declarations are stored per tag and a pair is TWO tags, so clamping the
    direct-answer half from the console clamps only the half that answers
    directly. Every thinking turn routes to the sibling, which keeps whatever
    its own entry allows - and to the operator that reads as the clamp having
    been ignored.

    Composed at resolution rather than by writing a second row: a row against
    the sibling would be a declaration nobody made, and it would outlive the
    one that caused it. Storage stays per-tag, which is what
    `test_a_clamp_on_one_tag_leaves_its_pair_sibling_alone` pins.

    MUTATION: drop the `pair_ceiling` argument at
    `turn_plan`'s routed re-resolution (or make it a max) and the sibling comes
    back agentic under a deterministic default.
    """
    # The sibling's own runtime report: a template that takes a tool list, so
    # its report-derived ceiling is assisted...
    vouched = ModelCapabilities(completion=True, tools=True)
    # ...and an operator armed it past that, which only a declaration can do.
    armed = PolicyView(ceiling="agentic")
    assert effective_agency_ceiling(vouched, policy=armed) == "agentic"
    # Under a default clamped to deterministic, it comes down with it - whether
    # its own ceiling came from a declaration...
    assert (
        effective_agency_ceiling(vouched, policy=armed, pair_ceiling="deterministic")
        == "deterministic"
    )
    # ...or from the runtime's report alone.
    assert effective_agency_ceiling(vouched, pair_ceiling="deterministic") == "deterministic"
    # One-directional: a permissive default cannot lift a clamped sibling.
    clamped = PolicyView(ceiling="assisted")
    assert effective_agency_ceiling(vouched, policy=clamped, pair_ceiling="agentic") == "assisted"


async def test_an_unknown_mode_is_refused_before_it_is_stored(db_sessionmaker) -> None:
    """Rule 3: the column is a plain string, so the writer validates.

    A value the fold cannot interpret reads back as "nothing declared", which
    is a button that appears to work and silently does not.

    MUTATION: drop the `DECLARABLE_CEILINGS` check and this stores "autonomous"
    into a column nothing can read.
    """
    with pytest.raises(UnprocessableEntityError, match="not an agency mode"):
        await model_declare.declare_ceiling(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            ceiling="autonomous",
        )


async def test_weights_with_no_digest_cannot_carry_a_ceiling(db_sessionmaker) -> None:
    """A claim about weights needs weights to name.

    MUTATION: store the row with `digest=None` instead of refusing and the
    declaration floats free of the checkpoint - surviving a re-pull onto
    different weights, which is the one thing this store refuses.
    """
    with pytest.raises(UnprocessableEntityError, match="no weights digest"):
        await model_declare.declare_ceiling(
            provider=provider(digests=False),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            ceiling="assisted",
        )


async def test_a_model_the_server_does_not_hold_is_a_404(db_sessionmaker) -> None:
    with pytest.raises(NotFoundError, match="not installed"):
        await model_declare.declare_ceiling(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model="never-pulled:1b",
            ceiling="assisted",
        )


# --- reasoning with tools ---------------------------------------------------


async def test_a_reasoning_grant_is_readable_at_once(db_sessionmaker) -> None:
    """The writer drops the read caches, exactly as the ceiling's does.

    MUTATION: delete `clear_policy_cache()` from `declare_thinking_with_tools`
    and the warmed read below keeps answering None.
    """
    assert await view(db_sessionmaker, CHAT) is None
    await model_declare.declare_thinking_with_tools(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_with_tools=True,
    )
    resolved = await view(db_sessionmaker, CHAT)
    assert resolved is not None and resolved.thinking_with_tools is True


async def test_a_reasoning_row_carries_no_other_axis(db_sessionmaker) -> None:
    """Rule 2 on this writer: granting a capability must not re-arm a ceiling.

    The specific silent re-arm: a ceiling declared against one checkpoint,
    copied forward onto this row, would outlive the re-pull that was supposed
    to retire it - and it would do so as a side effect of somebody answering an
    entirely different question.

    MUTATION: copy the folded `ceiling` into the ModelPolicy row that
    `declare_thinking_with_tools` builds and the last assertion fails.
    """
    await model_declare.declare_ceiling(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        ceiling="agentic",
    )
    await model_declare.declare_thinking_with_tools(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_with_tools=True,
    )
    from app.repositories.model_policy import ModelPolicyRepository

    async with db_sessionmaker() as session:
        rows = await ModelPolicyRepository(session).recent("fake", CHAT)
    newest = rows[0]
    assert newest.thinking_with_tools is True
    assert newest.ceiling is None


async def test_a_reasoning_grant_binds_to_the_installed_weights(db_sessionmaker) -> None:
    """A TRUST axis: the row names the digest installed now, and retires with it.

    This is the whole auto-revert story for the axis - a re-pull takes the
    grant with it, with no staleness code anywhere.

    MUTATION: write `digest=None` in the row and the first assertion fails;
    drop the digest gate from the fold's trust branch and the last one does.
    """
    stored = await model_declare.declare_thinking_with_tools(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_with_tools=True,
    )
    assert stored.digest == CHAT_DIGEST
    clear_policy_cache()
    repulled = FakeProvider(models=[ModelInfo(name=CHAT, digest="sha256:different")])
    resolved = await view(db_sessionmaker, CHAT, repulled)
    assert resolved is None or resolved.thinking_with_tools is None


async def test_weights_with_no_digest_cannot_carry_a_reasoning_grant(db_sessionmaker) -> None:
    """Refused, not stored unbound: a claim about weights needs weights to name.

    MUTATION: drop the `digest is None` guard and this raises nothing.
    """
    with pytest.raises(UnprocessableEntityError):
        await model_declare.declare_thinking_with_tools(
            provider=provider(digests=False),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_with_tools=True,
        )


async def test_a_reasoning_grant_for_a_model_the_server_lacks_is_a_404(db_sessionmaker) -> None:
    with pytest.raises(NotFoundError):
        await model_declare.declare_thinking_with_tools(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model="absent:7b",
            thinking_with_tools=True,
        )


async def test_a_revocation_supersedes_a_grant(db_sessionmaker) -> None:
    """The safe direction, through the ordinary newest-row-wins fold.

    Stored as an explicit False rather than by deleting the earlier row, which
    is what makes the append-only table able to say "withheld" at all - a
    missing row means "nobody declared", and the two must not be the same
    answer.
    """
    for granted in (True, False):
        await model_declare.declare_thinking_with_tools(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_with_tools=granted,
        )
    resolved = await view(db_sessionmaker, CHAT)
    assert resolved is not None and resolved.thinking_with_tools is False


async def test_a_reasoning_grant_records_the_probe_it_rested_on(db_sessionmaker) -> None:
    """Provenance: the measurement the operator was reading when they decided.

    Audit only - nothing reads it back as a resolution input - which is why an
    UNUSABLE id must cost a warning and not the declaration.
    """
    probe = str(uuid.uuid4())
    await model_declare.declare_thinking_with_tools(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_with_tools=True,
        probe=probe,
    )
    await model_declare.declare_thinking_with_tools(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_with_tools=False,
        probe="not-a-uuid",
    )
    from app.repositories.model_policy import ModelPolicyRepository

    async with db_sessionmaker() as session:
        rows = await ModelPolicyRepository(session).recent("fake", CHAT)
    assert rows[0].probe_id is None  # the malformed id, dropped
    assert rows[0].thinking_with_tools is False  # ...and the declaration kept
    assert str(rows[1].probe_id) == probe


# --- the thinking pair ------------------------------------------------------


async def test_a_pair_follows_the_tag_across_a_re_pull(db_sessionmaker) -> None:
    """A preference is not a claim about a checkpoint, so it carries no binding.

    MUTATION: give the pair row a digest AND move it to the trust half of the
    fold and this stops resolving after the digest changes - a chosen pairing
    silently forgotten by a routine update.
    """
    await model_declare.declare_thinking_pair(
        provider=provider(),
        provider_name="fake",
        sessionmaker=db_sessionmaker,
        tenant=TENANT,
        model=CHAT,
        thinking_model=THINK,
        partner_think_mode="always",
        partner_serves_chat=True,
    )
    clear_policy_cache()
    repulled = FakeProvider(
        models=[
            ModelInfo(name=CHAT, digest="sha256:new-chat"),
            ModelInfo(name=THINK, digest=THINK_DIGEST),
        ]
    )
    resolved = await view(db_sessionmaker, CHAT, repulled)
    assert resolved is not None and resolved.thinking_model == THINK


@pytest.mark.parametrize("mode", ["never", "unknown"])
async def test_a_partner_that_cannot_reason_is_refused(db_sessionmaker, mode) -> None:
    """The candidate set is MEASURED, and "unknown" is not a capability.

    Pairing two direct-answer models produces a thinking toggle that quietly
    changes nothing; offering an unmeasured partner is that same offer made on
    no evidence at all.

    MUTATION: widen `REASONING_THINK_MODES` to include "unknown" and the second
    parametrization stores a pair nothing has established can reason.
    """
    with pytest.raises(UnprocessableEntityError, match="can reason"):
        await model_declare.declare_thinking_pair(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_model=THINK,
            partner_think_mode=mode,
            partner_serves_chat=True,
        )


async def test_a_vision_model_that_reasons_is_still_refused_as_a_partner(
    db_sessionmaker,
) -> None:
    """The exact tag that made "can it reason" the wrong question.

    `qwen3-vl:2b` is a THINKING model: a think-mode check measures it `always`
    and it passes every reasoning test put to it. Its visible `content` channel
    comes back EMPTY. Pairing it therefore routes every thinking turn to a
    model that answers nothing at all, and the failure is silent - the user
    asks a hard question and gets back a blank.

    So a partner has to clear TWO independent bars: it must reason, and it must
    reply. The think mode only ever spoke to the first.

    MUTATION: delete the `partner_serves_chat` branch from
    `declare_thinking_pair` and this stores the pairing, because the think mode
    it is given here is a perfectly genuine "always".
    """
    vision = "qwen3-vl:2b"
    server = FakeProvider(
        models=[
            ModelInfo(name=CHAT, digest=CHAT_DIGEST),
            ModelInfo(name=vision, digest="sha256:vision-weights"),
        ]
    )
    with pytest.raises(UnprocessableEntityError, match="does not serve chat turns"):
        await model_declare.declare_thinking_pair(
            provider=server,
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_model=vision,
            # Genuinely measured, and genuinely irrelevant on its own.
            partner_think_mode="always",
            partner_serves_chat=False,
        )


async def test_a_model_cannot_be_its_own_thinking_partner(db_sessionmaker) -> None:
    with pytest.raises(UnprocessableEntityError, match="own thinking partner"):
        await model_declare.declare_thinking_pair(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_model=CHAT,
            partner_think_mode="hybrid",
            partner_serves_chat=True,
        )


async def test_a_partner_the_server_does_not_hold_is_a_404(db_sessionmaker) -> None:
    with pytest.raises(NotFoundError, match="not installed"):
        await model_declare.declare_thinking_pair(
            provider=provider(),
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_model="never-pulled:1b",
            partner_think_mode="always",
            partner_serves_chat=True,
        )


async def test_a_later_pair_supersedes_an_earlier_one(db_sessionmaker) -> None:
    # Append-only, so "changing" a pair is a newer row and the fold reads it.
    # Without this the store would be write-once and a mis-declared pair would
    # be unrepairable from the console.
    other = FakeProvider(
        models=[
            ModelInfo(name=CHAT, digest=CHAT_DIGEST),
            ModelInfo(name=THINK, digest=THINK_DIGEST),
            ModelInfo(name="other:8b", digest="sha256:other"),
        ]
    )
    for partner in (THINK, "other:8b"):
        await model_declare.declare_thinking_pair(
            provider=other,
            provider_name="fake",
            sessionmaker=db_sessionmaker,
            tenant=TENANT,
            model=CHAT,
            thinking_model=partner,
            partner_think_mode="always",
            partner_serves_chat=True,
        )
    resolved = await policy_for(
        provider_name="fake",
        model=CHAT,
        digests={CHAT: CHAT_DIGEST},
        sessionmaker=db_sessionmaker,
    )
    assert resolved is not None and resolved.thinking_model == "other:8b"
