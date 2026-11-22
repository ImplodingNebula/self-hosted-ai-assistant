"""Operator declarations made from the console: two trust axes and a pair.

The writers the ``model_policy`` table has beyond the think-mode confirmation
in :mod:`app.services.think_check` - the agency ceiling, the reasoning-with-
tools grant, and the thinking pair. That module wrote the rules down; this one
inherits them, and the two that bite hardest here are worth restating because
these are the axes with teeth:

1. **Drop the read caches after the commit, both of them.** A declaration is
   cached per (provider, model) for minutes and is only honoured while the
   digest it names is what the runtime holds, so the identity cache has to go
   with it - a fresh listing taken here can bind a row to weights the shared
   cache does not know about yet, and every reader would then gate the new row
   out for the length of the TTL.
2. **One action writes ONE row.** Only the axis this action set is written;
   every other column stays NULL. Merging on write would carry a ceiling
   declared against one checkpoint forward onto a re-pull as a side effect of
   somebody choosing a thinking partner, which is the silent re-arm the
   per-axis fold exists to prevent.
3. **Validate against the axis's legal set before storing.** The columns are
   plain strings and the fold drops what it cannot interpret, so an unvalidated
   write would be a declaration that reads back as nothing at all - a button
   that appears to work and silently does not.

And the two rules this module adds, which are the whole shape of the feature:

**A declaration sets the CEILING, and what that costs depends on the rung.**
An earlier draft of this docstring claimed a declaration "raises the ceiling
and never the floor", which is comfortable and not true. ``min(CEILING,
EARNED)`` does hold ``agentic`` behind a digest-bound deep certificate - but
EARNED is not purely measured: under the default grace regime it is
``max(_grace_floor(ceiling), certificate)``, and ``_grace_floor`` is
``min(ceiling, "assisted")`` (:mod:`app.services.earned`). So declaring
``assisted`` on a model the runtime's report floors at ``deterministic`` moves
what a turn does - tools start being offered - with nothing having measured
anything. That is the reviewed-declaration grace working as designed, and it is
exactly why this writer is capped rather than trusted.

**``agentic`` IS declarable from the console, pre-OIDC** (owner decision,
2026-08-08, superseding the older "UI declarations cap at assisted until
OIDC"). Recorded here because it looks like a hole and is not, and a future
reviewer should find the reasoning rather than re-raise it:

- Capping this ONE verb earns nothing. Every other verb on the same console -
  delete, pull, stored credentials, the integration registry - already sits
  behind the identical pre-OIDC gate, so an attacker who can reach the port has
  far more direct things to do than raise a ceiling.
- The real safeguard is not the cap, it is the TEST-FIRST gate, and that one is
  structural rather than advisory: an ``agentic`` declaration reaches a turn
  only through ``min(CEILING, EARNED)``, and EARNED requires a passing DEEP
  certificate at the running harness version, bound to the installed digest,
  measured on the serving endpoint (:mod:`app.services.earned`). A declaration
  on an unproven model is stored and inert.
- The dialog carries the reasoned warning that goes with it, read from the
  probe evidence rather than written as boilerplate.

A DOWN-clamp is the opposite of all of it and takes effect at once - ``min()``
needs no evidence to lower something - which is why the ceiling is the one axis
a console must never make hard to reach.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from uuid import UUID

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import NotFoundError, UnprocessableEntityError
from app.core.tenancy import TenantContext
from app.models.entities import ModelPolicy
from app.providers.base import (
    ModelProvider,
    installed_tag_matches,
    tag_aliases,
)
from app.repositories.model_policy import ModelPolicyRepository
from app.services.model_identity import clear_identity_cache, digest_of, installed_name_of
from app.services.model_names import require_model_name
from app.services.model_policy import clear_policy_cache
from app.services.model_resolution import AgencyMode, ThinkMode

logger = logging.getLogger(__name__)

# What an operator may declare on the ceiling axis - the full set, ``agentic``
# included (owner decision, 2026-08-08; see the module docstring).
#
# Restated on the WRITE side rather than imported from the reader's constant:
# the reader drops values it cannot interpret, so a writer leaning on it would
# store something that silently reads back as "nothing declared".
DECLARABLE_CEILINGS: frozenset[str] = frozenset({"deterministic", "assisted", "agentic"})

# The think modes that make a model usable as somebody else's thinking partner.
# "never" is the obvious exclusion; "unknown" is the load-bearing one - it is
# what the resolvers answer when NOTHING has established how the weights treat
# the flag, and offering it as a reasoning partner would be a guess dressed as
# a capability.
REASONING_THINK_MODES: frozenset[str] = frozenset({"always", "hybrid"})


@dataclass(frozen=True)
class DeclaredCeiling:
    """What one ceiling declaration stored, for the caller to report back.

    Attributes:
        model: The tag the row was written against - the INSTALLED spelling,
            which is what the fold reads under.
        ceiling: The mode that was stored.
        digest: The weights it is bound to. A re-pull past this digest retires
            the declaration on its own, with no code doing anything.
    """

    model: str
    ceiling: AgencyMode
    digest: str


@dataclass(frozen=True)
class DeclaredThinkingWithTools:
    """What one reasoning-with-tools declaration stored.

    Attributes:
        model: The tag the row was written against - the INSTALLED spelling,
            which is what the fold reads under.
        thinking_with_tools: The grant, or the revocation, that was stored.
        digest: The weights it is bound to. A re-pull past this digest retires
            the declaration on its own, with no code doing anything.
    """

    model: str
    thinking_with_tools: bool
    digest: str


@dataclass(frozen=True)
class DeclaredThinkingPair:
    """What one thinking-pair declaration stored.

    Attributes:
        model: The direct-answer tag the row was written against.
        thinking_model: The sibling that will serve its thinking turns.
    """

    model: str
    thinking_model: str


async def declare_ceiling(
    *,
    provider: ModelProvider,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
    ceiling: str,
    probe: str | None = None,
) -> DeclaredCeiling:
    """Store an operator's agency ceiling for ``model``.

    Args:
        provider: The model seam, for the installed listing.
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        sessionmaker: Session factory for the append.
        tenant: The acting principal, recorded as audit-only columns.
        model: The tag being declared about.
        ceiling: One of :data:`DECLARABLE_CEILINGS`.
        probe: The ``model_probes`` row the decision rested on, when the caller
            knows it - the measurement the operator was reading when they
            clicked. Audit only; nothing reads it back as an input, and a
            declaration made without one is legitimate.

    Returns:
        What was stored.

    Raises:
        UnprocessableEntityError: The name is not a usable model name, the mode
            is not one this build knows, or the runtime exposes no digest for
            these weights - a ceiling is a claim about WEIGHTS, and there would
            be nothing to bind it to.
        NotFoundError: The backend does not hold these weights.
    """
    tag = require_model_name(model).text
    if ceiling not in DECLARABLE_CEILINGS:
        raise UnprocessableEntityError(
            f"'{ceiling}' is not an agency mode. Declare one of: "
            f"{', '.join(sorted(DECLARABLE_CEILINGS))}."
        )
    # A FRESH listing, on exactly the precedent think_check.confirm_think_mode
    # set and for the sharper version of its reason: this row is bound to the
    # digest permanently, and a pull made outside the app leaves the shared
    # cache holding the old one - so a cached read would bind the declaration
    # to weights the machine no longer has, and it would read as working right
    # up until the TTL lapsed and it silently stopped applying.
    listing = await provider.list_models()
    installed = {info.name for info in listing}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(tag))):
        raise NotFoundError(
            f"'{tag}' is not installed, so there is nothing to declare a ceiling for."
        )
    digests = {info.name: info.digest for info in listing if info.digest}
    digest = digest_of(tag, digests)
    if digest is None:
        raise UnprocessableEntityError(
            f"The model server exposes no weights digest for '{tag}', so a ceiling "
            "cannot be bound to what is installed. A declaration that floated free "
            "of the weights would survive a re-pull onto a different checkpoint, "
            "which is the one thing this store refuses to allow."
        )
    name = installed_name_of(tag, digests) or tag
    # The measurement the operator was looking at when they decided, when there
    # is one. The recorded design for this verb is "logged with actor AND the
    # probe row relied on", and the actor half was already here - this is the
    # other half. Audit only: nothing reads it back as an input, and a
    # declaration with no probe behind it (a down-clamp, or a model nothing has
    # measured) is perfectly legitimate and simply records None.
    probe_id: UUID | None = None
    if probe is not None:
        try:
            probe_id = UUID(probe)
        except ValueError:
            # A malformed id must not cost an operator their declaration: the
            # column is provenance, not policy.
            logger.warning("Ignoring unusable probe id %r on a ceiling declaration.", probe)
    async with sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(
                provider=provider_name,
                model=name,
                digest=digest,
                # Rule 2: the ceiling and NOTHING else.
                ceiling=ceiling,
                declared_by_tenant_id=tenant.tenant_id,
                declared_by_user_id=tenant.user_id,
                probe_id=probe_id,
            )
        )
        await session.commit()
    # Rule 1, after the commit, and BOTH caches - see the module docstring.
    clear_policy_cache()
    clear_identity_cache()
    logger.info(
        "agency ceiling declared by tenant=%s user=%s: model=%r ceiling=%s digest=%s",
        tenant.tenant_id,
        tenant.user_id,
        name,
        ceiling,
        digest,
    )
    return DeclaredCeiling(model=name, ceiling=ceiling, digest=digest)  # type: ignore[arg-type]


async def declare_thinking_with_tools(
    *,
    provider: ModelProvider,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
    thinking_with_tools: bool,
    probe: str | None = None,
) -> DeclaredThinkingWithTools:
    """Store whether ``model`` may reason and call tools in the same turn.

    A TRUST axis, so the row is digest-bound exactly as a ceiling is: this is a
    claim about what these WEIGHTS can do, and it retires the moment the tag
    points somewhere else.

    Gated on the same principle as ``agentic`` and by the same mechanism - the
    fold, never a refusal here. A grant is STORED whatever the log holds, and
    :mod:`app.services.earned` decides whether it reaches a turn: a stored
    grant is evidence-gated in BOTH
    regimes, needing a passing DEEP verdict at the running harness version,
    bound to the pair's installed thinking weights, measured on the host this
    deployment serves from. A re-pull, a harness edit or a move to another
    model server retires it through that same fold, with no second staleness
    path anywhere.

    The asymmetry is deliberate and points the safe way. A REVOCATION (storing
    False) needs no evidence at all and applies to the next turn, because the
    failure this axis guards is SILENT - reasoning plus tools on an incapable
    pair returns an EMPTY answer rather than an error (ollama/ollama#10976) -
    and nothing downstream would catch it.

    Args:
        provider: The model seam, for the installed listing.
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        sessionmaker: Session factory for the append.
        tenant: The acting principal, recorded as audit-only columns.
        model: The tag being declared about.
        thinking_with_tools: The claim to store. True grants, False revokes.
        probe: The ``model_probes`` row the decision rested on, when the caller
            knows it. Audit only; nothing reads it back as an input, and a
            declaration made without one is legitimate - a revocation in
            particular rests on no measurement by design.

    Returns:
        What was stored.

    Raises:
        UnprocessableEntityError: The name is not a usable model name, or the
            runtime exposes no digest for these weights - this is a claim about
            WEIGHTS, and there would be nothing to bind it to.
        NotFoundError: The backend does not hold these weights.
    """
    tag = require_model_name(model).text
    # A FRESH listing, on the same rule declare_ceiling states: the row binds
    # to a digest permanently, and a cached read could bind it to weights the
    # machine no longer holds - which reads as working until the TTL lapses.
    listing = await provider.list_models()
    installed = {info.name for info in listing}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(tag))):
        raise NotFoundError(
            f"'{tag}' is not installed, so there is nothing to declare about how it reasons."
        )
    digests = {info.name: info.digest for info in listing if info.digest}
    digest = digest_of(tag, digests)
    if digest is None:
        raise UnprocessableEntityError(
            f"The model server exposes no weights digest for '{tag}', so a reasoning-with-"
            "tools claim cannot be bound to what is installed. A claim that floated free of "
            "the weights would survive a re-pull onto a differently-behaving checkpoint, and "
            "this is the axis whose failure is an empty answer rather than an error."
        )
    name = installed_name_of(tag, digests) or tag
    probe_id: UUID | None = None
    if probe is not None:
        try:
            probe_id = UUID(probe)
        except ValueError:
            # Provenance, not policy: a malformed id must not cost an operator
            # their declaration.
            logger.warning(
                "Ignoring unusable probe id %r on a thinking-with-tools declaration.", probe
            )
    async with sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(
                provider=provider_name,
                model=name,
                digest=digest,
                # Rule 2: this axis and NOTHING else. Merging on write would
                # carry a ceiling declared against one checkpoint forward onto
                # a re-pull as a side effect of somebody granting this.
                thinking_with_tools=thinking_with_tools,
                declared_by_tenant_id=tenant.tenant_id,
                declared_by_user_id=tenant.user_id,
                probe_id=probe_id,
            )
        )
        await session.commit()
    # Rule 1, after the commit, and BOTH caches - see the module docstring.
    clear_policy_cache()
    clear_identity_cache()
    logger.info(
        "thinking-with-tools declared by tenant=%s user=%s: model=%r granted=%s digest=%s",
        tenant.tenant_id,
        tenant.user_id,
        name,
        thinking_with_tools,
        digest,
    )
    return DeclaredThinkingWithTools(
        model=name, thinking_with_tools=thinking_with_tools, digest=digest
    )


async def declare_thinking_pair(
    *,
    provider: ModelProvider,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
    thinking_model: str,
    partner_think_mode: ThinkMode,
    partner_serves_chat: bool,
) -> DeclaredThinkingPair:
    """Store which installed tag serves ``model``'s thinking turns.

    A PREFERENCE axis, so the row carries no digest binding: "run this pair
    together" is a statement about how this deployment wants two tags used, not
    a claim about either checkpoint, and it follows the tags across a re-pull.

    Args:
        provider: The model seam, for the installed listing.
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        sessionmaker: Session factory for the append.
        tenant: The acting principal, recorded as audit-only columns.
        model: The direct-answer tag being paired.
        thinking_model: The sibling that should serve its thinking turns.
        partner_think_mode: How the PARTNER treats the think flag, as the
            caller resolved it through
            :func:`app.services.think_check.resolve_think_mode`. Taken as an
            argument rather than resolved here so that the candidate list a
            console offers and the check this writer makes against it come out
            of one function - a second resolution here is how the two would
            start disagreeing about which models can reason.
        partner_serves_chat: Whether the partner can actually answer a chat
            turn, as the caller resolved it from the
            runtime's own report. Load-bearing and separate from the think mode
            for a reason that bit: a vision model measured ``always`` looks
            like a perfect reasoning partner to a think-mode test and is not one
            at all. ``qwen3-vl:2b`` is exactly that shape - it reasons, and its
            visible ``content`` channel comes back EMPTY - so pairing it routes
            every thinking turn to a model that answers nothing.

    Returns:
        What was stored.

    Raises:
        UnprocessableEntityError: Either name is unusable, the pair names the
            same tag twice, the partner cannot serve a chat turn, or the
            partner is not a model this deployment has established can reason.
        NotFoundError: The backend does not hold one of the two.
    """
    tag = require_model_name(model).text
    partner = require_model_name(thinking_model).text
    listing = await provider.list_models()
    installed = {info.name for info in listing}
    for candidate in (tag, partner):
        if not any(
            installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(candidate))
        ):
            raise NotFoundError(f"'{candidate}' is not installed, so it cannot be paired.")
    digests = {info.name: info.digest for info in listing if info.digest}
    name = installed_name_of(tag, digests) or tag
    partner_name = installed_name_of(partner, digests) or partner
    if any(alias in tag_aliases(partner_name) for alias in tag_aliases(name)):
        raise UnprocessableEntityError(
            f"'{name}' cannot be its own thinking partner. A pair exists so a model that "
            "answers directly can hand a reasoning turn somewhere else; a model that can "
            "reason on its own needs no pair at all."
        )
    if not partner_serves_chat:
        # Checked BEFORE the think mode, because it is the stronger statement
        # and because the think mode is what makes the wrong partner look
        # right. A vision model measured "always" passes the reasoning test
        # perfectly and then answers a thinking turn with nothing at all.
        raise UnprocessableEntityError(
            f"'{partner_name}' does not serve chat turns, so it cannot answer "
            f"'{name}'s thinking ones. Reasoning is not the only thing a partner has "
            "to do - it has to reply - and a model bound to another role (vision, "
            "embedding) can be measured to reason while its visible answer comes back "
            "empty. Pick a partner that serves chat."
        )
    if partner_think_mode not in REASONING_THINK_MODES:
        # The one refusal that carries the feature's whole point. Pairing a
        # model that answers directly with another that answers directly would
        # produce a thinking toggle that quietly changes nothing, and an
        # "unknown" partner is the same offer made on no evidence at all.
        raise UnprocessableEntityError(
            f"Nothing establishes that '{partner_name}' can reason - its think mode "
            f"resolves to '{partner_think_mode}' - so it cannot serve '{name}'s thinking "
            "turns. Run a check against it first, or pick a partner whose mode already "
            "resolves to 'always' or 'hybrid'."
        )
    async with sessionmaker() as session:
        await ModelPolicyRepository(session).add(
            ModelPolicy(
                provider=provider_name,
                model=name,
                # No digest: a preference follows the TAG. Recording one here
                # would look like binding and read like nothing - the fold does
                # not gate this axis on it - which is worse than either.
                thinking_model=partner_name,
                declared_by_tenant_id=tenant.tenant_id,
                declared_by_user_id=tenant.user_id,
            )
        )
        await session.commit()
    clear_policy_cache()
    clear_identity_cache()
    logger.info(
        "thinking pair declared by tenant=%s user=%s: model=%r thinking_model=%r",
        tenant.tenant_id,
        tenant.user_id,
        name,
        partner_name,
    )
    return DeclaredThinkingPair(model=name, thinking_model=partner_name)
