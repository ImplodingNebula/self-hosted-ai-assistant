"""Confirming a measured think mode into the policy store.

The think-check measures how a model treats the think flag; this is the step
where an operator agrees to that measurement and it becomes a DECLARATION.
Until they do, the measurement moves one thing and one thing only - the routing
clamp in :func:`app.services.model_facts.effective_think_mode`, which forces the
flag back on for weights measured to reason regardless. Everything else waits
here, because the other directions REMOVE reasoning or send a flag to weights
that reject it, and those are decisions rather than repairs.

This is the first writer the ``model_policy`` table has. Its three rules come
from the store's own design (:mod:`app.services.model_policy`,
:class:`app.models.entities.ModelPolicy`) and every later writer inherits them:

1. **Drop the read cache after the commit.** Declarations are cached per
   (provider, model) for minutes; a confirmation nobody can see until the TTL
   lapses is a button that appears to do nothing.
2. **Never copy another axis forward.** One action writes ONE row carrying only
   the axes that action set. The obvious "merge on write" - copying the
   previous row's values so the newest row is always complete - would carry an
   agency ceiling declared against one set of weights onto a re-pull as a side
   effect of somebody confirming a think mode, which is the silent re-arm the
   per-axis fold exists to make impossible.
3. **Validate the value.** What is stored comes out of a JSON column on a probe
   row, and the column is machine-written, so a mode this build does not know
   must never reach a string column that later reads back as a declaration
   nothing can interpret. This writer satisfies the rule by CONSTRUCTION rather
   than by a second check: the only value it can store is one
   :func:`app.services.earned.measured_think_for` returned, and that read drops
   an unrecognized value before anything downstream sees it. A defensive
   re-check here would be a branch no test could reach, which is a worse thing
   to ship than the rule it appears to enforce - so the guard lives at the read,
   where it is exercised.

And one rule that is this writer's own: **a confirmation needs evidence.** The
value stored is the one a check MEASURED on the weights installed right now -
never a value a caller sent. There is no request field to disagree with the
log, so "confirm" cannot become "declare whatever I like", and a model nothing
has measured is refused outright.
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
    ModelCapabilities,
    ModelProvider,
    installed_tag_matches,
    tag_aliases,
)
from app.repositories.model_policy import ModelPolicyRepository
from app.services.earned import measured_think_for
from app.services.model_facts import effective_think_mode
from app.services.model_identity import clear_identity_cache, digest_of, installed_name_of
from app.services.model_policy import ReadHealth, clear_policy_cache, policy_for
from app.services.model_resolution import ThinkMode

logger = logging.getLogger(__name__)


async def resolve_think_mode(
    *,
    provider_name: str,
    model: str,
    report: ModelCapabilities | None,
    digests: dict[str, str],
    sessionmaker: async_sessionmaker[AsyncSession],
    health: ReadHealth | None = None,
) -> ThinkMode:
    """Resolve how ``model`` treats the think flag, over every source there is.

    The whole chain in one call - the stored declaration, the unconfirmed
    measurement, and the pure resolver that ranks them against the
    quirk seed and the runtime's template. It exists so a caller that needs the
    ANSWER does not have to assemble the chain itself: the resolver is pure by
    design and the two reads that feed it touch the database, so without this
    every such caller becomes another place the order could be got wrong, and
    "two screens, one answer" would be a convention rather than a shape.

    **A tag with no installed digest costs no database at all.** Both sources
    this function consults are bound to WEIGHTS - the stored ``think_mode`` is
    a trust axis, retired by the fold when its digest is not what is installed,
    and a measurement is digest-bound by construction - so for a tag the
    runtime holds no digest for, neither could change the answer. The skip is
    scoped to that: a caller wanting a PREFERENCE axis (a chosen context
    window, a sibling override) must read the store itself, because those
    follow the tag across a re-pull and are emphatically not covered here.
    Every listed tag a fresh deployment has not downloaded is in this case,
    which is most of the rows a model listing renders.

    Args:
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        model: The tag being described.
        report: The runtime's report for this tag, or None.
        digests: The installed name -> digest map the caller already read.
        sessionmaker: Session factory for the two reads (both cached).
        health: A caller-owned record marked when either read fails, so a
            caller resolving many models can stop (see
            :class:`app.services.model_policy.ReadHealth`).

    Returns:
        The resolved :data:`~app.services.model_resolution.ThinkMode`.
    """
    if digest_of(model, digests) is None:
        return effective_think_mode(model, report)
    policy = await policy_for(
        provider_name=provider_name,
        model=model,
        digests=digests,
        sessionmaker=sessionmaker,
        health=health,
    )
    if health is not None and health.failed:
        # The store is unreachable - the read above just found that out, or
        # another model's did. The probe log lives in the same database, so
        # asking it would buy a second timeout for the same answer.
        return effective_think_mode(model, report, policy=policy)
    found = await measured_think_for(
        provider_name=provider_name,
        model=model,
        digests=digests,
        sessionmaker=sessionmaker,
        health=health,
    )
    return effective_think_mode(
        model, report, policy=policy, measured=None if found is None else found.mode
    )


@dataclass(frozen=True)
class ConfirmedThinkMode:
    """What one confirmation stored, for the caller to report back.

    Attributes:
        model: The tag the declaration was recorded against - the INSTALLED
            spelling, which is what the fold reads under.
        think_mode: The mode that was stored.
        digest: The weights it is bound to. A re-pull past this digest retires
            the declaration on its own, with no code doing anything.
        probe_id: The measurement row the declaration rested on, recorded as
            audit on the row itself.
    """

    model: str
    think_mode: ThinkMode
    digest: str
    probe_id: str


async def confirm_think_mode(
    *,
    provider: ModelProvider,
    provider_name: str,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    model: str,
) -> ConfirmedThinkMode:
    """Store the measured think mode for ``model`` as an operator declaration.

    Args:
        provider: The model seam, for the installed listing (cached).
        provider_name: The ``MODEL_PROVIDER`` selector value rows key on.
        sessionmaker: Session factory for the append.
        tenant: The acting principal, recorded as audit-only columns.
        model: The already-validated tag being confirmed.

    Returns:
        What was stored.

    Raises:
        NotFoundError: The backend does not hold these weights, so there is
            nothing a declaration could be bound to.
        UnprocessableEntityError: No check has measured the installed weights
            with the running harness, or the measurement carries a mode this
            build does not know. Confirming without evidence is not a thing
            this endpoint does - the whole value of the row is that it records
            something somebody watched happen.
    """
    # A FRESH listing, exactly as the verify endpoint takes one and for the same
    # reason: "the backend does not hold these weights" is a different sentence
    # from "nothing has measured them", and a confirmation is a rare admin
    # action rather than a hot path, so it can afford to be sure which one it is
    # saying.
    #
    # The digest map comes off that SAME listing rather than the shared cached
    # one, and that matters here in a way it does not on a read path: this row
    # is BOUND to the digest, permanently. A pull made outside the app (nothing
    # clears the cache for that) would otherwise have the confirmation written
    # against weights the machine no longer holds - a declaration that reads as
    # working until the cache lapses and then silently stops applying.
    listing = await provider.list_models()
    installed = {info.name for info in listing}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(tag_aliases(model))):
        raise NotFoundError(
            f"'{model}' is not installed, so there is nothing to confirm a think mode for."
        )
    digests = {info.name: info.digest for info in listing if info.digest}
    found = await measured_think_for(
        provider_name=provider_name,
        model=model,
        digests=digests,
        sessionmaker=sessionmaker,
    )
    if found is None:
        raise UnprocessableEntityError(
            f"No tool check has measured how '{model}' treats the think flag on the "
            "weights installed right now. Run a check first - a confirmation records "
            "what was measured, so there is nothing to record until something has been. "
            "A runtime that exposes no weights digest reads the same way: there would "
            "be nothing to bind the declaration to."
        )
    # The INSTALLED spelling, because that is what the fold reads under (see
    # app.services.model_identity): a row written under the configured spelling
    # of a tag the runtime lists differently would be a declaration nobody ever
    # finds again.
    name = installed_name_of(model, digests) or model
    async with sessionmaker() as session:
        row = ModelPolicy(
            provider=provider_name,
            model=name,
            digest=found.digest,
            # Rule 2: think_mode and NOTHING else. Every other axis stays NULL
            # so this row says only what this action decided, and the fold
            # keeps each earlier statement attached to the weights it was made
            # about.
            think_mode=found.mode,
            declared_by_tenant_id=tenant.tenant_id,
            declared_by_user_id=tenant.user_id,
            probe_id=UUID(found.probe_id),
        )
        await ModelPolicyRepository(session).add(row)
        await session.commit()
    # Rule 1, and after the commit rather than before: a cache dropped first
    # would be refilled from the pre-write state by any read that landed in
    # between.
    #
    # BOTH caches, because a declaration is only visible when the digest it is
    # gated against is. This writer takes a fresh listing precisely so an
    # out-of-band pull cannot bind the row to weights the machine no longer
    # holds - and in exactly that case the shared installed-digest cache still
    # holds the OLD digest, so every reader would gate the new row out and the
    # confirmation would sit inert for up to five minutes. Dropping one cache
    # and not the other satisfies the letter of rule 1 and not its point.
    clear_policy_cache()
    clear_identity_cache()
    logger.info(
        "think mode confirmed by tenant=%s user=%s: model=%r mode=%s digest=%s probe=%s",
        tenant.tenant_id,
        tenant.user_id,
        name,
        found.mode,
        found.digest,
        found.probe_id,
    )
    return ConfirmedThinkMode(
        model=name, think_mode=found.mode, digest=found.digest, probe_id=found.probe_id
    )
