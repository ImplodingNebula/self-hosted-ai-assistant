"""Removing a model: what a delete changes, and the reasons one is refused.

Deleting weights is the one management verb that can break a running
deployment, so the guards are re-derived from live state on every request
rather than trusted from the client. The management view sends a ``can_delete``
flag with each row, but that flag is a snapshot of the moment the page
rendered: a second tab, an env change, or another user's pull can all invalidate
it before the button is pressed, and a client that has been open since this
morning is asserting something about a stack that has moved on.

A delete is more than a removal now. The two STORED tiers that can name a model
(see ``resolve_chat_models``) are handled in the same request instead of
refusing it:

- Conversation pins on the tag are CLEARED, tenant-wide, so those threads
  follow the workspace default from their next turn. Refusing over pins made
  every model anyone had ever picked effectively undeletable - an admin cannot
  walk every user's threads to unpin them, and the change forced on those
  threads is cheap, recoverable (pick a model again), and disclosed by the
  confirm dialog before the button is pressed.
- The WORKSPACE DEFAULT is replaced, not orphaned: deleting the stored default
  requires naming a successor in the same request, which is validated against
  what the server actually holds and written before the weights go. With no
  successor possible - no other manageable model installed - the delete is
  refused outright, because a workspace with no chat model cannot answer
  anything at all.

What still refuses: the name must be well formed, the backend must have the
verb, the weights must be here, the role must be one this surface manages
(where the embedding role narrows to its own stranded-vectors policy), and the
tag must not be the deployment's own chat pair - that one is a config line no
request can rewrite, so deleting under it would break the next turn with
nothing in the product able to fix it.

Every refusal carries a sentence naming what to change, and a count where there
is one. "Forbidden" with no reason is the failure mode that makes people go
looking for the file to edit by hand, which is exactly the state this endpoint
exists to avoid. Refusals are logged as well as successes: for a destructive
verb, the attempts turned away are the more interesting half of the trail.

Ordering inside a delete: the database changes are STAGED first, the provider
delete runs second, and the caller commits after. The likely failure is the
provider (a model server that is down), and this order leaves that failure
clean - the request raises, the session rolls back, and no pin or default
moved for a delete that did not happen.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import BadRequestError, ConflictError, NotFoundError
from app.core.tenancy import TenantContext
from app.providers.base import ModelProvider, installed_tag_matches, tag_aliases
from app.providers.rag.base import VectorStore
from app.repositories.conversations import ConversationRepository
from app.services.earned import clear_earned_cache
from app.services.embedding_resolution import resolve_embedding
from app.services.model_facts import clear_facts_cache, report_for, runtime_capabilities_many
from app.services.model_manager import (
    article_for,
    delete_block_reason,
    derive_role,
    embedding_delete_block,
    is_workspace_default,
    load_model_usage,
    management_of,
)
from app.services.model_names import require_model_name
from app.services.model_policy import clear_policy_cache
from app.services.vector_spaces import vector_space_report
from app.services.workspace_defaults import load_defaults, set_default_model

logger = logging.getLogger(__name__)

_NO_DELETE_VERB = (
    "This deployment's model backend cannot remove models on request - it "
    "serves what it was started with, so its model set changes by redeployment."
)


@dataclass(frozen=True)
class DeleteOutcome:
    """What a delete did: the tag removed, and the stored state moved with it.

    Attributes:
        removed: The normalized tag whose weights were removed.
        unpinned_conversations: Conversations whose pin on the tag was cleared.
        new_default: The successor written as the workspace default, when the
            deleted tag was the default; None when the default was not involved.
    """

    removed: str
    unpinned_conversations: int = 0
    new_default: str | None = None


async def delete_model(
    *,
    provider: ModelProvider,
    settings: Settings,
    session: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    name: str,
    replacement: str | None = None,
    store: VectorStore | None = None,
) -> DeleteOutcome:
    """Remove a model's weights, moving the stored state that named them.

    Guards, in the order they run: the name must be well formed; the backend
    must have the verb at all; the weights must actually be here; the role must
    be one this surface manages - where the embedding role narrows to its own
    per-row policy (the active embedder and any model that stored passages
    still reference are refused, and no answer from the vector store fails
    closed; see :func:`app.services.model_manager.embedding_delete_block`);
    and the tag must not be the chat pair the next turn is about to request.

    Past the guards, the stored tiers move: the workspace default is replaced
    by the validated ``replacement`` when the deleted tag is the default (and
    the request is refused when it is the default and no valid successor was
    named, or none could be), and every conversation pin on the tag is cleared
    so those threads follow the default.

    The "is it installed" check runs BEFORE the policy checks on purpose. A tag
    that is not here has nothing to protect, and answering a policy refusal for
    it would describe a state of affairs that does not exist.

    Args:
        provider: The model seam holding the weights.
        settings: The active settings (the active chat pair resolves from them).
        session: Database session for the stored-tier reads and writes. Staged
            and flushed here; the caller commits.
        sessionmaker: Session factory for the declaration read behind the
            active-pair guard. Separate from ``session`` because that read is
            cached across requests and must not join this one's transaction -
            and required rather than optional so this path cannot arm the pair
            guard differently from the management view that disabled the button.
        tenant: The acting principal, for scoping those reads and the audit line.
        name: The requested model name, as received.
        replacement: The successor for the workspace default, when the deleted
            tag is the default. Ignored when it is not: a stale dialog may name
            one after another tab already changed the default, and there is
            nothing for it to replace.
        store: The vector-store seam, for the embedding role's per-row policy.
            None fails that policy closed - it never widens it.

    Returns:
        The :class:`DeleteOutcome` describing everything that changed.

    Raises:
        UnprocessableEntityError: The name (or replacement) failed validation.
        BadRequestError: The backend has no delete verb.
        NotFoundError: The backend does not hold this model.
        ConflictError: A role lock or the active chat pair forbids it, the
            deleted tag is the workspace default with no usable successor, or
            the named successor is not one.
        ProviderUnavailableError: The backend could not be reached.
        ModelNotFoundError: The weights vanished between the listing and the
            delete (another caller got there first).
    """
    # Pre-OIDC the actor is a constant, and recording it anyway is the point:
    # the trail becomes meaningful the moment real identity populates it, and a
    # destructive verb with no record is not something to add auth to later.
    # Refusals are logged as well as successes - for a destructive verb, the
    # attempts that were turned away are the more interesting half.
    try:
        outcome = await _delete_model(
            provider=provider,
            settings=settings,
            session=session,
            sessionmaker=sessionmaker,
            tenant=tenant,
            name=name,
            replacement=replacement,
            store=store,
        )
    except Exception as exc:
        logger.info(
            "model delete refused by tenant=%s user=%s: model=%r reason=%s",
            tenant.tenant_id,
            tenant.user_id,
            name,
            getattr(exc, "detail", None) or type(exc).__name__,
        )
        raise
    logger.info(
        "model deleted by tenant=%s user=%s: model=%r unpinned=%d new_default=%r",
        tenant.tenant_id,
        tenant.user_id,
        outcome.removed,
        outcome.unpinned_conversations,
        outcome.new_default,
    )
    return outcome


async def _delete_model(
    *,
    provider: ModelProvider,
    settings: Settings,
    session: AsyncSession,
    sessionmaker: async_sessionmaker[AsyncSession],
    tenant: TenantContext,
    name: str,
    replacement: str | None = None,
    store: VectorStore | None = None,
) -> DeleteOutcome:
    """Run the guards and the removal; see :func:`delete_model` for the contract."""
    tag = require_model_name(name).text
    if not provider.management_capabilities().can_delete:
        raise BadRequestError(_NO_DELETE_VERB)

    # Every check below compares the REQUESTED tag against a provider listing,
    # against written-down configuration, or against a stored pick - and one
    # model has several spellings in all three. Matching the literal string
    # alone would let a spelling walk past a lock and delete the embedder or the
    # model being served; it would also answer "not installed" about a model
    # that plainly is, purely because the request wrote its name out in full.
    aliases = tag_aliases(tag)

    listing = await provider.list_models()
    installed = {info.name for info in listing}
    if not any(installed_tag_matches(alias, installed) for alias in sorted(aliases)):
        raise NotFoundError(f"'{tag}' is not installed, so there is nothing to delete.")

    # The role, from the RUNTIME's report about these weights - the same
    # derivation the management view groups rows by, so the button a page
    # disables and the request this refuses answer one question. A tag whose
    # probe failed reads as role-less and stays manageable: "the runtime would
    # not say" must not become a refusal a reader cannot act on.
    role = derive_role(await report_for(provider, settings.model_provider, tag))
    management, lock_reason = management_of(role)
    if management != "manageable":
        if role == "embedding":
            # The embedding lock narrows to a per-row policy: what actually
            # strands vectors is deleting the ACTIVE embedder or one that
            # stored passages still reference, and the report knows which.
            # Recomputed fresh here (never trusted from the page): an ingest
            # can land between the render and the click. No store, or a store
            # that does not answer, FAILS CLOSED - see embedding_delete_block.
            # Judged against the embedder a retrieval actually queries with, so
            # a workspace that named its own has the guard protecting the tag
            # its own turns use.
            active = await resolve_embedding(
                settings=settings,
                stored=await load_defaults(session, tenant),
                store=store,
                provider=provider,
                provider_name=settings.model_provider,
                refresh=True,
            )
            report = (
                await vector_space_report(store=store, active_model=active.model)
                if store is not None
                else None
            )
            blocked_embedding = embedding_delete_block(tag, report)
            if blocked_embedding is not None:
                raise ConflictError(blocked_embedding)
        else:
            raise ConflictError(lock_reason or f"'{tag}' is not managed from here.")

    # The SAME policy the management view renders, so the message a user gets
    # for pressing a button a stale page left enabled is the message that button
    # was disabled with.
    usage = await load_model_usage(
        settings=settings,
        session=session,
        tenant=tenant,
        provider_name=settings.model_provider,
        sessionmaker=sessionmaker,
        # Off the SAME listing the install check just made, so the guard and
        # the "is it here at all" test cannot be answered from two different
        # snapshots of the server.
        digests={info.name: info.digest for info in listing if info.digest},
    )
    blocked = delete_block_reason(tag, usage)
    if blocked is not None:
        raise ConflictError(blocked)

    new_default: str | None = None
    if is_workspace_default(tag, usage):
        new_default = _validated_successor(
            tag=tag,
            aliases=aliases,
            replacement=replacement,
            installed=installed,
            roles=await _roles_of(provider, settings.model_provider, sorted(installed)),
        )
        # Through the settings writer, not the repository, so the reassignment
        # inherits the read-merge-write lock. Without it an admin saving the
        # workspace-defaults page in the same window would have their other
        # fields overwritten by this update's stale copy of the blob.
        await set_default_model(session, tenant, new_default)

    unpinned = await ConversationRepository(session, tenant).clear_model_pins(aliases)

    await provider.delete_model(tag)
    # The cached report for this tag now describes weights that are gone, and
    # left to expire it would keep the deleted model looking capable for the
    # rest of its TTL - on the very page an operator is watching the deletion
    # happen on. This drops the WHOLE map rather than the one entry: the cache
    # has no per-tag eviction, and it does not need one. Every other entry is
    # re-probed on first use for the price of one metadata read, off any turn
    # that matters, and a delete is rare.
    #
    # The DIGEST-BOUND caches go with it, for a sharper reason than staleness.
    # Both the measured verdicts and the stored declarations bind to the digest
    # a tag currently resolves to, and the LISTING that answers "currently" is
    # itself cached - so leaving it would let a turn validate a verdict or a
    # policy row against weights this server no longer holds, for the rest of
    # the window. A tag re-pulled into the space just freed is exactly the case
    # where the two clocks disagree, and the name is free to be reused the
    # moment this returns.
    #
    # That listing lives behind clear_earned_cache. The policy call after it is
    # symmetry rather than a second mechanism - nothing here WRITES a
    # declaration, so those rows cannot be stale - and it is here so the three
    # caches one digest-bound answer depends on are dropped together.
    clear_facts_cache()
    clear_earned_cache()
    clear_policy_cache()
    return DeleteOutcome(removed=tag, unpinned_conversations=unpinned, new_default=new_default)


async def _roles_of(provider: ModelProvider, provider_name: str, tags: list[str]) -> dict[str, str]:
    """Derive each installed tag's role from the runtime's own report, at once.

    Bounded and cached by the facts service, and taken on a DELETE rather than
    on a turn - so a handful of metadata reads is the right trade for a guard
    that must not send a workspace's default to a model that cannot answer.

    A tag whose probe failed maps to the empty role, which reads as manageable:
    "the runtime would not say" is not evidence that a model cannot chat.
    """
    reports = await runtime_capabilities_many(provider, provider_name, tags)
    return {tag: derive_role(report) for tag, report in zip(tags, reports, strict=True)}


def _validated_successor(
    *,
    tag: str,
    aliases: set[str],
    replacement: str | None,
    installed: set[str],
    roles: dict[str, str],
) -> str:
    """Return the validated successor for the workspace default, or refuse.

    Deleting the stored default with no successor would leave every
    conversation that never chose a model - the common case - with nothing to
    resolve to, so the successor is part of the same request rather than a
    follow-up someone may forget. The candidate rule mirrors what the stored
    pick itself honours (see ``resolve_chat_models``): any model the server
    actually holds, except the roles that cannot drive a chat turn.

    Args:
        tag: The tag being deleted (already normalized).
        aliases: Its spellings, from ``tag_aliases``.
        replacement: The requested successor, as received; None when the
            request named none.
        installed: The provider's installed listing, by name.
        roles: Each installed tag's runtime-derived role, from
            :func:`_roles_of`. Passed in rather than probed here so this stays
            pure and the whole guard reads one snapshot of the server.

    Returns:
        The normalized successor tag to store.

    Raises:
        ConflictError: No successor was named, none is possible, or the named
            one is not installed, is the deleted tag itself, or serves a role
            that cannot drive chat.
        UnprocessableEntityError: The named successor is not a usable name.
    """
    candidates = sorted(
        candidate for candidate in installed if _can_succeed(candidate, aliases, roles)
    )
    if not candidates:
        raise ConflictError(
            f"'{tag}' is this workspace's default chat model and the only installed model "
            "that could drive a chat turn. Deleting it would leave the workspace unable to "
            "answer at all - pull another chat model first, then delete this one."
        )
    if replacement is None or not replacement.strip():
        raise ConflictError(
            f"'{tag}' is this workspace's default chat model. Deleting it needs a "
            "successor: name the model that takes over as the default in the same "
            "request ('replacement')."
        )
    successor = require_model_name(replacement).text
    if successor in aliases:
        raise ConflictError(
            f"'{successor}' is the model being deleted, so it cannot take over as the "
            "workspace default. Name a different model."
        )
    if not installed_tag_matches(successor, installed):
        raise ConflictError(
            f"'{successor}' is not installed, so it cannot take over as the workspace "
            "default. Name a model the server holds."
        )
    successor_role = _role_under_any_spelling(successor, roles)
    if management_of(successor_role)[0] != "manageable":
        raise ConflictError(
            f"'{successor}' is reported as {article_for(successor_role)} {successor_role} model, "
            "which cannot drive a chat turn, so it cannot take over as the workspace default."
        )
    return successor


def _role_under_any_spelling(tag: str, roles: dict[str, str]) -> str:
    """The runtime-derived role for ``tag`` under any of its spellings.

    The role map is keyed by the names the provider LISTED, and a request may
    spell a tag ":latest" or with the publisher namespace while the server
    lists it bare - the exact walk :func:`_delete_model` does for the tag being
    deleted. The successor checks must see through the same conventions, or a
    spelling walks a locked role past them and the embedder becomes the
    workspace's chat default.
    """
    aliases = tag_aliases(tag)
    for listed, role in roles.items():
        if tag_aliases(listed) & aliases:
            return role
    return ""


def _can_succeed(candidate: str, aliases: set[str], roles: dict[str, str]) -> bool:
    """True when an installed tag could take over as the workspace default.

    Not the deleted tag under any spelling, and not a locked role: the same
    rule :func:`_validated_successor` enforces for a NAMED successor, asked
    across the listing to learn whether any successor exists at all.
    """
    if candidate in aliases or installed_tag_matches(candidate, aliases):
        return False
    return management_of(_role_under_any_spelling(candidate, roles))[0] == "manageable"
