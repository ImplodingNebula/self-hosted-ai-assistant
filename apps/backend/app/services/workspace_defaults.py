"""The workspace defaults: reading them, writing them, and applying them.

``workspace_settings.settings`` sat behind read-only callers for two phases -
the chat turn, the warm endpoint, the title job and the management view all
consult it, and nothing but a one-off first-run seed ever wrote it. This module
is the writer: a settings page, a manager row action, the delete flow's
reassignment and a future orchestrator assignment all reach the same function,
so the validation, the row lock and the per-field merge cannot drift into four
slightly different versions of themselves.

One writer stays outside it, deliberately: the chat turn's first-run seed
(``chat_service._seed_single_model_default``) writes through the repository
directly, because it must SWALLOW a failure - two first turns can race the
insert, and a default that failed to save is not worth failing a turn over.
Every writer that a person triggered comes through here, where a failure is
the answer rather than something to log past.

Two of the six defaults are also APPLIED from here, because the layer they
clamp has no other place to learn about a tenant:

- :func:`clamp_num_ctx` (in ``model_facts``, beside the rest of the window
  rule) takes the ceiling this module reads.
- :func:`apply_reranker_default` turns the deployment's reranker into the one
  this workspace actually gets.

The other four - model, thinking, web search and the system prompt - already
flow through the resolution chain (``resolve_chat_models``,
``resolve_turn_defaults`` and ``compose_system_prompt``), so this module only
has to store them.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import Settings
from app.core.errors import ConflictError, NotFoundError
from app.core.tenancy import TenantContext
from app.providers.base import ModelProvider, installed_tag_matches
from app.providers.rag.base import Reranker
from app.repositories.workspace_settings import WorkspaceSettingsRepository
from app.schemas.settings import WorkspaceDefaults, WorkspaceDefaultsPatch
from app.services.model_names import require_model_name


@dataclass(frozen=True)
class RerankerAvailability:
    """Whether this deployment has a reranker for the workspace toggle to govern.

    The honesty half of the toggle. A reranker is a sidecar chosen by
    deployment configuration, not something a settings page can conjure, so a
    workspace with no sidecar gets a STATUS ("not configured", with the reason)
    rather than a switch that would silently do nothing - the same rule the
    management page follows for a verb its provider does not have.

    Attributes:
        configured: A reranker would run if the workspace permitted it.
        model: The sidecar's model tag, when there is one.
        detail: One sentence naming why it is unavailable, or describing what
            is wired. Rendered as visible text, never a tooltip.
    """

    configured: bool
    model: str | None
    detail: str


def reranker_availability(settings: Settings) -> RerankerAvailability:
    """Report whether a reranker is wired, and say why when it is not.

    ``RERANKER_URL`` is the gate, and the only one. Wiring a sidecar is an act
    of deployment - a container to start, memory to spend, a model to download
    - so the operator who did it has already said everything there is to say,
    and the same condition :func:`app.api.deps.get_reranker` builds on is named
    once here so the page and the dependency cannot disagree.

    Two gates used to have to agree with it, both of them lines in a committed
    file: one naming a reranker, and one flag saying whether it was on. Both
    are gone, because three votes on one question can only ever produce
    disagreements to explain - and the shipped configuration WAS one, a live
    URL beside a declaration that disabled reranking, which read as configured
    and did nothing.

    The model TAG comes from ``RERANKER_MODEL`` beside the URL, for the same
    reason: a rerank call has to carry a model name, and the only party that
    knows which weights the sidecar loaded is whoever started it with a
    ``--model-id``. The two keys move together or not at all. It cannot be
    blank by the time it arrives here - a blank one reads as unset and falls to
    the shipped default (see
    :meth:`app.core.config.Settings._blank_reranker_model_as_default`), because
    a nameless model would disable a sidecar that is plainly running.

    Args:
        settings: The active settings (``RERANKER_URL``, ``RERANKER_MODEL``).

    Returns:
        The :class:`RerankerAvailability` for this deployment.
    """
    if not settings.reranker_url:
        return RerankerAvailability(
            configured=False,
            model=None,
            detail=(
                "No reranker is configured for this deployment: RERANKER_URL is unset, so "
                "retrieval ranks on dense similarity alone."
            ),
        )
    return RerankerAvailability(
        configured=True,
        model=settings.reranker_model,
        detail=f"Retrieval re-scores its candidates with {settings.reranker_model}.",
    )


def apply_reranker_default(
    reranker: Reranker | None, defaults: WorkspaceDefaults
) -> Reranker | None:
    """Apply the workspace reranker toggle to the deployment's reranker.

    Clamps OFF and nothing else, for the same reason the context ceiling only
    ever clamps down: the sidecar exists (or does not) because of deployment
    configuration, and no stored preference can start a process. So ``False``
    withholds a wired reranker, while ``True`` and unset are the same
    permissive answer - the toggle remembers an operator's intent for the day
    a sidecar is wired, and never claims to have provided one.

    Passing this through even when nothing is wired is deliberate: ``None`` in
    means ``None`` out, so a caller never has to ask whether the clamp applies.

    Args:
        reranker: The reranker this deployment wired, or None.
        defaults: The tenant's stored defaults.

    Returns:
        The reranker retrieval should actually use.
    """
    if defaults.reranker is False:
        return None
    return reranker


async def load_defaults(session: AsyncSession, tenant: TenantContext) -> WorkspaceDefaults:
    """Read the tenant's stored defaults as the typed, lenient view.

    Args:
        session: An open session.
        tenant: The acting principal.

    Returns:
        The typed view; every field unset when nothing is stored.
    """
    return WorkspaceDefaults.from_stored(await WorkspaceSettingsRepository(session, tenant).load())


async def apply_defaults(
    session: AsyncSession, tenant: TenantContext, patch: WorkspaceDefaultsPatch
) -> WorkspaceDefaults:
    """Apply a per-field patch to the tenant's defaults and return the result.

    Read-merge-write under a row lock, so two controls saved at once serialize
    instead of one silently dropping the other's field. Flushed, not committed:
    the caller owns the transaction boundary.

    Both model tags - the chat default and the embedder - are parsed through
    the same validator every management verb uses. That check is about SHAPE,
    not existence: a name carrying a scheme, a traversal or a control character
    is refused before it can ever be handed to a model server as a default,
    while a merely unknown tag is stored and falls through the resolution chain
    exactly as a stale one does.

    Args:
        session: An open session.
        tenant: The acting principal.
        patch: The fields to change; absent fields are left alone and an
            explicit None clears one.

    Returns:
        The typed view of the defaults as they now stand.

    Raises:
        UnprocessableEntityError: A model tag is not a usable model name.
    """
    # The PARSED text is what gets stored, not the raw field. The parser trims,
    # so a tag pasted with trailing whitespace would otherwise be validated in
    # its clean form and stored in its dirty one - and the read side compares
    # stored tags against the provider's listing by equality, where
    # " qwen3:4b" matches nothing.
    cleaned = {
        field: require_model_name(value).text
        for field in ("model", "embedding_model")
        if (value := getattr(patch, field)) is not None
    }
    if cleaned:
        patch = patch.model_copy(update=cleaned)
    repo = WorkspaceSettingsRepository(session, tenant)
    stored = await repo.load_for_update()
    merged = patch.merged_into(stored)
    if merged != stored:
        await repo.replace_defaults(merged)
    return WorkspaceDefaults.from_stored(merged)


async def set_default_model(
    session: AsyncSession, tenant: TenantContext, model: str | None
) -> WorkspaceDefaults:
    """Set (or clear) the workspace's default chat model, touching nothing else.

    The narrow entry point for surfaces that change ONE field - a "set as
    default" action on a manager row, the delete flow's reassignment - so they
    inherit the lock and the name validation instead of writing the blob
    themselves.

    A blank tag is a CLEAR, not an error. These callers derive the tag rather
    than being handed one (a successor chosen for a model being deleted, a row
    the operator clicked), so the empty string means "there is nothing to put
    here" - and the patch model's own ``min_length`` would otherwise turn that
    into an unhandled validation error deep inside a delete.

    Args:
        session: An open session.
        tenant: The acting principal.
        model: The tag to store, or None/blank to clear the default.

    Returns:
        The typed view of the defaults as they now stand.

    Raises:
        UnprocessableEntityError: The tag is not a usable model name.
    """
    tag = (model or "").strip() or None
    return await apply_defaults(session, tenant, WorkspaceDefaultsPatch(model=tag))


async def validated_default_model(*, provider: ModelProvider, provider_name: str, name: str) -> str:
    """Check that ``name`` may be a workspace default, and normalize it.

    The validation half, split from the write on purpose: it touches no
    database, so it composes with the row-locking writer above. The rules are
    the SAME ones a delete's successor is held to
    (:func:`app.services.model_delete._validated_successor`) - the console must
    not be able to store a default the delete path would refuse as a successor.

    The role check reads the RUNTIME's report, through the SAME derivation the
    management view groups rows by and the delete path's successor rule
    applies. One derivation, because the two are held to one standard: the
    console must not be able to store a default the delete path would refuse
    as a successor, and a second local copy of "what counts as a chat model"
    is precisely how the two came to disagree.

    That derivation reads a report of neither capability - and an unread report
    - as the EMPTY role, which is manageable. Both are the permissive answer on
    purpose: "the runtime would not say" is not evidence that a model cannot
    chat, and refusing on a blipped metadata call would make a legitimate
    default unstorable for no reason a reader could see.

    Raises:
        UnprocessableEntityError: The name is not a usable model name.
        NotFoundError: The backend does not hold these weights.
        ConflictError: The tag serves a role that cannot drive a chat turn.
    """
    # Imported here, not at module top: model_manager reads load_defaults from
    # this module, so a top-level import back at it would be a cycle.
    from app.services.model_facts import report_for
    from app.services.model_manager import article_for, derive_role, management_of

    tag = require_model_name(name).text
    listing = await provider.list_models()
    if not installed_tag_matches(tag, {info.name for info in listing}):
        raise NotFoundError(
            f"'{tag}' is not installed, so it cannot be the workspace default. Download it first."
        )
    role = derive_role(await report_for(provider, provider_name, tag))
    if management_of(role)[0] != "manageable":
        raise ConflictError(
            f"'{tag}' is reported as {article_for(role)} {role} model, which cannot drive a "
            "chat turn, so it cannot be the workspace default."
        )
    return tag


async def set_validated_default_model(
    *,
    provider: ModelProvider,
    provider_name: str,
    session: AsyncSession,
    tenant: TenantContext,
    name: str,
) -> str:
    """Validate ``name`` as a workspace default, then store it under the row lock."""
    tag = await validated_default_model(provider=provider, provider_name=provider_name, name=name)
    await set_default_model(session, tenant, tag)
    return tag
