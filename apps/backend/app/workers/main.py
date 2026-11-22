"""The arq worker: background jobs that must not run in a request.

Three of these jobs are enqueued on demand, the rest are periodic maintenance.

``parse_and_ingest_attachment`` is enqueued by the upload endpoint. Parsing and
embedding a file takes seconds to minutes and must never block the HTTP
response; the row's ``status`` is how a client follows along.

``purge_deleted_attachments`` is enqueued when a conversation is deleted, so the
files that belonged to it stop taking up space promptly rather than at the next
sweep.

``generate_conversation_title`` is enqueued when a conversation's first exchange
completes. It asks the chat model to name the thread properly, replacing the
truncation the turn wrote instantly - work that belongs after the reply, not in
front of it, because the answer is what the reader is waiting for.

``sweep_expired_vectors`` is maintenance. Qdrant has no per-point expiry, so
session-scoped chunks carry a ``ttl_at`` timestamp and this worker deletes the
expired ones on a fixed cadence (plus once at startup, so a worker that was down
for a while catches up immediately).

``collect_attachment_garbage`` is the maintenance counterpart of the purge: the
safety net for the files of a deleted conversation whose purge job never ran,
and for blobs an upload wrote just before failing to commit its row. It also
carries the two-tier retention sweep, which empties files past their configured
window and leaves each row behind as a tombstone the interface can be honest
about.

Run with: ``arq app.workers.main.WorkerSettings`` (the compose ``worker``
service does exactly this on the backend image).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import timedelta
from typing import Any
from uuid import UUID

from arq import Retry, cron
from arq.connections import ArqRedis, RedisSettings
from arq.jobs import Job, JobStatus

from app.core.config import Settings, get_settings
from app.core.factories import build_model_provider, build_storage_backend
from app.core.logging import configure_logging
from app.core.tenancy import TenantContext
from app.db.engine import build_engine, build_sessionmaker
from app.models import (
    ATTACHMENT_FAILED,
    TITLE_ORIGIN_GENERATED,
    TITLE_ORIGIN_MANUAL,
    Conversation,
)
from app.parsing.base import ParserOutOfMemoryError
from app.parsing.registry import default_registry
from app.providers.base import (
    GenerationOptions,
    ModelInfo,
    ModelProvider,
    ProviderUnavailableError,
)
from app.providers.rag.base import Embedder, VectorStoreUnavailableError
from app.providers.rag.embedder import ProviderEmbedder
from app.providers.rag.qdrant_store import QdrantVectorStore
from app.repositories.attachments import AttachmentRepository
from app.repositories.conversations import ConversationRepository
from app.repositories.messages import MessageRepository
from app.schemas.settings import ConversationSettings
from app.services import think_check
from app.services.attachment_gc import (
    collect_garbage,
    fail_stalled_attachments,
    purge_attachments,
    retention_window,
)
from app.services.attachment_ingest import ingest_attachment
from app.services.embedding_resolution import UNCONFIGURED_DETAIL, resolve_embedding
from app.services.model_facts import (
    report_for,
    resolve_num_ctx,
)
from app.services.model_identity import installed_digests
from app.services.model_resolution import (
    ChatModelPair,
    ThinkMode,
    resolve_chat_models,
)
from app.services.model_routing import NoChatModelConfiguredError, route_around_missing
from app.services.thinking import normalize_think
from app.services.titles import build_title_messages, sanitize_title
from app.services.workspace_defaults import load_defaults

logger = logging.getLogger(__name__)

# How many times the job runs before giving up. The ingest job drives its own
# retries by raising arq's Retry (a plain exception is NOT retried - arq only
# requeues on Retry or CancelledError), and uses this to recognise its own last
# attempt so it leaves a row that explains itself rather than one stuck in
# `parsing` forever.
MAX_TRIES = 3

# How long to wait before each retry, indexed by the attempt just finished
# (attempt 1 waits 10s, attempt 2 waits 30s). A brief backoff lets a service
# that is restarting - object storage, the embedder, Qdrant - come back before
# the next try, without holding a worker slot busy in the meantime.
_RETRY_DEFER_SECONDS = (10, 30)

# Longer, when the reader was killed for memory, because the thing to wait out is
# different. A restarting service is back in seconds; the memory a model takes as
# it loads is held for a minute or more, and retrying inside that window just gets
# the reader killed again - three times, and then the file is blamed for it.
_MEMORY_RETRY_DEFER_SECONDS = (60, 120)

# The upper bound on one attachment's parse + embed. arq cancels the task past
# this, so a pathological file cannot wedge a worker slot. It is generous
# because embedding a large document on a small GPU genuinely takes minutes.
JOB_TIMEOUT_SECONDS = 900

_RETRIES_EXHAUSTED_MESSAGE = (
    "Processing failed repeatedly; a service it depends on may be unavailable. "
    "Try uploading the file again."
)

# Said instead when every attempt ran out of memory. It names both causes, in the
# order they are worth trying, because from in here they are genuinely
# indistinguishable: the machine may have been busy, or the file may simply be too
# big for it. Saying only "a service may be unavailable" would name neither.
_OUT_OF_MEMORY_MESSAGE = (
    "Reading this file ran out of memory on every attempt. Uploading it again may "
    "work if the machine was busy at the time; if it fails again, the file is too "
    "large to read here."
)

# arq cron fires on fixed minute-of-hour marks, so only divisors of 60 give a
# truly even cadence; other configured values are rounded DOWN to the nearest
# divisor (sweeping slightly more often errs on the private side).
_EVEN_INTERVALS = (30, 20, 15, 12, 10, 6, 5, 4, 3, 2, 1)

# arq keys a retry/dedup by this id, so re-enqueuing the same attachment (a
# double-clicked upload that somehow reached here twice) collapses to one job.
_JOB_ID_PREFIX = "attachment-ingest:"


# arq keys a job by this id too, so a conversation can only ever have one title
# job in flight - a resumed confirmation that finishes the same first exchange
# does not queue a second one.
_TITLE_JOB_ID_PREFIX = "conversation-title:"

# ...unless the turn asks for a DISTINCT one. Editing a conversation's opening
# question re-runs the naming, and arq remembers a finished job's id for a while,
# so a second run keyed on the conversation alone would be deduplicated against
# the first and silently never happen. Such a turn mints a discriminator of its
# own (and carries it across a confirmation pause, so the resume enqueues the
# same job rather than a second one).
_TITLE_JOB_ID_SEPARATOR = ":"

# A title is a handful of words. The cap is tight enough that a model which
# starts explaining its choice is cut off rather than left to run, and the low
# temperature keeps the answer descriptive instead of inventive.
_TITLE_MAX_TOKENS = 32
_TITLE_TEMPERATURE = 0.2


def title_job_id(conversation_id: UUID | str, discriminator: str | None = None) -> str:
    """The arq job id of a conversation's title job.

    Derived from the conversation and never stored, so both the fresh-turn and
    the resumed-confirmation enqueue compute the same id without coordinating.

    Args:
        conversation_id: The thread being named.
        discriminator: Set by a turn that re-names an ALREADY-named thread (an
            edited opening question), so its job is not deduplicated against the
            one the opening turn ran. None keeps the plain per-conversation id,
            which is what makes the first exchange's two enqueue points - the
            fresh turn and a resumed confirmation - collapse to one job.
    """
    if discriminator is None:
        return f"{_TITLE_JOB_ID_PREFIX}{conversation_id}"
    return f"{_TITLE_JOB_ID_PREFIX}{conversation_id}{_TITLE_JOB_ID_SEPARATOR}{discriminator}"


def ingest_job_id(attachment_id: UUID | str) -> str:
    """The arq job id of an attachment's parse job.

    Derived from the attachment and never stored, so anyone holding the row can
    recompute it. Two callers need that: the upload enqueue (a duplicate
    submission collapses onto the same id instead of parsing the file twice) and
    the stalled-upload sweep, which asks the queue whether this job still exists
    before concluding that nothing is ever going to finish the row. Derived in
    ONE place so those two can never disagree about which job they mean.
    """
    return f"{_JOB_ID_PREFIX}{attachment_id}"


# The job states that mean the queue is still going to run this job. Anything
# else (complete, not_found) means nothing will come back to the row.
_LIVE_JOB_STATUSES = frozenset({JobStatus.deferred, JobStatus.queued, JobStatus.in_progress})

# How often to look for uploads nothing is processing any more. Not a setting,
# and deliberately far more often than the hourly garbage collection this used
# to be a natural home for: the GC walks the whole object store, while this is a
# single indexed query, and someone is WATCHING a stalled upload. Waiting up to
# an hour to tell them it died is its own bug.
_STALL_CHECK_MINUTES = 5


def _sweep_minutes(interval_minutes: int, *, setting: str) -> set[int]:
    """Map a configured interval onto arq's minute-of-hour schedule.

    Args:
        interval_minutes: The requested cadence.
        setting: The env var it came from, so a warning names the knob to fix.
    """
    if interval_minutes >= 60:
        return {0}
    chosen = next((n for n in _EVEN_INTERVALS if n <= max(1, interval_minutes)), 1)
    if chosen != interval_minutes:
        logger.warning(
            "%s=%d does not divide an hour evenly; sweeping every %d minutes instead",
            setting,
            interval_minutes,
            chosen,
        )
    return set(range(0, 60, chosen))


async def sweep_expired_vectors(ctx: dict[str, Any]) -> int:
    """Delete session vectors whose TTL has passed; returns how many."""
    store: QdrantVectorStore = ctx["vector_store"]
    try:
        swept = await store.sweep_expired(time.time())
    except VectorStoreUnavailableError:
        # The next tick retries; expired vectors only outlive their TTL for
        # as long as the store stays unreachable.
        logger.warning("TTL sweep skipped: the vector store is unavailable")
        return 0
    if swept:
        logger.info("TTL sweep removed %d expired session vectors", swept)
    return swept


# "nobody put an embedder in the context", told apart from a deliberate None.
# The distinction is the whole safety of the override below - see _job_embedder.
_RESOLVE_PER_JOB = object()


async def _job_embedder(ctx: dict[str, Any], tenant: TenantContext) -> Embedder | None:
    """Build the embedder this job's workspace ingests with, or None.

    Resolved PER JOB rather than once at startup, because the choice is now a
    workspace's to make: a process that resolved it at boot would go on writing
    into the old vector space after the choice changed, and would write into a
    different one from the one the backend's queries read - silently, since
    retrieval against an empty collection is not an error.

    ``ctx["embedder"]`` overrides everything when it is set to ANYTHING,
    including ``None``, which is how tests inject a fake or exercise the
    unconfigured path. Production leaves it at :data:`_RESOLVE_PER_JOB`, which
    is what ``ctx.get`` answers with when nothing wrote the key at all. The
    sentinel exists so that the obvious wrong write - ``ctx["embedder"] =
    None`` at startup, meaning "there isn't one" - cannot silently disable
    per-job resolution for the life of the process.

    Args:
        ctx: The arq context.
        tenant: The principal this job runs for.

    Returns:
        The embedder, or None when no rung of the chain answered.
    """
    injected = ctx.get("embedder", _RESOLVE_PER_JOB)
    if injected is not _RESOLVE_PER_JOB:
        embedder: Embedder | None = injected
        return embedder
    settings: Settings = ctx["settings"]
    sessionmaker = ctx["db_sessionmaker"]
    async with sessionmaker() as session:
        stored = await load_defaults(session, tenant)
    choice = await resolve_embedding(
        settings=settings,
        stored=stored,
        store=ctx["vector_store"],
        provider=ctx["model_provider"],
        provider_name=settings.model_provider,
    )
    if choice.config is None:
        return None
    return ProviderEmbedder(
        ctx["model_provider"], choice.config, batch_size=settings.rag_embed_batch
    )


async def parse_and_ingest_attachment(
    ctx: dict[str, Any], attachment_id: str, tenant_id: str, user_id: str
) -> int:
    """Parse one uploaded file and ingest its text; returns the chunk count.

    A thin adapter around :func:`app.services.attachment_ingest.ingest_attachment`:
    it opens a session, loads the row through the tenant-scoped repository, and
    turns a transient failure into an arq retry.

    The tenant arrives as arguments rather than being read off the row, because
    the row is fetched through a repository bound to that tenant - a job carrying
    the wrong tenant finds nothing instead of touching someone else's file.

    ``ingest_attachment`` records every terminal failure (unreadable, unsupported,
    empty file) on the row itself and returns normally, so an exception here is by
    construction transient: a service it depends on is down, or the machine ran out
    of memory reading the file. Until the last attempt those are re-raised as
    :class:`arq.Retry` so arq requeues with a short backoff - a bare exception is
    NOT retried by arq. On the last attempt the row is marked ``failed`` with a
    reason, because a row left in ``parsing`` forever explains nothing to the person
    who uploaded the file.

    A row that has vanished is not simply skipped: the file may have been deleted
    while an earlier attempt was mid-ingest, so this purges any chunks that
    attempt left behind before returning. Nothing else can - with the row gone,
    the record id lives only in this job's arguments.

    Returns:
        The number of chunks written (0 when the file was unreadable, already
        ingested, or its row had vanished).

    Raises:
        arq.Retry: A transient failure with retries remaining.
        Exception: The original transient failure on the final attempt, after
            the row has been marked ``failed``.
    """
    tenant = TenantContext(tenant_id=UUID(tenant_id), user_id=UUID(user_id), role="member")
    sessionmaker = ctx["db_sessionmaker"]
    embedder = await _job_embedder(ctx, tenant)
    if embedder is None:
        # A deployment problem, not a file problem: nothing resolves an
        # embedding model. Raising keeps it loud in the logs.
        raise RuntimeError(UNCONFIGURED_DETAIL)

    async with sessionmaker() as session:
        repo = AttachmentRepository(session, tenant)
        attachment = await repo.get(UUID(attachment_id))
        if attachment is None:
            # Deleted (or never visible to this tenant) between enqueue and run.
            # Purge before skipping, because "gone" does not mean "left nothing".
            # An EARLIER attempt of this same job may have written the file's
            # chunks, found the row deleted, and then failed to clean up (the
            # vector store blinked) - which is what put the job back in the queue.
            # With the row gone, nothing else in the system knows the record id:
            # attachment chunks carry no TTL, so no sweep would ever find them,
            # and they would go on grounding answers forever. If the delete
            # already took them, this matches nothing and costs one count.
            logger.warning("Attachment %s is gone; purging any chunks it left", attachment_id)
            try:
                purged = await ctx["vector_store"].delete_record(tenant_id, attachment_id)
            except VectorStoreUnavailableError as exc:
                job_try = ctx.get("job_try", 1)
                if job_try >= MAX_TRIES:
                    # Out of retries, and there is no row left to record this on.
                    # Loud, because these chunks now need a human.
                    logger.exception(
                        "Could not purge the chunks left by deleted attachment %s", attachment_id
                    )
                    raise
                defer = _RETRY_DEFER_SECONDS[min(job_try, len(_RETRY_DEFER_SECONDS)) - 1]
                raise Retry(defer=defer) from exc
            if purged:
                logger.info(
                    "Purged %d chunk(s) left behind by deleted attachment %s",
                    purged,
                    attachment_id,
                )
            return 0
        try:
            result = await ingest_attachment(
                attachment=attachment,
                repo=repo,
                storage=ctx["storage"],
                registry=ctx["parser_registry"],
                embedder=embedder,
                store=ctx["vector_store"],
                settings=ctx["settings"],
            )
        except Exception as exc:
            job_try = ctx.get("job_try", 1)
            if job_try >= MAX_TRIES:
                # Out of retries: leave a row that explains itself, then let the
                # job fail for real (a re-raised plain exception ends it in arq).
                logger.exception("Attachment %s failed on the final attempt", attachment_id)
                await _mark_exhausted(repo, UUID(attachment_id), reason=_final_reason(exc))
                raise
            defer = _defer_seconds(exc, job_try)
            logger.warning(
                "Attachment %s ingestion failed on attempt %d/%d; retrying in %ds (%s)",
                attachment_id,
                job_try,
                MAX_TRIES,
                defer,
                exc,
            )
            raise Retry(defer=defer) from exc
        return result.chunks


async def purge_deleted_attachments(
    ctx: dict[str, Any], attachment_ids: list[str], tenant_id: str, user_id: str
) -> int:
    """Reclaim the files of a deleted conversation; returns how many went.

    Enqueued by the conversation-delete endpoint with the ids it saw before the
    row went away. The work itself is a sweep away from being unnecessary - and
    ``collect_attachment_garbage`` is exactly that sweep - so this job exists
    only to make the common case prompt: a user who deletes a conversation
    should not have their files sitting on disk until the next cron tick.

    Like the ingest job, the tenant arrives as an argument and the rows are
    loaded through a repository bound to it, so a job carrying the wrong tenant
    finds nothing rather than touching someone else's file.

    Raises:
        arq.Retry: Storage was unavailable and attempts remain.
        Exception: The original failure on the final attempt. Nothing is lost:
            the rows still have no conversation, which is precisely what the
            periodic sweep looks for.
    """
    tenant = TenantContext(tenant_id=UUID(tenant_id), user_id=UUID(user_id), role="member")
    sessionmaker = ctx["db_sessionmaker"]
    async with sessionmaker() as session:
        repo = AttachmentRepository(session, tenant)
        try:
            purged = await purge_attachments(
                attachment_ids=[UUID(value) for value in attachment_ids],
                repo=repo,
                storage=ctx["storage"],
                vector_store=ctx["vector_store"],
            )
        except Exception as exc:
            job_try = ctx.get("job_try", 1)
            if job_try >= MAX_TRIES:
                logger.exception(
                    "Purging %d attachment(s) failed on the final attempt; the periodic "
                    "sweep will retry them",
                    len(attachment_ids),
                )
                raise
            defer = _RETRY_DEFER_SECONDS[min(job_try, len(_RETRY_DEFER_SECONDS)) - 1]
            logger.warning(
                "Attachment purge failed on attempt %d/%d; retrying in %ds (%s)",
                job_try,
                MAX_TRIES,
                defer,
                exc,
            )
            raise Retry(defer=defer) from exc
    if purged:
        logger.info("Purged %d attachment(s) from a deleted conversation", purged)
    return purged


async def generate_conversation_title(
    ctx: dict[str, Any],
    tenant_id: str,
    user_id: str,
    conversation_id: str,
    title_seed: str,
) -> str | None:
    """Replace a conversation's truncated title with one the chat model wrote.

    Enqueued once, by the turn that completed a conversation's FIRST exchange.
    The truncated fallback is already in the sidebar by then (it is written
    before the model is even called), so this job is pure improvement: it reads
    the opening question and its reply, asks the chat model for a few words that
    say what the thread is about, and writes them over the truncation.

    Everything about it is best-effort. A missing conversation, an empty or
    unusable answer, an unreachable model, a database error - each leaves the
    fallback exactly where it is, because a slightly clumsy title is a far
    better outcome than a job that keeps failing loudly over a cosmetic write.
    Nothing is raised for the same reason, which also means arq never requeues
    it: one attempt per conversation, and no retry storm behind a model outage.

    Two things it deliberately does NOT do. It never writes over a title the
    user has since typed themselves - a rename during the seconds this job takes
    wins, and the guard is checked again under the row lock at write time. And
    it never bumps ``updated_at``: a background metadata write must not reorder
    the sidebar or resurface a thread the user has moved on from (the same rule
    the settings PATCH follows, for the same reason).

    Args:
        ctx: The arq worker context (settings, sessionmaker, model provider).
        tenant_id: The owning tenant, carried on the job rather than read off
            the row - a job with the wrong tenant finds nothing.
        user_id: The owning user, for the same reason.
        conversation_id: The conversation to name.
        title_seed: The title this job is allowed to replace - the truncation the
            opening turn wrote, or, when an edited first question re-runs the
            naming, whatever the thread is called now. Anything else in the
            column, or a ``manual`` provenance on it, means a human named the
            thread meanwhile and their name stands.

    Returns:
        The title written, or None when the fallback was left in place.
    """
    try:
        return await _write_generated_title(
            ctx,
            tenant=TenantContext(tenant_id=UUID(tenant_id), user_id=UUID(user_id), role="member"),
            conversation_id=UUID(conversation_id),
            title_seed=title_seed,
        )
    except Exception:  # noqa: BLE001 - a cosmetic write must not fail a job loudly
        logger.warning(
            "Could not generate a title for conversation %s; keeping the fallback",
            conversation_id,
            exc_info=True,
        )
        return None


def _may_retitle(conversation: Conversation, title_seed: str) -> bool:
    """Whether this job may still write over the conversation's current title.

    The rename guard, in ONE place because it is checked twice: once before the
    model call (so a renamed thread costs no GPU time) and once under the row
    lock that writes, which is the check that actually matters - a rename can
    land while the model is generating.

    Two conditions, and the second is not redundant. The title must still be the
    exact string this job was told it may replace, AND its provenance must not be
    MANUAL. String equality alone was enough while a thread was named once; it is
    not once a title can be regenerated (editing the opening question re-runs
    this job with the CURRENT title as its seed), because a person who renames a
    thread to the same words it already had would then be silently overwritten.
    Provenance answers "who chose this name", which is the actual question.
    """
    if conversation.title_origin == TITLE_ORIGIN_MANUAL:
        return False
    return conversation.title == title_seed


async def _title_model(
    provider: ModelProvider,
    provider_name: str,
    chat_pair: ChatModelPair,
    resolved_think: Callable[[str], Awaitable[ThinkMode]],
    listing: Sequence[ModelInfo] | None,
) -> str | None:
    """The installed, non-thinking model this job may generate on, or None.

    Two corrections to the resolved pair's default, both of which the chat turn
    already makes and neither of which a title job can skip.

    The first is thinking. An always-thinking model spends this job's whole
    token budget reasoning and returns nothing visible, so a default that
    resolves to one is stepped aside from rather than asked. There is no walk
    BACK from a thinking tag to a direct-answer partner: a pairing points one
    way, from the model a turn routes from to the model its reasoning turns
    route to, and inventing the reverse would be guessing at a relationship
    nobody declared.

    The second is what is actually pulled. A configured tag that was never
    pulled is substituted by :func:`route_around_missing` on every chat turn,
    so resolving without it would leave the title job asking for a model the
    turn itself never uses. The substitution can land on the thinking sibling
    (it is the only partner there is), which is why thinking is re-checked
    after it rather than only before.

    The gate reads the SAME resolved mode the think flag below it does - the
    stored declaration and the unconfirmed measurement included. The two used
    to disagree, and the gap was the whole point of the gate: a tag measured to
    reason regardless of the flag, over a declaration saying it never does,
    passed a declaration-only gate and then spent this job's entire 32-token
    budget reasoning, returning no title at all. A gate whose answer differs
    from the flag it is gating is not a gate.

    Args:
        provider: The active model provider, asked what is installed.
        provider_name: Its configured name, for the runtime-facts cache key.
        chat_pair: The conversation's resolved chat pair.
        resolved_think: The full think resolution for one tag, as the caller
            wires it. Injected rather than called directly so this stays the
            one place that decides WHICH tag, with the reads owned above it.
        listing: The provider's installed models, when the caller already took
            a listing the resolution was made against - handed down so the
            rescue reads the SAME snapshot rather than a second fetch that
            could disagree with it. None means that listing FAILED: a resolved
            tag then skips the rescue outright (see the body), while a blank
            one still goes through it for the typed first-run raise.

    Returns:
        The tag to generate the title on, or None when every candidate reasons,
        or when the server holds nothing that can chat at all - the caller then
        keeps the fallback without spending a model call.
    """
    model = chat_pair.default
    if model and await resolved_think(model) == "always":
        return None
    if listing is None and model:
        # The caller's listing failed, so there is no snapshot to judge
        # installation by - and the rescue fetching its own would be a second
        # round trip to a server just observed down, substituting against a
        # listing the resolution above never saw. The turn's routing closures
        # short-circuit the same way, so the resolved tag stands here too, and
        # a server that is still down answers the generation with its own
        # typed error. A BLANK tag does not take this exit: it has nothing to
        # stand on, and the rescue's typed raise below is the honest
        # step-aside when its own fetch cannot find a candidate either. The
        # gate above already answered the think question for this tag.
        return model
    try:
        model, _ = await route_around_missing(
            provider, provider_name, chat_pair, model, think=False, listing=listing
        )
    except NoChatModelConfiguredError:
        # A deployment whose model server holds nothing chat-capable. The chat
        # turn says so to the person who asked; naming a thread is cosmetic, so
        # this job simply steps aside rather than raising into the job log.
        logger.info(
            "The model server holds no chat model; leaving conversation titled by its first message"
        )
        return None
    return None if await resolved_think(model) == "always" else model


async def _write_generated_title(
    ctx: dict[str, Any],
    *,
    tenant: TenantContext,
    conversation_id: UUID,
    title_seed: str,
) -> str | None:
    """The title job's body; see :func:`generate_conversation_title`."""
    settings = ctx["settings"]
    sessionmaker = ctx["db_sessionmaker"]
    provider = ctx["model_provider"]

    async with sessionmaker() as session:
        conversation = await ConversationRepository(session, tenant).get(conversation_id)
        if conversation is None:
            # Deleted between enqueue and run. Nothing to name, nothing to log
            # loudly about.
            return None
        if not _may_retitle(conversation, title_seed):
            logger.info(
                "Conversation %s was renamed while its title job was queued; leaving it",
                conversation_id,
            )
            return None
        # The conversation's own model pick, so the title rides the model this
        # thread is already using instead of loading a second one - on a small
        # card that would evict the warm chat model to write six words.
        conversation_model = ConversationSettings.from_stored(conversation.settings or {}).model
        workspace_defaults = await load_defaults(session, tenant)
        message_repo = MessageRepository(session, tenant)
        first_user = await message_repo.first_of_role(conversation_id, "user")
        first_assistant = await message_repo.first_of_role(conversation_id, "assistant")

    if first_user is None or first_assistant is None or not first_assistant.content.strip():
        # An answerless (or empty) first exchange says nothing a title could be
        # built from; the truncated question is already the best available name.
        return None

    # The provider's listing, taken once and read twice. The resolution uses it
    # to honour-or-disprove the STORED rungs exactly as the turn does - a stale
    # conversation pin or workspace default falls through to the next rung
    # instead of standing - and the routing rescue below reads the same
    # snapshot, so the two cannot disagree about what is installed. Resolving
    # without it is how this job was once silently dead exactly where chat
    # worked: a stale workspace default stood here, and the rescue's partner
    # substitution then landed on the always-thinking sibling the gate refuses.
    try:
        listing: Sequence[ModelInfo] | None = await provider.list_models()
    except ProviderUnavailableError:
        listing = None
    installed = {info.name for info in listing} if listing is not None else None

    chat_pair = resolve_chat_models(
        settings,
        conversation_model=conversation_model,
        workspace_model=workspace_defaults.model,
        installed=installed,
    )

    async def resolved_think(tag: str) -> ThinkMode:
        """How ``tag`` treats the think flag, over every source there is.

        The ONE answer this job uses, for both the gate that picks a model and
        the flag it then sends. A gate reading a different source from the flag
        it gates is how a tag measured to reason regardless of the flag passed
        a declaration-only check and then returned no title at all.
        """
        return await think_check.resolve_think_mode(
            provider_name=settings.model_provider,
            model=tag,
            report=await report_for(provider, settings.model_provider, tag),
            digests=await installed_digests(provider, settings.model_provider),
            sessionmaker=sessionmaker,
        )

    model = await _title_model(
        provider, settings.model_provider, chat_pair, resolved_think, listing
    )
    if model is None:
        # Nothing to fall back to but the truncation, and this is the ONE place
        # that can tell before spending anything: an always-thinking model would
        # put all 32 tokens into reasoning and return no visible answer at all,
        # so asking it is a round trip that cannot succeed.
        logger.info(
            "No non-thinking chat model is available; leaving conversation %s titled by its "
            "first message",
            conversation_id,
        )
        return None
    # The pair's NON-thinking member, normalized so a never-thinking tag is not
    # sent the flag at all - through the SAME resolver the turn uses, report
    # included, so a tag whose template has no think plumbing is
    # not sent `think: false` here and nothing there. num_ctx matches what a
    # chat turn on this model asks for for the same reason: a different request
    # would make the runtime reload the model, which is the one thing this job
    # must not cost.
    #
    # The stored declaration and the unconfirmed measurement join for exactly
    # that reason too. This job rides the model a live thread is already using;
    # sending it a think value the turn would not send is a second request
    # shape for one model, which is a reload - and on a card that fits one
    # model, a reload for six words of title. Resolved through the same helper
    # the gate above used, so the two cannot answer differently.
    options = GenerationOptions(
        think=normalize_think(False, await resolved_think(model)),
        temperature=_TITLE_TEMPERATURE,
        num_ctx=await resolve_num_ctx(
            provider,
            settings.model_provider,
            model,
            default_window=settings.default_context_window,
            ceiling=workspace_defaults.context_ceiling,
        ),
        max_tokens=_TITLE_MAX_TOKENS,
    )
    parts: list[str] = []
    async for chunk in provider.chat_stream(
        build_title_messages(first_user.content, first_assistant.content),
        model=model,
        options=options,
    ):
        if chunk.type == "token":
            parts.append(chunk.delta)
        elif chunk.type == "error":
            logger.warning(
                "The model could not title conversation %s: %s", conversation_id, chunk.error
            )
            return None
        # Reasoning deltas are dropped rather than collected: the title is the
        # visible answer, never the thinking that led to it. A model that
        # reasons is not routed to here at all, so this is a belt-and-braces
        # guard for a tag whose behavior nobody has declared.

    title = sanitize_title("".join(parts))
    if title is None:
        logger.info("The model returned no usable title for conversation %s", conversation_id)
        return None

    async with sessionmaker() as session:
        repo = ConversationRepository(session, tenant)
        # FOR UPDATE, so a rename committing right now either lands before this
        # read (and the guard sees it) or waits behind this write.
        conversation = await repo.get_for_update(conversation_id)
        if conversation is None or not _may_retitle(conversation, title_seed):
            return None
        conversation.title = title
        # Stamped with the write, so a LATER regeneration (an edited opening
        # question) knows this name was the machine's and may replace it, while a
        # rename in between stamps MANUAL and stops it.
        conversation.title_origin = TITLE_ORIGIN_GENERATED
        # No touch(): updated_at stays where the turn left it. See the docstring.
        await repo.commit()
    logger.info("Titled conversation %s from its first exchange", conversation_id)
    return title


async def collect_attachment_garbage(ctx: dict[str, Any]) -> int:
    """Reclaim and expire attachments; returns how many rows and blobs it acted on.

    The count is the sum of three passes and NOT a count of things deleted: an
    expired attachment keeps its row (only its contents go), so it is included as
    work done rather than as an object reclaimed. The two log lines below tell
    the halves apart, which is what an operator actually reads.

    The safety net under every other cleanup path: it finds the files of deleted
    conversations whose purge job never ran, and the blobs of uploads that wrote
    their bytes and then failed to commit a row. A storage or database outage is
    handled inside ``collect_garbage`` - the next tick simply tries again.

    It also carries the retention sweep, which shares this cron rather than
    getting one of its own. Retention is configured in DAYS and this runs hourly,
    so a file is emptied within an hour of crossing its window - late, never
    early, which is the only direction that matters for a policy about how long
    something is KEPT. A cadence of its own would buy nothing but a second knob
    and a second full pass over the same table.
    """
    settings = ctx["settings"]
    sessionmaker = ctx["db_sessionmaker"]
    async with sessionmaker() as session:
        collected = await collect_garbage(
            session,
            ctx["storage"],
            ctx["vector_store"],
            grace=timedelta(minutes=settings.attachment_orphan_grace_minutes),
            generated_retention=retention_window(settings.generated_file_retention_days),
            upload_retention=retention_window(settings.attachment_retention_days),
        )
    if collected.rows or collected.blobs:
        logger.info(
            "Attachment GC reclaimed %d row(s) and %d orphaned blob(s)",
            collected.rows,
            collected.blobs,
        )
    if collected.expired:
        logger.info(
            "Retention expired %d attachment(s), keeping each row as a tombstone",
            collected.expired,
        )
    return collected.rows + collected.blobs + collected.expired


async def _has_live_ingest_job(redis: ArqRedis, attachment_id: UUID) -> bool:
    """Whether the queue still intends to run this attachment's parse job.

    The job id is derived from the attachment, so it can be asked about without
    anything having been stored. ``not_found`` means the queue has no record of
    it: the enqueue was lost, or the job's state went with a flushed Redis.
    ``complete`` means it ran and is over - and since a completed job that left
    its row mid-flight is a job that will never come back to it, both are dead.

    A queue that cannot be reached answers "live", because the honest answer is
    "unknown" and the safe reading of unknown is to leave the row alone.
    """
    try:
        status = await Job(ingest_job_id(attachment_id), redis).status()
    except Exception:
        logger.warning(
            "Could not ask the queue about attachment %s; leaving it alone",
            attachment_id,
            exc_info=True,
        )
        return True
    return status in _LIVE_JOB_STATUSES


async def reap_stalled_attachments(ctx: dict[str, Any]) -> int:
    """Fail uploads no job will ever finish; returns how many.

    The counterpart to every OTHER failure path in the parse job. Those all end
    with the row explaining itself - retries exhausted, file unreadable, storage
    down. This one covers the row whose job never got to run its own error
    handling, because there is no job: the queue was unreachable when the file
    was uploaded, or its state was lost. Nothing else will ever close that row,
    and the client polling it never stops waiting.

    A row whose job the queue STILL holds is left alone however old it is - it is
    queued, not dead, and this sweep only ever runs inside a live worker that is
    going to get to it. See :func:`fail_stalled_attachments` for why failing one
    of those is worse than leaving it.

    Deliberately not run at startup: the first thing a returning worker does is
    work off exactly the backlog this would be looking at.
    """
    settings = ctx["settings"]
    sessionmaker = ctx["db_sessionmaker"]
    redis: ArqRedis = ctx["redis"]

    async def has_live_job(attachment_id: UUID) -> bool:
        return await _has_live_ingest_job(redis, attachment_id)

    async with sessionmaker() as session:
        failed = await fail_stalled_attachments(
            session,
            stall=timedelta(minutes=settings.attachment_stall_minutes),
            has_live_job=has_live_job,
        )
    if failed:
        logger.warning(
            "Failed %d attachment(s) left mid-processing with no job to finish them",
            failed,
        )
    return failed


# The names arq registers these jobs under, derived from the functions so the
# enqueue side (the upload route, the delete route) can never drift from them:
# arq keys a job by its function's __name__, and a stale string literal at the
# call site would enqueue a job no worker knows how to run.
JOB_PARSE_AND_INGEST = parse_and_ingest_attachment.__name__
JOB_PURGE_ATTACHMENTS = purge_deleted_attachments.__name__
JOB_GENERATE_TITLE = generate_conversation_title.__name__


def _defer_seconds(exc: Exception, job_try: int) -> int:
    """How long to wait before retrying, given what went wrong and how often."""
    schedule = (
        _MEMORY_RETRY_DEFER_SECONDS
        if isinstance(exc, ParserOutOfMemoryError)
        else _RETRY_DEFER_SECONDS
    )
    return schedule[min(job_try, len(schedule)) - 1]


def _final_reason(exc: Exception) -> str:
    """What to tell the person who uploaded a file whose last attempt just failed."""
    if isinstance(exc, ParserOutOfMemoryError):
        return _OUT_OF_MEMORY_MESSAGE
    return _RETRIES_EXHAUSTED_MESSAGE


async def _mark_exhausted(repo: AttachmentRepository, attachment_id: UUID, *, reason: str) -> None:
    """Record that retries ran out, tolerating a session that is itself broken.

    The row is loaded AFTER the rollback, not carried in. Rolling back discards
    the ingest's half-finished changes (it had already stamped the content hash on
    the row), and the object it was holding comes back EXPIRED - so every attribute
    read on it, down to its tenant, becomes a database round trip that the caller's
    plain attribute access cannot await. That is not a hypothetical: it is what
    every failure past the hash - an embedder outage, a vector store outage, a
    reader killed for memory - would hit, on the one attempt whose whole purpose is
    to leave a row that explains itself. Fetching it again is a real await, and it
    reads the row as the rollback left it.
    """
    try:
        await repo.session.rollback()
        attachment = await repo.get(attachment_id)
        if attachment is None:
            # Deleted while the job was failing. Nobody is waiting on a row that
            # is not there, and the vector purge above has already run.
            return
        await repo.set_status(attachment, ATTACHMENT_FAILED, error=reason)
        await repo.commit()
    except Exception:
        # The database is the thing that is down. The original error is about to
        # be re-raised and is the one worth surfacing; this row stays in
        # `parsing`, and the stalled-upload sweep is what eventually closes it.
        logger.warning("Could not mark attachment %s failed", attachment_id, exc_info=True)


async def startup(ctx: dict[str, Any]) -> None:
    """Build the worker's own singletons (lazy connect; no boot race)."""
    settings = get_settings()
    configure_logging(settings)
    engine = build_engine(settings)
    provider = build_model_provider(settings)

    ctx["settings"] = settings
    ctx["db_engine"] = engine
    ctx["db_sessionmaker"] = build_sessionmaker(engine)
    ctx["vector_store"] = QdrantVectorStore(url=settings.qdrant_url)
    ctx["storage"] = build_storage_backend(settings)
    # The worker is where files are actually parsed, so it gets the OCR-capable
    # registry (settings drive whether OCR runs and how). The API's registry is
    # dispatch-only and needs no OCR.
    ctx["parser_registry"] = default_registry(settings)
    ctx["model_provider"] = provider
    # No embedder is built here, deliberately: each job resolves its own
    # workspace's (see :func:`_job_embedder`). What IS worth doing at boot is
    # saying so when nothing deployment-wide answers, because the ingest path
    # is then one workspace choice away from failing a file at a time, and a
    # line in the log at startup is where an operator will look.
    #
    # No vector store is passed, and that is not an oversight. The evidence
    # rung would put a round trip against Qdrant into worker BOOT, which is
    # exactly when Qdrant is least likely to be answering - and a failed
    # reading is cached like any other, so one slow start would keep that rung
    # unavailable to real jobs for the whole cache window.
    deployment = await resolve_embedding(
        settings=settings, provider=provider, provider_name=settings.model_provider
    )
    if deployment.config is None:
        logger.warning(
            "No embedding model is configured for this deployment. Ingestion resolves one per "
            "job from the workspace default, then from the model the stored passages were "
            "written by, and fails the file when neither answers."
        )


async def shutdown(ctx: dict[str, Any]) -> None:
    """Release every client the worker opened."""
    try:
        for key in ("vector_store", "model_provider", "storage"):
            holder = ctx.get(key)
            aclose = getattr(holder, "aclose", None)
            if callable(aclose):
                await aclose()
    finally:
        # The pool must be disposed even if a client close raises.
        engine = ctx.get("db_engine")
        if engine is not None:
            await engine.dispose()


class WorkerSettings:
    """arq worker configuration (see the module docstring for the run command)."""

    functions = [
        parse_and_ingest_attachment,
        purge_deleted_attachments,
        generate_conversation_title,
    ]
    cron_jobs = [
        cron(
            sweep_expired_vectors,
            minute=_sweep_minutes(
                get_settings().rag_sweep_interval_minutes,
                setting="RAG_SWEEP_INTERVAL_MINUTES",
            ),
            run_at_startup=True,
        ),
        cron(
            collect_attachment_garbage,
            minute=_sweep_minutes(
                get_settings().attachment_gc_interval_minutes,
                setting="ATTACHMENT_GC_INTERVAL_MINUTES",
            ),
            # At startup too: a worker that was down through a conversation
            # delete comes back and cleans up immediately, rather than leaving
            # the files until the next mark on the clock.
            run_at_startup=True,
        ),
        cron(
            reap_stalled_attachments,
            minute=set(range(0, 60, _STALL_CHECK_MINUTES)),
            # No run_at_startup - see the job's docstring; the backlog it would
            # be reaping is the backlog it is about to work off.
        ),
    ]
    on_startup = startup
    on_shutdown = shutdown
    redis_settings = RedisSettings.from_dsn(get_settings().redis_url)
    max_tries = MAX_TRIES
    job_timeout = JOB_TIMEOUT_SECONDS
    # Fresh health-check key every minute so `arq --check` (the compose
    # healthcheck) detects a wedged worker quickly, not after an hour.
    health_check_interval = 60
