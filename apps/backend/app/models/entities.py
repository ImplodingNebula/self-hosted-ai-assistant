"""SQLModel table models - the multi-tenant persistence schema.

Every domain row carries ``tenant_id`` (and usually ``user_id``) from day one
so real multi-tenancy needs no schema change when auth lands. Defaults are
client-side (``uuid4`` / ``datetime.now(UTC)``) rather than server-side SQL so
the same models work on the sqlite test database and on postgres. JSON payload
columns use the generic SQLAlchemy ``JSON`` type for the same reason.

There is deliberately NO model-registry table: what a deployment runs comes
from its own settings, and what is known about a model comes from the runtime's
own report. Two tables sit beside that rule as its signed-off exceptions, both
machine-scoped rather than tenant-scoped because they describe the shared model
server: ``model_probes`` (what a check MEASURED) and ``model_policy`` (what an
operator DECLARED). Neither mirrors an inventory, so neither has a source to
drift from.
"""

from __future__ import annotations

from datetime import UTC, datetime
from enum import StrEnum
from uuid import UUID, uuid4

from sqlalchemy import (
    JSON,
    BigInteger,
    DateTime,
    Index,
    LargeBinary,
    Text,
    UniqueConstraint,
    text,
)
from sqlmodel import Field, SQLModel


def utcnow() -> datetime:
    """Timezone-aware UTC now; the single timestamp default for all tables."""
    return datetime.now(UTC)


class Tenant(SQLModel, table=True):
    """An isolated customer/workspace; the root of every ownership chain."""

    __tablename__ = "tenants"

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    name: str = Field(unique=True)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class User(SQLModel, table=True):
    """A person within a tenant; ``role`` gates the admin surfaces."""

    __tablename__ = "users"
    # The identity a login resolves by. UNIQUE, but only over rows that HAVE
    # one: the seeded pre-auth user carries NULLs and must stay insertable
    # beside every real account. A partial index says exactly that, and it is
    # spelled for both dialects because the test database is sqlite and a
    # constraint that exists in only one of them is a guarantee only one of them
    # has.
    __table_args__ = (
        Index(
            "uq_users_oidc_identity",
            "oidc_issuer",
            "oidc_sub",
            unique=True,
            postgresql_where=text("oidc_issuer IS NOT NULL AND oidc_sub IS NOT NULL"),
            sqlite_where=text("oidc_issuer IS NOT NULL AND oidc_sub IS NOT NULL"),
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(foreign_key="tenants.id", index=True)
    email: str | None = Field(default=None)
    display_name: str
    role: str = Field(default="member")
    # Who vouched for this person and their opaque id there - together the key a
    # login resolves by. NULL on the seeded pre-auth user, which is what keeps it
    # unreachable by any sign-in: it owns the rows created before login existed
    # and is deliberately never resolved to again.
    oidc_issuer: str | None = Field(default=None)
    oidc_sub: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


# Where a conversation's current title came from, which is what decides whether
# anything is still allowed to rewrite it. FALLBACK is the truncation the opening
# turn wrote, GENERATED is the background job's wording, and MANUAL is a name a
# person typed - the only one nothing may overwrite. The distinction earns its
# keep the moment a title can be regenerated more than once (editing a
# conversation's first question re-names it): comparing the stored string against
# the seed cannot tell "still the fallback" from "renamed to the same words", and
# the second must not be quietly replaced.
TITLE_ORIGIN_FALLBACK = "fallback"
TITLE_ORIGIN_GENERATED = "generated"
TITLE_ORIGIN_MANUAL = "manual"
TITLE_ORIGINS = frozenset({TITLE_ORIGIN_FALLBACK, TITLE_ORIGIN_GENERATED, TITLE_ORIGIN_MANUAL})


class Conversation(SQLModel, table=True):
    """A chat thread; ``settings`` holds per-conversation overrides (model, thinking, prompt)."""

    __tablename__ = "conversations"
    # The sidebar query is "this user's conversations, newest activity first" -
    # scoped by tenant AND user since the repository is user-scoped - so the
    # index leads with both equality columns and ends on the ordering column,
    # serving the whole query without walking other users' rows to discard them.
    __table_args__ = (
        Index("ix_conversations_tenant_user_updated", "tenant_id", "user_id", "updated_at"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # No single-column index: the composite index's tenant_id prefix already
    # serves plain tenant lookups.
    tenant_id: UUID
    user_id: UUID
    title: str | None = Field(default=None)
    # Who last named this thread - see TITLE_ORIGINS. A plain string column like
    # every other enum-ish value here (no database enum type, so adding a value
    # later is a code change rather than a migration).
    title_origin: str = Field(default=TITLE_ORIGIN_FALLBACK)
    # When the thread was pinned to the top of the sidebar, or NULL when it is
    # not pinned. A timestamp rather than a boolean because the pinned group is
    # ordered newest pin first, which a flag cannot express; the server stamps
    # it, so a client's clock can never decide that order.
    pinned_at: datetime | None = Field(default=None, sa_type=DateTime(timezone=True))
    settings: dict = Field(default_factory=dict, sa_type=JSON)
    # A turn suspended on a side-effecting tool's confirmation (Phase 5), or NULL
    # when none is pending. It holds the whole checkpoint - the assembled prompt,
    # the loop's counters and citations, and the calls awaiting approval - so the
    # resume request can continue the exact turn without re-running its (costly,
    # non-deterministic) retrieval. Transient: cleared the moment it is resumed or
    # a new turn starts, and it carries its own ``expires_at``. It lives on the
    # conversation, not a table of its own, because a conversation has at most one
    # pending confirmation and the user-scoped conversation repository already
    # makes it unreachable by another user - the guarantee that resume needs.
    pending_confirmation: dict | None = Field(default=None, sa_type=JSON)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))
    updated_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class Message(SQLModel, table=True):
    """One turn in a conversation (user, assistant, or system)."""

    __tablename__ = "messages"
    # History rebuild reads "this conversation's messages in order" - the
    # composite index is that exact access path.
    __table_args__ = (Index("ix_messages_conversation_created", "conversation_id", "created_at"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID
    # The owning user, mirroring the conversation's owner. No index: every read
    # is already keyed by conversation_id (the composite index below), and
    # user_id only filters that narrow per-conversation set. It exists so the
    # message repository is user-scoped in its own right - a defence in depth
    # behind the conversation gate, not a reliance on it.
    user_id: UUID
    conversation_id: UUID = Field(foreign_key="conversations.id", ondelete="CASCADE")
    role: str
    content: str = Field(sa_type=Text)
    thinking: str | None = Field(default=None, sa_type=Text)
    model: str | None = Field(default=None)
    # "meta", not "metadata": SQLAlchemy reserves the latter on declarative classes.
    meta: dict = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class Collection(SQLModel, table=True):
    """A named RAG corpus owned by a tenant (persistent, unlike session scratch)."""

    __tablename__ = "collections"
    # Names address collections on the ingest API (get-or-create by name), so
    # duplicates would make addressing ambiguous.
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_collections_tenant_name"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID = Field(index=True)
    user_id: UUID | None = Field(default=None)
    name: str
    description: str | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


# Ingestion lifecycle. A file is PENDING the moment its bytes are stored, moves
# to PARSING when a worker picks it up, and settles on READY (its text is
# retrievable) or FAILED (``error`` says why, in words a person can act on).
ATTACHMENT_PENDING = "pending"
ATTACHMENT_PARSING = "parsing"
ATTACHMENT_READY = "ready"
ATTACHMENT_FAILED = "failed"
# Written by the retention sweep (``attachment_gc.expire_aged_attachments``) and
# by nothing else. It is a settled status like READY and FAILED, not a stage: the
# file's bytes and indexed text are gone and the row is kept deliberately, so the
# interface can say a file was here and has aged out rather than losing the
# record. A row never leaves it - re-uploading the same bytes stores them fresh.
ATTACHMENT_EXPIRED = "expired"
ATTACHMENT_STATUSES = frozenset(
    {
        ATTACHMENT_PENDING,
        ATTACHMENT_PARSING,
        ATTACHMENT_READY,
        ATTACHMENT_FAILED,
        ATTACHMENT_EXPIRED,
    }
)

# How the bytes arrived. A pasted block of text is a real, stored file that
# travels the identical pipeline; only its label differs, so a citation can read
# "Pasted content" rather than a synthetic filename nobody chose.
#
# GENERATED is the one origin no client may claim: the bytes came from a tool the
# assistant ran, not from anything a person sent, and the difference decides
# whether the file is offered back to the composer as a draft. The upload
# endpoint therefore accepts only the two a person can actually perform.
ATTACHMENT_ORIGIN_UPLOAD = "upload"
ATTACHMENT_ORIGIN_PASTE = "paste"
ATTACHMENT_ORIGIN_GENERATED = "generated"
ATTACHMENT_ORIGINS = frozenset(
    {ATTACHMENT_ORIGIN_UPLOAD, ATTACHMENT_ORIGIN_PASTE, ATTACHMENT_ORIGIN_GENERATED}
)


class Attachment(SQLModel, table=True):
    """An uploaded file; bytes live in object storage under ``storage_key``."""

    __tablename__ = "attachments"
    __table_args__ = (
        # Every read is "the attachments of this conversation, for this tenant":
        # the composer lists them, the chat turn resolves the ids it references,
        # and conversation delete collects the blobs to remove.
        Index("ix_attachments_tenant_conversation", "tenant_id", "conversation_id"),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    # No single-column index: the composite index's tenant_id prefix already
    # serves plain tenant lookups, and a second index would only add write cost.
    tenant_id: UUID
    user_id: UUID
    # SET NULL, not CASCADE: the row must outlive the conversation, because the
    # row is the only thing that knows where the bytes are. The NULL is a
    # tombstone - it is exactly what the attachment garbage collector looks for,
    # and it deletes the blob before finally dropping the row.
    conversation_id: UUID | None = Field(
        default=None, foreign_key="conversations.id", ondelete="SET NULL"
    )
    filename: str
    content_type: str
    size_bytes: int
    storage_key: str
    status: str = Field(default=ATTACHMENT_PENDING)
    # Why parsing failed, in words worth showing. Null unless status is failed.
    error: str | None = Field(default=None, sa_type=Text)
    # sha256 of the stored bytes. Two uploads of the same file parse and embed
    # once; it also lets a retried job recognise work it already finished.
    content_hash: str | None = Field(default=None, index=True)
    origin: str = Field(default=ATTACHMENT_ORIGIN_UPLOAD)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))
    # Stamped on every status change, so a job that died mid-parse is visible as
    # a row stuck in `parsing` rather than one indistinguishable from a new one.
    updated_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class WorkspaceSettings(SQLModel, table=True):
    """Tenant-wide defaults (one row per tenant, keyed by the tenant itself)."""

    __tablename__ = "workspace_settings"

    tenant_id: UUID = Field(primary_key=True, foreign_key="tenants.id")
    settings: dict = Field(default_factory=dict, sa_type=JSON)
    updated_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class McpCredentialScope(StrEnum):
    """Which credential a server's tool CALLS carry.

    Mirrors :class:`~app.providers.secrets.base.SecretScope` (that enum lives a
    layer up, so it is not imported here). Stored as the plain string value, the
    same way attachment statuses are.

    - ``WORKSPACE``: the shared, workspace-level header (today's behaviour) - one
      credential for every user of the tenant.
    - ``USER``: the acting user's own stored secret, injected per call, and
      FAILING CLOSED when the user has connected none - never falling back to the
      shared header (unlike the env-seeded PR1 path).
    """

    WORKSPACE = "workspace"
    USER = "user"


class McpAuthType(StrEnum):
    """HOW a server authenticates, where ``McpCredentialScope`` says WHOSE.

    Stored as the plain string value, like the scope beside it.

    - ``HEADER``: a static bearer credential (today's only mechanism) - pasted
      through the write-only credential endpoint, or the env bootstrap for the
      seeded chassis.
    - ``NONE``: the upstream takes no credential at all; nothing is attached to
      its requests even when a stored or env credential exists.
    - ``OAUTH``: calls carry the access token from the server's sealed OAuth
      grant (the consent flow populates it); no grant connected means no
      header, never a fallback to a pasted credential.
    """

    NONE = "none"
    HEADER = "header"
    OAUTH = "oauth"


class McpServer(SQLModel, table=True):
    """A registered MCP sidecar reachable over Streamable HTTP, per tenant."""

    __tablename__ = "mcp_servers"
    __table_args__ = (UniqueConstraint("tenant_id", "name", name="uq_mcp_servers_tenant_name"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID
    name: str
    base_url: str
    enabled: bool = Field(default=True)
    # Which credential this server's tool CALLS carry - a ``McpCredentialScope``
    # value. Client-side default (like every default here), so an existing row
    # and a new one both keep the shared workspace header until an operator opts
    # a server into per-user credentials.
    credential_scope: str = Field(default=McpCredentialScope.WORKSPACE.value)
    # How this server authenticates - a ``McpAuthType`` value. Defaults to
    # ``header`` (the pre-column behaviour) so existing rows and unmodified
    # clients change nothing.
    auth_type: str = Field(default=McpAuthType.HEADER.value)
    config: dict = Field(default_factory=dict, sa_type=JSON)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class OAuthAuthorization(SQLModel, table=True):
    """A pending OAuth consent - the state between "Connect" and the callback.

    Written when an admin starts a connect, consumed (deleted) exactly once by
    the callback that carries its ``state`` value - single use is what blocks
    an authorization code being injected into someone else's flow. Carries
    everything the callback needs to finish without re-running discovery: the
    endpoint snapshot, the redirect URI (the token exchange must repeat it
    verbatim), and the PKCE verifier SEALED with the secret cipher - a
    plaintext verifier plus an intercepted code would complete the exchange.
    Short-lived by design: rows expire and are swept opportunistically.
    """

    __tablename__ = "oauth_authorizations"
    # State is the lookup key the callback arrives with; global uniqueness is
    # what makes consume-by-state unambiguous (values are 32 random bytes, so
    # collisions do not occur in practice - the constraint is the backstop).
    __table_args__ = (UniqueConstraint("state", name="uq_oauth_authorizations_state"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID
    mcp_server_id: UUID
    state: str
    code_verifier_sealed: bytes = Field(sa_type=LargeBinary)
    client_id: str
    redirect_uri: str
    # The canonical RFC 8707 resource for an MCP-native flow; NULL on the
    # operator-supplied path (generic providers do not take the parameter).
    resource: str | None = Field(default=None)
    # Issuer / token / revocation endpoints + send_resource, snapshotted at
    # start so the callback and the sealed bundle agree on where tokens live.
    endpoints: dict = Field(default_factory=dict, sa_type=JSON)
    # Space-joined requested scopes; the bundle's granted_scope fallback when
    # the token response omits its scope (RFC 6749: absent means as-requested).
    scopes: str
    created_by: UUID
    expires_at: datetime = Field(sa_type=DateTime(timezone=True))
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class Secret(SQLModel, table=True):
    """An encrypted upstream credential; plaintext is never stored or returned.

    ``ciphertext`` is AES-GCM output from the ``SecretStore`` seam; ``last4``
    is the only readable hint the API ever exposes. A secret is either
    workspace-shared (``user_id`` NULL) or private to one user (``user_id`` set)
    - the two are different rows and never collide, so members can each connect
    their own account under the same name.
    """

    __tablename__ = "secrets"
    __table_args__ = (
        # Unique per (tenant, owner, name). ``postgresql_nulls_not_distinct``
        # makes it hold for workspace secrets too, whose owner is NULL - without
        # it Postgres treats two NULL owners as distinct and would let a
        # workspace name duplicate. The store upserts on this key, so a duplicate
        # normally never reaches the constraint; under a concurrent double-write
        # it is the backstop, and the store folds the loser into an update.
        UniqueConstraint(
            "tenant_id",
            "user_id",
            "name",
            name="uq_secrets_tenant_user_name",
            postgresql_nulls_not_distinct=True,
        ),
    )

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    tenant_id: UUID
    # NULL for a workspace-shared secret; the owning user for a per-user one.
    user_id: UUID | None = Field(default=None)
    name: str
    ciphertext: bytes = Field(sa_type=LargeBinary)
    last4: str
    # Which master key sealed this row, woven into the AES-GCM associated data.
    # A future key rotation bumps it; today every row is version 1.
    key_version: int = Field(default=1)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))
    updated_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class ModelProbe(SQLModel, table=True):
    """One tool-check harness run: a measurement bound to weights and runtime.

    Deliberately breaks two locked rules, both owner-approved (2026-07-15).
    It is MACHINE-scoped, not tenant-scoped: a probe row describes the shared
    model server, so the triggering tenant/user are audit-only columns, never
    filters. And it is NOT a model registry: a measurement has no source to
    drift from and costs GPU minutes to recompute, so rows are appended and
    kept as history - what passed on the previous runtime and fails on this
    one IS the data. Invalidation is structural: a re-pull changes ``digest``
    (old verdicts no longer describe the installed weights), a runtime upgrade
    changes ``runtime_version`` (stale, not deleted), and ``harness_version``
    fingerprints the case specs themselves.
    """

    __tablename__ = "model_probes"
    # The one read path: "the latest verdict for these weights on this
    # runtime", newest first.
    __table_args__ = (Index("ix_model_probes_lookup", "provider", "model", "digest", "created_at"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    provider: str
    model: str
    digest: str | None = Field(default=None)
    runtime_version: str | None = Field(default=None)
    # WHERE the run happened: scheme, host and port of the serving backend, with
    # credentials, path and query stripped (`serving_endpoint_of`). Throughput is
    # the host's hardware and tool behaviour is a (model, runtime) property of
    # the machine serving it, so a verdict must not migrate between servers - a
    # pod's certificate cannot arm a laptop. NULL on rows written before this
    # column, and on a backend that names no host; either way such a row
    # certifies nothing, while its FAILURES still clamp.
    serving_endpoint: str | None = Field(default=None)
    harness_version: str
    depth: str
    earned_agency: str
    thinking_with_tools: bool = Field(default=False)
    tokens_per_second: float | None = Field(default=None)
    # BigInteger: VRAM in bytes overflows int32 for any model past ~2GB.
    vram_bytes: int | None = Field(default=None, sa_type=BigInteger)
    # Per-case tallies and sample details, as the harness reported them.
    cases: dict = Field(default_factory=dict, sa_type=JSON)
    # Audit-only (see the class docstring): who triggered the run, never a scope.
    triggered_by_tenant_id: UUID | None = Field(default=None)
    triggered_by_user_id: UUID | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))


class ModelPolicy(SQLModel, table=True):
    """One operator declaration about a model: the axes it sets, and no others.

    The SECOND sanctioned exception to tenant scoping, on exactly the pattern
    :class:`ModelProbe` established and for the same reason: a declaration
    describes the SHARED model server. One workspace cannot decide that a set
    of weights reasons differently than another workspace sees them reason, so
    scoping these rows to a tenant would shard one machine's policy into
    per-tenant answers that contradict each other about one model. The
    declaring tenant and user ride along as audit columns - who said this, and
    when - and nothing filters on them.

    APPEND-ONLY, like the measurement log, and for a sharper reason. Each row
    declares only the axes that action set; every other column stays NULL, and
    the effective policy is FOLDED per axis from the recent rows (see
    :mod:`app.services.model_policy`). Copying prior axes forward into each new
    row - the obvious "merge on write" - is the one implementation that must
    not exist: it would carry a ceiling declared against old weights onto a new
    digest, silently re-arming a grant that a re-pull is supposed to retire.

    The two kinds of axis are not the same kind of statement, which is why the
    fold treats them differently:

    - TRUST axes (``ceiling``, ``thinking_with_tools``, ``think_mode``) say
      what these WEIGHTS may do, so they are honoured only while the declaring
      row's ``digest`` is still what is installed. A re-pull retires them with
      no code doing anything - the same structural invalidation that makes a
      probe verdict evaporate.
    - PREFERENCE axes (``num_ctx``, ``thinking_model``) say how this
      DEPLOYMENT wants the tag run. They follow the tag across a re-pull,
      because "give this model a 16k window" is not a claim about weights.
    """

    __tablename__ = "model_policy"
    # The one read path: "the recent declarations for this tag", newest first.
    __table_args__ = (Index("ix_model_policy_lookup", "provider", "model", "created_at"),)

    id: UUID = Field(default_factory=uuid4, primary_key=True)
    provider: str
    # The INSTALLED spelling, as the provider lists it - see
    # app.services.model_identity for why one matcher answers both sides.
    model: str
    # The digest installed when the declaration was made; the binding the trust
    # axes above are honoured against.
    digest: str | None = Field(default=None)
    ceiling: str | None = Field(default=None)
    thinking_with_tools: bool | None = Field(default=None)
    # never | hybrid | always, for a measured think-mode confirmation.
    think_mode: str | None = Field(default=None)
    num_ctx: int | None = Field(default=None)
    # A sibling tag override: which model serves this one's thinking turns.
    thinking_model: str | None = Field(default=None)
    # Audit-only (see the class docstring): who declared this, never a scope.
    declared_by_tenant_id: UUID | None = Field(default=None)
    declared_by_user_id: UUID | None = Field(default=None)
    # The model_probes row a declaration rested on, when a check backed it.
    probe_id: UUID | None = Field(default=None)
    created_at: datetime = Field(default_factory=utcnow, sa_type=DateTime(timezone=True))
