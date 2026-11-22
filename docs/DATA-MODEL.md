# Data Model

This document explains every table in the relational database: what each one stores, and - more importantly -
why it is shaped the way it is. The authoritative definitions live in code at
[`apps/backend/app/models/entities.py`](../apps/backend/app/models/entities.py) (SQLModel classes) and ship as
Alembic migrations (versioned, scripted schema updates) under
[`apps/backend/alembic/versions/`](../apps/backend/alembic/versions/). This page is the guided tour; when the
two ever disagree, the code wins and this page has a bug.

For how the data layer fits into the wider system (repositories, seams, request flow), see
[ARCHITECTURE.md](ARCHITECTURE.md) section 10. This page zooms in one level: the tables themselves.

A few terms used throughout, explained once:

- **Primary key (PK)** - the column that uniquely identifies a row. Here they are UUIDs (random 128-bit ids)
  rather than incrementing numbers, so ids are generated app-side, work offline in tests, and never leak "how
  many rows exist" or allow guessing a neighbouring record.
- **Foreign key (FK)** - a column that points at another table's primary key, with the database enforcing that
  the target exists and deciding what happens when it is deleted (see the cascade notes below).
- **Index** - a lookup structure the database maintains so a specific query pattern is fast. Indexes are not
  free (every write updates them), so each one here exists for a named query.
- **JSON column** - a column holding a free-form JSON object. Used where the shape is expected to evolve
  (settings, metadata), because adding keys to JSON needs no migration.

## The shape at a glance

Everything hangs off a tenant - an isolated workspace, think "one company using the platform":

```
tenants
  ├── users                     (people in the workspace; role: admin | member)
  ├── conversations             (chat threads; per-thread settings JSON)
  │     ├── messages            (turns: user / assistant / system)
  │     └── attachments         (uploaded files; bytes live in object storage)
  ├── collections               (named persistent RAG corpora)
  ├── workspace_settings        (one row of tenant-wide defaults)
  ├── mcp_servers               (registered tool servers; how each authenticates)
  │     └── oauth_authorizations (a consent in flight, consumed once by its callback)
  └── secrets                   (encrypted credentials and sealed OAuth grants)

model_probes                    (stands apart: measurements of the shared model
                                 server, so machine-scoped rather than per tenant)
model_policy                    (stands apart for the same reason: declarations
                                 ABOUT that shared server, made by an operator)
```

## Rules that shape every table

These five decisions repeat across the schema, so they are explained once:

1. **Every domain row carries `tenant_id` (and usually `user_id`).** The schema was multi-tenant from day one
   even while a single default tenant and user were seeded at boot, so that real login could arrive without
   retrofitting tenancy onto a live schema - one of the most painful migrations there is. That bet paid off only
   in part: when login landed, it replaced one dependency and started filling `tenant_id`/`user_id` from a
   verified session with no domain table changed, migrated, or backfilled - but the `users` table still needed
   two new columns (`oidc_issuer`, `oidc_sub`) plus a partial unique index, because nothing in the original
   schema said which provider had vouched for a person (see [`users`](#users)). The lesson generalizes: an
   OWNERSHIP column is free to add early, since its value is known the moment a row is written; an IDENTITY
   column cannot be, since its value does not exist until an identity provider is chosen.
2. **Scoping is enforced in the repository layer, not by per-column foreign keys.** Not every `tenant_id`
   column carries a database-level FK; instead, all data access goes through a base repository that injects
   `tenant_id` into every query, so a single forgotten filter cannot leak another tenant's data. Rows a
   per-user credential (or a per-user tool result) can reach - conversations, their messages, and attachments -
   go a step further through a **user-scoped** repository that injects the acting `user_id` alongside the tenant,
   so two users of one tenant are separated too. The acting user comes from the request context and is never a
   method argument, so naming another user's id is not expressible. FKs are declared where they carry real
   behavior - the delete rules on `messages` and `attachments` below.

   There is exactly one carve-out, and it is deliberate. Deleting a model's weights is a workspace-wide act -
   one delete removes them for everybody - so the guard that refuses to delete a model some conversation is
   still set to use has to look past the acting user. `ConversationRepository.pinned_model_counts` is
   therefore scoped to the tenant alone. Two things bound it: it returns **tag counts and nothing else**, so
   no other user's title, message, or setting can leave through it, and it is the only method on that
   repository that widens the scope, named so that reads plainly at the call site. The alternative was worse -
   a per-user count would report zero and let one admin quietly break every other user's threads.
3. **Defaults are computed in the app, not in SQL.** Ids (`uuid4`) and timestamps (timezone-aware UTC) are
   generated client-side rather than by database defaults, so the exact same models run against the in-memory
   SQLite database in tests and Postgres in production. Dialect-specific SQL would fork the two.
4. **Flexible payloads are JSON, strict payloads are columns.** Settings and metadata evolve phase by phase;
   making them JSON means new keys ship without a migration. Anything queried or constrained (ids, roles,
   timestamps, names) stays a real column.
5. **A settings JSON column reads leniently and writes strictly.** `conversations.settings` and
   `workspace_settings.settings` both take this shape: a read runs the stored blob through a typed view that
   drops a malformed or unknown field rather than failing the turn, so a row written by another code version
   stays usable. A write goes the other way - the PATCH endpoint merges PER FIELD under a row lock (a field
   absent from the request is left as stored, an explicit `null` clears it, an unknown key is rejected), so two
   controls saved at once serialize instead of clobbering each other's axis. Each table's own section below
   notes only what is specific to it - which keys it holds, and any extra validation those keys need.

## The tables

### `tenants`

The root of every ownership chain: one row per isolated workspace.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `name` | unique |
| `created_at` | |

Nothing references upward past a tenant; deleting one (a future admin operation) is the natural place to fan
out cleanup.

### `users`

A person within a tenant.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id` (FK -> tenants) | indexed |
| `email` | nullable - NULL on the seeded pre-login user; filled from verified claims at every sign-in |
| `display_name` | |
| `role` | `admin` or `member`, default `member` |
| `oidc_issuer` | nullable - the identity provider that vouched for this person |
| `oidc_sub` | nullable - that provider's stable, opaque id for them |
| `created_at` | |

**`(oidc_issuer, oidc_sub)` is the identity a sign-in resolves by**, under a UNIQUE index that is PARTIAL -
it covers only rows where both are non-NULL. Two decisions are packed into that:

- **Keyed on the subject claim, not the email.** A provider's `sub` is the claim the specification promises
  stays stable for the life of an account, while an address can be renamed, released, and reassigned to
  somebody else. Keying on the address would eventually hand one person another person's history. The email is
  still stored, and refreshed from the claims at every sign-in, but it is a label here rather than the key.
- **Partial, so the pre-login user still fits.** The seeded default user carries NULLs in both columns and has
  to remain insertable beside every real account. That is also what makes it unreachable: no sign-in can ever
  resolve to a row with no identity, so it stays the owner of everything created before login existed.

**The consequence of that last point is deliberate and visible:** turning login on does not hand the first
person who signs in the conversations, attachments and credentials created beforehand. Those rows keep
pointing at the seeded user, which now nobody signs in as. A deployment that wants them carried over has to
re-create them; the alternative - letting a first sign-in inherit the pre-login principal - would be privilege
escalation wearing a convenience hat, since whoever signed in first would also inherit that user's stored
integration credentials.

The `role` column predates anything reading it seriously, and a configured login is what makes it real:
`require_role("admin")` now genuinely refuses a member on the admin surfaces. An address listed in the
deployment's bootstrap admin setting is floored to `admin` at every sign-in, written through to this column,
so a workspace cannot demote its way out of having an administrator. The seeded default user is an `admin` for
the same reason - with no login configured every caller is that user, and the gate must not lock out the only
one there is.

### `conversations`

A chat thread, as shown in the sidebar.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id`, `user_id` | scoped by the repository layer |
| `title` | nullable; truncated from the first message on the turn, then rewritten by a background job |
| `title_origin` | who last named the thread: `fallback`, `generated`, or `manual` |
| `pinned_at` | nullable - when the thread was pinned to the top of the sidebar; NULL means unpinned |
| `settings` | JSON - per-conversation overrides (model, thinking mode, prompt) |
| `pending_confirmation` | JSON, nullable - a turn suspended on a side-effecting tool's confirmation, or NULL |
| `created_at`, `updated_at` | `updated_at` bumps on activity |

**Why it's shaped this way**

- **One composite index, `(tenant_id, user_id, updated_at)`.** The sidebar's only query is "this user's
  conversations, pinned first then newest activity" - the repository scopes by tenant and user, so the index
  leads with both equality columns and ends on the ordering column. The pinned prefix is sorted on top of that
  rather than by the index (see `pinned_at` below); one person's conversations are a small set, so a second
  index for that sort is not worth it. No standalone `tenant_id` index either: the composite already serves
  lookups on its leading column, and a second index would only slow writes.
- **`title` is written twice, deliberately.** The turn that opens a conversation writes a truncation of its
  first message immediately, so the sidebar is never blank while the reply streams. Once that exchange
  finishes, a background job asks the chat model for a few words describing the thread and writes them over the
  truncation - unless the column changed meanwhile, meaning a person renamed it and their name wins. That write
  leaves `updated_at` alone: naming a conversation is not activity in it, and bumping the column would reorder
  the sidebar under the reader.
- **`title_origin` records WHO chose the current name**, which decides whether anything may still change it:
  `fallback` (the opening truncation), `generated` (the background job), or `manual` (a typed rename, the only
  one nothing overrides). It exists because naming can happen more than once - rewording a conversation's
  opening question asks for a fresh name - and comparing the stored title against that new seed cannot tell
  "still the fallback" from "renamed to those same words"; provenance answers the actual question. Existing
  rows default to `fallback`, so a thread renamed by hand before the column existed reads as machine-named and
  may be re-titled once if its opening question is later edited - a one-shot cosmetic cost, cheaper than
  inventing provenance nothing recorded.
- **`pinned_at` is a timestamp, not a boolean, because pinning has an order:** the sidebar groups pinned threads
  above the recent list, newest pin first, which a flag cannot express. The client sends only its intent
  (`{"pinned": true/false}`) and the server stamps the clock, so a skewed or hostile client clock can never
  place a row in that group. Pinning an already-pinned thread is a no-op (restamping would jump it to the head
  for a request asking for nothing to change), nothing caps how many threads may be pinned, and the column is
  nullable with no backfill since an absent value already means "not pinned".
- **Pinning writes without bumping `updated_at`**, the same reasoning as a settings change: it is metadata about
  a thread, not activity in it. Because a pin is not activity, the list query has to carry it explicitly - rows
  order `pinned_at DESC NULLS LAST, updated_at DESC`, so pinned threads always lead the page (ordering by
  activity alone could let an old pinned thread fall off a page-limited sidebar entirely - the exact
  disappearance pinning exists to prevent). `NULLS LAST` is spelled out because the two dialects disagree on
  where a descending sort puts a NULL. No index backs the column: the composite index already narrows to one
  person's conversations, and sorting that handful by pin costs less than an index would cost every write.
- **`settings`** is the conversation-scoped half of the two-scope settings design (the workspace-scoped half is
  [`workspace_settings`](#workspace_settings)): a thread can pin a model, a thinking mode, or a web-search mode
  without touching any other thread. It follows the lenient-read / strict-write pattern common to every
  settings column (see [Rules that shape every table](#rules-that-shape-every-table), item 5).
- **`pending_confirmation` holds a turn stopped to ask before running a side-effecting tool** (Phase 5). When
  the model calls such a tool, the loop writes the whole suspended turn here - the assembled prompt, the loop's
  counters and cited sources, and the call awaiting approval - so a resume can continue the exact turn without
  re-running its retrieval. It rides the conversation rather than a table of its own because a conversation has
  at most one pending confirmation, and the user-scoped conversation repository already makes it unreachable by
  another user - the guarantee resume needs. It is transient, cleared the moment it is resumed or a new turn
  starts, and carries its own expiry so a forgotten prompt does not linger. It deliberately does NOT hold image
  pixels: base64 of a vision turn's pictures would be re-read in full on every later turn and locked for update
  on resume, running to megabytes, and would leave a durable image copy outside the object store's lifecycle
  where deleting the attachment could never reach it. The checkpoint records attachment ids instead, and resume
  re-reads those files - an honest trade: a file deleted inside the confirmation window cannot be re-read, and
  the resumed turn says which picture it lost rather than answering as though it had seen it, since a resume is
  never failed over an image once the user has already decided about a side-effecting action.

### `messages`

One turn in a conversation - user, assistant, or system.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id`, `user_id` | scoped by the repository layer |
| `conversation_id` (FK -> conversations) | deleting the conversation deletes its messages (CASCADE) |
| `role` | `user` / `assistant` / `system` |
| `content` | full text |
| `thinking` | nullable - the reasoning stream, when a thinking model produced one |
| `model` | nullable - which model generated an assistant turn |
| `meta` | JSON - e.g. flags an interrupted generation, the passage a question was replying to, or what a finished reply took to write |
| `created_at` | |

**Why it's shaped this way**

- **Composite index `(conversation_id, created_at)`** serves the hottest read in the system - the server-side
  history rebuild, "this conversation's messages, in order" - on every chat request. `user_id` gets no index of
  its own: every read is already keyed by `conversation_id`, so `user_id` only filters that narrow,
  already-fetched set. It mirrors the owning conversation's user and exists so the message repository is
  user-scoped in its own right - defence in depth behind the conversation gate, returning nothing for a user
  who does not own it even if a caller ever reaches a message without first resolving its conversation.
- **`thinking` and `model` are stored per message** so a reply stays auditable even as a thread mixes models and
  reasoning modes over its lifetime - each turn records what actually produced it. The column is `meta`, not
  `metadata`, only because SQLAlchemy reserves the latter on model classes.
- **Messages CASCADE on conversation delete** (contrast [`attachments`](#attachments), which survive) because a
  message has no meaning outside its thread.
- **Three writes DELETE rather than append**, for the two ways a thread can be rewound - answering a question
  again, and rewording it - and all three are scoped to the acting tenant and user like every other write, so
  one user cannot truncate or empty another's thread by naming its ids. A reply is only ever written while the
  question it answers still exists, which stops a stream in flight during a delete from landing an orphan.
  - **Regenerating the newest answer** drops the reply that question already has, kept narrow by four
    properties: it stops at the next question rather than filtering for replies (the run of `assistant` rows
    right after the anchor goes; the first row of any other role ends it - filtering the tail instead could
    sail past a later question and delete the answer to *that* one); the anchor is located by POSITION and the
    delete names explicit ids (the ordered thread is read, the question found in it by id, the run after it
    deleted by an id list, so the delete can never reach further than the read actually saw - a `created_at >`
    predicate could sweep up a row arriving between read and write); it runs in the same transaction that
    writes the replacement (a regenerate that produces nothing leaves the previous answer standing, the correct
    retry behavior; a turn paused on a side-effecting confirmation carries the pending swap in its checkpoint,
    so the resume still replaces rather than appends); and zero rows deleted is a normal outcome, since an
    interrupted turn that saved no text leaves none to delete.
  - **Regenerating an OLDER answer** cannot use that sweep - it discards a whole tail, since every turn below
    the answer being replaced was written in reply to something about to change. The rows are named while the
    user is still deciding (the confirmation dialog counts exactly them), carried through the turn, and deleted
    as that captured list in the transaction that writes the replacement - so a turn another tab appends
    meanwhile was never on the list and survives. A suspended turn carries the list in its checkpoint as a LIST
    rather than recomputing it on resume, since recomputing would sweep up whatever arrived during the
    confirmation window.
  - **Editing a question** is the widest of the three and the only one that runs BEFORE generation: it rewrites
    the row's text and removes every message after it, whatever the role, since those turns answered wording
    that no longer exists. The asymmetry with a regenerate is deliberate - a retry holds on to what it is
    retrying, an edit cannot, because the thing being retried is the question itself.
- **`meta.quote`** stores the passage verbatim (with the id it was taken from) when a question replies to a
  quoted passage. The text is authoritative and never re-read from the quoted row, which is what lets a quote
  outlive the message it came from being edited away or deleted; the id is provenance only.

### `attachments`

A file the conversation holds - one a person sent in, or one the assistant's own tools produced. Only
bookkeeping lives here; the actual bytes go to object storage (local disk in dev, S3-compatible storage in the
cloud) under `storage_key`.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id`, `user_id` | scoped by the repository layer |
| `conversation_id` (FK -> conversations) | nullable; SET NULL on conversation delete |
| `filename`, `content_type`, `size_bytes` | what the file is called, is, and weighs |
| `storage_key` | where the bytes live in object storage |
| `status` | lifecycle: `pending` -> `parsing` -> `ready` or `failed`; a `ready` file may later become `expired` (see below) |
| `error` | why parsing failed, in words worth showing; null unless `failed` |
| `content_hash` | sha256 of the stored bytes (indexed); null for a generated file (see below) |
| `origin` | `upload`, `paste`, or `generated` - where the bytes came from (see below) |
| `created_at` | |
| `updated_at` | stamped on every status change |

**Why it's shaped this way**

- `status` and `origin` are plain string columns, not database enums, so the schema stays portable across
  sqlite (tests) and postgres (production) and adding a value never needs a migration - the repository
  validates both against a known set on the way in, since nothing else would reject a typo.
- **A generated row (`origin = generated`) differs from an upload in three deliberate ways** - though it is
  created the same way an upload is: the turn stores the bytes and commits the row *before* announcing the file
  to the client, then records the same reference on the assistant message it saves. It is born
  `ready` - there is no parse job and no retrieval ingest, because the file is produced *for the reader*, not
  folded back into what the model can search. Its `content_hash` is null, so it can never match the
  upload-dedupe lookup and someone uploading the same bytes always gets their own row rather than the
  assistant's. And it is filtered out of the conversation-detail read's top-level attachments list (which
  exists to restore the composer's draft tray) - nobody chose that file, so it is not a draft - while still
  appearing on the message that produced it, with `origin` exposed everywhere a file reference appears so a
  client can tell the two kinds apart.
- **Retention (the `expired` status) is a policy, not a repair - the only cleanup path that acts on files that
  worked exactly as intended.** A periodic sweep is the sole writer. Its two tiers default oppositely:
  generated files expire after `GENERATED_FILE_RETENTION_DAYS` (30) since nobody chose to keep them, and that
  window is also what finally collects a generated row whose turn persisted no message - the one row both
  delete sweeps below would otherwise miss; uploads and pastes do not expire unless an operator sets
  `ATTACHMENT_RETENTION_DAYS` (unset by default, since a person chose to send those) - zero reads as off in
  both tiers, so a mistyped digit cannot mean "expire everything now". Age is measured from `created_at`; only
  `ready`/`failed` rows are candidates (a `pending`/`parsing` row belongs to the job still working on it), and
  a row whose conversation is already gone is left for the tombstone purge below.
- **Expiring removes content but keeps the row, on purpose.** The sweep drops the file's indexed text, then its
  bytes, then commits `expired` last - filename, size, type and conversation link all survive, so a transcript
  can go on saying a document was there and has aged out rather than losing the record. Committing last is
  what makes the sweep safe to interrupt from that direction: `expired` is the only thing that removes a row
  from the sweep's selection, so nothing already-emptied is mistaken for intact. Interrupted the other way has
  a real, bounded cost: the two deletes are not transactional with the row, so a failure between them rolls the
  row back to `ready` with its indexed text already gone - it stays listed and grounds nothing until the next
  pass finishes it. That ordering (text, then bytes, then status) is the least bad of the three: dropping text
  last would leave an `expired` row still quotable, and committing status first would strand the bytes forever.
  Each row commits as it finishes, not a batch at a time, to keep that window to the one row a failure
  interrupted. A database failure at that commit is handled like a store outage: the pass stops, logs, and
  trusts the next tick to retry.
- **Two consequences follow expiry:** an expired file is no longer retrievable or citable (its passages went
  with its bytes), and it is excluded from the duplicate check exactly like a `failed` row, so re-uploading the
  same bytes after they age out stores them fresh. Storage GC needs no special case for generated blobs either
  - they use the same `{tenant}/{conversation}/{attachment}` key an upload mints, so a generated file whose row
  never committed is reclaimed by the same orphan sweep described next.
- **Indexes:** a composite `(tenant_id, conversation_id)` serves the one real read pattern - "this
  conversation's attachments, for this tenant" (composer list, chat-turn resolution, or a conversation delete
  collecting blobs). No standalone `tenant_id` index, for the same reason `conversations` has none: the
  composite already serves a tenant-only lookup, and a second index would only cost writes. `content_hash` is
  indexed but **not unique** - the same bytes may legitimately sit in two threads, each with its own ephemeral
  vectors on its own clock.
- **Attachments survive conversation deletion** (the FK is `SET NULL`, unlike messages' `CASCADE`), because
  deleting a thread should not silently destroy the record of an uploaded file - the row is the anchor for
  cleaning up the stored bytes, and an orphaned blob with no row would leak storage forever. A `conversation_id`
  of NULL makes the row a **tombstone**: it exists only to say where the file's remains are. Two paths reclaim
  the same three things in the same order - indexed text, then bytes, then the row itself (text first because
  it is what can still answer questions; the row last because it is the only record of where the bytes are):
  - **Immediately.** Deleting a conversation reads its attachment ids first (afterwards the NULL has erased the
    link) and queues a job to remove all three, so a deleted thread's files stop occupying storage and stop
    being retrievable within seconds. The job removes the file's passages even though dropping the conversation
    already drops its whole ephemeral scope, because that drop is best-effort (the vector store may have been
    unreachable) and an ingest still running can write passages in just behind it - an attachment's passages
    carry no expiry of their own (they must outlive the conversation's two-hour ephemeral scope), so nothing
    else would ever catch either case.
  - **Eventually.** A periodic sweep collects tombstones whose job never ran (queue down, worker restarting) and
    separately walks object storage for blobs no row references at all. That walk only considers keys matching
    the app's own `{tenant}/{conversation}/{attachment}` shape, skips anything modified within a grace window
    (an upload mid-write looks like garbage for the moment before its commit), and deletes nothing while the
    database is unreachable, since every key would then look unreferenced.
- **`updated_at` makes a lifecycle stall visible.** A worker that dies mid-parse leaves a row stuck in
  `parsing`, indistinguishable by `created_at` alone from one just uploaded. A transient failure (storage,
  embedder, or Qdrant briefly down) retries a few times with backoff and settles on `ready` if a later attempt
  succeeds; only exhausted retries produce `failed` with a reason. A terminal failure (unreadable, unsupported,
  or empty) skips retries and fails immediately, since retrying the same bytes cannot change the outcome. One
  row nothing above would ever close: the one whose job never ran at all (the queue was unreachable at upload,
  or its state was later lost), which would otherwise poll `pending` forever. A periodic sweep
  (`ATTACHMENT_STALL_MINUTES`, default 30) is the second writer of `status`/`error`: rows untouched that long
  are marked `failed` with a reason to re-upload - safe, since the duplicate check skips failed rows - but only
  after asking the queue whether it still holds a job for that row, since age alone can just mean "tenth in
  line": failing one still queued would leave a re-upload and the original both landing, doubling the file. The
  retention sweep is a third writer of `status` (the only one that writes `expired`) and never touches `error`:
  nothing failed, so there is no fault to explain.
- **A failed file is still served.** `failed` records that ingestion failed - the platform could not read the
  document - and says nothing about the bytes, which were written before the row ever committed and are
  removed by no failure path. A `failed` upload or paste therefore stays downloadable even though nothing was
  drawn from it; withholding it would be the platform keeping a file it could not even use. The one status
  combination refused is a `generated` row reaching `failed`: a file the assistant produced exists only because
  the platform finished making it, so "it did not finish" leaves nothing behind to serve (no code path writes
  it today, since a generated row is born `ready` and nothing parses it). `expired` is the opposite case -
  those bytes are genuinely gone, and it is the only status allowed to say so.
- **Retention treats readable and unreadable uploads alike.** With no `ATTACHMENT_RETENTION_DAYS` set, a
  `failed` upload keeps its bytes - and stays downloadable - for as long as its conversation, same as a `ready`
  one. Once an operator sets the window, both kinds age into the same tombstone on the same clock: a file
  somebody can still fetch is one a retention policy has to be able to reach. The tombstone write also clears
  `error` - that message is real only until the bytes go, and a surviving `error` would make a months-old parse
  verdict read as a fresh fault rather than a normal, policy-driven expiry.
- **How a chat turn uses an attachment.** Once `ready`, a file's text sits in the conversation's ephemeral
  retrieval scope, so a question that resembles what is *inside* it already finds it by similarity. A question
  *about* the file often does not - "what does the screenshot say" shares almost no vocabulary with a page of
  terminal commands, so its chunks score below the relevance floor and get thrown away. Naming the file
  (`attachment_ids` on the chat request) buys three things similarity cannot: the turn **waits** (briefly,
  capped) for it to finish parsing rather than answering as though it were never sent; it is **read regardless
  of resemblance** - naming a file makes the reference itself the relevance, and its passages lead the context
  (both the source dedupe and the context-budget trim preserve input order) so a tight window cannot squeeze it
  out in favor of the knowledge base; and a turn **admits** when it could not use the file - parse failure,
  expiry, or a wait that ran out each produce their own standing notice, because silently ignoring a named file
  reads exactly like reading it and finding nothing (an expired file gets its own wording, since nothing went
  wrong and it is not worth a wait - `expired` is as settled as `ready` or `failed`). Only the turn a file
  arrives on carries its id (the composer's tray empties on send), so a follow-up naming the file, or pointing
  at it unmistakably ("the screenshot", "that pdf"), is read as re-naming it - the match errs toward including
  a file, since including one unintended costs a bounded slice of context a model can ignore, while missing one
  intended is an answer given as though the document were not there. A file becomes *many* retrieval documents
  (one per page, one per oversized part), each chunk carrying the attachment's id as its `record_id`, which is
  what lets a turn address "that file" as a whole rather than one piece of it.
- **A turn can be a file and nothing else.** Sending an attachment with no typed text is a real request, so
  `messages.content` may be empty on a user turn that carries attachments (and only then) - the stored message
  stays empty rather than inventing words the person never typed, while the model is asked about the files by
  name.
- **`origin` exists so the interface can be honest about pasted text.** A long paste (a stack trace, a config,
  a few pages of prose) becomes a real, stored attachment through the identical pipeline - written, parsed,
  chunked, embedded, cited like any file. `origin = paste` only changes presentation: a citation reads "Pasted
  content" instead of a filename nobody chose, and the composer renders a text card instead of a file chip - a
  column rather than a naming convention on `filename`, so neither the UI nor a citation has to infer intent
  from a string.

### `collections`

A named, persistent RAG corpus owned by a tenant - "our product docs", "support tickets" - as opposed to the
ephemeral, session-scoped vectors that back one-off file chats. The vectors themselves live in Qdrant; this
table is the durable catalog of which corpora exist. The ingest API addresses collections by name
(get-or-create), so names are unique per tenant - without that constraint, two concurrent first ingests could
create ambiguous duplicates.

| Column | Notes |
|---|---|
| `id` (PK) | UUID; stamped as `collection_id` on every Qdrant point in the corpus |
| `tenant_id` | indexed; unique together with `name` |
| `user_id` | nullable - a collection can be workspace-shared |
| `name`, `description` | `name` unique per tenant; `description` nullable |
| `created_at` | |

### `workspace_settings`

Tenant-wide defaults: how a turn behaves when nothing more specific decides.

| Column | Notes |
|---|---|
| `tenant_id` (PK, FK -> tenants) | the tenant id **is** the primary key |
| `settings` | JSON |
| `updated_at` | |

Using `tenant_id` itself as the primary key encodes "exactly one settings row per tenant" in the schema - no
uniqueness bug can ever create a second, conflicting row. An empty row is seeded at boot alongside the default
tenant.

The `settings` JSON is a free column read through a narrow typed view, so the keys below are the contract rather
than the schema. Each one is optional, and an ABSENT key is meaningfully different from a stored value: absent
means the next tier of the resolution chain decides that axis.

| Key | What it holds |
|---|---|
| `model` | the chat tag a turn defaults to |
| `embedding_model` | the tag retrieval embeds with - the vector space itself |
| `thinking` | `auto`, `on` or `off` - the reasoning default |
| `web_search` | `auto`, `on` or `off` - the web-search default |
| `system_prompt` | the persona every turn here opens with, replacing the built-in one |
| `context_ceiling` | the largest context window, in tokens, any turn may request |
| `reranker` | `false` forces dense-only retrieval; `true` or absent let a configured reranker run |

**Why it's shaped this way**

- **`system_prompt` is the top tier of the persona chain:** a stored value replaces both the built-in persona
  and any `SYSTEM_PROMPT` the environment set; an absent one leaves those exactly as they were. Whichever tier
  wins is a persona and nothing more - the tool-guidance layer is appended to it on any turn offering tools, so
  an operator who rewrites the persona cannot silently drop the honest-refusal and never-invent-a-tool
  instructions. Blank is stored as absent rather than an empty string, a deliberate asymmetry with the
  environment variable: `SYSTEM_PROMPT=""` disables the system prompt outright (guidance and injection framing
  included), a switch that stays with whoever configures the deployment, while an emptied settings field means
  "fall back", never "silence the prompt" - folding blank to absent on the way in makes the difference
  unrepresentable in the stored blob.
- **`embedding_model` is the only key that is not a preference.** The embedding model DEFINES the vector space -
  collections are named per model - so changing it points every query at a fresh, empty collection and leaves
  the old one full and unread. It lives here rather than in a committed file because a per-hardware-class file
  ships identically to every machine and says what that class was built with, not what a particular box has
  moved on to. Nothing seeds this key: absent, resolution falls back to the model this deployment's own stored
  passages were written by, and only then to the single installed embedder where exactly one exists - so an
  upgrade keeps reading the corpus it already has, with no frozen literal needed in a migration.
- **`context_ceiling` and `reranker` are limits, not preferences**, applied wherever a window or the reranker is
  resolved (the chat turn, the warm endpoint, the title job, the context-usage estimate, the management view's
  cards) so every reader asks the model server for the same number - a caller that disagreed would make the
  server reload the model between them. `context_ceiling` behaves in two different directions depending on what
  else sizes the model: against the window the model itself declares, it only ever clamps DOWN (a native window
  is a fact about the weights; allocating more cache than the model can take fails as an out-of-memory load or a
  silently truncated prompt, naming no setting as the cause); below that ceiling, it IS the deployment's
  affordance and moves the window in both directions (raising it genuinely raises the window; with no ceiling
  stored, the deployment-wide `DEFAULT_CONTEXT_WINDOW` setting takes over). Migration `e2f3a4b5c6d7` seeded this
  key so the second rule changed nothing for a deployment already running: `DEFAULT_CONTEXT_WINDOW` ships at the
  smallest of the old per-hardware-class numbers, so every workspace with no ceiling of its own was given the
  number its machine was already serving (8192, 16384, or 32768, frozen in the migration from the environment);
  a ceiling already stored was left untouched, and an absent row needed nothing, since absent already reads as
  unset. `reranker` only ever withholds: the sidecar is something deployment configuration starts, so a stored
  `true` is permissive, never creative, and where none is configured the settings surface reports that as a
  status rather than a switch that would do nothing - whether one is configured is the single question of
  whether `RERANKER_URL` names one, so the compose profile that starts the container and the retrieval path
  that calls it read the same signal and cannot disagree.
- This column follows the lenient-read / strict-write pattern common to every settings column (see
  [Rules that shape every table](#rules-that-shape-every-table), item 5), with one addition: `context_ceiling`
  and `system_prompt` are bounded on the way IN as well as out, because an out-of-range value in either actively
  cripples a turn (a stored `1` ceiling clamps every turn to a single token; an unbounded persona is prepended
  to every prompt) rather than merely being droppable.
- **Writers are narrow:** the settings surface's admin-gated PATCH; a model delete reassigning the default it is
  about to remove; and a first-run seed on the chat turn, writing the default model once and only when nothing
  is configured anywhere and the model server holds exactly one model that can chat - the seed that lets a
  fresh deployment's first turn answer instead of failing, after which every later turn finds the default
  already set.

### `mcp_servers`

The per-tenant registry of MCP tool sidecars (external tool servers the assistant can call, reached over
Streamable HTTP).

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id` | unique together with `name` |
| `name` | human label; unique per tenant |
| `base_url` | where the sidecar listens |
| `enabled` | on/off switch, default true |
| `credential_scope` | which credential this server's tool CALLS carry: `workspace` (the shared header, default) or `user` (each user's own, injected per call and failing closed when unconnected). Two pairings are refused at create and patch: `user` with `auth_type=oauth` (an OAuth grant is still sealed workspace-wide; per-user grants remain a follow-up) and `user` with `auth_type=none` (nothing is ever attached, so a per-user promise would go unkept - rows minted before the guard were normalized to `workspace` by migration `a8b9c0d1e2f3`) |
| `auth_type` | HOW the server authenticates, where `credential_scope` says whose: `header` (a static bearer credential, default - the pre-column behaviour), `none` (an unauthenticated upstream; nothing is attached even when a credential exists in the store), or `oauth` (calls carry the access token from the server's sealed grant; not connected means no header, never a fallback) |
| `config` | JSON - server-specific knobs: transport timeouts, extra headers, the per-tool enable map |
| `created_at` | |

**Why it's shaped this way**

- **Registration is data, not deployment:** pointing a tenant at a new tool server is an insert, not a
  config-file edit and restart. The `(tenant_id, name)` uniqueness gives each tenant a clean namespace of server
  names. A row does not imply a container - `base_url` may point at a sidecar the deployment runs, a
  vendor-hosted endpoint, or any other process speaking the protocol, and one server can wrap several upstream
  systems. The converse holds too: one process can mount several registered servers at distinct URL paths, each
  row keeping its own credentials, scope, and per-tool toggles - one deployment artifact, several logical
  integrations.
- **This is the live source registry as of Phase 5 slice 6b PR2:** the chat path reads a tenant's enabled rows
  (via `McpServerRepository` / `McpSourceProvider`) and builds one tool source per row, wiring a per-call,
  per-user credential for a `user`-scoped server. The env-configured sidecar URLs (`MCP_CHASSIS_URL`,
  `DOCGEN_MCP_URL`) are each upserted here once as a non-destructive bootstrap seed, so an env-only deployment
  works with zero config. The row's settings are authoritative thereafter - an edit or a disable survives every
  restart, since the seed never overwrites - but its existence stays config-owned: while the environment still
  names a seed, deleting its row is refused with a 409 (a delete would only last until the next boot re-seeded
  it), and the listing marks the row as seeded. Unsetting the variable releases the row.
- **The write path** (slice 6b PR3, `app/api/v1/integrations.py`) is admin-gated create/update/delete plus a
  per-server credential write (via [`secrets`](#secrets)) and a test-connection handshake. `config.disabled_tools`
  is a per-tool deny-list an operator toggles, filtered out of a listing so a server offers only what is left
  switched on. The UI over it (slice 6b PR4) is a standalone page reached from the admin console, folding into
  the Settings surface in [ROADMAP Phase 6b](ROADMAP.md).

### `oauth_authorizations`

A pending OAuth consent - the state between an admin clicking "Connect" and the provider's callback arriving.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id` | scoped by the repository layer |
| `mcp_server_id` | which server the consent is for |
| `state` | the callback's lookup key; globally unique, 32 random bytes |
| `code_verifier_sealed` | the PKCE verifier, AES-GCM sealed and AAD-bound to (tenant, state) - a database read alone cannot complete an exchange |
| `client_id` | the operator-registered OAuth client |
| `redirect_uri` | recorded so the token exchange repeats it verbatim |
| `resource` | the canonical RFC 8707 binding on the MCP-native path; NULL for generic providers |
| `endpoints` | JSON snapshot (issuer, token/revocation endpoints, send_resource) so the callback never re-runs discovery |
| `scopes` | space-joined requested scopes; the grant's fallback when the token response omits its scope |
| `created_by` | audit: which user started the consent |
| `expires_at`, `created_at` | rows expire (default 10 minutes) and are swept on the next insert |

A row is consumed - deleted - exactly once, by the callback carrying its `state`; single use is what blocks an
authorization code being injected into a second flow, and expiry plus the sweep-on-insert keeps the table
near-empty without a scheduled job. The sealed token bundle that a completed consent produces does NOT live
here: it goes into `secrets` under the server's stable-identity name, exactly where a pasted credential would.

### `secrets`

Encrypted credentials for upstream systems (for example, the API key of a datastore an MCP server wraps).

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `tenant_id`, `user_id`, `name` | unique together |
| `user_id` | nullable - NULL is a workspace-shared secret, a set value is one member's own |
| `name` | which credential this is |
| `ciphertext` | the encrypted secret bytes (a random nonce prepended to the AES-GCM output) |
| `last4` | the only readable hint ever exposed |
| `key_version` | which master key sealed this row, woven into the AES-GCM associated data |
| `created_at`, `updated_at` | |

**Why it's shaped this way**

- A secret is either **workspace-shared** (`user_id` NULL) or **private to one member** (`user_id` set), so
  members can each connect their own account under a shared name. Uniqueness is `(tenant_id, user_id, name)`,
  declared `NULLS NOT DISTINCT` so two workspace rows (both NULL owner) still cannot duplicate a name - without
  that flag Postgres would treat the NULLs as distinct.
- **Plaintext never rests anywhere.** Values are encrypted with AES-GCM (an authenticated encryption scheme -
  tampering with the ciphertext is detectable) using a master key from the environment; the read API returns
  only metadata plus `last4` (enough for "which key is this?" in a UI); decrypted values are injected into tool
  sidecars per request, never persisted downstream. The row's own identity - `tenant:owner:key_version:name` -
  is bound into each ciphertext as associated data, so a blob decrypts only under the exact row it was sealed
  for: were a query ever to hand one request another row's bytes, decryption would fail loudly rather than
  return the wrong credential. An unset master key is a first-class state - the store reports itself
  unavailable (the integration surface answers 503) rather than falling back to plaintext or minting a key.

### `model_probes`

The measurement log: one row per completed check, keeping what a (model, weights, runtime) combination
actually demonstrated. Every harness that writes here has two entry points - a command-line diagnostic and
an endpoint - and their rows look identical apart from the trigger columns, which only a request has an
actor to fill.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `provider`, `model` | which serving stack and tag the run drove |
| `digest` | the weights the tag pointed at, when the runtime exposes one - tags are mutable, digests are not |
| `runtime_version` | the serving runtime at run time (capability bugs have been runtime properties) |
| `serving_endpoint` | which model server answered: scheme, host and port only - see below |
| `harness_version` | fingerprints what the run measured (case specs, served prompt stack, or the embedder corpus); editing any of them invalidates what the old rows measured |
| `depth` | which harness ran: `fast` or `deep` for the tool check, `embedder` for the embedder check - see below |
| `earned_agency`, `thinking_with_tools` | the verdicts the run earned |
| `tokens_per_second`, `vram_bytes` | observed numbers, recorded because the run had them anyway |
| `cases` | JSON: what the run measured, in the shape its own harness produced (tallies, sample verdicts, the measured think mode, where the model sat) - see below |
| `triggered_by_tenant_id`, `triggered_by_user_id` | audit-only - see below |
| `created_at` | |

**Why it's shaped this way**

- **Two harnesses share the table, and `depth` tells them apart.** The tool check writes `fast` (screens, earns
  at most `assisted`) and `deep` (full, can earn `agentic`) - the only two values the agency resolver ever asks
  for. The embedder check writes `embedder`, a run of the retrieval-side harness that measures the similarity
  bands an embedding model produces and either validates the relevance floors declared for it or proposes a
  pair derived from them. The separation is structural: every agency-path read names the depth it wants, so a
  depth nobody asks for cannot reach a decision it says nothing about (an embedder row still fills
  `earned_agency` with `deterministic` rather than null, keeping the column's meaning consistent across both
  harnesses - a clamp on a model nothing routes to). `tokens_per_second` stays null on embedder rows, since an
  embedder generates no tokens and an invented rate would sit in the column the management grid reads as decode
  throughput. An embedder row's `cases` instead holds score bands, per-triplet detail, which query prefix was
  measured and where it came from, and the floors verdict or proposal (a proposal records `applied: false`
  beside a caveat that its numbers come from a corpus scoring higher than production text, since relevance
  floors are configuration and nothing reads those numbers back).
- **The floors verdict is a COMPOSED gate, not a per-floor pass/fail:** what the two floors do applied
  together, simulated per query the way retrieval composes them, kept with per-query detail so a leaking row
  names which query and passage. The per-floor band checks stored beside it are diagnostics that decide
  nothing - a failed band check beside a passing composed gate is the ordinary case. A second composed verdict
  covers the pair the deployment was actually retrieving with at run time: what a seed declares is a statement
  about the model, while `RAG_MIN_SCORE`/`RAG_REL_MARGIN` are what decided every real answer - a row keeping
  only the first could read "the floors hold" while the numbers in force dropped every true answer.
- **`serving_endpoint` records WHERE a run happened**, because throughput is a fact about that host's hardware
  and tool-calling behavior belongs to the runtime as that host runs it - so once a model server can be
  somewhere else (a rented pod, another box), a verdict that traveled with the log would let a pod's
  certificate vouch for a laptop that never ran a check. Stored as **scheme, host and port only**, stripped of
  userinfo/path/query, because a model-server URL routinely carries credentials and this value lands in a row,
  an API payload and a management page - all outliving the URL. Enforcement is asymmetric like every other
  validity axis: a verdict that would LIFT a model must name the host being served from right now; one that
  CLAMPS applies regardless of host, since moving a deployment does not rehabilitate a model that failed a
  guard. NULL (rows predating the column, or a backend naming no host) certifies nothing while its failures
  still count. No index - the comparison happens in memory against rows the
  `(provider, model, digest, created_at)` lookup already returned.
- **A tool-check sample records the two serve-time options it ran at**, `num_ctx` and `think`, because a
  deployment can change either without the weights changing (a context ceiling moves the window; a confirmed
  think-mode declaration or a re-pull moves the flag) - the digest says nothing about them, so a verdict naming
  neither could not be read against the turns it certifies. Both are purely descriptive: nothing reads them
  back into a decision, and a later change to either neither stales nor clamps a verdict already earned.
  `think` carries `true`, `false`, or `null` for the flag omitted from the request - a third thing to send, not
  a missing field, since a model whose serving stack has no reasoning plumbing is asked without the flag at all
  to avoid a rejection. A sample written before one of these fields existed carries neither, which reads as
  "not recorded" rather than a zero window, a `false` flag, or today's configuration.
- **The payload also carries `representative`**, the residency reading (GPU vs. spilled) behind it: a run that
  spilled off the card still counts against a model that failed a case, but cannot stand as the proof that
  lifts one. It lives in the JSON rather than a column because nothing queries it directly - the one lookup is
  still "newest verdict for these weights", and the marker is read off whatever row that returns. Rows written
  before the marker existed read as representative, exactly as trusted as when written.
- **`measured_think_mode`** (`never`, `hybrid`, `always`, or absent when undetectable) records which CHANNEL a
  model's reasoning comes out on, measured by the same run on the same weights, so it retires on a re-pull like
  the verdicts beside it. It is the value a `model_policy` confirmation stores, and the one field on this row
  that can change a chat turn before anyone confirms anything - see [`model_policy`](#model_policy) for the one
  direction in which it does.
- **This table breaks two rules above, both owner-approved (2026-07-15).** It is machine-scoped, not
  tenant-scoped: a probe describes the shared model server, so the trigger columns are audit trail, never
  filters, and the repository takes no identity at all. And it is the no-model-registry rule's sanctioned
  exception: a measurement mirrors no other source, so it has nothing to drift from, and its rows cost GPU
  minutes to recompute. It is append-only history - a re-pull that changes `digest` makes old verdicts
  invisible to a digest-bound lookup on its own (no code enforces the reversion), a runtime upgrade leaves old
  rows as a "passed there, fails here" comparison, and nothing updates a row after it is written. The one read
  path is "newest verdict for these weights", served by the `(provider, model, digest, created_at)` index.

### `model_policy`

What an operator DECLARED about a model, where `model_probes` records what a check MEASURED. One row per
declaration: a ceiling somebody set from the management view, a think mode a check confirmed, a context
window chosen for a tag.

| Column | Notes |
|---|---|
| `id` (PK) | UUID |
| `provider`, `model` | which serving stack and tag, the tag spelled as the provider lists it |
| `digest` | the weights installed when the declaration was made - what binds the trust axes below |
| `ceiling` | the strongest agency mode this model may attempt |
| `thinking_with_tools` | a granted reason-and-call-tools claim |
| `think_mode` | `never`, `hybrid` or `always`, from a check that measured the weights |
| `num_ctx` | a context window chosen for this tag |
| `thinking_model` | a sibling tag override: which model serves this one's thinking turns |
| `declared_by_tenant_id`, `declared_by_user_id` | audit-only, exactly as the probe log's trigger columns are |
| `probe_id` | the `model_probes` row a declaration rested on, when a check backed it |
| `created_at` | |

**Why it's shaped this way**

- **Every axis column is nullable, on purpose.** A row states what one action declared and stays silent about
  everything else; the effective policy is **folded per axis** at read time - for each axis, the newest row
  that sets it wins, carrying that row's own `digest` along. Copying prior axes forward (the obvious
  "always-complete row") is the one shape ruled out: it would carry a ceiling declared against old weights onto
  a re-pull as a side effect of somebody setting a context window, silently re-arming a grant the re-pull was
  meant to retire.
- **The axes split into two kinds, and the fold treats them differently.** Trust axes (`ceiling`,
  `thinking_with_tools`, `think_mode`) are claims about WEIGHTS, honored only while the declaring row's digest
  is still installed - a re-pull retires them with no code doing anything, the same structural invalidation
  that ages out a probe verdict. Preference axes (`num_ctx`, `thinking_model`) are claims about how this
  DEPLOYMENT wants a tag run, so they follow the tag across a re-pull: "give this model a 16k window" is not a
  statement about a checkpoint.
- **Machine-scoped and append-only, like [`model_probes`](#model_probes) and for the same reason:** it
  describes the shared model server, so one workspace cannot hold a different answer about a set of weights
  than another - the declaring tenant and user are audit trail, never filters. There is no update method,
  because an update is exactly how a grant would cross a re-pull undetected.
- **Read today by the chat turn's capability resolution and the Model Manager grid, through the same
  resolvers** (stored declaration, then the code-shipped quirk seed, then the runtime's own report, then the
  safe floor), so a card and a turn can never disagree about a model.
- **Four writers, all bound by three rules the first one set** - drop the read caches after commit, never copy
  another axis into the new row, and validate the stored value against the modes this build knows:
  - The **think-check confirmation** appends `think_mode` alone, plus a fourth rule of its own: the value
    stored is what a check MEASURED on the weights installed right now, never a value the caller sent, and a
    model nothing has measured is refused rather than declared.
  - The **ceiling declaration** appends `ceiling` alone, bound to the digest at click time. Direction matters
    here: lowering needs no evidence and applies next turn, while raising only withdraws a human objection -
    `min(CEILING, EARNED)` still holds the model at whatever a digest-bound deep check has proven. Declaring
    `agentic` on an unproven model is therefore stored, not refused: a durable statement of what an operator
    permits, applying the moment a check earns it. The endpoint answers with the resolved `earned` and
    `effective` beside the value written, so a page cannot report an escalation that did not happen. A tag the
    runtime exposes no digest for is refused outright - a claim about weights needs weights to name.
  - The **reasoning-with-tools declaration** appends `thinking_with_tools` alone, gated exactly like the
    ceiling (the fold, never a refusal at the endpoint): a grant is stored whatever any check has found, and
    reaches a turn only once a deep check has passed the reasoning case on these weights, at the running
    harness version, on the host being served from. Withdrawing needs no evidence and applies next turn,
    because the failure it guards is silent (see the asymmetry below).
  - The **thinking-pair declaration** appends `thinking_model` alone and carries **no digest**, since it is a
    preference - which two tags run together, not a claim about either checkpoint. The partner is refused
    unless its think mode already resolves to `always` or `hybrid`; pairing two direct-answer models would
    produce a control that silently changes nothing, and an unmeasured partner is that same offer on no
    evidence.
  - Every writer reads a FRESH provider listing rather than the shared cache, on the precedent the think-mode
    confirmation set: a pull made outside the app leaves the cached digest stale, and a declaration bound to
    weights the machine no longer holds would read as working until the cache lapsed. Each then drops both the
    declaration cache and the installed-digest cache, since a declaration is only visible while the digest it
    is gated against is.
- **One exception is deliberately not a write.** A check that watches a model reason with the think flag
  CLEARED forces the flag back on for routing immediately, without waiting for confirmation - the alternative
  is the model's reasoning arriving as raw markup inside the visible answer. Nothing is stored, and the reading
  is bound to the installed weights; no other direction acts early - removing reasoning, or offering the flag
  to weights that reject it, waits for a human.
- **`thinking_model` shares one resolution function with everything that reads it**, so no second copy can name
  a partner a turn ignores: the `THINKING_CHAT_MODEL` env override (the deployment's kill switch), then a
  stored pairing, then nothing. Four readers share it - the chat turn, the composer's model listing, the
  management view, and the delete guard - and the guard is enforced in the SERVICE the endpoint and the view
  both call, not by each remembering to ask (a first draft made it a parameter and the delete endpoint omitted
  it, disabling a button while the request behind it still went through). `num_ctx` is stored and not yet
  honored - the column exists ahead of its reader, so the slice that adds it needs no migration. Three readers
  still resolve a pair WITHOUT the stored declaration - the warm hook, the background title job, and the
  context-usage estimate - substituting on the environment's pairing alone when the configured default is
  missing, where a turn substitutes the declared partner instead; none changes what an answer says.
- **One asymmetry, because the two claims look alike:** a ceiling applies in both directions on the next turn,
  but a `thinking_with_tools` grant does not - it is a starting point a measurement then has to confirm,
  needing a deep verdict bound to the installed weights in every configuration, because reasoning and calling
  tools at once fails as an empty answer rather than an error on some model-and-runtime pairs. That is the one
  axis no declaration may enable on its own - its writer stores a permission rather than grants a capability,
  and withdrawing it is the direction that costs nothing.
- **This table replaced a committed per-model file, and nothing was migrated into it:** a migration reading
  that file would have written different rows on every machine, since the file described what a hardware class
  was built with rather than what a particular box holds. A deployment that leaned on those declarations
  re-states the ones it still wants, against the weights it actually has - every axis left unstated is simply
  unclaimed, the safe direction.

## What is deliberately absent

There is **no model-registry table**, and no committed catalog file either. Which models exist is whatever the
model server reports, read behind a short in-process cache; a stored copy of that would only drift from
reality. This is revisited only if per-tenant model allowlists become a requirement after sign-in lands. Two
tables above are the deliberate, signed-off exceptions, and neither is a registry mirror: `model_probes`
records what a check MEASURED, and `model_policy` records what an operator DECLARED. A measurement has no
other source to drift from, and a declaration IS the source - see each section for the full reasoning.

There are **no role tables yet**, and there is no session table either. Finer-grained role-based access
control ([ROADMAP Phase 8b](ROADMAP.md#phase-8b---identity--access)) adds `roles` (tenant-scoped, with seeded
`admin`/`member` built-ins), `role_permissions` (a typed permission key plus an optional resource selector),
and `user_roles` - designed now, migrated then. Until that lands, [`users.role`](#users) is the only
role-shaped column, and with a login configured it is genuinely enforced: the admin surfaces refuse a member.

Sessions live in Redis rather than in a table, keyed by an opaque cookie value. A session is a key with an
expiry by nature - created, read, slid, and gone - so a row would make every operation a write and would need
another background sweep to clean up after itself. Holding them server-side (rather than as a signed stateless
token) is what makes signing out and demotion take effect immediately: the record is the authority, so ending
it ends the session, and the role is re-read from the `users` row on every request rather than frozen into a
token at sign-in.

## Which tables are live today

Every table on this page has code behind it, but not every table is read and written by the same slice of the
app. This section is the reader/writer index; each table's own section above stays the place for WHY it is
shaped the way it is.

| Table | Written by | Read by |
|---|---|---|
| [`tenants`](#tenants) | seeded once at boot | every scoped query, on every chat turn |
| [`users`](#users) | seeded once at boot; sign-in upserts verified claims once login is configured | every scoped query, on every chat turn |
| [`conversations`](#conversations) | every chat turn (`title`, `settings`, `pending_confirmation`) | every chat turn's history rebuild; the model-delete guard's tenant-wide pin count (see [rule 2](#rules-that-shape-every-table)) |
| [`messages`](#messages) | every chat turn (append; the three delete paths under [Regenerating](#messages) above) | every chat turn's history rebuild |
| [`collections`](#collections) | the RAG ingest API (get-or-create by name); the knowledge console (`POST`/`DELETE /api/v1/knowledge/documents`) | the same two surfaces, to resolve a corpus by name |
| [`workspace_settings`](#workspace_settings) | the settings surface's admin-gated PATCH; a model delete reassigning its default; the first-run model seed; the Model Manager's "set default" action | every chat turn (resolution chain, context ceiling, reranker toggle); the Model Manager delete guard |
| [`attachments`](#attachments) | `POST /attachments`; the parse worker (`pending` -> `parsing` -> `ready`/`failed`); the retention sweep (-> `expired`); the stall sweep (-> `failed`); conversation delete's cleanup job; the chat turn (`origin=generated` rows) | the chat turn (`attachment_ids`); the composer's draft tray; conversation delete (collects blobs to remove) |
| [`mcp_servers`](#mcp_servers) | the Integrations API (admin CRUD, slice 6b PR3); an env-configured sidecar URL, upserted once as a bootstrap seed | the chat turn, to build a tenant's tool sources |
| [`oauth_authorizations`](#oauth_authorizations) | an admin's "Connect" click | the provider's callback, which deletes the row it consumes |
| [`secrets`](#secrets) | the Integrations credential endpoints (write-only) | a tool call, to inject the acting user's or the workspace's credential |
| [`model_probes`](#model_probes) | `make tool-check`; the Model Manager's check-run action; `make embedder-check` / `POST /models/embedder-check` | the agency resolver's earned-verdict lookup; the Model Manager grid |
| [`model_policy`](#model_policy) | the think-check confirmation, the ceiling declaration, and the thinking-pair declaration - all from the management view | the chat turn's capability resolution; the Model Manager grid; the model listing; the warm hook; the background title job |

A few facts do not fit that table because they are not about any one row:

- **The knowledge console has no table of its own.** Nothing about the knowledge base's contents lives in
  Postgres - its documents ARE the stored chunks. The console assembles its inventory (names, source types,
  passage counts, embedding provenance, ingest times) by reading chunk payloads back out of Qdrant, and a
  document's identity is the payload field naming its source rather than a primary key.
- **A reloaded transcript is rebuilt from `messages.meta`.** Besides `citations` (the same objects the SSE
  `citation` events stream), it carries `notices` (standing notes qualifying an answer), an `interrupted` flag,
  `attachments` (references to files the turn carried or produced, each tagged with its `origin` so the two are
  told apart even after the file itself is gone), and `tool_steps` (the tool-call/tool-result trace, including
  an `approved`/`denied` marker on a side-effecting call, and a `count` of the sources the tool produced when it
  could say). The rest of `meta` stays server-internal.
- **A finished reply keeps what it took to write, in `meta.usage`.** Two numbers: `completion_tokens`, summed
  over every round of the turn (a round that only asked for a tool wrote tokens too), and `generation_ms`, the
  runtime's own writing time for those rounds, so the pair divides into a real speed with no tool run, no wait
  on a confirmation and no model load inside it. `generation_ms` is null when any counted round came without a
  time: a speed over part of the tokens would be wrong, not approximate. The key is absent on a cut-short or
  failed reply (the round that was cut reported nothing), when the provider counted nothing, and on every row
  written before it existed - all of which read back as "no figures". It is the same object the stream's
  `done` event carries, so a reloaded reply shows exactly what its live stream closed on. It is a key rather
  than a column by [rule 4](#rules-that-shape-every-table): nothing queries or constrains it yet.
- **The model-delete guard's reads never write.** Counting a tenant's conversations that pin a tag, and reading
  the workspace default beside it, both go through the JSON accessor rather than string surgery, so one
  statement serves Postgres and the SQLite used in tests.
- **An empty `model_policy` table is a normal, safe state.** With nothing declared, every axis falls through to
  the sources beneath it (the quirk seed, the runtime's own report, the safe floor) - which is exactly what an
  unclaimed declaration store should mean.
- **The embedder check writes but nothing reads it back.** Its rows sit at the `embedder` depth, which none of
  the agency-resolving readers above ever asks for, so they are recorded history for a person to read rather
  than an input to anything the app decides.
- **`workspace_settings` is tenant-scoped where `model_policy` is not** (see that table's section): a default
  model is a workspace's own choice, not a claim about the shared model server.

## What lives in each volume

The tables above are only part of what a deployment holds. The stack keeps nine Docker named volumes - seven
declared at the end of [`infra/docker-compose.yml`](../infra/docker-compose.yml), plus the proxy's two, which
exist only under the deployment layout
([`infra/docker-compose.prod.yml`](../infra/docker-compose.prod.yml)). The difference that matters is not their
size - it is whether losing one loses anything that cannot be made again.

| Volume | Holds | In a backup |
|---|---|---|
| `postgres_data` | Every table on this page | Yes - dumped |
| `attachment_data` | The uploaded file bytes, under `STORAGE_BACKEND=local` | Yes - archived, when in use |
| `qdrant_data` | The retrieval vectors and their payloads | Yes - snapshotted |
| `redis_data` | Background job queue state | No |
| `ollama_models` | Downloaded model weights | No |
| `infinity_cache` | The reranker's downloaded weights | No |
| `minio_data` | Object storage under the opt-in `minio` profile | No - see below |
| `caddy_data` | TLS certificates, private keys and the ACME account | No |
| `caddy_config` | The proxy's autosaved last-known-good configuration | No |

**The database and the file store are two halves of one fact.** An `attachments` row records a storage key; the
bytes behind that key live wherever the storage seam points. A row without its bytes is a broken link, and
bytes without a row are unreachable and reclaimed by nothing - so the two are always copied in the same run,
never separately.

**`STORAGE_BACKEND` decides where the attachment bytes actually live**, which is what determines whether a
backup can cover them. Under `local` they are files in `attachment_data`, archived alongside the database;
under `s3` they are objects in a bucket, so that volume is empty or stale and a backup skips it rather than
archiving nothing useful. Mechanics - the manifest flag that records the skip, what a restore does with it, and
how an S3 deployment backs up its bucket instead - are in
[MAINTENANCE.md section 3](MAINTENANCE.md#3-backups-and-restores).

**Postgres is dumped and Qdrant is snapshotted, not copied, for the same live-consistency reason**: a volume
copied while its service is running can span several inconsistent moments. See
[MAINTENANCE.md section 3](MAINTENANCE.md#3-backups-and-restores) for the mechanics and where each archive
lands.

**Vectors are the one backed-up volume that could in principle be rebuilt**, by re-ingesting and re-embedding
every source. They are copied anyway because that rebuild costs hours of GPU time and needs the original
documents to still exist - which also makes a small drift between the database copy and the vector copy
tolerable: a missing vector degrades retrieval quality, where a missing attachment is a visible break.

**The exclusions, each for its own reason.** `redis_data` holds the arq job queue and nothing else - losing it
loses queued work, visibly rather than silently: a periodic worker sweep fails any upload row whose job no
longer exists, so a file stuck mid-processing stops claiming to be in progress. `ollama_models` and
`infinity_cache` are downloaded weights, replaced by downloading them again from the model manager and by the
reranker's own first start - and on a deployment whose model server runs on a separate machine, `ollama_models`
is not even on this host. The proxy's two are excluded on judgement, not because they hold nothing:
`caddy_data` holds real secrets (TLS private keys, issued certificates, the ACME account) and `caddy_config`
holds the proxy's last accepted configuration - both re-creatable by pointing the proxy at the domain again,
but re-issuing is not free either (certificate authorities rate-limit it, so a proxy container recreated a few
times in an afternoon can lose certificate access for hours), which argues for keeping the volume across
recreates (which compose already does) rather than copying private keys into a backup.

**`minio_data` is the one exclusion with a real gap behind it** - the opt-in `minio` profile is the self-hosted
form of `STORAGE_BACKEND=s3`, where "back the bucket up with the object store's own tooling" has no third party
to point at, since the deployment *is* the object store. See
[MAINTENANCE.md section 3](MAINTENANCE.md#3-backups-and-restores) for how it is covered instead.

Command mechanics (`make backup`, `make restore`, `make host-disk`) are in
[MAINTENANCE.md section 3](MAINTENANCE.md#3-backups-and-restores).

## How the schema changes

Schema changes are Alembic migrations, committed alongside the model change. The backend applies pending
migrations automatically before serving, so a fresh clone needs no manual database step (`make db-upgrade` runs
them by hand when needed). When a table changes, this document changes in the same pull request.
