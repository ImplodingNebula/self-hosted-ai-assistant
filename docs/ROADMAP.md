# Roadmap

The build order for the AI Chatbot Platform: what gets built next, in what order, and why that order. This is the
execution reference; [ARCHITECTURE.md](ARCHITECTURE.md) is the self-contained design reference and carries the
rationale behind every component named here. Each phase below is a runnable increment, written so its scope can be
understood without reading any code.

---

> **Pull request numbers (`#123`) and commit ids in this document refer to the private development repository this one is published from.** They are kept as references and do not link anywhere.

---

## 1. Status

Phases 0 through 3b are merged to `main`: the stack boots with one command, chat streams from a local model,
conversations persist, answers are grounded in retrieved passages with citations, and the assistant searches the
web when a question needs information newer than the model's training data.

Phase 4 (attachments) is closed and in daily use - a file can be attached from the composer, asked about on the
turn that sent it, and referred to again on later turns - across plain text and source code, PDFs both native and
scanned, the Office and OpenDocument families, RTF, EPUB and images. A round of fixes from that live use landed on
top of it, ending in a deliberate hardening pass over the parse-and-ingest path (see
[the fixes table](#after-the-slices-what-live-use-found)). Everything the phase set out to build is merged. Two
pieces were deferred out of it at the time and have both since landed: the S3 storage adapter, which waited for
the deployment that exercises it and shipped with
[Phase 8](#phase-8---hardening-ci-gates-deployment)'s run of work, and true vision - a model seeing a picture
rather than reading the text in it - which waited for [Phase 5](#phase-5---mcp-subsystem) and arrived there
alongside the pilot of a single model tag that does chat, reasoning, tools and vision at once. The slice table
in the Phase 4 section marks exactly what landed and what did not.

[Phase 5](#phase-5---mcp-subsystem), the tool layer, is complete. It makes the backend an MCP host, so the
assistant decides for itself when to search the web or the stored records instead of being handed results it never
asked for.

**The loop.** The seam, the registry and the agency policy landed first (slice 1), on a runtime measured and
pinned forward (slice 0). The provider seam then learned to carry tool calls in both directions (slice 2a), the
generation loop was reshaped to re-enter safely (slice 2b), and that loop now runs the tools a model asks for and
feeds the results back (slice 2c).

**The first real tools.** Web search and knowledge search are offered to the model as tools (slice 3) - the first
behaviour change - with their results and a tool server's returned records both cited like any other source
(slices 3c and 4b), and a tool-check harness and its measurement log deciding when a model has earned the drive
(slice 3b). The host reaches a real sidecar over Streamable HTTP and picks which tools to offer under schema
pressure (slice 4). Encrypted per-user credential storage landed alongside (slice 6a), and a side-effecting tool
now asks before it acts: the turn suspends on the call and resumes on the user's approve or deny (slice 5).

**In the UI.** Slice 7 surfaces all of it - the confirmation prompt that gates a side-effecting call, and the
activity trace of each call, its result and the model's narration, with the prompt folded into that trace once
answered (an approved run, or a denied one, badged in place). The whole trace persists on the message, so a reload
replays it.

**Connections.** Slice 6b makes a tool call carry the acting user's own credential (off by default, per server).
The chat turn builds its sources from the tenant's enabled `mcp_servers` rows, each declaring whether its calls
carry the shared workspace header or the acting user's own credential, failing closed when unconnected, with the
env server seeded in as a bootstrap row. Administration happens through the Integrations API and its UI: one
generic add-server form (MCP servers self-describe, so there is no per-integration UI code), write-only credential
entry (metadata and a `last4` hint on read), per-server manage cards, and a test-connection probe that does an
initialize-and-list handshake, never runs a tool, and authenticates exactly as a real turn would. The env
`MCP_CHASSIS_API_KEY` survives only as the seeded chassis's bootstrap fallback, so it can never reach an unrelated
server an admin later registers.

**Seeing, and consenting.** Slice 8 carries in the vision passthrough Phase 4 deferred: a picture reaches a model
that can take it, gated on the runtime's own report about the model serving the turn, sized against costs
measured on the serving runtime, with the honesty notice reporting per image which pictures were not shown and
why. Slice 9 replaces the pasted token with a consent screen - hardened authorization-server discovery, a PKCE
flow with single-use state and an exactly-bound redirect, a sealed grant bundle whose status a card can read even
on a deployment with no master key, and on-demand renewal that marks a dead grant instead of retrying it forever.

**Usable without reading a specification.** Slice 10 adds one-click presets for GitHub, Gmail, Google Drive, Slack
and Asana, each prefilling the add-server form with provider facts verified against primary documentation and live
probes, the deployment's literal redirect URI shown where it is needed, and a test-connection that reports what it
actually saw - reachable rather than connected. Slice 11 closes the phase with a documentation pass and the
security review the phase was always scheduled to get.

[Phase 6](#phase-6---model-manager), the Model Manager, is complete too. It gives the local model stack a surface -
what this deployment could run, what it is running right now, and what each model has actually earned - and it
gives a conversation the controls that decide how a turn is answered.

**Born themed.** The design system's token foundation landed first (slice 1), so every page the phase added arrived
already styled: aurora, a dark glass default, beside a minimal pair that follows the operating system. The chat page
was then split into components with no visual change at all (slice 2) and restyled to the design system's fidelity
on both themes (slice 2b).

**A conversation became something to keep.** Renaming in place, a per-row actions menu, titles written by the model
rather than the first sentence truncated, pinning, and an in-app confirmation for anything that cannot be undone
(slices 2c and 2f). A turn can be revisited rather than only added to: copy it, ask the question again, reword the
question in place, or quote a passage back into the next message - with whatever a rewind would discard counted out
loud before it runs (slices 2d and 2e).

**One inspectable plan per turn.** The per-turn decisions that had scattered across the chat service - model,
thinking, tools, budgets - consolidated into a single `plan_turn()` with behaviour byte-identical (slice 3), and the
model resolution chain made real: narrow typed per-conversation settings the turn actually reads, workspace defaults
read above the environment (slice 4), and the composer controls that drive them - a capability-gated model selector,
the thinking toggle and the web-search toggle (slices 9 and 9b).

**Model operations left the terminal.** The read API joins what the model server holds, what is known about each
model, and the measured `model_probes` facts at request time, with no new table behind it (slice 5); pull, delete and the free-text pull box
sit behind the provider seam's own verbs, so a backend that cannot pull never renders a dead button (slice 6); the
verify job wraps the tool-check harness - one probe at a time, a deliberate unload first, a GPU-residency assertion,
a re-warm of chat and embedder after (slice 7); and the stranded-vectors alarm watches the one change that silently
orphans every stored vector (slice 10). Slice 8 is the page all of that was built for, in an admin console shell the
later administrative surfaces reuse: role-grouped rows, live download progress that pauses and resumes, measured
facts with an honest "unmeasured" wherever nothing has been measured, and failure text that explains itself.

**Three later phases then landed ahead of their place in the order**, because running the platform on a real
server needed them first. [Phase 8](#phase-8---hardening-ci-gates-deployment)'s deployment core put the stack
behind one HTTPS origin with
readiness checks, per-client rate limits and volume backup and restore; [Phase 8b](#phase-8b---identity--access)
made sign-in real, with OpenID Connect, server-side sessions and an administrable `admin | member` split; and
[Phase 9a](#phase-9a---document-generation-tool-service) shipped whole, so the assistant can now produce a
spreadsheet, document, PDF or chart and hand it back as a downloadable file. Each is itemised in the phase table
below, with what remains of the partly-landed ones named there.

**The model layer was then corrected rather than extended.**
[Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts) reworked it around what a live trial against a
remote model server exposed: model identity and capability resolved through a committed catalog keyed by
hardware class rather than through what was actually installed, so a healthy connection to a different inventory
yielded a broken deployment. Installed models are the ground truth now, capabilities come from the runtime and
from measured checks, admin declarations replace catalog edits, and the tier has retired outright - which is why
nothing in this repository names a hardware class any more. Per-role model access (R7) and the guarded embedder
switch are what remain of it; the settings surface (6b) and the frontend polish (7) follow.

**The interface was then rebuilt on its default theme.** The first piece of
[Phase 7](#phase-7---frontend-polish--theming) is in: aurora redrawn as a night sky behind a veil, the
conversation laid out as a timeline, a boot screen that waits only on what the page actually blocks on, the
admin console in the same shell as the chat, and a sign-in card that says when a session has lapsed. The rest
of that phase is named in the table below.

As of this writing 3,622 backend tests are collected.

Two repository-maintenance items landed outside any single phase: CI now runs as path-scoped, cached,
concurrency-limited workflows per workspace instead of one broad run on every push
(#274), cutting billed Actions minutes; and
dependencies were bumped past two open security advisories
(#278).

This status line, the phase table below, and the test count are progress markers - advance all three at every
phase boundary, so a fresh clone always reads the true state.

| Phase | State |
|---|---|
| [0 - Repo & infra skeleton](#phase-0---repo--infra-skeleton) | Merged |
| [1 - Model seam + chat vertical slice](#phase-1---model-seam--chat-vertical-slice) | Merged (plus a hardening addendum) |
| [2 - Persistence + multi-tenant schema](#phase-2---persistence--multi-tenant-schema) | Merged |
| [3 - RAG subsystem](#phase-3---rag-subsystem) | Merged |
| [3b - Web search & grounding](#phase-3b---web-search--grounding) | Merged |
| [4 - Multimodal attachments](#phase-4---multimodal-attachments) | Merged, plus a round of fixes and a hardening pass from live use; the S3 storage adapter is deferred to Phase 8 and true vision passthrough to Phase 5 |
| [5 - MCP subsystem](#phase-5---mcp-subsystem) | Merged - every slice, from the tool seam to OAuth and the preset catalog (summarised above; itemised in the [slice table](#slices)) |
| [6 - Model Manager](#phase-6---model-manager) | Merged - every slice, from the theme foundation to the manager page and its console shell (summarised above; itemised in the [slice table](#slices-1)) |
| [6b - Settings surface v1](#phase-6b---settings-surface-v1) | Mostly landed - the Integrations tab and secrets shipped with Phase 5, the defaults half with Phase 6c, the workspace system prompt with #232; per-tool toggles and a health line remain |
| [6c - Model-layer rework & remote model hosts](#phase-6c---model-layer-rework--remote-model-hosts) | In progress - R0-R6, R8's code half and K1's console half merged, so the tier is gone and installed models are the ground truth; R7 and K1's embedder switch remain; amends two Phase 6 deferrals |
| [7 - Frontend polish + theming](#phase-7---frontend-polish--theming) | In progress - markdown rendering (#194), the aurora redesign, with its motion and its phone layout, and per-conversation drafts are merged; the citations panel, the remaining catalog themes and the accessibility pass are not started |
| [7b - Resumable streaming](#phase-7b---resumable-streaming) | Reliability half landed early; the resume itself is not started |
| [8 - Hardening, CI gates, deployment](#phase-8---hardening-ci-gates-deployment) | Partly landed - the deployment core is merged: the Caddy HTTPS origin and production layout (#210), rate limiting and `/readyz` (#211), and volume backup/restore with the host disk and engine-version documentation (#212); the S3 adapter it carried landed early (#204). Remaining: coverage gates, end-to-end tests, GHCR publishing, the two-runtime split, the tier suggester |
| [8b - Identity & access](#phase-8b---identity--access) | Partly landed - slices one and two are merged: the OIDC cutover (#224), the sign-in surface and pre-render session gate (#225), and the member-management page with the multi-user hardening beneath it (#226). Remaining: slice three's named permissions - the `roles` / `role_permissions` / `user_roles` tables and the role editor |
| [9a - Document-generation tool service](#phase-9a---document-generation-tool-service) | Landed, ahead of schedule - every deliverable is merged: the sidecar (#205), the S3 storage adapter it composes with (#204), the host-side artifact-to-attachment path (#207), the download route and its chip in the chat (#216), and generated-file retention (#214) |
| [9b - Sandboxed code interpreter](#phase-9b---sandboxed-code-interpreter-optional) | Optional, gated on 9a proving insufficient; not started |

## 2. How we build: layered, not all-at-once

Three ways to execute a project this size, and only one of them survives contact with a real codebase:

- **One-go** - build everything, then run it. ✗ Rejected. Too much surface to debug at once, no reviewable
  increments, and a single wrong assumption ripples everywhere before anything proves itself.
- **Layered vertical slices** - each phase is a runnable, testable increment. ✓ Chosen. It preserves code
  integrity, maps cleanly onto small pull requests, and lets the app be run and verified before the next layer
  goes on top of it.
- **Parallel work streams** - several people building at once. ✓ Used inside phases, selectively: only where the
  work is genuinely independent, and only after the shared interface is frozen.

**Freeze the contract before splitting the work.** This is the golden rule for parallelism, and it exists because
independent work streams cannot coordinate mid-task. Before splitting work at a *seam* - a place where two pieces
must interlock - write down the interface both sides will build against: the SSE event schema, the `ModelProvider`
signatures, the database schema. Freezing is cheap (an interface stub plus a short note, about fifteen minutes)
and the artifact is worth having in the pull request anyway. Where the work is truly independent - separate config
files, unrelated components - no freeze is needed, and the split can happen freely.

**At every phase boundary:** run the app, run the test suite, review the diff, then merge via pull request. Never
start the next phase on a red build.

## 3. Phases

Each phase is one or a few pull requests on a short-lived branch off `main`. The italic line under each heading
states what the phase unlocks from a user's point of view; the bullets underneath state how.

### Phase 0 - Repo & infra skeleton

*A user can now: see nothing yet - but a contributor can clone the repo and boot the stub stack with one command.*

The foundation every later phase lands on: the folders, the one-command boot, and the checks that guard `main`.

**Deliverables:**

- **Monorepo scaffold** - `apps/`, `services/`, `infra/`, `packages/`, `docs/`, and the git repository itself.
- **Repo hygiene** - `.gitignore`, `.editorconfig`, a pre-commit config, commitlint.
- **Compose topology** - a base `docker-compose` file plus `override` and `gpu`/`cpu` profiles over stub services,
  with `Caddyfile.dev` for the dev proxy.
- **Configuration sources** - [`infra/.env.example`](../infra/.env.example), where every environment variable is
  documented. (A committed model catalog sat beside it for the first six phases; see
  [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts) for why it was retired.)
- **Guarded CI** - a lint workflow and the frontend dependency-audit workflow, plus GitHub templates and
  CODEOWNERS. (CI is continuous integration: the checks that run automatically on every change.)
- **Setup notes** (now [SETUP.md](SETUP.md)) carrying the GPU and WSL2 checklist, so the first boot does not fail on a driver.

**Depends on:** nothing. **Parallelism:** HIGH - config files are independent of each other; keep `docker-compose`
under a single owner so the topology stays coherent.

### Phase 1 - Model seam + chat vertical slice

*A user can now: open the web UI and hold a streaming, multi-turn chat with a local model - with reasoning control
and clear errors when something breaks.*

The first end-to-end slice: browser to backend to model and back, one token at a time.

**Deliverables:**

- **The `ModelProvider` seam** - the one interface that hides whichever runtime runs the model - its
  `OllamaProvider` implementation, and a provider registry.
- **`POST /chat/stream`**, an SSE endpoint (Server-Sent Events: a single HTTP response the server holds open and
  pushes small events down, which is how tokens appear one by one) that normalizes Ollama's line-delimited JSON
  into a neutral `ChatChunk` stream.
- **A minimal Next.js chat page** rendering the streamed tokens.
- **A `FakeProvider`**, the first unit and contract tests, and health endpoints.

**Depends on:** 0. **Parallelism:** MEDIUM - freeze the SSE event contract first (`token`, `thinking`,
`tool_call`, `tool_result`, `citation`, `done`, `error`), after which backend and frontend proceed in parallel.

**How it landed** (the phase carried a hardening addendum):

- **History travels with each request** for now, an interim measure until Phase 2 rebuilds it server-side from
  `conversation_id`.
- **Reasoning is a first-class SSE channel** with an `auto`/`on`/`off` request control. `auto` is a deterministic
  services-layer policy that a later orchestrator can replace without touching the wire contract.
- **A never-thinking model can be paired with an always-thinking one** - `qwen3:4b-instruct` for normal turns and
  `qwen3:4b` for reasoning turns - while a single hybrid model toggles reasoning per request and needs no
  partner.
- **Provider failures surface as typed, human-safe `error` events**: `provider_unavailable`, `model_not_found`,
  `bad_request`, `internal_error`. No stack trace ever reaches a browser.
- **A read-only `GET /api/v1/models`** lists the models available to a turn.
- **Production compose unpublishes the backing-store ports**, containers run non-root, and images are pinned.

### Phase 2 - Persistence + multi-tenant schema

*A user can now: leave and come back - conversations persist, listed with auto-generated titles in a sidebar.*

The database layer, built multi-tenant from the first migration so that real authentication can arrive later
without a schema change.

**Deliverables:**

- **Postgres, SQLModel and Alembic** (versioned, scripted schema updates), with tenant, user, conversation,
  message, collection and attachment tables - `tenant_id` and `user_id` on every row, and a `users.role` column
  ready for later role-gating.
- **The settings schema** - a conversation `settings` JSON column, plus `workspace_settings`, `mcp_servers` and
  `secrets` tables that nothing writes yet.
- Deliberately **no model-registry table**: live model-server state behind a short in-process cache says which
  models exist. A database copy would only drift.
- **A tenant-injecting base repository** and a `get_current_tenant()` stub over a seeded default tenant, where real
  authentication will later plug in.
- **Conversation lifecycle** - chat turns persisted, list/switch/delete, auto-generated titles, a sidebar UI.
- **Server-side history** rebuilt from `conversation_id` under a token-budget truncation policy, and an SSE
  heartbeat comment roughly every 15 seconds so a proxy does not drop a stream that has gone quiet while the model
  loads.
- **The typed error taxonomy** mapped onto RFC 7807 responses (a small standard JSON shape for HTTP errors), with
  request-id structured-logging middleware.
- **A per-tenant concurrency cap** - over-cap requests fail fast with a 429 rather than queueing invisibly.
- **`packages/api-types` generated from the OpenAPI schema**, with a CI drift check that fails when the committed
  types go stale.
- **API and integration tests** through httpx's in-process transport.

**Depends on:** 1. **Parallelism:** LOW - migrations are a serial foundation and want a single owner; repositories
fan out per entity once the schema is in.

**How it landed:**

- **The `done` event carries `conversation_id`** alongside `message_id` and `usage`.
- **A malformed conversation id answers 404 by design.** A value that cannot be a UUID can never address a
  resource, so it is treated exactly like a missing one.
- **The legacy client-held `history` field is still accepted**, but only as pre-context when starting a brand-new
  conversation, and it is never persisted.
- **Interrupted or aborted generations persist the partial reply**, marked as interrupted.
- **Migrations apply automatically** before the backend serves its first request; `make db-upgrade` covers a
  manual run.

### Phase 3 - RAG subsystem

*A user can now: get answers grounded in ingested records, with citations pointing back at the sources.*

RAG (retrieval-augmented generation) means fetching relevant stored text first and handing it to the model, so an
answer comes from real data rather than the model's memory. It is built here as four small protocols over Qdrant,
with one ingestion entrypoint that every later source reuses.

**Deliverables:**

- **Qdrant** (a vector store: a database that finds text by meaning rather than by keywords) behind the
  `VectorStore` / `Embedder` / `Chunker` / `Reranker` protocols, with `qwen3-embedding` served through Ollama.
- **`ingest_structured()`** - the single ingestion entrypoint, feeding persistent collections and ephemeral
  session-scoped ones (`session_id` plus a TTL, "time to live": the moment a record expires).
- **Hybrid-ready retrieval** (dense today), `POST /rag/ingest` and `POST /rag/query`, and `citation` events wired
  into the chat stream.
- **A `FakeEmbedder`** and a fixture record corpus, so retrieval tests are reproducible.
- **Context-window budgeting** - an explicit `num_ctx` per turn, the system prompt reserved first, then a RAG
  budget, then history truncation.
- **Injection defense** - retrieved chunks are delimited as quoted data in the prompt, never as instructions, so a
  document that says "ignore previous instructions" is just text.
- **GPU scheduling** - batched embeds, ingestion running low-priority while a stream is active, and a CPU fallback
  for the small embedder.

**Depends on:** 2. **Parallelism:** MEDIUM - chunker, store and embedder are clean protocol-bounded units; wiring
retrieval into the chat service is the serial step.

**How it landed:**

- **Vector spaces never mix.** Collections are named per embedding model (`kb_chunks__{model}`,
  `session_chunks__{model}`) and shared across tenants behind an indexed `tenant_id` filter, so an embedder
  change lands in a fresh, empty space instead of colliding with stale vectors.
- **Retrieval is automatic and degrades rather than fails.** A cheap count decides whether to search at all, a
  relevance floor decides whether to ground, and a store or embedder failure yields a normal ungrounded answer,
  never a dead stream.
- **The grounding block travels in the user turn**, not the system prompt, so the warmed system-prefix cache
  survives. `make warm` sends the same `num_ctx` as a real chat turn, for the same reason.
- **Citations stream after the answer, and only for what the answer used.** They are emitted after the answer
  tokens and cover only the passages the finished reply actually references - part of the grounding repair that
  followed Phase 3b - and they persist in message metadata, so a reloaded conversation keeps its sources.
- **Ingestion is idempotent per document** - delete-then-upsert clears stale chunks, so re-running an interrupted
  ingest heals it. The background worker's first real job is the session-vector TTL sweep, and deleting a
  conversation purges its session vectors immediately.
- **The reranker ships as an Infinity sidecar** behind an opt-in compose profile (`--profile reranker`), started
  and consulted on the one signal: whether `RERANKER_URL` names one. A deployment that wires no sidecar retrieves
  dense-only.
- **`OLLAMA_MAX_LOADED_MODELS` defaults to 2 on dev**, so the CPU-pinned embedder cannot evict the warmed chat
  model. Collection names gained a per-tenant unique constraint.

### Phase 3b - Web search & grounding

*A user can now: ask about things newer than the model's training data - the assistant searches the web when the
question needs it, answers with cited sources, and a composer toggle turns searching off entirely.*

A local model knows nothing after its training date. Web search closes that gap as just another retrieval source:
the model never browses, the platform does.

**Deliverables:**

- **A SearXNG sidecar** - a self-hosted metasearch engine that aggregates public engines behind one JSON API. No
  account, no API key, one container - sitting behind a `SearchProvider` seam so a hosted engine can replace it.
- **A local-first search policy** - `web_search: auto | on | off` on the chat API, default `auto`. Local model
  knowledge and RAG context answer by default; the network is consulted only when a deterministic policy detects
  freshness cues (recency wording, a year at or after the current one, an explicit ask to search). The same
  heuristic-now, orchestrator-later pattern as the reasoning policy.
- **Result handling that reuses Phase 3 end to end** - fetch the top result pages → extract → chunk →
  session-scoped `ingest_structured(source="web")` → cited injection through the existing `citation` events.
- **`SEARXNG_URL` plus the compose service.** Privacy is the toggle's job: `off` keeps every request fully local,
  while `auto` and `on` send only the query to public engines through the self-hosted instance, tied to no account.
- **[Phase 5](#phase-5---mcp-subsystem) later offers `web_search` to the model as a tool it can call itself** - an
  in-process local tool rather than an MCP one - so a tool-capable model decides mid-answer to search and writes its
  own query. This deterministic pre-fetch stays the small-model fallback, and it is what every non-tool mode keeps.

**Depends on:** 3, whose chunking, ephemeral collections and citations it reuses wholesale. **Parallelism:** HIGH -
the sidecar and the provider are self-contained; only the policy hook into the chat service is serial.

**How it landed.** Defensive page fetching (SSRF guards, byte/character caps, per-page degrade-to-snippet),
extraction, the context-aware search query, one-citation-per-canonical-source, result caching, and why the
sidecar ships ungated all live in
[ARCHITECTURE.md section 6](ARCHITECTURE.md#web-search-fresh-knowledge-local-first). The composer toggle for
web search arrives with the other per-conversation controls in [Phase 6](#phase-6---model-manager).

### Phase 4 - Multimodal attachments

*A user can now: attach the files they actually have - a PDF, a screenshot, a spreadsheet, a document, a book, a
source file - to a chat and ask questions about their contents.*

Files come *in*: uploaded, parsed, indexed, and answerable in the turn that sent them. Everything an uploaded file
"is" collapses to text flowing through the one existing `ingest_structured()` entrypoint, so attachments add no
second ingestion path.

**Deliverables:**

- **A `StorageBackend` seam** with a local-filesystem implementation, and an upload endpoint that validates by
  magic bytes rather than the extension a client claims.
- **A `DocumentParser` registry seam** where a new file type is a new parser module and one registry line, never a
  change to the pipeline.
- **The PDF and image pipeline** - pypdfium2 for the text layer, a low-text-density heuristic that routes a
  sparse (scanned) page to Tesseract OCR (optical character recognition: pulling text out of a picture of text),
  rendering only the pages that need it.
- **A capability table** seeded from Ollama's `/api/show`.
- **Background ingestion** - arq worker jobs with a status a client can follow, into session-scoped collections.
- **A frontend attachment tray** - pick, drop or paste a file, watch it upload, delete it.

**Breadth is the point.** A general-purpose assistant that rejects a `.py` file feels broken, so the registry
covers plain text and source code in any language, markdown, csv and tsv, json, yaml and xml, PDFs both
native-text and scanned, images, the Office formats, their OpenDocument equivalents, RTF, and EPUB.

**Parsing untrusted files is treated as hostile input:** size, page and pixel caps, zip-bomb and entity-expansion
defenses on every zip-of-XML format (Office, OpenDocument and EPUB alike), and a hard timeout around every native
parser - all of it in a worker rather than a request handler.

**Input only.** Phase 4 brings files *in*. Producing files - spreadsheets, PDFs, charts - is
[Phase 9a](#phase-9a---document-generation-tool-service).

**Depends on:** 3. **Parallelism:** MEDIUM - storage adapters, parsers and the capability table are separate; the
worker and progress integration is serial.

The phase was built as a long series of small slices, each its own pull request, so that a broken file format could
never take the pipeline down with it. What each slice delivered, and where it stands:

| # | What it delivered | Landed as | State |
|---|---|---|---|
| 1 | Somewhere safe to put uploaded bytes; a hostile filename cannot decide where they land | `c4b4d96` (#38) | Merged |
| 2 | That store is reachable by the running app and survives a rebuild | `3cc9b41` (#39) | Merged |
| 3 | Any text or source file, in any encoding, turns into text without garbling | `7cd06d3` (#40) | Merged |
| 4 | An upload has a life story: pending, parsing, ready or failed, with a reason | `60a18eb` (#41) | Merged |
| 5 | Files parse and index in the background, so the app never stalls on an upload | `2186f3b` (#42) | Merged |
| 6 | An upload endpoint: send a file, poll it to ready | `b9c4132` (#43) | Merged |
| 7 | Deleted files stop costing disk; stray bytes get collected | `618f02b` (#44) | Merged |
| 8 | Ask a question about the file just attached and get an answer from it, or an honest note saying why not | `b1c0e54` (#45) + `02e39a7` (#46) | Merged |
| 9 | PDFs read page by page, so a citation can point at "page 7" | `cb45cd9` (#47) | Merged |
| 9b | Scanned PDFs - pictures of text - become readable too | `4a7fe91` (#51) | Merged |
| 10 | Word, Excel and PowerPoint read: a deck by slide, a workbook by sheet | `861b3a1` (#50) | Merged |
| 11 | Upload an image and the app reads the text in it (a screenshot, a photo of a page) | `7e8339a` (#52) | Merged, OCR half only |
| 11 (rest) | The model actually seeing a picture that holds no text | - | Deferred to [Phase 5](#phase-5---mcp-subsystem); an honest notice is emitted instead |
| 12 | Attach files from the composer: pick, drop or paste, and watch them upload | `1fd0a95` (#48) | Merged |
| 12b | Files ride the question they were sent with, survive a reload, and delete for real | `d6e6704` (#49) | Merged |
| 13 | A single model tag doing chat, reasoning, tools and vision at once | - | Deferred to [Phase 5](#phase-5---mcp-subsystem) |
| 14 | Docs close-out: this roadmap records what each slice shipped | `632a3ea` (#54) | Merged |
| 15 | OpenDocument, `.rtf` and `.epub` read, closing the breadth list | `0db3714` (#61) | Merged |
| - | The MinIO/S3 storage adapter | - | Deferred out of this phase, then built: `STORAGE_BACKEND=s3` selects it |

The merge order was not the plan order: office formats landed before PDF OCR, which landed before images. That is
the registry seam working as designed - each format is an independent module, so the slices could be reordered
freely around whatever was ready.

#### After the slices: what live use found

The slices closed the phase; using it every day found the rest. These landed on `main` after the phase itself,
and they are fixes and hardening on the pipeline the slices built, not new slices of it.

**`459fcd0`
(#53) - a stale chip no longer forks the
conversation.** A missing attachment used to answer with the same generic code as a missing conversation, which
wrongly detached the pane into a new, history-less thread; cancelling an in-flight upload no longer orphans its
file, a permanently-missing upload stops spinning, and four request-path tuning values reach the container for
the first time.

**`63c85ca`
(#55) - a model stops refusing the image it is
already reading.** Text recognised in an image is now labelled as *text read from the image*, so a model that had
the words in front of it stops refusing on the grounds that it cannot see pictures; re-uploading the same bytes
twice now un-stages the earlier chip instead of deleting the original.

**`ce6b384`
(#56) - nothing is left waiting on a worker that
died.** Services now restart on failure (a kernel-killed worker used to stay dead, hanging every later upload
forever), and a periodic sweep fails any upload stalled past `ATTACHMENT_STALL_MINUTES` - but only once the queue
confirms no job for it still exists, so a row merely waiting behind a backlog is never condemned.

**`aa6d5ad`
(#57) - a file can be referred to after the turn
that carried it.** A later turn naming a file, or pointing at one unmistakably ("that pdf"), pulls it back onto
the question, files sent with no words at all become the question themselves, and the waiting line now says what
is actually being read instead of always claiming to search stored records.

**`b30cc93`
(#60) - the parse path tells the truth when a
reader dies.** A dead reader is now noticed the moment it dies rather than blamed for slowness at the deadline; a
reader killed for want of memory is retried instead of marked permanently broken; a row that failed mid-ingest
says so instead of spinning; a deleted file's indexed text is reclaimed even when its conversation's own cleanup
could not; and images now carry a pixel ceiling so a small file that decodes to hundreds of megabytes is refused
before it is unpacked.

**How it landed** (the decisions worth recording):

- **Pure-Python office readers, not docling** - MIT-licensed, no system libraries, reading an office file as the
  zipped XML it actually is. Why docling was rejected as the default is in
  [ARCHITECTURE.md section 9](ARCHITECTURE.md#9-multimodal-image--pdf-session-scoped).
- **Every native parser runs in a killable subprocess.** PDFium, lxml and Tesseract are native code that cannot be
  interrupted from asyncio or unwound from a thread, so a corrupt or hostile file that sends one into an unbounded
  loop would wedge the ingest worker forever. A shared isolation module puts each behind a spawned process with a
  hard timeout that can simply be terminated.
- **An office file is a ZIP of XML, which is two attack surfaces** - both closed before any library opens the file:
  a zip-bomb check that reads the archive's declared sizes without decompressing a byte, and a scan of every XML
  part for a DOCTYPE or entity declaration. That holds no matter how each library happens to configure its own
  parser.
- **Attachment vectors persist instead of expiring.** Parsed text first landed in the ephemeral session scope with
  the same TTL as throwaway web results, so roughly two hours after upload the periodic sweep deleted a file's
  vectors while its row still read "ready" and its bytes still sat in storage - the file silently stopped grounding
  answers. `ingest_structured()` grew a `persist` flag, so a file's searchable text now lives exactly as long as
  its bytes and its conversation.
- **Naming a file in a turn is a pointer, not a permission.** Session retrieval already found an uploaded file, so
  `attachment_ids` exists to let a turn POINT at one: it waits briefly for parsing to finish, bypasses the
  relevance floor (because "summarise this" shares no vocabulary with the document and would otherwise be thrown
  away as a weak neighbour), folds the file's passages back together under its own larger token budget, and raises
  a standing notice naming the file and the real reason - unavailable, missing, or no room - whenever the answer
  could not use it.
- **OCR makes scans and images answerable by a plain text model on every tier.** Rather than gate image and
  scanned-PDF support on a vision-capable chat model that no tier currently runs, sparse pages and uploaded images
  are rendered and recognised with Tesseract - the one system binary this pipeline can genuinely lack, so it is
  optional everywhere, and the capable path and the graceful-degradation path are the same code behind a single
  gate.
- **Recognised text has to say what it is.** OCR output reached the model labelled exactly like an uploaded source
  file, so the model dutifully searched it for "image data", found none, and refused a question it could already
  answer. A passage read out of an image now says that is what it is. The lesson generalises: a retrieved passage
  carries its provenance, because the model reasons about the label as readily as about the text.
- **A reference to a file is recognised in the words, not just in the tray.** The composer's tray empties on send,
  so a follow-up question reached retrieval with nothing attached - and a question *about* a file shares almost no
  vocabulary with the text *inside* it, so the relevance floor threw the passage away. Lowering that floor would
  weaken every other turn, so instead a message that names a file, or points at one unmistakably, pulls it back
  onto the turn. Unlike the reasoning and web-search decisions, this one is deliberately *not* handed to the model
  later: [Phase 5](#phase-5---mcp-subsystem) keeps it deterministic in every mode, because the honest notices that
  admit a file could not be used are raised from this very path (see
  [the attachment invariant](#the-attachment-invariant)).
- **Nothing may leave an upload in limbo.** A worker killed by the kernel stayed dead, because no base service had
  a restart policy, and an upload only ever leaves `pending` or `parsing` when the worker moves it - so every file
  attached afterwards spun forever on a row that still claimed to be in progress. Services restart on failure, and
  a periodic sweep closes rows that nothing will ever finish. Age alone never condemns a row: the sweep asks the
  queue whether a job for it still exists, so a row waiting its turn behind a backlog is left alone.
- **A dead reader is not a slow file, and a starved one is not a broken file.** The parse runs in a child process
  precisely so it can be killed, which means the code that waits on it must be able to tell "still working" from
  "already dead" - and it could not. A child that the kernel's out-of-memory killer took said nothing at all, so
  the parent waited out the entire deadline and then reported that the file took too long to read. It took no time
  at all: the reader died. The answer now comes back over a pipe whose only write end lives in the child, so any
  death, however sudden and however far into the reply, arrives as an end of file rather than as silence. And
  running out of memory is the one parse failure that says nothing about the bytes - something else on the machine
  took the memory - so it is retried, with a backoff long enough to outlast a model loading beside it, instead of
  marking a perfectly good file permanently unreadable.

**What remains open**, none of it blocking:

- **The MinIO/S3 storage adapter, deferred out of this phase.** Writing it here would have meant writing it
  against nothing: it is the cloud deployment that exercises object storage for real. *It has since landed* -
  `STORAGE_BACKEND=s3` builds an S3-compatible backend behind the same seam, and the whole storage contract
  suite runs against it.
- **A model that can actually SEE a picture**, deferred to [Phase 5](#phase-5---mcp-subsystem). Text in an image
  is read today, by OCR; a photograph that holds no text is stored, and the answer says so plainly rather than
  guessing. Closing it needs a vision model on the chat role, which no tier currently has.

### Phase 5 - MCP subsystem

*A user can now: watch the assistant decide for itself to search the web or the stored records, call an external
tool, and answer from what it found - or say plainly that no available tool fits, instead of guessing.*

Every phase so far built a pipeline. The platform decided, before the model produced a single token, whether to
search the web and which stored passages to hand over; the model only ever wrote prose. Phase 5 gives it a hand on
the wheel.

**What an MCP host is.** The Model Context Protocol (MCP) is an open standard for offering *tools* to a model, a
tool being a name, a description, and a JSON Schema saying what arguments it takes. A *server* offers tools; a
*host* discovers them, shows them to the model, runs the ones the model asks for, and feeds the results back. The
backend becomes the host. The mechanics are a loop: the question goes to the model together with the list of
tools; if the model answers, the turn is over; if it instead asks for a tool, the platform runs the tool, appends
the result to the conversation as a new message, and asks again. The loop ends when the model answers or a budget
runs out. That loop is the phase. Everything else in it is about what is safe to put inside the loop, and what
happens when the model gets it wrong.

**MCP is a source of tools, not the tool architecture.** This is the piece most easily gotten backwards, so it
comes first. The two tools that matter on day one - web search and knowledge search - are ordinary in-process
Python behind a transport-agnostic tool seam. They are not MCP servers and they cross no network. MCP is *one
source* that contributes tools to that seam, alongside the local one, and it joins by implementing a single
`ToolSource` interface. A sidecar speaking Streamable HTTP is exactly the right shape for a tool that wraps
somebody else's API, and a pointless network hop for code that already lives in the backend with the retrieval
pipeline in its hands. Building the seam first and fitting MCP into it second is what keeps that choice open per
tool, instead of paying for a protocol everywhere.

**Deliverables:**

- **The tool seam** - a tool, its context, its result, and a *source* of tools, composed by one registry
  ([ARCHITECTURE.md's tool seam section](ARCHITECTURE.md#the-tool-seam)). Every later tool, local or remote,
  arrives through it.
- **Tool calls across the `ModelProvider` seam** - an additive extension of the frozen Phase 1 contract: an
  assistant message can carry tool calls, and a `tool_result` chunk joins `citation` as a service-owned event.
- **A re-entrant generation loop**, bounded three ways and always ending in an answer
  ([ARCHITECTURE.md's host-loop section](ARCHITECTURE.md#the-host-loop)).
- **Web search and knowledge search as local tools** - the same pipelines the deterministic path uses, now
  *offered* to the model rather than run on its behalf.
- **The `mcp-chassis` sidecar** over Streamable HTTP, with a mock upstream API, one read-only tool, one
  side-effecting tool, and a mode that rejects an unauthenticated call.
- **Tenant scoping enforced by the platform, never by the model** - a tool's JSON Schema carries no tenant, user,
  conversation or session id, so another tenant's data is not expressible in a call.
- **Record results through `ingest_structured(source="mcp_records")`**, cited exactly like any other source.
- **Local tool results become cited sources too** ([slice 3c](#slices)): `web_search` was first shipped with its
  own bespoke fetch, outside the session cache and the `[n]` citation flow; routing it through the same
  search-and-ingest path instead closes that gap.
- **`tool_call` and `tool_result` SSE events**, and the tool card that renders them as a persisted activity trace
  (`meta.tool_steps`) - the approve/deny prompt for a side-effecting call folds into that same trace once
  answered, badge and all, so a reload shows the authorization exactly as the live turn did.
- **Tool-result injection defenses** - reused, not reimplemented, from the same fencing the retrieval path
  already gets ([ARCHITECTURE.md's "Tool results are untrusted input" section](ARCHITECTURE.md#tool-results-are-untrusted-input)).
- **Read-only versus side-effecting classification**, confirmed before a side-effecting call runs - a missing
  annotation counts as side-effecting, since MCP's own hints are advisory
  ([detail](ARCHITECTURE.md#read-only-versus-side-effecting)).
- **A tool-guidance prompt layer** - prefer a tool over a guess, never invent one, refuse honestly when none fits.
- **A `thinking_with_tools` capability answer, and the tool-check case that earns it** - measured, never assumed
  (see [the agency model](ARCHITECTURE.md#three-agency-modes)).
- **The tool-check harness and its `model_probes` measurement log** ([slice 3b](#slices)).
- **Tool selection under schema pressure** ([slice 4](#slices)) - top-K relevant tools per turn plus a small
  always-offered core, detailed in
  [ARCHITECTURE.md's "Tool selection under schema pressure" section](ARCHITECTURE.md#tool-selection-under-schema-pressure).
- **Honesty notices for the ways a tool turn can go wrong** - a tool that could not run, a spent budget, or an
  agentic turn that grounded on nothing.
- **Per-user credentials behind a `SecretStore`** - encrypted at rest, write-only, injected per call and never
  persisted, so one workspace's members connect their own accounts.
- **The two pieces deferred out of [Phase 4](#phase-4---multimodal-attachments)** - vision passthrough and the
  natively-multimodal single-tag pilot.

**A custom system prompt now overrides the persona, not the tool guidance.** `SYSTEM_PROMPT` used to be a verbatim
replacement for the entire system prompt. From this phase a custom value replaces the *persona* and the tool
guidance is always appended to it; an empty string still disables the system prompt completely, which remains a
deliberate way to run bare. The reason is that tool guidance is a capability contract rather than a voice: an
operator who rewrites the persona would otherwise be handed tool schemas by the API with no instruction about
honest refusal or about never inventing a tool, and tool calling would break invisibly, with no error to read.

#### The agency model

How much the model drives is a setting, not a fact about the code - three modes, `deterministic` (no tools, the
pre-tool pipeline exactly), `assisted` (that same pipeline, plus tools offered on top), and `agentic` (the model
drives; ambient web pre-fetch and knowledge-base sweep stand down). The mode a turn actually runs in is a
`min()` over independent claims - an operator's declared ceiling, the latest earned verdict from the tool-check
harness ([slice 3b](#slices)), the deployment's `AGENCY_MODE`, a per-request clamp (a developer affordance, never
a composer control), and runtime floors (an empty tool set, no room for a tool result, or a reasoning turn on a
model that cannot reason and call tools at once) - so nothing can ever escalate its own privileges. `main` ships
`AGENCY_MODE=agentic`, with `assisted` as the grace floor a fresh clone starts at and `deterministic` the full
kill switch. The complete arbitration chain, including how a declaration alone only ever buys `assisted` and what
a passing deep check adds, lives in
[ARCHITECTURE.md's Three agency modes](ARCHITECTURE.md#three-agency-modes).

#### The attachment invariant

**A file the user attached is fed to the model in every mode, including agentic - the model is never asked to
decide whether to read it.** Attaching a file and pressing send is an explicit instruction about *this* turn, and
the honesty notices that admit a picture held no readable text or a promised file did not fit both depend on that
same forced path - routing files through a tool the model may forget to call would retire the honesty along with
it. Full detail (why it survives every mode, and why session retrieval survives agentic mode alongside it) is in
[ARCHITECTURE.md's tool section](ARCHITECTURE.md#three-agency-modes).

#### Reasoning with tools is a model capability, not a platform rule

Whether a model can reason *and* call tools in the same turn is a fact about that model **as served by a given
runtime**, measured and declared rather than assumed - `thinking_with_tools`, one of three per-model questions
tracked separately (`tools` mechanical, `agency` policy, `thinking_with_tools` capability). The default is
`false`, because the optimistic guess fails silently: qwen3 through the current Ollama returns *empty output*,
not an error, when reasoning and tools are requested together
([ollama/ollama#10976](https://github.com/ollama/ollama/issues/10976)). A model earns `true` only from a passing
deep [tool-check](#slices) case that asserts both a well-formed tool call and non-empty output, and the exclusion
is a workaround for that one bug - deleted the moment a (model, runtime) pair does not need it. The composer's
reasoning toggle therefore means what it says on a model that can do both; on one that cannot, arbitration falls
to the rule in
[ARCHITECTURE.md's "Whether reasoning and tools can both be live" section](ARCHITECTURE.md#whether-reasoning-and-tools-can-both-be-live-is-a-per-model-fact-measured-then-declared).

#### Why there is no model ranking and no boot-time probe

Scoring models and letting only the good ones drive was considered and rejected: parameter count is disqualified
empirically (a tool-tuned 3B model outscores a generic 4B on public function-calling benchmarks, especially on
multi-turn cases), a boot-time probe on a card that fits one model evicts the very model it is about to serve, and
a vendor capability flag carries no information once every model in play reports the same one. What earns a
model's agency ceiling instead is `make tool-check`, an opt-in diagnostic that never runs at boot or on a chat
turn - the rationale and the full case list live in
[ARCHITECTURE.md's tool-check harness section](ARCHITECTURE.md#the-tool-check-harness-and-the-model_probes-measurement-log).

#### Slices

The phase is built as a series of small pull requests, in the same way [Phase 4](#phase-4---multimodal-attachments)
was. What each slice delivers, and where it stands:

| # | What it delivers | Landed as | State |
|---|---|---|---|
| 0 | The runtime pin bump (measured), and a throwaway spike of how the model puts a tool call on the wire | #68 | Merged (spike throwaway) |
| 1 | The tool seam, the registry and the agency policy - wired to nothing, so behaviour is unchanged | `ad31a4f` (#67) | Merged |
| 2a | Tool calls carried across the `ModelProvider` seam, in both directions | #69 | Merged |
| 2b | The generation loop is reshaped to re-enter safely: the terminal chunk is caught and finalized once, below the stream (behaviour unchanged) | #70 | Merged |
| 2c | The loop runs a tool the model asked for, feeds the result back, and streams again | #71 | Merged |
| 3 | Web search and knowledge search offered to the model as tools - the first behaviour change | #74, #75, #76, #77 | Merged |
| 3b | The tool-check harness, the `model_probes` measurement log that keeps its verdicts, and the earned-verdict wiring that makes the chat path read them | #79, #80, #83 | Merged |
| 3c | Tool results become cited sources: `web_search` reuses the fetch-and-ingest pipeline (session cache, canonical URLs), and tool-grounded answers emit `citation` events like every other answer | #78 | Merged |
| 4 | The Streamable-HTTP spike, the MCP client and a real `mcp-chassis` sidecar, and top-K tool selection | #86 | Merged |
| 4b | Records from a tool server become cited sources: a wrapper returns structured records, which flow through the ingest pipeline into the conversation's session scope | `3b60aaa` (#87) | Merged |
| 5 | Side-effecting tools ask before they act: the turn suspends on the call and resumes on the user's approve or deny | `e06c64f` (#89) | Merged |
| 6a | Encrypted secret storage, the credential schema, and user-scoped repositories | #72, #73 | Merged |
| 6b | Per-user connections to a tool server, with the credential injected per call | #94, #95, #96, #97, #98, #99 | Merged - DB-backed registry, Integrations UI, per-call credential injection |
| 7 | The tool card in the UI: the activity trace (each call, its result, the model's narration), the approve/deny prompt for a side-effecting call folded into that trace once decided, and the whole trace persisted so a reload replays it | #90, #91, #92, #93 | Merged |
| 8 | Vision passthrough, and the single-tag multimodal pilot inherited from Phase 4 | #100, #102, #103, #104, #105, #106 | Merged, pilot included - mechanism in [ARCHITECTURE.md section 9](ARCHITECTURE.md#9-multimodal-image--pdf-session-scoped); verdict: the tag does chat, thinking, tools and vision, but only about a third fits a 4GB card, so a small card keeps its model pair |
| 9 | OAuth, so connecting an account is a consent screen rather than a pasted token | #111, #112, #113, #114, #115, #116 | Merged - the full flow (discovery, PKCE, sealed grants, renewal) is in [ARCHITECTURE.md's "Connecting an account" section](ARCHITECTURE.md#connecting-an-account); workspace-scoped only, per-user waits for [Phase 8b](#phase-8b---identity--access) |
| 10 | The first real integrations: GitHub, mail, and a widening catalog | #117, #118, #119, #122 | Merged: GitHub and Gmail presets first, widened to Google Drive, Slack and Asana in a follow-up (design in [ARCHITECTURE.md's "Presets are data" section](ARCHITECTURE.md#presets-are-data-not-integration-code)) |
| 11 | Docs close-out, and the security review this phase has always been scheduled to get | #126, #127, #128 | Merged - see [what the security review found](#what-the-security-review-found) |

**Slice 0 opens with three prerequisites, in order.** First, the Ollama image pin in
[`infra/docker-compose.yml`](../infra/docker-compose.yml) moves 0.30.7 -> 0.32.0, and `make perf` runs against the
incumbent model *before anything else changes* - 0.32.0 changed how the whole qwen3 family is matched to its chat
templates, so the bump must be measured alone rather than discovered inside the spike's noise. Second, every test
pull comes from the Ollama library, **never** from an `hf.co/...` path: HuggingFace's synthesized manifests carry
no renderer/parser metadata, so a model pulled that way emits its tool calls as prose with an empty `tool_calls`
array and no error anywhere - the spike would measure a broken packaging job and blame the model. Third, two
20-minute empirical checks close the remaining unknowns: an A/B of the same weights pulled from `hf.co` versus
the library, and confirming that the installed-models listing reports the `digest` field the measurement log
(slice 3b, below) is keyed on.

**Slices 0 through 3 are built first, and then the phase deliberately pauses.** They are the whole premise: a seam,
a loop, and two tools to put in it. Whether a small local model can actually drive that loop is an empirical
question that no amount of design settles, so the phase stops there for a period of live use before the rest is
committed to.

What that use finds decides the rest. If the model calls tools well, the remaining slices continue as listed. If it
calls them badly, the declared ceiling drops - one declaration, not a rewrite - and the agentic drive moves to a
larger machine. Little is wasted either way: the seam, the loop, the budgets and the honesty notices are the same
in every mode, and `assisted` keeps the old behaviour underneath them.

One scope note on what that live use proves: a small development box rehearses the host loop, the budgets, the
citations and the confirmation flow - everything above the provider seam. The wire-level tool path is a
per-(model, runtime) fact, verified per deployment by the harness below - which is why `runtime_version` is part
of the probe key.

**Slice 3b lands at the end of that pause, before slice 4, and turns the live-use verdict into a measurement.**
The tool-check harness is a pure service with a CLI entry point; the Model Manager's verify job
([Phase 6](#phase-6---model-manager)) later wraps the same service, so a terminal run and a UI click cannot
disagree. Verdicts persist in `model_probes`, a measurement log keyed
`(provider, model, digest, runtime_version, serving_endpoint, harness_version, depth)` that a re-pull, a runtime
upgrade, or a different serving host each invalidate in their own way - never on a schedule. The two depths, the
invalidation rules, and the two locked rules it deliberately breaks (signed off by the owner 2026-07-15) are
detailed in
[ARCHITECTURE.md's tool-check harness section](ARCHITECTURE.md#the-tool-check-harness-and-the-model_probes-measurement-log).

Slice 6a is the exception to the ordering. It imports nothing from MCP and needs no tool loop, so it runs alongside
slices 0 to 3 rather than queueing behind them - and it carries the highest-value security work in the phase, which
is worth having reviewed early rather than late.

Slice 4 leads with a spike of its own: mounting an MCP server's Streamable HTTP transport under the backend is the
one genuine unknown left in the phase. It sits at the head of slice 4 rather than in slice 0 because it blocks
nothing in slices 0 to 3, and because the MCP SDK's next major version is a breaking revision landing within weeks
of this being written - spiking the transport early would only mean spiking it twice.

Slice 4 also brings **tool selection under schema pressure**, because it is the slice that multiplies the tool
count: the moment MCP servers join, the offered set can grow past what a small model can hold. The tools are
ranked against the turn's message, the top-K are offered, and a small core - web search, knowledge search - is
always offered regardless of rank. Ranking happens in process over cached description vectors, and it does not
run at all below a configured tool count, so a deployment with a handful of tools pays nothing for it; the design,
and why a vector store would buy nothing here, is in [ARCHITECTURE.md](ARCHITECTURE.md#tool-selection-under-schema-pressure).

**Slice 4b is deliberately split out of slice 4.** Making a tool server's records citable routes them through the
same `ingest_structured()` pipeline a web page uses, which lands a document embed on the turn's critical path -
the same shape of cost that took [slice 3c](#slices)'s web tool from a 15s first token to 50s - so it earned its
own measurement rather than folding into the slice that introduces the transport. The structured-content contract
that lets a wrapper's records arrive as records rather than formatted prose is in
[ARCHITECTURE.md's "MCP servers as a tool source" section](ARCHITECTURE.md#mcp-servers-as-a-tool-source).

**Slice 5 turns "ask before acting" from a drop into a real question.** A read-only tool runs on sight; a
side-effecting one suspends the turn instead of running, checkpointing it onto the conversation so the client can
ask and the answer can resume the same turn later, even across a disconnect. The full checkpoint mechanism (the
row lock against double-approval, the TTL, the client-capability gate) is in
[ARCHITECTURE.md's "Read-only versus side-effecting" section](ARCHITECTURE.md#read-only-versus-side-effecting).

#### Performance: a tool turn is slower, by construction

An agentic tool turn cannot be as fast as today's. Before the first word of the answer exists, the model must
decide to call a tool and write the call out - a complete round trip through the model that today's turns do not
pay. Time-to-first-token therefore regresses on any turn that uses a tool, and merely *offering* tools costs a
little prefill on every turn, because the schemas ride in the prompt. Both are named and accepted rather than
discovered later.

They are defensible on one condition, and it is the condition that makes the whole design honest: **`deterministic`
mode must reproduce the pre-tool numbers exactly.** That is the control reading. For the slices that touch the
turn, three measurements are taken on the same machine - `main`, the branch, and the branch clamped to
`deterministic` - and the third must land within noise of the first. If it does not, the cost is not the price of
the feature; it is a bug in the policy layer, and it blocks the merge. `assisted` is measured for the same reason:
a model that never calls a tool must cost nothing beyond the schema prefill.

The harness needs one addition before any of that can be measured. Its only default scenario is a plain counting
prompt, which calls no tool at all - so a tool-exercising scenario (`PERF_TOOL=1`) ships with the slice that first
offers a tool, in the same pull request. Without it the three readings above measure nothing that uses a tool, and
the gate is blind to the exact regression this phase exists to introduce.

One measurement trap is specific to this phase. A turn with tools and a turn without them are *different prompt
prefixes* for the model, so the first tool turn after a warm-up pays a cold prefill that has nothing to do with the
change being measured. Each scenario is run twice, and the second reading is the one that counts.

**Depends on:** 4. **Parallelism:** MEDIUM-HIGH - the sidecar, the frontend card and the credential slice are
separate codebases behind a frozen contract; the generation loop is serial and wants a single owner. **The security
review lands here**, and retroactively over [Phase 2](#phase-2---persistence--multi-tenant-schema): this is the
phase that puts a tenant boundary, live third-party credentials, and attacker-controlled tool output into the same
request. It is also worth stating plainly that until real authentication is configured (the
[Phase 8b](#phase-8b---identity--access) OIDC cutover ships it, gated on `OIDC_ISSUER`), every caller is the same
seeded default user - so a deployment holding live integration credentials with sign-in left off must not be
exposed beyond the machine it runs on.

#### What the security review found

The review ran over the whole phase in six passes - the tenant boundary, credentials at rest and in flight, the
OAuth flow, prompt injection through attacker-controlled tool output, request forgery and the write surface, and
the tool loop's own limits - and each finding was then re-checked by an independent pass whose job was to refute
it. What it recorded is kept here rather than in a private note, because the accepted risks are decisions and a
decision nobody wrote down gets re-litigated or, worse, quietly forgotten.

**The mechanisms held.** Each ciphertext is bound to the row that owns it, so a wrong-row read fails loudly rather
than returning somebody else's token; the secret seam cannot express a cross-user lookup at all; the consent flow's
state is single-use and its verifier sealed; the confirmation gate decides against a frozen checkpoint rather than
a live re-listing, so a server cannot reclassify a tool between the question and the answer; and no endpoint exists
through which a stored credential can be read back.

**What it found was one shape, repeated: remote input crossing into the process unbounded.** Tool results were
measured against the turn's context budget only after being spent, which could push the prompt past the window -
and the runtime discards from the head, where the system prompt and its injection defenses live. A hostile server's
tool schema rode into that same system region with no size limit at all. Nothing capped how many bytes a remote
host could send, on any path. Alongside those sat one lifecycle hole: deleting an integration never revoked its
grant, and the secrets are keyed on the id and URL that the deletion destroys - so a live third-party token
survived where no part of the app could ever list it again. All of it is fixed.

**Accepted, with reasons, rather than fixed:**

- **An administrator may register a server whose URL points inside the network.** That is the documented tradeoff
  already: the surface is gated by the admin role and logged with an actor, not by a URL allow-list, because a
  legitimate deployment's tool servers usually *are* internal. What was treated as a defect is the *dial-time*
  gap - a redirect could re-point a call after registration - and that is closed.
- **The connected account (an email address on the Google presets) is visible to every member.** The grant is
  workspace-scoped, so every member already spends that account's authority on every call. Hiding whose account it
  is would make a shared credential less accountable, not safer.
- **A wholesale `config` PATCH can clear the per-tool deny-list.** No shipped path sends one, and a side-effecting
  tool still stops for confirmation even if it were re-offered. It is pinned to the settings work that will own
  that editor.

**Two findings were latent until real login, and were fixed with it in
[Phase 8b](#phase-8b---identity--access).** The vector layer filtered a session scope by tenant and not by user,
which left the router-level ownership check added here as the only boundary between two members of one workspace;
that scope is now filtered by user as well. And a single long tool turn could exhaust a whole workspace's
concurrent-stream budget - harmless while one person starves only themselves, and one member starving colleagues
the moment there is more than one - so the budget became per user instead.

### Phase 6 - Model Manager

*A user can now: browse, pull, delete and verify local models from the UI, and pick the model, thinking mode and
web-search mode per conversation from the composer.*

Model choices become visible and switchable, and the per-conversation controls designed in earlier phases finally
get their surface.

**The one prerequisite named here - a tag-first capability lookup that did not depend on which hardware class
bound a tag - was already delivered during Phase 5** by the flat-catalog restructure
(#107), so a model switched in from another
class no longer lost its declared context length (the failure that motivated it - an unsized model silently
truncating the prompt). What remained for this phase was presentation work, not the lookup itself.
[Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts) later removed the class distinction entirely, and
with it the question this prerequisite answered.

**Deliverables:**

- **A list/pull/delete proxy over Ollama**, originally merged against the committed model catalog of the time (its
  capability flags and a name allowlist) as a request-time join with no new table - later retired and inverted to
  a pure installed-inventory join by [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts).
- **Management verbs come from the provider, so the page cannot show a dead button** - `capabilities()` grows
  `can_pull` / `can_delete` / `can_list_installed`, each row deriving its buttons from those flags, grouped by
  role (chat manageable, embedding locked, reranker a status-only row). See
  [ARCHITECTURE.md section 11.4](ARCHITECTURE.md#114-the-serving-choices-behind-all-of-it).
- **A Model Manager page** listing the catalog of the time, installed-but-uncatalogued tags, and a free-text pull
  box pre-flighted against the registry manifest - `hf.co/...` reachable only there, behind a warning, clamped to
  `deterministic` until a human declares otherwise. (Amended by
  [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts): with no catalog left, the page is a pure
  inventory of what the model server holds.)
- **Pull progress streamed over SSE.** The download is content-addressed and resumable, so a dropped stream costs
  a re-click, not a re-download.
- **A successful pull auto-runs the fast tool check; the deep check is one click** (both wrap the
  [Phase 5 slice 3b](#phase-5---mcp-subsystem) harness) - never at boot, never on the chat path, at most one probe
  at a time; the ordering (unload, measure, re-warm) is in
  [ARCHITECTURE.md's verify-job section](ARCHITECTURE.md#the-verify-job-the-order-a-probe-has-to-happen-in).
- **The embedding role is locked in the UI**, because changing it silently orphans every stored vector - the
  **stranded-vectors alarm** ships instead (mechanism in
  [ARCHITECTURE.md section 6](ARCHITECTURE.md#6-retrieval-rag-design)).
- **Fit is observed, not predicted** - manifest download size before a pull, actual GPU residency after a load,
  never a computed estimate (a predicted-fit pill was cut after proving about 4.5x wrong on the project's own
  card; see [ARCHITECTURE.md section 11.3](ARCHITECTURE.md#113-hardware-is-expressed-by-what-is-pulled-and-how-wide-a-turn-may-be)).
- **A failed or degraded load explains itself** - a model resident on CPU after a load, a pull that failed
  preflight, a probe that could not get GPU residency - all turned into actionable text from the same observed
  facts as everything else, never a guess.
- **Catalog breadth grows by review, not discovery** - the committed catalog carried a small set of curated
  alternates per hardware class and role, listed like any other entry. (Amended by
  [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts): with the page inverted to installed-first,
  there is no list for breadth to be added to.)
- **The model resolution chain** - request > conversation setting > workspace default > env - with narrow typed
  conversation-settings fields, because the wholesale settings replace of the time would let two composer toggles
  clobber each other. Conversation settings were written but never read; the chain is the fix.
- **Composer controls in the chat page** - the capability-gated model selector, a two-state thinking toggle (Auto
  by default, or On), and a web-search toggle enabled by default (the local-first `auto` policy) that switches off
  to fully local.
- **UI ceiling declarations are gated by a passing check, not by a cap** (revised 2026-08-08) - an earlier plan
  capping console declarations at `assisted` until real sign-in existed was retired, since it protected one verb
  on a console whose every other verb already sat behind the same unauthenticated gate. The **test-first** rule
  that actually holds (`min(ceiling, earned)`, never escalating) is
  [the agency model](ARCHITECTURE.md#three-agency-modes). See [Phase 8b](#phase-8b---identity--access) for the
  sign-in work that makes "admin" name a person.

#### Slices

The phase is built as a series of small pull requests, like [Phase 5](#phase-5---mcp-subsystem) before it. The
first two frontend slices come from the design system's build order ([IMPLEMENTATION.md](design/IMPLEMENTATION.md)),
so every page this phase adds is born themed. Slices 1 and 3 touch different apps and can proceed in parallel.

| # | What it delivers | Landed as | State |
|---|---|---|---|
| 1 | The theme token foundation and theme picker, shipping the aurora (default) and minimal themes | #132 | Merged |
| 2 | The chat page split into components with zero visual change, so the composer controls later in the phase are small edits rather than surgery | #133 | Merged |
| 2b | The chat surface restyled to prototype fidelity on aurora and minimal | #136, #137 | Merged |
| 2c | Conversation management: rename in place, a per-row actions menu, a confirmation dialog, and model-written titles | #138, #139, #140, #141, #142, #143, #146 | Merged |
| 2d | Message actions: copy on any turn, and re-asking the newest question in place of its reply | #148 | Merged |
| 2e | Rewriting a turn: editing a question in place, regenerating any reply, and replying to a quoted passage | #149 | Merged |
| 2f | Pinned conversations, ordered ahead of recency in the sidebar | #157 | Merged |
| 3 | `plan_turn()`: the scattered per-turn decisions (model, thinking, tools, budgets) consolidated into one inspectable plan object, behaviour byte-identical | #131 | Merged |
| 4 | The model resolution chain made real: typed per-conversation settings the turn actually reads, workspace defaults above the environment | #134, #135 | Merged |
| 5 | The Model Manager read API: provider management verbs, and the request-time join of catalog, installed inventory, measured probe facts, and why-a-model-is-clamped | #152 | Merged |
| 6 | Pull and delete: a streamed pull with registry-manifest preflight, delete behind the seam, and the free-text pull box with its warnings and safe clamps | #159 | Merged - endpoints and preflight; the box that calls them lands with the page in slice 8 |
| 7 | The verify job: the tool-check harness behind an endpoint - one probe at a time, deliberate unload first, GPU-residency assertion, re-warm chat and embedder after | #161 | Merged - a CPU-spilled run is recorded as unrepresentative (clamps, never certifies); a successful pull auto-runs a fast check |
| 8 | The Model Manager page: the role-grouped grid, live pull progress, measured facts with an honest "unmeasured", and failure text that explains itself | #164 | Merged - with an admin console shell at `/console`, live resident-model status, and a second pass (pause/resume downloads, a narrowed per-row embedding lock) |
| 9 | Composer controls: the model selector (chat-role models only), the thinking toggle, and the web-search toggle, wired to per-conversation settings | #153 | Merged |
| 9b | Installed alternates in the picker: every installed, catalogued chat model becomes pickable, not only the active profile's | #156 | Merged |
| 10 | The stranded-vectors alarm: space enumeration on the vector-store seam, embedder provenance sampled from stored chunks, and the standing notice | #158 | Merged |

**Slice detail, for the cells above trimmed to a summary:**

- **Slice 1** also shipped the no-flash boot script; the rest of the theme catalog follows in
  [Phase 7](#phase-7---frontend-polish--theming).
- **Slice 2b** covers type hierarchy, tool-step anatomy, jump-to-latest, and the auto-growing composer, plus a
  follow-up bringing the integrations page onto the same system.
- **Slice 2c** also added humanized tool-activity labels and lifecycle fixes shaken out by live use.
- **Slice 2d**'s re-ask is a third request shape on the chat contract - a transactional swap, guarded on the
  question it answers still existing.
- **Slice 2e**'s regenerate names what it discards first, and a hand-renamed thread's title provenance means it
  is never machine-renamed afterward.
- **Slice 2f** adds a Pinned group above Recent (newest pin first, no cap), pin/unpin from the row menu, and a
  `pinned_at` column so a pin can never fall off the sidebar's first page.
- **Slice 4** landed as the read chain plus a follow-up that patches conversation settings per field, so two
  composer controls saving at once cannot clobber each other.
- **Slice 9b** also has a pair entry name its thinking sibling in its hint.

**Two deliverables above did not land here, both deliberately, and both since superseded by
[Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts):**

- **Ceiling declarations from the UI** originally waited for the admin role ([Phase 8b](#phase-8b---identity--access));
  Phase 6c shipped the flow early instead, since this console already granted install/delete at the same
  unauthenticated reach and a remote inventory needs declarations to lift a model off the floor at all.
- **Curated alternates in the catalog** originally meant an edit to the committed catalog file per reviewed
  recipe; Phase 6c retired that file into a quirk seed, so breadth stopped being a catalog question.

**A fix from live use closed the phase.** A slow-but-healthy model load was being aborted at five minutes by two
stacked ceilings - the runtime's own load timeout and the backend's read timeout - which on a contended card
turned a load that would have succeeded into a failure the model got blamed for. Both are now configurable, the
client's deliberately outlasting the server's so the server reports its own failures, and a test pins that
ordering across the two files that set them
(#165).

**Depends on:** 1, plus [Phase 5](#phase-5---mcp-subsystem) slice 3b for the verify flow; the web-search toggle
needs [Phase 3b](#phase-3b---web-search--grounding). **Parallelism:** HIGH - independent of the RAG and MCP work
apart from that one slice, so it can run alongside Phases 3 through 5.

### Phase 6b - Settings surface v1

*A user can now: set workspace defaults (model, thinking mode, system prompt), register tool servers, and enter
upstream secrets from a Settings page.*

The Phase 2 settings schema finally gets a face.

**Deliverables:**

- **Workspace defaults** - default model, default thinking mode, default web-search mode, a workspace system
  prompt. From here on, day-to-day behavior tweaks belong on this page: the resolution chain reads workspace
  settings above the environment, so `.env` recedes to genuine deployment configuration - URLs, keys, tier
  selection - rather than something to edit and restart for a default change.
  *The model/thinking/web-search defaults are pulled forward into
  [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts) (slice R4); the workspace system prompt landed
  with #232 - a stored persona wins over the
  environment's `SYSTEM_PROMPT`, an emptied field falls back rather than silencing anything, and the
  tool-guidance layer is appended to whichever persona wins. The rest below stays here.*
- **A session-specific system-prompt override.**
- **The Integrations tab** - the MCP server registry grown into a real surface. An MCP server describes itself:
  the host connects and asks what tools it offers, so one generic form covers any integration, with no
  per-integration UI code.
  *This shipped early under [Phase 5](#phase-5---mcp-subsystem) slices 6b and 10 - the add-server form,
  write-only credentials, test-connection and the preset gallery are all live on the console today; what
  remains below it is the per-tool enable map and the health line.* Credential scope and auth type are covered
  in [ARCHITECTURE.md's "Per-user credentials" section](ARCHITECTURE.md#per-user-credentials); presets in
  [its "Presets are data" section](ARCHITECTURE.md#presets-are-data-not-integration-code). Concretely:
  - **Add server** - name, base URL, auth type, and credential scope (workspace-shared or per-user). A
    registered server can be a sidecar, a vendor-hosted endpoint, or any Streamable-HTTP process; one server can
    wrap several upstream systems, and the converse also works - one container can mount several registered
    servers at distinct paths, each keeping its own credential scope and per-tool toggles.
  - **Test connection** - one button, one MCP handshake plus a tool listing (read-only versus side-effecting);
    it never invokes a tool, and failures surface as typed errors (unreachable, auth rejected, protocol
    mismatch).
  - **Per-tool enable toggles and a health line** (last successful call, standing error banner), living in the
    `config` JSON `mcp_servers` has carried since Phase 2 (see [DATA-MODEL.md](DATA-MODEL.md)).
  - **A short setup guide ships with the tab**, distinguishing a server that already speaks MCP from an
    arbitrary API that needs one embedded or fronted by a translating sidecar (the reference chassis under
    `services/`).
  - **The registry is authoritative**: environment variables pointing at bundled servers (`MCP_CHASSIS_URL`)
    recede to bootstrap seeds, and every registration is logged with its actor.
- **Write-only secrets entry.** Values are AES-GCM-encrypted at rest behind the `SecretStore` seam, the API returns
  only metadata plus the last 4 characters, and secrets are injected into MCP sidecars per request, never persisted
  there.
- Role gating (admin versus member) is not built here, but its call sites are: the admin surfaces this page adds
  are wrapped in a `require_role` dependency that passes for everyone until real identity claims exist, so
  [Phase 8b](#phase-8b---identity--access) turns enforcement on without touching a call site.

**Depends on:** 5, so that there is something real to configure, plus the Phase 2 settings schema. **Parallelism:**
HIGH - a separate page against a frozen API surface, so it runs alongside Phase 7.

### Phase 6c - Model-layer rework & remote model hosts

*A user can now: point the platform at any Ollama server - the local container or a cloud-hosted endpoint - and
see every installed model listed, pickable and honestly described; an admin manages what the server actually
holds instead of what a configuration file predicted.*

The tier system predated the platform's hardware-agnostic turn: `MODEL_TIER` selected a profile in a committed
model catalog that answered both a genuinely machine-scoped question (which hardware knobs apply) and *which
models run* - so pointing `OLLAMA_URL` at a healthy remote server with a different inventory produced a working
connection and a broken deployment (stored picks silently discarded, installed-but-uncatalogued models invisible
to the composer, an unlisted tag read as a total capability blackout). This phase replaces the catalog with
evidence resolved per key - installed inventory, the runtime's own capability report, measured checks, and admin
declarations - and retires it into a small code-shipped seed of quirks no runtime can report. The full resolution
design is [ARCHITECTURE.md section 11](ARCHITECTURE.md#11-models-what-runs-and-what-is-known-about-them).

#### Slices

The same small-PR discipline as Phases 4 through 6:

| # | What it delivers | PR | State |
|---|---|---|---|
| R0 | Remote enablement, security first: authenticated outbound calls, redirect detection (never followed), and a `remote` profile that starts no local model container | #173 | Merged |
| R1 | The picker and the resolution chain fixed together: installed models pickable regardless of the catalog, and a context window read from the runtime, never unknown | #175 + #177 + #180 | Merged |
| R2 | Capabilities from evidence: the runtime's capability report drives tools and vision for uncatalogued tags, with cards enriched from measured facts | #186 + #187 + #188 + #190 | Merged |
| R3 | The ledger and the policy store: the known-models quirk ledger, and a digest-bound `model_policy` table for admin declarations | #195 + #201 + #202 | Merged |
| R4 | The workspace settings writer - the defaults half of [Phase 6b](#phase-6b---settings-surface-v1), pulled forward | #209 | Merged |
| K1 | The knowledge console: browser uploads into the shared knowledge base, listing with provenance, and remove | #235 | Console half merged |
| R5 | The manager and picker reworked: installed-first, set-as-default as a row action, and agency/thinking-pair declarations from the UI | #182 + #215 | Merged |
| R6 | The tier retired: `MODEL_TIER`, `MODELS_YAML_PATH`, the catalog file and the never-wired vision role removed | #230 + #233 + #234 | Merged |
| R7 | Per-role model access groundwork: the grey areas decided before any schema - what a blocked user is told, how an unavailable pinned model falls through, whether reading history is ever gated | - | Not started |
| R8 | Measurements only a live remote can settle - probe records now carry the serving endpoint | #200 | Merged |

**Slice detail, for the cells above trimmed to a summary:**

- **R0** also added a typed `provider_unauthorized` error, `env-check` guardrails, and a committed reverse-proxy
  recipe for bare Ollama.
- **R1** exists because fixing only the list would still have accepted picks the turn itself ignored; it also
  added a typed first-run state and substitution notices.
- **R2** infers reasoning mode from the template, adds a role-gap card with a one-click suggested pull, and
  enriches cards with quantization, parameter size and the true context ceiling; the runtime pin advanced with a
  before-and-after measurement, and a measured verdict always outranks the report. Shipped in four pieces:
  inventory-truth surfaces and role-gap cards, the runtime pin advance, a weights-change warning (a re-pull
  orphans a verdict, surfaced on the banner, row pill and card), and the capability-report core landing last.
- **R3** also re-anchors the earned-agency grace on the automatic post-pull check, adds an embedder check that
  proposes retrieval floors from measured score bands, and a thinking check that classifies each model's real
  reasoning mode.
- **R4** covers default model, thinking and web-search defaults, a context-window ceiling, and the reranker
  toggle - all read by the turn, the warm and the verify re-warm through one row-locked store, with a console
  Defaults page to edit them.
- **K1** is one long synchronous request that keeps no file bytes (why re-ingest is not offered yet); the guarded
  embedder switch (stranded-vector preflight plus a background re-embed job) needs a stored-documents table
  first and remains open.
- **R5** adds a pre-pull look-up reporting size/quantization/parameter count, agency declarations behind the
  test-first gate with plain-language glosses, thinking-pair declarations on the cards (candidates drawn only
  from measured modes, with the composer's thinking toggle disabled - and saying why - when none is declared),
  vision badges plus an attach-time notice, the loaded-now GPU-share readout, and the standing banner over an
  unauthenticated remote endpoint. Three-tier delete protection (refuse the deployment pair, reassign the
  default with a hard guard, warn on pins) shipped early.
- **R6** also retired `make pull-models` and `make init PRESET=`, added `RERANKER_MODEL` as the deployment
  config a rerank call needs once nothing catalogues it, dropped the API's tier field and regenerated types, and
  swept every stale doc reference. Groundwork shipped ahead of the sweep: default context window and
  embedder/reranker choices re-homed to deployment config, and reasoning-with-tools declarable from the console
  behind the agency test-first gate.
- **R8**'s code half means a verdict from one machine cannot describe another (rows written before the column,
  or on another server, stop certifying - failure evidence still clamps). The proxy-timeout ceilings against
  long cold loads remain a live-use measurement, not a code change, and stay open.

Two follow-ons ride behind the phase rather than inside it:

- **History compaction.** The context-usage meter lands with R1 (the turn already computes the numbers), while
  the compaction action itself - summarising a long conversation's older turns so the window holds what
  matters - is its own later slice, since it touches prompt composition and pays a generation.
- **An optional routing orchestrator.** The `auto` decisions (reasoning, web search) are deterministic
  heuristics today; a deployment with headroom may configure a small chat model to make those calls per turn
  instead, reading the actual message and history. No new role: any installed chat-capable model qualifies,
  named by ONE nullable workspace setting (its writer is R4's, which is why this waits behind it). Unset - or
  unreachable, or answering garbage - falls back to the heuristics for that turn, so the feature can only ever
  add judgment, never take a turn down. The Model Manager marks the assigned model as in use the same way it
  marks the workspace default (deleting it warns and clears the setting rather than refusing - the fallback is
  the working heuristic, not a broken deployment). Adoption is measured before it is trusted: a small
  agreement harness scores the candidate against labelled prompts, and the cost is honest - one extra
  small-model call of time-to-first-token on every undecided turn, which is exactly why it is per-deployment
  opt-in and not a default. Model-choice routing ("pick the best model per turn") is deliberately out of
  scope: it would require a ranking, and rankings are a thing this platform does not invent.

#### Decisions recorded here so they are not re-litigated

- **No library autocomplete.** The model registry exposes no catalog or tag-list endpoints, and every third-party
  index is a scrape with no contract, so the add-model box stays free text, answering existence, download size,
  quantization and parameter count from the per-tag registry manifest instead.
- **The endpoint URL stays deployment configuration**, not a UI-editable field - the provider is constructed at
  boot in two processes, so the console gets a read-only display with a health probe, and model *choices* move
  into the UI instead.
- **An unauthenticated remote endpoint is tolerated, loudly.** Upstream Ollama has no native API authentication,
  so a bare cloud endpoint is usable - including its delete and create verbs - by anyone holding its URL. The
  platform works against one anyway, but says so persistently in the console and at `env-check`, and ships
  outbound-credential support plus a documented reverse-proxy recipe for the day a proxy closes it.
- **Facts never fit predictions, the agency vocabulary keeps its names, and reasoning mode is measured rather
  than trusted** - the same rules [section 11](ARCHITECTURE.md#11-models-what-runs-and-what-is-known-about-them)
  and [section 7](ARCHITECTURE.md#7-tools-design) already state, carried into this phase's UI rather than
  re-decided by it.

**Depends on:** 6 (the manager it reworks) and the [Phase 5](#phase-5---mcp-subsystem) harness.
**Parallelism:** MEDIUM - R0 is self-contained and ships first; R1 and R2 are serial on the resolution chain;
R4, K1 and R5 fan out once R3's stores exist.

### Phase 7 - Frontend polish + theming

*A user can now: use a polished, accessible, themed UI that works from a phone to a widescreen.*

Where a consistent design system and a shared component library get established, and the UI gets its final polish.

**The design system is already authored (2026-07-20)** and lives in [docs/design/](design/):
[DESIGN.md](design/DESIGN.md) (token architecture, state choreography, motion language, a11y
floors, and a 10-theme catalog), [IMPLEMENTATION.md](design/IMPLEMENTATION.md) (the PR-by-PR
build order with acceptance criteria - the token foundation lands at the *start* of Phase 6 so
its pages are born themed), pixel-real [prototypes](design/prototypes/) with a working state
harness, and per-theme [token maps](design/tokens/) with recorded contrast ratios. The
prototypes are the visual contract for this phase; the deliverables below are implemented
against them.

**Deliverables:**

- **Rendered answers - LANDED EARLY**
  (#194): assistant markdown
  rendered as GFM (headings, lists, tables, code, blockquotes) with inline `[n]` markers drawn
  as the design system's cite-marks, gated on the sources the message actually carries. Raw
  HTML stays escaped text, images are never fetched, and URLs pass an allowlist - the renderer
  treats model output as untrusted by design.
- **The aurora redesign - LANDED.** The default theme was redrawn against a new prototype
  ([the build order](design/IMPLEMENTATION.md#aurora-next---build-order-2026-10-09)), one slice per
  pull request, each compared with the prototype at a desktop width and at 390px and walked under
  reduced motion:
  - the sky, the veil and the surface tokens (#293);
  - the chat surface as a timeline, with its reply footer, code blocks and citation popover
    (#301);
  - the boot screen and the new chat (#302);
  - the chrome around the thread - composer controls, sidebar, floaters
    (#303);
  - the console and the sign-in, with the travel between chat and console
    (#331).

  Minimal shares the markup and was checked on every slice. A tool's row also says how many
  sources it produced (#304).
- **A citations panel.**
- **Theming per the dossier** - CSS custom-property tokens with a small hand-rolled
  ThemeProvider (`data-theme` / `data-mode` / `data-accent` on `<html>`, localStorage
  persistence, OS-following for paired themes, a no-flash inline script). The earlier
  `next-themes` idea is superseded: it models a single light/dark axis, not the
  theme x mode x accent triple the catalog needs. The foundation plus the first two
  themes - aurora (the default) and minimal - land at the start of
  [Phase 6](#phase-6---model-manager); the remaining catalog themes land here.
- **Micro-interactions** per the dossier's motion language (Framer Motion where a library
  earns its place; route-scoped GSAP stays reserved for the Model Manager's progress
  animation).
- **Responsive layout** - a three-pane desktop view collapsing to a single pane.
- **Accessibility** - focus management, `aria-live` for streaming text, WCAG AA contrast.
- **Per-conversation draft persistence - LANDED**
  (#336). An unsent message in the composer
  belongs to the conversation it was typed in, the way modern chat apps remember a half-written reply. The
  composer used to hold one draft for the whole app, so switching conversations carried it across and a reload
  lost it. Each conversation now keeps its own, persisted in the browser (`localStorage`, keyed by user and conversation id) so it survives a
  reload; the still-unbound "new chat" pane keeps its draft under a placeholder key that moves to the real id
  once the conversation is created. This mirrors how a conversation's unsent *attachments* already persist, but
  stays client-side: a draft is text the user is still editing, with no reason to reach the server before it is
  sent.

  Because it lives in the browser, three rules bound it. Keys carry the signed-in user's id, so two people
  sharing a browser never read each other's drafts, and nothing is read until the session is known. Signing out
  clears every draft in that browser; a session that merely lapses keeps them for the sign-in that follows. And
  a user holds at most twenty, none longer than fifty thousand characters, the oldest giving way.

**Depends on:** 6. **Parallelism:** MEDIUM - components fan out, but a single owner holds the shared theme tokens
and the layout shell.

### Phase 7b - Resumable streaming

*A user can now: switch conversations or close the tab while an answer is streaming, come back, and find that
answer still arriving - or already finished and saved - instead of losing it.*

Scheduled deliberately just before the cloud deployment, where flaky networks and closed laptops make a dropped
stream a routine event rather than an edge case.

**Deliverables:**

- **A server-side run buffer** recording a live generation's events - tokens, reasoning, citations - as they are
  emitted, keyed by the assistant message id.
- **A reattach endpoint** that replays the buffered prefix to a returning client and then continues streaming live,
  so a reconnect resumes rather than restarts.
- **A bounded grace window.** A disconnected generation keeps running only briefly, after which it is cancelled and
  its partial persisted - model inference slots are finite and must not be spent completing answers nobody will
  read. Unconditional background completion was considered and rejected for exactly this reason.
- **A frontend stream store** holding one handle per conversation, so switching chats detaches the *view* without
  dropping the *run*.
- Once more than one backend worker serves streams, the buffer moves behind Redis. Today's in-process SSE state and
  per-process concurrency cap are the two things that assume a single worker, and this is the phase that removes
  that assumption.

**Depends on:** 2 (persistence) only. **Parallelism:** MEDIUM - the buffer, the reattach endpoint and the frontend
store are separable once the event contract is fixed; the grace-window semantics are serial.

**How it landed (partly).** The reliability half of this phase shipped early, ahead of the rest:

- **A client disconnect can no longer lose a partial reply.** The disconnect cancels the streaming generator inside
  a cancel scope where every subsequent `await` is torn, so the save-the-partial write is *shielded* and completes
  independently. A partial reply can no longer vanish between the last token and the database.
- **A cut-short reply says so.** The resulting message exposes an `interrupted` flag through the API, and the UI
  badges it, so a truncated answer never reads as a complete one.
- What remains for this phase is resuming the stream itself, rather than merely surviving its loss.

### Phase 8 - Hardening, CI gates, deployment

*A user can now: reach the app at a real HTTPS address, running on a cloud VM.*

The phase that turns a stack which runs on a laptop into one that runs in front of people.

**Deliverables:**

- **Coverage gates** - roughly 75% backend, 60% frontend.
- **Playwright end-to-end tests** over 3-5 core flows, plus nightly smoke tests against a real Ollama.
- **Production compose and the Caddy service** as the single HTTPS origin, with port hygiene verified end to end.
  *Landed* (#210): `DEPLOY_ENV` selects the
  layout, so the same `make up` runs the development overlay on a laptop and the hardened one on a server. The
  loopback-bound-ports and single-HTTPS-origin mechanics are in
  [ARCHITECTURE.md section 12](ARCHITECTURE.md#12-containers--deployment).
- **Rate limiting** - deliberately generous (abuse protection, not a quota) - and `/readyz` dependency checks.
  *Landed* (#211) - the route classes, the
  `TRUSTED_PROXY_IPS` mechanism, the fail-open behavior, and the readiness split are described in
  [ARCHITECTURE.md section 12](ARCHITECTURE.md#12-containers--deployment).
- **Log and disk management on a long-running host.** Container logs are capped in the compose files already.
  *Landed*: the minimum Docker engine version is stated as a prerequisite in both setup paths ([SETUP.md](SETUP.md) and [DEPLOYMENT.md](DEPLOYMENT.md))
  (23.0 / API 1.42 - the floor `make prune`'s guard already enforced but nobody was told about), where the
  volumes live is documented as a deployment choice with the two supported ways to move them, and
  `make host-disk` reports free space on the daemon's filesystem beside the size of each data volume and of
  the local backup copies. *Still owed*: a floor on free disk before a model pull starts - the manager knows
  the download size but cannot see free space, because the model volume is mounted only into the model
  server, and on a deployment whose model server is a separate machine it is not on this host at all.
- **Backup and restore for the named volumes.** *Landed.* `make backup` / `make restore` cover the volumes a
  reinstall cannot reproduce, with the `pg_dump` / Qdrant-snapshot / attachment-tar mechanics, the manifest
  checksums, and the refuse-before-destroy behavior described in
  [ARCHITECTURE.md section 12](ARCHITECTURE.md#12-containers--deployment) and
  [DATA-MODEL.md](DATA-MODEL.md#what-lives-in-each-volume) (the full volume-by-volume accounting).
- **A retention decision for attachments.** *Landed*
  (#214) - what expires, what does not, and
  the sweeps that enforce it are in [MAINTENANCE.md section 6](MAINTENANCE.md#6-what-expires-and-what-does-not).
- **GHCR image publishing** gated by a human-reviewed release process, then the deploy to the cloud VM.
- ~~**The MinIO/S3 `StorageBackend` implementation**~~, deferred out of
  [Phase 4](#phase-4---multimodal-attachments) and now **landed early**: `STORAGE_BACKEND=s3` selects an
  S3-compatible backend (AWS or the bundled MinIO) behind the existing seam. What remains for the deployment is
  pointing it at a real bucket.
- **A first-clone preset switch and a hardware suggester.** A small hardware scan (observed VRAM, RAM,
  GPU-versus-CPU class) that recommends the deployment knobs a machine of that size wants - a context window,
  an embedding placement - and prints its suggestion. It never predicts per-model fit, and it never applies
  anything on its own. Deterministic and scriptable, never an interactive wizard.
  *The narrow form landed early* (with the first server deployment to point it at): `make init` writes `.env`
  from the template with a fresh `SECRET_MASTER_KEY` - see the
  [server deployment steps](DEPLOYMENT.md#deploying-to-a-server). The hardware scan and any preset overlay above
  it remain here.
  *Amended by [Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts):* the preset argument this once
  carried selected a model tier, and retired with it.
- **Two model runtimes side by side.** The deployment target serves chat (and, with it, thinking and vision on a
  multimodal tag) from a vLLM process, because continuous batching is what lets concurrent users share one GPU
  rather than queue behind one another, while embeddings stay on Ollama or move to the Infinity sidecar that
  already serves the reranker. The seam is already per-role rather than per-process, so the work is a provider
  implementation plus factories that select one **per role** - nothing above the `ModelProvider` seam changes.
  See
  [ARCHITECTURE.md section 4.1](ARCHITECTURE.md#41-model-seam---modelprovider).
- **A finalized [ARCHITECTURE.md](ARCHITECTURE.md).**

**Depends on:** 5, 6b, 7, 7b. **Parallelism:** LOW - it is cross-cutting, and wants a single owner with the
whole-system view, though test authoring can still fan out.

### Phase 8b - Identity & access

*A user can now: sign in with a real work account. An admin can now: see who is in the workspace and change who
administers it.*

Deployment is what creates real users, so identity follows it immediately. Two things had to exist first: the
stable HTTPS origin that OIDC redirect URIs require ([Phase 8](#phase-8---hardening-ci-gates-deployment)), and
the multi-tenant schema that has carried `tenant_id`, `user_id` and a `users.role` column since
[Phase 2](#phase-2---persistence--multi-tenant-schema) precisely so the sign-in cutover is a dependency swap, not
a migration.

**Slice one: the OIDC cutover. MERGED**
(#224, with the sign-in page and the
pre-render session gate in #225). The cutover
replaced exactly one dependency - `get_current_tenant()` stopped returning the seeded default and now resolves
(or creates, on first login) the `users` row keyed on the token's issuer and subject claims, with every downstream
call site unchanged. `OIDC_ISSUER` left unset keeps the pre-sign-in behaviour byte for byte. The mechanism (admission
rules, session cookies, role re-read per request) is described in
[ARCHITECTURE.md section 10](ARCHITECTURE.md#10-data-model--multi-tenancy).

The `require_role` gates that had accumulated on the admin surfaces started enforcing the `admin | member` split
the schema has held since Phase 2, including the Model Manager's ceiling declarations (shipped early by
[Phase 6c](#phase-6c---model-layer-rework--remote-model-hosts)) - the test-first and reasoned-warning safeguards
on that flow (decided 2026-07-15) already existed by then and are covered under
[the agency model](ARCHITECTURE.md#three-agency-modes). Declarations are logged with their actor and the probe
row they relied on.

**Two multi-user gaps closed alongside it**, in the same commit as slice two
(#226), both recorded as latent during
[Phase 5's security review](#what-the-security-review-found): the ephemeral vector scope is now filtered by user
as well as tenant, and the concurrent-stream budget (`MAX_STREAMS_PER_TENANT`) is now spent per user rather than
per workspace - see [ARCHITECTURE.md section 10](ARCHITECTURE.md#10-data-model--multi-tenancy) for why each was
only latent while a deployment had one user.

**Slice two shipped as minimal member management, and the permission model stays deferred. MERGED**
(#226): an admin-gated members API and a
**Console -> Members** page, with the last-administrator and `OIDC_ADMIN_EMAILS` refusals and the per-route
admin gating described in
[ARCHITECTURE.md's settings section](ARCHITECTURE.md#8-settings--configuration). Every role change is logged with
its actor, target and the roles either side of it; there is deliberately no delete or deactivate verb, since
access comes from the identity provider.

**What that leaves for a later slice** is everything below: the three tables, the typed permission enum, the
per-role tool and integration visibility, and the role editor. Nothing about the minimal version blocks them -
the `users.role` column it writes is the same column a `user_roles` join would supersede.

**One hardening follow-up rides here too, not with the permission model:** re-keying the [Phase 8](#phase-8---hardening-ci-gates-deployment)
rate limiter on the authenticated principal instead of client IP (see
[ARCHITECTURE.md section 12](ARCHITECTURE.md#12-containers--deployment)) - deliberately out of scope for the
slices above.

**Slice three (deferred): roles beyond the binary.** Application-level RBAC, deliberately *not* SQL-level roles:
the backend connects to Postgres as a single service account (end users are not database principals), most of
what needs authorizing is not a table row ("may this role use `web_search`", "may it pull a model"), and
authorization already has one home - the repository layer and its dependencies - which must not gain a rival.
Row-level security stays available as defense-in-depth underneath, never as the mechanism.

- **Three tables**: `roles` (tenant-scoped, with seeded, undeletable `admin` and `member` built-ins),
  `role_permissions` (a typed permission key from a closed enum - `tools.use`, `integrations.connect`,
  `models.pull`, `kb.write`, `secrets.manage`, `settings.write`, `members.manage`, and kin - plus an optional
  resource selector such as a tool name), and `user_roles`. The selector is what lets "may use web search but not
  the ticket tools" arrive later without a migration. See [DATA-MODEL.md](DATA-MODEL.md).
- **Enforcement reuses seams that already exist.** Routers gain a `require_permission` dependency (the same
  one-dependency pattern as tenancy); the per-turn tool set - already computed through a stack of clamps - gains
  a role allowlist as one more intersection, clamping down and never up; registered MCP servers gain per-role
  visibility.
- **Roles broaden what a user may *do*, never whose data they may *see*.** The user-scoped boundary sits beneath
  RBAC, and no permission overrides it: an admin managing roles still cannot read another user's conversations.
  Admin access to user data would be its own explicit decision, and the default is no.
- **Guardrails.** The last admin can be neither demoted nor deleted, and every role change is logged with its
  actor, the same way tool calls already are. Both already hold for the shipped binary.
- **A role editor on the members page.** The roster and its per-user assignment already exist; what this adds is
  the editor beside them - permission checkboxes grouped by category, with per-tool and per-integration toggles.

Deliberately deferred until a real deployment demands them: per-collection knowledge permissions, teams and
groups, and conversation sharing. The permission enum and its selector column admit each without schema change.

**Depends on:** 8 (the HTTPS origin) and 2 (the schema). **Parallelism:** MEDIUM - the OIDC cutover was small and
serial; the roles schema, the enforcement points and the editor fan out behind it.

### Phase 9a - Document-generation tool service

*A user can now: ask for a real spreadsheet, document, PDF or chart, and download the generated file from the chat.*

The output half of the platform's file story: Phase 4 brings files in, Phase 9a produces them.

**Deliverables:**

- **A curated MCP sidecar for document generation** - structured JSON in, a real file out, via pinned libraries:
  `.xlsx` (openpyxl), `.docx` (python-docx), `.pdf` (reportlab), `.csv` and `.md` (standard library).
- **Charts and graphs** rendered to `.png` or `.svg` (matplotlib on the headless `Agg` backend), standalone or
  embedded in the documents above.
- **Generated files stored through `StorageBackend`** and returned as downloadable attachments in the chat, so
  generation reuses the Phase 4 storage surface rather than inventing a second one.

This is the safe first answer to any request for a generated file: the model fills a schema, pinned code renders
it, and no arbitrary code executes. Only when a request genuinely exceeds what these generators can express does the
sandboxed interpreter in 9b become the answer.

**Depends on:** 4 (the storage and attachment surface) and 5 (the MCP loop); scheduled after 8. **Parallelism:**
HIGH - a self-contained sidecar behind the MCP contract.

### Phase 9b - Sandboxed code interpreter (optional)

*A user can now: ask for computations and files beyond the curated generators, executed in an isolated sandbox.*

Built only if 9a proves insufficient.

**Deliverables:**

- **An isolated MCP sidecar exposing `run_python`** - standard output and files out, with no network egress, CPU,
  memory and time limits, an ephemeral filesystem, and pinned Python libraries.

Deliberately sequenced after the Phase 8 hardening pass, and gated to the mid and server tiers: a 4B model writing
correct spreadsheet code is unreliable.

**Depends on:** 8 and 9a. **Parallelism:** HIGH - a self-contained sidecar.

## 4. Definition of done (every phase)

- Runs via `docker compose up`.
- Tests green in CI; lint, format and type checks pass; the dependency audit is clean.
- The diff is reviewed, and `packages/api-types` is regenerated after any API change.
- Docs updated wherever a decision changed.
- **No unexplained performance regression.**
- Merged to `main` via pull request.

**On that performance bar.** A change can be correct, well tested, and still leave the assistant too slow to enjoy -
and no test suite notices, because latency is not an assertion. Phase 3b's grounding had to be repaired on both
counts at once: it cited the wrong sources *and* it was slow, and only the wrong citations were visible to the
tests. `make perf` measures the latency of a chat turn - time-to-first-token and tokens-per-second - so it is run
for any change that can move that: the generation path, retrieval and grounding, streaming, web search, or model
and context-window selection. Such a change is not done until `make perf` has been run on the same machine before
and after it, and any regression is either absent or named and deliberately accepted. A change that cannot touch a
turn - background ingestion, docs, CI - does not need it. Absolute numbers mean nothing across machines; only the
delta on one machine does. Warm the stack first (`make warm`), or a cold model load swamps the measurement.
