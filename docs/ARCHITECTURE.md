# Architecture

How the AI Chatbot Platform is put together and why. This is the self-contained design reference;
[ROADMAP.md](ROADMAP.md) covers the build order. It is written to be readable without prior familiarity with the
specific tools - every jargon term is explained the first time it appears.

---

## 1. What this platform is

A self-hosted chat assistant. It runs open language models on local hardware (no cloud AI account required),
grounds its answers in uploaded documents and in records fetched live from external systems, understands images
and PDFs, and serves all of it through a clean, modern web UI. Its defining trait is modularity: the model
runtime, the retrieval layer, and the tool layer each sit behind a swappable interface - a "seam" - so any one of
them can be replaced without touching the rest of the app.

The backend is a single FastAPI application - a "modular monolith": one deployable process, cleanly layered
inside - surrounded by small helper containers called "sidecars". One backend is far simpler to run, debug, and
reason about than a fleet of microservices, while the internal layering keeps it from turning into spaghetti.

```
Browser (Next.js)
      │  HTTPS + SSE (streamed tokens)
      ▼
   Caddy (reverse proxy)
      │
      ▼
  FastAPI backend ────────────────►  Ollama              (chat / embeddings / vision)
   routers → services → providers ►  Qdrant              (vector search)
                     │            ►  Postgres            (conversations, settings, tenants)
                     │            ►  Redis + arq         (background jobs + cache)
                     │            ►  Storage             (attachments: a Docker volume, or S3-compatible)
                     │            ►  Reranker sidecar    (Infinity: bge-reranker-v2-m3)
                     └── MCP host ►  mcp-chassis ─►  external API-backed datastore
```

One line per box:

- **Browser (Next.js)** - the chat UI; receives replies as a live stream.
- **Caddy** - the reverse proxy: the single public entrance that terminates HTTPS and forwards traffic inward.
- **FastAPI backend** - the brain: builds prompts, orchestrates retrieval and tools, streams replies.
- **Ollama** - the model runtime: runs the actual language models (chat, embeddings, vision) on the local GPU.
- **Qdrant** - the vector store: a database that finds text by meaning rather than by keywords.
- **Postgres** - the relational database holding conversations, messages, settings, and tenants.
- **Redis + arq** - the job queue where slow work (document ingestion) runs outside the request path.
- **Storage** - uploaded files; a Docker volume on the server, or an S3-compatible object store.
- **Reranker (Infinity)** - a small sidecar model that re-scores search hits so only the best reach the model.
- **mcp-chassis** - a tool server wrapping an external HTTP API (tickets, orders, a CRM) so the model can query it.

**Backend layers** (a request only ever flows downward): `api/` routers do HTTP/streaming only; `services/`
orchestrate ("build the prompt, retrieve context, run the model, drive the tool loop"); `providers/` are the
swappable backends; `repositories/` do tenant-scoped database access; `core/` holds config, logging, errors, and
the auth boundary.

## 2. What each phase delivers

The build is layered - each phase is a runnable increment ([ROADMAP.md](ROADMAP.md) has the execution detail).
From a user's point of view:

| Phase | What becomes possible for a user | Key machinery behind it |
|---|---|---|
| 0 | Nothing visible yet - the whole stack starts with one command, and every later change ships through guarded CI (continuous integration: the checks that run automatically on every change) | Monorepo scaffold, docker compose profiles, CI + dependency audits |
| 1 | Chat with a local model in the browser: replies stream in live, follow-ups remember the conversation, reasoning models show their thinking, and failures surface as plain messages instead of dead streams | `ModelProvider` seam + `OllamaProvider`, SSE chat endpoint (Server-Sent Events - explained in section 3), client-held history, thinking channel + model-pair routing, typed error events, `GET /api/v1/models` |
| 2 | Conversations survive restarts and appear in a sidebar with auto-generated titles; switch and delete them | Postgres + tenant-scoped repositories, server-side history rebuild + token-budget truncation, settings schema, RFC 7807 errors + request-id logs, SSE heartbeats, per-tenant concurrency cap, generated TS API types |
| 3 | Answers can draw on a knowledge collection and cite the passages they came from | Qdrant + the embed/chunk/retrieve pipeline, context-window budgeting, chunk delimiting as injection defense |
| 3b | Ask about anything newer than the model's training data - the assistant searches the web only when needed, cites its sources, and a toggle keeps everything fully local | SearXNG sidecar + `SearchProvider` seam, local-first search policy, the same ingest + citation pipeline |
| 4 | Attach a PDF or image to a message and ask about it | `StorageBackend`, parsing + OCR pipeline, vision capability detection, background ingestion with progress |
| 5 | The assistant decides for itself when to search the web or look up live records in an external system mid-answer - and says plainly when no tool fits | `Tool` seam + registry (local tools in-process, MCP servers as one source), bounded host loop, agency modes, read-only vs side-effecting confirmation, tool-result injection defenses, honest-refusal prompt layer, per-user credentials behind `SecretStore` |
| 6 | Browse, download, verify, and switch models from a Model Manager page; pick the model, thinking mode, and web-search mode per conversation right in the composer | Ollama proxy with pull progress, the model-knowledge sources of section 11, composer controls |
| 6b | A Settings page: workspace defaults, a workspace system prompt, the tool-server registry, secret entry | `workspace_settings` / `mcp_servers` / `secrets` tables, write-only encrypted secrets |
| 6c | Point the platform at any Ollama server - local or remote - and every installed model is listed, pickable, and honestly described, whatever a config file predicted | Evidence-based resolution (provider inventory, the runtime's capability report, digest-bound `model_probes` measurements, admin `model_policy` declarations) replacing the retired tier catalog; workspace defaults writer |
| 7 | A polished, themed, accessible UI on any screen size | Design tokens, dark/light theming, citations panel, accessibility pass |
| 7b | A dropped connection no longer loses an answer - a reconnect finds a reply still arriving or already saved, marked interrupted if it was cut short | Shielded persistence across a disconnect, an `interrupted` message flag; the server-side run buffer and reattach endpoint remain future work |
| 8 | The platform runs on a real server over HTTPS, stable under load | Caddy TLS, rate limiting, prod compose hardening, coverage gates, end-to-end tests |
| 8b | Sign in with a real work account; admins see who is in the workspace and change who administers it | OIDC behind the one tenancy dependency, server-side sessions, admission from deployment config, a members page over the `admin`/`member` split (typed permission bundles remain future work) |
| 9a | Ask for a real file - a spreadsheet or a PDF report - and download it | Document-generation MCP service: structured JSON in, `xlsx`/`pdf` out |
| 9b | Computations beyond the curated tools, via model-written Python run safely | Sandboxed code-interpreter sidecar (no network, hard resource limits) - built only if 9a proves insufficient |

## 3. A chat turn, end to end

The full flow once every layer is in place - a message with an attachment, retrieved context, and a record
lookup. Three terms first. **SSE** (Server-Sent Events) is a plain HTTP response the server keeps open and pushes
small text events down - it is how tokens appear one by one. **RAG** (Retrieval-Augmented Generation) means
fetching relevant stored text first and handing it to the model, so answers come from real data rather than the
model's memory. **MCP** (Model Context Protocol) is an open standard for exposing tools a model can call.

1. A user types a message, optionally attaches a PDF or image, and picks a collection and a model.
2. **The attachment uploads first.** Bytes stream to storage; a background job extracts, chunks, and embeds the
   content into an ephemeral (session-only) Qdrant collection, streaming progress events back.
3. The browser opens `POST /api/v1/chat/stream` - an SSE endpoint. The request carries the message, a
   `conversation_id` (absent when starting a new conversation), an optional model, a thinking mode, and a
   web-search mode (each `auto | on | off`, default `auto`).
4. The chat service composes the system prompt and applies the reasoning and web-search policies (sections 5
   and 6).
5. **RAG** retrieves the most relevant chunks from Qdrant (filtered by tenant, session, and collection) - and,
   when the question needs information newer than the model's training data, fresh web results fetched through
   the search sidecar - optionally reranks them, and injects them into the prompt, delimited as quoted data.
   Retrieval queries carry the embedding model's instruct prefix (the wrapper its model card prescribes, held
   in the model-quirk seed of [section 11](#111-four-sources-of-model-knowledge-none-of-which-can-go-stale-quietly));
   documents embed raw - retrieval-tuned embedders are asymmetric, and skipping the prefix collapses the
   similarity scale the relevance floors are calibrated against.
6. The service composes the tool schemas this turn is allowed to offer (section 7) and calls
   `provider.chat_stream(...)`.
7. If the model asks for a tool - a local one such as `web_search`, or a `search_records` tool exposed by an MCP
   server, which authenticates upstream and hard-enforces tenant scoping - the platform executes it, feeds a
   bounded result back as a tool message, and re-enters the model. Side-effecting tools require a UI
   confirmation first.
8. Passages a tool produced flow through the same `ingest_structured()` pipeline (session-scoped) and become
   citable like any other source. The loop is bounded, and its last pass is entered with no tools at all, so the
   turn always ends in an answer.
9. Tokens stream to the browser as they are generated; on completion the assistant message and citations are
   persisted. If the client disconnects or aborts mid-stream, the partial reply is persisted anyway and marked
   as interrupted, so nothing typed or generated is silently lost - and because every stream opens by naming
   its conversation (the `meta` frame below), an aborted first turn still leaves the client bound to the
   thread it started: follow-ups continue it, and the background title job still names it when reply text
   streamed before the cut.

The SSE wire contract (extended additively as phases land):

```
event: meta       data: {"conversation_id": "..."}          # first frame of every stream: names the thread
event: thinking   data: {"delta": "..."}                    # reasoning text, streamed before the answer
event: token      data: {"delta": "..."}                    # answer text
event: status     data: {"message": "...", "code": "..."}   # progress note, e.g. a model loading or records being searched
event: citation   data: {"id","source","snippet","score","index",...}  # a retrieved source backing this turn
event: tool_call / tool_result                              # a tool the model asked for, and its bounded result (Phase 5)
event: tool_confirmation  data: {"id","conversation_id","calls":[...]}  # a side-effecting call awaiting the user's OK; terminal (Phase 5)
event: file       data: {"attachment_id","filename","content_type","size_bytes","origin"}  # a file a tool produced, stored and ready
event: error      data: {"message": "...", "code": "..."}   # provider_unavailable | model_not_found | bad_request | internal_error
event: done       data: {"message_id": "...", "conversation_id": "...", "usage": {...}, "turn_usage": {...}}
```

`done` carries two usage objects on purpose. `usage` is the provider's report for the final generation
alone. `turn_usage` is the chat service's sum over every generation the turn ran (a turn that called a tool
is at least two): the tokens written and the runtime's own writing time for them. The same object is
stored with the reply, so the figures under a reply are identical live and after a reload.

The `file` event is how a turn tells the client that one of its tools produced a document. Only the file's
metadata travels the stream - the bytes go to object storage and are recorded as an attachment on the
conversation, and the event is emitted only once that row is committed, so an announced file always names a
row that is really there. The model itself never sees the bytes either: it reads whatever one-line summary the
tool wrote, which keeps a large payload out of a prompt that is budgeted to the token.

Provider failures never leak stack traces: the services layer translates them into those four typed, human-safe
error codes. Outside the stream, errors are RFC 7807 `problem+json` responses (a small standard JSON shape for
HTTP errors) carrying a request id that also appears in the structured JSON logs, so any failure can be traced
end to end. The server emits an SSE comment line (a `:`-prefixed ping) roughly every 15 seconds as a heartbeat,
so proxies do not drop a stream that has gone quiet while a model loads or thinks. A per-tenant concurrency cap
bounds simultaneous streams - the GPU can only serve so many at once - and over-cap requests fail fast with a
429 rather than queueing invisibly.

History is rebuilt server-side from `conversation_id` under a token-budget truncation policy: the newest turns
are kept, oldest dropped first, within a message-count and approximate-token cap. A client-supplied `history`
array is still accepted, but only as pre-context when starting a brand-new conversation, and it is never
persisted.

### 3.1 The four request shapes

A chat request is one of four things, and the request model's validator is what keeps them apart - each shape
is exclusive, because a field the backend would silently ignore is worse than one it refuses:

- **A new turn** - words, files (`attachment_ids`), or both. The ordinary case. It may also carry a `quote`
  (see below), which decorates it rather than changing its shape.
- **A confirmation resume** - `confirmation` carrying an approve or deny, continuing a turn that suspended on a
  side-effecting tool (section 7). It brings no new words: it answers a question the backend asked.
- **A regenerate** - `regenerate: true` with the `conversation_id`, and optionally the `message_id` of the reply
  to redo. It answers a stored question AGAIN, in place of the reply that question already produced.
- **An edit** - `edit` naming a stored question and its new wording. It rewrites that question and answers the
  new wording.

The last two are the same turn, re-run rather than replayed. The prepare phase locks the conversation, finds the
question, and rebuilds history up to but not including it - then takes the files and the ordering timestamp off
the stored row rather than off the request, which is why neither request carries them. An edit supplies the one
thing that is NOT already stored, the new words; a regenerate supplies nothing at all. The thinking and
web-search modes come from the request as they would on any turn, so asking again with the switches moved is how
a *different kind* of answer is asked for. Everything after that is the ordinary path, `meta` frame included.
Files behave slightly differently here and deliberately so: a client naming a file that does not exist has made
a mistake worth a 404, but a question whose file was deleted since has made none, so a stale reference is
skipped and the turn answers from what remains.

**Where the discarding happens is what separates the two**, and it follows from what each turn is retrying.

A regenerate drops its old reply at the **save**, in the same transaction that writes the replacement - never up
front. One that produces nothing (an empty generation, a provider failure before the first token, a client that
disconnected) therefore leaves the previous answer standing, which is what a retry should do; deleting eagerly
would destroy it on the way to failing, and the client puts its transcript back whenever the backend did not
take the turn. Redoing the NEWEST reply sweeps the run of replies after its question and stops hard at the next
question, so a second tab sending a new turn mid-stream can at worst gain an extra reply, never lose one.
Redoing an OLDER one discards a whole tail instead - every turn below it replied to an answer that is about to
change - so those rows are named at prepare time, exactly the rows the client counted for the user, and that
captured list is what the save deletes. Rows that arrive in between were never on it and survive.

An edit truncates in the **prepare** phase instead. The reply below the question answers wording that no longer
exists anywhere, so keeping it while the question changes would put words in the model's mouth, and the user has
already been shown what it costs. A generation that then fails leaves the reworded question standing with no
reply - which is precisely the state a regenerate is the retry for. Rewording the question a conversation is
NAMED after also hands the naming back to the background job, under a job id of its own so it is not
deduplicated against the one that named the thread the first time.

Because a regenerate's discarding lives in the save, it has to survive anything that splits a turn in two - and
a confirmation pause is exactly that. Such a turn records the debt in its checkpoint (the sweep flag, or the
captured id list) and honours it when the resume saves, or the resumed answer lands beside the one it was sent
to replace instead of on top of it.

One further guard belongs to every shape rather than to any one of them. A reply is written only while the
question it answers still exists - one primary-key read on the session the save already opens. Three real races
separate the two while a stream is in flight: a stopped turn's shielded partial landing after the thread was
truncated, a conversation deleted mid-stream, and a suspended turn resumed onto a tail that has since moved
(which is why the confirmation checkpoint records the question's id alongside everything else it carries). The
opening `meta` frame names that same question, so a client can key its optimistic bubbles on server ids from the
stream's first frame rather than on their position in the transcript.

### 3.2 Replying to a passage

A turn may carry a `quote`: the passage the user is responding to, taken from anywhere in the transcript. It is
not a shape of its own - it decorates a new turn and is refused beside the other three, which carry no fresh
words for a quotation to be part of.

Two properties define it. The text is **authoritative**: it is what reaches the model and what is stored with
the message, and it is never re-read from the row it came from, so quoting a passage keeps working after that
message has been edited away or deleted. And it is **untrusted**, because what sits in a transcript is not all
the user's - an assistant reply may itself be quoting a fetched web page, so quoting one hands that page's words
back to the model inside the user turn, which is the part of a prompt a chat template trusts most. It therefore
enters the prompt through the same fence and the same "reference DATA, not instructions" framing that retrieved
passages and tool results use, rather than a second, weaker treatment beside them (section 7 covers that defense
in full). It is length-bounded and refused rather than truncated above the bound: a silently shortened quotation
would attribute words to a passage that does not contain them.

## 4. The three seams

A "seam" is an interface that hides a concrete implementation so it can be swapped without touching its callers.
The services layer talks only to these three; it never imports Ollama, Qdrant, or a specific MCP server directly.

### 4.1 Model seam - `ModelProvider`

One interface stands between the app and whatever runs the model - local Ollama today, a hosted API tomorrow.

```
class ModelProvider(Protocol):
    def chat_stream(messages, tools=None, model=None, options=None) -> AsyncIterator[ChatChunk]  # options: GenerationOptions (think, temperature, num_ctx, ...)
    def embed(texts, model=None) -> list[Vector]
    def list_models() -> list[ModelInfo]
    def pull_model(name) -> AsyncIterator[dict]    # the backend's own progress events
    def delete_model(name) -> None
    def unload_model(name) -> bool                 # drop it from memory now; False = this backend cannot
    def residency(model) -> ModelResidency | None  # loaded? how much of it is on the GPU?
    def capabilities(model) -> ModelCapabilities   # vision? tools? thinking? context length
    def serving_endpoint() -> str | None           # the host this provider talks to right now, for endpoint-bound tool-check certificates
    def runtime_version() -> str | None            # the serving runtime's version, for staleness banners
    def management_capabilities() -> ManagementCapabilities  # can this backend pull/delete/list/unload at all?
```

The management verbs are paired with a declaration rather than assumed, because they are not
universally meaningful. A registry-style backend downloads and removes weights while it keeps running; a
process-per-model runtime loads what it was started with and changes only by redeployment. Asking
`management_capabilities()` costs nothing (it is a fact about the backend class, not a round trip), so a
management surface can honestly report "changed via deployment" instead of rendering a button that could
only fail. Pull progress crosses the seam as the backend's own dicts on purpose: what a download reports is
genuinely its vocabulary, and a normalized progress record here would only be a second dialect to keep in
step with the first. The services layer maps those dicts onto the typed events clients see.

**Unload and residency are the two hooks a measurement needs.** `unload_model` is deliberate eviction: on a
card that fits one model at a time, loading anything evicts something, and a caller that wants a clean
measurement is far better off choosing what goes than racing the runtime's choice. It is a request, not a
guarantee, and a backend without the verb returns `False` rather than pretending. `residency` answers where a
loaded model actually sits - the total it occupies *and* the share of that in GPU memory. Both halves matter:
a VRAM figure alone cannot tell a model that spilled entirely to the CPU from one that is not loaded, and
those two call for opposite reactions. `None` means "this backend does not report residency", which is never
the same as "not on the GPU".

`OllamaProvider` is the first implementation (async httpx over Ollama's REST API). An OpenAI-compatible, vLLM,
or hosted provider can be added later with **zero** changes to routers, RAG, or MCP. Shared value types
(`ChatMessage`, `ChatChunk`, `ModelInfo`, `ModelCapabilities`) live at this boundary so nothing downstream ever
sees an Ollama-specific shape.

The contract is frozen, not closed - it grows only deliberately, and the tool phase adds to it in exactly two
places: a `ChatMessage` may now carry the tool calls the model asked for (they have to be echoed back before the
tool's answer can be attached to them), and the chunk stream gains a `tool_result` type so a tool's outcome
reaches the UI on the same wire as tokens and citations. Providers still know nothing about the tool registry;
they parse and emit tool calls, and the services layer does the rest.

That last sentence makes this seam the **dialect-normalization point**, and it is worth stating outright: how a
tool call looks on the wire is a property of the serving stack, not of the weights - the same model can emit
native tool calls under one runtime and emit them as prose under another. Each provider parses its own runtime's
dialect into the neutral tool-call chunk right here, so nothing downstream ever learns which dialect was spoken,
and no separate normalization layer exists to build.

**One seam, several runtimes at once.** A provider is selectable per *role*, not per process, so one deployment
can serve chat from one runtime while embeddings and reranking come from another - and that is the shape the
cloud deployment is expected to take: a vLLM process holding the chat model, with embeddings served beside it by
Ollama or by the Infinity sidecar that already runs the reranker.

The split follows the workloads rather than taste. Chat is the concurrent, latency-sensitive load, and continuous
batching is exactly what lets several users share one GPU instead of queueing behind each other. Embeddings and
vision are bursty and small, which is where Ollama's hot-swapping is an asset - and vLLM cannot offer it, because
it serves one model per process and reserves that model's memory up front. Today the factories resolve a single
provider from `MODEL_PROVIDER`; naming a provider per role is what turns the split on, and it belongs with the
phase that has a real deployment to run it
([Phase 8](ROADMAP.md#phase-8---hardening-ci-gates-deployment)), not before.

### 4.2 RAG seam - `VectorStore` / `Embedder` / `Chunker` / `Reranker`

The retrieval pipeline is four small protocols over `qdrant-client` - deliberately **not** a framework like
LlamaIndex, so the path stays small, debuggable, and swappable. `ingest_structured(source, tenant_id, items)` is
the **single** ingestion entrypoint; uploaded files and records pulled via MCP both flow through it, so there is
no duplicate code path.

Every data method on `VectorStore` takes a tenant id and scopes to it. Four methods deliberately do not, because
the questions they answer belong to the deployment rather than to any one tenant: `ensure_ready` (create a
space), `sweep_expired` (drop expired points, which is cross-tenant by nature), `aclose`, and `list_spaces`.

`list_spaces()` enumerates the physical collections the store holds - name, kind, point count, and
the embedding-model tags read off a small sample of real stored points. It is the input to the stranded-vectors
alarm in [section 6](#6-retrieval-rag-design). Two properties of its contract carry weight. The provenance is
**sampled**, never parsed out of the collection name: the name carries a lossy rendering of the model tag (every
character a collection name cannot hold collapses to a placeholder), so it cannot be turned back into the tag
that wrote the points. And a store that cannot enumerate must **raise**, never return an empty list - "there is
nothing stored" and "I cannot tell you" are different answers, and a caller unable to tell them apart would
report a healthy store while every vector in it sat unreachable.

### 4.3 Tool seam - `Tool`, with MCP as one source

Tools sit behind a transport-agnostic `Tool` abstraction: a spec (name, description, JSON Schema parameters,
side-effecting or not), one async call, and a registry that composes tool *sources*. Local tools - web search,
knowledge search - are ordinary in-process Python. **MCP is one source behind that seam, not the tool
architecture itself.** For the MCP source, the backend is the MCP *host*: it discovers tools from MCP *servers*
(sidecar containers reached over Streamable HTTP, MCP's plain-HTTP transport), adapts each into the same `Tool`,
advertises them to the model, and runs the call-a-tool, feed-the-result-back loop. Which servers exist is
per-tenant configuration, not code. Section 7 has the full design.

## 5. Reasoning ("thinking") models

Some models can reason before answering: they first emit a private chain of thought, then the reply. That extra
pass measurably improves hard questions (math, multi-step planning) but costs real seconds on local hardware, so
the platform treats reasoning as a first-class, controllable channel rather than an always-on default.

- **On the wire,** reasoning streams as its own `thinking` SSE events ahead of the `token` events, so the UI can
  render it collapsed while the answer appears normally.
- **Per request,** the chat API takes `thinking: "auto" | "on" | "off"` (default `auto`). In `auto`, a
  deterministic policy in the services layer decides per message whether reasoning is worth the latency; a later
  orchestrator phase can replace that policy without touching the wire contract. The composer later gains a
  two-state toggle - Auto (default) / On - next to the model selector (Phase 6).
- **A model that cannot toggle gets a partner.** The plain `qwen3:4b` tag points at Qwen's 2507-refresh
  *thinking* variant, which always reasons - the think flag cannot turn it off - while `qwen3:4b-instruct` is
  its never-thinking sibling. Pairing the two gives one deployment both behaviours: `qwen3:4b-instruct` answers
  normal turns fast and `qwen3:4b` serves thinking turns. Which tags are paired is an operator declaration made
  on the console against the installed weights, and never inferred, because a wrong guess routes a turn to a
  model nobody chose. `THINKING_CHAT_MODEL` overrides it as the deployment's kill switch (the literal `none`
  disables pair routing outright; empty means unset).
- **A hybrid model needs no partner.** `qwen3:8b/14b/32b` are original hybrids where Ollama's top-level `think`
  flag genuinely switches reasoning on and off per request.
- Asking a model without the thinking capability to think returns an Ollama 400 ("does not support thinking");
  the provider retries that turn once without the think flag, so the answer still streams - reasoning is simply
  skipped. Other malformed requests surface as the typed `bad_request` error.

## 6. Retrieval (RAG) design

An *embedding* turns a piece of text into a long list of numbers that captures its meaning; texts about similar
things end up with similar numbers. A *vector store* (Qdrant) indexes those number lists so "find the stored
passages most similar to this question" becomes one fast query - that similarity search is what grounds answers
in real documents.

- **One Qdrant instance, two collection types, scoped per vector space.** *Persistent*: `kb_chunks__{model}`,
  the knowledge base. *Ephemeral*: `session_chunks__{model}`, session-scoped points (`session_id` = the
  conversation id) carrying a `ttl_at` timestamp - TTL, "time to live", the moment a record expires. Qdrant has
  no native per-point expiry, so a periodic arq job sweeps expired points, and deleting a conversation purges
  its session points immediately. Collections are shared across tenants and partitioned by an indexed
  `tenant_id` payload filter that the store layer applies on every read and stamps on every write - the same
  single-chokepoint rule the repository layer enforces for Postgres, and the multitenancy layout Qdrant itself
  recommends (a collection per tenant costs a separate index graph each and scales poorly).
- **Vector spaces never mix.** The collection *name* carries the embedding model, so an embedder change simply
  lands in a fresh, empty space - vectors of different models (or dimensions) cannot collide, and there is no
  error state to recover from. Old-model collections linger untouched until manually removed; each point also
  records its `embed_model` for provenance.
- **Stranded vectors are detected and said out loud.** The flip side of that safety is silence: change the
  embedding model and every query goes to a fresh, empty space while the full one sits unread. Nothing fails, no
  error is raised, and the deployment simply looks as though it forgot everything it was taught. So the platform
  looks for exactly that. It resolves the embedding model actually in use, enumerates the store's
  spaces (`list_spaces()`, [section 4.2](#42-rag-seam---vectorstore--embedder--chunker--reranker)), and compares
  the active tag against the provenance sampled from stored points. Four rules keep the answer honest:

  1. **An empty store is not stranded**, and neither is one whose points were all written by the model in use.
     The alarm fires on evidence, not on absence.
  2. **Only the persistent collections raise it.** Session points carry a TTL and the sweep clears them within
     hours on their own, so a stranded session space is a fact that fixes itself before anyone could act on it.
  3. **A store that cannot be enumerated reports "unknown", never "fine".** So does a deployment with no
     embedding model resolved at all - though that report still carries the full inventory, because a machine
     nobody has finished configuring is exactly when someone needs to see what is actually stored.
  4. **Out-of-reach points are an alarm only when nothing readable is left.** Changing the embedder and
     re-ingesting - the remediation the alarm itself prescribes - leaves a full new collection beside a full old
     one, since old collections linger until they are removed by hand. Judged collection by collection that
     state looks identical to the emergency, so the report counts the readable side too: nothing readable plus
     stranded points is the alarm; a working corpus plus stranded points is a **cleanup note** that names the
     leftovers and says retrieval is fine. Without this split, the single most likely state a migrated
     deployment is in would raise a permanent, unclearable alarm.

  The enumeration is cached in process for fifteen minutes - what it finds changes only when someone edits
  configuration or re-ingests - and is bounded by a short timeout of its own, well under the store client's,
  because a status line must never be the slowest thing on the page it annotates. The verdict is rebuilt from
  that cached reading rather than cached itself, since the same inventory now answers a second question: which
  embedder wrote the stored passages, as the evidence rung of the embedding resolution chain.

  That composition has one honest limit, and it is worth stating rather than discovering. Where the chain FALLS
  BACK to the corpus - nobody has chosen an embedder - the active model is by construction one that wrote
  something, so the alarm can only speak about the other collections. It has teeth exactly where a tag was
  chosen independently of what is stored, which is what a workspace default is.

  It surfaces in two places. The model-management view carries the full report (recomputed per page load, since
  that page is where an operator who just changed the embedder looks), which is the detective half of the
  protection the guarded embedder switch gives - the guard covers a swap made *from that page*, and a corpus
  that changed underneath for any other reason is invisible to it. And a chat turn raises a standing
  `vectors_stranded` notice naming both models.
  Both judge against the same tag - the embedder actually in use - so a workspace that named its own cannot see
  the console raise an alarm the transcript stays quiet about.

  That notice is raised **only from a branch where retrieval has already come back empty**, which is what keeps
  a healthy turn from paying anything for it. There are two such branches, because there are two ways a turn
  reaches the knowledge base: the automatic pre-fetch, whose lookup found nothing, and - in agentic mode, where
  the pre-fetch is deliberately suppressed - a `search_knowledge` call that matched nothing. Either is the
  moment the distinction matters to a reader, since "nothing was on file about that" and "everything on file is
  unreadable" otherwise produce the same ungrounded answer.

  A miss is not on its own evidence of anything, though - most misses are simply an off-corpus question. So
  either branch only earns the right to *ask* the wholesale question, "does this turn's scope hold anything at
  all", and the notice is raised only on a definite no. A store that could not be reached answers neither yes
  nor no, and is never asked a second time. The counts stay off that sentence on purpose: the enumeration
  behind them is tenant-free, so they are deployment-wide totals that belong on an operator's page rather than
  in a line addressed to one person.
- **Payload schema:** `{tenant_id, user_id, session_id|null, collection_id|null, source_type
  (records|upload|mcp_records|web), doc_id, record_id|null, title|null, text, chunk_index, created_at, ttl_at,
  embed_model, url|null, content_hash|null}`. Timestamps are numeric (epoch seconds) so the TTL sweep is a
  plain range filter. Point ids are deterministic UUIDs derived from (tenant, scope, document, chunk index);
  re-ingesting a document first deletes its previous chunks by filter, so even a shrunken document leaves no
  stale tail behind - unless its `content_hash` (a fingerprint of the chunking config plus document text) is
  unchanged, in which case the embed-and-rewrite is skipped entirely.
- **Embeddings run in Ollama** (`qwen3-embedding:*` via `/api/embed`) - no separate embedding service.
- **Retrieval:** dense-only where no reranker sidecar is wired, and reranking wherever `RERANKER_URL` names one
  - the *reranker* re-scores the top hits so only the sharpest context reaches the model.
  A chat turn searches the tenant's whole knowledge base (all collections) plus the conversation's session
  points; retrieval runs only when a cheap count says there is anything to find, and only hits above a
  relevance floor ground the answer - an empty or irrelevant store means a normal, ungrounded reply. A
  per-conversation retrieval override is reserved for the settings surface (the `rag` key in conversation
  settings, wired with the Phase 6 composer controls).
- **Context budgeting:** every turn asks for an explicit context window, resolved the one way
  [section 11.3](#113-hardware-is-expressed-by-what-is-pulled-and-how-wide-a-turn-may-be) describes, and warming
  uses the same value - the model server restarts its runner when the requested window changes, which would
  throw away the primed prompt cache. The per-turn envelope reserves the system prompt
  and an answer allowance first (larger on reasoning turns, whose thinking shares the window), then grants
  retrieved context up to the RAG budget, trimmed to whole chunks; when nothing fits, grounding is skipped
  outright rather than shipped truncated - citations must never point at text the model did not actually see.
- **Injection defense:** retrieved chunks are injected delimited as quoted data, never as instructions - a
  document that says "ignore previous instructions" is just text. The grounding block travels in the user turn
  rather than the system prompt, for two reasons: the warmed system-prefix cache survives untouched, and chat
  templates that special-case system messages stay on their tested path. Only the prompt copy is
  delimiter-sanitized; stored text and citation snippets stay verbatim.
- **Grounded answers cite what they use.** Retrieved passages are OFFERED to the model in the grounded prompt,
  but citations are emitted - as SSE `citation` events after the answer tokens, before `done` - only for the
  passages the finished answer references inline by [n] marker, and only those persist in the assistant
  message's metadata. An answer that ignored the offered context carries no sources, so small-talk turns in a
  conversation that once searched the web no longer display stale source chips. Used sources are renumbered
  1..k in order of first reference (an answer citing only the fourth offered passage shows [1], not [4]); when
  that rewrite changed the text, `done` carries the final answer in `content` and the client swaps its streamed
  copy. Retrieval failures never fail a chat turn: an unreachable store or embedder degrades to an ungrounded
  answer with a logged warning.
- **GPU scheduling:** embeds are batched; ingestion pauses between batches while a chat stream is active; and
  `EMBEDDING_DEVICE` defaults to `cpu` so the embedder never steals VRAM from - or evicts - the chat model on a
  card that fits one.
- **Pipeline:** parse → chunk (token-aware, structural: paragraphs, then sentences) → embed → upsert.
  Structured records ingest synchronously through `POST /rag/ingest` (idempotent per document - re-running an
  interrupted ingest heals it); an uploaded file takes the same path behind a background arq job, because
  parsing and embedding a document takes seconds to minutes and must not hold an HTTP response open. The
  attachment row's `status` is how a client follows along.

### Web search: fresh knowledge, local-first

Local models have a knowledge cutoff - they know nothing after their training date. Web search closes that gap,
and it is deliberately just another retrieval source: search the web, fetch the top result pages, extract their
readable text, chunk it, and inject it as cited context through the exact same pipeline as document chunks. The
model never browses; the platform does. Fetching the pages, rather than grounding on the engine's short snippets,
matters because those snippets are often the site's boilerplate meta description ("Breaking News, Latest News and
Videos") instead of the content that answers the question; the full-page text gives retrieval the passage that
actually does. Extraction is a lightweight standard-library HTML-to-text pass (no third-party parser) that drops
page chrome - navigation, headers, footers, banners - and, when a page marks a semantic `<main>` region, keeps
only that: without this the menus and section taxonomies dominate the text and bury the article (or, under the
character cap, truncate it away entirely). One caveat is inherent to a fetcher that runs no
browser: pages that build their content with client-side JavaScript (many news landing pages) expose little text
to a static fetch, so grounding on those stays thin. It sits behind a `PageFetcher` seam beside `SearchProvider`,
and `WEB_FETCH_PAGES=false` restores the older snippet-only, single-round-trip path for the smallest systems.

- **The engine is self-hosted and free.** A SearXNG sidecar (a metasearch engine that aggregates public engines
  such as Google, Bing, and DuckDuckGo behind one JSON API) runs as one more compose container - no account, no
  API key, no per-query cost. It sits behind a `SearchProvider` seam, so a hosted search API could replace it
  without touching the callers.
- **Local-first by policy.** The chat API takes `web_search: "auto" | "on" | "off"` (default `auto`). In `auto`,
  the model's own knowledge and any RAG context answer by default; the web is consulted only when a deterministic
  policy detects freshness cues - recency wording ("latest", "today", "current price", "breaking news"), a year at
  or after the current one, or an explicit ask to search. Like the reasoning policy, it starts as a testable
  heuristic and can be
  replaced by the orchestrator later. On tool-capable models, `web_search` is also exposed as a tool the model
  can call mid-answer (Phase 5) - a *local*, in-process tool behind the same seam MCP servers plug into, not an
  MCP tool; the deterministic pre-fetch remains the small-model fallback. Switching web search off withholds the
  tool as well as the pre-fetch: off means fully local, whatever the model would have liked to do.
- **Privacy is the toggle's job.** The composer's web-search toggle is enabled by default (= `auto`); switching
  it off keeps every request fully local. When a search does run, only the query leaves the machine - sent by the
  self-hosted instance to public engines, tied to no account.
- **Fetching is defensive and bounded.** Result pages are attacker-influenceable, so the fetcher is http(s)-only,
  resolves each host and refuses private, loopback, and link-local addresses (an SSRF guard against internal or
  cloud-metadata endpoints) - re-checked on every redirect hop, since redirects are followed by hand rather than
  by the HTTP client and capped at three - verifies the response is HTML, and caps both bytes downloaded and
  characters extracted. Pages are fetched concurrently, bounded by `WEB_FETCH_CONCURRENCY`, and every per-page failure
  degrades to that result's snippet - a slow or hostile page never breaks the turn.
- **Results are cited like everything else:** the extracted page text (falling back to the snippet) flows through
  `ingest_structured(source="web")` into the session-scoped collection (it expires on the same session TTL) and
  streams back as the same `citation` events - each carrying the result's `url`, so a web citation renders as a
  clickable source chip while a local document citation stays a plain label. A search that finds nothing, times
  out, or hits an unreachable engine degrades to a local answer, never a broken stream (the same degrade-not-fail
  rule as document retrieval).
- **Repeat searches are nearly free.** Full page text is fetched for only the top `WEB_FETCH_MAX_PAGES` results
  (the rest ground on snippets), a page already ingested into the conversation within
  `WEB_RESULT_CACHE_MINUTES` is reused as-is, and a re-ingested document whose content fingerprint is unchanged
  skips embedding entirely - so a follow-up search of the same topic costs one engine round trip instead of the
  full fetch-and-embed pipeline (which dominates web-turn latency on CPU-pinned embedders).
- **The first search of a fresh stack is warmed.** A cold metasearch engine pays DNS resolution, an empty
  connection pool, and a lazy-loaded worker on its first outbound call, so that first fan-out is the slow one -
  reliably slow enough that a too-eager client timeout would cut it off ("the first search fails, the retry
  works"). Three measures remove that edge: the backend's client timeout is held at or above the engine's own
  per-upstream wait, so it never pre-empts a slow-but-successful fan-out; the `SearchProvider` retries once on a
  transient connection fault (never on a genuine read timeout, and never on an empty result set, which is a
  valid "found nothing"); and a throwaway query at startup - the same warming pattern the chat and embedding
  models use - primes the engine before the first real turn arrives.
- **The search query is context-aware.** A follow-up that leans on an earlier turn ("and how is it currently")
  leads the query with its own words and trails the recent user turns as a topic anchor, before the query reaches
  both the engine and the retrieval ranking. The `auto` decision is history-aware the same way: a
  context-dependent follow-up in a thread whose recent turn already warranted a search inherits that freshness.
  This is the deterministic small-model path; a tool-capable model composes its own query instead (section 7),
  and this remains its fallback.
- **One citation per source.** A long page or record splits into many chunks that share a URL or document id, so
  retrieved chunks are collapsed to the single best-scoring one per canonical source before grounding - each
  source stays one numbered passage and one citation, never several for the same page.
- **The sidecar is not profile-gated.** Web search defaults to `auto`, so SearXNG runs in the standard `make up`
  set rather than behind an opt-in profile; a mounted `infra/searxng/settings.yml` enables its JSON API (off by
  default upstream) and holds its secret key. The whole feature stays off unless `SEARXNG_URL` is set, and a
  failed or empty search degrades to a local answer rather than a broken turn.

## 7. Tools design

A *tool* is a function the model can ask the platform to run in the middle of an answer: search the web, search
the stored records, look up a ticket in an external system. The model never runs anything itself. It emits a
structured request, the platform decides whether to honour it, executes it, and hands the result back as more
context - and the model then writes the answer.

The seam is a small, **transport-agnostic** `Tool` abstraction. The natural reading of "the MCP phase" is that
every tool is an MCP tool, and that is not the design: **local tools are ordinary in-process Python, and MCP is
one *source* of tools behind the same seam.** Sending a search that already lives in this process out over HTTP
to a sidecar and straight back is pure ceremony - it buys a network round trip on the most latency-sensitive
path in the app in exchange for nothing at all.

### The tool seam

- **`ToolSpec`** - what the model is shown: a name, a description, a JSON Schema for the parameters, whether the
  tool has side effects, and which source it came from. The description is load-bearing; good descriptions are
  what make small local models call the right tool.
- **`Tool`** - one async `call(arguments, context)`, plus an `available(context)` check that does no I/O.
- **`ToolContext`** - the tenant, the user, the conversation, and the collaborators a tool needs (embedder,
  vector store, reranker, search provider, page fetcher), plus the token budget its result may spend.
- **`ToolResult`** - text for the model, any passages the call produced (so they can be cited), and an honest
  error when it failed.
- **`ToolRegistry`** - composes tool *sources*. A `ToolSource` lists tools: `LocalToolSource` holds the
  in-process ones, and an MCP source lists what a server advertises. The registry always holds every tool; the
  per-turn policy alone decides which ones are *offered*.

Phase 5 ships two local tools, each with exactly one parameter - `web_search(query)` and
`search_knowledge(query)`. One parameter is a rule for the whole tool surface, not a coincidence: every optional
parameter is another chance for a small model to fill it with garbage.

Tool names follow MCP's own naming rule even for local tools, because the model echoes a name back verbatim and
the two vocabularies must not drift apart. A tool's JSON Schema never contains a tenant, user, conversation, or
session id - the platform injects those from the request's own principal, so the model cannot even *express* a
call against somebody else's data.

### MCP servers as a tool source

- The backend is the MCP *host*: using the official `mcp` Python SDK (pinned to the 1.x line - v2 is a breaking
  revision of both the SDK and the spec), it speaks Streamable HTTP to sidecar servers on internal-only ports,
  lists their tools, and adapts each one into a `Tool`. That adapter is the seam's whole payoff - a new tool
  server becomes configuration, not code.
- `mcp-chassis` wraps an external API-backed datastore - a JS/HTTP API fronting a database; it never
  touches the database directly. It exposes a small, well-described tool surface (`search_records`,
  `get_record`, `create_record`) rather than mirroring the raw API. It authenticates upstream per request,
  resolves pagination server-side into a bounded array, and **hard-enforces `tenant_id` scoping regardless of
  what the model passes** (a data-leak guard).
- Records a tool returns flow through the same `ingest_structured()` pipeline as an uploaded file or a web page,
  into the conversation's session-scoped collection - so they are retrievable by the next turn, citable, and
  they expire with the session rather than lingering in the knowledge base. A wrapper opts in by returning its
  records in the protocol's structured-content field against a small `{records: [{id, title, text}]}` contract;
  a result without it still reaches the model as fenced reference data, uncited. Because a tool server's records
  can be per-user once credentials arrive, `ingest_structured()` refuses to write `mcp_records` anywhere but the
  session scope - a raised error, so records can never reach the tenant-shared knowledge base.

### Tool selection under schema pressure

Every offered tool rides in the prompt as its full JSON Schema, so the offered set is paid for in context on
every turn - and the moment MCP servers join the registry, the tool count can grow past what a small model can
hold, in window space and in judgement alike. So selection becomes retrieval: the tools are ranked against the
turn's message, the top-K are offered, and a small core - web search, knowledge search - is always offered
regardless of rank.

Two properties keep that affordable, and both are deliberate.

**Selection is inert until there is pressure.** At or below a configured tool count (`TOOL_SELECTION_THRESHOLD`)
the whole set is offered and selection does not run at all: no ranking, no embedding, no work of any kind. That is
what a small deployment sees. With a handful of tools there is no schema pressure to relieve, and ranking them
would only spend an embedding call to arrive at the one answer available anyway - offer them all. It matters more
than it looks: an agentic turn otherwise touches the embedder exactly zero times, so a ranking call would be the
one thing waking a cold model, and it would land in front of the first word of the reply.

**Ranking happens in process, over cached description vectors.** It reuses the `Embedder` seam and nothing else.
The per-turn cost of selection is embedding the message, and a vector store would not remove that cost - what a
store would hold is the tool descriptions, which are the cheap half: few, static between deploys, and cached on
their content hash, so a re-embed happens only when a server actually changes a description. Ranking tens of
descriptions in process is arithmetic. Keeping them in the vector store instead would mean a third collection
kind threaded through every method of the `VectorStore` seam, a tenant id stamped on data that belongs to no
tenant, and vectors that strand on the next embedder change - infrastructure bought to solve the half that was
never expensive.

### The host loop

Generation becomes re-entrant. The model is offered the turn's tool schemas alongside the prompt; instead of
tokens it may answer with tool calls; the platform executes them, appends the results to the conversation as
`tool`-role messages, and re-enters the model with everything it now knows. Answer tokens still stream to the
browser exactly as before - a tool turn simply opens with `tool_call` and `tool_result` events where a plain turn
would already be producing text.

The loop is bounded three ways: a cap on tool-executing passes per turn, a cap on total tool calls per turn, and a
token budget for the results themselves (see below) - all documented in
[`infra/.env.example`](../infra/.env.example). **The final pass is always
entered with no tools at all**, so a model that would happily keep calling tools forever is instead forced to
produce an answer from what it has. Every turn ends in exactly one `done` event and exactly one persisted
assistant message, however many passes it took, and a client that disconnects mid-loop still gets its partial
reply persisted and marked interrupted.

**A tool exchange lives for the turn that ran it.** It is persisted for display, and never replayed into a later
turn's prompt: a 900-token tool result carried forward into every subsequent turn would eat a large share of a
modest context window for work that has already happened, and would multiply the injection surface
permanently. What a tool found stays reachable from the conversation's session collection instead - cheaper, and
precisely why agentic mode never suppresses that scope (see below).

Failure is routed to the model first and to the user always. A hallucinated tool name, malformed arguments, or a
tool that ran and failed all come back as an honest `[tool error]` message that the model can recover from within
the same turn; an identical repeated call is answered from cache and named as a repeat. But small models
routinely swallow that text and answer from memory anyway, so the turn *also* carries a standing status notice
that persists with the message - `tool_error` when a tool could not run, `tool_limit_reached` when the budget was
spent and the answer was forced, and `tools_not_used` when a turn that was free to look something up looked
nothing up and grounded on nothing. The last one is the honest label for an answer written entirely from the
model's own memory, which may be out of date.

### Three agency modes

How far the model is trusted to drive a turn is one setting, resolved per turn, and it can only ever be clamped
downward.

| Mode | What a turn does |
|---|---|
| `deterministic` | The pre-tool pipeline exactly: heuristic web pre-fetch, automatic retrieval, no tools offered. The kill switch. |
| `assisted` | The deterministic preamble runs **and** tools are offered on top of it. |
| `agentic` | No pre-fetch and no ambient knowledge-base retrieval - the model looks things up itself, or does not. |

- **The mode a turn runs in is a minimum over five keys** -
  `effective_agency = min(ceiling, earned, AGENCY_MODE, request, runtime floors)`. The **ceiling** answers from
  a stored operator declaration when there is one; with nothing declared, the runtime's report answers and
  stops one rung short of the top - `assisted` for a tag whose template takes a tool list, `deterministic`
  otherwise, including for a report that could not be taken, since "the runtime would not say" is not evidence.
  **`agentic` is only ever a human declaration.** A runtime saying "my template accepts a tool list" is a
  mechanical fact about a chat template, not a decision about how far these weights may be trusted, and a probe
  never raises a ceiling either.
- **The `earned` key is the latest valid verdict from the tool-check harness** (below). A declaration alone
  buys a **grace floor, and that floor is `assisted`** - tools offered while the deterministic pre-fetch and the
  ambient knowledge base still run, so a fresh deployment works on day one and is exactly where it was before
  tools existed, plus an offer its model may decline. `agentic` is never granted on paper: handing the turn
  over is a claim about how these weights behave in a loop, which only a **passing DEEP check bound to the
  installed digest and the serving endpoint** can make. Re-pull the weights, edit the harness, or point at a
  different model server, and the verdict stops applying - the model falls back to the grace floor until it is
  re-checked. (Failure evidence clamps regardless of endpoint; the asymmetry is deliberate.)
  `AGENCY_REQUIRE_VERIFICATION` removes the grace entirely for a hardened deployment, leaving only certificates.
  Never probed *and* never declared resolves to `deterministic`. Two keys, because each catches what the other
  cannot: a tag drives only when a human said it may *and* a measurement agreed it can - which structurally
  defuses the mispackaged-model trap, since a model that cannot emit a parseable tool call fails the probe and
  never gets the loop. The deployment then clamps with `AGENCY_MODE`, a single request may clamp again, and the
  floors below clamp last; every key is a minimum, so nothing anywhere can escalate its own privileges.
- **The per-request clamp is not a user-facing control.** It exists as a developer and testing affordance, and
  the composer deliberately never surfaces it. Thinking and web search are choices a user can reason about; *how
  far the model drove a turn* is an internal routing decision, and putting it in front of a user would only ask
  them to guess.
- **`main` ships `AGENCY_MODE=agentic`**, which leaves the deployment axis wide open so the model's own ceiling
  and earned verdict decide - a tool path that is never taken is a tool path that is never fixed. A deployment
  that wants the old behaviour back changes one environment variable, not one line of code, and
  `deterministic` reproduces the pre-tool turn exactly, which is what makes it a genuine kill switch and not a
  euphemism.
- **In `assisted`, a model that never calls a tool behaves exactly as it does today.** It is the mode for a
  deployment that cannot afford *any* change to the non-tool path: the deterministic preamble still runs
  underneath, so tools can only add. Shipping `agentic` by default deliberately accepts the other trade - a model
  that declines to look something up answers from its own memory, on a turn the knowledge base would previously
  have grounded. That is exactly what the `tools_not_used` notice exists to make visible rather than silent.
- **Some floors are not negotiable.** A turn resolves to `deterministic` on its own when nothing survives to be
  offered, when there is no room left in the context window for a tool result (see below), or when it must reason
  on a model not cleared to reason and call tools in the same turn (see below) - which a model whose reasoning
  cannot be turned off at all is always in. A model that *can* do both hits no such floor.
- **The user's switches gate the tool set, not merely the mode.** With web search off, the `web_search` tool is
  not offered at all - the model must not be able to make a live fetch the user explicitly refused.
- **Agentic suppresses the ambient knowledge base and the pre-fetch, never the conversation's own session
  scope.** What a previous tool call found still lives in the session collection, and the model must be able to
  find it again on the next turn.
- **Attached files are never handed to the model's discretion.** A file attached to a turn is force-fed into that
  turn's prompt in every mode, agentic included. The honesty notices - "this picture was not shown to the model",
  "this attachment could not be read" - are raised from that same retrieval path, so replacing it with a tool the
  model may forget to call would retire the honesty along with it, and a user staring at the file they just
  attached would never learn it had been ignored.

### Whether reasoning and tools can both be live is a per-model fact, measured then declared

**A model that can reason and call tools in the same turn is allowed to do exactly that, and the platform does
nothing to stand in its way.** Reasoning streams as usual - whether the request asked for it explicitly or the
`auto` policy chose it - the turn's tools are offered as usual, and the model interleaves the two: think, call a
tool, read the result, think again, answer. That is the flagship behaviour, it is the goal, and on a model capable
of it the agency resolver arbitrates *nothing*, because there is nothing to arbitrate.

Whether a given model can do it is one of the three questions
[section 11](#111-four-sources-of-model-knowledge-none-of-which-can-go-stale-quietly) keeps apart, and the only
one of them that no source may answer optimistically:

| Question | Kind | Answered by |
|---|---|---|
| `tools` - can the chat template take a tool list at all | Mechanical | The runtime's own report |
| `agency` - the strongest mode this model may attempt | Policy | An operator's declaration, floored by the report |
| `thinking_with_tools` - can it reason AND call tools in one turn | Capability | A deep measurement, then a declaration standing on it |

**It is a property of a (model, runtime, digest) triple, not of the weights alone.** Today's `false` answers
exist because of one upstream bug in one runtime:
[ollama/ollama#10976](https://github.com/ollama/ollama/issues/10976) makes a qwen3 model that is asked to think
and to use tools in the same request return empty output. That is a defect in the serving layer, not in the
weights - the issue remains open without a maintainer root-cause, and the most detailed community trace points
at the server's request handling (an omitted think flag being promoted to "on" for any model advertising the
thinking capability) compounding with the model's own sampling under that regime - so a different runtime with
separate tool-call and reasoning parsers (vLLM, for one) may well have no such limitation with the very same
model. And a model tag is mutable - a re-pull can swap the weights underneath it - so the digest pins which
weights a measurement actually saw, and the probe record names the runtime and the serving endpoint beside it.

**The default is `false`, and deliberately so.** Being wrong in the optimistic direction fails catastrophically
and silently: #10976 produces *empty output*, not an error - no exception to catch, no typed error event, just a
blank reply that every correctness test can watch sail past. Being wrong in the cautious direction costs a little
reasoning on tool turns. So a model does not inherit `true` from its family, its size, or a vendor capability
flag; it earns it.

**`make tool-check` is how it is earned.** The opt-in diagnostic that already earns a model its measured agency
verdict (above) carries one more case: *reasoning enabled and a tool offered*, asserting both a well-formed tool
call **and** non-empty output. A model passes that case at the full depth (below), or the answer stays `false`.
The measurement log records what was measured, never what was hoped, and a stored `true` is a permission an
operator granted on the back of it - which is why withdrawing the claim costs nothing and granting it costs a
deep verdict bound to the installed weights.

**The fallback, and only for models that cannot do both.** Only when the answer is `false` does a turn have to
choose. In agentic mode:

- **Thinking on `auto` - tools win.** The turn keeps its tools and does not reason.
- **Thinking explicitly on - thinking wins.** The turn reasons, and runs deterministic with no tools.

In `assisted`, thinking wins in both cases: the deterministic preamble has already gathered the turn's context,
so a reasoning turn gives up nothing by declining its tools.

The mechanism differs by how the deployment reasons; the arbitration does not. Where a never-thinking model is
paired with an always-thinking partner (section 5), "tools win" means staying on the first instead of routing
to the second. Where one hybrid model toggles, it means sending `think: false` and keeping the tools.

The composer's thinking toggle (Phase 6) therefore reads differently depending on the model in force. On a model
that cannot do both, **Auto** is "look things up if that helps" and **On** is "reason about it, and accept that it
will not search". On a model that can, the toggle only decides whether to reason - the tools stay on the table
either way, and the trade-off disappears.

Which is the point worth being blunt about: reasoning and tool use are not naturally at odds. The exclusion is a
workaround for somebody else's bug, scoped to the models that actually have it, and it should be **deleted** the
moment it is not needed - because the upstream fix lands, or because the deployed model never had the problem.
Keeping it a per-model answer rather than a platform policy is what makes that deletion a check and a click
instead of a redesign.

### The context budget: a tool result is paid for before the tool is offered

This is the subtle one, and it is invisible to every correctness test. The retrieval budget deliberately *fills*
the context window: it grants retrieved context whatever room is left once the system prompt, the history, and
the answer allowance are reserved (section 6). Tool results arrive *after* that, on top of a prompt that is
already full - and a model runtime makes room by evicting the **head** of the prompt. The head is the system
prompt, which carries the persona, the tool guidance, and the injection defense. Left alone, the tool surface
would quietly eat the very instructions that keep it honest, and the answer would look perfectly fine.

So the tool-result budget is computed per turn from the actually-assembled prompt, reserved before retrieval is
granted its share, and drained as results come in. One result is capped; the turn's results are capped together;
and if what remains falls below a floor, **the turn does not offer tools at all** and runs deterministic instead.
That last rule is a matter of honesty as much as arithmetic: offering a tool whose result could not possibly fit
in the window is a lie told to the model.

### Tool results are untrusted input

A fetched web page or an upstream record is attacker-influenceable text, and it enters the prompt under the
`tool` role - which some chat templates present to the model as *more* authoritative than the user's own words.
That is exactly why the defense here is the existing one, reused rather than reimplemented: results are fenced
and labelled as quoted data, never as instructions, they are size-capped, and a truncated result carries a
visible marker inside the fence so it cannot pretend to be complete. A second, weaker copy of a defense the
retrieval path already gets right would be the worst of both worlds.

### Citations from tools

Passages a tool produced are citable exactly like any other source. They join one turn-wide offer list, numbered
continuously across the loop's passes and deduplicated by canonical source - so a page reached by both the
deterministic pre-fetch and the model's own search is offered, and cited, once. The usual usage gate then applies
(section 6): only the passages the finished answer actually references survive as `citation` events and in the
stored metadata, and they are renumbered 1..k, once, when the answer completes.

### Read-only versus side-effecting

Classification is deliberately conservative: **a tool is side-effecting unless it is explicitly marked read-only
*and* its server is trusted.** MCP lets a server annotate a tool as read-only, but the specification is explicit
that those annotations are *hints* from a possibly-untrusted server and must not be relied on for safety. So a
missing annotation means side-effecting, and the platform materializes that default itself rather than assuming
the SDK will apply it.

A side-effecting call does not simply run. When the model asks for one, the loop checkpoints the suspended turn -
the assembled prompt, its counters and cited sources, the call awaiting approval - onto the conversation, emits a
terminal `tool_confirmation` event carrying the pending action, and stops without running anything. The client
shows what is about to happen and asks; the user's answer resumes the *same* turn in a second request, and the
backend loads and consumes the checkpoint under a row lock (so a double-tapped approve cannot run an action
twice), runs the tool on an approval or feeds a synthetic refusal on a denial, and streams the answer either way.
A checkpoint carries its own expiry, is superseded the moment a new turn starts on the thread, and is refused
once consumed - so a stale or replayed approval is a clean 404, never a late action.

The gate is opt-in by capability: a side-effecting tool is offered *only* to a client that advertises it can
resume a confirmation. To any other client - and to every request before this landed - such a tool stays
withheld, exactly as the earlier fail-closed drop left it, because a call that suspends a turn the client cannot
continue is worse than no tool at all. A newly-added MCP server whose tools default to "destructive" therefore
cannot be called by accident from either kind of client.

### Per-user credentials

Integrations belong to people, not to workspaces. An administrator registers a tool server once; each user
connects their own account to it; and the token used at tool-call time belongs to the user whose turn it is.

There is no separate connection table. The server row itself declares how its calls are authenticated
(`auth_type`) and whose credential they carry (`credential_scope`), the credential lives in `secrets` under that
scope, and the human-readable link - which account a grant belongs to - is resolved from the provider at consent
time and kept on the server row's `config` blob. An in-flight consent is the one thing that gets its own table
(`oauth_authorizations`), because it is state between two HTTP requests rather than a lasting relationship, and
it is consumed exactly once. Every column and delete rule is in [DATA-MODEL.md](DATA-MODEL.md).

- **Secrets live behind a `SecretStore` seam**, encrypted at rest (AES-GCM, master key supplied from the
  environment and never auto-generated, because a key invented at boot looks perfect in development and destroys
  every stored credential on the next restart). Each ciphertext is cryptographically bound to the row that owns
  it, so a read for the wrong principal fails loudly instead of returning somebody else's token. The API is
  write-only: metadata and the last four characters are all that ever come back out. Credentials are injected
  into a sidecar at request time and never persisted there.
- **Tool clients are not pooled across users.** The obvious optimisation - cache one MCP client per server - would
  reuse the first user's `Authorization` header for the next user's call, and no database guard catches it,
  because the correct token was fetched and simply never reached the wire. One client per call, until the risk
  can be retired by a cache key that includes the user.
- **Tenant scoping grows a user dimension where credentials make it load-bearing.** "A user's tool output is only
  visible in their own conversation" is only true if a conversation belongs to exactly one user - which the
  repository layer does not enforce today, because it filters by tenant alone. Conversations, messages, and
  attachments therefore become user-scoped in the same change that introduces per-user tokens. It is a no-op
  with a single seeded user, and it is the difference between safe and leaking on the day real logins arrive.
- **A deployment with sign-in switched off needs saying plainly.** With no `OIDC_ISSUER` configured the tenant
  dependency returns a seeded default user, so anyone who can reach the port *is* that user and holds that user's
  tokens by proxy: such a deployment, holding live integration credentials, is an unauthenticated credential
  vault. That is the right shape for a laptop and the wrong shape for anything reachable. Keep it off the open
  internet - loopback-bound, on a private network, or restricted upstream of the host. The production layout
  narrows the surface as far as topology can, to one HTTPS origin, but it adds no authentication and is not a
  substitute for it. What limits the damage either way: no endpoint ever reads a secret back out - the only thing
  a caller can do is invoke a tool - and every tool call is logged with its tenant and user id, which is an audit
  trail worth having the moment those ids name real people (section 10).

### Connecting an account

A pasted API token is the wrong shape for an integration a person owns. It is long-lived, it is usually scoped to
far more than the tool needs, and nothing about it records who granted it or lets them take it back. So a server
can instead declare `auth_type: oauth`, and connecting it becomes a consent screen.

**Discovery asks the server, rather than trusting configuration.** An MCP-native server publishes protected-resource
metadata ([RFC 9728](https://www.rfc-editor.org/rfc/rfc9728)) naming the authorization server that guards it, which
in turn publishes its own metadata ([RFC 8414](https://www.rfc-editor.org/rfc/rfc8414)) naming the endpoints. The
platform follows that chain rather than asking an administrator to find the endpoints by hand.

Because the chain starts at a URL somebody typed in, it is walked defensively:

- **HTTPS**, with cleartext permitted only against loopback, so local development still works.
- **No redirects followed at all** - not even same-host ones.
- **Response bodies capped**, so a hostile server cannot answer a metadata fetch with a gigabyte.
- **Every server-side fetch target must resolve publicly** when the registered server did, so a metadata document
  cannot steer a credential-bearing request at an internal address.
- **A declared resource identifier that covers the URL being registered** - and not so much more that the binding
  stops meaning anything.

Providers that are not MCP-native (a plain OAuth API behind a wrapper) skip discovery entirely - the endpoints come
from the form.

**The consent flow itself is unremarkable, which is the point.** Authorization code with
[PKCE](https://www.rfc-editor.org/rfc/rfc7636), a single-use `state` row that the callback consumes exactly once (so
an authorization code cannot be injected into somebody else's flow), a redirect URI bound exactly and repeated
verbatim at the token exchange, and the PKCE verifier sealed with the same cipher the credentials use - a plaintext
verifier plus an intercepted code would be enough to complete the exchange. The redirect URI is deployment config,
not per-server config, so the page shows the literal URI this deployment will use: it has to be registered with the
provider *before* the first connection, and a template a person has to fill in by hand is a template they will fill
in wrongly.

**A token is bound to the resource it was issued for** ([RFC 8707](https://www.rfc-editor.org/rfc/rfc8707)), using
the identifier the server's own metadata declares rather than one the platform invents - and the refresh repeats it,
so a renewed token keeps the audience the original had.

**What is stored is one sealed bundle**, not a scatter of columns: the access token, the refresh token if there is
one, when it expires, where it came from, and what it is bound to. Renewal happens on demand at call time rather than
on a schedule, and a grant that is genuinely dead is *marked* dead - so the UI can say "reconnect" instead of the
platform retrying a hopeless refresh on every turn. Because a card only needs the bundle's metadata to report whether
a server is connected, that status still reads correctly on a deployment with no master key configured, where the
bundle itself cannot be opened at all. OAuth consents are workspace-scoped: the flow seals one grant for the whole
workspace. A per-user grant needs a real login to belong to, and that login now exists
([Phase 8b](ROADMAP.md#phase-8b---identity--access)) - per-user HEADER credentials already bind to the signed-in
user's row - so switching the OAuth consent flow itself to per-user scope is the remaining step, no longer blocked
on there being a principal to attach it to.

### Presets are data, not integration code

MCP servers self-describe - handshake one and it reports its tools - so a new integration needs no UI code, only
its coordinates. A preset is therefore just a row of verified facts: the endpoint, the scopes worth asking for, any
extra parameters the provider requires, where to look up the account's name once connected, and the setup steps that
cannot be automated (registering an application, enabling an API, what an administrator has to unblock). Choosing one
prefills the same add-server form anybody could fill in by hand, and the form stays the only path in.

Two consequences follow from keeping it data. Presets are cheap - the shipped set covers GitHub, Gmail, Google Drive,
Slack and Asana, and the effort in each is research rather than engineering, because a wrong scope or a stale endpoint
is a support ticket that arrives as "connect just fails". And the link survives: the card remembers *which* preset it
came from rather than a copy of its values, so its prefilled fields are rebuilt from the catalog on every render and
cannot be lost by navigating away mid-setup.

Each is answered separately, because they are not the same question:

- **`tools: native | none`** is *mechanical*: can this model's chat template accept a tool list at all? There is
  deliberately no third value. An `emulated` mode - scraping tool calls out of a model's plain text when its
  template cannot take a list - is not offered, because it would be fragile in precisely the place it would be
  needed: on the small models it would exist to serve. A documented capability is a promise.
- **`agency: deterministic | assisted | agentic`** is *policy*: how far should this model be trusted to drive a
  turn? It is a ceiling - the strongest mode the model may attempt - not a switch.
- **`thinking_with_tools: true | false`** is a *capability*: can this model, as served by this runtime, reason and
  call tools in the same turn? It defaults to `false` and is earned, never assumed (above).

A model can be entirely capable of *receiving* a tool list and still be untrustworthy in a multi-step loop; and a
model can be trustworthy in that loop and still break when asked to reason inside it. Three independent facts,
answered by three different kinds of source.

Not one of them is a fact about weights alone. Even the mechanical one is a claim about a serving stack: tool
dialect is a property of the runtime, so the same model can emit native tool calls under one stack and emit them
as prose under another, with no error raised anywhere. The `ModelProvider` seam is already where those dialects
are normalized (section 4.1) - each provider parses its own runtime's wire format into the neutral tool-call
chunk, so nothing downstream knows which dialect was spoken and no new normalization layer is built. What a
report or a declaration then needs is verification, and that is the harness below.

A model earns its measured agency verdict - and its `thinking_with_tools: true`, if it gets one - from an opt-in diagnostic
(`make tool-check`) that exercises the cases which actually break small models: must call, must *not* call,
choosing among several tools, refusing a tool that does not exist, continuing after a result is fed back,
recovering from a tool that returned an error, and reasoning while a tool is on offer - run against the warm model
at the settings it is really served with. It is opt-in, and it never runs at boot or on a chat turn: on a box
whose card fits one model, a probe would evict the very model it is about to serve, and one successful call says
nothing about reliability across a multi-step loop, which is where small models actually fail. Neither parameter
count nor a vendor capability flag is evidence either: a tool-tuned 3B model outscores a generic 4B on the public
[Berkeley Function-Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html), and the models that
matter here all report the same vendor flag, so that flag carries no information at all.

What the platform *does* check by itself is consistency: where a tool check's own measurement of how the
installed weights treat the think flag disagrees with what the resolution otherwise declares, the mismatch is
surfaced rather than silently resolved. That check is deliberately
not a per-turn round trip - the capability lookup is a call out to the runtime, so a briefly-wedged model server
would otherwise be able to fail a chat path whose entire design principle is to degrade rather than break. It is
the honest middle ground between probing nothing and believing a claim blindly. `thinking_with_tools` gets no
equivalent check, because no runtime advertises it: the only way to know is to run such a turn and look at what
comes back, which is exactly what the diagnostic does.

**When no tool fits: honesty over improvisation.** The model can only use tools the platform provides - it
cannot install libraries or invent capabilities at runtime. Coverage grows in three deliberate steps:

1. **Curated tools + honest refusal (Phase 5).** A tool-guidance prompt layer instructs the model to say plainly
   when no available tool can satisfy a request and to offer the nearest alternative - an inline CSV or Markdown
   table instead of a real `.xlsx`, for example.
2. **Document-generation service (Phase 9a).** A curated MCP service: structured JSON in, real `xlsx`/`pdf`
   files out via pinned libraries (openpyxl, reportlab), stored through `StorageBackend` and returned as
   downloadable attachments. Covers the common "produce a real file" requests deterministically.
3. **Sandboxed code interpreter (Phase 9b - only if 9a proves insufficient).** An isolated MCP sidecar with no
   network egress, CPU/memory/time limits, an ephemeral filesystem, and pinned Python libraries, exposing
   `run_python` (stdout + files out). Deliberately sequenced after Phase 8 hardening, and gated to deployments
   running a larger chat model - a 4B model writing correct spreadsheet code is unreliable.

### The tool-check harness and the `model_probes` measurement log

The diagnostic above is a **pure service** - no database, no HTTP, no queue inside - with a CLI entry point; the
Model Manager's verify job wraps the same service, so a terminal run and a UI click cannot disagree about
what was measured.

It runs at two depths, and **the depth caps what a verdict can earn**:

- **Fast** (about 45 seconds): the must-call and must-not-call cases, one sample each - earns at most `assisted`.
- **Deep** (every case, three samples each): earns up to `agentic`, and it alone can earn `thinking_with_tools`.

Three details behind those cases are worth keeping in mind:

- **The runtime's own capability flag is not itself evidence.** Ollama reports `tools: true` when a model's chat
  template merely contains the literal substring `tools` or `tool_call` - a syntactic fact about the template,
  not a behavioral claim about the model - and every model run against this platform reports it, which is why
  the harness scores behavior instead of trusting the flag.
- **The must-call case is framed personally** - "how many days do I have to get a refund on my order" - rather
  than only as a freshness ask, because a 4B model (measured 2026-07-16) fell into an "I don't have access"
  refusal on that framing that no amount of prompt wording broke. Whether a model escapes it is exactly the kind
  of per-model fact the harness exists to record.
- **The must-not-call case is the sharpest in the set.** A model that reaches for a tool to answer "what is 2 + 2"
  will reach for one everywhere, so declining an offered tool it plainly does not need is the strongest single
  signal that a model can be trusted to decide when *not* to act.

Where a chat model is paired with a reasoning partner (section 5), a verify covers **both** tags, because a
reasoning turn routes to a different tag than a plain one and a verdict for one says nothing about the other.

Verdicts persist in `model_probes`, a measurement log keyed
`(provider, model, digest, runtime_version, serving_endpoint, harness_version, depth)`, with history kept - it is
what answers "this passed on the previous runtime and fails on this one". Invalidation is structural, never
scheduled:

- **A re-pull that changes the digest** reverts the verdict to unverified on its own - the key simply no longer
  matches.
- **A runtime upgrade** marks results stale behind a banner; nothing re-probes automatically, because an upgrade
  day must not become a probe storm.
- **A different serving host** withdraws the verdict outright. Decode throughput is that machine's hardware and
  tool-calling behaviour belongs to the runtime as that machine runs it, so a check earned against a remote pod
  certifies nothing for a laptop. Only certification is bound this way: a FAILURE still clamps whatever host
  produced it, because moving a deployment is not a way to rehabilitate a model that failed a guard.
- **The harness version fingerprints its own case specs, the served prompt stack, and the scoring rules that
  live in code**, so editing a case - or the persona / tool-guidance layers the harness drives, whose wording is
  measured to move verdict-deciding cases - invalidates everything the old cases measured. The third of those is
  a hand-maintained declaration rather than something derived, because a rule written in code changes what a
  verdict means while every constant it is computed from stays byte-identical: how a sample's think flag is
  resolved, and what the reasoning-with-tools case demands, are both such rules.

Every sample runs at the **serve-time options a turn on this deployment would actually request** - the context
window and the think flag - each resolved through the same rule the chat turn, the warm and the title job use.

For the window, a tag runs at the smaller of the runtime's native window and what the deployment affords - the
workspace context ceiling where one is set, and the deployment-wide default where none is.

For the think flag it is the full resolution chain: a stored operator declaration and a previous check's
measurement first, then the code-shipped model-quirk seed, then the runtime's own chat template. Resolving from
any single one of those would be enough to drive an always-thinking model with the flag cleared, and the raw
reasoning block would then land in the visible answer the check is scoring.

A probe run at any other configuration certifies one nothing serves, and makes the model reload between the
check and the re-warm behind it. Both options are recorded per sample on the row, so a verdict names the
configuration it measured and not only the weights.

The reasoning-with-tools case is the one place a resolved flag needs a guard rather than just an answer. It
asks whether a model can reason AND call tools in the same turn, so on a tag the resolution says never reasons
it is sent no flag at all - and a well-behaved model then calls its tool and answers, satisfying every other
condition. It therefore passes only on reasoning, a call, and an answer together. This is the axis whose
optimistic answer fails as silent empty output rather than as an error, so it is never earned by default.

Both recorded options are **descriptive, and deliberately not invalidation axes**. The axes above are facts
about the weights or the machine; a context ceiling is a setting somebody may raise or lower at any time, and a
think-mode declaration is another, so letting either void verdicts would throw away GPU-minutes of evidence -
or silently re-clamp a model that passed - on the strength of an edit to a settings page.

Alongside verdicts, a probe records the numbers it produces anyway - measured tokens/sec and the model's resident
VRAM after load. Those are what the Model Manager grid later shows as measured facts, with an honest "unmeasured"
where no probe has run; the platform never displays a predicted figure (a fit estimator was evaluated and cut
after proving about 4.5x wrong on real hardware).

#### The verify job: the order a probe has to happen in

The harness scores a model. Running it against a live deployment is a second problem, and the order it happens
in is the whole of it - so that order lives in exactly one place (`app/services/model_verify.py`), which both
the endpoint and the download hand-off go through.

1. **One probe at a time, process-wide.** A probe evicts models and generates tokens for minutes. Two at once do
   not merely cost double, they measure each other: the second one's model load is what the first one's
   residency reading sees. The guard is a single slot rather than one per model for the same reason, and a
   second request is refused with a conflict before anything starts. The slot is released twice over, by the
   request and by the probe itself, because the two cover opposite failures - a client that hangs up before the
   response body is ever read leaves the probe's own release unreachable, while a client that hangs up midway
   unwinds the request while the probe is still running. The release is keyed on the holder, so whichever
   arrives second changes nothing.
2. **Unload deliberately, before measuring** - the target first, then the deployment's own chat pair and the
   embedder, all of which the probe would have evicted anyway. Unloading first also guarantees a *fresh* load at
   serve-time options rather than a runner somebody warmed with a different context size.
3. **Measure**, at the window this deployment's turns request, streaming each finished sample to the client as
   it lands. The workspace settings are read before the probe rather than only for the re-warm at the end, so
   the certificate belongs to the configuration that actually runs here.
4. **Assert where the model actually sat**, through the residency hook above.
5. **Record it**, bound to the weights digest and the runtime version, through the same row builder the CLI uses.
6. **Re-warm chat and the embedder, unconditionally** - including when the probe failed. A diagnostic that
   leaves the deployment cold has made the app slower for the next person to use it, and the failure paths are
   exactly the ones that leave the most evicted behind.

**An unrepresentative run is marked, never quietly trusted.** A model that spilled entirely to the CPU still
answers every case - slowly - so a passing run says nothing about how the deployment serves. Such a run is
recorded with the fact attached, and what that costs it follows the standing rule that a probe may only ever
clamp, never raise: the pass/fail verdicts are kept exactly as measured and still clamp a model that failed a
guard (a spill does not make a model's tool calls better), but the run cannot act as the *certificate* that
lifts a model under `AGENCY_REQUIRE_VERIFICATION`, and its throughput is dropped rather than published, because
a CPU-bound decode rate describes the spill and not the deployment. A *partial* GPU split is not marked at all:
spilling part of a model is how small cards run models in the first place.

A run where the model server could not be reached for **any** case is not recorded at all. Such a sample is
scored as a case failure, so a single blip in a two-case screen earns `deterministic` - and writing that would
clamp a perfectly healthy model on the strength of a network outage. Discarding a partly-measured run is much
cheaper, and this check now runs unattended after a download, where nobody is watching a table.

The same error carries a second meaning, so the discarded samples' detail lines are logged before they go: the
serving runtime reports "this model needs more memory than the machine has" through the same channel as an
unreachable server. Losing that sentence would leave someone re-running a probe that can never pass. Telling the
two apart properly needs a distinct error code at the model seam, which is a change to the chat wire contract
and belongs with the phase that touches it.

**A successful download continues into the fast check** on the same connection, unless the caller opts out or
the tag's declared role is not chat. The download's own terminal event is sent first and still reports success
on its own terms, so a check that fails afterwards can never be mistaken for a download that failed. Either way
the download drops whatever the chat path had cached about that tag: new weights make every cached verdict about
the old ones, and the digest-bound reversion has to happen at the download rather than whenever a cache entry
happens to expire.

The table deliberately breaks two locked rules; it was signed off on 2026-07-15 with both understood:

- **It is a measurement log, not an inventory mirror.** The no-model-registry rule (section 10) bans derived
  copies that drift; a measurement has no source to drift from, and recomputing one costs GPU minutes - which is
  the definition of data worth persisting. `GET /api/v1/models` still derives its list from live runtime state.
- **It is machine-scoped, not tenant-scoped.** A probe row describes the shared model server, the way a schema
  version does - so it does not take the tenant-injecting base repository, and the triggering tenant and user are
  audit-only columns, never scoping.

#### The embedder check: the same idea, one seam over

Retrieval has two calibrated numbers of its own - `RAG_MIN_SCORE`, the absolute relevance floor, and
`RAG_REL_MARGIN`, the relative one - and they belong to a similarity scale produced by **one** embedding model
under **one** query wrapper. Swap either and the floors describe a scale nothing is querying on, silently: a
wrong floor is not an error, it is a citation that should not be there or an answer that quietly lost its
grounding. That failure has already happened here once, when unprefixed queries turned tiny boilerplate chunks
into similarity hubs while genuinely relevant passages fell under a floor calibrated for a different scale.

`make embedder-check` measures the scale instead of assuming it, and `POST /models/embedder-check` is the same
service behind an endpoint. A fixed corpus of triplets - a query, a passage that answers it, a passage that
does not - is embedded through the model seam; every query is scored against every passage, twice, once with
the query prefix under test and once without. What comes back is two bands, what a related passage scores and
what an unrelated one does, and the corridor between them that a floor has to sit in.

**The verdict is the floors COMPOSED, never either one alone.** Retrieval applies `RAG_MIN_SCORE` as a coarse
gate and then drops anything more than `RAG_REL_MARGIN` below the top surviving hit, so a check on one floor
against one band is a decomposition of a rule the system never applies that way. The composed check simulates
retrieval's dense-only path per query and asks the two questions a user would notice - *was the real answer
retrieved, and did any noise come with it*. Everything else is reported beside it:

- **Band diagnostics.** The weakest relevant passage clearing the floor with headroom, the strongest
  non-answer sitting below it with clearance. Informational: they say which floor to look at when the composed
  gate fails, and a band check failing beside a gate that holds is an ordinary, healthy reading. The first live
  run was exactly that - the noise peak sat above `min_score` while nothing leaked, because the margin was
  doing the work.
- **The pruning trade.** A margin that drops a weak corroborating passage is doing its documented job, so it
  is reported as a measured trade rather than a fault.
- **A proposal**, wherever a floor is undeclared or a composed gate fails: the absolute floor at the
  *conservative* end of the corridor (the lowest value clearing the corpus's noise peak, not its midpoint) and
  the widest margin no query leaks at. The pair is then run back through the same simulation a declared pair
  faces, so nothing is recommended that the gate itself would reject.
- **Nothing is applied.** The floors are configuration, and a number that changes what every answer in the
  deployment may cite is a person's decision after reading how it was derived.

**A proposal always carries its transplant caveat.** The probe corpus is purpose-written question-and-answer
text and scores systematically higher than production prose, so a floor derived from it sits too high: the
first live run proposed 0.51 against a production calibration that had measured genuinely relevant chunks at
0.40-0.60. The numbers are still worth having as an upper bound to compare a real calibration against; what
must never travel without them is the reason they cannot be pasted into configuration.

**What counts as noise is a corpus decision.** A query's noise pool is the corpus minus its own answers and
minus every other triplet's *supporting* passage - those are written as near-paraphrases of their own topic and
are the least honest noise available, which the first live run demonstrated by making one of them the noise
peak. Other triplets' primary answers stay in the pool deliberately: a real knowledge base holds documents
about adjacent topics, and a pool with no topical adjacency would measure an unrealistically easy problem.

**Two pairs of floors are judged, not one.** What a model-quirk seed declares for an embedder is a statement
about the model; `RAG_MIN_SCORE` and `RAG_REL_MARGIN` are what retrieval actually applies, and the two coincide
only until somebody edits an environment file. So a run reports both - the declared verdict answers "is what the
repo believes about this model still true", and the configured verdict answers "is this deployment retrieving
correctly". Reporting only the first could say "the floors hold" while the numbers in force sat above the
related band and dropped true answers on every turn, which is the failure the whole check exists to catch.

The prefix comparison is part of the answer rather than a curiosity: on an asymmetric embedder the wrapper is
supposed to widen the corridor, so a wrapper that does not is a finding about the floors sitting beside it.

Readings persist to the same `model_probes` log at the `embedder` depth, which none of the agency reads ever
asks for - an embedder has no tool loop to earn anything in, so its row is history for a person to read and an
input to nothing the app decides.

## 8. Settings & configuration

Two scopes, and deliberately no third:

- **Conversation-scoped** - model, thinking mode, web-search mode. Persisted per conversation in a settings JSON
  column (Phase 2); surfaced as composer controls - model selector + thinking toggle + web-search toggle
  (Phase 6).
- **Workspace-scoped** - the same three axes as a default for every turn, plus two a conversation has no
  business holding: a context-window ceiling and a reranker toggle, both deployment-shaped choices about how
  much machine a turn may spend. Stored tenant-scoped in Postgres (Phase 2) and written from the console's
  Workspace defaults page. The MCP server registry (name, base URL, enabled, secret reference) and the upstream
  API secrets are workspace-scoped too and are managed from the Integrations page (Phase 5).

Neither scope carries a system-prompt override. The persona composition has no override seam yet, so there is
nothing for such a field to feed; a session-specific override arrives when it grows one.

**Secrets are never plaintext and never leave the server.** A `SecretStore` seam, first implemented as AES-GCM
encryption at rest with a master key supplied via the environment. The API is write-only: values go in; only
metadata plus the last 4 characters ever come back. Secrets are injected into MCP sidecars per request and never
persisted there. A secret is either workspace-scoped - one service account standing in for everyone - or scoped
to the individual user who connected it (section 7).

**Role gating is live.** A `require_role` dependency reads the role off the acting principal and admits only an
administrator. Where an identity provider is configured that role comes from the signed-in user's `users` row and
is re-read on every request, so a demotion takes effect on the next one; where it is not, the seeded principal is
an administrator and the gate passes, which is the single-user behaviour the platform has always had. The
`users.role` column has existed since Phase 2, so turning this on needed no migration.

**What sits behind that gate is a per-route decision, not a per-page one.** The member list is administrators
only, roster and role change alike - the roster names every colleague's address, which is not something a member
needs. The integrations registry splits: reading which servers exist is a workspace fact any member may see, while
registering one, re-pointing it, deleting it or storing a workspace credential is admin-gated, because those
verbs reach an arbitrary URL on the deployment's behalf. The Model Manager splits the same way: its inventory
reads openly, but the hardware-SPENDING mutations - pulling weights, deleting them, running a verify or
embedder-check probe, and the registry preflight a pull needs - are admin-gated, the same routes the rate limiter
also gives their own small budget. Warming a model for a turn is deliberately NOT in that set: a member picks a
model to chat with, so warming it is part of ordinary use and stays open to any signed-in member.

**Who administers a workspace is deliberately not a settings field.** It is the members list (section 10), plus
one floor from the deployment environment: `OIDC_ADMIN_EMAILS` names addresses that are administrators at every
sign-in and that the app refuses to demote. That floor and the last-administrator guard exist for the same
reason - a workspace must not be able to lock itself out of its own console - and both refusals name where the
change actually belongs.

**The default system prompt is composed, layered, and owned by the services layer.** The base persona: concise,
direct, answer-first; admits uncertainty and missing capabilities plainly; asks one clarifying question when a
request is genuinely ambiguous; no flattery - plus an English-language anchor that keeps small local models from
drifting between languages. Later phases append citation and tool-guidance layers; the per-session override
noted above waits on an override seam in this composition.

The `SYSTEM_PROMPT` env var controls the **persona**: unset means the composed default, an empty string disables
the system prompt entirely (a deliberate choice to run bare, and it is honoured), and any other value replaces
the persona. It does **not** replace the tool guidance, which is appended to whatever persona is in force. The
guidance is a capability contract rather than a voice: an operator who rewrote the persona would otherwise be
handed a model that the API is offering tool schemas to, with no instruction to refuse honestly when no tool
fits and never to invent one - and tool calling would break invisibly, which is the worst way for it to break.

All configuration flows through `pydantic-settings` from the environment, documented in `infra/.env.example`.
`CORS_ALLOW_ORIGINS` accepts either a comma-separated string or a JSON array.

## 9. Multimodal (image + PDF), session-scoped

- **Images are read for their text first.** An uploaded image is OCR'd (optical character recognition: pulling
  text out of a picture of text), so a screenshot of an error or a photo of a page is answerable by an ordinary
  text model - the graceful path that works on any hardware. An image that holds no readable text is still
  accepted and stored (not rejected for having no text), because its value is not only its text: a vision model
  can look at the picture itself.
- **Vision-model passthrough** - handing the raw picture to a model that can see it - is built, and gated on the
  runtime's report about the chat model actually serving the turn. Nothing written down could have been enough
  on its own: a tag is mutable, and a re-pull swaps the weights beneath the name. There is no separate vision
  role and no second model to select - a deployment running a chat model the runtime reports as vision-capable
  gets passthrough, and one that is not gets the OCR path above plus an honest "this picture was not shown to
  the model" notice. A report that cannot be taken resolves to *not seen*, never to seen: "the runtime would not
  say" is not permission.

  Images are sized before they are sent, against costs measured on the serving runtime rather than read off a
  model card. A visual token covers a 32x32 pixel cell, and the runtime clamps every image into a band: a floor of
  about 1030 tokens however small the picture, and a ceiling a few thousand above it. Two things follow. Shrinking
  below the floor saves nothing and only loses detail, so it is never done. And the budget is spent on cells, not
  on the longest side, so an image is fitted by AREA - a 16:9 picture squeezed into a square box wastes almost half
  the budget and measurably loses text that an area fit keeps.

  Pixels SUPPLEMENT the extracted text, never replace it, so a model that reads a picture badly still has whatever
  OCR found. Delivery fails soft per image: an unreadable file, a deleted blob, a wedged store or a failed probe
  costs that one picture and is reported, never the answer. Because image tokens are invisible to a text-based
  token estimate, their cost is handed to the retrieval budget explicitly - without that, retrieval hands back room
  the pixels already took and the runtime silently truncates the prompt from the head, dropping the system prompt
  with the tool guidance and injection defense inside it.
- **PDFs are never sent as raw bytes.** Text is always extracted into RAG; a scanned page whose text layer is
  sparse is rasterized and OCR'd so text-only models still read it. Rasterizing a page *for a vision model to see*
  follows the same capability gate as images above.
- **Lifecycle:** validate (MIME + magic bytes + size limits) → store (keyed `tenant_id/session_id/attachment_id`)
  → reference within the session → cleanup deletes both the stored files and the ephemeral vectors.

Document parsing stack (all permissive licenses, keeping a closed-source option open):

| Library | License | Role |
|---|---|---|
| pypdfium2 | Apache-2.0 / BSD | Text extraction + page rasterization |
| python-docx, openpyxl, python-pptx | MIT | Text from Word, Excel and PowerPoint files |
| striprtf | BSD-3-Clause | Text from RTF documents |
| defusedxml | PSF-2.0 | Parsing the XML inside a zip container without resolving entities |
| pytesseract + Tesseract | Apache-2.0 | OCR for scanned pages and uploaded images |
| Pillow | MIT-CMU (HPND) | Decoding an uploaded image, and rendering a PDF page, for OCR |

OpenDocument and EPUB need no library of their own: both are zip archives of XML, so they are read with the
standard library plus the container defences the office formats already brought.

> **PyMuPDF / pymupdf4llm are excluded** - they are AGPL-3.0, incompatible with a closed-source/commercial pivot
> without a paid license. Likewise `unstructured`'s hi-res extra (it pulls AGPL `ultralytics`).

> **docling is an opt-in heavy dependency, not the default path for office files.** It reads a document by running
> layout and table-structure MODELS over it, so a plain install pulls PyTorch and downloads weights on first run -
> gigabytes in the image, paid for by every build and every deployment, and forbidden outright by a hermetic test
> run and an air-gapped deploy. The three MIT readers above extract the same content directly (an office file is
> zipped XML), in pure Python, with no system dependency and no model. The trade is worth naming: docling
> reconstructs hard layout - multi-column pages, merged cells - far better than a direct reader can, so it remains
> the answer where fidelity demands it, as a targeted optional dependency rather than a default the platform carries.

## 10. Data model & multi-tenancy

Postgres holds tenants, users (with a `role` column from day one), conversations (title plus a settings JSON
column), messages (role, content, any captured thinking, the producing model, and a metadata JSON that flags
interrupted generations), collections, attachments, `workspace_settings`, `mcp_servers` (each row a registered
tool server, carrying how it authenticates and whose credential its calls use), `secrets` (every credential,
encrypted at rest and bound to the row that owns it), `oauth_authorizations` (a consent in flight, consumed once
by the callback that finishes it), `model_probes` (the measurement log behind a model's declared
tool-calling verdicts), and `model_policy` (append-only operator declarations - agency ceiling,
thinking-with-tools, think mode, context window, thinking pair - digest-bound and folded per axis).
`workspace_settings` is written from the console's Workspace defaults page; the tool phase brought
`mcp_servers`, `secrets` and their neighbours to life (section 7). **Every domain row carries
`tenant_id` (+ `user_id`).** A base repository auto-injects `tenant_id` on every query, and
routers never touch raw DB sessions - so a single forgotten filter cannot leak another tenant's data. A test
asserts that a model- or client-supplied tenant id can never override server-side scoping.

Conversations are managed through a small REST surface (list, detail, rename, delete) that backs the sidebar UI;
titles are auto-generated from the first message. Schema changes ship as Alembic migrations (versioned, scripted
schema updates), and the backend applies any pending migrations automatically before it starts serving - a fresh
clone needs no manual database step (`make db-upgrade` runs them by hand when needed).

There is deliberately **no model-registry table**, and no committed catalog file either: live model-server state
behind a short in-process cache is the source of truth for which models exist - a stored copy would only drift.
The `model_probes` measurement log in section 7 and the `model_policy` declaration store are the two deliberate
exceptions, and neither mirrors an inventory: one records what a probe MEASURED, the other what an operator
DECLARED, and both are facts no other system holds. Revisit the rule itself only if per-tenant model allowlists
arrive after auth.

**`get_current_tenant()` is the whole of the auth boundary, and it now resolves real people.** With an OIDC
issuer configured it reads the session cookie, finds the signed-in user's row, and returns a context carrying
that tenant, that user, and that user's role; with none configured it returns the seeded default tenant exactly
as it always did. Every repository, service and router downstream consumes the same context either way, so the
sign-in cutover changed one dependency and no call sites - which is what the multi-tenant schema had been
carrying `tenant_id`, `user_id` and `users.role` since [Phase 2](ROADMAP.md#phase-2---persistence--multi-tenant-schema)
in order to make possible.

The one thing the schema could NOT carry ahead of time was the identity KEY. The claim-shape promise held - every
downstream consumer of the tenant context was already right - but resolving a user by their provider identity
needed two new `users` columns (`oidc_issuer`, `oidc_sub`) and a partial unique index over them, which the cutover
added as an Alembic migration. So "no schema change" was true of the tenancy plumbing and not of the identity key;
[DATA-MODEL.md](DATA-MODEL.md) records the same amendment against the same promise.

Two consequences worth stating. **The role is read fresh from the database on every authenticated request**
rather than baked into the session, so demoting somebody takes effect on their next request instead of whenever
their session happens to expire. And **the user dimension stopped being decorative**: `UserScopedRepository`
filters conversations, messages and attachments by user as well as tenant, and the ephemeral vector scope a
conversation searches is filtered the same way - defence in depth beneath the ownership check the routers already
make, so a passage one colleague uploaded cannot ground another's answer.

Roles are the coarse `admin | member` pair the schema has always held. Named permissions - application-level
roles as bundles of typed permissions, enforced through these same repository-and-dependency seams, deliberately
not SQL-level roles since the backend connects as one service account and most permissions ("may this role use a
tool") are not rows - remain future work
([ROADMAP Phase 8b](ROADMAP.md#phase-8b---identity--access)).

A table-by-table reference - every column, index, and delete rule, with the reasoning behind each - lives in
[DATA-MODEL.md](DATA-MODEL.md).

## 11. Models: what runs, and what is known about them

Two questions look like one and are not. *Which* models a deployment runs is a configuration answer that only
that deployment can give. *What is true about* a model - can its chat template take a tool list, does it always
reason, how wide is its window - is a fact about weights, and belongs nowhere near a configuration file.

The platform used to answer both from one committed catalog keyed by hardware class. That file could only ever
describe what a class was *built with*: point the stack at a model server holding a different inventory and it
produced a healthy connection and a broken deployment - stored model picks discarded for not being listed,
installed models missing from the picker, and unlisted tags treated as capability blackouts (no tools, no
window, `deterministic` agency). It is gone, and the two questions are answered separately below.

### 11.1 Four sources of model knowledge, none of which can go stale quietly

Everything the platform believes about a model comes from one of four places. They are ranked, strongest
first, and every surface reads them through the same resolvers - so a card in the management view and the turn
it describes can never disagree.

| Source | What it answers | Why it cannot rot |
|---|---|---|
| Operator declarations | Policy | Bound to the installed digest |
| Measured verdicts | What the model demonstrably did | Measured on the weights in front of it |
| The model-quirk seed | Family facts no runtime reports | Ships with the code that reads it |
| The runtime's report | Mechanics | Read from the server doing the serving |

What each covers, in order:

- **Operator declarations** set the agency ceiling, the reasoning mode, the thinking partner, and the permission
  to reason and call tools at once. They are stored per tag against the digest the tag pointed at, so a re-pull
  orphans them rather than letting them describe different weights.
- **Measured verdicts** come from the checks: how far a model got through the tool-calling harness, whether it
  actually reasons when the flag says so, what score bands an embedder produces.
- **The model-quirk seed** is a small code-shipped file of family facts nothing reports - which tag of a family
  always reasons, an asymmetric embedder's query wrapper, a packaging channel that strips the tool parser. Every
  entry carries dated provenance, so a reader who disagrees with a line can go find the measurement it names.
- **The runtime's report** answers the mechanical questions: does this template accept a tool list, does it
  accept images, what is the model's native context window.

Four properties of that stack are worth stating outright, because each was paid for:

- **A declaration outranks a measurement for *policy*, and never for *capability*.** An operator may say how
  much a model is allowed to drive; nobody may declare that it succeeds. The effective agency mode is the
  lowest of the declared ceiling, what a check actually proved, and what the deployment permits
  ([section 7](#7-tools-design) has the full arbitration).
- **Declarations are digest-bound.** A tag is a mutable pointer - one in this project's own history was
  re-pointed from a toggleable model to an always-reasoning refresh under the same name - so a declaration
  names the weights it was made about. Re-pull the tag and it reverts to unclaimed rather than silently
  describing something else.
- **The seed gates nothing.** It informs priors and carries data for named readers. A model it has never heard
  of works completely: it starts unknown and earns its clearances through the same checks as anything else.
  That is the inversion the retired catalog was worth making - a stale catalog broke new models, while a stale
  seed merely knows less about them.
- **The runtime's report is mechanics, not behaviour.** A flag saying "thinking" reports that the serving
  integration has think plumbing, not that the weights reason - a never-reasoning instruct tag advertises it.
  So reasoning mode is measured and confirmed, never inferred from the advertisement, and a mismatch clamps
  toward safety (treated as always-reasoning, so raw reasoning markup can never reach an answer).

Capability therefore lives with the **(model, runtime, digest)** triple rather than with the weights alone. The
same checkpoint behind a different provider can legitimately behave differently - a tool call parsed natively
under one runtime arrives as prose under another - which is why a probe records the runtime version and the
serving endpoint alongside the digest, and why a verdict earned on one machine does not certify another.

### 11.2 Two chains decide what a deployment runs

Both are ordinary configuration lookups, and both are resolved in exactly one place so no surface can answer
differently from the turn.

**The chat model:** a conversation's own pinned model, then the workspace default set on the console, then
`DEFAULT_CHAT_MODEL`, then nothing. A stored pick is honoured unless the caller holds the provider's inventory
*and* that inventory says the tag is absent, in which case it falls through rather than wedging the
conversation.

"Nothing" is a first-run state rather than a fault, and it does not last. The first turn against a server
holding exactly one chat model adopts that model as the workspace default; a server holding nothing that can
chat answers with a typed error naming the fix. A hardcoded fallback tag here would be a guess at somebody
else's inventory, and it used to be exactly that.

**The embedding model:** the workspace's stored choice, then the model this deployment's own stored passages
were written by (sampled from the vector store), then the single installed embedding-capable model when there
is exactly one. Several installed and nothing selecting one is **refused**, deliberately - picking between two
embedders would be a ranking, and the wrong pick writes vectors into a space the other model can never read.
None installed means retrieval is unavailable and says so.

That second rung is what makes an upgrade silent in the right way: a deployment that never touches a settings
page keeps embedding with the model its own corpus was built by, instead of quietly opening a second vector
space beside it. The [stranded-vectors alarm](#6-retrieval-rag-design) watches the case where it cannot.

`GET /api/v1/models` is the one list both the composer and the management view draw from, and it reports what
the model server actually holds - annotated with role, measured facts, and why a model is clamped where it is.
There are no rows for models nobody downloaded and no listed-versus-unlisted distinction, because there is no
list. Downloading weights belongs to the management view, which can see what is already there.

### 11.3 Hardware is expressed by what is pulled and how wide a turn may be

Nothing detects the hardware and nothing decides on its behalf. A laptop and a cloud GPU run the same
containers and differ in exactly two respects:

- **Which models are downloaded onto the machine.** A 4GB card is comfortable with a 4B-class chat model and a
  small embedder; a 16GB card runs a 14B-class one. That choice is made in the Model Manager, against the
  registry manifest's stated download size and, for a model a check has run against, the VRAM the probe found
  resident - fact against fact, never a computed fit prediction, because a predicted fit is how an
  out-of-memory load gets blamed on the model.
- **How much of that machine one turn may spend.** `DEFAULT_CONTEXT_WINDOW` is the deployment-wide answer, and
  a workspace context ceiling set on the console replaces it - upward as well as downward, still bounded by the
  model's own native window. The 8192 default is a boot-anywhere floor, not a recommendation: a modern tag's
  native window (131072 on the llama 3 family, 262144 on the qwen3 2507 refresh) allocates a KV cache that fits
  almost nowhere, so the floor has to be small enough to start on a small card.

The remaining machine-scoped decisions are single environment keys for the same reason: `EMBEDDING_DEVICE`
decides where embedding work runs (`cpu` by default, since an embedder sharing a one-model card evicts the chat
model on every retrieval), and `RERANKER_URL` decides whether a reranker exists at all - setting it *is* the
switch, and `make up` starts the sidecar exactly when it is set, so the container and the code cannot disagree.
`RERANKER_MODEL` names the weights that sidecar loaded, which nothing else can look up.

### 11.4 The serving choices behind all of it

- **Ollama-first** because one REST API covers chat, embeddings, vision, and model pull/list/remove (which
  powers the Model Manager UI), with GPU/CPU fallback. vLLM or hosted APIs stay available behind the
  `ModelProvider` seam, and per role rather than per deployment (section 4.1) - the Model Manager stays honest
  across providers, too: management verbs (pull, delete, list installed) are provider capabilities the UI reads
  per row, so a provider without a pull API (vLLM loads its model at process start) shows "changed via
  deployment" rather than a dead button.
- **The Qwen3 family** is the best-tested locally-runnable tool-caller as of mid-2026, which is why the quirk
  seed knows it best. Note the 2507 split on the small tag: plain `qwen3:4b` is the always-thinking refresh
  variant and `qwen3:4b-instruct` the never-thinking one, which is the case model pairing exists for
  (section 5). Larger tags in the family (`qwen3:8b/14b/32b`) are hybrids whose reasoning toggles per request
  and need no partner.
- **A newer family is not automatically a better tool-caller.** The Ollama renderer bug that keeps qwen3.5's
  larger tags out of the tool-calling flow remains open upstream as of mid-2026 (claims that it was fixed do
  not survive checking the issue itself), which is why a model has to demonstrate the loop on the runtime that
  actually serves it before it is trusted with one. Three separate questions are tracked per model, and
  conflating them is the mistake this split exists to prevent: **tools** (mechanical - can the chat template
  take a tool list at all), **agency** (policy - the strongest mode this model may attempt), and
  **thinking_with_tools** (capability - can it reason and call tools in the same turn). The first is read off
  the runtime's report; the last two are earned from the tool-check diagnostic, never from a version number
  (section 7).
- **Only the reranker needs a sidecar** (Infinity) because Ollama has no rerank endpoint. It asks for CPU and a
  second or two per search rather than a share of the GPU, which is why it belongs in the standard set; a host
  too memory-constrained to hold it beside the warmed models leaves `RERANKER_URL` unset and retrieves
  dense-only.

## 12. Containers & deployment

- `docker compose` with a base file carrying every service definition, plus exactly one of two overlays: `override`
  (dev hot-reload) or `prod` (restart policies, nothing published but the proxy). `gpu`/`cpu`/`remote` profiles
  choose where models are served, independently of which overlay is in play. Named volumes persist Ollama models,
  Postgres, Qdrant, storage, and the proxy's certificates.
- **One command, and the machine decides what it means.** A `DEPLOY_ENV` variable in the environment file picks
  the overlay, so the same `make up` is correct on a laptop and on a server, and a routine deploy cannot
  reassemble a server from the development layout by accident. The development overlay hot-reloads bind-mounted
  code and publishes the app's own ports, bound to loopback by default (`BIND_HOST`) - closing a gap a host
  firewall cannot on Linux, where a published port is matched by Docker's own firewall chain before `ufw` ever
  sees it. The production layout adds restart policies and publishes nothing but Caddy, so Postgres, Redis,
  Qdrant, the backend and Ollama are reachable only from inside the compose network. Day-to-day commands
  (`make warm`, `make status`, `make db-upgrade`) behave identically in both layouts, so switching overlays is a
  configuration change, never a different playbook.
- **Hardened by default:** containers run as non-root users, images are pinned to specific versions rather than
  floating `latest` tags, and the prod compose publishes only Caddy's ports - the backend, the web UI, Postgres,
  Redis, Qdrant, and Ollama are unreachable from outside the internal Docker network.
- **Same-origin by design:** in dev the Next.js server proxies API calls to the backend, so the browser sees one
  origin and CORS (the browser rule restricting cross-site requests) never triggers; in prod Caddy serves the UI
  and API from one origin, terminates TLS, and proxies with response buffering disabled so token streams reach
  the browser as they are produced. The backend keeps a thin CORS middleware only as an escape hatch for
  non-proxied clients.
- **TLS needs no manual step.** Caddy obtains and renews certificates itself: publicly-trusted ones once `DOMAIN`
  names a real hostname (with `ACME_EMAIL` for the Let's Encrypt registration), and an internally-signed pair
  before then - an unset or blank `DOMAIN` defaults to `localhost`, so a box with no DNS yet still runs the
  production layout, HTTPS included, end to end.
- **The streaming path is the one to keep in mind when changing the proxy.** The browser's server-sent events do
  not arrive over the versioned API prefix: they come through the frontend's own route handlers, which relay the
  backend stream on the same origin. Buffering configured only on the API prefix would therefore look correct and
  still deliver every answer in one lump at the end.
- **No cross-worker stream fan-out.** The GPU is the bottleneck: one async worker saturates one Ollama, so SSE
  streams stay in-process instead of being relayed through Redis. Redis remains for background jobs and caching;
  revisit only if horizontal backend scaling ever becomes real.
- **Backend base image `python:3.13-slim`** - the standard GIL build; the free-threaded variant buys nothing for
  an I/O-bound web service and its native dependencies are not ready for it.
- **Rate limiting is abuse protection, not a quota, and it lives in the app rather than the proxy** - the app is
  the only layer that knows which routes are expensive, and the only one that still applies against a bare
  `docker compose up` with no Caddy in front. Each request is classified into one of three per-minute budgets -
  `general` (everything else under `/api/v1/`), `heavy` (model pulls, ingestion, uploads, the OAuth callback,
  connecting or disconnecting an OAuth grant, testing an integration's connection, downloading attachment
  content, and the probe endpoints that evict and reload a model), and `light` (the one route a healthy client
  legitimately polls many times a minute, the attachment status check) - counted in fixed one-minute windows in Redis and
  refused with a `429` plus `Retry-After` once a budget is spent. `TRUSTED_PROXY_IPS` (CIDR ranges, empty by
  default) names which peers' `X-Forwarded-For` header may be believed, and the caller's key is read from the
  rightmost entry that is not itself a trusted proxy - reading the leftmost, client-supplied entry would let any
  caller mint a fresh identity per request and walk straight through the limiter. It fails **open**: a counter
  store that cannot be reached lets the request through rather than turning a bookkeeping outage into a site
  outage, and every lapse is logged, throttled so an outage cannot flood the log. `POST /chat/stream` is limited
  as `general` traffic only - how many streams may run *at once* is a separate axis, the `StreamLimiter`
  (`MAX_STREAMS_PER_TENANT`). One follow-up is deferred to
  [Phase 8b](ROADMAP.md#phase-8b---identity--access): the limiter keys on client IP, so once real users exist a
  shared office or VPN address counts as one caller for everybody behind it - re-keying it on the authenticated
  principal, with IP as the fallback for the unauthenticated login routes, stays open. A decision on SSE
  resume/replay after a disconnect is likewise open (see [Phase 7b](ROADMAP.md#phase-7b---resumable-streaming)).
- **`GET /readyz` splits its dependencies into hard and informational.** Postgres, Redis and the vector store are
  hard - any one down answers `503`, because the app genuinely cannot serve without them. The model server, the
  search sidecar and the reranker are reported for visibility but never flip the verdict: the model server is
  routinely remote and can take on the order of a minute to cold-load its first model, and the other two are
  optional by design, so a slow or absent one degrades the app rather than marking a serviceable deployment
  unready. `GET /healthz` stays a bare liveness check; both are unauthenticated and mounted outside `/api/v1` so
  an orchestrator's probe configuration outlives any API version, and the rate limiter exempts them explicitly -
  a limiter that could answer `429` to a readiness probe would take the deployment down by itself.
- **Backup and restore cover the three named volumes a reinstall cannot reproduce** - `postgres_data`,
  `qdrant_data` and `attachment_data` - by dumping or snapshotting each through its own service rather than
  copying it file by file, since a live volume copied while running can span several inconsistent moments and
  the result may not restore. Under `STORAGE_BACKEND=s3` the attachment bytes instead live in the object store
  (the bundled MinIO profile's own `minio_data` volume) with its own backup story, so that volume is skipped and
  the skip is recorded loudly in the backup's manifest. `make restore` refuses while the application is up and
  refuses a backup whose manifest checksums do not match, because restoring is the one operation here that
  destroys the current state. Mechanics live in
  [MAINTENANCE.md section 3](MAINTENANCE.md#3-backups-and-restores); the full volume-by-volume accounting is in
  [DATA-MODEL.md](DATA-MODEL.md#what-lives-in-each-volume).

## 13. CI & supply-chain security

CI runs the dependency audit as a committed check, so the guarantee does not depend on any individual machine's
setup - though it is advisory rather than enforced, for the reason given below the workflow. Sketch of
`.github/workflows/frontend-audit.yml`:

```yaml
name: js-security-audit
on:
  # main only: a push to an open PR branch already audits via pull_request below.
  push:
    branches: [main]
    paths: ['apps/frontend/**', 'packages/api-types/package*.json', '.github/workflows/frontend-audit.yml']
  pull_request: { paths: ['apps/frontend/**', 'packages/api-types/package*.json'] }
  schedule: [{ cron: '0 4 * * *' }]        # catch newly-disclosed CVEs against an unchanged lockfile
jobs:
  npm-audit:
    runs-on: ubuntu-latest
    # EVERY workspace with a lockfile, not just the app: a gate that watches one
    # directory silently exempts the rest, which is how two high advisories once
    # sat unnoticed in the type-generator package.
    strategy:
      fail-fast: false
      matrix: { workspace: [apps/frontend, packages/api-types] }
    defaults: { run: { working-directory: '${{ matrix.workspace }}' } }
    steps:
      - uses: actions/checkout@v7
      - uses: actions/setup-node@v7
        with: { node-version: '22', cache: 'npm', cache-dependency-path: apps/frontend/package-lock.json }
      - run: npm ci
      - run: npm audit --audit-level=moderate
  osv-scanner:
    uses: google/osv-scanner-action/.github/workflows/osv-scanner-reusable.yml@v2
    with: { scan-args: "-r --lockfile=apps/frontend/package-lock.json" }
```

Plus **Dependabot** (npm security + version updates, grouping the `next`/`react` ecosystem). The `npm-audit` job
runs today on every push and pull request that touches the frontend, and nightly; the `osv-scanner` job above is
the intended shape and is not wired yet ([Phase 8](ROADMAP.md#phase-8---hardening-ci-gates-deployment)). Neither
is a *required* status check, because branch protection is not available on this repository's plan - a red audit
blocks a merge by convention rather than by enforcement, which is worth knowing before trusting the merge button.
Next.js is kept at or above the latest patched release and bumped promptly on security releases; where a
transitive dependency carries the advisory, it is pinned forward with an override rather than by downgrading the
framework.

CI also guards the API contract itself: the TypeScript types in `packages/api-types` are generated from the
backend's OpenAPI schema, and a drift check regenerates them on every change and fails when the committed output
is stale - so the frontend and backend cannot quietly disagree on request or response shapes.
