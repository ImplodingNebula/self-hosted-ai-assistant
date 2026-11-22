"""Application settings, loaded from the environment (and an optional .env)."""

from __future__ import annotations

import json
from functools import lru_cache
from typing import Annotated, Literal
from urllib.parse import urlparse

from pydantic import Field, ValidationInfo, field_validator, model_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# A century, as the ceiling on either retention window. Any real policy is a
# handful of years; the bound exists so an accidental digit is refused at boot
# rather than raising OverflowError inside a background sweep an hour later.
_MAX_RETENTION_DAYS = 36_500

# The reranker sidecar this repo's compose file starts. Named once so the field
# default and the blank-value rescue below cannot drift into two spellings of
# one model id.
DEFAULT_RERANKER_MODEL = "bge-reranker-v2-m3"


class Settings(BaseSettings):
    """Typed application configuration.

    Values come from environment variables (see infra/.env.example); fields fall
    back to sensible local-dev defaults so the app runs with zero config.
    """

    # hide_input_in_errors: a failed validation must never echo the raw value
    # back into the error text - several fields here are secrets (auth headers,
    # the master key), and a boot-time ValidationError goes straight to stderr
    # and the container log.
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        protected_namespaces=(),
        hide_input_in_errors=True,
    )

    app_name: str = "AI Chatbot Platform"
    # Which ModelProvider implementation to use. A Literal so a typo in
    # MODEL_PROVIDER fails fast at startup instead of silently misrouting.
    model_provider: Literal["ollama", "fake"] = "ollama"
    ollama_url: str = "http://ollama:11434"
    # Extra HTTP headers sent VERBATIM on every backend call to the model
    # server, given as a JSON object (e.g. {"Authorization": "Bearer ..."}).
    # Ollama authenticates nothing itself, so a server reachable beyond this
    # deployment is only ever protected by a reverse proxy in front of it, and
    # these are the credentials that proxy expects. THE VALUES ARE SECRETS:
    # never log them, never echo them, never fold them into an error message.
    # NoDecode: the raw string must reach the validator below, which reads an
    # empty value as "no headers" - compose forwards an unset key as an empty
    # string, and the default JSON decode would fail the whole boot on it.
    ollama_auth_headers: Annotated[dict[str, str], NoDecode] = {}
    # Read timeout for the backend's calls to Ollama, in seconds. This is the
    # ceiling on a COLD MODEL LOAD, not a cost paid per request - and it must
    # OUTLAST the server's own OLLAMA_LOAD_TIMEOUT (set beside it in compose),
    # because a client that hangs up mid-load makes Ollama ABORT the load: the
    # shorter this is, the more it kills slow-but-healthy loads on a busy box.
    # Measured live 2026-07-29: a contended host stretched a healthy 2.5GB
    # load past 6 minutes, and the old 300s ceiling (plus Ollama's default 5m)
    # aborted every warm for a day.
    ollama_read_timeout_seconds: float = 960.0
    # The REMOTE registry a model name is checked against before a download
    # (does this tag exist, how many bytes would arrive). Only matters where a
    # mirror or a proxy sits in front of the public registry; the default is
    # the public one. Never combined with anything a user types: a request
    # supplies a NAME, and the host it is looked up on comes from here alone.
    ollama_registry_url: str = "https://registry.ollama.ai"
    # The deployment's chat model, below the two stored rungs (a conversation's
    # pin and the workspace default) and above nothing. Unset or empty on a box
    # nobody has configured is a first-run state, not a fault: the first turn
    # against a server holding one chat model adopts it as the workspace
    # default. Resolution lives in model_resolution.resolve_chat_models.
    default_chat_model: str = ""
    # OPTIONAL override for the thinking-pair sibling. Unset or empty defers to
    # whatever an operator declared for the tag from the console; the literal
    # value "none" disables pair-routing outright; any other value overrides the
    # tag. Empty CANNOT mean "disabled": compose forwards an unset key as an
    # empty string, so empty must read as unset.
    thinking_chat_model: str | None = None
    # NoDecode: pydantic-settings must hand the raw env string to the validator
    # below instead of insisting on JSON for a list-typed field.
    cors_allow_origins: Annotated[list[str], NoDecode] = ["http://localhost:3000"]
    # System prompt policy: None (unset) -> the composed default from
    # app.services.prompts; "" -> no system prompt at all; any other string ->
    # full override, used verbatim.
    system_prompt: str | None = None
    # Async SQLAlchemy URL (env DATABASE_URL). The default targets a local
    # docker-published postgres so bare-metal dev tools (alembic, scripts) work
    # without compose networking; inside compose the env var points at the
    # `postgres` service host instead.
    database_url: str = "postgresql+asyncpg://chatbot:chatbot@localhost:5432/chatbot"
    # Echo SQL statements (debugging aid only - very noisy under streaming).
    database_echo: bool = False
    # Server-side history rebuild: cap the rebuilt context by approximate size
    # (1 token ~ 4 chars - cheap and close enough for truncation) so long
    # conversations cannot blow the model's context window.
    history_token_budget: int = 4000
    # ... and by message count, so a burst of tiny messages stays bounded too.
    history_max_messages: int = 20
    # Idle gap between SSE keep-alive comment frames; proxies (Caddy, browsers)
    # drop quiet connections otherwise. <= 0 disables heartbeats (tests).
    sse_heartbeat_seconds: float = 15.0
    # Concurrent chat streams allowed to EACH USER of a workspace; the GPU
    # serializes generations anyway, so a small cap fails fast (429) instead of
    # queueing invisible work. The key kept its name when the budget moved from
    # the workspace to the person (a shared budget let one member's long
    # agentic turns lock colleagues out); renaming it would have broken every
    # deployment's .env for a wording change.
    max_streams_per_tenant: int = 2
    # Root log level for the structured logging setup.
    log_level: str = "INFO"
    # The context window a turn asks for when no workspace ceiling names a
    # number. It is pure deployment config: how much of this machine one turn
    # may spend, which no platform constant can know. The window that actually
    # runs is the smaller of this and the model's own native window, so raising
    # it never asks a model for more than it has.
    #
    # 8192 is a conservative STARTING value, not a recommendation. A runtime
    # asked for a model's full native window allocates KV cache for all of it,
    # and a modern tag's native window (131072 on the llama 3 family, 262144
    # on the qwen3 2507 refresh) is an allocation that fits almost nowhere -
    # so the floor has to be small enough to boot on a 4GB card. A deployment
    # with real memory should raise this to what its hardware affords; a
    # workspace that names a context ceiling replaces it outright.
    #
    # Floored at the same 4096 the workspace ceiling is floored at, because it
    # feeds the same slot from the other direction and a typo here is silent:
    # a thinking turn already reserves 2048 tokens for its answer, so anything
    # smaller leaves nothing for the system prompt and the question, and the
    # runtime truncates the prompt from the HEAD. Refused at boot instead. The
    # literal is repeated rather than imported (settings must not depend on a
    # schema module) and pinned equal to MIN_CONTEXT_CEILING by a test.
    default_context_window: int = Field(default=8192, ge=4096)
    # Where embedding work runs, and the ONLY thing that decides it. Placement
    # is a fact about the MACHINE - how much card there is and what else wants
    # it - so it belongs in deployment configuration rather than in a model
    # file that ships identically to every box. "cpu" is the default because
    # the failure it prevents is the expensive one: an embedder sharing a card
    # with the chat model evicts it on every retrieval, and a multi-second
    # reload per turn reads as a broken deployment rather than as a placement
    # choice. A box whose GPU has room to spare sets "auto" and gets faster
    # ingestion for it.
    embedding_device: Literal["cpu", "auto"] = "cpu"
    # Vector store (Qdrant) base URL; the default targets the compose service.
    qdrant_url: str = "http://qdrant:6333"
    # Redis URL for the arq job broker (the ephemeral-vector TTL sweep).
    redis_url: str = "redis://redis:6379/0"
    # Reranker sidecar base URL, and the whole gate: a URL here means an
    # operator wired a sidecar and retrieval re-scores through it, while None
    # keeps retrieval dense-only. Nothing else votes. Three gates used to have
    # to agree (this URL, a committed file naming a reranker, and that file's
    # enabled flag), which meant the shipped dev default - a URL set beside a
    # declaration that disabled reranking - looked configured and did nothing.
    # A workspace toggle can still WITHHOLD a wired sidecar; nothing can conjure
    # one, because a settings page cannot start a process.
    reranker_url: str | None = None
    # The model id the reranker sidecar serves, as the sidecar itself was
    # started with (`infinity v2 --model-id ...`). A rerank call carries a model
    # name, and the sidecar is the only party that knows which weights it
    # loaded - so this is deployment configuration beside the URL rather than a
    # fact anything can look up. Changed only alongside the sidecar's command.
    reranker_model: str = DEFAULT_RERANKER_MODEL
    # Which StorageBackend implementation holds uploaded bytes. A Literal so a
    # typo in STORAGE_BACKEND fails at boot rather than silently writing to the
    # wrong store. Unlike the reranker and search sidecars, storage has no
    # graceful-degrade mode - an attachment the app cannot read is an attachment
    # it cannot answer about - so one backend is always built. "local" is the
    # single-node default; "s3" is what several backend replicas need, since the
    # replica serving a download is rarely the one that took the upload.
    storage_backend: Literal["local", "s3"] = "local"
    # Where the local-fs backend writes. Backed by a named volume in compose, so
    # the files survive a container recreate. Ignored when STORAGE_BACKEND=s3.
    storage_local_dir: str = "/data/storage"
    # S3-compatible service settings, read only when STORAGE_BACKEND=s3.
    # Base URL of the service. None targets AWS's own endpoint for the region;
    # a MinIO deployment sets its service URL (http://minio:9000 in compose).
    s3_endpoint_url: str | None = None
    # The bucket every object lives in. Created on the first upload if it is
    # not already there, exactly as the local backend makes its root directory.
    s3_bucket: str = "attachments"
    # Credentials. THE SECRET IS A SECRET: never log it, never echo it, never
    # fold it into an error message. Both may be left unset on AWS, where
    # botocore's own chain (an instance role, the shared config file) supplies
    # them; anywhere else - MinIO above all - they are required, and
    # build_storage_backend refuses to start without them.
    s3_access_key_id: str | None = None
    s3_secret_access_key: str | None = None
    # The region the client signs for, and where a created bucket is placed.
    # MinIO ignores it but still requires the signature to name one.
    s3_region: str = "us-east-1"
    # Largest single upload the attachments endpoint accepts, in bytes. A
    # too-large file is refused up front (413) rather than buffered and parsed:
    # the whole point of the cap is to bound what one request can pull into
    # memory. 25 MB comfortably covers documents and code while staying well
    # under what the small-tier embedder can chew through in a reasonable time.
    attachment_max_bytes: int = 25_000_000
    # How long a chat turn waits for an attachment it names to finish parsing.
    # An upload returns the moment its bytes are stored, so asking about a file
    # immediately after attaching it is the NORMAL case, not a race: without a
    # wait the answer would silently ignore the file. Capped because a chat turn
    # cannot hang - past this the turn proceeds and says plainly that the file
    # was not ready.
    attachment_wait_seconds: float = 30.0
    # How often that wait re-checks the row (the worker updates it in another
    # process, so this is a poll, not a notification).
    attachment_poll_seconds: float = 0.5
    # How many passages a turn draws from EACH attachment it explicitly names.
    # Separate from rag_top_k: a named file is not competing with the knowledge
    # base for a slot, it is the thing being asked about. Generous enough that a
    # document which fits the window arrives whole rather than as a few excerpts.
    rag_attachment_top_k: int = 12
    # Token ceiling for the files a turn names, shared between them - the
    # equivalent of rag_token_budget, and deliberately far larger. That 1200-token
    # cap exists to stop a knowledge base flooding the window with material nobody
    # asked for; a file someone attached and asked about is the opposite case, and
    # holding it to the same ceiling means a 10KB document is answered from a
    # quarter of itself. Both budgets still shrink to what the model's window
    # actually has room for, so this cannot overflow a small context.
    rag_attachment_token_budget: int = 4000
    # How many images one turn may show the model. Small on purpose, and not by
    # analogy with hosted platforms - they allow 10 to 20 because they carry
    # 128k-plus windows. The binding fact here was measured on this project's
    # runtime: an image costs a FLOOR of about 1030 prompt tokens no matter how
    # small it is, so on the dev tier's 8192-token window every picture is an
    # irreducible eighth of the prompt. Two is a quarter of the window; more
    # would start crowding out the conversation the picture is about.
    vision_max_images: int = 2
    # The grid an image is resized to fit, in 32x32-pixel cells (one cell is one
    # visual token). 1024 cells is the runtime's own floor - below it the cost
    # stops falling while the detail keeps going - so this is the largest image
    # that is still charged the minimum. Measured: normalizing to this is free
    # up to about 1440p and costs only the finest print above it. Raising it
    # buys detail at a proportional token cost, which is what a per-image
    # "high detail" control would do.
    vision_image_budget_cells: int = 1024
    # Bytes one turn will pull out of storage for images, across all of them. A
    # bound on memory, not on quality: the cell budget already decides what the
    # model sees, and this stops a turn of several 25 MB uploads being decoded
    # at once just to be shrunk.
    vision_max_source_bytes: int = 24_000_000
    # How often the worker reclaims attachment garbage: the files of deleted
    # conversations, and blobs no row points at. Deleting a conversation already
    # queues its files for immediate removal, so this cadence only bounds how
    # long the leftovers of a failure survive - hourly is ample, and the sweep
    # walks the whole store, so more often would cost more than it saves.
    attachment_gc_interval_minutes: int = 60
    # How long an unreferenced blob must sit untouched before the sweep treats
    # it as garbage. An upload writes the bytes and commits the row moments
    # later, so a young unreferenced blob is far more likely to be an upload in
    # flight than a leak; waiting an hour makes deleting one impossible in
    # practice, at the cost of an hour of stale bytes on a failure that is
    # already rare.
    attachment_orphan_grace_minutes: int = 60
    # How long an attachment may sit in `pending` or `parsing` with nothing moving
    # before the worker declares it dead and marks it failed. Nothing else ever
    # closes such a row: a worker that dies mid-parse - or that was never up to
    # claim the job at all - leaves an upload that says "processing" forever, and
    # a client polling it spins forever with it.
    #
    # Generous on purpose. One parse attempt may legitimately run for the whole
    # job timeout (15 minutes), and every attempt re-stamps `updated_at`, so 30
    # minutes is twice the longest quiet spell a healthy job can produce. Age is
    # only ever the first of two tests, though - the sweep also asks the queue
    # whether it still holds a job for the row, and never fails one it does.
    #
    # Must be positive: at 0 the cutoff is "now" and every upload in flight fails
    # on the next tick.
    attachment_stall_minutes: int = Field(default=30, gt=0)
    # How long a file the assistant's own tools produced is kept. Generated files
    # are the one kind nobody chose to keep - a tool renders a document as a side
    # effect of answering, and the next question renders another - so they are the
    # only tier that expires by default. Past the window the sweep removes the
    # bytes and the indexed text and marks the row `expired`, which is what lets
    # the transcript keep saying the file existed instead of losing the record.
    #
    # 0 turns it off (keep every generated file forever), and the ceiling is a
    # century: past roughly a million days a timedelta cannot be built at all, so
    # a fat-fingered digit would raise OverflowError inside the hourly sweep and
    # take the whole garbage-collection pass down with it, every hour, for a
    # value that plainly meant "never".
    generated_file_retention_days: int = Field(default=30, ge=0, le=_MAX_RETENTION_DAYS)
    # How long a file a PERSON sent - an upload or a paste - is kept. Unset, which
    # is the default, means retention is off for them: they live as long as the
    # conversation they belong to and are reclaimed with it. That is deliberately
    # the opposite default from generated files, because somebody chose to send
    # these, and a platform that quietly deletes what a user uploaded is worse
    # than one that keeps it. A deployment with a real data-retention policy sets
    # a number of days here; 0 also reads as off, so the switch has no cutoff at
    # which every stored upload disappears at once. Same century ceiling as above.
    attachment_retention_days: int | None = Field(default=None, ge=0, le=_MAX_RETENTION_DAYS)
    # OCR (reading the text off a SCANNED page, which carries a picture of text and
    # no text layer). On by default so a scanned PDF is answerable out of the box,
    # but it needs the Tesseract system binary: where that is absent the parser
    # degrades to what it did before OCR existed - a scanned page yields nothing -
    # rather than failing the upload. Turning this off skips the check entirely.
    ocr_enabled: bool = True
    # Tesseract language(s) to recognise, e.g. "eng" or "eng+deu". The matching
    # language data must be installed in the image (the Dockerfile installs eng).
    ocr_language: str = "eng"
    # Resolution a page is rendered at before OCR. Higher reads small type better
    # but is slower and uses more memory; 200-300 is the usual band.
    ocr_dpi: int = 200
    # A page with fewer than this many non-whitespace characters in its text layer
    # is treated as a scan and sent to OCR. A born-digital page sails past it; a
    # scan sits at zero, and the small floor also ignores a page bearing only a
    # watermark or a page number.
    ocr_min_chars: int = 16
    # The most pages of one document to OCR. OCR is seconds per page, so a long
    # scan is bounded rather than left to run for minutes; pages past the cap keep
    # whatever sparse text they had.
    ocr_max_pages: int = 20
    # How long a PDF gets to parse WHEN OCR is enabled - larger than the text-only
    # ceiling because rendering and recognising pages is genuinely slow. It only
    # bites a document that actually OCRs many pages; a born-digital file finishes
    # long before it.
    ocr_timeout_seconds: float = 300.0
    # Master switch for retrieval-augmented chat turns. Ingest/query endpoints
    # keep working either way; this only gates the automatic chat wiring.
    rag_enabled: bool = True
    # How many retrieved chunks reach the prompt (after optional reranking).
    rag_top_k: int = 5
    # How many candidates dense search fetches when a reranker will re-score.
    rag_fetch_k: int = 20
    # Minimum cosine similarity for a chunk to count as relevant at all.
    # Calibrated against the qwen3-embedding family WITH the query-side
    # instruct prefix its model card prescribes applied,
    # measured live 2026-07-09 across stored records AND web chunks: strongly
    # related passages score 0.40-0.60, while unrelated text (small talk vs
    # stored content) peaks around 0.31 - so 0.35 cuts the noise with margin
    # on both sides. Without the prefix the scale is compressed and inverted
    # for short queries (unrelated boilerplate reached 0.57), so this floor
    # and the prefix must move together. Re-tune if the embedder or its
    # prefix changes.
    rag_min_score: float = 0.35
    # Relative-relevance gap: once a query has a clear best match, drop hits
    # scoring more than this far below it. Weak topical neighbors clear the
    # absolute floor but are noise the answer should not cite; this adaptive
    # gap trims them per query while leaving a genuine cluster of near-top
    # hits intact (prefixed relevant clusters span ~0.08). Set to 0 to disable.
    rag_rel_margin: float = 0.10
    # Token ceiling for retrieved context per turn (1 token ~ 4 chars).
    rag_token_budget: int = 1200
    # Chunking targets, in approximate tokens.
    rag_chunk_tokens: int = 350
    rag_chunk_overlap_tokens: int = 50
    # Texts per embedding call; bounds request size and GPU/CPU bursts.
    rag_embed_batch: int = 32
    # Lifetime of session-scoped (ephemeral) vectors before the sweep removes them.
    rag_session_ttl_minutes: int = 120
    # How often the background worker sweeps expired session vectors.
    rag_sweep_interval_minutes: int = 10
    # Pause between ingest embedding batches while a chat stream is active, so
    # ingestion yields the GPU to token generation.
    rag_ingest_pause_seconds: float = 0.5
    # SearXNG JSON API base URL for web search. None disables web search (the
    # feature is off when the sidecar is unreachable), the same opt-in shape as
    # reranker_url; infra/.env.example ships it active because the sidecar runs
    # by default (nothing else gates it - web search defaults to on).
    searxng_url: str | None = None
    # Master switch for the automatic web-search chat wiring. The auto/on policy
    # only ever fires when this is True AND searxng_url is set.
    web_search_enabled: bool = True
    # How many search-result snippets are ingested into the session scope per
    # web-search turn; retrieval then floors and ranks them like any other chunk.
    web_search_top_k: int = 5
    # Read timeout for the SearXNG query (seconds). It MUST stay at or above
    # SearXNG's own outgoing.request_timeout (10s in infra/searxng/settings.yml)
    # plus overhead: SearXNG holds the response open while it waits on its
    # slowest engine, so a lower client ceiling cuts off slow-but-successful
    # engines - the "first (cold) search fails, second works" bug. 12s clears
    # the 10s engine wait with headroom while still bounding chat latency.
    web_search_timeout_seconds: float = 12.0
    # Fetch the top result PAGES and ground on their extracted readable text
    # rather than the (often thin, description-only) search snippet - retrieval
    # then surfaces the relevant page chunks. Best-effort per result: any
    # fetch/extract failure falls back to that result's snippet. Set False for a
    # fully snippet-only, lower-latency path on small systems.
    web_fetch_pages: bool = True
    # Per-page fetch read timeout (seconds); bounds how long a slow page can
    # hold up a web-search turn (fetches run concurrently).
    web_fetch_timeout_seconds: float = 5.0
    # Hard cap on bytes downloaded per page (defends against huge responses).
    web_fetch_max_bytes: int = 2_000_000
    # Cap on extracted characters kept per page before chunking (bounds the
    # ingest/token cost of one page - embedding fetched chunks inside the
    # stream is the dominant web-turn latency on CPU embedders).
    web_fetch_max_chars: int = 6000
    # Full page text is fetched for only this many TOP results per turn; the
    # rest ground on their search snippet. Fewer pages = fewer chunks to embed
    # before the first token.
    web_fetch_max_pages: int = 3
    # Max pages fetched concurrently per web-search turn.
    web_fetch_concurrency: int = 5
    # A result URL already ingested into this conversation within this window
    # is reused as-is: no re-fetch, no re-embed. Repeat and follow-up searches
    # then cost one SearXNG call instead of the full fetch+embed pipeline.
    # 0 disables the reuse window.
    web_result_cache_minutes: int = 15
    # How much the model may drive a turn (Phase 5 tool calling). Clamps DOWN
    # from each model's resolved ceiling, never up:
    # "agentic" lets the model decide when to search the web or the stored
    # records; "assisted" keeps today's automatic pre-fetch AND offers tools on
    # top (a model that never calls one behaves byte-for-byte as today);
    # "deterministic" is the kill switch (no tools, the pre-tool pipeline
    # verbatim); "auto" defers to the model's own ceiling. Shipped default
    # "agentic" - dev is a testing sandbox, a deployment clamps down as it
    # likes. Consumed from the tool-loop slice; inert until then.
    agency_mode: Literal["auto", "agentic", "assisted", "deterministic"] = "agentic"
    # Remove the remaining grace from the EARNED half of the agency resolution.
    # By default a reviewed declaration already counts for at most "assisted" -
    # the no-regression mode, where tools are merely offered alongside the
    # pre-fetch that already ran - while "agentic" must be EARNED from a DEEP
    # tool-check verdict bound to the exact installed weights, in either
    # regime. When true, even that floor goes: a model may drive only on what
    # model_probes proves for these weights, so never probed, re-pulled since
    # probing, or weights the runtime cannot digest-identify all resolve to
    # deterministic. Default false, which is what lets a fresh clone answer
    # grounded turns on day one without running a check first.
    agency_require_verification: bool = False
    # Loop bounds for one agentic turn: the most model<->tool round trips, and
    # the most individual tool calls, before the turn is forced to answer with
    # no tools offered. Kept small: multi-turn tool accuracy falls off sharply
    # past a few passes on a small local model, so a runaway loop is a bug, not
    # a feature.
    tool_max_iterations: int = 4
    tool_max_calls_per_turn: int = 8
    # Wall-clock budget for a SINGLE tool call. 75s since tool results became
    # cited sources: a healthy web_search now composes SearXNG's 12s query
    # timeout, the page fetches, chunk EMBEDDING and ingestion (a cold CPU
    # embed load alone is 6-8s), and the read-back retrieval - measured live,
    # the first post-boot web call ran past the old 45s and was marked failed
    # while perfectly healthy. A tight deadline here strips a working tool for
    # the rest of the turn.
    tool_timeout_seconds: float = 75.0
    # Tool-result window sizing (approx tokens, 1 token ~ 4 chars). A tool
    # result is RESERVED room in the window, not merely capped, because the RAG
    # budget deliberately fills num_ctx: appending a result on top would
    # otherwise evict the prompt HEAD - the system prompt, carrying the tool
    # guidance and the injection defense. tool_result_max_tokens caps ONE
    # result; tool_result_total_max_tokens ceilings the turn-wide budget the
    # loop computes from the actually-assembled prompt; tool_result_min_tokens
    # is the floor below which tools are NOT OFFERED at all (offering a tool
    # whose result cannot fit is a lie). The dynamic budget arithmetic lands
    # with the tool loop; these are its clamps and are inert until then.
    tool_result_max_tokens: int = 1000
    tool_result_total_max_tokens: int = 2000
    tool_result_min_tokens: int = 256
    # Files a tool HANDS BACK (a generated document), stored as attachments and
    # announced to the client. Deliberately below attachment_max_bytes: a file
    # nobody chose to upload should not be able to occupy more of the store than
    # one somebody did, and the whole artifact is resident in memory (decoded
    # from the tool's response) while it is written. An over-large or over-count
    # artifact is DROPPED with a one-line notice in the result the model reads
    # and on the tool card the user sees - never silently.
    #
    # The value is pinned to what the TRANSPORT can actually deliver, not to a
    # round number. An MCP response body is capped at _MAX_RESPONSE_BYTES = 2 MB
    # (app/providers/mcp/source.py) and an artifact travels as base64, which
    # costs 4 bytes per 3: 2_000_000 * 3/4 = 1_500_000 bytes of decoded file per
    # RESPONSE, minus the JSON envelope and whatever text rides beside it. 1.4 MB
    # sits just under that, so this cap is the one that fires - dropping the file
    # with a notice - instead of the transport's, which fails the whole CALL.
    #
    # So the two move TOGETHER or not at all. Raising this alone buys nothing (the
    # transport refuses the body first); raising the transport alone re-opens an
    # OOM surface on the single process serving every concurrent stream. A
    # deployment that needs larger documents raises both, deliberately, after
    # re-doing the arithmetic recorded beside _MAX_RESPONSE_BYTES.
    tool_artifact_max_bytes: int = 1_400_000
    # The per-TURN count, not per call: several calls in one turn share it, so a
    # loop that calls a generating tool repeatedly cannot mint files without end.
    tool_artifact_max_per_turn: int = 10
    # How long a suspended side-effecting-tool confirmation (Phase 5) stays
    # resumable. A turn that pauses for the user's OK checkpoints itself on the
    # conversation; past this many seconds the resume is refused (the world may
    # have moved on, and a pending action should not lurk indefinitely) and the
    # user starts the turn over. Long enough to read the action and decide,
    # short enough that a forgotten prompt does not linger for hours.
    tool_confirmation_ttl_seconds: float = 900.0
    # Base URL of the MCP chassis sidecar's wrapper server. UNSET is a
    # first-class state, not a misconfiguration: a deployment without the
    # sidecar simply offers no MCP tools, which is a normal deployment. The URL
    # carries the wrapper's mount path (the bundled example wrapper answers at
    # /example/mcp), because one chassis container mounts several wrappers at
    # distinct paths. This is the BOOTSTRAP SEED only: the DB-backed per-tenant
    # server registry (Phase 6b) becomes authoritative, and this recedes to the
    # value that seeds it.
    mcp_chassis_url: str | None = None
    # Static credential for that server, sent as "Authorization: Bearer <key>".
    # Unset is normal on a private network - the chassis accepts unauthenticated
    # requests when ITS OWN key is unset. Workspace-level: one shared secret for
    # everyone in the tenant. Per-USER credentials (SecretStore, scoped per
    # principal) are Phase 6b.
    mcp_chassis_api_key: str | None = None
    # Inject a PER-USER credential into each tool CALL (Phase 5 slice 6b) instead
    # of the shared workspace key above. OFF by default, so the path is
    # byte-identical to the shared-key behaviour until a deployment opts in: when
    # on, the acting user's stored credential (SecretStore, USER scope) is fetched
    # per call and sent as the Authorization header, falling back to the static
    # workspace header when the user has stored none. LISTING stays on the
    # workspace header either way, so the listing cache never mixes callers.
    mcp_per_user_credentials: bool = False
    # The SecretStore name a user's chassis credential is stored under (USER
    # scope). Keyed by the server's name so "connect your account" maps to one
    # secret; the DB-backed registry (later in Phase 6b) names it per server.
    mcp_chassis_credential_name: str = "mcp-chassis"
    # Base URL of the document-generation sidecar's MCP server (it answers at
    # /docgen/mcp). Unset is a first-class state exactly as the chassis URL is:
    # a deployment without the sidecar simply offers no document tools. Also a
    # BOOTSTRAP SEED only - the DB-backed registry is authoritative once the row
    # exists, so re-pointing or disabling it in the UI survives a restart.
    docgen_mcp_url: str | None = None
    # Static credential for that server, sent as "Authorization: Bearer <key>".
    # Unset is normal on a private network. WORKSPACE-scoped by nature: the
    # sidecar renders what it is handed and reaches no upstream system, so there
    # is no per-user account for anyone to connect.
    docgen_api_key: str | None = None
    # The registry name (and so the credential name) the seeded doc-gen row
    # takes. Distinct from the chassis's, so each server's stored credential and
    # tool toggles stay its own.
    docgen_credential_name: str = "doc-gen"
    # The origin the OAuth consent flow returns to - used to build both the
    # redirect_uri registered with a provider and the post-callback hop to the
    # Integrations page. Defaults to the frontend origin: its /api proxy
    # forwards the callback to the backend, so ONE origin serves the whole
    # round trip in dev, and Phase 8's Caddy makes the same shape real in prod.
    oauth_redirect_base_url: str = "http://localhost:3000"
    # How long a started consent stays completable before its state expires.
    oauth_authorization_ttl_seconds: float = 600.0
    # Wall-clock budget for ONE MCP tool call.
    mcp_timeout_seconds: float = 30.0
    # Budget for a tool LISTING - deliberately far tighter than a call's. A
    # listing is discovery that runs BEFORE the first token of every agentic
    # turn, so a slow or hanging sidecar must degrade fast to "this server
    # offers no tools this turn" rather than stalling the turn. A CALL is work
    # the user asked for and can afford to wait on; a LISTING is latency nobody
    # requested.
    mcp_list_timeout_seconds: float = 5.0
    # How long a good listing is reused. Tool sets change on deploys, not per
    # turn, so re-listing every turn buys nothing and costs a round trip.
    mcp_list_cache_seconds: float = 300.0
    # How long an EMPTY listing is reused. Shorter than a good one so a sidecar
    # that comes back is picked up quickly, but non-zero so a dead one is not
    # re-dialled on every single turn.
    mcp_list_cache_error_seconds: float = 30.0
    # Tool selection (rank tool descriptions against the turn, offer the top K)
    # is THRESHOLD-GATED: at or below this many candidates the whole set is
    # offered and selection does not run AT ALL - zero embed calls, zero IO.
    # The measured why: a plain agentic turn pays NO embedding today, and the
    # dev embedder is CPU-pinned where a cold load alone is 6-8s, so ranking
    # tools would put multiple seconds on a turn that currently pays nothing.
    # The shipped default discovers 7 tools - 2 local (web_search,
    # search_knowledge), 3 from the chassis example wrapper (search_records,
    # get_record, create_record) and 2 from the doc-gen sidecar (create_document,
    # render_chart) - of which the side-effecting create_record is dropped before
    # the set is offered to a client that cannot confirm it, leaving 6. Either
    # count sits below the threshold, so selection never runs out of the box; it
    # stays inert until the tool count is a real problem.
    tool_selection_threshold: int = 8
    # How many NON-CORE tools are offered once selection does run. web_search
    # and search_knowledge are always offered on top of this.
    tool_selection_top_k: int = 5

    # Base64-encoded AES master key for the SecretStore (credentials at rest).
    # UNSET is a first-class state: the integration surface answers 503 rather
    # than storing anything. It NEVER falls back to plaintext and is NEVER
    # auto-generated at boot - a minted key looks perfect in dev and silently
    # destroys every stored credential on the next redeploy, when a different
    # key is minted. Must decode to exactly 32 bytes (AES-256) - pinning the
    # length rejects a raw pass-phrase pasted where a base64 key belongs, which
    # would otherwise decode to a silently-wrong key. Consumed from the
    # integration slices; inert until then.
    secret_master_key: str | None = None

    # ---- Login (OIDC, Phase 8b) ----------------------------------------
    # The issuer URL of the OpenID Connect provider, e.g.
    # https://accounts.google.com. UNSET OR BLANK IS THE MASTER SWITCH: with no
    # issuer there is no login at all and the platform behaves exactly as it did
    # before this existed - every request acts as the seeded default principal.
    # Same shape as MODEL_PROVIDER=fake: one key decides whether a whole
    # subsystem participates, so a dev box and CI never need a provider.
    #
    # Provider-agnostic by construction: nothing below names Google. The issuer
    # is asked for its own metadata (RFC 8414 / OIDC discovery) and everything
    # else - authorize, token and JWKS endpoints - comes from what it answers.
    oidc_issuer: str | None = None
    # The client registered with that issuer, and its secret. Plain environment
    # values, deliberately NOT the SecretStore: the SecretStore needs the
    # database and a master key to be reachable before a login can happen, and a
    # credential that gates logging IN cannot depend on being logged in. Same
    # posture as SECRET_MASTER_KEY - never logged, never echoed, never minted.
    oidc_client_id: str | None = None
    oidc_client_secret: str | None = None
    # Bootstrap admins, comma separated. Two jobs in one key: the FIRST login
    # from a listed address is an admin (nothing else could grant the first one),
    # and the listing is a STANDING FLOOR - the role is re-applied at every
    # login, so a listed person cannot be demoted from inside the app. A refusal
    # to demote names this key as the place to change it.
    oidc_admin_emails: Annotated[list[str], NoDecode] = []
    # Admission control. A verified email is admitted when it is listed in
    # OIDC_ADMIN_EMAILS, listed here, or its domain is listed in
    # OIDC_ALLOWED_DOMAINS. Anything else is refused - DENY BY DEFAULT, because
    # the alternative on a public issuer like Google is that every account on
    # the internet may sign in and own a workspace.
    oidc_allowed_domains: Annotated[list[str], NoDecode] = []
    oidc_allowed_emails: Annotated[list[str], NoDecode] = []
    # How long a signed-in session lives without use. Sliding: every
    # authenticated request re-arms it, so an active person is never signed out
    # mid-work and an abandoned browser stops being a key after a week.
    session_ttl_seconds: int = Field(default=604_800, gt=0)

    # Request-rate limiting (Phase 8). ABUSE PROTECTION, not a quota: the
    # defaults sit far above what a person's session produces, so the limiter
    # is invisible in normal use and only bites a script hammering the port. A
    # limit tight enough for a real user to feel would be a bug here.
    rate_limit_enabled: bool = True
    # Requests per minute per client for ordinary /api/v1 traffic. A chat turn
    # is ONE request that then streams for many seconds, so a person cannot
    # approach this; a loop can.
    rate_limit_per_minute: int = 240
    # ... and for the handful of routes where a single call costs real
    # resources: a model download, an embedding pass over a document, an
    # upload's bytes, or an outbound call to a third party on the caller's
    # behalf. Deliberately an order of magnitude smaller.
    rate_limit_heavy_per_minute: int = 10
    # ... and for the one route a healthy client POLLS. The composer polls
    # every parsing attachment concurrently every 1.2s, so N files in flight
    # is N*50 requests a minute from one entirely legitimate user - five files
    # would trip the general budget by themselves. Raising the general limit to
    # cover that would raise it for everything, so the polled route gets its
    # own generous ceiling and the general one stays meaningful.
    rate_limit_light_per_minute: int = 1200
    # Peer addresses whose X-Forwarded-For header may be believed, comma
    # separated. Entries may be a plain address or a CIDR range - on a compose
    # network the address a proxy dials from is assigned at boot, so exact
    # match alone would be a setting nobody could actually fill in. EMPTY IS
    # THE SAFE DEFAULT and the one to keep unless a reverse proxy really does
    # sit in front of the app: trusting the header from an arbitrary peer lets
    # any caller mint a fresh identity per request and walk straight through
    # the limiter. List the proxy, and only the proxy.
    trusted_proxy_ips: Annotated[list[str], NoDecode] = []

    # Which SearchProvider implementation to use, the same shape as
    # MODEL_PROVIDER above: a Literal, so a typo fails the boot instead of
    # silently disabling web search. Until now the choice was implicit - web
    # search existed if SEARXNG_URL happened to be set - which made "is web
    # search off on purpose?" unanswerable from the configuration. "none"
    # states it; "searxng" with no URL still yields no provider, exactly as
    # before, so nothing existing changes behaviour.
    search_provider: Literal["searxng", "none"] = "searxng"

    @field_validator("trusted_proxy_ips", mode="before")
    @classmethod
    def _parse_trusted_proxies(cls, value: object) -> object:
        """Read the trusted-proxy list from a comma-separated string.

        An unset or empty value yields an empty list - "trust nobody" - which
        is also what compose forwards for a key nobody set.
        """
        if isinstance(value, str):
            return [entry.strip() for entry in value.split(",") if entry.strip()]
        return value

    @field_validator("cors_allow_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, value: object) -> object:
        """Accept a JSON array or a comma-separated string for CORS origins."""
        if isinstance(value, str):
            text = value.strip()
            if text.startswith("["):
                return json.loads(text)
            return [origin.strip() for origin in text.split(",") if origin.strip()]
        return value

    @field_validator("reranker_model", mode="before")
    @classmethod
    def _blank_reranker_model_as_default(cls, value: object) -> object:
        """Treat a blank RERANKER_MODEL as unset, not as "no model named".

        Compose forwards a key it lists but cannot resolve as an EMPTY STRING,
        and a blank model name is never a meaningful answer - a rerank call has
        to carry one. Reading it literally would silently disable reranking on
        a machine whose sidecar is running, which is the exact shape of the bug
        the three-gate collapse was meant to end.
        """
        if isinstance(value, str) and not value.strip():
            return DEFAULT_RERANKER_MODEL
        return value

    @field_validator("thinking_chat_model", mode="before")
    @classmethod
    def _empty_thinking_model_as_none(cls, value: object) -> object:
        """Treat THINKING_CHAT_MODEL="" as None (unset - defer to a declaration)."""
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("s3_endpoint_url", "s3_access_key_id", "s3_secret_access_key", mode="before")
    @classmethod
    def _empty_s3_value_as_none(cls, value: object) -> object:
        """Treat an empty S3 URL or credential as unset, not as a configured blank.

        Compose forwards an unset key as an empty string, and the template ships
        these commented out with no value, so ``""`` is the ORDINARY shape here.
        Read literally it is poison twice over: an empty access key is still an
        explicit credential, so botocore signs with it instead of falling back
        to the instance role this file promises; and an empty endpoint makes
        :func:`app.core.factories.build_storage_backend`'s credential check
        skip itself, so the deployment boots clean and fails every upload.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("s3_bucket", "s3_region", mode="before")
    @classmethod
    def _empty_s3_name_keeps_the_default(cls, value: object, info: ValidationInfo) -> object:
        """Treat an empty S3 bucket or region as unset, so the default applies.

        Same empty-string origin as above, but these fields have no "off"
        reading: a request signed for no region against no bucket is not a
        configuration anybody meant, and it fails on the first upload with an
        error that names neither key.
        """
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("generated_file_retention_days", "attachment_retention_days", mode="before")
    @classmethod
    def _empty_retention_keeps_the_default(cls, value: object, info: ValidationInfo) -> object:
        """Treat an empty retention window as unset, so the field's default applies.

        The template ships both commented out, so the shape an operator produces
        by uncommenting one and clearing it - ``ATTACHMENT_RETENTION_DAYS=`` - is
        an ordinary way to write "no policy". Read literally it is not a number
        at all and would fail the boot of both the API and the worker over a knob
        that was being turned OFF. Falling back to the declared default lands on
        exactly what each field means when nobody set it: 30 days for generated
        files, no policy at all for uploads. Turning the generated window off is
        a deliberate ``0``, not a blank.
        """
        if isinstance(value, str) and not value.strip():
            return cls.model_fields[info.field_name].default
        return value

    @field_validator("ollama_auth_headers", mode="before")
    @classmethod
    def _parse_auth_headers(cls, value: object) -> object:
        """Parse the header map from JSON, reading an unset value as no headers.

        Malformed JSON deliberately fails the boot: a deployment that meant to
        send a credential and mistyped it would otherwise talk to a public
        model server with no credential at all and never be told. The error is
        SHAPE-ONLY on purpose - the value is a credential, and quoting it back
        would land the secret in stderr and the container log.
        """
        if value is None:
            return {}
        if isinstance(value, str):
            text = value.strip()
            if not text:
                return {}
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                # from None: the chained JSONDecodeError carries a snippet of
                # the document - which here IS the credential.
                raise ValueError(
                    "OLLAMA_AUTH_HEADERS is not valid JSON; expected an object "
                    'like {"Authorization": "Bearer ..."} (value withheld - it '
                    "is a credential)."
                ) from None
        return value

    @field_validator("oidc_issuer", "oidc_client_id", "oidc_client_secret", mode="before")
    @classmethod
    def _empty_oidc_value_as_none(cls, value: object) -> object:
        """Treat a blank OIDC issuer, client id or secret as unset.

        Compose forwards a key nobody set as an empty string, and the template
        ships these commented out, so ``""`` is the ORDINARY shape for "no
        login". Read literally an empty issuer would be a configured issuer that
        happens to be blank, which turns the master switch ON and fails every
        request with a discovery error instead of running with no login at all.
        """
        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator(
        "oidc_admin_emails", "oidc_allowed_emails", "oidc_allowed_domains", mode="before"
    )
    @classmethod
    def _parse_admission_list(cls, value: object) -> object:
        """Read a comma-separated admission list, lowercased for comparison.

        Everything these lists are matched against comes from an identity
        provider, which is free to send ``Person@Example.COM``; comparing that
        against a hand-typed ``person@example.com`` verbatim would refuse a
        person who is plainly listed. Lowercasing once here means every
        comparison downstream is a plain ``in``.

        ``str.lower`` specifically, NOT ``str.casefold``: the members surface
        (lane C) decides who is a protected env-admin with ``lower`` on both
        sides, and the two must agree on exactly which addresses those are. They
        diverge on a handful of Unicode code points (German ``ss`` is the
        classic), and a split there is the worst kind of bug - an entry that
        floors nobody at login while the members API still refuses to demote it.
        The whole feature normalizes email with ``lower`` for that reason.

        A domain entry may be written with or without the leading ``@``, since
        both readings of "allow this company" are natural to type.
        """
        if isinstance(value, str):
            entries = [entry.strip().lower() for entry in value.split(",")]
            return [entry.lstrip("@") for entry in entries if entry.strip("@")]
        return value

    @model_validator(mode="after")
    def _oidc_configuration_is_complete(self) -> Settings:
        """Refuse to boot on a half-configured login.

        Every missing key here fails LATE and confusingly otherwise: no client
        secret means the token exchange is rejected by the provider on the first
        callback, and no admin email means a deployment where every admitted
        person is a member and nothing can ever be administered - a workspace
        locked out of itself, discovered after the first user signs in.

        Only the presence of the issuer switches this on, so a deployment with
        no login is never asked for any of it. The message names the keys rather
        than the fields, because the keys are what an operator edits.
        """
        if self.oidc_issuer is None:
            return self
        missing = [
            key
            for key, value in (
                ("OIDC_CLIENT_ID", self.oidc_client_id),
                ("OIDC_CLIENT_SECRET", self.oidc_client_secret),
                ("OIDC_ADMIN_EMAILS", self.oidc_admin_emails),
            )
            if not value
        ]
        if missing:
            raise ValueError(
                "OIDC_ISSUER is set, so login is enabled, but "
                f"{', '.join(missing)} {'is' if len(missing) == 1 else 'are'} missing. "
                "Set them in .env, or clear OIDC_ISSUER to run with no login."
            )
        # An issuer reached over plain http is the whole security model handed
        # to whoever is on the network path: its metadata document names the
        # signing keys, so substituting it forges accepted logins. Refused at
        # BOOT rather than at the first sign-in, because a deployment that
        # discovers this from a failed login has already been running.
        #
        # The rule itself is imported, never restated - it is the same one every
        # OAuth endpoint in this codebase passes, loopback-http exception
        # included, and a second copy of a security check is a second copy to
        # get wrong.
        from app.services.oauth.bundle import validate_endpoint

        try:
            validate_endpoint(self.oidc_issuer)
        except ValueError as exc:
            raise ValueError(f"OIDC_ISSUER is not a usable issuer URL: {exc}") from exc
        return self

    @property
    def oidc_enabled(self) -> bool:
        """Whether login is configured - the one question every caller asks.

        A property rather than a repeated ``settings.oidc_issuer is not None``,
        so "is there login here" has exactly one spelling and reads the same at
        the dependency, in the routes, and in the tests.
        """
        return self.oidc_issuer is not None


# Hosts that are this machine or this deployment's own compose network. The
# same list is implemented in scripts/remote_check.sh, which the make targets
# use to decide whether there is a local container to exec into - change one
# and change the other.
_LOCAL_OLLAMA_HOSTS = frozenset(
    {
        "ollama",
        "ollama-gpu",
        "ollama-cpu",
        "localhost",
        "127.0.0.1",
        "::1",
        "host.docker.internal",
    }
)


def ollama_url_host(url: str) -> str:
    """Return the hostname of a model-server URL, or "" when it names none.

    The hostname ALONE, because this value is reported over the API and a URL
    can carry a password in its userinfo - nothing but the host may survive the
    parse. ``urlparse`` already lowercases it and strips the brackets from an
    IPv6 literal, so ``[::1]`` reads back as ``::1``.

    Args:
        url: The configured model-server URL.

    Returns:
        The lowercased hostname, or "" when the value has no parseable host.
    """
    try:
        return urlparse(url).hostname or ""
    except ValueError:
        # A malformed URL is not worth an exception here: every caller is
        # describing configuration, never acting on it.
        return ""


def ollama_url_is_remote(url: str) -> bool:
    """Report whether a model-server URL points outside this deployment.

    The distinction matters because Ollama authenticates nothing itself. A
    server on this machine or on the compose network can only be reached by
    processes already inside the deployment; anything else is an endpoint on a
    network nobody here controls, and reaching it without a credential means
    the model server answers whoever finds it. Callers use this to WARN, never
    to refuse - a deliberately open server on a trusted network is a legitimate
    deployment.

    Only the hostname decides: scheme, port, path and userinfo are ignored, and
    a value with no parseable host counts as local so an odd setting degrades
    to the behaviour it had before this check existed.

    Args:
        url: The configured model-server URL.

    Returns:
        True when the URL names a host outside this deployment.
    """
    host = ollama_url_host(url)
    if not host:
        return False
    return host not in _LOCAL_OLLAMA_HOSTS


@lru_cache
def get_settings() -> Settings:
    """Return the cached application settings."""
    return Settings()
