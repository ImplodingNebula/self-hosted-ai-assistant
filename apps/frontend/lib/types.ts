/**
 * Local DTO and stream-event types mirroring the backend REST/SSE contract.
 *
 * Hand-written for now; a generated `packages/api-types` package replaces
 * these once the OpenAPI type-generation pipeline lands.
 */

/** One conversation row as returned by the list endpoint. */
export interface ConversationSummary {
  /** Conversation UUID (serialized as a string). */
  id: string;
  /** User-visible title; null until auto-titling or a rename sets one. */
  title: string | null;
  /**
   * ISO-8601 UTC time the conversation was pinned; null when it is not pinned.
   * The sidebar groups on its presence and orders that group by its value,
   * newest pin first. Only the server ever writes it.
   */
  pinned_at: string | null;
  /** ISO-8601 UTC creation timestamp. */
  created_at: string;
  /** ISO-8601 UTC last-activity timestamp (lists order by this, descending). */
  updated_at: string;
}

/** Paginated conversation list: `GET /api/v1/conversations`. */
export interface ConversationListResponse {
  /** One page of conversations, most recently updated first. */
  items: ConversationSummary[];
  /** Total rows for the tenant, independent of limit/offset. */
  total: number;
}

/** One retrieved source backing an assistant turn (SSE `citation` payload and persisted DTO). */
export interface CitationData {
  /** Citation UUID. */
  id: string;
  /** Human-readable source label (title when present, else the document id). */
  source: string;
  /** Excerpt of the retrieved chunk; surfaced in the citation popover (marker or chip). */
  snippet: string;
  /** Retrieval relevance score; null when the retriever reports none. */
  score: number | null;
  /** 1-based marker the answer may reference inline as [1]; null when unnumbered. */
  index: number | null;
  /** Source document title; null when the document has none. */
  title: string | null;
  /** Id of the source document; null when not tied to a stored document. */
  doc_id: string | null;
  /** Origin of the source (e.g. an upload or a fetched record); null when unknown. */
  source_type: string | null;
  /** Clickable source link for web results; null for local records. */
  url?: string | null;
}

/**
 * A file a turn carried, as recorded on the message it arrived with.
 *
 * A reference, not the live attachment row: the composer's tray empties the
 * moment a message is sent, so this is what tells a reloaded transcript which
 * files the question came with. It outlives the attachment's own deletion by
 * design - the turn really did carry that file.
 */
export interface MessageAttachment {
  /** Attachment UUID; may no longer name a live attachment. */
  id: string;
  /** Filename as uploaded. */
  filename: string;
  /** Declared media type. */
  content_type: string;
  /** Size in bytes. */
  size_bytes: number;
  /**
   * How the file arrived: `upload`/`paste` from a person, `generated` from a
   * tool the assistant ran. Optional because a snapshot written before the
   * field existed has none, and every one of those is a user turn - so an
   * absent value reads as an upload, exactly as the backend's default does.
   */
  origin?: string;
}

/**
 * A passage a question was replying to, as recorded on it.
 *
 * `text` is authoritative and stored verbatim: it is what reached the model and
 * what the chip renders, so a quote outlives the message it was taken from being
 * edited or deleted. `message_id` is provenance only - it may be null, or may
 * name a row that no longer exists, so nothing resolves it.
 */
export interface MessageQuote {
  /** Where the passage came from; null when the source had no server id yet. */
  message_id: string | null;
  /** The quoted passage itself. */
  text: string;
}

/**
 * A standing notice qualifying the answer it is stored with.
 *
 * Mirrors the SSE `status` event for the standing codes (`model_substituted`,
 * `web_search_unavailable`, `web_search_no_results`, `attachment_unavailable`,
 * `image_not_seen`, `tool_error`, `tool_limit_reached`, `tools_not_used`): each
 * explains something about THIS reply, so it is shown beside it and survives a
 * reload.
 */
export interface MessageNotice {
  /** Machine-readable status code. */
  code: string;
  /** Human-readable explanation, shown as-is. */
  message: string;
}

/** One tool call the model requested (SSE `tool_confirmation` payload member). */
export interface ToolCallData {
  /** Call id, correlating this call with its result across the turn. */
  id: string;
  /** The tool the model asked to run. */
  name: string;
  /** The call arguments the model produced; shape depends on the tool. */
  arguments: Record<string, unknown>;
}

/**
 * A side-effecting tool call (or batch) awaiting the user's approval.
 *
 * Mirrors the SSE `tool_confirmation` event: the turn SUSPENDS and the stream
 * ends (no `done` follows). The client shows the pending action(s), and the user
 * resumes the same turn by sending back a decision carrying `id` and naming
 * `conversation_id`. An approval runs the tool and streams the answer; a denial
 * answers without it.
 */
export interface ToolConfirmationData {
  /** The confirmation id, echoed back with the decision to resume the turn. */
  id: string;
  /** The conversation the suspended turn belongs to. */
  conversation_id: string;
  /** The side-effecting call(s) the user is being asked to authorize. */
  calls: ToolCallData[];
}

/** The outcome of a tool a turn ran (SSE `tool_result` payload). */
export interface ToolResultData {
  /** The call id this result answers, matching a `tool_call`. */
  id: string;
  /** The tool that produced it. */
  name: string;
  /** A short, display-safe summary of what the tool returned (not the full result). */
  result: string;
  /** True when the tool could not run; `result` then explains why. */
  is_error?: boolean;
  /** True when a side-effecting call was declined: it ran nothing, so `result` is empty. */
  denied?: boolean;
  /** How many sources, records or results the tool produced; null/absent when it cannot say. */
  count?: number | null;
}

/**
 * One tool the assistant ran this turn, as persisted on the message.
 *
 * The stored form of the live `tool_call` + `tool_result` pair: a reloaded
 * transcript rebuilds its activity trace from these, so the cards the live stream
 * showed reappear. `result` is null when the tool never returned one (the turn
 * was cut off mid-run).
 */
export interface ToolStepData {
  /** Call id, correlating the call with its result. */
  id: string;
  /** The tool that ran. */
  name: string;
  /** The arguments the model produced for the call. */
  arguments: Record<string, unknown>;
  /** Text the model streamed before the call ("let me look that up"); null when none. */
  narration?: string | null;
  /** Short display summary of what the tool returned; null when it never finished. */
  result?: string | null;
  /** True when the tool could not run. */
  is_error?: boolean;
  /** True for a side-effecting call the user approved at the confirmation prompt. */
  approved?: boolean;
  /** True for a side-effecting call the user declined; it ran nothing. */
  denied?: boolean;
  /** How many sources, records or results the tool produced; null when it could not say. */
  count?: number | null;
}

/**
 * What a reply took to write, summed over every round of its turn: the
 * tokens the model wrote (reasoning and tool-call rounds included) and the
 * runtime's own writing time for them. The same object closes a live
 * stream (`done.turn_usage`) and is stored with the reply, so the footer
 * reads one shape either way.
 */
export interface TurnUsageData {
  /** Tokens written this turn. */
  completion_tokens: number;
  /** Milliseconds the runtime spent writing them; null when it measured none. */
  generation_ms?: number | null;
}

/** One persisted chat message inside a conversation detail response. */
export interface ConversationMessage {
  /** Message UUID. */
  id: string;
  /** Author of the turn. */
  role: "user" | "assistant" | "system";
  /** Message text. */
  content: string;
  /** Model reasoning captured for assistant turns; null when absent. */
  thinking: string | null;
  /** Sources cited by an assistant turn; empty for user turns and old messages. */
  citations?: CitationData[];
  /** Files this turn carried; user turns only. */
  attachments?: MessageAttachment[];
  /** The passage this question was replying to; user turns only, null when none. */
  quote?: MessageQuote | null;
  /** Standing notices qualifying this reply; assistant turns only. */
  notices?: MessageNotice[];
  /** The tools this turn ran, in order - the activity trace; assistant turns only. */
  tool_steps?: ToolStepData[];
  /** True when the reply was cut short (client disconnect or provider error mid-stream). */
  interrupted?: boolean;
  /**
   * What the reply took to write; null on a cut-short reply, on one whose
   * provider counted nothing, and on every reply saved before it was kept.
   */
  usage?: TurnUsageData | null;
  /** Model that produced an assistant turn; null for user turns. */
  model: string | null;
  /** ISO-8601 UTC creation timestamp. */
  created_at: string;
}

/** Full conversation with transcript: `GET /api/v1/conversations/{id}`. */
export interface ConversationDetail {
  /** Conversation UUID. */
  id: string;
  /** User-visible title; null until auto-titling or a rename sets one. */
  title: string | null;
  /** Conversation-scoped settings JSON (model, thinking mode, ...). */
  settings: Record<string, unknown>;
  /** ISO-8601 UTC creation timestamp. */
  created_at: string;
  /** ISO-8601 UTC last-activity timestamp. */
  updated_at: string;
  /** Messages in ascending created_at order. */
  messages: ConversationMessage[];
  /**
   * Every file the conversation holds, sent or not. A client restores the ones
   * no message carries into its composer: they were uploaded and ingested but
   * never sent, and the server still has them.
   */
  attachments?: Attachment[];
}

/** Payload of the SSE `meta` event: the first frame, naming the stream's conversation. */
export interface MetaEventData {
  /** Conversation this turn belongs to (new or continued). */
  conversation_id: string;
  /**
   * The question this turn is answering - the row a new turn just persisted, or
   * the stored one a regenerate or an edit is answering again. Optional: an
   * older backend does not send it, so nothing may depend on it being there.
   */
  user_message_id?: string | null;
}

/**
 * How full the model's context window was on the prompt a turn sent - or, for
 * the eager pre-turn reading, how full it already is right now.
 *
 * Both numbers are the BACKEND's own accounting, not the model server's:
 * `prompt_tokens` is the character-based estimate the turn's budgets were
 * computed against, and `num_ctx` is the window the turn asked for. Their
 * ratio is an estimate, so anything showing it says so.
 */
export interface ContextUsageData {
  /** Estimated tokens the assembled prompt carried. */
  prompt_tokens: number;
  /** Context window the turn requested, in tokens. */
  num_ctx: number;
  /**
   * Tag the reading describes - the model the turn actually ran (substitutions
   * included), or the one the estimate was asked about. Absent from an older
   * backend's frames, so nothing may assume it is there.
   */
  model?: string | null;
  /**
   * CLIENT-side flag, never on the wire: true when this reading came from the
   * pre-turn estimate endpoint (system prompt + history, before the next
   * message and any retrieved context exist to count) rather than a completed
   * turn. The meter shifts its tense on it.
   */
  estimated?: boolean;
}

/**
 * Payload of the SSE `file` event: a file one of this turn's tools produced.
 *
 * Emitted once per generated file, after its row is committed, so the id always
 * names something the download route can serve. Metadata only - the bytes went
 * to object storage and never travel the stream.
 */
export interface FileEventPayload {
  /** Attachment UUID; the id the download route takes. */
  attachment_id: string;
  /** Filename the host settled on (a remote tool's suggestion is re-validated). */
  filename: string;
  /** Declared media type. */
  content_type: string;
  /** Size in bytes. */
  size_bytes: number;
  /** Always `generated` today; sent explicitly so a client keys on one field. */
  origin: string;
}

/** Payload of the SSE `done` event that closes a chat stream. */
export interface DoneEventData {
  /** Id of the persisted assistant message; "" on synthetic error frames. */
  message_id: string;
  /** Conversation the turn was persisted to; null on synthetic error frames. */
  conversation_id?: string | null;
  /** The provider's own report for the final round alone, when available. */
  usage?: Record<string, unknown> | null;
  /**
   * The whole turn's figures, as stored with the reply; null when the
   * provider counted nothing, absent on an older backend.
   */
  turn_usage?: TurnUsageData | null;
  /**
   * Final persisted answer, set only when it differs from the streamed text
   * (inline citation markers renumbered to match the emitted citations);
   * clients replace their streamed copy with it.
   */
  content?: string | null;
  /**
   * Window fill for this turn's prompt; null when the window is unknown, and
   * absent on an older backend - so nothing may assume it is there.
   */
  context_usage?: ContextUsageData | null;
}

/**
 * Ingestion status of an attachment.
 *
 * `pending` and `parsing` are transient (the client polls through them);
 * `ready` (retrievable), `failed` and `expired` are terminal. `expired` means
 * retention removed the bytes while the row stayed: the file is named by the
 * turn that carried it but can no longer be downloaded.
 */
export type AttachmentStatus =
  | "pending"
  | "parsing"
  | "ready"
  | "failed"
  | "expired";

/** One attachment as returned by the upload and status endpoints (`AttachmentOut`). */
export interface Attachment {
  /** Attachment UUID (serialized as a string). */
  id: string;
  /** Conversation this file belongs to; null once its conversation is deleted. */
  conversation_id: string | null;
  /** Client-supplied filename (a label; never a path). */
  filename: string;
  /** Declared media type, or the octet-stream default when unknown. */
  content_type: string;
  /** Stored size in bytes. */
  size_bytes: number;
  /** Ingestion status the client polls after upload. */
  status: AttachmentStatus;
  /** How the file arrived: a picked/dropped upload or pasted text. */
  origin: string;
  /** Human-readable failure reason; null unless `status` is `failed`. */
  error: string | null;
  /** ISO-8601 UTC creation timestamp. */
  created_at: string;
  /** ISO-8601 UTC last-update timestamp. */
  updated_at: string;
}

/** What a model can do, as the serving runtime reports it. */
export interface ModelCapabilities {
  /** Reads images handed to it directly. */
  vision: boolean;
  /** Its chat template can take a tool list. */
  tools: boolean;
  /** Serves embeddings rather than chat. */
  embedding: boolean;
  /** The runtime has think plumbing for it (NOT "these weights reason"). */
  thinking: boolean;
  /** Context window the runtime reports; null when it reports none. */
  context_length: number | null;
}

/** One model the provider currently has installed (`ModelInfo`). */
export interface ModelInfo {
  /** Exact tag as the provider knows it. */
  name: string;
  /** On-disk size in bytes; null when the runtime reports none. */
  size?: number | null;
  /** Digest of the weights the (mutable) tag points at; null when unexposed. */
  digest?: string | null;
  /** Capability flags the runtime reports. */
  capabilities: ModelCapabilities;
}

/**
 * One model the server holds, described by what the runtime reports about it.
 *
 * Every entry is a tag the provider actually has: there is no catalog of models
 * this deployment could run, only an inventory of the ones it does.
 *
 * `role` is what the report establishes the tag can serve - `chat` when it can
 * complete, `embedding` when it can only embed, empty when it establishes
 * neither - and the composer's selector offers the chat ones. It REPEATS across
 * the list. `capabilities` carries the remaining reported keys verbatim
 * (`tools`, `vision`, and `thinking`/`thinking_model` once something has
 * established them), so a key this client does not know about rides along
 * rather than being dropped.
 */
export interface ModelEntry {
  /** Role the report establishes; not unique across the listing, and may be empty. */
  role: string;
  /** Serving backend holding the tag (e.g. `ollama`). */
  provider: string;
  /** Exact model tag. */
  model: string;
  /** The remaining reported keys, passed through untouched. */
  capabilities: Record<string, unknown>;
}

/**
 * Where the model server is, and whether it is reached with credentials.
 *
 * Read from configuration alone, so it is as true under a stub provider as a
 * live one. It is what lets a console say out loud that a model server outside
 * this deployment is being reached with nothing proving who is asking - the
 * server authenticates nothing itself, so anyone who can route to it can drive
 * it, and that is a fact about the deployment rather than about any one model.
 *
 * Optional, because a backend that predates the block sends no `server` key at
 * all. Absent must read as "nothing to say", never as "remote and unprotected".
 */
export interface ModelServerStatus {
  /** The model server is somewhere other than this deployment's own host. */
  remote: boolean;
  /** Requests to it carry credentials. */
  auth_configured: boolean;
  /** Host as configured, for naming it in a notice; empty when unknown. */
  host: string;
}

/** Model listing: `GET /api/v1/models`. */
export interface ModelsResponse {
  /** Every model the provider currently has installed. */
  models: ModelInfo[];
  /** The same tags again, each described by what the runtime reports about it. */
  entries: ModelEntry[];
  /** Where the models are served from, and how that host is reached. */
  server?: ModelServerStatus;
}

/** RFC 7807 problem body the backend returns on errors. */
export interface ProblemDetail {
  /** Problem type URI; "about:blank" for plain status-code semantics. */
  type: string;
  /** Short human-readable summary of the problem class. */
  title: string;
  /** HTTP status code duplicated in the body. */
  status: number;
  /** Human-readable explanation specific to this occurrence. */
  detail?: string;
  /** Machine-readable error code (not_found, bad_request, ...). */
  code?: string;
  /** Correlation id for tracing the failure in backend logs. */
  request_id?: string;
}
