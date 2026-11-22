/**
 * Browser-side helpers for the conversation REST endpoints.
 *
 * All calls go through the same-origin `/api/conversations` proxy routes, so
 * the backend base URL never reaches the client.
 */

import { ApiError, raise } from "@/lib/api-error";
import type {
  ContextUsageData,
  ConversationDetail,
  ConversationListResponse,
  ConversationSummary,
} from "@/lib/types";

// Re-exported so existing importers (`@/lib/conversations`) keep their path.
export { ApiError };

/**
 * Create a new, empty conversation and return its summary.
 *
 * A conversation is normally born with its first chat turn; this mints one up
 * front so a file can be attached before that turn exists (an upload must name
 * the conversation it belongs to). The new conversation has no title until its
 * first message.
 *
 * @returns The freshly created conversation summary (title is null).
 */
export async function createConversation(): Promise<ConversationSummary> {
  const resp = await fetch("/api/conversations", {
    method: "POST",
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return withPin((await resp.json()) as ConversationSummary);
}

/**
 * Force `pinned_at` to a concrete value on a summary from the wire.
 *
 * A backend one version behind this client omits the field entirely, and the
 * sidebar classifies with strict null checks - so an absent field MUST read
 * as unpinned here, at the one boundary summaries enter through. Without
 * this, the mixed-version window a rolling deploy produces (old backend, new
 * frontend) renders every conversation pinned.
 */
function withPin(c: ConversationSummary): ConversationSummary {
  return { ...c, pinned_at: c.pinned_at ?? null };
}

/**
 * List the tenant's conversations, most recently updated first.
 *
 * @param limit - Page size (server default: 50).
 * @param offset - Rows to skip before the page starts.
 * @returns The requested page plus the tenant-wide total.
 */
export async function listConversations(
  limit = 50,
  offset = 0,
): Promise<ConversationListResponse> {
  const resp = await fetch(
    `/api/conversations?limit=${limit}&offset=${offset}`,
    { cache: "no-store" },
  );
  if (!resp.ok) await raise(resp);
  const page = (await resp.json()) as ConversationListResponse;
  return { ...page, items: page.items.map(withPin) };
}

/**
 * Fetch one conversation with its full transcript.
 *
 * @param id - Conversation UUID.
 * @returns Summary fields plus messages in ascending created_at order.
 * @throws ApiError - With `status === 404` when the conversation is gone.
 */
export async function getConversation(id: string): Promise<ConversationDetail> {
  const resp = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ConversationDetail;
}

/**
 * Ask how full a model's window already is on this conversation, pre-turn.
 *
 * The eager reading behind the context meter: what the next prompt opens with
 * (system prompt plus the capped history), over the window the named model
 * would be granted - character arithmetic on the server, no model call. The
 * response echoes the tag it measured, so a caller that switched picks while
 * the request was in flight can discard a stale answer.
 *
 * @param id - Conversation UUID.
 * @param model - The composer's pinned tag, or null for the resolved default.
 * @returns The estimated usage; `estimated` is NOT set here - the caller marks
 *   the reading when it stores it.
 * @throws ApiError - With `status === 404` when the conversation is gone.
 */
export async function getContextEstimate(
  id: string,
  model: string | null,
): Promise<ContextUsageData> {
  const query = model ? `?model=${encodeURIComponent(model)}` : "";
  const resp = await fetch(
    `/api/conversations/${encodeURIComponent(id)}/context${query}`,
    { cache: "no-store" },
  );
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ContextUsageData;
}

/**
 * The settings half of a conversation PATCH - a per-field merge, never a blob.
 *
 * Presence is what the server reads, and the three states are distinct: a key
 * left OFF the object keeps whatever is stored, an explicit `null` CLEARS the
 * pin (so the axis falls through to the workspace default), and a value
 * overwrites. `JSON.stringify` drops `undefined` and keeps `null`, so the two
 * survive the wire exactly as written here.
 */
export interface ConversationSettingsPatch {
  /** Chat model tag to pin, or null to unpin. */
  model?: string | null;
  /** Thinking mode to pin, or null to clear. */
  thinking?: "auto" | "on" | "off" | null;
  /** Web-search mode to pin, or null to clear. */
  web_search?: "auto" | "on" | "off" | null;
}

/** Editable conversation fields: the display title, the pin, and the settings. */
export interface ConversationPatch {
  /** New display title; the server accepts 1-200 characters. */
  title?: string;
  /**
   * Whether the conversation is pinned to the top of the sidebar. Intent only -
   * the server stamps `pinned_at` itself, so no client clock can decide where
   * the row sits in the pinned group. Omitting the key leaves the pin as it is.
   */
  pinned?: boolean;
  /** Per-field settings merge; see {@link ConversationSettingsPatch}. */
  settings?: ConversationSettingsPatch;
}

/**
 * Update a conversation's editable fields.
 *
 * The server treats an update that changes nothing as a no-op, so a rename
 * back to the current title cannot reorder the sidebar.
 *
 * @param id - Conversation UUID.
 * @param patch - Fields to change; anything omitted is left alone.
 * @returns The conversation summary as the server now holds it.
 */
export async function updateConversation(
  id: string,
  patch: ConversationPatch,
): Promise<ConversationSummary> {
  const resp = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!resp.ok) await raise(resp);
  return withPin((await resp.json()) as ConversationSummary);
}

/**
 * Persist the composer's per-conversation settings.
 *
 * A thin, well-named front door onto {@link updateConversation}: the settings
 * object merges PER FIELD server-side (under a row lock), so a control writes
 * only its own axis and two saves racing cannot clobber each other the way a
 * wholesale replace would.
 *
 * @param id - Conversation UUID.
 * @param settings - The axes to change; see {@link ConversationSettingsPatch}.
 * @returns The conversation summary as the server now holds it.
 */
export async function updateConversationSettings(
  id: string,
  settings: ConversationSettingsPatch,
): Promise<ConversationSummary> {
  return updateConversation(id, { settings });
}

/**
 * Pin or unpin a conversation.
 *
 * A thin, well-named front door onto {@link updateConversation}. Only the
 * intent travels: the server stamps the pin time, so the returned summary is
 * what the pinned group must be ordered by - never a locally guessed value.
 *
 * @param id - Conversation UUID.
 * @param pinned - True to pin, false to unpin.
 * @returns The conversation summary as the server now holds it.
 */
export async function setConversationPinned(
  id: string,
  pinned: boolean,
): Promise<ConversationSummary> {
  return updateConversation(id, { pinned });
}

/**
 * Delete a conversation (its messages cascade server-side).
 *
 * A 404 is treated as success: the conversation is already gone, which is the
 * end state the caller asked for.
 *
 * @param id - Conversation UUID.
 */
export async function deleteConversation(id: string): Promise<void> {
  const resp = await fetch(`/api/conversations/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
  if (!resp.ok && resp.status !== 404) await raise(resp);
}
