/**
 * Browser-side helper for the model listing endpoint.
 *
 * Goes through the same-origin `/api/models` proxy route, so the backend base
 * URL never reaches the client.
 */

import { raise } from "@/lib/api-error";
import type { ModelEntry, ModelsResponse } from "@/lib/types";

/**
 * List what the model server holds, and what its runtime reports about each tag.
 *
 * @returns The model listing as the backend now holds it.
 * @throws ApiError - 503 when the model server itself cannot be reached (the
 *   backend refuses to serve an inventory it cannot verify).
 */
export async function listModels(): Promise<ModelsResponse> {
  const resp = await fetch("/api/models", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ModelsResponse;
}

/**
 * The entries a chat turn could run on.
 *
 * Chat-role entries only: a tag that only embeds, or one whose report
 * establishes no role at all, is not something a conversation picks between,
 * and offering it would let a pin be stored that the turn's resolution chain
 * then silently refuses.
 *
 * Every entry is a tag the server already holds, so there is nothing here that
 * cannot be picked and nothing to sort inert rows below. The order is the
 * server's own listing order and is kept as it arrived - it is the only
 * ordering anything here has a reason to prefer.
 *
 * @param response - A model listing.
 * @returns The chat-role entries, in the order the server listed them.
 */
export function chatModels(response: ModelsResponse): ModelEntry[] {
  return response.entries.filter((entry) => entry.role === "chat");
}

/**
 * The entries that can create embeddings.
 *
 * Read off the role rather than off a capability flag, because the role already
 * carries the precedence the deployment's own embedder resolution applies: a tag
 * that can ALSO hold a conversation is listed as a chat model, so counting it
 * here would offer one model for two jobs and make a server with a single
 * embedder look like a choice nobody had made.
 *
 * @param response - A model listing.
 * @returns The embedding-role entries, in the order the server listed them.
 */
export function embeddingModels(response: ModelsResponse): ModelEntry[] {
  return response.entries.filter((entry) => entry.role === "embedding");
}

/** The listing entry for an exact tag, or null when nothing describes it. */
export function entryFor(models: ModelEntry[], tag: string | null): ModelEntry | null {
  if (tag === null) return null;
  return models.find((entry) => entry.model === tag) ?? null;
}

/**
 * Whether a turn on this entry could reason at all.
 *
 * Two ways to qualify, and they are genuinely different mechanisms: the model
 * reasons itself (`thinking` resolves to `always` or `hybrid`), or it answers
 * directly and hands thinking turns to a partner (`thinking_model`). Either one
 * makes the composer's Thinking control mean something; neither one present
 * means the control would change nothing at all.
 *
 * A missing `thinking` key is NOT read as "cannot". The backend writes that key
 * only when something has actually established how the weights treat the flag,
 * so absence is "nobody knows" - and disabling a control on an unknown would
 * take a working feature away on no evidence, which is the wrong direction for
 * a guess to fail in.
 */
export function entryCanReason(entry: ModelEntry): boolean {
  const thinking = entry.capabilities.thinking;
  const partner = entry.capabilities.thinking_model;
  if (typeof partner === "string" && partner !== "") return true;
  if (thinking === undefined) return true;
  return thinking === "always" || thinking === "hybrid";
}

/**
 * Whether this entry accepts images.
 *
 * Strict where {@link entryCanReason} is lenient, and deliberately: this one
 * drives a NOTICE rather than a control, so the failure direction is a sentence
 * nobody needed rather than a capability taken away. The honest-notice pattern
 * that governs image handling elsewhere makes the same trade - saying "this
 * model may not read the image" is cheap, and silently dropping an attachment
 * a user watched themselves attach is not.
 */
export function entryCanSee(entry: ModelEntry): boolean {
  return entry.capabilities.vision === true;
}
