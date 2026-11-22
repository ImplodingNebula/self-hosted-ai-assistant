/**
 * Browser-side helpers for the workspace-defaults endpoint.
 *
 * Goes through the same-origin `/api/workspace/defaults` proxy route, so the
 * backend base URL never reaches the client. The shapes mirror the backend's
 * `app/schemas/workspace.py` and `app/schemas/settings.py` wire contract by
 * hand, which is this app's convention (nothing imports the generated
 * `packages/api-types`).
 */

import { raise } from "@/lib/api-error";

/** The request-axis vocabulary shared with the composer and `ChatRequest`. */
export type SettingMode = "auto" | "on" | "off";

/** Whether this deployment has a reranker for the workspace toggle to govern. */
export interface RerankerStatus {
  /** A reranker would run if the workspace permitted it. */
  configured: boolean;
  /** The sidecar's model tag, when this deployment names one. */
  model: string | null;
  /** One sentence for the page to render as visible text. */
  detail: string;
}

/**
 * The workspace's stored defaults, plus what this deployment can honour.
 *
 * Every stored field is nullable and null means UNSET: the next level of the
 * resolution chain decides that axis. That is deliberately distinct from a
 * value - a null thinking default lets the per-turn heuristic run, where
 * `"auto"` stored pins the heuristic explicitly.
 */
export interface WorkspaceDefaults {
  /** The default chat tag every turn falls back to, or null. */
  model: string | null;
  /**
   * The tag retrieval embeds with, or null to leave the answer to the deployment.
   *
   * Not a preference the way the chat default is: the embedding model DEFINES
   * the vector space, so a change points every query at a fresh, empty one and
   * leaves the passages already stored where they are. Null resolves against
   * what wrote those passages, then against the one installed embedder when the
   * server holds exactly one - so null is "nobody chose", never "no embedder".
   */
  embedding_model: string | null;
  /** The default thinking mode, or null. */
  thinking: SettingMode | null;
  /** The default web-search mode, or null. */
  web_search: SettingMode | null;
  /**
   * The persona every turn here opens with, or null for the deployment's own.
   *
   * Never an empty string: blank is normalized to null on the way in, because
   * the environment's "" kill switch - which disables the system prompt
   * outright, tool guidance included - has no workspace twin.
   */
  system_prompt: string | null;
  /** The token ceiling every turn's window clamps to, or null. */
  context_ceiling: number | null;
  /** False forces dense-only retrieval; true or null let a wired sidecar run. */
  reranker: boolean | null;
  /** What this deployment can actually do about the reranker toggle. */
  reranker_status: RerankerStatus;
  /** The smallest ceiling the API accepts. */
  min_context_ceiling: number;
  /** The largest. */
  max_context_ceiling: number;
  /** The longest persona the API accepts, in characters. */
  max_system_prompt_chars: number;
}

/**
 * A per-field change to the defaults: absent leaves alone, null clears.
 *
 * The tri-state is the whole point of the endpoint - a body that named every
 * field would overwrite an axis another tab had just changed - so a caller
 * builds this from what actually moved, never from the whole form.
 */
export interface WorkspaceDefaultsPatch {
  model?: string | null;
  embedding_model?: string | null;
  thinking?: SettingMode | null;
  web_search?: SettingMode | null;
  system_prompt?: string | null;
  context_ceiling?: number | null;
  reranker?: boolean | null;
}

/**
 * Read the workspace defaults.
 *
 * @returns The stored defaults and this deployment's reranker status.
 * @throws ApiError - the backend refused or could not be reached.
 */
export async function getWorkspaceDefaults(): Promise<WorkspaceDefaults> {
  const resp = await fetch("/api/workspace/defaults", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as WorkspaceDefaults;
}

/**
 * Change some of the workspace defaults.
 *
 * @param patch - Only the fields that moved; an explicit null clears one.
 * @returns The defaults as they now stand.
 * @throws ApiError - 422 when a value is refused, 403 without the admin role.
 */
export async function updateWorkspaceDefaults(
  patch: WorkspaceDefaultsPatch,
): Promise<WorkspaceDefaults> {
  const resp = await fetch("/api/workspace/defaults", {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(patch),
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as WorkspaceDefaults;
}

/**
 * The fields that differ between what was loaded and what the form holds.
 *
 * Building the patch from the DIFF rather than from the whole form is what
 * makes the backend's per-field merge worth having: a save that named every
 * axis would clobber a change made in another tab between the load and the
 * click, which is exactly the bug the endpoint's shape exists to prevent.
 *
 * @param loaded - The defaults as last read from the backend.
 * @param draft - The defaults the form currently holds.
 * @returns A patch carrying only the changed fields (empty when none moved).
 */
export function defaultsDiff(
  loaded: WorkspaceDefaults,
  draft: WorkspaceDefaultsPatch,
): WorkspaceDefaultsPatch {
  const patch: WorkspaceDefaultsPatch = {};
  if (draft.model !== loaded.model) patch.model = draft.model ?? null;
  if (draft.embedding_model !== loaded.embedding_model) {
    patch.embedding_model = draft.embedding_model ?? null;
  }
  if (draft.thinking !== loaded.thinking) patch.thinking = draft.thinking ?? null;
  if (draft.web_search !== loaded.web_search) patch.web_search = draft.web_search ?? null;
  if (draft.system_prompt !== loaded.system_prompt) {
    patch.system_prompt = draft.system_prompt ?? null;
  }
  if (draft.context_ceiling !== loaded.context_ceiling) {
    patch.context_ceiling = draft.context_ceiling ?? null;
  }
  if (draft.reranker !== loaded.reranker) patch.reranker = draft.reranker ?? null;
  return patch;
}
