/**
 * Browser-side helpers for the knowledge-base endpoints.
 *
 * All calls go through the same-origin `/api/knowledge/documents` proxy route.
 * The shapes mirror the backend's `app/schemas/knowledge.py` wire contract. Every
 * endpoint is admin-gated, so a member's browser gets a 403 `ApiError` from all
 * three.
 *
 * Adding a file is a SINGLE request that returns when the text is genuinely
 * searchable - there is no job to poll. That is the backend's shape (the
 * knowledge base has no table behind it, so there is nowhere to record a
 * status), and it decides how the page shows progress: the request being in
 * flight IS the progress, and every failure arrives as this call's rejection
 * rather than minutes later on a row nobody is watching.
 */

import { raise } from "@/lib/api-error";

/** One source stored in the knowledge base. */
export interface KnowledgeDocument {
  /** What the delete call addresses. Opaque - not a database id. */
  id: string;
  /** The label a citation shows; for an uploaded file, its filename. */
  title: string | null;
  /** `records` (ingest API), `upload`, `web`, or `mcp_records`. */
  source_type: string;
  url: string | null;
  /** How many RAG documents the source became (a page, a sheet, a part). */
  documents: number;
  /** Embedded passages, summed over every vector space holding a copy. */
  chunks: number;
  /** The embedding tags that actually wrote those passages. */
  embed_models: string[];
  /**
   * Whether a query issued today can find this source.
   *
   * False means the passages are stored but no answer can cite them - what an
   * embedding-model change leaves behind. The row is still deletable.
   */
  reachable: boolean;
  ingested_at: string | null;
  content_hash: string | null;
  collection_id: string | null;
}

/** The knowledge base's contents, newest ingest first. */
export interface KnowledgeListing {
  items: KnowledgeDocument[];
  collection: string;
  /** The tag `reachable` was judged against; null when none is configured. */
  active_embedding_model: string | null;
  /** The store held more than this list returned. */
  truncated: boolean;
}

/** What one added file produced. */
export interface KnowledgeUpload {
  id: string;
  title: string;
  documents: number;
  chunks: number;
  collection: string;
}

/** What a delete actually removed. */
export interface KnowledgeDeletion {
  id: string;
  chunks_deleted: number;
}

/** List what the workspace knowledge base holds, newest ingest first. */
export async function listKnowledgeDocuments(): Promise<KnowledgeListing> {
  const resp = await fetch("/api/knowledge/documents", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as KnowledgeListing;
}

/**
 * Add one file to the knowledge base, resolving when its text is searchable.
 *
 * Rejects with an `ApiError` carrying the backend's own sentence: 415 for a file
 * nothing can read, 422 for one that was read and yielded no text, 413 for one
 * over the size limit. Those are terminal - the same file sent again fails the
 * same way - so the message is worth showing rather than retrying behind.
 *
 * @param file - The file to read.
 * @param signal - Aborts the upload; the caller keeps the controller.
 */
export async function addKnowledgeDocument(
  file: File,
  signal?: AbortSignal,
): Promise<KnowledgeUpload> {
  const form = new FormData();
  form.append("file", file, file.name);
  const resp = await fetch("/api/knowledge/documents", {
    method: "POST",
    body: form,
    cache: "no-store",
    signal,
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as KnowledgeUpload;
}

/**
 * Remove one source's passages from the knowledge base.
 *
 * The id rides as a query parameter, not a path segment: a source id is opaque
 * text chosen by whoever ingested it, and one containing a slash cannot survive
 * a path.
 */
export async function deleteKnowledgeDocument(
  id: string,
): Promise<KnowledgeDeletion> {
  const query = new URLSearchParams({ document_id: id });
  const resp = await fetch(`/api/knowledge/documents?${query.toString()}`, {
    method: "DELETE",
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as KnowledgeDeletion;
}
