/**
 * Browser-side helpers for the attachments endpoints, and the chip policy that
 * reads their results.
 *
 * All calls go through the same-origin `/api/attachments` proxy routes, so the
 * backend base URL never reaches the client. Upload sends a multipart form;
 * status is polled by id until it reaches a terminal state.
 *
 * The presentation half - what a file's lifecycle state means for the chip that
 * names it, and the sentences those chips say - lives here rather than in the
 * component that renders it, for one reason: this workspace has no component
 * runner, and those are the decisions worth checking directly. The markup stays
 * in `components/attachment-tray.tsx`.
 */

import { raise } from "@/lib/api-error";
import type { Attachment, AttachmentStatus } from "@/lib/types";

/**
 * Client-side size cap, mirroring the backend `attachment_max_bytes` default so
 * an over-large file is rejected before it is uploaded. The backend is the
 * authority (it answers 413); this is only fast local feedback, and the two can
 * drift if the backend default is overridden - a slightly stale value here just
 * means the server does the rejecting instead.
 */
export const ATTACHMENT_MAX_BYTES = 25_000_000;

/**
 * `accept` hint for the file picker. Only a hint: the backend decides what it
 * can read by sniffing the bytes, so this neither blocks nor guarantees a type.
 * Covers the formats this build parses today: plain text and source code, PDF, the
 * modern Office formats, their OpenDocument equivalents, RTF, EPUB, and images
 * (read for any text they contain via OCR).
 *
 * The LEGACY Office formats (.doc, .xls, .ppt) are a different container entirely
 * and are not readable, so they are left out rather than offered and then refused
 * after the upload.
 */
export const ATTACHMENT_ACCEPT = [
  ".txt,.md,.markdown,.csv,.tsv,.json,.yaml,.yml,.log",
  ".pdf,.docx,.docm,.xlsx,.xlsm,.pptx,.pptm",
  ".odt,.ott,.ods,.ots,.odp,.otp,.rtf,.epub",
  ".png,.jpg,.jpeg,.webp,.gif,.bmp,.tif,.tiff",
  "text/*,application/pdf,image/*",
  "application/rtf,application/epub+zip",
  "application/vnd.oasis.opendocument.text",
  "application/vnd.oasis.opendocument.spreadsheet",
  "application/vnd.oasis.opendocument.presentation",
].join(",");

/** How a file entered the pipeline; only the citation label differs. */
export type AttachmentOrigin = "upload" | "paste";

/**
 * Whether the content route will hand back this file's bytes.
 *
 * The client-side mirror of the backend's own servability gate, and the single
 * definition of it: a chip offers a download exactly when this says yes, so the
 * UI cannot grow an affordance the server refuses (or hide one it would serve).
 *
 * `failed` is downloadable, which reads as a contradiction until the two facts
 * are separated. What failed is INGESTION - the platform could not read the
 * document - and the bytes are untouched, so the file is still exactly what its
 * sender handed over and is still theirs to take back. That holds only for a
 * file a PERSON sent: one the assistant generated exists solely because the
 * platform finished making it, so there is no original behind a half-made one.
 *
 * `pending` and `parsing` are refused because the file has not settled yet, and
 * `expired` because its bytes are genuinely gone. An unknown status, and an
 * unknown ORIGIN, are refused too - both are allowlists, matching the shape of
 * the backend's own gate rather than merely its current membership, so a fourth
 * origin added one day cannot silently grow an affordance the server refuses.
 * A dead affordance is worse than a missing one.
 *
 * @param status - The live row's status, or undefined when no row remains.
 * @param origin - How the file arrived; absent reads as an upload, exactly as
 *   the backend's own default does.
 * @returns True when a download would succeed.
 */
export function servesBytes(
  status: AttachmentStatus | undefined,
  origin?: string,
): boolean {
  if (status === "ready") return true;
  if (status !== "failed") return false;
  return origin === undefined || origin === "upload" || origin === "paste";
}

/** Format a byte count for a chip label ("3.4 KB", "12 MB"). */
export function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  const kb = bytes / 1024;
  if (kb < 1024) return `${kb < 10 ? kb.toFixed(1) : Math.round(kb)} KB`;
  const mb = kb / 1024;
  return `${mb < 10 ? mb.toFixed(1) : Math.round(mb)} MB`;
}

/**
 * The fact an expired chip states, shared by every place a file chip appears so
 * the tombstones cannot drift into describing the same event differently. What
 * FOLLOWS from it differs by where the chip sits, so each adds its own sentence.
 */
export const EXPIRED_FILE_NOTE = "This file is no longer stored.";

/** The tray's consequence: a dead file cannot ride the next question. */
export const EXPIRED_DRAFT_NOTE = `${EXPIRED_FILE_NOTE} It will not be sent with the next question.`;

/** The reply's consequence: the assistant can simply make the file again. */
export const EXPIRED_GENERATED_NOTE = `${EXPIRED_FILE_NOTE} Asking for it again produces a fresh copy.`;

/** The sent question's consequence: the turn stands, the file cannot be fetched. */
export const EXPIRED_SENT_NOTE = `${EXPIRED_FILE_NOTE} It can no longer be downloaded.`;

/** A deleted file's chip: the record of the turn remains, the file does not. */
export const DELETED_FILE_NOTE = "This file has been deleted.";

/**
 * What a failed file's download actually offers - which is not what its red
 * says. The platform could not READ the document, so nothing in the answer came
 * from it; the bytes were never touched, so the original is exactly what was
 * sent and is still its sender's to take back.
 */
export const FAILED_FILE_NOTE =
  "This file could not be read, so the answer drew nothing from it. " +
  "The original is still stored, exactly as it was sent.";

/** The lifecycle values a stored row can hold, for narrowing a wider status. */
const LIFECYCLE_STATUSES = new Set<string>([
  "pending",
  "parsing",
  "ready",
  "failed",
  "expired",
]);

/** What a send should record about the files its turn just took. */
export interface CarriedFiles {
  /** Live state to record per file, id first - the shape a Map is built from. */
  states: [string, AttachmentStatus][];
  /** Ids still ingesting, which nothing watches once their chip leaves the tray. */
  unsettled: string[];
}

/**
 * Read the files a turn is taking into what the pane should remember about them.
 *
 * Two decisions, and both are easy to get subtly wrong in the middle of a send.
 *
 * The first is that a carried file's state is recorded AS IT IS, never as a
 * flat "ready". Send waits for the upload, not for the ingest behind it, so a
 * file can ride a turn while still `pending` or `parsing` - and a chip claiming
 * otherwise would offer a download the server answers 409 to.
 *
 * The second follows from it: those unsettled files need finding again. Their
 * tray chip is dropped the moment the turn goes, and the tray's poll is the only
 * thing that was watching them, so without a deliberate read at the end of the
 * turn a file that finished perfectly well would show no download arrow - and
 * one that failed would say nothing at all - until the conversation was
 * reopened.
 *
 * Entries with no server id, and the two CLIENT-only states, are dropped rather
 * than recorded: neither names a row the server would recognise. The turn's own
 * filter already excludes them; this does not depend on that.
 *
 * @param carried - The tray entries the turn is taking, in order.
 * @returns The states to record, and the subset that still has to settle.
 */
export function carriedFileStates(
  carried: readonly { serverId: string | null; status: string }[],
): CarriedFiles {
  const states: [string, AttachmentStatus][] = [];
  const unsettled: string[] = [];
  for (const entry of carried) {
    if (entry.serverId === null) continue;
    if (!LIFECYCLE_STATUSES.has(entry.status)) continue;
    const status = entry.status as AttachmentStatus;
    states.push([entry.serverId, status]);
    if (status === "pending" || status === "parsing") unsettled.push(entry.serverId);
  }
  return { states, unsettled };
}

/** What a sent chip says about its file, and how it carries itself saying it. */
export interface SentFileState {
  /** Short label in the chip's meta slot: a size, or the word for its state. */
  label: string;
  /** The sentence explaining a non-ordinary state; null when there is nothing to add. */
  note: string | null;
  /** Struck through: the file is not there any more (deleted, or aged out). */
  gone: boolean;
  /** Marked as a fault: ingestion failed, so the answer drew nothing from it. */
  problem: boolean;
}

/**
 * Read one sent chip's live status into what it should say.
 *
 * The POLICY behind those chips, kept out of the component that renders it for
 * two reasons. It is one readable table rather than four ternaries buried in
 * JSX; and it is the part worth checking directly - which state reads as a
 * fault and which as a quiet tombstone - in a workspace with no component
 * runner to check the markup around it.
 *
 * The distinction it exists to keep is between "gone" and "wrong". A deleted
 * file and one that aged out are both simply not there, and neither is anyone's
 * mistake, so both read as the same quiet struck-through record. A file whose
 * ingestion FAILED is the opposite case: it is still there, but the answer
 * above it drew nothing from it, which is the one thing a reader needs to be
 * told rather than left to infer.
 *
 * @param status - The live row's status, or undefined when no row remains.
 * @param sizeBytes - The size recorded on the turn, for the ordinary case.
 * @returns What the chip should show, and how.
 */
export function sentFileState(
  status: AttachmentStatus | undefined,
  sizeBytes: number,
): SentFileState {
  if (status === undefined) {
    return { label: "Removed", note: DELETED_FILE_NOTE, gone: true, problem: false };
  }
  if (status === "expired") {
    return { label: "Expired", note: EXPIRED_SENT_NOTE, gone: true, problem: false };
  }
  if (status === "failed") {
    return { label: "Failed", note: FAILED_FILE_NOTE, gone: false, problem: true };
  }
  return { label: formatBytes(sizeBytes), note: null, gone: false, problem: false };
}

/**
 * Upload one file and return its attachment row in the `pending` state.
 *
 * @param file - The file to upload.
 * @param conversationId - The conversation to attach it to (must already exist).
 * @param origin - Whether the file was picked/dropped ("upload") or pasted.
 * @param signal - Optional abort signal (cancels an in-flight upload).
 * @returns The created attachment; poll {@link getAttachment} until terminal.
 * @throws ApiError - On a rejected upload (413 too large, 415 unreadable, ...).
 */
export async function uploadAttachment(
  file: File,
  conversationId: string,
  origin: AttachmentOrigin = "upload",
  signal?: AbortSignal,
): Promise<Attachment> {
  const form = new FormData();
  form.append("file", file, file.name);
  form.append("conversation_id", conversationId);
  form.append("origin", origin);
  const resp = await fetch("/api/attachments", {
    method: "POST",
    body: form,
    cache: "no-store",
    signal,
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as Attachment;
}

/**
 * Fetch one attachment's current state, for polling its ingestion status.
 *
 * @param id - Attachment UUID.
 * @param signal - Optional abort signal.
 * @returns The attachment; its `status` drives the poll loop.
 * @throws ApiError - With `status === 404` when the attachment is gone.
 */
export async function getAttachment(
  id: string,
  signal?: AbortSignal,
): Promise<Attachment> {
  const resp = await fetch(`/api/attachments/${encodeURIComponent(id)}`, {
    cache: "no-store",
    signal,
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as Attachment;
}

/**
 * Save one attachment's bytes to disk, or say why they cannot be had.
 *
 * ONE request, and the app reads its outcome. That is the whole design, and it
 * is a correction of the obvious one: pointing a `<a download>` at the URL and
 * letting the browser navigate is fewer lines and hands the browser its own
 * progress and save dialog, but it also hands the browser whatever comes back.
 * A refusal - the file aged out, the client is over its rate limit, storage
 * blinked - is a small JSON document, and a navigation saves it to disk under
 * the file's own name. The reader gets `quarterly-report.pdf` full of
 * `{"title":"Too Many Requests"}` and the app never learns anything happened.
 * Fetching the bytes here makes every one of those visible instead, in the
 * backend's own words.
 *
 * The cost is that the file passes through this process's memory, and it is
 * bounded by the platform rather than by hope: an upload cannot exceed
 * `attachment_max_bytes` (25 MB) and a tool-generated file
 * `tool_artifact_max_bytes` (1.4 MB), both enforced server-side at the moment
 * the bytes are stored. One blob of at most 25 MB, released as soon as the
 * browser has taken it.
 *
 * The name comes from the row the chip already holds, not from parsing
 * `Content-Disposition`: an object URL has no filename of its own, and
 * re-implementing RFC 6266 parsing in the client to recover one the caller is
 * already holding would be a second, weaker copy of something the server got
 * right. The header still rides on the response for anyone who navigates to the
 * route directly.
 *
 * @param id - Attachment UUID.
 * @param filename - Name to save as; the caller's row is the authority.
 * @throws ApiError - 410 when the file expired or never became readable, 404
 *   when its row or its bytes are gone, 409 while it is still being processed,
 *   429 when the client is over its download budget.
 */
export async function downloadAttachment(
  id: string,
  filename: string,
): Promise<void> {
  const url = `/api/attachments/${encodeURIComponent(id)}/content`;
  const resp = await fetch(url, { cache: "no-store" });
  // Before the blob is read, so a refusal is never buffered and never saved.
  if (!resp.ok) await raise(resp);
  const objectUrl = URL.createObjectURL(await resp.blob());
  try {
    const link = document.createElement("a");
    link.href = objectUrl;
    link.download = filename;
    link.rel = "noopener";
    document.body.append(link);
    link.click();
    link.remove();
  } finally {
    // Deferred, because the click only STARTS the save and revoking while it is
    // in flight leaves the browser writing an empty file - silently, with the
    // right name and nothing to say it went wrong, which is the same class of
    // lie this whole path exists to prevent. A second rather than a tick: the
    // ceiling here is a 25 MB blob and a browser under load can easily take
    // longer than one turn of the event loop to commit it. In a `finally` so a
    // throw between creating the URL and clicking it cannot leak the blob for
    // the page's lifetime.
    setTimeout(() => URL.revokeObjectURL(objectUrl), 1000);
  }
}

/** What a failed download means for the chip that asked for it. */
export type LivenessVerdict = "expired" | "forget" | "keep";

/**
 * Decide what a refusal says about the row behind the chip.
 *
 * Three outcomes, because a chip has three honest things it can become, and
 * collapsing any two of them tells the reader something false. `expired`: the
 * row is still there and its bytes aged out, which is a retention policy doing
 * its job and reads as a quiet tombstone. `forget`: nothing is left to describe
 * - the row or its bytes are gone - so the chip can only say the record
 * remains. `keep`: the file is presumably fine and this attempt was not about
 * it (still ingesting, over the rate limit, storage briefly down), so the chip
 * keeps its arrow and the reason goes in a banner.
 *
 * Keyed on the CODE as well as the status because the two 410s mean opposite
 * things: `attachment_expired` is a row that outlived its bytes, while an
 * unreadable one is a file that was never finished. Only the first has a state
 * worth showing.
 *
 * @param status - HTTP status of the refusal.
 * @param code - Problem+json `code`, when the response carried one.
 * @returns How the pane's liveness record should change.
 */
export function livenessAfterRefusal(
  status: number,
  code?: string,
): LivenessVerdict {
  if (status === 410) return code === "attachment_expired" ? "expired" : "forget";
  if (status === 404) return "forget";
  return "keep";
}

/**
 * Delete an attachment: its indexed text, its stored bytes, and its row.
 *
 * Taking a file off a turn has to actually remove it. An upload is ingested the
 * moment it lands, so a file that was merely hidden from the composer would
 * still be retrievable, and the assistant would go on answering from a document
 * its owner believes is gone.
 *
 * A 404 counts as success: the file is already gone, which is the end state the
 * caller asked for.
 *
 * @param id - Attachment UUID.
 * @throws ApiError - When the file could not be removed (it is still usable).
 */
export async function deleteAttachment(id: string): Promise<void> {
  const resp = await fetch(`/api/attachments/${encodeURIComponent(id)}`, {
    method: "DELETE",
    cache: "no-store",
  });
  if (!resp.ok && resp.status !== 404) await raise(resp);
}
