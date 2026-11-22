"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api-error";
import {
  ATTACHMENT_MAX_BYTES,
  type AttachmentOrigin,
  deleteAttachment,
  formatBytes,
  getAttachment,
  uploadAttachment,
} from "@/lib/attachments";
import type { Attachment, AttachmentStatus } from "@/lib/types";

// How often a still-parsing attachment is re-polled. Ingestion is usually a few
// seconds; a sub-second poll would hammer the endpoint for no gain.
const POLL_INTERVAL_MS = 1200;

/**
 * Lifecycle of a tray attachment. "uploading" and "deleting" are client-only
 * states either side of the server row; the rest mirror the ingestion status.
 */
export type TrayStatus = "uploading" | "deleting" | AttachmentStatus;

/** One file in the composer tray, from the moment it is picked. */
export interface TrayAttachment {
  /** Stable client key, assigned before the server responds. */
  localId: string;
  /** Display name. */
  filename: string;
  /** Declared media type; recorded on the message the file is sent with. */
  contentType: string;
  /** Size in bytes (from the File; the server re-checks the real length). */
  sizeBytes: number;
  /** Current lifecycle state. */
  status: TrayStatus;
  /** Server attachment id once the upload responds; null until then. */
  serverId: string | null;
  /** Failure reason for a rejected upload or a failed parse; null otherwise. */
  error: string | null;
}

/** What {@link useAttachments} exposes to the composer. */
export interface UseAttachments {
  /** The tray contents, in the order files were added. */
  attachments: TrayAttachment[];
  /** Add files: validate size, upload, then poll each to a terminal state. */
  addFiles: (files: File[], origin?: AttachmentOrigin) => Promise<void>;
  /**
   * Delete one file for good (text, bytes and row), then drop its chip.
   *
   * Resolves with the deleted file's server id so the caller can forget it
   * everywhere else it is named - the same file can appear on a sent message as
   * well as in the tray. Resolves with null when the chip never reached the
   * server and there was nothing to delete.
   *
   * Rejects if the server refused, leaving the chip in place and marked: the
   * file is still there, so the caller must say so rather than let it look gone.
   */
  remove: (localId: string) => Promise<string | null>;
  /** Drop named chips WITHOUT deleting anything (the turn that sent them owns them now). */
  drop: (localIds: string[]) => void;
  /** Put chips back after a send the backend never accepted. */
  restore: (entries: TrayAttachment[]) => void;
  /** Empty the tray WITHOUT deleting anything (a pane switch or a new chat). */
  clear: () => void;
  /** Replace the tray with files a reopened conversation is still holding. */
  seed: (attachments: Attachment[]) => void;
  /** True while any file is still uploading (Send waits for these). */
  uploading: boolean;
}

/**
 * The tray entries a turn should carry: uploaded, still live, each named once.
 *
 * The single definition of that policy. A plain function rather than a value on
 * the hook because the composer must apply it to the tray AS IT IS AT SEND TIME
 * (read from a ref), and a render-time value would be a stale copy.
 *
 * A failed file has no text to offer, an expired one has no bytes left at all,
 * and one mid-delete is on its way out, so naming any of them on a turn would
 * point the answer at nothing. A chip carrying an error is left out for the same
 * reason: a delete that failed on the client may still have landed server-side,
 * so naming it risks pointing the turn at a row that is already gone (the chip
 * stays visible with its error either way).
 * Duplicate server ids collapse to one entry - the same bytes uploaded twice
 * into a conversation dedupe to a single row server-side, and the turn should
 * reference it once.
 */
export function attachmentsForTurn(tray: TrayAttachment[]): TrayAttachment[] {
  const seen = new Set<string>();
  const carried: TrayAttachment[] = [];
  for (const a of tray) {
    if (
      !a.serverId ||
      a.status === "failed" ||
      a.status === "expired" ||
      a.status === "deleting" ||
      a.error
    ) {
      continue;
    }
    if (seen.has(a.serverId)) continue;
    seen.add(a.serverId);
    carried.push(a);
  }
  return carried;
}

/** Patch shape for {@link makeUpdate}; only mutable lifecycle fields. */
type Patch = Partial<Pick<TrayAttachment, "status" | "error" | "serverId">>;

// What a browser sends when it cannot guess a type; the backend sniffs the bytes.
const UNKNOWN_TYPE = "application/octet-stream";

/** Build the entry a picked file starts as, rejecting an over-large one up front. */
function makeEntry(file: File): TrayAttachment {
  const tooBig = file.size > ATTACHMENT_MAX_BYTES;
  return {
    localId: crypto.randomUUID(),
    filename: file.name,
    contentType: file.type || UNKNOWN_TYPE,
    sizeBytes: file.size,
    status: tooBig ? "failed" : "uploading",
    serverId: null,
    error: tooBig
      ? `This file is larger than the ${formatBytes(ATTACHMENT_MAX_BYTES)} limit.`
      : null,
  };
}

/**
 * Composer attachment tray: upload state plus status polling.
 *
 * The tray owns the client lifecycle of each file (upload -> pending/parsing ->
 * ready/failed). Uploading needs a conversation to attach to, so the caller
 * passes an idempotent `ensureConversationId` that mints one on first use and
 * returns the same id thereafter - every file in a tray lands on one thread.
 *
 * @param ensureConversationId - Resolves the conversation to upload into,
 *   creating it on first call. Must be idempotent (return the same id).
 */
export function useAttachments(
  ensureConversationId: () => Promise<string>,
): UseAttachments {
  const [attachments, setAttachments] = useState<TrayAttachment[]>([]);

  // localIds the user removed WHILE their upload was still in flight. The chip
  // cannot just be dropped: the POST may land after, creating a stored, ingested
  // row. uploadOne reads this the instant the server hands back an id and deletes
  // the file for good, so a cancelled upload never leaves an answerable-but-
  // undeletable orphan behind. A ref, not state - it is read in a callback and
  // must not trigger a render.
  const pendingRemovalRef = useRef<Set<string>>(new Set());

  // Update one entry, keeping the array reference stable when nothing actually
  // changed - a no-op poll must not re-render or restart the poll timer.
  const update = useCallback((localId: string, patch: Patch) => {
    setAttachments((list) => {
      const idx = list.findIndex((a) => a.localId === localId);
      if (idx === -1) return list; // removed while in flight: do not resurrect
      const current = list[idx];
      const merged = { ...current, ...patch };
      if (
        merged.status === current.status &&
        merged.error === current.error &&
        merged.serverId === current.serverId
      ) {
        return list;
      }
      const next = list.slice();
      next[idx] = merged;
      return next;
    });
  }, []);

  const uploadOne = useCallback(
    async (
      localId: string,
      file: File,
      origin: AttachmentOrigin,
      conversationId: string,
    ) => {
      try {
        const att = await uploadAttachment(file, conversationId, origin);
        if (pendingRemovalRef.current.has(localId)) {
          // Removed while this was still uploading. The upload landed anyway - a
          // stored, ingested row - so deleting it is the only way to honor the
          // removal; dropping the chip alone would leave a file the user was told
          // is gone that the assistant could still answer from.
          pendingRemovalRef.current.delete(localId);
          try {
            await deleteAttachment(att.id);
          } catch {
            // Best-effort; the orphan/GC sweep is the backstop. Drop the chip
            // regardless so the tray reflects the user's intent.
          }
          setAttachments((list) => list.filter((a) => a.localId !== localId));
          return;
        }
        update(localId, {
          serverId: att.id,
          status: att.status,
          error: att.error,
        });
      } catch (e) {
        if (pendingRemovalRef.current.has(localId)) {
          // Removed mid-upload and the upload failed: nothing was stored, so just
          // drop the chip.
          pendingRemovalRef.current.delete(localId);
          setAttachments((list) => list.filter((a) => a.localId !== localId));
          return;
        }
        update(localId, {
          status: "failed",
          error: e instanceof ApiError ? e.message : "Upload failed.",
        });
      }
    },
    [update],
  );

  const addFiles = useCallback(
    async (files: File[], origin: AttachmentOrigin = "upload") => {
      if (files.length === 0) return;
      const staged = files.map((file) => ({ file, entry: makeEntry(file) }));
      setAttachments((list) => [...list, ...staged.map((s) => s.entry)]);

      const toUpload = staged.filter((s) => s.entry.status === "uploading");
      if (toUpload.length === 0) return; // all rejected up front (too large)

      let conversationId: string;
      try {
        conversationId = await ensureConversationId();
      } catch (e) {
        const message =
          e instanceof ApiError
            ? e.message
            : "Couldn't start a conversation for the upload.";
        for (const s of toUpload) {
          const localId = s.entry.localId;
          if (pendingRemovalRef.current.has(localId)) {
            // Removed while the conversation was still being created, so nothing
            // was ever uploaded: honor the removal and clear the intent, rather
            // than resurrect the chip as failed (uploadOne, which normally does
            // this, never runs for these).
            pendingRemovalRef.current.delete(localId);
            setAttachments((list) => list.filter((a) => a.localId !== localId));
            continue;
          }
          update(localId, { status: "failed", error: message });
        }
        return;
      }
      await Promise.all(
        toUpload.map((s) =>
          uploadOne(s.entry.localId, s.file, origin, conversationId),
        ),
      );
    },
    [ensureConversationId, update, uploadOne],
  );

  const remove = useCallback(
    async (localId: string): Promise<string | null> => {
      const entry = attachments.find((a) => a.localId === localId);
      if (!entry || entry.status === "deleting") return null;
      if (!entry.serverId) {
        if (entry.status === "uploading") {
          // The upload is in flight, so dropping the chip now would strand the
          // file: if the POST lands after, it creates a stored, ingested row with
          // no chip and no id on the client - answerable but undeletable. Mark it
          // and let uploadOne delete it the instant the server returns its id.
          pendingRemovalRef.current.add(localId);
          update(localId, { status: "deleting", error: null });
          return null;
        }
        // Rejected up front (too large): nothing was ever sent, so just drop it.
        setAttachments((list) => list.filter((a) => a.localId !== localId));
        return null;
      }
      const previous = entry.status;
      const serverId = entry.serverId;
      update(localId, { status: "deleting", error: null });
      try {
        await deleteAttachment(serverId);
        // EVERY chip for that file goes, not just the one clicked. Attaching the
        // same bytes twice dedupes to ONE row server-side, so two chips can name
        // a single file - and the deleted file's twin would sit there looking
        // ready while pointing at a row that no longer exists.
        setAttachments((list) => list.filter((a) => a.serverId !== serverId));
        return serverId;
      } catch (e) {
        // The file is still stored and still retrievable, so the chip STAYS,
        // marked with the reason. Removing it here would tell the user the file
        // is gone while the assistant could still answer from it - the exact lie
        // this delete exists to prevent. Rethrown so the composer can say it out
        // loud too: a chip that quietly turns red is easy to miss, and the
        // difference between "removed" and "still there" is the whole point.
        update(localId, {
          status: previous,
          error: e instanceof ApiError ? e.message : "Couldn't remove this file.",
        });
        throw e;
      }
    },
    [attachments, update],
  );

  const drop = useCallback((localIds: string[]) => {
    if (localIds.length === 0) return;
    const dropped = new Set(localIds);
    setAttachments((list) => list.filter((a) => !dropped.has(a.localId)));
  }, []);

  const restore = useCallback((entries: TrayAttachment[]) => {
    if (entries.length === 0) return;
    setAttachments((list) => {
      // Idempotent: a restore that races anything which already put a chip back
      // must not double it.
      const present = new Set(list.map((a) => a.localId));
      const back = entries.filter((e) => !present.has(e.localId));
      return back.length === 0 ? list : [...back, ...list];
    });
  }, []);

  const clear = useCallback(() => setAttachments([]), []);

  const seed = useCallback((rows: Attachment[]) => {
    setAttachments(
      rows.map((row) => ({
        localId: crypto.randomUUID(),
        filename: row.filename,
        contentType: row.content_type,
        sizeBytes: row.size_bytes,
        status: row.status,
        serverId: row.id,
        error: row.error,
      })),
    );
  }, []);

  // Poll every uploaded-but-not-terminal attachment until it settles. The
  // effect re-subscribes only when a status actually changes (see `update`), so
  // a steady-state parse polls at a fixed cadence rather than restarting each tick.
  useEffect(() => {
    const pending = attachments.filter(
      (a) => a.serverId && (a.status === "pending" || a.status === "parsing"),
    );
    if (pending.length === 0) return;
    let active = true;
    const timer = setInterval(() => {
      void Promise.all(
        pending.map(async (a) => {
          try {
            const fresh = await getAttachment(a.serverId as string);
            if (active) update(a.localId, { status: fresh.status, error: fresh.error });
          } catch (e) {
            // A permanent 404 means the row is gone for good (deleted from
            // another surface, or swept): stop polling it and fail the chip so it
            // is not carried onto the next turn. Anything else is a transient
            // blip - keep trying.
            if (active && e instanceof ApiError && e.status === 404) {
              update(a.localId, {
                status: "failed",
                error: "This file is no longer available.",
              });
            }
          }
        }),
      );
    }, POLL_INTERVAL_MS);
    return () => {
      active = false;
      clearInterval(timer);
    };
  }, [attachments, update]);

  const uploading = attachments.some((a) => a.status === "uploading");

  return { attachments, addFiles, remove, drop, restore, clear, seed, uploading };
}
