"use client";

import { Icon } from "@/components/chat/step";

import { useRef } from "react";

import {
  ATTACHMENT_ACCEPT,
  EXPIRED_DRAFT_NOTE,
  EXPIRED_GENERATED_NOTE,
  formatBytes,
  sentFileState,
  servesBytes,
} from "@/lib/attachments";
import type { TrayAttachment, TrayStatus } from "@/hooks/use-attachments";
import type { AttachmentStatus, MessageAttachment } from "@/lib/types";

/** The live state of the files a transcript names, by attachment id. */
export type LiveFiles = ReadonlyMap<string, AttachmentStatus>;

/** Short human label for a tray attachment's lifecycle state. */
const STATUS_LABEL: Record<TrayStatus, string> = {
  uploading: "Uploading",
  deleting: "Removing",
  pending: "Queued",
  parsing: "Processing",
  ready: "Ready",
  failed: "Failed",
  // Retention took the bytes while the row stayed. Rare in a draft tray - a file
  // has to sit unsent for a long time to get here - but a chip that could reach
  // this state and had no word for it would spin "Processing" forever.
  expired: "Expired",
};

// suppressHydrationWarning on every svg and its shape children: dark-mode
// browser extensions stamp stroke/fill attributes onto inline SVGs before React
// hydrates (see the note in page.tsx). React suppresses per element, not per
// subtree, so each glyph carries it on both the svg and its paths.
function Spinner() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-3 w-3 animate-spin"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M8 1.5a6.5 6.5 0 1 1-6.5 6.5" />
    </svg>
  );
}

function CheckIcon() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-3 w-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M3.5 8.5l3 3 6-7" />
    </svg>
  );
}

function AlertIcon() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-3 w-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M8 5v3.5M8 11h.01" />
      <circle suppressHydrationWarning cx="8" cy="8" r="6.5" />
    </svg>
  );
}

/**
 * The status glyph shown at the head of a chip.
 *
 * The three branches are ordered by what the reader most needs to know, and the
 * default is the SPINNER - so any state without a branch reads as "still
 * working". That is why `gone` is a parameter rather than something inferred
 * from an unrecognised status: a file whose bytes have aged out is finished, and
 * would otherwise spin forever waiting for an ingestion that already ended.
 */
function StatusGlyph({
  status,
  problem,
  gone,
}: {
  status: TrayStatus;
  problem: boolean;
  gone: boolean;
}) {
  if (problem) {
    return (
      <span className="chip-glyph problem">
        <AlertIcon />
      </span>
    );
  }
  if (gone) {
    // Not an error - nothing went wrong, the file simply is not there any more,
    // which is the same thing a deleted file's chip on a sent turn says.
    return (
      <span className="chip-glyph">
        <PaperclipIcon />
      </span>
    );
  }
  if (status === "ready") {
    return (
      <span className="chip-glyph ok">
        <CheckIcon />
      </span>
    );
  }
  return (
    <span className="chip-glyph">
      <Spinner />
    </span>
  );
}

/** The X glyph shared by the chip remove buttons. */
function CloseIcon() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-3 w-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M4 4l8 8M12 4l-8 8" />
    </svg>
  );
}

/**
 * The row of attachment chips shown above the composer input.
 *
 * Each chip carries its filename, size, and lifecycle state, and exposes any
 * failure reason as a tooltip.
 *
 * The remove button means one of two things, and says which. For a file no
 * question has been sent with yet, it DELETES the file for good - its indexed
 * text, its stored bytes and its row - because nothing else refers to it. But
 * attaching the same bytes twice in one conversation dedupes to a single file
 * server-side, so a chip can name a file an EARLIER question already carries;
 * destroying that would strip the file out of a turn already answered from it.
 * Such a chip is only un-staged from the next turn, and `staged` marks it so the
 * label reads "Remove from this message" rather than "Delete". The file itself
 * can still be deleted from the chip on the question that carries it.
 */
export function AttachmentTray({
  attachments,
  onRemove,
  staged,
}: {
  attachments: TrayAttachment[];
  onRemove: (localId: string) => void;
  /** Server ids an already-sent question carries: removing these un-stages, never deletes. */
  staged?: ReadonlySet<string>;
}) {
  if (attachments.length === 0) return null;
  return (
    <ul className="chip-row" aria-label="Attachments">
      {attachments.map((a) => {
        // ANY error makes the chip read as a problem, not just a failed parse:
        // a refused delete leaves a live chip whose file is still stored and
        // still answerable, and a chip that looked ordinary would be taken for
        // one that was quietly removed.
        const problem = a.status === "failed" || a.error !== null;
        // An expired file is not a problem, it is a tombstone: nothing went
        // wrong, its bytes aged out. It reads as the same struck-through chip a
        // deleted file gets on a sent turn, rather than as a red failure - but
        // it must still be visibly finished, because it can no longer ride a
        // turn (see attachmentsForTurn) and a chip that looked ordinary would
        // promise a file the next question cannot carry. An error outranks it:
        // a failed action is news, and the chip has one line to give.
        const gone = !problem && a.status === "expired";
        const busy = a.status === "deleting";
        // An earlier question already carries this exact file (see the note on
        // this component): its X un-stages, it does not destroy.
        const unstagesOnly = a.serverId !== null && (staged?.has(a.serverId) ?? false);
        return (
          <li
            key={a.localId}
            title={a.error ?? (gone ? EXPIRED_DRAFT_NOTE : undefined)}
            className={`chip${problem ? " problem" : ""}${gone ? " gone" : ""}${busy ? " busy" : ""}`}
          >
            <StatusGlyph status={a.status} problem={problem} gone={gone} />
            <span className={`chip-name${gone ? " gone" : ""}`} title={a.filename}>
              {a.filename}
            </span>
            <span className="chip-meta">
              {a.status === "failed" || a.status === "expired"
                ? STATUS_LABEL[a.status]
                : formatBytes(a.sizeBytes)}
            </span>
            {gone && <span className="sr-only">{EXPIRED_DRAFT_NOTE}</span>}
            <button
              type="button"
              onClick={() => onRemove(a.localId)}
              disabled={busy}
              title={
                unstagesOnly
                  ? `Remove ${a.filename} from this message. An earlier message keeps it.`
                  : `Delete ${a.filename} for good`
              }
              aria-label={
                unstagesOnly
                  ? `Remove ${a.filename} from this message`
                  : `Delete ${a.filename} for good`
              }
              className="chip-x"
            >
              <CloseIcon />
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/**
 * The files a sent turn carried, shown on its message bubble.
 *
 * The chips are permanent, but the files they name are not. A chip stays after
 * its file is deleted, because the turn really did carry it and a transcript
 * that quietly forgot would misrepresent what the answer was given - it just
 * stops claiming the file is still there.
 *
 * That distinction is the whole point of showing them at all. An answer only
 * CITES a file it quotes inline, so a reply that read a document and paraphrased
 * it cites nothing - and without these chips the file that shaped the answer
 * would be invisible, with no way to tell it apart from one that never arrived.
 *
 * Which is why the chip carries a download. The transcript is where a sent file
 * lives once the composer's tray has emptied, so this is the only place it can
 * be had back - and handing a file back to whoever sent it is the plainest
 * thing a file chip can offer. It appears exactly when the server would serve
 * the bytes (see `servesBytes`), which notably includes a file whose INGESTION
 * failed: the assistant could not read it, the original is untouched, and the
 * chip says both.
 *
 * @param attachments - The files recorded on this message.
 * @param liveFiles - Status by id for the files that still exist; an id absent
 *   from it was deleted.
 * @param deletingIds - Ids whose delete is in flight.
 * @param fetchingIds - Ids whose file is being transferred. It spans the WHOLE
 *   download, because the app fetches the bytes itself rather than handing the
 *   route to the browser - so on a large file it is a real wait, and the chip
 *   dims for all of it.
 * @param onDelete - Delete a file for good (text, bytes and row). Omit for a
 *   read-only transcript.
 * @param onDownload - Fetch one file back (and reconcile the chip if it is gone).
 */
export function MessageAttachments({
  attachments,
  liveFiles,
  deletingIds,
  fetchingIds,
  onDelete,
  onDownload,
}: {
  attachments: MessageAttachment[];
  liveFiles: LiveFiles;
  deletingIds: ReadonlySet<string>;
  fetchingIds: ReadonlySet<string>;
  onDelete?: (id: string) => void;
  onDownload: (id: string) => void;
}) {
  if (attachments.length === 0) return null;
  return (
    <ul className="chip-row on-turn" aria-label="Attached files">
      {attachments.map((a) => {
        const status = liveFiles.get(a.id);
        const { label, note, gone, problem } = sentFileState(status, a.size_bytes);
        const deleting = deletingIds.has(a.id);
        const fetching = fetchingIds.has(a.id);
        const downloadable = servesBytes(status, a.origin);
        return (
          <li
            key={a.id}
            title={note ?? undefined}
            className={`chip${problem ? " problem" : ""}${gone ? " gone" : ""}${
              deleting || fetching ? " busy" : ""
            }`}
          >
            <span className={`chip-glyph${problem ? " problem" : ""}`}>
              {problem ? <AlertIcon /> : <PaperclipIcon />}
            </span>
            <span className={`chip-name${gone ? " gone" : ""}`} title={a.filename}>
              {a.filename}
            </span>
            <span className="chip-meta">{label}</span>
            {note && <span className="sr-only">{note}</span>}
            {downloadable && (
              // A button, not a link, and the difference is the slice's whole
              // guarantee. A link exposes two paths this component cannot
              // intercept - middle-click and "save link as" both bypass React's
              // onClick entirely - and each of them hands the browser the raw
              // route, where a refusal is a small JSON document that gets saved
              // to disk under the file's own name. There is no way to guard
              // those paths, so there is no link to take them. Native keyboard
              // activation comes with the button.
              <button
                type="button"
                className="chip-get"
                disabled={fetching}
                aria-busy={fetching || undefined}
                aria-label={`Download ${a.filename}`}
                title={problem ? "Download the original file" : "Download this file"}
                onClick={() => onDownload(a.id)}
              >
                {/* The arrow becomes the tool row's running ring while the
                    file is on its way. The app fetches the bytes itself, so
                    that can be a real wait, and a chip that only dimmed would
                    leave the reader guessing whether the click registered. */}
                {fetching ? <span className="chip-spin" aria-hidden="true" /> : <DownloadIcon />}
              </button>
            )}
            {status !== undefined && onDelete && (
              <button
                type="button"
                onClick={() => onDelete(a.id)}
                // Busy as a unit: a delete racing its own file's transfer buys
                // nothing, and either order of the two ends somewhere odd.
                disabled={deleting || fetching}
                aria-label={`Delete ${a.filename}`}
                title="Delete this file"
                className="chip-x"
              >
                <CloseIcon />
              </button>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** The tray-down arrow on the chips that fetch a file. */
function DownloadIcon() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-3 w-3"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M8 2.5v7M5 7l3 3 3-3" />
      <path suppressHydrationWarning d="M3 11.5v1a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1v-1" />
    </svg>
  );
}

/**
 * The files an assistant turn PRODUCED, shown under its reply.
 *
 * The mirror of {@link MessageAttachments}, and deliberately not the same
 * component: those chips are about a file the reader gave the assistant and can
 * take back, while these are about one the assistant made and the reader can
 * only take away. So the affordance is a download, there is no delete, and the
 * dead state means something different - a generated file that is gone can be
 * asked for again, which is worth saying rather than mourning.
 *
 * Each chip is a BUTTON rather than a link, for the reason spelled out on its
 * sibling: a link's middle-click and "save link as" reach the route without
 * passing through any handler, and a refusal fetched that way is saved to disk
 * as the file. Nothing in this component can intercept those, so the chip does
 * not offer them. Keyboard activation is the button's own.
 *
 * @param attachments - The message's files; only the generated ones render.
 * @param liveFiles - Status by id for the files still believed to name stored
 *   bytes; anything the server would not serve is shown as expired.
 * @param busyIds - Ids whose file is being transferred. The app fetches the
 *   bytes itself, so this spans the whole download rather than a check in front
 *   of one, and the chip stays dimmed until it lands.
 * @param onDownload - Fetch one file (and reconcile the chip if it is gone).
 */
export function GeneratedFiles({
  attachments,
  liveFiles,
  busyIds,
  onDownload,
}: {
  attachments: MessageAttachment[];
  liveFiles: LiveFiles;
  busyIds: ReadonlySet<string>;
  onDownload: (id: string) => void;
}) {
  const generated = attachments.filter((a) => a.origin === "generated");
  if (generated.length === 0) return null;
  return (
    <ul className="chip-row on-reply" data-part="turn-files" aria-label="Files this reply produced">
      {generated.map((a) => {
        const live = servesBytes(liveFiles.get(a.id), a.origin);
        const busy = busyIds.has(a.id);
        if (!live) {
          return (
            <li key={a.id} className="chip gone" title={EXPIRED_GENERATED_NOTE}>
              <span className="chip-glyph">
                <PaperclipIcon />
              </span>
              <span className="chip-name gone" title={a.filename}>
                {a.filename}
              </span>
              <span className="chip-meta">Expired</span>
              <span className="sr-only">{EXPIRED_GENERATED_NOTE}</span>
            </li>
          );
        }
        return (
          <li key={a.id} className={`chip${busy ? " busy" : ""}`}>
            <button
              type="button"
              className="chip-download"
              disabled={busy}
              aria-busy={busy || undefined}
              onClick={() => onDownload(a.id)}
            >
              <span className="chip-glyph">
                {/* Same running ring the sent-question chip shows, and the tool
                    row before it: one vocabulary for "working". */}
                {busy ? <span className="chip-spin" aria-hidden="true" /> : <DownloadIcon />}
              </span>
              <span className="chip-name" title={a.filename}>
                {a.filename}
              </span>
              <span className="chip-meta">{formatBytes(a.size_bytes)}</span>
              <span className="sr-only">Download</span>
            </button>
          </li>
        );
      })}
    </ul>
  );
}

/** The paperclip glyph, shared by the attach button and the message chips. */
function PaperclipIcon({ className = "h-3 w-3" }: { className?: string }) {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 24 24"
      className={className}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M21 11.5l-8.5 8.5a5 5 0 0 1-7-7l8.5-8.5a3.3 3.3 0 0 1 4.7 4.7L8.6 17.8a1.7 1.7 0 0 1-2.4-2.4l7.9-7.9" />
    </svg>
  );
}

/**
 * Paperclip button that opens the file picker.
 *
 * The picker's `accept` list is only a hint; the backend decides what it can
 * read by sniffing the bytes. Selecting the same file twice in a row still
 * fires (the input value is reset after each pick).
 */
export function AttachButton({
  onPick,
  disabled,
}: {
  onPick: (files: File[]) => void;
  disabled?: boolean;
}) {
  const inputRef = useRef<HTMLInputElement>(null);
  return (
    <>
      <input
        ref={inputRef}
        type="file"
        multiple
        accept={ATTACHMENT_ACCEPT}
        className="hidden"
        onChange={(e) => {
          const files = Array.from(e.target.files ?? []);
          // Reset so picking the same file again still triggers onChange.
          e.target.value = "";
          if (files.length > 0) onPick(files);
        }}
      />
      <button
        type="button"
        onClick={() => inputRef.current?.click()}
        disabled={disabled}
        aria-label="Attach files"
        title="Attach files"
        className="ctl"
      >
        <Icon name="clip" />
      </button>
    </>
  );
}
