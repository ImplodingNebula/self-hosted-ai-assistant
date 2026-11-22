"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ConfirmDialog } from "@/components/confirm-dialog";
import { ApiError } from "@/lib/api-error";
import {
  ATTACHMENT_ACCEPT,
  ATTACHMENT_MAX_BYTES,
  formatBytes,
} from "@/lib/attachments";
import {
  type KnowledgeDocument,
  type KnowledgeListing,
  addKnowledgeDocument,
  deleteKnowledgeDocument,
  listKnowledgeDocuments,
} from "@/lib/knowledge";

/**
 * The proxy's code for "delivered, still running" - see `lib/backend-proxy.ts`.
 *
 * Branching on a code rather than on the 504 alone: this page must react only
 * to the ceiling the proxy itself imposes, never to a gateway timeout that
 * happened to arrive from somewhere else.
 */
const STILL_WORKING_CODE = "backend_still_working";

/** Human line for a failed call; ApiError carries the problem+json detail. */
function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Is the backend running?";
}

/** What a row calls this entry, falling back to its own id. */
function nameOf(doc: KnowledgeDocument): string {
  return doc.title || doc.id;
}

/** The ingest date, in the reader's own locale, or nothing if unparseable. */
function ingested(doc: KnowledgeDocument): string {
  if (!doc.ingested_at) return "";
  const at = new Date(doc.ingested_at);
  if (Number.isNaN(at.getTime())) return "";
  return at.toLocaleString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit",
  });
}

/**
 * Where the content came from, in words rather than the wire value.
 *
 * No `mcp_records` entry: a tool server's results may have been fetched under
 * one person's credential, so the backend refuses to put them in a shared
 * corpus at all. An unmapped value falls through to itself rather than to a
 * guess, which is the honest rendering of a source type this build predates.
 */
const ORIGINS: Record<string, string> = {
  upload: "Uploaded file",
  records: "Ingest API",
  web: "Web page",
};

function originOf(doc: KnowledgeDocument): string {
  return ORIGINS[doc.source_type] ?? doc.source_type;
}

/** A plural that reads, without a helper library for two words. */
function passages(count: number): string {
  return `${count} passage${count === 1 ? "" : "s"}`;
}

/** A removal the page must ask about before it makes it. */
interface Pending {
  doc: KnowledgeDocument;
}

/**
 * Knowledge: what every grounded answer is drawn from.
 *
 * The workspace corpus, and the first place a person can actually put something
 * into it - before this page the only way in was the ingest API, which takes
 * JSON records rather than files.
 *
 * Adding a file is ONE long request rather than a queued job, because the
 * backend has nowhere to record a half-finished one (see `lib/knowledge.ts`).
 * That decides the shape of the progress here: the control stays busy for as
 * long as the work takes and says so, and every failure arrives as the answer
 * to the request the reader is already watching.
 */
export default function KnowledgePage() {
  const [listing, setListing] = useState<KnowledgeListing | null>(null);
  const [loading, setLoading] = useState(true);
  // Two error slots, as the members page splits them and for the same reason:
  // an upload is refused and then the list is re-read, so a single slot would
  // have the refetch wipe the refusal a few hundred milliseconds after it
  // appeared - the only explanation of what went wrong, gone. `loadError` is
  // the listing's own failure; `actionError` belongs to an add or a remove, and
  // a refresh never touches it.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // What the page is doing, named rather than a bare boolean: an upload can run
  // for minutes, and a reader deserves to be told which long thing is running
  // and on what.
  const [adding, setAdding] = useState<string | null>(null);
  const [removing, setRemoving] = useState<string | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  // Aborts an upload in flight when the page goes away, so a navigation does
  // not leave a request whose answer nobody will ever read.
  const abortRef = useRef<AbortController | null>(null);

  // Promise callbacks (not async/await) keep every setState out of the
  // synchronous call graph, which the set-state-in-effect lint rule rejects for
  // the mount-time call below (the same idiom as the members page).
  const refresh = useCallback(
    () =>
      listKnowledgeDocuments()
        .then((next) => {
          setListing(next);
          setLoadError(null);
        })
        // Only the listing's own slot - a refused upload waiting on screen is
        // not the refetch's to clear.
        .catch((err: unknown) => setLoadError(errorMessage(err)))
        .finally(() => setLoading(false)),
    [],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => () => abortRef.current?.abort(), []);

  const onPick = useCallback(
    (file: File) => {
      if (adding !== null || removing !== null) return;
      const controller = new AbortController();
      abortRef.current = controller;
      setAdding(file.name);
      setActionError(null);
      setNotice(null);
      addKnowledgeDocument(file, controller.signal)
        .then((added) => {
          setNotice(
            added.chunks === 0
              ? `${added.title} was already stored, unchanged - nothing was re-indexed.`
              : `${added.title} is searchable: ${passages(added.chunks)} stored.`,
          );
        })
        .catch((err: unknown) => {
          // Aborted by the unmount above - there is nobody left to tell.
          if (err instanceof DOMException && err.name === "AbortError") return;
          // The proxy's five-minute ceiling, which is not a failure: the
          // backend does not abandon the work when its caller hangs up, so the
          // file is still being indexed and the refresh below will very likely
          // show it. Putting this in the error slot would announce a failure
          // over a document that is about to appear.
          if (err instanceof ApiError && err.code === STILL_WORKING_CODE) {
            setNotice(err.message);
            return;
          }
          setActionError(errorMessage(err));
        })
        .finally(() => {
          // Released only once the list has been re-read, so the new row is on
          // screen before the control frees up.
          void refresh().finally(() => {
            abortRef.current = null;
            setAdding(null);
          });
        });
    },
    [adding, refresh, removing],
  );

  const onRemove = useCallback(
    (doc: KnowledgeDocument) => {
      if (adding !== null || removing !== null) return;
      setRemoving(doc.id);
      setActionError(null);
      setNotice(null);
      deleteKnowledgeDocument(doc.id)
        .then((removed) => {
          setNotice(
            removed.chunks_deleted === 0
              ? `${nameOf(doc)} was already gone - nothing was stored under it.`
              : `${nameOf(doc)} removed: ${passages(removed.chunks_deleted)} deleted.`,
          );
        })
        .catch((err: unknown) => setActionError(errorMessage(err)))
        .finally(() => {
          void refresh().finally(() => setRemoving(null));
        });
    },
    [adding, refresh, removing],
  );

  const busy = adding !== null || removing !== null;
  const items = listing?.items ?? [];
  const unreachable = items.filter((doc) => !doc.reachable).length;

  return (
    <div data-part="knowledge">
      <header className="kb-header">
        <h1 className="mm-title">Knowledge</h1>
        {listing && (
          <p className="mm-meta">
            {items.length === 1
              ? "1 document in this workspace"
              : `${items.length} documents in this workspace`}
            {listing.active_embedding_model
              ? ` - searched with ${listing.active_embedding_model}`
              : ""}
          </p>
        )}
      </header>

      {/* One line before the first control, so the page's verbs are in the
          first viewport. The not-kept fact lives in the remove dialog, where
          it decides something; the default-collection scope note sits under
          the list, where it is read while reconciling rather than adding. */}
      <p className="kb-lede">
        Everything here is shared by the whole workspace, and any passage of it
        can be quoted - and cited - in an answer.
      </p>

      <section className="mm-section">
        <h2 className="mm-section-title">Add a document</h2>
        {/* The one fact that decides whether a file belongs here at all, said
            at the moment somebody is choosing one. */}
        <p className="mm-section-note">
          Everyone in this workspace can search what is added here, so only add
          what every member may read.
        </p>
        <p className="mm-section-note">
          Text and source files, PDF, Office and OpenDocument, RTF, EPUB, and
          images (any text in them is read), up to{" "}
          {formatBytes(ATTACHMENT_MAX_BYTES)}.
        </p>
        <div className="kb-addbox">
          {/* The input is driven by the button beside it: a bare file input
              cannot be styled to the console's control height, and the button
              is what carries the busy word. */}
          <input
            ref={inputRef}
            type="file"
            className="kb-file"
            accept={ATTACHMENT_ACCEPT}
            disabled={busy}
            onChange={(e) => {
              const file = e.target.files?.[0];
              // Reset so picking the same file again still fires onChange.
              e.target.value = "";
              if (file) onPick(file);
            }}
          />
          <button
            type="button"
            className="btn btn-primary"
            disabled={busy}
            onClick={() => inputRef.current?.click()}
          >
            {adding !== null ? "Indexing..." : "Choose a file"}
          </button>
          {adding !== null && (
            <div className="kb-progress">
              {/* The activity panel's shape: a named progressbar with NO
                  aria-valuenow, because an indeterminate bar must not claim a
                  value. There is genuinely no fraction to report - the backend
                  answers once, when the whole file is indexed. */}
              <div
                className="mm-bar indeterminate"
                role="progressbar"
                aria-label={`Indexing ${adding}`}
                aria-valuemin={0}
                aria-valuemax={100}
              >
                <i />
              </div>
              {/* The leave-the-page-open instruction lives here, in the busy
                  state, rather than in standing copy above the button: it
                  applies exactly while this line is on screen. */}
              <span role="status">
                Reading and indexing {adding}. This can take a few minutes for
                a large file - leave the page open until it finishes.
              </span>
            </div>
          )}
        </div>
      </section>

      {/* Announced, not merely displayed: a refused upload is this page's
          principal failure mode, and the backend's own sentence is the only
          explanation of it. Its own slot, so re-reading the list underneath
          cannot wipe it. */}
      {actionError && (
        <p className="mm-banner error" role="alert">
          {actionError}
        </p>
      )}

      {notice && (
        <p className="mm-note ok" role="status">
          {notice}
        </p>
      )}

      {/* The listing's own failure, kept apart from a refused action so the two
          never clear each other. */}
      {loadError && <p className="mm-banner error">{loadError}</p>}

      {/* The one thing a reader must not do is take this page for the whole
          corpus. Deliberately does NOT claim these are the newest: the flag
          also fires when the walk behind the list stopped early, and it stops
          in storage order rather than in date order, so what survived is an
          arbitrary sample of the corpus rather than the top of it. The same
          cut-off is why the counts are floors. */}
      {listing?.truncated && (
        <p className="mm-banner warn">
          This workspace holds more than can be listed at once. What is shown is
          part of the corpus, not all of it - other entries are stored and
          searchable without appearing here, and each passage count below is at
          least what that entry holds rather than necessarily all of it.
        </p>
      )}

      {unreachable > 0 && (
        <p className="mm-banner warn">
          {unreachable === 1
            ? "1 entry was indexed by a different embedding model"
            : `${unreachable} entries were indexed by a different embedding model`}
          , so no answer can search them. Add those files again with the model
          in use now, or remove them.
        </p>
      )}

      {loading && <p className="mm-muted">Loading...</p>}

      {listing && items.length > 0 && (
        <ul className="kb-list" aria-busy={busy}>
          {items.map((doc) => (
            <DocumentRow
              key={doc.id}
              doc={doc}
              busy={busy}
              removing={removing === doc.id}
              onRemove={() => setPending({ doc })}
            />
          ))}
        </ul>
      )}

      {listing && items.length === 0 && !loadError && (
        <p className="mm-muted">
          Nothing has been added yet. Until something is, answers come only from
          what the model already knows and from whatever a conversation is
          given.
        </p>
      )}

      {/* Said rather than glossed over, but at the foot: a turn retrieves
          across ALL of the tenant's corpora while this page manages one, and
          without the sentence the list reads as the whole of what grounds an
          answer. It matters when reconciling this list against what answers
          cite, not while choosing a file to add. */}
      {listing && (
        <p className="mm-muted">
          This page manages the default collection. Records sent to the ingest
          API under another collection name are searched and cited just the
          same, but are not listed here.
        </p>
      )}

      {pending && (
        <ConfirmDialog
          title="Remove this document?"
          body={
            `Everything indexed from ${nameOf(pending.doc)} will be deleted - ` +
            `${passages(pending.doc.chunks)} the assistant can currently quote. ` +
            "The file itself was never kept, so putting it back means adding it " +
            "again."
          }
          confirmLabel="Remove"
          onConfirm={() => {
            onRemove(pending.doc);
            setPending(null);
          }}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
}

/** One stored entry: what it is, where it came from, and what it holds. */
function DocumentRow({
  doc,
  busy,
  removing,
  onRemove,
}: {
  doc: KnowledgeDocument;
  busy: boolean;
  removing: boolean;
  onRemove: () => void;
}) {
  const at = ingested(doc);
  return (
    <li className="kb-row">
      <div className="kb-identity">
        <span className="kb-name">{nameOf(doc)}</span>
        <span className="mm-meta">
          {originOf(doc)}
          {at ? ` - added ${at}` : ""}
        </span>
      </div>

      <div className="kb-control">
        {/* Color plus a word, never color alone - the console's pill rule. */}
        <span className={doc.reachable ? "mm-state ok" : "mm-state warn"}>
          {doc.reachable ? "Searchable" : "Not searchable"}
        </span>
        <button
          type="button"
          className="btn btn-danger-ghost"
          disabled={busy}
          onClick={onRemove}
        >
          {removing ? "Removing..." : "Remove"}
        </button>
      </div>

      <dl className="kb-facts">
        <div className="mm-fact">
          <dt>Passages</dt>
          <dd>{doc.chunks}</dd>
        </div>
        <div className="mm-fact">
          <dt>Parts</dt>
          <dd>{doc.documents}</dd>
        </div>
        <div className="mm-fact">
          <dt>Indexed by</dt>
          {/* An exact tag somebody compares against configuration, so it reads
              in the monospace face - the same reasoning as .mm-tag. */}
          <dd className="kb-mono">
            {doc.embed_models.length > 0
              ? doc.embed_models.join(", ")
              : "unknown"}
          </dd>
        </div>
        <div className="mm-fact">
          <dt>Fingerprint</dt>
          {/* Only for a source that IS one document. The hash is computed per
              document, so a file that split into several has one apiece, and
              printing whichever happened to be folded in first would put a
              12-character identifier under a heading it does not identify. */}
          <dd className="kb-mono">
            {doc.documents === 1 && doc.content_hash
              ? doc.content_hash.slice(0, 12)
              : "per part"}
          </dd>
        </div>
      </dl>

      {doc.url && (
        <p className="kb-link">
          <a href={doc.url} rel="noreferrer noopener" target="_blank">
            {doc.url}
          </a>
        </p>
      )}
    </li>
  );
}
