"use client";

import { type ReactNode, useState } from "react";

import {
  GeneratedFiles,
  type LiveFiles,
  MessageAttachments,
} from "@/components/attachment-tray";
import { CitationHighlightProvider } from "@/components/chat/citation-context";
import { CitationChip } from "@/components/chat/citation-popover";
import { DisclosureRow } from "@/components/chat/disclosure-row";
import { MarkdownContent } from "@/components/chat/markdown-content";
import { MessageEditor } from "@/components/chat/message-editor";
import { InfoNote } from "@/components/chat/notice";
import { QuoteChip } from "@/components/chat/quote-chip";
import { Icon, MatrixLoader, Step } from "@/components/chat/step";
import { ToolConfirmationCard } from "@/components/chat/tool-confirmation-card";
import { ToolStepRow } from "@/components/chat/tool-step-card";
import { TurnActions } from "@/components/chat/turn-actions";
import type { LitKind } from "@/hooks/use-quote-jump";
import { dedupeNotices, type Message } from "@/lib/messages";
import { clockLabel, elapsedLabel, thoughtLabel, tokensPerSecond } from "@/lib/timeline";

/** The ms behind an ISO stamp, or undefined when it does not parse. */
function stampMs(iso: string | undefined): number | undefined {
  if (!iso) return undefined;
  const ms = new Date(iso).getTime();
  return Number.isNaN(ms) ? undefined : ms;
}

/**
 * One rendered turn, as a timeline.
 *
 * A question is one row: the bubble right-aligned with its clock time in the
 * gutter, its files and the passage it replies to above the words, its action
 * row beneath. A reply is a stack of rows on one rail - thought, each tool,
 * a pending action, the answer, and the closing row that holds the sources,
 * the notices and the footer - every row carrying its elapsed time since the
 * question and a glyph in the colour of its kind. Rows exist only for what
 * happened; a reply with no reasoning and no tools is the answer and its
 * footer.
 *
 * A question being rewritten is the one case where the bubble is not text: it
 * becomes the editor in place, and its action row stands down while it is open.
 */
export function MessageBubble({
  message,
  questionAt,
  latest,
  failure,
  live,
  status,
  modelName,
  modelLoading,
  liveFiles,
  deletingFiles,
  onDeleteFile,
  fetchingFiles,
  onDownloadFile,
  onConfirm,
  confirmBusy,
  onRegenerate,
  onReply,
  onEdit,
  editing,
  onEditSave,
  onEditCancel,
  onQuoteJump,
  quoteGone,
  lit,
}: {
  message: Message;
  /** When the question this reply answers was sent; the rail's clock zero. */
  questionAt?: string;
  /** The newest reply on screen, whose footer shows at rest. */
  latest?: boolean;
  /**
   * The pane's error, when this reply is the one its turn failed on: it is
   * rendered as the reply's closing row (the rail's danger row) in place of
   * the footer.
   */
  failure?: ReactNode;
  /** True while this message is the one currently receiving stream deltas. */
  live: boolean;
  /** Backend progress note shown until the first real delta arrives. */
  status: string | null;
  /** The model the live turn is reaching, for the waiting row; null when unnamed. */
  modelName?: string | null;
  /** The live turn is waiting on the model to load from disk. */
  modelLoading?: boolean;
  /** Status by id for this conversation's files that still exist. */
  liveFiles: LiveFiles;
  /** Ids whose delete is in flight. */
  deletingFiles: ReadonlySet<string>;
  /** Delete one of this turn's files for good. */
  onDeleteFile: (id: string) => void;
  /**
   * Ids whose file is being transferred. The app fetches the bytes itself
   * rather than handing the route to the browser, so this spans the whole
   * download and its chip is dimmed for all of it.
   */
  fetchingFiles: ReadonlySet<string>;
  /** Fetch a file this turn carried or produced, or find out it is no longer there. */
  onDownloadFile: (id: string) => void;
  /** Answer a pending side-effecting-tool confirmation on this message. */
  onConfirm: (decision: "approve" | "deny") => void;
  /** A stream is in flight, so the confirmation buttons are disabled. */
  confirmBusy: boolean;
  /**
   * Answer this turn again, replacing the reply shown (and, further up the
   * thread, what came after it). Passed on any settled reply while the pane can
   * accept the request; the button is absent otherwise.
   */
  onRegenerate?: () => void;
  /** Quote this reply into the composer, to reply to it specifically. */
  onReply?: () => void;
  /** Open this question for rewriting. Absent while the pane cannot take a turn. */
  onEdit?: () => void;
  /** True while this question's bubble is an editor rather than text. */
  editing?: boolean;
  /** Commit the rewrite; the page closes the editor and runs the turn. */
  onEditSave?: (text: string) => void;
  /** Abandon the rewrite, leaving the question as it was. */
  onEditCancel?: () => void;
  /** Follow this question's quote to its source, which is in the thread. */
  onQuoteJump?: () => void;
  /** Why this question's quote leads nowhere: its source is not in the thread. */
  quoteGone?: string;
  /**
   * This turn was just landed on, and wears a cue for a moment: the sweep
   * over its prose where the whole of it was quoted, else the ring.
   */
  lit?: LitKind;
}) {
  // The reader's own say over the thinking row, once they have toggled it;
  // until then it follows the turn (open while reasoning streams, closed after).
  const [thinkToggled, setThinkToggled] = useState<boolean | null>(null);

  if (message.role === "user") {
    const attachments = message.attachments ?? [];
    // The band, wherever the turn shows it: over the editor, inside the
    // bubble, or alone on a file-only turn.
    const chip = message.quote && (
      <QuoteChip text={message.quote.text} onJump={onQuoteJump} gone={quoteGone} />
    );
    return (
      <div
        className="turn turn-user"
        data-message-id={message.id}
        data-lit={lit}
      >
        <Step time={clockLabel(message.created_at)}>
          {/* The files the turn carried, above the question they came with. The
              composer's tray is empty by now: this is where they live, and the
              only place they can be fetched back or deleted once sent. */}
          <MessageAttachments
            attachments={attachments}
            liveFiles={liveFiles}
            deletingIds={deletingFiles}
            fetchingIds={fetchingFiles}
            onDelete={onDeleteFile}
            onDownload={onDownloadFile}
          />
          {editing && onEditSave && onEditCancel ? (
            <>
              {chip}
              {/* The bubble IS the editor while it is open: the turn being
                  changed is the one on screen, and moving its words elsewhere
                  to change them loses that. */}
              <MessageEditor
                initial={message.content}
                onSave={onEditSave}
                onCancel={onEditCancel}
              />
            </>
          ) : (
            <>
              {/* The quote rides inside the bubble, a band at its top: the
                  same strip the composer showed while the reply was pending,
                  so a question keeps the shape it was written in and there is
                  one bubble. It is stored with the turn, so it survives a
                  reload and the message it came from being edited away.
                  A turn can be a file and nothing else, and then the chips
                  above ARE the question: an empty bubble under them would
                  read as a message that failed to send, so there is none,
                  and a quote on such a turn stands alone. */}
              {message.content !== "" ? (
                <div data-part="bubble-user">
                  {chip}
                  {message.content}
                </div>
              ) : (
                chip
              )}
              <TurnActions
                text={message.content}
                align="end"
                onEdit={onEdit}
                onReply={onReply}
              />
            </>
          )}
        </Step>
      </div>
    );
  }

  const thinking = message.thinking ?? "";
  const citations = message.citations ?? [];
  // Word-for-word repeats collapse: a per-call notice on a multi-call round says
  // the same sentence twice, and only the first tells the reader anything.
  const notices = dedupeNotices(message.notices ?? []);
  const toolSteps = message.toolSteps ?? [];
  // A tool is running: its row carries the loader, so no waiting row is needed.
  const toolInFlight = toolSteps.some((step) => step.running);
  // Reasoning streams before the answer: the row stands open and its glyph
  // runs until content, a tool or the end of the turn takes over.
  const reasoningInFlight = live && thinking !== "" && message.content === "" && !toolInFlight;
  // Fold the approve/deny card into the activity trace. Once DECIDED, a confirmed
  // call shows as its own (badged) tool step - approved (it ran) or denied (it was
  // blocked) - so the card drops it: no two cards for one action. While UNDECIDED
  // (including after a failed resume rolled the decision back), every call stays on
  // the card, so the buttons - the only way to resolve the pause - are never
  // hidden. Keying the fold on "a step now exists for this call" is what keeps a
  // failed resume (which creates no step) from stranding the turn behind a hidden
  // card: the decision reverts, the card returns with its buttons.
  const stepCallIds = new Set(toolSteps.map((s) => s.id));
  const confirmCalls = message.confirmation?.calls ?? [];
  const displayConfirmCalls =
    message.confirmationDecision != null
      ? confirmCalls.filter((c) => !stepCallIds.has(c.id))
      : confirmCalls;
  // The turn is paused on a yes: the composer is blocked and the closing row
  // waits, since nothing has closed.
  const paused = message.confirmation != null && message.confirmationDecision == null;
  const settled = !live && !paused;
  // The turn failed with this reply on screen: the error row closes it.
  const failed = message.failed === true;
  // The answer text has arrived; while still live, a caret sits at its edge.
  const showContent = message.content !== "";
  // Which [n] markers in the answer name a real source. A citation can arrive
  // unnumbered (index null), and one that did is a chip only - nothing in the
  // text points at it - so it cannot mark anything. Joined to a STRING here
  // rather than passed as a list: this array is rebuilt on every delta, and a
  // fresh identity would re-render (and re-parse) every settled answer on
  // screen for each token of the one being written.
  const markers = citations
    .filter((c) => c.index != null)
    .map((c) => String(c.index))
    .join(" ");
  // Reaching the model, with nothing else to show for it yet. A finished turn
  // can still be answerless - it died after raising a notice - and a loader
  // under a note nothing is coming would promise an answer that never arrives.
  const showWaiting = !showContent && live && thinking === "" && !toolInFlight;
  // Namespaces this message's citation markers/chips and its tool-trace DOM id
  // (the "View tool step" popover action's scroll target) - the same fallback
  // MarkdownContent already uses for footnote ids, reused here so a live turn
  // (no server id yet) still resolves to one stable key.
  const idKey = message.id ?? "live";
  const files = message.attachments ?? [];

  // The rail's clocks. A live turn stamps its own; a reloaded one knows only
  // when the reply was persisted, which is the end of the turn.
  const thinkingEndedAt = toolSteps[0]?.at ?? message.answerAt ?? message.doneAt;
  const endAt = message.doneAt ?? (settled ? stampMs(message.created_at) : undefined);
  const thinkOpen = thinkToggled ?? reasoningInFlight;
  // Both figures are the server's, live or reloaded: the turn's count over
  // the runtime's own writing time.
  const tokens = message.completionTokens;
  const meta = {
    model: message.model ?? null,
    tokens,
    tokensPerSecond: tokensPerSecond(tokens, message.generationMs),
  };
  const showEnd =
    settled &&
    !failed &&
    (showContent ||
      citations.length > 0 ||
      notices.length > 0 ||
      files.length > 0 ||
      message.interrupted === true ||
      onRegenerate !== undefined);
  const modelLabel = modelName ?? "the model";
  const waitText =
    status ?? (modelLoading ? `Waiting for ${modelLabel} to load` : `Sending to ${modelLabel}`);

  return (
    <div
      className={`turn${live ? " live" : ""}`}
      data-part="bubble-assist"
      data-latest={latest ? "true" : undefined}
      data-message-id={message.id}
      data-lit={lit}
    >
      {/* Scopes this message's citation markers (inside MarkdownContent, below)
          and its chip row (further down) to ONE shared popover: hovering or
          pinning either view highlights both, keyed on citation index. The two
          live in different rows of this turn, which is why they need a common
          ancestor rather than a prop passed straight from one to the other. */}
      <CitationHighlightProvider citations={citations} idKey={idKey} onDownloadFile={onDownloadFile}>
        {showWaiting && (
          <Step kind="wait" glyph={<MatrixLoader variant="twinkle" />}>
            <div className="wait-row">
              {/* Keyed on the words, so a change fades in as a new line rather
                  than rewriting the old one in place. */}
              <span key={waitText} className="tok">
                {waitText}
              </span>
            </div>
          </Step>
        )}
        {thinking !== "" && (
          <Step
            kind="thought"
            part="thinking"
            time={elapsedLabel(questionAt, message.thinkingAt)}
            glyph={
              reasoningInFlight ? (
                <MatrixLoader variant="orbit" rounded color="var(--c-step-thought)" />
              ) : (
                <Icon name="thought" />
              )
            }
          >
            <DisclosureRow
              open={thinkOpen}
              onToggle={() => setThinkToggled(!thinkOpen)}
              summary={
                <>
                  <span>
                    {reasoningInFlight ? "Thinking" : thoughtLabel(message.thinkingAt, thinkingEndedAt)}
                  </span>
                  {reasoningInFlight && <span className="tag-live">thinking</span>}
                </>
              }
            >
              <div className="think-body">{thinking}</div>
            </DisclosureRow>
          </Step>
        )}
        {toolSteps.length > 0 && (
          // Wrapped (rather than left as bare siblings) so a citation's "View
          // tool step" action has one stable id to scroll to and expand - see
          // CitationHighlightProvider.revealToolTrace. The wrapper is
          // `display: contents`, so the rows still sit on the turn's rail.
          <div id={`tool-trace-${idKey}`} data-part="tool-trace-list">
            {toolSteps.map((step, i) => (
              // The tools this turn used, in order, with any narration the model
              // gave before each - the agentic trace, above the final answer.
              <ToolStepRow key={`${i}-${step.id}`} step={step} questionAt={questionAt} />
            ))}
          </div>
        )}
        {message.confirmation && displayConfirmCalls.length > 0 && (
          // The side-effecting action(s) this turn paused on that are still
          // unresolved as a step: while undecided (or rolled back), approve/deny;
          // once denied, the outcome. An approved call drops out the moment its
          // execution step appears, so the run is shown once, not twice.
          <Step
            kind="confirm"
            glyph={<Icon name="gate" />}
            time={elapsedLabel(questionAt, message.confirmAt)}
          >
            <ToolConfirmationCard
              confirmation={{ ...message.confirmation, calls: displayConfirmCalls }}
              decision={message.confirmationDecision}
              busy={confirmBusy}
              onConfirm={onConfirm}
            />
          </Step>
        )}
        {showContent && (
          <Step
            kind="answer"
            glyph={<Icon name="answer" />}
            time={elapsedLabel(questionAt, message.answerAt)}
          >
            <div data-part="content" className="prose">
              {/* The answer is markdown - and the caret is placed inside it, at the
                  end of the streamed text, rather than after the whole block. */}
              <MarkdownContent
                content={message.content}
                live={live}
                markers={markers}
                idKey={idKey}
              />
            </div>
          </Step>
        )}
        {showEnd && (
          <Step kind="end" end glyph={<Icon name="check" />} time={elapsedLabel(questionAt, endAt)}>
            {notices.map((n, i) => (
              // Standing notes about THIS reply (a substitute model answered, an
              // announced search found nothing, an attached file could not be
              // used). Stored with the message, so they are still here after a
              // reload.
              <InfoNote key={`${i}-${n.code}`} message={n.message} />
            ))}
            {message.interrupted && (
              // A reloaded reply that was cut short mid-stream (a client disconnect
              // or a provider error); the partial text above is all that was saved.
              <div>
                <span className="interrupted-chip" title="This reply was cut short before it finished.">
                  Interrupted
                </span>
              </div>
            )}
            {citations.length > 0 && (
              // Sources the answer's inline [n] markers point at, each a trigger for
              // the shared citation popover (icon, label, passage, and the one action
              // its kind derives - see CitationChip). A web result (source_type
              // "web") carries a url and keeps its own link-click on top of that.
              <div data-part="citations">
                {citations.map((c, i) => (
                  // Position-keyed: ids are backend-derived and not guaranteed
                  // unique within one turn's list.
                  <CitationChip key={`${i}-${c.id}`} citation={c} />
                ))}
              </div>
            )}
            {/* Files this reply's tools produced, under the sources it drew on.
                Only generated files render here - a question's own uploads stay
                on the question, where they can still be deleted. */}
            <GeneratedFiles
              attachments={files}
              liveFiles={liveFiles}
              busyIds={fetchingFiles}
              onDownload={onDownloadFile}
            />
            <TurnActions
              text={message.content}
              onRegenerate={onRegenerate}
              onReply={onReply}
              meta={meta}
            />
          </Step>
        )}
        {failed && failure && (
          <Step kind="error" end glyph={<Icon name="x" />} time={elapsedLabel(questionAt, message.failedAt)}>
            {failure}
          </Step>
        )}
      </CitationHighlightProvider>
    </div>
  );
}
