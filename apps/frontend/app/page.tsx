"use client";

import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
} from "react";

import { AnimatedTitle } from "@/components/animated-title";
import {
  AttachButton,
  AttachmentTray,
  type LiveFiles,
} from "@/components/attachment-tray";
import {
  ComposerInput,
  type ComposerInputHandle,
} from "@/components/chat/composer-input";
import { ComposerNotice } from "@/components/chat/composer-notice";
import { ContextMeter } from "@/components/chat/context-meter";
import { ErrorNote } from "@/components/chat/error-note";
import { MessageBubble } from "@/components/chat/message-bubble";
import { ModelSelector } from "@/components/chat/model-selector";
import { ModelStatus, type ModelWhereabouts } from "@/components/chat/model-status";
import { NoChatModelNote } from "@/components/chat/no-chat-model-note";
import { QuoteChip } from "@/components/chat/quote-chip";
import { SelectionQuote } from "@/components/chat/selection-quote";
import { Icon, Step } from "@/components/chat/step";
import { ThinkingToggle } from "@/components/chat/thinking-toggle";
import { WebSearchToggle } from "@/components/chat/web-search-toggle";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { Sidebar } from "@/components/sidebar";
import {
  attachmentsForTurn,
  type TrayAttachment,
  useAttachments,
} from "@/hooks/use-attachments";
import { useAuthSession } from "@/hooks/use-auth-session";
import { useFollowStream } from "@/hooks/use-follow-stream";
import { useQuoteJump } from "@/hooks/use-quote-jump";
import { useModelWhereabouts } from "@/hooks/use-model-whereabouts";
import { useVeilWindow } from "@/hooks/use-veil-window";
import { Boot } from "@/components/boot";
import { Welcome } from "@/components/chat/welcome";
import { useBoot } from "@/hooks/use-boot";
import { reachItems } from "@/lib/boot";
import {
  carriedFileStates,
  deleteAttachment,
  downloadAttachment,
  getAttachment,
  livenessAfterRefusal,
} from "@/lib/attachments";
import { handleAuthRequiredFrame } from "@/lib/auth";
import { stripTrailingBareMarker } from "@/lib/composer-lists";
import { carryDraft, draftStore, forgetDraft, readDraft, writeDraft } from "@/lib/drafts";
import {
  type ComposerPins,
  type DirtyAxes,
  hasPins,
  NO_DIRTY_AXES,
  NO_PINS,
  pinsDiff,
  readPins,
  revertPins,
  thinkingOn,
  webSearchOn,
} from "@/lib/composer-settings";
import {
  ApiError,
  createConversation,
  deleteConversation,
  getContextEstimate,
  getConversation,
  listConversations,
  setConversationPinned,
  updateConversation,
  updateConversationSettings,
} from "@/lib/conversations";
import {
  appendCitation,
  appendDelta,
  appendFile,
  appendNotice,
  attachConfirmation,
  beginToolStep,
  clampQuoteText,
  isWholeQuote,
  clearConfirmation,
  completeToolStep,
  dropEmptyTrailingAssistant,
  dropTrailingUserTurn,
  editQuestionAt,
  fileLiveness,
  markConfirmationDecided,
  markToolStepsApproved,
  markTrailingFailed,
  markTrailingInterrupted,
  type Message,
  recordTrailingFigures,
  replaceReplyAt,
  replaceTrailingContent,
  revertConfirmationIfUnanswered,
  settleRunningToolSteps,
  stampQuestionId,
  stampTrailingId,
  toPaneMessages,
} from "@/lib/messages";
import { modelChoice } from "@/lib/model-choice";
import { chatModels, entryCanReason, entryCanSee, entryFor, listModels } from "@/lib/models";
import { parseSSE, TURN_ACCEPTED_HEADER } from "@/lib/sse";
import { toolStatusLabel } from "@/lib/tool-labels";
import {
  clearView,
  readView,
  setTurnState,
  setView,
  signalSkyHandoff,
  signalSkyJump,
} from "@/lib/turn-state";
import { rememberConversation, takeResumedConversation } from "@/lib/resume";
import { arrivingByRoute, transitionView } from "@/lib/view-transition";
import { getWorkspaceDefaults } from "@/lib/workspace-defaults";
import type {
  CitationData,
  ContextUsageData,
  ConversationSummary,
  DoneEventData,
  FileEventPayload,
  MessageAttachment,
  MessageQuote,
  MetaEventData,
  ModelEntry,
  ToolCallData,
  ToolConfirmationData,
  ToolResultData,
} from "@/lib/types";

// A pasted block at least this long becomes a text attachment instead of
// flooding the composer; anything shorter pastes into the box as usual.
const PASTE_AS_FILE_CHARS = 4000;

// Status codes whose message must persist past the first token as a standing
// note under the answer, rather than flashing in the transient status line: a
// model substitution, the web-search outcomes that explain why an announced
// search contributed no sources (unavailable backend, or no relevant results),
// and an attached file the answer could not use. Each explains something about
// the answer being read, so it is kept ON that answer - the backend stores it
// with the message too, so it survives a reload. A transient line would be wiped
// by the next frame and the reader would never learn the file was ignored.
const STANDING_STATUS_CODES = new Set([
  "model_substituted",
  "web_search_unavailable",
  "web_search_no_results",
  "attachment_unavailable",
  "image_not_seen",
  // Tool outcomes that qualify the answer: a tool the model called could not
  // run, the tool budget was spent and the reply was forced, or an agentic turn
  // answered from its own knowledge without a live search (so anything
  // time-sensitive may be stale). tool_running is transient and omitted.
  "tool_error",
  "tool_limit_reached",
  "tools_not_used",
  // A side-effecting call the user declined at the confirmation step: the answer
  // was written without it, which is exactly the kind of thing a reader coming
  // back to the thread needs to still see. Standing on the backend too, and this
  // set has to agree with it or the live view drops a notice the reload shows -
  // it flashes in the transient line and the next frame wipes it.
  "tool_declined",
  // Stored knowledge exists but a different embedding model wrote it, so no
  // query can reach it and the answer is ungrounded - a fact about the reply
  // that has to survive the next frame.
  "vectors_stranded",
]);

/** A destructive action held up for an answer, and what it would remove.
 *
 * The two rewind kinds carry an INDEX rather than an id because that is what the
 * transcript is addressed by, plus the count the dialog quotes - both read at the
 * moment the button was pressed, which is what the user was looking at.
 */
type PendingConfirm =
  | { kind: "conversation"; id: string }
  | { kind: "attachment"; id: string; filename: string }
  | { kind: "regenerate"; index: number; count: number }
  | { kind: "edit"; index: number; count: number; text: string };

/**
 * What went wrong on the pane, and (when the stream said so) which kind.
 *
 * Message and code are ONE value rather than two states so a stale code cannot
 * outlive the failure it described: most failures here are local (a fetch that
 * threw, a save that was refused) and carry no code at all, and a leftover one
 * would dress the next of them up as something it is not.
 */
/**
 * What the pane could not do. `turn` marks an error the turn itself ended on
 * (a typed stream error, a request that died): only that kind closes the
 * reply's row and offers to answer again; a file or list error stands
 * beside the transcript without touching the last reply.
 */
type PaneError = { message: string; code?: string; detail?: string; turn?: boolean };

/**
 * What a request that died should say.
 *
 * A fetch that never reached the backend throws the browser's own words
 * ("Failed to fetch"), which name nothing a reader can act on; the pane says
 * what could not be reached and what to check. Any other failure keeps the
 * sentence it carried, since those are already written for people.
 */
function describeFailure(e: unknown): PaneError {
  if (e instanceof TypeError) {
    return {
      message: "The assistant could not be reached.",
      detail: "Check that the backend is running, then try again.",
      turn: true,
    };
  }
  return { message: e instanceof Error ? e.message : "Something went wrong.", turn: true };
}

/**
 * The stream's way of saying the model server holds nothing that can chat.
 *
 * A first-run state rather than a fault, and the only error code the thread
 * renders specially - everything else takes the generic note.
 */
const NO_CHAT_MODEL_CODE = "no_chat_model_configured";

/** How the confirmation dialog counts what a rewind is about to discard. */
function discardedBody(count: number): string {
  const turns = count === 1 ? "1 message" : `${count} messages`;
  return `The ${turns} below it will be removed. This cannot be undone.`;
}

/** Server-side cap on a conversation title. */
const TITLE_MAX_CHARS = 200;

/** What the file-delete dialog says, naming the file it is about to destroy. */
function fileDeleteBody(filename: string): string {
  return `${filename} will be removed for good, and later replies cannot use it.`;
}

/** First look for a model-written title, after the turn that seeds it ends. */
const TITLE_POLL_FIRST_MS = 2000;
/** Gap between looks while the title has not changed yet. */
const TITLE_POLL_INTERVAL_MS = 3000;
/**
 * How long the watch keeps looking before the next natural refresh takes over.
 * Sized for the contended case: a title job queued behind released GPU work
 * has been observed landing past the minute. Costs nothing in the normal
 * case - the watch stops the moment the title arrives - so the extra window
 * only ever polls when it is actually rescuing a slow title.
 */
const TITLE_POLL_WINDOW_MS = 90_000;
/**
 * How long a title reveal may wait for a surface to play it. Comfortably past
 * the longest type-out (~1.4s at the title cap): if no surface consumed the
 * reveal by then - the row sat in a rename editor, the pane was elsewhere - no
 * surface is going to, and a reveal that plays later narrates nothing.
 */
const TITLE_REVEAL_EXPIRY_MS = 4000;

/** Popovers that own the keyboard while focus is inside them, by `data-part`.
 *
 * The type-anywhere listener below has to name them: their focus sits on a
 * BUTTON, which none of its editable-element tests catch, so without the
 * exclusion a printable key inside a menu would pull focus out to the
 * composer. A popover cannot defend itself with `stopPropagation` - the App
 * Router hydrates `document` itself, so React's delegated handler and the
 * listener below are two listeners on the SAME node, and neither can stop the
 * other.
 */
const KEY_OWNING_PARTS = [
  '[data-part="theme-picker"]',
  '[data-part="convo-menu"]',
  '[data-part="model-select"]',
].join(",");

/**
 * Where a question's quote leads: its source message, when that is in the
 * thread, else why it leads nowhere (the greyed band's tooltip). The source's
 * text comes along for the click to tell a quote of the whole message from a
 * passage (`isWholeQuote`): that comparison runs on the click, not here, as
 * this runs for every quoted question on every render of the thread.
 */
function quoteTarget(
  quote: MessageQuote,
  messages: readonly Message[],
): { source: string; content: string; text: string } | { gone: string } {
  if (quote.message_id === null) {
    return { gone: "The quote has no record of the reply it was taken from." };
  }
  const source = messages.find((m) => m.id === quote.message_id);
  if (source === undefined) {
    return { gone: "The quoted reply is no longer in this conversation." };
  }
  return { source: quote.message_id, content: source.content, text: quote.text };
}

export default function ChatPage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  // What the composer's three controls hold for the OPEN conversation: the
  // model pin, the thinking pin, the web-search pin. One piece of state rather
  // than three, because they are read, written and reset together - a
  // conversation switch reloads all three, a pane reset clears all three.
  const [pins, setPins] = useState<ComposerPins>(NO_PINS);
  // The chat-role entries the model selector offers. Empty until the listing
  // lands - and for good if the model server is unreachable, in which case the
  // selector honestly offers only Default.
  const [chatCatalog, setChatCatalog] = useState<ModelEntry[]>([]);
  // Whether that listing FAILED, as opposed to simply not having landed. The
  // two look identical in the menu (an empty list) and mean opposite things -
  // "the server holds no chat model" versus "the list could not be fetched" -
  // so the menu is told which it is and says so.
  const [modelsUnavailable, setModelsUnavailable] = useState(false);
  // Whether it has ANSWERED at all, either way. The third state the menu needs:
  // only once an answer has landed is an empty list evidence of an empty
  // server, and before then the menu must not say so.
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [error, setError] = useState<PaneError | null>(null);
  // How full the model's window was on the last turn that finished; null until
  // one reports it. Only a `done` writes here, so an interrupted turn leaves
  // the previous reading standing rather than blanking the meter.
  const [contextUsage, setContextUsage] = useState<ContextUsageData | null>(null);
  // Backend progress note (e.g. a model being loaded into memory); shown in
  // the live bubble until the first real delta arrives.
  const [status, setStatus] = useState<string | null>(null);
  // Server-side conversation this pane is bound to; null for a fresh pane
  // until the first reply's done event reports the id the backend created.
  const [conversationId, setConversationId] = useState<string | null>(null);
  // Mirrored for the unmount, which remembers where the reader was.
  const conversationRef = useRef<string | null>(null);
  useEffect(() => {
    conversationRef.current = conversationId;
  }, [conversationId]);
  // The conversation the reader left for another page, opened again on the
  // way back (lib/resume.ts). Read only when a route transition carries the
  // page in: a full load has no way back to be on, and the server renders
  // none, so this never differs between the two.
  const [resuming] = useState(() => (arrivingByRoute() ? takeResumedConversation() : null));
  const [conversations, setConversations] = useState<ConversationSummary[]>([]);
  const [listError, setListError] = useState<string | null>(null);
  // Whether a list fetch has ever APPLIED a result. An empty list and a fetch
  // that has not answered yet are the same picture, so on a cold boot - where
  // the dev server compiles the route before the request is even served - the
  // panel's "No conversations yet." reads as lost data. Set once and never
  // cleared: later refetches refresh a list the user can already see, and
  // blanking it back to a loading note would be a worse lie than a stale row.
  const [listLoaded, setListLoaded] = useState(false);
  // Sidebar row whose transcript is being fetched; drives the row's loading
  // indicator so a slow load (cold dev server, busy disk) never reads as a
  // dead click. Null once the load applies, fails, or is superseded.
  const [pendingId, setPendingId] = useState<string | null>(null);
  // The main column and the composer box, for the veil's window (aurora cuts
  // the sky's shade open exactly where the composer sits).
  const mainColRef = useRef<HTMLDivElement>(null);
  const composerBoxRef = useRef<HTMLDivElement>(null);
  useVeilWindow(mainColRef, composerBoxRef);
  // Small-screen overlay state; the sidebar is always visible on md+.
  const [sidebarOpen, setSidebarOpen] = useState(false);
  // On a desktop the sidebar is a panel the toggle hides entirely; on a
  // phone it is a drawer the same toggle opens. Which is which is the
  // stylesheet's breakpoint, watched so the toggle can say which state it
  // is reporting.
  const [navCollapsed, setNavCollapsed] = useState(false);
  const [narrow, setNarrow] = useState(false);
  useEffect(() => {
    const mq = window.matchMedia("(max-width: 767px)");
    const sync = () => setNarrow(mq.matches);
    sync();
    mq.addEventListener("change", sync);
    return () => mq.removeEventListener("change", sync);
  }, []);
  // Who is signed in, asked once on load. Null - unknown, or auth switched off
  // entirely - is not an error state here: it renders the pane exactly as it
  // rendered before sign-in existed, which is the whole promise of the switch.
  const session = useAuthSession();
  // The boot screen has the window until the waits the shell blocks on are
  // over (see useBoot); the shell is mounted beneath it, hidden, so its own
  // fetches run meanwhile. The handoff into the new chat is one view
  // transition: the mark into the brand, the headline into the greeting,
  // each node into its reach item, while the sky moves the chart.
  // Carried in by a route transition (the console's Back to chat): the shell
  // renders at once so the bar and the rail travel back the way they came,
  // and the boot's waits run behind it for the reach line rather than in
  // front of it. The backend answered seconds ago; nothing is waited on.
  const [carriedIn] = useState(() => arrivingByRoute());
  const [booting, setBooting] = useState(!carriedIn);
  const boot = useBoot(booting || carriedIn);
  useEffect(() => {
    if (!booting || boot.phase !== "done") return;
    // A quick load whose waits ended before the screen was painted has no
    // chart to move and no movers to carry: the sky is not told (it eases
    // the chart in where it lives), and the shell cross-fades in whole.
    if (boot.shown) signalSkyHandoff();
    void transitionView(boot.shown ? "boot" : "shell", () => {
      setView("new");
      setBooting(false);
    });
  }, [booting, boot.phase, boot.shown]);
  const reach = useMemo(() => reachItems(boot.reach), [boot.reach]);
  // Bumped whenever a different transcript takes over the pane (new chat,
  // selection, deletion). Keys off this so switching remounts the bubbles -
  // an expanded reasoning block never leaks across transcripts - while the
  // done-event conversation adoption leaves the rendered turn untouched.
  const [paneEpoch, setPaneEpoch] = useState(0);
  // The context reading describes ONE conversation's last exchange, so it goes
  // when the pane does. Keyed on the epoch every switch already bumps rather
  // than cleared beside each of the pane's reset paths - that is a line a
  // future reset path can forget - and adjusted DURING render rather than in
  // an effect, which would paint one frame of the old conversation's number
  // under the new transcript.
  const [usageEpoch, setUsageEpoch] = useState(paneEpoch);
  if (usageEpoch !== paneEpoch) {
    setUsageEpoch(paneEpoch);
    setContextUsage(null);
  }
  // Irreversible action waiting on its dialog; null whenever none is open.
  const [pendingConfirm, setPendingConfirm] = useState<PendingConfirm | null>(null);
  // The passage the next send is replying to, if any. ONE at a time (a second
  // Reply or Quote replaces it): a chip that stacked would turn the composer
  // into a clipboard, and the backend takes one quote per turn.
  const [quote, setQuote] = useState<MessageQuote | null>(null);
  // Which question's bubble is currently an editor, addressed by its SERVER id -
  // an optimistic bubble the backend has not named yet cannot be edited, because
  // there is nothing to name in the request. Cleared by anything that takes the
  // pane over, so an editor never outlives the transcript it was opened in.
  const [editingId, setEditingId] = useState<string | null>(null);
  // Conversation whose title the topbar is editing, and the draft being typed
  // into it. Holding the ID rather than a flag makes a pane switch abandon the
  // edit on its own: the field renders only while the two still agree.
  const [renamingTitleId, setRenamingTitleId] = useState<string | null>(null);
  const [titleDraft, setTitleDraft] = useState("");
  // Escape unmounts the field, and an unmounted field fires no blur, so the
  // cancel flag is cleared when the next edit starts rather than on blur.
  const titleCancelledRef = useRef(false);
  // Titles renamed here but not yet acknowledged by the server. A list refresh
  // landing mid-flight (a stream finishing, a delete) would otherwise overwrite
  // the shown name with the stale server one and flip it back seconds later.
  const pendingTitlesRef = useRef(new Map<string, string>());
  // Sidebar pins toggled here but not yet acknowledged by the server, shielded
  // from a list refresh for the same reason as the titles above: a poll landing
  // mid-flight would drop the row back into Recent and lift it again a moment
  // later. Holds the pin STAMP (or null for unpinned), which is what the pinned
  // group is ordered by. Unrelated to the composer's pinned SETTINGS below,
  // which this file calls pins too.
  const pendingPinnedAtRef = useRef(new Map<string, string | null>());
  // Which toggle currently owns each shielded row, and the counter that mints
  // the claims. The shielded VALUE cannot answer that question: every unpin
  // writes the same `null`, so a superseded unpin would recognize a later one's
  // entry as its own and get to speak for the row again. A claim is unique per
  // toggle, so only the toggle that took it can answer for the row.
  const pinClaimRef = useRef(new Map<string, number>());
  const pinClaimSeqRef = useRef(0);
  // Conversation whose just-generated title is being typed into place - the
  // reveal the title watch triggers. Null while no reveal is playing. Pane
  // switches deliberately leave a playing reveal alone (the sidebar row
  // persists and finishes it); the expiry timer bounds every case where no
  // surface is mounted to play it at all.
  const [titleRevealId, setTitleRevealId] = useState<string | null>(null);
  const revealExpiryRef = useRef<number | null>(null);
  const endTitleReveal = useCallback(() => {
    if (revealExpiryRef.current !== null) {
      window.clearTimeout(revealExpiryRef.current);
      revealExpiryRef.current = null;
    }
    setTitleRevealId(null);
  }, []);
  // Never leave the expiry timer behind pointing at an unmounted pane.
  useEffect(
    () => () => {
      if (revealExpiryRef.current !== null) {
        window.clearTimeout(revealExpiryRef.current);
      }
    },
    [],
  );
  const scrollRef = useRef<HTMLElement>(null);
  // Following the stream (see useFollowStream): the view follows the end
  // while the reader is there and never yanks them back once they have left.
  // The hook's result is mirrored into a ref so the send paths, which are
  // memoized callbacks, can re-stick without re-creating themselves.
  const streamingRef = useRef(false);
  const follow = useFollowStream(scrollRef, streamingRef);
  const followRef = useRef(follow);
  useEffect(() => {
    followRef.current = follow;
  });
  // Following a quote to its source and back (see useQuoteJump).
  const quoteJump = useQuoteJump(scrollRef, follow, paneEpoch);
  // Where the model is (see useModelWhereabouts), readable from the stream
  // handler: a model on disk means a load before the first frame.
  const whereaboutsRef = useRef<ModelWhereabouts>("unknown");
  const refreshWhereaboutsRef = useRef<() => void>(() => {});
  // The live turn is waiting on the model to load from disk; the waiting row,
  // the topbar's status and the docked notice all say so.
  const [modelLoading, setModelLoading] = useState(false);
  const [noticeDismissed, setNoticeDismissed] = useState(false);
  // The workspace's default chat model, which an unpinned conversation runs
  // on; null until read, or when the backend cannot say. `defaultsRead` is
  // what separates the two, since only a known answer may be acted on.
  const [defaultModel, setDefaultModel] = useState<string | null>(null);
  const [defaultsRead, setDefaultsRead] = useState(false);
  useEffect(() => {
    getWorkspaceDefaults()
      .then((defaults) => setDefaultModel(defaults.model))
      // A read that failed is still an answer: there is no default to
      // suggest, and the composer should not wait on one.
      .catch(() => {})
      .finally(() => setDefaultsRead(true));
  }, []);
  const abortRef = useRef<AbortController | null>(null);
  // Set when a send starts; Stop is ignored for a short grace window so a
  // double-click on Send cannot instantly abort the stream it just started.
  const sentAtRef = useRef(0);
  // True from a send's reveal until its request is over (see send).
  const sendingRef = useRef(false);
  // Monotonic token for pane-changing actions: a conversation-detail fetch
  // only applies if no newer select/send/new-chat happened while it was
  // in flight (guards against a slow response clobbering the pane).
  const selectSeqRef = useRef(0);
  // One-shot guard for the conversation-detail warmup fetch below.
  const warmedRef = useRef(false);
  const composerRef = useRef<ComposerInputHandle>(null);
  // The draft, readable synchronously. The composer reports every edit, but the
  // state commit behind it lags a render, and a failed send has to know RIGHT
  // THEN whether the box is still empty before it hands the words back.
  const inputRef = useRef("");

  /** Put the cursor in the composer, or refuse quietly.
   *
   * Refuses on touch-primary devices (focus pops the on-screen keyboard over a
   * tap that may only have meant to read) and whenever another editable holds
   * focus (a rename, a form field) - autofocus must never steal a cursor. The
   * composer is an editable itself, so a composer that already has the cursor
   * is left exactly where it is rather than jumped to the end.
   *
   * One behaviour moved when the plain text box became an editor: focusing an
   * UNFOCUSED composer now restores the caret where it was last parked, because
   * the editor keeps its selection and only falls back to the end of the draft
   * when it holds none. The text box always jumped to the end. Continuing a
   * draft from where it was left is the better of the two, and it is the one
   * every other editor does.
   */
  const focusComposer = useCallback(() => {
    if (!window.matchMedia("(pointer: fine)").matches) return;
    const active = document.activeElement;
    if (
      active instanceof HTMLElement &&
      (active.tagName === "INPUT" ||
        active.tagName === "TEXTAREA" ||
        active.isContentEditable)
    ) {
      return;
    }
    composerRef.current?.focus();
  }, []);

  // Land ready to type: the composer is the page's primary affordance.
  useEffect(() => {
    focusComposer();
  }, [focusComposer]);

  // Abort an in-flight stream if the page unmounts mid-response.
  useEffect(() => () => abortRef.current?.abort(), []);

  // The pane's conversation id, readable from async completions in the same
  // tick it changes. A state commit (and the effect behind it) lags a
  // macrotask behind the update, but a stream's finally block or a resolved
  // DELETE runs its microtasks before that and would read yesterday's pane -
  // so every adoption writes the ref FIRST, synchronously, and the effect
  // below is only a backstop for a future write that bypasses the setter.
  const conversationIdRef = useRef<string | null>(null);
  useEffect(() => {
    conversationIdRef.current = conversationId;
  }, [conversationId]);

  /** Bind the pane to a conversation - ref synchronously, state with it. */
  const adoptConversationId = useCallback((next: string | null) => {
    conversationIdRef.current = next;
    setConversationId(next);
  }, []);

  // Whose drafts the composer keeps (lib/drafts.ts). Null until the session
  // is known, and nothing is read or kept before then: a draft shown before
  // the reader is known could be someone else's.
  const draftUserRef = useRef<string | null>(null);
  // The conversation load in flight, by its sequence number; 0 when none.
  // Its own completion opens the draft of whichever pane it lands on, so
  // the session arriving mid-load must not open one for the pane being left.
  const loadSeqRef = useRef(0);

  /** Take the draft the composer just reported, and keep it for this pane. */
  const onDraftChange = useCallback((markdown: string) => {
    inputRef.current = markdown;
    setInput(markdown);
    const store = draftStore();
    const user = draftUserRef.current;
    if (store !== null && user !== null) {
      writeDraft(store, user, conversationIdRef.current, markdown);
    }
  }, []);

  /**
   * The pane now shows another conversation: its composer takes the draft
   * left there (or empties). The words being replaced were kept under their
   * own conversation as they were typed, so nothing is lost by the swap.
   * The edit comes back through onDraftChange like any other, which is what
   * refreshes the page's own reading of the text.
   */
  const openDraft = useCallback((id: string | null) => {
    const store = draftStore();
    const user = draftUserRef.current;
    if (store === null || user === null) return;
    composerRef.current?.setMarkdown(readDraft(store, user, id));
  }, []);

  /** A conversation is gone: so is the draft that was waiting in it. */
  const dropDraft = useCallback((id: string) => {
    const store = draftStore();
    const user = draftUserRef.current;
    if (store !== null && user !== null) forgetDraft(store, user, id);
  }, []);

  /**
   * Bind the pane to a conversation WITHOUT changing what its composer
   * holds: the pane gained its conversation (a first send, an attachment),
   * or lost one that turned out to be gone. The draft follows the pane, so
   * words typed while the id was still unknown are not left behind under
   * the placeholder to turn up in the next new chat.
   */
  const bindConversationId = useCallback(
    (next: string | null) => {
      const prev = conversationIdRef.current;
      adoptConversationId(next);
      const store = draftStore();
      const user = draftUserRef.current;
      if (store !== null && user !== null) {
        carryDraft(store, user, prev, next, inputRef.current);
      }
    },
    [adoptConversationId],
  );

  // The session says whose drafts these are. When it lands, an empty
  // composer takes the draft left on this pane; one already typed into is
  // kept as it stands, the newer words winning over the stored ones.
  const draftUser = session?.user_id ?? null;
  useEffect(() => {
    draftUserRef.current = draftUser;
    const store = draftStore();
    if (store === null || draftUser === null) return;
    if (inputRef.current !== "") {
      writeDraft(store, draftUser, conversationIdRef.current, inputRef.current);
    } else if (loadSeqRef.current === 0 || loadSeqRef.current !== selectSeqRef.current) {
      openDraft(conversationIdRef.current);
    }
  }, [draftUser, openDraft]);

  // Pins chosen on a pane that has no conversation yet. The controls work
  // before the first turn - picking a model IS how the first turn is aimed -
  // so the choice waits here until the backend names the thread, and the
  // effect below writes it the moment one exists.
  const pendingPinsRef = useRef<ComposerPins | null>(null);

  // The model pin the SERVER is known to hold, and the conversation it was
  // read from. What it gates is whether a turn carries `model` as a request
  // FIELD, and that is a routing decision, not a cosmetic one: an explicit
  // request model is honoured verbatim by the backend, which short-circuits
  // both halves of its resolution. Pair-routing goes first - a thinking turn
  // would stay on the non-thinking tag, so the Thinking control would silently
  // stop working - and the stale-tag fallthrough goes with it: a stored tag
  // that has since been deleted falls through to the next level of the
  // resolution chain when it is READ from the conversation, but fails the turn
  // loudly when it arrives as a request field.
  //
  // So the field is used ONLY while the conversation cannot answer for the pin
  // itself: a fresh pane whose thread does not exist yet, or a save still in
  // flight or refused. Once the pin is stored, the turn says nothing about the
  // model and the resolution chain reads it, routing and fallthrough intact.
  //
  // Only this axis is tracked. Thinking and web search ride on every request by
  // design (an unpinned axis sends "auto", which defers to what is stored), and
  // neither can land the turn on different weights.
  const storedModelRef = useRef<{ conversationId: string | null; model: string | null }>({
    conversationId: null,
    model: null,
  });
  // Axes whose stored value the composer's controls cannot express (see
  // readPins). Belongs to the open conversation, and is cleared by the write
  // that clears the stored value itself.
  const dirtyPinsRef = useRef<DirtyAxes>(NO_DIRTY_AXES);

  /** Forget what the server holds - a pane reset, or a thread that went away. */
  const forgetStoredPins = useCallback(() => {
    storedModelRef.current = { conversationId: null, model: null };
    dirtyPinsRef.current = NO_DIRTY_AXES;
  }, []);

  /**
   * The model tag this turn must carry itself, or null to let settings decide.
   *
   * Null is the ordinary answer - either nothing is pinned, or the pin is
   * already stored on the conversation, which is the case that must NOT send
   * the field (see storedModelRef).
   *
   * A pin being CLEARED is null too, and unavoidably so: the request has no way
   * to say "no model", only to name one, so a turn sent in the gap before the
   * clear lands still resolves from the tag the conversation is still storing.
   */
  const unsavedModel = useCallback(
    (id: string | null): string | null => {
      if (pins.model === null) return null;
      const stored = storedModelRef.current;
      const persisted =
        id !== null && stored.conversationId === id && stored.model === pins.model;
      return persisted ? null : pins.model;
    },
    [pins.model],
  );

  /**
   * Apply a composer pin: optimistically, then persisted, then reverted if the
   * server refuses.
   *
   * The control has to move under the finger, so the state goes first - but a
   * control showing a setting the server never accepted is a lie the next turn
   * would expose, so a refusal puts it back and says which axis failed. Only
   * the changed axis is sent, and only the changed axis is put back: the three
   * save independently, so one being refused must not disturb a switch the
   * user flipped while it was in flight. A refusal that arrives after the pane
   * has moved to another conversation is dropped entirely - those pins belong
   * to a thread this one knows nothing about.
   */
  const changePins = useCallback(
    (next: ComposerPins, axis: string) => {
      const previous = pins;
      setPins(next);
      const id = conversationIdRef.current;
      if (id === null) {
        pendingPinsRef.current = next;
        return;
      }
      const patch = pinsDiff(previous, next, dirtyPinsRef.current);
      void updateConversationSettings(id, patch)
        .then(() => {
          // What the server holds for a thread the pane has left is not this
          // pane's business - the pins on show belong to another conversation
          // entirely, and recording this one against them would have the next
          // turn resolve its model from the wrong thread.
          if (conversationIdRef.current !== id) return;
          // Every dirty axis is named by the patch (see pinsDiff), so a write
          // that landed has cleared all of them.
          dirtyPinsRef.current = NO_DIRTY_AXES;
          // Recorded only when this write actually carried the model. A
          // thinking-only save landing while a model save is still in flight
          // says nothing about the model, and taking it as proof would drop the
          // request field while the server still holds the OLD tag - the next
          // turn would quietly run on a model the composer is not showing.
          if ("model" in patch) {
            storedModelRef.current = { conversationId: id, model: next.model };
          }
        })
        .catch((e: unknown) => {
          setPins((current) =>
            conversationIdRef.current === id ? revertPins(current, next, previous) : current,
          );
          // The same reading as the revert beside it: a refusal that arrives
          // after the pane has moved on belongs to a thread this one knows
          // nothing about, so its banner would sit above an unrelated
          // transcript blaming it for a control the user cannot see.
          if (conversationIdRef.current !== id) return;
          const reason = e instanceof Error ? ` ${e.message}` : "";
          setError({
            message: `Couldn't change ${axis} for this conversation.${reason}`,
          });
        });
    },
    [pins],
  );

  // The workspace default is a suggestion: when it names nothing the server
  // lists (a tag since removed, or none set), the first listed model stands
  // in for a FRESH conversation and is pinned, so the composer names the
  // model that will answer (lib/model-choice.ts). Fresh only: the pin then
  // travels with the create and no settings write is sent, so a server that
  // refuses has nothing to refuse and a stored thread is never rewritten.
  // Only once both the listing and the defaults have answered.
  useEffect(() => {
    if (conversationId !== null || !modelsLoaded || modelsUnavailable || !defaultsRead) return;
    if (pins.model !== null) return;
    const { standIn } = modelChoice(null, defaultModel, chatCatalog.map((m) => m.model), true);
    if (standIn === null) return;
    // A tick later: a pin is a change of settings, not part of the render
    // that noticed the gap.
    const t = window.setTimeout(() => changePins({ ...pins, model: standIn }, "the model"), 0);
    return () => window.clearTimeout(t);
  }, [conversationId, modelsLoaded, modelsUnavailable, defaultsRead, defaultModel, chatCatalog, pins, changePins]);

  // The current reading, readable from async completions without joining the
  // effect's dependencies - depending on it would make the estimate's own
  // write refire the fetch in a loop.
  const contextUsageRef = useRef<ContextUsageData | null>(null);
  useEffect(() => {
    contextUsageRef.current = contextUsage;
  }, [contextUsage]);
  // Newest estimate request wins; anything slower is a stale answer.
  const estimateSeqRef = useRef(0);

  // The eager pre-turn reading: on pane load and on a model switch, ask the
  // backend what the next prompt already opens with (system prompt + stored
  // history over the model's window) instead of leaving the meter blank until
  // a turn happens to report. This is what makes the ring answer "is it still
  // reading all of this?" the moment a long thread is REOPENED - the moment
  // the question is actually asked - and lets the number move when a switch
  // aims the next turn at a model with a different window.
  //
  // A completed turn's own reading outranks the estimate: it measured the
  // prompt actually sent, retrieval and tool schemas included, where the
  // estimate is a floor. So a measured reading stands - and no request is
  // made - unless the PICK has moved out from under it: unpinned, whatever
  // the turn ran IS the resolution in force (the done frame's tag survives
  // routing and substitution, which the composer cannot second-guess); pinned,
  // the reading may honestly name either the pin or its declared thinking
  // sibling, since pair-routing sends thinking turns to the sibling without
  // the pin changing. Comparing against both is what keeps a thinking turn's
  // measured number from being demoted to the floor seconds after it lands.
  // Mid-stream the effect stands down entirely: the turn is about to report
  // the real number.
  useEffect(() => {
    // Retired FIRST, before any guard: every dependency change - including
    // the ones that stand down - must orphan whatever request is still in
    // flight, or a stale answer for yesterday's pick lands through today's
    // checks and repaints the meter with a reading nothing on screen asked
    // for.
    const seq = ++estimateSeqRef.current;
    if (conversationId === null || streaming) return;
    const held = contextUsageRef.current;
    if (held !== null && !held.estimated) {
      if (pins.model === null || held.model === pins.model) return;
      const sibling = chatCatalog.find((entry) => entry.model === pins.model)
        ?.capabilities.thinking_model;
      if (typeof sibling === "string" && held.model === sibling) return;
    }
    const id = conversationId;
    void getContextEstimate(id, pins.model)
      .then((estimate) => {
        if (seq !== estimateSeqRef.current) return;
        if (conversationIdRef.current !== id) return;
        const current = contextUsageRef.current;
        if (current !== null && !current.estimated && current.model === estimate.model) {
          return;
        }
        setContextUsage({ ...estimate, estimated: true });
      })
      .catch(() => {
        // The meter is a nicety: a failed estimate leaves whatever is showing
        // (usually nothing) rather than raising a banner over the transcript.
      });
  }, [chatCatalog, conversationId, paneEpoch, pins.model, streaming]);

  // Write the pins a fresh pane collected, once the conversation exists. Every
  // stream names its thread on the FIRST frame (meta), and an attachment
  // upload mints one earlier still, so this normally lands well before the
  // answer does. Nothing depends on it winning a race: the turn already
  // carried the same choices as request fields, so this only makes them the
  // conversation's standing defaults.
  useEffect(() => {
    if (conversationId === null) return;
    const pending = pendingPinsRef.current;
    if (pending === null) return;
    pendingPinsRef.current = null;
    // Nothing was actually chosen (or it was chosen and undone): a brand-new
    // conversation stores nothing, so there is nothing to clear.
    if (!hasPins(pending)) return;
    void updateConversationSettings(conversationId, pinsDiff(NO_PINS, pending))
      .then(() => {
        if (conversationIdRef.current !== conversationId) return;
        // The conversation answers for the model pin from here on, so turns
        // stop carrying it themselves (see storedModelRef).
        storedModelRef.current = { conversationId, model: pending.model };
      })
      .catch((e: unknown) => {
        setPins((current) =>
          conversationIdRef.current === conversationId
            ? revertPins(current, pending, NO_PINS)
            : current,
        );
        // Dropped with the revert when the pane has moved on: the settings that
        // failed to save are another conversation's (see changePins).
        if (conversationIdRef.current !== conversationId) return;
        const reason = e instanceof Error ? ` ${e.message}` : "";
        setError({
          message: `Couldn't save the composer settings for this conversation.${reason}`,
        });
      });
  }, [conversationId]);

  // The chat models the selector offers, fetched once per page load. A failure
  // never blocks chatting - the selector degrades to Default, which is exactly
  // what an unpinned conversation already runs on - but it is not swallowed
  // either: the menu carries a line saying the list could not be fetched, so an
  // unreachable model server does not read as a server holding nothing.
  useEffect(() => {
    void listModels()
      .then((listing) => setChatCatalog(chatModels(listing)))
      .catch(() => setModelsUnavailable(true))
      // Either way the question has been answered, which is the fact the menu
      // needs before an empty list may be read as an empty server.
      .finally(() => setModelsLoaded(true));
  }, []);

  // Every list fetch takes a ticket and only the newest applies. Without the
  // ordering, a slow request started before a delete or a create resolves
  // last and puts a vanished row back - or hides a brand-new one.
  const listSeqRef = useRef(0);

  // Fetches, applies, and RETURNS the list (null on failure or staleness), so
  // the title poll below can read what actually came back. setState only after
  // the await, which keeps it out of the synchronous call graph the
  // set-state-in-effect lint rule rejects for the mount-time call below.
  const fetchConversations = useCallback(async () => {
    const seq = ++listSeqRef.current;
    try {
      const list = await listConversations();
      if (seq !== listSeqRef.current) return null;
      // A rename or a pin still in flight keeps what the user asked for;
      // everything else takes the server's word.
      const pendingTitles = pendingTitlesRef.current;
      const pendingPins = pendingPinnedAtRef.current;
      setConversations(
        pendingTitles.size === 0 && pendingPins.size === 0
          ? list.items
          : list.items.map((c) => {
              const title = pendingTitles.get(c.id);
              const pinnedAt = pendingPins.get(c.id);
              return {
                ...c,
                ...(title === undefined ? {} : { title }),
                ...(pinnedAt === undefined ? {} : { pinned_at: pinnedAt }),
              };
            }),
      );
      setListError(null);
      setListLoaded(true);
      // Warm the conversation-detail path once per page load: the dev
      // server compiles each route on its first request (and the backend
      // opens its connection pool lazily), so on a fresh boot the first
      // sidebar click would otherwise stall for the compile. Fetching the
      // most recent conversation in the background - result discarded -
      // pays that cost before any human can click.
      if (!warmedRef.current && list.items.length > 0) {
        warmedRef.current = true;
        getConversation(list.items[0].id).catch(() => {});
      }
      return list.items;
    } catch {
      // A dead list endpoint must not block chatting; note it in the panel -
      // unless a newer fetch owns the list, in which case its verdict stands.
      if (seq === listSeqRef.current) {
        setListError("Couldn't load conversations.");
        setListLoaded(true);
      }
      return null;
    }
  }, []);

  const refreshConversations = useCallback(() => {
    void fetchConversations();
  }, [fetchConversations]);

  // The backend names a new conversation twice: instantly, from a truncation of
  // the first message, and then properly - a background job asks the model for a
  // real title once the first exchange is complete. That job finishes AFTER the
  // stream closes and there is no push channel for it, so the sidebar watches
  // for the rename instead: a short poll, anchored on the end of generation,
  // that stops the moment the title changes from its baseline. Bounded by a
  // wall-clock window - a fast title shows within seconds, a slow one (the job
  // queues behind a GPU a long tool turn just released) is caught by a later
  // look, and past the window the next natural refresh takes over.
  //
  // The pending timer, plus a generation counter: a cancelled or superseded
  // poll can have a fetch mid-flight, and that fetch must not reschedule
  // itself into a poll that no longer exists.
  const titlePollTimerRef = useRef<number | null>(null);
  const titlePollGenRef = useRef(0);
  // The row the watch is anchored to while it runs, so a manual rename of
  // exactly that row - and no other - can call the watch off (handleRename).
  // Nulled when the watch ends, whichever way it ends.
  const titleWatchIdRef = useRef<string | null>(null);

  const cancelTitleRefreshes = useCallback(() => {
    titlePollGenRef.current += 1;
    titleWatchIdRef.current = null;
    if (titlePollTimerRef.current !== null) {
      clearTimeout(titlePollTimerRef.current);
      titlePollTimerRef.current = null;
    }
  }, []);

  const scheduleTitleRefreshes = useCallback(
    (id: string | null, anchor: ConversationSummary[] | null) => {
      cancelTitleRefreshes();
      // No conversation was created (the turn failed before the backend made
      // one): nothing exists to be titled.
      if (id === null) return;
      titleWatchIdRef.current = id;
      const gen = titlePollGenRef.current;
      const deadline = Date.now() + TITLE_POLL_WINDOW_MS;
      // The baseline comes from the anchor fetch when it saw the row: that
      // list was requested in the same breath the title job was enqueued, so
      // it still carries the instant fallback - a title job fast enough to
      // beat the FIRST look can therefore never be mistaken for the baseline
      // and watched for a change that already happened. If the anchor missed
      // the row, the first look that sees it records the baseline instead.
      let baseline: string | null | undefined;
      const anchored = anchor?.find((c) => c.id === id);
      if (anchored !== undefined) baseline = anchored.title ?? null;
      const look = () => {
        void fetchConversations().then((items) => {
          if (gen !== titlePollGenRef.current) return;
          // Only a fetch that really returned a list can move the judgement
          // (a failed or superseded one says nothing): a change from the
          // baseline - the model's title, or the row gone entirely - means
          // the watch is over.
          if (items !== null) {
            const title = items.find((c) => c.id === id)?.title ?? null;
            if (baseline === undefined) {
              baseline = title;
            } else if (title !== baseline) {
              // The generated name is in: type it into place (sidebar row,
              // and the topbar if this conversation is open) rather than
              // snapping. A vanished row reads as a change too - a deletion
              // ends the watch with nothing left to animate.
              if (title !== null) {
                setTitleRevealId(id);
                if (revealExpiryRef.current !== null) {
                  window.clearTimeout(revealExpiryRef.current);
                }
                revealExpiryRef.current = window.setTimeout(
                  () => setTitleRevealId(null),
                  TITLE_REVEAL_EXPIRY_MS,
                );
              }
              titleWatchIdRef.current = null;
              return;
            }
          }
          if (Date.now() >= deadline) {
            titleWatchIdRef.current = null;
            return;
          }
          titlePollTimerRef.current = window.setTimeout(look, TITLE_POLL_INTERVAL_MS);
        });
      };
      titlePollTimerRef.current = window.setTimeout(look, TITLE_POLL_FIRST_MS);
    },
    [cancelTitleRefreshes, fetchConversations],
  );

  // Never leave a timer behind pointing at an unmounted pane.
  useEffect(() => cancelTitleRefreshes, [cancelTitleRefreshes]);

  // De-dupes concurrent conversation creations: while a create is in flight,
  // every caller awaits the same promise, so a tray of files lands on one thread.
  const createConvRef = useRef<Promise<string> | null>(null);

  // Resolve the conversation to attach uploads to, creating one on first use.
  // A fresh pane has no conversation until its first reply, but an attachment
  // needs one sooner (the upload must name it), so this mints it up front and
  // binds the pane to it. Idempotent: returns the bound id once it exists.
  const ensureConversationId = useCallback(async (): Promise<string> => {
    if (conversationId) return conversationId;
    if (createConvRef.current) return createConvRef.current;
    // Snapshot the pane generation: if a New chat or conversation switch happens
    // while the create is in flight (a slow round-trip), that action bumps
    // selectSeqRef, and this must NOT bind the now-abandoned pane to the
    // conversation it created (the upload still lands on it; the pane moves on).
    const seq = selectSeqRef.current;
    const pending = (async () => {
      const summary = await createConversation();
      if (selectSeqRef.current === seq) {
        bindConversationId(summary.id);
        // Surface the new (title-less) row in the sidebar right away.
        refreshConversations();
      }
      return summary.id;
    })();
    createConvRef.current = pending;
    try {
      return await pending;
    } finally {
      // Only clear if still ours: a pane reset may have detached this ref and a
      // newer create installed its own promise in the meantime.
      if (createConvRef.current === pending) createConvRef.current = null;
    }
  }, [bindConversationId, conversationId, refreshConversations]);

  const tray = useAttachments(ensureConversationId);
  const {
    clear: clearTray,
    drop: dropTray,
    restore: restoreTray,
    seed: seedTray,
    remove: removeFile,
  } = tray;
  // Mirror the tray into a ref so send() can read it at click time without
  // depending on a per-render array (which would rebuild send() and its whole
  // callback graph every render).
  const trayRef = useRef<TrayAttachment[]>([]);
  useEffect(() => {
    trayRef.current = tray.attachments;
  }, [tray.attachments]);
  // Mirrored for the same reason: removing a tray chip has to know whether an
  // earlier question in this thread already carries that file (see
  // removeAttachment), without rebuilding the callback on every new message.
  const messagesRef = useRef<Message[]>([]);
  useEffect(() => {
    messagesRef.current = messages;
  }, [messages]);
  // Whether the tray holds anything the next turn could actually carry. A file
  // is a question on its own, so this is half of what enables the Send button -
  // but only files that would really go: a chip that failed to upload is still
  // sitting there, and it must not make an empty composer look sendable.
  const hasFilesToSend = useMemo(
    () => attachmentsForTurn(tray.attachments).length > 0,
    [tray.attachments],
  );

  // Which of this conversation's files still exist, IN WHAT STATE, and which are
  // being deleted. A message's chips are permanent, but the files behind them
  // are not: a chip has to say whether its file is still stored (and so still
  // usable, still fetchable, still deletable), whether it aged out of retention,
  // or whether only the record of it remains. The status is what tells those
  // three apart - a bare set of live ids can only ever say the last one - and it
  // is what decides whether a chip offers a download at all. The conversation
  // detail reports it on open; a send adds the files it just took, and a delete
  // removes what it took away.
  const [liveFiles, setLiveFiles] = useState<LiveFiles>(new Map());
  const [deletingFiles, setDeletingFiles] = useState<ReadonlySet<string>>(new Set());
  // Ids whose download is in flight. Its own set rather than a flag: a reply can
  // produce several files, and clicking one must not make the others look busy.
  const [fetchingFiles, setFetchingFiles] = useState<ReadonlySet<string>>(new Set());

  // Drag-and-drop highlight for the composer drop zone.
  const [dragActive, setDragActive] = useState(false);
  useEffect(() => {
    streamingRef.current = streaming;
  }, [streaming]);
  // The load is only ever shown while its turn is in flight: a turn that ends
  // before its first frame (Stop, a dropped request, an early error) must not
  // leave the load standing in the topbar, the notice and the next waiting row.
  const loadActive = modelLoading && streaming;

  useEffect(() => {
    refreshConversations();
    // Compile-warm the send route once per page load: the dev server builds
    // each route on its first request, so without this the first send of a
    // boot stalls for the compile. Deliberately NOT tied to the conversation
    // list succeeding - compiling the route does not need the backend at all
    // (the proxy buffers an upstream failure into an error frame). A live
    // backend rejects the empty body during validation (422) without
    // creating a conversation, writing to the DB, or touching the model.
    fetch("/api/chat", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "{}",
    }).catch(() => {});
  }, [refreshConversations]);

  /** Glide the transcript to its end (a cut under reduced motion) and
   * re-arm following; the sky streaks its stars with it. */
  const jumpToBottom = useCallback(() => {
    followRef.current.jump();
    signalSkyJump();
  }, []);

  // Consume an SSE chat stream, dispatching each event into the TRAILING assistant
  // message. Shared by an initial send and a confirmation resume: both stream into
  // the same trailing bubble, so the append helpers (which all key off the last
  // message) write the resume's answer into the very bubble that showed the
  // pending action. Returns when the stream ends - a `done`, a `tool_confirmation`
  // suspend (terminal, no `done` follows), or the body simply closing.
  const consumeStream = useCallback(
    async (body: ReadableStream<Uint8Array>) => {
      // The backend narrates tool progress in prose that names the raw tool
      // identifier ("Running web_search..."). Rather than parse that prose, the
      // frames around it are tracked: a tool_call opens the running phase and
      // its tool_result hands over to the reading phase, so the line can be
      // reworded from the same labels the trace card uses. Per stream, because
      // the tracking means nothing once this one ends.
      //
      // SINGLE USE: each frame arms exactly one rewording, and the tool_running
      // frame that consumes it disarms it again. The two are adjacent today, but
      // this file must not silently depend on that: an unpaired tool_running
      // frame (a reorder, a new emission site, a batched round) would otherwise
      // be labelled confidently with the PREVIOUS tool's copy, and a wrong label
      // is worse than the backend's own prose, which is what falling through
      // gets. Disarmed, anything unexpected reads verbatim.
      let activeTool: { name: string; phase: "running" | "reading" } | null = null;
      // Reaching a model that is on disk means a load first: the sky goes
      // amber, the waiting row and the docked notice say so, until the first
      // frame of the model's own work arrives.
      if (whereaboutsRef.current === "disk") {
        setModelLoading(true);
        setNoticeDismissed(false);
        setTurnState("loading");
      }
      let firstFrame = true;
      const arrived = () => {
        if (!firstFrame) return;
        firstFrame = false;
        setModelLoading(false);
        refreshWhereaboutsRef.current();
      };
      for await (const msg of parseSSE(body)) {
        if (msg.event === "token") {
          const { delta } = JSON.parse(msg.data) as { delta: string };
          arrived();
          setStatus(null);
          setTurnState("streaming");
          setMessages((m) => appendDelta(m, "content", delta));
        } else if (msg.event === "thinking") {
          const { delta } = JSON.parse(msg.data) as { delta: string };
          arrived();
          setStatus(null);
          setTurnState("thinking");
          setMessages((m) => appendDelta(m, "thinking", delta));
        } else if (msg.event === "citation") {
          // Citations arrive after the answer tokens (the backend emits only the
          // sources the reply actually referenced, known once generation ends);
          // appending is position-independent, so nothing assumes ordering here
          // beyond "before done".
          const citation = JSON.parse(msg.data) as CitationData;
          setMessages((m) => appendCitation(m, citation));
        } else if (msg.event === "status") {
          const { message, code } = JSON.parse(msg.data) as {
            message: string;
            code?: string;
          };
          // Standing notes (model substitution, web-search outcome, a file the
          // answer could not use, a declined action) explain something about THIS
          // answer, so they are pinned to it rather than to the pane: they must
          // survive later status frames, the first token, and a reload (the
          // backend stores them on the message). Everything else is transient.
          if (code && STANDING_STATUS_CODES.has(code)) {
            setMessages((m) => appendNotice(m, { code, message }));
          } else if (code === "tool_running" && activeTool) {
            setStatus(toolStatusLabel(activeTool.name, activeTool.phase));
            setTurnState("tool");
            activeTool = null;
          } else {
            setStatus(message);
          }
        } else if (msg.event === "tool_call") {
          // The model asked to run a tool: open a step for the activity trace,
          // folding any narration streamed just before it onto the step. Clears
          // the transient status so the step's own "running" state carries the
          // wait.
          const call = JSON.parse(msg.data) as ToolCallData;
          arrived();
          activeTool = { name: call.name, phase: "running" };
          setStatus(null);
          setTurnState("tool");
          setMessages((m) => beginToolStep(m, call));
        } else if (msg.event === "tool_result") {
          // The tool finished: close its step with a short summary of what came
          // back (the model already has the full result).
          const result = JSON.parse(msg.data) as ToolResultData;
          activeTool = { name: result.name, phase: "reading" };
          setMessages((m) => completeToolStep(m, result));
        } else if (msg.event === "file") {
          // A tool wrote a file. The event arrives after its row is committed,
          // so the id already names something the download route can serve -
          // which is why the chip goes up immediately and counts as live rather
          // than waiting for a reload to confirm it.
          const file = JSON.parse(msg.data) as FileEventPayload;
          setMessages((m) => appendFile(m, file));
          // Born ready: a generated file has no parse job and no ingest, so the
          // status it is committed with is the one it keeps until retention
          // takes it.
          setLiveFiles((live) => new Map(live).set(file.attachment_id, "ready"));
        } else if (msg.event === "tool_confirmation") {
          // The model asked for a side-effecting tool; the turn is SUSPENDED, not
          // finished. Pin the pending action to the bubble (which shows the
          // approve/deny prompt) and adopt the conversation it belongs to - a
          // fresh pane has none yet, and the resume must name the thread. This
          // event is terminal: no `done` follows, so the loop just ends after it.
          const confirmation = JSON.parse(msg.data) as ToolConfirmationData;
          setTurnState("confirm");
          setMessages((m) => attachConfirmation(m, confirmation));
          // The ref holds the latest adopted value (write-through), so it can
          // stand in for the functional-updater form here.
          bindConversationId(conversationIdRef.current ?? confirmation.conversation_id);
        } else if (msg.event === "error") {
          const { message, code } = JSON.parse(msg.data) as {
            message: string;
            code?: string;
          };
          // The code rides along with the message: the thread renders one of
          // them (the first-run "nothing to chat with" state) as its own
          // notice, and every other code takes the generic note unchanged.
          setError({ message, code, turn: true });
          setMessages((m) => markTrailingFailed(m));
          setTurnState("error");
          // auth_required: the session lapsed mid-conversation. The REST
          // clients reach the same handler through `raise`; this path cannot,
          // because the chat proxy rewrites a pre-stream refusal into an
          // in-band frame - so without this the turn would end on a banner
          // and a dead composer instead of the sign-in page.
          handleAuthRequiredFrame(code);
          // not_found: the pane is bound to a conversation deleted elsewhere;
          // unbind so the next send starts fresh instead of failing again.
          if (code === "not_found") {
            bindConversationId(null);
            // The only path that moves the binding WITHOUT bumping paneEpoch -
            // the transcript deliberately stays on screen, so a bump would
            // remount it and collapse every expanded reasoning block. The
            // reading has to go by hand instead: the next send opens a brand
            // new conversation, and the dead one's fill would sit in the
            // composer until that turn finished.
            setContextUsage(null);
          }
        } else if (msg.event === "meta") {
          // The stream names its conversation up front, so a turn that never
          // reaches `done` - a stop press, a dropped connection - still binds
          // the pane: the topbar can name the thread, the title watch can arm
          // from the end-of-turn refresh, and the next send appends to this
          // conversation instead of forking a fresh one. Same functional
          // guard as `done`: a mid-stream "New chat" click wins.
          const meta = JSON.parse(msg.data) as MetaEventData;
          if (meta.conversation_id) {
            const id = meta.conversation_id;
            bindConversationId(conversationIdRef.current ?? id);
          }
          // The frame also names the question this turn is answering, so a
          // just-sent one can be edited straight away rather than only after a
          // reload - an edit has to name the row it rewrites. Optional on the
          // wire (an older backend sends no such field), so its absence simply
          // leaves the question anonymous until `done`, as before.
          if (meta.user_message_id) {
            const questionId = meta.user_message_id;
            setMessages((m) => stampQuestionId(m, questionId));
          }
        } else if (msg.event === "done") {
          const done = JSON.parse(msg.data) as DoneEventData;
          arrived();
          setTurnState("done");
          // The footer's figures: the model from the context reading, the
          // count and the writing time from the turn's usage - the object
          // the server stored, so a reload reads the same.
          setMessages((m) =>
            recordTrailingFigures(m, {
              model: done.context_usage?.model ?? null,
              usage: done.turn_usage ?? null,
              doneAt: Date.now(),
            }),
          );
          // The server sends the final answer only when it rewrote the streamed
          // text (inline citation markers renumbered to match the emitted
          // citations); swap the streamed copy for it.
          if (typeof done.content === "string" && done.content !== "") {
            const content = done.content;
            setMessages((m) => replaceTrailingContent(m, content));
          }
          // The reply now has a server-side identity; record it on the bubble
          // that streamed it, so a live turn and a reloaded one hold the same
          // shape. Empty on the proxy's synthetic frame (a turn the backend
          // never took), which is the case worth skipping - the backend's own
          // frame always carries an id, falling back to a fresh uuid when a
          // turn persisted no row, so this is not a "was it saved" test.
          if (done.message_id) {
            const id = done.message_id;
            setMessages((m) => stampTrailingId(m, id, new Date().toISOString()));
          }
          // First turn of a fresh pane: adopt the conversation the backend
          // created so subsequent turns append to it. The functional guard keeps a
          // mid-stream "New chat" click from being overridden.
          if (done.conversation_id) {
            const id = done.conversation_id;
            bindConversationId(conversationIdRef.current ?? id);
          }
          // How full the window was on the prompt this turn sent. Absent on an
          // older backend and null when the window is unknown; either way the
          // previous reading stands rather than being blanked by a turn that
          // simply had nothing to say about it.
          if (done.context_usage) {
            setContextUsage(done.context_usage);
          }
          break;
        }
        // Any other event type is a future protocol addition: ignore it.
      }
    },
    [bindConversationId],
  );

  // The trailing turn is paused on a side-effecting confirmation, waiting for an
  // approve/deny. While it is, the composer is blocked and a new send is refused:
  // the paused turn owns the trailing bubble the resume streams into, and starting
  // a fresh turn would both strand that bubble and (backend-side) supersede the
  // pending confirmation.
  const lastMessage = messages[messages.length - 1];
  const awaitingConfirmation =
    lastMessage?.role === "assistant" &&
    lastMessage.confirmation != null &&
    lastMessage.confirmationDecision == null;

  // The turn state the themes key on (<body data-state>, see lib/turn-state).
  // The stream handlers write the live states as frames arrive; this settles
  // the state once nothing is streaming: a paused turn holds "confirm", a
  // standing error holds "error", and anything else rests. Idle and done share
  // one look (done's pulse fires on the transition, which the handler made),
  // so a settled "done" is left alone rather than flipped to idle.
  useEffect(() => {
    // The boot writes its own waits' weather until the shell shows.
    if (booting) return;
    if (awaitingConfirmation) setTurnState("confirm");
    else if (streaming) return;
    else if (error !== null) setTurnState("error");
    else if (document.body.dataset.state !== "done") setTurnState("idle");
  }, [booting, awaitingConfirmation, streaming, error]);
  // The state belongs to this page: leaving it (for the console, say) with an
  // error standing would carry the error's sky to a page that has none.
  useEffect(() => () => setTurnState("idle"), []);

  // Hand focus back the moment a confirmation is answered: the composer was
  // disabled while the card waited, so the focus must chase the re-enable (an
  // effect on the derived flag), not the click itself.
  const wasAwaitingRef = useRef(false);
  useEffect(() => {
    if (wasAwaitingRef.current && !awaitingConfirmation) focusComposer();
    wasAwaitingRef.current = awaitingConfirmation;
  }, [awaitingConfirmation, focusComposer]);

  // Type-anywhere: a printable keystroke with nothing else focused belongs in
  // the composer. Shortcuts, IME composition, and menu keyboard flows (focus
  // inside them blocks the steal) all pass through untouched.
  //
  // The character is inserted EXPLICITLY, which the plain text box did not
  // need: focusing a rich editable mid-keydown does not reliably route the
  // browser's own insertion into it, and a keystroke that focused the box and
  // then vanished is worse than one that never focused it.
  useEffect(() => {
    const onGlobalKeyDown = (event: KeyboardEvent) => {
      if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey) return;
      if (event.key.length !== 1) return;
      if (sidebarOpen) return;
      // A modal is answered before anything else on the page, and the composer
      // it would steal to is behind the scrim. Keyed on the state rather than
      // on where focus sits, because clicking the panel's own text parks focus
      // on the body, which every test below waves through.
      if (pendingConfirm !== null) return;
      // The composer is blocked while a side-effecting action waits: a
      // keystroke must not be swallowed into a box that cannot take it.
      if (awaitingConfirmation) return;
      if (!window.matchMedia("(pointer: fine)").matches) return;
      const active = document.activeElement;
      if (
        active instanceof HTMLElement &&
        active !== document.body &&
        (active.tagName === "INPUT" ||
          active.tagName === "TEXTAREA" ||
          active.tagName === "SELECT" ||
          active.isContentEditable ||
          active.closest(KEY_OWNING_PARTS) !== null)
      ) {
        return;
      }
      event.preventDefault();
      composerRef.current?.focus();
      composerRef.current?.insertText(event.key);
    };
    document.addEventListener("keydown", onGlobalKeyDown);
    return () => document.removeEventListener("keydown", onGlobalKeyDown);
  }, [awaitingConfirmation, pendingConfirm, sidebarOpen]);

  /**
   * Re-read the files a finished turn carried that had not settled when it went.
   *
   * Send only waits for the UPLOAD, not for the ingest behind it, so a file can
   * ride a turn while still `pending` or `parsing` - and the moment its chip
   * leaves the composer, the tray's poll (the only thing that was watching it)
   * stops. Nothing else would ever learn how it ended, so its chip on the sent
   * question would sit at the state it left in: no download arrow for a file
   * that finished perfectly well, and no word about one that failed, until the
   * conversation was reopened.
   *
   * One pass, not a poll: the turn itself waits for these files before it
   * answers, so by the time it is over they have almost always settled, and the
   * rare one that has not is caught on the next reopen like everything else.
   * Best-effort throughout - a chip that keeps its old state is the same
   * outcome as not having run.
   */
  const settleCarriedFiles = useCallback(async (ids: string[]) => {
    const read = await Promise.all(
      ids.map(async (id) => {
        try {
          return [id, (await getAttachment(id)).status] as const;
        } catch {
          return null;
        }
      }),
    );
    const found = read.filter((entry) => entry !== null);
    if (found.length === 0) return;
    setLiveFiles((live) => {
      const next = new Map(live);
      for (const [id, status] of found) {
        // Only files the pane still believes in. A delete that landed while
        // this was in flight has already forgotten the id, and writing it back
        // would resurrect a chip over a row that no longer exists.
        if (next.has(id)) next.set(id, status);
      }
      return next;
    });
  }, []);

  const send = useCallback(async () => {
    // The marker on an unfilled list item is an offer, not content: trimming
    // leaves "-" behind, which sends as a message saying nothing.
    //
    // Read from the ref, not the state: the composer reports every edit but the
    // commit behind it lags a render, so a send fired from the last keystroke
    // (Enter, which is that keystroke) would leave the final characters behind.
    // The restore path below already reads the ref, and the two have to agree.
    const text = stripTrailingBareMarker(inputRef.current).trim();
    // The ref covers the frame between a send and its state landing (the
    // first send of a new chat reveals inside a view transition): a second
    // Enter in that frame must not send the message twice.
    if (streaming || awaitingConfirmation || sendingRef.current) return;
    // Enter calls send() directly, so the Send button's disabled state is not
    // the guard - this is. A file still uploading has no server id yet, so the
    // turn would go out without it while its chip sits in the composer looking
    // for all the world like it was attached, and the answer would be given
    // without ever seeing the file.
    if (trayRef.current.some((a) => a.status === "uploading")) return;
    // A file with no words is a real request - it is how every mainstream
    // assistant is used, and the natural reply to "attach the logs and I will
    // take a look". Words with no file are one too. Neither is not.
    if (!text && attachmentsForTurn(trayRef.current).length === 0) return;

    // Whether this send opens the conversation - the turn the backend titles.
    // Read from the transcript, not from conversationId: attaching a file mints
    // the conversation early, so a pane can hold an id and still be about to
    // send its very first message.
    const firstExchange = messagesRef.current.length === 0;

    // Sending supersedes any in-flight conversation selection: a late-arriving
    // transcript must not clobber the live stream (see selectConversation).
    selectSeqRef.current += 1;
    setPendingId(null);
    // A refresh still waiting on the PREVIOUS send's title is stale now.
    cancelTitleRefreshes();
    // One mutation of the transcript at a time. An editor left open over a
    // question while a new turn streams in below it is an offer the pane cannot
    // honour - saving it would rewind the thread past the answer just given -
    // so sending closes it, exactly as Cancel would.
    setEditingId(null);

    const controller = new AbortController();
    abortRef.current = controller;
    sentAtRef.current = Date.now();

    setError(null);
    // Both halves, because the editor owns the text and the page owns the
    // reading of it: neither can be inferred from the other in time.
    inputRef.current = "";
    setInput("");
    composerRef.current?.clear();
    // Taken by THIS turn, so the chip goes with the words - and is captured
    // first, because the state is cleared before the request is built.
    const carriedQuote = quote;
    setQuote(null);

    // The files this turn carries. The backend waits for any still parsing, so a
    // just-attached file can be asked about immediately. Whenever this list is
    // non-empty the pane is already bound to the conversation those files were
    // uploaded to (attaching created it), so conversationId is the right thread -
    // no separate resolve is needed here.
    const carried = attachmentsForTurn(trayRef.current);
    const sentFiles: MessageAttachment[] = carried.map((a) => ({
      id: a.serverId as string,
      filename: a.filename,
      content_type: a.contentType,
      size_bytes: a.sizeBytes,
    }));
    const attachmentIds = sentFiles.map((f) => f.id);
    // EVERY chip pointing at a file this turn takes, not just the entries
    // `attachmentsForTurn` named. Attaching the same bytes twice leaves two
    // chips sharing ONE server row (identical uploads dedupe server-side) and
    // the turn names that row once, so dropping only the named entry would
    // strand its twin in the composer looking unsent - and that twin's remove
    // button would delete the very file the answer above it was grounded on.
    const carriedIds = new Set(carried.map((a) => a.serverId as string));
    const sentChips = trayRef.current.filter(
      (a) => a.serverId !== null && carriedIds.has(a.serverId),
    );
    // The chips this turn takes leave the tray the moment it is sent, exactly as
    // the text box empties: those files belong to the turn now, and its bubble
    // carries them. They are NOT deleted - dropping a chip here only stops the
    // file riding the NEXT turn as well. Only the sent ones go: a file that
    // FAILED to upload rode nothing, so binning its chip would quietly erase a
    // failure the user has not seen yet.
    dropTray(sentChips.map((a) => a.localId));
    // They are stored, so the chips about to appear on the question can say so -
    // and offer to fetch or delete them - without waiting for a reload to
    // confirm it. What each chip may claim, and which of them still need
    // chasing once the turn is over, is carriedFileStates' decision.
    const { states, unsettled } = carriedFileStates(carried);
    if (states.length > 0) {
      setLiveFiles((live) => new Map([...live, ...states]));
    }

    // Sending implies wanting to see the reply: re-stick even if scrolled up.
    followRef.current.pin();
    // The client clock stamps both bubbles: the backend's own timestamps are a
    // round trip away (and the reply's does not exist until `done`), and "when
    // did I send this" is answered well enough by when the browser sent it.
    const sentAt = new Date().toISOString();
    const reveal = () => {
      setMessages((m) => [
        ...m,
        {
          role: "user",
          content: text,
          created_at: sentAt,
          ...(sentFiles.length > 0 ? { attachments: sentFiles } : {}),
          ...(carriedQuote !== null ? { quote: carriedQuote } : {}),
        },
        // The reply's own timestamp arrives with `done` (see stampTrailingId):
        // stamping the question's send time here would have the tooltip disagree
        // with the reloaded transcript by the whole generation.
        { role: "assistant", content: "" },
      ]);
      setStreaming(true);
      setTurnState("waiting");
    };
    // The first message of a new chat docks the composer to the foot: the
    // box travels (a view transition), the greeting leaves, the thread
    // arrives. The reveal lands a frame later inside the transition, so the
    // request waits for it: a fast failure must find the bubbles it marks.
    sendingRef.current = true;
    if (readView() === "new") {
      await transitionView("composer", () => {
        setView("thread");
        reveal();
      });
    } else reveal();

    // True once the backend has the turn. It persists the user message before
    // the first token, so from that moment the turn happened - whatever goes
    // wrong afterwards is part of it, and rolling the composer back would erase
    // a turn the transcript really has. Until then, nothing was recorded.
    //
    // The HTTP status cannot answer this: the proxy turns every transport
    // failure into an in-band error frame on a 200 so there is one parsing path,
    // which makes a dead backend indistinguishable from a healthy stream. It
    // marks those responses with a header (see the chat route) and that is what
    // this reads.
    let handedOff = false;

    // The model pin this turn has to carry itself, if any (see unsavedModel).
    const unsaved = unsavedModel(conversationId);

    try {
      const resp = await fetch("/api/chat", {
        method: "POST",
        headers: { "content-type": "application/json" },
        // The server rebuilds context from persisted turns, so no history is
        // sent. A fresh pane omits conversation_id and the backend creates one.
        body: JSON.stringify({
          message: text,
          // The two switches ride on the REQUEST as well as being stored, which
          // is what makes them work on a fresh pane: the conversation they
          // would be stored on does not exist yet when this turn is sent. An
          // unpinned axis sends "auto" - the request-level way of saying "no
          // choice here, defer to what is stored".
          thinking: thinkingOn(pins) ? "on" : "auto",
          web_search: webSearchOn(pins) ? "auto" : "off",
          // The model is the exception: it is sent only while the conversation
          // cannot answer for it, because an explicit request model takes the
          // backend's routing out of the loop (see storedModelRef).
          ...(unsaved !== null ? { model: unsaved } : {}),
          // This client can render a pause and resume it, so the backend may
          // offer side-effecting tools (which suspend the turn for a confirmation
          // rather than running unasked). A client that omits this is never
          // offered them - see the confirmation flow.
          supports_confirmation: true,
          ...(conversationId ? { conversation_id: conversationId } : {}),
          ...(attachmentIds.length > 0 ? { attachment_ids: attachmentIds } : {}),
          ...(carriedQuote !== null ? { quote: carriedQuote } : {}),
        }),
        signal: controller.signal,
      });
      if (!resp.ok) {
        throw new Error(`The server responded with HTTP ${resp.status}.`);
      }
      if (!resp.body) throw new Error("The server returned no response stream.");
      handedOff = resp.headers.get(TURN_ACCEPTED_HEADER) !== "0";

      await consumeStream(resp.body);
    } catch (e) {
      // A stop (button or unmount) aborts the fetch; keep whatever partial
      // text arrived and show no error banner.
      if (!controller.signal.aborted) {
        setError(describeFailure(e));
        setMessages((m) => markTrailingFailed(m));
      }
    } finally {
      sendingRef.current = false;
      abortRef.current = null;
      setStreaming(false);
      setStatus(null);
      // A tool left mid-run (Stop, disconnect) must stop pulsing "Running".
      setMessages(settleRunningToolSteps);
      // A placeholder with nothing in it - no answer, no reasoning, no notice -
      // would pulse forever.
      setMessages(dropEmptyTrailingAssistant);
      if (controller.signal.aborted) {
        // The user stopped this reply; say so NOW. The backend persists the
        // same flag, but waiting for a reload to admit the cut looks like
        // nothing happened. (Ordered after the empty-placeholder drop: a stop
        // before any output leaves nothing to badge.)
        setMessages(markTrailingInterrupted);
      }
      if (!handedOff && !controller.signal.aborted) {
        // The request never reached the backend (it is down, the network
        // dropped), so the turn did not happen: nothing was persisted, and a
        // reload would show no trace of it. Put the composer back exactly as it
        // was - the question AND its files - so retrying is one click instead of
        // a retype and a re-upload, and take the optimistic bubble back down
        // rather than leave a question standing that the transcript never got.
        // (An abort is the user's own doing and the backend may well have the
        // turn already, so it is left alone.)
        setMessages(dropTrailingUserTurn);
        restoreTray(sentChips);
        if (inputRef.current === "") composerRef.current?.setMarkdown(text);
        // The quote belonged to the words being handed back, so it goes back
        // with them - unless a new one has been picked in the meantime, which
        // is the more recent intent.
        setQuote((current) => current ?? carriedQuote);
      } else if (unsettled.length > 0) {
        // The turn happened, so its files are on a question now and their tray
        // chips are gone. Any that were still ingesting when it went have had
        // nothing watching them since; this is the one read that finds out how
        // they ended. Skipped when the turn never landed - those chips are back
        // in the tray, where the poll picks them up again.
        void settleCarriedFiles(unsettled);
      }
      // The single sidebar refresh point: once per send attempt, whatever the
      // outcome (done, error, or user abort). Done-only would miss the row a
      // stopped first turn already created; first-token would double up with
      // an end-of-stream refresh anyway. One place keeps it predictable.
      //
      // On the send that opened the thread, the same refresh anchors a watch
      // for the model-written title, which lands a moment behind the reply.
      // The id comes through the ref: this closure predates the conversation.
      const seededId = firstExchange ? conversationIdRef.current : null;
      if (seededId !== null) {
        void fetchConversations().then((items) => {
          scheduleTitleRefreshes(seededId, items);
        });
      } else {
        refreshConversations();
      }
    }
  }, [
    awaitingConfirmation,
    cancelTitleRefreshes,
    consumeStream,
    conversationId,
    dropTray,
    fetchConversations,
    pins,
    quote,
    refreshConversations,
    restoreTray,
    scheduleTitleRefreshes,
    settleCarriedFiles,
    streaming,
    unsavedModel,
  ]);

  // Resume a turn suspended on a side-effecting confirmation: send the decision
  // back and stream the continuation into the SAME (trailing) assistant bubble.
  // Not a new turn - no user message, no tray, no optimistic bubble - so it does
  // not touch the composer state send() manages; it just continues the paused one.
  const resume = useCallback(
    async (confirmation: ToolConfirmationData, decision: "approve" | "deny") => {
      if (streaming) return;
      const controller = new AbortController();
      abortRef.current = controller;
      sentAtRef.current = Date.now();
      setError(null);
      // Record the decision now: the buttons disable and the card shows the
      // outcome even before the continuation streams.
      setMessages((m) => markConfirmationDecided(m, decision));
      followRef.current.pin();
      setStreaming(true);
      setTurnState("waiting");
      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            conversation_id: confirmation.conversation_id,
            confirmation: { id: confirmation.id, decision },
          }),
          signal: controller.signal,
        });
        if (!resp.ok) {
          // A 404/410 means the pending confirmation is gone server-side
          // (expired, superseded, or already used): clear the card so the turn is
          // not stuck behind buttons that cannot work, and let the user re-ask.
          // Any other failure is treated as retryable and handled by the finally.
          const expired = resp.status === 404 || resp.status === 410;
          if (expired) {
            setError({
              message:
                "That pending action expired or was already handled - please ask again.",
            });
            setMessages(clearConfirmation);
            return;
          }
          throw new Error(`The server responded with HTTP ${resp.status}.`);
        }
        if (!resp.body) throw new Error("The server returned no response stream.");
        await consumeStream(resp.body);
      } catch (e) {
        if (!controller.signal.aborted) {
          setError(describeFailure(e));
        setMessages((m) => markTrailingFailed(m));
        }
      } finally {
        abortRef.current = null;
        setStreaming(false);
        setStatus(null);
        // Badge the approved calls' steps so the authorization stays visible even
        // if a later round replaces the live confirmation. Keyed off THIS resume's
        // confirmation (not message state), so it is correct mid-turn.
        if (decision === "approve") {
          const approvedIds = new Set(confirmation.calls.map((c) => c.id));
          setMessages((m) => markToolStepsApproved(m, approvedIds));
        }
        // Undo an optimistic decision that produced no answer (a failed or
        // backend-down resume), so the card returns to pending rather than
        // showing an outcome that never happened; a resume that streamed a reply
        // (or already ran the action) keeps it. Then drop the bubble only if
        // nothing at all remains.
        setMessages(revertConfirmationIfUnanswered);
        setMessages(settleRunningToolSteps);
        setMessages(dropEmptyTrailingAssistant);
        // A resume is where a paused turn actually finishes, so it is also where
        // a paused FIRST turn gets its generated title: the end-of-resume
        // refresh doubles as the watch's anchor. Watched unconditionally rather
        // than worked out from the transcript - a resume is rare, and the cost
        // of guessing wrong is a few extra list fetches. The confirmation names
        // its thread, so no ref-guessing is needed for the id.
        void fetchConversations().then((items) => {
          scheduleTitleRefreshes(confirmation.conversation_id, items);
        });
      }
    },
    [consumeStream, fetchConversations, scheduleTitleRefreshes, streaming],
  );

  // Re-answer a question the backend already holds - by asking it again, or by
  // rewording it first. A cousin of send() rather than a variant of it: the same
  // stream and the same teardown ladder, but no files and no composer state,
  // because everything the turn needs is already stored.
  //
  // The two shapes differ in ONE place, and it is the restore. A regenerate
  // discards its old reply at SAVE time, so a turn that produced nothing changed
  // nothing server-side and the transcript is put back. An edit rewrites and
  // truncates at PREPARE time, so once the backend has taken the turn the old
  // wording is gone for good - restoring it would show a transcript the server
  // does not have. That is what `restoreOnEmptyReply` selects.
  const runRewind = useCallback(
    async (
      body: Record<string, unknown>,
      apply: (messages: Message[]) => Message[],
      { restoreOnEmptyReply }: { restoreOnEmptyReply: boolean },
    ): Promise<boolean> => {
      // Returns whether the turn was actually attempted, so a caller with
      // follow-up work (arming the title watch) does not do it for a turn the
      // pane refused to start.
      if (streaming || awaitingConfirmation) return false;
      if (!conversationId) return false;

      // The transcript as it stands, to be put back if the turn never landed.
      // Unlike a send - where a refused turn hands the words back to the
      // composer - the thing to restore here IS the transcript: it was rewritten
      // optimistically, and nothing else remembers what it said.
      const snapshot = messagesRef.current;

      // Same pane-ownership move as a send: this supersedes an in-flight
      // conversation selection, and any watch left over from the previous turn's
      // title is stale. The sequence is kept: it is how the teardown below can
      // tell whether the pane still belongs to this attempt.
      const seq = (selectSeqRef.current += 1);
      setPendingId(null);
      cancelTitleRefreshes();
      // One transcript mutation at a time (see send()).
      setEditingId(null);

      const controller = new AbortController();
      abortRef.current = controller;
      sentAtRef.current = Date.now();
      setError(null);

      // Asking again implies wanting to watch the new answer arrive.
      followRef.current.pin();
      // Rewrite the transcript and open a fresh bubble at its end, so every
      // stream handler keeps writing into the trailing message exactly as it
      // does on a send. The transition re-checks the row it is about to take
      // off against the LATEST state rather than the render that offered the
      // button - taking a question off instead would delete the very turn being
      // re-asked from the pane.
      setMessages(apply);
      setStreaming(true);
      setTurnState("waiting");

      // True once the backend has the turn (see send() for why the HTTP status
      // cannot answer this). For an edit it is the whole question: past that
      // point the rewrite is committed.
      let handedOff = false;

      // The model pin this turn has to carry itself, if any (see unsavedModel).
      const unsaved = unsavedModel(conversationId);

      try {
        const resp = await fetch("/api/chat", {
          method: "POST",
          headers: { "content-type": "application/json" },
          body: JSON.stringify({
            conversation_id: conversationId,
            // The controls are read fresh, so moving a switch - or picking a
            // different model - and asking again is how a different KIND of
            // answer is asked for. The model rides on the request only while
            // the conversation has not stored it yet (see storedModelRef); once
            // it has, the backend reads the same pick from there.
            web_search: webSearchOn(pins) ? "auto" : "off",
            thinking: thinkingOn(pins) ? "on" : "auto",
            ...(unsaved !== null ? { model: unsaved } : {}),
            supports_confirmation: true,
            ...body,
          }),
          signal: controller.signal,
        });
        if (!resp.ok) {
          throw new Error(`The server responded with HTTP ${resp.status}.`);
        }
        if (!resp.body) throw new Error("The server returned no response stream.");
        handedOff = resp.headers.get(TURN_ACCEPTED_HEADER) !== "0";

        await consumeStream(resp.body);
      } catch (e) {
        if (!controller.signal.aborted) {
          setError(describeFailure(e));
        setMessages((m) => markTrailingFailed(m));
        }
      } finally {
        abortRef.current = null;
        setStreaming(false);
        setStatus(null);
        setMessages(settleRunningToolSteps);
        setMessages(dropEmptyTrailingAssistant);
        if (controller.signal.aborted) {
          // Stopped mid-answer: the backend saved what streamed, and the badge
          // says so now rather than after a reload. (A stop before any output
          // has nothing to badge - its empty placeholder is gone by now, and
          // the restore below is what handles it.)
          setMessages(markTrailingInterrupted);
        }
        // The sequence check is pane ownership, and it gates every restore: New
        // chat and delete both abort THIS stream and then empty the pane on
        // purpose, so without it the restore would refill a pane the user just
        // dismissed. Any error frame has already been shown, and a not_found
        // among them has already unbound the pane (see consumeStream).
        if (selectSeqRef.current === seq) {
          if (restoreOnEmptyReply) {
            // A REGENERATE that left no replacement on screen - refused before
            // the stream opened, or stopped or dead before the first token -
            // deleted nothing: an empty reply is never persisted, so the
            // save-time swap never ran and the reply dropped optimistically is
            // still the stored one. The role check reads the LATEST state, so a
            // turn that DID stream something (interrupted partials included)
            // still trails with an assistant bubble and is left alone.
            setMessages((m) => (m[m.length - 1]?.role === "assistant" ? m : snapshot));
          } else if (!handedOff && !controller.signal.aborted) {
            // An EDIT rewrites and truncates in its PREPARE phase, so "the turn
            // did not stream" does not mean "nothing happened": the backend may
            // have committed the rewrite and then failed on something after it.
            // The old wording is gone in that case, and putting it back on
            // screen would show a transcript the server does not have - while
            // leaving the optimistic one would be equally wrong if the request
            // never arrived. Neither guess is safe, so the server is ASKED.
            //
            // A reload that fails means the backend is unreachable, which is
            // also the case where it almost certainly never took the turn - so
            // that is the one branch where the snapshot is the right answer.
            // (An abort is left alone entirely: it is the user's own doing and
            // the turn may well have landed, the same reading a stopped send
            // makes.)
            void getConversation(conversationId)
              .then((detail) => {
                if (selectSeqRef.current !== seq) return;
                setMessages(toPaneMessages(detail.messages));
              })
              .catch(() => {
                if (selectSeqRef.current !== seq) return;
                setMessages(snapshot);
              });
          }
        }
        // One refresh per attempt, like a send - the thread changed, so its
        // activity moved. No title watch is armed here: a re-titling edit is
        // handled where it is decided (see performEdit).
        refreshConversations();
      }
      return true;
    },
    [
      awaitingConfirmation,
      cancelTitleRefreshes,
      consumeStream,
      conversationId,
      pins,
      refreshConversations,
      streaming,
      unsavedModel,
    ],
  );

  /** Answer the reply at `index` again, discarding it and anything below it.
   *
   * The newest reply is the ordinary case and names nothing; an older one is
   * named explicitly, so the backend re-asks the question above THAT reply
   * rather than the newest question in the thread.
   */
  const performRegenerate = useCallback(
    (index: number) => {
      const target = messagesRef.current[index];
      if (target?.role !== "assistant") return;
      const newest = index === messagesRef.current.length - 1;
      // An older reply MUST be named, and a bubble the server never named
      // cannot be: sending the bare shape would re-answer the NEWEST question
      // while the pane optimistically truncated at this one, and the two would
      // never reconcile. A reply loses its id whenever its stream was stopped or
      // dropped before `done`, and stops being trailing the moment anything is
      // sent after it - so this is reachable, not theoretical. The button is
      // withheld for the same case (see the render), and this is the guard that
      // does not depend on the render being right.
      if (!newest && target.id === undefined) return;
      void runRewind(
        {
          regenerate: true,
          // Omitted for the newest reply, which keeps that turn byte-identical
          // to what it was before older replies became regenerable.
          ...(newest ? {} : { message_id: target.id }),
        },
        (m) => replaceReplyAt(m, index),
        { restoreOnEmptyReply: true },
      );
    },
    [runRewind],
  );

  /** Rewrite the question at `index` and answer the new wording. */
  const performEdit = useCallback(
    (index: number, text: string) => {
      const target = messagesRef.current[index];
      if (target?.role !== "user" || target.id === undefined) return;
      const first = messagesRef.current.find((m) => m.role === "user");
      void runRewind(
        { edit: { message_id: target.id, message: text } },
        (m) => editQuestionAt(m, index, text),
        { restoreOnEmptyReply: false },
      ).then((ran) => {
        // Rewording the question a thread is NAMED after asks the backend for a
        // fresh name, which lands a moment behind the reply exactly as it does
        // on a first exchange - so the same watch catches it. A hand-picked
        // title is never re-generated, and the watch simply finds nothing.
        if (!ran || first?.id !== target.id) return;
        const id = conversationIdRef.current;
        if (id === null) return;
        void fetchConversations().then((items) => scheduleTitleRefreshes(id, items));
      });
    },
    [fetchConversations, runRewind, scheduleTitleRefreshes],
  );

  /** Take a passage into the composer as what the next message replies to.
   *
   * Replaces any pending quote rather than stacking: the composer sends one, and
   * a growing pile of chips would read as a clipboard rather than a reply.
   */
  const quotePassage = useCallback(
    (text: string, messageId?: string) => {
      const clamped = clampQuoteText(text);
      if (clamped === "") return;
      setQuote({ message_id: messageId ?? null, text: clamped });
      // The next thing to do is type the reply, so land there.
      focusComposer();
    },
    [focusComposer],
  );

  /** Start a regenerate, asking first when it would take more than one reply. */
  const requestRegenerate = useCallback(
    (index: number) => {
      const after = messagesRef.current.length - 1 - index;
      if (after > 0) {
        // Not the newest reply: everything under it goes too, and that is not
        // implied by the word "regenerate", so it is said out loud with a count.
        setPendingConfirm({ kind: "regenerate", index, count: after });
        return;
      }
      performRegenerate(index);
    },
    [performRegenerate],
  );

  /** Commit an edit, asking first when it would take more than its own reply. */
  const requestEdit = useCallback(
    (index: number, text: string) => {
      // Its own reply is what the edit inherently replaces - the same bargain
      // regenerating the newest reply makes, and it needs no dialog. Anything
      // BEYOND that reply is a thread the user has not been warned about.
      const after = messagesRef.current.length - 1 - index;
      if (after > 1) {
        // The editor stays open BEHIND the dialog, so answering "no" leaves the
        // draft exactly where it was rather than throwing away the rewrite the
        // user just typed. The turn itself closes it (see runRewind).
        setPendingConfirm({ kind: "edit", index, count: after, text });
        return;
      }
      performEdit(index, text);
    },
    [performEdit],
  );

  const stop = () => {
    if (Date.now() - sentAtRef.current < 300) return;
    abortRef.current?.abort();
  };

  /** Reset the pane to a fresh, unpersisted conversation. */
  const resetPane = useCallback(() => {
    // Starting fresh mid-stream abandons the old stream deliberately; the
    // backend still persists whatever it generated.
    selectSeqRef.current += 1;
    // Disown any in-flight conversation create so its late setConversationId is
    // gated out (see ensureConversationId) and the fresh pane mints its own.
    createConvRef.current = null;
    abortRef.current?.abort();
    adoptConversationId(null);
    // The composer goes with the pane: whatever was being written in the
    // conversation left behind stays there, and the new chat's own unsent
    // words (if any) come back.
    openDraft(null);
    setPendingId(null);
    setMessages([]);
    setError(null);
    setStatus(null);
    setTurnState("idle");
    clearTray();
    // The quote, any open editor, and any dialog waiting on a row of the old
    // transcript all belong to the pane being left behind. The two rewind
    // dialogs address their target by POSITION, so one left standing would point
    // into a transcript that is no longer there.
    setQuote(null);
    setEditingId(null);
    setPendingConfirm(null);
    setLiveFiles(new Map());
    setDeletingFiles(new Set());
    setFetchingFiles(new Set());
    // The composer's controls belong to the conversation being left: a fresh
    // pane has chosen nothing, so it shows the defaults. Any choice that was
    // still waiting for a conversation to be written to is abandoned with it,
    // and so is everything known about what the server holds for it.
    setPins(NO_PINS);
    pendingPinsRef.current = null;
    forgetStoredPins();
    setSidebarOpen(false);
    setPaneEpoch((e) => e + 1);
    focusComposer();
  }, [adoptConversationId, clearTray, focusComposer, forgetStoredPins, openDraft]);

  const newChat = useCallback(() => {
    // Leaving a thread for the new chat: the composer box travels up to the
    // greeting (a view transition); starting over on an already-new pane has
    // nothing to move.
    const apply = () => {
      setView("new");
      resetPane();
    };
    if (readView() === "thread") void transitionView("composer", apply);
    else apply();
  }, [resetPane]);

  /** Load a persisted conversation's transcript into the pane. */
  const selectConversation = useCallback(
    async (id: string) => {
      setSidebarOpen(false);
      if (id === conversationId) return;
      // Already loading this row: re-clicks must not stack duplicate fetches
      // onto a server that is slow precisely when re-clicks happen.
      if (id === pendingId) return;
      // Switching panes orphans any in-flight stream: kill it first.
      abortRef.current?.abort();
      // Disown any in-flight conversation create (see ensureConversationId), so
      // its late setConversationId cannot clobber the pane being opened.
      createConvRef.current = null;
      setError(null);
      setStatus(null);
      // The tray is deliberately NOT cleared here. The load below can fail, and
      // the pane stays on the CURRENT conversation when it does - emptying its
      // composer first would silently drop draft files that are still uploaded,
      // still ingested, and still perfectly usable. On success, seeding replaces
      // the tray wholesale with the opened conversation's own unsent files.
      // Mark the row as loading immediately; on a cold dev server the fetch
      // below can take seconds and the click must never look ignored.
      setPendingId(id);
      const seq = ++selectSeqRef.current;
      loadSeqRef.current = seq;
      try {
        const detail = await getConversation(id);
        // A newer pane action (select/send/new chat) superseded this load;
        // that action owns pendingId now, so leave it untouched.
        if (seq !== selectSeqRef.current) return;
        /** Put the loaded conversation in the pane. */
        const land = () => {
          setPendingId(null);
          loadSeqRef.current = 0;
          adoptConversationId(id);
          // The composer shows what was left unsent in THIS conversation. Done
          // only now the load has succeeded, for the tray's reason above: a
          // failed load leaves the pane, and its draft, where they were.
          openDraft(id);
          // All three belong to the transcript being replaced (see newChat).
          setQuote(null);
          setEditingId(null);
          setPendingConfirm(null);
          // The composer's controls now show what THIS conversation stored, so
          // reopening a thread reads its model and switches back exactly as it
          // was left. Anything still queued for the pane being left goes with it.
          pendingPinsRef.current = null;
          const stored = readPins(detail.settings);
          setPins(stored.pins);
          // The pins came FROM the conversation, so it can answer for them: its
          // turns carry no model field and let the backend resolve the pick (see
          // storedModelRef). Any axis whose stored value the controls cannot show
          // is remembered as dirty, for the next write to clear.
          storedModelRef.current = { conversationId: id, model: stored.pins.model };
          dirtyPinsRef.current = stored.dirty;
          const shown = detail.messages.filter(
            (m) => m.role === "user" || m.role === "assistant",
          );
          setMessages(toPaneMessages(detail.messages));
          // Restore the files this conversation is still holding that were never
          // sent: uploaded (and ingested) but left in the composer, so that is
          // where they belong again. Without this, switching away mid-draft and
          // back would look like the upload was lost, when the server still has it.
          //
          // Two tests, and the timestamp is the one that matters. A conversation
          // that predates message-level attachment records has files that no
          // message claims yet were plainly sent with one - seeding those would
          // resurrect long-gone files into an old thread's composer. A file
          // uploaded AFTER the newest turn cannot have ridden any of them, which
          // is exactly what "still in the draft" means.
          const carriedIds = new Set(
            shown.flatMap((m) => (m.attachments ?? []).map((a) => a.id)),
          );
          const lastTurnAt = shown.reduce(
            (latest, m) => Math.max(latest, Date.parse(m.created_at)),
            0,
          );
          const held = detail.attachments ?? [];
          seedTray(
            held.filter(
              (a) => !carriedIds.has(a.id) && Date.parse(a.created_at) > lastTurnAt,
            ),
          );
          // What the conversation still holds, and in what state. The chips
          // themselves come from each message's static snapshot, which records
          // what the turn CARRIED and never changes again; this is the live half
          // joined to it by id, and it is what lets a chip offer a download, mark
          // a file that aged out, or stop claiming a deleted one is there. See
          // fileLiveness for why generated files are folded in rather than read
          // out of `held`.
          setLiveFiles(fileLiveness(shown, held));
          setDeletingFiles(new Set());
          setFetchingFiles(new Set());
          // Open at the latest turn, like a live stream would.
          followRef.current.pin();
          setPaneEpoch((e) => e + 1);
          // Switching threads is intent to engage with this one: land typing.
          focusComposer();
        };
        // Opening a thread from the new chat docks the composer to the foot,
        // the travel a first send makes: the box moves, the greeting leaves,
        // the thread arrives. Only when there is a thread to arrive - an
        // empty conversation (one an attachment created) stays a new chat,
        // and thread to thread moves nothing.
        const hasTurns = detail.messages.some(
          (m) => m.role === "user" || m.role === "assistant",
        );
        if (readView() === "new" && hasTurns) {
          await transitionView("composer", () => {
            // The snapshot takes a frame, and a newer pane action may have
            // landed in it: that action owns the pane now.
            if (seq !== selectSeqRef.current) return;
            setView("thread");
            land();
          });
        } else land();
      } catch (e) {
        if (seq !== selectSeqRef.current) return;
        setPendingId(null);
        loadSeqRef.current = 0;
        if (e instanceof ApiError && e.status === 404) {
          // Deleted elsewhere (another tab/client): drop the stale list entry
          // and clear the pane rather than chat into a dead conversation. The
          // tray empties with it - those chips are the conversation the pane is
          // LEAVING, and carrying them into a fresh unbound pane would drag its
          // files onto whatever gets sent next. They are still on the server,
          // and reopening that conversation seeds them back.
          adoptConversationId(null);
          // The dead conversation's draft has nowhere to be sent; the
          // unbound pane shows its own.
          dropDraft(id);
          openDraft(null);
          setMessages([]);
          clearTray();
          setQuote(null);
          setEditingId(null);
          setPendingConfirm(null);
          setLiveFiles(new Map());
          setDeletingFiles(new Set());
          setFetchingFiles(new Set());
          setPins(NO_PINS);
          pendingPinsRef.current = null;
          forgetStoredPins();
          setError({ message: "That conversation no longer exists." });
          setPaneEpoch((e2) => e2 + 1);
          refreshConversations();
        } else {
          setError({
            message: e instanceof Error ? e.message : "Couldn't open the conversation.",
          });
          // The pane stays where it was. If the session landed while this
          // load was out, its first read of the draft was held back for the
          // load to finish - and this is the load finishing.
          if (inputRef.current === "") openDraft(conversationIdRef.current);
        }
      }
    },
    [
      adoptConversationId,
      clearTray,
      conversationId,
      dropDraft,
      focusComposer,
      forgetStoredPins,
      openDraft,
      pendingId,
      refreshConversations,
      seedTray,
    ],
  );

  /** Delete server-side, then reconcile the pane and the list.
   *
   * Confirmation happens before this runs (see the dialog at the foot of the
   * page); by the time it is called the answer is already yes.
   */
  const handleDelete = useCallback(
    async (id: string) => {
      try {
        await deleteConversation(id);
      } catch (e) {
        setError({
          message: e instanceof Error ? e.message : "Couldn't delete the conversation.",
        });
        return;
      }
      dropDraft(id);
      // Judged against the pane's id NOW (the ref), never this closure's: the
      // DELETE can take seconds, and the user may have moved on - opened
      // another thread, or started a brand-new one whose reply is streaming.
      // The closure's id would still match the deleted conversation, and the
      // reset below would abort that unrelated live stream mid-generation.
      if (id === conversationIdRef.current) {
        // The open transcript no longer exists server-side: reset the pane.
        selectSeqRef.current += 1;
        createConvRef.current = null;
        abortRef.current?.abort();
        adoptConversationId(null);
        openDraft(null);
        setMessages([]);
        setError(null);
        setStatus(null);
        clearTray();
        setQuote(null);
        setEditingId(null);
        setPendingConfirm(null);
        setLiveFiles(new Map());
        setDeletingFiles(new Set());
        setFetchingFiles(new Set());
        setPins(NO_PINS);
        pendingPinsRef.current = null;
        forgetStoredPins();
        setPaneEpoch((e) => e + 1);
      }
      // A reveal pointed at the deleted row has nothing left to type into.
      setTitleRevealId((cur) => (cur === id ? null : cur));
      refreshConversations();
    },
    [adoptConversationId, clearTray, dropDraft, forgetStoredPins, openDraft, refreshConversations],
  );

  /** Rename a conversation, showing the new name before the server agrees.
   *
   * The list is patched in place first: the topbar title is derived from this
   * same array, so anything less would leave the sidebar and the topbar
   * disagreeing for the length of the round trip. A refusal puts the old title
   * back and says why, rather than leaving a name the server never accepted.
   */
  const handleRename = useCallback(
    async (id: string, title: string) => {
      // A manual rename pre-empts the generated title (the worker's rename
      // guard yields to it), so a watch on this row has nothing left to catch
      // - and a reveal mid-play must not re-type a name the user just chose.
      if (titleWatchIdRef.current === id) cancelTitleRefreshes();
      setTitleRevealId((cur) => (cur === id ? null : cur));
      const previous = conversations.find((c) => c.id === id)?.title ?? null;
      const pending = pendingTitlesRef.current;
      pending.set(id, title);
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, title } : c)),
      );
      /** Stop shielding this row, unless a newer rename has claimed it. */
      const releaseShield = () => {
        if (pending.get(id) === title) pending.delete(id);
      };
      try {
        await updateConversation(id, { title });
      } catch (e) {
        releaseShield();
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, title: previous } : c)),
        );
        // Name the title that was refused: a sidebar rename can fail while a
        // DIFFERENT conversation is open, and this banner sits above that one.
        const reason = e instanceof Error ? ` ${e.message}` : "";
        setError({
          message: `Couldn't rename that conversation to "${title}".${reason}`,
        });
        return;
      }
      releaseShield();
      // A title change counts as activity server-side, so the list may reorder.
      refreshConversations();
    },
    [cancelTitleRefreshes, conversations, refreshConversations],
  );

  /**
   * Pin or unpin a conversation, moving its row between the sidebar groups.
   *
   * The row moves immediately - a pin whose effect waited on a round trip
   * would read as a click that missed - but the local stamp is a placeholder
   * ONLY: the server owns the pin time (a client clock cannot be trusted to
   * order the group), so the response's value replaces it. A refusal puts the
   * row back where it was and says so, rather than leaving a pin the server
   * never recorded.
   */
  const handleTogglePin = useCallback(
    async (id: string, pinned: boolean) => {
      const previous = conversations.find((c) => c.id === id)?.pinned_at ?? null;
      const optimistic = pinned ? new Date().toISOString() : null;
      const shield = pendingPinnedAtRef.current;
      const claims = pinClaimRef.current;
      const claim = ++pinClaimSeqRef.current;
      shield.set(id, optimistic);
      claims.set(id, claim);
      setConversations((prev) =>
        prev.map((c) => (c.id === id ? { ...c, pinned_at: optimistic } : c)),
      );
      /**
       * Whether this toggle still owns the row.
       *
       * A toggle fired while this one was in flight has claimed it, and its own
       * request answers for the row - landing this outcome on top, success or
       * failure, would undo what the user last asked for.
       */
      const stillOurs = () => claims.get(id) === claim;
      /**
       * Drop the shield, and with it every list answer older than this one.
       *
       * A list fetch issued before the PATCH committed carries the pre-toggle
       * pin, and it still holds the newest list ticket - so the moment the
       * shield stops rewriting it, it would land on top of the reconciled row
       * and quietly put it back in the other group. No error, and wrong until
       * some unrelated refresh happened to correct it. Refreshing here takes a
       * newer ticket, which retires that stale answer and re-reads the row.
       */
      const settle = () => {
        shield.delete(id);
        claims.delete(id);
        refreshConversations();
      };
      try {
        const saved = await setConversationPinned(id, pinned);
        if (!stillOurs()) return true;
        // Reconcile on the response: the placeholder above is a guess at the
        // stamp, and only the stored one orders the group correctly.
        setConversations((prev) =>
          prev.map((c) =>
            c.id === id ? { ...c, pinned_at: saved.pinned_at } : c,
          ),
        );
        settle();
        return true;
      } catch (e) {
        if (!stillOurs()) return true;
        setConversations((prev) =>
          prev.map((c) => (c.id === id ? { ...c, pinned_at: previous } : c)),
        );
        settle();
        // Say which way it failed: a sidebar toggle can fail while a DIFFERENT
        // conversation is open, and this banner sits above that one.
        const reason = e instanceof Error ? ` ${e.message}` : "";
        setError({
          message: `Couldn't ${pinned ? "pin" : "unpin"} that conversation.${reason}`,
        });
        // The revert moved the row a SECOND time, costing the caret again -
        // the sidebar re-arms its one-shot restore on this signal.
        return false;
      }
    },
    [conversations, refreshConversations],
  );

  /** Open the topbar title for editing, seeded with the name on show. */
  const startTitleRename = useCallback(() => {
    titleCancelledRef.current = false;
    // Editing takes the title over from a reveal mid-play; the field seeds
    // from the full name either way.
    setTitleRevealId((cur) => (cur === conversationId ? null : cur));
    setTitleDraft(
      conversations.find((c) => c.id === conversationId)?.title ?? "",
    );
    setRenamingTitleId(conversationId);
  }, [conversationId, conversations]);

  /** Commit the topbar draft, unless it is empty or unchanged. */
  const commitTitleRename = useCallback(() => {
    // The conversation the edit STARTED on, so a rename cannot land on a thread
    // that took over the pane while the field was open.
    const id = renamingTitleId;
    setRenamingTitleId(null);
    if (id === null) return;
    const next = titleDraft.trim();
    const current = conversations.find((c) => c.id === id)?.title ?? "";
    if (next === "" || next === current) return;
    void handleRename(id, next);
  }, [conversations, handleRename, renamingTitleId, titleDraft]);

  /** Route a batch of files (picker, drop, or paste) into the tray. */
  const addFiles = useCallback(
    (files: File[], origin: "upload" | "paste" = "upload") => {
      void tray.addFiles(files, origin);
    },
    [tray],
  );

  /**
   * Delete an attachment for good, saying so out loud if the server refuses.
   *
   * A refusal means the file is still stored and still answerable, which the
   * user has every reason to think is untrue after clicking a remove button. The
   * chip stays and turns red, and the banner spells it out.
   */
  /**
   * Forget a file that no longer exists, everywhere the pane still names it.
   *
   * The tray and the transcript can name the SAME file: attaching identical
   * bytes twice in one conversation dedupes to a single row server-side, so a
   * draft chip and a chip on an earlier question can share one id. Whichever of
   * them deletes it, both have to stop claiming it is there.
   */
  const forgetFile = useCallback((id: string) => {
    setLiveFiles((live) => {
      if (!live.has(id)) return live;
      const next = new Map(live);
      next.delete(id);
      return next;
    });
  }, []);

  /**
   * Record that a file aged out of retention, without forgetting it.
   *
   * The difference from {@link forgetFile} is the whole story a chip tells. A
   * deleted file has no row left, so its chip can only say the record is all
   * that remains; an EXPIRED one still has its row, and the honest thing to
   * show is a file that was there and passed its retention window. Dropping it
   * from the map instead would report someone's own retention policy as if
   * somebody had deleted their file.
   */
  const expireFile = useCallback((id: string) => {
    setLiveFiles((live) => {
      // Guarded against INSERTING, not just against a no-op write: a delete
      // that landed while this download was in flight has already forgotten the
      // id, and putting it back would resurrect a chip - delete button and all -
      // over a row that no longer exists.
      if (!live.has(id) || live.get(id) === "expired") return live;
      return new Map(live).set(id, "expired");
    });
  }, []);

  /**
   * The files an already-sent question in this thread carries.
   *
   * Attaching the same bytes twice in one conversation dedupes to ONE file
   * server-side, so a draft chip can name a file an earlier question already
   * has. Those files must survive the composer being tidied (see
   * removeAttachment), and their chips say so rather than offering a delete.
   */
  const filesAnEarlierTurnCarries = useMemo(() => {
    const ids = new Set<string>();
    for (const m of messages) {
      for (const a of m.attachments ?? []) ids.add(a.id);
    }
    return ids;
  }, [messages]);

  /**
   * The listing entry for the model THIS conversation pins, or null.
   *
   * Only a pin resolves to a known tag here. On "Default" the turn resolves
   * through the workspace default and then this deployment's own configuration
   * - a chain this page holds no answer for - so the two notices below stay
   * silent rather than describing whichever entry happens to lead the list.
   * Saying nothing is right: both exist to prevent a surprise, and inventing
   * one would be the surprise.
   */
  const pinnedEntry = useMemo(
    () => entryFor(chatCatalog, pins.model),
    [chatCatalog, pins.model],
  );

  /** Why the Thinking control cannot do anything here, or null when it can. */
  const thinkingUnavailable = useMemo(() => {
    if (pinnedEntry === null || entryCanReason(pinnedEntry)) return null;
    return (
      `${pinnedEntry.model} answers directly and has no reasoning partner, so asking ` +
      "it to reason would change nothing. Pair one with it in the model manager, or " +
      "pick a model that reasons."
    );
  }, [pinnedEntry]);

  /**
   * Whether the staged files include an image the pinned model cannot read.
   *
   * The honesty-notice pattern, moved EARLIER: the backend already says
   * `image_not_seen` beside an answer that could not look at the picture, which
   * is true and late - by then the question has been asked and the answer paid
   * for. This says the same thing while the file is still a chip in the
   * composer, where the fix is free (pick a model that sees, or send the text
   * instead).
   *
   * Keyed on the file being an IMAGE specifically. A PDF or a spreadsheet is
   * parsed into text upstream and reaches any model perfectly well, so warning
   * about "attachments" as a class would be false for most of them.
   */
  const blindToImages = useMemo(() => {
    if (pinnedEntry === null || entryCanSee(pinnedEntry)) return null;
    const images = tray.attachments.filter((a) => a.contentType.startsWith("image/"));
    if (images.length === 0) return null;
    return (
      `${pinnedEntry.model} cannot read images, so ${
        images.length === 1 ? "this picture" : "these pictures"
      } will not be looked at. The rest of the message still goes through.`
    );
  }, [pinnedEntry, tray.attachments]);

  const removeAttachment = useCallback(
    (localId: string) => {
      setError(null);
      // A file an earlier question already carries must NOT be destroyed just by
      // tidying the composer: this draft chip and the chip on that question are
      // the SAME file, so deleting it here would strip the file out of a turn
      // that was already answered from it, and leave that answer citing something
      // the reader can no longer open. Removing it here only UN-STAGES it from
      // the next turn; the question keeps its file, and the chip on that question
      // is where it can still be deleted for good (with the confirmation this
      // quiet X deliberately does not have).
      const serverId = trayRef.current.find((a) => a.localId === localId)?.serverId;
      const carriedByASentTurn =
        serverId !== null &&
        serverId !== undefined &&
        messagesRef.current.some((m) =>
          m.attachments?.some((a) => a.id === serverId),
        );
      if (carriedByASentTurn) {
        dropTray([localId]);
        return;
      }
      void removeFile(localId)
        .then((deletedId) => {
          // Deleting from the composer destroys the file itself, so a message
          // chip naming it must stop offering to delete it a second time - and
          // stop implying a later reply could still draw on it.
          if (deletedId) forgetFile(deletedId);
        })
        .catch((e) => {
          setError({
            message:
              e instanceof Error
                ? e.message
                : "Couldn't remove that file; it is still attached.",
          });
        });
    },
    [dropTray, forgetFile, removeFile],
  );

  /**
   * Delete a file a past turn carried, from the chip on that turn.
   *
   * The chip stays afterwards - the question really did arrive with this file,
   * and an answer above it may have been written from it - but it stops saying
   * the file is still there. Confirmed first (see the dialog at the foot of the
   * page), because this is not a tidy-up: the file leaves storage and its text
   * leaves the index, so no later turn in this conversation can draw on it again.
   */
  const deleteMessageAttachment = useCallback(
    async (id: string) => {
      if (deletingFiles.has(id)) return;
      setError(null);
      setDeletingFiles((busy) => new Set([...busy, id]));
      try {
        await deleteAttachment(id);
        forgetFile(id);
        // The composer may be holding a chip for this very file (see forgetFile).
        // Leaving it would be worse than a cosmetic lie: the next turn would name
        // a file that no longer exists, and the backend rejects the whole turn
        // when it cannot resolve one - so every send would fail until the chip
        // was found and removed by hand.
        dropTray(
          trayRef.current.filter((a) => a.serverId === id).map((a) => a.localId),
        );
      } catch (e) {
        // Still stored, still answerable. The chip must go on saying so.
        setError({
          message:
            e instanceof Error
              ? e.message
              : "Couldn't delete that file; it is still attached.",
        });
      } finally {
        setDeletingFiles((busy) => {
          const next = new Set(busy);
          next.delete(id);
          return next;
        });
      }
    },
    [deletingFiles, dropTray, forgetFile],
  );

  /**
   * Save a file this thread holds - one a question carried, or one a reply
   * produced - or find out it is no longer there.
   *
   * A file can stop existing long after the turn that named it was written, and
   * a generated one especially: the conversation's own file list deliberately
   * omits those (they are not composer drafts), so the server's refusal is the
   * first honest signal about them there is.
   *
   * What a refusal means for the chip is livenessAfterRefusal's decision, taken
   * through the SAME liveness map every file chip reads. Whatever it says, the
   * refusal itself is shown: the transfer is a fetch this app reads, so no
   * failure can end up saved to disk wearing the file's name.
   *
   * A chip retired in error is not permanent either: liveness is re-derived
   * from the conversation whenever it is reopened.
   */
  const downloadMessageFile = useCallback(
    async (id: string) => {
      if (fetchingFiles.has(id)) return;
      const named = messagesRef.current
        .flatMap((m) => m.attachments ?? [])
        .find((a) => a.id === id);
      setError(null);
      setFetchingFiles((busy) => new Set([...busy, id]));
      try {
        await downloadAttachment(id, named?.filename ?? "download");
      } catch (e) {
        // NAMED, because there is one banner and a turn can carry several
        // files: "This file's contents are no longer stored." floats free
        // beside four chips, and a second failure would replace the first
        // without either ever saying which file it meant. The server's own
        // sentence follows the name untouched.
        const about = named?.filename ?? "That file";
        if (e instanceof ApiError) {
          const verdict = livenessAfterRefusal(e.status, e.code);
          if (verdict === "expired") expireFile(id);
          else if (verdict === "forget") forgetFile(id);
          setError({ message: `${about}: ${e.message}`, code: e.code });
        } else {
          setError({
            message: `${about}: ${
              e instanceof Error ? e.message : "Couldn't download that file."
            }`,
          });
        }
      } finally {
        setFetchingFiles((busy) => {
          const next = new Set(busy);
          next.delete(id);
          return next;
        });
      }
    },
    [expireFile, fetchingFiles, forgetFile],
  );

  /** Ask before destroying a file a past turn carried, naming which one. */
  const requestDeleteMessageAttachment = useCallback(
    (id: string) => {
      if (deletingFiles.has(id)) return;
      // A turn can carry several files and the dialog covers the chips, so it
      // has to say which one is about to go. Read from the transcript rather
      // than passed down, so the chip's own props stay as they are.
      const named = messagesRef.current
        .flatMap((m) => m.attachments ?? [])
        .find((a) => a.id === id);
      setPendingConfirm({
        kind: "attachment",
        id,
        filename: named?.filename ?? "This file",
      });
    },
    [deletingFiles],
  );

  /**
   * Turn a paste into an attachment when it carries files or a large text block.
   *
   * Offered every paste before the composer takes it. The whole clipboard is
   * read, not just its files, because the size cap below is a decision about
   * the TEXT - and it has to be made before the box swallows it.
   *
   * @returns True when the paste became an attachment and must not also be typed.
   */
  const onPasteIntercept = (clipboard: DataTransfer): boolean => {
    // Mid-stream a new attachment would be uploaded then wiped when the turn
    // completes, so leave paste alone (text still lands in the box).
    if (streaming) return false;
    const files = Array.from(clipboard.files);
    if (files.length > 0) {
      // Real pasted files (e.g. an image on the clipboard) become uploads.
      addFiles(files, "upload");
      return true;
    }
    const text = clipboard.getData("text/plain");
    if (text.length >= PASTE_AS_FILE_CHARS) {
      // A large pasted block becomes a text attachment rather than flooding the
      // composer; the backend labels it as pasted for its citation.
      addFiles([new File([text], "pasted.txt", { type: "text/plain" })], "paste");
      return true;
    }
    // Anything shorter pastes into the composer as usual.
    return false;
  };

  const onDrop = (e: React.DragEvent<HTMLDivElement>) => {
    // Always swallow the drop so the browser never navigates to the file, but
    // ignore attachments mid-stream (they would be wiped when the turn ends) and
    // while a confirmation is pending (the composer is blocked; the disabled
    // attach button says so, and the drop-zone must agree).
    e.preventDefault();
    setDragActive(false);
    if (streaming || awaitingConfirmation) return;
    const files = Array.from(e.dataTransfer.files);
    if (files.length > 0) addFiles(files, "upload");
  };

  // The topbar editor belongs to ONE conversation, so a pane switch abandons
  // it for good. Adjusted here rather than in an effect because removing a
  // focused input from the DOM fires no blur: a switch that skipped the commit
  // path (a deep link, a shortcut) would leave the id set and the stale draft
  // would reappear on the way back to that conversation.
  if (renamingTitleId !== null && renamingTitleId !== conversationId) {
    setRenamingTitleId(null);
    setTitleDraft("");
  }

  // The model a turn would run on, as far as the page can name it: the pin,
  // else the model the last turn actually ran, else the workspace default.
  const modelTag = pins.model ?? contextUsage?.model ?? defaultModel;
  const whereabouts = useModelWhereabouts(modelTag);
  useEffect(() => {
    whereaboutsRef.current = whereabouts.state;
    refreshWhereaboutsRef.current = whereabouts.refresh;
  }, [whereabouts.state, whereabouts.refresh]);

  // A conversation being resumed reads as a thread while it loads, so the
  // greeting never shows between the travel and the messages.
  const empty = messages.length === 0 && !(resuming !== null && pendingId === resuming);
  // The view on the body, for the stylesheets and the sky. The transitions
  // above write it inside their own flush; this keeps it true everywhere
  // else (a reload of a thread, a load that fails back to the empty pane).
  useLayoutEffect(() => {
    setView(booting ? "boot" : empty ? "new" : "thread");
  }, [booting, empty]);
  // Leaving for another page takes the view with it: the console draws
  // nothing of the new chat's (the chart, the veil) over its own. Where the
  // reader was is kept for the way back.
  useLayoutEffect(
    () => () => {
      clearView();
      rememberConversation(conversationRef.current);
    },
    [],
  );
  // A failure lands in one of two places (the empty pane, or the foot of the
  // thread), so which note it gets is decided once here rather than twice
  // below. The first-run state is the ONLY code with a note of its own -
  // nothing is broken, and there is a page that fixes it; every other code
  // takes the generic note exactly as before.
  let errorNote: React.ReactNode = null;
  if (error !== null) {
    if (error.code === NO_CHAT_MODEL_CODE) {
      // The action is the point of this one, so it stands on its own even if
      // the message somehow arrived empty.
      errorNote = <NoChatModelNote message={error.message} />;
    } else if (error.message !== "") {
      // The old test was on the STRING (`{error && ...}`), so an empty message
      // rendered nothing. Moving the message inside an object must not quietly
      // turn that into an empty red box with no text in it.
      //
      // The retry answers the last reply again, when there is one to answer:
      // the error row is that reply's closing row, so the verb sits where the
      // footer's "Answer again" would.
      const last = messages.length - 1;
      const canRetry =
        messages[last]?.role === "assistant" &&
        messages[last].failed === true &&
        !streaming &&
        !awaitingConfirmation;
      errorNote = (
        <ErrorNote
          message={error.message}
          detail={error.detail}
          action={canRetry ? "Try again" : undefined}
          onAction={canRetry ? () => requestRegenerate(last) : undefined}
        />
      );
    }
  }
  // Topbar title: the open conversation's name (untitled -> a neutral label),
  // or a fresh-pane label when nothing is bound yet. Pure read of existing
  // state; the brand now lives in the sidebar (matching the prototype topbar).
  // The row is looked up, never assumed: the meta frame binds the pane on the
  // stream's FIRST frame, well before the end-of-turn refresh puts the row in
  // the sidebar list. Until the row is known the pane is still visually a
  // fresh one, so the label (and the rename affordance below) key on the ROW
  // - anything less flashes the untitled-row placeholder for the whole first
  // generation and offers a rename seeded from nothing.
  const activeRow = conversationId
    ? conversations.find((c) => c.id === conversationId)
    : undefined;
  const activeTitle = activeRow ? (activeRow.title ?? "New conversation") : "New chat";
  // The topbar reveals only a REAL generated name: if the row vanished from
  // the list mid-reveal, activeTitle is a placeholder and the plain path
  // shows it - a typewriter must never spell out "New conversation".
  // The way back lands on the conversation the reader left: opened as soon
  // as the pane is up. A load that is not the way back forgets any memory
  // left over, so a later travel cannot open something stale.
  useEffect(() => {
    if (!carriedIn) rememberConversation(null);
    if (resuming === null) return;
    const open = window.setTimeout(() => void selectConversation(resuming), 0);
    return () => window.clearTimeout(open);
    // once, on mount: the id is fixed for the page's life
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const revealTitle =
    titleRevealId !== null && titleRevealId === conversationId
      ? (activeRow?.title ?? null)
      : null;

  return (
    <>
      {booting && boot.shown && <Boot state={boot} />}
    <div className="chat-shell" data-nav-collapsed={navCollapsed ? "true" : undefined}>
      <Sidebar
        conversations={conversations}
        activeId={conversationId}
        pendingId={pendingId}
        open={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
        onNewChat={newChat}
        onSelect={(id) => void selectConversation(id)}
        onDelete={(id) => setPendingConfirm({ kind: "conversation", id })}
        onTogglePin={handleTogglePin}
        onRename={(id, title) => void handleRename(id, title)}
        revealId={titleRevealId}
        onRevealDone={endTitleReveal}
        error={listError}
        loading={!listLoaded}
        session={session}
      />

      <div className="main-col" ref={mainColRef}>
        {pendingId !== null && (
          // Pane-level loading cue, a sibling of the scroll container so it
          // stays put regardless of scroll position. On small screens the
          // sidebar slide-over (and its row spinner) closes the instant a row
          // is tapped, so the pane itself must acknowledge the load. The
          // delayed fade-in (@starting-style) keeps fast loads free of a
          // flash; aria-busy on the pane announces the state, so this stays
          // decorative.
          <div
            aria-hidden="true"
            className="pane-loading transition-opacity delay-150 duration-200 starting:opacity-0"
          >
            <span>Opening conversation...</span>
          </div>
        )}
        <header data-part="topbar">
          <button
            onClick={() => (narrow ? setSidebarOpen((o) => !o) : setNavCollapsed((c) => !c))}
            aria-label="Conversation list"
            aria-expanded={narrow ? sidebarOpen : !navCollapsed}
            aria-controls="conversation-list"
            className="topbar-toggle"
          >
            <Icon name="panel" />
          </button>
          {activeRow === undefined ? (
            // No known row - nothing is bound, or the first turn is streaming
            // ahead of the list refresh - so there is no name to change yet.
            <span className="title">{activeTitle}</span>
          ) : renamingTitleId === conversationId ? (
            <span className="title-size" data-value={titleDraft}>
            <input
              value={titleDraft}
              onChange={(e) => setTitleDraft(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  e.preventDefault();
                  commitTitleRename();
                } else if (e.key === "Escape") {
                  e.preventDefault();
                  titleCancelledRef.current = true;
                  setRenamingTitleId(null);
                }
              }}
              onBlur={() => {
                if (titleCancelledRef.current) {
                  titleCancelledRef.current = false;
                  return;
                }
                commitTitleRename();
              }}
              autoFocus
              onFocus={(e) => e.currentTarget.select()}
              maxLength={TITLE_MAX_CHARS}
              aria-label="Conversation title"
              className="title-input"
            />
            </span>
          ) : (
            <button
              type="button"
              onClick={startTitleRename}
              aria-label={`Rename conversation: ${activeTitle}`}
              className="title-edit"
            >
              {/* Pinned hint, mirroring the sidebar row's glyph: decorative
                  (the Pinned group is where readers learn the state). */}
              {activeRow.pinned_at !== null && (
                <svg
                  suppressHydrationWarning
                  viewBox="0 0 16 16"
                  className="title-pin h-3 w-3"
                  fill="none"
                  stroke="currentColor"
                  strokeWidth="1.5"
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  aria-hidden="true"
                >
                  <path suppressHydrationWarning d="M6 2.5h4M7 2.7v3.6L4.8 8.9h6.4L9 6.3V2.7M8 8.9v4.1" />
                </svg>
              )}
              <span className="title">
                {revealTitle !== null ? (
                  // A passive mirror of the sidebar's reveal: no onDone, so
                  // one logical reveal has one owner (the row, which always
                  // exists while a reveal plays; the expiry timer backstops).
                  <AnimatedTitle text={revealTitle} />
                ) : (
                  activeTitle
                )}
              </span>
              <svg
                suppressHydrationWarning
                viewBox="0 0 16 16"
                className="title-pencil h-3.5 w-3.5"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path suppressHydrationWarning d="M11.5 2.5l2 2L6 12l-2.5.5.5-2.5z" />
              </svg>
            </button>
          )}
          <ModelStatus model={modelTag} state={loadActive ? "loading" : whereabouts.state} />
        </header>

        {empty && (
          // The new chat: the greeting and the reach line above the composer,
          // with anything the pane has to say before the first message.
          <Welcome items={reach}>{errorNote}</Welcome>
        )}
        <main
          ref={scrollRef}
          aria-live="polite"
          aria-busy={streaming || pendingId !== null}
          data-part="thread"
        >
          {!empty && (
            <div className="thread-inner">
              {messages.map((m, i) => {
                const target = m.role === "user" && m.quote ? quoteTarget(m.quote, messages) : null;
                return (
                  <MessageBubble
                    // Epoch-scoped keys force a remount on pane switch so an
                    // expanded reasoning block never leaks across transcripts.
                    key={`${paneEpoch}-${i}`}
                    message={m}
                    // The rail's clock zero: when the question this reply
                    // answers was sent.
                    questionAt={
                      m.role === "assistant" && messages[i - 1]?.role === "user"
                        ? messages[i - 1].created_at
                        : undefined
                    }
                    latest={i === messages.length - 1}
                    failure={
                      m.failed === true && i === messages.length - 1 ? errorNote : undefined
                    }
                    modelName={modelTag}
                    modelLoading={loadActive}
                    live={streaming && i === messages.length - 1}
                    status={streaming && i === messages.length - 1 ? status : null}
                    liveFiles={liveFiles}
                    deletingFiles={deletingFiles}
                    onDeleteFile={requestDeleteMessageAttachment}
                    fetchingFiles={fetchingFiles}
                    onDownloadFile={(id) => void downloadMessageFile(id)}
                    onConfirm={(decision) => {
                      if (m.confirmation) void resume(m.confirmation, decision);
                    }}
                    confirmBusy={streaming}
                    // Any settled reply may be answered again; an older one takes
                    // what is under it, which the dialog says before it runs.
                    // Never mid-stream (there is no finished answer to discard)
                    // and never while a confirmation owns this very bubble - and
                    // never for an OLDER reply the server never named, which the
                    // request has no way to point at.
                    onRegenerate={
                      m.role === "assistant" &&
                      (m.id !== undefined || i === messages.length - 1) &&
                      !streaming &&
                      !awaitingConfirmation &&
                      // Not while an editor holds a draft: starting a rewind would
                      // close it and throw the typed rewrite away unasked. Absent,
                      // never disabled, per the row's contract - Cancel the edit
                      // and the button returns.
                      editingId === null
                        ? () => requestRegenerate(i)
                        : undefined
                    }
                    // Replying to a whole answer quotes it into the composer;
                    // replying to one passage of it is the selection affordance.
                    // Not on the turn still being streamed: a half-written answer
                    // can finish differently from the words a quote would freeze.
                    onReply={
                      m.content !== "" &&
                      !awaitingConfirmation &&
                      !(streaming && i === messages.length - 1)
                        ? () => quotePassage(m.content, m.id)
                        : undefined
                    }
                    // A question can be rewritten once the server has named it: an
                    // optimistic bubble has no id for the request to point at.
                    onEdit={
                      m.role === "user" &&
                      m.id !== undefined &&
                      !streaming &&
                      !awaitingConfirmation &&
                      // One editor at a time, and no silent draft loss: while one
                      // is open, the way to edit elsewhere is to close it first.
                      editingId === null
                        ? () => setEditingId(m.id ?? null)
                        : undefined
                    }
                    editing={m.role === "user" && m.id !== undefined && m.id === editingId}
                    onEditSave={(text) => requestEdit(i, text)}
                    onEditCancel={() => setEditingId(null)}
                    // The quote's way back to its source while the source is in
                    // the thread; the reason it has none when it is not.
                    onQuoteJump={
                      target !== null && "source" in target
                        ? () =>
                            quoteJump.jumpToQuoted(
                              target.source,
                              m.id,
                              // A whole message is marked as one; a passage is
                              // looked for among the words.
                              isWholeQuote(target.content, target.text) ? null : target.text,
                            )
                        : undefined
                    }
                    quoteGone={target !== null && "gone" in target ? target.gone : undefined}
                    lit={
                      quoteJump.lit !== null && quoteJump.lit.id === m.id
                        ? quoteJump.lit.kind
                        : undefined
                    }
                  />
                );
              })}
              {errorNote && lastMessage?.failed !== true && (
                // An error with no reply to close: the rail's danger row on
                // its own, under the transcript.
                <div className="turn turn-note">
                  <Step kind="error" end glyph={<Icon name="x" />}>
                    {errorNote}
                  </Step>
                </div>
              )}
            </div>
          )}
        </main>

        <footer data-part="composer">
          {!empty && (
            // Jump-to-latest: floats above the composer, shown only when scrolled
            // up (data-visible fades it in/out; kept mounted so both directions
            // animate, and pulled out of the tab order while hidden).
            <button
              type="button"
              data-part="jump-to-latest"
              data-visible={follow.away ? "true" : undefined}
              data-new={follow.unseen ? "true" : undefined}
              aria-hidden={!follow.away}
              tabIndex={follow.away ? 0 : -1}
              // While a quote was followed up the thread, this is the way back
              // down to the question that was left, and says so.
              aria-label={quoteJump.returnTo !== null ? "Back to your question" : "Jump to latest"}
              data-return={quoteJump.returnTo !== null ? "true" : undefined}
              onClick={quoteJump.returnTo !== null ? quoteJump.returnToQuoting : jumpToBottom}
            >
              <svg
                suppressHydrationWarning
                viewBox="0 0 16 16"
                className="jl-glyph jl-down h-4 w-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path suppressHydrationWarning d="M8 3v10M4 9l4 4 4-4" />
              </svg>
              {/* The way back: a return arrow, over and down the thread to the
                  question that was left (an arrow onto a line reads as a
                  download). */}
              <svg
                suppressHydrationWarning
                viewBox="0 0 16 16"
                className="jl-glyph jl-back h-4 w-4"
                fill="none"
                stroke="currentColor"
                strokeWidth="1.75"
                strokeLinecap="round"
                strokeLinejoin="round"
                aria-hidden="true"
              >
                <path suppressHydrationWarning d="M3.5 3h5a3 3 0 0 1 3 3v6.5M8.5 10l3 3 3-3" />
              </svg>
              {/* The dot: something arrived while the reader was away. */}
              <i />
            </button>
          )}
          {/* A notice that belongs to the composer docks above it, in the
              layout: a model loading from disk, for as long as the load. */}
          <ComposerNotice
            open={loadActive && !noticeDismissed}
            title={`${modelTag ?? "The model"} is loading from disk`}
            body="It will be ready shortly - keep typing."
            loading
            onDismiss={() => setNoticeDismissed(true)}
          />
          <div
            onDragOver={(e) => {
              e.preventDefault();
              setDragActive(true);
            }}
            onDragLeave={(e) => {
              e.preventDefault();
              setDragActive(false);
            }}
            onDrop={onDrop}
            data-drag={dragActive ? "true" : undefined}
            data-disabled={awaitingConfirmation ? "true" : undefined}
            className="composer-box"
            ref={composerBoxRef}
          >
            <AttachmentTray
              attachments={tray.attachments}
              onRemove={removeAttachment}
              staged={filesAnEarlierTurnCarries}
            />
            {/* Directly under the chips it is about, and above the box the
                question is typed in: the whole point of moving this notice
                earlier is that it arrives while the fix is still free - swap
                the model, or send the words without the picture. The backend's
                own `image_not_seen` still lands beside the answer if the
                message goes anyway; this does not replace it, it precedes it. */}
            {blindToImages !== null && (
              <p className="composer-notice" role="status">
                {blindToImages}
              </p>
            )}
            {quote !== null && (
              // What the next message is replying to, above the box it is typed
              // in - the same place the file chips sit, for the same reason.
              <QuoteChip text={quote.text} onDismiss={() => setQuote(null)} />
            )}
            <div className="composer-editor">
              <ComposerInput
                ref={composerRef}
                ariaLabel="Message"
                onChangeMarkdown={onDraftChange}
                onPasteIntercept={onPasteIntercept}
                onSend={() => void send()}
                // Blocked while a side-effecting action awaits an answer: the
                // paused turn owns the trailing bubble, and a new message would
                // supersede its pending confirmation server-side.
                disabled={awaitingConfirmation}
                placeholder={
                  awaitingConfirmation
                    ? "Approve or deny the pending action above to continue..."
                    : "Ask a question, or paste something to work on"
                }
              />
            </div>
            <div className="composer-row">
              <AttachButton onPick={addFiles} disabled={streaming || awaitingConfirmation} />
              {/* The three controls stay live DURING a stream: moving one and
                  asking again is how a different kind of answer is asked for,
                  and each change applies from the next turn. They are blocked
                  only while a confirmation is pending, because that turn is
                  suspended mid-flight and would resume against the changed
                  setting rather than the one it was planned with. */}
              <ThinkingToggle
                value={thinkingOn(pins)}
                onChange={(on) =>
                  changePins({ ...pins, thinking: on ? "on" : null }, "thinking")
                }
                disabled={awaitingConfirmation}
                unavailableReason={thinkingUnavailable}
              />
              <WebSearchToggle
                value={webSearchOn(pins)}
                onChange={(enabled) =>
                  changePins({ ...pins, webSearch: enabled ? null : "off" }, "web search")
                }
                disabled={awaitingConfirmation}
              />
              {/* Leads the right dock (owner decision 2026-08-04): the dock
                  is what runs the turn AND what it costs - a quiet ring whose
                  numbers are one click away, docked where the eye already
                  checks which model is answering. */}
              <ContextMeter usage={contextUsage} />
              <ModelSelector
                models={chatCatalog}
                unavailable={modelsUnavailable}
                loaded={modelsLoaded}
                value={pins.model}
                defaultModel={defaultModel}
                fresh={conversationId === null}
                onChange={(model) => changePins({ ...pins, model }, "the model")}
                disabled={awaitingConfirmation}
              />
              <button
                onClick={() => (streaming ? stop() : void send())}
                aria-label={streaming ? "Stop" : "Send"}
                // Attachments alone are enough to send: a file IS a question
                // ("attach the logs and I will take a look"). What is never
                // enough is nothing at all, or a file that has not finished
                // uploading - that would go out without the file while its chip
                // sits there looking attached. And nothing goes while a
                // confirmation is pending - it must be answered first.
                disabled={
                  !streaming &&
                  (awaitingConfirmation ||
                    // Same reading of the draft that send() takes, so a lone
                    // list marker leaves the button off rather than dead.
                    (stripTrailingBareMarker(input).trim() === "" && !hasFilesToSend) ||
                    tray.uploading)
                }
                title={
                  tray.uploading && !streaming
                    ? "Waiting for attachments to finish uploading"
                    : streaming
                      ? "Stop"
                      : "Send"
                }
                className="send"
              >
                {streaming ? (
                  <svg
                    suppressHydrationWarning
                    viewBox="0 0 16 16"
                    className="h-3.5 w-3.5"
                    fill="currentColor"
                    aria-hidden="true"
                  >
                    <rect suppressHydrationWarning x="3.5" y="3.5" width="9" height="9" rx="1.5" />
                  </svg>
                ) : (
                  <svg
                    suppressHydrationWarning
                    viewBox="0 0 16 16"
                    className="h-4 w-4"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth="1.75"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <path suppressHydrationWarning d="M8 13V3M4 7l4-4 4 4" />
                  </svg>
                )}
              </button>
            </div>
          </div>
          <div className="hint">
            Local model. Answers can be wrong, so check the sources.
          </div>
        </footer>
      </div>

      {/* Selecting words anywhere in the transcript offers to quote them. It
          lives outside the scroll container so a bubble's own overflow cannot
          clip it, and positions itself over the selection. */}
      <SelectionQuote containerRef={scrollRef} onQuote={(text, id) => quotePassage(text, id)} />

      {/* Conditionally rendered, never toggled with `hidden`: the author
          `display` on the scrim would beat the user agent's rule for the
          attribute and the dialog would ship open. */}
      {pendingConfirm !== null &&
        (pendingConfirm.kind === "conversation" ? (
          <ConfirmDialog
            title="Delete this conversation?"
            body="Every message in it goes too. This cannot be undone."
            onCancel={() => setPendingConfirm(null)}
            onConfirm={() => {
              const { id } = pendingConfirm;
              setPendingConfirm(null);
              void handleDelete(id);
            }}
          />
        ) : pendingConfirm.kind === "attachment" ? (
          <ConfirmDialog
            title="Delete this file?"
            body={fileDeleteBody(pendingConfirm.filename)}
            onCancel={() => setPendingConfirm(null)}
            onConfirm={() => {
              const { id } = pendingConfirm;
              setPendingConfirm(null);
              void deleteMessageAttachment(id);
            }}
          />
        ) : pendingConfirm.kind === "regenerate" ? (
          <ConfirmDialog
            title="Answer this again?"
            body={discardedBody(pendingConfirm.count)}
            confirmLabel="Regenerate"
            onCancel={() => setPendingConfirm(null)}
            onConfirm={() => {
              const { index } = pendingConfirm;
              setPendingConfirm(null);
              performRegenerate(index);
            }}
          />
        ) : (
          <ConfirmDialog
            title="Edit this message?"
            body={discardedBody(pendingConfirm.count)}
            confirmLabel="Save and answer"
            onCancel={() => setPendingConfirm(null)}
            onConfirm={() => {
              const { index, text } = pendingConfirm;
              setPendingConfirm(null);
              performEdit(index, text);
            }}
          />
        ))}
    </div>
    </>
  );
}
