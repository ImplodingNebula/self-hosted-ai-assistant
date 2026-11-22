/**
 * The chat pane's client-side message model and its pure state transitions.
 *
 * `Message` and `ToolStep` are the shapes a rendered turn holds; every function
 * here takes the current `Message[]` and returns a new one, mutating nothing.
 * They all key off the TRAILING message, because the SSE loop always streams into
 * the turn it just appended, so a single "operate on the last message" convention
 * keeps the whole stream-handling surface consistent. Kept free of React and the
 * SSE layer so they can be reasoned about (and tested) on their own.
 */
import type {
  Attachment,
  AttachmentStatus,
  CitationData,
  ConversationMessage,
  FileEventPayload,
  MessageAttachment,
  MessageNotice,
  MessageQuote,
  ToolCallData,
  ToolConfirmationData,
  ToolResultData,
  TurnUsageData,
} from "@/lib/types";

/**
 * Longest passage the backend will accept as a quote (it refuses, never
 * truncates - a silently shortened quotation would attribute words to a passage
 * that does not contain them). Mirrored here so the composer can clamp its own
 * excerpt visibly instead of sending a turn that comes back 422.
 */
export const QUOTE_MAX_CHARS = 2000;

/**
 * Reduce a passage to something quotable, marking it when it had to be cut.
 *
 * A prefix that ends in an ellipsis is an honest excerpt - every word in it
 * really is in the passage, and the marker says there was more. That is the one
 * shortening allowed anywhere near a quote; the backend refuses the rest.
 */
export function clampQuoteText(text: string): string {
  const trimmed = text.trim();
  if (trimmed.length <= QUOTE_MAX_CHARS) return trimmed;
  return `${trimmed.slice(0, QUOTE_MAX_CHARS - 3).trimEnd()}...`;
}

/**
 * Whether a quote is of a whole message rather than of words in it: its text
 * is the message's own, as the Reply verb stores it (clamped the same way,
 * marker and all), so no flag is stored or sent for this. A plain message
 * selected in full is the same thing; a formatted reply selected in full is
 * not, since a selection yields the rendered words and the reply holds its
 * markdown - that one is looked for among the prose like any passage.
 */
export function isWholeQuote(content: string, text: string): boolean {
  return clampQuoteText(content) === text;
}

/**
 * One tool the assistant used this turn, for the activity trace.
 *
 * `narration` is any text the model streamed just before the call ("let me
 * search instead") - captured here rather than lost, so the agentic reasoning is
 * preserved above the clean final answer. `running` is true between the call and
 * its result; `result` is a short summary of what came back.
 */
export interface ToolStep {
  id: string;
  name: string;
  arguments: Record<string, unknown>;
  narration?: string;
  result?: string;
  isError?: boolean;
  /** How many sources the tool produced; absent when it could not say. */
  count?: number;
  running: boolean;
  /**
   * True when this step ran a side-effecting call the user approved. Marked on the
   * step itself (not derived from the live confirmation, which a later round can
   * replace) so the "approved" badge survives a second pause in the same turn.
   */
  approved?: boolean;
  /**
   * True when this step is a side-effecting call the user DECLINED: it ran nothing
   * (no result), and the card is badged "denied" - the mirror of `approved`.
   */
  denied?: boolean;
  /**
   * When the call was opened, in client clock ms; live turns only. The
   * timeline rail reads it against the question's time (`+5s`). A reloaded
   * step has none - the server stores no per-step clock - and its cell stays
   * empty rather than guessed.
   */
  at?: number;
}

export interface Message {
  role: "user" | "assistant";
  content: string;
  /**
   * Server-side message id, once there is one.
   *
   * Absent while a turn is optimistic (the bubble exists before the backend has
   * named it) and on any reply whose stream never reached the backend at all.
   * Present is NOT proof a row was persisted: a turn that saved nothing still
   * gets an id on `done` so the wire contract holds. Nothing keys off it today -
   * it is what a per-message action addresses once one needs to name a row
   * rather than "the last one".
   */
  id?: string;
  /**
   * When this turn happened, ISO-8601. A reloaded transcript carries the
   * server's timestamp; a live turn stamps the client clock as it is appended,
   * so the bubble can say when it was sent before `done` has confirmed anything.
   */
  created_at?: string;
  /** Streamed model reasoning, shown above the answer; assistant turns only. */
  thinking?: string;
  /** Client clock ms when the first reasoning delta landed; live turns only. */
  thinkingAt?: number;
  /** Client clock ms when the first answer token landed; live turns only. */
  answerAt?: number;
  /** Client clock ms when `done` closed the turn; live turns only. */
  doneAt?: number;
  /**
   * The model that produced this reply, as the server names it: stored on a
   * reloaded turn, stamped from `done` on a live one. Null when unknown.
   */
  model?: string | null;
  /** Tokens the model wrote this turn, when the provider counted them. */
  completionTokens?: number;
  /** Milliseconds the runtime spent writing them, when it measured that. */
  generationMs?: number;
  /** Client clock ms when the turn paused on a confirmation; live turns only. */
  confirmAt?: number;
  /**
   * The turn ended on an error (a typed stream error, a request that died)
   * with this reply on screen: the pane's error row is its closing row, and
   * "Try again" answers it again. Client-side only, like the clocks.
   */
  failed?: boolean;
  /** Client clock ms when the turn failed; live turns only. */
  failedAt?: number;
  /** The tools this turn used, in order, with the model's narration; assistant only. */
  toolSteps?: ToolStep[];
  /** Retrieved sources cited by the answer; assistant turns only. */
  citations?: CitationData[];
  /**
   * The files this turn holds. On a question they are what it CARRIED (the tray
   * empties when it is sent); on a reply they are what its tools PRODUCED, which
   * `origin` is what tells apart.
   */
  attachments?: MessageAttachment[];
  /** The passage this question was replying to; user turns only. */
  quote?: MessageQuote;
  /** Standing notes qualifying this reply; assistant turns only. */
  notices?: MessageNotice[];
  /** True when a reloaded reply was cut short (disconnect/error mid-stream). */
  interrupted?: boolean;
  /**
   * A side-effecting tool call awaiting the user's approve/deny; assistant only.
   * Set when the turn suspended on a `tool_confirmation` event. While present and
   * undecided it blocks the composer - the turn is paused, not finished.
   */
  confirmation?: ToolConfirmationData;
  /** The decision made on `confirmation`, once the user answered it. */
  confirmationDecision?: "approve" | "deny";
  /**
   * The `content` at the moment the turn paused for `confirmation` - the model's
   * pre-pause narration, if any. A failed resume leaves `content` unchanged from
   * this, so it is how "the resume streamed no answer" is told apart from "the
   * resume answered" when no tool step was created (see revertConfirmationIfUnanswered).
   */
  confirmationContentBaseline?: string;
}

/**
 * Map a stored transcript onto the pane's message model.
 *
 * The single translation point between the wire shape and what a bubble renders,
 * so opening a conversation and re-reading one after a turn cannot drift apart -
 * a difference between the two would show as a transcript that changes shape
 * depending on how it was loaded. System turns are dropped: they are prompt
 * scaffolding, not something anyone said.
 */
export function toPaneMessages(stored: ConversationMessage[]): Message[] {
  return stored
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: m.content,
      id: m.id,
      created_at: m.created_at,
      thinking: m.thinking ?? undefined,
      citations: m.citations && m.citations.length > 0 ? m.citations : undefined,
      attachments:
        m.attachments && m.attachments.length > 0 ? m.attachments : undefined,
      quote: m.quote ?? undefined,
      model: m.role === "assistant" ? m.model : undefined,
      ...(m.role === "assistant" ? turnFigures(m.usage) : {}),
      notices: m.notices && m.notices.length > 0 ? m.notices : undefined,
      // Replay the tool trace: stored steps are all finished (a saved turn is
      // not mid-run), so each rehydrates with running=false.
      toolSteps:
        m.tool_steps && m.tool_steps.length > 0
          ? m.tool_steps.map((s) => ({
              id: s.id,
              name: s.name,
              arguments: s.arguments,
              narration: s.narration ?? undefined,
              result: s.result ?? undefined,
              isError: s.is_error ?? false,
              count: s.count ?? undefined,
              approved: s.approved ?? false,
              denied: s.denied ?? false,
              running: false,
            }))
          : undefined,
      interrupted: m.interrupted ?? undefined,
    }));
}

/**
 * The live state of every file a transcript's chips could name.
 *
 * The chips themselves come from each message's STATIC snapshot: it records
 * what the turn carried and never changes again, which is exactly right for a
 * transcript and useless for saying whether the file is still there. This is
 * the other half - the conversation's live rows, joined to those snapshots by
 * id - and it is what lets a sent chip offer a download, mark a file that aged
 * out, or stop claiming a deleted file exists.
 *
 * An id ABSENT from the result names no live row, and a chip reads that as
 * deleted. That is only sound because `held` is the whole set of rows the
 * conversation still has, with ONE deliberate exception: files the assistant
 * generated are left out of it server-side (that list exists to restore a
 * composer, and a document nobody chose is not a draft). Absence is therefore
 * no evidence about them at all, so they are folded back in here as `ready` -
 * believed live until a download says otherwise, which is what a download does
 * when a file has aged out. Without that, reopening a conversation would mark
 * every file the assistant ever made as gone.
 *
 * A live row wins over that assumption where the two overlap, so nothing here
 * can promote a row the server described.
 *
 * @param messages - The transcript's turns, each with its attachment snapshot.
 * @param held - The conversation's live attachment rows (generated ones absent).
 * @returns Status by attachment id, for every file believed to exist.
 */
export function fileLiveness(
  messages: readonly { attachments?: MessageAttachment[] }[],
  held: readonly Attachment[],
): Map<string, AttachmentStatus> {
  const live = new Map<string, AttachmentStatus>();
  for (const message of messages) {
    for (const file of message.attachments ?? []) {
      if (file.origin === "generated") live.set(file.id, "ready");
    }
  }
  for (const row of held) live.set(row.id, row.status);
  return live;
}

/**
 * Append a streamed delta to the given field of the trailing (assistant) message.
 *
 * The FIRST delta on each channel also stamps when it landed (`thinkingAt`,
 * `answerAt`): the timeline rail reads those against the question's clock,
 * and the thinking row's "Thought for N seconds" is the gap between them. An
 * existing stamp is never moved, so a tool round that streams more reasoning
 * keeps the time the reasoning began.
 *
 * @param now - The client clock, injectable so the stamp can be checked.
 */
export function appendDelta(
  messages: Message[],
  field: "content" | "thinking",
  delta: string,
  now: number = Date.now(),
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const updated =
    field === "content"
      ? { ...last, content: last.content + delta, answerAt: last.answerAt ?? now }
      : {
          ...last,
          thinking: (last.thinking ?? "") + delta,
          thinkingAt: last.thinkingAt ?? now,
        };
  return [...messages.slice(0, -1), updated];
}

/**
 * Read a reply's figures off the wire, from `done` or from a stored turn.
 *
 * One reader for both, so a reloaded reply cannot show a different number
 * than its live stream closed on. A count that is not a positive number is
 * no figures at all; a time that is not is a count with no speed.
 */
export function turnFigures(
  usage: TurnUsageData | null | undefined,
): { completionTokens?: number; generationMs?: number } {
  const tokens = usage?.completion_tokens;
  if (typeof tokens !== "number" || !(tokens > 0)) return {};
  const elapsed = usage?.generation_ms;
  return typeof elapsed === "number" && elapsed > 0
    ? { completionTokens: tokens, generationMs: elapsed }
    : { completionTokens: tokens };
}

/**
 * Close the trailing (assistant) reply's figures once `done` names them.
 *
 * The footer's small print - which model answered, how many tokens it wrote
 * and how fast - comes from the done frame: the model from its context
 * reading, the count and the writing time from its turn usage, which is the
 * same object the server stores with the reply. The speed is therefore the
 * runtime's own, over every round of the turn, and reads the same after a
 * reload. Fields the frame leaves out stay as they were.
 */
export function recordTrailingFigures(
  messages: Message[],
  figures: { model?: string | null; usage?: TurnUsageData | null; doneAt?: number },
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const updated: Message = { ...last, ...turnFigures(figures.usage) };
  if (figures.model !== undefined && figures.model !== null) updated.model = figures.model;
  if (figures.doneAt !== undefined) updated.doneAt = figures.doneAt;
  return [...messages.slice(0, -1), updated];
}

/** Append a streamed citation to the trailing (assistant) message's sources. */
export function appendCitation(
  messages: Message[],
  citation: CitationData,
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const updated = { ...last, citations: [...(last.citations ?? []), citation] };
  return [...messages.slice(0, -1), updated];
}

/**
 * Record a file this turn's tools produced on the trailing (assistant) message.
 *
 * The live counterpart of the `attachments` a reloaded reply carries: the same
 * field, filled from the `file` stream event, so the chip a stream shows and the
 * chip a reload shows are rendered from one shape rather than two.
 *
 * The event is announce-only and arrives after the row is committed, so the id
 * always names something the download route can serve. Appended in arrival
 * order, and never deduplicated: one turn CAN legitimately produce two files
 * with the same name (two calls to the same tool), and they are separate rows.
 */
export function appendFile(
  messages: Message[],
  file: FileEventPayload,
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const attachment: MessageAttachment = {
    id: file.attachment_id,
    filename: file.filename,
    content_type: file.content_type,
    size_bytes: file.size_bytes,
    origin: file.origin,
  };
  const attachments = [...(last.attachments ?? []), attachment];
  return [...messages.slice(0, -1), { ...last, attachments }];
}

/**
 * Attach a standing notice to the trailing (assistant) message.
 *
 * A LIST, not a single note: one turn can legitimately raise several - a
 * substituted model AND a file it could not use, or even two
 * `attachment_unavailable` notes covering disjoint sets of files. The old
 * single-slot pane state showed only whichever landed last, so the reader
 * silently lost the rest. Appended in arrival order, exactly as the backend
 * stores them, so the live view and a reloaded one agree.
 */
export function appendNotice(messages: Message[], notice: MessageNotice): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const notices = [...(last.notices ?? []), notice];
  return [...messages.slice(0, -1), { ...last, notices }];
}

/**
 * Drop notices that repeat one already in the list, word for word.
 *
 * Keyed on the WHOLE (code, message) pair, never the code alone: two
 * `attachment_unavailable` notes covering disjoint sets of files say different
 * things and both belong (see `appendNotice`). Two notes with identical text
 * say the same thing twice, and the reader learns nothing from the second.
 *
 * That happens for real: a turn declines a round of several side-effecting
 * calls and the backend raises its `tool_declined` note once PER call, so a
 * two-call refusal reads the identical sentence twice. Applied at the render
 * site rather than on the way in, so a live turn and a reloaded one - which
 * take different paths into `notices` - collapse it the same way.
 */
export function dedupeNotices(notices: MessageNotice[]): MessageNotice[] {
  const seen = new Set<string>();
  return notices.filter((n) => {
    const key = `${n.code} ${n.message}`;
    if (seen.has(key)) return false;
    seen.add(key);
    return true;
  });
}

/**
 * Record the server's id on the trailing (assistant) message, once `done` names
 * it - and, when the bubble has no timestamp yet, the moment it finished. The
 * completion time is what a reloaded transcript will show (the server stamps the
 * reply as it persists it), so stamping it here keeps the live tooltip and the
 * reloaded one telling the same story; an existing timestamp is never touched.
 */
export function stampTrailingId(messages: Message[], id: string, at?: string): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || last.id === id) return messages;
  const stamped =
    at !== undefined && last.created_at === undefined
      ? { ...last, id, created_at: at }
      : { ...last, id };
  return [...messages.slice(0, -1), stamped];
}

/**
 * Record the server's id on the QUESTION this stream is answering.
 *
 * The `meta` frame names it on the stream's first frame, well before `done`, so
 * a just-sent question stops being anonymous immediately - which is what lets it
 * be edited without a reload, since an edit has to name the row it rewrites.
 * Targets the last user message rather than the trailing one, because by now the
 * assistant placeholder is already sitting under it.
 *
 * Never overwrites an id that is already there: on a regenerate or an edit the
 * question is a stored row whose id the transcript is holding, and the frame is
 * simply naming the same one back.
 */
export function stampQuestionId(messages: Message[], id: string): Message[] {
  const index = messages.findLastIndex((m) => m.role === "user");
  if (index === -1 || messages[index].id !== undefined) return messages;
  return [
    ...messages.slice(0, index),
    { ...messages[index], id },
    ...messages.slice(index + 1),
  ];
}

/** Replace the trailing (assistant) message's content with the final server copy. */
export function replaceTrailingContent(messages: Message[], content: string): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  return [...messages.slice(0, -1), { ...last, content }];
}

/**
 * Attach a pending confirmation to the trailing (assistant) message.
 *
 * Clears any earlier decision: a turn can pause more than once (a second
 * side-effecting call after the first was resolved), and each pause is a fresh
 * question, so the buttons must come back rather than stay stuck on the last
 * answer.
 */
export function attachConfirmation(
  messages: Message[],
  confirmation: ToolConfirmationData,
  now: number = Date.now(),
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  // Snapshot the content at pause time (the model's pre-pause narration, if any).
  // A failed resume leaves content at this value; a resume that answers grows it -
  // which is how the revert below tells the two apart when no step was created.
  const updated = {
    ...last,
    confirmation,
    confirmationDecision: undefined,
    confirmationContentBaseline: last.content,
    confirmAt: last.confirmAt ?? now,
  };
  return [...messages.slice(0, -1), updated];
}

/** Record the user's approve/deny on the trailing message's pending confirmation. */
export function markConfirmationDecided(
  messages: Message[],
  decision: "approve" | "deny",
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.confirmation) return messages;
  return [...messages.slice(0, -1), { ...last, confirmationDecision: decision }];
}

/**
 * Undo an optimistic approve/deny that produced no answer.
 *
 * The decision is recorded the instant a button is clicked, for immediate
 * feedback - but a resume can then fail (the confirmation expired, the backend
 * was unreachable), and a card left reading "You approved this action" over an
 * action that never ran would be a lie the user cannot recover from. So when a
 * resume ends with the bubble still answerless, the decision is rolled back and
 * the pending card (buttons) returns. A resume that DID stream an answer keeps
 * its outcome - the action ran, and the reply is proof.
 */
export function revertConfirmationIfUnanswered(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.confirmation) return messages;
  // If a call from this confirmation already produced a tool step, the resume got
  // past the point of no return (the action ran, or was recorded as declined), so
  // rolling back to a pending "approve or deny?" would misrepresent what occurred
  // and re-block the composer behind a card the fold hides. Keep the decision: the
  // composer unblocks and the step (badged approved/denied, or "did not finish")
  // tells the true story.
  const stepIds = new Set((last.toolSteps ?? []).map((s) => s.id));
  if (last.confirmation.calls.some((c) => stepIds.has(c.id))) return messages;
  // No step was created, so the resume never reached the backend (an HTTP failure,
  // not a decision that took effect). It answered only if `content` GREW past the
  // pre-pause narration snapshot; if it is unchanged, nothing streamed - roll the
  // optimistic decision back so the buttons return. (Testing `content !== ""` here
  // would wrongly keep the decision whenever the model narrated before the pause.)
  if (last.content !== (last.confirmationContentBaseline ?? "")) return messages;
  return [...messages.slice(0, -1), { ...last, confirmationDecision: undefined }];
}

/** Badge the given calls' steps as user-approved (durable across a later pause). */
export function markToolStepsApproved(messages: Message[], ids: Set<string>): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.toolSteps) return messages;
  const toolSteps = last.toolSteps.map((step) =>
    ids.has(step.id) ? { ...step, approved: true } : step,
  );
  return [...messages.slice(0, -1), { ...last, toolSteps }];
}

/** Drop the pending confirmation from the trailing message (it is gone server-side). */
export function clearConfirmation(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.confirmation) return messages;
  const rest = { ...last };
  delete rest.confirmation;
  delete rest.confirmationDecision;
  delete rest.confirmationContentBaseline;
  return [...messages.slice(0, -1), rest];
}

/**
 * Open a tool step on the trailing message (the model asked to run a tool).
 *
 * The text streamed since the last step is the model narrating what it is about
 * to do ("let me search instead"): it is captured on the step and cleared from
 * `content`, so the final answer - which streams after the tool results - stands
 * alone below the activity trace rather than being wiped when `done` arrives.
 */
export function beginToolStep(
  messages: Message[],
  call: ToolCallData,
  now: number = Date.now(),
): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant") return messages;
  const step: ToolStep = {
    id: call.id,
    name: call.name,
    arguments: call.arguments,
    narration: last.content || undefined,
    running: true,
    at: now,
  };
  return [
    ...messages.slice(0, -1),
    { ...last, content: "", toolSteps: [...(last.toolSteps ?? []), step] },
  ];
}

/** Close the matching open tool step on the trailing message with its result. */
export function completeToolStep(messages: Message[], result: ToolResultData): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.toolSteps) return messages;
  let filled = false;
  const toolSteps = last.toolSteps.map((step) => {
    // The first still-running step with this id (a tool can be called more than
    // once in a turn, each its own step).
    if (!filled && step.running && step.id === result.id) {
      filled = true;
      // A declined side-effecting call ran nothing: badge it "denied" and keep no
      // result, so the card reads as a blocked action, not a tool that returned.
      return {
        ...step,
        running: false,
        result: result.denied ? undefined : result.result,
        isError: result.is_error ?? false,
        denied: result.denied ?? false,
        count: result.count ?? undefined,
      };
    }
    return step;
  });
  return [...messages.slice(0, -1), { ...last, toolSteps }];
}

/**
 * Close any still-open tool steps on the trailing message when a turn ends.
 *
 * A step only clears on its own tool_result, so a turn that ends any other way -
 * the user hit Stop mid-tool, a disconnect, a backend error between the call and
 * its result - would leave a step pulsing "Running" forever. Called on teardown,
 * this settles those to a finished-without-a-result state (rendered "Did not
 * finish", not "Running"), so nothing spins after the turn is over.
 */
export function settleRunningToolSteps(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || !last.toolSteps?.some((step) => step.running)) {
    return messages;
  }
  const toolSteps = last.toolSteps.map((step) =>
    step.running ? { ...step, running: false } : step,
  );
  return [...messages.slice(0, -1), { ...last, toolSteps }];
}

/**
 * Mark the trailing assistant reply as the one its turn failed on.
 *
 * Called before the end-of-turn cleanup: a failed reply with nothing in it is
 * still dropped, and then no reply carries the flag - the error stands on its
 * own under the transcript, with nothing to answer again.
 */
export function markTrailingFailed(messages: Message[], now: number = Date.now()): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || last.failed) return messages;
  return [...messages.slice(0, -1), { ...last, failed: true, failedAt: now }];
}

/**
 * Mark the trailing assistant reply as cut short.
 *
 * The backend persists the interrupted flag when a stream dies, so a RELOADED
 * transcript shows the "Interrupted" chip - but the live pane only learns the
 * flag from this call. Without it, hitting Stop showed nothing until the user
 * left the conversation and came back to the server's copy.
 */
export function markTrailingInterrupted(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (!last || last.role !== "assistant" || last.interrupted) return messages;
  return [...messages.slice(0, -1), { ...last, interrupted: true }];
}

/**
 * Drop a trailing assistant placeholder that produced nothing at all.
 *
 * "Nothing at all" includes its notices, a pending confirmation, and any file it
 * produced: a turn that raised a notice (a file it could not read), one that
 * SUSPENDED to ask before a side-effecting action, and one whose tool wrote a
 * document before the stream died have all streamed no answer yet and are none
 * of them empty - the bubble stays to carry that. Only a truly empty placeholder
 * goes - otherwise it would pulse forever.
 */
export function dropEmptyTrailingAssistant(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (
    last &&
    last.role === "assistant" &&
    !last.content &&
    !last.thinking &&
    !last.notices?.length &&
    !last.confirmation &&
    !last.attachments?.length &&
    !last.toolSteps?.length
  ) {
    return messages.slice(0, -1);
  }
  return messages;
}

/** Drop a trailing user message whose turn the backend never accepted. */
export function dropTrailingUserTurn(messages: Message[]): Message[] {
  const last = messages[messages.length - 1];
  if (last && last.role === "user") return messages.slice(0, -1);
  return messages;
}

/**
 * Take a reply and everything under it off, and open a fresh bubble in its place.
 *
 * The optimistic half of regenerating a reply anywhere in the thread. It is the
 * one transition here that does NOT key off the trailing message, so it re-checks
 * what it is about to remove against the state it is handed rather than trusting
 * the render that offered the button: taking a question off instead would delete
 * the very turn being re-asked. An index naming anything but a reply leaves the
 * transcript exactly as it was.
 */
export function replaceReplyAt(messages: Message[], index: number): Message[] {
  if (messages[index]?.role !== "assistant") return messages;
  // Timestamped at `done`, like a sent turn's reply (see stampTrailingId).
  return [...messages.slice(0, index), { role: "assistant", content: "" }];
}

/**
 * Reword a question, drop everything it had answered, and await a new reply.
 *
 * The optimistic half of editing. Unlike a regenerate this discards later
 * QUESTIONS too, because each of them was asked in a thread that no longer
 * exists as they saw it - the same rule the backend applies, so the two agree
 * without the client having to be told. Same re-check as above: an index naming
 * anything but a question changes nothing.
 *
 * The turn keeps its files and its own quote: an edit changes the words, not
 * what the question arrived with.
 */
export function editQuestionAt(
  messages: Message[],
  index: number,
  content: string,
): Message[] {
  const target = messages[index];
  if (target?.role !== "user") return messages;
  return [
    ...messages.slice(0, index),
    { ...target, content },
    { role: "assistant", content: "" },
  ];
}
