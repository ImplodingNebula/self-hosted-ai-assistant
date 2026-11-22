"use client";

import { Fragment, useCallback, useEffect, useRef, useState } from "react";

import {
  type Activity,
  ActivityPanel,
  isChecking,
  newActivity,
} from "@/components/console/activity-panel";
import { ModelRow, StatusRow } from "@/components/console/model-row";
import { ConfirmDialog } from "@/components/confirm-dialog";
import { ApiError } from "@/lib/api-error";
import {
  AGENCY_GLOSS,
  THINKING_WITH_TOOLS_GLOSS,
  THINKING_WITH_TOOLS_HAZARD,
  type AgencyMode,
  type ManagerRow,
  type ManagerStreamEvent,
  type ModelManagerView,
  type PreflightResult,
  type PullProgress,
  type ResidencyView,
  type RoleGap,
  type RoleGapCode,
  canPartner,
  confirmThinkMode,
  declareCeiling,
  declareThinkingPair,
  declareThinkingWithTools,
  deleteModel,
  getManagerView,
  getResidency,
  preflightModel,
  setDefaultModel,
  streamPull,
  streamVerify,
  thinkModeOf,
  weightsChangeOf,
} from "@/lib/model-manager";
import { listModels } from "@/lib/models";
import { setTurnState } from "@/lib/turn-state";
import type { ModelServerStatus } from "@/lib/types";

/** Human line for a failed call; ApiError carries the problem+json detail. */
function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Is the backend running?";
}

/** Bytes as GB to one decimal - the one size unit this page uses. */
function gb(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * How much of a resident model is on the GPU, in words a reader can act on.
 *
 * Three states and not a percentage everywhere, because the percentage only
 * carries meaning in the middle one. Fully resident is the expected case and
 * says so; a full spill is "on CPU", which is the sentence that explains an
 * order-of-magnitude slowdown far better than "0% GPU" does; and a partial
 * split is where the actual number is the fact - it is ordinary on a small
 * card, it is not a fault, and it is what a reader compares against the next
 * model they are thinking of loading.
 *
 * A runtime that reports nothing gets nothing rendered. Filling in "100%" for
 * an unreported reading would be the one kind of confident wrongness this page
 * exists to avoid.
 */
function gpuShare(fraction: number | null): string {
  if (fraction === null) return "";
  if (fraction < 0.05) return " - on CPU";
  if (fraction > 0.99) return " - fully on GPU";
  return ` - ${Math.round(fraction * 100)}% on GPU`;
}

/**
 * The scan line for each gap: the consequence, in the fewest words that carry
 * it.
 *
 * A summary of the backend's own message and never a claim beyond it - the
 * sentence underneath is what explains and what proposes a fix. Keyed on the
 * code rather than the severity because the pairs are fixed in opposite ways,
 * and a reader who acts on the wrong one downloads weights that change nothing.
 */
const GAP_TITLES: Record<RoleGapCode, string> = {
  embedder_missing: "Retrieval is unavailable - the embedding model is not here",
  embedder_unselected: "Retrieval is unavailable - no embedding model is selected",
  tools_incapable: "No installed model can take tools",
  tools_clamped: "No installed model is cleared to use tools",
};

/**
 * The scan line for a gap, surviving a code this build has not learned.
 *
 * The map above is exhaustive over the KNOWN codes; the wire is not a closed
 * set, and a backend that ships a fifth code before this frontend learns it
 * must still get a title over its sentence - severity is the one axis every
 * code carries, so the fallback reads off that.
 */
function gapTitle(gap: RoleGap): string {
  const known = GAP_TITLES as Partial<Record<string, string>>;
  return (
    known[gap.code] ??
    (gap.severity === "fatal"
      ? "A model role is unavailable"
      : "The inventory could work better")
  );
}

/**
 * Sentence-case a role name for a section heading.
 *
 * The empty role is a real group and not a leftover bin: it holds the tags the
 * runtime's report says can neither complete nor embed, which is a fact about
 * them rather than a gap in anything.
 */
function roleTitle(role: string): string {
  if (role === "") return "Other";
  return role.charAt(0).toUpperCase() + role.slice(1);
}

/**
 * Installed rows first, each group's own order kept within the two halves.
 *
 * Every row the server sends is an installed tag, so this changes nothing
 * today; it stays because the grid answers "what is on this machine" and a
 * payload that ever carried a row that is not must not scatter the actionable
 * ones through it. Stable within each half deliberately: the order the rows
 * arrived in is the model server's own listing order, and re-sorting inside a
 * half would throw that away to no purpose.
 */
function installedFirst(rows: ManagerRow[]): ManagerRow[] {
  return [...rows].sort((a, b) => Number(b.installed) - Number(a.installed));
}

/** One option in a dialog's picker. */
interface Choice {
  value: string;
  label: string;
  /** The consequence of picking this, in plain language; omitted where obvious. */
  hint?: string;
}

/** A question the page must ask before it acts. */
interface Pending {
  title: string;
  body: string;
  confirmLabel: string;
  /** Confirm-button hue: "accent" for consequential-but-not-destructive. */
  tone: "danger" | "accent";
  /**
   * Deleting the workspace default only: who could take over as the default.
   * Present-and-empty means nobody can, which withholds the confirm button -
   * the body says to pull another chat model first. Absent on every other
   * question.
   */
  successors?: string[];
  /**
   * A choice the dialog collects before it can act, for the questions whose
   * answer IS a value rather than a yes.
   *
   * Deliberately separate from `successors` even though both render a select:
   * a successor is a consequence of the action being confirmed (and an empty
   * list withholds the confirm), while this is the action's own subject.
   * Collapsing them would make "no options" mean two different things.
   */
  choice?: { label: string; options: Choice[] };
  /**
   * A caution that has to be READ rather than skimmed - why the current state
   * is what it is, or what the action will cost. Rendered apart from the body
   * so it keeps the warn slot the console reserves for one per subject.
   */
  caution?: string;
  /**
   * The same warn slot, for a dialog whose caution depends on what is
   * SELECTED rather than on the row alone.
   *
   * A function rather than a value because `pending` is captured when the
   * dialog opens and the selection moves afterwards: a fixed string would go
   * on explaining a choice the reader has since changed away from.
   */
  cautionFor?: (picked: string) => string | null;
  /**
   * The question cannot be answered at all, so Confirm is withheld and the
   * body explains what would make it answerable.
   *
   * Stated by the caller rather than inferred from an absent `choice`: a
   * dialog with no picker is the ORDINARY shape (a delete, a check), and
   * deriving "unanswerable" from that would withhold Confirm on every plain
   * yes/no question in the page.
   */
  unanswerable?: boolean;
  run: (picked?: string) => void;
}

/**
 * The Model Manager: what this deployment holds, what it is running on, and
 * what anything has actually measured about either.
 *
 * An INVENTORY, so every row is a tag the model server really has. Every number
 * here was either reported by the runtime, declared to this deployment, or
 * observed by a probe, and the page says which. Nothing is predicted - there is
 * no fit formula and no ranking, because a prediction that is wrong on the
 * machine it ships on is worse than an honest "unmeasured".
 */
export default function ModelManagerPage() {
  const [view, setView] = useState<ModelManagerView | null>(null);
  const [residency, setResidency] = useState<ResidencyView | null>(null);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState<string | null>(null);

  // ONE operation at a time, because the backend is single-flight: a second
  // download or check is refused before its stream opens.
  const [activity, setActivity] = useState<Activity | null>(null);
  const abortRef = useRef<AbortController | null>(null);
  const [pending, setPending] = useState<Pending | null>(null);
  // Whatever the open dialog's select is showing - a delete's successor, or a
  // declaration's own value. Page state rather than dialog state so onConfirm
  // reads the live pick, not the value captured when the dialog was opened.
  const [picked, setPicked] = useState("");
  // Where the models are served from, for the standing notice below. Its own
  // read and its own failure: the console must not go dark because one banner's
  // source did not answer.
  const [server, setServer] = useState<ModelServerStatus | null>(null);

  // Transfers that stopped short of finishing, by model: what their rows
  // offer Resume against, and how far each got. Session-scoped by nature -
  // the runtime has no API that lists partial downloads, so the page can only
  // remember the ones it watched stop.
  const [pausedPulls, setPausedPulls] = useState<Record<string, PullProgress>>({});
  const lastProgressRef = useRef<PullProgress | null>(null);
  const pullDoneRef = useRef(false);

  // The free-text pull box: a name, its preflight answer, and any refusal.
  const [draft, setDraft] = useState("");
  const [preflight, setPreflight] = useState<PreflightResult | null>(null);
  const [preflightError, setPreflightError] = useState<string | null>(null);
  const [preflightBusy, setPreflightBusy] = useState(false);
  // Bumped on every lookup AND every edit, so a slow answer can never land on
  // a question that has since changed.
  const preflightSeq = useRef(0);

  const busy = activity?.running ?? false;

  // Promise callbacks (not async/await) keep every setState out of the
  // synchronous call graph, which the set-state-in-effect lint rule rejects for
  // the mount-time call below (the same idiom as the integrations page).
  const refresh = useCallback(
    () =>
      getManagerView()
        .then((next) => {
          setView(next);
          setPageError(null);
        })
        .catch((err: unknown) => setPageError(errorMessage(err)))
        .finally(() => setLoading(false)),
    [],
  );

  // Residency is its own read and its own failure: the runtime not answering
  // what it holds must not take the whole page down with it.
  const refreshResidency = useCallback(
    () =>
      getResidency()
        .then(setResidency)
        .catch(() => setResidency(null)),
    [],
  );

  // Read once at mount, not on every operation: where the deployment points its
  // model server is deployment configuration, and nothing this page does moves
  // it. A failure leaves the notice unsaid rather than guessed at.
  const refreshServer = useCallback(
    () =>
      listModels()
        .then((listing) => setServer(listing.server ?? null))
        .catch(() => setServer(null)),
    [],
  );

  useEffect(() => {
    void refresh();
    void refreshResidency();
    void refreshServer();
  }, [refresh, refreshResidency, refreshServer]);

  // Leaving the page stops what it started. The backend's pull and check claims
  // are held for the life of the REQUEST, so an orphaned stream keeps the
  // single probe slot for minutes with nothing on screen saying so - and the
  // next check would be refused with a 409 nobody could explain.
  useEffect(() => () => abortRef.current?.abort(), []);

  // The sky follows the one long operation: amber while the runtime fetches,
  // the tool's teal while a model is being worked, rest otherwise (and on the
  // way out, so the chat does not inherit a console's weather).
  const running = activity?.running ?? false;
  const checking = activity !== null && isChecking(activity);
  useEffect(() => {
    setTurnState(!running ? "idle" : checking ? "tool" : "download");
  }, [running, checking]);
  useEffect(() => () => setTurnState("idle"), []);

  /** Fold one stream frame into the running activity. */
  const onEvent = useCallback((event: ManagerStreamEvent) => {
    // Bookkeeping for the resume story, outside the reducer: the last frame
    // is what a paused row shows, and the done frame is what retires it.
    if (event.type === "progress") lastProgressRef.current = event.data;
    if (event.type === "done") pullDoneRef.current = true;
    setActivity((current) => {
      if (current === null) return current;
      switch (event.type) {
        case "progress":
          return { ...current, progress: event.data };
        case "done":
          // The download's terminal frame. The progress bar goes with it: what
          // follows is either nothing or a check, and neither has bytes.
          return { ...current, pull: event.data, progress: null };
        case "error":
          return { ...current, failure: event.data.message };
        case "verify_progress":
          return { ...current, cases: [...current.cases, event.data] };
        case "verify_done":
          return { ...current, verdict: event.data };
        case "verify_error":
          return { ...current, failure: event.data.message };
      }
    });
  }, []);

  /**
   * Run one streamed operation start to finish, then re-read the page.
   *
   * A refusal decided before the stream (an unusable name, a probe already
   * running) arrives as a thrown ApiError carrying the backend's own sentence;
   * an abort is the reader's own Cancel and is reported as such, never as a
   * failure.
   */
  // The panel is the one thing on the page that changes while it is watched,
  // and the verb that starts it may sit a screen below it (a row's Run check,
  // a Pull at the foot of the list): the page glides to its top once the
  // panel is on it, as a cut under reduced motion. Only a START does this -
  // a frame landing or a verdict never moves the reader.
  const revealRef = useRef(false);
  useEffect(() => {
    if (!revealRef.current || activity === null) return;
    revealRef.current = false;
    const panel = document.querySelector<HTMLElement>('[data-part="mm-activity"]');
    const pane = panel?.closest(".console-scroll");
    if (!panel || !pane) return;
    // A panel already in view stays where it is; otherwise the page goes to
    // its top, where the panel sits under the title with the page's name
    // still in view (tucking the panel under the top edge lost that).
    const p = panel.getBoundingClientRect();
    const v = pane.getBoundingClientRect();
    if (p.top >= v.top && p.bottom <= v.bottom) return;
    const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    pane.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });
  }, [activity]);

  const runStreamed = useCallback(
    async (started: Activity, start: (signal: AbortSignal) => Promise<void>) => {
      const controller = new AbortController();
      abortRef.current = controller;
      lastProgressRef.current = null;
      pullDoneRef.current = false;
      revealRef.current = true;
      setActivity(started);
      try {
        await start(controller.signal);
        // A stream that ends without its terminal frame is a failure that
        // never introduced itself. Left silent, it rendered as a download
        // stuck at its last percent with nothing to act on - so the page
        // names it, and the Resume path picks up whatever landed on disk.
        setActivity((current) => {
          if (current === null || current.paused || current.failure !== null) return current;
          if (current.kind === "pull" && current.pull === null) {
            return {
              ...current,
              failure:
                "The connection closed before the download finished. " +
                "Resume continues from where it stopped.",
            };
          }
          if (
            current.verdict === null &&
            (current.kind === "verify" || (current.pull?.verifying ?? false))
          ) {
            return { ...current, failure: "The connection closed before the check finished." };
          }
          return current;
        });
      } catch (err) {
        const aborted = controller.signal.aborted;
        // The reader's own abort: a download pauses (its partial is kept), a
        // check is stopped (its run is discarded, the last verdict stands). A
        // check hung up on after its verdict was only putting the models back:
        // nothing was stopped, and it keeps the verdict it has.
        setActivity((current) =>
          current === null
            ? current
            : aborted
              ? isChecking(current)
                ? current.verdict !== null || current.failure !== null
                  ? current
                  : { ...current, stopped: true }
                : { ...current, paused: true }
              : { ...current, failure: errorMessage(err) },
        );
      } finally {
        abortRef.current = null;
        setActivity((current) => (current === null ? current : { ...current, running: false }));
        // Remember an unfinished transfer's last frame so its row can offer
        // Resume and show how far it got; forget it the moment one finishes.
        // Gated on a frame having ARRIVED: a refusal decided before the
        // stream left nothing on disk to resume.
        if (started.kind === "pull") {
          const partial = lastProgressRef.current;
          if (pullDoneRef.current) {
            setPausedPulls((all) => {
              if (!(started.model in all)) return all;
              const rest = { ...all };
              delete rest[started.model];
              return rest;
            });
          } else if (partial !== null) {
            setPausedPulls((all) => ({ ...all, [started.model]: partial }));
          }
        }
        // The grid carries install flags, digests and measured facts, all of
        // which an operation can have just changed; residency changes with any
        // load or eviction.
        await refresh();
        await refreshResidency();
      }
    },
    [refresh, refreshResidency],
  );

  const startPull = useCallback(
    (name: string) =>
      void runStreamed(newActivity("pull", name), (signal) =>
        streamPull(name, { verify: true, onEvent, signal }),
      ),
    [onEvent, runStreamed],
  );

  const startCheck = useCallback(
    (name: string, depth: "fast" | "deep") =>
      void runStreamed(newActivity("verify", name, depth), (signal) =>
        streamVerify(name, { depth, onEvent, signal }),
      ),
    [onEvent, runStreamed],
  );

  /** Ask before a check: it evicts the warm model and costs real GPU minutes. */
  const confirmCheck = (row: ManagerRow, depth: "fast" | "deep") =>
    setPending({
      title: depth === "deep" ? "Run the deep check?" : "Run a quick check?",
      body:
        depth === "deep"
          ? `The deep check runs every case three times against ${row.model}. It unloads whatever is warm, holds the GPU for several minutes, and puts the chat model and the embedder back afterwards. Chat is slow to answer until it does.`
          : `The quick check runs the two load-bearing cases against ${row.model}. It unloads whatever is warm, takes about a minute, and puts the chat model and the embedder back afterwards.`,
      confirmLabel: depth === "deep" ? "Run deep check" : "Run check",
      // Consequential (it evicts the warm model), not destructive: nothing is
      // lost that a re-warm does not restore.
      tone: "accent",
      run: () => startCheck(row.model, depth),
    });

  /** Ask before re-pulling installed weights: an update can reset the verdict. */
  const confirmPullUpdates = (row: ManagerRow) =>
    setPending({
      title: "Pull updates?",
      body:
        `Fetches whatever the registry now serves as '${row.model}'. If nothing changed, ` +
        "this costs one manifest read and finishes at once." +
        // The row's own answer, so the promise of a re-check is made exactly
        // where the backend would actually run one.
        (row.checkable
          ? " If the weights did change, the recorded tool check no longer applies - a " +
            "quick check runs after the download, so the verdict is re-earned on the new " +
            "weights rather than assumed."
          : ""),
      confirmLabel: "Pull updates",
      // The foot places this verb as the benign one; a blood-red confirm
      // would un-say that at the moment of commitment.
      tone: "accent",
      run: () => startPull(row.model),
    });

  /**
   * Ask before storing a measured think mode: it is a DECLARATION, and it
   * changes how every later turn routes this model until the weights move.
   *
   * The measurement and the declaration it disagrees with are both named in
   * the dialog, because "confirm" is only meaningful when a reader can see
   * what they are confirming over. Nothing here sends a mode - the backend
   * stores the measured one, so the body describes rather than chooses.
   */
  const confirmThinkModeFor = (row: ManagerRow) => {
    const note = thinkModeOf(row);
    if (note === null || note.measured === null) return;
    setPending({
      title: "Confirm the measured think mode?",
      body:
        `A tool check measured ${row.model} as '${note.measured}' on the weights ` +
        `installed now, where the declaration says '${note.declared}'. Confirming ` +
        "stores the measurement, so turns route on it from here on. It is bound to " +
        "these weights: pulling the tag again retires it, and a fresh check earns it " +
        "back.",
      confirmLabel: "Confirm",
      // Consequential, not destructive: it changes routing and can be
      // superseded by the next confirmation or a re-pull.
      tone: "accent",
      run: () => {
        confirmThinkMode(row.model)
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => void refresh());
      },
    });
  };

  /**
   * Why this model is held where it is, read from the evidence and never guessed.
   *
   * The backend already wrote the sentence - which case failed, that a re-pull
   * outran the verdict, that the check ran somewhere else - and repeating it
   * here in different words would be a second opinion about one model. What
   * this adds is only what the dialog needs and the card does not carry: that
   * raising a ceiling past what a check has proven changes nothing until one
   * does, and what earning it would cost.
   */
  const ceilingCaution = (row: ManagerRow, target: AgencyMode): string | null => {
    if (target !== "agentic") return null;
    const clamp = row.agency?.clamp_detail ?? null;
    const proven = row.agency?.earned === "agentic";
    if (proven) return null;
    const cost =
      "Storing this now is not wasted - it is remembered, and it starts applying the " +
      "moment a deep check passes on these weights. Running one evicts whatever is warm " +
      "and holds the GPU for several minutes.";
    return clamp === null ? cost : `${clamp} ${cost}`;
  };

  /**
   * Ask before declaring an agency ceiling: it is the axis with teeth.
   *
   * The dialog carries three things a bare confirm would not. What each word
   * MEANS, because "agentic" is not self-explanatory and the difference between
   * the three is what the deployment does when the model declines to look
   * something up. Why the model sits where it does right now, read from the
   * probe evidence the backend already explained. And, for a raise, that a
   * declaration alone moves nothing - the earned axis still gates.
   */
  const confirmDeclareCeiling = (row: ManagerRow) => {
    const current = row.agency?.ceiling ?? "deterministic";
    setPicked(current);
    const source = row.ceiling_declared
      ? "A declaration made here sets the current ceiling"
      : "The model server's own capability report sets the current ceiling";
    setPending({
      title: "Declare how much this model may drive",
      body:
        `${source} of '${current}' for ${row.model}. A ceiling is permission and not a ` +
        "promise: raising it withdraws the objection, and what the model may actually do " +
        "stays the lower of this and what a check has measured on the weights installed " +
        "now. Lowering it needs no evidence and applies to the next turn. The declaration " +
        "is bound to these weights - pulling the tag again retires it.",
      confirmLabel: "Declare",
      // Consequential, not destructive: it is superseded by the next
      // declaration and retired by a re-pull.
      tone: "accent",
      choice: {
        label: "Ceiling",
        options: (["deterministic", "assisted", "agentic"] as AgencyMode[]).map((mode) => ({
          value: mode,
          label: mode,
          hint: AGENCY_GLOSS[mode],
        })),
      },
      // Recomputed from the LIVE selection at render time, not captured here:
      // a caution about what earning `agentic` costs, left standing over a
      // `deterministic` selection, contradicts the body's own (correct) "this
      // one needs no evidence".
      cautionFor: (mode: string) => ceilingCaution(row, mode as AgencyMode),
      run: (mode?: string) => {
        // A confirm that changed nothing is a no-op, not a fresh declaration.
        // Storing one would look harmless and is not: the row rebinds to the
        // digest installed NOW, so re-confirming an unchanged ceiling silently
        // re-arms a declaration that a re-pull was about to retire.
        if (!mode || mode === current) return;
        declareCeiling(row.model, mode as AgencyMode)
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => void refresh());
      },
    });
  };

  /**
   * Why a grant would not take effect yet, read from the evidence and never guessed.
   *
   * The mirror of {@link ceilingCaution}, and gated on the same test-first rule:
   * a stored grant reaches a turn only once a DEEP check has passed the
   * reasoning case on these weights. Revoking is the safe direction and carries
   * no caution at all - it applies to the next turn, which is exactly what the
   * body already says.
   */
  const thinkingWithToolsCaution = (row: ManagerRow, target: string): string => {
    // The hazard leads on every answer, because it is what makes this axis
    // worth a dialog at all - a reader withdrawing the claim should see the
    // reason it exists as much as one granting it should.
    if (target !== "true" || row.thinking_with_tools_effective) {
      // Withheld, or already in force: nothing is being promised that a check
      // has not measured, so the test-first line would be noise.
      return THINKING_WITH_TOOLS_HAZARD;
    }
    // The MEASUREMENT, never the effective value. Reaching this branch says
    // only that the capability is not in force, and the commonest reason for
    // that is nobody having declared it yet - the resolution stops at the
    // missing declaration and never consults the log. Reading a failure out of
    // it describes the ordinary order (check, read the result, then declare)
    // exactly backwards, telling someone whose deep check just PASSED that
    // granting it changes nothing.
    const passed = row.measured?.thinking_with_tools_passed ?? null;
    if (passed === true) {
      return (
        `${THINKING_WITH_TOOLS_HAZARD} The deep check on these weights established it, so this ` +
        "grant takes effect immediately - the next turn that reasons will be offered tools."
      );
    }
    if (passed === false) {
      return (
        `${THINKING_WITH_TOOLS_HAZARD} The deep check on these weights did not establish it, so ` +
        "granting it changes nothing yet. The grant is remembered and starts applying the moment " +
        "a check does establish it. Running one evicts whatever is warm and holds the GPU for " +
        "several minutes."
      );
    }
    // Not established HERE - which is not the same as measured absent, and
    // must not be worded as one. Null covers a missing verdict, a verdict this
    // deployment cannot count, AND a row whose measurement was never read (a
    // deterministic ceiling resolves without touching the log). So state the
    // RULE that decides rather than a claim about what any check found: the
    // rule is true in all three, an assertion about evidence is not.
    const elsewhere = row.agency?.clamp_code === "endpoint_mismatch";
    return (
      `${THINKING_WITH_TOOLS_HAZARD} ` +
      (elsewhere
        ? "A check established this on another model server, which does not certify this one. "
        : "") +
      "A deep check on the weights installed here is what settles this: the grant is stored " +
      "either way, and it applies from the moment such a check has established it. Running one " +
      "evicts whatever is warm and holds the GPU for several minutes."
    );
  };

  /**
   * Ask before declaring reason-and-call-tools: the failure it guards is silent.
   *
   * The dialog carries what each answer MEANS for a turn, the hazard in plain
   * words (an unearned grant returns blank answers, not errors), and - for a
   * grant - that a declaration alone moves nothing until a deep check agrees.
   * The same test-first gate the ceiling's `agentic` rung sits behind, said out
   * loud rather than enforced by a disabled control: the backend stores the
   * declaration either way and the fold is what withholds it.
   */
  const confirmDeclareThinkingWithTools = (row: ManagerRow) => {
    const current = row.thinking_with_tools ? "true" : "false";
    setPicked(current);
    const source = row.thinking_with_tools_declared
      ? "A declaration made here"
      : "This deployment's own default";
    // Only a GRANT can differ from what turns do, so only a grant gets the
    // clause. Appending "and no turn acts on it yet" to "cannot reason" would
    // read as a second, contradictory claim about the same model.
    const state = row.thinking_with_tools
      ? `can reason and call tools in the same turn, and ${
          row.thinking_with_tools_effective ? "turns act on it" : "no turn acts on it yet"
        }`
      : "cannot reason and call tools in the same turn";
    setPending({
      title: "Declare whether this model reasons while calling tools",
      body:
        `${source} says ${row.model} ${state}. Granting it is permission and not a promise: ` +
        "what a turn actually does stays the lower of this and what a deep check has measured " +
        "on the weights installed now. Withdrawing it needs no evidence and applies to the " +
        "next turn. The declaration is bound to these weights - pulling the tag again retires it.",
      confirmLabel: "Declare",
      // Consequential, not destructive: superseded by the next declaration and
      // retired by a re-pull.
      tone: "accent",
      choice: {
        label: "Reasons while calling tools",
        options: (["true", "false"] as const).map((value) => ({
          value,
          label: value === "true" ? "Granted" : "Withheld",
          hint: THINKING_WITH_TOOLS_GLOSS[value],
        })),
      },
      // Recomputed from the LIVE selection, never captured: a caution about
      // what earning the grant costs, left standing over a withdrawal, would
      // contradict the body's own (correct) "this one needs no evidence".
      cautionFor: (value: string) => thinkingWithToolsCaution(row, value),
      run: (value?: string) => {
        // A confirm that changed nothing is a no-op, not a fresh declaration.
        // Storing one re-binds the row to the digest installed NOW, silently
        // re-arming a claim that a re-pull was about to retire.
        if (!value || value === current) return;
        declareThinkingWithTools(row.model, value === "true")
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => void refresh());
      },
    });
  };

  /**
   * Installed models that could serve this one's thinking turns.
   *
   * Drawn from what the rows say each model MEASURES as, through the one
   * predicate the backend refuses on - never from what anything declared about
   * pairing. A menu built from existing pairings would offer a partner because
   * somebody once picked it, which is the circular reasoning this whole axis
   * exists to avoid.
   *
   * {@link canPartner} and not `canReason`: a partner has to reply as well as
   * reason, and a vision model measured `always` passes the reasoning half
   * while answering a thinking turn with nothing at all.
   *
   * Every group is searched but the LOCKED ones are skipped: a locked role
   * cannot serve a chat turn at all, so offering one would build a pairing the
   * backend refuses. That deliberately keeps the empty-role group in - a tag
   * whose report establishes no role is manageable, and a chat-capable one
   * among them is exactly the partner a remote inventory has to offer.
   */
  const partnersFor = (row: ManagerRow): string[] => {
    if (view === null) return [];
    return view.groups
      .filter((g) => g.management === "manageable")
      .flatMap((g) => g.rows)
      .filter((candidate) => candidate.model !== row.model && canPartner(candidate))
      .map((candidate) => candidate.model);
  };

  /** Ask before pairing: it changes where every thinking turn on this model goes. */
  const confirmDeclarePair = (row: ManagerRow) => {
    const partners = partnersFor(row);
    setPicked(partners[0] ?? "");
    const held = row.thinking_model;
    setPending({
      title: "Choose a reasoning partner",
      body:
        partners.length === 0
          ? `Nothing installed has been established to reason, so ${row.model} has no partner to hand a thinking turn to. A model earns that by being measured - run a check against a model that reasons, or install one.`
          : `${row.model} answers directly. Picking a partner sends its thinking turns to that model instead, and leaves everything else on this one.` +
            (held === null
              ? ""
              : ` It currently uses ${held}${row.thinking_model_declared ? "" : ", from this deployment's own default"}.`) +
            " The pairing follows the tags rather than the weights, so it survives a re-pull.",
      confirmLabel: "Pair",
      tone: "accent",
      unanswerable: partners.length === 0,
      choice:
        partners.length === 0
          ? undefined
          : {
              label: "Thinking partner",
              options: partners.map((name) => ({ value: name, label: name })),
            },
      run: (partner?: string) => {
        if (!partner) return;
        declareThinkingPair(row.model, partner)
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => void refresh());
      },
    });
  };

  /** Ask before moving the workspace default: every unpinned thread follows it. */
  const confirmSetDefault = (row: ManagerRow) => {
    setPending({
      title: "Make this the default model?",
      body:
        `Every conversation that has not chosen a model of its own follows the workspace ` +
        `default, so this changes what they answer on from their next turn. Threads that ` +
        `pinned a model keep it. ${row.model} takes over.`,
      confirmLabel: "Set as default",
      tone: "accent",
      run: () => {
        setDefaultModel(row.model)
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => void refresh());
      },
    });
  };

  /**
   * Installed models that could take over as the workspace default: every row
   * of a manageable group, minus the one being deleted. Mirrors the backend's
   * own successor rule (the locked roles cannot drive a chat turn), so the
   * picker never offers a name the request would refuse.
   *
   * The empty-role group is manageable and is therefore included, deliberately:
   * a tag the runtime holds and describes no role for can still be the only
   * thing left that answers, and hiding it would offer a delete no successor
   * could complete.
   */
  const successorsFor = (row: ManagerRow): string[] => {
    if (view === null) return [];
    return view.groups
      .filter((group) => group.management === "manageable")
      .flatMap((group) => group.rows)
      .filter((r) => r.installed && r.model !== row.model)
      .map((r) => r.model);
  };

  /**
   * Ask before a delete: the weights have to be downloaded again to come back,
   * and the stored state that names the tag moves with it - pins are cleared,
   * and the workspace default hands over to a successor picked right here.
   */
  const confirmDelete = (row: ManagerRow) => {
    const successors = row.workspace_default ? successorsFor(row) : undefined;
    if (successors !== undefined) setPicked(successors[0] ?? "");
    const pins =
      row.pinned_conversations > 0
        ? (row.pinned_conversations === 1
            ? " 1 conversation is set to use it and"
            : ` ${row.pinned_conversations} conversations are set to use it and`) +
          " will follow the workspace default instead."
        : "";
    const body =
      successors === undefined
        ? `${row.model} will be removed from this machine.${pins} It can be downloaded again - which costs the whole transfer.`
        : successors.length > 0
          ? `${row.model} will be removed from this machine, and it is the workspace default - every conversation that has not chosen a model follows it. Pick what takes over.${pins}`
          : `${row.model} is the workspace default and the only installed model that could take over. Deleting it would leave the workspace unable to answer at all - pull another chat model first, then delete this one.`;
    setPending({
      title: "Delete these weights?",
      body,
      confirmLabel: "Delete",
      tone: "danger",
      successors,
      run: (successor?: string) => {
        deleteModel(row.model, successor)
          .then(() => setPageError(null))
          .catch((err: unknown) => setPageError(errorMessage(err)))
          .finally(() => {
            void refresh();
            void refreshResidency();
          });
      },
    });
  };

  /**
   * Ask the registry about the typed name; the answer gates the Pull button.
   *
   * Sequenced, because the answer is what Pull acts on: a lookup that resolves
   * after the box has been edited would re-arm Pull for a name nobody can see
   * any more, and pull THAT. Editing the box bumps the counter too, so an
   * in-flight lookup is stale the moment its question changes.
   */
  const checkAvailability = () => {
    const seq = (preflightSeq.current += 1);
    setPreflightBusy(true);
    setPreflight(null);
    setPreflightError(null);
    preflightModel(draft.trim())
      .then((result) => {
        if (seq === preflightSeq.current) setPreflight(result);
      })
      .catch((err: unknown) => {
        if (seq === preflightSeq.current) setPreflightError(errorMessage(err));
      })
      .finally(() => {
        if (seq === preflightSeq.current) setPreflightBusy(false);
      });
  };

  // Pulling stays possible for a name the registry could not answer for, as
  // long as the backend attached a warning to it: an hf.co path is never
  // checkable at all, and the warning is what the reader consents to.
  const pullable =
    preflight !== null &&
    (preflight.status === "available" ||
      (preflight.status === "unknown" && preflight.warning !== null));

  // The model whose bytes are moving right now, if any: its own row carries
  // the live readout, so the click that started it is never answered only by
  // a panel that may be scrolled far off-screen.
  const transferringModel =
    activity !== null && activity.running && activity.kind === "pull" && activity.pull === null
      ? activity.model
      : null;

  // The model being CHECKED right now, and its progress - the same story the
  // download above tells, for the other half of a stream.
  //
  // Read off the ACTIVITY's own model, never off a progress frame's. The
  // reasoning-with-tools case drives the pair's THINKING tag, so the frames
  // name two different models over one run, and a row bound to them would
  // hand its readout to a sibling card partway through the check. What the
  // reader clicked is what the panel titles itself with, and it never moves.
  //
  // Retired on verdict OR failure, not on stream end: the stream outlives the
  // check by the whole re-warm (verify_done lands, then the generator's
  // finally cold-loads chat + embedder back), and a row still reading
  // "Checking" beside a panel saying the check is over would contradict it
  // for that entire window. Same predicate the panel's own bar gate uses.
  const checkingModel =
    activity !== null &&
    activity.running &&
    isChecking(activity) &&
    activity.verdict === null &&
    activity.failure === null
      ? activity.model
      : null;
  const lastSample = activity?.cases[activity.cases.length - 1] ?? null;
  const checkProgress = {
    completed: lastSample?.completed ?? null,
    total: lastSample?.total ?? null,
  };

  const vectors = view?.vectors ?? null;
  const stranded = vectors?.stranded === true;
  // Leftovers from a completed embedder change: worth saying, never an alarm.
  const leftovers = vectors !== null && !vectors.stranded && vectors.message !== null;

  // Models whose installed weights outran their tool check, in the page's own
  // order. Derived from the rows already on screen rather than asked for
  // separately, so the banner can never name a model whose card says otherwise.
  //
  // No role filter: only a chat row is ever explained, so only a chat row can
  // carry the note (see _weights_change in model_manager.py). A second copy of
  // "which roles have checks" here is one copy too many - the day it drifted,
  // the banner would count rows the cards do not.
  const weightsChanged = (view?.groups ?? [])
    .flatMap((group) => group.rows)
    .filter((row) => weightsChangeOf(row) !== null)
    .map((row) => row.model);

  return (
    <div data-part="model-manager" className="mm-page">
      {/* What the inventory cannot do comes first, in the server's own order
          (worst first). Above the vectors alarm deliberately: a missing
          embedder is why stored vectors went out of reach, and the cause
          reads before its consequence. */}
      {view?.role_gaps.map((gap) => (
        <RoleGapNotice
          key={`${gap.role}:${gap.code}`}
          gap={gap}
          // Whether the box the card points at is on the page at all. The
          // suggestion is the backend's and carries no such gating, so without
          // this a provider with no download verb gets a card naming a section
          // that was deliberately not rendered.
          canPull={view.provider.can_pull}
        />
      ))}

      {/* The page-level notice OWNS the alarm; the embedding section explains
          quietly beneath it rather than raising it a second time. */}
      {stranded && vectors?.message && (
        <div className="mm-banner warn">
          <p>{vectors.message}</p>
        </div>
      )}
      {/* Beneath the gap cards and the alarm, and deliberately: those say a
          feature area is DEAD or that stored passages cannot be reached, while
          this says the deployment works and nobody has measured what it is
          working with. Reading top-down therefore goes broken, then
          unreachable, then merely unverified. Sliding it any higher would
          split the gap-cards-above-the-vectors-alarm adjacency, which is a
          recorded contract (cause above consequence) and worth more than this
          banner's own position.

          Above the LEFTOVERS note, though - that one is neutral by design
          ("worth saying, never an alarm"), and the two are mutually exclusive
          with the alarm, so this keeps a warn box from sitting under a quiet
          one in the only combination where both render. */}
      {weightsChanged.length > 0 && <WeightsChangedNotice models={weightsChanged} />}

      {/* Below the notices about what this deployment's models cannot do, and
          above the neutral leftovers note: it is a standing warn like the two
          alarms, but it describes the DEPLOYMENT rather than any answer the
          models are giving, so it must not push a broken feature area down. */}
      {server !== null && server.remote && !server.auth_configured && (
        <UnauthenticatedServerNotice host={server.host} />
      )}

      {leftovers && vectors?.message && (
        <div className="mm-banner">
          <p>{vectors.message}</p>
        </div>
      )}

      <header className="mm-header">
        <div className="mm-headline">
          <h1 className="mm-title">Model manager</h1>
          {view && (
            <p className="mm-meta">
              {view.provider.name}
              {view.provider.runtime_version ? ` ${view.provider.runtime_version}` : ""}
            </p>
          )}
        </div>
        <LoadedNow residency={residency} />
      </header>

      {pageError && <p className="mm-banner error">{pageError}</p>}

      {activity && (
        <ActivityPanel
          activity={activity}
          onCancel={() => abortRef.current?.abort()}
          onResume={() => startPull(activity.model)}
          onDismiss={() => setActivity(null)}
        />
      )}

      {loading && <p className="mm-muted">Loading...</p>}

      {/* Above the grid, not beneath it: the box is how anything the server
          does not already hold arrives at all, and at the foot of a long list
          it read as fine print. Hidden entirely rather than disabled when the
          backend has no download verb - an inert box is a dead promise. */}
      {view?.provider.can_pull && (
        <section className="mm-section">
          <h2 className="mm-section-title">Add a model</h2>
          <p className="mm-section-note">
            The registry cannot be browsed, so a model is named rather than
            picked. Checking availability asks the registry whether the name
            exists and what the download would cost.
          </p>
          <div className="mm-addbox">
            <input
              value={draft}
              onChange={(e) => {
                setDraft(e.target.value);
                // The answer belonged to the old name; keeping it would let a
                // typo be pulled on the strength of another name's check. The
                // counter retires any lookup still in flight for the same
                // reason - it would otherwise land and re-arm Pull.
                preflightSeq.current += 1;
                setPreflight(null);
                setPreflightError(null);
                setPreflightBusy(false);
              }}
              placeholder="qwen3:8b"
              aria-label="Model name"
              className="mm-input"
            />
            <button
              type="button"
              disabled={preflightBusy || busy || draft.trim() === ""}
              onClick={checkAvailability}
              className="btn btn-ghost"
            >
              {preflightBusy ? "Checking..." : "Check availability"}
            </button>
            <button
              type="button"
              disabled={!pullable || busy}
              onClick={() => startPull(preflight?.name ?? draft.trim())}
              className="btn btn-primary"
            >
              Pull
            </button>
          </div>
          {preflightError && <p className="mm-note error">{preflightError}</p>}
          {preflight && <PreflightNote result={preflight} />}
          <p className="mm-hint">
            A quick tool check runs automatically once a chat model finishes
            downloading, so its agency verdict is measured rather than assumed.
          </p>
        </section>
      )}

      {view?.groups.map((group) => (
        <section key={group.role} className="mm-section">
          <h2 className="mm-section-title">{roleTitle(group.role)}</h2>
          {group.lock_reason && <p className="mm-section-note">{group.lock_reason}</p>}
          {/* What this group IS, said once - otherwise "Other" is a heading a
              reader has to guess at. Kept to the fact and no further: the
              report establishes nothing about these tags, so anything about
              what they are good for would be the page's own invention. */}
          {group.role === "" && (
            <p className="mm-section-note">
              Models the server holds whose report establishes neither chat
              completion nor embedding, so nothing here describes what they
              are for.
            </p>
          )}
          {/* The alarm lives at the top of the page; here it is a pointer, at
              the volume a section note is read at. */}
          {group.role === "embedding" && stranded && (
            <p className="mm-section-note">
              Some stored vectors are out of reach of this embedder - see the
              note at the top of the page.
            </p>
          )}
          <ul className="mm-list">
            {installedFirst(group.rows).map((row) =>
              group.management === "status_only" ? (
                <StatusRow key={row.model} row={row} />
              ) : (
                <ModelRow
                  key={row.model}
                  row={row}
                  management={group.management}
                  busy={busy}
                  activePull={
                    transferringModel === row.model
                      ? { percent: activity?.progress?.percent ?? null }
                      : null
                  }
                  activeCheck={checkingModel === row.model ? checkProgress : null}
                  pausedPull={row.installed ? null : (pausedPulls[row.model] ?? null)}
                  onPull={(target) => startPull(target.model)}
                  onPullUpdates={confirmPullUpdates}
                  onCheck={confirmCheck}
                  onConfirmThinkMode={confirmThinkModeFor}
                  onDeclareCeiling={confirmDeclareCeiling}
                  onDeclareThinkingWithTools={confirmDeclareThinkingWithTools}
                  onDeclarePair={confirmDeclarePair}
                  onSetDefault={confirmSetDefault}
                  onDelete={confirmDelete}
                />
              ),
            )}
          </ul>
        </section>
      ))}

      {pending && (
        <ConfirmDialog
          title={pending.title}
          body={pending.body}
          confirmLabel={pending.confirmLabel}
          tone={pending.tone}
          confirmDisabled={
            (pending.successors !== undefined && pending.successors.length === 0) ||
            (pending.unanswerable ?? false)
          }
          onConfirm={() => {
            pending.run(
              pending.successors !== undefined || pending.choice !== undefined
                ? picked
                : undefined,
            );
            setPending(null);
          }}
          onCancel={() => setPending(null)}
        >
          {pending.successors !== undefined && pending.successors.length > 0 && (
            <label className="dialog-field">
              <span>New default model</span>
              <select value={picked} onChange={(e) => setPicked(e.target.value)}>
                {pending.successors.map((name) => (
                  <option key={name} value={name}>
                    {name}
                  </option>
                ))}
              </select>
            </label>
          )}
          {pending.choice && (
            <>
              <label className="dialog-field">
                <span>{pending.choice.label}</span>
                <select value={picked} onChange={(e) => setPicked(e.target.value)}>
                  {pending.choice.options.map((option) => (
                    <option key={option.value} value={option.value}>
                      {option.label}
                    </option>
                  ))}
                </select>
              </label>
              {/* The gloss for what is CURRENTLY selected, not a glossary of
                  all three: a dialog that explained every option at once would
                  be read as boilerplate and skipped, which is the opposite of
                  what a word like "agentic" needs. */}
              {pending.choice.options.find((option) => option.value === picked)?.hint && (
                <p className="dialog-hint">
                  {pending.choice.options.find((option) => option.value === picked)?.hint}
                </p>
              )}
            </>
          )}
          {/* The console reserves one warn slot per subject, and in a dialog
              the subject is the decision being made - so this is the only warn
              line here, and it carries the reason the clamp exists rather than
              repeating the body in a louder colour. */}
          {(pending.cautionFor?.(picked) ?? pending.caution ?? null) !== null && (
            <p className="dialog-caution">
              {pending.cautionFor?.(picked) ?? pending.caution}
            </p>
          )}
        </ConfirmDialog>
      )}
    </div>
  );
}

/**
 * One inventory gap, at the volume its severity earns.
 *
 * A gap is the ABSENCE of a row, so it has no card of its own to hang on and
 * lives at the top of the page instead. The two severities are two different
 * interruptions: a fatal gap means a whole feature area is dead until somebody
 * acts and takes the error tone and the assertive role; a note means the
 * deployment works and could work better, and reads at the volume of every
 * other page banner. Only the title takes the hue - a whole paragraph in the
 * danger colour is harder to read than the state it is warning about, and the
 * wording carries the meaning without it.
 */
function RoleGapNotice({ gap, canPull }: { gap: RoleGap; canPull: boolean }) {
  const fatal = gap.severity === "fatal";
  return (
    <div
      className={`mm-banner mm-gap${fatal ? " error" : ""}`}
      // Only the fatal one interrupts: a screen reader that announced every
      // note over whatever the reader was doing would train them to ignore
      // the one announcement that matters.
      role={fatal ? "alert" : undefined}
    >
      <p className="mm-gap-title">{gapTitle(gap)}</p>
      {/* The backend's own sentence, whole: it names what stopped working AND
          what fixes it, and paraphrasing it here would drop one of the two. */}
      <p className="mm-gap-body">{gap.message}</p>
      {/* Named, not offered as a button. The grid lists what the server HOLDS,
          so a model that would close a gap has no row here to pull from - the
          free-text box takes the name, and a second control for the same
          single-flight download would be a race with nothing to gain. Where the
          fix is a CHOICE rather than a download the backend sends no
          suggestion, and its own sentence above points at the workspace
          defaults instead.

          The box is named only where it EXISTS: a provider with no download
          verb hides that section entirely, and the backend's suggestion is not
          gated on the same thing - so without this the card would send a
          reader to a section that was deliberately never rendered. The tag is
          still worth naming there; it is what an operator downloads by
          whatever means that deployment actually has. */}
      {gap.suggested_model && (
        <p className="mm-gap-fix">
          Closed by downloading <code className="mm-tag">{gap.suggested_model}</code>
          {canPull ? " under Add a model" : ""}
        </p>
      )}
    </div>
  );
}

/**
 * The standing notice that some installed weights outran their tool check.
 *
 * Discovery, not explanation: the sentence a reader needs in order to know
 * there is anything to look at, with the model's own card carrying the detail.
 * A collapsed grid hides the row notes entirely, so without this an admin only
 * learns of a re-pull's effect by expanding the right card by chance.
 *
 * The stranded-vectors alarm's pattern verbatim (`mm-banner warn`, one
 * paragraph), because it is the same kind of statement: a standing fact about
 * the deployment that no single row owns. Note severity and NO `role="alert"` -
 * the console contract reserves interruption for a fatal gap, and nothing here
 * is broken. The models still answer; what is missing is a measurement.
 *
 * It says weights CHANGED and never how or why. A re-pull, a registry update
 * and a hand-moved tag are indistinguishable from a digest, and there is no
 * ledger of who moved what - so any word implying tampering would be a claim
 * this deployment cannot make.
 *
 * It also stops short of naming the check to run, which the CARD does. A row
 * can have a current verdict at one depth and an orphaned one at another - a
 * re-pull runs the quick check automatically, so a deep verdict is routinely
 * the only thing left behind - and a banner promising "run a check" would send
 * a reader to a button that clears nothing. Only the row knows which depth is
 * missing.
 */
function WeightsChangedNotice({ models }: { models: string[] }) {
  const many = models.length > 1;
  return (
    <div className="mm-banner warn">
      <p>
        {`Installed weights changed under ${many ? `${models.length} models` : "one model"}: `}
        {/* Named rather than counted alone: the tags are what a reader acts
            on, and in the monospace face the page uses for every tag. Not
            truncated to the first few either - a hidden name is a model
            nobody goes and checks. */}
        {models.map((model, index) => (
          <Fragment key={model}>
            {index > 0 && ", "}
            <code className="mm-tag">{model}</code>
          </Fragment>
        ))}
        {many
          ? ". A recorded tool-check verdict on each was earned on the weights they replaced, and each card names that verdict and the check that re-earns it."
          : ". A recorded tool-check verdict on it was earned on the weights it replaced, and its card names that verdict and the check that re-earns it."}
      </p>
    </div>
  );
}

/**
 * The standing notice that this deployment reaches its model server unprotected.
 *
 * A statement about the DEPLOYMENT, not about any model or any answer, which is
 * why it is a page banner and why no row carries it. The model server
 * authenticates nothing on its own: anything that can route to that host can
 * load models onto it, evict what is warm, and read every prompt sent through
 * it - so a remote host reached with no credential is a fact worth saying once,
 * plainly, and leaving on screen.
 *
 * Warn severity and no `role="alert"`, on the same rule the weights-change
 * notice follows: the console reserves interruption for a fatal gap, and
 * nothing here is broken. What is missing is a credential, not a capability.
 *
 * Deliberately silent about how to fix it. The fix is a deployment credential
 * whose name and shape depend on how the host is fronted, and a console
 * confidently naming one setting would send an operator to edit the wrong file.
 */
function UnauthenticatedServerNotice({ host }: { host: string }) {
  return (
    <div className="mm-banner warn">
      <p>
        {"The model server "}
        {/* Named where the deployment knows it, in the same monospace face
            every other address and tag on this page uses. Unnamed rather than
            guessed at when configuration did not say. */}
        {host === "" ? "this deployment uses" : <code className="mm-tag">{host}</code>}
        {
          " is on another host and is reached without credentials. It authenticates nothing itself, so anything that can reach it can load and remove models there and can read what is sent to it."
        }
      </p>
    </div>
  );
}

/** The registry's answer about a typed name, rendered fact by fact. */
function PreflightNote({ result }: { result: PreflightResult }) {
  const word =
    result.status === "available"
      ? "Available"
      : result.status === "not_found"
        ? "Not found"
        : "Not checkable";
  return (
    <div className="mm-preflight">
      <p className={`mm-note${result.status === "available" ? " ok" : ""}`}>
        <strong>{word}</strong>
        {result.download_size_bytes !== null
          ? ` - ${gb(result.download_size_bytes)} to download`
          : ""}
        {/* What the size MEANS, where the registry says. Two tags of one model
            can be a gigabyte apart on quantization alone, so a byte count with
            neither number beside it is a figure nobody can weigh against the
            card it has to fit on. Each is present only when published - an
            absent one is left out rather than rendered as "unknown", which
            would invent a gap where the registry simply says nothing. */}
        {result.parameter_size !== null ? ` - ${result.parameter_size} parameters` : ""}
        {result.quantization !== null ? ` - ${result.quantization}` : ""}
      </p>
      {/* The registry's own explanation is load-bearing (it is the only one
          the reader gets), so it takes the AA tier, not the decorative one. */}
      {result.detail && <p className="mm-hint reason">{result.detail}</p>}
      {/* The hf.co caveat, and anything like it: the reader consents to this
          by pulling anyway, so it is never folded into the line above. */}
      {result.warning && <p className="mm-note warn">{result.warning}</p>}
    </div>
  );
}

/** What the runtime holds in memory right now, beside the page's title. */
function LoadedNow({ residency }: { residency: ResidencyView | null }) {
  if (residency === null) return null;
  return (
    <div className="mm-loaded">
      <h2 className="mm-loaded-title">Loaded now</h2>
      {!residency.reported ? (
        // Not the same answer as "nothing is loaded", and saying so is the
        // whole point of the flag.
        <p className="mm-muted">This runtime does not report what it holds.</p>
      ) : residency.models.length === 0 ? (
        <p className="mm-muted">Nothing loaded.</p>
      ) : (
        <ul className="mm-loaded-list">
          {residency.models.map((entry) => (
            <li key={entry.model} className="mm-loaded-row">
              <span className="ms-dot" aria-hidden="true" />
              <code className="mm-tag">{entry.model}</code>
              <span className="mm-meta">
                {entry.vram_bytes === null ? "VRAM unreported" : `${gb(entry.vram_bytes)} VRAM`}
                {/* How much of the model is actually on the card. A partial
                    split is the ordinary state on a small GPU, not a fault -
                    and it is the single number that explains why a model that
                    "is loaded" answers at a fraction of its usual rate. Read
                    off the runtime's own reading; absent stays absent rather
                    than becoming a confident 100%. */}
                {gpuShare(entry.gpu_fraction)}
              </span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}
