"use client";

import { Fragment } from "react";

import { Icon } from "@/components/chat/step";
import type { PullDone, PullProgress, VerifyDone, VerifyProgress } from "@/lib/model-manager";

/**
 * The one operation the page is running, as a single state object.
 *
 * One at a time is not a simplification: the backend is single-flight, and a
 * second download or check is refused before its stream opens. Modelling it as
 * one object is what lets every other button on the page disable off a single
 * flag, and what lets a download that continues into its automatic check render
 * as one continuous activity rather than as two.
 */
export interface Activity {
  /** What was asked for. A "pull" can still end up showing check frames. */
  kind: "pull" | "verify";
  model: string;
  /** The check depth, when a check was asked for directly. */
  depth: "fast" | "deep" | null;
  /** The stream is still open. Every other verb on the page waits on this. */
  running: boolean;
  /** The most recent download frame; null once the download finishes. */
  progress: PullProgress | null;
  /** The download's terminal frame, which says whether a check follows. */
  pull: PullDone | null;
  /** Finished check samples, in arrival order. */
  cases: VerifyProgress[];
  /** The check's verdict frame. */
  verdict: VerifyDone | null;
  /** The backend's OWN failure text - never a paraphrase of it. */
  failure: string | null;
  /** The reader paused the download; the partial is kept and resumes. */
  paused: boolean;
  /** The reader stopped the check; the run is discarded, the last verdict stands. */
  stopped: boolean;
}

/** Build the activity an operation starts from. */
export function newActivity(
  kind: Activity["kind"],
  model: string,
  depth: Activity["depth"] = null,
): Activity {
  return {
    kind,
    model,
    depth,
    running: true,
    progress: null,
    pull: null,
    cases: [],
    verdict: null,
    failure: null,
    paused: false,
    stopped: false,
  };
}

/**
 * Whether this activity is checking a model rather than downloading one.
 *
 * Deliberately NOT the same question as "has the download finished": a pull can
 * land and no check follow it at all, which is what happens every time a
 * non-chat model is pulled (the harness measures tool-calling, so the backend
 * skips the check and says why). Reading the two as one question left that
 * download rendering as an in-progress transfer forever.
 */
export function isChecking(activity: Activity): boolean {
  return (
    activity.kind === "verify" ||
    activity.cases.length > 0 ||
    activity.verdict !== null ||
    (activity.pull?.verifying ?? false)
  );
}

/** Whether this activity's weights are still transferring. */
export function isTransferring(activity: Activity): boolean {
  return activity.kind === "pull" && activity.pull === null;
}

/** Bytes as GB to one decimal - the one size unit this page uses. */
function gb(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * How far a check has got, as a percentage - or null before it can be said.
 *
 * Null until the first sample lands, and that is not pedantry: a check unloads
 * whatever is warm and cold-loads the model under test, which on a small card
 * takes minutes before sample one. A bar sitting at 0% through all of it reads
 * as a run that has stalled, where an indeterminate one reads as work
 * happening - which is what is true.
 */
function checkPercent(samples: VerifyProgress[]): number | null {
  const last = samples[samples.length - 1];
  if (last === undefined || last.total <= 0) return null;
  return (last.completed / last.total) * 100;
}

/**
 * What each check case is actually asking the model, in one line.
 *
 * The case names are identifiers, not English: nothing about "loop-guard"
 * says what passing it means, and a reader watching 26 samples go by has no
 * way to tell a case that matters from one that does not. One gloss per BATCH
 * of samples, never per sample - the harness runs each case's samples
 * together, so the line reads as a heading over the rows it describes.
 *
 * Display-side, and it must stay that way: the harness's own case specs are
 * hashed into the version every stored verdict is keyed on, so a word added
 * to them there would stale every verdict in the log. A case name this build
 * does not know renders NOTHING - a placeholder would be a sentence the page
 * invented about a case it cannot describe.
 */
const CASE_GLOSSES: Record<string, string> = {
  "think-on":
    "Reasoning asked for - it must reason on its own channel and still answer cleanly.",
  "think-off":
    "Reasoning switched off - whatever it does, no raw <think> markup may reach the answer.",
  "must-call-freshness": "A time-sensitive question only a live web search can answer.",
  "must-call-workspace": "A question about stored records that needs a knowledge search.",
  "must-not-call": "Trivial math - calling any tool here means it over-triggers everywhere.",
  selection: "Both tools offered - it must pick the knowledge search, and only that.",
  "hallucination-guard":
    "A web-shaped question with only knowledge search offered - inventing an unoffered tool is the one hard fail.",
  continuation: "It must request a search, then actually answer from the result it gets back.",
  "loop-guard": "The same exchange, scored on stopping - endlessly chasing tools fails.",
  "thinking-with-tools":
    "Reasoning switched on - it must both make the tool call and still produce an answer.",
};

/** One case's run of samples, as they arrived. */
interface CaseBatch {
  case: string;
  samples: VerifyProgress[];
}

/**
 * Split the flat sample stream into the batches it arrived in.
 *
 * Grouped by CONSECUTIVE name rather than by name outright: the harness loops
 * cases outer and samples inner, so consecutive is what the wire actually
 * guarantees, and collecting a name that reappeared later into an earlier
 * group would reorder the run on screen. A case that legitimately ran twice
 * simply gets two batches, which is what happened.
 */
function caseBatches(samples: VerifyProgress[]): CaseBatch[] {
  const batches: CaseBatch[] = [];
  for (const sample of samples) {
    const last = batches[batches.length - 1];
    if (last !== undefined && last.case === sample.case) last.samples.push(sample);
    else batches.push({ case: sample.case, samples: [sample] });
  }
  return batches;
}

/** Props for {@link ActivityPanel}. */
export interface ActivityPanelProps {
  activity: Activity;
  /** Pause the download; only ever offered while one is transferring. */
  onCancel: () => void;
  /** Start the paused or failed download again; it resumes where it stopped. */
  onResume: () => void;
  /** Clear a finished activity from the page. */
  onDismiss: () => void;
}

/**
 * The running operation, rendered where the page can only ever have one.
 *
 * A download shows its stage wording and a progress bar; a check shows each
 * finished sample as it lands. When a download continues into its automatic
 * check the panel simply moves from the first rendering to the second on the
 * same stream, because it is the same activity.
 */
export function ActivityPanel({ activity, onCancel, onResume, onDismiss }: ActivityPanelProps) {
  const checking = isChecking(activity);
  const transferring = isTransferring(activity);
  // A check that has its verdict (or failed) is only putting the models back.
  const settled = activity.verdict !== null || activity.failure !== null;
  const percent = activity.progress?.percent ?? null;
  const completed = activity.progress?.completed ?? null;
  const total = activity.progress?.total ?? null;
  const percentDone = checkPercent(activity.cases);
  // An unfinished transfer - paused or died - can always be started again:
  // the download is content-addressed, so "resume" and "retry" are the same
  // request and the same button.
  const resumable = transferring && !activity.running && (activity.paused || activity.failure !== null);

  return (
    <section
      data-part="mm-activity"
      className="mm-activity"
      // The theme's running tell (aurora's comet round the panel) keys on this.
      data-running={activity.running ? "" : undefined}
    >
      <div className="mm-activity-head">
        <h2 className="mm-activity-title">
          {checking
            ? activity.stopped
              ? "Stopped "
              : settled
                ? "Checked "
                : "Checking "
            : activity.paused
              ? "Paused "
              : transferring
                ? "Downloading "
                : "Downloaded "}
          <code className="mm-tag">{activity.model}</code>
        </h2>
        {/* A download pauses (content-addressed, it resumes from where it
            stopped); a check can only be stopped. Stopping is an abort of the
            stream, and the backend's abort path puts the chat model and the
            embedder back exactly as its terminal frame does, so the hint
            below is true. */}
        {/* A check's stream outlives its verdict by the re-warm it does on the
            way out; there is nothing left to stop then, so the verb goes with
            the hint. */}
        {activity.running && !(checking && settled) && (
          <button type="button" onClick={onCancel} className="btn btn-ghost">
            {checking ? "Stop" : "Pause"}
          </button>
        )}
        {!activity.running && (
          <span className="mm-activity-verbs">
            {resumable && (
              <button type="button" onClick={onResume} className="btn btn-primary">
                Resume
              </button>
            )}
            <button type="button" onClick={onDismiss} className="btn btn-ghost">
              Dismiss
            </button>
          </span>
        )}
      </div>

      {transferring && (
        <>
          {/* the paused sentence below is the one note then */}
          {!activity.paused && (
            <p className="mm-note" aria-live="polite">
              {activity.progress?.status ?? activity.pull?.status ?? "Starting..."}
            </p>
          )}
          <div
            className={`mm-bar${percent === null ? " indeterminate" : ""}`}
            role="progressbar"
            aria-label={`Downloading ${activity.model}`}
            aria-valuemin={0}
            aria-valuemax={100}
            // Omitted while a stage reports no transfer: an indeterminate bar
            // must not claim a value, and React drops an undefined attribute.
            aria-valuenow={percent === null ? undefined : Math.round(percent)}
          >
            <i style={percent === null ? undefined : { width: `${percent}%` }} />
          </div>
          {/* Only while it runs: the paused sentence below carries the same
              numbers, and saying them twice was the round's own finding. */}
          {activity.running && (
            <p className="mm-meta">
              {percent === null
                ? "This stage reports no transfer, so there is no progress to show."
                : `${percent.toFixed(1)}%${
                    completed !== null && total !== null
                      ? ` - ${gb(completed)} of ${gb(total)}`
                      : ""
                  }`}
            </p>
          )}
          {/* Only while it runs: after a pause this line and the paused note
              below would say the same thing in the same breath. */}
          {activity.running && (
            <p className="mm-hint">
              Pausing is safe: the download is content-addressed and resumes
              from where it stopped.
            </p>
          )}
        </>
      )}

      {/* Outside the checking gate: a download that lands with no check after
          it is the ordinary outcome for every non-chat model, and the reason
          the backend gives for skipping one is the whole answer there. */}
      {activity.pull && (
        <p className="mm-note ok">
          Downloaded.
          {activity.pull.verifying
            ? " A quick tool check follows on the same connection."
            : activity.pull.verify_skipped_reason
              ? ` No check ran: ${activity.pull.verify_skipped_reason}`
              : ""}
        </p>
      )}

      {checking && (
        <>
          {/* The bar retires with the verdict, exactly as the download's does
              when its terminal frame lands: past that point the tallies are
              the answer and a full bar is one more thing to read. A check that
              died mid-run keeps its bar frozen where it stopped, which is the
              one place a part-finished fraction is still worth having - but a
              run that died before its first sample has no fraction to freeze,
              and an indeterminate bar over a failure message would claim work
              is still happening.

              The failure test is separate from `running` deliberately: the
              backend yields its error frame and only THEN re-warms the chat
              model and the embedder, which is minutes on a cold card, so the
              stream - and `running` with it - outlives the failure by a long
              way. */}
          {activity.verdict === null &&
            (activity.cases.length > 0 || (activity.running && activity.failure === null)) && (
              <div
                className={`mm-bar${percentDone === null ? " indeterminate" : ""}`}
                role="progressbar"
                aria-label={`Checking ${activity.model}`}
                aria-valuemin={0}
                aria-valuemax={100}
                // Omitted before the first sample: an indeterminate bar must
                // not claim a value, and React drops an undefined attribute.
                aria-valuenow={percentDone === null ? undefined : Math.round(percentDone)}
              >
                <i style={percentDone === null ? undefined : { width: `${percentDone}%` }} />
              </div>
            )}
          {activity.cases.length > 0 && (
            <>
              <p className="mm-meta" aria-live="polite">
                {`${activity.cases[activity.cases.length - 1].completed}/${
                  activity.cases[activity.cases.length - 1].total
                } samples`}
              </p>
              <ul className="mm-cases">
                {caseBatches(activity.cases).map((batch, batchIndex) => (
                  <Fragment key={`${batch.case}-${batchIndex}`}>
                    {/* Once per batch, above its samples. Not aria-hidden: it
                        is the only thing on the page that says what the case
                        below it means, and hiding it would leave a screen
                        reader with the identifier alone. role="presentation"
                        keeps the prose readable while excluding it from the
                        list's item count, so a deep run still announces as 26
                        samples rather than 36 items. */}
                    {CASE_GLOSSES[batch.case] && (
                      <li role="presentation" className="mm-case-gloss">
                        {CASE_GLOSSES[batch.case]}
                      </li>
                    )}
                    {batch.samples.map((sample, index) => (
                      <li
                        // The same case runs several times on a deep check, so
                        // the index is the only stable key here.
                        key={`${batch.case}-${batchIndex}-${index}`}
                        className={`mm-case${sample.passed ? " pass" : " fail"}`}
                      >
                        <Icon
                          name={sample.passed ? "check" : "x"}
                          className="i mm-case-glyph"
                        />
                        <span>
                          <span className="mm-case-name">{sample.case}</span>
                          <span className="mm-case-word">
                            {sample.passed ? "pass" : "fail"}
                          </span>
                        </span>
                        {/* A pass is a time; a failure is the backend's reason. */}
                        <span className="mm-case-detail">
                          {sample.passed ? `${sample.latency_s.toFixed(1)} s` : sample.detail}
                        </span>
                      </li>
                    ))}
                  </Fragment>
                ))}
              </ul>
            </>
          )}
          {activity.verdict && <Verdict verdict={activity.verdict} />}
          {activity.running && activity.verdict && (
            <p className="mm-hint">
              The check is putting the chat model and the embedder back now. That
              takes a few seconds and needs nothing from here.
            </p>
          )}
          {/* Only while the samples land: what Stop would cost, said beside it. */}
          {activity.running && !settled && (
            <p className="mm-hint reason">
              Stopping discards this run: the last verdict stands, and the chat
              model and the embedder are put back.
            </p>
          )}
          {activity.stopped && (
            <p className="mm-note">
              Stopped. The last verdict stands; the chat model and the embedder
              are being put back.
            </p>
          )}
        </>
      )}

      {activity.paused && (
        <p className="mm-note">
          {`Paused${percent === null ? "" : ` at ${percent.toFixed(1)}%`}`}
          {completed !== null && total !== null ? ` - ${gb(completed)} of ${gb(total)}` : ""}
          {". Resume continues from where it stopped."}
        </p>
      )}
      {activity.failure && (
        <p className="mm-note error">
          {checking ? "Check failed. " : "Download failed. "}
          {activity.failure}
        </p>
      )}
    </section>
  );
}

/** The verdict frame: the word, the per-case tallies, then any caveat. */
function Verdict({ verdict }: { verdict: VerifyDone }) {
  const tallies = Object.entries(verdict.tallies);
  const observed = [
    verdict.tokens_per_second === null ? null : `${verdict.tokens_per_second.toFixed(1)} tok/s`,
    verdict.residency.vram_bytes === null ? null : `${gb(verdict.residency.vram_bytes)} VRAM`,
    verdict.runtime_version === null ? null : `runtime ${verdict.runtime_version}`,
  ]
    .filter((part) => part !== null)
    .join(" - ");
  return (
    <>
      {/* Tone follows the outcome: a run that earned nothing above the safe
          floor is a real answer, and painting it the same green as a passing
          one would make the word the only honest channel left. */}
      <p className={`mm-note${verdict.earned === "deterministic" ? "" : " ok"}`}>
        {`Earned ${verdict.earned}`}
        {verdict.thinking_with_tools ? ", and it reasons while calling tools" : ""}
        {` (${verdict.depth} check).`}
      </p>
      {/* The think mode is a MEASUREMENT and not part of the verdict, so it
          gets its own neutral line rather than a clause on the one above:
          none of the three modes is a better or worse thing for weights to
          be, and tinting it would read as a grade. What to DO about a
          measurement that disagrees with the declaration belongs on the
          model's card, which holds both sides of that comparison. */}
      {verdict.measured_think_mode && (
        <p className="mm-meta">
          {`Think flag measured: ${verdict.measured_think_mode}`}
          {verdict.measured_think_mode === "always"
            ? " - it reasons even when the flag is cleared."
            : verdict.measured_think_mode === "never"
              ? " - it produced no reasoning either way."
              : " - it reasons only when asked."}
        </p>
      )}
      {tallies.length > 0 && (
        <ul className="mm-tallies">
          {tallies.map(([name, [passed, run]]) => (
            // The same gloss the running list showed, as hover text: the
            // verdict is a scan grid and a sentence per chip would bury the
            // numbers it exists to line up.
            //
            // A `title` on a non-interactive element reaches neither keyboard
            // nor touch, so this is a mouse-only affordance and nothing more.
            // It is acceptable here only because the tallies READ without it -
            // the chip is a case name and two numbers - and not because the
            // reader already saw the glosses go by, which is false for anyone
            // who loaded the page after the run.
            <li key={name} className="mm-tally" title={CASE_GLOSSES[name]}>
              {name} <b>{`${passed}/${run}`}</b>
            </li>
          ))}
        </ul>
      )}
      {/* A FAILED case is the one chip whose gloss is load-bearing after the
          fact - it is what the reader came to understand - so it gets a quiet
          line of its own, reachable without a mouse. Clean chips keep hover
          only: repeating every gloss would bury the failures. */}
      {tallies
        .filter(([name, [passed, run]]) => passed < run && CASE_GLOSSES[name])
        .map(([name]) => (
          <p key={name} className="mm-meta">
            {`${name}: ${CASE_GLOSSES[name]}`}
          </p>
        ))}
      {/* A run the model spilled off the GPU for is recorded and can still
          clamp, but it certifies nothing - so its caveat is a line of its own,
          never a footnote on the verdict. */}
      {verdict.warning && <p className="mm-note warn">{verdict.warning}</p>}
      {!verdict.representative && !verdict.warning && (
        <p className="mm-note warn">
          This run was not representative: the model did not sit on the GPU, so
          it certifies nothing.
        </p>
      )}
      {/* Only what the run OBSERVED, and only the parts it observed: an empty
          line here would read as a measurement that came back blank. */}
      {observed !== "" && <p className="mm-meta">{observed}</p>}
    </>
  );
}
