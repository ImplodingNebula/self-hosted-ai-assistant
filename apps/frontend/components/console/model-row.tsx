"use client";

import { useState } from "react";

import {
  AGENCY_GLOSS,
  type ManagerRow,
  type MeasuredFacts,
  type PullProgress,
  type RoleGroup,
  thinkMismatchOf,
  thinkModeOf,
  weightsChangeOf,
} from "@/lib/model-manager";

/** Bytes as GB to one decimal - the one size unit this page uses. */
function gb(bytes: number): string {
  return `${(bytes / 1e9).toFixed(1)} GB`;
}

/**
 * An ISO timestamp as a brief month-name date ("20 Jul 2026").
 *
 * Month-name form on purpose: a slashed date is ambiguous across locales, and
 * every date on this page sits beside numbers that have to be trusted.
 */
function shortDate(iso: string): string {
  return new Date(iso).toLocaleDateString(undefined, {
    day: "numeric",
    month: "short",
    year: "numeric",
  });
}

/**
 * The row's tool-check state, in one word-led phrase.
 *
 * Check-worded, not measure-worded: the page shows hardware facts (declared
 * size, on disk) the moment a model lands, and a pill that said "measured"
 * read as one of them - when what it reports is whether the TOOL CHECK has
 * run against these weights. Never blank and never an estimate: an unchecked
 * model reads "Not checked", a fact about the deployment rather than a gap in
 * the page. The CPU-spill case gets its own words here and its full sentence
 * in the body - without them a probe that recorded zero VRAM and no
 * throughput reads as a corrupt measurement instead of a model that ran on
 * the processor. Staleness is two words here and says WHICH kind beside the
 * numbers it qualifies, in the body: this pill is a fixed-width column that
 * has to scan down the list.
 *
 * A weights change outranks every other word, because it is the one state a
 * COLLAPSED row would otherwise hide completely: the note explaining it lives
 * in the body, and a reader who never expands the card would see "Not checked"
 * and read it as a model nobody has got round to yet. It is also the truest
 * summary available - whatever the log holds was measured on other weights.
 *
 * A think-mode disagreement sits directly beneath it, for the same reason and
 * one more: it is the state where the check DID run and found something the
 * declaration does not say, so "Checked" would be true and useless.
 */
function checkWord(row: ManagerRow): string {
  if (weightsChangeOf(row) !== null) return "Weights changed";
  // Only the MEASURED disagreement claims this column. The hazard note fires
  // precisely when nothing has measured these weights, so wording it as a
  // difference would assert a comparison nobody made - and would hide the
  // "Not checked" that is both true and the thing to act on.
  if (thinkMismatchOf(row) !== null) return "Think mode differs";
  const measured = row.measured;
  if (measured === null) return "Not checked";
  if (!measured.representative) return "Checked on CPU";
  if (isStale(measured)) return "Check stale";
  return "Checked";
}

/**
 * Whether a probe's verdict no longer describes what is being served.
 *
 * One predicate for all three kinds, because three surfaces ask - the pill's
 * word, the pill's tone, and the body's note - and a set that differed between
 * them is how a quiet pill ends up over a warning. The three kinds qualify a
 * verdict differently and the NOTE says which; whether it is stale at all is
 * this one question.
 *
 * `endpoint_stale` is absent on backends that predate the endpoint binding, and
 * `?? false` is what makes that read as "not stale" rather than as `undefined`
 * quietly becoming truthy through some later edit.
 */
function isStale(measured: MeasuredFacts): boolean {
  return measured.runtime_stale || measured.harness_stale || (measured.endpoint_stale ?? false);
}

/**
 * Whether the check pill is an operator-actionable gap (the warn hue).
 *
 * Deliberately one rule and not two: the pill warns exactly when the body has
 * a warn note to show. Anything else would put a quiet pill over a warning, or
 * the reverse.
 */
function checkWarn(row: ManagerRow): boolean {
  if (weightsChangeOf(row) !== null || thinkModeOf(row) !== null) return true;
  const measured = row.measured;
  if (measured === null) return false;
  return !measured.representative || isStale(measured);
}

/** The measured facts as facts, with the provenance that makes them readable. */
function measuredFacts(row: ManagerRow): string | null {
  const measured = row.measured;
  if (measured === null) return null;
  const when = shortDate(measured.measured_at);
  if (!measured.representative) {
    return `Checked on CPU - not representative - ${measured.depth} check, ${when}`;
  }
  const parts = [
    measured.tokens_per_second === null
      ? null
      : `${measured.tokens_per_second.toFixed(1)} tok/s`,
    measured.vram_bytes === null ? null : `${gb(measured.vram_bytes)} VRAM`,
    `${measured.depth} check, ${when}`,
  ];
  return parts.filter((part) => part !== null).join(" - ");
}

/**
 * The staleness note that rides beside a probe's numbers, when one applies.
 *
 * One note, worst first. The endpoint leads because it is the only one that
 * withdraws the verdict outright - the other two qualify a measurement that
 * still describes the server being served from, while this one says the
 * measurement was taken somewhere else entirely.
 */
function stalenessNote(row: ManagerRow): string | null {
  const measured = row.measured;
  if (measured === null) return null;
  if (measured.endpoint_stale ?? false) {
    return "Stale - there is no record of this check running against the model server serving now, so its verdict does not carry. Re-run the check here.";
  }
  if (measured.runtime_stale) {
    return "Stale - runtime upgraded since this ran, so it describes a different serving stack.";
  }
  if (measured.harness_stale) {
    return "Stale - the check's cases have changed since this ran.";
  }
  return null;
}

/**
 * The weights-change note: verdicts a re-pull left behind, in one sentence.
 *
 * Cause first, then what it means, then the named fix - and only from the
 * backend's own facts. Which depth, when it ran and what it earned all come
 * off the payload; nothing here guesses WHY the weights changed, because a
 * re-pull, a registry update and a manual tag move are indistinguishable from
 * a digest and only one of them would be worth naming.
 *
 * Two shapes, keyed on whether any verdict still describes the installed
 * weights. With none, the model is unmeasured as it stands; with one, the
 * orphaned depth is the only thing missing.
 */
function weightsChangeNote(row: ManagerRow): string | null {
  const note = weightsChangeOf(row);
  if (note === null) return null;
  const lead = note.depths[note.depths.length - 1];
  const when = shortDate(note.measured_at);
  // The buttons are "Run check" (fast) and "Deep check", so the fix is named
  // in the words the reader is about to look for.
  const fix = lead === "deep" ? "running a deep check re-earns it" : "running a check re-earns it";
  if (row.measured === null) {
    return (
      `The installed weights changed since the last tool check. The ${lead} verdict from ` +
      `${when} (${note.earned}) was earned on the previous weights, so nothing has measured ` +
      `what is installed now - ${fix}.`
    );
  }
  return (
    `The installed weights changed since the ${lead} check. Its verdict from ${when} ` +
    `(${note.earned}) was earned on the previous weights, so it says nothing about what is ` +
    `installed now - ${fix}.`
  );
}

/** One declared-capability chip: its words, and whether it is a capability. */
interface CapabilityChip {
  label: string;
  /** True for a genuine capability (tinted); false for a mode word. */
  cap: boolean;
}

/** The capability words a row declares, as short chips. */
function capabilityChips(row: ManagerRow): CapabilityChip[] {
  const chips: CapabilityChip[] = [];
  if (row.tools === "native") chips.push({ label: "Tools", cap: true });
  if (row.think !== "unknown") chips.push({ label: `Thinking: ${row.think}`, cap: false });
  // The EFFECTIVE value, not the claim. A chip is read as "this is what the
  // model does", and on this one axis a claim the fold has withheld is worse
  // than no chip at all: reasoning with tools on a pair that cannot manage it
  // returns a BLANK answer, so a card promising it would be the only place a
  // reader could have caught the mistake.
  if (row.thinking_with_tools_effective) {
    chips.push({ label: "Reasons while calling tools", cap: true });
  }
  if (row.vision) chips.push({ label: "Vision", cap: true });
  return chips;
}

/**
 * The standing sentence about this row's reason-and-call-tools state, or null.
 *
 * Raised only where the claim is GRANTED and the fold has withheld it, which is
 * the one state a chip cannot express: a chip is present or absent, and
 * "granted but not yet earned" needs a sentence. The opposite pairing cannot
 * arise - the resolution starts from the declaration, so nothing withheld is
 * ever effective - and agreement in either direction is the ordinary case and
 * says nothing.
 */
function thinkingWithToolsNote(row: ManagerRow): string | null {
  if (!row.thinking_with_tools || row.thinking_with_tools_effective) return null;
  // `thinking_with_tools` is true only when a stored declaration made it so -
  // nothing else can claim this axis - so `thinking_with_tools_declared` is
  // always true here too; there is no other source to branch on.
  return (
    "A declaration made here grants " +
    "reasoning with tools, and no deep check has proven it on the weights installed now - so " +
    "turns still do one or the other. The grant is stored and starts applying the moment one " +
    "passes."
  );
}

/** Down-chevron glyph for a row's expand affordance. */
function Chevron() {
  return (
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
      <path suppressHydrationWarning d="M4 6l4 4 4-4" />
    </svg>
  );
}

/** Props for one {@link ModelRow}. */
export interface ModelRowProps {
  row: ManagerRow;
  /** How much the group holding this row may be managed. */
  management: RoleGroup["management"];
  /** True while some operation holds the backend's single flight. */
  busy: boolean;
  /** This row's own download, while one is in flight; null otherwise. */
  activePull: { percent: number | null } | null;
  /**
   * This row's own tool check, while one runs; null otherwise.
   *
   * Both counts are null until the first sample lands - a check unloads
   * whatever is warm and cold-loads the model under test, so there is a real
   * wait before there is a fraction to show.
   */
  activeCheck: { completed: number | null; total: number | null } | null;
  /** The unfinished transfer this session remembers for this row, if any. */
  pausedPull: PullProgress | null;
  /** Download this row's weights. */
  onPull: (row: ManagerRow) => void;
  /** Re-pull installed weights to take whatever the registry now serves. */
  onPullUpdates: (row: ManagerRow) => void;
  /** Run a tool check at the given depth. */
  onCheck: (row: ManagerRow, depth: "fast" | "deep") => void;
  /** Store the measured think mode as this deployment's declaration. */
  onConfirmThinkMode: (row: ManagerRow) => void;
  /** Declare how much this model may be allowed to drive. */
  onDeclareCeiling: (row: ManagerRow) => void;
  /** Grant or withhold reasoning and calling tools in the same turn. */
  onDeclareThinkingWithTools: (row: ManagerRow) => void;
  /** Choose which installed model serves this one's thinking turns. */
  onDeclarePair: (row: ManagerRow) => void;
  /** Make this the workspace default every unpinned conversation follows. */
  onSetDefault: (row: ManagerRow) => void;
  /** Remove this row's weights. */
  onDelete: (row: ManagerRow) => void;
}

/**
 * One model in the grid: a summary row over an expandable body of facts.
 *
 * The summary answers the two questions a list is scanned for - is it here, and
 * has anything measured it - and the body carries everything else: the declared
 * facts, what a probe observed, how much the model may drive a turn, and the
 * destructive zone.
 */
export function ModelRow({
  row,
  management,
  busy,
  activePull,
  activeCheck,
  pausedPull,
  onPull,
  onPullUpdates,
  onCheck,
  onConfirmThinkMode,
  onDeclareCeiling,
  onDeclareThinkingWithTools,
  onDeclarePair,
  onSetDefault,
  onDelete,
}: ModelRowProps) {
  const [open, setOpen] = useState(false);
  // A locked role keeps its facts and loses its verbs; a status-only role is
  // not served through this provider at all, so it has no verb to honour.
  const manageable = management === "manageable";
  const facts = measuredFacts(row);
  const weightsChange = weightsChangeNote(row);
  // One warn line in the check zone, and weights-change wins it. The two can
  // coexist only where a bound verdict is stale AND another depth was left
  // behind by a re-pull; there the re-pull is both the newer event and the one
  // with a fix, and "the runtime moved under this verdict" is a footnote to a
  // verdict that no longer describes the weights anyway.
  const staleness = weightsChange === null ? stalenessNote(row) : null;
  const thinkNote = thinkModeOf(row);
  const ttwNote = thinkingWithToolsNote(row);
  const chips = capabilityChips(row);
  // The backend's own answer, mirroring what the verify endpoint refuses -
  // which is why an installed chat model nothing describes gets its check
  // here, where a role test would have hidden it.
  const checkable = row.checkable;
  // A row a chat turn could route to. `agency` is the tell rather than the role
  // string: only a chat row is ever given an agency report, and a tag the
  // runtime vouches for gets one even where its report establishes no role -
  // which is exactly the row a remote-inventory deployment needs these
  // controls on.
  const chatRow = row.agency !== null;
  // The check zone exists wherever there is a check story to tell: a role the
  // harness measures, a probe that already ran, a weights change to explain,
  // or a clamp explaining itself. The third term is redundant today (only a
  // chat row carries a note, and a chat row always carries an agency report)
  // and is here anyway: the summary pill now renders on it, and a warn pill
  // over a body with no zone under it would be a dead end.
  const checkZone =
    checkable ||
    row.measured !== null ||
    weightsChange !== null ||
    thinkNote !== null ||
    row.agency !== null;
  // A pull on installed weights fetches updates - a body verb, never the
  // summary's call to action (see _row_verbs in model_manager.py: the locked
  // embedding role deliberately never gets it).
  const updatable = row.can_pull && row.installed;
  // Null until the first sample lands, which is also what makes the mini read
  // as indeterminate through the model's cold load rather than pinned at zero.
  const checkFraction =
    activeCheck === null ||
    activeCheck.completed === null ||
    activeCheck.total === null ||
    activeCheck.total <= 0
      ? null
      : (activeCheck.completed / activeCheck.total) * 100;

  return (
    <li className="mm-row">
      <div className="mm-head">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="mm-summary"
        >
          <span className="mm-main">
            <span className="mm-name">
              <code className="mm-tag">{row.model}</code>
              {/* Which model answers when nobody chose one - the question a
                  grid is scanned for as often as "what is installed", and one
                  that previously needed a card expanded to answer. Exactly ONE
                  row wears it (the backend resolves the default, falling back
                  to the deployment binding when nothing is stored), so it reads
                  as a pointer rather than as a property.

                  Accent, not warn: it is a routing FACT, not something wrong.
                  It leads the badge cluster because it is about the whole
                  deployment, where the ones after it describe this model. */}
              {row.is_default && <span className="badge cap">Default</span>}
              {/* Vision is the one capability whose ABSENCE a user meets as a
                  surprise - they attach an image and the answer never mentions
                  it - so it earns a place in the collapsed row rather than
                  waiting in the chip list below. Shown only where it is true:
                  most models cannot see, and a "no vision" badge on nearly
                  every row would be noise that hides the few that can. */}
              {row.vision && <span className="badge cap">Vision</span>}
            </span>
          </span>
          <span className={`mm-state${row.installed ? " ok" : ""}`}>
            <span className="pill-label">Weights: </span>
            {row.installed ? "Installed" : "Not installed"}
          </span>
          {checkable || row.measured !== null || weightsChange !== null || thinkNote !== null ? (
            <span
              className={`mm-state${checkWarn(row) ? " warn" : checkWord(row) === "Checked" ? " ok" : ""}`}
            >
              <span className="pill-label">Tool check: </span>
              {checkWord(row)}
            </span>
          ) : (
            // The column stays open so the weights pills align down the whole
            // page; a role the check cannot measure has nothing to say in it,
            // and saying "Not checked" there would read as a gap to close.
            <span className="mm-state-space" aria-hidden="true" />
          )}
          <span className="mm-chevron">
            <Chevron />
          </span>
        </button>
        {/* The slot always renders so every row's pills sit in one column;
            a verb the provider does not have is ABSENT inside it, never
            disabled - a dead button is a promise the backend cannot keep.
            While this row's own download runs, the slot is where the click
            landed - so it carries the live readout rather than going mute
            with the pull happening off-screen in the panel above. A check
            gets the same treatment, and the two are consecutive on one stream
            (a pull runs a check when it lands), so the slot simply moves from
            the percent to the sample count when it does. */}
        <span className="mm-action">
          {activePull ? (
            <button
              type="button"
              onClick={() =>
                document
                  .querySelector('[data-part="mm-activity"]')
                  ?.scrollIntoView({ block: "nearest" })
              }
              className="mm-pullmini"
              aria-label={`Downloading ${row.model} - show the progress panel`}
            >
              <span>
                {activePull.percent === null ? "..." : `${Math.round(activePull.percent)}%`}
              </span>
              <span
                className={`mm-minibar${activePull.percent === null ? " indeterminate" : ""}`}
                aria-hidden="true"
              >
                <i
                  style={
                    activePull.percent === null
                      ? undefined
                      : { width: `${activePull.percent}%` }
                  }
                />
              </span>
            </button>
          ) : activeCheck ? (
            <button
              type="button"
              onClick={() =>
                document
                  .querySelector('[data-part="mm-activity"]')
                  ?.scrollIntoView({ block: "nearest" })
              }
              className="mm-pullmini"
              aria-label={`Checking ${row.model} - show the progress panel`}
            >
              {/* Samples, not a percentage: a check is a countable number of
                  discrete runs, and "7/26" says how much is left in a unit
                  the panel below uses too. A percentage over 26 samples would
                  move in steps of about four points, which reads as a bar
                  that jumps rather than one that fills. */}
              <span>
                {checkFraction === null
                  ? "..."
                  : `${activeCheck.completed}/${activeCheck.total}`}
              </span>
              <span
                className={`mm-minibar${checkFraction === null ? " indeterminate" : ""}`}
                aria-hidden="true"
              >
                <i style={checkFraction === null ? undefined : { width: `${checkFraction}%` }} />
              </span>
            </button>
          ) : row.can_pull && !row.installed ? (
            <button
              type="button"
              disabled={busy}
              onClick={() => onPull(row)}
              className="btn btn-primary"
            >
              {pausedPull ? "Resume" : "Pull"}
            </button>
          ) : null}
        </span>
      </div>

      {open && (
        <div className="mm-body">
          {/* Identity first (what these weights ARE), then the two window
              numbers, then sizes and the digest - so a body reads from the
              answer a person came for down to the numbers they check. Every
              one of these is present only when the payload carries it: a
              never-installed row has no quantization to report, and a fact
              cell reading "unknown" invents a gap where there is simply
              nothing to say yet. */}
          <dl className="mm-facts">
            {/* Empty where nothing names a serving backend for these weights,
                and an empty cell beside filled ones reads as a page that
                failed rather than as a fact nobody has. */}
            {row.provider !== "" && <Fact label="Provider" value={row.provider} />}
            {row.family !== null && <Fact label="Family" value={row.family} />}
            {row.parameter_size !== null && (
              <Fact label="Parameters" value={row.parameter_size} />
            )}
            {row.quantization !== null && (
              <Fact label="Quantization" value={row.quantization} />
            )}
            {row.num_ctx !== null && (
              <Fact label="Context window" value={`${row.num_ctx.toLocaleString()} tokens`} />
            )}
            {/* The ceiling, beside what this deployment actually asks for -
                without it a small window reads as a property of the model
                when it is a choice this machine made. Labelled by the
                question it answers rather than by the field's name, since a
                fact cell has no room to gloss "native". */}
            {row.native_context_length !== null && (
              <Fact
                label="Trained for"
                value={`${row.native_context_length.toLocaleString()} tokens`}
              />
            )}
            {row.installed_size_bytes !== null && (
              <Fact label="On disk" value={gb(row.installed_size_bytes)} />
            )}
            {row.digest !== null && (
              // Short form: the full digest is 64 characters of noise, and the
              // first dozen are what identifies the weights at a glance.
              <Fact label="Digest" value={row.digest.slice(0, 12)} mono />
            )}
            {row.thinking_model !== null && (
              // Labelled by WHERE it came from, because the two are changed in
              // different places: a declared pair is edited from this console,
              // while the environment's pair override is deployment
              // configuration and outranks anything set here.
              <Fact
                label={row.thinking_model_declared ? "Thinking partner" : "Thinking sibling"}
                value={row.thinking_model}
                mono
              />
            )}
          </dl>

          {/* The unfinished transfer this session remembers: how far it got,
              frozen, and that resuming is free. Session-scoped by nature - the
              runtime has no API that lists partial downloads. */}
          {pausedPull && !row.installed && (
            <div className="mm-partial">
              {pausedPull.percent !== null && (
                <div className="mm-bar">
                  <i style={{ width: `${pausedPull.percent}%` }} />
                </div>
              )}
              <p className="mm-note">
                {`Paused download${
                  pausedPull.percent === null ? "" : ` at ${pausedPull.percent.toFixed(1)}%`
                }`}
                {pausedPull.completed !== null && pausedPull.total !== null
                  ? ` - ${gb(pausedPull.completed)} of ${gb(pausedPull.total)}`
                  : ""}
                {". Resume continues from where it stopped."}
              </p>
            </div>
          )}

          {chips.length > 0 && (
            <ul className="mm-chips">
              {chips.map((chip) => (
                <li key={chip.label} className={`mm-chip${chip.cap ? " cap" : ""}`}>
                  {chip.label}
                </li>
              ))}
            </ul>
          )}

          {/* Everything the tool check has to say, under one name - so the
              measured line, the agency verdict and the buttons that change
              them read as one story, apart from the declared facts above.
              Absent entirely for a role the harness cannot measure: an
              embedder's body says nothing about checks rather than
              "unmeasured", which read as a gap somebody should close. */}
          {checkZone && (
            <div className="mm-zone">
              <p className="mm-overline">Tool check</p>
              {(row.installed || row.measured !== null) && (
                <p className="mm-measured">
                  {facts ?? "No tool check has run against these weights."}
                </p>
              )}
              {weightsChange && <p className="mm-note warn">{weightsChange}</p>}
              {staleness && <p className="mm-note warn">{staleness}</p>}

              {/* The backend's own sentence, whole: it names the disagreement
                  AND what it means, and paraphrasing would drop one of them.
                  The two extra lines are the only things this side adds - what
                  is already happening to turns, and the one action that
                  settles it - because neither is in the sentence and both are
                  what a reader came for. */}
              {thinkNote && (
                <div className="mm-thinknote">
                  <p className="mm-note warn">{thinkNote.message}</p>
                  {/* What is happening RIGHT NOW, both halves of it. The
                      backend's sentence above already names the tool
                      consequence when it applies; this line is the standing
                      state, in the present tense a reader checks a card for. */}
                  {thinkNote.routing_clamped && (
                    <p className="mm-hint reason">
                      {/* The EFFECTIVE capability, matching the backend's own
                          sentence above and matching what the turn does: a
                          granted-but-unearned model is offered no tools, so
                          reading the claim here would promise the wrong half. */}
                      {row.thinking_with_tools_effective
                        ? "Turns are already sending the flag on for this model, so nothing is leaking into answers while this is decided."
                        : "Turns are already sending the flag on for this model, so nothing is leaking into answers - and while it is forced, this model is offered no tools. Answers stay grounded through the usual pre-fetch until the mode is confirmed."}
                    </p>
                  )}
                  {/* Only where a measurement bound to the installed weights
                      exists. The hazard note has nothing to confirm, and a
                      button there would promise to store something nobody
                      measured. */}
                  {manageable && thinkNote.confirmable && (
                    <button
                      type="button"
                      disabled={busy}
                      onClick={() => onConfirmThinkMode(row)}
                      className="btn btn-ghost"
                    >
                      Confirm measured mode
                    </button>
                  )}
                </div>
              )}

              {/* What the model RUNS AT, said once and in words.

                  This replaced a compact triplet ("agentic - ceiling agentic,
                  earned agentic, deployment clamp agentic") that packed three
                  resolution inputs and the answer into one line. It was dense,
                  it repeated the same word up to four times, and it asked the
                  reader to do the min() themselves to find out what actually
                  happens. The answer leads now; the sentence under it, when
                  there is one, names the single factor that decided and what
                  would change it - and it is the BACKEND's sentence, so the
                  card and the dialog cannot tell two stories. */}
              {row.agency && (
                <div className="mm-agency">
                  <p className="mm-verdict">
                    <span className="mm-verdict-mode">{row.agency.effective}</span>
                    {/* Declared HERE is worth a word: it tells a reader the
                        ceiling is theirs to change from this console rather
                        than read off the model server's own report. */}
                    {row.ceiling_declared && <span className="badge">Declared</span>}
                  </p>
                  <p className="mm-verdict-gloss">{AGENCY_GLOSS[row.agency.effective]}</p>
                  {/* Only ever the backend's own explanation: inventing one here
                      would be a guess about why a model is held back. Load-bearing,
                      so it reads at the AA tier, not the decorative one. */}
                  {row.agency.clamp_detail && (
                    <p className="mm-hint reason">{row.agency.clamp_detail}</p>
                  )}
                </div>
              )}

              {/* The other axis a deep check earns, said only when the claim
                  and what a turn does have come apart. The chip above already
                  carries the ordinary case in both directions; this is the
                  gap between them, which a chip cannot show. */}
              {ttwNote && <p className="mm-hint reason">{ttwNote}</p>}

              {/* One action row, at the FOOT of the zone rather than scattered
                  through it: everything here acts on the check story above, and
                  the two verbs that measure belong beside the one that decides.
                  Each is absent, never disabled, where it does not apply. */}
              {manageable && row.installed && (checkable || row.agency !== null) && (
                <div className="mm-zone-actions">
                  {checkable && (
                    <>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => onCheck(row, "fast")}
                        className="btn btn-primary"
                      >
                        Run check
                      </button>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => onCheck(row, "deep")}
                        className="btn btn-ghost"
                      >
                        Deep check
                      </button>
                    </>
                  )}
                  {/* A declaration is bound to a digest, so an uninstalled row
                      has nothing to bind one to - which the gate above covers. */}
                  {row.agency !== null && (
                    <>
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => onDeclareCeiling(row)}
                        className="btn btn-ghost"
                      >
                        Declare ceiling
                      </button>
                      {/* Offered on the same gate and never disabled by the
                          missing check: the backend STORES a grant either way
                          and the fold is what withholds it, so a control
                          greyed out until a deep check passed would refuse the
                          one order that works - declare it, then earn it. The
                          dialog carries the gate in words instead. */}
                      <button
                        type="button"
                        disabled={busy}
                        onClick={() => onDeclareThinkingWithTools(row)}
                        className="btn btn-ghost"
                      >
                        {row.thinking_with_tools
                          ? "Change reasoning with tools"
                          : "Declare reasoning with tools"}
                      </button>
                    </>
                  )}
                </div>
              )}
            </div>
          )}

          {/* What this model is SET to be, as distinct from what it can do.
              Kept out of the check zone deliberately: nothing here is measured,
              and putting a routing choice under a heading about measurement is
              how a reader comes to think the two are the same kind of fact. */}
          {manageable && row.installed && (chatRow || row.workspace_default) && (
            <div className="mm-zone">
              <p className="mm-overline">Routing</p>
              <p className="mm-measured">
                {row.workspace_default
                  ? "The workspace default - every conversation that has not chosen a model follows this one."
                  : "Not the workspace default."}
              </p>
              {/* Absent, never disabled: a row that already IS the default has
                  no such verb to offer, and a dead button would be a promise
                  the backend would refuse. */}
              {!row.workspace_default && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onSetDefault(row)}
                  className="btn btn-ghost"
                >
                  Set as default
                </button>
              )}
              {/* Only for a model that answers DIRECTLY. A model that reasons
                  on its own has no thinking turn to hand anywhere, so pairing
                  it would be a control with nothing behind it - and the backend
                  refuses a self-pair for the same reason. */}
              {/* A model that reasons on its own has no thinking turn to hand
                  anywhere, so it gets no control. "unknown" is included rather
                  than excluded: nothing has established how these weights treat
                  the flag, so pairing a partner is exactly the move that makes
                  the composer's toggle mean something - the strictness belongs
                  on the CANDIDATE side, where offering an unmeasured partner
                  would dress a guess as a capability. */}
              {chatRow && (row.think === "never" || row.think === "unknown") && (
                <>
                  <p className="mm-hint reason">
                    {row.thinking_model === null
                      ? "This model answers directly and has no reasoning partner, so a conversation that asks it to reason gets a direct answer instead."
                      : `Thinking turns go to ${row.thinking_model}${row.thinking_model_declared ? "." : ", which this deployment's environment pairs with it."}`}
                  </p>
                  <button
                    type="button"
                    disabled={busy}
                    onClick={() => onDeclarePair(row)}
                    className="btn btn-ghost"
                  >
                    {row.thinking_model_declared ? "Change partner" : "Choose partner"}
                  </button>
                </>
              )}
            </div>
          )}

          {/* The housekeeping foot: what maintains these weights, benign verb
              before destructive one. Pull updates lives HERE rather than
              beside the check buttons - it maintains the weights, it does not
              measure them, and on a row with no checks it floated alone.
              Gated on the ROW's verbs, deliberately not `manageable`: the
              locked embedding role grants delete per row (an installed
              embedder nothing references strands nothing), and the row's
              flags are the backend's own per-row answer. */}
          {(updatable || row.can_delete || row.delete_blocked_reason !== null) && (
            <div className="mm-danger">
              {updatable && (
                <button
                  type="button"
                  disabled={busy}
                  onClick={() => onPullUpdates(row)}
                  className="btn btn-ghost"
                >
                  Pull updates
                </button>
              )}
              {(row.can_delete || row.delete_blocked_reason !== null) && (
                <button
                  type="button"
                  disabled={busy || !row.can_delete}
                  onClick={() => onDelete(row)}
                  className="btn btn-danger-ghost"
                >
                  Delete
                </button>
              )}
              {/* Visible text, never a title tooltip: a title on a disabled
                  button never opens in Chrome or Safari, and hover-only text
                  never reaches touch or keyboard at all. */}
              {row.delete_blocked_reason && (
                <p className="mm-reason">{row.delete_blocked_reason}</p>
              )}
            </div>
          )}
        </div>
      )}
    </li>
  );
}

/**
 * A status-only row: a model this provider does not serve at all.
 *
 * There is no verb to offer and nothing behind an expander - the name is the
 * whole answer. Nothing renders a status word here any more: the one role that
 * ever took this shape was the reranker, which runs in its own sidecar rather
 * than on the model server and so never appears in an inventory of installed
 * tags. Its configured-or-not state lives on the Defaults page, which reads it
 * from the workspace payload. The shape is kept because the group's management
 * union still carries `status_only`, and a backend that returns one must render
 * as a row rather than as nothing at all.
 */
export function StatusRow({ row }: { row: ManagerRow }) {
  return (
    <li className="mm-row">
      <div className="mm-head static">
        <span className="mm-main">
          <span className="mm-name">
            <code className="mm-tag">{row.model}</code>
          </span>
        </span>
        {/* The same trailing reservations a summary row carries (the two pill
            columns, the chevron, the action gutter), so this row's name lines
            up with every other name the page scans down. */}
        <span className="mm-state-space" aria-hidden="true" />
        <span className="mm-state-space" aria-hidden="true" />
        <span className="mm-chevron-space" aria-hidden="true" />
        <span className="mm-action" aria-hidden="true" />
      </div>
    </li>
  );
}

/** One labelled fact in the body's fact list. */
function Fact({
  label,
  value,
  mono,
}: {
  label: string;
  value: string;
  mono?: boolean;
}) {
  return (
    <div className="mm-fact">
      <dt>{label}</dt>
      <dd className={mono ? "mm-tag" : undefined}>{value}</dd>
    </div>
  );
}
