/**
 * The boot's waits, run in order, with everything they touch injected: the
 * fetches, the clock and the sleep. The hook (hooks/use-boot.ts) hands in the
 * real ones and a sink that writes React state; the battery hands in fakes
 * and watches the sink, so the retry, slow, skip and abort rules are tested
 * as behaviour rather than read off the source.
 */

import {
  BOOT_WAITS,
  type BootPace,
  type BootPhase,
  backendDetail,
  countingLabel,
  dataDetail,
  dwellFor,
  holdLeft,
  type LitLine,
  modelCheck,
  modelDetail,
  type Reach,
  READY_DWELL_MS,
  READY_HEAD,
  type ReadinessReport,
  RETRY_EVERY_MS,
  SLOW_AFTER_MS,
  slowHead,
  vectorStoreCheck,
} from "@/lib/boot";
import { sameTag } from "@/lib/model-choice";
import type { TurnState } from "@/lib/turn-state";

/** How long an optional fact (defaults, residency, counts) may take before the boot goes on without it. */
export const OPTIONAL_TIMEOUT_MS = 5_000;

/** What the runner asks of the world. */
export interface BootDeps {
  readiness: (signal: AbortSignal) => Promise<ReadinessReport>;
  defaults: () => Promise<{ model: string | null; web_search: Reach["webSearch"] } | null>;
  residency: () => Promise<{
    reported: boolean;
    models: { model: string; total_bytes: number | null }[];
  } | null>;
  conversations: () => Promise<{ items: unknown[] }>;
  knowledge: () => Promise<{ items: unknown[]; truncated: boolean } | null>;
  servers: () => Promise<{ servers: { enabled: boolean }[] } | null>;
  /** Resolves after `ms`, or at once when the signal aborts. */
  sleep: (ms: number, signal: AbortSignal) => Promise<void>;
  now: () => number;
  /** Full on the session's first load, quick after (lib/boot.ts). */
  pace: BootPace;
  /** When the boot screen was painted on the clock, or null while it has not been. */
  shownAt: () => number | null;
}

/** What the runner tells the screen. */
export interface BootSink {
  node: (index: number, phase: BootPhase, detail: string) => void;
  head: (line: LitLine) => void;
  sub: (text: string) => void;
  /** The slow box's copy, or null to take it down. */
  slow: (copy: string | null) => void;
  weather: (state: TurnState) => void;
  reach: (reach: Reach) => void;
  phase: (phase: "booting" | "ready" | "done") => void;
  /** A long wait that may be given up on offers this; null once it may not. */
  skippable: (skip: (() => void) | null) => void;
}

export const NO_REACH: Reach = {
  documents: null,
  truncated: false,
  knowledgeReady: false,
  webSearch: null,
  tools: null,
};

/** A failure the server meant: asking again would get the same answer. */
function isClientError(e: unknown): boolean {
  const status = (e as { status?: unknown } | null)?.status;
  return typeof status === "number" && status >= 400 && status < 500;
}

interface WaitControl {
  /** The person gave up on this wait. */
  skipped: () => boolean;
  /** A subline while the wait runs (dropped once it ended). */
  sub: (text: string) => void;
}

/**
 * Run the three waits and hand the shell over.
 *
 * Backend: the readiness relay answers with a report whose required
 * services are up; one that is not names what is down and is asked again,
 * as is a backend that does not answer at all. Model runtime: the report's
 * model line is ok or unconfigured (a deployment without one is a normal
 * deployment); an unreachable runtime is asked again until it answers, or
 * until the person, once the wait is slow, chooses to go on without it - the
 * shell does not need the runtime to show, and the page that fixes it lives
 * behind the shell. Conversations: the list, with the counts the reach line
 * wants beside it; an answer the server meant (a lapsed session's 401) ends
 * the wait with nothing, since the shell owns sign-in. Optional facts are
 * given a few seconds and then done without. Each lit node stays lit for a
 * beat even when its wait ends at once, so the rail reads; a wait past the
 * limit is called slow and counts. Nothing is written after the signal
 * aborts.
 */
export async function runBootWaits(
  deps: BootDeps,
  sink: BootSink,
  signal: AbortSignal,
): Promise<void> {
  const aborted = () => signal.aborted;
  let reach: Reach = NO_REACH;
  sink.phase("booting");
  BOOT_WAITS.forEach((_, i) => sink.node(i, "pending", ""));
  sink.reach(NO_REACH);
  sink.skippable(null);

  /* an optional fact: a failure or a long silence reads as not learned */
  const optional = <T,>(ask: () => Promise<T | null>): Promise<T | null> =>
    Promise.race([
      ask().catch(() => null),
      deps.sleep(OPTIONAL_TIMEOUT_MS, signal).then(() => null),
    ]);

  /* ask until it answers; a 4xx will not change by asking again, nor will
     anything once the person gave up on the wait */
  const keepTrying = async <T,>(
    ask: () => Promise<T>,
    ctl?: WaitControl,
  ): Promise<T | null> => {
    for (;;) {
      if (aborted()) throw new Error("aborted");
      if (ctl?.skipped()) return null;
      try {
        return await ask();
      } catch (e) {
        if (aborted()) throw e;
        if (isClientError(e)) return null;
        await deps.sleep(RETRY_EVERY_MS, signal);
      }
    }
  };

  /* one wait: its node lights, the headline names it, the sky takes its
     colour; the slow clock runs beside the work; the node stays lit a beat */
  const wait = async <T,>(
    i: number,
    canSkip: boolean,
    work: (ctl: WaitControl) => Promise<[T, string]>,
  ): Promise<T> => {
    const w = BOOT_WAITS[i];
    const from = deps.now();
    let ended = false;
    let skipped = false;
    let slow = false;
    sink.node(i, "live", w.waiting);
    sink.head(w.head);
    sink.sub(w.waiting);
    sink.slow(null);
    sink.weather(w.sky);
    const ctl: WaitControl = {
      skipped: () => skipped,
      /* once the wait is slow the subline counts; the work's own words give way */
      sub: (text) => {
        if (!ended && !slow) sink.sub(text);
      },
    };
    void deps.sleep(SLOW_AFTER_MS, signal).then(async () => {
      if (ended || aborted()) return;
      slow = true;
      sink.head(slowHead(w.head));
      sink.slow(w.slow);
      sink.weather("boot-slow");
      if (canSkip) {
        sink.skippable(() => {
          skipped = true;
        });
      }
      while (!ended && !aborted()) {
        sink.sub(countingLabel(deps.now() - from));
        await deps.sleep(1000, signal);
      }
    });
    const [value, detail] = await work(ctl);
    ended = true;
    sink.skippable(null);
    const left = dwellFor(deps.pace, w.dwell) - (deps.now() - from);
    if (left > 0) await deps.sleep(left, signal);
    if (!aborted()) sink.node(i, "done", detail);
    return value;
  };

  // 1. the backend: a report whose required services are up
  let report = await wait<ReadinessReport>(0, false, async (ctl) => {
    const r = await keepTrying(async () => {
      const got = await deps.readiness(signal);
      const verdict = backendDetail(got);
      if (!verdict.done) {
        ctl.sub(verdict.detail);
        throw new Error(verdict.detail);
      }
      return got;
    });
    if (r === null) throw new Error("boot: the readiness relay refused");
    return [r, backendDetail(r).detail];
  });
  if (aborted()) return;

  // 2. the model runtime: ok or unconfigured ends the wait; unreachable is
  //    asked again until it answers, or until the person goes on without it
  const [defaults, residency] = await Promise.all([
    optional(deps.defaults),
    optional(deps.residency),
  ]);
  if (aborted()) return;
  reach = { ...reach, webSearch: defaults?.web_search ?? null };
  const model = defaults?.model ?? null;
  const resident =
    residency?.reported === true
      ? (residency.models.find((m) => model !== null && sameTag(m.model, model)) ?? null)
      : null;
  report = await wait<ReadinessReport>(1, true, async (ctl) => {
    let r = report;
    let check = modelCheck(r);
    if (check !== null && check.state === "unreachable") {
      /* the runtime's name and state, not the report's developer string */
      ctl.sub(`${check.name} ${check.state}`);
      const again = await keepTrying(async () => {
        const got = await deps.readiness(signal);
        const c = modelCheck(got);
        if (c !== null && c.state === "unreachable") {
          ctl.sub(`${c.name} ${c.state}`);
          throw new Error(c.state);
        }
        return got;
      }, ctl);
      if (again === null) return [r, "model runtime unreachable"];
      r = again;
      check = modelCheck(r);
    }
    const detail =
      check !== null && check.state === "unconfigured"
        ? "model runtime not configured"
        : modelDetail(model, resident);
    return [r, detail];
  });
  if (aborted()) return;

  // 3. conversations, with the reach line's counts beside them
  await wait<unknown>(2, false, async () => {
    const [list, knowledge, servers] = await Promise.all([
      keepTrying(deps.conversations),
      optional(deps.knowledge),
      optional(deps.servers),
    ]);
    const documents = Array.isArray(knowledge?.items) ? knowledge.items.length : null;
    const tools = Array.isArray(servers?.servers)
      ? servers.servers.filter((s) => s.enabled).length
      : null;
    reach = {
      ...reach,
      documents,
      truncated: knowledge?.truncated === true,
      knowledgeReady: vectorStoreCheck(report)?.state === "ok",
      tools,
    };
    sink.reach(reach);
    const detail =
      list === null ? "could not be listed" : dataDetail(list.items.length, documents);
    return [list, detail];
  });
  if (aborted()) return;

  // ready: a beat, then the shell; a screen that was painted stands its least
  sink.head(READY_HEAD);
  sink.sub("");
  sink.weather("boot-ready");
  sink.phase("ready");
  const stand = Math.max(dwellFor(deps.pace, READY_DWELL_MS), holdLeft(deps.shownAt(), deps.now()));
  if (stand > 0) await deps.sleep(stand, signal);
  if (aborted()) return;
  sink.phase("done");
}
