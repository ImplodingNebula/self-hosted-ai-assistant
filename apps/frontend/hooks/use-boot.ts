"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import {
  BOOT_DEFER_MS,
  BOOT_WAITS,
  type BootPace,
  type BootPhase,
  type LitLine,
  paceFor,
  type Reach,
} from "@/lib/boot";
import { type BootSink, NO_REACH, runBootWaits } from "@/lib/boot-runner";
import { listConversations } from "@/lib/conversations";
import { listServers } from "@/lib/integrations";
import { listKnowledgeDocuments } from "@/lib/knowledge";
import { getResidency } from "@/lib/model-manager";
import { getReadiness } from "@/lib/readiness";
import { setTurnState } from "@/lib/turn-state";
import { getWorkspaceDefaults } from "@/lib/workspace-defaults";

/** One node of the boot rail. */
export interface BootNodeState {
  phase: BootPhase;
  /** The mono detail under the label: what is being waited on, or what arrived. */
  detail: string;
}

/** What the boot screen shows, and what the shell learns from it. */
export interface BootState {
  /** `booting` while a wait runs, `ready` for the beat after the last, `done` once the shell may show. */
  phase: "booting" | "ready" | "done";
  /** Whether the boot screen is painted: at once on a full load, only once a wait runs long on a quick one. */
  shown: boolean;
  nodes: readonly BootNodeState[];
  head: LitLine;
  sub: string;
  /** The running wait has gone long: the screen says so, counts, and offers a retry. */
  slow: string | null;
  /** The long wait may be given up on (the model runtime); null when it may not. */
  skip: (() => void) | null;
  /** What the install can reach, for the greeting's reach line. */
  reach: Reach;
  /** Run the waits again from the first. */
  retry: () => void;
}

const PENDING: readonly BootNodeState[] = BOOT_WAITS.map(() => ({ phase: "pending", detail: "" }));

/** The session's mark that it has booted once; the next load is quick. */
const SEEN_KEY = "boot:seen";

/**
 * Whether this browser session has booted before, marking it so for the next
 * load. Marked as the waits start, not as they end: a reload mid-boot has
 * still shown the screen once, and a second full run would be a tax.
 */
function sessionSeen(): boolean {
  try {
    const seen = window.sessionStorage.getItem(SEEN_KEY) !== null;
    window.sessionStorage.setItem(SEEN_KEY, "1");
    return seen;
  } catch {
    return false;
  }
}

/** A sleep that ends early, resolved, when the signal aborts: loops check the signal and leave. */
const sleep = (ms: number, signal: AbortSignal) =>
  new Promise<void>((resolve) => {
    if (signal.aborted) return resolve();
    const t = window.setTimeout(done, ms);
    function done() {
      window.clearTimeout(t);
      signal.removeEventListener("abort", done);
      resolve();
    }
    signal.addEventListener("abort", done, { once: true });
  });

/**
 * Run the waits the shell blocks on (lib/boot-runner.ts) and say where they
 * stand. The sky keeps each wait's colour through the turn state.
 */
export function useBoot(enabled: boolean): BootState {
  const [run, setRun] = useState(0);
  const [phase, setPhase] = useState<BootState["phase"]>(enabled ? "booting" : "done");
  const [nodes, setNodes] = useState<readonly BootNodeState[]>(PENDING);
  const [head, setHead] = useState<LitLine>(BOOT_WAITS[0].head);
  const [sub, setSub] = useState(BOOT_WAITS[0].waiting);
  const [slow, setSlow] = useState<string | null>(null);
  const [skip, setSkip] = useState<(() => void) | null>(null);
  const [reach, setReach] = useState<Reach>(NO_REACH);
  const retry = useCallback(() => setRun((r) => r + 1), []);
  // The pace is the session's, read once the page is on screen (the server
  // cannot know the session, so the screen is never in its HTML): a full load
  // paints the boot on the next tick, a quick one only once a wait has run
  // long. A painted screen then stands its least (the runner reads
  // `shownAt`), so a slow wait never flashes it past. A retry keeps both.
  const pace = useRef<BootPace | null>(null);
  const [shown, setShown] = useState(false);
  const shownAt = useRef<number | null>(null);
  const paint = useCallback(() => {
    shownAt.current ??= Date.now();
    setShown(true);
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const p = (pace.current ??= paceFor(sessionSeen()));
    const ac = new AbortController();
    const { signal } = ac;
    // Cleared when the run is cancelled, and when the waits are done: a
    // screen painted in the frame between done and the shell would flicker.
    const defer = window.setTimeout(paint, p === "quick" ? BOOT_DEFER_MS : 0);
    signal.addEventListener("abort", () => window.clearTimeout(defer), { once: true });
    const sink: BootSink = {
      node: (i, phase, detail) =>
        setNodes((prev) => prev.map((n, j) => (j === i ? { phase, detail } : n))),
      head: setHead,
      sub: setSub,
      slow: setSlow,
      weather: setTurnState,
      reach: setReach,
      phase: (next) => {
        if (next === "done") window.clearTimeout(defer);
        setPhase(next);
      },
      skippable: (fn) => setSkip(() => fn),
    };
    void runBootWaits(
      {
        readiness: getReadiness,
        defaults: () => getWorkspaceDefaults(),
        residency: () => getResidency(),
        conversations: () => listConversations(),
        knowledge: () => listKnowledgeDocuments(),
        servers: () => listServers(),
        sleep,
        now: () => Date.now(),
        pace: p,
        shownAt: () => shownAt.current,
      },
      sink,
      signal,
    ).catch((e: unknown) => {
      // A cancelled run ends quietly; anything else is a fault worth hearing about.
      if (!signal.aborted) console.error("boot: the waits stopped unexpectedly", e);
    });
    return () => ac.abort();
  }, [enabled, run, paint]);

  return { phase, shown, nodes, head, sub, slow, skip, reach, retry };
}
