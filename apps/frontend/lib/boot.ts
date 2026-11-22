/**
 * The boot screen's pure half: the waits the shell blocks on, in order, and
 * the words for each; the reach line under the new chat's greeting.
 *
 * Data and arithmetic only (DESIGN.md, "Sign-in, boot and new chat"), so the
 * battery can hold the copy and the pacing without a browser. The screen
 * itself is components/boot.tsx; the waits are run by hooks/use-boot.ts.
 */

import type { SkyState } from "@/lib/sky";

/** The three waits, in the order the rail shows them. */
export type BootNode = "backend" | "model" | "data";

/** Where one wait is: not started, running, or over. */
export type BootPhase = "pending" | "live" | "done";

/** A headline with its one lit word: the part before, the word, the part after. */
export type LitLine = readonly [string, string, string];

/** What the screen says for one wait. */
export interface BootWait {
  node: BootNode;
  /** The node's label on the rail. */
  label: string;
  /** The headline while this wait runs. */
  head: LitLine;
  /** The mono subline while this wait runs. */
  waiting: string;
  /** The sky's weather while this wait runs. */
  sky: Extract<SkyState, "boot" | "boot-model" | "boot-data">;
  /**
   * The least this wait's node stays lit, even when the wait ends at once,
   * so the rail reads. The model's is longer: its violet eases in over a
   * second, and a blink of it would read as a fault.
   */
  dwell: number;
  /** What a long wait says, for the slow box. */
  slow: string;
}

/** The least a lit node stays lit before the next wait starts, so the rail reads. */
export const NODE_MIN_MS = 450;

/** The model's node stays lit long enough for the sky's violet to arrive (it eases in over 1.2s). */
export const MODEL_MIN_MS = 1_000;

export const BOOT_WAITS: readonly BootWait[] = [
  {
    node: "backend",
    label: "Backend",
    head: ["Reaching the ", "backend", ""],
    waiting: "waiting for the API",
    sky: "boot",
    dwell: NODE_MIN_MS,
    slow:
      "The backend is taking longer than usual to answer. If the stack was just started it may still be coming up; otherwise check that the backend is running.",
  },
  {
    node: "model",
    label: "Model runtime",
    head: ["Waking the ", "model", ""],
    waiting: "reaching the model runtime",
    sky: "boot-model",
    dwell: MODEL_MIN_MS,
    slow:
      "The model runtime is taking longer than usual to answer. If the stack was just started it may still be coming up; otherwise check that the model service is running.",
  },
  {
    node: "data",
    label: "Conversations",
    head: ["Gathering your ", "conversations", ""],
    waiting: "fetching what you have",
    sky: "boot-data",
    dwell: NODE_MIN_MS,
    slow:
      "Your conversations are taking longer than usual to arrive. If the stack was just started it may still be coming up; otherwise check that the database is running.",
  },
];

/** The headline once every wait is over. */
export const READY_HEAD: LitLine = ["", "Ready", ""];

/** A wait that runs this long is called slow: the screen says so and offers a retry. */
export const SLOW_AFTER_MS = 15_000;

/** How long "Ready" stands before the shell arrives. */
export const READY_DWELL_MS = 700;

/** How often the backend and the model runtime are asked again while they do not answer. */
export const RETRY_EVERY_MS = 2_000;

/**
 * How the boot paces itself. The first load of a browser session gets the
 * full choreography: each node holds its beat and "Ready" stands before the
 * shell. Every later load is `quick`: the same waits, but no beat is held,
 * so against a backend that answers at once the shell is a few hundred
 * milliseconds away and the boot screen is never painted at all.
 */
export type BootPace = "full" | "quick";

/** The pace for a load, given whether this browser session has booted before. */
export function paceFor(seen: boolean): BootPace {
  return seen ? "quick" : "full";
}

/** A beat's length under a pace: held in full, dropped in quick. */
export function dwellFor(pace: BootPace, ms: number): number {
  return pace === "quick" ? 0 : ms;
}

/** On a quick load the boot screen is painted only once a wait has run this long. */
export const BOOT_DEFER_MS = 600;

/** A boot screen that was painted stays at least this long, so it never flickers past. */
export const BOOT_SHOWN_MIN_MS = 700;

/**
 * How much longer a painted boot screen must stand, `now` being the clock
 * and `shownAt` when it was painted (null when it never was).
 */
export function holdLeft(shownAt: number | null, now: number): number {
  if (shownAt === null) return 0;
  return Math.max(0, shownAt + BOOT_SHOWN_MIN_MS - now);
}

/** The lit share of the rail's line for a number of finished waits. */
export function litShare(done: number, total: number = BOOT_WAITS.length): number {
  if (total <= 1) return done > 0 ? 1 : 0;
  return Math.min(1, Math.max(0, done / (total - 1)));
}

/** The headline for a wait that has run long: "Still reaching the backend". */
export function slowHead(head: LitLine): LitLine {
  const before = head[0];
  return [`Still ${before.charAt(0).toLowerCase()}${before.slice(1)}`, head[1], head[2]];
}

/** The subline that counts a long wait: "17 s and counting". */
export function countingLabel(elapsedMs: number): string {
  return `${Math.max(0, Math.round(elapsedMs / 1000))} s and counting`;
}

/** "4 conversations", "1 document". */
export function countOf(n: number, noun: string, plural: string = `${noun}s`): string {
  return `${n} ${n === 1 ? noun : plural}`;
}

/** Bytes as the rail writes them: "2.5 GB", "640 MB". */
export function sizeLabel(bytes: number): string {
  const gb = bytes / 1024 ** 3;
  if (gb >= 1) return `${gb < 10 ? gb.toFixed(1) : Math.round(gb)} GB`;
  return `${Math.round(bytes / 1024 ** 2)} MB`;
}

/** One backing service as the readiness report describes it. */
export interface DependencyCheck {
  name: string;
  required: boolean;
  state: "ok" | "unreachable" | "unconfigured";
  latency_ms: number;
  detail: string | null;
}

/** The readiness report, as `/readyz` answers it. */
export interface ReadinessReport {
  status: "ready" | "unready";
  checks: DependencyCheck[];
}

/** The report's line for the model runtime, whatever the deployment calls it. */
export function modelCheck(report: ReadinessReport): DependencyCheck | null {
  return (
    report.checks.find((c) => c.name === "ollama" || c.name === "vllm" || /model/.test(c.name)) ??
    null
  );
}

/** The report's line for the vector store (the knowledge base). */
export function vectorStoreCheck(report: ReadinessReport): DependencyCheck | null {
  return report.checks.find((c) => c.name === "qdrant" || /vector/.test(c.name)) ?? null;
}

/**
 * What the backend's node says once it answered. A report that is not ready
 * names the required service that is down; the wait goes on.
 */
export function backendDetail(report: ReadinessReport): { done: boolean; detail: string } {
  if (report.status === "ready") return { done: true, detail: "api ready" };
  const down = report.checks.find((c) => c.required && c.state !== "ok");
  return { done: false, detail: down ? `${down.name} ${down.state}` : "not ready" };
}

/**
 * What the model node says: the default model and its size when the runtime
 * holds it, "on disk" when it will load on the first turn, or that none is
 * chosen, which the composer's first-run notice then explains.
 */
export function modelDetail(
  model: string | null,
  resident: { model: string; total_bytes: number | null } | null,
): string {
  if (model === null) return "no model chosen";
  if (resident !== null) {
    return resident.total_bytes === null ? model : `${model}, ${sizeLabel(resident.total_bytes)}`;
  }
  return `${model}, on disk`;
}

/** What the data node says: "4 conversations, 128 documents". */
export function dataDetail(conversations: number, documents: number | null): string {
  const parts = [countOf(conversations, "conversation")];
  if (documents !== null) parts.push(countOf(documents, "document"));
  return parts.join(", ");
}

/** One item of the reach line: the rail's glyph, a label, and the fact. */
export interface ReachItem {
  glyph: "search" | "globe" | "plug";
  label: string;
  value: string;
}

/** What the install can reach, as far as the page could learn it. */
export interface Reach {
  /** The knowledge base's document count; null when the person may not list it. */
  documents: number | null;
  /** The store held more than the count returned. */
  truncated: boolean;
  /** The vector store answers, so the knowledge base is there to search. */
  knowledgeReady: boolean;
  /** The workspace's web search default. */
  webSearch: "auto" | "on" | "off" | null;
  /** Connected tool servers; null when they could not be listed. */
  tools: number | null;
}

/** The web search word for the reach line: auto lets the model search, so it reads as on. */
export function webSearchWord(mode: Reach["webSearch"]): string | null {
  if (mode === null) return null;
  return mode === "off" ? "off" : "on";
}

/**
 * The reach line's items. A fact the page could not learn leaves its item
 * out rather than guessing; a knowledge base the person may not list but
 * which answers reads as "ready".
 */
export function reachItems(reach: Reach): ReachItem[] {
  const items: ReachItem[] = [];
  if (reach.documents !== null) {
    items.push({
      glyph: "search",
      label: "Knowledge base",
      value: `${reach.documents}${reach.truncated ? "+" : ""} ${reach.documents === 1 && !reach.truncated ? "document" : "documents"}`,
    });
  } else if (reach.knowledgeReady) {
    items.push({ glyph: "search", label: "Knowledge base", value: "ready" });
  }
  const web = webSearchWord(reach.webSearch);
  if (web !== null) items.push({ glyph: "globe", label: "Web search", value: web });
  if (reach.tools !== null) {
    items.push({
      glyph: "plug",
      label: "Tools",
      value: reach.tools === 0 ? "none connected" : `${reach.tools} connected`,
    });
  }
  return items;
}
