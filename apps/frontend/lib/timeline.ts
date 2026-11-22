/**
 * The timeline turn's small arithmetic: what the rail says beside each step,
 * what the thinking row says once it has closed, and what the reply footer's
 * figures are.
 *
 * Pure, so the battery can pin the wording. The clock is always passed in;
 * nothing here reads `Date.now()`.
 */

/** The rail's elapsed-time label for a step, `+6s` or `+1m04s`; null when unknown. */
export function elapsedLabel(questionAt: string | undefined, at: number | undefined): string | null {
  if (!questionAt || at === undefined) return null;
  const from = new Date(questionAt).getTime();
  if (Number.isNaN(from)) return null;
  const seconds = Math.max(0, Math.round((at - from) / 1000));
  if (seconds < 60) return `+${seconds}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = seconds % 60;
  return `+${minutes}m${String(rest).padStart(2, "0")}s`;
}

/** A turn's clock time in the reader's locale (`14:26`), or null when unknown. */
export function clockLabel(iso: string | undefined): string | null {
  if (!iso) return null;
  const at = new Date(iso);
  if (Number.isNaN(at.getTime())) return null;
  return at.toLocaleTimeString(undefined, { hour: "2-digit", minute: "2-digit" });
}

/**
 * The closed thinking row's label.
 *
 * "Thought for N seconds" when the reasoning's span is known (it began at
 * `thinkingAt` and ended when the next thing started: a tool, the answer, or
 * the turn's end); "Thought" alone on a reloaded turn, where no clock says.
 */
export function thoughtLabel(thinkingAt: number | undefined, endedAt: number | undefined): string {
  if (thinkingAt === undefined || endedAt === undefined || endedAt < thinkingAt) return "Thought";
  const seconds = Math.max(1, Math.round((endedAt - thinkingAt) / 1000));
  return `Thought for ${seconds} second${seconds === 1 ? "" : "s"}`;
}

/**
 * Tokens per second over the time the runtime spent writing them, rounded
 * to a whole number. Both figures are the server's (the turn's count, the
 * runtime's own writing time), so a tool's run time or a wait on the reader
 * is never in the span. Null when either is missing, or the span is too
 * short to say anything (under half a second reads as noise).
 */
export function tokensPerSecond(
  tokens: number | undefined,
  generationMs: number | undefined,
): number | null {
  if (tokens === undefined || generationMs === undefined) return null;
  const span = generationMs / 1000;
  if (!(span >= 0.5) || tokens <= 0) return null;
  return Math.round(tokens / span);
}

/** The kind of glyph a tool's rail step wears, from its identifier. */
export type ToolGlyph = "search" | "globe" | "plug";

/**
 * The tool row's trailing count: "3 sources", "1 source", or null.
 *
 * Null for an absent or non-positive count - the backend sends null when the
 * tool cannot say, and a zero would only repeat the "nothing found" summary.
 */
export function sourceCountLabel(count: number | null | undefined): string | null {
  if (typeof count !== "number" || !(count >= 1)) return null;
  return count === 1 ? "1 source" : `${count} sources`;
}

/**
 * Which glyph narrates a tool on the rail.
 *
 * The two shipped tools get the glyph of their medium (a magnifier for the
 * knowledge base, a globe for the web); anything else reads as a connected
 * tool. Name-based, so a remote tool called `web_lookup` is still a plug: the
 * glyph says what kind of thing ran, and only the exact built-ins are known.
 */
export function toolGlyph(name: string): ToolGlyph {
  if (name === "search_knowledge") return "search";
  if (name === "web_search") return "globe";
  return "plug";
}
