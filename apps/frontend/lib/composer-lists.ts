/**
 * Send-time hygiene for the composer's list markers and empty code fences.
 *
 * Pure string work over the draft as it would go out. The composer used to be a
 * plain text box that continued list markers itself; that job now belongs to
 * the editor, which renders items as glyphs. What survives here is the one rule
 * that is about SENDING rather than typing - a marker offered and never filled
 * in is not a message - and it stays a string rule so the Send button and the
 * send path read the draft the same way.
 *
 * Only the markers people actually type into a chat box are recognized -
 * bullets, task items, ordered items - and only when whitespace follows the
 * marker, so "-item" and "1.5" stay ordinary prose. An EMPTY code fence pair
 * counts as the same kind of offer: the editor keeps an empty code block alive
 * so it can be typed into, and a draft that ends on its bare opener + closer
 * says as little as a bare "- ".
 */

// Tried in this order, because a task item IS a bullet with a checkbox: the
// plain-bullet pattern would match "- [x] done" first and read it as a bare
// "- ", quietly treating a finished checklist entry as an empty offer.
const TASK_ITEM = /^([ \t]*)([-*+])[ \t]+\[[ xX]\][ \t]+/;
const BULLET_ITEM = /^([ \t]*)([-*+])[ \t]+/;
const ORDERED_ITEM = /^([ \t]*)(\d+)([.)])[ \t]+/;

// A code fence: its backtick run, then whatever follows it on the line. An
// opener may name a language; a closer is the run and nothing else. Which one a
// given line is depends on what came before it AND on the run's length - a
// three-backtick line inside a four-backtick block is content, not the end.
const CODE_FENCE = /^[ \t]*(`{3,})([^`]*)$/;

// Blank lines at the end of a draft. Note what it does NOT eat: the space that
// makes "- " a marker, which sits on the last line with content rather than
// after it.
const TRAILING_BLANK_LINES = /(?:\r?\n[ \t]*)+$/;

/** Whether `line` opens with a list marker that has nothing after it. */
function isBareMarker(line: string): boolean {
  for (const pattern of [TASK_ITEM, BULLET_ITEM, ORDERED_ITEM]) {
    const matched = pattern.exec(line);
    if (matched !== null) return line.slice(matched[0].length).trim() === "";
  }
  return false;
}

/** Start index of the line containing `index`. */
function lineStartAt(value: string, index: number): number {
  return value.lastIndexOf("\n", index - 1) + 1;
}

/**
 * Whether the line starting at `lineStart` sits inside an unclosed code fence.
 *
 * A fence left open above it means the text there is code, where a "-" is a
 * minus sign and a "1." is a decimal point - trimming either would corrupt what
 * the user is quoting.
 *
 * Fences are matched by RUN LENGTH, not counted. A block whose content holds
 * three backticks is written with four, and counting lines would read that
 * inner run as a closer, leave the count odd, and then treat every marker after
 * the real closing fence as untouchable code.
 */
function insideCodeFence(value: string, lineStart: number): boolean {
  if (lineStart === 0) return false;
  let open = 0;
  for (const line of value.slice(0, lineStart - 1).split("\n")) {
    const fence = CODE_FENCE.exec(line);
    if (fence === null) continue;
    const width = fence[1].length;
    // Only an opener may name a language, so a line that does is never a close.
    if (open === 0) open = width;
    else if (width >= open && fence[2].trim() === "") open = 0;
  }
  return open !== 0;
}

/**
 * Start index of the empty fence pair ending `body`, or null when it ends on
 * anything else.
 *
 * The pair is an opener DIRECTLY above a closer - the one shape an abandoned
 * empty code block serializes to. A closer is a run at least as long as the
 * opener's with nothing else on the line; only an opener may name a language.
 * A fence holding anything, even a blank line, has been typed into and is a
 * message. A pair sitting inside a WIDER open fence is content, not an offer.
 */
function emptyFencePairStart(body: string, lineStart: number): number | null {
  const closer = CODE_FENCE.exec(body.slice(lineStart));
  if (closer === null || closer[2].trim() !== "" || lineStart === 0) return null;
  const openStart = lineStartAt(body, lineStart - 1);
  const opener = CODE_FENCE.exec(body.slice(openStart, lineStart - 1));
  if (opener === null || closer[1].length < opener[1].length) return null;
  return insideCodeFence(body, openStart) ? null : openStart;
}

/**
 * Drop trailing content that only ever existed as an offer, on the way out.
 *
 * Two shapes qualify. A marker with nothing after it: a draft can end on a
 * freshly continued item the user never filled in, and trimming does not save
 * them - `"- ".trim()` is `"-"`, a perfectly sendable message that says
 * nothing. And an empty fence pair: the editor keeps an empty code block alive
 * to be typed into, so a draft can end on one the user meant to delete, and a
 * bare opener + closer is as empty an offer as the bare marker. Stripping
 * repeats until the tail is honest, so a marker stranded above a dropped pair
 * goes with it.
 *
 * The last line with anything on it is the one judged, not the last line: text
 * arriving by paste or drop routinely ends in a newline, and a blank line after
 * the marker would otherwise hide it right up until trimming brought it back.
 *
 * Left alone inside a code fence, where the line is content rather than an offer.
 *
 * @param value - The whole draft, as it would be sent.
 * @returns The draft without its trailing bare markers and empty fence pairs,
 *   or unchanged.
 */
export function stripTrailingBareMarker(value: string): string {
  let out = value;
  for (;;) {
    const body = out.replace(TRAILING_BLANK_LINES, "");
    const lineStart = lineStartAt(body, body.length);
    // Cheap tests first: this runs on every render (the Send button reads the
    // same draft), and scanning for fences is only worth it once a tail shape
    // has matched.
    if (isBareMarker(body.slice(lineStart))) {
      if (insideCodeFence(body, lineStart)) return out;
      out = body.slice(0, lineStart);
      continue;
    }
    const openStart = emptyFencePairStart(body, lineStart);
    if (openStart === null) return out;
    out = body.slice(0, openStart);
  }
}
