/**
 * LaTeX delimiters, rewritten into the only math syntax the parser is given.
 *
 * Models write display math as `\[...\]` and inline math as `\(...\)`, but
 * `remark-math` is configured with single-dollar math OFF (so "costs $5 and
 * $10" can never become a formula) and understands only `$$`. What remains of
 * its syntax is: a `$$` fence on its own lines for display, and `$$x$$` inside
 * a line for inline. This module is the bridge, run on the raw text before it
 * is parsed. Pure and React-free so it can be tested on its own.
 *
 * Models also write inline math as `$x$`. The parser's own single-dollar mode
 * stays off (it cannot tell "$5 and $10" from a formula), so this module is the
 * only gate: `$body$` becomes inline math when it follows pandoc's rule and
 * does not read as money (see `singleDollar`). A run of three or more `$` is
 * escaped to literal text.
 *
 * A `$$body$$` the model wrote itself means display math, but the parser reads
 * it inline when text shares its line; it is lifted into a fenced block (see
 * `doubleDollar`). Our own `\(...\)` and `$...$` rewrites also come out as
 * inline `$$x$$`, but they are emitted, never rescanned, so the two cannot be
 * confused. A multi-line `$$` that opens a line with text and closes on a later
 * line of its own is not recognised and stays literal.
 *
 * Left untouched on purpose: fenced code blocks, inline code spans, Markdown
 * link destinations, an unclosed delimiter (a streaming answer whose closer has
 * not arrived yet), a backslash-escaped one (`\\[`), anything that would span a
 * blank line, and bodies that contain `$$` (they would end the rewritten form
 * early). A model-written `$$` fence that never closes is neutralised to literal
 * text, since the parser would otherwise run it to the end of the answer.
 *
 * Known limits: indented (4-space) code blocks are NOT recognised, so a `\(`
 * inside one is rewritten like any other text; and a formula containing `|`
 * inside a table row is split into cells by the table parser before math is
 * read, so write such formulas outside the table.
 *
 * Every scan is linear in the input: this runs on each streamed token, so a
 * pathological answer must not cost more than its length. Searches that fail
 * once are remembered, because a failed search from one position means every
 * later search for the same thing fails too.
 */

const FENCE_OPEN = /^( {0,3})(`{3,}|~{3,})/;
const DOLLAR_LINE = /^[ \t]*(?:>[ \t]*)*\$\$[ \t\r]*$/;
const PARAGRAPH_GAP = /(\r?\n[ \t]*\r?\n)/;
const LINE_PREFIX = /^[ \t>]*/;
const QUOTE_LEAD = /^[ \t]*(?:>[ \t]*)*/;
const LINE_END = /[ \t\r]*(?:\n|$)/y;
const NON_SPACE = /\S/g;
const WORD_CHAR = /[\p{L}\p{N}]/u;
const DIGIT = /\d/;
const SPACE = /\s/;
/** Anything that reads as notation rather than a price: `^`, `=`, `\alpha`, `n-1`... */
const MATH_SIGNAL = /[\\^_=+<>{]|[A-Za-z][-*/]|[-*/][A-Za-z]/;

/** Longest `$...$` body converted; a longer span is prose between two prices. */
const MAX_INLINE_BODY = 200;

/** Longest link destination looked at; a longer one is treated as prose. */
const MAX_DESTINATION = 512;

/** Rewrite `\[...\]` and `\(...\)` outside code into `$$` math. */
export function normalizeMathDelimiters(source: string): string {
  if (!source.includes("\\[") && !source.includes("\\(") && !source.includes("$")) {
    return source;
  }

  const lines = source.split("\n");
  const out: string[] = [];
  let prose: string[] = [];
  let fence: { char: string; length: number } | null = null;
  // Index in `out` of a `$$` line that opened a block not yet closed.
  let dollarOpen: number | null = null;

  const flushProse = () => {
    if (prose.length > 0) out.push(...rewriteProse(prose.join("\n")).split("\n"));
    prose = [];
  };

  for (const line of lines) {
    if (fence) {
      out.push(line);
      const close = FENCE_OPEN.exec(line);
      if (
        close &&
        close[2][0] === fence.char &&
        close[2].length >= fence.length &&
        line.slice(close[0].length).trim() === ""
      ) {
        fence = null;
      }
      continue;
    }
    if (dollarOpen !== null) {
      out.push(line);
      if (DOLLAR_LINE.test(line)) dollarOpen = null;
      continue;
    }
    if (DOLLAR_LINE.test(line)) {
      flushProse();
      dollarOpen = out.length;
      out.push(line);
      continue;
    }
    const open = FENCE_OPEN.exec(line);
    if (open) {
      flushProse();
      fence = { char: open[2][0], length: open[2].length };
      out.push(line);
      continue;
    }
    prose.push(line);
  }
  flushProse();
  if (dollarOpen !== null) out[dollarOpen] = out[dollarOpen].replace("$$", "\\$\\$");
  return out.join("\n");
}

/** One run of non-fence lines: split into paragraphs so nothing spans a blank line. */
function rewriteProse(text: string): string {
  const parts = text.split(PARAGRAPH_GAP);
  // Odd entries are the captured gaps, kept verbatim.
  return parts.map((part, k) => (k % 2 === 1 ? part : rewriteParagraph(part))).join("");
}

/** A formula body as trimmed lines, quote markers stripped from continuations. */
function bodyLines(body: string, quote: boolean): string[] {
  return body
    .split("\n")
    .map((line, k) => (k > 0 && quote ? line.replace(QUOTE_LEAD, "") : line).trim())
    .filter((line) => line !== "");
}

function rewriteParagraph(text: string): string {
  const n = text.length;
  const out: string[] = [];

  // What the line being written looks like so far: `prefix` is its leading
  // whitespace and quote markers while nothing else has been written on it.
  let prefix: string | null = "";
  let quoted = false;
  // The prefix of the current line once content has followed it.
  let lineLead = "";
  let lastChar = "";
  const push = (s: string) => {
    if (s === "") return;
    out.push(s);
    lastChar = s[s.length - 1];
    let rest = s;
    const cut = s.lastIndexOf("\n");
    if (cut >= 0) {
      prefix = "";
      quoted = false;
      rest = s.slice(cut + 1);
    }
    if (prefix !== null) {
      const head = LINE_PREFIX.exec(rest)![0];
      prefix += head;
      if (head.length < rest.length) {
        quoted = prefix.includes(">");
        lineLead = prefix;
        prefix = null;
      }
    }
  };
  const inQuote = () => (prefix !== null ? prefix.includes(">") : quoted);
  /** Drop spaces left dangling at the end of the output before a line break. */
  const trimTail = () => {
    while (out.length > 0) {
      const trimmed = out[out.length - 1].replace(/[ \t]+$/, "");
      if (trimmed !== "") {
        out[out.length - 1] = trimmed;
        break;
      }
      out.pop();
    }
    lastChar = out.length > 0 ? out[out.length - 1].slice(-1) : "";
  };

  // Failed searches: nothing of this kind exists at or after the position.
  const noRunFrom = new Map<number, number>();
  const noCloserFrom: Record<string, number> = { "]": Infinity, ")": Infinity };
  // The last successful closer search: [from, closer position].
  const foundCloser: Record<string, [number, number]> = { "]": [-1, -1], ")": [-1, -1] };
  let nextPair = -2; // cached position of the next `$$` at or after a point

  const findRun = (from: number, length: number): number => {
    const dead = noRunFrom.get(length);
    if (dead !== undefined && from >= dead) return -1;
    let i = from;
    while (i < n) {
      if (text[i] !== "`") {
        i++;
        continue;
      }
      let run = 1;
      while (text[i + run] === "`") run++;
      if (run === length) return i;
      i += run;
    }
    noRunFrom.set(length, from);
    return -1;
  };

  const findCloser = (from: number, closer: string): number => {
    if (from >= noCloserFrom[closer]) return -1;
    const [cachedFrom, cachedEnd] = foundCloser[closer];
    // No closer lies between the cached search's start and its result, so any
    // search starting inside that span ends at the same place.
    if (from >= cachedFrom && from <= cachedEnd && cachedFrom !== -1) return cachedEnd;
    for (let j = from; j < n; j++) {
      if (text[j] !== "\\") continue;
      if (text[j + 1] === closer) {
        foundCloser[closer] = [from, j];
        return j;
      }
      j += 1;
    }
    noCloserFrom[closer] = from;
    return -1;
  };

  const hasPairBefore = (from: number, end: number): boolean => {
    if (nextPair === -2 || (nextPair !== -1 && nextPair < from)) {
      nextPair = text.indexOf("$$", from);
    }
    return nextPair !== -1 && nextPair + 1 < end;
  };

  const isBlank = (from: number, end: number): boolean => {
    NON_SPACE.lastIndex = from;
    const hit = NON_SPACE.exec(text);
    return hit === null || hit.index >= end;
  };

  /**
   * End of a `(...)` link destination starting at `open`, or -1. A destination
   * still streaming in (no closer yet, nothing but its own characters) runs to
   * the end, so its parentheses do not flicker into formulas as it grows.
   */
  const destinationEnd = (open: number): number => {
    let depth = 0;
    const limit = Math.min(n, open + MAX_DESTINATION);
    for (let j = open; j < limit; j++) {
      const c = text[j];
      if (c === "\\") {
        j++;
      } else if (c === "(") {
        depth++;
      } else if (c === ")") {
        depth--;
        if (depth === 0) return j + 1;
      } else if (c === " " || c === "\t" || c === "\n") {
        return -1;
      }
    }
    // Ran off the end with no whitespace: a destination still arriving.
    return limit === n ? n : -1;
  };

  /**
   * A model-written `$$body$$` opening at `open`: [display block, index after
   * it], or null when there is no usable closer.
   *
   * By convention `$$...$$` means display math, but the parser reads it inline
   * when it shares a line with text. So it is lifted into its own fence, with
   * a blank line either side where text sits next to it, keeping the quote or
   * indent prefix of the line it was written on. The search stops at the first
   * `$`, which must start a lone `$$`, so it is bounded by the next dollar.
   */
  const doubleDollar = (open: number): [string, number] | null => {
    const k = text.indexOf("$", open + 2);
    if (k === -1 || text[k + 1] !== "$" || text[k + 2] === "$") return null;
    const body = text.slice(open + 2, k);
    if (body.includes("`")) return null;
    const lines = bodyLines(body, inQuote());
    if (lines.length === 0) return null;
    let after = k + 2;
    let head = "";
    let lead = prefix;
    if (lead === null) {
      lead = lineLead;
      trimTail();
      head = `\n${lead.trimEnd()}\n${lead}`;
    }
    LINE_END.lastIndex = after;
    const ownsTail = LINE_END.test(text);
    const tail = ownsTail ? "" : `\n${lead.trimEnd()}\n${lead}`;
    // The text after the block starts a fresh line; its leading spaces would
    // otherwise pile up as indent, one more with every block on a line.
    if (!ownsTail) while (text[after] === " " || text[after] === "\t") after++;
    const block = `${head}$$\n${lines.map((line) => lead + line).join("\n")}\n${lead}$$${tail}`;
    return [block, after];
  };

  /** The `$$` form of one formula, or null when it should stay as written. */
  const rewrite = (body: string, display: boolean, after: number): string | null => {
    LINE_END.lastIndex = after;
    const ownsLine = prefix !== null && LINE_END.test(text);
    const quote = inQuote();
    if (display && ownsLine) {
      const lines = bodyLines(body, quote);
      if (lines.length === 0) return null;
      return `$$\n${lines.map((line) => prefix + line).join("\n")}\n${prefix}$$`;
    }
    let inner = body.replace(quote ? /\s*\n[ \t]*(?:>[ \t]*)*/g : /\s*\n[ \t]*/g, " ").trim();
    if (inner === "") return null;
    // A formula touching a `$` would fuse its fence with it; the space keeps
    // the fences apart and is stripped by the parser when it pads both sides.
    if (inner.startsWith("$") || inner.endsWith("$")) inner = ` ${inner} `;
    const lead = lastChar === "$" ? " " : "";
    const trail = text[after] === "$" ? " " : "";
    return `${lead}$$${inner}$$${trail}`;
  };

  /**
   * A `$body$` formula opening at `open`: [rewritten text, index after it], or
   * null when it should stay as written.
   *
   * Pandoc's rule keeps prices out: the opener must not follow a word
   * character and must touch its body, the closer must touch the body and must
   * not be followed by a digit ("$5 and $10" fails on the space before the
   * second `$`; "$5,$10" on the digit). Bodies that open with a digit and show
   * no notation are money too ("$5$"). The scan is one line and bounded, so a
   * stray `$` costs a constant.
   */
  const singleDollar = (open: number): [string, number] | null => {
    const before = open > 0 ? text[open - 1] : "";
    if (before !== "" && (before === "\\" || WORD_CHAR.test(before))) return null;
    const first = text[open + 1];
    if (first === undefined || SPACE.test(first)) return null;
    const limit = Math.min(n, open + 2 + MAX_INLINE_BODY);
    for (let j = open + 1; j < limit; j++) {
      const c = text[j];
      if (c === "\n") return null;
      if (c !== "$") continue;
      const last = text[j - 1];
      const next = text[j + 1];
      if (SPACE.test(last) || last === "\\") return null;
      if (next !== undefined && (DIGIT.test(next) || next === "$")) return null;
      const body = text.slice(open + 1, j);
      if (body.includes("`")) return null;
      if (DIGIT.test(body[0]) && !MATH_SIGNAL.test(body)) return null;
      const done = rewrite(body, false, j + 1);
      return done === null ? null : [done, j + 1];
    }
    return null;
  };

  let i = 0;
  while (i < n) {
    const ch = text[i];
    if (ch === "`") {
      // A code span closes on a run of exactly the same length; with no
      // closer the backticks are literal text (CommonMark).
      let run = 1;
      while (text[i + run] === "`") run++;
      const close = findRun(i + run, run);
      if (close === -1) {
        push(text.slice(i, i + run));
        i += run;
      } else {
        push(text.slice(i, close + run));
        i = close + run;
      }
      continue;
    }
    if (ch === "]" && text[i + 1] === "(") {
      const end = destinationEnd(i + 1);
      if (end !== -1) {
        push(text.slice(i, end));
        i = end;
      } else {
        push("]");
        i += 1;
      }
      continue;
    }
    if (ch === "\\") {
      const next = text[i + 1];
      if (next === "[" || next === "(") {
        const closer = next === "[" ? "]" : ")";
        const end = findCloser(i + 2, closer);
        if (end !== -1 && !isBlank(i + 2, end) && !hasPairBefore(i + 2, end)) {
          const done = rewrite(text.slice(i + 2, end), next === "[", end + 2);
          if (done !== null) {
            push(done);
            i = end + 2;
            continue;
          }
        }
      }
      // An escape pair is consumed whole so `\\[` is not read as `\[`.
      push(text.slice(i, i + 2));
      i += 2;
      continue;
    }
    if (ch === "$") {
      let run = 1;
      while (text[i + run] === "$") run++;
      if (run === 1) {
        const hit = singleDollar(i);
        if (hit !== null) {
          push(hit[0]);
          i = hit[1];
          continue;
        }
      } else if (run === 2) {
        const hit = doubleDollar(i);
        if (hit !== null) {
          push(hit[0]);
          i = hit[1];
          continue;
        }
        // Unclosed at the start of a line, the parser reads `$$` as a math
        // fence and runs it to the end of the answer; escaped it stays prose.
        push(prefix !== null ? "\\$\\$" : "$$");
        i += 2;
        continue;
      }
      // A run of three or more is never a fence; escaped, it cannot pair up
      // with a later `$$` into a formula that swallows the prose between.
      push(run >= 3 ? "\\$".repeat(run) : text.slice(i, i + run));
      i += run;
      continue;
    }
    let j = i + 1;
    while (j < n && text[j] !== "`" && text[j] !== "\\" && text[j] !== "]" && text[j] !== "$") j++;
    push(text.slice(i, j));
    i = j;
  }
  return out.join("");
}
