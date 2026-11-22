/**
 * The two things a rendered answer needs that markdown itself cannot express:
 * inline citation markers, and the streaming caret.
 *
 * Both are structural edits to the parsed document rather than string surgery on
 * the model's text, which is what makes them safe: a `[2]` inside a code fence
 * stays code, and the caret lands on a real node instead of being concatenated
 * onto text that the parser would then have to re-read. Kept free of React so
 * the tree walk can be reasoned about on its own.
 */
import type { Element, Root, RootContent, Text } from "hast";

/** What the answer plugin needs to know about the message it is rendering. */
export interface AnswerMarkOptions {
  /**
   * The `[n]` markers this message actually has a source for, as the exact
   * decimal strings the backend numbered them with. A number outside it stays
   * plain text: a bare `[3]` in an answer with two sources is prose the model
   * wrote, not a reference, and dressing it as one would invent a citation.
   *
   * Strings rather than numbers on purpose - `"01"` is not `"1"`, so a padded
   * marker cannot coerce its way onto a source it was never given.
   */
  citationIndexes: ReadonlySet<string>;
  /** True while this message is still receiving deltas; places the caret. */
  live: boolean;
}

/**
 * Elements whose text is not prose, so a `[n]` inside them is left alone.
 *
 * `a` is here because a marker inside a link is part of the link's own label,
 * and wrapping it would nest a citation in an anchor.
 */
const OPAQUE_TO_MARKERS: ReadonlySet<string> = new Set(["a", "code", "pre"]);

/**
 * Class names that mark a formula (remark-math's wrappers and KaTeX's own
 * output): its text is notation, so a `[1]` inside it is an interval or index,
 * never a citation.
 */
const MATH_CLASSES: ReadonlySet<string> = new Set([
  "math-inline",
  "math-display",
  "katex",
  "katex-display",
  "katex-error",
]);

function isMath(node: Element): boolean {
  const classes = node.properties?.className;
  return Array.isArray(classes) && classes.some((name) => MATH_CLASSES.has(String(name)));
}

/**
 * Elements the caret may sink into, looking for the end of the streamed text.
 *
 * Everything here reads as running prose, so a caret at its end sits where the
 * next character will appear. A `pre`, `code` or `table` terminal is deliberately
 * absent: those lay their contents out themselves, and a blinking block inside
 * one reads as part of the code or the data.
 */
const CARET_DESCENDS_INTO: ReadonlySet<string> = new Set([
  "p",
  "h1",
  "h2",
  "h3",
  "h4",
  "h5",
  "h6",
  "li",
  "ul",
  "ol",
  "blockquote",
  "em",
  "strong",
  "del",
]);

/**
 * A `[n]` or `[n, m]` run, captured so the digits can be checked against the
 * sources.
 *
 * PARALLEL PATH - this grammar mirrors `_CITATION_MARKER` in the backend's
 * `app/services/prompts.py`, which is what counts an answer's markers and
 * renumbers them. The two must move in lockstep: a marker the backend counts
 * but this misses renders as bare prose next to a source chip nothing points
 * at, and one this matches but the backend does not gets dressed as a citation
 * that was never issued. Bounded to two digits for the same reason it is there
 * ("[2026]" is a year, not a source).
 *
 * The lookbehind is deliberately NOT copied: it is spelled out per match below
 * instead, since regex lookbehind is not something to lean on across browsers.
 * One knowing divergence: the backend parses each part as an int, so a padded
 * "[01]" counts as source 1 there while staying plain prose here (see
 * `namesSources`) - that answer shows a chip no styled marker points at, which
 * beats dressing a spelling the backend never emits.
 */
const CITATION_MARKER = /\[(\d{1,2}(?:\s*,\s*\d{1,2})*)\]/g;

/** Stands in for the backend's `(?<!\w)`; see `CITATION_MARKER`. */
const WORD_CHARACTER = /\w/;

/** The streamed-answer caret, as the element every theme's `.caret` rule styles. */
function caretElement(): Element {
  return {
    type: "element",
    tagName: "span",
    // The string, not the boolean: hast properties carry attribute values, and
    // aria-hidden is a "true"/"false" enumerated attribute rather than a flag.
    properties: { className: ["caret"], ariaHidden: "true" },
    children: [],
  };
}

/**
 * Split a captured marker group into its individual index strings:
 * `"2, 3"` -> `["2", "3"]`. Shared by the source check and the element build so
 * the two can never read the group differently.
 */
function groupIndexes(group: string): string[] {
  return group.split(",").map((part) => part.trim());
}

/**
 * One `[n]` marker, kept as the exact text it replaced.
 *
 * Carries the source indexes it names as a `data-citation-indexes` hast
 * property (a plain string array, never serialized) so the interactive
 * `CitationMarker` component - `components/chat/citation-popover.tsx` - can
 * resolve them straight off the parsed node (`react-markdown`'s `passNode`)
 * without re-parsing the label text.
 */
function citationElement(label: string, indexes: readonly string[]): Element {
  return {
    type: "element",
    tagName: "span",
    properties: { className: ["cite-mark"], "data-citation-indexes": [...indexes] },
    children: [{ type: "text", value: label }],
  };
}

/**
 * True when every number in a captured group names a source this message has.
 *
 * All-or-nothing: `[2, 3]` on a message with only source 2 stays plain text
 * rather than becoming a half-true reference, which matches how the backend
 * treats the group - one marker, one set of indexes.
 */
function namesSources(indexes: readonly string[], citationIndexes: ReadonlySet<string>): boolean {
  for (const part of indexes) {
    // Compared as text, never parsed: `Number("01")` is 1, and a padded marker
    // would otherwise borrow source 1's chip.
    if (!citationIndexes.has(part)) return false;
  }
  return true;
}

/**
 * Split one text node around the markers that name a real source.
 *
 * Returns null when the text carries none, so the common case replaces nothing
 * and the original node (and its position, which the renderer keys on) survives.
 */
function splitOnMarkers(
  value: string,
  citationIndexes: ReadonlySet<string>,
): Array<Element | Text> | null {
  // Reset rather than rebuilt: a global regex carries `lastIndex` between
  // calls, so the next node would be scanned from the middle.
  CITATION_MARKER.lastIndex = 0;
  const parts: Array<Element | Text> = [];
  let cursor = 0;
  let match: RegExpExecArray | null;
  while ((match = CITATION_MARKER.exec(value)) !== null) {
    // A bracket welded to a word is a subscript the model wrote - `argv[1]`,
    // `items[2]` - not a reference. `]` is not a word character, so adjacent
    // chains like `[1][2]` still both count.
    const before = match.index > 0 ? value[match.index - 1] : "";
    if (before !== "" && WORD_CHARACTER.test(before)) continue;
    const indexes = groupIndexes(match[1]);
    if (!namesSources(indexes, citationIndexes)) continue;
    if (match.index > cursor) {
      parts.push({ type: "text", value: value.slice(cursor, match.index) });
    }
    parts.push(citationElement(match[0], indexes));
    cursor = match.index + match[0].length;
  }
  if (parts.length === 0) return null;
  if (cursor < value.length) {
    parts.push({ type: "text", value: value.slice(cursor) });
  }
  return parts;
}

/**
 * hast splits a root's children (`RootContent`) from an element's
 * (`ElementContent`), and this walk crosses between the two. It only ever reads
 * existing children and inserts elements and text - members of both - so one
 * array type carries the whole tree without widening anything at runtime.
 */
function childrenOf(node: Element | Root): RootContent[] {
  return node.children as RootContent[];
}

/** Rewrite every prose text node in the subtree, depth first, in place. */
function markCitations(node: Element | Root, citationIndexes: ReadonlySet<string>): void {
  const children = childrenOf(node);
  // Backwards, so splicing extra nodes in cannot shift the cursor onto text that
  // was already rewritten.
  for (let i = children.length - 1; i >= 0; i--) {
    const child = children[i];
    if (child.type === "text") {
      const parts = splitOnMarkers(child.value, citationIndexes);
      if (parts) children.splice(i, 1, ...parts);
    } else if (child.type === "element" && !OPAQUE_TO_MARKERS.has(child.tagName) && !isMath(child)) {
      markCitations(child, citationIndexes);
    }
  }
}

/** Index of the last child that is not layout whitespace, or -1 when there is none. */
function lastMeaningfulIndex(children: RootContent[]): number {
  for (let i = children.length - 1; i >= 0; i--) {
    const child = children[i];
    if (child.type === "text" && child.value.trim() === "") continue;
    return i;
  }
  return -1;
}

/**
 * Put the caret at the end of the streamed text.
 *
 * It follows the tree's right edge down through prose (a list, into its last
 * item, into that item's paragraph) so it ends up where the next token will
 * land, rather than under the whole answer. The descent stops at anything that
 * lays itself out - a code block, a table - and the caret becomes its sibling.
 * The block separators markdown leaves between nodes are skipped, or a trailing
 * newline would strand the caret at the top level of a finished paragraph.
 */
function placeCaret(tree: Root): void {
  let target: Element | Root = tree;
  let children = childrenOf(target);
  let at = lastMeaningfulIndex(children);
  while (at !== -1) {
    const last = children[at];
    if (last.type !== "element" || !CARET_DESCENDS_INTO.has(last.tagName)) break;
    target = last;
    children = childrenOf(target);
    at = lastMeaningfulIndex(children);
  }
  // Directly after the content rather than at the array's end, so trailing
  // whitespace nodes do not push the caret away from the words.
  children.splice(at + 1, 0, caretElement());
}

/**
 * Build the rehype plugin that turns a parsed answer into a rendered one.
 *
 * One pass per render: the citation walk rewrites prose text nodes, then the
 * caret is placed down the tree's right edge. Both are omitted when they have
 * nothing to do, so a settled message with no sources costs a single traversal
 * that changes nothing.
 */
export function answerMarkPlugin({ citationIndexes, live }: AnswerMarkOptions) {
  return function attach() {
    return function transform(tree: Root): undefined {
      if (citationIndexes.size > 0) markCitations(tree, citationIndexes);
      if (live) placeCaret(tree);
    };
  };
}
