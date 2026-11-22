/**
 * The wire between the composer's editor nodes and the draft string.
 *
 * The composer renders list bullets, ordinals, task checkboxes and code blocks
 * as glyphs, but what gets SENT is still the markdown a person would have typed
 * into a plain box. These two functions are that contract, and they are written
 * by hand rather than taken from the editor library's own markdown pair for one
 * reason: a generic exporter escapes prose. `a_b_c` comes back `a\_b\_c`,
 * `2*3=6` comes back `2\*3=6`, and the user never typed a backslash. Ordinary
 * text must survive BYTE-IDENTICAL, so the text of a node is written out
 * exactly as it reads.
 *
 * The node set is closed - paragraph, text, line break, list, list item, code -
 * which is what makes hand-writing both directions small enough to trust.
 * Anything outside it (bold, headings, links) is never produced by the editor
 * and is never read back into it: typed markdown outside the scope stays
 * literal text, in the composer and on the wire.
 *
 * Indentation of a nested list is the ENCLOSING ITEM'S CONTENT COLUMN - the
 * width of its marker, checkbox excluded ("- " is 2, "10. " is 4). That is the
 * canonical form, and it is what the answer-side renderer (remark-gfm) parses
 * back into the same tree; a fixed width would break the moment an ordinal
 * reached three digits.
 *
 * Two accepted consequences of canonicalizing, both of which the reader sees
 * exactly as the composer showed it:
 *
 * - A wrapped continuation line that itself opens with a marker comes back as a
 *   NESTED list rather than as the wrapped text it was written as. The wire is
 *   what a plain text box would have carried, and the answer-side renderer
 *   reads it the same way the composer does, so the two ends agree.
 * - A top-level list is re-emitted at column 0 whatever column it was written
 *   at, and the lines that follow it lose the same indentation, so that reading
 *   the composer's own output back yields the same tree it was written from.
 */

import { $createCodeNode, $isCodeNode } from "@lexical/code-core";
import {
  $createListItemNode,
  $createListNode,
  $isListItemNode,
  $isListNode,
  type ListItemNode,
  type ListNode,
  type ListType,
} from "@lexical/list";
import {
  $createLineBreakNode,
  $createParagraphNode,
  $createTextNode,
  $getRoot,
  $isElementNode,
  $isLineBreakNode,
  $isTextNode,
  type ElementNode,
  type LexicalNode,
} from "lexical";

/** What a bullet item opens with. One glyph, so `*` and `+` normalize to it. */
const BULLET_MARKER = "- ";
/** What a task item opens with; the box carries the state. */
const TASK_MARKER = (checked: boolean): string => `- [${checked ? "x" : " "}] `;

// The constructs the composer understands. Each demands whitespace after its
// marker, which is what keeps "-item" and "1.5" ordinary prose - the same line
// the plain-textarea composer drew before the editor replaced it.
const TASK_LINE = /^([ \t]*)[-*+][ \t]+\[([ xX])\][ \t]+(.*)$/;
const BULLET_LINE = /^([ \t]*)[-*+][ \t]+(.*)$/;
const ORDERED_LINE = /^([ \t]*)(\d{1,9})\.[ \t]+(.*)$/;
const FENCE_OPEN = /^[ \t]*(`{3,})[ \t]*([^\s`]*)[ \t]*$/;
const FENCE_CLOSE = /^[ \t]*(`{3,})[ \t]*$/;

// A clipboard filled on Windows carries CRLF, and every pattern above is
// anchored to the end of the line: a stray CR would leave "- item\r" matching
// nothing and put a control character on the wire.
const CARRIAGE_RETURN = /\r\n?/g;

/** A tab counts as four columns, the width the editor indents lists by. */
const TAB_COLUMNS = 4;

/** How many columns of indentation `whitespace` occupies. */
function columnsOf(whitespace: string): number {
  let columns = 0;
  for (const character of whitespace) {
    columns += character === "\t" ? TAB_COLUMNS : 1;
  }
  return columns;
}

/** How many columns of indentation `line` opens with. */
function indentColumnsOf(line: string): number {
  return columnsOf(line.slice(0, line.length - line.trimStart().length));
}

/** Drop up to `columns` columns of leading whitespace from `line`. */
function dedent(line: string, columns: number): string {
  let dropped = 0;
  let index = 0;
  while (index < line.length && dropped < columns) {
    const character = line[index];
    if (character !== " " && character !== "\t") break;
    dropped += character === "\t" ? TAB_COLUMNS : 1;
    index += 1;
  }
  return line.slice(index);
}

/**
 * Rewrite every flavour of line ending as a plain newline.
 *
 * A clipboard filled by a Windows application carries CRLF, and text dragged
 * out of one carries it too. Every pattern in this file is anchored to the end
 * of a line, so a surviving CR would leave a pasted list matching nothing and
 * put a control character on the wire.
 *
 * @param text - Text arriving from outside the editor.
 * @returns The same text with `\n` line endings.
 */
export function normalizeNewlines(text: string): string {
  return text.replace(CARRIAGE_RETURN, "\n");
}

/** The opening line of a fenced block, split into its two halves. */
export interface FenceOpener {
  /** The backtick run, which the closing fence has to match or beat. */
  ticks: string;
  /** The info string, empty when the fence names no language. */
  language: string;
}

/**
 * Read the fence a line opens, or null when it opens none.
 *
 * Shared with the editor so a fence typed into the box and a fence pasted into
 * it are judged by one pattern rather than two that can drift apart.
 *
 * @param line - One line of a draft.
 * @returns The fence's run and language, or null.
 */
export function readFenceOpener(line: string): FenceOpener | null {
  const match = FENCE_OPEN.exec(line);
  return match === null ? null : { language: match[2], ticks: match[1] };
}

// ---------------------------------------------------------------------------
// Nodes -> markdown
// ---------------------------------------------------------------------------

/**
 * The draft as the user would have typed it, read from the editor's root.
 *
 * Must run inside an editor read or update.
 *
 * @returns The whole draft as markdown.
 */
export function $serializeComposerMarkdown(): string {
  // Blocks join on ONE newline, not the blank line a markdown writer would use:
  // a block here is a typed LINE, so this is what makes a plain paragraph of
  // prose come back exactly as it went in.
  return $getRoot().getChildren().map($serializeBlock).join("\n");
}

/** One top-level block of the draft. */
function $serializeBlock(node: LexicalNode): string {
  if ($isListNode(node)) return $serializeList(node, "");
  if ($isCodeNode(node)) return $serializeCode(node.getTextContent(), node.getLanguage(), "");
  if ($isElementNode(node)) return $serializeInline(node, "");
  return node.getTextContent();
}

/**
 * The text of one element, line breaks included.
 *
 * `continuation` is written after every break so a wrapped line stays inside
 * the list item it belongs to rather than closing the list. A child that is
 * itself a block - a nested list, or a fenced block inside a list item - is
 * handed to the serializer that knows its markers; falling through to bare
 * text would drop the markers and the fence with them.
 */
function $serializeInline(element: ElementNode, continuation: string): string {
  let out = "";
  for (const child of element.getChildren()) {
    if ($isLineBreakNode(child)) {
      out += `\n${continuation}`;
    } else if ($isTextNode(child)) {
      // Verbatim - see the header. No escaping, ever.
      out += child.getTextContent();
    } else if ($isListNode(child)) {
      out += `\n${$serializeList(child, continuation)}`;
    } else if ($isCodeNode(child)) {
      out += `\n${$serializeCode(child.getTextContent(), child.getLanguage(), continuation)}`;
    } else if ($isElementNode(child)) {
      out += $serializeInline(child, continuation);
    } else {
      out += child.getTextContent();
    }
  }
  return out;
}

/** A fenced block, with a longer fence when the code itself holds backticks. */
function $serializeCode(
  text: string,
  language: string | null | undefined,
  indent: string,
): string {
  const runs = text.match(/`{3,}/g);
  const width = runs === null ? 3 : Math.max(3, ...runs.map((run) => run.length + 1));
  const fence = "`".repeat(width);
  const lines = [`${fence}${language ?? ""}`];
  if (text !== "") lines.push(...text.split("\n"));
  lines.push(fence);
  // A blank line stays blank: trailing whitespace on an empty line is invisible
  // in the box and would make the round trip depend on it.
  return lines.map((line) => (line === "" ? "" : indent + line)).join("\n");
}

/**
 * The nested lists this item is only a wrapper around, or null.
 *
 * Normally exactly one - that is the shape the editor's own nesting builds and
 * the shape an item is written back from. Several are tolerated because the
 * library reads nesting off the FIRST child alone: an item holding two lists
 * still draws as a nesting level, so refusing to serialize it would write a
 * marker for a line the box never showed. Each list is written at the same
 * depth, which is where the box drew them.
 */
function $nestedListsOf(item: ListItemNode): ListNode[] | null {
  const children = item.getChildren();
  if (children.length === 0 || !children.every($isListNode)) return null;
  return children;
}

/** One list and everything nested under it, indented by `indent`. */
function $serializeList(list: ListNode, indent: string): string {
  const type = list.getListType();
  const start = list.getStart();
  const lines: string[] = [];
  let index = 0;
  // The column an item's own text starts at, which is where a nested list and a
  // wrapped line both belong. Seeded for the case of a list that opens on a
  // nested one, which has no item above it to measure.
  let contentColumn = type === "number" ? `${start}. `.length : BULLET_MARKER.length;

  for (const item of list.getChildren()) {
    if (!$isListItemNode(item)) continue;
    const nested = $nestedListsOf(item);
    if (nested !== null) {
      const deeper = indent + " ".repeat(contentColumn);
      for (const child of nested) lines.push($serializeList(child, deeper));
      continue;
    }
    const value = start + index;
    let marker: string;
    if (type === "number") {
      marker = `${value}. `;
      contentColumn = marker.length;
    } else if (type === "check") {
      marker = TASK_MARKER(item.getChecked() === true);
      // The checkbox is CONTENT, not marker: a nested list under "- [ ] x"
      // indents by 2, and indenting by 6 would make it an indented code block.
      contentColumn = BULLET_MARKER.length;
    } else {
      marker = BULLET_MARKER;
      contentColumn = BULLET_MARKER.length;
    }
    lines.push(indent + marker + $serializeInline(item, indent + " ".repeat(contentColumn)));
    index += 1;
  }
  return lines.join("\n");
}

// ---------------------------------------------------------------------------
// Markdown -> nodes
// ---------------------------------------------------------------------------

/**
 * A list the parser is currently inside.
 *
 * Two column pairs, because the two disagree and both matter: the SOURCE
 * columns are what the incoming text was written at and are what every line
 * below is measured against, while the CANONICAL columns are where the
 * serializer will put the list back - a top-level list at 0, a nested one at
 * its parent item's content column. Re-reading the composer's own output has to
 * land on the tree it came from, and that only holds if a line that follows the
 * list loses exactly the difference.
 */
interface OpenList {
  node: ListNode;
  type: ListType;
  /** Column its markers will be written back at. */
  startColumn: number;
  /** Column its items' text will be written back at. */
  contentColumn: number;
  /** Column its markers arrived at. */
  sourceColumn: number;
  /** Column its items' text arrived at; a deeper line belongs to the item. */
  sourceContentColumn: number;
  /** The item a wrapped line would continue, or null before the first one. */
  lastItem: ListItemNode | null;
  /** Items appended so far, which is the next one's offset from the start. */
  items: number;
}

/** One list item read off a line. */
interface ItemLine {
  type: ListType;
  checked: boolean;
  startColumn: number;
  markerWidth: number;
  value: number;
  text: string;
}

/** The list item this line opens, or null if it is not one. */
function readItemLine(line: string): ItemLine | null {
  const task = TASK_LINE.exec(line);
  if (task !== null) {
    const [, indent, box, text] = task;
    return {
      checked: box.toLowerCase() === "x",
      markerWidth: BULLET_MARKER.length,
      startColumn: columnsOf(indent),
      text,
      type: "check",
      value: 1,
    };
  }
  const ordered = ORDERED_LINE.exec(line);
  if (ordered !== null) {
    const [, indent, digits, text] = ordered;
    return {
      checked: false,
      markerWidth: `${digits}. `.length,
      startColumn: columnsOf(indent),
      text,
      type: "number",
      value: Number.parseInt(digits, 10),
    };
  }
  const bullet = BULLET_LINE.exec(line);
  if (bullet !== null) {
    const [, indent, text] = bullet;
    return {
      checked: false,
      markerWidth: BULLET_MARKER.length,
      startColumn: columnsOf(indent),
      text,
      type: "bullet",
      value: 1,
    };
  }
  return null;
}

/** The text of a code block, as nodes: Lexical forbids a newline in a text node. */
function $codeChildren(text: string): LexicalNode[] {
  const children: LexicalNode[] = [];
  const lines = text.split("\n");
  for (let index = 0; index < lines.length; index += 1) {
    if (index > 0) children.push($createLineBreakNode());
    if (lines[index] !== "") children.push($createTextNode(lines[index]));
  }
  return children;
}

/**
 * Read a draft string back into editor nodes.
 *
 * Only the composer's own constructs are recognized; everything else becomes
 * paragraph text exactly as written, so a pasted `**bold**` stays four
 * asterisks and a word rather than turning into formatting the composer does
 * not offer.
 *
 * Must run inside an editor update.
 *
 * @param markdown - The draft to read.
 * @returns Top-level nodes, never empty.
 */
export function $buildComposerNodes(markdown: string): LexicalNode[] {
  const lines = normalizeNewlines(markdown).split("\n");
  const blocks: LexicalNode[] = [];
  const stack: OpenList[] = [];
  const top = (): OpenList => stack[stack.length - 1];

  /** Start a list - nested under the open item when there is one. */
  const openList = (item: ItemLine, startColumn: number): void => {
    const node = $createListNode(item.type, item.type === "number" ? item.value : undefined);
    if (stack.length === 0) {
      blocks.push(node);
    } else {
      // The editor models nesting as a list inside an item of its own, with no
      // marker of its own; the serializer knows that shape too.
      const wrapper = $createListItemNode();
      top().node.append(wrapper);
      wrapper.append(node);
    }
    stack.push({
      contentColumn: startColumn + item.markerWidth,
      items: 0,
      lastItem: null,
      node,
      sourceColumn: item.startColumn,
      sourceContentColumn: item.startColumn + item.markerWidth,
      startColumn,
      type: item.type,
    });
  };

  /**
   * The width of the marker the SERIALIZER will write for the next item.
   *
   * Not the width that arrived: an ordered list is re-emitted as a run from its
   * start, so "100." written second in a list beginning at 1 comes back as
   * "2." - two columns narrower. Measuring the source width here is what made a
   * second read of the composer's own output pull the following lines a column
   * too far and, at the extreme, fold a separate list into the one above it.
   */
  const canonicalMarkerWidth = (open: OpenList, item: ItemLine): number =>
    open.type === "number" ? `${open.node.getStart() + open.items}. `.length : item.markerWidth;

  for (let index = 0; index < lines.length; index += 1) {
    const line = lines[index];
    const column = indentColumnsOf(line);

    const fence = readFenceOpener(line);
    if (fence !== null) {
      const body: string[] = [];
      let scan = index + 1;
      // An unclosed fence runs to the end, which is what a markdown parser does
      // with one - the closing fence is written back on the way out.
      for (; scan < lines.length; scan += 1) {
        const close = FENCE_CLOSE.exec(lines[scan]);
        if (close !== null && close[1].length >= fence.ticks.length) break;
        body.push(lines[scan]);
      }
      index = scan;
      // A fence indented to the open item's own text is that item's content,
      // not a block that ends the list: the item keeps it, its indentation is
      // stripped here, and the serializer writes the indentation back.
      const open = stack.length > 0 ? top() : null;
      const host =
        open !== null && open.lastItem !== null && column >= open.sourceContentColumn
          ? open.lastItem
          : null;
      const code = $createCodeNode(fence.language === "" ? undefined : fence.language);
      const strip = host === null ? 0 : (open as OpenList).sourceContentColumn;
      const text = body.map((bodyLine) => dedent(bodyLine, strip)).join("\n");
      if (text !== "") code.append(...$codeChildren(text));
      if (host !== null) {
        host.append(code);
      } else {
        blocks.push(code);
        stack.length = 0;
      }
      continue;
    }

    const item = readItemLine(line);
    if (item !== null) {
      // Leave every list whose markers sit deeper than this line's.
      while (stack.length > 0 && item.startColumn < top().sourceColumn) stack.pop();
      if (stack.length === 0) {
        // A top-level list is written back at column 0 whatever column it
        // arrived at; every line under it is measured against that.
        openList(item, 0);
      } else if (item.startColumn >= top().sourceContentColumn) {
        // Nested, and the serializer will put it at the parent item's content
        // column rather than at the one it was written at.
        openList(item, top().contentColumn);
      } else if (top().type !== item.type) {
        // A different kind of list at the same level is a different list, and
        // it belongs exactly where the one it replaces lived.
        stack.pop();
        openList(item, stack.length === 0 ? 0 : top().contentColumn);
      }
      const listItem = $createListItemNode(item.type === "check" ? item.checked : undefined);
      if (item.text !== "") listItem.append($createTextNode(item.text));
      const open = top();
      open.node.append(listItem);
      open.lastItem = listItem;
      // Canonical columns are measured against the marker that will be WRITTEN;
      // source columns stay as they arrived, because that is what the lines
      // below this one are matched against.
      open.contentColumn = open.startColumn + canonicalMarkerWidth(open, item);
      open.sourceContentColumn = item.startColumn + item.markerWidth;
      open.items += 1;
      continue;
    }

    // A line indented past the open item's text is that item wrapping, not a
    // new block. Its own indentation is NOT kept: it is rewritten at the item's
    // content column, which is where the serializer puts a continuation. The
    // accepted consequence is that a line indented four or more columns inside
    // an item stops reading as an indented code block.
    if (stack.length > 0 && line.trim() !== "") {
      const open = top();
      if (open.lastItem !== null && column >= open.sourceContentColumn) {
        open.lastItem.append($createLineBreakNode(), $createTextNode(line.trimStart()));
        continue;
      }
    }

    // The list ends here. It is written back at a shallower column than it
    // arrived at, so this line loses the same difference - otherwise re-reading
    // what the composer writes would pull the line back INTO the item it just
    // left.
    const shift = stack.length > 0 ? top().sourceContentColumn - top().contentColumn : 0;
    const text = dedent(line, shift);
    stack.length = 0;
    const paragraph = $createParagraphNode();
    if (text !== "") paragraph.append($createTextNode(text));
    blocks.push(paragraph);
  }

  return blocks.length === 0 ? [$createParagraphNode()] : blocks;
}

/**
 * Whether `text` holds anything the composer would render as a glyph.
 *
 * Text with none of it can be inserted as-is, which keeps a plain paste cheap
 * and leaves the caret where the editor's own insertion puts it.
 *
 * @param text - The text about to be inserted.
 * @returns True when at least one line opens a list, task or code block.
 */
export function hasComposerConstructs(text: string): boolean {
  return normalizeNewlines(text)
    .split("\n")
    .some((line) => readFenceOpener(line) !== null || readItemLine(line) !== null);
}
