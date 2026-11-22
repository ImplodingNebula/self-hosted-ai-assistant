/**
 * What the composer's editor does beyond typing, and the shape it is built in.
 *
 * Kept out of the React component so the behaviour is reachable without a
 * browser: the keystrokes this file claims - Enter sends, Shift+Enter breaks,
 * Tab leaves the box - are the ones a person's message depends on, and they are
 * checkable against a headless editor rather than by hand.
 *
 * The pieces belong together for one reason: Enter sends, so Shift+Enter has to
 * take over the newline the editor would otherwise put there, and text arriving
 * by paste or drop has to reach the page before either of them.
 */

import { $createCodeNode, $isCodeNode, CodeNode } from "@lexical/code-core";
import {
  $createListItemNode,
  $createListNode,
  $isListItemNode,
  $isListNode,
  ListItemNode,
  ListNode,
} from "@lexical/list";
import {
  CODE,
  ORDERED_LIST,
  registerMarkdownShortcuts,
  type Transformer,
  UNORDERED_LIST,
} from "@lexical/markdown";
import { DRAG_DROP_PASTE } from "@lexical/rich-text";
import { $findMatchingParent, calculateZoomLevel, mergeRegister } from "@lexical/utils";
import {
  $createParagraphNode,
  $createRangeSelection,
  $getNearestNodeFromDOMNode,
  $getNodeByKey,
  $getRoot,
  $getSelection,
  $isElementNode,
  $isLineBreakNode,
  $isParagraphNode,
  $isRangeSelection,
  $isTextNode,
  $normalizeSelection__EXPERIMENTAL,
  $setSelection,
  COMMAND_PRIORITY_CRITICAL,
  DROP_COMMAND,
  type EditorThemeClasses,
  type ElementNode,
  FORMAT_TEXT_COMMAND,
  HISTORIC_TAG,
  INSERT_PARAGRAPH_COMMAND,
  isHTMLElement,
  KEY_ARROW_DOWN_COMMAND,
  KEY_ENTER_COMMAND,
  KEY_TAB_COMMAND,
  type Klass,
  type LexicalEditor,
  type LexicalNode,
  PASTE_COMMAND,
  type PointType,
  TextNode,
} from "lexical";

import {
  $buildComposerNodes,
  $serializeComposerMarkdown,
  hasComposerConstructs,
  normalizeNewlines,
  readFenceOpener,
} from "@/lib/composer-markdown";

/**
 * The ONLY markdown the composer speaks.
 *
 * No bold, italic, heading, quote or link transformer may join this list.
 * Out-of-scope markdown stays literal text in both directions, so what is typed
 * is what is sent - and a draft that LOOKED formatted while the wire carried
 * plain text would be a lie the user only finds out about after sending.
 *
 * CHECK_LIST is deliberately absent. It converts a bare "[ ] " with no dash,
 * which GFM does not recognize as a task item at all - so typing it would have
 * drawn a checkbox that the reader, and the composer's own reader of a pasted
 * draft, both see as ordinary prose. The task items the composer does offer
 * come from the "- [ ] " transform below, which is the only GFM-valid form.
 */
export const COMPOSER_TRANSFORMERS: Transformer[] = [UNORDERED_LIST, ORDERED_LIST, CODE];

// The checkbox a task item opens with, once the line is already a list item.
const TYPED_CHECKBOX = /^\[([ xX])\][ \t]/;

/** The closer a person types to end a block. Exactly three, nothing beside it. */
const CLOSING_FENCE = "```";

/** Backticks and nothing else, which is all a closer line can be made of. */
const BACKTICK_RUN = /^`+$/;

/**
 * Everything the composer renders beyond paragraphs and text.
 *
 * `CodeNode` comes from `@lexical/code-core` rather than `@lexical/code`: the
 * full package pulls in the Prism highlighter chain (~3.5 MB) to colour code
 * the composer shows as plain monospace anyway. The cost of the lean import is
 * a development-only "Using CodeNode without CodeExtension is deprecated"
 * warning, which is known and accepted.
 */
export const COMPOSER_NODES: Klass<LexicalNode>[] = [ListNode, ListItemNode, CodeNode];

/** Class names the editor stamps on what it renders; chat.css styles them. */
export const COMPOSER_THEME: EditorThemeClasses = {
  code: "composer-code",
  list: {
    checklist: "composer-checklist",
    listitem: "composer-listitem",
    listitemChecked: "composer-listitem-checked",
    listitemUnchecked: "composer-listitem-unchecked",
    nested: { listitem: "composer-nested-listitem" },
    ol: "composer-ol",
    ul: "composer-ul",
  },
  paragraph: "composer-paragraph",
};

/** What the composer hands back to whoever mounted it. */
export interface ComposerBehaviourHandlers {
  /** The draft after every edit, as the markdown a person would have typed. */
  onChangeMarkdown: (markdown: string) => void;
  /** Enter, with no modifier and no IME candidate open. */
  onSend: () => void;
  /** Offered every paste and drop before the editor takes it; true means consumed. */
  onPasteIntercept?: (clipboard: DataTransfer) => boolean;
}

/** The two DOM caret-from-point APIs, which no engine implements both of. */
interface CaretDocument {
  caretPositionFromPoint?: (x: number, y: number) => { offsetNode: Node; offset: number } | null;
  caretRangeFromPoint?: (x: number, y: number) => Range | null;
}

/** The DOM position under `x`/`y`, or null when the engine cannot say. */
function domCaretFromPoint(
  owner: CaretDocument,
  x: number,
  y: number,
): { node: Node; offset: number } | null {
  const range = owner.caretRangeFromPoint?.(x, y);
  if (range !== null && range !== undefined) {
    return { node: range.startContainer, offset: range.startOffset };
  }
  const position = owner.caretPositionFromPoint?.(x, y);
  if (position !== null && position !== undefined) {
    return { node: position.offsetNode, offset: position.offset };
  }
  return null;
}

/**
 * Put the caret where a drop landed.
 *
 * The editor suppresses selection changes for the duration of a drag, so the
 * selection at drop time is still wherever the caret was parked before it -
 * inserting there would drop the text somewhere the user never pointed. Falls
 * back to the end of the draft, which at least never loses what was dropped.
 */
function $selectDropPoint(x: number, y: number): void {
  const owner =
    typeof document === "undefined" ? null : (document as unknown as CaretDocument);
  const hit = owner === null ? null : domCaretFromPoint(owner, x, y);
  const node = hit === null ? null : $getNearestNodeFromDOMNode(hit.node);
  if (hit === null || node === null) {
    $getRoot().selectEnd();
    return;
  }
  const selection = $createRangeSelection();
  if ($isTextNode(node)) {
    selection.anchor.set(node.getKey(), hit.offset, "text");
    selection.focus.set(node.getKey(), hit.offset, "text");
  } else {
    const parent = node.getParent();
    if (parent === null) {
      $getRoot().selectEnd();
      return;
    }
    const offset = node.getIndexWithinParent() + 1;
    selection.anchor.set(parent.getKey(), offset, "element");
    selection.focus.set(parent.getKey(), offset, "element");
  }
  $setSelection($normalizeSelection__EXPERIMENTAL(selection));
}

/** How many positions `point` is measured against - characters, or children. */
function $pointSize(point: PointType): number {
  const node = point.getNode();
  return $isElementNode(node) ? node.getChildrenSize() : node.getTextContentSize();
}

// Nodes are compared by KEY throughout this file, never by identity: an edit
// clones the node it touches, so the object a variable is holding stops being
// the object the tree hands back the moment anything is written.

/** Whether `point` sits before everything in `block`. */
function $atBlockStart(point: PointType, block: ElementNode): boolean {
  if (point.offset !== 0) return false;
  let node: LexicalNode = point.getNode();
  while (node.getKey() !== block.getKey()) {
    if (node.getPreviousSibling() !== null) return false;
    const parent: ElementNode | null = node.getParent();
    if (parent === null) return false;
    node = parent;
  }
  return true;
}

/** Whether `point` sits after everything in `block`. */
function $atBlockEnd(point: PointType, block: ElementNode): boolean {
  if (point.offset !== $pointSize(point)) return false;
  let node: LexicalNode = point.getNode();
  while (node.getKey() !== block.getKey()) {
    if (node.getNextSibling() !== null) return false;
    const parent: ElementNode | null = node.getParent();
    if (parent === null) return false;
    node = parent;
  }
  return true;
}

/** Leave the caret at the end of what was just put in. */
function $selectAfter(nodes: LexicalNode[]): void {
  const last = nodes[nodes.length - 1];
  if ($isElementNode(last)) last.selectEnd();
}

/** Chain `nodes` in as siblings after `block`, in order. */
function $insertBlocksAfter(block: LexicalNode, nodes: LexicalNode[]): void {
  let after: LexicalNode = block;
  for (const node of nodes) {
    after.insertAfter(node);
    after = node;
  }
  $selectAfter(nodes);
}

/** Chain `nodes` in as siblings before `block`, in order. */
function $insertBlocksBefore(block: LexicalNode, nodes: LexicalNode[]): void {
  let after: LexicalNode = nodes[0];
  block.insertBefore(after);
  for (const node of nodes.slice(1)) {
    after.insertAfter(node);
    after = node;
  }
  $selectAfter(nodes);
}

/**
 * Put a code-leading payload after the whole list the caret is in.
 *
 * Splitting inside a list item makes a new ITEM, not a new block, so a fence
 * dropped there is swallowed by the list wherever the caret sits - the block
 * has to leave the list entirely. Emptiness is judged on the ITEM the caret is
 * in, never on the list around it: a list holding one empty item is a marker
 * the user has just opened, and reading the LIST as empty is what let the fence
 * merge into it.
 */
function $insertCodeAfterList(list: ElementNode, item: ListItemNode, nodes: LexicalNode[]): void {
  $insertBlocksAfter(list, nodes);
  // A marker offered and never filled in is not content - the same rule the
  // send path applies to a trailing one - so the block takes its place rather
  // than hanging under an empty bullet. Only the last item, and only a direct
  // child: removing one mid-list would delete a line the user made.
  if (item.getTextContentSize() !== 0 || item.getNextSibling() !== null) return;
  const parent = item.getParent();
  if (parent === null || parent.getKey() !== list.getKey()) return;
  item.remove();
  if (parent.getChildrenSize() === 0) parent.remove();
}

/**
 * Put `nodes` in at the caret without letting a block dissolve into the line.
 *
 * A fenced block inserted into a line is MERGED into it by the editor's own
 * insertion: the fence and its language vanish, and the prose on the far side
 * of the caret is swallowed INTO the code. So a code-leading payload is placed
 * as a sibling block instead - before the line, after it, or between the two
 * halves of a line split at the caret.
 *
 * Splitting a line mid-word is the same normalization the composer already
 * applies to a wrapped continuation, carried up to blocks: the box has to show
 * exactly what the recipient's renderer will show, and no renderer opens a
 * fence halfway through a line of prose.
 *
 * A LIST-leading payload keeps merging into the line it lands in, which is what
 * continuing a list by pasting into it should do, and is what a plain text box
 * would have produced. Only the placement rules are shared: a payload landing
 * at the very start of a block goes in without splitting first, because
 * splitting there leaves the empty half behind as a blank first line.
 */
function $insertComposerNodes(nodes: LexicalNode[]): void {
  const ranged = $getSelection();
  if (!$isRangeSelection(ranged)) return;
  const first = nodes[0];
  if (!$isListNode(first) && !$isCodeNode(first)) {
    ranged.insertNodes(nodes);
    return;
  }
  // Take a selection's text out first, so every rule below reads ONE caret.
  if (!ranged.isCollapsed()) ranged.removeText();
  const selection = $getSelection();
  if (!$isRangeSelection(selection)) return;
  const anchor = selection.anchor;
  const block = anchor.getNode().getTopLevelElement();
  if (!$isElementNode(block)) {
    selection.insertNodes(nodes);
    return;
  }
  const atStart = $atBlockStart(anchor, block);

  if ($isCodeNode(first)) {
    const item = $findMatchingParent(anchor.getNode(), $isListItemNode);
    if ($isListItemNode(item)) {
      $insertCodeAfterList(block, item, nodes);
      return;
    }
    const atEnd = $atBlockEnd(anchor, block);
    // An empty line has no halves to split and nothing to push aside.
    if (atStart && atEnd) selection.insertNodes(nodes);
    else if (atStart) $insertBlocksBefore(block, nodes);
    else if (atEnd) $insertBlocksAfter(block, nodes);
    else {
      selection.insertParagraph();
      const tail = selection.anchor.getNode().getTopLevelElement();
      if ($isElementNode(tail)) $insertBlocksBefore(tail, nodes);
      else selection.insertNodes(nodes);
    }
    return;
  }

  // A caret with nothing before it in its block has no first half to keep, and
  // splitting there leaves the empty one behind - as a blank first line in
  // prose, as a naked marker in a list. Neither is anything the user asked for,
  // so the payload goes straight in and merges as it would anywhere else.
  if (!atStart) selection.insertParagraph();
  selection.insertNodes(nodes);
}

/**
 * Insert `text` at the caret, reading the composer's own constructs out of it.
 *
 * Inside a code block everything is content, and in prose with no markers there
 * is nothing to read: both go in as typed, which is what makes a plain paste
 * merge into the line the way a text box would.
 */
function $insertComposerText(text: string): void {
  const selection = $getSelection();
  if (!$isRangeSelection(selection)) return;
  const inCode = $findMatchingParent(selection.anchor.getNode(), $isCodeNode) !== null;
  if (inCode || !hasComposerConstructs(text)) {
    selection.insertRawText(text);
    return;
  }
  $insertComposerNodes($buildComposerNodes(text));
}

/**
 * Turn a fence typed on its own line into a code block.
 *
 * ONE undo hands the fence back as the text it was typed as, and nothing here
 * asks for that: replacing a paragraph with a code block is a STRUCTURAL edit,
 * which the history plugin classifies as unlike the character-by-character
 * typing before it and therefore never merges into it. The pairing is pinned by
 * the "a fence opened on shift+enter gives back the text" check, which fails
 * the moment this update becomes part of the keystrokes that led to it.
 *
 * @returns True when the paragraph was a fence opener and became code.
 */
function $openTypedFence(): boolean {
  const selection = $getSelection();
  if (!$isRangeSelection(selection) || !selection.isCollapsed()) return false;
  const block = selection.anchor.getNode().getTopLevelElement();
  if (!$isParagraphNode(block)) return false;
  const fence = readFenceOpener(block.getTextContent());
  if (fence === null) return false;
  const code = $createCodeNode(fence.language === "" ? undefined : fence.language);
  block.replace(code);
  code.selectEnd();
  return true;
}

/**
 * Take the newline Shift+Enter owes, inside a code block, without the escape.
 *
 * The base editor's paragraph insertion carries a rule of its own for code: two
 * TRAILING linebreaks leave the block, and it deletes them on the way out. That
 * is a double-Shift+Enter escape, and it fires on the most ordinary thing there
 * is to type inside a code block - the blank lines between two functions, which
 * is where PEP 8 asks for two. Measured before this landed: "def a():", a body
 * line, three Shift+Enters and "def b():" left the block after `pass`, both
 * blank lines gone and the second definition sitting outside as prose.
 *
 * So the newline is inserted here instead of delegated: inside a block the key
 * means a line, every time, chained without limit. Leaving the block is what
 * the two gestures below {@link $isStrandedBlock} are for - a typed closing
 * fence and ArrowDown - neither of which touches this key.
 *
 * Nothing else is lost with the delegation: the base path's other job there is
 * re-creating a line's leading indentation, which it reads off highlighted code
 * nodes that this composer never builds.
 *
 * @returns True when the linebreak was inserted here.
 */
function $breakInsideCode(): boolean {
  const selection = $getSelection();
  if (!$isRangeSelection(selection)) return false;
  if ($findMatchingParent(selection.anchor.getNode(), $isCodeNode) === null) return false;
  selection.insertLineBreak();
  return true;
}

/**
 * The item a nested list hangs under, or null when the list is top-level.
 *
 * The editor models nesting as a list held inside a list item that carries no
 * text of its own - so a nested list's PARENT is an item, and everything that
 * treats a list as a block has to reach one level further up.
 */
function $wrapperOf(list: ListNode): ListItemNode | null {
  const parent = list.getParent();
  return $isListItemNode(parent) ? parent : null;
}

/**
 * Put `made` immediately after `list`, at the same depth.
 *
 * A wrapper item holds exactly ONE list - that is the shape the serializer
 * reads a nesting level off, and the shape the editor's own nesting produces.
 * Dropping a second list into the same wrapper leaves a node that still LOOKS
 * nested (its first child is a list, which is all the library asks) while the
 * serializer no longer recognizes it, and writes a bare marker for the item on
 * a line the box never showed. So a sibling list gets a wrapper of its own.
 */
function $insertListAfter(list: ListNode, made: ListNode): void {
  const wrapper = $wrapperOf(list);
  if (wrapper === null) {
    list.insertAfter(made);
    return;
  }
  const sibling = $createListItemNode();
  wrapper.insertAfter(sibling);
  sibling.append(made);
}

/** Drop `list`, and the wrapper item it was the whole content of. */
function $removeList(list: ListNode): void {
  const wrapper = $wrapperOf(list);
  list.remove();
  if (wrapper !== null && wrapper.getChildrenSize() === 0) wrapper.remove();
}

/**
 * Turn "[ ] " typed at the head of a BULLET item into a task item.
 *
 * "- [ ] " cannot arrive through the markdown shortcuts at all: the "- " has
 * already turned the line into a bullet item by the time the box is typed, and
 * a block-level transformer only ever fires on a top-level paragraph. So the
 * box is read off the item's own text instead, and ONLY that item becomes a
 * task - the list around it is left as it was.
 *
 * Bullet lists only. In an ORDERED item the box stays literal text, which is
 * what the same line PASTED into the box already does: "1. [ ] milk" reads as
 * an ordered item whose text opens with a bracket, and GFM's task syntax is
 * defined on the item's content either way. Converting it instead would swap
 * the ordinal for a dash - silently destroying a valid construct, and making
 * the typed and pasted forms of one line disagree.
 *
 * @param label - The item's first text node, as the keystroke left it.
 * @returns True when the item became a task.
 */
function $convertTypedCheckbox(label: TextNode): boolean {
  // The box has to OPEN the item. A wrapped line that happens to start with
  // "[ ] " is prose the user wrote, not a checkbox they asked for.
  if (label.getPreviousSibling() !== null) return false;
  const typed = TYPED_CHECKBOX.exec(label.getTextContent());
  if (typed === null) return false;
  // The box must have JUST been completed - the caret sits immediately behind
  // it, the same discipline the library's own markdown shortcuts use. Matching
  // the text alone fires on EVERY later keystroke in the item, so a conversion
  // the user undid would come straight back on the next character they typed,
  // and a literal "[ ] " would be untypeable.
  const selection = $getSelection();
  if (!$isRangeSelection(selection) || !selection.isCollapsed()) return false;
  const anchor = selection.anchor;
  if (anchor.type !== "text" || anchor.key !== label.getKey()) return false;
  if (anchor.offset !== typed[0].length) return false;
  const item = label.getParent();
  if (!$isListItemNode(item)) return false;
  const list = item.getParent();
  if (!$isListNode(list) || list.getListType() !== "bullet") return false;

  const following = item.getNextSiblings();
  const tasks = $createListNode("check");
  // Not `list.insertAfter` - a NESTED list's sibling belongs in a wrapper item
  // of its own; see $insertListAfter for what a shared wrapper corrupts.
  $insertListAfter(list, tasks);
  if (following.length > 0) {
    // The items below carry on as the bullets they already were.
    const rest = $createListNode("bullet");
    $insertListAfter(tasks, rest);
    for (const sibling of following) rest.append(sibling);
  }
  tasks.append(item);
  if (list.getChildrenSize() === 0) $removeList(list);

  const text = label.getTextContent().slice(typed[0].length);
  if (text === "") label.remove();
  else label.setTextContent(text);
  item.setChecked(typed[1].toLowerCase() === "x");
  item.selectStart();
  return true;
}

/** The flavour the base editor stamps on a drag it started, holding its key. */
const LEXICAL_DRAG_TYPE = "application/x-lexical-drag";

/**
 * Whether this drop ends a drag that started in THIS editor.
 *
 * The base editor writes the flavour on drag start and reads it on drop to tell
 * a move from a paste; comparing the key it carries is what keeps the answer
 * "this editor" rather than "some editor on the page".
 */
function isSameEditorDrag(transfer: DataTransfer, editor: LexicalEditor): boolean {
  if (!transfer.types.includes(LEXICAL_DRAG_TYPE)) return false;
  let marker: unknown;
  try {
    marker = JSON.parse(transfer.getData(LEXICAL_DRAG_TYPE));
  } catch {
    return false;
  }
  if (typeof marker !== "object" || marker === null) return false;
  return (marker as { editorKey?: unknown }).editorKey === editor.getKey();
}

/**
 * Whether a pointer at `x`/`y` is in the gutter that draws `item`'s checkbox.
 *
 * The gutter is the item's own left padding, the whole of it, over the item's
 * full height. The drawn box is 0.9em - about 14x15 CSS pixels - which is well
 * under any usable target; widening the GLYPH to match would make the composer
 * disagree with the message it becomes, so the TARGET is widened instead and
 * the glyph is left exactly where it is.
 *
 * Height, not first line: a wrapped item's remaining lines hang off the same
 * indent, and the base plugin's own band already runs the item's whole height,
 * so clamping this one to the first line would make the left half of that
 * column toggle and the right half not.
 */
function inCheckGutter(item: HTMLElement, x: number, y: number): boolean {
  const view = item.ownerDocument.defaultView;
  if (view === null) return false;
  const gutter = Number.parseFloat(view.getComputedStyle(item).paddingLeft);
  if (!Number.isFinite(gutter) || gutter <= 0) return false;
  const rect = item.getBoundingClientRect();
  return x >= rect.left && x <= rect.left + gutter && y >= rect.top && y <= rect.bottom;
}

/**
 * The task item whose checkbox gutter `event` landed in, if any.
 *
 * Answered from the NODE, not from the theme classes on the element: what the
 * click toggles is a list item in a check list, and reading that off the tree
 * is what keeps a bullet's gutter - which is the list's padding, not the
 * item's - from ever being mistaken for a box.
 */
function checkGutterTarget(editor: LexicalEditor, event: MouseEvent): HTMLElement | null {
  const target = event.target;
  if (!isHTMLElement(target)) return null;
  const item = target.closest("li");
  if (item === null) return null;
  // An item that opens with a nested list is a container for the items below
  // it, and draws no box of its own.
  const first = item.firstElementChild;
  if (first !== null && (first.tagName === "UL" || first.tagName === "OL")) return null;
  const zoom = calculateZoomLevel(item);
  if (!inCheckGutter(item, event.clientX / zoom, event.clientY / zoom)) return null;
  const isTask = editor.read(() => {
    const node = $getNearestNodeFromDOMNode(item);
    if (!$isListItemNode(node)) return false;
    const list = node.getParent();
    return $isListNode(list) && list.getListType() === "check";
  });
  return isTask ? item : null;
}

/** Whether this pointer came from something with a visible cursor. */
function isPrecisePointer(event: MouseEvent): boolean {
  return !(event instanceof PointerEvent) || event.pointerType !== "touch";
}

/** Whether `node` is a root-level list holding nothing but one empty item. */
function $isStrandedList(node: LexicalNode): boolean {
  if (!$isListNode(node) || node.getChildrenSize() !== 1) return false;
  const only = node.getFirstChild();
  return $isListItemNode(only) && only.getChildrenSize() === 0;
}

/**
 * Whether `node` is a code block with nothing left inside it.
 *
 * Text content, not child count: a block holding one line break is a blank
 * LINE the user typed and still has a caret in, which is content.
 */
function $isStrandedCode(node: LexicalNode): boolean {
  return $isCodeNode(node) && node.getTextContentSize() === 0;
}

/**
 * Whether the selection of the state being read sat wholly inside `key`.
 *
 * Read against the PREVIOUS state, where it answers "was the edit that just
 * emptied this block made from inside it" - a person clearing a block's lines
 * to retype them, rather than a gesture that swept the block away with the rest
 * of the draft.
 *
 * Must run inside a read of the state the selection belongs to.
 */
function $selectionWasInside(key: string): boolean {
  const selection = $getSelection();
  if (!$isRangeSelection(selection)) return false;
  return [selection.anchor, selection.focus].every(
    (point) => point.getNode().getTopLevelElement()?.getKey() === key,
  );
}

/**
 * Whether a block emptied down to a shell should be kept rather than collapsed.
 *
 * A code block cleared from the INSIDE is being retyped: collapsing it there
 * takes the block and its language away mid-edit, and retyping in place - the
 * ordinary way to change what a block holds - becomes impossible.
 *
 * The two are only distinguishable while the draft holds something else.
 * Measured: over a draft that is nothing BUT the block, a manual selection of
 * its text and Ctrl+A leave the identical selection (both points anchored in
 * the block's own text), because there is nothing outside the block for a
 * whole-draft selection to reach. Accepted edge, deliberately resolved toward
 * collapsing: a sole block cleared any way at all gives the empty paragraph
 * back, which is what keeps the box typeable and the placeholder honest.
 */
function $keepsItsShell(block: LexicalNode, blocks: number): boolean {
  return $isCodeNode(block) && blocks > 1;
}

/**
 * Whether `node` is a block emptied down to a shell that traps the box.
 *
 * A naked list marker and an empty fence fail the same way, for the same
 * reason - the block outlives its content, so the placeholder never returns,
 * the markdown shortcuts stop firing (they only run on a top-level paragraph)
 * and everything typed next lands inside the shell. The fence is the worse of
 * the two: it also serializes to a bare fence PAIR, which is a non-empty draft,
 * so the send button stays lit over a message with nothing in it.
 */
function $isStrandedBlock(node: LexicalNode): boolean {
  return $isStrandedList(node) || $isStrandedCode(node);
}

/**
 * Put an empty paragraph below `code` and move the caret into it.
 *
 * The one way out of a code block, shared by both gestures that offer one, so
 * they cannot disagree about what leaving looks like.
 *
 * A block left EMPTY becomes the paragraph rather than gaining one beneath it.
 * Inserting below would leave a shell that the stranded-code collapse turns
 * into a SECOND empty paragraph one update later - two blank lines and a caret
 * in the lower one, from a keystroke that was supposed to produce one line.
 *
 * Accepted edge, measured: an empty block left this way is GONE, and its
 * opener's language with it - "```py" then ArrowDown gives back a plain
 * paragraph, whatever else the draft holds. Leaving a block with nothing in it
 * is a request for the line below it, not for two empty blocks, and one undo
 * hands the block and its language back. (The block a person is RETYPING is a
 * different case, and is spared - see {@link $keepsItsShell}.)
 */
function $exitCodeBlock(code: CodeNode): void {
  const paragraph = $createParagraphNode();
  if ($isStrandedCode(code)) code.replace(paragraph);
  else code.insertAfter(paragraph);
  paragraph.select();
}

/**
 * Whether `code` is a block of its own rather than one nested in a list item.
 *
 * A nested block has no room below it for the paragraph either gesture would
 * insert: the caret would land in a paragraph inside a list item, which is not
 * a shape the composer's serializer or the reader's parser speaks.
 */
function $isTopLevelCode(code: CodeNode): boolean {
  return code.getTopLevelElement()?.getKey() === code.getKey();
}

/**
 * Close a code block whose last line has just had a fence typed onto it.
 *
 * Markdown muscle memory: someone who puts the trailing fence in place means
 * the block to be finished. The closer line is consumed - its text and the
 * linebreak that opened it - and an empty paragraph below the block takes the
 * caret. It never touches the newline key, so blank lines inside a block still
 * chain on Shift+Enter without limit.
 *
 * The trigger discipline is the typed checkbox's, for the same reasons: the
 * caret has to sit immediately behind the third backtick, so a fence that is
 * merely PRESENT on the line never fires, a conversion the user undid does not
 * come straight back on the next keystroke, and typing elsewhere in the block
 * is left alone.
 *
 * Accepted edge, not fought: no fence line can be typed in place as a block's
 * last line, because the THIRD backtick closes the block - measured, typing
 * "```js" there leaves the block closed and "js" in the paragraph below, and a
 * fourth backtick or a trailing space lands in that paragraph too. Recovery is
 * undo, one step per thing typed: from "```js" it takes TWO (the "js" in the
 * paragraph, then the close), and from a bare "```" it takes one. Nothing else
 * reaches this: a pasted or dropped fence arrives whole rather than a character
 * at a time (see the caller's typed-growth gate), and the serializer already
 * widens a block's own fence around backtick content.
 *
 * @param node - The text node the keystroke landed in.
 * @returns True when the block was closed.
 */
function $closeTypedFence(node: TextNode): boolean {
  const code = node.getParent();
  if (!$isCodeNode(code) || !$isTopLevelCode(code)) return false;
  // Nothing after it inside the block, which answers BOTH questions at once:
  // the caret is at the end of its line, and the line is the block's last. A
  // "```" with code under it is content, and must stay content.
  if (node.getNextSibling() !== null) return false;
  const selection = $getSelection();
  if (!$isRangeSelection(selection) || !selection.isCollapsed()) return false;
  const anchor = selection.anchor;
  if (anchor.type !== "text" || anchor.key !== node.getKey()) return false;
  if (anchor.offset !== node.getTextContentSize()) return false;

  // The whole line, back to the break that opened it. Judged whole rather than
  // by its last piece, and what that rejects is a line that ARRIVED holding
  // more than a fence - a pasted or dropped snippet, a planted line, a restored
  // draft - where an indent, a language or a trailing character leaves a line
  // that is not a closer. It is not what stops a person typing past the fence:
  // typing "```js" closes on the third backtick and puts "js" in the paragraph
  // below, so the extra characters never reach this line at all. The other job
  // is plumbing - the editor may hold one line as several text nodes.
  const line: TextNode[] = [node];
  let text = node.getTextContent();
  let previous = node.getPreviousSibling();
  while (previous !== null && !$isLineBreakNode(previous)) {
    if (!$isTextNode(previous)) return false;
    line.push(previous);
    text = previous.getTextContent() + text;
    previous = previous.getPreviousSibling();
  }
  if (text !== CLOSING_FENCE) return false;

  for (const part of line) part.remove();
  // The linebreak that opened the closer goes with it, or the block keeps a
  // blank last line nobody typed.
  if (previous !== null) previous.remove();
  $exitCodeBlock(code);
  return true;
}

/**
 * Whether `point` sits on the LAST line of `code` - no break left below it.
 *
 * The column inside the line is deliberately irrelevant: "down" from anywhere
 * on the last line means the same thing, and a rule that also asked for the end
 * of the line would leave the key dead wherever the caret happened to be.
 */
function $onCodeLastLine(point: PointType, code: CodeNode): boolean {
  if (point.key === code.getKey()) {
    // An element point counts CHILDREN, so everything from its offset on is
    // what still sits below the caret.
    return !code.getChildren().slice(point.offset).some($isLineBreakNode);
  }
  // Climb to the child of the block the caret is somewhere inside, which is
  // the node the line's remaining pieces are siblings of.
  let node: LexicalNode = point.getNode();
  let parent = node.getParent();
  while (parent !== null && parent.getKey() !== code.getKey()) {
    node = parent;
    parent = node.getParent();
  }
  if (parent === null) return false;
  for (let after = node.getNextSibling(); after !== null; after = after.getNextSibling()) {
    if ($isLineBreakNode(after)) return false;
  }
  return true;
}

/**
 * Leave a code block that ends the draft, on the key that does nothing there.
 *
 * ArrowDown on the last line of a trailing code block has nowhere to move the
 * caret to - the keypress is dead - so it is spent going below the block, which
 * is what "down" reads as. Without it a block typed at the end of a draft is a
 * trap: Enter sends, Shift+Enter only adds lines inside, and there is no
 * keyboard way out at all.
 *
 * Every other case is left to native navigation - a block with content under
 * it, a caret on any earlier line, prose, a list, a selection that is not a
 * caret - so nothing that already moved the caret stops doing so. Pressing it
 * again from the new paragraph does nothing: the caret is no longer in a block.
 *
 * @returns True when the caret left the block.
 */
function $exitCodeOnArrowDown(): boolean {
  const selection = $getSelection();
  if (!$isRangeSelection(selection) || !selection.isCollapsed()) return false;
  const block = selection.anchor.getNode().getTopLevelElement();
  // The LAST block, so the key is only taken where it had nothing to do.
  if (!$isCodeNode(block) || block.getNextSibling() !== null) return false;
  if (!$onCodeLastLine(selection.anchor, block)) return false;
  $exitCodeBlock(block);
  return true;
}

/**
 * Wire the composer's own behaviour onto `editor`.
 *
 * @param editor - The editor to register against.
 * @param read - Returns the current handlers, so a re-render never costs a
 *   registration: a dropped Enter handler is a message the user watched vanish.
 * @returns A teardown that unregisters everything.
 */
export function registerComposerBehaviour(
  editor: LexicalEditor,
  read: () => ComposerBehaviourHandlers,
): () => void {
  let previous: string | null = null;

  return mergeRegister(
    registerMarkdownShortcuts(editor, COMPOSER_TRANSFORMERS),

    // The task conversion runs in an update of ITS OWN, one tick behind the
    // keystroke that triggered it, exactly as the library's own markdown
    // shortcuts do. That deferral is the whole undo story: the keystroke's own
    // update commits the literal "[ ] " first, and the structural conversion
    // that follows is classified as unlike the typing before it, so the history
    // plugin never merges the two. Converting inside the keystroke's update
    // would leave undo landing on a half-typed box, and a literal "- [ ] "
    // would be untypeable. The "undo: a task gives back the literal box" check
    // fails the moment the conversion stops being its own step.
    editor.registerUpdateListener(({ dirtyLeaves, editorState, tags }) => {
      if (tags.has(HISTORIC_TAG) || dirtyLeaves.size === 0) return;
      if (editor.isComposing()) return;
      const candidates = editorState.read(() =>
        [...dirtyLeaves].filter((key) => {
          const node = $getNodeByKey(key);
          return $isTextNode(node) && TYPED_CHECKBOX.test(node.getTextContent());
        }),
      );
      if (candidates.length === 0) return;
      editor.update(() => {
        for (const key of candidates) {
          const node = $getNodeByKey(key);
          if ($isTextNode(node)) $convertTypedCheckbox(node);
        }
      });
    }),

    // A fence typed onto a block's last line closes it, one update behind the
    // keystroke for the same reason the task conversion is: consuming the line
    // is a structural edit, and running it inside the keystroke's own update
    // would leave undo landing on a half-typed fence rather than on the literal
    // line. The "undo: a typed closer gives back the literal fence" check fails
    // the moment this stops being its own step.
    editor.registerUpdateListener(({ dirtyLeaves, editorState, prevEditorState, tags }) => {
      if (tags.has(HISTORIC_TAG) || dirtyLeaves.size === 0) return;
      if (editor.isComposing()) return;
      const runs = editorState.read(() =>
        [...dirtyLeaves].flatMap((key) => {
          const node = $getNodeByKey(key);
          if (!$isTextNode(node) || !$isCodeNode(node.getParent())) return [];
          const text = node.getTextContent();
          return BACKTICK_RUN.test(text) ? [{ key, text }] : [];
        }),
      );
      if (runs.length === 0) return;
      // The third backtick has to have been TYPED: the leaf grew by exactly the
      // one character this update put there. Text that ARRIVES holding a fence
      // - a pasted snippet, a drop, a restored draft - leaves the same line and
      // the same caret behind it, and closing the block on that would eat a
      // line nobody offered as a closer.
      const candidates = prevEditorState.read(() =>
        runs
          .filter(({ key, text }) => {
            const before = $getNodeByKey(key);
            return $isTextNode(before) && `${before.getTextContent()}\`` === text;
          })
          .map(({ key }) => key),
      );
      if (candidates.length === 0) return;
      editor.update(() => {
        for (const key of candidates) {
          const node = $getNodeByKey(key);
          if ($isTextNode(node)) $closeTypedFence(node);
        }
      });
    }),

    // A block emptied of everything it held stays behind as a shell (see
    // $isStrandedBlock), so it is put back to a plain paragraph. A block the
    // user just OPENED - "- " typed into an empty line, a fence turned into a
    // code block by Shift+Enter or by the markdown shortcut - looks IDENTICAL
    // to an emptied one and is waiting for its first character, so the two are
    // told apart by whether the block existed a moment ago. Collapsing a brand
    // new one would make a bullet, and a code block, untypeable.
    editor.registerUpdateListener(({ dirtyElements, editorState, prevEditorState, tags }) => {
      if (tags.has(HISTORIC_TAG) || dirtyElements.size === 0) return;
      const stranded = editorState.read(() => {
        const blocks = $getRoot().getChildren();
        return blocks
          .filter((node) => dirtyElements.has(node.getKey()) && $isStrandedBlock(node))
          .map((node) => ({ key: node.getKey(), retyped: $keepsItsShell(node, blocks.length) }));
      });
      if (stranded.length === 0) return;
      const emptied = prevEditorState.read(() =>
        stranded
          .filter(({ key }) => $getNodeByKey(key) !== null)
          // A code block cleared from inside itself is being retyped, not
          // abandoned - see $keepsItsShell. Lists are never spared: a naked
          // marker has no language to lose and nothing to retype into.
          .filter(({ key, retyped }) => !retyped || !$selectionWasInside(key))
          .map(({ key }) => key),
      );
      if (emptied.length === 0) return;
      editor.update(() => {
        for (const key of emptied) {
          const block = $getNodeByKey(key);
          if (block === null || !$isStrandedBlock(block)) continue;
          const selection = $getSelection();
          const anchored =
            $isRangeSelection(selection) &&
            selection.anchor.getNode().getTopLevelElement()?.getKey() === block.getKey();
          const paragraph = $createParagraphNode();
          block.replace(paragraph);
          // Only when the caret was already in there: moving it otherwise would
          // pull focus into a composer the user had deliberately left.
          if (anchored) paragraph.select();
        }
      });
    }),

    editor.registerUpdateListener(({ editorState }) => {
      const markdown = editorState.read($serializeComposerMarkdown);
      if (markdown === previous) return;
      previous = markdown;
      read().onChangeMarkdown(markdown);
    }),

    // Enter sends, and has since the first build. Making that depend on what
    // the caret sits next to would turn the one reflex the composer relies on
    // into a guess - so the newline the editor would have inserted, list
    // continuation and all, moves to Shift+Enter instead.
    editor.registerCommand(
      KEY_ENTER_COMMAND,
      (event) => {
        if (event === null) return false;
        // An IME candidate is being confirmed, not submitted - and not asked to
        // break a line either, so the key is swallowed whole. The default is
        // left alone deliberately: confirming the candidate is the browser's
        // job. (Unreachable in a real browser, which never dispatches this
        // mid-composition; it is the headless dispatch that can.)
        if (event.isComposing || editor.isComposing()) return true;
        event.preventDefault();
        if (event.shiftKey) {
          // A fence typed on its own line becomes a block here: the library's
          // own Enter trigger sits below this handler and never sees the key.
          // Mid-typing the same fence converts one character LATE (the library
          // fires on the character AFTER the trigger) - accepted, not fought.
          if ($openTypedFence()) return true;
          // Inside a code block the line is taken here rather than delegated,
          // which is what keeps the key a newline and only a newline - see
          // $breakInsideCode for the escape it is holding shut.
          if ($breakInsideCode()) return true;
          // Otherwise exactly what a plain Enter does everywhere else: an item
          // continues, an empty one leaves the list.
          editor.dispatchCommand(INSERT_PARAGRAPH_COMMAND, undefined);
          return true;
        }
        read().onSend();
        return true;
      },
      COMMAND_PRIORITY_CRITICAL,
    ),

    // ArrowDown is the caret's way out of a code block that ends the draft -
    // the one place the key had nothing to do. See $exitCodeOnArrowDown for
    // everything it deliberately leaves to native navigation; the default is
    // prevented only when the caret really did leave, so no other press of it
    // changes.
    editor.registerCommand(
      KEY_ARROW_DOWN_COMMAND,
      (event) => {
        if (!$exitCodeOnArrowDown()) return false;
        event?.preventDefault();
        return true;
      },
      COMMAND_PRIORITY_CRITICAL,
    ),

    // Tab is never an indent key. It is the browser's focus move - the only way
    // out of a text box for someone not using a mouse - in prose, in a list and
    // in a code block alike; nesting arrives by paste and by draft restore, not
    // by keystroke. The command is swallowed so the rich-text base cannot claim
    // it, and the default is deliberately NOT prevented, which is what leaves
    // the focus ring free to move.
    editor.registerCommand(KEY_TAB_COMMAND, () => true, COMMAND_PRIORITY_CRITICAL),

    // Ctrl+B and friends are wired up by the rich-text base whether or not
    // anything renders them. Swallowing the command is what keeps the box
    // honest: a draft that LOOKED bold would still go out as plain text,
    // because bold is not one of the constructs this composer speaks.
    editor.registerCommand(FORMAT_TEXT_COMMAND, () => true, COMMAND_PRIORITY_CRITICAL),

    editor.registerCommand(
      PASTE_COMMAND,
      (event) => {
        if (!(event instanceof ClipboardEvent)) return false;
        const clipboard = event.clipboardData;
        if (clipboard === null) return false;
        // A paste with no caret is nothing this handler can place; the base
        // editor knows what to do with it, and swallowing it would lose it.
        if (!$isRangeSelection($getSelection())) return false;
        // The page gets first refusal - it turns clipboard files, and text past
        // its size cap, into attachments.
        if (read().onPasteIntercept?.(clipboard) === true) {
          event.preventDefault();
          return true;
        }
        // The builders normalize CR themselves, and so does the editor's raw
        // insert; normalizing here as well is what keeps that independent of
        // either one's internals rather than a fact this file relies on.
        const text = normalizeNewlines(clipboard.getData("text/plain"));
        // Taken over even for plain text, which is what keeps a rich-text paste
        // from smuggling in formatting the composer cannot send: the HTML
        // flavour of the clipboard is never read.
        event.preventDefault();
        // ...and a clipboard carrying ONLY html is dropped rather than handed
        // back: handing it back reaches the rich-text importer, which builds
        // formatted nodes out of that html - the box would show bold while the
        // wire carried plain text, the one lie this module exists to stop.
        // Files are already gone by here, into the page's attachment path.
        if (text === "") return true;
        editor.update(() => $insertComposerText(text));
        return true;
      },
      COMMAND_PRIORITY_CRITICAL,
    ),

    // A drop is a paste that arrived by mouse, and it has to be treated as one:
    // the rich-text base reads the text/html flavour of a drop, which is the
    // one door formatting the composer cannot send could walk through. Routing
    // it here also puts dropped text under the page's paste-size cap.
    editor.registerCommand(
      DROP_COMMAND,
      (event) => {
        if (!(event instanceof DragEvent)) return false;
        const transfer = event.dataTransfer;
        if (transfer === null) return false;
        // Dropped files belong to the page, which turns them into attachments.
        // Nothing is handed over here: the drop keeps bubbling to the page's
        // own handler on the box around this editor.
        if (transfer.files.length > 0) {
          event.preventDefault();
          return true;
        }
        // Text dragged from one place in this box to another is a MOVE, not a
        // copy: taking it over here would insert the text and leave the
        // original where it was, because only the base editor's own drop path
        // removes the source range. Handing that one case back is safe for the
        // same reason reading text/html is not - the base reads the editor's
        // OWN node serialization first, which every drag it starts carries and
        // which is the highest-priority flavour, so the HTML door stays shut;
        // and content that came out of this editor can only hold the nodes this
        // editor registers. A drag from any OTHER editor keeps the plain-text
        // path below, where the page's cap applies and formatting cannot ride
        // along.
        if (isSameEditorDrag(transfer, editor)) return false;
        if (read().onPasteIntercept?.(transfer) === true) {
          event.preventDefault();
          return true;
        }
        const text = normalizeNewlines(transfer.getData("text/plain"));
        event.preventDefault();
        // An html-only payload is dropped rather than handed back, for the same
        // reason the paste door drops one: the base importer would read that
        // html and build formatting the wire cannot carry. Files left for the
        // page above, and a drag out of this box was handed back above too.
        if (text === "") return true;
        const { clientX, clientY } = event;
        editor.update(() => {
          $selectDropPoint(clientX, clientY);
          $insertComposerText(text);
        });
        return true;
      },
      COMMAND_PRIORITY_CRITICAL,
    ),

    // The base editor's own answer to a file drop or a file paste. Nothing
    // above ever lets it be dispatched; swallowing it keeps that true if a
    // future base registration finds another way to raise it.
    editor.registerCommand(DRAG_DROP_PASTE, () => true, COMMAND_PRIORITY_CRITICAL),
  );
}

/**
 * Widen a task item's click target from its drawn box to its whole gutter.
 *
 * A task box is a target, and the drawn 0.9em glyph - about 14x15 CSS pixels -
 * is not a big enough one. The base check-list plugin measures the GLYPH and
 * toggles only inside it; this answers the item's whole indent gutter instead,
 * which is the largest honest target there is: past it the pointer is over the
 * item's own text, where a click belongs to the caret.
 *
 * Both listeners CAPTURE, and the CLICK one stops propagation: the base plugin
 * toggles on a BUBBLE-phase click on the same element, so a click on the glyph
 * - which is inside both bands - would otherwise be toggled twice. The
 * mousedown listener stops nothing: the base's own mousedown only suppresses
 * the caret, and the page's menus close on a mousedown that reaches the
 * document. Touch is deliberately left to the base plugin: its tap target is
 * already 32px wider than this gutter, and it answers a tap on `pointerup`,
 * which a click handler cannot take back.
 *
 * Registered apart from {@link registerComposerBehaviour} because it is the one
 * piece of the composer that needs a real root element - a headless editor has
 * none, and refuses the listener outright.
 *
 * @param editor - The editor to register against.
 * @returns A teardown that unregisters the listeners.
 */
export function registerCheckGutter(editor: LexicalEditor): () => void {
  return editor.registerRootListener((root) => {
    if (root === null) return;
    const onPointerDown = (event: MouseEvent): void => {
      if (!isPrecisePointer(event) || checkGutterTarget(editor, event) === null) return;
      // The gutter is the box's target, not a place to put the caret. The
      // default is all that is taken: the event is left to travel, because the
      // page's open menus - the model picker, the context card, the theme and
      // row menus - all close on a mousedown that reaches the document, and a
      // click that left one of them standing would be the composer's only
      // click that does. Nothing toggles here, so nothing is toggled twice by
      // letting it through; the click handler below is where that is held.
      //
      // Measured, and only half won: over the drawn glyph - the inner ~14px of
      // this 22px band - the base plugin's own capture pointerdown handler
      // preventDefaults, and a prevented pointerdown suppresses the
      // compatibility mousedown outright, so no listener anywhere sees one. A
      // menu left standing by a click on the glyph is that, not this, and it
      // cannot be answered from here; the outer gutter is what this reaches.
      event.preventDefault();
    };
    const onClick = (event: MouseEvent): void => {
      if (!isPrecisePointer(event) || !editor.isEditable()) return;
      const item = checkGutterTarget(editor, event);
      if (item === null) return;
      event.preventDefault();
      event.stopPropagation();
      editor.update(() => {
        const node = $getNearestNodeFromDOMNode(item);
        if ($isListItemNode(node)) node.toggleChecked();
      });
      // The item carries role="checkbox" and tabindex="-1" for exactly this:
      // focusing it is what lets Space toggle the box that was just clicked.
      item.focus();
    };
    root.addEventListener("mousedown", onPointerDown, { capture: true });
    root.addEventListener("click", onClick, { capture: true });
    return () => {
      root.removeEventListener("mousedown", onPointerDown, { capture: true });
      root.removeEventListener("click", onClick, { capture: true });
    };
  });
}
