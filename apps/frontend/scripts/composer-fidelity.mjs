/**
 * Round-trip and behaviour battery for the composer's editor.
 *
 * The composer renders markers as glyphs but sends the string a person would
 * have typed, so the two conversions in `lib/composer-markdown.ts` are the
 * whole contract. This drives them through a real headless editor - the same
 * nodes the browser builds - and checks four things:
 *
 *   1. Ordinary prose survives byte-identical. Generic markdown exporters
 *      escape `*`, `_`, `[` and friends; nobody typed a backslash.
 *   2. The composer's own constructs come back in their canonical typed form,
 *      and reading the composer's own output back gives the same draft again.
 *   3. What comes out parses, under the ANSWER side's own parser, into the
 *      shape it was composed in - so a nested list in the box is a nested list
 *      in the message.
 *   4. The keystrokes, the paste and the drop behave, driven through the real
 *      command handlers with the base editor registered underneath them.
 *
 * Run it as the workspace's test command:
 *
 *     npm test        (or: node scripts/composer-fidelity.mjs)
 *
 * It reports every failing case and exits non-zero if there was one.
 *
 * It imports the app's TypeScript modules directly, with no build step, so it
 * needs a Node that strips types on its own - 22.18 or newer. The check below
 * says so; without it the failure is an unexplained "unknown file extension".
 */

import { $createCodeNode, CodeNode } from "@lexical/code-core";
import { createHeadlessEditor } from "@lexical/headless";
import { createEmptyHistoryState, registerHistory } from "@lexical/history";
import {
  $createListItemNode,
  $createListNode,
  ListItemNode,
  ListNode,
  registerList,
} from "@lexical/list";
import { registerRichText } from "@lexical/rich-text";
import { mergeRegister } from "@lexical/utils";
import { register } from "node:module";

import {
  $createLineBreakNode,
  $createTextNode,
  $getRoot,
  $getSelection,
  $isRangeSelection,
  $setSelection,
  COMMAND_PRIORITY_HIGH,
  DROP_COMMAND,
  FORMAT_TEXT_COMMAND,
  KEY_ARROW_DOWN_COMMAND,
  KEY_DELETE_COMMAND,
  KEY_ENTER_COMMAND,
  KEY_TAB_COMMAND,
  PASTE_COMMAND,
  SELECT_ALL_COMMAND,
  UNDO_COMMAND,
} from "lexical";
import remarkGfm from "remark-gfm";
import remarkParse from "remark-parse";
import { unified } from "unified";

// Checked before the first TypeScript module is reached, which is why that one
// is loaded on demand rather than named in an import above: a static import is
// resolved before any line of this file runs, so the guard would never speak.
const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `composer fidelity: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

const { $buildComposerNodes, $serializeComposerMarkdown, hasComposerConstructs } =
  await import("../lib/composer-markdown.ts");

/** A fresh editor with the composer's node set and nothing registered on it. */
function makeEditor(namespace) {
  return createHeadlessEditor({
    namespace,
    nodes: [ListNode, ListItemNode, CodeNode],
    onError: (error) => {
      throw error;
    },
  });
}

// Two editors, because the two halves of the battery want different things:
// `editor` is bare, so a round trip measures the conversions and nothing else,
// while `live` carries the full registration the browser gets.
const editor = makeEditor("composer-fidelity");
const live = makeEditor("composer-fidelity-live");

/** Read `markdown` into editor nodes, then write it back out. */
function roundTrip(markdown) {
  let out = "";
  editor.update(
    () => {
      const root = $getRoot();
      root.clear();
      root.append(...$buildComposerNodes(markdown));
    },
    { discrete: true },
  );
  editor.getEditorState().read(() => {
    out = $serializeComposerMarkdown();
  });
  return out;
}

// The answer-side renderer's own parser, so both ends of the app read one
// grammar: whatever this accepts is what the reader will see.
const gfmParser = unified().use(remarkParse).use(remarkGfm).freeze();

/** The syntax tree the message renderer would build from `markdown`. */
function parseGfm(markdown) {
  return gfmParser.parse(markdown);
}

let failures = 0;
let checks = 0;

function check(label, actual, expected) {
  checks += 1;
  const ok = JSON.stringify(actual) === JSON.stringify(expected);
  if (!ok) {
    failures += 1;
    console.error(`FAIL  ${label}`);
    console.error(`  expected ${JSON.stringify(expected)}`);
    console.error(`  actual   ${JSON.stringify(actual)}`);
  }
  return ok;
}

/** The draft must come back exactly as it went in. */
function stable(label, markdown) {
  check(label, roundTrip(markdown), markdown);
}

// ---------------------------------------------------------------------------
// 1. Prose survives byte-identical (the escaping battery)
// ---------------------------------------------------------------------------

for (const prose of [
  "2*3=6",
  "a_b_c",
  "[not a link]",
  "1.5 liters",
  "-dashes-",
  "#hashtag",
  "**bold** stays literal",
  "~~strike~~ stays literal",
  "> quoted line stays literal",
  "# heading stays literal",
  "path\\to\\file",
  "back`tick` in prose",
  "[link](https://example.com)",
  "1) not an ordered item",
  "-item",
  "3 - 5 and 3-5",
  "| a | b |",
  "  leading and trailing  ",
]) {
  stable(`prose: ${prose}`, prose);
}

check(
  "prose: no backslash is ever introduced",
  roundTrip("2*3=6 a_b_c [x] ~t~ `c`").includes("\\"),
  false,
);

// Multi-line prose: each typed line comes back as a typed line, blank lines
// included. A hard-break marker (trailing double space or backslash) is never
// invented.
stable("prose: two lines", "first line\nsecond line");
stable("prose: blank line between", "first\n\nsecond");
stable("prose: trailing blank line", "first\n");

// ---------------------------------------------------------------------------
// 2. In-scope constructs take their canonical typed form
// ---------------------------------------------------------------------------

stable("bullet: one item", "- item");
stable("bullet: two items", "- one\n- two");
stable("bullet: bare trailing marker", "- one\n- ");
check("bullet: star normalizes to dash", roundTrip("* one\n* two"), "- one\n- two");
check("bullet: plus normalizes to dash", roundTrip("+ one"), "- one");

stable("ordered: from one", "1. one\n2. two");
stable("ordered: from another start", "5. five\n6. six");
check("ordered: renumbers to the sequence", roundTrip("1. one\n1. two\n1. three"), "1. one\n2. two\n3. three");
check("ordered: a paren delimiter stays prose", roundTrip("1) one"), "1) one");

stable("task: unchecked", "- [ ] todo");
stable("task: checked", "- [x] done");
stable("task: mixed", "- [ ] todo\n- [x] done");
check("task: capital X normalizes", roundTrip("- [X] done"), "- [x] done");

stable("code: fenced with language", "```python\nprint(1)\n```");
stable("code: fenced without language", "```\nplain\n```");
stable("code: empty block", "```\n```");
stable("code: markers inside are content", "```\n- not a bullet\n1. not ordered\n```");
stable("code: a blank line inside survives", "```\na\n\nb\n```");
check(
  "code: an unclosed fence is closed on the way out",
  roundTrip("```js\nlet x = 1"),
  "```js\nlet x = 1\n```",
);
check(
  "code: a block holding a fence gets a longer one",
  roundTrip("````\n```\n````"),
  "````\n```\n````",
);

stable("mixed: prose then list then prose", "before\n- one\n- two\nafter");
stable("mixed: list then code", "- one\n```\ncode\n```");

// Nested lists indent by the PARENT MARKER's width, checkbox excluded.
stable("nested: bullet under bullet", "- one\n  - deep");
stable("nested: three levels", "- one\n  - two\n    - three");
stable("nested: bullet under ordered (3 columns)", "1. one\n   - deep");
stable("nested: bullet under a wide ordinal (4 columns)", "10. ten\n    - deep");
stable("nested: ordered under bullet", "- one\n  1. deep");
stable("nested: bullet under a task (2 columns, not 6)", "- [ ] todo\n  - deep");
stable("nested: back out to the parent level", "- one\n  - deep\n- two");

// A wrapped line inside an item is written back at the item's content column.
stable("wrap: continuation inside a bullet item", "- one\n  wrapped");
stable("wrap: continuation inside an ordered item", "1. one\n   wrapped");

// A fence indented to the item's own text is the item's CONTENT: it must not
// escape the list, and it must not keep the item's indentation in the code.
stable("fence in item: under an ordered item", "1. one\n   ```py\n   x = 1\n   ```");
stable("fence in item: the list carries on after it", "- one\n  ```\n  code\n  ```\n- two");
stable("fence in item: an unindented fence still ends the list", "- one\n```\ncode\n```");

// A carriage return arrives on every Windows clipboard, and every pattern here
// is anchored to the end of the line.
check("crlf: a pasted list is still a list", roundTrip("- a\r\n- b"), "- a\n- b");
check("crlf: a pasted fence is still code", roundTrip("```js\r\nx\r\n```"), "```js\nx\n```");
check("crlf: a bare CR is normalized too", roundTrip("- a\r- b"), "- a\n- b");
check("crlf: no carriage return survives", roundTrip("hello\r\nworld").includes("\r"), false);
// A fence is the strictest of the patterns - it has to reach the end of the
// line with nothing after the language - so it is the one a stray CR hides.
check("crlf: a fence is still recognized through it", hasComposerConstructs("```js\r\nx"), true);

// Lexical forbids a newline inside a text node; a restored code block has to
// carry its line endings as break nodes instead.
function textNodesHoldingNewlines(markdown) {
  let found = [];
  editor.update(
    () => {
      const root = $getRoot();
      root.clear();
      root.append(...$buildComposerNodes(markdown));
    },
    { discrete: true },
  );
  editor.getEditorState().read(() => {
    found = $getRoot()
      .getAllTextNodes()
      .map((node) => node.getTextContent())
      .filter((text) => text.includes("\n"));
  });
  return found;
}
check("code: no text node holds a newline", textNodesHoldingNewlines("```py\na\nb\nc\n```"), []);
check(
  "code: no text node holds a newline (in a list item)",
  textNodesHoldingNewlines("- one\n  ```\na\nb\n  ```"),
  [],
);

// A block child reached through an element - a nested list, or a fence inside a
// list item - must be handed to the serializer that knows its markers.
function handBuilt(build) {
  let out = "";
  editor.update(
    () => {
      const root = $getRoot();
      root.clear();
      build(root);
    },
    { discrete: true },
  );
  editor.getEditorState().read(() => {
    out = $serializeComposerMarkdown();
  });
  return out;
}

check(
  "inline: a list beside text in an item keeps its markers",
  handBuilt((root) => {
    const list = $createListNode("bullet");
    const item = $createListItemNode();
    item.append($createTextNode("one"));
    const nested = $createListNode("bullet");
    const deep = $createListItemNode();
    deep.append($createTextNode("deep"));
    nested.append(deep);
    item.append(nested);
    list.append(item);
    root.append(list);
  }),
  "- one\n  - deep",
);
check(
  "inline: a code block beside text in an item keeps its fence",
  handBuilt((root) => {
    const list = $createListNode("bullet");
    const item = $createListItemNode();
    item.append($createTextNode("one"));
    const code = $createCodeNode("py");
    code.append($createTextNode("x = 1"));
    item.append(code);
    list.append(item);
    root.append(list);
  }),
  "- one\n  ```py\n  x = 1\n  ```",
);

// A wrapper item is meant to hold exactly ONE list, and the library reads a
// nesting level off the FIRST child alone - so an item holding two draws as one
// nesting level while a serializer that demanded one child fell through and
// wrote a bare marker for a line the box never showed. Both lists belong at the
// nested depth, which is where they were drawn.
check(
  "inline: an item holding two lists writes both, and no bare marker",
  handBuilt((root) => {
    const list = $createListNode("bullet");
    const item = $createListItemNode();
    item.append($createTextNode("a"));
    list.append(item);
    const wrapper = $createListItemNode();
    const tasks = $createListNode("check");
    const task = $createListItemNode(false);
    task.append($createTextNode("b"));
    tasks.append(task);
    const rest = $createListNode("bullet");
    const plain = $createListItemNode();
    plain.append($createTextNode("c"));
    rest.append(plain);
    wrapper.append(tasks, rest);
    list.append(wrapper);
    root.append(list);
  }),
  "- a\n  - [ ] b\n  - c",
);

// ---------------------------------------------------------------------------
// 3. What comes out is what the ANSWER side parses back
// ---------------------------------------------------------------------------

/** The list nesting the answer-side parser finds in `markdown`. */
function listShape(markdown) {
  const shape = [];
  const walk = (node, depth) => {
    if (node.type === "list") {
      for (const item of node.children) {
        const label = item.checked === null || item.checked === undefined ? "item" : `task:${item.checked}`;
        shape.push(`${"  ".repeat(depth)}${node.ordered ? "ol" : "ul"}/${label}`);
        for (const child of item.children) walk(child, depth + 1);
      }
      return;
    }
    for (const child of node.children ?? []) walk(child, depth);
  };
  walk(parseGfm(markdown), 0);
  return shape;
}

check("gfm: bullet under bullet nests", listShape(roundTrip("- one\n  - deep")), [
  "ul/item",
  "  ul/item",
]);
check("gfm: bullet under ordered nests", listShape(roundTrip("1. one\n   - deep")), [
  "ol/item",
  "  ul/item",
]);
check("gfm: bullet under a wide ordinal nests", listShape(roundTrip("10. ten\n    - deep")), [
  "ol/item",
  "  ul/item",
]);
check("gfm: three levels nest", listShape(roundTrip("- one\n  - two\n    - three")), [
  "ul/item",
  "  ul/item",
  "    ul/item",
]);
check("gfm: a task list keeps its boxes", listShape(roundTrip("- [ ] todo\n- [x] done")), [
  "ul/task:false",
  "ul/task:true",
]);
check("gfm: a nested list under a task nests, not code", listShape(roundTrip("- [ ] todo\n  - deep")), [
  "ul/task:false",
  "  ul/item",
]);

/** Node types the answer-side parser finds at the top level. */
function blockTypes(markdown) {
  return parseGfm(markdown).children.map((node) => node.type);
}

check("gfm: a nested list is never read as indented code", blockTypes(roundTrip("- one\n  - deep")), [
  "list",
]);
check("gfm: a wrapped item line stays inside the item", blockTypes(roundTrip("- one\n  wrapped")), [
  "list",
]);
check("gfm: a fenced block stays code", blockTypes(roundTrip("```py\nx = 1\n```")), ["code"]);
check(
  "gfm: a fence inside an item stays inside the list",
  blockTypes(roundTrip("1. one\n   ```py\n   x = 1\n   ```")),
  ["list"],
);
check(
  "gfm: and the reader sees it as code, not prose",
  (() => {
    const tree = parseGfm(roundTrip("1. one\n   ```py\n   x = 1\n   ```"));
    const item = tree.children[0].children[0];
    return item.children.map((node) => node.type);
  })(),
  ["paragraph", "code"],
);

// Prose the composer leaves alone must ALSO reach the reader as prose - the
// answer side parses what the composer sends, so an escape the composer never
// wrote is an escape the reader never sees.
check("gfm: prose emphasis markers reach the reader intact", (() => {
  const sent = roundTrip("2*3=6 and a_b_c");
  const tree = parseGfm(sent);
  return tree.children[0].children.map((n) => n.type);
})(), ["text"]);

// ---------------------------------------------------------------------------
// 3b. Reading the composer's own output back gives the same draft
// ---------------------------------------------------------------------------
// The wire is written in canonical columns and read in source ones, so a draft
// that lands in the box, gets sent, and comes back (a failed send restores it)
// must not drift. A seeded generator, never Math.random: a case that fails has
// to fail again on the next run.

let fuzzSeed = 20260806;
function nextRandom() {
  fuzzSeed = (fuzzSeed * 1103515245 + 12345) % 2147483648;
  return fuzzSeed / 2147483648;
}

const FUZZ_LINES = [
  "hello",
  "",
  "- a",
  "  - b",
  "    - c",
  "      - d",
  "1. one",
  "   2. two",
  "10. ten",
  " 9. nine",
  // A three-digit ordinal, so the corpus can generate a marker that loses TWO
  // or more columns when the list is renumbered. Without one the widest change
  // it could produce was a single column, which is why the drift below went
  // unseen for a whole round.
  "100. hundred",
  "- [ ] todo",
  "- [x] done",
  "  - [x] deep task",
  "```py",
  "```",
  "  ```",
  "   ```js",
  "````",
  "  wrapped",
  "   deep",
  "     deeper",
  "\ttabbed",
  "  \t mixed",
  "2*3=6",
  "  - [ ] t",
  "* star",
  "+ plus",
  "1) paren",
  "    indented code?",
  "> quote",
  "#hash",
];

let drifted = null;
let fuzzed = 0;
for (let round = 0; round < 3000; round += 1) {
  const length = 1 + Math.floor(nextRandom() * 8);
  const draft = Array.from(
    { length },
    () => FUZZ_LINES[Math.floor(nextRandom() * FUZZ_LINES.length)],
  ).join("\n");
  const once = roundTrip(draft);
  fuzzed += 1;
  if (roundTrip(once) !== once) {
    drifted = { draft, once, twice: roundTrip(once) };
    break;
  }
}
check(`round trip: ${fuzzed} generated drafts are idempotent`, drifted, null);

for (const draft of [
  "  - one\n   deep",
  "  - one\n  - two",
  "  - one\n    - deep",
  "   1. a\n      wrapped\n   2. b",
  "- [x] done\n    - c\n     deeper",
  // An ordinal that loses two columns when the list is renumbered. The line
  // below it has to lose the same two, or the second read pulls it in.
  "1. a\n100. b\n    x",
  "1. a\n999. b\n     x",
  "9. a\n100. b\n      x",
]) {
  const once = roundTrip(draft);
  check(`round trip: idempotent for ${JSON.stringify(draft)}`, roundTrip(once), once);
}

/** The top-level blocks the parser builds, so absorption shows as a shape change. */
function parsedShape(markdown) {
  let out = [];
  editor.update(
    () => {
      const root = $getRoot();
      root.clear();
      root.append(...$buildComposerNodes(markdown));
    },
    { discrete: true },
  );
  editor.getEditorState().read(() => {
    out = $getRoot()
      .getChildren()
      .map((node) => (node.getType() === "list" ? node.getListType() : node.getType()));
  });
  return out;
}

// Idempotence of the STRING is not the whole story: a renumbered marker that
// leaves the line below it over-indented changes the TREE on the second read -
// the paragraph after the list is pulled into the last item and stops being a
// paragraph at all. The reader sees a different message from the one composed.
for (const draft of ["1. a\n100. b\n    x", "1. a\n999. b\n     x", "9. a\n100. b\n      x"]) {
  check(
    `round trip: the block after a re-numbered ordinal survives ${JSON.stringify(draft)}`,
    parsedShape(roundTrip(draft)),
    parsedShape(draft),
  );
}

// Every ordinal width against every ordinal width, at every indentation a
// following line can carry. The fuzz above cannot promise this corner is
// covered; this can, and it costs milliseconds.
{
  const opening = [1, 2, 9, 10, 100];
  const following = [1, 2, 3, 9, 10, 100, 999];
  const indents = 9;
  let unstable = null;
  for (const first of opening) {
    for (const second of following) {
      for (let indent = 0; indent < indents; indent += 1) {
        const draft = `${first}. a\n${second}. b\n${" ".repeat(indent)}x`;
        const once = roundTrip(draft);
        if (roundTrip(once) !== once) {
          unstable = { draft, once, twice: roundTrip(once) };
          break;
        }
      }
    }
  }
  check(
    `round trip: ${opening.length * following.length * indents} ordinal widths x indentations are idempotent`,
    unstable,
    null,
  );
}

// ---------------------------------------------------------------------------
// 4. The keystrokes, paste and drop, through the real command handlers
// ---------------------------------------------------------------------------

register("./workspace-alias-hook.mjs", import.meta.url);
const {
  COMPOSER_TRANSFORMERS,
  registerComposerBehaviour,
} = await import("../lib/composer-editor.ts");

const { CHECK_LIST, CODE, ORDERED_LIST, TEXT_FORMAT_TRANSFORMERS, UNORDERED_LIST } =
  await import("@lexical/markdown");

// Node has no clipboard or drag events; the handlers gate on them by type, so
// the battery supplies the two shapes rather than skipping the paths that carry
// every construct into the box from outside.
class FakeDataTransfer {
  constructor(data = {}, files = []) {
    this.entries = data;
    this.files = files;
    this.asked = [];
    this.types = [...Object.keys(data), ...(files.length > 0 ? ["Files"] : [])];
  }

  getData(type) {
    this.asked.push(type);
    return this.entries[type] ?? "";
  }
}

globalThis.ClipboardEvent = class ClipboardEvent {
  constructor(clipboardData) {
    this.clipboardData = clipboardData;
    this.defaultPrevented = false;
  }

  preventDefault() {
    this.defaultPrevented = true;
  }
};

globalThis.DragEvent = class DragEvent {
  constructor(dataTransfer) {
    this.dataTransfer = dataTransfer;
    this.clientX = 0;
    this.clientY = 0;
    this.defaultPrevented = false;
  }

  preventDefault() {
    this.defaultPrevented = true;
  }
};

// The behaviour under test sits ON TOP of the base registrations the browser
// gets from RichTextPlugin, ListPlugin and HistoryPlugin - the paragraph
// insertion it delegates to, and the undo stack it pushes onto, live there.
// Registering them in the same order is what makes the precedence real: the
// composer's Enter has to beat the rich-text one, and this is where that is
// proved.
let sends = 0;
let intercept = null;
const teardown = mergeRegister(
  registerRichText(live),
  registerList(live),
  registerHistory(live, createEmptyHistoryState(), 300),
  registerComposerBehaviour(live, () => ({
    onChangeMarkdown: () => {},
    onPasteIntercept: (clipboard) => (intercept === null ? false : intercept(clipboard)),
    onSend: () => {
      sends += 1;
    },
  })),
);

/** A keyboard event that records whether the handler took it. */
function keyEvent(init = {}) {
  return {
    isComposing: false,
    prevented: false,
    preventDefault() {
      this.prevented = true;
    },
    shiftKey: false,
    ...init,
  };
}

/**
 * Let every deferred update land.
 *
 * Two of the composer's repairs - the typed checkbox and the stranded-list
 * collapse - run one update BEHIND the change that triggered them, on purpose:
 * that is what gives each its own entry on the undo stack.
 */
function flush() {
  for (let round = 0; round < 3; round += 1) live.update(() => {}, { discrete: true });
}

/**
 * Put `markdown` in the live editor and place the caret.
 *
 * `where` is `"end"` - the end of the whole draft, and the default - or
 * `"all"`, the whole draft selected, or a position inside one of the draft's
 * text leaves: `{leaf, anchor, focus}`, where `leaf` indexes the leaves in
 * document order and omitting `focus` leaves the caret collapsed.
 *
 * The insertion rules turn on where the caret sits INSIDE its own block, so a
 * matrix that only ever seeds the end measures one of them and guesses at the
 * rest. That is exactly how a mid-word paste got to eat the tail of the line.
 */
function seed(markdown, where = "end") {
  live.update(
    () => {
      const root = $getRoot();
      root.clear();
      root.append(...$buildComposerNodes(markdown));
      root.selectEnd();
    },
    { discrete: true },
  );
  flush();
  if (where === "end") return;
  live.update(
    () => {
      const root = $getRoot();
      if (where === "all") {
        root.select(0, root.getChildrenSize());
        return;
      }
      const leaf = root.getAllTextNodes()[where.leaf ?? 0];
      leaf.select(where.anchor, where.focus ?? where.anchor);
    },
    { discrete: true },
  );
  flush();
}

/** The draft the live editor is holding. */
function wire() {
  let out = "";
  live.getEditorState().read(() => {
    out = $serializeComposerMarkdown();
  });
  return out;
}

/** The live editor's top-level blocks, as type names. */
function blocks() {
  let out = [];
  live.getEditorState().read(() => {
    out = $getRoot()
      .getChildren()
      .map((node) => (node.getType() === "list" ? node.getListType() : node.getType()));
  });
  return out;
}

/**
 * The live editor's top-level blocks, as [kind, text] pairs.
 *
 * A list reports its KIND, because that is the difference the draft string
 * cannot show: a bullet item reading "[ ] " and a task item serialize alike.
 */
function nodes() {
  let out = [];
  live.getEditorState().read(() => {
    out = $getRoot()
      .getChildren()
      .map((node) => [
        node.getType() === "list" ? node.getListType() : node.getType(),
        node.getTextContent(),
      ]);
  });
  return out;
}

/**
 * Where the caret is: the index and kind of the top-level block holding it.
 *
 * A gesture that moves the caret OUT of a block is only half done if the caret
 * stayed behind, and the draft string cannot show that at all - both outcomes
 * serialize identically.
 */
function caret() {
  let out = null;
  live.getEditorState().read(() => {
    const selection = $getSelection();
    if (!$isRangeSelection(selection)) return;
    const block = selection.anchor.getNode().getTopLevelElement();
    if (block === null) return;
    out = [block.getIndexWithinParent(), block.getType() === "list" ? block.getListType() : block.getType()];
  });
  return out;
}

/** Press a key, then read the draft it left behind. */
function press(command, event) {
  live.dispatchCommand(command, event);
  flush();
  return wire();
}

/** The newline key, which is the one key neither exit gesture may touch. */
function shiftEnter() {
  return press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true }));
}

/** Type `text` one character at a time, the way a person does. */
function type(text) {
  for (const character of text) {
    live.update(
      () => {
        const selection = $getSelection();
        if ($isRangeSelection(selection)) selection.insertText(character);
      },
      { discrete: true },
    );
    flush();
  }
}

/** Empty the live editor and put the caret in it. */
function blank() {
  live.update(
    () => {
      $getRoot().clear();
    },
    { discrete: true },
  );
  live.update(
    () => {
      $getRoot().selectEnd();
    },
    { discrete: true },
  );
  flush();
}

// Plain Enter sends, wherever the caret is - that is the whole point of the
// inversion, and a caret inside a list must not turn it into a guess.
for (const [label, draft] of [
  ["in prose", "hello"],
  ["in a list item", "- one"],
  ["in a code block", "```\nx\n```"],
]) {
  const before = sends;
  seed(draft);
  const after = press(KEY_ENTER_COMMAND, keyEvent());
  check(`enter: sends ${label}`, sends - before, 1);
  check(`enter: leaves the draft alone ${label}`, after, draft);
}

// ...except while an IME candidate is open, where Enter is a confirmation. The
// message must not go out half-composed, and the draft must not gain a line
// either: the key is swallowed whole and the default is left to the browser,
// whose job confirming the candidate is.
{
  const before = sends;
  seed("hello");
  const event = keyEvent({ isComposing: true });
  const after = press(KEY_ENTER_COMMAND, event);
  check("enter: an IME confirmation does not send", sends - before, 0);
  check("enter: an IME confirmation does not change the draft", after, "hello");
  check("enter: an IME confirmation leaves the default alone", event.prevented, false);
}

// Shift+Enter is the newline, and carries the list semantics with it.
seed("- one");
check("shift+enter: continues a list item", press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })), "- one\n- ");
seed("1. one");
check(
  "shift+enter: continues an ordered list with the next ordinal",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "1. one\n2. ",
);
seed("- [x] done");
check(
  "shift+enter: continues a task unchecked",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "- [x] done\n- [ ] ",
);
seed("- one\n- ");
check(
  "shift+enter: an empty item leaves the list",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "- one\n",
);
seed("hello");
check(
  "shift+enter: breaks a line in prose",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "hello\n",
);
check(
  "shift+enter: takes a newline inside a code block",
  (seed("```\ncode\n```"), press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true }))),
  "```\ncode\n\n```",
);
{
  const before = sends;
  seed("hello");
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true }));
  check("shift+enter: never sends", sends - before, 0);
}

// A fence typed on its own line opens a code block on Shift+Enter. The
// library's own Enter trigger sits below the composer's handler and never sees
// the key, so without this a fence would sit there as two paragraphs.
blank();
type("```js");
check("shift+enter: a typed fence opens a code block", press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })), "```js\n```");
check("shift+enter: and it really is a code node", blocks(), ["code"]);
blank();
type("```");
check(
  "shift+enter: a fence with no language opens one too",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "```\n```",
);
seed("``` not a fence");
check(
  "shift+enter: a line that only starts with backticks stays prose",
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true })),
  "``` not a fence\n",
);

// Tab is never an indent key: it is the browser's focus move, everywhere in the
// box. The command is swallowed so the rich-text base cannot claim it, and the
// default is left alone so the focus ring can move.
for (const [label, draft] of [
  ["in prose", "hello"],
  ["in a list item", "- one\n- two"],
  ["in a code block", "```\nx\n```"],
]) {
  seed(draft);
  const event = keyEvent();
  const handled = live.dispatchCommand(KEY_TAB_COMMAND, event);
  flush();
  check(`tab: leaves the draft alone ${label}`, wire(), draft);
  check(`tab: is swallowed ${label}`, handled, true);
  check(`tab: does not prevent the default ${label}`, event.prevented, false);
}
{
  seed("- one\n  - two");
  const event = keyEvent({ shiftKey: true });
  live.dispatchCommand(KEY_TAB_COMMAND, event);
  flush();
  check("shift+tab: does not outdent either", wire(), "- one\n  - two");
  check("shift+tab: does not prevent the default", event.prevented, false);
}

// Typing "- [ ] " into a list item turns THAT item into a task and leaves the
// list around it alone. The "- " has already made the line a bullet by the time
// the box is typed, so this is the only path a typed task ever takes.
/**
 * Type `text` at the head of the item at `index`, the way a keystroke would.
 *
 * The caret lands immediately behind what was typed, because that is what a
 * keystroke leaves behind and the conversion reads it: a helper that left the
 * caret elsewhere would be testing a state no keystroke can produce.
 *
 * Reports the BLOCKS as well as the draft, because the draft alone cannot tell
 * the two outcomes apart: a bullet item whose text happens to read "[ ] two"
 * serializes to the same "- [ ] two" a real task item does.
 */
function typeIntoItem(markdown, index, text) {
  seed(markdown);
  live.update(
    () => {
      const item = $getRoot().getFirstChild().getChildren()[index];
      const label = item.getFirstChild();
      label.setTextContent(text + label.getTextContent());
      label.select(text.length, text.length);
    },
    { discrete: true },
  );
  flush();
  return { blocks: blocks(), markdown: wire() };
}

/**
 * How many lists each wrapper item in the live draft holds.
 *
 * The editor models a nesting level as a list inside a list item of its own,
 * and the library reads "this is a nesting level" off the item's FIRST child.
 * So an item holding TWO lists still draws as one level - the box looks right -
 * while the serializer no longer recognizes the item and writes a bare marker
 * for a line nobody typed. Anything but a run of 1s is that shape, and it is
 * the only place the corruption is visible before it reaches the wire.
 */
function wrapperListCounts() {
  const out = [];
  live.getEditorState().read(() => {
    const walk = (node) => {
      if (node.getChildren === undefined) return;
      const children = node.getChildren();
      if (node.getType() === "listitem") {
        const lists = children.filter((child) => child.getType() === "list").length;
        if (lists > 0) out.push(lists);
      }
      for (const child of children) walk(child);
    };
    for (const block of $getRoot().getChildren()) walk(block);
  });
  return out;
}

/**
 * Type `text` at the head of the item whose own text reads `label`.
 *
 * `typeIntoItem` indexes the FIRST block's top-level list, which cannot reach a
 * nested item; the keystroke it plants is otherwise identical, caret and all.
 *
 * Reports three readings of one outcome, because no one of them is enough: the
 * wire is what gets sent, the wrapper counts are the node shape the wire is
 * written from, and the parsed shape is what the reader builds back.
 */
function typeIntoNested(markdown, label, text) {
  seed(markdown);
  live.update(
    () => {
      const leaf = $getRoot()
        .getAllTextNodes()
        .find((node) => node.getTextContent() === label);
      leaf.setTextContent(text + leaf.getTextContent());
      leaf.select(text.length, text.length);
    },
    { discrete: true },
  );
  flush();
  const markdownOut = wire();
  return { markdown: markdownOut, read: listShape(markdownOut), wrappers: wrapperListCounts() };
}

check("typed task: a lone bullet becomes a task", typeIntoItem("- todo", 0, "[ ] "), {
  blocks: ["check"],
  markdown: "- [ ] todo",
});
check("typed task: a checked box is honoured", typeIntoItem("- done", 0, "[x] "), {
  blocks: ["check"],
  markdown: "- [x] done",
});
check("typed task: the bullets above it stay bullets", typeIntoItem("- one\n- two", 1, "[ ] "), {
  blocks: ["bullet", "check"],
  markdown: "- one\n- [ ] two",
});
check(
  "typed task: the items below it stay, in order",
  typeIntoItem("- one\n- two\n- three", 1, "[ ] "),
  { blocks: ["bullet", "check", "bullet"], markdown: "- one\n- [ ] two\n- three" },
);
check("typed task: a bracket that is not a box is left alone", typeIntoItem("- todo", 0, "[not a box] "), {
  blocks: ["bullet"],
  markdown: "- [not a box] todo",
});
// The box only counts when it OPENS the item; a wrapped line that happens to
// start with one is prose the user wrote.
stable("typed task: a checkbox on a wrapped line is prose", "- one\n  [ ] not a box");

// An ORDERED item keeps its ordinal and its literal box. GFM defines a task by
// the item's content, so "1. [ ] milk" is a valid ordered task list; converting
// it would swap the ordinal for a dash - destroying what was typed, and making
// the typed form disagree with the pasted one.
check(
  "typed task: an ordered item keeps its ordinal and the literal box",
  typeIntoItem("1. one\n2. two\n3. three", 1, "[ ] "),
  { blocks: ["number"], markdown: "1. one\n2. [ ] two\n3. three" },
);
check("typed task: a lone ordered item keeps its ordinal", typeIntoItem("1. buy milk", 0, "[ ] "), {
  blocks: ["number"],
  markdown: "1. [ ] buy milk",
});
check(
  "typed task: an ordered list is not split in two by a box",
  typeIntoItem("1. one\n2. two", 1, "[ ] ").blocks,
  ["number"],
);
// ...and the two doors agree on it, byte for byte and shape for shape.
check("agreement: a pasted ordered box stays an ordered item", roundTrip("1. [ ] buy milk"), "1. [ ] buy milk");
check(
  "agreement: a typed ordered box sends what the pasted one does",
  typeIntoItem("1. buy milk", 0, "[ ] ").markdown,
  roundTrip("1. [ ] buy milk"),
);
{
  // Typed char by char, which is the path a person actually takes.
  blank();
  type("1. [ ] buy milk");
  check("agreement: typing the ordered form leaves an ordered item", blocks(), ["number"]);
  check("agreement: and it sends as typed", wire(), "1. [ ] buy milk");
}

// A box typed into a NESTED item splits the nested list in two, and each half
// needs a wrapper item of its own. Dropped into the SAME wrapper the two lists
// still draw as one nesting level - the box looks right - but the serializer
// stops recognizing the item and writes a bare "- " for a line nobody typed,
// which no amount of send-time trimming can reach because it is not the last
// line. Every case reports the wire, the wrapper counts and what the reader
// parses back, because the wire alone cannot show the shape it came from.
check(
  "typed task: a box at the head of a nested run",
  typeIntoNested("- a\n  - b\n  - c", "b", "[ ] "),
  {
    markdown: "- a\n  - [ ] b\n  - c",
    read: ["ul/item", "  ul/task:false", "  ul/item"],
    wrappers: [1, 1],
  },
);
check(
  "typed task: a box in the middle of a nested run",
  typeIntoNested("- a\n  - b\n  - c\n  - d", "c", "[ ] "),
  {
    markdown: "- a\n  - b\n  - [ ] c\n  - d",
    read: ["ul/item", "  ul/item", "  ul/task:false", "  ul/item"],
    wrappers: [1, 1, 1],
  },
);
check(
  "typed task: a box at the end of a nested run",
  typeIntoNested("- a\n  - b\n  - c", "c", "[ ] "),
  {
    markdown: "- a\n  - b\n  - [ ] c",
    read: ["ul/item", "  ul/item", "  ul/task:false"],
    wrappers: [1, 1],
  },
);
// The nested list empties out entirely, so the wrapper it lived in has to go
// with it - an item left holding nothing is a bullet the box never drew.
check("typed task: a box on a lone nested item", typeIntoNested("- a\n  - b", "b", "[ ] "), {
  markdown: "- a\n  - [ ] b",
  read: ["ul/item", "  ul/task:false"],
  wrappers: [1],
});
check(
  "typed task: a box in a run nested under an ordered item",
  typeIntoNested("1. a\n   - b\n   - c", "b", "[ ] "),
  {
    markdown: "1. a\n   - [ ] b\n   - c",
    read: ["ol/item", "  ul/task:false", "  ul/item"],
    wrappers: [1, 1],
  },
);
// ...and the draft it leaves behind is one the composer can read back, which is
// what a failed send hands it.
for (const [draft, label] of [
  ["- a\n  - b\n  - c", "b"],
  ["- a\n  - b\n  - c\n  - d", "c"],
  ["1. a\n   - b\n   - c", "b"],
]) {
  const sent = typeIntoNested(draft, label, "[ ] ").markdown;
  check(`typed task: the nested draft ${JSON.stringify(draft)} reads back unchanged`, roundTrip(sent), sent);
}

// Typed and pasted have to agree, or the box shows one thing and sends another.
// GFM needs the dash, so a bare "[ ] " is prose on BOTH paths.
blank();
type("[ ] x");
check("agreement: a typed bare box is prose", blocks(), ["paragraph"]);
check("agreement: a typed bare box sends as typed", wire(), "[ ] x");
check("agreement: a pasted bare box is prose too", roundTrip("[ ] x"), "[ ] x");
check("agreement: and the reader reads it as prose", blockTypes(roundTrip("[ ] x")), ["paragraph"]);
blank();
type("- [ ] x");
check("agreement: a typed dashed box is a task", blocks(), ["check"]);
check("agreement: a typed dashed box sends as typed", wire(), "- [ ] x");
check("agreement: a pasted dashed box is a task too", roundTrip("- [ ] x"), "- [ ] x");

// Text formatting is swallowed, so a draft can never LOOK like something the
// wire cannot carry.
{
  seed("hello");
  live.update(
    () => {
      $getRoot().selectStart();
      const selection = $getSelection();
      selection.focus.set(selection.anchor.key, 5, "text");
    },
    { discrete: true },
  );
  live.dispatchCommand(FORMAT_TEXT_COMMAND, "bold");
  flush();
  let bold = null;
  live.getEditorState().read(() => {
    bold = $getRoot().getAllTextNodes()[0].hasFormat("bold");
  });
  check("format: bold is refused", bold, false);
}

// ---------------------------------------------------------------------------
// 4b. Undo lands on what was typed
// ---------------------------------------------------------------------------
// A conversion that shares its update with the keystroke that triggered it
// leaves no history entry holding the literal text: undo would land on a
// half-typed marker, and the marker itself would be untypeable.

/** Type `sequence` into an empty box, undo once, and report the blocks. */
function typeThenUndo(sequence) {
  blank();
  type(sequence);
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  return nodes();
}

check("undo: a bullet gives back the literal marker", typeThenUndo("- "), [["paragraph", "- "]]);
check("undo: an ordinal gives back the literal marker", typeThenUndo("1. "), [["paragraph", "1. "]]);
check("undo: a fence gives back the literal fence", typeThenUndo("```js "), [
  ["paragraph", "```js "],
]);
check("undo: a task gives back the literal box", typeThenUndo("- [ ] "), [["bullet", "[ ] "]]);
{
  blank();
  type("```js");
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true }));
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("undo: a fence opened on shift+enter gives back the text", nodes(), [
    ["paragraph", "```js"],
  ]);
}

// An undo that is immediately undone by the next keystroke is not an undo. The
// task conversion is offered only while the caret is still sitting right behind
// the box it completed; keyed on the item's TEXT alone it fires again on every
// later character, and a literal "[ ] " becomes untypeable.
//
// The draft reads "- [ ] x" either way, so the node KINDS are what is asserted:
// a bullet item holding literal text, not a task item.
{
  blank();
  type("- [ ] ");
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  type("x");
  check("undo: an undone box stays literal as typing carries on", nodes(), [["bullet", "[ ] x"]]);
  check("undo: and the draft carries the same string either way", wire(), "- [ ] x");
}
{
  // ...and the box is not lost for the rest of the item's life: put the caret
  // back behind the marker, retype the space, and the conversion is offered
  // again.
  blank();
  type("- [ ] ");
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  live.update(
    () => {
      const label = $getRoot().getFirstChild().getFirstChild().getFirstChild();
      label.setTextContent("[ ]");
      label.select(3, 3);
    },
    { discrete: true },
  );
  flush();
  check("undo: an incomplete marker is left alone", blocks(), ["bullet"]);
  type(" ");
  check("undo: retyping the space at the marker converts again", blocks(), ["check"]);
}

// ---------------------------------------------------------------------------
// 4c. Emptying a block gives the box back
// ---------------------------------------------------------------------------

/** Strip every child out of the item at `index` of the first block. */
function emptyItem(draft, index) {
  seed(draft);
  live.update(
    () => {
      const item = $getRoot().getFirstChild().getChildren()[index];
      for (const child of item.getChildren()) child.remove();
    },
    { discrete: true },
  );
  flush();
  return { blocks: blocks(), markdown: wire() };
}

check("stranded list: an emptied lone item becomes a paragraph", emptyItem("- one", 0), {
  blocks: ["paragraph"],
  markdown: "",
});
check("stranded list: an emptied lone task becomes a paragraph", emptyItem("- [ ] one", 0), {
  blocks: ["paragraph"],
  markdown: "",
});
check("stranded list: a list with other items is left alone", emptyItem("- one\n- two", 0), {
  blocks: ["bullet"],
  markdown: "- \n- two",
});
{
  // A list the user just OPENED by typing "- " looks exactly like an emptied
  // one. Collapsing it would make a bullet untypeable, so it must survive.
  blank();
  type("- ");
  check("stranded list: a freshly typed marker survives", blocks(), ["bullet"]);
  check("stranded list: and it is still a marker", wire(), "- ");
}

// Select-all then Delete is the one gesture that empties a CODE block without
// removing it: the deletion merges the whole draft into its first block, and
// when that block is a fence the shell is what is left. The shell is worse than
// a naked bullet - it serializes to a bare fence PAIR, so the draft reads as
// non-empty, the send button stays lit, and everything typed next is swallowed
// as literal code.

/** Select the whole draft and press Delete, as Ctrl+A then Delete does. */
function clearAll(draft) {
  seed(draft);
  live.dispatchCommand(SELECT_ALL_COMMAND, undefined);
  flush();
  live.dispatchCommand(KEY_DELETE_COMMAND, keyEvent({ target: null }));
  flush();
  return { nodes: nodes(), markdown: wire() };
}

check(
  "stranded code: a deleted code draft leaves one empty paragraph",
  clearAll("```js\nlet a = 1;\n```"),
  { nodes: [["paragraph", ""]], markdown: "" },
);
check(
  "stranded code: a deleted mixed draft leaves one empty paragraph",
  clearAll("hello\n\n- one\n\n```js\nlet a = 1;\n```"),
  { nodes: [["paragraph", ""]], markdown: "" },
);
// The fence FIRST, which is the arrangement that leaves the shell behind: the
// merge target is the draft's opening block, whatever kind it is.
check(
  "stranded code: a deleted code-first mixed draft leaves one empty paragraph",
  clearAll("```js\nlet a = 1;\n```\n\nhello\n\n- one"),
  { nodes: [["paragraph", ""]], markdown: "" },
);
// The control: the same gesture over a draft with no block in it was already
// right, and the collapse must not have changed what it leaves behind.
check("stranded code: a deleted prose draft is left as it was", clearAll("hello\n\nworld"), {
  nodes: [["paragraph", ""]],
  markdown: "",
});

{
  // A fence JUST opened is empty for the only innocent reason there is - it is
  // waiting for its first line. It is NEW in the update that made it, which is
  // what tells it apart from a shell, and collapsing it would make a code block
  // untypeable.
  blank();
  type("```js");
  press(KEY_ENTER_COMMAND, keyEvent({ shiftKey: true }));
  check("stranded code: a freshly opened fence survives", blocks(), ["code"]);
  check("stranded code: and it is still an empty block", wire(), "```js\n```");
}

{
  // The point of the collapse: what is typed next has to reach a paragraph. In
  // the shell it would have gone in as literal code, and no marker in it would
  // ever have converted, because the shortcuts only run on a paragraph.
  clearAll("```js\nlet a = 1;\n```");
  type("- ");
  check("stranded code: typing after a collapse lands in a paragraph", blocks(), ["bullet"]);
  check("stranded code: and the markdown shortcuts fire again", wire(), "- ");
}

// ...but a block cleared from INSIDE is being retyped, not abandoned. Selecting
// a block's lines and deleting them is the ordinary way to change what it holds,
// and collapsing there took the block and its language away mid-edit: retyping
// in place was impossible, and the language had to be typed again.

/** The language of every top-level code block in the live draft, in order. */
function languages() {
  let out = [];
  live.getEditorState().read(() => {
    out = $getRoot()
      .getChildren()
      .filter((node) => node.getType() === "code")
      .map((node) => node.getLanguage() ?? null);
  });
  return out;
}

/**
 * Select the whole text of the code block at `index` and press Delete.
 *
 * The selection is anchored in the block's OWN text leaves, which is what a
 * person dragging across its lines leaves behind - and what tells this apart
 * from the whole-draft selection that Ctrl+A makes.
 */
function emptyCodeFromInside(draft, index) {
  seed(draft);
  live.update(
    () => {
      const leaves = $getRoot()
        .getChildren()
        [index].getChildren()
        .filter((node) => node.getType() === "text");
      const selection = leaves[0].select(0, 0);
      const last = leaves[leaves.length - 1];
      selection.focus.set(last.getKey(), last.getTextContentSize(), "text");
    },
    { discrete: true },
  );
  flush();
  live.dispatchCommand(KEY_DELETE_COMMAND, keyEvent({ target: null }));
  flush();
  return { languages: languages(), nodes: nodes() };
}

check(
  "retyped code: clearing a block from inside keeps the block and its language",
  emptyCodeFromInside("```js\nlet a = 1;\n```\nhello", 0),
  { languages: ["js"], nodes: [["code", ""], ["paragraph", "hello"]] },
);
check(
  "retyped code: and what is typed next goes back into it",
  (() => {
    emptyCodeFromInside("```js\nlet a = 1;\n```\nhello", 0);
    type("let b = 2;");
    return { nodes: nodes(), wire: wire() };
  })(),
  {
    nodes: [["code", "let b = 2;"], ["paragraph", "hello"]],
    wire: "```js\nlet b = 2;\n```\nhello",
  },
);
check(
  "retyped code: a multi-line block cleared from inside survives too",
  emptyCodeFromInside("```py\ndef a():\n    pass\n```\nafter", 0),
  { languages: ["py"], nodes: [["code", ""], ["paragraph", "after"]] },
);
check(
  "retyped code: one block of two, cleared from inside, leaves the other alone",
  emptyCodeFromInside("```js\na\n```\n```py\nb\n```", 0),
  { languages: ["js", "py"], nodes: [["code", ""], ["code", "b"]] },
);
// The accepted edge, pinned rather than assumed: over a draft that is NOTHING
// but the block there is no selection outside it, so a manual sweep of its text
// and Ctrl+A leave the identical selection. It is resolved toward collapsing,
// which is what keeps the box typeable and the placeholder honest - and what
// the whole-draft delete relies on.
check(
  "retyped code: a SOLE block cleared from inside still gives the paragraph back",
  emptyCodeFromInside("```js\nlet a = 1;\n```", 0),
  { languages: [], nodes: [["paragraph", ""]] },
);
// A naked list marker has no language to lose and nothing to retype into, so
// the list collapse is untouched by any of the above.
check(
  "retyped code: a list emptied from inside still collapses",
  (() => {
    seed("- one\nafter");
    live.update(
      () => {
        const leaf = $getRoot().getFirstChild().getFirstChild().getFirstChild();
        leaf.select(0, leaf.getTextContentSize());
      },
      { discrete: true },
    );
    flush();
    live.dispatchCommand(KEY_DELETE_COMMAND, keyEvent({ target: null }));
    flush();
    return nodes();
  })(),
  [["paragraph", ""], ["paragraph", "after"]],
);

// ---------------------------------------------------------------------------
// 4c-2. Getting the caret back OUT of a code block
// ---------------------------------------------------------------------------
// A code block that ends the draft was a keyboard trap: Enter sends,
// Shift+Enter only piles newlines up inside, and ArrowDown had nothing below
// the block to move to. Two gestures open it, and NEITHER touches the newline
// key - blank lines inside a block still chain without limit, which is the
// flow that puts two of them between functions.

/**
 * Press ArrowDown, reporting what it did and whether it was handed back.
 *
 * The stand-in sits directly below the composer's own handler, exactly as the
 * drop battery's does, for two reasons: it is how "handed back" is OBSERVED,
 * and the rich-text base's own ArrowDown reaches for a root element that a
 * headless run does not have. `passedDown: 1` is native caret navigation,
 * untouched.
 */
function arrowDown() {
  let passedDown = 0;
  const spy = live.registerCommand(
    KEY_ARROW_DOWN_COMMAND,
    () => {
      passedDown += 1;
      return true;
    },
    COMMAND_PRIORITY_HIGH,
  );
  const event = keyEvent();
  live.dispatchCommand(KEY_ARROW_DOWN_COMMAND, event);
  flush();
  spy();
  return { caret: caret(), nodes: nodes(), passedDown, prevented: event.prevented };
}

/**
 * Put `stem` on a fresh last line of the trailing code block, then TYPE `typed`.
 *
 * The keystroke is what the gesture keys on, so the finished line can never be
 * planted whole: a helper that did that would walk past the typed-growth gate
 * and every guard behind it, and report green for a line that never fired.
 */
function typeOntoLastLine(draft, stem, typed) {
  seed(draft);
  live.update(
    () => {
      const code = $getRoot().getLastChild();
      code.append($createLineBreakNode(), $createTextNode(stem));
      code.getLastChild().select(stem.length, stem.length);
    },
    { discrete: true },
  );
  flush();
  type(typed);
  return { blocks: blocks(), nodes: nodes() };
}

// The gesture, end to end: the closer line is consumed, the block keeps what
// was above it, and the caret lands in a paragraph below.
blank();
type("```js");
shiftEnter();
type("code");
shiftEnter();
type("```");
check("typed closer: the closer is consumed and a paragraph takes the caret", nodes(), [
  ["code", "code"],
  ["paragraph", ""],
]);
check("typed closer: the caret is in the paragraph", caret(), [1, "paragraph"]);
check("typed closer: nothing of the closer is left in the draft", wire(), "```js\ncode\n```\n");
type("after");
check("typed closer: what is typed next is prose below the block", wire(), "```js\ncode\n```\nafter");
check("typed closer: and the reader sees code then a paragraph", blockTypes(wire()), [
  "code",
  "paragraph",
]);
check("typed closer: with no prose swallowed into the block", codeBlockValues(wire()), ["code"]);

// The block's ONLY content is the closer, so consuming it empties the block.
// The stranded-code collapse would then fire too, and a paragraph inserted
// BELOW the shell would leave the user two blank lines and a caret in the
// lower one. The block becomes the paragraph instead: exactly one, caret in it.
blank();
type("```js");
shiftEnter();
type("```");
check("typed closer: a closer that is the whole block leaves ONE paragraph", nodes(), [
  ["paragraph", ""],
]);
check("typed closer: with the caret in it", caret(), [0, "paragraph"]);
check("typed closer: and an empty draft", wire(), "");
type("- ");
check("typed closer: and the markdown shortcuts fire in it", blocks(), ["bullet"]);

// A closer typed under content keeps the block, so the paragraph goes after it.
seed("```js\nx\n```");
shiftEnter();
type("```");
check("typed closer: a block with content above the closer keeps it", nodes(), [
  ["code", "x"],
  ["paragraph", ""],
]);

// The closer works on ANY top-level block, not only the draft's last - putting
// the trailing fence in place means the same thing wherever the block sits.
// (ArrowDown deliberately asks for the last block: elsewhere it still has a
// caret to move to, and taking it would cost real navigation.)
seed("```js\nx\n```\nafter", { anchor: 1, leaf: 0 });
shiftEnter();
type("```");
check("typed closer: a block mid-draft closes too", nodes(), [
  ["code", "x"],
  ["paragraph", ""],
  ["paragraph", "after"],
]);
check("typed closer: with the caret between the two", caret(), [1, "paragraph"]);

// A fence with code UNDER it is content, and the serializer widens the block's
// own fence around it rather than letting it close anything.
seed("```js\na\nb\n```", { leaf: 0, anchor: 1 });
shiftEnter();
type("```");
check("typed closer: a fence mid-block is content", blocks(), ["code"]);
check("typed closer: and the block's own fence widens around it", wire(), "````js\na\n```\nb\n````");

// Exactly three backticks, alone on the line. The two below that a person can
// reach by typing - three backticks, and an indented run - are seeded exactly
// as the keystrokes leave them.
check("typed closer: three backticks close the block", typeOntoLastLine("```js\nx\n```", "``", "`"), {
  blocks: ["code", "paragraph"],
  nodes: [
    ["code", "x"],
    ["paragraph", ""],
  ],
});
check("typed closer: an indented fence does not", typeOntoLastLine("```js\nx\n```", " ``", "`"), {
  blocks: ["code"],
  nodes: [["code", "x\n ```"]],
});

// The three below are GROWTH-GATE tests, and their seeds say so: each PLANTS a
// whole "```" and grows it by one more character, which is the shape a paste, a
// drop or a restored draft leaves - a line that ARRIVED holding a fence. In
// natural typing none of them is reachable, because the third backtick closes
// the block before a fourth, a language or a space can be typed; what those
// keystrokes really do is measured directly further down.
check(
  "typed closer: a planted fence grown to four backticks does not close",
  typeOntoLastLine("```js\nx\n```", "```", "`"),
  { blocks: ["code"], nodes: [["code", "x\n````"]] },
);
check(
  "typed closer: a planted fence grown by a language letter does not close",
  typeOntoLastLine("```js\nx\n```", "```", "j"),
  { blocks: ["code"], nodes: [["code", "x\n```j"]] },
);
check(
  "typed closer: a planted fence grown by a trailing space does not close",
  typeOntoLastLine("```js\nx\n```", "```", " "),
  { blocks: ["code"], nodes: [["code", "x\n``` "]] },
);

/**
 * Open a fresh last line on the trailing code block and TYPE `typed` onto it.
 *
 * Nothing is planted: the line starts empty, so every character goes through
 * the gesture in the order a person produces it. That is the whole point -
 * seeding a line that already holds a fence measures a different path, and the
 * two disagree about what typing past the third backtick does.
 */
function typeOnFreshLastLine(draft, typed) {
  seed(draft);
  live.update(
    () => {
      const code = $getRoot().getLastChild();
      code.append($createLineBreakNode());
      code.selectEnd();
    },
    { discrete: true },
  );
  flush();
  type(typed);
  return { nodes: nodes(), wire: wire() };
}

// What typing those same characters ACTUALLY does: the block closes on the
// third backtick, and everything after it is typed into the paragraph the close
// just made. So there is no way to type a fence line in place as a block's last
// line - the accepted edge the gesture is documented with.
check("typed closer: typing ```js closes the block and ejects the language", typeOnFreshLastLine("```js\nx\n```", "```js"), {
  nodes: [
    ["code", "x"],
    ["paragraph", "js"],
  ],
  wire: "```js\nx\n```\njs",
});
check("typed closer: and the reader sees code then a paragraph", blockTypes(wire()), [
  "code",
  "paragraph",
]);
check("typed closer: with no prose swallowed into the block", codeBlockValues(wire()), ["x"]);
check("typed closer: typing a fourth backtick puts it below the block", typeOnFreshLastLine("```js\nx\n```", "````"), {
  nodes: [
    ["code", "x"],
    ["paragraph", "`"],
  ],
  wire: "```js\nx\n```\n`",
});
check("typed closer: typing a trailing space puts it below the block", typeOnFreshLastLine("```js\nx\n```", "``` "), {
  nodes: [
    ["code", "x"],
    ["paragraph", " "],
  ],
  wire: "```js\nx\n```\n ",
});
{
  // Recovery from a close nobody wanted is undo, one step per thing typed: the
  // ejected characters first, then the close. Two of them for "```js".
  typeOnFreshLastLine("```js\nx\n```", "```js");
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("undo: the first step takes back the ejected language", nodes(), [
    ["code", "x"],
    ["paragraph", ""],
  ]);
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("undo: the second gives the literal fence back inside the block", nodes(), [
    ["code", "x\n```"],
  ]);
}

// ...and a closer that merely APPEARS on the line - because it arrived whole,
// by paste or by drop - is code the user meant to keep. Only the keystroke that
// completes the fence fires the gesture.
check(
  "typed closer: a pasted closer stays content",
  paste("```js\nx\n```", "\n```"),
  "````js\nx\n```\n````",
);
check("typed closer: and its block survives the paste", blocks(), ["code"]);
check(
  "typed closer: a dropped closer stays content too",
  drop("```js\nx\n```", new FakeDataTransfer({ "text/plain": "\n```" })).wire,
  "````js\nx\n```\n````",
);

// The caret has to sit immediately behind the third backtick. Typed at the HEAD
// of the line the same three characters are a fence the user is building out,
// and firing there would eat a line mid-edit.
{
  seed("```js\nx\n```");
  live.update(
    () => {
      const code = $getRoot().getLastChild();
      code.append($createLineBreakNode(), $createTextNode("``"));
      code.getLastChild().select(0, 0);
    },
    { discrete: true },
  );
  flush();
  type("`");
  check("typed closer: completing the fence from its head leaves it content", nodes(), [
    ["code", "x\n```"],
  ]);
}
{
  // The fence is on the block's last line, but the keystroke is up in the
  // prose: a conversion must never reach back for a line the caret has left.
  seed("hello\n```js\nx\n```");
  live.update(
    () => {
      $getRoot().getLastChild().append($createLineBreakNode(), $createTextNode("```"));
      $getRoot().getFirstChild().selectEnd();
    },
    { discrete: true },
  );
  flush();
  type("`");
  check("typed closer: typing elsewhere never fires it retroactively", nodes(), [
    ["paragraph", "hello`"],
    ["code", "x\n```"],
  ]);
}
{
  // ...and the same with the fence itself completing while the caret is away.
  // The paragraph holding the caret is three characters long ON PURPOSE: the
  // caret is then at the offset the closer's own end sits at, so only asking
  // WHICH node the caret is in can tell the two apart.
  seed("abc\n```js\nx\n```");
  live.update(
    () => {
      $getRoot().getLastChild().append($createLineBreakNode(), $createTextNode("``"));
    },
    { discrete: true },
  );
  flush();
  live.update(
    () => {
      $getRoot().getLastChild().getLastChild().setTextContent("```");
      $getRoot().getFirstChild().selectEnd();
    },
    { discrete: true },
  );
  flush();
  check("typed closer: a fence completed with the caret elsewhere never fires", nodes(), [
    ["paragraph", "abc"],
    ["code", "x\n```"],
  ]);
}

// A block nested in a list item has no room below it for a paragraph - the
// caret would land in a paragraph inside a list item, which is a shape neither
// the serializer nor the reader's parser speaks.
{
  seed("- one\n  ```py\n  x\n  ```");
  live.update(
    () => {
      const code = $getRoot().getFirstChild().getFirstChild().getLastChild();
      code.append($createLineBreakNode(), $createTextNode("``"));
      code.getLastChild().select(2, 2);
    },
    { discrete: true },
  );
  flush();
  type("`");
  check("typed closer: a block nested in a list item is left alone", blocks(), ["bullet"]);
  check(
    "typed closer: and its fence line stays content",
    wire(),
    "- one\n  ````py\n  x\n  ```\n  ````",
  );
}

// One undo hands the literal fence back INSIDE the block, which only holds
// while the conversion is its own history step.
{
  blank();
  type("```js");
  shiftEnter();
  type("code");
  shiftEnter();
  type("```");
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("undo: a typed closer gives back the literal fence", nodes(), [["code", "code\n```"]]);
  type("x");
  check("undo: and an undone closer stays literal as typing carries on", nodes(), [
    ["code", "code\n```x"],
  ]);
}
{
  blank();
  type("```js");
  shiftEnter();
  type("```");
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("undo: a closer that emptied its block gives the block back", nodes(), [["code", "```"]]);
}

// ArrowDown, the other door. It was a DEAD key on the last line of a trailing
// block, so taking it there costs no behaviour that existed.
seed("```js\nx\n```");
check("arrowdown: exits a trailing block from its last line", arrowDown(), {
  caret: [1, "paragraph"],
  nodes: [
    ["code", "x"],
    ["paragraph", ""],
  ],
  passedDown: 0,
  prevented: true,
});
type("after");
check("arrowdown: what is typed next is prose below the block", wire(), "```js\nx\n```\nafter");
check("arrowdown: and the reader sees code then a paragraph", blockTypes(wire()), [
  "code",
  "paragraph",
]);

// The column within the last line is irrelevant - "down" reads the same from
// anywhere on it, and asking for the end would leave the key dead again.
for (const [where, at] of [
  ["its head", { anchor: 0, leaf: 1 }],
  ["mid-word", { anchor: 2, leaf: 1 }],
  ["its end", { anchor: 3, leaf: 1 }],
]) {
  seed("```js\nabc\ndef\n```", at);
  check(`arrowdown: exits from ${where} of the last line`, arrowDown().nodes, [
    ["code", "abc\ndef"],
    ["paragraph", ""],
  ]);
}

// Everything else is handed back, so native caret navigation is untouched.
seed("```js\nabc\ndef\n```", { anchor: 1, leaf: 0 });
check("arrowdown: a caret on an earlier line is left to native navigation", arrowDown(), {
  caret: [0, "code"],
  nodes: [["code", "abc\ndef"]],
  passedDown: 1,
  prevented: false,
});
seed("```js\nx\n```\nafter", { anchor: 1, leaf: 0 });
check("arrowdown: a block with content below it is left alone", arrowDown(), {
  caret: [0, "code"],
  nodes: [
    ["code", "x"],
    ["paragraph", "after"],
  ],
  passedDown: 1,
  prevented: false,
});
seed("hello");
check("arrowdown: prose is left alone", arrowDown(), {
  caret: [0, "paragraph"],
  nodes: [["paragraph", "hello"]],
  passedDown: 1,
  prevented: false,
});
seed("- one");
check("arrowdown: a list is left alone", arrowDown(), {
  caret: [0, "bullet"],
  nodes: [["bullet", "one"]],
  passedDown: 1,
  prevented: false,
});
seed("```js\nabc\n```", { anchor: 0, focus: 3, leaf: 0 });
check("arrowdown: a selection that is not a caret is left alone", arrowDown(), {
  caret: [0, "code"],
  nodes: [["code", "abc"]],
  passedDown: 1,
  prevented: false,
});

// Pressing it again from the new paragraph adds nothing: the gesture needs the
// caret INSIDE a block, and after the first press it is not.
{
  seed("```js\nx\n```");
  arrowDown();
  check("arrowdown: a second press adds no second paragraph", arrowDown(), {
    caret: [1, "paragraph"],
    nodes: [
      ["code", "x"],
      ["paragraph", ""],
    ],
    passedDown: 1,
    prevented: false,
  });
  check("arrowdown: nor does a third", arrowDown().nodes, [
    ["code", "x"],
    ["paragraph", ""],
  ]);
}

// An EMPTY block becomes the paragraph rather than gaining one below it, for
// the same reason the typed closer's only-content case does: the shell left
// behind is what the stranded-code collapse would turn into a second one.
{
  blank();
  type("```js");
  shiftEnter();
  check("arrowdown: an empty block becomes the paragraph it is left into", arrowDown(), {
    caret: [0, "paragraph"],
    nodes: [["paragraph", ""]],
    passedDown: 0,
    prevented: true,
  });
}
{
  // The accepted edge of that, pinned: the block is GONE and its opener's
  // language with it, whatever else the draft holds - leaving a block with
  // nothing in it asks for the line below, not for two empty blocks. One undo
  // hands the block and its language back.
  blank();
  type("hello");
  shiftEnter();
  type("```py");
  shiftEnter();
  check("arrowdown: an empty trailing block goes even beside other content", arrowDown().nodes, [
    ["paragraph", "hello"],
    ["paragraph", ""],
  ]);
  check("arrowdown: and its language went with it", languages(), []);
  live.dispatchCommand(UNDO_COMMAND, undefined);
  flush();
  check("arrowdown: one undo gives the block back", nodes(), [
    ["paragraph", "hello"],
    ["code", ""],
  ]);
  check("arrowdown: and its language with it", languages(), ["py"]);
}

// THE REGRESSION THAT MATTERS: Shift+Enter is unconditionally a newline inside
// a code block and chains without limit. Neither gesture is allowed near it -
// an escape built on a doubled Shift+Enter would collide with exactly this.
{
  blank();
  type("```py");
  shiftEnter();
  type("x");
  shiftEnter();
  shiftEnter();
  shiftEnter();
  shiftEnter();
  check("shift+enter: four in a row never leave the block", blocks(), ["code"]);
  check("shift+enter: they are four blank lines inside it", wire(), "```py\nx\n\n\n\n\n```");
  check("shift+enter: and the caret is still in the block", caret(), [0, "code"]);
  type("y");
  check("shift+enter: so what is typed next is still code", nodes(), [["code", "x\n\n\n\ny"]]);
}
{
  // The flow the rule exists for: two blank lines between functions.
  blank();
  type("```py");
  shiftEnter();
  type("def a():");
  shiftEnter();
  type("    pass");
  shiftEnter();
  shiftEnter();
  shiftEnter();
  type("def b():");
  check("shift+enter: two blank lines between functions stay in one block", blocks(), ["code"]);
  check(
    "shift+enter: with every line inside it",
    wire(),
    "```py\ndef a():\n    pass\n\n\ndef b():\n```",
  );
}
{
  // Enter still sends from the very position ArrowDown now claims.
  const before = sends;
  seed("```js\nx\n```");
  const after = press(KEY_ENTER_COMMAND, keyEvent());
  check("enter: still sends from the last line of a trailing block", sends - before, 1);
  check("enter: and leaves the draft alone there", after, "```js\nx\n```");
}

// ---------------------------------------------------------------------------
// 4d. Paste, at every caret, for every kind of payload
// ---------------------------------------------------------------------------
// The editor's own insertion MERGES the first pasted block into the block the
// caret sits in, which silently eats a fence and its language. Every cell here
// is the wire the paste has to leave behind.

/** Paste `text` into `draft` with the caret placed by `where`. */
function pasteAt(draft, where, text) {
  seed(draft, where);
  const event = new ClipboardEvent(new FakeDataTransfer({ "text/plain": text }));
  live.dispatchCommand(PASTE_COMMAND, event);
  flush();
  return wire();
}

/** Paste `text` with the caret at the end of `draft`. */
function paste(draft, text) {
  return pasteAt(draft, "end", text);
}

const PASTE_CARETS = {
  "an empty draft": "",
  "non-empty prose": "hello",
  "a list item": "- one",
  "a task item": "- [ ] todo",
  "a code block": "```\nx\n```",
};

const PASTE_MATRIX = {
  "": {
    list: "- a\n- b",
    code: "```py\nprint(1)\n```",
    mixed: "text\n- a\n```py\nz\n```",
    prose: "world",
  },
  // A payload that OPENS with prose merges into the line, the way a text box
  // would; only a payload that opens with a block splits it.
  hello: {
    list: "hello\n- a\n- b",
    code: "hello\n```py\nprint(1)\n```",
    mixed: "hellotext\n- a\n```py\nz\n```",
    prose: "helloworld",
  },
  "- one": {
    list: "- one\n- a\n- b",
    code: "- one\n```py\nprint(1)\n```",
    mixed: "- onetext\n- a\n```py\nz\n```",
    prose: "- oneworld",
  },
  "- [ ] todo": {
    list: "- [ ] todo\n- a\n- b",
    code: "- [ ] todo\n```py\nprint(1)\n```",
    mixed: "- [ ] todotext\n- a\n```py\nz\n```",
    prose: "- [ ] todoworld",
  },
  // Inside a code block everything is content, markers and fences alike - so a
  // pasted fence goes in as text and the block's OWN fence widens to hold it.
  "```\nx\n```": {
    list: "```\nx- a\n- b\n```",
    code: "````\nx```py\nprint(1)\n```\n````",
    mixed: "````\nxtext\n- a\n```py\nz\n```\n````",
    prose: "```\nxworld\n```",
  },
};

const PASTE_PAYLOADS = {
  code: "```py\nprint(1)\n```",
  list: "- a\n- b",
  mixed: "text\n- a\n```py\nz\n```",
  prose: "world",
};

for (const [where, draft] of Object.entries(PASTE_CARETS)) {
  for (const [what, text] of Object.entries(PASTE_PAYLOADS)) {
    check(`paste: ${what} into ${where}`, paste(draft, text), PASTE_MATRIX[draft][what]);
  }
}

// The fence has to survive as a fence, whatever it was pasted into - that is
// the failure the matrix above exists to stop, stated once in the reader's
// terms.
for (const [where, draft] of Object.entries(PASTE_CARETS)) {
  if (draft.startsWith("```")) continue;
  check(
    `paste: a fence pasted into ${where} still reads as code`,
    blockTypes(paste(draft, "```py\nprint(1)\n```")).includes("code"),
    true,
  );
}

// ---------------------------------------------------------------------------
// 4d-2. The same payloads, at every caret in the line
// ---------------------------------------------------------------------------
// The matrix above always pastes at the END of the draft, which is the one
// position where the editor's own insertion happens to behave. At any other
// caret a code-leading payload used to SWALLOW the far side of the line into
// the block ("abcdef" + a fence, caret at 3, came out as "abc" then a block
// holding "ZZdef"), and replacing a selection left the empty half of the split
// behind as a blank first line. Both are corruption of the draft, not a
// rendering quirk, so every caret gets a cell.
//
// The rule the code cells state: a fenced block NEVER absorbs prose. It goes
// before the line, after it, or between the two halves of a line split at the
// caret - and a caret anywhere in a list item puts it after the whole list,
// because splitting inside an item makes another ITEM, not another block.
//
// The rule the list cells state: a pasted LIST still merges into the line it
// lands in, which is what continuing a list by pasting into it should do and
// what a plain text box would have produced.

const CARET_DRAFTS = {
  "an ordered item": "1. abcdef",
  "a bullet item": "- abcdef",
  "a task item": "- [ ] abcdef",
  prose: "abcdef",
};

const CARETS = {
  "part of a word selected": { anchor: 0, focus: 3 },
  "the end of the draft": "end",
  "the head of the line": { anchor: 0 },
  "the middle of a word": { anchor: 3 },
  "the whole draft selected": "all",
};

const CARET_CODE = "```py\nZZ\n```";
const CARET_LIST = "- a\n- b";

// [what a code payload leaves, what a list payload leaves]
const CARET_MATRIX = {
  prose: {
    "the head of the line": ["```py\nZZ\n```\nabcdef", "- a\n- b\nabcdef"],
    "the middle of a word": ["abc\n```py\nZZ\n```\ndef", "abc\n- a\n- b\ndef"],
    "the end of the draft": ["abcdef\n```py\nZZ\n```", "abcdef\n- a\n- b"],
    "part of a word selected": ["```py\nZZ\n```\ndef", "- a\n- b\ndef"],
    "the whole draft selected": ["```py\nZZ\n```", "- a\n- b"],
  },
  "a bullet item": {
    "the head of the line": ["- abcdef\n```py\nZZ\n```", "- a\n- babcdef"],
    "the middle of a word": ["- abcdef\n```py\nZZ\n```", "- abc\n- a\n- bdef"],
    "the end of the draft": ["- abcdef\n```py\nZZ\n```", "- abcdef\n- a\n- b"],
    "part of a word selected": ["- def\n```py\nZZ\n```", "- a\n- bdef"],
    "the whole draft selected": ["```py\nZZ\n```", "- a\n- b"],
  },
  "an ordered item": {
    "the head of the line": ["1. abcdef\n```py\nZZ\n```", "- a\n- babcdef"],
    "the middle of a word": ["1. abcdef\n```py\nZZ\n```", "1. abc\n- a\n- bdef"],
    "the end of the draft": ["1. abcdef\n```py\nZZ\n```", "1. abcdef\n- a\n- b"],
    "part of a word selected": ["1. def\n```py\nZZ\n```", "- a\n- bdef"],
    "the whole draft selected": ["```py\nZZ\n```", "- a\n- b"],
  },
  "a task item": {
    "the head of the line": ["- [ ] abcdef\n```py\nZZ\n```", "- a\n- babcdef"],
    "the middle of a word": ["- [ ] abcdef\n```py\nZZ\n```", "- [ ] abc\n- a\n- bdef"],
    "the end of the draft": ["- [ ] abcdef\n```py\nZZ\n```", "- [ ] abcdef\n- a\n- b"],
    "part of a word selected": ["- [ ] def\n```py\nZZ\n```", "- a\n- bdef"],
    "the whole draft selected": ["```py\nZZ\n```", "- a\n- b"],
  },
};

for (const [line, draft] of Object.entries(CARET_DRAFTS)) {
  for (const [at, where] of Object.entries(CARETS)) {
    const [byCode, byList] = CARET_MATRIX[line][at];
    check(`paste: a fence into ${line} at ${at}`, pasteAt(draft, where, CARET_CODE), byCode);
    check(`paste: a list into ${line} at ${at}`, pasteAt(draft, where, CARET_LIST), byList);
  }
}

/** What the answer-side parser finds inside every code block of `markdown`. */
function codeBlockValues(markdown) {
  const found = [];
  const walk = (node) => {
    if (node.type === "code") found.push(node.value);
    for (const child of node.children ?? []) walk(child);
  };
  walk(parseGfm(markdown));
  return found;
}

// Stated once more in the READER's terms, since it is the whole point: whatever
// the caret was doing, the block reaches them as one fenced block holding
// exactly what was pasted. The corruption this replaces was invisible in the
// draft string until it was read - "ZZdef" inside the fence looks like code
// that was always there.
for (const [line, draft] of Object.entries(CARET_DRAFTS)) {
  for (const [at, where] of Object.entries(CARETS)) {
    const sent = pasteAt(draft, where, CARET_CODE);
    check(`paste: the fence into ${line} at ${at} still reads as code`, blockTypes(sent).includes("code"), true);
    check(`paste: and it swallowed no prose, into ${line} at ${at}`, codeBlockValues(sent), ["ZZ"]);
  }
}

// A marker the user has just opened and not filled in yet. Its whole LIST is
// empty, which is what made the round-1 gate read it as an empty block and let
// the fence merge into the marker.
for (const [marker, kind] of [
  ["- ", "bullet"],
  ["- [ ] ", "check"],
  ["1. ", "number"],
]) {
  check(`paste: a bare ${kind} marker gives way to a pasted fence`, paste(marker, CARET_CODE), CARET_CODE);
  check(`paste: and nothing is left of the marker`, blocks(), ["code"]);
  check(`paste: a bare ${kind} marker gives way to a pasted list`, paste(marker, CARET_LIST), CARET_LIST);
}

check("paste: a CRLF list arrives as a list", paste("", "- a\r\n- b"), "- a\n- b");
check("paste: a CRLF fence arrives as code", paste("", "```py\r\nz\r\n```"), "```py\nz\n```");
check("paste: no carriage return reaches the draft", paste("hello", " a\r\nb").includes("\r"), false);

{
  // The page gets first refusal, and what it takes never reaches the box.
  seed("hello");
  intercept = () => true;
  const event = new ClipboardEvent(new FakeDataTransfer({ "text/plain": "- a\n- b" }));
  const handled = live.dispatchCommand(PASTE_COMMAND, event);
  flush();
  intercept = null;
  check("paste: an intercepted paste is consumed", handled, true);
  check("paste: an intercepted paste never reaches the box", wire(), "hello");
  check("paste: an intercepted paste is not left to the browser", event.defaultPrevented, true);
}
{
  // The HTML flavour is the one door formatting the composer cannot send would
  // walk through, so it is never opened.
  seed("");
  const transfer = new FakeDataTransfer({
    "text/html": "<b>bold</b>",
    "text/plain": "bold",
  });
  live.dispatchCommand(PASTE_COMMAND, new ClipboardEvent(transfer));
  flush();
  check("paste: the html flavour is never read", transfer.asked.includes("text/html"), false);
  check("paste: the plain flavour is what lands", wire(), "bold");
}
/**
 * Paste `transfer` and report whether it reached the handler BELOW the composer.
 *
 * The mirror of `dropPastComposer`, and for the same reason: handing a paste
 * back reaches the base editor's own importer, which needs a DOM parser this
 * run has none of - so a stand-in sits directly under the composer's handler
 * and records the handing back without the base path ever running.
 */
function pastePastComposer(draft, transfer) {
  let passedDown = 0;
  const spy = live.registerCommand(
    PASTE_COMMAND,
    () => {
      passedDown += 1;
      return true;
    },
    COMMAND_PRIORITY_HIGH,
  );
  seed(draft);
  const event = new ClipboardEvent(transfer);
  const handled = live.dispatchCommand(PASTE_COMMAND, event);
  flush();
  spy();
  return { handled, nodes: nodes(), passedDown, prevented: event.defaultPrevented, wire: wire() };
}

{
  // A clipboard carrying ONLY html has no plain flavour to insert, and handing
  // it back is not the harmless no-op it looks like: the base importer DOES
  // read text/html and builds formatted nodes out of it, so the box would show
  // bold over a wire carrying plain text - the one lie this module exists to
  // stop. It is swallowed instead, and its html is still never read.
  const transfer = new FakeDataTransfer({ "text/html": "<b>bold</b>" });
  const result = pastePastComposer("hello", transfer);
  check("paste: an html-only clipboard is not passed down", result.passedDown, 0);
  check("paste: an html-only clipboard is swallowed", result.handled, true);
  check("paste: an html-only clipboard is not left to the browser", result.prevented, true);
  check("paste: an html-only clipboard inserts nothing", result.nodes, [["paragraph", "hello"]]);
  check("paste: and its html flavour is never read", transfer.asked.includes("text/html"), false);
}
{
  // ...and the swallow sits BELOW the page's attachment path, which is what a
  // screenshot paste depends on: an image on the clipboard carries no
  // text/plain at all, so a swallow placed above it would drop the file.
  let sawFiles = 0;
  intercept = (clipboard) => {
    if (clipboard.files.length === 0) return false;
    sawFiles += 1;
    return true;
  };
  const transfer = new FakeDataTransfer({ "text/html": "<img>" }, [{ name: "shot.png" }]);
  const result = pastePastComposer("hello", transfer);
  intercept = null;
  check("paste: a file clipboard reaches the page's attachment path first", sawFiles, 1);
  check("paste: and the page's answer is what consumes it", result.handled, true);
  check("paste: with nothing typed into the box", result.wire, "hello");
}
{
  // No caret, nothing to place: the paste is left to the base editor rather
  // than swallowed, which would lose it outright.
  seed("hello");
  live.update(
    () => {
      $setSelection(null);
    },
    { discrete: true },
  );
  const event = new ClipboardEvent(new FakeDataTransfer({ "text/plain": "- a" }));
  const handled = live.dispatchCommand(PASTE_COMMAND, event);
  flush();
  check("paste: a paste with no caret is not swallowed", handled, false);
  check("paste: a paste with no caret is left to the browser", event.defaultPrevented, false);
}

// ---------------------------------------------------------------------------
// 4e. Drop is a paste that arrived by mouse
// ---------------------------------------------------------------------------

/** Drop `transfer` onto the live editor holding `draft`. */
function drop(draft, transfer) {
  seed(draft);
  const event = new DragEvent(transfer);
  const handled = live.dispatchCommand(DROP_COMMAND, event);
  flush();
  return { handled, prevented: event.defaultPrevented, wire: wire() };
}

check(
  "drop: dropped text is read as the composer's constructs",
  drop("", new FakeDataTransfer({ "text/plain": "- a\n- b" })),
  { handled: true, prevented: true, wire: "- a\n- b" },
);
{
  const transfer = new FakeDataTransfer({
    "text/html": "<b>bold</b>",
    "text/plain": "bold",
  });
  const result = drop("", transfer);
  check("drop: the html flavour is never read", transfer.asked.includes("text/html"), false);
  check("drop: the plain flavour is what lands", result.wire, "bold");
}
check("drop: a CRLF payload is normalized", drop("", new FakeDataTransfer({ "text/plain": "- a\r\n- b" })).wire, "- a\n- b");
{
  intercept = () => true;
  const result = drop("hello", new FakeDataTransfer({ "text/plain": "- a" }));
  intercept = null;
  check("drop: the page's cap applies to dropped text too", result, {
    handled: true,
    prevented: true,
    wire: "hello",
  });
}
{
  const result = drop("hello", new FakeDataTransfer({ "text/plain": "ignored" }, [{ name: "a.png" }]));
  check("drop: a dropped file never becomes text", result, {
    handled: true,
    prevented: true,
    wire: "hello",
  });
}

// A drop lands where the POINTER is, and dropping into the middle of a word is
// the ordinary drop rather than an edge case - so the caret matrix runs through
// this door too, and has to come out identical to the paste door's.
//
// The editor asks the DOM where the pointer was, which a headless run has no
// answer for; the stand-in below hands back the node and offset the caret was
// already placed at, keyed the way the editor reads a DOM node. A drop always
// COLLAPSES to the point under the pointer, so the two selection rows of the
// matrix belong to the paste door alone.

/** A stand-in for the DOM node under the pointer, keyed as the editor reads it. */
function domStandIn(key) {
  return { [`__lexicalKey_${live.getKey()}`]: key, parentElement: null };
}

/** Drop `text` into `draft` at the position `where` names. */
function dropAt(draft, where, text) {
  seed(draft, where);
  let hit = null;
  live.getEditorState().read(() => {
    const selection = $getSelection();
    hit = { key: selection.anchor.key, offset: selection.anchor.offset };
  });
  const previous = globalThis.document;
  globalThis.document = {
    caretRangeFromPoint: () => ({
      startContainer: domStandIn(hit.key),
      startOffset: hit.offset,
    }),
  };
  live.dispatchCommand(DROP_COMMAND, new DragEvent(new FakeDataTransfer({ "text/plain": text })));
  flush();
  globalThis.document = previous;
  return wire();
}

const DROP_CARETS = ["the head of the line", "the middle of a word", "the end of the draft"];
for (const [line, draft] of Object.entries(CARET_DRAFTS)) {
  for (const at of DROP_CARETS) {
    const [byCode, byList] = CARET_MATRIX[line][at];
    check(`drop: a fence into ${line} at ${at}`, dropAt(draft, CARETS[at], CARET_CODE), byCode);
    check(`drop: a list into ${line} at ${at}`, dropAt(draft, CARETS[at], CARET_LIST), byList);
  }
}
for (const [marker, kind] of [
  ["- ", "bullet"],
  ["- [ ] ", "check"],
  ["1. ", "number"],
]) {
  // The caret is already in the empty item, so the plain door reaches it.
  check(
    `drop: a bare ${kind} marker gives way to a dropped fence`,
    drop(marker, new FakeDataTransfer({ "text/plain": CARET_CODE })).wire,
    CARET_CODE,
  );
}

// A drag that STARTED in this box is a MOVE, not a copy. Taking it over here
// would insert the text and leave the original where it was, because only the
// base editor's own drop path removes the source range - so that one case is
// handed back untouched, and the base does the move.
//
// What is checkable without a browser is the handing back. The base path itself
// needs a pointer and a real root element, so a stand-in listener sits directly
// below the composer's own handler and records that the drop reached it; the
// real move is verified in a browser.
/**
 * Drop `transfer` and report whether it reached the handler BELOW the composer.
 *
 * The stand-in sits between the composer's handler and the base editor's, so it
 * records the handing back without the base path - which needs a pointer and a
 * real root element - ever running.
 */
function dropPastComposer(draft, transfer) {
  let passedDown = 0;
  const spy = live.registerCommand(
    DROP_COMMAND,
    () => {
      passedDown += 1;
      return true;
    },
    COMMAND_PRIORITY_HIGH,
  );
  seed(draft);
  const event = new DragEvent(transfer);
  const handled = live.dispatchCommand(DROP_COMMAND, event);
  flush();
  spy();
  return { handled, passedDown, prevented: event.defaultPrevented, wire: wire() };
}

{
  const transfer = new FakeDataTransfer({
    "application/x-lexical-drag": JSON.stringify({ editorKey: live.getKey() }),
    "text/html": "<b>alpha </b>",
    "text/plain": "alpha ",
  });
  const result = dropPastComposer("alpha beta gamma", transfer);
  check("drop: a drag from this box is passed down to the base editor", result.passedDown, 1);
  check("drop: and the composer inserts nothing of its own", result.wire, "alpha beta gamma");
  check("drop: and does not consume the event", result.prevented, false);
  check("drop: and never reads the html flavour on the way past", transfer.asked.includes("text/html"), false);
}
{
  // A drag from a DIFFERENT editor keeps the intercepted plain-text path, where
  // the page's cap applies and the html flavour stays shut. Passing THIS one
  // down would hand a foreign payload to an importer that does read html.
  const transfer = new FakeDataTransfer({
    "application/x-lexical-drag": JSON.stringify({ editorKey: "someone-else" }),
    "text/html": "<b>bold</b>",
    "text/plain": "- a\n- b",
  });
  const result = dropPastComposer("", transfer);
  check("drop: a drag from another editor is not passed down", result.passedDown, 0);
  check("drop: a drag from another editor is still intercepted", result.handled, true);
  check("drop: and it arrives as the composer's own constructs", result.wire, "- a\n- b");
  check("drop: and its html flavour is never read", transfer.asked.includes("text/html"), false);
}
{
  // A drop carrying only html goes the same way a paste carrying only html
  // does, and for the same reason - the base importer reads that html. Files
  // and a drag out of this box are both answered ABOVE this, so neither is
  // caught by it.
  const transfer = new FakeDataTransfer({ "text/html": "<b>bold</b>" });
  const result = dropPastComposer("hello", transfer);
  check("drop: an html-only payload is not passed down", result.passedDown, 0);
  check("drop: an html-only payload is swallowed", result.handled, true);
  check("drop: an html-only payload is not left to the browser", result.prevented, true);
  check("drop: an html-only payload inserts nothing", result.wire, "hello");
  check("drop: and its html flavour is never read", transfer.asked.includes("text/html"), false);
}
{
  // The file door stays above the swallow: a dropped file is the page's, and
  // the drop keeps bubbling to the handler on the box around the editor.
  const transfer = new FakeDataTransfer({ "text/html": "<img>" }, [{ name: "shot.png" }]);
  const result = dropPastComposer("hello", transfer);
  check("drop: a file drop is answered above the html swallow", result.passedDown, 0);
  check("drop: a file drop reads no flavour at all", transfer.asked, []);
  check("drop: and nothing is typed into the box", result.wire, "hello");
}
{
  // A malformed marker is not a same-editor drag: it must not become a hole
  // through which the base editor's html path is reached.
  const transfer = new FakeDataTransfer({
    "application/x-lexical-drag": "not json",
    "text/html": "<b>bold</b>",
    "text/plain": "plain",
  });
  const result = dropPastComposer("", transfer);
  check("drop: a malformed drag marker is not passed down", result.passedDown, 0);
  check("drop: a malformed drag marker falls back to the plain path", result.handled, true);
  check("drop: and the plain flavour is what lands", result.wire, "plain");
}

// ---------------------------------------------------------------------------
// 4f. A fence whose body is nothing but blank lines
// ---------------------------------------------------------------------------
// A failed send hands the draft back through setMarkdown, so what the editor
// makes of a blank-bodied block decides whether the draft survives the trip.
// Two measured facts fix the answer, and neither is a choice this file made:
//
//  - The editor cannot hold a body of exactly ONE blank line. A block's lines
//    are its text split on newlines, so n blank lines are n-1 line breaks, and
//    n = 1 would be a block with no children - which IS the empty block.
//    Appending an empty text node to invent the difference does not work: the
//    library drops it (measured, children size 0 immediately after).
//  - The reader does not draw the distinction either: remark-gfm parses
//    "```\n```" and "```\n\n```" into the identical code node, value "".
//
// So one blank line canonicalizes to the empty pair - losing nothing the reader
// would have seen - and two or more survive byte for byte, which is all the
// serializer can WRITE and therefore all a restored draft can carry.

stable("code: a body of two blank lines survives", "```js\n\n\n```");
stable("code: a body of three blank lines survives", "```js\n\n\n\n```");
{
  // Through the editor the browser runs, not through the conversions alone:
  // this is the path setMarkdown takes on a failed-send restore.
  seed("```js\n\n\n```");
  check("restore: a two-blank-line body comes back unchanged", wire(), "```js\n\n\n```");
  check("restore: and it really is a block with a break in it", nodes(), [["code", "\n"]]);
  seed("abc\n```js\n\n\n```");
  check("restore: ...and beside other content too", wire(), "abc\n```js\n\n\n```");
}
check(
  "code: a body of ONE blank line canonicalizes to the empty pair",
  roundTrip("```js\n\n```"),
  "```js\n```",
);
check(
  "code: which the reader reads as the same block either way",
  [codeBlockValues("```js\n\n```"), codeBlockValues("```js\n```")],
  [[""], [""]],
);
{
  seed("```js\n\n```");
  check("restore: a one-blank-line body restores as the empty block", nodes(), [["code", ""]]);
  check("restore: and reports the canonical pair", wire(), "```js\n```");
}

teardown();

// ---------------------------------------------------------------------------
// 5. Send-time hygiene
// ---------------------------------------------------------------------------
// A marker the editor offered and nobody filled in is not a message: trimming
// alone leaves "-", which sends perfectly and says nothing.

const { stripTrailingBareMarker } = await import("../lib/composer-lists.ts");

check("hygiene: a trailing bare bullet is dropped", stripTrailingBareMarker("- one\n- "), "- one\n");
check("hygiene: a trailing bare ordinal is dropped", stripTrailingBareMarker("1. one\n2. "), "1. one\n");
check("hygiene: a trailing bare task is dropped", stripTrailingBareMarker("- [ ] "), "");
check("hygiene: a lone marker sends nothing", stripTrailingBareMarker("- ").trim(), "");
check("hygiene: a filled item is kept", stripTrailingBareMarker("- one"), "- one");
check("hygiene: the last line WITH content is judged", stripTrailingBareMarker("- one\n- \n\n"), "- one\n");
check("hygiene: prose is untouched", stripTrailingBareMarker("-item"), "-item");

// The fence guard reads drafts the SERIALIZER writes, so the cases that matter
// are the ones it can actually produce: fences are always closed, and a block
// whose content holds backticks is written with a longer run.
check(
  "hygiene: a marker after a closed block is still a marker",
  stripTrailingBareMarker(roundTrip("```\nx\n```\n- one\n- ")),
  "```\nx\n```\n- one\n",
);
check(
  "hygiene: a widened fence closes, so the marker after it is dropped",
  stripTrailingBareMarker(roundTrip("````\n```\n````\n- ")),
  "````\n```\n````\n",
);
check(
  "hygiene: a marker inside a widened block is content",
  stripTrailingBareMarker("````\n```\n- "),
  "````\n```\n- ",
);
check(
  "hygiene: a fence naming a language never closes a block",
  stripTrailingBareMarker("```\n```js\n- "),
  "```\n```js\n- ",
);

// An abandoned code block is the same offer in fence form: the editor keeps an
// empty block alive to be typed into, so a draft can end on its bare pair -
// which must not light the Send button the way it once did.
check("hygiene: a bare fence pair sends nothing", stripTrailingBareMarker("```js\n```"), "");
check(
  "hygiene: a bare fence pair after prose is dropped",
  stripTrailingBareMarker("abc\n```js\n```"),
  "abc\n",
);
check("hygiene: a widened bare pair is dropped too", stripTrailingBareMarker("````\n````"), "");
check(
  "hygiene: a fence holding code is a message",
  stripTrailingBareMarker("```js\ncode\n```"),
  "```js\ncode\n```",
);
// A block holding blank lines HAS been typed into, so it is a message. Judged
// on the draft the editor can actually produce - two blank lines, which is what
// a Shift+Enter inside a fresh block leaves - because a one-blank-line body
// never reaches here: it canonicalizes to the bare pair on the way in (see 4f),
// and the bare pair is what the check above already covers.
check(
  "hygiene: a fence holding blank lines is a message",
  stripTrailingBareMarker("```js\n\n\n```"),
  "```js\n\n\n```",
);
check(
  "hygiene: ...and one after prose is kept too",
  stripTrailingBareMarker("abc\n```js\n\n\n```"),
  "abc\n```js\n\n\n```",
);
check(
  "hygiene: a marker stranded above a dropped pair goes with it",
  stripTrailingBareMarker("- \n```js\n```"),
  "",
);
check(
  "hygiene: a bare pair inside a wider block is content",
  stripTrailingBareMarker("`````\n```js\n```"),
  "`````\n```js\n```",
);

// ---------------------------------------------------------------------------
// 6. The transformer restriction, checked where it is declared
// ---------------------------------------------------------------------------

check("transformers: exactly three are registered", COMPOSER_TRANSFORMERS.length, 3);
check(
  "transformers: they are the list and code ones",
  COMPOSER_TRANSFORMERS.every((transformer) =>
    [UNORDERED_LIST, ORDERED_LIST, CODE].includes(transformer),
  ),
  true,
);
check(
  "transformers: the bare-checkbox one is NOT registered",
  COMPOSER_TRANSFORMERS.includes(CHECK_LIST),
  false,
);
check(
  "transformers: no text formatting is registered",
  COMPOSER_TRANSFORMERS.some((transformer) => TEXT_FORMAT_TRANSFORMERS.includes(transformer)),
  false,
);
check(
  "transformers: none is a text-format or text-match kind",
  COMPOSER_TRANSFORMERS.some(
    (transformer) => transformer.type === "text-format" || transformer.type === "text-match",
  ),
  false,
);

console.log(
  failures === 0
    ? `composer fidelity: ${checks} checks passed`
    : `composer fidelity: ${failures} of ${checks} checks FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
