/**
 * Battery for math in answers: the delimiter rewrite (`lib/math-delimiters.ts`)
 * and the real parser chain it feeds (remark-math, KaTeX, the citation walk).
 *
 * Two layers. The rewrite is run directly on strings, including everything it
 * must leave alone (code, currency, unclosed or escaped delimiters, citation
 * markers). Then answers are pushed through the same plugin chain the renderer
 * uses and the resulting tree is inspected: KaTeX markup must come out for real
 * formulas, nothing may come out for currency, and a `[1]` inside a formula
 * must not become a citation chip while one after it still does.
 *
 *     npm test        (or: node scripts/math-rendering.mjs)
 *
 * Needs a Node that strips TypeScript types on its own - 22.18 or newer.
 */

import { readFileSync } from "node:fs";
import { register } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import remarkParse from "remark-parse";
import remarkRehype from "remark-rehype";
import { unified } from "unified";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `math rendering: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const { normalizeMathDelimiters: norm } = await import("../lib/math-delimiters.ts");
const { answerRehypePlugins, answerRemarkPlugins } = await import("../lib/math-markdown.ts");

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (rel) => readFileSync(join(root, rel), "utf8");

let passed = 0;
const failures = [];
function check(name, ok) {
  if (ok) passed += 1;
  else failures.push(name);
}

// ------------------------------------------------------- the delimiter rewrite
check(
  "rewrite: a display block on its own lines becomes a $$ fence",
  norm("A\n\\[\n\\Delta = 1\n\\]\nB") === "A\n$$\n\\Delta = 1\n$$\nB",
);
check("rewrite: one-line display on its own line is still a fence", norm("\\[x^2\\]") === "$$\nx^2\n$$");
check("rewrite: inline form is $$x$$", norm("so \\(a+b\\) holds") === "so $$a+b$$ holds");
check(
  "rewrite: display mid-sentence falls back to the inline form",
  norm("where \\[x\\] is") === "where $$x$$ is",
);
check("rewrite: a bare $$ block is left alone", norm("$$\ny\n$$") === "$$\ny\n$$");
check("rewrite: a model-written $$x$$ lifts to a block", norm("$$x$$ and\n$$\ny\n$$") === "$$\nx\n$$\n\nand\n$$\ny\n$$");
check(
  "rewrite: display inside a list item keeps its indent",
  norm("- item\n  \\[\n  a\n  \\]") === "- item\n  $$\n  a\n  $$",
);

// Currency
for (const text of [
  "it costs $5 and $10",
  "250 USD for the pair, then $20 more",
  "$5 \\(x\\) $10".replace("\\(x\\)", "plain"),
]) {
  check(`currency: "${text}" is untouched`, norm(text) === text);
}
check("currency: a \\( next to a price only rewrites the formula", norm("$5 or \\(x\\) or $10") === "$5 or $$x$$ or $10");

// Code
const fenced = "```\n\\[x\\]\n\\(y\\)\n```";
check("code: a backtick fence is untouched", norm(fenced) === fenced);
// A blank line inside the fence breaks the paragraph the inline-span rule works
// in, so only the fence rule itself can protect this one.
const gapped = "```tex\n\\[x\\]\n\n\\(y\\)\n```";
check("code: a fence with a blank line inside is untouched", norm(gapped) === gapped);
const tilde = "~~~tex\n\\(y\\)\n~~~";
check("code: a tilde fence is untouched", norm(tilde) === tilde);
const openFence = "```\n\\[x\\]";
check("code: an unclosed (streaming) fence stays code to the end", norm(openFence) === openFence);
check("code: math after a closed fence is rewritten", norm(fenced + "\n\\(z\\)") === fenced + "\n$$z$$");
check("code: an inline code span is untouched", norm("use `\\(x\\)` here") === "use `\\(x\\)` here");
check("code: a double-backtick span is untouched", norm("use ``a \\[x\\] b`` here") === "use ``a \\[x\\] b`` here");
check("code: math beside a code span still converts", norm("`a` \\(x\\) `b`") === "`a` $$x$$ `b`");
check(
  "code: a lone backtick is literal, not a span opener",
  norm("a ` b \\(x\\)") === "a ` b $$x$$",
);

// Streaming, escapes, boundaries, citations
check("stream: an unclosed \\[ is untouched", norm("so \\[ \\Delta = 1") === "so \\[ \\Delta = 1");
check("stream: an unclosed \\( is untouched", norm("so \\(x + ") === "so \\(x + ");
check("stream: a half-arrived closer is untouched", norm("\\[x\\") === "\\[x\\");
check("escape: \\\\[ is a literal backslash then a bracket", norm("a \\\\[x\\\\] b") === "a \\\\[x\\\\] b");
check(
  "boundary: a formula never spans a blank line",
  norm("\\(a\n\nb\\)") === "\\(a\n\nb\\)",
);
check("boundary: an empty body is untouched", norm("\\(  \\)") === "\\(  \\)");
check("citations: [1] and [2] are untouched", norm("fact [1] and more [2].") === "fact [1] and more [2].");
check(
  "citations: markers survive beside math",
  norm("see [1] \\(x\\) then [2]") === "see [1] $$x$$ then [2]",
);

// -------------------------------------------------- the real parser chain
function parse(markdown, citations = []) {
  const processor = unified()
    .use(remarkParse)
    .use(answerRemarkPlugins)
    .use(remarkRehype, { allowDangerousHtml: true })
    .use(answerRehypePlugins({ citationIndexes: new Set(citations), live: false }));
  const tree = processor.parse(norm(markdown));
  return processor.runSync(tree);
}

function classesOf(node) {
  const raw = node.properties?.className;
  return Array.isArray(raw) ? raw.map(String) : [];
}
function walk(node, visit) {
  visit(node);
  for (const child of node.children ?? []) walk(child, visit);
}
function findAll(tree, predicate) {
  const hits = [];
  walk(tree, (node) => {
    if (node.type === "element" && predicate(node)) hits.push(node);
  });
  return hits;
}
const hasClass = (name) => (node) => classesOf(node).includes(name);
function textOf(node) {
  let out = "";
  walk(node, (n) => {
    if (n.type === "text") out += n.value;
  });
  return out;
}
/** Text that is not inside a formula, so what a reader sees as prose. */
function proseText(tree) {
  let out = "";
  (function go(node) {
    if (node.type === "text") out += node.value;
    if (node.type === "element" && classesOf(node).some((c) => c === "katex" || c === "katex-display")) return;
    for (const child of node.children ?? []) go(child);
  })(tree);
  return out;
}

const SAMPLE = [
  "**Difference in unit volume between peak and weakest week:**",
  "\\[",
  "\\Delta = \\text{Week }4 - \\text{Week}3 = 583 - 395 = 188 \\text{ bags}",
  "\\]",
].join("\n");

const display = parse(SAMPLE);
check("render: \\[...\\] produces a display formula", findAll(display, hasClass("katex-display")).length === 1);
check("render: the display formula carries real KaTeX markup", findAll(display, hasClass("katex")).length === 1 && findAll(display, hasClass("katex-html")).length === 1);
check(
  "render: the formula source is kept for assistive tech",
  findAll(display, (n) => n.tagName === "annotation").some((n) => textOf(n).includes("\\Delta = \\text{Week }4")),
);
check("render: no raw bracket or LaTeX is left in the prose", !/[\[\]\\]/.test(proseText(display)));

const inline = parse("so \\(a^2 + b^2 = c^2\\) holds");
check("render: \\(...\\) produces an inline formula", findAll(inline, hasClass("katex")).length === 1);
check("render: the inline formula has KaTeX markup and no display wrapper", findAll(inline, hasClass("katex-display")).length === 0);
check("render: the sentence around it survives", proseText(inline) === "so  holds");

const dollars = parse("$$\nx^2\n$$\n\nand $$y$$ inline");
check("render: $$ blocks and a mid-line $$y$$ both render as display", findAll(dollars, hasClass("katex-display")).length === 2 && findAll(dollars, hasClass("katex")).length === 2);

for (const text of ["it costs $5 and $10", "250 USD for the pair, then $20 more", "$5 - $10 and $20"]) {
  const tree = parse(text);
  check(
    `currency render: "${text}" is plain text`,
    findAll(tree, (n) => classesOf(n).some((c) => c.startsWith("katex"))).length === 0 &&
      proseText(tree) === text,
  );
}

const unclosed = parse("so \\[ \\Delta = 1");
check(
  "stream render: an unclosed delimiter is plain text and does not throw",
  findAll(unclosed, (n) => classesOf(n).some((c) => c.startsWith("katex"))).length === 0 &&
    // Markdown's own escape drops the backslash before `[`; that is the same
    // text the reader saw before math support, and it is not a formula.
    proseText(unclosed) === "so [ \\Delta = 1",
);
const fencedTree = parse(fenced);
check(
  "code render: a fenced block keeps its source and has no formula",
  findAll(fencedTree, hasClass("katex")).length === 0 && textOf(fencedTree).includes("\\[x\\]"),
);

let badFormula = null;
try {
  badFormula = parse("see \\(\\frac{1}{\\) now and \\(\\undefinedmacro\\)");
} catch {
  badFormula = null;
}
check("error: a broken formula does not throw out of the parser chain", badFormula !== null);
check(
  "error: a broken formula renders as an error span, not a crash",
  badFormula !== null && findAll(badFormula, hasClass("katex-error")).length >= 1,
);

// Citations
const chips = (tree) => findAll(tree, hasClass("cite-mark"));
const inMath = parse("range \\(x_{[1]}\\) done", ["1"]);
check("cite: [1] inside an inline formula is not a chip", chips(inMath).length === 0);
const inDisplay = parse("\\[\n[1] + [2]\n\\]", ["1", "2"]);
check("cite: [1] [2] inside a display formula are not chips", chips(inDisplay).length === 0);
const mixed = parse("the bound \\(f(x) = [1]\\) holds [1]", ["1"]);
check("cite: [1] after the formula in the same paragraph is still a chip", chips(mixed).length === 1);
check(
  "cite: that chip sits outside the formula",
  chips(mixed).every((chip) => {
    let inside = false;
    for (const wrapper of findAll(mixed, (n) => classesOf(n).includes("katex"))) {
      walk(wrapper, (n) => {
        if (n === chip) inside = true;
      });
    }
    return !inside;
  }),
);
const plain = parse("fact [1] and [2].", ["1", "2"]);
check("cite: markers in ordinary prose are still chips", chips(plain).length === 2);

// ------------------------------------------------- shapes the first pass missed
// Adjacent and `$`-touching formulas must not fuse their fences.
const noError = (tree) => findAll(tree, hasClass("katex-error")).length === 0;
const adjacent = parse("\\(a\\)\\(b\\)");
check("adjacent: two inline formulas render as two", findAll(adjacent, hasClass("katex")).length === 2 && noError(adjacent));
const dollarEnd = parse("\\(5\\$\\)");
check("dollar: a body ending in $ is one clean formula", findAll(dollarEnd, hasClass("katex")).length === 1 && noError(dollarEnd));
const dollarAfter = parse("\\(x\\)$5");
check("dollar: a price touching a formula stays a price", findAll(dollarAfter, hasClass("katex")).length === 1 && noError(dollarAfter) && proseText(dollarAfter) === " $5");

// Blockquote display math keeps its quote prefix.
check("quote: display math in a blockquote keeps its prefix", norm("> \\[\n> x\n> \\]") === "> $$\n> x\n> $$");
const quoted = parse("> \\[\n> x\n> \\]");
const quoteBlocks = findAll(quoted, (n) => n.tagName === "blockquote");
check(
  "quote: it renders as one display formula inside the blockquote",
  quoteBlocks.length === 1 && findAll(quoteBlocks[0], hasClass("katex-display")).length === 1 && noError(quoted),
);
check("quote: inline math in a quote wraps cleanly", norm("> see \\(a\n> + b\\) now") === "> see $$a + b$$ now");

// Line endings, link destinations
check("crlf: a blank line with CR still ends the paragraph", norm("\\(a\r\n\r\nb\\)") === "\\(a\r\n\r\nb\\)");
check("crlf: display math converts", norm("A\r\n\\[\r\nx\r\n\\]\r\nB") === "A\r\n$$\nx\n$$\r\nB");
check("link: a destination is left alone", norm("[x](http://a/\\(b\\))") === "[x](http://a/\\(b\\))");
check("link: math after a link still converts", norm("[x](u) \\(y\\)") === "[x](u) $$y$$");

// A model-written $$ fence that never closes is neutralised.
check("dollar fence: an unclosed block becomes literal text", norm("$$\n\\frac{1}{") === "\\$\\$\n\\frac{1}{");
check("dollar fence: a closed block is untouched", norm("$$\n\\frac{1}{2}\n$$") === "$$\n\\frac{1}{2}\n$$");
check("dollar fence: nothing inside it is rewritten", norm("$$\n\\(x\\)\n$$") === "$$\n\\(x\\)\n$$");
check("dollar fence: an unclosed block renders no formula", findAll(parse("$$\n\\frac{1}{"), hasClass("katex")).length === 0);

// KaTeX size bounds
const huge = parse("\\(\\rule{1000000em}{1000000em}\\)");
const styles = [];
walk(huge, (n) => {
  if (n.type === "element" && typeof n.properties?.style === "string") styles.push(n.properties.style);
});
check("bounds: a huge rule is clamped", styles.length > 0 && !styles.join(";").includes("1000000"));

// Cost: each input is hostile to a rescanning implementation.
norm("warm \\(x\\) up \\[ y");
const hostile = {
  "unclosed display openers": "\\[ ".repeat(20000),
  "unclosed inline openers": "\\(x ".repeat(15000),
  "many formulas on one line": "\\(x\\) ".repeat(9000),
  "lone backticks": "` ".repeat(30000) + "\\(x\\)",
  "link openers": "](".repeat(30000) + "\\(x\\)",
  "single-dollar openers": "$x ".repeat(20000),
  "single-dollar formulas": "$x$ ".repeat(15000),
  "double-dollar pairs": "$$x$$ ".repeat(9000),
  "double-dollar blanks": "$$ ".repeat(20000),
  "double-dollar openers": "$$ x ".repeat(15000) + "$",
  "openers with unusable bodies": "\\(a$$ ".repeat(10000) + "\\)",
};
for (const [label, text] of Object.entries(hostile)) {
  const start = performance.now();
  norm(text);
  const ms = performance.now() - start;
  check(`cost: ${label} finishes quickly (${ms.toFixed(0)} ms)`, ms < 150);
  if (process.env.MATH_TIMINGS) console.log(`  ${label}: ${ms.toFixed(1)} ms`);
}

// Single-dollar math: what models actually write, with prices kept as prices.
const isFormula = (tree) => findAll(tree, hasClass("katex")).length;
const stayText = [
  "it costs $5 and $10",
  "$5 - $10 and $20",
  "a notebook costs $5 and a pen $2.",
  "between $5 and $10, or $20",
  "$100/month vs $90",
  "US$5 and A$10",
  "price: $5.00 (was $7.50)",
  "`$PATH` and $HOME and $USER",
  "costs $$$5$,$ a pen $ is priced at $$$2$.",
  "$5$ and $5,$ stay text",
  "cost \\$x$ here",
];
for (const text of stayText) {
  const tree = parse(text);
  check(`dollar text: "${text}" renders no formula`, isFormula(tree) === 0 && noError(tree));
}
for (const text of stayText.filter((t) => !t.includes("$$$"))) {
  check(`dollar text: "${text}" is not rewritten`, norm(text) === text);
}
const asMath = [
  ["($b^2 - 4ac$)", "($$b^2 - 4ac$$)"],
  ["where $x$ is real", "where $$x$$ is real"],
  ["$a_1 + a_2 = 3$", "$$a_1 + a_2 = 3$$"],
  ["$\\alpha$", "$$\\alpha$$"],
  ["$E=mc^2$.", "$$E=mc^2$$."],
  ["if $n > 0$ then", "if $$n > 0$$ then"],
];
for (const [text, expected] of asMath) {
  const tree = parse(text);
  check(`dollar math: "${text}" becomes inline math`, norm(text) === expected && isFormula(tree) === 1 && noError(tree));
}
const keptAsWritten = [
  ["a code span", "use `$x$` here"],
  ["a fence", "```\n$x$\n```"],
  ["a link destination", "[a](http://x/$y$)"],
  ["a dollar block", "$$\n$x$\n$$"],
  ["an unclosed dollar", "so $x + 1"],
  ["a closer followed by a digit", "pay $x$5 now"],
  ["a body over a line", "$a\nb$"],
  ["a body with a backtick", "use $x `y$ here"],
  ["a body over 200 characters", "$" + "x".repeat(250) + "$"],
];
for (const [label, text] of keptAsWritten) {
  check(`dollar guard: ${label} is left alone`, norm(text) === text);
}
check("dollar mix: a price and a formula in one sentence", norm("$5 or $x$ or $10") === "$5 or $$x$$ or $10");

// Streaming: every prefix of a realistic answer parses without error markup.
const answers = [
  "**Difference:**\n\\[\n\\Delta = \\text{Week }4 - \\text{Week}3 = 188\n\\]\nPer the report [1] it grew; inline \\(x_{n+1} = x_n + [2]\\) as shown [2].\n\n```py\nprint(\"\\\\[x\\\\]\")\n```\nDone.",
  "$$\n\\frac{a}{b}\n$$\n\nCosts $5 and $10, so \\(a+b\\) [1].",
  "The discriminant ($b^2 - 4ac$) tells us $x$ is real if $n > 0$ then $E=mc^2$. [1]",
  "> \\[\n> E = mc^2\n> \\]\n\n- item \\(y\\) [1]\n- see [link](http://a/\\(b\\)) and \\(z\\)",
];
let prefixes = 0;
let prefixFailure = null;
for (const [which, answer] of answers.entries()) {
  for (let end = 1; end <= answer.length; end++) {
    prefixes += 1;
    try {
      const tree = parse(answer.slice(0, end), ["1", "2"]);
      if (!noError(tree)) prefixFailure ??= `answer ${which} prefix ${end}: katex-error`;
      // The math-free-of-$$ answers must never leak a fence into the prose.
      if (which !== 1 && proseText(tree).includes("$$")) prefixFailure ??= `answer ${which} prefix ${end}: $$ in prose`;
    } catch (error) {
      prefixFailure ??= `answer ${which} prefix ${end}: threw ${error}`;
    }
  }
}
// A $$...$$ the model wrote means display math, wherever it sits.
const RAW = "$$x = \\frac{-b \\pm \\sqrt{b^{2} - 4ac}}{2a}$$The discriminant ($b^2-4ac$) tells us the roots.";
const displays = (tree) => findAll(tree, hasClass("katex-display"));
const inlineFormulas = (tree) =>
  findAll(tree, hasClass("katex")).filter(
    (node) => !displays(tree).some((wrapper) => findAll(wrapper, (n) => n === node).length > 0),
  );
const rawTree = parse(RAW);
check("display: the raw model string has one display formula", displays(rawTree).length === 1);
check("display: ...and one inline formula, with no errors", inlineFormulas(rawTree).length === 1 && noError(rawTree));
check(
  "display: the sentence is outside the display node",
  displays(rawTree).length === 1 &&
    !textOf(displays(rawTree)[0]).includes("discriminant") && proseText(rawTree).includes("The discriminant ("),
);
check(
  "display: no paragraph holds the display formula",
  findAll(rawTree, (n) => n.tagName === "p").every((p) => displays(p).length === 0),
);
const between = parse("text $$a+b$$ more text");
const betweenKinds = between.children.filter((n) => n.type === "element").map((n) => (displays(n).length > 0 || classesOf(n).includes("katex-display") ? "display" : n.tagName));
check("display: mid-sentence becomes a block between two paragraphs", betweenKinds.join(",") === "p,display,p");
check("display: the two paragraphs keep their text", findAll(between, (n) => n.tagName === "p").map((p) => textOf(p).trim()).join("|") === "text|more text");
const wholeLine = parse("$$x$$");
check("display: a whole-line $$x$$ is display", norm("$$x$$") === "$$\nx\n$$" && displays(wholeLine).length === 1 && inlineFormulas(wholeLine).length === 0);
const ownInline = parse("a \\(x\\) and $y$ here");
check("display: \\(x\\) and $x$ stay inline", displays(ownInline).length === 0 && inlineFormulas(ownInline).length === 2 && noError(ownInline));
check(
  "display: inside a blockquote the prefix is kept",
  norm("> text $$a$$ more") === "> text\n>\n> $$\n> a\n> $$\n>\n> more",
);
const quotedDisplay = parse("> text $$a$$ more");
const quoteNode = findAll(quotedDisplay, (n) => n.tagName === "blockquote");
check("display: ...and renders inside the blockquote", quoteNode.length === 1 && displays(quoteNode[0]).length === 1 && noError(quotedDisplay));
check("display: a multi-line body is one block", norm("$$a\nb$$") === "$$\na\nb\n$$");
check("display: two on one line are two blocks", displays(parse("$$a$$ then $$b$$")).length === 2);
for (const [label, text] of [
  ["a code span", "use `$$x$$` here"],
  ["a fence", "```\n$$x$$\n```"],
  ["a link destination", "[a](http://x/$$y$$)"],
  ["an unclosed mid-line", "so $$x + 1"],
  ["a blank body", "a $$ $$ b"],
  ["a backtick body", "a $$x `y$$ b"],
]) {
  check(`display guard: ${label} is left alone`, norm(text) === text);
}
check("display guard: an unclosed line-start $$ is escaped", norm("$$x = \\frac{") === "\\$\\$x = \\frac{");
check("display guard: ...and renders no formula", findAll(parse("$$x = \\frac{"), hasClass("katex")).length === 0);
check("display guard: a $$$ run is not a display", displays(parse("costs $$$5 now")).length === 0 && noError(parse("costs $$$5 now")));

// Before the closing $$ arrives, nothing renders as a formula.
const closeAt = RAW.indexOf("$$", 2) + 2;
let rawPrefixFailure = null;
for (let end = 1; end <= RAW.length; end++) {
  const tree = parse(RAW.slice(0, end));
  if (!noError(tree)) rawPrefixFailure ??= `prefix ${end}: katex-error`;
  if (end < closeAt && findAll(tree, hasClass("katex")).length !== 0) rawPrefixFailure ??= `prefix ${end}: formula before the closer`;
}
check(`display stream: every prefix of the raw string is clean (${rawPrefixFailure ?? "ok"})`, rawPrefixFailure === null);

// A price must never flicker into a formula on any prefix.
const priceAnswer =
  "It costs $5 and $10, a pen $2.50, `$PATH` and $HOME, $100/month vs $90, US$5 and A$10, price: $5.00 (was $7.50), between $5 and $10, or $20 [1].";
let priceFailure = null;
for (let end = 1; end <= priceAnswer.length; end++) {
  const tree = parse(priceAnswer.slice(0, end), ["1"]);
  if (isFormula(tree) !== 0 || !noError(tree)) priceFailure ??= `prefix ${end}`;
  prefixes += 1;
}
check(`stream: no prefix of a price answer becomes a formula (${priceFailure ?? "ok"})`, priceFailure === null);
check(`stream: all ${prefixes} prefixes parse with no error markup (${prefixFailure ?? "ok"})`, prefixFailure === null);

// The component can only be loaded as TSX, which this script cannot do, so
// these are SOURCE checks: the behaviour above runs through the shared builders
// the component imports, and these pin that the component still uses them.
const component = read("components/chat/markdown-content.tsx");
check("source: the component parses with the shared remark chain", component.includes("remarkPlugins={answerRemarkPlugins}"));
check("source: the component builds rehype plugins with the shared builder", component.includes("answerRehypePlugins({ citationIndexes, live })"));
check(
  "source: delimiters are rewritten only after the pathological check",
  component.includes("isPathological(deferred) ? deferred : normalizeMathDelimiters(deferred)") && component.includes("{parsed}"),
);
check("source: the layout loads the KaTeX stylesheet from the package", read("app/layout.tsx").includes('import "katex/dist/katex.min.css";'));
const plugins = read("lib/math-markdown.ts");
check("source: KaTeX is registered before the citation walk", plugins.indexOf("[...REHYPE_KATEX]") !== -1 && plugins.indexOf("[...REHYPE_KATEX]") < plugins.indexOf("answerMarkPlugin(options)"));

if (failures.length > 0) {
  console.error(`math rendering: ${failures.length} failed, ${passed} passed`);
  for (const name of failures) console.error(`  FAIL ${name}`);
  process.exit(1);
}
console.log(`math rendering: all ${passed} checks passed`);
