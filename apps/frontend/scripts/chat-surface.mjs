/**
 * Battery for the chat surface: the timeline turn and what it reads.
 *
 * Three kinds of check. The pure helpers behind the rail (`lib/timeline.ts`),
 * the code tokenizer (`lib/code-tokens.ts`) and the message stamps
 * (`lib/messages.ts`) are run directly. The stylesheets are read as text and
 * pinned on the rules the design contract names: the one row grid, the
 * serif answer column on aurora alone, the rail zeroed on minimal, the
 * un-carded tool row on aurora, the docked notice's track, the held toggle.
 * And the components are read as text for the hooks the themes and the
 * other components key on, which a refactor could drop without any type
 * noticing.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/chat-surface.mjs)
 *
 * It imports the app's TypeScript directly and so needs a Node that strips
 * types on its own - 22.18 or newer.
 */

import { readFileSync } from "node:fs";
import { register } from "node:module";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `chat surface: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const { clockLabel, elapsedLabel, sourceCountLabel, thoughtLabel, tokensPerSecond, toolGlyph } =
  await import("../lib/timeline.ts");
const { tokenizeCode } = await import("../lib/code-tokens.ts");
const { appendDelta, beginToolStep, completeToolStep, recordTrailingFigures, toPaneMessages, turnFigures } =
  await import("../lib/messages.ts");

const root = join(dirname(fileURLToPath(import.meta.url)), "..");
const read = (rel) => readFileSync(join(root, rel), "utf8");

let passed = 0;
const failures = [];
function check(name, ok) {
  if (ok) passed += 1;
  else failures.push(name);
}

// ---------------------------------------------------------------- the rail
check("elapsed: seconds", elapsedLabel("2026-10-10T10:00:00Z", Date.parse("2026-10-10T10:00:06Z")) === "+6s");
check(
  "elapsed: minutes carry a padded seconds figure",
  elapsedLabel("2026-10-10T10:00:00Z", Date.parse("2026-10-10T10:01:04Z")) === "+1m04s",
);
check("elapsed: rounds to the nearest second", elapsedLabel("2026-10-10T10:00:00.000Z", Date.parse("2026-10-10T10:00:00.600Z")) === "+1s");
check("elapsed: never negative", elapsedLabel("2026-10-10T10:00:10Z", Date.parse("2026-10-10T10:00:00Z")) === "+0s");
check("elapsed: null without a question clock", elapsedLabel(undefined, 1) === null);
check("elapsed: null without a step clock", elapsedLabel("2026-10-10T10:00:00Z", undefined) === null);
check("elapsed: null on an unparseable clock", elapsedLabel("yesterday", 1) === null);
check("clock: null on nothing", clockLabel(undefined) === null && clockLabel("nope") === null);
check("clock: a time, not a date", /^\d{1,2}:\d{2}/.test(clockLabel("2026-10-10T14:26:00Z") ?? ""));
check("thought: plain when the span is unknown", thoughtLabel(undefined, 5000) === "Thought" && thoughtLabel(1000, undefined) === "Thought");
check("thought: counts whole seconds", thoughtLabel(1000, 7400) === "Thought for 6 seconds");
check("thought: singular second", thoughtLabel(1000, 1900) === "Thought for 1 second");
check("thought: never zero", thoughtLabel(1000, 1100) === "Thought for 1 second");
check("tokens/s: rounds over the runtime's writing time", tokensPerSecond(412, 10_800) === 38);
check("tokens/s: null under half a second", tokensPerSecond(10, 400) === null);
check("tokens/s: null without a count or a time", tokensPerSecond(undefined, 1000) === null && tokensPerSecond(10, undefined) === null);
check("glyph: the two built-ins have their medium", toolGlyph("search_knowledge") === "search" && toolGlyph("web_search") === "globe");
check("glyph: anything else is a connected tool", toolGlyph("web_lookup") === "plug" && toolGlyph("post_status_update") === "plug");

// ---------------------------------------------------------- the tokenizer
const SAMPLE = [
  "# point the app at the new host, then restart it",
  'export QDRANT_URL="https://new-host:6333"',
  "make deploy",
  "if [ $? -ne 0 ]; then echo \"roll back\"; fi",
  "curl -s $URL | jq '.result.points_count'   # 48213",
].join("\n");
const tokens = tokenizeCode(SAMPLE);
const kinds = (kind) => tokens.filter((t) => t.kind === kind).map((t) => t.text);
check("tokenizer: concatenation restores the source", tokens.map((t) => t.text).join("") === SAMPLE);
check("tokenizer: a hash comment, to the line's end", kinds("comment")[0] === "# point the app at the new host, then restart it");
check("tokenizer: a trailing comment after code", kinds("comment").includes("# 48213"));
check("tokenizer: double-quoted strings", kinds("string").includes('"https://new-host:6333"'));
check("tokenizer: single-quoted strings", kinds("string").includes("'.result.points_count'"));
check("tokenizer: shell keywords", ["export", "if", "then", "fi"].every((k) => kinds("keyword").includes(k)));
check("tokenizer: a bare number", kinds("number").includes("0"));
check("tokenizer: a digit inside a comment stays a comment", !kinds("number").includes("48213"));
check("tokenizer: a word followed by ( is a function", tokenizeCode("print(x)")[0].kind === "function");
check("tokenizer: a keyword followed by ( stays a keyword", tokenizeCode("if (x)")[0].kind === "keyword");
check("tokenizer: a shebang is not a comment", tokenizeCode("#!/bin/sh\n")[0].kind === null);
check("tokenizer: a hex literal", tokenizeCode("0x1F")[0].kind === "number" && tokenizeCode("0x1F")[0].text === "0x1F");
check("tokenizer: a digit-led identifier is plain", tokenizeCode("3abc").every((t) => t.kind === null));
check("tokenizer: block comments span lines", tokenizeCode("/* a\nb */ x")[0].text === "/* a\nb */");
check("tokenizer: an unterminated string ends at the line", tokenizeCode("'open\nnext")[0].text === "'open");
check("tokenizer: SQL keywords in upper case", tokenizeCode("SELECT id FROM t")[0].kind === "keyword");
check("tokenizer: -- comments only where a comment can start", tokenizeCode("a -- note").some((t) => t.kind === "comment" && t.text === "-- note") && tokenizeCode("x--").every((t) => t.kind === null));
check("tokenizer: a member access keeps the receiver plain", tokenizeCode("obj.call()").find((t) => t.kind === "function")?.text === "call" && tokenizeCode("obj.call()")[0].kind === null);
check("tokenizer: empty input is empty", tokenizeCode("").length === 0);
// the false positives the first draft had: a flag, a colour, a URL, everyday names
const plainOnly = (src) => tokenizeCode(src).every((t) => t.kind !== "comment");
check("tokenizer: a --flag is not a comment", plainOnly("docker run --rm -it app") && plainOnly("npm i --save x"));
check("tokenizer: #fff is not a comment", plainOnly("color: #fff;") && plainOnly("#include <stdio.h>"));
check("tokenizer: a URL is not a comment", plainOnly("curl http://localhost:8000/x") && plainOnly("url: https://a.b/c"));
check("tokenizer: everyday names are not keywords", ["set(x)", "type(x)", "key = 1", "select id, name", "git checkout source"].every((src) => tokenizeCode(src).every((t) => t.kind !== "keyword")));
check("tokenizer: a method call is a function, not a keyword", tokenizeCode("map.delete(k)").find((t) => t.kind === "function")?.text === "delete");
check("tokenizer: a control keyword before ( stays a keyword", tokenizeCode("while (x)")[0].kind === "keyword" && tokenizeCode("return (x)")[0].kind === "keyword");
// a property: for any text, it never throws and the runs restore the source
{
  let seed = 20261010;
  const rand = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
  const alphabet = "abc xyz_$ 019 \n\t\"'`/#-*:.(){}[]=<>!;\eE+";
  let ok = true;
  for (let i = 0; i < 400 && ok; i += 1) {
    let src = "";
    const len = Math.floor(rand() * 40);
    for (let k = 0; k < len; k += 1) src += alphabet[Math.floor(rand() * alphabet.length)];
    try {
      const runs = tokenizeCode(src);
      ok = runs.map((t) => t.text).join("") === src && runs.every((t) => t.text !== "");
    } catch {
      ok = false;
    }
  }
  check("tokenizer: never throws and always restores the source (400 random inputs)", ok);
}

// ---------------------------------------------------------- the stamps
const reply = [{ role: "user", content: "q", created_at: "2026-10-10T10:00:00Z" }, { role: "assistant", content: "" }];
const thought = appendDelta(reply, "thinking", "hm", 1000);
const thoughtMore = appendDelta(thought, "thinking", "...", 2000);
check("stamps: the first reasoning delta stamps thinkingAt", thoughtMore[1].thinkingAt === 1000);
check("stamps: a later delta keeps it", thoughtMore[1].thinking === "hm..." && thoughtMore[1].thinkingAt === 1000);
check("stamps: reasoning does not stamp the answer", thoughtMore[1].answerAt === undefined);
const answered = appendDelta(appendDelta(thoughtMore, "content", "A", 3000), "content", "B", 4000);
check("stamps: the first token stamps answerAt once", answered[1].answerAt === 3000 && answered[1].content === "AB");
const stepped = beginToolStep(thoughtMore, { id: "c1", name: "web_search", arguments: { q: "x" } }, 2500);
check("stamps: a tool call stamps the step", stepped[1].toolSteps?.[0].at === 2500);
check("stamps: a tool call still clears the narration into the step", stepped[1].content === "" && stepped[1].toolSteps?.[0].running === true);
const doneStep = completeToolStep(stepped, { id: "c1", name: "web_search", result: "3 results", count: 3 });
check("count: a tool_result's count lands on its step", doneStep[1].toolSteps?.[0].count === 3);
const uncounted = completeToolStep(stepped, { id: "c1", name: "web_search", result: "ok", count: null });
check("count: a null count leaves the step uncounted", uncounted[1].toolSteps?.[0].count === undefined);
check("count: plural label", sourceCountLabel(3) === "3 sources");
check("count: singular label", sourceCountLabel(1) === "1 source");
check("count: nothing for null, absent, zero or garbage", sourceCountLabel(null) === null && sourceCountLabel(undefined) === null && sourceCountLabel(0) === null && sourceCountLabel(Number.NaN) === null);
const figured = recordTrailingFigures(answered, { model: "qwen3:4b", usage: { completion_tokens: 58, generation_ms: 2900 }, doneAt: 6000 });
check("figures: model, tokens, writing time and the clock land on the reply", figured[1].model === "qwen3:4b" && figured[1].completionTokens === 58 && figured[1].generationMs === 2900 && figured[1].doneAt === 6000);
const partial = recordTrailingFigures(figured, { model: null, usage: null });
check("figures: nulls leave what was there", partial[1].model === "qwen3:4b" && partial[1].completionTokens === 58 && partial[1].generationMs === 2900 && partial[1].doneAt === 6000);
check("figures: a count with no time is a count with no speed", JSON.stringify(turnFigures({ completion_tokens: 58, generation_ms: null })) === '{"completionTokens":58}' && JSON.stringify(turnFigures({ completion_tokens: 58 })) === '{"completionTokens":58}' && JSON.stringify(turnFigures({ completion_tokens: 58, generation_ms: 0 })) === '{"completionTokens":58}');
check("figures: no count is no figures, whatever the time", [null, undefined, {}, { completion_tokens: 0, generation_ms: 900 }, { completion_tokens: "58", generation_ms: 900 }, { generation_ms: 900 }].every((u) => JSON.stringify(turnFigures(u)) === "{}"));
check("figures: a user turn is left alone", recordTrailingFigures([reply[0]], { model: "x" })[0].model === undefined);
const stored = toPaneMessages([
  { id: "u", role: "user", content: "q", thinking: null, model: null, created_at: "2026-10-10T10:00:00Z" },
  { id: "a", role: "assistant", content: "r", thinking: null, model: "qwen3:4b-instruct", created_at: "2026-10-10T10:00:14Z" },
]);
const storedFigured = toPaneMessages([
  { id: "u", role: "user", content: "q", thinking: null, model: null, created_at: "2026-10-10T10:00:00Z", usage: { completion_tokens: 9, generation_ms: 900 } },
  { id: "a", role: "assistant", content: "r", thinking: null, model: "qwen3:4b-instruct", created_at: "2026-10-10T10:00:14Z", usage: { completion_tokens: 58, generation_ms: 2900 } },
]);
check("stored: a reply rehydrates the figures its stream closed on", storedFigured[1].completionTokens === 58 && storedFigured[1].generationMs === 2900);
check("stored: the reloaded speed is the live one", tokensPerSecond(storedFigured[1].completionTokens, storedFigured[1].generationMs) === tokensPerSecond(figured[1].completionTokens, figured[1].generationMs) && tokensPerSecond(58, 2900) === 20);
check("stored: a question never carries figures", storedFigured[0].completionTokens === undefined && storedFigured[0].generationMs === undefined);
check("stored: a reply saved before the figures has none", stored[1].completionTokens === undefined && stored[1].generationMs === undefined);
const { attachConfirmation, markTrailingFailed } = await import("../lib/messages.ts");
const failedReply = markTrailingFailed(answered, 9000);
check("failed: the trailing reply is marked, with its clock", failedReply[1].failed === true && failedReply[1].failedAt === 9000);
check("failed: a question is left alone", markTrailingFailed([reply[0]], 1)[0].failed === undefined);
const paused = attachConfirmation(answered, { id: "c", conversation_id: "x", calls: [] }, 7000);
check("confirm: the pause stamps its clock once", paused[1].confirmAt === 7000 && attachConfirmation(paused, { id: "c", conversation_id: "x", calls: [] }, 8000)[1].confirmAt === 7000);
const storedCounted = toPaneMessages([
  { id: "a", role: "assistant", content: "r", thinking: null, model: null, created_at: "2026-10-10T10:00:14Z", tool_steps: [{ id: "c1", name: "web_search", arguments: {}, result: "x", count: 2 }, { id: "c2", name: "web_search", arguments: {}, result: "x", count: null }] },
]);
check("count: a stored step rehydrates its count, null stays uncounted", storedCounted[0].toolSteps?.[0].count === 2 && storedCounted[0].toolSteps?.[1].count === undefined);
check("stored: a reply carries the model that produced it", stored[1].model === "qwen3:4b-instruct" && stored[0].model === undefined);

// ---------------------------------------------------------- the stylesheets
const base = read("app/themes/tokens.base.css");
const chat = read("app/themes/chat.css");
const aurora = read("app/themes/aurora.css");
const minimal = read("app/themes/minimal.css");
const has = (text, needle) => text.includes(needle);
const rule = (text, selector) => {
  const i = text.indexOf(selector);
  if (i === -1) return null;
  const open = text.indexOf("{", i);
  const close = text.indexOf("}", open);
  return text.slice(open + 1, close);
};

check("tokens: the rail is sized by tokens", ["--rail-time:", "--rail-glyph:", "--rail-gap:", "--step-gap:", "--col:", "--gutter:"].every((t) => has(base, t)));
check("tokens: the prose column has its own size and measure", has(base, "--text-prose: var(--text-body)") && has(base, "--prose-measure: none"));
check("tokens: the step colours derive from the roles", ["--c-step-thought:", "--c-step-tool:", "--c-step-confirm:", "--c-step-answer:", "--c-step-error:"].every((t) => has(base, t)));
check("tokens: syntax colour has defaults", ["--c-code-com:", "--c-code-key:", "--c-code-str:", "--c-code-num:", "--c-code-fn:"].every((t) => has(base, t)));

const step = rule(chat, "\n.step {");
check("chat: one grid per row, time | glyph | body", step !== null && /grid-template-columns:\s*var\(--rail-time\)\s+var\(--rail-glyph\)\s+minmax\(0,\s*1fr\)/.test(step));
check("chat: the rail hangs outside the column", step !== null && has(step, "padding-right: var(--gutter)"));
check("chat: the reading column is the composer's width plus the gutters", has(rule(chat, ".thread-inner {") ?? "", "calc(var(--col) + 2 * var(--gutter))") && has(rule(chat, "\n.composer-box {") ?? "", "max-width: var(--col)"));
check("chat: a hairline joins each glyph to the next row", has(rule(chat, ".step-glyph::after {") ?? "", "background: var(--s-border-strong)"));
check("chat: the rail ends at the closing row", has(chat, ".step-end .step-glyph::after,") && has(chat, ".turn > .step:last-child .step-glyph::after"));
check("chat: the tool rows' wrapper is not a box", /\[data-part="tool-trace-list"\]\s*\{\s*display:\s*contents;/.test(chat));
check("chat: the user's side has no rail", has(rule(chat, ".turn-user .step-glyph {") ?? "", "display: none"));
// The time and the glyph centre on the row's first line: the chrome's 20px
// line by default, the prose line on an answer. A fixed 2px offset put the
// glyph four pixels above a serif paragraph's first line.
check("chat: the rail centres on the row's line", has(rule(chat, "\n.step {") ?? "", "--row-lh: 20px") && has(rule(chat, ".step-time {") ?? "", "line-height: var(--row-lh)") && has(rule(chat, ".step-glyph > svg.i {") ?? "", "margin-top: calc((var(--row-lh) - 16px) / 2)"));
check("chat: the answer row's rail centres on the prose line", has(rule(chat, '.step[data-kind="answer"] {') ?? "", "--row-lh: calc(var(--text-prose) * var(--text-prose-line))"));
check("chat: the rail hairline starts below the glyph wherever it sits", has(rule(chat, ".step-glyph::after,\n.step-glyph::before {") ?? "", "top: calc(var(--row-lh) / 2 + 14px)"));
check("chat: the rail glyph is a shade heavier than the chrome's", has(rule(chat, ".step-glyph > svg.i {") ?? "", "stroke-width: 1.75"));
// An inline box sits on the body's taller baseline and drops below the glyph.
check("chat: a disclosure row sits on the row's top, on the rail's line", (() => {
  const r = rule(chat, ".disc-btn {") ?? "";
  return has(r, "display: flex") && has(r, "width: fit-content") && has(r, "line-height: 20px") && !has(r, "inline-flex");
})());
// The closing row usually opens with the 28px chips or verbs row: its check
// and time centre on that, not on a 20px line above it.
check("chat: a closing row opening with chips or verbs centres on them", has(rule(chat, '.step-end:has(> .step-body > :first-child:is([data-part="citations"], [data-part="turn-actions"])) {') ?? "", "--row-lh: 28px"));
// The glyph cell is a flex column that stretches its children; the loader
// keeps its own height so its dots sit where the glyph would.
check("chat: the rail loader keeps its own height", (() => {
  const r = rule(chat, ".step-glyph .t-matrix {") ?? "";
  return has(r, "margin-top: calc((var(--row-lh) - 14px) / 2)") && has(r, "align-self: flex-start");
})());
check("chat: the thread's lower edge fades under a mask", has(rule(chat, '[data-part="thread"] {') ?? "", "mask-image: linear-gradient("));
check("chat: the fade steps aside while a citation card is open (the mask is a backdrop root)", has(rule(chat, '[data-part="thread"]:has(.citation-popover) {') ?? "", "mask-image: none"));
check("chat: the answer reads in the prose face at the prose size", has(rule(chat, '\n[data-part="content"] {') ?? "", "font-family: var(--font-prose)") && has(rule(chat, '\n[data-part="content"] {') ?? "", "font-size: var(--text-prose)"));
check("chat: the prose measure spares code and tables", has(chat, '[data-part="content"] > :not([data-part="code-block"]):not(.md-table-wrap) {'));
check("chat: the code block is a component with a head", has(chat, '[data-part="code-block"] {') && has(chat, ".code-head {") && has(chat, ".tk-c {"));
check("chat: inline code reads its own fill token", has(rule(chat, '[data-part="content"] code:not(pre code) {') ?? "", "var(--c-inline-code-bg)"));
check("chat: the disclosure panel opens from 0fr to 1fr", has(rule(chat, ".disc-panel {") ?? "", "grid-template-rows: 0fr") && has(rule(chat, '.disc-btn[aria-expanded="true"] + .disc-panel {') ?? "", "grid-template-rows: 1fr"));
check("chat: a closed panel leaves the accessibility tree", has(rule(chat, ".disc-panel {") ?? "", "visibility: hidden"));
check("chat: the tool row is a hairline chip in the base", has(rule(chat, '[data-part="tool-trace"] .disc-btn {') ?? "", "background: var(--c-tool-bg)"));
check("chat: a failed call takes the danger colour", has(chat, '[data-part="tool-trace"][data-state="danger"] {'));
check("chat: the footer shows at rest on the latest reply only", has(chat, '.turn[data-latest] [data-part="turn-actions"]') && has(rule(chat, '\n[data-part="turn-actions"] {') ?? "", "opacity: 0"));
check("chat: a verb taken holds the row open", has(chat, '.turn[data-hold] [data-part="turn-actions"]'));
check("chat: the figures are small mono", has(rule(chat, ".turn-meta {") ?? "", "font-family: var(--font-mono)"));
check("chat: the matrix loader runs on one pulse with a per-dot delay", has(rule(chat, ".t-matrix i {") ?? "", "animation-delay: calc(var(--d, 0) * 1ms)"));
check("chat: the thinking badge appears only under reduced motion", /prefers-reduced-motion: reduce\)\s*\{\s*\.turn\.live \.tag-live\s*\{\s*display: inline-block/.test(chat) && has(rule(chat, "\n.tag-live {") ?? "", "display: none"));
check("chat: the error has no box", !has(rule(chat, '\n[data-part="error"] {') ?? "x", "border"));
check("chat: the error's own glyph yields to the rail's", has(chat, '.step[data-kind="error"] [data-part="error"] .err-glyph {'));
check("chat: the notice is a line, not a card", !has(rule(chat, '\n[data-part="notice"] {') ?? "x", "border"));
check("chat: the model status keeps the name in mono", has(rule(chat, '[data-part="model-status"] .ms-name {') ?? "", "font-family: var(--font-mono)"));
check("chat: on disk the dot is a ring", has(rule(chat, '[data-model-state="disk"] .ms-dot {') ?? "", "box-shadow: inset"));
check("chat: the docked notice is a track above the composer", has(rule(chat, '\n[data-part="composer-notice"] {') ?? "", "grid-template-rows: 0fr") && has(rule(chat, '[data-part="composer-notice"][data-open] {') ?? "", "grid-template-rows: 1fr"));
check("chat: the notice box joins the composer's top edge", has(rule(chat, ".cn-box {") ?? "", "border-bottom: none"));
check("chat: the jump floater carries a dot for new content", has(chat, '[data-part="jump-to-latest"][data-new="true"] i {'));
check("chat: a held toggle is struck, not pressed", has(rule(chat, ".ctl[data-held] {") ?? "", "color: var(--s-text-3)") && has(rule(chat, ".ctl[data-held] .lbl {") ?? "", "text-decoration: line-through"));
check("chat: the copy glyph cross-fades to a tick", has(rule(chat, ".copy-btn .ic .ok {") ?? "", "opacity: 0") && has(rule(chat, ".copy-btn[data-copied] .ic .ok {") ?? "", "opacity: 1"));
check("chat: under 768px the rail time and the gutters go, the glyph stays", /max-width: 767px\)\s*\{\s*:root\s*\{\s*--rail-time: 0px;[^}]*--gutter: 0px;/.test(chat) && /\.step\s*\{\s*grid-template-columns: var\(--rail-glyph\) minmax\(0, 1fr\)/.test(chat));
check("chat: no backdrop filter on a row, a card or the code block", !/(\.step|\[data-part="code-block"\]|\.tool-detail|\.cn-box)[^{]*\{[^}]*backdrop-filter/.test(chat));

check("aurora: the answer column is the serif, at 16.5px / 1.68 on 68ch", has(aurora, "--font-prose: var(--font-source-serif)") && has(aurora, "--text-prose: 16.5px") && has(aurora, "--text-prose-line: 1.68") && has(aurora, "--prose-measure: 68ch"));
check("aurora: the shell keeps the body size", !has(aurora, "--text-body:"));
check("aurora: inline code is a chip on the panel", has(aurora, "--c-inline-code-bg: var(--s-bg-panel)"));
check("aurora: the tool row is no chip at all", has(rule(aurora, '[data-part="tool-trace"] .disc-btn {') ?? "", "background: none"));
check("aurora: the list marker is a star with its glow", has(aurora, 'ul:not(.contains-task-list) > li::before {') && has(rule(aurora, "ul:not(.contains-task-list) > li::before {") ?? "", "0 0 7px"));
check("aurora: the thinking body is bare text", has(aurora, "--c-thinking-rule: transparent"));
check("aurora: the code block and the tool detail are lit panes", has(aurora, 'html[data-theme="aurora"] [data-part="code-block"],') && has(aurora, 'html[data-theme="aurora"] .tool-detail,'));
check("aurora: the running chip sweep is gone", !has(aurora, "aurora-tool-sweep"));
check("aurora: the answer carries no pane", !has(aurora, '[data-part="bubble-assist"] > [data-part="content"]'));
check("minimal: no rail", has(minimal, "--rail-time: 0px; --rail-glyph: 0px;") && /\.step-time,\s*html\[data-theme="minimal"\] \.step-glyph\s*\{\s*display: none/.test(minimal));
check("minimal: no serif", !has(minimal, "--font-prose"));

// ---------------------------------------------------------- the components
const bubble = read("components/chat/message-bubble.tsx");
const toolRow = read("components/chat/tool-step-card.tsx");
const page = read("app/page.tsx");
const context = read("components/chat/citation-context.tsx");
const markdown = read("components/chat/markdown-content.tsx");
const stepSource = read("components/chat/step.tsx");
const followSource = read("hooks/use-follow-stream.ts");

check("toolrow: a finished step shows its count in the meta slot", has(toolRow, 'state === "ok"') && has(toolRow, "sourceCountLabel(step.count)") && has(toolRow, '<span className="tool-meta">{meta}</span>'));
check("bubble: the reply's turn carries the part the selection quote reads", has(bubble, 'data-part="bubble-assist"') && has(bubble, 'className={`turn${live ? " live" : ""}`}'));
check("bubble: the latest reply is marked for the footer", has(bubble, 'data-latest={latest ? "true" : undefined}'));
check("bubble: every reply row is a step", ['kind="wait"', 'kind="thought"', 'kind="confirm"', 'kind="answer"', 'kind="end" end'].every((k) => has(bubble, k)));
check("bubble: the thinking row opens while reasoning streams", has(bubble, "const thinkOpen = thinkToggled ?? reasoningInFlight;"));
check("bubble: the closed thinking row says how long", has(bubble, "thoughtLabel(message.thinkingAt, thinkingEndedAt)"));
check("bubble: the waiting row names the model and the load", has(bubble, "`Waiting for ${modelLabel} to load`") && has(bubble, "`Sending to ${modelLabel}`"));
check("bubble: no closing row while the turn is live or paused", has(bubble, "const settled = !live && !paused;") && has(bubble, "settled &&"));
check("bubble: a failed reply's closing row is the error row", has(bubble, "const failed = message.failed === true;") && has(bubble, '{failed && failure && (') && has(bubble, 'kind="error" end'));
check("chat: the closing row's parts breathe", has(rule(chat, ".step-end .step-body {") ?? "", "gap: 10px"));
check("chat: the citation card reads in the UI face", has(rule(chat, ".citation-popover {") ?? "", "font-family: var(--font-ui)"));
check("chat: a still loader lights its odd dots", has(chat, ".t-matrix i:nth-child(odd) {"));
check("chat: the citation chip is the prototype's shape with a hit floor", has(rule(chat, "\n.cite-chip {") ?? "", "border-radius: var(--radius-s)") && has(chat, ".cite-chip::after {"));
check("minimal: the error keeps its glyph and the live thinking pulses", has(minimal, '[data-part="error"] .err-glyph {') && has(minimal, 'content: "...";'));
check("aurora: the thinking body is flush", has(aurora, '[data-part="thinking"] .think-body {'));
check("bubble: the footer carries the figures", has(bubble, "tokensPerSecond(tokens, message.generationMs)"));
check("tool row: the loader runs while the call runs", has(toolRow, 'step.running ? (\n          <MatrixLoader variant="scan" color="var(--c-step-tool)" />'));
check("tool row: a row with nothing to open is a plain line", has(toolRow, "onToggle={hasDetail ? () => setOpen((o) => !o) : undefined}"));
check("context: a citation opens the disclosure rows", has(context, 'button[aria-expanded="false"]'));
check("markdown: fenced blocks go through the code block", has(markdown, "pre({ children }) {") && has(markdown, "<CodeBlock"));
check("step: the rail cell is hidden from assistive tech", has(stepSource, 'className="step-time" aria-hidden="true"'));
check("step: the loader has three variants on one cycle", has(stepSource, "const CYCLE_MS = 1200;") && has(stepSource, 'variant === "scan"') && has(stepSource, 'variant === "twinkle"'));
check("page: the thread follows through the hook", has(page, "useFollowStream(scrollRef, streamingRef)") && !has(page, "stuckToBottomRef"));
check("page: every send re-sticks", (page.match(/followRef\.current\.pin\(\);/g) ?? []).length >= 3);
check("page: the jump glides and streaks the sky", has(page, "followRef.current.jump();\n    signalSkyJump();"));
check("page: the floater shows the away and unseen states", has(page, 'data-visible={follow.away ? "true" : undefined}') && has(page, 'data-new={follow.unseen ? "true" : undefined}'));
check("page: the topbar carries the model status", has(page, "<ModelStatus model={modelTag}"));
check("page: a model on disk writes the loading state before the first frame", has(page, 'if (whereaboutsRef.current === "disk") {\n        setModelLoading(true);') && has(page, 'setTurnState("loading");'));
check("page: the first frame ends the load", (page.match(/arrived\(\);/g) ?? []).length >= 4);
check("page: done records the figures", has(page, "recordTrailingFigures(m, {") && has(page, "usage: done.turn_usage ?? null,") && has(page, "doneAt: Date.now()"));
check("page: the reply knows its question's clock", has(page, "questionAt={"));
check("page: the error is the rail's danger row", has(page, '<Step kind="error" end glyph={<Icon name="x" />}>'));
check("page: the notice docks above the composer", has(page, "<ComposerNotice") && has(page, "open={loadActive && !noticeDismissed}"));
check("follow: at the end means within 80px", has(followSource, "const NEAR_END_PX = 80;"));
check("follow: the glide is 420ms plus distance, capped at 1100", has(followSource, "const GLIDE_BASE_MS = 420;") && has(followSource, "const GLIDE_MAX_MS = 1100;") && has(followSource, "const GLIDE_PER_PX = 0.35;"));
check("follow: a wheel notch upward is intent to leave", has(followSource, "takeOver(e.deltaY < 0, WHEEL_INTENT_MS)"));
check("follow: a paging key is intent to leave", has(followSource, '"PageUp"') && has(followSource, '"Home"'));
check("follow: height changes drive the follow", has(followSource, "new ResizeObserver(follow)"));
check("follow: reduced motion cuts", has(followSource, "if (reduceMotion()) {\n      el.scrollTop = el.scrollHeight;"));
check("follow: a glide to the end ends stuck", has(followSource, 'if (goalRef.current === "end") {\n      stuckRef.current = true;\n      follow();'));
check("follow: the content column is re-observed when the thread's children change", has(followSource, "new MutationObserver(observeColumn)") && has(followSource, "for (const child of el.children) ro.observe(child);"));
check("follow: a paging key marks only the scroll that follows it as leaving", has(followSource, 'document.addEventListener("keydown", onKeyDown)') && has(followSource, "pendingLeaveRef.current = LEAVE_KEYS.has(e.key);"));
check("follow: a second jump cancels the glide in flight", has(followSource, 'cancelGlide();\n        goalRef.current = "end";\n        const from = el.scrollTop;'));
check("follow: a chase stood down by an intent window tries again when it closes", has(followSource, "chaseLaterRef.current = window.setTimeout("));
check("page: the load is shown only while its turn is in flight", has(page, "const loadActive = modelLoading && streaming;") && has(page, "modelLoading={loadActive}"));
check("page: only the reply the turn failed on takes the error row", has(page, "m.failed === true && i === messages.length - 1 ? errorNote : undefined") && has(page, "lastMessage?.failed !== true &&"));
check("page: the retry answers the failed reply again, and only that", has(page, "messages[last].failed === true &&"));
check("page: a dead request says what could not be reached", has(page, 'message: "The assistant could not be reached."') && (page.match(/setMessages\(\(m\) => markTrailingFailed\(m\)\);/g) ?? []).length >= 4);

/* ---------- the chrome: the composer's controls, the model choice, the sidebar, the glass ----------
   A whole-page inventory against the prototype found the chrome around the
   thread had kept its earlier anatomy. These pin what was moved over, and
   the one rule the prototype did not have: which model a conversation runs
   on when nothing is pinned. */

const { modelChoice, isChosen, sameTag } = await import("../lib/model-choice.ts");
const same = (a, b) => JSON.stringify(a) === JSON.stringify(b);
check("choice: a pin wins over everything", same(modelChoice("qwen3:4b", "gemma", ["gemma", "qwen3:4b"], false).model, "qwen3:4b"));
check("choice: an unpinned conversation runs on the workspace default when it is listed", same(modelChoice(null, "gemma", ["qwen3:4b", "gemma"], false), { model: "gemma", defaultListed: true, standIn: null }));
check("choice: a default the server no longer lists gives way to the first listed model on a fresh conversation, which is to be pinned", same(modelChoice(null, "gone:7b", ["qwen3:4b", "gemma"], true), { model: "qwen3:4b", defaultListed: false, standIn: "qwen3:4b" }));
check("choice: no default set behaves the same", same(modelChoice(null, null, ["qwen3:4b"], true), { model: "qwen3:4b", defaultListed: false, standIn: "qwen3:4b" }));
check("choice: a stored conversation on an unlisted default takes no stand-in and names nothing", same(modelChoice(null, "gone:7b", ["qwen3:4b"], false), { model: null, defaultListed: false, standIn: null }));
check("choice: nothing listed means nothing known to answer", same(modelChoice(null, "gemma", [], true), { model: null, defaultListed: false, standIn: null }));
check("choice: a pin the listing lacks still names the answerer and asks for no stand-in", same(modelChoice("old:1b", null, ["qwen3:4b"], true), { model: "old:1b", defaultListed: false, standIn: null }));
check("choice: tags match across case and a :latest suffix", sameTag("Qwen3:4B", "qwen3:4b:latest") && !sameTag("qwen3:4b", "qwen3:8b"));
check("choice: the default's row is the checked one across a :latest suffix", isChosen("gemma:latest", modelChoice(null, "gemma", ["gemma:latest"], false)));

const selector = read("components/chat/model-selector.tsx");
check("selector: there is no separate Default entry", !has(selector, "DEFAULT_HINT") && !/>\s*Default\s*</.test(selector));
check("selector: the default's own row wears the badge and choosing it clears the pin", has(selector, "<em>default</em>") && has(selector, "pick(isDefault ? null : entry.model)"));
check("selector: the trigger names the model that will answer, or says whose resolution it is", has(selector, "{choice.model ?? unnamed}") && has(selector, 'const unnamed = listingKnown && models.length === 0 ? "No model" : "Workspace default";'));
check("selector: the trigger carries the chevron glyph", has(selector, '<Icon name="chev" className="i chev" />'));
check("page: only a fresh conversation pins the stand-in, and only once the listing and the defaults have answered", /if \(conversationId !== null \|\| !modelsLoaded \|\| modelsUnavailable \|\| !defaultsRead\) return;\s*if \(pins\.model !== null\) return;\s*const \{ standIn \} = modelChoice\(null, defaultModel, chatCatalog\.map\(\(m\) => m\.model\), true\)/.test(page));
check("page: defaults that could not be read still count as answered", has(page, ".finally(() => setDefaultsRead(true));"));
check("page: the toggle reports the state of the panel it controls", has(page, "aria-expanded={narrow ? sidebarOpen : !navCollapsed}") && has(page, 'aria-controls="conversation-list"') && has(read("components/sidebar.tsx"), 'id="conversation-list"'));
check("page: the composer's placeholder is the prototype's", has(page, ': "Ask a question, or paste something to work on"'));
check("page: a question offers Reply like an answer does", !/onReply=\{\s*m\.role === "assistant"/.test(page) && /onReply=\{\s*m\.content !== ""/.test(page) && /align="end"\s*onEdit=\{onEdit\}\s*onReply=\{onReply\}/.test(bubble));
const actions = read("components/chat/turn-actions.tsx");
check("actions: a question's verbs run Edit, Reply, Copy; an answer's Copy, Reply, Again", has(actions, "{isReply && copy}") && has(actions, "{!isReply && copy}") && actions.indexOf("{isReply && copy}") < actions.indexOf('title="Edit"') && actions.indexOf("{!isReply && copy}") < actions.indexOf('title="Answer again"'));

const toggles = read("components/chat/thinking-toggle.tsx") + read("components/chat/web-search-toggle.tsx") + read("components/attachment-tray.tsx");
check("controls: the toggles and attach are chips with a glyph and a label", (toggles.match(/className="ctl"/g) ?? []).length === 3 && has(toggles, '<Icon name="thought" />') && has(toggles, '<Icon name="globe" />') && has(toggles, '<Icon name="clip" />') && !has(toggles, 'className="toggle"'));
const ctl = chat.slice(chat.indexOf("\n.ctl {\n"), chat.indexOf("\n}\n", chat.indexOf("\n.ctl {\n")));
check("chat: a control is a borderless 32px chip in the label size", has(ctl, "height: 32px") && !has(ctl, "border:") && has(ctl, "font-size: var(--text-label)") && has(ctl, "border-radius: var(--radius-s)"));
const on = rule(chat, '.ctl[aria-pressed="true"] {') ?? "";
check("chat: a control that is on is ink alone, neither boxed nor filled", has(on, "color: var(--s-accent)") && !has(on, "border") && !has(on, "background"));
const trigger = chat.slice(chat.indexOf("\n.model-trigger {\n"), chat.indexOf("\n}\n", chat.indexOf("\n.model-trigger {\n")));
check("chat: the model trigger is a chip in mono with no border of its own", has(trigger, "font-family: var(--font-mono)") && !has(trigger, "border") && has(read("components/chat/model-selector.tsx"), 'className="ctl model-trigger"'));
check("chat: the default badge is a quiet accent word", has(rule(chat, ".model-tag em {") ?? "", "color: var(--s-accent)") && has(rule(chat, ".model-tag em {") ?? "", "font-size: var(--text-micro)"));
check("chat: Send is the prototype's 34px square, two steps rounder than the chips on any theme", has(rule(chat, ".send {") ?? "", "border-radius: calc(var(--radius-s) + 2px)") && has(rule(chat, ".send {") ?? "", "width: 34px"));
check("chat: focus tints the composer's edge and rings it, as the prototype draws it", has(rule(chat, ".composer-box:focus-within {") ?? "", "border-color: color-mix(in srgb, var(--s-accent) 70%, transparent)") && has(rule(chat, ".composer-box:focus-within {") ?? "", "0 0 0 3px") && !has(aurora, "focus is a ring outside the pane"));
check("chat: the placeholder never takes a second line over the controls", has(rule(chat, ".composer-placeholder {") ?? "", "white-space: nowrap") && has(rule(chat, ".composer-placeholder {") ?? "", "text-overflow: ellipsis"));
check("chat: the narrow composer is the prototype's inset pane, placed after the base rule it must beat", chat.indexOf('@media (max-width: 767px) {\n  [data-part="composer"] {\n    padding: 0 10px 10px;') > chat.indexOf('\n[data-part="composer"] {\n') && !/@media \(max-width: 767px\) \{[\s\S]*?\n  \.ctl \{/.test(chat));
check("menu: the drop-up is the prototype's 300px panel", has(rule(chat, "\n.model-menu {") ?? "", "width: min(300px, calc(100vw - 2rem))") && !has(rule(chat, "\n.model-menu {") ?? "", "min-width"));
const nudgeHook = read("hooks/use-floater-nudge.ts");
check("floaters: the menu and the card hinge on their buttons, and slide back inside the box only where they would hang off it", has(rule(chat, "\n.model-menu {") ?? "", "right: 0") && !has(chat, ".model-select {\n    position: static;") && has(rule(chat, ".context-dock {") ?? "", "position: relative") && has(rule(chat, ".context-pop {") ?? "", "left: 0") && has(nudgeHook, "if (r.right > bounds.right - PAD) next = bounds.right - PAD - r.right;") && has(nudgeHook, "if (r.left + next < bounds.left + PAD) next = bounds.left + PAD - r.left;") && has(selector, "translate: nudge !== 0 ? `${nudge}px` : undefined,") && has(read("components/chat/context-meter.tsx"), "translate: nudge !== 0 ? `${nudge}px` : undefined,"));
check("menu: the residency column is fixed, so the dots stand in a line", has(rule(chat, ".model-opt > .model-where {") ?? "", "min-width: 4.5rem") && has(rule(chat, ".model-opt > .model-where {") ?? "", "justify-content: flex-start"));
check("chat: a lit chip stays lit under the pointer, a step brighter", has(rule(chat, '.ctl[aria-pressed="true"]:not(:disabled):hover {') ?? "", "color: var(--s-accent-hover)"));
check("cite: the marker fills under the pointer and takes the ink when open", has(rule(chat, ".cite-mark:hover {") ?? "", "var(--s-accent) 26%") && has(rule(chat, '.cite-mark[data-highlighted="true"] {') ?? "", "var(--s-accent) 30%") && has(rule(chat, '.cite-mark[data-highlighted="true"] {') ?? "", "color: var(--s-text)") && !has(rule(chat, '.cite-mark[data-highlighted="true"] {') ?? "", "box-shadow") && has(rule(aurora, 'html[data-theme="aurora"] .cite-mark:hover {') ?? "", "var(--s-accent) 26%") && has(rule(aurora, 'html[data-theme="aurora"] .cite-mark[data-highlighted="true"] {') ?? "", "var(--s-accent) 30%"));
const popover = read("components/chat/citation-popover.tsx");
check("cite: the card opens above its trigger and flips down only when the top is nearer", has(rule(chat, ".citation-popover {") ?? "", "bottom: calc(100% + 8px)") && !has(rule(chat, ".citation-popover {") ?? "", "top: calc") && has(rule(chat, '.citation-popover[data-flip="down"] {') ?? "", "top: calc(100% + 8px)") && has(popover, "const flippedDown = popRect.height + pad > spaceAbove && spaceBelow > spaceAbove;") && has(popover, 'data-flip={flippedDown ? "down" : undefined}'));
const stillInReduced = (sheet) => {
  const block = sheet.indexOf("@media (prefers-reduced-motion: reduce) {");
  const still = sheet.indexOf('html[data-theme="aurora"] .composer-box::after { animation: none; }');
  return block !== -1 && still > block && still < sheet.indexOf("\n}\n", block);
};
const halo = rule(aurora, 'html[data-theme="aurora"] .composer-box::after {') ?? "";
const haloOn = rule(aurora, 'html[data-theme="aurora"] .composer-box:focus-within::after {') ?? "";
check("glass: the focused composer breathes on aurora, fading in and out while its cycle pauses, still under reduced motion", has(halo, "animation: aurora-composer-breathe 5.2s ease-in-out infinite") && has(halo, "animation-play-state: paused") && has(halo, "transition: opacity 320ms") && has(halo, "opacity: 0") && has(haloOn, "opacity: 1") && has(haloOn, "animation-play-state: running") && has(aurora, "@keyframes aurora-composer-breathe") && stillInReduced(aurora));
check("chat: the toggle's glyph is the chrome's 16px", has(rule(chat, ".topbar-toggle .i {") ?? "", "width: 16px"));
check("chat: the chevron takes the chip's ink and the chosen row is the wash alone", has(rule(chat, ".model-trigger .chev {") ?? "", "color: inherit") && !has(rule(chat, '.model-opt[aria-checked="true"] {') ?? "", "box-shadow"));
check("menu: each row says where its model is, read when the menu opens", has(selector, "getResidency()") && has(selector, 'className="model-where"') && has(selector, '? "in memory" : "on disk"') && has(rule(chat, ".model-where[data-loaded=\"true\"]::before {") ?? "", "background: var(--s-ok)"));
check("menu: the row menu wears the drop-up's surface", has(rule(chat, '[data-part="convo-menu"] {') ?? "", "border: 1px solid var(--s-border-strong)") && has(aurora, 'html[data-theme="aurora"] [data-part="convo-menu"],'));
check("glass: the ring's glow is aurora's, not every theme's", !has(rule(chat, ".ring-fill {") ?? "", "drop-shadow") && has(rule(aurora, 'html[data-theme="aurora"] .ring-fill {') ?? "", "drop-shadow"));
check("chat: the narrow row drops the chips' labels and keeps their glyphs", /@media \(max-width: 767px\) \{[\s\S]*?\.ctl \.lbl,[\s\S]*?display: none;/.test(chat));

check("meter: the ring is the accent, and glows a little on aurora", has(rule(chat, ".context-ring {") ?? "", "color: var(--s-accent)") && has(rule(aurora, 'html[data-theme="aurora"] .ring-fill {') ?? "", "drop-shadow"));
check("meter: the figure appears once the window is filling", has(read("components/chat/context-meter.tsx"), '{level !== undefined && <span className="ring-pct">{percent}%</span>}'));
check("meter: nearly full is the danger hue", has(rule(chat, '.context-dock[data-level="high"] .context-ring {') ?? "", "color: var(--s-danger)"));
check("meter: the card's sentences read at the meta size under a label-sized head", has(rule(chat, ".context-pop p {") ?? "", "font-size: var(--text-meta)") && has(rule(chat, ".context-pop-head {") ?? "", "font-size: var(--text-label)"));
const bubbleTsx = read("components/chat/message-bubble.tsx");
const quoteChipTsx = read("components/chat/quote-chip.tsx");
check("quote: the sent chip rides inside the bubble as a band at its top, the composer's strip carried over", /<div data-part="bubble-user">\s*\{chip\}\s*\{message\.content\}/.test(bubbleTsx) && has(rule(chat, '[data-part="bubble-user"] > [data-part="quote-chip"] {') ?? "", "margin: -5px -10px 8px") && ["max-width: calc(100% + 20px)", "display: grid", "grid-template-columns: auto minmax(0, 1fr)", "row-gap: 0"].every((d) => has(rule(chat, '[data-part="bubble-user"] > [data-part="quote-chip"] {') ?? "", d)) && ["content: attr(data-text)", "grid-column: 1 / -1", "height: 0", "min-width: 0", "padding-left: 10px", "max-width: calc(28rem - 32px)", "overflow: hidden", "visibility: hidden", "white-space: nowrap"].every((d) => has(rule(chat, '[data-part="bubble-user"] > [data-part="quote-chip"]::before {') ?? "", d)) && ["contain: inline-size", "white-space: normal", "display: -webkit-box", "-webkit-box-orient: vertical", "-webkit-line-clamp: 2", "line-clamp: 2"].every((d) => has(rule(chat, '[data-part="bubble-user"] > [data-part="quote-chip"] > .quote-text {') ?? "", d)) && (quoteChipTsx.match(/data-text=\{text\}/g) ?? []).length === 2 && !has(chat, "question-stack") && !has(chat, ".turn > [data-part=\"quote-chip\"]"));
const messagesTs = read("lib/messages.ts");
const quoteJumpTs = read("hooks/use-quote-jump.ts");
const followTs = read("hooks/use-follow-stream.ts");
const passageTs = read("lib/passage-highlight.ts");
const selectionTsx = read("components/chat/selection-quote.tsx");
check("quote link: the band is a button to its source while the source is in the thread, greyed with the reason when it is not", has(quoteChipTsx, 'title="Go to the quoted message"') && has(quoteChipTsx, "onClick={onJump}") && has(quoteChipTsx, 'data-dead={gone !== undefined ? "true" : undefined}') && has(quoteChipTsx, "title={gone}") && has(page, "function quoteTarget(") && has(page, "The quoted reply is no longer in this conversation.") && has(page, "The quote has no record of the reply it was taken from.") && has(page, 'target !== null && "source" in target'));
check("quote link: turns carry their message id and the ring attribute, on both sides", (bubbleTsx.match(/data-message-id=\{message\.id\}/g) ?? []).length === 2 && (bubbleTsx.match(/data-lit=\{lit\}/g) ?? []).length === 2 && has(bubbleTsx, "lit?: LitKind;") && (bubbleTsx.match(/\{chip\}/g) ?? []).length === 2 && has(bubbleTsx, "const chip = message.quote && (") && has(bubbleTsx, "<QuoteChip text={message.quote.text} onJump={onQuoteJump} gone={quoteGone} />"));
check("quote link: a selection quote records the one turn it was taken from, and the Reply verb its message", has(selectionTsx, "messageId?: string") && has(selectionTsx, "startTurn !== null && startTurn === endTurn ? startTurn.dataset.messageId : undefined") && has(selectionTsx, "onQuote(pending.text, pending.messageId)") && has(page, "onQuote={(text, id) => quotePassage(text, id)}") && has(page, "() => quotePassage(m.content, m.id)"));
check("quote link: the glide to a place lands away unless that is the end, and only a landing runs the arrival", has(followTs, "glideTo: (target: () => number, onArrive?: () => void) => void;") && has(followTs, 'if (goalRef.current === "end" && gap() < 2) endGlide();') && has(followTs, 'goalRef.current === "end" && gap() < 2) endGlide();') && (followTs.match(/stuckRef\.current = gap\(\) < NEAR_END_PX;/g) ?? []).length === 2 && has(rule(followTs, "const cancelGlide = useCallback(() => {") ?? "", "arriveRef.current = null;") && has(followTs, 'goalRef.current = "end";') && has(followTs, 'goalRef.current = "place";'));
check("quote link: the landing lights the passage where its words are found, eased by hand, and rings the turn for a moment", has(quoteJumpTs, "export const LIT_MS = 1800") && has(quoteJumpTs, "findPassage(body, passage)") && has(quoteJumpTs, `el.querySelector('[data-part="content"], [data-part="bubble-user"]') ?? el`) && has(quoteJumpTs, "if (range === null) {\n        if (kind === \"whole\") layRailLight(el);\n        setLit({ id, kind });\n      } else lightPassage(range, el, LIT_MS);") && has(quoteJumpTs, "follow.glideTo(topFor(el, range), () => {") && has(quoteJumpTs, 'light(sourceId, el, range, passage === null ? "whole" : "ring");') && has(passageTs, '[data-part="quote-chip"]') && has(passageTs, "CSS.highlights.set(NAME, new Highlight(range))") && has(passageTs, "CSS.highlights.delete(NAME)") && has(passageTs, "a = easeInOut(t / RISE_MS)") && has(passageTs, 'if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {\n    at(1);') && has(passageTs, 'probe.style.color = "var(--c-passage-light)";') && has(passageTs, "paint.style.textDecorationColor = color;") && has(passageTs, 'passage.replace(/\\.\\.\\.$/, "")'));
check("quote link: the floater is the way back while there is one, rings the question it returns to, and forgets the way at the end", has(page, 'data-return={quoteJump.returnTo !== null ? "true" : undefined}') && has(page, "onClick={quoteJump.returnTo !== null ? quoteJump.returnToQuoting : jumpToBottom}") && has(page, 'aria-label={quoteJump.returnTo !== null ? "Back to your question" : "Jump to latest"}') && has(page, 'd="M3.5 3h5a3 3 0 0 1 3 3v6.5M8.5 10l3 3 3-3"') && has(quoteJumpTs, "if (!follow.away) setReturnTo(null);") && has(quoteJumpTs, "setReturnTo(follow.stuckRef.current ? null : (fromId ?? null));") && has(quoteJumpTs, "el.focus({ preventScroll: true });") && has(quoteChipTsx, 'onJump !== undefined ? "Go to the quoted message: " : gone !== undefined ? `${gone} ` : null') && has(rule(chat, ".turn:focus {") ?? "", "outline: none") && has(quoteJumpTs, 'follow.glideTo(topFor(el, null), () => light(id, el, null, "ring"))') && has(rule(chat, '[data-part="jump-to-latest"][data-return="true"] .jl-back {') ?? "", "opacity: 1") && has(rule(chat, '[data-part="jump-to-latest"][data-return="true"] .jl-down {') ?? "", "opacity: 0"));
check("quote link: a whole reply is told from a passage by its words being the source's own, and marked as a message: the sweep under the prose as light from the rail with the pane edge over it, the rail in the accent, a cut under reduced motion", has(messagesTs, "export function isWholeQuote(content: string, text: string): boolean {\n  return clampQuoteText(content) === text;\n}") && has(page, "isWholeQuote(target.content, target.text) ? null : target.text,") && has(page, "return { source: quote.message_id, content: source.content, text: quote.text };") && has(page, "quoteJump.lit !== null && quoteJump.lit.id === m.id\n                        ? quoteJump.lit.kind\n                        : undefined") && has(quoteJumpTs, "const range = passage !== null && canLightPassage() ? findPassage(body, passage) : null;") && has(quoteJumpTs, 'export type LitKind = "whole" | "ring";') && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]:not(:has(.citation-popover)) {') ?? "", "z-index: 0") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "z-index: -1") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "animation: lit-sweep 1800ms var(--ease-standard) both") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "mask-size: 100% 300%") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "background: linear-gradient(\n    100deg,\n    color-mix(in srgb, var(--c-passage-light) 84%, transparent),\n    color-mix(in srgb, var(--c-passage-light) 50%, transparent) 22%,") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "color-mix(in srgb, var(--c-passage-light) 6%, transparent)\n  );") && has(rule(chat, '.turn[data-lit="whole"] [data-part="content"]::before {') ?? "", "box-shadow: var(--lit-under, none)") && has(chat, "@keyframes lit-sweep {\n  0% {\n    mask-position: 0 100%;") && !has(chat, '.turn[data-lit="whole"] .step-glyph::after {') && !has(chat, "@keyframes lit-rail {") && !has(chat, "@media (prefers-reduced-motion: reduce) {\n  .turn[data-lit=\"whole\"] .step-glyph::after {"));
check("quote link: a marked message lands centred on its prose or bubble, not its turn, with its top kept below the margin, and a lit passage rests its first line on the landing line", has(quoteJumpTs, "const LAND_AT = 0.28") && has(quoteJumpTs, "const body = el.querySelector('[data-part=\"content\"], [data-part=\"bubble-user\"]') ?? el;\n      const r = body.getBoundingClientRect();") && has(quoteJumpTs, "const KEEP_TOP = 0.08") && has(quoteJumpTs, "return scroller.scrollTop + (rects[0].top - sr.top) - scroller.clientHeight * LAND_AT;") && has(quoteJumpTs, "const turnTop = scroller.scrollTop + (el.getBoundingClientRect().top - sr.top);\n      const centred = top + r.height / 2 - scroller.clientHeight / 2;\n      return Math.min(centred, turnTop - scroller.clientHeight * KEEP_TOP);"));
check("quote link: the rail's light travels with the wash as one front down the whole rail: the ring's bloom over each segment, the segment a masked column the core copies, every segment's mask the prose's slid by its measured offset", has(quoteJumpTs, "function layRailLight(turn: HTMLElement): void {") && has(quoteJumpTs, 'turn.style.setProperty("--rail-h", `${Math.max(1, r.height)}px`);') && has(quoteJumpTs, 'const inset = parseFloat(getComputedStyle(glyph, "::after").top) || 0;') && has(quoteJumpTs, 'glyph.style.setProperty("--seg-top", `${top}px`);') && has(quoteJumpTs, 'if (kind === "whole") layRailLight(el);') && has(rule(chat, ".step-glyph::after {") ?? "", "width: 30px;\n  margin-left: -15px;\n  pointer-events: none;\n  background: var(--s-border-strong);\n  mask-image: linear-gradient(\n    to right,\n    transparent 50%,\n    #000 50%,\n    #000 calc(50% + 1px),\n    transparent calc(50% + 1px)\n  );") && has(chat, "@keyframes lit-rail-light {\n  0% {\n    mask-position:\n      0 calc(-29 * var(--rail-h, 600px) - var(--seg-top, 0px)),\n      0 0;\n    opacity: 1;\n  }\n  38%,\n  72% {\n    mask-position:\n      0 calc(-27 * var(--rail-h, 600px) - var(--seg-top, 0px)),\n      0 0;\n    opacity: 1;") && has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "z-index: 1") && has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "width: 30px;\n  margin-left: -15px;") && has(chat, ".step-glyph::after,\n.step-glyph::before {\n  position: absolute;\n  left: 50%;\n  top: calc(var(--row-lh) / 2 + 14px);\n  bottom: calc(-1 * var(--step-gap) + 2px);\n}") && !has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "top:") && has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "animation: lit-rail-light 1800ms var(--ease-standard) both;\n  mask-image:\n    linear-gradient(to bottom, #000 93.3333%, transparent 95%),\n    linear-gradient(to bottom, transparent, #000 12px, #000 calc(100% - 12px), transparent);\n  mask-size:\n    100% calc(var(--rail-h, 600px) * 30),\n    100% 100%;\n  mask-repeat: no-repeat;\n  mask-composite: intersect;") && has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "color-mix(in srgb, var(--s-accent) 9%, transparent) 32%,\n      color-mix(in srgb, var(--s-accent) 26%, transparent) 50%,\n      color-mix(in srgb, var(--s-accent) 9%, transparent) 68%,") && has(rule(chat, '.turn[data-lit="whole"] .step-glyph::before {') ?? "", "transparent 50%,\n      var(--s-accent) 50%,\n      var(--s-accent) calc(50% + 1px),\n      transparent calc(50% + 1px)") && has(chat, ".step-end .step-glyph::after,\n.step-end .step-glyph::before,") && has(chat, ".step:has(+ .step-end) .step-glyph::after,\n.step:has(+ .step-end) .step-glyph::before {"));
check("quote link: the question returned to wears a glow that swells once over its edge light, on the sweep's clock, the ring rising and falling with it", has(rule(chat, '.turn[data-lit] [data-part="bubble-user"] {') ?? "", "animation: lit-glow 1800ms var(--ease-standard) both") && has(chat, "@keyframes lit-glow {") && (chat.match(/var\(--lit-under, 0 0 0 0 transparent\)/g) ?? []).length === 2 && has(chat, "@keyframes lit-glow {\n  0%,\n  100% {") && has(chat, "  0%,\n  100% {\n    outline-color: transparent;\n    box-shadow: var(--lit-under, 0 0 0 0 transparent), 0 0 0 0 transparent, 0 0 0 0 transparent;") && has(chat, "  22%,\n  72% {\n    outline-color: color-mix(in srgb, var(--s-accent) 55%, transparent);\n    box-shadow:\n      var(--lit-under, 0 0 0 0 transparent),\n      0 0 0 5px") && has(aurora, 'html[data-theme="aurora"] [data-part="bubble-user"],\n  html[data-theme="aurora"] [data-part="content"] {\n    --lit-under: var(--a-pane-edge);') && has(quoteJumpTs, "export const LIT_MS = 1800") && (chat.match(/ 1800ms var\(--ease-standard\) both/g) ?? []).length === 4 && has(chat, "@keyframes lit-rail-light {") && has(chat, "  100% {\n    mask-position:\n      0 calc(-27 * var(--rail-h, 600px) - var(--seg-top, 0px)),\n      0 0;\n    opacity: 0;") && has(chat, '.turn[data-lit="ring"] [data-part="content"] {\n  animation: lit-ring 1800ms var(--ease-standard) both;\n}') && has(chat, "@keyframes lit-ring {\n  0%,\n  100% {\n    outline-color: transparent;\n  }\n  22%,\n  72% {\n    outline-color: color-mix(in srgb, var(--s-accent) 55%, transparent);") && has(minimal, "--lit-under: inset 0 0 0 1px var(--s-border);"));
check("quote link: the pointer deepens the band and lights the mark, the press deepens it, the dead band greys, the ring is an outline that eases", has(rule(chat, 'button[data-part="quote-chip"] {') ?? "", "width: calc(100% + 20px)") && has(rule(chat, 'button[data-part="quote-chip"] {') ?? "", "cursor: pointer") && has(rule(chat, 'button[data-part="quote-chip"]:hover {') ?? "", "background: var(--c-quote-bg-hover)") && has(rule(chat, 'button[data-part="quote-chip"]:hover > .quote-mark {') ?? "", "background: var(--s-accent-hover)") && has(rule(chat, 'button[data-part="quote-chip"]:active {') ?? "", "background: var(--c-quote-bg-active)") && has(rule(chat, 'button[data-part="quote-chip"]:focus-visible {') ?? "", "outline: 2px solid var(--s-accent)") && has(rule(chat, '[data-part="quote-chip"][data-dead="true"] {') ?? "", "background: var(--c-bubble-user-bg)") && has(rule(chat, '[data-part="quote-chip"][data-dead="true"] > .quote-mark {') ?? "", "background: var(--s-text-3)") && has(chat, "transition: outline-color var(--dur-slow) var(--ease-standard);") && has(rule(chat, '.turn[data-lit] [data-part="bubble-user"],\n.turn[data-lit="ring"] [data-part="content"] {') ?? "", "outline-color: color-mix(in srgb, var(--s-accent) 55%, transparent)") && has(rule(chat, 'button[data-part="quote-chip"]::after {') ?? "", "inset: -4px 0") && has(rule(chat, 'button[data-part="quote-chip"] {') ?? "", "position: relative") && ["--c-quote-bg-hover:", "--c-quote-bg-active:", "--c-passage-light:"].every((t) => has(base, t) && has(aurora, t)) && has(minimal, "--c-passage-light:"));
check("quote: both surfaces stand on the quote tint, the bubble's pane with a touch more accent", has(rule(chat, '[data-part="quote-chip"] {') ?? "", "background: var(--c-quote-bg)") && has(base, "--c-quote-bg: color-mix(in srgb, var(--s-accent) 8%, var(--c-bubble-user-bg));") && has(aurora, "--c-quote-bg: color-mix(in srgb, var(--p-aurora-teal) 16%, var(--p-night-1));"));
check("meter: the card docks the later action under a hairline", has(read("components/chat/context-meter.tsx"), '<p className="context-later">{LATER}</p>') && has(rule(chat, ".context-pop .context-later {") ?? "", "border-top"));

check("fonts: the three faces ship with the app, none fetched at build time", has(read("app/layout.tsx"), 'from "geist/font/sans"') && has(read("app/layout.tsx"), 'from "next/font/local"') && !has(read("app/layout.tsx"), "next/font/google"));
check("fonts: the serif's files are the package's", has(read("app/layout.tsx"), "@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-normal.woff2"));
const globals = read("app/globals.css");
check("type: the chrome reads at 1.5 and the prose and bubble keep the reading measure", /body \{[^}]*line-height: 1\.5;/.test(globals) && has(rule(chat, '[data-part="bubble-user"] {') ?? "", "line-height: var(--text-body-line)") && has(chat, "line-height: var(--text-prose-line)"));
check("type: emphasis in the prose is a step up, not a shout", has(rule(chat, '[data-part="content"] strong,') ?? "", "font-weight: 600"));

check("sidebar: New chat is a quiet bordered verb", has(rule(chat, ".new-chat {") ?? "", "border: 1px solid var(--s-border-strong)") && !has(rule(chat, ".new-chat {") ?? "", "var(--s-accent)"));
check("sidebar: section labels are sentence case at 500", !has(rule(chat, ".side-label {") ?? "", "uppercase") && has(rule(chat, ".side-label {") ?? "", "font-weight: 500"));
check("sidebar: the current row and a hovered row take the raised fill", has(rule(chat, '.convo[aria-current="true"] {') ?? "", "background: var(--s-bg-raised)") && has(rule(chat, ".convo:hover {") ?? "", "background: var(--s-bg-raised)"));
check("sidebar: a row is 40px", has(rule(chat, ".convo {") ?? "", "min-height: 40px"));
check("sidebar: the brand is plain ink", has(rule(chat, ".brand {") ?? "", "color: var(--s-text)") && !/\.brand \{[^}]*background-clip/.test(aurora));
check("sidebar: the console row wears its glyph", /<Icon name="gear" \/>\s*Console/.test(read("components/sidebar.tsx")) && has(rule(chat, ".side-foot .convo > .i {") ?? "", "width: 13px"));
check("sidebar: Sign out meets the hit floor", has(rule(chat, ".user-chip-out {") ?? "", "min-height: 40px"));
check("sidebar: the toggle shows on every width and hides the panel on a desktop", has(rule(chat, ".topbar-toggle {") ?? "", "display: inline-flex") && /@media \(min-width: 768px\) \{\s*\.chat-shell\[data-nav-collapsed\] \[data-part="sidebar"\] \{\s*display: none;/.test(chat) && has(page, "setNavCollapsed((c) => !c)") && has(page, 'data-nav-collapsed={navCollapsed ? "true" : undefined}'));
check("chat: a chip borrows the 40px floor and the wrapped row's gap keeps two bands apart", has(rule(chat, ".ctl::after {") ?? "", "inset: -4px 0") && has(rule(chat, ".context-ring::before {") ?? "", "inset: -4px 0") && has(rule(chat, ".composer-row {") ?? "", "row-gap: 8px"));

const floatBg = aurora.match(/--c-float-bg: rgba\(16, 19, 46, (\.\d+)\);/);
const lift = aurora.match(/--c-menu-backdrop: blur\(16px\) saturate\(1\.35\) brightness\((1\.\d+)\);/);
check("glass: floaters carry a fuller fill and only a little lift, so their labels hold against the sky", floatBg !== null && Number(floatBg[1]) >= 0.6 && lift !== null && Number(lift[1]) <= 1.2);
check("glass: the user bubble is a solid navy pane", has(aurora, "--c-bubble-user-bg: color-mix(in srgb, var(--p-aurora-teal) 11%, var(--p-night-1));"));
check("glass: code and citation chips stand on solid ground", has(aurora, "--c-code-bg: #080a1a;") && has(aurora, "--c-citation-bg: var(--s-bg-panel);"));
check("glass: Send carries no glow", !/html\[data-theme="aurora"\] \.send \{/.test(aurora));
check("glass: jump-to-latest is neutral at rest", has(rule(aurora, 'html[data-theme="aurora"] [data-part="jump-to-latest"] {') ?? "", "color: var(--s-text-2)") && !has(rule(aurora, 'html[data-theme="aurora"] [data-part="jump-to-latest"] {') ?? "", "var(--s-accent)"));
check("glass: focus keeps the composer's edge under its ring", has(rule(aurora, 'html[data-theme="aurora"] .composer-box:focus-within {') ?? "", "var(--a-sky-edge), 0 0 0 3px"));

// ------------------------------------------------- per-conversation drafts
const { MAX_DRAFT_CHARS, MAX_DRAFTS, carryDraft, forgetAllDrafts, forgetDraft, readDraft, stopKeepingDrafts, writeDraft } =
  await import("../lib/drafts.ts");
/** A Storage stand-in over a Map; `refuse` makes the named calls throw. */
function fakeStore(seed = {}, refuse = []) {
  const map = new Map(Object.entries(seed));
  const guard = (name, fn) => (...args) => {
    if (refuse.includes(name)) throw new Error("storage refused");
    return fn(...args);
  };
  return {
    map,
    get length() {
      if (refuse.includes("length")) throw new Error("storage refused");
      return map.size;
    },
    key: guard("key", (i) => [...map.keys()][i] ?? null),
    getItem: guard("getItem", (k) => (map.has(k) ? map.get(k) : null)),
    setItem: guard("setItem", (k, v) => void map.set(k, String(v))),
    removeItem: guard("removeItem", (k) => void map.delete(k)),
  };
}
const drafts = fakeStore({ "app.theme": "aurora" });
writeDraft(drafts, "ann", "c1", "half a thought", 1);
writeDraft(drafts, "ann", null, "not sent anywhere yet", 2);
check("drafts: a conversation gives back what was typed in it", readDraft(drafts, "ann", "c1") === "half a thought");
check("drafts: another conversation has none of it", readDraft(drafts, "ann", "c2") === "");
check("drafts: the unbound pane keeps its own under a placeholder", readDraft(drafts, "ann", null) === "not sent anywhere yet" && drafts.map.has("chat:draft:ann:new"));
check("drafts: another user reads none of them", readDraft(drafts, "bob", "c1") === "" && readDraft(drafts, "bob", null) === "");
writeDraft(drafts, "ann", "c1", "   \n ", 3);
check("drafts: whitespace alone is no draft, and clears the kept one", readDraft(drafts, "ann", "c1") === "" && !drafts.map.has("chat:draft:ann:c1"));
writeDraft(drafts, "ann", "c1", "short", 4);
writeDraft(drafts, "ann", "c1", "x".repeat(MAX_DRAFT_CHARS + 1), 5);
check("drafts: one too long to keep leaves no stale shorter version", readDraft(drafts, "ann", "c1") === "");
writeDraft(drafts, "ann", "c1", "x".repeat(MAX_DRAFT_CHARS), 5);
check("drafts: the limit itself is kept", readDraft(drafts, "ann", "c1").length === MAX_DRAFT_CHARS);
carryDraft(drafts, "ann", null, "c9", "not sent anywhere yet", 6);
check("drafts: the placeholder's draft follows the pane to its conversation", readDraft(drafts, "ann", "c9") === "not sent anywhere yet" && readDraft(drafts, "ann", null) === "");
carryDraft(drafts, "ann", "c9", "c9", "", 7);
check("drafts: a binding to the same id touches nothing", readDraft(drafts, "ann", "c9") === "not sent anywhere yet");
writeDraft(drafts, "ann", null, "waiting in the new chat", 8);
carryDraft(drafts, "ann", "c9", null, "  ", 8);
check("drafts: an empty composer carried over forgets its own and erases nothing waiting there", readDraft(drafts, "ann", "c9") === "" && readDraft(drafts, "ann", null) === "waiting in the new chat");
forgetDraft(drafts, "ann", null);
forgetDraft(drafts, "ann", "c1");
check("drafts: forgetting one removes that one", readDraft(drafts, "ann", "c1") === "" && drafts.map.get("app.theme") === "aurora");

const crowded = fakeStore();
writeDraft(crowded, "bob", "theirs", "bob's own", 0);
for (let i = 0; i < MAX_DRAFTS; i += 1) writeDraft(crowded, "ann", "c" + i, "draft " + i, 100 + i);
const held = () => [...crowded.map.keys()].filter((k) => k.startsWith("chat:draft:ann:")).length;
check("drafts: a user holds up to the limit", held() === MAX_DRAFTS);
writeDraft(crowded, "ann", "c3", "draft 3, edited", 500);
check("drafts: editing one already held evicts nothing", held() === MAX_DRAFTS && readDraft(crowded, "ann", "c0") === "draft 0");
writeDraft(crowded, "ann", "one-more", "the newest", 600);
check("drafts: one more gives the oldest's place away", held() === MAX_DRAFTS && readDraft(crowded, "ann", "c0") === "" && readDraft(crowded, "ann", "c1") === "draft 1" && readDraft(crowded, "ann", "one-more") === "the newest");
check("drafts: another user's are never the ones evicted", readDraft(crowded, "bob", "theirs") === "bob's own");
crowded.map.set("chat:draft:ann:junk", "{not json");
writeDraft(crowded, "ann", "after-junk", "newer still", 700);
check("drafts: an entry it cannot read goes before any real draft", !crowded.map.has("chat:draft:ann:junk") && held() === MAX_DRAFTS && readDraft(crowded, "ann", "c2") === "draft 2" && readDraft(crowded, "ann", "after-junk") === "newer still");

const odd = fakeStore({ "chat:draft:ann:a": "{not json", "chat:draft:ann:b": '{"text":7,"at":1}', "chat:draft:ann:c": '"just a string"', "chat:draft:ann:d": '{"text":"no clock"}' });
check("drafts: an unreadable entry reads as no draft", ["a", "b", "c", "d"].every((id) => readDraft(odd, "ann", id) === ""));

const refusing = fakeStore({ "chat:draft:ann:c1": '{"text":"kept","at":1}' }, ["getItem", "setItem", "removeItem", "key", "length"]);
let quiet = true;
try {
  writeDraft(refusing, "ann", "c1", "new words", 9);
  carryDraft(refusing, "ann", null, "c1", "new words", 9);
  forgetDraft(refusing, "ann", "c1");
  forgetAllDrafts(refusing);
  quiet = readDraft(refusing, "ann", "c1") === "";
} catch {
  quiet = false;
}
check("drafts: a store that refuses is silence, never a throw", quiet);
const full = fakeStore({ "chat:draft:ann:c1": '{"text":"an older version","at":1}' }, ["setItem"]);
writeDraft(full, "ann", "c1", "what was just typed", 9);
check("drafts: a full store drops the entry rather than keep stale words", !full.map.has("chat:draft:ann:c1"));

const shared = fakeStore({ "app.theme": "aurora", "chat:draft:ann:c1": '{"text":"a","at":1}', "chat:draft:bob:new": '{"text":"b","at":1}', "chat:draft:bob:c2": '{"text":"c","at":1}' });
forgetAllDrafts(shared);
check("drafts: signing out clears every draft and nothing else", shared.map.size === 1 && shared.map.get("app.theme") === "aurora");

// The page's wiring, read as text: these pin that each site the pane's id
// changes still makes its call. What the calls DO is the store's, above, and
// the browser walk recorded with the change.
const authSrc = read("lib/auth.ts");
check("drafts: every edit is kept under the pane's conversation", has(page, "writeDraft(store, user, conversationIdRef.current, markdown);"));
check("drafts: nothing is read before the reader is known", has(page, "if (store === null || user === null) return;\n    composerRef.current?.setMarkdown(readDraft(store, user, id));"));
check("drafts: a fresh pane takes the new chat's own draft", has(page, "openDraft(null);\n    setPendingId(null);"));
check("drafts: an opened conversation shows its draft only once loaded", has(page, "loadSeqRef.current = 0;\n          adoptConversationId(id);") && has(page, "          openDraft(id);\n"));
check("page: opening a thread from the new chat docks the composer", has(page, 'if (readView() === "new" && hasTurns) {\n          await transitionView("composer", () => {'));
check("page: a load superseded inside the travel lands nothing", has(page, '            if (seq !== selectSeqRef.current) return;\n            setView("thread");\n            land();'));
check("page: any other opening lands at once", has(page, "        } else land();\n      } catch (e) {"));
check("drafts: the draft follows a pane that gains its conversation", has(page, "bindConversationId(summary.id);") && (page.match(/bindConversationId\(conversationIdRef\.current \?\? /g) ?? []).length === 3);
check("drafts: a deleted conversation takes its draft with it", has(page, "      dropDraft(id);\n      // Judged against the pane's id NOW"));
check("drafts: the session landing mid-load opens nothing for the pane being left", has(page, "} else if (loadSeqRef.current === 0 || loadSeqRef.current !== selectSeqRef.current) {\n      openDraft(conversationIdRef.current);"));
check("drafts: signing out clears them before the request can hang", authSrc.indexOf("forgetAllDrafts(store)") > 0 && authSrc.indexOf("forgetAllDrafts(store)") < authSrc.indexOf('await fetch("/api/auth/logout"'));
check("drafts: a load that fails leaves the pane its own draft", has(page, '          if (inputRef.current === "") openDraft(conversationIdRef.current);\n        }\n'));
// Last, because it is one-way for the page that ran it.
check("drafts: signing out also stops the page keeping more", authSrc.indexOf("stopKeepingDrafts();") > authSrc.indexOf("forgetAllDrafts(store)") && authSrc.indexOf("stopKeepingDrafts();") < authSrc.indexOf('await fetch("/api/auth/logout"'));
stopKeepingDrafts();
const afterSignOut = fakeStore();
writeDraft(afterSignOut, "ann", "c1", "typed while the sign-out was still going out", 1);
check("drafts: nothing is kept once the page has signed out", afterSignOut.map.size === 0);

if (failures.length > 0) {
  console.error(`chat surface: ${failures.length} failing`);
  for (const name of failures) console.error("  - " + name);
  process.exit(1);
}
console.log(`chat surface: ${passed} checks passed`);
