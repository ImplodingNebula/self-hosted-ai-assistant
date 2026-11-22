/**
 * Battery for the boot screen and the new chat (DESIGN.md, "Sign-in, boot
 * and new chat"; "What travels between views").
 *
 * The pure half (`lib/boot.ts`) is run: the waits and their words, the
 * rail's lit share, the slow headline and its counter, the node details and
 * the reach line. The rest is pinned statically: the stylesheet rules that
 * make a new chat the greeting and the composer together, the boot's rail
 * and its narrow-screen stacking, the view-transition names carried only
 * while a transition runs, and the page's wiring (the boot gates the shell,
 * the first send and New chat travel the composer, the readiness relay).
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/boot-contract.mjs)
 *
 * It reports every failing case and exits non-zero if there was one. Like the
 * batteries beside it, it imports the app's TypeScript directly and so needs
 * a Node that strips types on its own - 22.18 or newer.
 */

import { readFileSync } from "node:fs";
import { register } from "node:module";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `boot contract: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const {
  backendDetail,
  BOOT_DEFER_MS,
  BOOT_SHOWN_MIN_MS,
  BOOT_WAITS,
  countingLabel,
  countOf,
  dataDetail,
  dwellFor,
  holdLeft,
  litShare,
  MODEL_MIN_MS,
  modelCheck,
  modelDetail,
  NODE_MIN_MS,
  paceFor,
  reachItems,
  READY_DWELL_MS,
  READY_HEAD,
  RETRY_EVERY_MS,
  sizeLabel,
  SLOW_AFTER_MS,
  slowHead,
  vectorStoreCheck,
  webSearchWord,
} = await import("../lib/boot.ts");

let failures = 0;
let checks = 0;

function check(label, actual, expected = true) {
  checks += 1;
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1;
    console.error(`FAIL ${label}`);
    console.error(`  expected ${JSON.stringify(expected)}`);
    console.error(`  actual   ${JSON.stringify(actual)}`);
  }
}

const read = (rel) => readFileSync(new URL(rel, import.meta.url), "utf8");
const has = (text, needle) => text.includes(needle);
const rule = (text, selector) => {
  const i = text.indexOf(selector);
  if (i === -1) return null;
  const open = text.indexOf("{", i);
  const close = text.indexOf("}", open);
  return text.slice(open + 1, close);
};

/* ---------- the waits ---------- */

check("waits: three, in the order the shell blocks", BOOT_WAITS.map((w) => w.node), ["backend", "model", "data"]);
check("waits: each names its lit word", BOOT_WAITS.map((w) => w.head[1]), ["backend", "model", "conversations"]);
check("waits: each has the sky's colour", BOOT_WAITS.map((w) => w.sky), ["boot", "boot-model", "boot-data"]);
check("waits: each says what a long wait means", BOOT_WAITS.every((w) => w.slow.length > 40));
check("waits: the ready headline is the one lit word", READY_HEAD, ["", "Ready", ""]);
check("pace: a wait is slow after about fifteen seconds", SLOW_AFTER_MS, 15000);
check("pace: a lit node stays lit for a beat", NODE_MIN_MS >= 300 && NODE_MIN_MS <= 900);
check("pace: the model's node stays lit long enough for its violet to land", MODEL_MIN_MS >= 900 && MODEL_MIN_MS <= 1500);
check("waits: each names its dwell, the model's the longer", BOOT_WAITS.map((w) => w.dwell), [NODE_MIN_MS, MODEL_MIN_MS, NODE_MIN_MS]);
check("pace: ready stands for a beat before the shell", READY_DWELL_MS >= 400 && READY_DWELL_MS <= 1200);
check("pace: a silent backend is asked again every two seconds", RETRY_EVERY_MS, 2000);
check("pace: the session's first load is full, every later one quick", [paceFor(false), paceFor(true)], ["full", "quick"]);
check("pace: a beat is held in full and dropped in quick", [dwellFor("full", NODE_MIN_MS), dwellFor("quick", NODE_MIN_MS), dwellFor("quick", READY_DWELL_MS)], [NODE_MIN_MS, 0, 0]);
check("pace: on a quick load the screen is painted only once a wait has run long", BOOT_DEFER_MS >= 250 && BOOT_DEFER_MS <= 600);
check("pace: a painted screen stands its least, never flickering past", BOOT_SHOWN_MIN_MS >= 500 && BOOT_SHOWN_MIN_MS <= 1200);
check("pace: a screen never painted has nothing to stand", holdLeft(null, 5000), 0);
check("pace: a screen painted just now stands the whole least", holdLeft(1000, 1000), BOOT_SHOWN_MIN_MS);
check("pace: a screen up for part of its least stands the rest", holdLeft(1000, 1000 + BOOT_SHOWN_MIN_MS - 200), 200);
check("pace: a screen up past its least stands no longer", holdLeft(1000, 1000 + BOOT_SHOWN_MIN_MS + 1), 0);

/* ---------- the rail ---------- */

check("rail: nothing lit before a wait ends", litShare(0), 0);
check("rail: one wait lights half the line", litShare(1), 0.5);
check("rail: two waits light the whole line", litShare(2), 1);
check("rail: the share never passes one", litShare(5), 1);
check("rail: a single-node rail lights whole", litShare(1, 1), 1);

/* ---------- the words ---------- */

check("slow: the headline keeps its lit word", slowHead(BOOT_WAITS[1].head), ["Still waking the ", "model", ""]);
check("slow: the counter counts seconds", countingLabel(17400), "17 s and counting");
check("slow: the counter never goes negative", countingLabel(-5), "0 s and counting");
check("count: plurals", [countOf(1, "document"), countOf(4, "conversation")], ["1 document", "4 conversations"]);
check("size: gigabytes to a decimal under ten", sizeLabel(3_100_000_000), "2.9 GB");
check("size: whole gigabytes past ten", sizeLabel(12 * 1024 ** 3), "12 GB");
check("size: megabytes below a gigabyte", sizeLabel(640 * 1024 ** 2), "640 MB");

const ready = {
  status: "ready",
  checks: [
    { name: "postgres", required: true, state: "ok", latency_ms: 1, detail: null },
    { name: "qdrant", required: true, state: "ok", latency_ms: 1, detail: null },
    { name: "ollama", required: false, state: "unreachable", latency_ms: 2000, detail: "Unreachable (ConnectError)." },
  ],
};
const unready = {
  status: "unready",
  checks: [{ name: "postgres", required: true, state: "unreachable", latency_ms: 2000, detail: null }],
};
check("backend: a ready report ends the wait", backendDetail(ready), { done: true, detail: "api ready" });
check("backend: an unready report names what is down and waits on", backendDetail(unready), { done: false, detail: "postgres unreachable" });
check("model: the report's model line is found by its runtime's name", modelCheck(ready)?.state, "unreachable");
check("model: a report without one is null", modelCheck(unready), null);
check("store: the vector store line is found", vectorStoreCheck(ready)?.state, "ok");
check("model detail: resident, with its size", modelDetail("qwen3:4b", { model: "qwen3:4b", total_bytes: 3_100_000_000 }), "qwen3:4b, 2.9 GB");
check("model detail: on disk", modelDetail("qwen3:4b", null), "qwen3:4b, on disk");
check("model detail: none chosen", modelDetail(null, null), "no model chosen");
check("data detail: both counts", dataDetail(4, 128), "4 conversations, 128 documents");
check("data detail: conversations alone when documents are unknown", dataDetail(1, null), "1 conversation");

/* ---------- the reach line ---------- */

const full = { documents: 128, truncated: false, knowledgeReady: true, webSearch: "auto", tools: 3 };
check(
  "reach: knowledge, web and tools, in the rail's glyphs",
  reachItems(full).map((i) => [i.glyph, i.label, i.value]),
  [["search", "Knowledge base", "128 documents"], ["globe", "Web search", "on"], ["plug", "Tools", "3 connected"]],
);
check("reach: auto reads as on, off as off", [webSearchWord("auto"), webSearchWord("on"), webSearchWord("off"), webSearchWord(null)], ["on", "on", "off", null]);
check("reach: a truncated count says so", reachItems({ ...full, truncated: true })[0].value, "128+ documents");
check("reach: one document", reachItems({ ...full, documents: 1 })[0].value, "1 document");
check("reach: a knowledge base the person may not list but which answers reads as ready", reachItems({ ...full, documents: null })[0].value, "ready");
check("reach: a knowledge base that neither answers nor lists is left out", reachItems({ ...full, documents: null, knowledgeReady: false }).map((i) => i.glyph), ["globe", "plug"]);
check("reach: no tools connected says so", reachItems({ ...full, tools: 0 })[2].value, "none connected");
check("reach: facts not learned leave their items out", reachItems({ documents: null, truncated: false, knowledgeReady: false, webSearch: null, tools: null }), []);

/* ---------- the stylesheets ---------- */

const chat = read("../app/themes/chat.css");
const aurora = read("../app/themes/aurora.css");
check("chat: a new chat hides the thread and the jump", /body\[data-view="new"\] \[data-part="thread"\],\s*body\[data-view="new"\] \[data-part="jump-to-latest"\]\s*\{\s*display: none/.test(chat));
check("chat: the greeting and the composer sit together above centre", has(rule(chat, ".welcome {") ?? "", "margin-top: auto") && has(rule(chat, 'body[data-view="new"] [data-part="composer"] {') ?? "", "margin-bottom: auto"));
check("chat: the greeting is the prose face", has(rule(chat, ".welcome-head {") ?? "", "font-family: var(--font-prose)"));
check("chat: the lit word is italic in the accent", has(rule(chat, ".welcome-head em,") ?? "", "color: var(--s-accent)"));
check("chat: the reach line uses the rail's tool colour", has(rule(chat, ".reach svg.i {") ?? "", "color: var(--c-step-tool)"));
check("chat: the shell is hidden beneath the boot, and before the view is written", /body\[data-view="boot"\] \.chat-shell,\s*body:not\(\[data-view\]\) \.chat-shell \{\s*display: none/.test(chat));
check("chat: the boot has the window", has(rule(chat, "\n.boot {") ?? "", "position: fixed"));
check("chat: the mark draws itself in, and reduced motion shows the mark whole", (() => {
  const r = rule(chat, ".boot-mark path {") ?? "";
  return has(r, "stroke-dasharray: 1") && has(r, "animation: boot-draw 1100ms var(--ease-standard) both") && !has(r, "stroke-dashoffset") && /@keyframes boot-draw \{\s*from \{\s*stroke-dashoffset: 1;/.test(chat);
})());
check("chat: the rail's lit line grows with --lit", has(rule(chat, "}\n.boot-rail::after {") ?? "", "var(--lit, 0)"));
check("chat: a live node spins and pulses", has(rule(chat, '.boot-node[data-step="live"] .boot-dot::before {') ?? "", "animation: boot-spin") && has(rule(chat, '.boot-node[data-step="live"] .boot-dot::after {') ?? "", "animation: boot-pulse"));
check("chat: a done node fills with its check", has(rule(chat, '.boot-node[data-step="done"] .boot-dot {') ?? "", "background: var(--s-accent)"));
check("chat: under 560px the chart goes and the rail stacks", /@media \(max-width: 560px\)[\s\S]*?\.boot-chart \{\s*display: none;[\s\S]*?\.boot-rail \{\s*grid-template-columns: 1fr/.test(chat));
check("chat: the composer is named only while its transition runs", has(rule(chat, 'html[data-vt="composer"] .composer-box {') ?? "", "view-transition-name: composer-box") && !/^\.composer-box \{[^}]*view-transition-name/m.test(chat));
check("chat: the composer travels in 420ms", has(rule(chat, "::view-transition-group(composer-box),") ?? "", "animation-duration: 420ms"));
check("chat: the root cross-fades in 260ms", has(rule(chat, "::view-transition-old(root),") ?? "", "animation-duration: 260ms"));
check("chat: behind the composer's travel the page is live, its old image dropped", has(rule(chat, 'html[data-vt="composer"]::view-transition-old(root) {') ?? "", "display: none") && has(rule(chat, 'html[data-vt="composer"]::view-transition-new(root) {') ?? "", "animation: none"));
check("chat: what the two views do not share is named only while the travel runs", [[".welcome", "pane-welcome"], ['[data-part="thread"]', "pane-thread"], ['[data-part="jump-to-latest"][data-visible="true"]', "pane-jump"], ['[data-part="topbar"]', "pane-topbar"], ['[data-part="composer"] > .hint', "composer-hint"]].every(([sel, name]) => has(rule(chat, 'html[data-vt="composer"] ' + sel + " {") ?? "", "view-transition-name: " + name)) && !/^(\.welcome|\[data-part="thread"\]|\[data-part="jump-to-latest"\]|\[data-part="topbar"\]|\.hint) \{[^}]*view-transition-name/m.test(chat));
check("chat: the greeting, the thread and the bar fade on the root's clock", has(chat, "::view-transition-old(pane-welcome),\n::view-transition-new(pane-welcome),\n::view-transition-old(pane-thread),\n::view-transition-new(pane-thread),\n::view-transition-old(pane-jump),\n::view-transition-new(pane-jump),\n::view-transition-old(pane-topbar),\n::view-transition-new(pane-topbar) {\n  animation-duration: 260ms;"));
check("chat: the bar stands over the thread through the travel", has(rule(chat, "::view-transition-group(pane-topbar) {") ?? "", "z-index: 1"));
check("chat: the line under the composer travels with the box, over the fading page", has(chat, "::view-transition-group(composer-box),\n::view-transition-group(composer-hint) {\n  z-index: 1;\n  animation-duration: 420ms;") && has(rule(chat, "::view-transition-old(composer-hint),\n::view-transition-new(composer-hint) {") ?? "", "object-fit: none"));
check("chat: the mark travels into the brand", has(chat, 'html[data-vt="boot"] .boot-mark,\nhtml[data-vt="boot"] .brand-mark {'));
check("chat: the headline becomes the greeting", has(chat, 'html[data-vt="boot"] .boot-head,\nhtml[data-vt="boot"] .welcome-head {'));
check("chat: each node becomes its reach item", [1, 2, 3].every((n) => has(chat, `html[data-vt="boot"] .boot-node:nth-child(${n}),\nhtml[data-vt="boot"] .reach > li:nth-child(${n}) {`)));
check("chat: the boot's movers take 560ms", has(rule(chat, "::view-transition-group(brand-mark),") ?? "", "animation-duration: 560ms"));
check("chat: text is never stretched to the moving box", has(rule(chat, "::view-transition-old(headline),\n::view-transition-new(headline) {") ?? "", "object-fit: none"));
check("chat: what a transition carries in has no entrance of its own", has(rule(chat, "html[data-vt] .welcome {") ?? "", "transition: none") && has(rule(chat, "html[data-vt] .boot-name,\nhtml[data-vt] .boot-rail {") ?? "", "animation: none"));
check("chat: the greeting's entrance can play only as it first renders", has(rule(chat, "\n.welcome {") ?? "", "opacity var(--dur-slow) var(--ease-entrance),\n    transform var(--dur-slow) var(--ease-entrance);") && !has(rule(chat, "\n.welcome {") ?? "", "animation:") && has(chat, "@starting-style {\n  .welcome {\n    opacity: 0;\n    transform: translateY(4px);\n  }\n}"));
check("chat: the jump button's group carries no frost of its own", has(rule(chat, "::view-transition-group(pane-jump) {") ?? "", "backdrop-filter: none"));
check("aurora: the new chat's veil is smaller", has(rule(aurora, 'body[data-view="new"] .main-col::before {') ?? "", "clamp(46rem, 88%, 74rem)"));
check("aurora: the boot's pool of shade", has(rule(aurora, ".boot::before {") ?? "", "radial-gradient"));
check("aurora: the mark keeps a halo", has(rule(aurora, ".boot-mark::before {") ?? "", "animation: boot-halo"));

/* ---------- the page, the components, the sky ---------- */

const page = read("../app/page.tsx");
const sky = read("../components/sky.tsx");
const boot = read("../components/boot.tsx");
const welcome = read("../components/chat/welcome.tsx");
const hook = read("../hooks/use-boot.ts");
const state = read("../lib/turn-state.ts");
check("page: the boot screen gates the shell, painted only when the hook says", has(page, "const boot = useBoot(booting || carriedIn);") && has(page, "{booting && boot.shown && <Boot state={boot} />}"));
check("page: leaving the chat page takes its view off the body", has(page, "useLayoutEffect(\n    () => () => {\n      clearView();\n      rememberConversation(conversationRef.current);\n    },\n    [],\n  );"));
check("page: a boot never painted moves no chart, the shell fades in on its own", has(page, "if (boot.shown) signalSkyHandoff();"));
check("hook: the paint timer dies with the waits, so no screen is painted in the frame before the shell", /phase: \(next\) => \{\s*if \(next === "done"\) window\.clearTimeout\(defer\);/.test(hook));
check("page: the handoff is one transition with the sky told first, and a quick load cross-fades the shell whole", /if \(boot\.shown\) signalSkyHandoff\(\);\s*void transitionView\(boot\.shown \? "boot" : "shell"/.test(page) && has(read("../lib/view-transition.ts"), 'export type TransitionKind = "composer" | "boot" | "shell" | "console";'));
check("page: the first send travels the composer and waits for the reveal before its request", /if \(readView\(\) === "new"\) \{\s*await transitionView\("composer", \(\) => \{\s*setView\("thread"\);\s*reveal\(\);/.test(page));
check("page: a second Enter in the reveal's frame does not send twice", has(page, "if (streaming || awaitingConfirmation || sendingRef.current) return;") && has(page, "sendingRef.current = true;") && has(page, "sendingRef.current = false;"));
check("page: New chat travels the composer back", /if \(readView\(\) === "thread"\) void transitionView\("composer", apply\)/.test(page));
check("page: the boot's weather is not overwritten by the turn's", /if \(booting\) return;\s*if \(awaitingConfirmation\) setTurnState\("confirm"\)/.test(page));
check("page: the greeting stands in the empty pane", has(page, "<Welcome items={reach}>{errorNote}</Welcome>"));
check("page: the view is written to the body", has(page, 'setView(booting ? "boot" : empty ? "new" : "thread")'));
check("boot: polite, and busy only while a wait runs quietly", has(boot, 'aria-live="polite"') && has(boot, 'aria-busy={state.phase === "booting" && state.slow === null}'));
check("boot: the headline replays its entrance per wait", has(boot, 'className="boot-head swap-in" key={headKey}'));
check("boot: the slow box offers a retry", has(boot, "onClick={state.retry}"));
check("welcome: the greeting's lit word", has(welcome, "What shall we shed <em>light</em> on?"));
check("hook: the real waits run through the runner with the real fetches", has(hook, "runBootWaits(") && has(hook, "readiness: getReadiness") && has(hook, "conversations: () => listConversations()"));
check("hook: a sleep ends early when the run is cancelled", has(hook, 'signal.addEventListener("abort", done, { once: true })'));
check("hook: the slow wait's skip reaches the screen", has(hook, "skippable: (fn) => setSkip(() => fn)") && has(boot, "onClick={state.skip}"));
check("hook: the pace is the session's, read once and kept through a retry", has(hook, "const p = (pace.current ??= paceFor(sessionSeen()));"));
check("hook: a full load paints the screen at once, a quick one only once a wait has run long", has(hook, 'const defer = window.setTimeout(paint, p === "quick" ? BOOT_DEFER_MS : 0);'));
check("hook: the screen is never in the server's HTML", !has(hook, "useLayoutEffect") && has(hook, "const [shown, setShown] = useState(false);"));
check("hook: the runner is told when the screen was painted", has(hook, "shownAt: () => shownAt.current,") && has(hook, "shownAt.current ??= Date.now();"));
check("hook: the session is marked seen for the next load", has(hook, 'window.sessionStorage.setItem(SEEN_KEY, "1");'));
check("state: the console is a view of its own", has(state, 'export type View = "boot" | "new" | "thread" | "console" | "login";') && has(state, 'v === "boot" || v === "new" || v === "console" || v === "login" ? v : "thread"'));
check("state: a view can be taken off the body", has(state, "export function clearView(): void") && has(state, "delete document.body.dataset.view;"));
check("console: the console names its view while mounted and clears it after", has(read("../app/console/layout.tsx"), '<ViewMarker view="console" />') && has(read("../components/view-marker.tsx"), "setView(view);\n    if (state !== undefined) setTurnState(state);\n    return () => {\n      clearView();"));
check("state: the boot's weather is a turn state", ['"boot"', '"boot-model"', '"boot-data"', '"boot-ready"', '"boot-slow"'].every((s) => has(state, `  | ${s}`)));
check("sky: the chart is drawn on the boot's own canvas above its pool", has(sky, 'document.querySelector<HTMLCanvasElement>(".boot-chart")'));
check("sky: the handoff fades the chart where it was, rests, and fades it in where it lives", has(sky, "if (dt < 260)") && has(sky, "else if (dt < 420) dim = 0;") && has(sky, "f = (dt - 420) / 560"));
check("sky: the chart is drawn before the stars", /drawChart\(now, t, sky\);[\s\S]*?drawStars\(now, t, sky\);/.test(sky));
check("sky: the handoff is heard", has(sky, "document.addEventListener(SKY_HANDOFF_EVENT, onHandoff)"));

/* ---------- the runner, with the world faked ----------
   The waits are run for real against fake fetches and a fake clock: every
   sleep takes one turn of the event loop and moves the clock by the time
   it asked for, so the slow verdict and the optional timeouts arrive at
   once and in order. A watchdog cancels a run that never ends, so a fault
   that stalls the boot reads as a failure instead of hanging the battery. */

const { NO_REACH, OPTIONAL_TIMEOUT_MS, runBootWaits } = await import("../lib/boot-runner.ts");

const report = (model = "ok", required = "ok") => ({
  status: required === "ok" ? "ready" : "unready",
  checks: [
    { name: "postgres", required: true, state: required, latency_ms: 1, detail: null },
    { name: "qdrant", required: true, state: "ok", latency_ms: 1, detail: null },
    { name: "ollama", required: false, state: model, latency_ms: 1, detail: model === "ok" ? null : "Unreachable (ConnectError)." },
  ],
});

function harness(overrides = {}, sinkOverrides = {}) {
  let clock = 0;
  const log = [];
  const sleeps = [];
  const ac = new AbortController();
  const sink = {
    node: (i, phase, detail) => log.push(["node", i, phase, detail]),
    head: (line) => log.push(["head", line.join("")]),
    sub: (text) => log.push(["sub", text]),
    slow: (copy) => log.push(["slow", copy !== null]),
    weather: (state) => log.push(["weather", state]),
    reach: (reach) => log.push(["reach", reach]),
    phase: (phase) => log.push(["phase", phase]),
    skippable: (fn) => log.push(["skippable", fn !== null]),
    ...sinkOverrides,
  };
  const deps = {
    readiness: async () => report(),
    defaults: async () => ({ model: "qwen3:4b", web_search: "auto" }),
    residency: async () => ({ reported: true, models: [{ model: "qwen3:4b", total_bytes: 3_100_000_000 }] }),
    conversations: async () => ({ items: [1, 2, 3, 4] }),
    knowledge: async () => ({ items: new Array(128).fill(0), truncated: false }),
    servers: async () => ({ servers: [{ enabled: true }, { enabled: false }] }),
    sleep: (ms, signal) =>
      new Promise((resolve) => {
        sleeps.push(ms);
        setTimeout(() => {
          if (!signal.aborted) clock += ms;
          resolve();
        }, 0);
      }),
    now: () => clock,
    pace: "full",
    shownAt: () => 0,
    ...overrides,
  };
  // runs the waits; "done" when they ended, "aborted" when the watchdog had to cancel them
  const settle = async (ms = 3000) => {
    const t = setTimeout(() => ac.abort(), ms);
    try {
      await runBootWaits(deps, sink, ac.signal);
      return "done";
    } catch (e) {
      return ac.signal.aborted ? "aborted" : `threw: ${e.message}`;
    } finally {
      clearTimeout(t);
    }
  };
  const dones = () => log.filter((e) => e[0] === "node" && e[2] === "done").map((e) => e[3]);
  const weather = () => log.filter((e) => e[0] === "weather").map((e) => e[1]);
  const heads = () => log.filter((e) => e[0] === "head").map((e) => e[1]);
  const subs = () => log.filter((e) => e[0] === "sub").map((e) => e[1]);
  const phases = () => log.filter((e) => e[0] === "phase").map((e) => e[1]);
  const lastReach = () => log.filter((e) => e[0] === "reach").at(-1)?.[1];
  return { ac, log, sleeps, settle, dones, weather, heads, subs, phases, lastReach };
}

// 1. a run-through
{
  const h = harness();
  check("runner: a run-through ends", await h.settle(), "done");
  check("runner: the three nodes end in order with their details", h.dones(), ["api ready", "qwen3:4b, 2.9 GB", "4 conversations, 128 documents"]);
  check("runner: the sky takes each wait's colour, then the pulse", h.weather(), ["boot", "boot-model", "boot-data", "boot-ready"]);
  check("runner: the headlines name each wait, then ready", h.heads(), ["Reaching the backend", "Waking the model", "Gathering your conversations", "Ready"]);
  check("runner: booting, ready, done", h.phases(), ["booting", "ready", "done"]);
  check("runner: every lit node stays lit for the beat, the model's longer so its violet lands", [h.sleeps.filter((ms) => ms === NODE_MIN_MS).length, h.sleeps.filter((ms) => ms === MODEL_MIN_MS).length], [2, 1]);
  check("runner: ready stands its beat", h.sleeps.includes(READY_DWELL_MS));
  check("runner: the reach line learns every count", h.lastReach(), { documents: 128, truncated: false, knowledgeReady: true, webSearch: "auto", tools: 1 });
  check("runner: nothing may be skipped on a run-through", h.log.some((e) => e[0] === "skippable" && e[1] === true), false);
}
// 2. a silent backend is asked again until it answers
{
  let calls = 0;
  const h = harness({ readiness: async () => { calls += 1; if (calls < 3) throw new Error("no report (503)"); return report(); } });
  check("runner: a silent backend is asked again", await h.settle(), "done");
  check("runner: the backend was asked three times, two seconds apart", [calls, h.sleeps.filter((ms) => ms === RETRY_EVERY_MS).length], [3, 2]);
}
// 3. an unready report names what is down, then clears
{
  let calls = 0;
  const h = harness({ readiness: async () => { calls += 1; return calls < 2 ? report("ok", "unreachable") : report(); } });
  check("runner: an unready report is waited out", await h.settle(), "done");
  check("runner: the subline named the service that was down", h.subs().includes("postgres unreachable"));
}
// 4. a deployment without a model runtime is a normal deployment
{
  const h = harness({ readiness: async () => report("unconfigured") });
  check("runner: an unconfigured runtime ends the wait", await h.settle(), "done");
  check("runner: the node says the runtime is not configured", h.dones()[1], "model runtime not configured");
  check("runner: no asking again for a runtime that is not there", h.sleeps.includes(RETRY_EVERY_MS), false);
}
// 5. an unreachable runtime is asked again, called slow, and may be given up on
{
  // the person gives up a few turns after the offer, while the asking goes on
  const h = harness({ readiness: async () => report("unreachable") }, { skippable: (fn) => { if (fn !== null) setTimeout(fn, 8); } });
  check("runner: an unreachable runtime can be gone on without", await h.settle(), "done");
  check("runner: the node says the runtime was unreachable", h.dones()[1], "model runtime unreachable");
  check("runner: the wait went slow first, with the sky amber", h.heads().includes("Still waking the model") && h.weather().includes("boot-slow"));
  check("runner: the slow box counted", h.subs().some((s) => /\d+ s and counting/.test(s)));
  const slowAt = h.log.findIndex((e) => e[0] === "slow" && e[1] === true);
  const doneAt = h.log.findIndex((e, i) => i > slowAt && e[0] === "node" && e[2] === "done");
  const whileSlow = h.log.slice(slowAt, doneAt).filter((e) => e[0] === "sub").map((e) => e[1]);
  check("runner: once slow, the count is not overwritten by the asking", whileSlow.length > 0 && whileSlow.every((s) => /\d+ s and counting/.test(s)));
  check("runner: the subline names the runtime and its state, not the report's developer string", h.subs().includes("ollama unreachable") && !h.subs().some((s) => /ConnectError/.test(s)));
  check("runner: the shell still arrives", h.phases().at(-1), "done");
}
// 6. the backend's wait cannot be skipped: it goes slow and keeps asking
{
  const h = harness({ readiness: async () => { throw new Error("no report (503)"); } }, { skippable: (fn) => { if (fn !== null) throw new Error("the backend was offered as skippable"); } });
  check("runner: a backend that never answers is waited on until cancelled", await h.settle(400), "aborted");
  check("runner: the wait was called slow", h.heads().includes("Still reaching the backend") && h.log.some((e) => e[0] === "slow" && e[1] === true));
  const at = h.log.length;
  await new Promise((r) => setTimeout(r, 30));
  check("runner: nothing is written after the run is cancelled", h.log.length, at);
}
// 7. an answer the server meant ends the list's wait with nothing: the shell owns sign-in
{
  const h = harness({ conversations: async () => { const e = new Error("auth"); e.status = 401; throw e; } });
  check("runner: a 401 on the list does not loop", await h.settle(), "done");
  check("runner: the node says the list could not be read", h.dones()[2], "could not be listed");
  check("runner: a 401 is not asked again", h.sleeps.includes(RETRY_EVERY_MS), false);
}
// 8. a hung optional fact is given its seconds, then done without
{
  const h = harness({ knowledge: () => new Promise(() => {}), residency: () => new Promise(() => {}) });
  check("runner: a hung optional fact does not stall the boot", await h.settle(), "done");
  check("runner: the facts not learned are left out", [h.dones()[1], h.dones()[2]], ["qwen3:4b, on disk", "4 conversations"]);
  check("runner: the optional facts were given their seconds", h.sleeps.includes(OPTIONAL_TIMEOUT_MS));
  check("runner: the reach line leaves the unknown count out", h.lastReach().documents, null);
}
// 9. a defaults failure is not a stalled boot
{
  const h = harness({ defaults: async () => { throw new Error("503"); } });
  check("runner: a failed defaults read still boots", await h.settle(), "done");
  check("runner: with no default named, the node says so", h.dones()[1], "no model chosen");
}
// 10. a run cancelled during a node's beat writes nothing more, not even the node's end
{
  let abortNow = () => {};
  const h = harness({
    sleep: (ms) => {
      if (ms === NODE_MIN_MS) abortNow();
      return new Promise((resolve) => setTimeout(resolve, 0));
    },
  });
  abortNow = () => h.ac.abort();
  check("runner: a run cancelled mid-beat ends quietly", await h.settle(), "done");
  check("runner: the node whose beat was cut is never marked done", h.dones(), []);
  check("runner: the shell is never told to show", h.phases().includes("done"), false);
}
// 11. a quick load against a backend that answers at once: no beat is held
{
  const h = harness({ pace: "quick", shownAt: () => null });
  check("runner: a quick load runs through", await h.settle(), "done");
  check("runner: a quick load holds no beat and lets ready stand none", h.sleeps.filter((ms) => ms === NODE_MIN_MS || ms === MODEL_MIN_MS || ms === READY_DWELL_MS), []);
  check("runner: a quick load still ends every node with its detail", h.dones().length, 3);
  check("runner: a quick load still says booting, ready, done", h.phases(), ["booting", "ready", "done"]);
}
// 12. a quick load whose screen was painted: it stands its least before the shell
{
  const h = harness({ pace: "quick", shownAt: () => 100, now: () => 300 });
  check("runner: a painted quick boot runs through", await h.settle(), "done");
  check("runner: a painted quick boot stands the rest of its least before done", h.sleeps.at(-1), BOOT_SHOWN_MIN_MS - 200);
  check("runner: that stand is after ready and before done", h.phases(), ["booting", "ready", "done"]);
}
// 13. NO_REACH is the empty reach
check("runner: the empty reach has nothing learned", NO_REACH, { documents: null, truncated: false, knowledgeReady: false, webSearch: null, tools: null });

/* ---------- the relay ---------- */

const relay = read("../app/api/ready/route.ts");
check("relay: /api/ready forwards the backend's /readyz", has(relay, "`${BACKEND_URL}/readyz`"));
check("relay: a 503 report is relayed, not swallowed", has(relay, "return relay(upstream);"));

if (failures > 0) {
  console.error(`boot contract: ${failures} of ${checks} checks failed`);
  process.exit(1);
}
console.log(`boot contract: ${checks} checks passed`);
