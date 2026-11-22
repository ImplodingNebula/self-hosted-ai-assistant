/**
 * Battery for the sky's pure half (`lib/sky.ts`) and the stylesheet rules
 * that keep its cost bounded.
 *
 * The sky is the one piece of the interface that runs a render loop, so the
 * things that bound its cost are worth pinning: which renderer a browser
 * gets, the frame caps, and the dossier's energy and tint per state (a quiet
 * edit to one number changes what every turn looks like). The stylesheet
 * half checks the surface rule the dossier calls "glass is three jobs":
 * exactly two blurs past 14px, both on the sidebar and the rail, and never a
 * hand-written `-webkit-backdrop-filter` (the compiler then emits ONLY the
 * prefixed form, which current Chrome does not implement).
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/sky-contract.mjs)
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
    `sky contract: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const {
  AMBER,
  approach,
  DANGER,
  FIELD_TALL,
  fieldY,
  FRAME_TOLERANCE_MS,
  frameDue,
  MAGENTA,
  PARALLAX,
  parallaxOffset,
  pickSkyRenderer,
  shaderFrameInterval,
  shaderSize,
  SKY_STATES,
  TEAL,
  VIOLET,
  weatherFor,
} = await import("../lib/sky.ts");

let failures = 0;
let checks = 0;

function check(label, actual, expected) {
  checks += 1;
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1;
    console.error(`FAIL  ${label}`);
    console.error(`  expected ${JSON.stringify(expected)}`);
    console.error(`  actual   ${JSON.stringify(actual)}`);
  }
}

/* ---------- energy and tint per state (DESIGN.md, "The sky") ---------- */

// Pinned by VALUE: the table in the dossier is the contract, and this is the
// copy the renderer reads.
const table = Object.fromEntries(
  Object.entries(SKY_STATES).map(([k, v]) => [k, [v.e, v.tint, v.err ?? 0, v.pulse === true]]),
);
check("sky: idle rests at .36 teal", table.idle, [0.36, TEAL, 0, false]);
check("sky: done is idle's energy with a pulse", table.done, [0.36, TEAL, 0, true]);
check("sky: waiting .48 teal", table.waiting, [0.48, TEAL, 0, false]);
check("sky: loading .30 amber", table.loading, [0.3, AMBER, 0, false]);
check("sky: thinking .92 violet", table.thinking, [0.92, VIOLET, 0, false]);
check("sky: tool .72 teal", table.tool, [0.72, TEAL, 0, false]);
check("sky: confirm .40 magenta", table.confirm, [0.4, MAGENTA, 0, false]);
check("sky: streaming .56 teal", table.streaming, [0.56, TEAL, 0, false]);
check("sky: error .30 washed toward danger", table.error, [0.3, DANGER, 1, false]);
check("sky: download .46 amber", table.download, [0.46, AMBER, 0, false]);
check("sky: an unknown state reads as idle", weatherFor("nonsense"), SKY_STATES.idle);
check("sky: no state reads as idle", weatherFor(null), SKY_STATES.idle);
// Thinking is the ceiling and the flat night the floor: nothing may outshine
// the one state the tell exists for, and nothing goes fully dark.
check(
  "sky: thinking is the brightest state",
  Object.values(SKY_STATES).every((w) => w.e <= SKY_STATES.thinking.e),
  true,
);
check(
  "sky: no state empties the sky",
  Object.values(SKY_STATES).every((w) => w.e >= 0.3),
  true,
);

/* ---------- which renderer a browser gets ---------- */

const capable = { reducedTransparency: false, saveData: false, deviceMemory: 8, webgl: true };
check("renderer: a capable browser gets the shader", pickSkyRenderer(capable), "shader");
check(
  "renderer: reduced transparency turns the sky off",
  pickSkyRenderer({ ...capable, reducedTransparency: true }),
  "off",
);
check(
  "renderer: reduced transparency wins even without WebGL",
  pickSkyRenderer({ ...capable, reducedTransparency: true, webgl: false }),
  "off",
);
check("renderer: no WebGL falls to the CSS curtains", pickSkyRenderer({ ...capable, webgl: false }), "css");
check("renderer: Save-Data turns the sky off", pickSkyRenderer({ ...capable, saveData: true }), "off");
check(
  "renderer: a 2 GB device gets the flat night",
  pickSkyRenderer({ ...capable, deviceMemory: 2 }),
  "off",
);
check(
  "renderer: unknown memory is not held against the browser",
  pickSkyRenderer({ ...capable, deviceMemory: null }),
  "shader",
);

/* ---------- pacing ---------- */

check("pace: 20 fps once settled at rest", shaderFrameInterval(true, 0.36), 50);
check("pace: 30 fps while the weather changes", shaderFrameInterval(false, 0.36), 1000 / 30);
check("pace: 30 fps while the turn is busy", shaderFrameInterval(true, 0.56), 1000 / 30);
// Display frames jitter: two 60Hz frames are 33.3ms give or take, and a cap
// compared strictly lets half of them through and waits a third frame for
// the rest, a judder on everything drawn. A frame a touch early is due.
check("pace: a frame may come a touch early and still be due", FRAME_TOLERANCE_MS >= 1 && FRAME_TOLERANCE_MS <= 4, true);
check("pace: two 60Hz frames are due at the 30 fps cap", frameDue(33.0, 1000 / 30), true);
check("pace: one 60Hz frame is not due at the 30 fps cap", frameDue(16.7, 1000 / 30), false);
check("pace: three 60Hz frames are due at the 20 fps cap", frameDue(49.5, 50), true);
check("pace: two 60Hz frames are not due at the 20 fps cap", frameDue(33.4, 50), false);
check("size: half resolution", shaderSize(1200, 800), [600, 400]);
check("size: capped at 960 wide", shaderSize(2560, 1440), [960, 540]);
check("size: never zero", shaderSize(1, 1), [1, 1]);
// The ease: a state change settles over about half a second (tau 520ms) and
// never overshoots - the dossier's "weather, not a cut".
check("ease: half a time constant covers about 39%", Math.round(approach(0, 1, 260) * 100), 39);
check("ease: three time constants are within 5%", approach(0, 1, 1560) > 0.95, true);
check("ease: no overshoot", approach(0, 1, 100000) <= 1, true);
check("ease: zero time moves nothing", approach(0.5, 1, 0), 0.5);

/* ---------- the star field's parallax ---------- */

// The field follows the thread at a tenth, upward, and is never capped: on a
// long conversation a cap is where every scroll would start, and the stars
// would stand still (the sky reads as broken).
check("field: the parallax is a tenth of the scroll, upward", parallaxOffset(1000), -100);
check("field: the parallax share is the constant the dossier names", PARALLAX, 0.1);
check("field: a tenth of a long thread is not capped", parallaxOffset(50000), -5000);
// The band is a quarter taller than the window, centred, and wraps.
check("field: the band is a quarter taller than the window", FIELD_TALL, 1.25);
check("field: a star at rest draws where it sits, less the band's half extra", fieldY(500, 0, 800), 400);
check("field: the band is centred: the top of the band is above the window", fieldY(0, 0, 800), -100);
check("field: a star carried past the top comes back in at the bottom", fieldY(50, -100, 800), 850);
check("field: a whole band of scroll lands where it started", fieldY(500, -1000, 800), 400);
check("field: a far scroll stays inside the band", (() => {
  const y = fieldY(500, -123456, 800);
  return y >= -100 && y < 900;
})(), true);
check("field: a zero height draws the star where it is", fieldY(42, -10, 0), 42);
// The renderer draws through these, so the pure rules are the ones in force.
const SKY = new URL("../components/sky.tsx", import.meta.url);
const sky = readFileSync(SKY, "utf8");
check("sky.tsx: stars are placed through fieldY", sky.includes("fieldY(st.y * H, oy * st.z, H, TALL)"), true);
check("sky.tsx: the parallax target is parallaxOffset", sky.includes("parallaxOffset(thread.scrollTop)"), true);
check("sky.tsx: no clamp on the parallax", /Math\.min\([^)]*scrollTop/.test(sky), false);
check("sky.tsx: both loops pace themselves through frameDue", (sky.match(/frameDue\(/g) ?? []).length, 2);
check("sky.tsx: no loop compares its cap strictly", /< STARS_FRAME_INTERVAL\b|< shaderFrameInterval\(/.test(sky), false);
// A wrapping band draws any row anywhere in time, so a star's place must be
// weighted by its distance from the centre column alone: a vertical edge
// term would seed full-width rows that then travel through the prose.
check("sky.tsx: stars are placed by horizontal distance from the centre", sky.includes("edge = Math.abs(x - 0.5) * 2 * (0.75 + 0.5 * Math.abs(y / TALL - 0.5))"), true);
check("sky.tsx: no vertical edge term in the placement", /edge = Math\.max\(/.test(sky), false);
check("sky.tsx: the shader paints its first curtain frame as it starts, before the stars' first", sky.includes("          draw();\n          raf = requestAnimationFrame(loop);"), true);
check("sky.tsx: a view change re-measures the pane before the chart draws", /if \(moved\) \{\s*if \(lastView === "new" && view === "thread" && !reduceMotion\.matches\) departAt = now;\s*lastView = view;\s*paneAt = -1e9;\s*pane\(now\);/.test(sky), true);
check("sky.tsx: a chart leaving for a thread is timed from the change, a cut under reduced motion", sky.includes('if (lastView === "new" && view === "thread" && !reduceMotion.matches) departAt = now;\n          lastView = view;'), true);
check("sky.tsx: the leaving chart fades over 260ms where it stood, from the brightness it had", sky.includes('if (view === "thread" && dd >= 0 && dd < 260) {') && sky.includes("let f = 1 - dd / 260; f = f * f * (3 - 2 * f);") && sky.includes("if (shown * f > 0.01) chart(ctx, t, sky, gx, gy, gR, shown * f);") && sky.includes("        shown = dim;\n      };"), true);
check("sky.tsx: the fade runs at the full frame rate", sky.includes("now - departAt < 300 ||"), true);
check("sky.tsx: a sky started again has no chart left to fade", /const start = \(\) => \{[^}]*?size\(\);[^}]*?shown = 0;/.test(sky), true);
check("sky.tsx: a chart arriving without a handoff fades in over 560ms, a cut under reduced motion", sky.includes('if ((view === "new" || view === "login") && !reduceMotion.matches && now - handoffAt > 1200) arriveAt = now;') && sky.includes("if (da < 560) { let f = da / 560; f = f * f * (3 - 2 * f); dim *= f; }"), true);
check("sky.tsx: the chart's anchors are followed with an ease, snapped on a view change, under reduced motion and through the handoff", sky.includes("const dt = Math.max(0, now - gAt);") && sky.includes("const k = snap || reduceMotion.matches || dt > 1000 ? 1 : 1 - Math.exp(-dt / 160);") && sky.includes("follow(now, g.x, g.y, g.R, moved || dt < 980);") && sky.includes("chart(ctx, t, sky, gx, gy, gR, dim);"), true);
// A line carried across the wrap fades rather than vanishing in one frame.
check("sky.tsx: a line cut by the wrap fades out", sky.includes("l.cut = now") && sky.includes("(now - l.cut) / 350"), true);

/* ---------- the stylesheet: glass is three jobs ---------- */

const AURORA = new URL("../app/themes/aurora.css", import.meta.url);
const css = readFileSync(AURORA, "utf8");

// Never hand-written: the compiler would dedupe the pair and keep only the
// prefixed form, which current Chrome does not implement. The components'
// inline React styles are a different path (no compiler) and are not read.
check(
  "css: no hand-written -webkit-backdrop-filter",
  (css.match(/-webkit-backdrop-filter/g) ?? []).length,
  0,
);

// Every blur radius the sheet declares. Over 14px is allowed only through the
// side filter token, and that token is read by exactly the sidebar's pseudo
// and the rail.
const blurs = [...css.matchAll(/backdrop-filter:\s*blur\((\d+)px\)/g)].map((m) => Number(m[1]));
check("css: every literal backdrop blur stays at or under 14px", blurs.every((px) => px <= 14), true);
const sideFilter = css.match(/--a-side-filter:\s*blur\((\d+)px\)/);
check("css: the side filter is the 22px exception", sideFilter && Number(sideFilter[1]), 22);
const sideUses = [...css.matchAll(/^([^\n]*)backdrop-filter:\s*var\(--a-side-filter\)/gm)].length;
check("css: the side filter is used by exactly the sidebar and the rail", sideUses, 2);
const sideRules = css.split("}").filter((r) => r.includes("backdrop-filter: var(--a-side-filter)"));
check(
  "css: the side filter's rules name only the sidebar and the console rail",
  sideRules.every((r) => /\[data-part="(sidebar|console-rail)"\]/.test(r)),
  true,
);

// The cards never blur: a backdrop-filter in the same rule as a card part is
// the old glass coming back.
const cardParts = ['[data-part="bubble-user"]', '[data-part="tool-trace"]', '[data-part="thinking"]',
  '[data-part="confirm"]', '[data-part="notice"]', ".mm-row", ".server-card", ".composer-box"];
const blurRules = css.split("}").filter((r) => /backdrop-filter:\s*(blur|var)/.test(r));
check(
  "css: no card and not the composer carries a backdrop-filter",
  blurRules.filter((r) => cardParts.some((p) => r.includes(p))).map((r) => r.trim().slice(0, 60)),
  [],
);

// Reduced transparency: the sidebar's and the rail's blur are switched off
// with !important, so no later layered rule can bring them back.
const reduced = css.slice(css.indexOf("prefers-reduced-transparency: reduce"));
check(
  "css: reduced transparency nulls the sidebar and rail blur",
  /\[data-part="sidebar"\]::before,[\s\S]*?\[data-part="console-rail"\],[\s\S]*?backdrop-filter: none !important/.test(reduced),
  true,
);
check(
  "css: reduced transparency makes the floaters solid",
  /--c-menu-backdrop: none;/.test(reduced) && /--c-float-bg: #[0-9a-f]{6};/.test(reduced),
  true,
);

// The veils key on the renderer being on, so the flat night never carries a
// shade over nothing.
const veils = [...css.matchAll(/html\[data-theme="aurora"\]([^{]*)(\.main-col|\.console-main|\[data-part="login"\])::before/g)];
// Three veils, and the new chat's smaller cut of the thread's.
check("css: three veils (chat, console, sign-in) and the new chat's cut", veils.length, 4);
check(
  "css: the new chat's veil is the thread's, smaller",
  veils.some((m) => m[1].includes('body[data-view="new"]') && m[2] === ".main-col"),
  true,
);
// The boot's pool of shade behind its words keys on the renderer too.
check(
  "css: the boot's pool is gated on the sky being on",
  /html\[data-theme="aurora"\]:not\(\[data-sky="off"\]\) \.boot::before/.test(css),
  true,
);
check(
  "css: every veil is gated on the sky being on",
  veils.every((m) => m[1].includes(':not([data-sky="off"])')),
  true,
);
// The veil's window travels with the composer box: its place is a set of
// registered lengths, eased only while the composer's transition runs.
check(
  "css: the window's place is four registered lengths",
  ["x", "b", "w", "h"].every((k) => css.includes("@property --pane-" + k + ' { syntax: "<length>"; inherits: true; initial-value: 0px; }')),
  true,
);
const windowTravel = css.slice(css.indexOf('html[data-theme="aurora"][data-vt="composer"] .main-col {')).split("}")[0];
check(
  "css: the window travels on the box's clock and curve",
  ["x", "b", "w", "h"].every((k) => windowTravel.includes("--pane-" + k + " 420ms cubic-bezier(0.2, 0.8, 0.2, 1)")),
  true,
);
check(
  "css: at rest the window is placed at once",
  [...css.matchAll(/transition:[^;]*--pane-/g)].length,
  1,
);
const veilWindow = readFileSync(new URL("../hooks/use-veil-window.ts", import.meta.url), "utf8");
check(
  "veil window: what places the box is watched by its border box",
  veilWindow.includes('const watch = (el: Element) => ro.observe(el, { box: "border-box" });'),
  true,
);
check(
  "veil window: the pane, the box and the pane's children are watched",
  veilWindow.includes("    watch(paneEl);\n    watch(boxEl);\n    for (const el of paneEl.children) watch(el);"),
  true,
);
check(
  "veil window: a child that arrives is watched, and the window re-placed at once",
  veilWindow.includes("if (node instanceof Element) watch(node);\n      }\n      measure();") && veilWindow.includes("mo.observe(paneEl, { childList: true });"),
  true,
);

/* ---------- the turn-state hook ---------- */

const PAGE = new URL("../app/page.tsx", import.meta.url);
const page = readFileSync(PAGE, "utf8");
// Every live state the dossier's table names for the thread is written from
// the stream handler; a state nobody sets is a sky that never answers it.
const handler = page.slice(page.indexOf("const consumeStream"), page.indexOf("const send = "));
for (const state of ["thinking", "tool", "confirm", "streaming", "error", "done"]) {
  check(`page: the stream handler writes "${state}"`, handler.includes(`setTurnState("${state}")`), true);
}
check("page: a send writes waiting", (page.match(/setTurnState\("waiting"\)/g) ?? []).length >= 3, true);
// A paused turn is not streaming, so the settle effect (not the handler) is
// what holds "confirm" for as long as the card waits, and what rests the sky
// once nothing is in flight.
check(
  "page: the settle effect holds confirm and rests the sky",
  /if \(awaitingConfirmation\) setTurnState\("confirm"\);[\s\S]*?setTurnState\("idle"\)/.test(page),
  true,
);

if (failures > 0) {
  console.error(`\nsky contract: ${failures} of ${checks} checks failed`);
  process.exit(1);
}
console.log(`sky contract: ${checks} checks passed`);
