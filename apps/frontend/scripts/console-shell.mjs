/**
 * Battery for the console shell and the sign-in page, as the dossier draws
 * them (DESIGN.md "Console shell", "Sign-in, boot and new chat", "Rows and
 * pills"; the aurora-next prototype's console and sign-in views).
 *
 * What is pinned here fails SILENTLY when it goes: a rail a few rem narrower
 * still works, a chevron that drifts inside the card still opens the row, a
 * comet that never runs still leaves a download that finishes, and a sky
 * that no longer follows the console's one long operation is simply a sky.
 * None of it is behaviour a type or a lint sees, so the rules and the wiring
 * are read from the sources the way `proxy-contract.mjs` reads the route
 * files - this workspace has no component runner.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/console-shell.mjs)
 *
 * It reports every failing case and exits non-zero if there was one.
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
    `console shell: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const { loginHref } = await import("../lib/auth.ts");
const { SKY_STATES } = await import("../lib/sky.ts");

let failures = 0;
let checks = 0;

function check(label, actual, expected = true) {
  checks += 1;
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1;
    console.error(`FAIL  ${label}`);
    console.error(`  expected ${JSON.stringify(expected)}`);
    console.error(`  actual   ${JSON.stringify(actual)}`);
  }
}

const read = (rel) => readFileSync(new URL(rel, import.meta.url), "utf8");
const has = (text, needle) => text.includes(needle);
/* the body of the first rule whose selector line is `head` */
function rule(css, head) {
  // anchored at a line start, so `.mm-meta {` is not found inside `.x .mm-meta {`
  const i = css.indexOf("\n" + head);
  if (i < 0) return null;
  const at = i + 1;
  const close = css.indexOf("\n}", at);
  return close < 0 ? null : css.slice(at, close);
}

const consoleCss = read("../app/themes/console.css");
const auth = read("../app/themes/auth.css");
const aurora = read("../app/themes/aurora.css");
const chat = read("../app/themes/chat.css");
const base = read("../app/themes/tokens.base.css");
const integrations = read("../app/themes/integrations.css");
const minimal = read("../app/themes/minimal.css");
const login = read("../app/login/page.tsx");
const models = read("../app/console/models/page.tsx");
const panel = read("../components/console/activity-panel.tsx");
const sky = read("../components/sky.tsx");
const marker = read("../components/view-marker.tsx");
const turnState = read("../lib/turn-state.ts");
const authTs = read("../lib/auth.ts");
const transition = read("../lib/view-transition.ts");
const travel = read("../components/travel-link.tsx");
const sidebar = read("../components/sidebar.tsx");

/* ---------- the shell ---------- */

const rail = rule(consoleCss, '[data-part="console-rail"] {') ?? "";
check(
  "rail: the sidebar's width, so the two swap in place",
  has(rail, "width: 272px;") && has(rail, "padding: 14px 10px;"),
);
const entry = rule(consoleCss, ".console-rail-item {") ?? "";
check(
  "rail: entries at the 40px floor in the label size",
  has(entry, "min-height: 40px;") &&
    has(entry, "padding: 0 12px;") &&
    has(entry, "font-size: var(--text-label);"),
);
check(
  "rail: the current entry is marked by fill AND weight, on the raised pane",
  has(
    rule(consoleCss, '.console-rail-item[aria-current="page"] {') ?? "",
    "background: var(--s-bg-raised);\n  color: var(--s-text);\n  font-weight: 500;",
  ),
);
check(
  "rail: hover lifts to the raised pane, never the inset cut-out",
  has(
    rule(consoleCss, ".console-rail-item:hover {") ?? "",
    "background: var(--s-bg-raised);",
  ),
);
check(
  "topbar: the back verb is a 32px quiet chip that lifts on hover",
  has(rule(consoleCss, ".console-back {") ?? "", "min-height: 32px;") &&
    has(
      rule(consoleCss, ".console-back:hover {") ?? "",
      "background: var(--s-bg-raised);",
    ),
);
check(
  "page: the work column's padding",
  has(rule(consoleCss, ".console-scroll {") ?? "", "padding: 32px 28px 72px;"),
);
check(
  "page: every page's h1 is the serif display at 30px, one rule for three",
  has(
    consoleCss,
    ".mm-title,\n.int-title,\n.mbr-title {\n  font-family: var(--font-prose);\n  font-size: 30px;\n  font-weight: 400;",
  ) &&
    !has(integrations, ".int-title {\n  font-size") &&
    !has(consoleCss, ".mbr-title {\n  font-size"),
);
check(
  "page: the meta line is mono and tabular",
  has(rule(consoleCss, ".mm-meta {") ?? "", "font-family: var(--font-mono);") &&
    has(
      rule(consoleCss, ".mm-meta {") ?? "",
      "font-variant-numeric: tabular-nums;",
    ),
);
check(
  "loaded now: a dot and a tag per row, the size at the right",
  has(rule(consoleCss, ".mm-loaded {") ?? "", "min-width: 16rem;") &&
    has(
      rule(consoleCss, ".mm-loaded-row .mm-meta {") ?? "",
      "margin-left: auto;",
    ) &&
    has(
      models,
      '<span className="ms-dot" aria-hidden="true" />\n              <code className="mm-tag">{entry.model}</code>',
    ),
);

/* ---------- the rows ---------- */

check(
  "row: the whole head tints under the pointer",
  has(
    rule(consoleCss, ".mm-head:hover {") ?? "",
    "background: color-mix(in srgb, var(--s-text) 4%, transparent);",
  ),
);
const summary = rule(consoleCss, ".mm-summary,\n.mm-head.static {") ?? "";
check(
  "row: the summary is a grid with a reserved slot and the chevron's column last",
  has(
    summary,
    "grid-template-columns: minmax(0, 1fr) auto auto var(--slot, 6.5rem) 16px;",
  ) && has(summary, "min-height: 52px;"),
);
check(
  "row: the chevron sits in the last column, at the card's right edge",
  has(rule(consoleCss, ".mm-chevron {") ?? "", "grid-column: 5;") &&
    has(rule(consoleCss, ".mm-chevron-space {") ?? "", "grid-column: 5;"),
);
const action = rule(consoleCss, ".mm-action {") ?? "";
check(
  "row: the action overlays the slot from outside the trigger, and an empty slot lets a click fall through",
  has(action, "position: absolute;") &&
    has(action, "right: calc(14px + 16px + 12px);") &&
    has(action, "width: 6.5rem;") &&
    has(action, "pointer-events: none;") &&
    has(rule(consoleCss, ".mm-action > * {") ?? "", "pointer-events: auto;"),
);
check(
  "row: narrow, the slot keeps its column only while something is in it",
  has(
    consoleCss,
    "  .mm-head:has(.mm-action > *) .mm-summary {\n    --slot: 6.5rem;\n  }",
  ) &&
    has(
      consoleCss,
      "  .mm-chevron,\n  .mm-chevron-space {\n    grid-column: 3;\n    grid-row: 1 / span 3;\n  }",
    ),
);
check(
  "pill: the dot carries the hue; a muted state is a ring",
  has(
    rule(consoleCss, ".mm-state::before {") ?? "",
    "box-shadow: inset 0 0 0 1.5px var(--s-text-3);",
  ) &&
    has(
      rule(consoleCss, ".mm-state.ok::before {") ?? "",
      "background: var(--s-ok);\n  box-shadow: none;",
    ) &&
    has(
      rule(consoleCss, ".mm-state.warn {") ?? "",
      "color: color-mix(in srgb, var(--s-warn) 70%, var(--s-text));",
    ),
);
check(
  "pill: the ceiling the dossier sets",
  has(rule(consoleCss, ".mm-state {") ?? "", "min-width: 7.5rem;"),
);
check(
  "tag: a mono chip on the panel fill",
  has(
    rule(consoleCss, ".mm-tag {") ?? "",
    "border-radius: 5px;\n  background: var(--s-bg-panel);",
  ),
);
check(
  "badge: sentence case at the text tier, 18px",
  has(rule(integrations, ".badge {") ?? "", "height: 18px;") &&
    !has(rule(integrations, ".badge {") ?? "", "uppercase"),
);
check(
  "chip: a 24px outlined chip at the radius, not a pill",
  has(
    rule(consoleCss, ".mm-chip {") ?? "",
    "height: 24px;\n  padding: 0 9px;\n  border-radius: var(--radius-s);",
  ),
);
check(
  "readout: the row's live readout is 6rem, mono, in the text ink, its bar glowing",
  has(rule(consoleCss, ".mm-pullmini {") ?? "", "width: 6rem;") &&
    has(
      rule(consoleCss, ".mm-pullmini {") ?? "",
      "font-family: var(--font-mono);",
    ) &&
    has(
      aurora,
      'html[data-theme="aurora"] .mm-minibar i { box-shadow: 0 0 8px color-mix(in srgb, var(--s-accent) 70%, transparent); }',
    ),
);
check(
  "bar: the panel's bar glows and glides between frames",
  has(
    aurora,
    'html[data-theme="aurora"] .mm-bar i { box-shadow: 0 0 10px color-mix(in srgb, var(--s-accent) 70%, transparent); }',
  ) &&
    has(
      rule(consoleCss, ".mm-bar i {") ?? "",
      "transition: width 900ms var(--ease-standard);",
    ) &&
    !has(
      consoleCss,
      "box-shadow: 0 0 10px color-mix(in srgb, var(--s-accent) 70%, transparent);",
    ),
);
check(
  "cases: a glyph, the case and its word, the time or the reason at the right",
  has(
    rule(consoleCss, ".mm-case {") ?? "",
    "grid-template-columns: 16px minmax(0, 1fr) auto;",
  ) &&
    has(panel, 'name={sample.passed ? "check" : "x"}') &&
    has(
      panel,
      "{sample.passed ? `${sample.latency_s.toFixed(1)} s` : sample.detail}",
    ),
);
check(
  "tallies: mono chips with the count set apart",
  has(rule(consoleCss, ".mm-tally b {") ?? "", "font-weight: 500;") &&
    has(panel, "{name} <b>{`${passed}/${run}`}</b>"),
);

/* ---------- the one long operation ---------- */

check(
  "panel: positioned, so the theme's running tell rides its own pseudos",
  has(
    rule(consoleCss, '[data-part="mm-activity"] {') ?? "",
    "position: relative;",
  ),
);
check(
  "panel: the running tell keys on data-running",
  has(panel, 'data-running={activity.running ? "" : undefined}'),
);
check(
  "panel: a check can be stopped, a download paused, both from the one verb",
  has(panel, '{checking ? "Stop" : "Pause"}') &&
    has(
      panel,
      '{activity.running && !(checking && settled) && (\n          <button type="button" onClick={onCancel} className="btn btn-ghost">',
    ) &&
    has(
      panel,
      "const settled = activity.verdict !== null || activity.failure !== null;",
    ),
);
// A check's stream outlives its verdict by the re-warm it does on the way
// out: nothing is left to stop then, so neither the verb nor an abort may
// mark a verdict-bearing check as stopped.
check(
  "page: a hang-up after the verdict stops nothing and keeps it",
  has(
    models,
    "? current.verdict !== null || current.failure !== null\n                  ? current\n                  : { ...current, stopped: true }",
  ),
);
check(
  "panel: what Stop costs is said beside it while the samples land",
  has(
    panel,
    "Stopping discards this run: the last verdict stands, and the chat\n              model and the embedder are put back.",
  ),
);
check(
  "panel: a stopped check says so and keeps the last verdict",
  has(panel, "stopped: boolean;") &&
    has(
      panel,
      '? activity.stopped\n              ? "Stopped "\n              : settled\n                ? "Checked "\n                : "Checking "',
    ) &&
    has(
      panel,
      "Stopped. The last verdict stands; the chat model and the embedder\n              are being put back.",
    ),
);
check(
  "page: an abort stops a check and pauses a download",
  has(
    models,
    "? isChecking(current)\n                ? current.verdict !== null || current.failure !== null\n                  ? current\n                  : { ...current, stopped: true }\n                : { ...current, paused: true }",
  ),
);
check(
  "page: the sky follows the operation (amber fetching, the tool's teal checking) and rests on the way out",
  has(
    models,
    'setTurnState(!running ? "idle" : checking ? "tool" : "download");',
  ) && has(models, 'useEffect(() => () => setTurnState("idle"), []);'),
);
check(
  "sky: download is amber at .46 and the console may write it",
  SKY_STATES.download.e === 0.46 &&
    has(turnState, '  | "download"\n  | "login"\n>;'),
);
check(
  "comet: the angle is registered so the lap interpolates",
  has(
    aurora,
    '@property --mm-ang { syntax: "<angle>"; inherits: false; initial-value: 0deg; }',
  ),
);
const comet =
  rule(
    aurora,
    'html[data-theme="aurora"] [data-part="mm-activity"]::before,\nhtml[data-theme="aurora"] [data-part="mm-activity"]::after {',
  ) ?? "";
check(
  "comet: a ring cut from a conic sweep in the sky's running colour, hidden at rest",
  has(comet, "conic-gradient(from var(--mm-ang)") &&
    has(comet, "var(--sky-tint)") &&
    has(comet, "opacity: 0; transition: opacity 900ms var(--ease-standard);") &&
    has(comet, "mask-composite: exclude;"),
);
check(
  "comet: the ring is the border's width, the glow blurred behind it",
  has(
    aurora,
    'html[data-theme="aurora"] [data-part="mm-activity"]::before { padding: 1.5px; }',
  ) &&
    has(
      aurora,
      'html[data-theme="aurora"] [data-part="mm-activity"]::after { inset: -7px; padding: 8px; z-index: -1; filter: blur(7px); }',
    ),
);
check(
  "comet: it runs while something runs, one lap in 4.8s",
  has(
    aurora,
    'html[data-theme="aurora"] [data-part="mm-activity"][data-running]::before,\nhtml[data-theme="aurora"] [data-part="mm-activity"][data-running]::after {\n  opacity: 1; animation: mm-lap 4.8s linear infinite;\n}',
  ) && has(aurora, "@keyframes mm-lap { to { --mm-ang: 360deg; } }"),
);
check(
  "comet: aurora's alone (minimal subtracts), and the base blanket stills it",
  (aurora.match(/\[data-part="mm-activity"\]::(before|after)/g) ?? []).length >=
    2 &&
    !has(consoleCss, "mm-lap") &&
    !has(consoleCss, '[data-part="mm-activity"]::') &&
    !has(minimal, '[data-part="mm-activity"]::') &&
    has(
      base,
      "@media (prefers-reduced-motion: reduce) {\n  *,\n  *::before,\n  *::after {\n    animation: none !important;",
    ),
);

/* ---------- the sign-in ---------- */

check(
  "sign-in: one card on the sky at 23.5rem, arriving",
  has(rule(auth, ".login-card {") ?? "", "width: min(100%, 23.5rem);") &&
    has(rule(auth, ".login-card {") ?? "", "padding: 36px 32px 28px;") &&
    has(
      rule(auth, ".login-card {") ?? "",
      "animation: enter 600ms var(--ease-entrance) both;",
    ),
);
check(
  "sign-in: the mark, the name, the lit headline, the lede, the button, the foot",
  has(
    login,
    '<div className="login-mark" aria-hidden="true">\n          <Icon name="logo" />',
  ) &&
    has(login, '<div className="login-name">{appName()}</div>') &&
    has(login, "Sign in to <em>continue</em>") &&
    has(login, '<p className="login-foot">'),
);
check(
  "sign-in: the mark at 52px in the accent, with aurora's halo",
  has(rule(auth, ".login-mark {") ?? "", "width: 52px;") &&
    has(
      aurora,
      'html[data-theme="aurora"] .login-mark::before {\n    content: ""; position: absolute; inset: -30px;',
    ),
);
check(
  "sign-in: the headline in the serif at 28px with its lit word in the accent",
  has(
    rule(auth, ".login-head {") ?? "",
    "font-family: var(--font-prose);\n  font-size: 28px;\n  font-weight: 400;",
  ) && has(rule(auth, ".login-head em {") ?? "", "color: var(--s-accent);"),
);
check(
  "sign-in: the one control is 44px and full width",
  has(rule(auth, ".login-btn {") ?? "", "width: 100%;") &&
    has(rule(auth, ".login-btn {") ?? "", "min-height: 44px;"),
);
check(
  "sign-in: a session that lapsed says so and names the way back",
  has(login, 'const lapsed = single(params.lapsed) === "1";') &&
    has(login, "Your session <em>ended</em>") &&
    has(
      login,
      '"Sign in again to get back to where you were. Nothing you sent was lost."',
    ),
);
check(
  "sign-in: the lapsed flag rides the signed-out reaction, not the proxy's bounce",
  has(
    authTs,
    "window.location.replace(loginHref(`${window.location.pathname}${window.location.search}`, true));",
  ),
);
check(
  "login href: the flag is a query word beside the path",
  [
    loginHref("/", true),
    loginHref("/console/models", true),
    loginHref("/console/models"),
  ],
  [
    "/login?lapsed=1",
    "/login?next=%2Fconsole%2Fmodels&lapsed=1",
    "/login?next=%2Fconsole%2Fmodels",
  ],
);
check(
  "sign-in: the page names its view and its rest to the sky",
  has(login, '<ViewMarker view="login" state="login" />') &&
    has(marker, "if (state !== undefined) setTurnState(state);") &&
    has(marker, 'if (state !== undefined) setTurnState("idle");'),
);
check(
  "sky: the sign-in draws the chart at the boot's place and brightness, in over 560ms, gone under 560px",
  has(
    sky,
    'if (view === "login") {\n          if (W < 560) return;\n          const g = bootGeometry();',
  ) &&
    has(
      sky,
      'if ((view === "new" || view === "login") && !reduceMotion.matches && now - handoffAt > 1200) arriveAt = now;',
    ),
);
check(
  "sky: login rests at .44 teal",
  [SKY_STATES.login.e, SKY_STATES.login.tint],
  [0.44, SKY_STATES.idle.tint],
);
check(
  "base: the one entrance exists, so every arrival that names it arrives",
  has(
    base,
    "@keyframes enter {\n  from {\n    opacity: 0;\n    transform: translateY(4px);\n  }\n}",
  ),
);

/* ---------- chat into console ---------- */

check(
  "travel: the console is a transition kind",
  has(
    transition,
    'export type TransitionKind = "composer" | "boot" | "shell" | "console";',
  ),
);
check(
  "travel: a route transition waits for the destination's view, and lets go after a moment",
  has(
    transition,
    "const stop = observeView((now) => {\n            if (now === view) settle();\n          });",
  ) &&
    has(transition, "timer = window.setTimeout(settle, ROUTE_SETTLE_MS);") &&
    has(transition, "const ROUTE_SETTLE_MS = 1500;"),
);
check(
  "travel: reduced motion and a missing API simply navigate",
  has(
    transition,
    'typeof doc.startViewTransition !== "function" ||\n    reduce ||\n    readView() === view\n  ) {\n    navigate();\n    return;\n  }',
  ),
);
check(
  "travel: the sidebar's console row travels",
  has(
    sidebar,
    '<TravelLink href="/console" kind="console" view="console" className="convo">',
  ) &&
    has(
      travel,
      'transitionRoute(kind, typeof view === "function" ? view() : view, () =>',
    ),
);
check(
  "travel: the bar and the rail are named only while the transition runs, the rail only where it stands beside the page",
  has(
    chat,
    'html[data-vt="console"] [data-part="topbar"] {\n  view-transition-name: topbar;\n}',
  ) &&
    has(
      chat,
      '@media (min-width: 768px) {\n  html[data-vt="console"] [data-part="sidebar"],\n  html[data-vt="console"] [data-part="console-rail"] {\n    view-transition-name: side-rail;\n  }\n}',
    ) &&
    !/^\[data-part="console-rail"\] \{[^}]*view-transition-name/m.test(
      consoleCss,
    ),
);
check(
  "travel: the way back travels too, landing on the conversation the reader left",
  has(read("../app/console/layout.tsx"), "<BackToChat>") &&
    has(
      read("../components/console/back-to-chat.tsx"),
      'view={() => (peekResumedConversation() === null ? "new" : "thread")}',
    ) &&
    has(transition, "routeInFlight = id;") &&
    has(transition, "if (routeInFlight === id) routeInFlight = 0;") &&
    has(transition, "root.dataset.vtTo = view;"),
);
const page = read("../app/page.tsx");
check(
  "chat: carried in by a route, the shell renders at once with the waits behind it",
  has(page, "const [carriedIn] = useState(() => arrivingByRoute());") &&
    has(page, "const [booting, setBooting] = useState(!carriedIn);") &&
    has(page, "const boot = useBoot(booting || carriedIn);"),
);
check(
  "chat: where the reader was is remembered on the way out and opened on the way back, as a thread meanwhile",
  has(
    page,
    "clearView();\n      rememberConversation(conversationRef.current);",
  ) &&
    has(
      page,
      "const [resuming] = useState(() => (arrivingByRoute() ? takeResumedConversation() : null));",
    ) &&
    has(
      page,
      "if (!carriedIn) rememberConversation(null);\n    if (resuming === null) return;\n    const open = window.setTimeout(() => void selectConversation(resuming), 0);",
    ) &&
    has(
      page,
      "const empty = messages.length === 0 && !(resuming !== null && pendingId === resuming);",
    ),
);
check(
  "resume: the memory is session-scoped and taken once",
  has(read("../lib/resume.ts"), 'const KEY = "chat:resume";') &&
    has(read("../lib/resume.ts"), "window.sessionStorage.setItem(KEY, id);") &&
    has(
      read("../lib/resume.ts"),
      "if (id !== null) rememberConversation(null);\n  return id;",
    ),
);
check(
  "panel: a start glides the page to its top, eased, a cut under reduced motion, and a visible panel or a frame never moves it",
  has(models, "revealRef.current = true;\n      setActivity(started);") &&
    has(
      models,
      "if (!revealRef.current || activity === null) return;\n    revealRef.current = false;",
    ) &&
    has(models, "if (p.top >= v.top && p.bottom <= v.bottom) return;") &&
    has(
      models,
      'pane.scrollTo({ top: 0, behavior: reduce ? "auto" : "smooth" });',
    ),
);
check(
  "travel: no box is morphed across the corner - the side column stands full height, the bar wipes over it",
  has(
    chat,
    "::view-transition-group(side-rail) {\n  z-index: 1;\n  animation: none;\n  transform: none;\n  width: 272px;\n  height: 100dvh;\n}",
  ) &&
    has(chat, "::view-transition-group(topbar) {\n  z-index: 2;") &&
    has(
      chat,
      "::view-transition-old(topbar),\n::view-transition-new(topbar) {\n  height: 100%;\n  object-fit: none;\n  object-position: right top;\n}",
    ) &&
    has(
      chat,
      'html[data-vt-to="console"]::view-transition-new(side-rail),\nhtml[data-vt-to="new"]::view-transition-old(side-rail),\nhtml[data-vt-to="thread"]::view-transition-old(side-rail) {\n  object-position: left 48px;\n}',
    ),
);
check(
  "travel: the bar and the rail take 360ms",
  has(
    rule(chat, "::view-transition-group(topbar) {") ?? "",
    "animation-duration: 360ms;",
  ),
);

if (failures > 0) {
  console.error(`\nconsole shell: ${failures} of ${checks} checks failed`);
  process.exit(1);
}
console.log(`console shell: ${checks} checks passed`);
