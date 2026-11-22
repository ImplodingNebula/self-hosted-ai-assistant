# Design Implementation Playbook

How the design language in [DESIGN.md](DESIGN.md) becomes shipped frontend code, in order, with
acceptance criteria. This document exists so implementation requires as few design decisions as
possible: the decisions are already made, in the dossier and in [prototypes/](prototypes/).

Reading order for anyone starting this work cold: [DESIGN.md](DESIGN.md) first, then open
[prototypes/index.html](prototypes/index.html) in a browser and walk each theme's state harness,
then this file.

---

## Ground rules

- **The prototypes are the contract.** Every visual question ("how rounded is the user bubble in
  Grove?") is answered by opening the prototype, not by inventing. Deviations need an explicit
  owner decision, recorded in DESIGN.md.
- **Verify visually, every change.** After any UI change, render the running app and compare a
  screenshot against the matching prototype - same theme, same state. Styling merged without a
  side-by-side look is not done. Automating this (a browser-driving test or screenshot script)
  is encouraged but the manual loop is the floor.
- **Token discipline is reviewable.** Component code never contains a raw color. A diff that adds
  a hex value outside `tokens/` or a flourish layer is wrong by inspection.
- **All existing behavior is preserved.** This work restyles and restructures presentation; the
  SSE handling, confirmation flow, attachments, and API calls in `apps/frontend` do not change
  behavior. Any PR here should be demonstrably a no-op on the wire.

---

## Sequencing

The foundation lands at the **start of Phase 6**, so Model Manager and collections pages are born
themed; the full theme slate and polish land as **Phase 7**. Retrofitting themes after Phase 6
would touch every page twice.

| PR | Scope | Phase |
|---|---|---|
| F1 | Token foundation + ThemeProvider + picker | 6 (first) |
| F2 | Componentize the chat page (no visual change) | 6 |
| - | Phase 6 feature pages, built on tokens | 6 |
| P1 | Thread + message polish to prototype fidelity | 7 |
| P2 | Composer polish (model drop-up, toggles, send) | 7 |
| P3 | Sidebar, top bar, Integrations restyle | 7 |
| P4 | State choreography (thinking tell, tool cards, confirm) | 7 |
| P5+ | Theme slate, 2-3 themes per PR | 7 |
| P9 | Motion + reduced-motion + contrast audit | 7 (last) |

Every PR in this table is frontend-render-only, so the perf gate does not apply; note that in
each PR body ("perf: not run - frontend-render-only").

---

## F1 - Token foundation

> **Scope note (decided 2026-07-23):** F1 ships exactly two themes - **aurora (the default)** and
> **minimal** (the OS-following pair). The remaining catalog themes land with the Phase 7 theming
> pass; the registry, provider, and picker are built for the full catalog from day one, so adding
> a theme later is one CSS file plus one registry entry.

**Files:**

- `apps/frontend/app/themes/tokens.base.css` - the base sheet: tier 3 derivations from tier 2,
  shape/motion defaults, the reduced-motion blanket override (copied from the prototype skeleton's
  sections 5-6).
- `apps/frontend/app/themes/<theme>.css` - one file per theme, copied from
  [tokens/](tokens/) (authored there first; the app copies are the same content under
  `html[data-theme="..."]` selectors).
- `apps/frontend/lib/themes.ts` - the registry:

```ts
export interface ThemeDef {
  id: string;            // "aurora" | "minimal" | ...
  label: string;         // "Aurora"
  modes: "paired" | "light" | "dark";  // paired = follows data-mode + OS
  accents?: string[];    // phosphor: ["green", "amber", "red"]
  description: string;   // one line, shown in the picker
}
export const THEMES: ThemeDef[] = [ /* shipped themes, DESIGN.md order - aurora + minimal for now */ ];
export const DEFAULT_THEME = "aurora";
```

- `apps/frontend/components/theme-provider.tsx` - client component: owns
  `data-theme` / `data-mode` / `data-accent` on `<html>`, persists to `localStorage`
  (`app.theme`, `app.mode`, `app.accent`), follows `prefers-color-scheme` via `matchMedia` for
  paired themes when the user has not pinned a mode, exposes a `useTheme()` hook.
- A **no-flash script** inlined in `app/layout.tsx` `<head>`, before any styles, so the first
  paint is already themed (`suppressHydrationWarning` is already on `<html>`):

```html
<script>(function(){try{var t=localStorage.getItem("app.theme")||"aurora";
var m=localStorage.getItem("app.mode")||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light");
var d=document.documentElement;d.dataset.theme=t;d.dataset.mode=m;
var a=localStorage.getItem("app.accent");if(a)d.dataset.accent=a;}catch(e){}})()</script>
```

- **Tailwind v4 wiring** in `globals.css`: import the theme files, then alias semantic tokens
  into Tailwind's theme so utilities stay ergonomic:

```css
@import "tailwindcss";
@import "./themes/tokens.base.css";
@import "./themes/aurora.css";  /* + minimal.css; one line per shipped theme */

@theme inline {
  --color-app: var(--s-bg-app);
  --color-panel: var(--s-bg-panel);
  --color-raised: var(--s-bg-raised);
  --color-inset: var(--s-bg-inset);
  --color-fg: var(--s-text);
  --color-fg-2: var(--s-text-2);
  --color-fg-3: var(--s-text-3);
  --color-accent: var(--s-accent);
  --color-border-line: var(--s-border);
  /* fonts: --font-ui defaults to var(--font-geist-sans); themes may re-point it */
}
```

  Components then use `bg-panel`, `text-fg-2`, `border-border-line`, or CSS vars directly for
  anything not worth aliasing.

- **Picker:** a compact popover from the sidebar footer ("Theme: Aurora"): theme list with the
  registry descriptions, a mode toggle shown only for paired themes, accent dots shown only when
  `accents` is set. Swatches use each theme's real `--s-accent` / `--s-bg-app`.

**Acceptance:** the app under `aurora` matches the aurora prototype side-by-side (this is a
deliberate change of default look - the app is dark glass on first visit, and no longer follows
the OS scheme until the user picks `minimal`); `minimal` + system mode follows the OS and matches
its prototype in both modes; switching themes is instant with no flash on reload; reduced motion
freezes aurora's mesh drift and band per its catalog entry and kills all ambient animation.

---

## F2 - Componentize the chat page

`app/page.tsx` is ~1,900 lines. The pure `Message[]` helpers at the top (`appendDelta`,
`appendCitation`, `beginToolStep`, `completeToolStep`, confirmation helpers, ...) move to
`lib/messages.ts` unchanged. The presentational pieces already defined at the bottom
(`MessageBubble`, `ToolStepCard`, `ToolConfirmationCard`, `InfoNote`, `ErrorNote`, `ArgsView`,
`WebSearchToggle`) move to `components/chat/`, one file each. The remaining `ChatPage` keeps the
SSE loop and state, and renders:

```
components/chat/
  message-list.tsx      # the thread; owns scroll anchoring          data-part: thread
  message-bubble.tsx    # user + assistant turns                     bubble-user / bubble-assist
  thinking-pane.tsx     # collapsible reasoning                      thinking
  tool-step-card.tsx    # per-call trace card                        tool-trace
  tool-confirmation-card.tsx                                         confirm
  citation-row.tsx      # usage-gated chips / footnotes              citations
  notice.tsx            # standing honesty notices                   notice
  composer.tsx          # input + control row + send                 composer
  model-select.tsx      # the drop-up                                model-select
  web-search-toggle.tsx / thinking-toggle.tsx
```

Each component sets its `data-part` attribute - that is what themes and flourish CSS key on, and
it must match the prototypes exactly.

**Acceptance:** zero visual or behavioral diff (screenshot before/after per state); no component
imports the SSE layer; `data-part` hooks present on every part listed in DESIGN.md.

---

## SSE events to UI states

The stream handlers in `ChatPage` already distinguish these; the state machine the themes key on
(`<body data-state>` in the prototypes, a `turnState` value in the app) maps as:

| Wire | UI state | Visible parts (live turn) |
|---|---|---|
| request sent, nothing yet | `waiting` | waiting dots |
| `thinking` deltas | `thinking` | thinking pane (open) + ambient tell |
| `status: tool_running` | `tool` | thinking (collapsed) + tool card running |
| confirmation payload | `confirm` | + confirm card, composer disabled |
| content deltas | `streaming` | + content with caret |
| `error` | `error` | + error block, retry |
| `done` | `done` | citations fade in; renumbered content swapped with a 200ms cross-fade on the markers (`done.content`, only when changed) |
| standing status codes | notice added | `tools_not_used`, `tool_error`, `tool_limit_reached`, `image_not_seen`, `attachment_unavailable` |

The prototype harness in every theme file plays exactly this sequence ("Play full turn") - it is
the reference for timing and choreography.

---

## Theme slate PRs (P5+)

Land 2-3 themes per PR, each PR containing: the `themes/<name>.css` token file, the flourish
layer (a `@layer theme-flourish` block scoped under `html[data-theme="<name>"]`, appended to that
theme's css file), the registry entry, and a screenshot set (all states x light/dark where
paired). Order: minimal, nocturne (low flourish) first; phosphor, aurora, manuscript (heavy
flourish) after the flourish plumbing has proven itself; abyss, grove, ember, sorbet fill in.

Per-theme QA (from DESIGN.md, compressed): every state expressed; contrast pairs pass (ratios are
recorded in each tokens file header); reduced motion fully static and legible; focus visible;
signature reads on the thread AND on model-select + Integrations surfaces.

---

## Aurora next - build order (2026-10-09)

The 2026-10-09 amendment in [DESIGN.md](DESIGN.md) and
[prototypes/aurora-next.html](prototypes/aurora-next.html) are the contract for this work. It is
frontend-render-only with one backend exception, named in the last slice. Each slice is its own
PR: screenshot-compared against the prototype in the same view and state, at a desktop width and
at 390px, walked under reduced motion and reduced transparency, and design-reviewed before it is
presented. Minimal shares the markup, so every slice is also checked on minimal - nothing the sky
brought may show there.

1. **Sky and tokens.** The renderer and `data-sky` (shader, css fallback, off), the veil and its
   composer window, the constellations, the horizon, the three-job surface tokens, the sidebar's
   shade, and Source Serif 4 served locally. It lands first because every other slice sits on it,
   and it is measured in the browser before anything on the thread is touched: frame time and
   main-thread share at rest and while a turn streams, on the same machine, with the 30 and 20 fps
   caps as the budget and the css renderer as the floor. `make perf` is unaffected (nothing on the
   turn path moves) and the PR says so.
2. **Chat surface.** The timeline turn and its disclosure rows, the answer column, the reply
   footer, the code block, the citation popover, the model status with its waiting rows and the
   docked notice, the held toggle, jump-to-latest and the following rules. A no-op on the wire.
3. **Boot and new chat.** The boot screen on the waits the shell actually blocks on, the greeting
   and reach line, the view transitions and the chart handoff.
3b. **The chrome.** A whole-page inventory against the prototype, measured rather than read, found
   the chrome around the thread had kept its earlier anatomy. This pass moves it over: the
   composer's controls as one kind of chip and the selector's default rule, the sidebar's rows and
   foot, the floaters' contrast, the faces shipped inside the app, the user turn's verbs, the solid
   grounds under code and chips, and the sidebar toggle on every width.
4. **Console and sign-in.** The rail at the sidebar's width, lit rows with the edge chevron and
   the overlaid action, the comet ring, the sky's download and check states, and the sign-in card.
   Stop on a check ships only with its backend half, a PR of its own: an abort path on the
   tool-check stream that re-warms the chat model and the embedder as the stream's terminal frame
   does, with a test that fails when the re-warm is removed. The backend half was already there
   (the stream's cleanup re-warms on any exit, a client's hang-up included), so its PR is the
   test alone. The slice also carries the travel between chat and console (a link runs its route
   change inside a view transition, released when the destination names its view; a chat carried
   in renders at once on the conversation it left, with the boot's waits behind it), the page
   gliding to its top when an operation starts, and the sign-in's lapsed flag; what it leaves
   out, and why, is recorded beside the sign-in in DESIGN.md.

---

## Known traps

- **Hydration flash:** theme attributes must be set by the inline head script, not in a
  `useEffect` - an effect paints one wrong frame first.
- **`backdrop-filter` (Aurora)** is expensive on low-end GPUs: keep blur <= 14px, never animate
  the blur radius, and let the theme degrade to a solid tint when
  `prefers-reduced-transparency` is set. The one exception (2026-10-09) is the sidebar and the
  console rail at 22px: a single pane, never animated, and the only blurred surface at rest -
  floaters stay at 14px, and nothing else blurs at all.
- **Never hand-write `-webkit-backdrop-filter` (found 2026-07-26, the hard way):** when a rule
  contains both the standard property and the hand-written prefix, the CSS compiler dedupes them
  and emits ONLY the `-webkit-` form - which current Chrome does not implement (the prefix was
  always Safari's). Every glass surface in the app shipped flat on modern Chrome for exactly this
  reason, silently. Write the standard `backdrop-filter` alone; the compiler adds the Safari
  prefix itself. `browserslist` in `apps/frontend/package.json` pins the compile targets so this
  stays deterministic. To verify what actually ships, fetch the compiled chunk text - computed
  styles and devtools rule panels can both mislead here.
- **The sky's WebGL context is kept across remounts, never released.** A released context
  (`WEBGL_lose_context`) comes back from the same canvas as a LOST context on the next mount, so
  the shader silently falls to the CSS curtains after any remount - the development double-mount
  included - and the page looks right while running the wrong renderer. The context is left on
  the canvas and goes with the element; `<html data-sky>` says which renderer is actually running,
  so check it rather than the picture.
- **A `mask-image` on a scroll container is a backdrop root** (found 2026-10-10): the thread's
  lower-edge fade is a mask, and a blurred floater inside the thread (the citation card)
  samples nothing through it - it ships as a clear pane with the transcript showing through,
  computed `backdrop-filter` intact. The card stays in the thread's DOM (it is its trigger's
  next Tab stop), so the fade steps aside while a card is open
  (`[data-part="thread"]:has(.citation-popover) { mask-image: none }`). A new floater that
  blurs inside the thread needs the same escape, or a portal outside the masked scroller.
- **`clip-path` clips backgrounds, not outlines** - focus rings draw via `outline` +
  `outline-offset` outside the clip (the prototypes rely on this; do not switch focus styles to
  `box-shadow`, which does get clipped).
- **The reduced-motion override lives in the base sheet** with `!important`, so no theme can
  forget it. Never move it into per-theme files.
- **`!important` inside `@layer theme-flourish` BEATS the unlayered reduced-motion blanket**
  (CSS Cascade 5: for important declarations, layer precedence inverts). Rule of thumb: flourish
  layers avoid `!important` entirely; a flourish that truly needs it must ship its own in-layer
  `@media (prefers-reduced-motion: reduce)` reset (the sorbet and abyss prototypes show the
  pattern - both were caught leaking animation past the blanket in review).
- **The `hidden` attribute loses to any author `display` rule** (the UA's `[hidden]` style is
  not `!important`). Every element that gets an author `display` and is toggled via `hidden`
  needs an explicit `[hidden] { display: none !important; }` companion - the prototypes'
  integrations overlay shipped with exactly this bug and rendered permanently open until review
  caught it. In the app, prefer conditional rendering over the `hidden` attribute.
- **Paired vs single-mode themes:** `prefers-color-scheme` influences `data-mode` only for
  `modes: "paired"` themes; a single-mode theme ignores the OS. The picker must not offer a mode
  toggle for single-mode themes.
- **Markdown rendering** is a separate decision from theming, and it LANDED ahead of the Phase 7
  polish (react-markdown + remark-gfm + remark-breaks, plus remark-math + rehype-katex for LaTeX). Its output
  styles through `chat.css`'s `[data-part="content"]` rules on the shared tokens - `--font-prose`,
  `--c-code-*`, `--s-*` - never a stylesheet of its own, and the `pre` / inline-code treatment is
  the prototypes' contract verbatim. Raw HTML in an answer renders as escaped text and images are
  never fetched; both are safety decisions, not styling ones - do not "fix" them from the theme
  layer.
- **Do not surface agency in the composer.** The thinking toggle (Auto / On) and web-search
  toggle are the only mode controls a user sees; `ChatRequest.agency` stays a developer
  affordance (locked decision - see the chat-behavior section of the architecture docs).
- **The model drop-up lists chat-role models only** (from `GET /api/v1/models` + the catalog),
  with the one-line role descriptions from the registry; embedding/reranker roles never appear
  in it.

---

## Definition of done (any UI PR)

1. `npm run lint` and `npm run typecheck` pass.
2. Screenshot matrix reviewed against the prototypes: affected surfaces x affected themes x the
   states the change touches, at a desktop and a narrow viewport.
3. Reduced-motion pass: OS setting on, walk the changed surfaces.
4. No raw colors outside token/flourish files (inspect the diff).
5. PR body includes the perf line ("not run - frontend-render-only") or a real reading if the
   change somehow touches the turn path.
