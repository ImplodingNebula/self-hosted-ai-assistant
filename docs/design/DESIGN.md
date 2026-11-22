# Design Language

The visual and interaction contract for the platform's web UI. This document defines the rules that
generalize; the pixel-real reference lives in [prototypes/](prototypes/) - one self-contained HTML file
per theme, openable directly in a browser. **When prose and prototype disagree, the prototype wins.**
Aurora, the default, has a second and newer reference, [prototypes/aurora-next.html](prototypes/aurora-next.html):
it covers the chat surface, sign-in, boot, the new chat and the Model Manager, and wins over
`aurora.html` (the skeleton the other themes derive from) wherever the two overlap.
The build order and acceptance criteria live in [IMPLEMENTATION.md](IMPLEMENTATION.md).

> Consolidated 2026-10-09 from the amendments recorded since 2026-07-20. Each rule appears once,
> with its reasoning; git history holds what it replaced.

---

## Principles

1. **Calm surface, live details.** The chrome is quiet; energy is spent where the system is actually
   doing something - streaming, thinking, running a tool. A user should be able to tell the app's
   state from across the room, without reading a word.
2. **The answer is the interface.** Nothing competes with the message thread. Sidebars collapse,
   toolbars stay monochrome until hovered, and decoration never sits between the user and the text.
3. **States are honest.** The backend emits honesty signals (`tools_not_used`, `image_not_seen`,
   `attachment_unavailable`, tool errors). The UI presents them as first-class, designed elements -
   quiet but unmissable - never as afterthought toasts.
4. **A theme is a personality, not a palette swap.** Every theme answers the same six questions (see
   [Theme anatomy](#theme-anatomy)) with its own voice. Presentation pulls people in; a dozen
   well-made themes is a feature, not a gimmick.
5. **One component tree.** Themes restyle; they never fork markup. All theming happens through
   design tokens and a constrained flourish layer keyed on stable `data-part` hooks.
6. **Motion is meaning.** Every animation encodes a state change or draws the eye to new content.
   Ambient motion is subtle, slow, and fully removable via `prefers-reduced-motion`.

---

## Layout anatomy

The app is a two-region layout: a **sidebar** and the **main column**. The main column holds a slim
**top bar**, the **thread**, and the **composer**.

```
+------------+--------------------------------------+
| sidebar    | top bar (title, model, status)       |
|            +--------------------------------------+
| new chat   |                                      |
| convo list |   thread (centered, 50rem column)    |
| ...        |     user message                     |
|            |     assistant message                |
|            |       [thinking pane]                |
| console    |       [tool trace]                   |
| theme      |       content + citations            |
+------------+   composer (sticky bottom)           |
             +--------------------------------------+
```

- **Sidebar** (`data-part="sidebar"`): 272px, hidden entirely by its toggle on desktop, a solid
  drawer over a scrim under 760px. It holds the conversation list, the new-chat action, one link to
  the console and the theme picker trigger. Its width is the console rail's, so the two swap in
  place, and it is the one blurred pane of the glass system (see
  [Surfaces](#surfaces-glass-is-three-jobs)).

  Pinned conversations lift into a **Pinned** group above Recent, headed by the same `.side-label`,
  newest pin first, and a pinned row appears there ONLY. The group and its label disappear when
  nothing is pinned, so an unused feature costs no vertical space; both labels scroll with the list.
  Pinning is a single row-menu item worded after the row (Pin / Unpin). A pinned row carries a
  small LEADING pushpin glyph (12px, `--s-text-3`, aria-hidden - the group label is what readers
  hear), leading so it never fights the menu trigger for the row's right edge; the topbar mirrors
  the glyph beside an open pinned conversation's title. Managing a row is in
  [The row menu and rename](#the-row-menu-and-rename).
- **Top bar** (`data-part="topbar"`): conversation title, the model's whereabouts (see
  [Model whereabouts](#model-whereabouts-and-the-composer-notice)) and connection state. One line,
  hairline bottom border, never taller than 48px. While a turn is busy its lower edge takes the
  running colour (the horizon - see [The sky](#the-sky-energy-and-tint-per-state)). On narrow
  screens the model status keeps only its dot.
- **Thread** (`data-part="thread"`): a single centered reading column, `50rem` wide - exactly the
  composer's width. The timeline rail and a user turn's clock time hang OUTSIDE the column in equal
  gutters, so the composer and the answer share one left edge. 24px between turns. Following,
  the jump to latest and the lower-edge fade are in [Following the stream](#following-the-stream).
  Under 760px the rail time and both gutters go; the glyph column stays. The console's `52rem`
  column is a different measure.
- **Composer** (`data-part="composer"`): an auto-growing textarea, an attachment tray above it when
  populated, a hint line beneath ("Local model. Answers can be wrong, so check the sources."), and a
  compact control row; Send morphs into Stop while streaming. On aurora it is a translucent pane
  straight onto the sky, its placeholder stepped up to `--s-text-2` for contrast at the sky's
  brightest. A notice that belongs to the composer docks ABOVE it (see
  [Model whereabouts](#model-whereabouts-and-the-composer-notice)), and a round jump-to-latest
  floater hangs above it whenever the reader has left the end, with a dot once something arrived
  while they were away.

  What is typed and not sent belongs to the conversation it was typed in. Opening another thread
  shows that thread's own unsent words (or an empty box), coming back restores these, a new chat
  keeps its own, and a reload loses none of them. The swap is the box changing its text and
  nothing else: no notice, no motion, and the caret moves only if the box already had it. Sending
  takes the draft with the message; a send that never reached the server hands it back. Drafts
  are kept in the browser, per signed-in user, and signing out clears them.

  The control row has two groups. Per-turn adjustments on the LEFT: attach, the thinking toggle
  (Auto / On), the web-search toggle. What runs the turn, and what it costs, in the RIGHT dock: the
  **context meter** (see [The context meter](#the-context-meter)), the **model selector**
  (`data-part="model-select"`), then Send. Every control is the same quiet chip: a glyph and a
  `--text-label` word, no border, 32px tall with the 40px hit floor borrowed around it, lit in the
  accent ink alone when it is on (a boxed or filled chip beside the others would read as the primary
  action and compete with Send, the one filled control); under the pointer a chip takes a 7% wash,
  and a lit one stays lit, a step brighter (`--s-accent-hover`). The selector's chip is mono with a
  chevron; Send is the 34px accent square at a 10px radius, with no glow. The meter leads the dock because it is itself a control
  (click for the numbers) and the dock is where the eye already checks which model is answering.
  The selector carries the row's auto-margin and its menu is right-anchored, so the 300px panel
  cannot overflow the composer; below 767px the row reverts to flow order with Send taking the
  trailing margin. The menu hinges on the chip's right edge at every width; where that would hang
  it off the screen it slides back inside the box, measured as it opens, and the context card does
  the same from the arc. Each row says where its model is in a fixed column at the right (a filled dot
  "in memory", a ring "on disk"), so the dots stand in a line whatever the word beside them.

  The selector is a drop-up listing the catalog's chat models with a one-line, sentence-cased role
  each; installed models lead (see [Rows and pills](#rows-and-pills)). **The workspace default is a
  suggestion, not an entry:** its own row wears a small *default* badge, choosing that row clears
  the conversation's pin (so the conversation follows the default wherever an admin later moves
  it), and the trigger names the model that will answer rather than the word "Default". When the
  default names nothing the server lists (a tag since removed, or none set), the first listed model
  stands in for a fresh conversation and is pinned, so the trigger is never a promise the turn
  cannot keep. It is a floater on
  `--c-float-bg` with the floaters' 14px blur, which works only because the composer itself carries
  no filter (a blur nested inside another blur samples nothing). Each entry says whether it is in
  memory or on disk; choosing one on disk starts the load and the composer notice. The list carries
  the server's own inventory beside the catalogued entries, so a row the catalog never described
  gets the hint "On the model server - not described by the catalog"; when every row is inert, a
  non-focusable `model-note` line names the one move that changes that ("Nothing installed can chat
  yet - pull a model in the Model Manager").

  Two honesties keep the row from offering what would change nothing. A toggle that is on but will
  not apply this turn (thinking on, and a model that cannot reason and call tools at once) wears
  `data-held`: struck label at `--s-text-3`, tooltip saying why and what to change. And with a
  pinned model that answers directly and no partner paired with it, the Thinking toggle is DISABLED
  with a title naming the fix - disabled rather than absent, against the console's default, because
  the control is one of three in a fixed row and removing it would reflow the row on a model change.
  It is gated on a PIN only: on "Default" the turn resolves through a chain the page holds no answer
  for, and disabling on a guess would take a working feature away on no evidence.

### Console shell

The admin console is the same shell answering like a turn, not a second app. Management surfaces
gather at `/console`, which redirects to its first section rather than asking the reader to choose
again.

- **The rail** is a 272px glass column - the sidebar's width, so the transition between chat and
  console swaps one for the other in place - of 40px `--text-label` entries on `--c-sidebar-bg`, the
  current one marked by fill AND weight. Under 760px it becomes a horizontal, scrollable tab strip
  above the page: a fixed column beside content that already wraps leaves neither room to read.
  Every entry goes somewhere that exists; a "coming soon" entry is a promise the console cannot keep.
- **The topbar** carries a back-to-chat verb and "Console" for every section, so each page names
  itself in its own column with an `h1` (`.mm-title`, `.int-title`): the page title in the serif
  display (30px) with a mono meta line. The topbar and the rail travel between the chat and console
  views (see [What travels between views](#what-travels-between-views)). A **Loaded now** box lists
  what is in memory - the topbar's status as a list.
- **One measure, `52rem`,** under the thread's veil cut to its width. Sibling sections whose content
  edge jumps on every rail switch read as inconsistency, so every section's column is `52rem` (the
  Model Manager's row anatomy sets the floor) and anything that reads better narrow constrains
  itself inside it at the column's left edge - prose at `46ch`, the add-server form at `44rem`.
- **The interactive floor is 40px** (`.console-body` gives every button a 40px minimum).
- **Rows are lit panes** whose whole head tints on hover (see
  [Surfaces](#surfaces-glass-is-three-jobs)).
- **Starting the one long operation brings its panel into view.** The verb may sit a screen below
  the activity panel (a row's Run check, a Pull at the foot of the list), so the page glides to its
  top as the operation begins, where the panel sits under the page's name (a cut under reduced
  motion). A panel already in view stays where it is; a frame landing or a verdict never moves the
  reader.

Rules every console page follows:

- **A verb the backend lacks is ABSENT, not disabled.** The exception is a control refused for a
  REASON - a Delete something still depends on - which stays visible and disabled with the reason as
  text beside it at `--s-text-2` (the `int-hint.reason` tier): a title on a disabled button never
  opens in Chrome or Safari, and hover-only text never reaches touch or keyboard. Load-bearing prose
  uses this tier, never `--s-text-3` (which misses AA at small sizes on the light theme). Every
  failure surface shows the backend's own sentence.
- **Numbers are facts with their provenance, never verdicts,** and a prediction is never shown where
  a measurement does not exist.
- **An outcome is three channels, never one sentence**: a status word, a self-sizing monospace chip
  grid of the specifics, and any caveat as a warn line of its own, because a special case folded
  into a long line is a special case lost. The integrations test outcome and the Model Manager
  verdict both take this shape.
- **One warn slot per SUBJECT** (see [Verdicts and warn slots](#verdicts-and-warn-slots)); a dialog
  inherits the rule with the decision as its subject.
- **Per-theme voice, from the token system.** Aurora blurs the rail like the sidebar and lights the
  panels as panes, the activity panel a touch heavier since it is the one thing on the page that
  changes while it is watched. Minimal subtracts: rows flatten to hairline-separated entries at the
  page edge with a tinted body when expanded, nothing decorative added. That subtraction is written
  UNLAYERED under a `html[data-theme]` prefix rather than layered with `!important`, so the base
  sheet's reduced-motion blanket stays the one thing nothing can outrank.

### Sign-in, boot and new chat

- **A new chat is the greeting and the composer, together.** With no thread to protect, the serif
  greeting ("What shall we shed *light* on?" - one word in italic, in the sky's colour), a reach line
  beneath it (knowledge base, N documents; web search on, off, or "stands down while thinking";
  tools, N connected - in the glyphs the rail will use) and the composer sit a little above centre,
  and the rest of the pane is open sky. Sending docks the composer to the foot: the box travels
  (420ms), the greeting leaves, the thread arrives.
- **Boot has the whole window.** Before the shell can mean anything - the backend not yet answering,
  or the first signed-in load - the sky and the stars have the window, with a chart of rings turning
  against each other behind the words. The mark draws itself in (1.1s) and keeps a slow halo; the
  name sits under it in tracked capitals; the headline names what is being waited on with the
  waited-on word lit ("Reaching the *backend*", "Waking the *model*", "Gathering your
  *conversations*", "*Ready*"), with a mono subline for the detail. A rail of nodes lights as each
  wait ends - a live node spins an arc and pulses, a done node fills with a check, the lit line
  grows - and the sky keeps each wait's colour (violet while the model wakes). A wait past about
  fifteen seconds says so ("Still waking the *model*"), counts ("N s and counting"), offers a retry,
  and the sky goes amber. The nodes are the waits the shell actually blocks on. Under 560px the
  chart goes and the rail stacks. The handoff to the new chat is one choreography: the mark travels
  into the sidebar's brand, the headline becomes the greeting and each node its reach item (560ms,
  text cross-fading in place, never stretched to the moving box), while the chart fades out where it
  was (260ms), rests a beat (160ms) and fades in at its new-chat place, dimmer (560ms). The chart
  never cuts: the sky re-measures its pane the moment the view changes (under the boot the shell
  is hidden, so the pane read then is the whole window), and wherever the chart arrives without a
  handoff - a quick load, a new chat opened from a thread - it fades in where it lives (560ms); a
  later move of its anchors (the sidebar closing, a resize) is eased, not stepped.

  The choreography is paid for once per browser session. The first load holds each lit node a
  beat (450ms, the model's a second so the violet lands) and stands on "Ready" (700ms), about
  2.6s against a backend that answers at once. Every later load in the session runs the same
  waits with no beat held: the screen is painted only if a wait is still open after 600ms, and
  once painted it stands at least 700ms so it never flickers past. Against a quick backend a
  reload is a few hundred milliseconds of sky, then the shell cross-fading in whole (260ms, no
  movers: a boot never painted has nothing to carry) with the chart arriving behind it; a slow or
  down service still gets the whole screen, since the wait itself takes the time.
- **Sign-in is one glass card on the sky,** with the chart at the boot's full brightness behind a
  23.5rem card: the mark with its halo, the name, a serif headline with one lit word ("Sign in to
  *continue*"), one line of lede, one 44px button, a foot line ("Conversations live in the
  workspace."). A session that lapsed mid-task says so ("Your session *ended*", "Sign in again to
  get back to where you were. Nothing you sent was lost.", and the foot "Only the session
  ended."): the lapse is a flag the signed-out reaction adds beside the remembered
  destination (`/login?next=...&lapsed=1`), never something the proxy's own bounce claims. The
  lede cannot name the conversation: the lapse is learnt from a refusal before anything of the
  page has rendered, so the page has no title to say. Under 560px the chart goes, as it does under
  the boot. Signing in leads into boot, which draws the mark in afresh: the identity provider's
  round trip is a cross-origin navigation no transition can span, so nothing carries across it.

---

## Message anatomy

Each assistant turn is a stack of parts; every layer is optional and independently themed:

| Part (`data-part`) | Content |
|---|---|
| `thinking` | Collapsible reasoning pane, streamed via the `thinking` SSE channel |
| `tool-trace` | One row per tool call: proposed, running, done, error |
| `confirm` | Side-effecting tool halt: summary of the pending action + approve / deny |
| `content` | The streamed answer (markdown), with inline `[n]` citation markers |
| `citations` | Usage-gated source chips, renumbered 1..k on clean completion |
| `notice` | Standing honesty notices (`tools_not_used`, `tool_error`, `tool_limit_reached`, ...) |
| `turn-actions` | Quiet action row: copy, edit, reply, answer again |
| `quote-chip` | The passage a turn is replying to: pending in the composer, or recorded on a sent question |
| `message-editor` | A question's bubble, swapped for a box to reword it in |
| `selection-quote` | The floating Quote/Copy group that appears over a text selection's start |
| `turn-files` | Files the reply itself produced: one download chip per generated file |

User turns are a single `bubble-user` part, closed by the same `turn-actions` row. Attachments
render as chips inside the turn that sent them; files a reply *produced* render as download chips in
its own `turn-files` row.

### The turn as a timeline

On aurora a turn is a timeline, not a stack of cards. Every row of an assistant turn shares one
grid: elapsed time | glyph | body. The rail carries the time since the question (`+2s`, `+5s`, in
tabular mono at `--s-text-3`) and one glyph per step, joined by a hairline to the next row. A step's
kind sets its colour through `--c-step-*` tokens - thought violet, tool teal, confirmation magenta,
answer `--s-text-2`, error `--s-danger`, the closing row `--s-ok` - and a running step swaps its
glyph for the matrix loader in the same colour. Thinking and tool calls are inline disclosure rows:
one quiet line that opens in place (a `0fr` to `1fr` grid row, no mounting). The user's turn has no
rail: the bubble sits right-aligned with the clock time in the right gutter. The answer's row is
typeset prose with no card (see [Content and markdown](#content-and-markdown)); the closing row is
the reply footer (see [Turn actions](#turn-actions)). A live turn's first row says "Sending to
<model>", or "Waiting for <model> to load" when it is loading; the two swap in place.

### Thinking

The reasoning pane streams from the `thinking` channel. In the timeline it is a disclosure row that
stands open while live and collapses to "Thought for N seconds" after. Its ambient tell is a
state-level signal, not a pane decoration (see [State choreography](#state-choreography)); under
reduced motion the tell is a static badge on the pane.

### Tool trace

- **One summary-row anatomy in the base sheet, all themes:** status glyph (spinner while running, a
  check for success, a cross for a failed or denied call, a neutral dot for finished-inconclusive),
  the tool's label, an inline preview of the call's leading string argument (truncated, quoted, in
  mono), any approval badge, and a trailing meta word only for the states the glyph cannot carry
  alone ("running", "failed", "did not finish"). A result meta such as "1 source" is admissible
  only as a factual count, never as decoration. The visible verb ("Used ...") lives in a
  visually-hidden span for screen readers.
- **The trace narrates, so its name is a humanized label,** derived deterministically from the
  identifier (`web_search` reads "Web search"; an unrecognized `search_records` reads "Search
  records"), with the raw identifier on the element's `title`. The confirmation gate keeps the
  identifier verbatim (see [Confirm](#confirm)). `.tool-name` keeps `--font-mono` for now: the class
  is shared by both surfaces and the gate still carries a literal identifier, so changing the face
  means splitting the class first - a recorded follow-up.
- **Minimal re-cards the trace** as the base hairline chip on a panel-gray tint (`--c-tool-bg:
  gray-1` - not gray-2, so the chip never reads as a user bubble), because bare text lines read as
  stray prose rather than activity.
- **Aurora renders no chip at all:** the trace is the timeline row - teal glyph, matrix loader while
  running, label and query on the line, the sources as a lit pane beneath. The luminous edge that
  once travelled a running chip's border lives on in one place, the console activity panel's comet.

### Confirm

The `confirm` part is the halt before a side-effecting tool: a summary of the pending action with
approve and deny, while the composer is disabled except for those two. **The card keeps the tool
identifier verbatim.** Humanizing is lossy - a registered `web-search` renders exactly like the
built-in `web_search` without colliding with it, since the catalog rejects only an exact duplicate -
and an approve/deny gate exists to show what would actually run. In the timeline it is the magenta
step, and answering it re-sticks the thread to the end.

### Content and markdown

The answer renders as markdown. Every rule lives in the base sheet's `[data-part="content"]` scope
on shared tokens, so a theme restyles by re-pointing tokens, never by re-deriving the list. On aurora
the answer has no card: typeset prose in the one reading column (see [Typography](#typography) for
its serif face), the list marker a star (a 5px point in the accent with its glow, a point of the
constellation beside each item), inline code a chip on `--s-bg-panel`.

- **Headings are landmarks, not page titles, but landmarks outrank each other:** h1 1.35em, h2
  1.2em, h3 1.1em (all 650), h4 1em / 700, h5 0.95em / 700, h6 0.9em / 700 in `--s-text-2`, so all
  six stay tellable apart (a flatter scale read as six identical bold lines). h1/h2 take 16px 0 8px
  margins, the rest 14px 0 6px, with first/last-child trims so an answer that opens on a heading or
  closes on a list sits flush on the stack gap.
- **Blockquote:** 2px left rule `--s-border-strong`, `--s-text-2` text. **hr:** 1px `--s-border`.
  **Table:** `--c-table-border` cells, a `--c-table-head-bg` header band (base derivations
  `--s-border` / `--s-bg-inset`), `th` 600 and LEFT-aligned (the UA centers headings, which reads
  as misalignment over left cells; an explicitly aligned column still wins). A table ships inside a
  `.md-table-wrap` horizontal scroller and its cells reset break-anywhere wrapping, so a wide table
  scrolls instead of shattering words or widening the pane.
- **Task lists** drop the list marker (the checkbox IS the marker), drawn as the `.md-check` glyph,
  never the native control: a task item reports what the answer said rather than offering a control,
  and a disabled native input is washed out on every browser and theme. Unchecked is a
  `--s-border-strong` outline; checked fills `--s-accent` with an `--s-on-accent` mark.
- **Code is a component:** a head with the language label in mono at `--s-text-3` and a copy button
  whose glyph cross-fades to a tick, then syntax colour on the `--c-code-*` tokens (comment, keyword,
  string, number, function).
- **Raised, not recessed, on glass themes.** In-answer surfaces join a glass theme's material system
  as RAISED planes; the base derivations sank the code slab and table band into the card while their
  10% borders faded. Aurora lifts `--c-code-bg` to its night-2 primitive with the glass line and
  inset top highlight on `pre`, and overrides `--c-table-head-bg` and `--c-table-border`.
- **`.md-image`:** an image is never fetched; it renders as an inert chip on the inline-code recipe
  labelled with its alt text. **`.md-fallback`:** an answer the parser cannot handle falls back to
  plain pre-wrap text. Both are safety decisions; themes may restyle the chip, never re-enable the
  fetch.
- **Recorded deviation:** the composer's code slab insets at `padding: 10px 12px` / `margin: 4px 0`
  against the answer's `14px 16px` / `10px 0`, because it sits INSIDE the composer box's own padding
  where the answer's slab sits directly on the column; carrying the figures across would stack the
  two paddings.

### Citations

`citations` are usage-gated source chips, renumbered 1..k on clean completion (a designed moment -
see [State choreography](#state-choreography)). **Citations open where they are read:** a marker in
the text or a chip in the footer opens one shared popover (number, title, file, the passage, where it
sits in the source) on hover or focus, holds it on click, and lets it go on scroll, Escape or a
click elsewhere; marker and chip light together while it is open. The popover opens ABOVE its
trigger (the lines above a marker have been read, the lines below are next), below only when the
pane's top edge is nearer, and it is a floater and blurs. A marker fills under the pointer (26%)
and, open or lit by its chip, fills further (30%) with its number in the text ink; a chip takes
the accent edge and, on aurora, a soft glow. Manuscript renders the chips as a footnote list on identical markup.

### Notice

`notice` carries the standing honesty notices (`tools_not_used`, `tool_error`,
`tool_limit_reached`, `image_not_seen`, ...) beside the answer, never as toasts. A staged image on a
model that cannot read it is also raised BEFORE the turn, in the composer, where the fix is still
free; that notice precedes `image_not_seen`, it does not replace it.

### Turn actions

- **The assistant footer is the closing timeline row:** a check glyph in `--s-ok`, a little lower
  than the answer and in a smaller voice - the citation chips, then three icon verbs (Copy answer,
  Reply, Answer again), then the figures in small mono (model, tokens, tokens per second). The figures
  are the server's and are stored with the reply, so a reloaded turn shows the ones its live stream
  closed on: the count covers every round of the turn, the speed is that count over the runtime's own
  writing time (never a clock that ran through a tool or a confirmation), and a reply with no count, or
  one saved before the figures were kept, shows its model alone. A footer, not a toolbar. **A user
  turn** offers Edit, Reply and Copy in the gap under its bubble's right edge. The turn owns the
  actions; the bubble owns the answer.
- **Shown at rest only on the latest reply;** older replies reveal the footer when pointed at or
  focused into, and a verb taken from a row holds it open for about two seconds so the copied tick
  is seen before the row fades. The reveal is OPACITY, never mounting: the buttons keep their box, so
  a hover cannot reflow the turn, and they stay in the tab order. Where there is no hover at all
  (`@media (hover: none)`) the row simply stays. At rest the row is `--s-text-3` at `--text-micro`;
  hovering a button lifts its ink to `--s-text` over a `--c-tool-bg` fill (on aurora a tint on the
  veil, no blur).
- **Every action is icon-only,** with the word in the tooltip and the accessible name; the pencil is
  the same glyph the topbar rename uses, so one gesture reads one way across the app. An action is
  ABSENT rather than disabled when the pane cannot take it: while a stream is in flight or a
  confirmation is waiting there is nothing to replace, and a disabled control invites a click and
  explains nothing.
- **Copy on every turn that has words.** Success swaps the glyph to a tick in `--s-ok` for about
  1.5 seconds; failure changes nothing, because a tick over a clipboard that never took the text
  sends someone to paste nothing.
- **A dialog appears only when something OTHER than the turn's own reply would go.** Answering the
  newest reply again and editing the last question replace exactly what their button promises and
  ask nothing. Further up the thread both also discard what came after, so the confirmation dialog
  states the count ("The 4 messages below it will be removed"); cancelling an edit's dialog leaves
  the draft in the editor. The irreversible-action dialog stays reserved for what cannot be
  re-derived (deleting a conversation, deleting a file).
- **A file-only question carries the pencil alone:** no words to copy, not a reply, but it can be
  rewritten - it is exactly the question someone comes back to add words to. A row is omitted only
  where every action is genuinely absent.
- **Recorded cost:** because the row holds its space while invisible, every turn that renders one is
  about 30px taller than the prototype's. Collapsing the row when hidden would make the thread jump
  under the pointer on every hover, which is the worse defect. Worth a look on a long transcript if
  the surface starts to feel loose.

### Quote chip

One clipped line, on two surfaces: in the composer it is the pending quote with a dismiss; on a sent
question it sits above the bubble as the record of what was replied to, and survives a reload because
the passage is stored with the turn. Both are the same `quote-chip` part - an accent rule and one
ellipsized line, nothing else. On a sent question the band is the way back to its source while the
message it came from is in the thread: a button (the pointer deepens the tint and lights the mark,
the press deepens it further; its press reaches the 40px floor) that glides the thread to that
message and marks what was quoted. The quoted words, where they are found in the rendered answer,
are lit (a wash and an underline in one colour, `--c-passage-light`, so it is not read as a
selection: a rise, a hold and a fall). A whole answer - the Reply verb, or a selection of all of a plain
one, told apart by the quote's words being the answer's own, so no second field is stored - is
marked as a message: a softer wash of the same colour arrives from the top of the prose and settles over it
while the turn's rail, where the theme draws one, takes the accent, a glow travelling down the rail
with the wash's front (the ring's bloom at half strength over the lit segment, its ends fading beside
the glyphs, one front down the whole rail passing over the glyphs, revealed by the sweep's mask on
its clock), the extent
of the message shown by passing over it. The wash is light from the rail, a lit pane like the theme's others: fullest
beside the rail, half of that within the first fifth of the measure and almost nothing at the far
edge, with the pane edge the theme draws over it (laid flat and even it read as a tint on the pane,
not light on it, and an even ramp left the tail a grey tint; minimal, with no rail and no edge
light, bounds it with its hairline). A marked message lands centred (the prose or the bubble, not
the turn with its rows above), the whole of it in view, the turn's top never higher than a margin
below the topbar (a message taller than the viewport shows its beginning, rows and all); lit
words rest in the upper part of the viewport, with the rest of the message below them for reading
on. Where
the words cannot be found (the answer was edited since, or quoted as its raw markdown) the prose is
ringed - an outline in the accent at half strength (at full it is the focus ring's drawing) that
rises and falls on the same clock as the washes. Never two of these at once: a ring around a tall answer next to its lit words says
the same thing twice. Under reduced motion the glide, the ring, the washes and the rail's light are
cuts. The jump floater is then the way back down: it shows the return glyph (a
hooked arrow over and down the thread; an arrow onto a line reads as a download), lands on the
question that was left and rings its bubble, the ring swelling once as a glow over the bubble's edge
light, on the same clock as the washes, the outline rising and falling with the glow (a bare outline
on a tinted pane read as flat; a second beat read as a flicker; an outline that outstayed the glow
read as a second state), and forgets the way once the reader is at the end again. Where the source is gone - edited away, replaced, or quoted before it had an id - the
band is inert and has lost its accent: the mark greys and the tint drops to the bubble's own ground,
while the words stay readable (they are content, not meta; greyed text on a tinted pane read as a
disabled control), with the reason as its tooltip and, for readers, before the words: a dead link
that looked live would be worse than none, and a link offered only where it leads somewhere is not
that. Not a rich card. One quote at
a time; a second Reply or Quote replaces the chip, because a pile of chips reads as a clipboard.
On a sent question the chip rides INSIDE the bubble as a band at its top, inset 6px from the
bubble's edges on the quote tint (the bubble's pane with a touch more accent, `--c-quote-bg`): the
strip the composer shows while the reply is pending, carried over, so a question keeps the shape it
was written in and there is one bubble (a second bubble stacked on the first read as two objects
however the corners were cut). The width a quote can force on the bubble stops at 28rem, seven
tenths of the measure (a tiny question under a long quote gets a bubble that wide, not the full
40rem, while a short quote keeps it compact; a long question still widens the bubble and the band
stretches with it; on a phone the column wins), and the passage runs two lines there before it clips, which carries more of it than
one line at the full measure did. In the composer it is the pending line, one line, at the top of
the pane on the same tint. Aurora lights both as panes.

### Message editor

Editing happens in the bubble, not in a dialog: the question's bubble becomes a panel at the bubble's
alignment, widened to the thread's width (rewording wants room), with an auto-growing textarea and a
Cancel/Save pair, so the turn being changed stays the turn on screen. Escape cancels, Ctrl+Enter (or
Cmd+Enter) saves, plain Enter is a newline - a question being reworked is longer than one being typed,
and a stray keystroke must not throw away a reply. Save is disabled while the draft is empty or
unchanged. Aurora lets the editor borrow the user bubble's pane so it does not read as a form dropped
into the thread.

### Selection quote

Selecting text offers Quote (a reply glyph) and Copy (a copied tick that announces itself politely)
in a two-action group that lasts as long as the selection and goes at the first sign it is stale - a
scroll, a resize, a click elsewhere. It is sized to be FOUND (dialog-button scale, glyph-labelled,
raised on the menu surface) because an affordance that exists only while a selection does cannot
afford to be missed. It anchors over the selection's START, clamped to the viewport: a multi-line
selection's box spans the whole column and a centred control lands nowhere the eye is. Positioned in
viewport coordinates above everything, so no bubble's overflow clips it; a press on either action
does not clear the selection, and Copy keeps both the selection and the group, so taking a passage
out and then quoting it is one gesture. Aurora keeps it a blurred floater (it passes over text).

### Turn files and the tombstone

`turn-files` holds one download chip per file the reply itself produced.

**The tombstone treatment**, for a file whose bytes are no longer held: dashed `--s-border`, struck
name, `--s-text-3`, and the gray is never the only channel (label, strikethrough, dashed border and
the failure banner all carry the state). It is implemented on all three chip surfaces - the
`turn-files` chips, the composer tray, and an upload's chip on a user message - and on each it is
reached from evidence the client has: the download's own refusal, or the file's stored status.
Liveness for a sent question's chips is keyed on the status the conversation payload carries, joined
by id against each message's attachment snapshot, so the three states are told apart: a deleted file
reads "Removed", an aged-out one "Expired" (same tombstone, no download), and a file the platform
could not READ is neither - it keeps the problem styling and still offers its bytes, because
ingestion failing says nothing about the file, and its copy says so.

**Row fill anchors to the bubble.** When a sent question's chips wrap, the row nearest the message
packs full first and leftovers hang above it (`flex-wrap: wrap-reverse`), so the files read as
growing out of the message. The reply row needs no flip. Tab order visits the bottom row first; the
chips are an unordered set. The row sits the same 10px from its bubble that the reply's file row
keeps, so the two sides of the conversation breathe identically.

**The chip's action pair:** two 16px icon buttons sharing one shape - a download arrow, then the X.
Buttons, not links: a link's middle-click and "save link as" reach the route without any handler, and
a refusal fetched that way is written to disk under the file's own name. While a transfer is in
flight the arrow becomes the running ring, the chip dims and the X is disabled with it - the chip is
busy as a unit. *Recorded deviation:* the buttons carry `::after` overlays widening their targets to
22x30 and 24x30 (each growing away from its neighbour, never past the chip's 31px height), both
under the 40px floor, because growing the chip would trade the prototype's density for a notional
gain - the same call the sidebar's row trigger makes.

### Model whereabouts and the composer notice

The topbar's `model-status` reads the model name in mono, a state word (in memory / loading / on
disk) and a dot: filled `--s-ok` in memory, a ring on disk, the matrix loader in `--s-warn` while
loading. A notice that belongs to the composer docks ABOVE it, in the layout (`composer-notice`):
"<model> is loading from disk - it will be ready shortly, keep typing", "<model> is still loading -
the message is queued, no need to resend". It opens as a row, never as a toast over the thread, so
it can never overlap the status row. The composer's warn subject is the message about to be sent,
so the staged-image notice (same left rule and wash as the dialog caution) is the only other line
there.

### Error

An `error` state renders an inline error block in the turn with a retry affordance; in the timeline
it is the `--s-danger` step, and the sky washes toward `--s-danger` and dims. Every typed error code
keeps this note except `no_chat_model_configured` (see [The first-run notice](#the-first-run-notice)).

---
## Token architecture

Three tiers of CSS custom properties. Components may reference **only tier 2 and tier 3**; raw
palette values never appear in component CSS.

- **Tier 1 - primitives** (`--p-*`): the theme's raw ramps. Free-form per theme.
- **Tier 2 - semantic roles** (`--s-*`): the meaning layer every theme must fully define:
  - Surfaces: `--s-bg-app`, `--s-bg-panel`, `--s-bg-raised`, `--s-bg-inset`
  - Text: `--s-text`, `--s-text-2`, `--s-text-3`, `--s-text-inverse`
  - Accent: `--s-accent`, `--s-accent-hover`, `--s-on-accent`
  - Status: `--s-info`, `--s-ok`, `--s-warn`, `--s-danger`
  - Lines: `--s-border`, `--s-border-strong`, `--s-focus`
  - Effects: `--s-on-status` (glyph color on status-colored badges), `--s-scrim` (overlay
    backdrop), `--s-shadow` (floating-element shadow color)
- **Tier 3 - component tokens** (`--c-*`): defaults derive from tier 2 in the base sheet; themes
  override sparsely. Families: `--c-bubble-user-*`, `--c-bubble-assist-*`, `--c-thinking-*`,
  `--c-tool-*`, `--c-composer-*`, `--c-sidebar-*`, `--c-citation-*`, `--c-code-*`, `--c-step-*`,
  `--c-menu-*`, `--c-dialog-*`, `--c-picker-bg`.
- **Shape and motion tokens** (also tier 3): `--radius-s/m/l`, `--radius-bubble`, `--clip-panel`
  (a `clip-path` or `none` - how Phosphor gets diagonal-cut corners without forked markup),
  `--dur-fast/base/slow`, `--ease-standard/entrance/spring`, `--font-ui`, `--font-prose`,
  `--font-mono`.

### Theme selection attributes

Set on `<html>`, persisted per user, applied before first paint (see IMPLEMENTATION.md for the
no-flash script):

- `data-theme="porcelain|minimal|phosphor|aurora|manuscript|nocturne|abyss|grove|ember|sorbet"`
- `data-mode="light|dark"` - only meaningful for paired themes (Porcelain, Minimal); paired themes
  follow the OS by default, single-mode themes ignore it.
- `data-accent="green|amber|red"` - Phosphor only; other themes ignore it.
- `data-sky="shader|css|off"` - Aurora only, and set by the app rather than chosen as a theme:
  the sky's renderer (see [The sky](#the-sky-energy-and-tint-per-state)).

### The flourish layer

Signature geometry and decoration live in a per-theme `@layer theme-flourish` block that may target
only `[data-part]` hooks. Allowed: backgrounds, borders, shadows, `clip-path`, `::before/::after`
decoration, animation, filters, typography. Forbidden: changing the layout of the app regions
(`sidebar`, `topbar`, `thread`, `composer`) - their display, position, order, or sizing beyond
padding. Within message-level parts, display changes are allowed when they preserve content order
and accessibility - this is how Manuscript renders citation chips as a footnote list while
Phosphor cuts panel corners diagonally, on identical markup.

### Surfaces: glass is three jobs

Glass is three jobs, and only one of them blurs.

- **The sidebar and the console rail** carry the one backdrop blur (22px, saturate 1.2 - the one
  surface allowed past the 14px ceiling, being a single pane that never animates its radius) under
  a shade fill that deepens toward the outer edge, where the labels start, and opens along the edge
  that meets the thread.
- **The composer** is a pane straight onto the sky: the veil is opened behind it, so a translucent
  gradient fill is all the frost it needs. Focused, it breathes: a soft accent halo past the focus
  ring, in over 320ms and then slowly to and fro on the horizon's 5.2s cycle (still, at its resting
  value, under reduced motion).
- **Cards** - the user bubble, sources, code, a confirmation, console rows - sit on the veil where
  nothing sharp is ever behind them, so they are lit panes: a 4% light fill and an inset edge light,
  no filter.
- **Floaters** - the model menu, the citation popover, the jump button, the row menu, the dialog,
  every drop-up - pass over text, so they carry a real 16px blur with only a little lift (the
  fuller fill carries the contrast; lifted further, the sky's brightest curtain came through as a
  wash the labels could not hold against), and their hosts carry none, so a
  floater always samples the page. The hover fill under an action row is a tint, not a frost.

Under `prefers-reduced-transparency` the sidebar and rail go solid, the sky goes off, every blurred
floater goes solid, and every lit pane keeps its fill.

The recipes that keep this true:

- **A surface that hosts a positioned descendant carries its frost on a pseudo-element.** A surface
  that frosts itself with `backdrop-filter` becomes a containing block for fixed descendants (which
  is what once put an open select list at the viewport origin). The frost sits on `::before`
  (absolute, `inset: 0`, `z-index: -1`, `border-radius: inherit`) with a stacking context on the
  host from `z-index`, never `isolation` - `isolation` re-creates the containing block the move
  exists to remove. The dialog and its scrim follow this; **frost directly only on a surface with
  nothing positioned inside it.** A view-transition name is the same kind of trap (see
  [What travels between views](#what-travels-between-views)).
- **A top-layer surface needs a solid token of its own.** The open list of a customizable select
  cannot carry a `::before`, so it cannot frost, and a glass dialog fill would let the page read
  through the options. A frosting theme declares `--c-picker-bg` solid (aurora reuses the
  `#141730` its reduced-transparency block already pairs with `--s-text`), and the shared rule reads
  `var(--c-picker-bg, var(--c-dialog-bg, var(--s-bg-raised)))`, which keeps minimal legible in both
  modes with no declaration at all.
- **Floating fills.** The row menu runs the `--a-glass-fill` recipe through `--c-menu-backdrop` /
  `--c-menu-bg`; the dialog is a touch denser (a destructive decision over dense text), and its
  scrim blurs the page 6px, static for the dialog's lifetime, so the question reads first and the
  page becomes context.
- **Raised, not recessed:** in-answer surfaces on glass are raised planes (see
  [Content and markdown](#content-and-markdown)).
- **Scrollbars are one quiet treatment everywhere:** `scrollbar-width: thin` with a thumb derived
  from `--s-text-3`, set once on the root and inherited; themes tint it through their own tokens.

---

## State choreography

The SSE stream drives a small UI state machine. Every theme must express all of these states; the
prototypes include a harness that toggles each one.

| State | Trigger | Base expression (themes restyle, never remove) |
|---|---|---|
| `idle` | no request in flight | composer active, ambient background at rest |
| `waiting` | request sent, no first token | themed "reaching" indicator in the assistant slot |
| `streaming` | answer tokens arriving | caret glyph at the text edge; tokens fade in over 60ms |
| `thinking` | `thinking` deltas arriving | thinking pane open + the theme's ambient tell |
| `tool-running` | `tool_running` status | tool row in running state, named tool + spinner |
| `confirm` | side-effect halt | confirm card; composer disabled except approve / deny |
| `error` | typed SSE error | inline error block in the turn, retry affordance |
| `done` | `done` event | caret removed; citations renumbered and settled |

Two rules with teeth:

- **The thinking tell is ambient, not modal.** Each theme picks one background-level signal (a
  breathing gradient, a sonar pulse, a scanline shimmer) that runs only while `thinking` is active.
  It must be visible in peripheral vision and invisible in a screenshot's worth of attention - and
  it must disappear entirely under `prefers-reduced-motion`, replaced by a static badge on the
  thinking pane.
- **Citation renumbering is a designed moment.** On clean completion the backend rewrites `[n]`
  markers to 1..k; the UI swaps its streamed copy from `done.content`. The swap gets a 200ms
  cross-fade on the affected markers - never a flash of the whole message.

### The sky: energy and tint per state

Aurora's ambient tell is a sky behind the app, held back from the reading column and the composer
by a veil - a radial shade sized to the pane (84% night at its centre, open at the edges) that
widens on the thread, shrinks to a pool on a new chat, and carries a feathered window where the
composer sits. Three renderers exist and the app, not the person, picks one through `data-sky`:
`shader` (one WebGL plane at half resolution capped at 960px wide, 30 fps while the weather changes
or a turn is busy, 20 at rest, one still frame under reduced motion), `css` (three pre-blurred
curtain layers moving on transforms only) when WebGL is unavailable, and `off` (the flat night) under
`prefers-reduced-transparency` or on a low-power device. The sky answers the turn with an energy and
a tint per state, eased toward over about half a second so a change reads as weather, not a cut:

| State | Energy | Tint |
|---|---|---|
| idle, done | .36 | teal (done pulses once) |
| waiting | .48 | teal |
| loading (model from disk) | .30 | amber |
| thinking | .92 | violet |
| tool, and a console check | .72 | teal |
| confirm | .40 | magenta |
| streaming | .56 | teal |
| error | .30 | washed toward `--s-danger` |
| download (the runtime fetching, console) | .46 | amber |

The tint is a registered custom property (`--sky-tint`) that eases over 1.2s, so everything on the
CSS side that borrows it moves with the sky. Two things do. The **horizon**: the topbar's lower edge
takes the running colour while a turn is busy - a 2px core line and a bloom that falls onto the
thread - breathing on a 5.2s cycle; it is the tell on narrow screens and, held still, the static
tell under reduced motion. The **constellations**: a sharp 2D canvas layer of stars over the soft
sky, placed at the edges and corners with the centre nearly clear, among which a spark now and then
traces a line through a few neighbours in the sky's tint - more often while thinking, resting while
a confirmation waits, dimmer on an error, every star lit for a moment by the done pulse - drifting a
little with the thread and streaking outward on a jump.

---

## Motion language

- Durations: `--dur-fast: 120ms` (hover, toggles), `--dur-base: 200ms` (entrances, fades),
  `--dur-slow: 320ms` (panel slides). Ambient loops run 4s or slower.
- Entrances: fade + 4px rise, `--ease-entrance` (decelerate). Exits: fade only, faster.
- Streaming text: each appended chunk fades in over 60ms; no per-character animation.
- Springs (overshoot) are a theme choice (`--ease-spring`), used only by themes that declare them
  (Sorbet) and only on small elements (chips, toggles) - never on text or panels.
- `prefers-reduced-motion: reduce` disables every ambient loop and every entrance transform
  (opacity-only fades remain), enforced in the base sheet with a blanket override so a theme cannot
  forget it (see [Accessibility](#accessibility) for the aurora parts).

### Following the stream

Following is a distance, never a direction. "At the end" means within 80px of it, re-measured on
every scroll; only a scroll the reader made (a wheel, touch, pointer or key signal just before it)
counts, and every other scroll event - anchoring, a panel collapsing, the notice closing, the
thread's own chase - is followed. Following eases toward the end over a few frames so streamed lines
land without a snap; the jump glides there (420 to 1100ms by distance, eased in and out) and ends at
the end even if the content grew on the way. A wheel notch upward or a paging key is intent to leave
whatever the distance; sending, and answering a confirmation, re-stick. Reduced motion jumps. The
thread's lower edge fades out under a mask, so text dissolves before the composer instead of being
cut by it.

### What travels between views

Thread and new chat: the composer box (420ms), by every route between the two - a first send,
opening a thread from the new chat, and New chat from a thread. Thread to thread moves nothing, and
an opened conversation with no turns yet stays a new chat. Boot to new chat: the mark, the headline, the nodes
(560ms; see [Sign-in, boot and new chat](#sign-in-boot-and-new-chat)). Sign-in to boot: the mark.
Chat and console, both ways (360ms). The chat's chrome is a full-height sidebar beside a bar and
the console's a full-width bar over a rail, so the two never share a shape at the top-left corner
and no box is morphed across it: the side column stands full height for the whole travel, each
chrome drawn at its natural size where it stood, the old fading as the new arrives in place; the
bar's box does travel, from the sidebar's edge to the window's, keeping its right edge, so the
console's bar is revealed from the left like a wipe over the sidebar's head (and cropped away
again on the way back, fading quicker than it shrinks). The bar paints over the column, so the
corner is the brand fading under the arriving bar, never a hole. On a phone the sidebar is a
drawer off the edge, so only the bar travels. The way back lands on the conversation the reader
left (remembered for the session on the way out) or on a new chat; a chat carried in by the travel
skips its boot screen and renders its shell at once, running the boot's waits behind it for the
reach line, so the travel never waits on the backend and the thread's content arrives behind the
chrome that moved. The root cross-fades in 260ms. A
view-transition name is carried only while a transition runs - a named element is a backdrop root,
and the menu inside it would blur nothing - and an element a transition carries in has its own
entrance suppressed, or it replays as a flicker. Reduced motion cuts between views.

Between a thread and a new chat the page behind the composer is not cross-faded as one image. It
stays live, and only what the two views do not share fades (260ms): the greeting, the thread with
its jump button, the bar with its title. The line under the composer travels with the box, and the
rest of the page (the sidebar's lit row) changes on its own transitions. Two cross-fading images
of the page would each carry aurora's veil window where the composer stood in that view, so the
sky would glow through both ends of the trip while the box was still between them; live, the
window travels with the box, on its clock and curve. The star chart leaves as it arrives: out over
260ms where it stood when a thread takes the pane. The window is placed from the composer's box
whenever anything above it changes size, arrives or leaves, so it follows the box on a phone too,
where the box is the same width in both views.

Two rules any part named for a travel has to keep. A named element's backdrop filter moves to
its transition group, which is a plain rectangle, so a frosted disc would leave a square of blur
behind as it faded: the jump button is named only while it shows, and its group drops the filter.
And an entrance held back with `animation: none` starts the moment the hold is lifted, a second
arrival as the travel ends: the greeting's entrance is a transition out of its starting style,
which can play only as the greeting first renders, and is switched off while a travel carries
it in.

### The generated title reveal

A background job names a new conversation moments after its first exchange, and the sidebar watches
for the rename. The swap is the one moment the app visibly writes copy on its own, so it is staged as
exactly that: the name TYPES into place over about a second (28ms per character, longer titles
proportionally faster so every reveal lands in the same beat) with a steady 2px caret in
`--s-accent`; the sidebar row and, when that conversation is open, the topbar play it together.

- **Only the watch reveals.** A manual rename swaps plainly - re-typing it back would claim
  authorship the app does not have. Editing or deleting a row mid-reveal ends the reveal at once, and
  a reveal with no surface to play on expires quietly instead of firing later out of context.
- **Layout never moves.** The reveal types inside the row's reserved footprint and the settled title
  ellipsizes exactly as static text does; mid-reveal clipping of an over-long name is accepted for
  the sub-second it can exist.
- **Per-theme voice.** Minimal types in plain row ink. Aurora paints the reveal on an overlay of the
  same glyphs (`--p-aurora-teal` into `--p-aurora-magenta` sweeping through resting ink) while the
  base title holds its fill invisible beneath - the band must be the glyphs' own fill against the
  dark backdrop, or its antialiased edges wash out to a ghost - and the two layers cross-fade over
  about 200ms when the last character lands, so the settled title is pixel-identical to static text.
- **Reduced motion is an instant swap** (no caret, no sweep), and forced-colors mode drops the
  clipped gradient for plain ink. **Assistive tech never hears the typewriter:** the animated
  fragment is `aria-hidden` beside a visually-hidden copy of the full name.

---

## Typography

- Families are tokens (`--font-ui`, `--font-prose`, `--font-mono`); the app ships Geist Sans and
  Geist Mono, and themes may re-point the tokens (Manuscript uses a serif prose stack; Phosphor sets
  all three to mono).
- **Prose:** `--text-body` in the thread, never justified, measure the thread's `50rem` column.
  **Aurora overrides it with a serif:** Source Serif 4 at 16.5px / 1.68 with a 68ch measure, the one
  face the theme adds, served locally like the others.
- Code: `--font-mono` at 13px in blocks; inline code picks up `--c-code-*` chip styling.
- Hierarchy comes from weight and spacing, not size explosions. The chat surface is content-first:
  the thread content is the reading anchor and the chrome steps down from it, on a discrete token
  scale in the base sheet rather than sprinkled literals:
  - `--text-display` 20px - the one display size, for the empty-state headline (full contrast).
  - `--text-body` 15.5px / 1.75 - thread prose and user bubbles, the reading size.
  - `--text-heading` 17px - page-section headings on management surfaces.
  - `--text-title` 14px semibold - topbar title and sidebar brand.
  - `--text-label` 13px - interactive chrome (conversation rows, buttons, primary actions).
  - `--text-meta` 12px - quiet chrome (tool lines, citations, notices, meta).
  - `--text-micro` 11px - micro chrome (section labels, hints, footer notes, badges).

  Where this scale conflicts with the prototypes' flatter "everything 12-15px" scale, the scale
  wins. Console page titles use the serif display at 30px (see [Console shell](#console-shell)).

---
## Shared components

Patterns used by more than one surface, built from tokens the system already defines.

### The confirm dialog

A confirmation for what cannot be undone: a scrim at `--s-scrim`, a centered `min(400px, 92vw)`
panel at `--radius-l` over a hairline border, fading and rising in at `--dur-base`, holding a
`--text-title` question, one `--text-label` line saying what the action costs, and the
`.btn-ghost` / `.btn-danger` pair, safe answer first, each a full 40px. Aurora frosts and lifts it,
minimal ships the hairline alone (fills in [Surfaces](#surfaces-glass-is-three-jobs)). The floor:
conditional rendering (never the `hidden` attribute), `role="dialog"` with `aria-modal` and an
`aria-labelledby` heading, initial focus on the safe answer, Tab cycling inside the panel, Escape
and the scrim both cancelling, focus returned to whatever opened it.

- **Tone.** The confirm button defaults to the danger fill; a `tone: "accent"` variant serves
  actions that are consequential but not destructive - Pull updates, the tool checks, a declaration,
  a confirmed measurement. A benign verb confirmed in blood red un-says its own benignity.
- **The count.** Where an action also discards what came after it, the body states the count (see
  [Turn actions](#turn-actions)).
- **The one input.** The dialog may carry ONE input the confirmed action needs, between the body and
  the buttons (the first user: the delete-the-default dialog's successor picker), in the console
  input's clothes at its 42px rendered height, inside the focus trap. With an input present the
  confirm may be WITHHELD while still rendered: when the only honest answer is Cancel (no successor
  exists), the body says why and the disabled confirm keeps its box, so the dialog's shape never
  changes with its answer - the explanation is the body, not a tooltip.
- **The caution.** A dialog inherits the one-warn-slot-per-subject rule with the DECISION as its
  subject: exactly one warn line (`.dialog-caution`) carrying what the action will cost or why the
  current state is what it is, never restating the body in a louder colour, tinted by a left rule
  and a wash rather than by colouring the sentence (a whole paragraph in the warn hue measures worse
  than the state it warns about).
- **Keyboard capture.** While a menu or a dialog is open it owns the keyboard: the chat page's
  type-anywhere shortcut is suspended, since focus inside either sits on a button and would be
  pulled away mid-interaction. Any future overlay joins that exclusion when it lands.

#### The open select list

A page built from selects cannot leave the OPEN list to the operating system - a white sheet with a
system-blue highlight would be the one part of the console that is not the console. Where the engine
offers the customizable select (Chromium's `base-select`), the open list joins the design system
through one shared recipe: the menu surface's fill, border and shadow at the trigger's width, 40px
options with the accent-bar checked treatment (the UA checkmark hidden - the bar already says it),
anchored BELOW the trigger, scrolling past 16rem. The list never declares its own `display` (a
closed picker is hidden by a plain UA `display: none`, so any author `display` renders it while
shut); row spacing is sibling margin, not flex gap. On aurora the popup is solid through
`--c-picker-bg`. Engines without `base-select` keep the native popup, which the `color-scheme`
wiring renders dark. Every theme sheet loads on every route, so a new surface joins by adding a
selector.

A UA-anchored list needs the UA's own knobs, not hand-written insets, and there are two independent
traps. A picker that renders at the viewport's top-left corner has a frosting ancestor acting as its
containing block (the frost-on-`::before` recipe fixes it; `anchor()` insets do not). A picker that
opens UPWARD with a clean ancestor chain is Chrome's UA `position-try-order: most-block-size`
reordering the try list toward whichever side is taller; the fix is `position-area: block-end` with
`position-try-order: normal`, keeping `position-try-fallbacks: flip-block` for genuine no-room.
When a UA-owned popover sits wrong, check the UA's try-order before reaching for insets. A custom
menu the app owns (the composer's model picker) anchors in a `position: relative` wrapper and is
unaffected by any of it.

### The row menu and rename

- **One action slot per row, holding a menu.** The row keeps a single trailing kebab, and every row
  action lives behind it, so a fourth action is a menu item, never a third icon crowding a 272px
  panel. The panel follows the model drop-up's anatomy and the theme picker's behaviour (outside
  mousedown and Escape close, `aria-expanded` / `aria-controls`), plus the semantics a menu owes:
  `role="menu"` / `role="menuitem"`, arrow keys walking the items, focus back on the trigger on
  close. Items (Pin/Unpin, Rename, Delete) carry a 14px stroke glyph in the topbar pencil's idiom
  inheriting the item's colour, so the danger item's icon reds with its `--s-danger` text.
- **The panel renders at the APP LAYER** (a portal, viewport-anchored), dropping down-right from the
  kebab over the content: the scrolling list clips anything positioned inside it, and the sidebar is
  itself a blur layer inside which a nested blur samples nothing. It wraps its items (`min-width`
  7.5rem), dismisses on any scroll or resize (it cannot follow its row), and flips above the trigger
  only when the viewport would clip it. Aurora frosts it through `--c-menu-backdrop` /
  `--c-menu-bg`.
- **The whole row answers for its kebab:** hovering the trigger lights its row, and the row stays lit
  while its menu is open, so the menu never leaves doubt about which conversation it acts on.
- **Renaming happens in place, at the size of the thing being renamed.** A row's title becomes an
  input at `--text-label` in the row's footprint, the topbar title an input at `--text-title`, and
  the list must not jump when the edit commits. In the topbar the title itself is the trigger, with
  a pencil fading in on hover (simply present where there is no hover). Enter commits, Escape
  cancels, blur commits, and an empty or unchanged value is a quiet cancel that sends no request.
  The topbar input fits its text and grows with it (a hidden replica with identical metrics shares
  its grid cell) and mirrors the title button's inset, so entering, typing and leaving never shift
  the first character. The sidebar rename field draws its focus ring inside itself (rows sit flush,
  so an outward ring is clipped and painted over by a neighbour's hover), and a focused row raises
  itself one z-index step.
- **Hit targets.** The menu trigger is a 40px-wide target spanning the row's full height, the visible
  chip staying the small quiet square; menu items are a full 40px; rows keep the prototype's
  density. *Recorded deviation:* the trigger's vertical extent is capped by the row pitch rather
  than reaching 40px on both axes.

### The context meter

The backend reports how full the model's context window was on each turn. The meter
(`data-part="context-meter"`) is a 16px ring at the head of the composer's right dock, filled to how
full the window stood on the LAST completed exchange. Clicking it opens a drop-up card (the model
menu's surface and geometry) hinged on the arc's left edge and opening to its right, above the
model and Send rather than over the composer's own text; its head is label-sized and its sentences
meta-sized. It carries a bar, the numbers ("About 340 of 8,192 tokens used last turn.")
and the consequence ("When the window fills, the oldest messages stop being sent.") - answering "is
it still reading all of this?" before the answer shows up as forgetting. A ring rather than a
standing label because the percentage only matters once it is high; the card puts the sentence
where touch can reach it, and the trigger's `aria-label` carries the whole story.

- **The number is an ESTIMATE, and the wording admits it:** the figures are the backend's
  character-based accounting, so the card says "about" and the ring draws the RATIO.
- **The ring is the theme's one dial:** a hairline track in the quiet ink, the accent arc with a
  little of the sky's glow behind it, 32px tall like the chips beside it. The card docks one more
  line under a hairline, in the quiet tint: the later action (compressing older turns to make
  room), named so a full window reads as something with a remedy coming rather than a wall.
- **Thresholds: quiet below 75%, `--s-warn` from 75% with the figure beside the ring, `--s-danger`
  from 90% with a wash behind it.** Colour first, chrome only at the top step, catchable on the
  next glance without interrupting the sentence being
  typed. No animation or transition (a dial easing into a colour reads as something happening), and
  the geometry never changes with the level, so escalation cannot reflow the row.
- **Recorded deviation:** the ring's TRACK is a sub-3:1 hairline (`--s-border-strong`). The arc is
  the state-bearing part and holds the 3:1 floor; a track dark enough to pass would be
  indistinguishable from the quiet arc, so the recessive floor is recorded instead of met.
- **Hidden until the conversation's first turn reports a reading, and per-conversation** (a pane
  switch clears it and closes the card); an interrupted turn leaves the previous reading standing.

### The first-run notice

When a turn ends with `no_chat_model_configured`, the thread renders the server's message verbatim
plus one action, "Open the Model Manager" (`/console/models`, 40px), as
`data-part="no-chat-model"` - deliberately NOT in the error note's danger tint: nothing is broken,
and painting a setup step red reads as a bug rather than a step not yet taken.

### Status pills

A status pill is colour, a dot and a WORD, the word primary. The same language serves the
integrations cards, the Model Manager rows and the knowledge roster.

- **Fixed width, so dots form a column.** Pills in a right cluster share a `min-width` sized to the
  longest label the column can hold plus a few px of slack; surplus reservation lands between a
  short label and the next dot and reads as misalignment. The ceiling is 7.5rem, set by "Think mode
  differs". A name-adjacent badge (Vision, the routing badge) is NOT a state pill and never joins
  the column.
- **The tones.** `--s-warn` states are gaps an operator must close, `--s-ok` means the thing can do
  its job, muted means none is needed. The word always leads: a server backed by a deployment
  environment variable reads "Deployment key", not "Connected".
- **A pill renders only where its question has an answer;** otherwise an empty slot holds the column
  open (see [Rows and pills](#rows-and-pills)). Under 700px an integrations summary wraps, name and
  URL on a full row and the pills beneath.

### Banners

A standing banner (`mm-banner`) states a fact about the deployment that no single row owns: a
neutral box, a warn tone and an error tone, with the severity hue on the TITLE line only and the
body at `--s-text` (a paragraph set entirely in the danger colour reads as a wall of alarm).
`role="alert"` is reserved for fatal gap cards - announcing every note would train a screen-reader
user to ignore the one that matters - and standing notices are not live regions: facts a reader
arrives at, not events announced at them. Placement is in
[Gap cards and standing notices](#gap-cards-and-standing-notices).

---

## Console

The shell and the rules every console page follows are in [Console shell](#console-shell). The
console's only prototype is the Model Manager view in `aurora-next.html`; the rest is built from the
token system.

### Model Manager

#### Rows and pills

A manager row reuses the integrations card anatomy: a summary grid - the model tag in the monospace
face and its badges, the state pills, a reserved action column, the expand chevron - over a body of
detail. The pills are the shared status-pill language, reserved to a common width.

- **The chevron lives at the card's right edge, always.** The row's action (Pull, Resume, the live
  readout) overlays the reserved column OUTSIDE the expand trigger, and an empty slot lets a click
  fall through, so a click anywhere on the head opens the row. Focus rings are inset on these
  edge-to-edge triggers. On a phone the pills stack under the name and the slot keeps its column
  only while something is in it.
- **The check pill is check-worded.** The page shows hardware facts the moment a model lands, and a
  pill saying "Measured" read as one of them when it reports the TOOL CHECK. The vocabulary: `Not
  checked` for a model nothing has probed (never a blank, never an estimate), `Checked on CPU` for a
  probe the model spilled off the GPU for, `Check stale` for one the runtime or harness has moved
  past, `Checked` otherwise. The body spells out what the pill abbreviates - the CPU case in full
  ("Checked on CPU - not representative"), because 0 GB and no throughput with nothing saying why
  reads as a corrupt measurement.
- **Empty slot for unmeasurable roles.** The pill renders only where a check is possible or a probe
  exists; an embedder or a vision model holds the column open with an empty slot, since "Not
  checked" on a row that CANNOT be checked reads as a gap somebody should close.
- **Name-adjacent badges.** Vision earns a place in the COLLAPSED row - it is the one capability
  whose absence a user meets as a surprise - as a `.badge.cap` beside the tag in the capability
  tint, and only when true (a "no vision" badge on nearly every row would bury the few that have
  it). Exactly one row wears a routing badge, "which model answers when nobody chose?", resolved
  through the same chain a turn uses (the stored workspace default, else the deployment binding), in
  the accent tone and LEADING the cluster because it is about the whole deployment. Neither badge
  touches the pill reservation.
- **Installed rows lead every model list** - the grid and the composer picker both. A catalogue of a
  dozen rows with two installed, listed with the un-choosable first, reads as a broken control. An
  ordering, never a filter, and stable within each half so the backend's own order survives.
- **Per-theme voice.** Aurora lights the row as a pane; Minimal flattens it.

#### Facts

Measured facts read "9.5 tok/s - 2.4 GB VRAM - deep check, 20 Jul 2026"; a catalog's declared size
reads "declared 9.3 GB" beside them. The page draws no comparison the payload does not contain and
shows no predicted fit: a prediction wrong on the machine it ships on is worse than an honest "Not
checked". **A fraction is rendered only where the number carries meaning:** the loaded-now readout
says "fully on GPU", "on CPU", or "N% on GPU", because the percentage is the fact only in the middle
case, and "0% GPU" explains an order-of-magnitude slowdown far worse than "on CPU" does. A runtime
that reports nothing renders nothing; "100%" for an unreported reading is the confident wrongness
this page exists to avoid.

#### The check zone

In the row body, an overline in the fact-label voice ("TOOL CHECK") gathers the measured line, the
staleness note, the agency verdict and the check buttons above a hairline, apart from the declared
facts - a body reads facts, then the check's story, then the destructive zone. Roles with no check
story have no zone. Capability chips step up to `--text-meta`, and a genuine capability (Tools,
Vision, reasoning with tools) carries a quiet accent tint mixed toward `--s-text` for AA on the light
theme, while mode words stay neutral. **A zone's verbs collect at its foot in one row** - the two that
measure, the one that declares - each absent where it does not apply; they measure weights rather
than maintain them, so they never sit beside the housekeeping verbs. **Routing gets its own zone**
(same anatomy, overline "Routing"): nothing in it is measured, and a routing choice under a heading
about measurement teaches that the two are the same kind of fact.

#### Verdicts and warn slots

- **A verdict is three channels:** the status word ("Earned assisted"), the per-case tallies as a
  self-sizing monospace chip grid, and any caveat as its own warn line.
- **A resolution is reported as its ANSWER, not its inputs:** the mode on a line of its own, a
  plain-language line saying what it permits, and - only when something clamps - the backend's own
  sentence naming the one factor that decided and what would change it. (Listing ceiling, earned and
  clamp in one run of text left the reader to do the `min()` themselves.) The glosses are shared
  source with the declare dialog's, so a card and the dialog that changes it cannot tell two stories.
- **A clamp names the cases that clamped** - worst first, capped, read from the probe row whose
  evidence drives the clamp, never a case that merely dropped a sample while passing its bar. Tally
  chips carry their gloss as hover text, and every FAILED chip's gloss also renders as a quiet line
  beneath the grid, because hover reaches neither keyboard nor touch.
- **One warn slot per SUBJECT.** The zone's notes judge two subjects. *The verdict's validity:* at
  most one validity note per row, a weights-change note outranking a runtime, harness or endpoint
  staleness note (the newer event, and the one with a named fix). *The think flag:* a routing hazard,
  standing BESIDE a validity note and rendering last - verdict first, hazard second. Suppressing
  either would drop load-bearing information, which costs more than a second amber paragraph.
- **The weights-change note is cause first, backend facts only.** Derived per depth from the probe
  log against the installed digest, never stored, self-clearing the moment a check runs on the
  installed weights. It names which depth's verdict was orphaned, when it ran and what it earned,
  and closes with the fix in the words of the buttons beside it ("running a deep check re-earns
  it"). It never guesses WHY the weights changed - a re-pull, a registry update and a manual tag
  move are indistinguishable from a digest - and "changed" is the strongest word allowed, since
  there is no ledger of who moved a tag. Warn tier (`mm-note warn`), not error: nothing is broken,
  the model simply stands unmeasured as installed.
- **The same fact surfaces at two more volumes,** derived on the client from the row payload: the
  collapsed pill reads "Weights changed" in the warn tone (a verdict measured on other weights makes
  every other pill word a worse summary, and "Not checked" reads as a model nobody has got round
  to), and the page carries ONE standing notice while any row's weights outran its check. **One
  predicate decides for all four surfaces** (pill word, pill hue, body note, page notice): the pill
  warns exactly when the body has a warn note to show.
- **"Think mode differs"** is a pill word in the warn tone, under the same predicate. **A measurement
  can be confirmed, never declared:** when a current-harness check's measured think mode disagrees
  with the catalog's declaration, the note names both sides and a confirm control stores the
  MEASURED value (the request names a model and nothing else), absent when nothing is measured, in
  the accent tone because agreeing with a measurement is not destructive.
- **A measured "always" model states its consequences as a standing hint:** its turns run with think
  forced on and, while thinking-with-tools stays gated off, with no tools offered; the hint says so
  and names the grounded pre-fetch that returns in their place. Load-bearing prose, hint-reason tier.
- **The endpoint staleness note joins the validity family,** claiming only what the flag supports
  ("no record of this check running against the model server serving now"): a row written before
  endpoints were tracked and a run on a different server are indistinguishable in the payload.

#### The activity panel

Exactly one long operation can run (the backend is single-flight), so it renders once, in one panel
above the grid, and every other action on the page disables while it does. It is the heavier pane,
and while something runs a comet goes round its border - a conic sweep in the sky's tint, masked to
a 1.5px ring with a blurred glow behind it, one steady 4.8s lap, fading in and out over 900ms - while
the sky follows (amber while the runtime fetches, teal while a model is being worked). One activity,
two renderings: **a download** shows its stage wording over a progress bar (a `--s-bg-inset` track
with an `--s-accent` fill whose width transitions, collapsed to instant under reduced motion), and
**a check** shows each finished sample as a pass/fail line as it lands. A download that continues
into its automatic check is ONE activity moving between the two, because that is what it is on the
wire.

- **The click is answered where it happened.** While a row's own download runs, its action slot
  becomes a live readout (tabular percent over a 3px accent hairline) rather than going mute while
  the progress renders only in a panel that may be scrolled off-screen; activating it brings the
  panel into view. The row being checked carries the same mini with `n/N` samples (the total is
  small and the unit is the story), retiring the moment the verdict or failure lands - the stream
  outlives the check by the whole re-warm, and a row saying "Checking" beside a panel saying the
  check is over would be the page disagreeing with itself.
- **The indeterminate stripe.** Both bars carry a moving diagonal stripe whenever no fraction can
  honestly be stated (a check before its first sample, a download stage without byte counts); the
  fill's base width is zero, since an unknown fraction painting a FULL track reads as finished
  through an entire cold load. The stripe is full accent ink made quiet by coverage rather than
  alpha (it must clear the 3:1 floor on minimal light); under reduced motion it stands still and
  still reads "working, no number yet"; forced-colors modes drop it and the adjacent text is the
  tell. The moving layer is a static overlay panned by transform one true pattern period, never an
  animated background position: a repeating gradient rasterizes to a finite tile whose fractional
  period fits no whole pixel, and a position animation surfaces that as a seam travelling the bar
  (observed 2026-08-06).
- **One gloss line per case batch.** The live sample list groups by consecutive case name, each batch
  headed by one muted line saying what the case asks (`--s-text-2` at the meta size - the only plain
  English for an identifier like "loop-guard"), as presentation rather than an item, so a deep run
  still announces as 26 samples and a fast run as 4. The copy is a display-side map keyed by case
  name, living in the client because the harness fingerprints its own spec constants and wording
  beside them could never be edited without invalidating stored verdicts; an unknown case gets no
  line.

#### Stop and Pause

Cancellability follows the cost of stopping. Both confirmations are the shared dialog.

- **A download PAUSES, never cancels.** The transfer is content-addressed, so stopping keeps the
  partial and resuming is free: the button says "Pause" and why that is safe, the terminal state
  says "Paused at N%" over the frozen bar, and Resume sits beside Dismiss (a failed transfer gets the
  same Resume - resume and retry are one request). The page remembers unfinished transfers for the
  session (the runtime lists no partials), so the model's own row offers Resume in place of Pull with
  the frozen bar in its body. A stream that dies without its terminal frame is named as a failure
  with the same Resume path, never left rendering as a download stuck at its last percent.
- **A check STOPS.** The running check panel carries Stop beside its progress, with the hint
  "Stopping discards this run: the last verdict stands, and the chat model and the embedder are put
  back." The terminal state is "Stopped <model>" over the samples that landed, with "Stopped. The
  last verdict stands; the chat model and the embedder are being put back." and Dismiss alone (a
  stopped check offers no Resume: it would be a new run). Never pause: a stopped check re-runs from
  the start. This binds the backend as much as the
  page: the abort path on the check stream must re-warm the chat model and the embedder exactly as
  the stream's own terminal frame does, or the hint is not true.

#### The add-a-model box

Above the grid, after the activity panel and before the first role section: it is how anything
outside the catalog arrives at all, and at the foot of a long list it read as fine print.

#### Housekeeping verbs

Installed manageable rows keep a "Pull updates" verb in the row's foot, before Delete (benign verb,
then destructive one), never as the summary's call to action, behind the confirm dialog in the
accent tone, which names the consequence: changed weights reset the recorded check, and a quick
check re-earns the verdict after the download. The locked embedding role never gets it - changed
weights under an unchanged tag would shift the vector space silently. Deletion on that role is
per-row: an installed embedder neither active nor referenced by any stored passage offers Delete,
refused rows carry the backend's reason beside the disabled button, and a vector store that does not
answer fails the verb closed.

#### Gap cards and standing notices

The Model Manager opens with zero or more gap cards, one per feature the server's inventory leaves
broken or degraded, from the backend's own findings, worst first.

- **A gap card is the banner box wearing a severity** on its title line only: the error tone for a
  fatal gap, the neutral box for a note.
- **Scan-line titles are keyed on the gap's CODE, never its severity.** Two gaps of equal severity
  can demand opposite actions (download a missing embedder; select one already installed). An
  unknown code degrades to a generic severity-worded title over the backend's sentence.
- **The suggested fix is named, not offered as a second control:** the named model's row already
  carries the one Pull verb, and a duplicate control for a single-flight operation is a race with
  nothing to gain. The tag renders in the monospace face inside the fix line.
- **Only a fatal gap interrupts** (`role="alert"` on fatal cards alone).

The standing notices follow in a fixed order: gap cards, the stranded-vectors alarm, then the
weights-change notice and the remote-server banner, then the quiet leftovers note - broken, then
unreachable, then merely unverified. Gap cards sit above the vectors alarm because a missing
embedder is why vectors strand, and that cause-above-consequence adjacency is worth more than any
other banner's position.

- **One weights-change notice while any row's weights outran its check,** in the vectors alarm's
  anatomy (`mm-banner warn`, one paragraph), naming the models untruncated in the tag face - a
  hidden name is a model nobody goes and checks. Note severity, no `role="alert"`: the models answer;
  what is missing is a measurement. **The notice names no fix; the card does.** A row can hold a
  current verdict at one depth and an orphaned one at another, and that is the ORDINARY case (a
  re-pull runs the quick check automatically, so the deep verdict is routinely the one left behind);
  a page-level "run a check" would send a reader to a button that clears nothing.
- **A standing banner may describe the DEPLOYMENT rather than any model.** A remote model server
  reached without credentials gets one `mm-banner warn` at the top of the console, note severity,
  stating the exposure and stopping short of naming a fix: the fix depends on how the host is
  fronted, and a console naming one setting would send an operator to edit the wrong file.

#### The declare dialog and the pair control

These are the patterns that WRITE a fact, and a control that changes what the deployment does needs
more of the reader than one that describes it.

- **A dialog whose answer is a VALUE gets a picker, not a row of verbs.** Three buttons would read as
  three actions and put the most consequential one a mis-click from the safest. One select, one
  confirm in the accent tone (a declaration is superseded by the next one and retired by a re-pull,
  so it is consequential, not destructive).
- **A dialog glosses the SELECTED option, never all of them.** "Agentic" is not self-explanatory, and
  the gloss is the only place a person is told what they are permitting; a glossary of all three
  reads as boilerplate and gets skipped. It sits under the picker in the meta tier.
- **The dialog's one warn line carries the decision's cost,** read from the backend's clamp
  explanation, never re-worded here.
- **A raise says out loud that it changes nothing yet.** `min(CEILING, EARNED)` means declaring
  `agentic` on an unproven model stores permission and moves no turn until a deep check passes; the
  dialog says so and names the cost of earning it (the check evicts the warm model and holds the GPU
  for minutes) rather than offering a second button that would race the single-flight operation.
- **The pair control is scoped to models that answer directly** (a model that reasons on its own has
  no thinking turn to hand anywhere, so the control is ABSENT there). Candidates are drawn only from
  models whose think mode is established as `always` or `hybrid` - "unknown" would dress a guess as a
  capability - while the HOST side deliberately includes an unmeasured model: declaring a measured
  partner is exactly the move that makes the composer's toggle mean something there.

#### The second value axis

The console can declare whether a model may reason and call tools in the same turn - the first axis
whose wrong answer produces no error at all (on some model and runtime pairs the turn returns an
empty reply), and that property drives everything below.

- **A chip renders the EFFECTIVE value, never the claim.** A chip is read as "this is what the model
  does", so a capability declared but not yet earned must not wear one; the failure downstream is a
  blank answer no log records.
- **The gap between a claim and what a turn does gets a SENTENCE.** "Granted, not yet established"
  is a third state a chip cannot hold. It renders in the check zone's hint tier only while the two
  disagree, and names where the claim came from, since a declaration made here and a line in the
  profile are undone in different places.
- **The declare dialog takes the axis on the ceiling's pattern:** one picker, two options, the
  selected one glossed as what a TURN does, confirm in the accent tone, glosses shared with the
  card's.
- **The hazard leads EVERY caution on this axis, including the withdrawal:** the warn slot opens with
  the failure mode itself - silent, empty answers - on every selection, because a reader lowering
  the claim is owed the reason the control exists as much as one raising it.
- **A caution states what is KNOWN, and asks the server which of three things that is:** established
  (the grant applies immediately), measured absent (stored and inert), or not established here (only
  the deciding rule stated - deliberately asserting nothing about evidence, since it also covers a
  measurement never READ). Inferring the outcome from the effective value is WRONG: it reads "not in
  force" as "the check failed", when the commonest reason is that nobody has declared it yet, and it
  told an operator whose deep check had just passed that granting would change nothing. The verdict
  is its own server-side field.

### Integrations

Integrations lives at `/console/integrations`, in the console shell with its own `h1`
(`.int-title`). The cards and forms carry the surface treatment (lit panes on aurora, hairline rows
on minimal) so the page reads composed over the theme rather than as a floating slab.

- **Server rows are expandable cards.** The collapsed summary: name, credential-scope badge, base
  URL, enable switch (a sibling of the expand trigger, never nested inside it), and two status
  pills. The expanded body varies by auth type - an OAuth connect panel, a header-credential
  connect/replace/disconnect flow, or a no-credential note - plus test-connection and delete.
- **Two status pills, not one.** The credential state sits left of the enabled pill: the scope badge
  answers *whose* credential, never *whether there is one*, so identically-badged rows hid the one
  server that could actually authenticate.
- **Standing credential state reads the same wherever it is found.** The body's state line carries
  the matching tone and dot on every auth type; the first clause is the state, and the guidance
  after it steps back to `--s-text-2`, hung on a 15px indent rather than a flex row (a wide guidance
  column crushed the state clause into a sliver). A card never repeats a deployment-level alarm at
  full volume - the page-level notice owns that. A stored key's tail renders as `****1234`, not
  `...1234`: the mask convention says "end of a hidden key" where an ellipsis reads as a truncated
  word.
- **Forms are token-built:** outlined input wells, accent radios and checkboxes, one accent-filled
  primary action per surface, ghost secondaries, red danger actions; every control must read as a
  control in both themes.
- **The page is grouped for the coming role split.** Workspace-level setup (registering servers, the
  preset gallery, workspace credentials) is its own titled section (`--text-heading`) apart from the
  per-user card actions, so later admin gating is a conditional render rather than a redesign.
- **A test outcome is three things:** the short status inline beside its button ("Reachable - 13
  tools." / "Failed (auth_rejected). HTTP 401"), the advertised tools as a self-sizing monospace chip
  grid on their own row, and the answered-without-a-sign-in caveat as a warn line of its own.

### Workspace defaults

The section (`data-part="workspace-defaults"`) holds the defaults a turn falls back to - chat model,
reasoning, web search, a context ceiling, the reranker toggle - and is the console's first form that
WRITES a setting.

- **The page says it is a floor, not a ceiling, before it shows a control:** one lede at `--s-text-2`
  states that a conversation's own settings and a single request both win over it. Without it the
  page reads as an override switch, and the first thing a reader would learn is that a pinned thread
  ignored what they just saved.
- **"Not set" is a first-class option, worded as an outcome** ("Not set - the deployment's own
  default decides"), never a blank or a "Default" that begs the question of whose; the enum options
  are sentences for the same reason, and the console has the width.
- **Fields keep a narrower measure than the column:** panels at `52rem`, controls `36rem`, hints
  `62ch`, all at the column's left edge.
- **A section is a panel, not a card of rows.** A form answers "what should happen", with no per-item
  state to align, so the grouping is two labelled panels (Answering, Machine budget) wearing the
  manager row's surface; Minimal subtracts them to hairline-separated sections with no tint, because
  the controls carry their own fill and border.
- **A control the deployment cannot honour is a STATUS, not a disabled switch.** The reranker toggle
  renders only where a sidecar is configured; otherwise the field shows the pill "Not configured"
  with the backend's sentence beneath. The ceiling's bounds come from the payload too, so `min`/`max`
  and the refusal wording cannot drift from what the API takes.
- **Saving is explicit, and it saves only what moved:** one Save with a Discard beside it, carrying
  the DIFF so an admin in another tab who changed one axis is not silently reverted. The outcome is
  one line on the action row ("Saved. New turns use these defaults."), not a toast.
- **The switch is never alone:** the word rides beside the 36x20 pill, and the row carries the 40px
  floor the pill itself is under. An open select is themed through the shared recipe.

### Knowledge

The section (`data-part="knowledge"`) lists what the workspace knowledge base holds, with provenance,
and lets an administrator add a file or remove one - the first page whose principal action takes
MINUTES with no progress to report, and whose subject is content rather than configuration.

- **The row is the roster's; the detail is the manager's.** One source is one row, whatever it split
  into internally: identity left, a state pill and its verb in a right cluster, full-width detail
  beneath on the `.mm-fact` label-over-value pair on the SAME `11rem` track the manager uses (a
  narrower one wraps an embedding tag mid-string in the very column built to show it). Minimal
  subtracts the row as it subtracts the roster's.
- **A wait with no fraction says so, in three places:** the verb moves onto the button
  ("Indexing..."), the indeterminate stripe runs beside a sentence NAMING the file, and the bar
  carries `role="progressbar"` with an accessible name and NO `aria-valuenow`, because a bar claiming
  a value it does not have is worse than one admitting it has none.
- **A ceiling the plumbing imposes is a NOTICE, not a failure.** Past the same-origin proxy's ceiling
  the work is still running, so that outcome renders in the success slot with what is true ("still
  being processed - reload in a few minutes"), never in the error slot: an error over a file that WAS
  stored is the one wrong answer, and the one naive handling gives.
- **Two error slots, because a refusal and a re-read collide.** A refused upload is followed by
  re-reading the list, and one slot would have the refetch wipe the refusal moments after it
  appeared. The listing's own failure and a refused action get separate slots, and a refresh never
  touches the second - the rule wherever a refused action is followed by a re-read.
- **The sharing rule is stated AT the control:** one sentence between the section title and the file
  picker says everyone in the workspace can search what is added. The lede is read once; this
  decision is made every time somebody picks a file, and it is the one that cannot be undone by
  deleting afterwards.
- **A partial list says its numbers are floors:** what is shown is part of the corpus, other entries
  are stored and searchable without appearing, and each count is AT LEAST what that entry holds -
  three sentences, because each is separately wrong to assume (the read stops in storage order, not
  date order).
- **Provenance is shown as fact, and absence as "cannot tell".** Each row carries where the content
  came from, when it was last written, how many passages it became, which embedding model wrote them,
  and a pill saying whether a query today can still reach it. A source no current query can find is
  LISTED, not hidden - it still occupies room, and hiding it would leave an operator unable to remove
  what a model change stranded. Empty provenance renders as "unknown", never as a guess.

---

## Accessibility

- **Contrast:** `--s-text` on `--s-bg-app` and `--s-bg-panel` meets WCAG AA (4.5:1) in every theme;
  `--s-text-2` meets 4.5:1 on `--s-bg-app`; `--s-text-3` (decorative/meta) meets 3:1. Accent-on-
  accent pairs used for buttons meet 4.5:1. Each theme's tokens file records its checked ratios.
- **Focus:** `--s-focus` renders a visible 2px ring (3:1 against its surroundings) in every theme,
  including the dark and pastel ones. Flourish clip-paths must not clip focus rings - rings draw
  via `outline-offset`, outside the clip.
- **Reduced motion:** see [Motion language](#motion-language); the blanket override is base-sheet
  law, not per-theme courtesy. For the aurora parts: the shader draws one still frame, the stars
  hold, the horizon holds, the comet ring and the matrix loaders go static, there are no view
  transitions, and following jumps.
- **Reduced transparency:** the sidebar and rail go solid, the sky goes off, every blurred floater
  goes solid (see [Surfaces](#surfaces-glass-is-three-jobs)); every lit pane keeps its fill.
- **Hit targets:** 40px minimum on all interactive elements. The recorded exceptions are named where
  they occur: the sidebar row trigger's vertical extent and the turn-file chip's action pair.
- **Color is never the only channel:** status states pair color with an icon or label.

---

## Theme anatomy

Every theme answers six questions. The catalog below answers them per theme; the prototypes make
the answers concrete.

1. **Mood** - one sentence of intent.
2. **Surfaces** - palette direction and how depth is expressed.
3. **Shape** - bubble and panel geometry (radius, clip, borders).
4. **Thinking tell** - the ambient signal while reasoning streams.
5. **Tool voice** - how tool calls, traces and confirms present.
6. **Signature** - the one detail someone screenshots.

### Catalog

> **Default and build order (decided 2026-07-23):** Aurora is the default theme, and the first
> build ships exactly two themes - Aurora and Minimal (the OS-following pair, for anyone who wants
> a plain light/dark look). The remaining catalog entries land with the Phase 7 theming pass, in
> this document's order. The "(default)" label previously on Porcelain is superseded.

#### 1. Porcelain (light + dark pair, follows OS)

- **Mood:** warm, literate calm - the house style. Cream and bone in light; deep warm charcoal in
  dark, never blue-black.
- **Surfaces:** paper-warm neutrals; depth via 1px hairlines and faint warm shadows, not elevation
  stacks. Accent: terracotta clay.
- **Shape:** 14px bubbles, user turns filled soft clay-tint, assistant turns borderless on the app
  surface (text-first).
- **Thinking tell:** the thread background warms by 2-3% and a soft rule beside the thinking pane
  slowly pulses.
- **Tool voice:** quiet card with a hairline border; running state shows a small orbiting dot;
  confirm card uses the warm danger tone with a plain-language summary line.
- **Signature:** serif-italic empty-state greeting and citation chips styled like tiny book plates.

#### 2. Minimal (light + dark pair, follows OS)

- **Mood:** nothing but the conversation; for people who find personality themes noisy.
- **Surfaces:** pure white / pure near-black (#111), one gray ramp, accent is the system blue.
- **Shape:** 8px radius, user bubble filled light gray, assistant plain text. The flourish layer
  is subtractive only: it removes shadows and card chrome, never adds decoration.
- **Thinking tell:** a static "thinking" label with an opacity-pulsing ellipsis of three dots
  (`...`), nothing ambient.
- **Tool voice:** single-line entries, no cards: `web_search - running` becomes
  `web_search - 3 sources` on completion.
- **Signature:** the absence of one. This pair is also the print-friendly and screenshot-neutral
  choice.

#### 3. Phosphor (dark; accent variants: green, amber, red)

- **Mood:** a CRT terminal that grew up - the machine-room theme.
- **Surfaces:** near-black with a 1-2% phosphor tint of the accent; depth via brightness, not
  shadow. Faint static scanlines on the app background (a repeating-linear-gradient at 2-3%
  opacity; removed under reduced motion along with the shimmer).
- **Shape:** `--clip-panel` cuts one corner of panels and cards diagonally (12px, console style);
  bubbles are near-square (4px). All three font tokens point at mono.
- **Thinking tell:** a slow scanline shimmer sweeps the background every 6s while reasoning; the
  thinking pane is dimmed "stderr" text prefixed with `#`.
- **Tool voice:** tool calls render as command lines inside the trace card:
  `$ web_search "qdrant migration guide"` then `> 200 OK - 3 results ingested`. Confirms read as a
  privileged prompt: `! send_email requires approval [y/N]` with real buttons.
- **Signature:** the streaming caret is a blinking block cursor; user messages carry a `>` prompt
  prefix rendered by CSS (`::before`), so copied text stays clean.

#### 4. Aurora (default; dark, glass)

Reference: [prototypes/aurora-next.html](prototypes/aurora-next.html);
`aurora.html` is the earlier skeleton.

- **Mood:** ambient instrument - a deep night with a real sky in the margins, which the work in
  front of it is read against.
- **Surfaces:** indigo night base; the sky (a shader, with a CSS fallback) held back from the
  reading column by a veil. Glass is three jobs: the sidebar and console rail are the one blurred
  pane, the composer is a translucent pane onto the sky, and cards are lit panes - a light fill and
  an edge light, no filter. Only floaters that pass over text blur.
- **Shape:** 12 to 16px radius; the user's bubble is a lit teal-tinted pane, the answer has no
  card at all and is set in a serif.
- **Thinking tell:** the sky turns violet and rises (energy .92), the constellations trace more
  often, and the topbar's horizon takes the colour. Reduced motion: one still frame of the sky and
  the horizon held still.
- **Tool voice:** an inline timeline row with a teal glyph and the matrix loader while running;
  on the console a comet runs round the activity panel's border in the running colour.
- **Signature:** the sky takes the colour of the step that is running - violet to think, teal to
  work, magenta to wait on a yes, amber while a model loads - and rests when nothing is happening.

#### 5. Manuscript (light)

- **Mood:** a well-set book page; the theme for long reading.
- **Surfaces:** warm paper, faint ruled baseline grid in the margins, ink-black text, oxblood
  accent. Depth via rules, never shadows.
- **Shape:** no assistant bubble at all - typeset prose on the page; user turns are indented
  quotations with a left rule. Panels square (2px radius) with double-rule top borders.
- **Thinking tell:** italic "marginalia" - the thinking pane sits slightly inset with a vertical
  rule, styled as editor's notes; while active, the rule breathes in width (1px to 2px).
- **Tool voice:** proofreader's marks - a small caret glyph and a terse line ("consulted the web,
  3 sources"), expanding to the full trace on click.
- **Signature:** citations are real footnotes: superscript markers in the text and a rule-separated
  footnote block at the end of the message, numbered in first-reference order (which is exactly
  what the backend's renumbering guarantees).

#### 6. Nocturne (dark, OLED)

- **Mood:** maximum restraint for night work; true black, one live color.
- **Surfaces:** #000 app, #0a0a0a panels, hairline #1f1f1f borders; a single electric violet accent
  carries every interactive state.
- **Shape:** 10px radius, everything outlined, nothing filled except the send button.
- **Thinking tell:** a 1px accent filament runs along the thinking pane's left edge with a slow
  luminance crawl.
- **Tool voice:** outline cards; the running spinner is a rotating 90-degree arc of the accent.
- **Signature:** on OLED screens the app visually merges with the bezel; the composer's focus ring
  is the brightest thing on screen.

#### 7. Abyss (dark, oceanic)

- **Mood:** deep-sea instrument panel; pressure-calm, bioluminescent.
- **Surfaces:** layered navies (#050c14 to #0d1b2a) with a barely-visible caustic light pattern
  drifting on the app background; cyan-teal biolum accent, coral warn.
- **Shape:** 12px radius; panels carry a faint top edge glow like light from above.
- **Thinking tell:** a sonar pulse - a soft radial ring expands from the thinking pane every 4s.
  Reduced motion: a static ring badge.
- **Tool voice:** depth-gauge cards - a thin vertical depth bar fills while a tool runs; done
  state leaves a small tick at the reached depth.
- **Signature:** citation chips glow faintly on hover like disturbed plankton.

#### 8. Grove (light, botanical)

- **Mood:** a reading nook by a window - moss, sage, cream, bark.
- **Surfaces:** warm cream base, sage panels, moss accent, bark-brown text; depth via soft
  green-gray shadows.
- **Shape:** leaf-asymmetric bubbles (18px radius with one 4px corner, mirrored per role); organic,
  never cartoonish.
- **Thinking tell:** dappled light - two large, very soft radial gradients drift slowly across the
  header area, like foliage shadows.
- **Tool voice:** seed-to-sprout: the running indicator is a two-frame sprout glyph; completed
  traces show a small leaf tick.
- **Signature:** hairline vine rules (CSS gradients, no images) separating sidebar sections.

#### 9. Ember (dark, warm)

- **Mood:** dusk and firelight; the cozy dark theme for people who find OLED themes cold.
- **Surfaces:** deep plum-brown base with a low horizon gradient (amber at the bottom 15% of the
  viewport, 4% opacity); amber-coral accent ramp.
- **Shape:** 14px radius, user bubbles filled with a subtle amber-to-coral gradient.
- **Thinking tell:** breathing coals - the horizon gradient slowly brightens and dims on an 8s
  cycle while reasoning.
- **Tool voice:** warm cards whose left border smolders (accent at 60%) while running and cools to
  the hairline color when done.
- **Signature:** the streaming caret is an ember - a small glyph with a soft warm glow that
  brightens as tokens arrive.

#### 10. Sorbet (light, pastel)

- **Mood:** friendly and light-hearted without sacrificing legibility - the approachable theme.
- **Surfaces:** off-white cream base; pastel lilac, mint and peach for fills; text stays a real
  dark ink (contrast is non-negotiable in pastel).
- **Shape:** squishy 20px bubbles; action chips look like stickers (2px offset shadow, no blur).
- **Thinking tell:** a soft blob (one blurred oval) behind the thinking pane slowly morphs
  border-radius; reduced motion swaps it for a static blob.
- **Tool voice:** sticker chips that tilt 1 degree on hover; running state pulses the chip's
  shadow; springs (`--ease-spring`) are allowed here and nowhere else.
- **Signature:** the send button is a pill that squashes 4% on press - the only overshoot
  animation in the catalog.

---

## Theme authoring recipe

Adding a theme is a data change plus one prototype:

1. Copy an existing tokens file in [tokens/](tokens/); redefine tier 1 and tier 2 fully, override
   tier 3 sparsely. Record contrast checks in the header comment.
2. Answer the six [theme anatomy](#theme-anatomy) questions in a catalog entry above.
3. Add a flourish layer only if the theme needs geometry or decoration beyond tokens - and only on
   `data-part` hooks, within the allowed property set.
4. Copy the prototype skeleton (structure of [prototypes/porcelain.html](prototypes/porcelain.html)),
   swap the token block and flourish layer, and walk the state harness through every state.
5. Register the theme (name, modes, accents, description) in the theme registry so the picker,
   persistence and no-flash script see it.

A theme is done when: every state in [State choreography](#state-choreography) is expressed, every
contrast rule in [Accessibility](#accessibility) passes, reduced motion produces a fully static but
still legible experience, and the prototype reads as *designed* - not as a palette swap.
