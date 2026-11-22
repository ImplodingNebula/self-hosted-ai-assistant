# frontend

Next.js (App Router, TypeScript) UI for the AI Chatbot Platform: Tailwind v4,
a clean, modern chat design language, and a themed surface - aurora, a night
sky default, beside a minimal pair that follows the operating system.

## Status

Current through **ROADMAP Phase 8b, partially** - OpenID Connect sign-in, the
deployment core and the model-manager stack are merged, while Phase 7's
remaining polish and Phase 8b's named-permission slice are outstanding. The
surfaces are the chat page, the admin console (model manager, integrations,
workspace defaults, members) and the sign-in page - all hand-rolled React over
the same-origin proxy routes, drawn from the semantic design tokens in
`app/themes/` rather than hard-coded colours.

**Chat** (`app/page.tsx`, `components/`, `hooks/`)

- Streams assistant tokens as they arrive. For thinking-capable models the
  reasoning streams too and renders as a collapsed block above the answer.
  A response can be stopped mid-stream (the partial text is kept), each request
  sends the new message with a `conversation_id` and thinking mode but no
  client-held history (the backend rebuilds context from persisted turns), and
  the view auto-follows the stream only while scrolled to the bottom.
- `citation` events collect on the in-flight assistant message and render as
  numbered source chips under it (hover for the excerpt), live and after a
  reload; web sources open in a new tab.
- The tool activity trace: each call the model made, its result, and the model's
  narration around it. A side-effecting call renders an approve/deny prompt that
  suspends the turn, and folds into the trace once answered - badged approved or
  denied. The whole trace persists on the message, so reopening a conversation
  replays it.
- Status notices render inline and honestly: a tool that failed, a spent tool
  budget, an answer written without tools, a picture the model could not see.
- `components/attachment-tray.tsx` + `hooks/use-attachments.ts` - the composer's
  file tray: attach, watch parsing progress, remove, and see which files a turn
  actually carried.
- `components/chat/turn-actions.tsx`, `message-editor.tsx`, `selection-quote.tsx`
  - revisiting a turn rather than only adding to it: copy, ask again, reword the
  question in place, or quote a passage into the next message. Anything a rewind
  discards is named in a confirmation before it runs.
- `components/chat/model-selector.tsx`, `thinking-toggle.tsx`,
  `web-search-toggle.tsx` - the composer controls, saved per conversation and
  read by the turn; the selector offers chat-role models the backend actually
  holds, and each control patches its own field so two of them saving at once
  cannot clobber each other.
- `components/sidebar.tsx` + `convo-row-menu.tsx` - the conversation list panel
  (new chat, switch, rename in place, pin, delete), with a Pinned group above
  Recent; a slide-over on small screens, always visible on wider ones. Opening a
  conversation marks the clicked row with a spinner, and the detail route is
  fetched once in the background at page load so the first real click does not
  pay the dev server's first-request route compile.

**Console** (`app/console/`, `components/console/`)

- The administrative shell: a rail at `/console` holding four sections - Model
  manager, Integrations, Workspace defaults and Members - which share its layout
  and panel anatomy. What each section's verbs require is a per-route decision on
  the backend, not a property of the shell.
- `app/console/models/page.tsx` - the Model Manager. Rows grouped by role over a
  request-time join of catalog, installed inventory and measured probe facts:
  each row a summary over an expandable body of facts (digest, context window,
  declared and on-disk sizes, capabilities, why a model is clamped), with pull,
  update, delete and tool-check verbs drawn from what the backend says the
  provider can actually do.
- `components/console/activity-panel.tsx` - the one running operation, streamed:
  download progress that pauses and resumes from where it stopped, continuing
  into the automatic tool check on the same stream, then the verdict with its
  per-case tallies. Every other verb disables while it runs, and a stream that
  dies without its terminal frame is reported as a failure rather than left
  frozen at its last percent.

**Integrations** (`app/console/integrations/page.tsx`, `lib/integrations.ts`)

- Register a tool server from one generic form - MCP servers self-describe, so
  there is no per-integration UI code - or start from a preset (GitHub, Gmail,
  Google Drive, Slack, Asana), which prefills the form with that provider's
  verified endpoints, scopes and setup notes.
- Per-server cards: credentials entered write-only (a `last4` hint is all that
  reads back), OAuth connect and disconnect with the grant's status and the
  connected account shown on the card, per-server enable toggles, and a
  test-connection that handshakes and lists a server's tools without ever
  running one.

**Workspace defaults** (`app/console/defaults/page.tsx`,
`lib/workspace-defaults.ts`)

- The axes a turn falls back to when neither the request nor the conversation
  names one: default model, thinking mode, web-search mode, a workspace system
  prompt, a context-window ceiling and the reranker toggle. Every axis is
  optional and individually clearable, so a workspace can set one and leave the
  rest to the deployment's own configuration.
- The embedding model sits beside them and is a different kind of choice: it
  defines the vector space stored passages live in, so changing it points
  retrieval at a fresh, empty one that re-ingestion has to fill. Left unset, the
  deployment resolves it from whichever model wrote the stored vectors, or from
  the single installed embedder when the server holds exactly one. With several
  installed and nothing chosen, retrieval refuses rather than guess - this page
  is where that is settled.

**Members** (`app/console/members/page.tsx`, `lib/members.ts`)

- Who has an account in this workspace and what each of them may do: the roster
  with display name, address and joining date, and an admin/member role control
  per row. The last administrator cannot be demoted - the refusal comes from the
  backend and names itself, so the page never has to guess which change is the
  one that would lock everybody out.

**Sign in** (`proxy.ts`, `app/login/page.tsx`, `lib/auth.ts`,
`components/user-chip.tsx`)

- A deployment can require sign-in through an identity provider, or not require
  it at all. When it does not, this surface is inert and invisible: no account
  chip, and the sign-in page redirects into the app rather than offering a
  button that leads nowhere.
- `app/login/page.tsx` - the sign-in card, rendered on the server with no client
  JavaScript. Its one control is a plain link, because beginning the round trip
  is a navigation to the provider's own domain.
- `proxy.ts` - the gate in front of every page route. Reacting inside the app
  can only ever happen after a page has been sent, which meant a signed-out
  visitor watched the whole interface flash past before being bounced; this asks
  who they are before any markup leaves the server. Assets and `/api` requests
  are outside it, so the cost is one question per page request the gate matches
  - a navigation plus whatever links are prefetched around it, not one per
  resource. A backend that cannot be reached lets the request through rather
  than trapping anyone in front of a sign-in page that could not work either. On
  a deployment with sign-in switched off the answer is always "nobody needs to
  sign in here", and every request passes through exactly as it did before.
- `lib/auth.ts` - the single place the app reacts to being signed out, and the
  single place that decides what "signed out" even is: the gate above reads the
  same rule from here, because a gate and an app that disagreed about it would
  bounce a visitor between them with nothing on screen to explain why. Every
  REST failure already funnels through one error helper, so a reply that says a
  session is required sends the visitor to sign in from there rather than from
  each of the thirty call sites; the chat stream reaches the same handler from
  the frame it reads. A return path is re-checked before it is used - it has
  been through a browser and is not the app's to trust.
- `components/user-chip.tsx` - who is signed in and the way out, in the sidebar
  footer: the display name over the address, since the address is what tells two
  accounts apart, and a sign-out that ends on the sign-in page whether or not
  the server confirmed it.

**Plumbing**

- `app/api/**/route.ts` - same-origin proxies to the backend, relaying status and
  `problem+json` error bodies untouched via the shared helpers in
  `lib/backend-proxy.ts`. Same-origin means no CORS, and it works in Docker where
  the browser cannot resolve the internal host. The chat proxy pipes the SSE
  response straight back.
- Because every call reaches the backend from this server rather than from the
  person who made it, those helpers carry two things across the hop in each
  direction: the caller's address, so per-client rate limits and the audit trail
  name the right client, and the session cookie, so an authenticated call is not
  anonymous by the time it lands. Responses carry any cookie the backend set
  back out again - a sign-in and a sign-out each say everything they have to say
  in that one header, so a leg that drops it half-works in silence.
- `scripts/proxy-contract.mjs` checks both statically across every route file at
  once, rather than by a test per route that the next new route would not have.
- `lib/sse.ts` - a small `text/event-stream` parser for `fetch` streams; accepts
  CRLF, LF, or CR event-block separators per the SSE spec.
- `lib/types.ts` - local DTO and SSE event types mirroring the backend contract
  (`packages/api-types` holds the schema generated from the backend OpenAPI spec).

The design language for the polished UI is authored ahead of the work. Its token
foundation is live (`app/themes/`, `components/theme-provider.tsx` and
`theme-picker.tsx`, with a boot script that applies the stored choice before
first paint), shipping two of its themes; the rest of the catalog and the richer
stack (headless chat primitives, a query layer, a client store, motion) land in
Phase 7 - see
[../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) and
[../../docs/ROADMAP.md](../../docs/ROADMAP.md).

## Develop

Run the whole stack (recommended) from the repo root:

```bash
make up                           # GPU (default); frontend on http://localhost:3000
# or, without a GPU:  make up PROFILE=cpu
```

Or run the frontend alone (needs the backend reachable at `BACKEND_INTERNAL_URL`,
default `http://localhost:8000`):

```bash
npm install
npm run dev        # http://localhost:3000
```

## Scripts

| Command | Purpose |
|---|---|
| `npm run dev` | Dev server (hot reload) |
| `npm run build` | Production build (standalone output) |
| `npm run start` | Serve the production build |
| `npm run lint` | ESLint |
| `npm run typecheck` | `tsc --noEmit` |
| `npm run test` | The check batteries (composer fidelity, message transitions, proxy contract, auth contract, console defaults). Needs Node 22.18+ |

> This project pins Next.js at a patched release and holds three transitive
> dependencies forward via `overrides`: `minimatch` at `^10.2.5` (a
> brace-expansion denial of service reaching the tree through the eslint and
> redocly chains, whose 1.x line has no patched release), and `postcss` at
> `^8.5.23` (GHSA-r28c-9q8g-f849) and `sharp` at `^0.35.0` (inherited libvips
> advisories), both of which sit behind Next's exact pins. An override fixes
> those two without taking the downgrade the advisory suggests - moving Next
> backwards would trade one vulnerability for the thirteen its own patched
> release fixed. Each override is droppable once the upstream ships a patched
> range. Re-run `npm audit` after any dependency change - see the repo's
> security rule.
