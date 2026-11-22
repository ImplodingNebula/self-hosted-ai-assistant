# Testing

This document explains how the platform is tested: what each seam's fake stands in for, what the backend and
frontend suites actually cover, what runs in CI and when, and how to test a signed-in session without a browser.
The goal is not "100% coverage" - it is **confidence that a change did not break anything**, without a suite that
is painful to maintain.

---

## How the seams tame a nondeterministic model

The single most important idea in this project's tests: **the three seams (`ModelProvider`, RAG, MCP) are exactly
where we substitute fakes.** Because the services layer only talks to those interfaces (see
[ARCHITECTURE.md](ARCHITECTURE.md)), a test can inject:

- `FakeProvider` - streams a scripted sequence of `ChatChunk`s (thinking, status, error and tool-call frames are
  all scriptable) so the chat loop is fully deterministic, tool calls included.
- `FakeEmbedder` - turns text into a vector by hashing, so the same text always embeds the same way and RAG ranking
  is reproducible.
- A **mock upstream API** - the `mcp-chassis`'s upstream is replaced with recorded JSON, so tests never touch
  a real upstream system.

Now hard assertions *are* possible: "given this retrieved context and this scripted model output, the endpoint
streams these SSE events and persists a message with these citations." Deterministic and fast.

The intent is to keep a **few** real-model tests (marked `@pytest.mark.ollama`) that actually hit Ollama, to
catch integration drift, and to run them apart from the fast suite so it stays fast. (None are marked yet - the
first ones land with the `ModelProvider` contract tests, one suite run against both the fake and the real
provider so the two cannot silently drift apart, so today the whole suite runs on fakes, and the marker is
deselected in CI against a set that is currently empty.)

## What we use

**Backend (Python):**
- `pytest` + `pytest-asyncio` - the test runner (async-aware).
- `httpx.AsyncClient(ASGITransport(app))` - calls the FastAPI app **in-process** (no server to start), so "API
  tests" are still fast.
- `app.dependency_overrides` - FastAPI's built-in switch to inject `FakeProvider` etc. for a test.
- `httpx.MockTransport` - intercepts outgoing HTTP (e.g. the provider's calls to Ollama) and returns recorded
  responses; `respx` / `pytest-httpx` are the upgrade path if interception needs outgrow it.
- An **in-memory SQLite** database (`aiosqlite`) with the schema created from the SQLModel metadata - fast and
  hermetic, no Postgres to start.
- An **in-memory Qdrant** for the RAG tests: `qdrant-client` ships a local mode (`AsyncQdrantClient(":memory:")`)
  that runs the real client code paths - filters, counts, delete-by-filter - entirely in-process, so retrieval
  tests exercise genuine store behavior with no server and no containers. Its one blind spot: payload-INDEX
  creation is a no-op locally, so index definitions are verified against the real server as a deployment check
  instead.

**Frontend (TypeScript) - installed today:**
- Five **headless check batteries**, run together by `npm test` - plain Node scripts, no runner and no browser.
  They cover the message composer, the chat pane's message transitions, the `/api/*` proxy contract, the
  sign-in helpers and the console defaults page's ceiling presets. See
  [the frontend check batteries](#the-frontend-check-batteries) below.

**Frontend - planned, not installed:**
- `vitest` + React Testing Library - component and streaming-render tests.
- **MSW (Mock Service Worker)** - intercepts the frontend's network calls and returns fake responses; shared
  between vitest and Playwright so the mocks are written once.
- `Playwright` - real-browser E2E for 3-5 critical flows only.

## What to test first (so the task isn't overwhelming)

Prioritize the code where a bug is costly or subtle. In rough order:

1. **Tenant scoping** - assert a request can't read another tenant's data, and that a model-supplied `tenant_id`
   can't override the server's. (Security-critical.)
2. **The chat SSE loop** - with `FakeProvider`: correct event sequence, citations attached, message persisted even
   on disconnect.
3. **RAG ingest/retrieve** - with `FakeEmbedder`: chunks land in the right (ephemeral vs persistent) collection;
   filters work; TTL/cleanup removes session data.
4. **The MCP tool loop** - with the mock upstream API: a tool call is executed, results are ingested and fed back,
   and tenant scoping is enforced in the sidecar.
5. **Parsers** - a sample text PDF, a table PDF, and a scanned (OCR) PDF each produce sensible chunks.

**Defer:** exhaustive UI-pixel tests, and LLM *quality* evals (ragas/promptfoo). Those measure answer quality, not
correctness - nice later, not a gate now.

## Coverage targets (enforceable, not punishing)

- Backend business logic: **~75%** (exclude thin router/stream glue and generated code).
- Frontend: **~60%** (exclude pure animation/presentational bits).
- Chase *meaningful* coverage of the 5 priorities above before raising the number. A green suite worth trusting beats a
  high percentage that isn't.

## What CI runs

CI is split into path-scoped workflows, so a change pays only for the trees it touches rather than running the
whole suite over a docs edit or a single-app fix:

| Workflow | Runs when | What it does |
|---|---|---|
| `ci.yml` | Every pull request | The Conventional-Commit check over the PR's own commits |
| `ci-backend.yml` | `apps/backend`, `infra`, `scripts`, `services`, or the `Makefile` changes | `ruff check`, `ruff format --check`, then `pytest -m "not ollama and not ocr"` |
| `ci-frontend.yml` | `apps/frontend` changes | `npm run lint`, `typecheck`, `build`, then `npm test` (the five check batteries) |
| `ci-api-types.yml` | `apps/backend` or `packages/api-types` changes | Regenerates the OpenAPI schema and its types, and fails if the committed copies drifted |
| `frontend-audit.yml` | Any `apps/frontend` change, `packages/api-types` package-file changes, plus nightly | `npm audit` across every JS workspace with a lockfile |

The four path-scoped workflows (everything but `ci.yml`, which only makes sense on a pull request's own commits)
also run on every push to `main`, so the record on the trunk stays complete even though a pull request only pays
for what it touches.

The slow tier a fuller test pyramid would add - Playwright end-to-end runs and `@pytest.mark.ollama` real-model
smoke tests - is not built yet. No workflow has a schedule or a label trigger that would run one; the only
scheduled job in the repository is the nightly audit in `frontend-audit.yml`, which is a security check rather
than a test tier. Both layers land with the frontend polish phase (see
[ROADMAP.md](ROADMAP.md#phase-7---frontend-polish--theming)).

## Running tests locally

```bash
# Backend
cd apps/backend && uv run pytest                 # fast suite (fakes; no Ollama needed)
cd apps/backend && uv run pytest -m ollama        # real-model tests (needs Ollama; none marked yet)

# Frontend - the checks that exist today
cd apps/frontend && npm run lint                  # eslint
cd apps/frontend && npm run typecheck             # tsc --noEmit
cd apps/frontend && npm test                      # all five check batteries
```

**There is no general frontend test RUNNER yet** - no vitest, no Playwright, no `*.test.tsx` files. What
`npm test` runs is five plain Node scripts under `apps/frontend/scripts/`, and they are the only automated
frontend checks besides lint and type-check. `make frontend-test` and the CI step both call
`npm run test --if-present`, so they pick them up automatically. A vitest and Playwright layer is still the
intended shape for component and end-to-end coverage, and lands with the frontend polish phase (see
[ROADMAP.md](ROADMAP.md#phase-7---frontend-polish--theming)).

### The frontend check batteries

Each of the five pins one contract that lint and type-check cannot see. They share a shape: every one imports
the app's TypeScript directly with no build step in between, so all five need **Node 22.18 or newer** - the
first release that strips type annotations on its own - and stop with a one-line message on anything older.
Each prints every failing case with what it expected and what it got, and exits non-zero if there was one.

Any one of them can be run alone while working on the code it covers, for example
`cd apps/frontend && node scripts/composer-fidelity.mjs`.

#### Composer wire fidelity - `composer-fidelity.mjs`

The message composer is a rendered editor: list bullets, ordered numbers, task checkboxes and code blocks
appear as real glyphs while they are typed, but what gets SENT is still the plain markdown a person would
have typed into an ordinary text box. Those are two different representations of one draft, and everything
downstream - the send button's enablement, the restore after a failed send, the wire itself, the renderer
that draws the message at the other end - depends on them meaning the same thing.

That equivalence is what the battery checks, by driving a real editor with no browser attached (a *headless*
editor: the same node objects the browser builds, without the DOM). It covers:

- **Prose survives byte-identical.** A generic markdown writer escapes characters like `*` and `_`; nobody
  typed a backslash, so anything outside the composer's small grammar has to come back exactly as written.
- **The constructs take one canonical form** - `*` and `+` bullets normalize to `-`, ordinals renumber,
  unclosed fences close - and reading the composer's own output back gives that same output again. That last
  property is checked over three thousand generated drafts from a fixed seed, so a failure is reproducible.
- **The reader agrees.** Everything the composer emits is parsed by the same markdown parser the message
  renderer uses, and the resulting shape is compared against what was composed.
- **The keystrokes, the paste and the drop**, dispatched through the real command handlers with the editor's
  own base behaviour registered underneath them: Enter sends, Shift+Enter breaks a line, Tab moves focus out
  of the box, pasted or dropped text is read as constructs at any caret position, and formatting the composer
  cannot send is refused rather than shown.

#### Message transitions - `message-transitions.mjs`

`lib/messages.ts` is where a stream frame becomes a rendered turn: every function in it takes the current
message array and returns a new one, keyed off the trailing message. That makes the whole file checkable
without React, a DOM or a component runner - none of which this workspace has - so this battery drives it
directly, along with the composer's decision about which attachments a question carries.

It also covers the attachment **download** path, including its one guarantee that is invisible from every
other angle: a refusal must never reach the browser's file saver. Without that check a reader gets a JSON
error document on disk, under the file's own name, with nothing anywhere saying so.

#### The proxy contract - `proxy-contract.mjs`

A route under `app/api/` runs on the server, so the backend sees the request coming from that process rather
than from the person who made it. Two invariants follow, and both fail silently: every outgoing `fetch` has to
carry the caller's address and cookies forward, or the per-client rate limit quietly becomes one shared budget
and the audit trail names the wrong client; and a route must not build a response of its own, or a
`Set-Cookie` that mints or clears a session is dropped and sign-in appears to succeed with nobody signed in.

Both are checked **statically, across every file that names the backend URL at once**, rather than by a test
per route - so the next route added is covered without anyone remembering to cover it.

#### Sign-in helpers and the session gate - `auth-contract.mjs`

Four guards in `lib/auth.ts` and `proxy.ts`, each of which fails quietly if it is ever weakened: the OIDC
return path is attacker-reachable by the time it is navigated to, so a path check that admits `//elsewhere/`
is an open redirect on the sign-in page; the signed-out reaction must fire on exactly one status code, or a
deployment with sign-in switched off starts bouncing people to a page that should not exist for it; the
session probe must answer "do not know" for every way of not knowing, rather than showing an identity chip
naming nobody; and the pre-render gate must run on page routes and nowhere else, since on `/login` it is an
infinite bounce and on `/api/*` it answers a fetch with HTML.

#### The console defaults - `console-defaults.mjs`

The Defaults page offers card-sized presets for the context-window ceiling, and every one of those numbers is
a promise checked against the backend's own bounds and step - a preset outside the accepted range, or off the
1,024-token step, fails a named check rather than failing an administrator later. The battery also pins the
select's state derivation (which preset a given field value displays, when "Custom" appears, what "Not set"
clears) and the wiring that routes a pick through the page's ordinary draft-and-save machinery.

## Testing a signed-in session

Once an identity provider is configured, every `/api/v1` route wants a session cookie, and the only way to
get one through the front door is a person completing a consent screen at that provider. That is deliberate -
it is what makes sign-in mean anything - but it also puts the whole signed-in surface out of reach of any
check that is not a person clicking, which is most of them.

The way through is a decision the session store already made. Sessions here are **server-side**: the cookie
carries an opaque id, and the record it names lives in Redis (see
[`apps/backend/app/services/auth/store.py`](../apps/backend/app/services/auth/store.py)). The record IS the
credential, so writing one and handing over its cookie leaves exactly the state a completed sign-in leaves -
same key, same value, read back by the same code on the next request. That is what `make dev-session` does,
through the app's own session-creation function rather than a copy of it:

```bash
make dev-session                            # the seeded default user
make dev-session EMAIL=person@example.com   # a member who has signed in at least once
make dev-session TTL=30                     # a shorter window (default: 120 minutes)
```

It prints the cookie's name and value plus two ready-to-paste forms - one for a browser console or a
browser-automation script, one for `curl`. The cookie goes on the origin the app is served from, not the
API's: the app relays it through to the backend, and a cookie set on the wrong origin is simply never sent.

Naming a member works only once a users row exists for them, and a row is written by a first sign-in - so an
address that has never signed in has nothing to mint for. The roster is the console's Members page
(`GET /api/v1/members`), and the addresses listed there are the ones that can be named.

Two things worth knowing before trusting a result:

- **A short window is short only until the session is used.** Reading a session re-arms it with the
  deployment's own sliding window (`SESSION_TTL_SECONDS`), so the first request carrying the cookie stretches
  it to the same lifetime any other session gets.
- **With no issuer configured, sessions are not read at all.** The API answers every request as the seeded
  default principal, so a signed-in surface exercised against that stack has been tested with no sign-in in
  front of it. The command says so in its own output rather than letting the hour go quietly.

### Why this is safe to keep in the repository

The tool runs inside the backend container, so reaching it requires a shell on the host running the stack -
and whoever holds that can already read the database and write a session record by hand. It hands out nothing
that access did not already carry: a shortcut past typing, not past a boundary. Testing against
authentication always takes some version of this shape, either a stand-in identity provider or a session
written directly. The backend's own suite takes the first (it verifies tokens from a fake issuer); this is
the second, for the stack that is actually running.

What it must never become is something that runs where nobody meant it to, so it **refuses under the
production compose layout**, and refuses again when the layout is not stated at all - an unknown layout is
declined rather than assumed to be a development one. Nothing on the login path is relaxed to make any of it
work: what it writes is an ordinary session, and it expires, slides, and signs out like every other.
