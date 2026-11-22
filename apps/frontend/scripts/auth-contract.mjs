/**
 * Battery for the sign-in helpers in `lib/auth.ts`, and for the server-side
 * gate in `proxy.ts` that judges by the same rule.
 *
 * Four of them are guards rather than conveniences, and all four fail
 * quietly if they are ever weakened:
 *
 *   - the return path is attacker-reachable. It travels to an identity
 *     provider and back, so by the time it is navigated to it is just a string
 *     someone else may have chosen. A path check that lets `//elsewhere/`
 *     through is an open redirect on the sign-in page - the exact page a
 *     person is most willing to trust.
 *   - the signed-out reaction must fire on ONE code and no other. If a bare
 *     401 triggered it, a deployment with sign-in switched off would start
 *     bouncing people to a page that should not exist for it.
 *   - the probe must answer "do not know" for every way of not knowing. A
 *     backend too old to have the endpoint, no session, a mangled body: each
 *     has to render as today's UI, not as an identity chip naming nobody.
 *   - the gate in front of the app must run on pages and NOWHERE else. Left on
 *     `/login` it is an infinite bounce; left on `/api/*` it answers a fetch
 *     with HTML; left off the page routes it is a no-op and the flash it
 *     exists to remove comes back. Every one of those is silent.
 *
 * Every function here is pure or reaches exactly one stubbed global, so the
 * whole file runs without React, a DOM, or a component runner - none of which
 * this workspace has. Like the batteries beside it, it imports the app's
 * TypeScript directly and so needs a Node that strips types on its own -
 * 22.18 or newer.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/auth-contract.mjs)
 */

import { execFileSync } from "node:child_process";
import { register } from "node:module";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `auth contract: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

/**
 * Compile `config.matcher` the way the build does, in a process of its own.
 *
 * The source string is NOT the regex Next runs: Next wraps it with the App
 * Router's transport spellings of the same page (`.rsc`, `.segments/...`) and
 * an optional data prefix, and that wrapping is part of what has to be checked.
 * So the real compiler is used rather than a hand-rolled regex, which would be
 * a second copy of the thing under test and would agree with itself no matter
 * what it said.
 *
 * In a child because Next's build half and its runtime half each set up
 * module-level async storage on import, and loading the build half AFTER
 * `next/server` leaves it reaching for a runtime this plain Node process does
 * not have. That failure is an invariant error thrown from a place that names
 * nothing, and - the reason a comment would not have been enough - it escapes
 * `try`/`catch` and kills the run outright. Two processes cannot violate an
 * ordering they do not share, so the hazard stops existing rather than being
 * documented.
 *
 * @param matcher - The `matcher` array exported by the gate.
 * @returns One compiled regex source per entry.
 */
function compileMatchers(matcher) {
  const child = `
    const { getMiddlewareMatchers } = require("next/dist/build/analysis/get-page-static-info.js");
    const matchers = getMiddlewareMatchers(JSON.parse(process.argv[1]), {});
    process.stdout.write(JSON.stringify(matchers.map((m) => m.regexp)));
  `;
  return JSON.parse(
    execFileSync(process.execPath, ["-e", child, JSON.stringify(matcher)], {
      // Resolved from the workspace root so the child finds `next` wherever the
      // battery was invoked from.
      cwd: new URL("../", import.meta.url),
      encoding: "utf8",
    }),
  );
}

const {
  AUTH_REQUIRED_CODE,
  authGateDecision,
  fetchAuthSession,
  handleAuthRequired,
  handleAuthRequiredFrame,
  LOGIN_PATH,
  LOGOUT_TIMEOUT_MS,
  loginHref,
  redirectToLogin,
  safeNextPath,
  startLoginHref,
} = await import("../lib/auth.ts");
// The REST funnel, imported so the WIRING is checked and not just the handler
// it is supposed to call: every non-2xx in the app passes through `raise`, so
// if that one line is ever lost, nothing else notices.
const { ApiError, raise } = await import("../lib/api-error.ts");

let freshCount = 0;
/**
 * A second copy of the module, whose one-shot redirect latch is unspent.
 *
 * The latch is module state by design - it exists so a burst of failures
 * navigates once - which means only the FIRST scenario in a run can observe a
 * redirect. A distinct specifier gets a distinct module instance, so each
 * scenario that needs a fresh latch can have one instead of the checks after
 * the first silently passing for the wrong reason.
 */
async function freshAuth() {
  freshCount += 1;
  return import(`../lib/auth.ts?fresh=${freshCount}`);
}

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

/* ---------- the return path is not trustworthy ---------- */

check("next path: an ordinary in-app path survives", safeNextPath("/console/models"), "/console/models");
check("next path: a query string survives", safeNextPath("/?q=1&r=2"), "/?q=1&r=2");
check("next path: an absolute URL is refused", safeNextPath("https://elsewhere.example/x"), "/");
check("next path: a protocol-relative URL is refused", safeNextPath("//elsewhere.example/x"), "/");
check("next path: the backslash spelling is refused", safeNextPath("/\\elsewhere.example"), "/");
// Browsers STRIP these out of a URL before parsing, so each of the three
// normalizes to "//elsewhere.example" - the protocol-relative case above,
// wearing a disguise that a plain startsWith check cannot see through.
check("next path: a tab-disguised protocol-relative URL is refused", safeNextPath("/\t/elsewhere.example"), "/");
check("next path: a newline-disguised one is refused", safeNextPath("/\n/elsewhere.example"), "/");
check("next path: a carriage-return-disguised one is refused", safeNextPath("/\r/elsewhere.example"), "/");
check("next path: a backslash anywhere is refused", safeNextPath("/console\\..\\elsewhere"), "/");
check("next path: a NUL is refused", safeNextPath("/console\u0000.example"), "/");
check("next path: a scheme with no slashes is refused", safeNextPath("javascript:alert(1)"), "/");
check("next path: a bare relative path is refused", safeNextPath("console"), "/");
check("next path: nothing at all becomes the root", safeNextPath(""), "/");
check("next path: a missing value becomes the root", safeNextPath(undefined), "/");
check("next path: a null value becomes the root", safeNextPath(null), "/");

check("login href: the root needs no query at all", loginHref("/"), LOGIN_PATH);
check(
  "login href: a real destination is remembered, encoded",
  loginHref("/console/models"),
  "/login?next=%2Fconsole%2Fmodels",
);
check(
  "login href: a hostile destination is dropped, not carried",
  loginHref("//elsewhere.example/x"),
  LOGIN_PATH,
);
// The flag is what lets the sign-in page say the session ended rather than
// greet a first visit; it rides beside the destination, and alone at the root.
check(
  "login href: a lapsed session is flagged beside its destination",
  loginHref("/console/models", true),
  "/login?next=%2Fconsole%2Fmodels&lapsed=1",
);
check("login href: a lapsed session at the root is flagged alone", loginHref("/", true), "/login?lapsed=1");
check(
  "start login href: the provider hand-off carries a checked path",
  startLoginHref("/console"),
  "/api/v1/auth/login?next=%2Fconsole",
);
check(
  "start login href: a hostile destination collapses to the root",
  startLoginHref("https://elsewhere.example/x"),
  "/api/v1/auth/login?next=%2F",
);

/* ---------- the signed-out reaction fires on one code and no other ---------- */

check("auth required: the code is what the backend promises", AUTH_REQUIRED_CODE, "auth_required");

// The judgement itself, with no navigation and no server attached. Both the
// client reaction below and the server-side gate at the bottom of this file
// delegate to it, so this is the one rule stated once.
check("gate decision: a 401 with the code means sign in first", authGateDecision(401, "auth_required"), "redirect");
check("gate decision: a 401 with another code is somebody else's problem", authGateDecision(401, "not_found"), "pass");
check("gate decision: a 401 with no code is not the signal", authGateDecision(401, undefined), "pass");
check("gate decision: the code on another status is not the signal", authGateDecision(403, "auth_required"), "pass");
check("gate decision: a signed-in answer passes", authGateDecision(200, undefined), "pass");
check("gate decision: a backend without the endpoint passes", authGateDecision(404, undefined), "pass");
// The fail-open case, and the reason the status is nullable at all: a backend
// that could not be reached must not lock every page behind a sign-in page
// that could not work either.
check("gate decision: no answer at all passes", authGateDecision(null, undefined), "pass");

// No window yet, so nothing navigates and the one-shot latch stays unarmed -
// these check the DECISION, and the navigation is checked on its own below.
check("auth required: a 401 with the code is a signed-out failure", handleAuthRequired(401, "auth_required"), true);
check("auth required: a 401 with another code is not", handleAuthRequired(401, "not_found"), false);
check("auth required: a 401 with no code is not", handleAuthRequired(401, undefined), false);
check("auth required: the code on another status is not", handleAuthRequired(403, "auth_required"), false);
check("auth required: an ordinary server error is not", handleAuthRequired(500, undefined), false);
check("auth required frame: the stream path agrees", handleAuthRequiredFrame("auth_required"), true);
check("auth required frame: and only on that code", handleAuthRequiredFrame("backend_unreachable"), false);

/* ---------- the redirect happens once, and never onto itself ---------- */

const assigned = [];
/**
 * Stand in for the browser's one global this module touches.
 *
 * Only `replace` is provided, deliberately: a page that answered 401 (or one
 * the person has just signed out of) must not be left in history for Back to
 * return to, so a regression to `assign` should break loudly here rather than
 * quietly change the history behaviour.
 */
function stubWindow(pathname, search = "") {
  globalThis.window = {
    location: { pathname, search, replace: (href) => assigned.push(href) },
  };
}

stubWindow(LOGIN_PATH);
redirectToLogin();
check("redirect: the sign-in page never redirects to itself", assigned, []);

// Driven through `raise` rather than by calling the handler directly: this is
// the path every REST client in the app actually takes, so it is the one worth
// pinning. It must still throw - a navigation is not instant, and each
// caller's own failure path has to keep running in the meantime.
stubWindow("/console/models", "?tab=x");
let threw = null;
try {
  await raise(
    new Response(JSON.stringify({ detail: "Sign in required.", code: AUTH_REQUIRED_CODE }), {
      status: 401,
      headers: { "content-type": "application/problem+json" },
    }),
  );
} catch (e) {
  threw = e;
}
check(
  "rest funnel: a lapsed session sends the visitor to sign in, page and query remembered, flagged as lapsed",
  assigned,
  ["/login?next=%2Fconsole%2Fmodels%3Ftab%3Dx&lapsed=1"],
);
check(
  "rest funnel: and still throws, so the caller's own failure path runs",
  threw instanceof ApiError && threw.status === 401 && threw.code === AUTH_REQUIRED_CODE,
  true,
);

// A page load fires several calls at once (list, catalog, estimate); each
// failing one must not stack its own history entry on top of the others.
redirectToLogin();
redirectToLogin();
check("redirect: a burst of failures still navigates once", assigned.length, 1);

delete globalThis.window;

// A failure that is NOT a lapsed session must leave navigation alone; checked
// with the window gone, so any attempt would be visible as a crash.
let ordinary = null;
try {
  await raise(
    new Response(JSON.stringify({ detail: "Gone.", code: "not_found" }), {
      status: 404,
      headers: { "content-type": "application/problem+json" },
    }),
  );
} catch (e) {
  ordinary = e;
}
check(
  "rest funnel: an ordinary failure is still just an error",
  ordinary instanceof ApiError && ordinary.status === 404 && ordinary.code === "not_found",
  true,
);

/* ---------- the probe answers "do not know" for every way of not knowing ---------- */

const realFetch = globalThis.fetch;
// What the stub was last called with. Recorded rather than ignored so the
// REQUEST is pinned too: a stub that accepts anything would let the probe be
// renamed to a path with no route behind it and stay green, while the chip
// silently never appears.
let lastCall = null;
/** Answer the next probe with this status and body. */
function stubFetch(status, body) {
  globalThis.fetch = async (input, init) => {
    lastCall = { input, init };
    return new Response(body === undefined ? null : JSON.stringify(body), {
      status,
      headers: { "content-type": "application/json" },
    });
  };
}

stubFetch(200, {
  auth: "none",
  user_id: "00000000-0000-0000-0000-000000000001",
  display_name: "Default user",
  role: "member",
});
check("probe: sign-in switched off is a real answer, not a failure", await fetchAuthSession(), {
  auth: "none",
  user_id: "00000000-0000-0000-0000-000000000001",
  display_name: "Default user",
  role: "member",
  email: null,
});
check(
  "probe: it asks the route that actually exists, uncached",
  { url: lastCall?.input, cache: lastCall?.init?.cache },
  { url: "/api/auth/me", cache: "no-store" },
);

stubFetch(200, {
  auth: "oidc",
  user_id: "u-1",
  email: "person@example.com",
  display_name: "A Person",
  role: "admin",
});
check("probe: a signed-in member comes back whole", await fetchAuthSession(), {
  auth: "oidc",
  user_id: "u-1",
  display_name: "A Person",
  role: "admin",
  email: "person@example.com",
});

stubFetch(200, { auth: "oidc", user_id: "u-2", email: "quiet@example.com", display_name: "  ", role: "member" });
check(
  "probe: a provider that returned no name falls back to the address",
  (await fetchAuthSession())?.display_name,
  "quiet@example.com",
);

stubFetch(200, { auth: "oidc", user_id: "u-3", role: "wizard" });
check(
  "probe: an unknown role is not trusted upward",
  (await fetchAuthSession())?.role,
  "member",
);

stubFetch(200, { auth: "oidc", user_id: "u-5", email: "   ", display_name: " ", role: "member" });
const blank = await fetchAuthSession();
check(
  "probe: an address that is only whitespace is not an identity",
  { name: blank?.display_name, email: blank?.email },
  { name: "Signed in", email: null },
);

stubFetch(404, { detail: "Not Found" });
check("probe: a backend without the endpoint is unknown, not broken", await fetchAuthSession(), null);

stubFetch(401, { code: "auth_required" });
check("probe: no session yet is unknown", await fetchAuthSession(), null);

// The probe is an API reply like any other, so the signed-out rule has to hold
// for it too - it must not be the one place that swallows the signal. Fresh
// module copies because the latch is one-shot and already spent above; the
// no-code case runs first precisely because it should not spend its own.
const quiet = await freshAuth();
const quietSeen = [];
globalThis.window = { location: { pathname: "/", search: "", replace: (h) => quietSeen.push(h) } };
stubFetch(401, { code: "not_auth_at_all" });
await quiet.fetchAuthSession();
check("probe: a 401 without the code does not bounce anyone", quietSeen, []);

stubFetch(401, { code: "auth_required" });
await quiet.fetchAuthSession();
check("probe: a 401 with the code sends the visitor to sign in, flagged as lapsed", quietSeen, [
  `${LOGIN_PATH}?lapsed=1`,
]);

// Auth switched off must never trigger it, whatever else is true.
const off = await freshAuth();
const offSeen = [];
globalThis.window = { location: { pathname: "/", search: "", replace: (h) => offSeen.push(h) } };
stubFetch(200, { auth: "none", user_id: "u", display_name: "Default user", role: "member" });
await off.fetchAuthSession();
check("probe: sign-in switched off never bounces anyone", offSeen, []);

stubFetch(404, { detail: "Not Found" });
await off.fetchAuthSession();
check("probe: a backend without the endpoint never bounces anyone", offSeen, []);

delete globalThis.window;

stubFetch(200, { auth: "sideways", user_id: "u-4" });
check("probe: a body of the wrong shape is refused", await fetchAuthSession(), null);

stubFetch(200, { auth: "oidc", display_name: "Nameless" });
check("probe: a session with no user id is refused", await fetchAuthSession(), null);

stubFetch(200);
check("probe: an empty body is refused", await fetchAuthSession(), null);

globalThis.fetch = async () => {
  throw new TypeError("network down");
};
check("probe: an unreachable backend is unknown, and never throws", await fetchAuthSession(), null);

/* ---------- signing out always ends somewhere honest ---------- */

// Fresh module copies again: logout claims the one-shot latch on purpose, so
// each scenario needs its own or only the first could navigate.
const bye = await freshAuth();
const byeSeen = [];
globalThis.window = { location: { pathname: "/", search: "", replace: (h) => byeSeen.push(h) } };
stubFetch(204);
await bye.logout();
check(
  "logout: it POSTs the route that actually exists",
  { url: lastCall?.input, method: lastCall?.init?.method },
  { url: "/api/auth/logout", method: "POST" },
);
check("logout: and lands on the sign-in page", byeSeen, [LOGIN_PATH]);
// Checked structurally rather than by hanging a stub for the full timeout: the
// deadline exists so a backend that never answers cannot leave someone reading
// "Signing out" indefinitely, and a battery that reproduced that by waiting
// would have to wait exactly as long to find out.
//
// Both halves are needed, and the first alone was decorative: a bare
// `new AbortController().signal` is an AbortSignal that nobody ever fires, so
// it passes the instanceof and never expires. The exported constant is what
// makes the deadline a NUMBER this can hold to a sane range.
check(
  "logout: the request carries a signal at all",
  lastCall?.init?.signal instanceof AbortSignal,
  true,
);
check(
  "logout: and that signal is a real deadline, in a sane range",
  typeof LOGOUT_TIMEOUT_MS === "number" &&
    LOGOUT_TIMEOUT_MS > 0 &&
    LOGOUT_TIMEOUT_MS <= 10000,
  true,
);

// The case the docstring is about: a person left staring at their own name
// because the server did not answer is the failure, not the failed request.
const stuck = await freshAuth();
const stuckSeen = [];
globalThis.window = { location: { pathname: "/", search: "", replace: (h) => stuckSeen.push(h) } };
globalThis.fetch = async () => {
  throw new TypeError("network down");
};
await stuck.logout();
check("logout: a refused request still signs the person out", stuckSeen, [LOGIN_PATH]);

// The one that cost real time to write and is worth it: a backend that accepts
// the connection and then says NOTHING. Both structural checks above are
// satisfied by `new AbortController().signal` - an AbortSignal nobody ever
// fires - so only driving the behaviour tells the two apart. The stub honors
// the signal, exactly as a real fetch does, and the watchdog exists so the
// failure reads as a red check rather than a battery that hangs forever.
const hung = await freshAuth();
const hungSeen = [];
globalThis.window = { location: { pathname: "/", search: "", replace: (h) => hungSeen.push(h) } };
globalThis.fetch = (input, init) =>
  new Promise((_, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
  });
const outcome = await Promise.race([
  hung.logout().then(() => "returned"),
  new Promise((r) => setTimeout(() => r("hung"), LOGOUT_TIMEOUT_MS + 2000)),
]);
check("logout: a backend that never answers cannot hold the sign-out open", outcome, "returned");
check("logout: and the person still lands on the sign-in page", hungSeen, [LOGIN_PATH]);

// A 401 landing from some other in-flight call must not beat the sign-out and
// hand it a `next` back into the app the person just asked to leave.
const raced = await freshAuth();
const racedSeen = [];
globalThis.window = {
  location: { pathname: "/console/models", search: "", replace: (h) => racedSeen.push(h) },
};
stubFetch(204);
const leaving = raced.logout();
raced.redirectToLogin();
await leaving;
check("logout: an in-flight 401 cannot smuggle a return path past it", racedSeen, [LOGIN_PATH]);

delete globalThis.window;

/* ---------- the gate decides before anything is rendered ---------- */

// From here on there is no `window` at all, which is the honest environment:
// this code runs on the server, in front of the render, and anything reaching
// for the browser here is a bug the deletion above will expose as a crash.

const { NextRequest } = await import("next/server");
const { PROBE_TIMEOUT_MS, config: proxyConfig, proxy } = await import("../proxy.ts");

/**
 * Run the real gate over one path, with the backend stubbed to one answer.
 *
 * `status: null` stands for a backend that never answered at all; `raw` hands
 * the stub a whole Response for the cases a status and a JSON body cannot
 * describe.
 */
async function runGate(pathAndQuery, { status, body, raw }) {
  if (raw !== undefined) {
    globalThis.fetch = async (input, init) => {
      lastCall = { input, init };
      return raw();
    };
  } else if (status === null) {
    globalThis.fetch = async () => {
      throw new TypeError("network down");
    };
  } else {
    stubFetch(status, body);
  }
  const response = await proxy(
    new NextRequest(`http://localhost:3000${pathAndQuery}`, {
      headers: { cookie: "session_id=abc123" },
    }),
  );
  return {
    status: response.status,
    location: response.headers.get("location"),
    next: response.headers.get("x-middleware-next"),
  };
}

/**
 * What letting the request through looks like from outside.
 *
 * The continue TOKEN is read, not merely the absence of a redirect, and that is
 * the difference between a real check and a comforting one. Next treats a
 * middleware response carrying no rewrite, no continue token and no Location as
 * TERMINAL and serves it as the answer - so a plain `new NextResponse()` here
 * is also a 200 with no Location, passes any status-and-Location comparison,
 * and blanks every gated page in the app. That is a far bigger outage than the
 * flash this file exists to remove, and it would have shipped green.
 */
const PASSED = { status: 200, location: null, next: "1" };

/** The other of the two shapes: turned away, towards a named destination. */
function BOUNCED(path) {
  return { status: 307, location: `http://localhost:3000${path}`, next: null };
}

check(
  "gate: a lapsed session never gets the page, and its destination is remembered",
  await runGate("/console/models?tab=x", { status: 401, body: { code: "auth_required" } }),
  BOUNCED("/login?next=%2Fconsole%2Fmodels%3Ftab%3Dx"),
);
check(
  "gate: the probe carried the caller's own cookie to the backend",
  lastCall?.init?.headers?.cookie,
  "session_id=abc123",
);
// The host is whatever BACKEND_INTERNAL_URL says and is deployment config, so
// only the path is pinned - a renamed probe endpoint would otherwise 404 and,
// failing open, silently take the whole gate out of service.
check(
  "gate: and asked the endpoint that exists, uncached",
  {
    path: String(lastCall?.input).replace(/^https?:\/\/[^/]+/, ""),
    cache: lastCall?.init?.cache,
  },
  { path: "/api/v1/auth/me", cache: "no-store" },
);
check(
  "gate: the probe is on a deadline, so a hung backend cannot hang the page",
  lastCall?.init?.signal instanceof AbortSignal,
  true,
);

// The root needs no `next` at all - and the bare `/login` it produces is the
// one URL the sign-in page's own bounce-to-`/` reacts to, so this is the case
// where a loop would live if the two ever disagreed.
check(
  "gate: the root sends the visitor to a bare sign-in page",
  await runGate("/", { status: 401, body: { code: "auth_required" } }),
  BOUNCED(LOGIN_PATH),
);

// A soft navigation asks for the page's PAYLOAD - same path, plus the router's
// own cache-busting marker. Carried into the return path, that marker would
// land a freshly signed-in visitor on a screenful of serialized React instead
// of the page they were going to.
check(
  "gate: the router's transport marker is not remembered as part of the destination",
  await runGate("/console/models?tab=x&_rsc=1a2b3c", {
    status: 401,
    body: { code: "auth_required" },
  }),
  BOUNCED("/login?next=%2Fconsole%2Fmodels%3Ftab%3Dx"),
);

// Four ways of not being the signal, all of which must let the page render.
check(
  "gate: a 401 that is somebody else's refusal is not a sign-out",
  await runGate("/console/models", { status: 401, body: { code: "rate_limited" } }),
  PASSED,
);
check(
  "gate: a signed-in visitor is not interrupted",
  await runGate("/console/models", {
    status: 200,
    body: { auth: "oidc", user_id: "u-1", display_name: "A Person", role: "member" },
  }),
  PASSED,
);
// The locked guarantee: with sign-in switched off the backend answers this to
// every probe, so a deployment that has no login is byte-identical to one
// without this file.
check(
  "gate: sign-in switched off passes everything through",
  await runGate("/console/models", {
    status: 200,
    body: { auth: "none", user_id: "u-0", display_name: "Operator", role: "member" },
  }),
  PASSED,
);
check(
  "gate: an unreachable backend fails OPEN, never into a redirect loop",
  await runGate("/console/models", { status: null }),
  PASSED,
);
// A 401 that is not JSON at all. An ingress proxy, a WAF or a misconfigured
// gateway answers HTML, and reading the code off it throws - inside the gate,
// where an uncaught throw is a 500 on every page rather than one bad request.
check(
  "gate: a 401 that is not JSON is survived, not raised",
  await runGate("/console/models", {
    raw: () =>
      new Response("<html><body>Unauthorized</body></html>", {
        status: 401,
        headers: { "content-type": "text/html" },
      }),
  }),
  PASSED,
);

// The deadline, driven rather than inspected. Both structural checks above are
// satisfied by an `AbortController().signal` that nobody ever fires, and the
// difference between that and a real timeout is a backend that accepts the
// connection and then says nothing: every page render waits forever. The stub
// honors the signal exactly as a real fetch does, and the race is what turns a
// battery that would otherwise HANG into a red check.
check(
  "gate: the deadline is a real number, in a sane range",
  typeof PROBE_TIMEOUT_MS === "number" && PROBE_TIMEOUT_MS > 0 && PROBE_TIMEOUT_MS <= 5000,
  true,
);
globalThis.fetch = (input, init) =>
  new Promise((_, reject) => {
    init?.signal?.addEventListener("abort", () => reject(new Error("aborted")));
  });
const hungGate = await Promise.race([
  proxy(new NextRequest("http://localhost:3000/console/models")).then((r) => ({
    status: r.status,
    location: r.headers.get("location"),
    next: r.headers.get("x-middleware-next"),
  })),
  new Promise((r) => setTimeout(() => r("hung"), PROBE_TIMEOUT_MS + 2000)),
]);
check("gate: a backend that never answers cannot hold the page open", hungGate, PASSED);

// The redirect must not be cacheable: kept, a proxy in front of the app would
// go on serving it after the person signed in.
globalThis.fetch = async () =>
  new Response(JSON.stringify({ code: "auth_required" }), {
    status: 401,
    headers: { "content-type": "application/problem+json" },
  });
const bounced = await proxy(new NextRequest("http://localhost:3000/console"));
check("gate: the redirect is not cacheable", bounced.headers.get("cache-control"), "no-store");

globalThis.fetch = realFetch;

/* ---------- and it runs on pages, and on nothing else ---------- */

let matcherRegexp = null;
try {
  matcherRegexp = compileMatchers(proxyConfig.matcher)[0];
} catch (e) {
  check("gate matcher: the matcher compiled at all", `failed: ${e.message}`, "compiled");
}

if (matcherRegexp !== null) {
  const matches = (pathname) => new RegExp(matcherRegexp).test(pathname);
  const gated = [
    ["the app root", "/"],
    ["a console page", "/console/models"],
    // Prefetch and soft-navigation spellings of a protected page. Excluding
    // these is the hole that the tempting "anything with a dot is a static
    // file" matcher opens: the markup still leaves the server, just under a
    // different filename.
    ["a prefetched page payload", "/console/models.rsc"],
    ["a segment payload", "/console/models.segments/x.segment.rsc"],
    // None of these four is a route today, and that is the point: they pin the
    // WORD BOUNDARY on each exclusion. Written as bare prefixes - `login`,
    // `api`, `_next` - the matcher still passes every other check in this file
    // while quietly exempting any future page whose name merely starts with
    // one of those words. The mistake looks like a tidier version of the rule.
    ["a page that only starts like the sign-in one", "/login-help"],
    ["a page that only starts like the API prefix", "/api-docs"],
    ["a page that only starts like the framework prefix", "/_nextsteps"],
    ["a path under something named like the favicon", "/favicon.ico/x"],
  ];
  for (const [what, pathname] of gated) {
    check(`gate matcher: ${what} is gated`, matches(pathname), true);
  }

  const exempt = [
    // Gating this is an infinite bounce: the page a signed-out visitor is sent
    // to would itself send them away.
    ["the sign-in page", "/login"],
    ["its prefetch payload", "/login.rsc"],
    // Gating these answers a `fetch` with an HTML redirect and retires the
    // client-side 401 handling - and the sign-in legs must stay reachable to
    // someone who is by definition not signed in yet.
    ["the API relay", "/api/conversations"],
    ["the sign-in relay", "/api/v1/auth/login"],
    // Dozens per page load, none of them protected content.
    ["a script chunk", "/_next/static/chunks/main.js"],
    ["the image optimizer", "/_next/image"],
    ["the favicon", "/favicon.ico"],
  ];
  for (const [what, pathname] of exempt) {
    check(`gate matcher: ${what} is exempt`, matches(pathname), false);
  }
}

console.log(
  failures === 0
    ? `auth contract: ${checks} checks passed`
    : `auth contract: ${failures} of ${checks} checks FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
