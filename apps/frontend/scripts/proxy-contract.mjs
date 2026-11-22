/**
 * The invariants every `/api/*` proxy route must hold, in both directions.
 *
 * A route handler runs on the server, so the backend sees the request coming
 * from THIS process rather than from the person who made it. `proxyHeaders`
 * carries the caller's address and cookies forward; a handler that forgets it
 * hands the backend one identity for the whole deployment, and the per-client
 * rate limit silently becomes a single shared budget while the audit trail
 * names the wrong client - or, once sessions exist, the call arrives
 * anonymously and answers 401 while the browser holds a good session. Nothing
 * about the first failure is visible: the route works perfectly until two
 * people use it at once.
 *
 * The return leg has the same shape of hazard. A backend response can carry a
 * `Set-Cookie` that mints or clears a session, and a leg that builds its own
 * response headers drops it without a word - login appears to succeed and
 * nobody is signed in.
 *
 * Both are checked statically, over every file that names `BACKEND_URL` at
 * once, rather than by a test per route that the next new route would not
 * have:
 *
 *   1. every `fetch(` in such a file forwards the caller, via `proxyHeaders(`
 *      in a route handler or `forwardedHeaders(` in a server component;
 *   2. a route handler constructs NO response of its own, except the abort for
 *      a caller who has already hung up. Earlier versions of this rule tried
 *      to allow hand-built responses and merely require a cookie call beside
 *      them; both attempts were escapable, because nothing tied a given call
 *      to a given response. The relay helpers now cover every shape a route
 *      needs, so the rule can be structural and have no inference in it.
 *
 * These files exist to relay to the backend and reach nothing else, so the
 * rules need no exceptions - and a route that one day legitimately fetches
 * something else should have to come here and say so.
 *
 * Below the static pass, the cookie helpers in `lib/backend-proxy.ts` are
 * exercised for real, because "multi-value Set-Cookie" is where this goes
 * wrong quietly: `Headers.get` folds the values into one comma-joined line,
 * and cookie `Expires` attributes contain a comma, so the folded form cannot
 * be taken apart again. Running the helpers needs a Node that strips types on
 * its own - 22.18 or newer, like the batteries beside it.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/proxy-contract.mjs)
 */

import { readdirSync, readFileSync } from "node:fs";
import { register } from "node:module";
import path from "node:path";
import { fileURLToPath } from "node:url";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `proxy contract: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const {
  applySetCookies,
  proxyHeaders,
  relay,
  relayDownload,
  relayRedirect,
  relayStream,
  setCookiesFrom,
} = await import("../lib/backend-proxy.ts");

const APP_DIR = fileURLToPath(new URL("../app/", import.meta.url));

/**
 * Every file beneath `app/` that talks to the backend, as absolute paths.
 *
 * Not just `route.ts`. A server component can call the backend too - the
 * sign-in page probes it before deciding whether to render at all - and a scan
 * that knew only about route handlers would have left that call, and the
 * identity headers it has to carry, outside the rule entirely. Naming
 * `BACKEND_URL` is the only way to reach the backend from this origin, which
 * makes it the honest definition of "in scope" rather than a guess about where
 * such code tends to live.
 */
function backendCallers(dir) {
  const found = [];
  for (const entry of readdirSync(dir, { withFileTypes: true })) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) found.push(...backendCallers(full));
    else if (/\.tsx?$/.test(entry.name) && readFileSync(full, "utf8").includes("BACKEND_URL")) {
      found.push(full);
    }
  }
  return found.sort();
}

/** Route handlers carry the response-construction rules; other callers do not. */
function isRouteHandler(file) {
  return path.basename(file) === "route.ts";
}

/**
 * Blank out comments, preserving every other character's position.
 *
 * Done BEFORE any paren matching, and not as a nicety: these files are heavily
 * commented in prose, and prose is full of apostrophes. A scanner that tracked
 * quotes without knowing what a comment is reads "the caller's address" as the
 * start of a string literal and swallows the rest of the file - which is
 * exactly how the first version of this script reported the chat route as
 * having no fetch at all.
 */
function stripComments(source) {
  let out = "";
  let i = 0;
  let quote = null;
  while (i < source.length) {
    const c = source[i];
    const next = source[i + 1];
    if (quote) {
      out += c;
      if (c === "\\") { out += next ?? ""; i += 2; continue; }
      if (c === quote) quote = null;
      i += 1;
      continue;
    }
    if (c === "/" && next === "/") {
      while (i < source.length && source[i] !== "\n") { out += " "; i += 1; }
      continue;
    }
    if (c === "/" && next === "*") {
      const end = source.indexOf("*/", i + 2);
      const stop = end === -1 ? source.length : end + 2;
      // Newlines are kept so a reported position still lines up with the file.
      for (; i < stop; i += 1) out += source[i] === "\n" ? "\n" : " ";
      continue;
    }
    if (c === '"' || c === "'" || c === "`") quote = c;
    out += c;
    i += 1;
  }
  return out;
}

/**
 * The source text of every call matching `pattern`, argument list only.
 *
 * Parenthesis counting over comment-free source, with string and template
 * literals skipped so a `)` inside a URL cannot end the call early. Crude next
 * to a real parser, and enough for the shape these files are written in.
 */
function callArgs(clean, pattern) {
  const calls = [];
  const CALL = new RegExp(pattern, "g");
  let match;
  while ((match = CALL.exec(clean)) !== null) {
    let depth = 1;
    let i = match.index + match[0].length;
    const start = i;
    let quote = null;
    while (i < clean.length && depth > 0) {
      const c = clean[i];
      if (quote) {
        if (c === "\\") i += 1;
        else if (c === quote) quote = null;
      } else if (c === '"' || c === "'" || c === "`") {
        quote = c;
      } else if (c === "(") depth += 1;
      else if (c === ")") depth -= 1;
      i += 1;
    }
    if (depth === 0) calls.push(clean.slice(start, i - 1));
  }
  return calls;
}

/** Each `fetch(...)` call's arguments, from raw (still commented) source. */
function fetchCalls(source) {
  return callArgs(stripComments(source), "\\bfetch\\s*\\(");
}

/**
 * Every way a route file can mint a response of its own, as argument text.
 *
 * Deliberately not just `new Response(`. The static factories build exactly the
 * same thing and read as the more idiomatic choice, so a rule that only knew
 * the constructor would wave through `Response.json(await upstream.json())` -
 * a route that reconstructs the body by hand and drops every header with it,
 * which is precisely the shape this rule exists to catch.
 */
function responseCalls(clean) {
  return callArgs(
    clean,
    "\\b(?:new\\s+(?:Next)?Response|(?:Next)?Response\\s*\\.\\s*(?:json|redirect|next))\\s*\\(",
  );
}

/**
 * Whether a constructed response is the client-gone abort.
 *
 * The one case with genuinely nothing to carry: the caller's connection is
 * already closed, so no cookie can reach a browser that is not listening. The
 * exemption is written as "status 499" rather than "has no body" on purpose -
 * a bodyless response is NOT inherently safe, and a hand-built 204 relaying a
 * sign-out would drop the cookie-clearing header that is its entire payload.
 */
function isClientGone(args) {
  return /\bstatus:\s*499\b/.test(args);
}

let failures = 0;
let checks = 0;

function check(label, ok, detail) {
  checks += 1;
  if (!ok) {
    failures += 1;
    console.error(`FAIL  ${label}`);
    if (detail) console.error(`  ${detail}`);
  }
}

const files = backendCallers(APP_DIR);

check(
  "proxy contract: the backend callers were found at all",
  files.filter(isRouteHandler).length > 0,
  `no route.ts naming BACKEND_URL under ${APP_DIR}`,
);

for (const file of files) {
  const rel = path.relative(path.join(APP_DIR, ".."), file).replace(/\\/g, "/");
  const source = readFileSync(file, "utf8");
  const calls = fetchCalls(source);
  check(
    `proxy contract: ${rel} relays at least one call`,
    calls.length > 0,
    "a file naming BACKEND_URL with no fetch is either dead or reaching the backend some other way",
  );
  calls.forEach((args, index) => {
    // Either spelling satisfies it: a route handler holds a NextRequest and
    // uses proxyHeaders, a server component has only what next/headers gives
    // it and uses forwardedHeaders. Both derive the same two headers from the
    // one definition in lib/backend-proxy.ts, which is the point of the split.
    check(
      `proxy contract: ${rel} fetch #${index + 1} forwards the caller`,
      /(?:proxyHeaders|forwardedHeaders)\s*\(/.test(args),
      "pass `headers: proxyHeaders(request)` (route handler) or `headers: forwardedHeaders(await headers())` (server component) - without it the backend keys its rate limit and its audit on this server rather than on the person who asked, and the call arrives with no session",
    );
  });

  // The response-construction rules below are about RELAYING, so they ask only
  // route handlers. A server component never hands the backend's response to
  // anyone - it reads an answer and decides what to render - so it has no
  // return leg to drop a cookie from.
  if (!isRouteHandler(file)) continue;

  // The return leg, checked per CONSTRUCTED RESPONSE rather than per file.
  //
  // The first version of this rule asked the question once per file, and that
  // was hollow in the way that matters: the chat route builds two responses,
  // so deleting the cookie from its streaming leg left the other call sight
  // of the regex and the battery stayed green - while every chat turn quietly
  // stopped sliding the session forward. It also keyed on the identifier
  // `upstream`, so the same route written with any other variable name was
  // exempt from a rule whose whole purpose is to catch the NEXT new route.
  //
  // Two narrower rules replace it, both keyed on the construction itself:
  const clean = stripComments(source);
  const constructed = responseCalls(clean);

  // (1) A streamed relay must go through the helper. This is the leg that
  //     needs headers `relay` cannot give it, and therefore the one a route is
  //     tempted to hand-build; `relayStream` carries the cookie so no route
  //     has to remember to.
  constructed.forEach((args, index) => {
    check(
      `proxy contract: ${rel} response #${index + 1} does not hand-build a stream`,
      !/\.body\b/.test(args),
      "return `relayStream(upstream, HEADERS)` instead - a hand-built streaming response is one more place the session cookie has to be remembered, and forgetting it is invisible",
    );
  });

  // (2) A route file constructs NO responses of its own, full stop - only the
  //     abort for a caller who has already hung up.
  //
  //     This rule was twice too clever before. Asking "does the file mention
  //     applySetCookies" let a second, cookie-dropping construction hide
  //     behind a first one that was fine. Counting the calls against the
  //     constructions was better and still guessable: nothing tied a
  //     particular call to a particular response, so aliasing the body into a
  //     local and building from that kept the arithmetic balanced while the
  //     cookie went missing on every chat turn.
  //
  //     There is no clever version of this. The fix is that route handlers
  //     have nothing left to construct - relay, relayRedirect, relayDownload,
  //     relayStream and relayFrame cover every shape - so the rule can be the
  //     flat structural one, checked per construction, with no inference in it
  //     at all.
  constructed.forEach((args, index) => {
    check(
      `proxy contract: ${rel} response #${index + 1} is the abort, not a hand-built relay`,
      isClientGone(args),
      "route files must not construct responses - return relay / relayRedirect / relayDownload / relayStream / relayFrame, which carry the backend's Set-Cookie for you. Only `new Response(null, { status: 499 })` (the caller already hung up) is built here",
    );
  });
}

/* ---------- the cookie helpers, exercised for real ---------- */

// The exact shape that punishes a folded read: an Expires attribute whose
// weekday is followed by a comma. Joined into one header these two cannot be
// separated again by any comma split.
const SESSION_COOKIE =
  "session_id=abc123; Path=/; Expires=Wed, 21 Oct 2026 07:28:00 GMT; HttpOnly; SameSite=Lax";
const OTHER_COOKIE = "csrf=zzz; Path=/; Expires=Thu, 22 Oct 2026 07:28:00 GMT; SameSite=Lax";

/** A stand-in backend response carrying both cookies. */
function upstreamWithCookies(body, init = {}) {
  const headers = new Headers(init.headers ?? {});
  headers.append("set-cookie", SESSION_COOKIE);
  headers.append("set-cookie", OTHER_COOKIE);
  return new Response(body, { ...init, headers });
}

/** The `Set-Cookie` values a relayed response actually carries. */
function cookiesOn(response) {
  return response.headers.getSetCookie();
}

const withCookie = proxyHeaders(
  new Request("http://localhost/api/thing", {
    headers: { cookie: "session_id=abc123", "x-forwarded-for": "203.0.113.7" },
  }),
);
check(
  "proxy headers: the caller's cookie travels to the backend",
  withCookie.cookie === "session_id=abc123",
  "without it every relayed call reaches the backend signed out",
);

// The OIDC round trip rides on TWO cookies, not one: login sets a `login_state`
// CSRF cookie on its 302, and callback needs it back ALONGSIDE the session
// cookie to check the returning request against the one it started. Forwarding
// the header verbatim carries every cookie the browser holds - which is the
// whole reason this must never filter by name. A guard against a well-meaning
// future edit that "forwards just the session cookie" and silently breaks the
// CSRF check.
const bothForward = proxyHeaders(
  new Request("http://localhost/api/v1/auth/callback?code=x&state=y", {
    headers: { cookie: "session_id=abc123; login_state=csrf-nonce-42" },
  }),
);
check(
  "proxy headers: the callback carries BOTH the session and the CSRF cookie back",
  bothForward.cookie === "session_id=abc123; login_state=csrf-nonce-42",
  "callback checks the returning request against the login_state it set on the 302; forwarding only session_id would fail every real sign-in as a CSRF mismatch",
);
check(
  "proxy headers: the caller's address still travels",
  withCookie["x-forwarded-for"] === "203.0.113.7",
  "the rate-limit/audit key regressed",
);
check(
  "proxy headers: no cookie is invented when the caller sent none",
  !("cookie" in proxyHeaders(new Request("http://localhost/api/thing"))),
  "a synthesized header would put a value nobody vouched for in front of the backend",
);

const parsed = setCookiesFrom(upstreamWithCookies(null));
check(
  "set-cookie: both cookies survive as separate values",
  parsed.length === 2 && parsed[0] === SESSION_COOKIE && parsed[1] === OTHER_COOKIE,
  `folding them into one line makes the Expires comma unparseable - got ${JSON.stringify(parsed)}`,
);

const relayed = relay(
  upstreamWithCookies('{"ok":true}', {
    status: 200,
    headers: { "content-type": "application/json" },
  }),
);
check(
  "relay: the backend's cookies reach the browser",
  JSON.stringify(cookiesOn(relayed)) === JSON.stringify([SESSION_COOKIE, OTHER_COOKIE]),
  `got ${JSON.stringify(cookiesOn(relayed))}`,
);
check(
  "relay: the content type is still forwarded",
  relayed.headers.get("content-type") === "application/json",
  "problem+json errors would lose their media type",
);

// The logout leg: 204, no body, and the cleared cookie is the ENTIRE payload.
const relayed204 = relay(upstreamWithCookies(null, { status: 204 }));
check(
  "relay: a 204 still carries its cookies",
  relayed204.status === 204 && cookiesOn(relayed204).length === 2 && relayed204.body === null,
  "logout says everything it has to say in this one header",
);

const downloaded = relayDownload(
  upstreamWithCookies("bytes", {
    status: 200,
    headers: { "content-type": "application/pdf", "content-disposition": 'attachment; filename="a.pdf"' },
  }),
);
check(
  "relay download: a download is an authenticated request like any other",
  cookiesOn(downloaded).length === 2 &&
    downloaded.headers.get("content-disposition") === 'attachment; filename="a.pdf"',
  `got ${JSON.stringify(cookiesOn(downloaded))}`,
);

const redirected = relayRedirect(
  upstreamWithCookies(null, { status: 302, headers: { location: "/console" } }),
);
check(
  "relay redirect: the session and the redirect travel together",
  redirected.status === 302 &&
    redirected.headers.get("location") === "/console" &&
    JSON.stringify(cookiesOn(redirected)) === JSON.stringify([SESSION_COOKIE, OTHER_COOKIE]),
  "the callback return leg mints the session in the SAME response as its redirect; dropping either one signs nobody in",
);

// The login leg specifically: it is a 302 to the identity provider that ALSO
// sets a `login_state` CSRF cookie, so this relay carries a Set-Cookie on the
// OUTBOUND hop too - not only the callback's return hop. Same helper, but named
// as its own case because it is the one the outbound leg was easiest to assume
// carried no cookie.
const LOGIN_STATE_COOKIE =
  "login_state=csrf-nonce-42; Path=/; Expires=Fri, 23 Oct 2026 07:28:00 GMT; HttpOnly; SameSite=Lax";
const loginUpstream = new Response(null, {
  status: 302,
  headers: new Headers({ location: "https://accounts.example/authorize?..." }),
});
loginUpstream.headers.append("set-cookie", LOGIN_STATE_COOKIE);
const loginRelayed = relayRedirect(loginUpstream);
check(
  "relay redirect: the login 302 carries its CSRF cookie to the provider hop",
  loginRelayed.status === 302 &&
    loginRelayed.headers.get("location") === "https://accounts.example/authorize?..." &&
    JSON.stringify(cookiesOn(loginRelayed)) === JSON.stringify([LOGIN_STATE_COOKIE]),
  "callback rejects a return with no matching login_state; dropping this Set-Cookie makes every sign-in a CSRF failure",
);

const SSE_HEADERS = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache, no-transform",
  "x-accel-buffering": "no",
};
const streamed = relayStream(
  upstreamWithCookies("event: token\ndata: hi\n\n", { status: 200 }),
  SSE_HEADERS,
);
check(
  "relay stream: the session slides forward on a streaming turn too",
  JSON.stringify(cookiesOn(streamed)) === JSON.stringify([SESSION_COOKIE, OTHER_COOKIE]),
  "a user who only ever chats would be signed out at the absolute TTL with nothing to say why",
);
check(
  "relay stream: the stream's own headers survive",
  streamed.headers.get("content-type") === "text/event-stream; charset=utf-8" &&
    streamed.headers.get("x-accel-buffering") === "no" &&
    streamed.status === 200,
  "these are the headers relay() cannot give it, and the reason this helper exists",
);

const notARedirect = relayRedirect(
  upstreamWithCookies("Access denied.", {
    status: 403,
    headers: { "content-type": "text/plain; charset=utf-8" },
  }),
);
check(
  "relay redirect: a response with no Location falls back to a plain relay",
  notARedirect.status === 403 &&
    notARedirect.headers.get("content-type") === "text/plain; charset=utf-8" &&
    notARedirect.body !== null,
  "an admission refusal must reach the browser in the backend's own words, not as a blank page",
);

// A guard for the guard: `applySetCookies` must APPEND, so a header the route
// set for its own reasons is not clobbered and neither cookie is lost.
const built = new Headers({ "content-type": "text/event-stream" });
applySetCookies(upstreamWithCookies(null), built);
check(
  "apply set-cookies: appends onto headers the route already set",
  built.get("content-type") === "text/event-stream" && built.getSetCookie().length === 2,
  `got ${JSON.stringify(built.getSetCookie())}`,
);

console.log(
  failures === 0
    ? `proxy contract: ${checks} checks passed across ${files.length} route files`
    : `proxy contract: ${failures} of ${checks} checks FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
