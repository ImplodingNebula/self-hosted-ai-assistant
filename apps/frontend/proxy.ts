/**
 * The sign-in gate, decided before a single byte of a page is sent.
 *
 * Without it an unauthenticated visitor gets the whole app: the HTML arrives,
 * React hydrates, the first API call comes back 401, and only THEN does the
 * client bounce to the sign-in page. What that looks like is a full flash of
 * the signed-in interface - the model manager even paints its red "Sign in to
 * continue" - for a person who is not signed in and never was. Protected
 * markup reaching a browser at all is the problem; a redirect that happens
 * afterwards does not un-send it.
 *
 * So the question is asked here, on the server, in front of the render. This
 * is the file Next 16 calls `proxy` - the convention that replaced
 * `middleware`, which still works but prints a deprecation warning on every
 * build and every dev boot, and which defaults to the edge runtime where this
 * needs none of edge's constraints.
 *
 * A deployment with sign-in switched off is UNTOUCHED by all of this, and that
 * is a guarantee rather than a happy accident. The backend answers such a probe
 * `200 {auth: "none"}`, which is not the one answer that redirects, so every
 * request passes through exactly as it did before this file existed.
 *
 * The gate cannot loop with the sign-in page, and the reasoning is worth
 * writing down because a loop here would be invisible - two redirects with no
 * page in between. With sign-in ON: `/login` is outside the matcher, so the
 * gate never runs on it, and the page's own bounce-to-`/` fires only when its
 * probe says sign-in is unavailable, which a live auth system never says. With
 * sign-in OFF: the gate never redirects anyone in the first place, so nothing
 * arrives at `/login` for the page to bounce. The two agree because they read
 * the same endpoint and judge it by the same rule - {@link authGateDecision},
 * which is shared code and not a second copy.
 *
 * That agreement is the whole proof, so it is worth naming what it rests on:
 * both sides asking ONE backend that answers the question the same way twice.
 * A rolling deploy that leaves two replicas disagreeing about whether sign-in
 * is configured is the one state where they can disagree - and a visitor sent
 * from `/` arrives at a bare `/login` with no `next`, which is exactly the
 * evidence the sign-in page's own circuit breaker looks for and will not find.
 * Nothing here defends against that; a marker the breaker also honoured would,
 * if a mixed-version window ever becomes a real deployment shape.
 */
import type { NextRequest } from "next/server";
import { NextResponse } from "next/server";

import { authGateDecision, loginHref } from "@/lib/auth";
import { BACKEND_URL, forwardedHeaders } from "@/lib/backend-proxy";

/**
 * Which requests the gate runs on.
 *
 * Page routes, and only page routes. The three exclusions each prevent a
 * concrete failure rather than merely saving work:
 *
 *   - `/api/*` is the relay the client calls with `fetch`. Gating it would put
 *     an HTML redirect where a caller expects JSON and would retire the
 *     client-side 401 machinery that has to keep working for lapsed sessions
 *     mid-session. It also carries the sign-in flow's own legs, which must
 *     stay reachable to a visitor who is by definition not signed in yet.
 *   - `/_next/*` is chunks and assets, requested by the dozen per page; a
 *     backend round trip each would be absurd, and none of them is protected
 *     content.
 *   - `/login` must render for someone who cannot sign in yet. Gating it is
 *     the loop.
 *
 * Everything else is in, including the App Router's transport spellings of a
 * page (`.rsc`, `.segments/...`) - Next appends those forms to whatever source
 * is written here, and they carry exactly the protected markup this gate
 * exists to withhold. That is why the usual "anything with a dot" exclusion is
 * deliberately NOT used: it reads as a rule about static files and is in fact
 * a hole that exempts a prefetched protected page. `public/` holds no files,
 * so the only real static path a browser asks for unprompted is the favicon,
 * and it is named.
 *
 * Two consequences of that choice, both worth knowing before changing anything
 * near it. Adding a file to `public/` puts it behind the gate - a probe per
 * request, and a 307 to sign in while signed out - so a genuinely public asset
 * has to be named here alongside the favicon. And the dev server's own overlay
 * endpoints (`/__nextjs_*`) are gated too, so a signed-out development session
 * loses its error overlay until it signs in.
 *
 * Each exclusion is written to end at a word boundary rather than as a bare
 * prefix. `login` alone would exempt a future `/login-help`, and `api` alone a
 * future `/api-docs` - real page routes, silently ungated, with the mistake
 * looking exactly like a tidier version of the same rule.
 */
export const config = {
  matcher: ["/((?!api(?:$|/)|_next(?:$|/)|login(?:$|[/.])|favicon\\.ico$).*)"],
};

/**
 * How long to wait for the probe before letting the request through anyway.
 *
 * Shorter than the sign-in page's own probe, because this one sits in front of
 * every matched request rather than one page: when the backend is down, the
 * cost of the gate is a delay added to a request that was going to fail on its
 * own merits, and it should be brief.
 *
 * Exported so a check can hold the deadline to a real number. A `signal` is not
 * evidence on its own - an `AbortController().signal` nobody ever fires passes
 * every structural test for one and expires never, which here would hang not a
 * button but every page render.
 */
export const PROBE_TIMEOUT_MS = 2500;

/** What a probe answered, with `null` standing for "it never answered". */
interface ProbeAnswer {
  /** HTTP status, or null when the request itself failed. */
  status: number | null;
  /** The problem+json `code`, when there was one. */
  code?: string;
}

/**
 * Gate one page request on whether the visitor is signed in.
 *
 * One backend round trip per MATCHED REQUEST - which is a navigation plus
 * whatever links Next prefetches around it, not one per page and not one per
 * resource (assets and API calls are outside the matcher entirely). Opening
 * the console is a handful of probes rather than one. They are cheap and they
 * slide the session forward, but the number is worth stating honestly, since
 * it is the cost being weighed against caching the answer.
 *
 * That price buys "protected markup never reaches the browser", and it is paid
 * even when a `session_id` cookie is sitting right there: a cookie is evidence
 * that somebody signed in once, not that the session is still alive, and
 * trusting its mere presence reproduces the exact flash this file removes for
 * anyone whose session has expired.
 *
 * The answer is deliberately not cached anywhere. A cached "signed in" outlives
 * the sign-out that should have ended it, which is a worse bug than the round
 * trips are a cost.
 *
 * @param request - The incoming page request.
 * @returns A redirect to sign in, or permission to carry on.
 */
export async function proxy(request: NextRequest): Promise<NextResponse> {
  const answer = await probeSession(request);
  if (authGateDecision(answer.status, answer.code) === "pass") {
    return NextResponse.next();
  }
  const response = NextResponse.redirect(
    new URL(loginHref(returnPath(request)), request.url),
  );
  // The redirect describes one moment's session state and nothing more. Left
  // cacheable, a proxy in front of the app could keep answering it after the
  // person signed in, pinning them on the sign-in page with a valid session.
  response.headers.set("cache-control", "no-store");
  return response;
}

/**
 * Where to send the visitor back to once they have signed in.
 *
 * The URL as asked for, minus Next's own transport marker. A soft navigation
 * inside the app does not request the page, it requests the page's payload -
 * same path, plus a `_rsc` cache-busting param the router added. Remembering
 * that param verbatim would hand somebody, at the end of a successful sign-in,
 * a document request for a payload: a screenful of serialized React rather than
 * the page they were going to. Next strips the same param wherever it turns one
 * of these requests back into a URL, for the same reason.
 */
function returnPath(request: NextRequest): string {
  const url = new URL(request.nextUrl);
  url.searchParams.delete(RSC_QUERY_PARAM);
  return `${url.pathname}${url.search}`;
}

/** Next's App Router marks a payload request with this query parameter. */
const RSC_QUERY_PARAM = "_rsc";

/**
 * Ask the backend who is making this request.
 *
 * The caller's cookies are forwarded through the same helper every relay route
 * uses. Without them the probe reaches the backend anonymously, every visitor
 * looks signed out, and the gate bounces the entire deployment to a sign-in
 * page it will bounce them away from again - a total lockout, from one missing
 * header.
 *
 * Nothing here throws. Every way of not knowing collapses to a `null` status,
 * which passes.
 */
async function probeSession(request: NextRequest): Promise<ProbeAnswer> {
  let resp: Response;
  try {
    resp = await fetch(`${BACKEND_URL}/api/v1/auth/me`, {
      headers: forwardedHeaders(request.headers),
      cache: "no-store",
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
  } catch {
    return { status: null };
  }
  // Only a 401 has a code worth reading; anything else is decided by status
  // alone, and its body is released rather than parsed.
  if (resp.status !== 401) {
    await drain(resp);
    return { status: resp.status };
  }
  try {
    const problem = (await resp.json()) as { code?: unknown };
    return {
      status: 401,
      code: typeof problem.code === "string" ? problem.code : undefined,
    };
  } catch {
    return { status: 401 };
  }
}

/** Release a response body that is not going to be read. */
async function drain(resp: Response): Promise<void> {
  try {
    await resp.body?.cancel();
  } catch {
    // Already closed or errored; there is nothing left to release.
  }
}
