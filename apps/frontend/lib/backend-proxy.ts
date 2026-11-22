/**
 * Server-side helpers shared by the `/api/*` proxy route handlers.
 *
 * The browser talks only to same-origin Next routes; these helpers relay
 * requests to the backend over the internal network and pass responses back
 * verbatim, so problem+json error bodies reach the client untouched.
 */
import type { NextRequest } from "next/server";

// Server-side only: the backend base URL. In Docker the frontend container
// reaches the API over the internal network (`http://backend:8000`); in bare
// local dev it's the host-published port. The browser never sees this.
export const BACKEND_URL =
  process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

/**
 * Headers for a relayed backend call, carrying the caller's address forward.
 *
 * Every browser request reaches the backend from this server, not from the
 * person who made it, so without this the backend sees one client for the
 * whole deployment and its per-client rate limits become a single shared
 * budget. Passing `X-Forwarded-For` through restores the real caller.
 *
 * The header is forwarded VERBATIM and only when the incoming request already
 * carries one - nothing is ever synthesized here. That matters for trust: the
 * backend believes this chain only from a configured trusted proxy, and reads
 * it from the right, so the entry the ingress proxy appended (the real client)
 * wins and anything a caller supplied further left is inert. Inventing an
 * entry would put a value nobody vouched for into that position.
 *
 * With no proxy in front - ordinary local dev - there is no header to forward
 * and the backend falls back to keying on the address it sees. That is this
 * server for every request, which is correct for a single developer.
 *
 * The `cookie` header rides along for the same structural reason, and it is
 * what makes a signed-in session work at all: the browser holds the session
 * cookie against THIS origin, so a relay that does not carry it reaches the
 * backend anonymously and every authenticated call answers 401 while the
 * browser is holding a perfectly good session. Forwarded verbatim and only
 * when present, exactly like the address - nothing is ever synthesized.
 *
 * Sending the whole header rather than one named cookie is deliberate: this
 * process is not the authority on which cookies the backend uses, and picking
 * favourites here would quietly break the next one it adds. Both ends are the
 * same deployment behind one origin, so there is no third party to leak to.
 *
 * @param request - The incoming Next request being proxied.
 * @param extra - Headers the call needs of its own (e.g. `content-type`).
 * @returns Headers to pass to `fetch`.
 */
export function proxyHeaders(
  request: NextRequest,
  extra?: Record<string, string>,
): Record<string, string> {
  return { ...extra, ...forwardedHeaders(request.headers) };
}

/**
 * The two headers that must cross the hop, taken from any inbound header bag.
 *
 * Split out of {@link proxyHeaders} because a route handler is not the only
 * thing on this origin that calls the backend: a server component doing the
 * same has no `NextRequest` to hand, only what `next/headers` gives it, and
 * the alternative was for it to re-derive this by hand. Two copies of a rule
 * about which headers carry identity is one copy too many - the second is
 * where the cookie gets forgotten.
 *
 * @param source - Headers of the request being served.
 * @returns Just the headers worth forwarding; empty when there are none.
 */
export function forwardedHeaders(source: Headers): Record<string, string> {
  const headers: Record<string, string> = {};
  const forwardedFor = source.get("x-forwarded-for");
  if (forwardedFor) headers["x-forwarded-for"] = forwardedFor;
  const cookie = source.get("cookie");
  if (cookie) headers["cookie"] = cookie;
  return headers;
}

/**
 * Every `Set-Cookie` the backend sent, as separate values.
 *
 * `Set-Cookie` is the one header that must never be folded into a single
 * comma-joined line: its `Expires` attribute contains a comma of its own, so a
 * folded pair is not just ugly, it is unparseable, and the browser drops or
 * mangles it. `Headers.get` folds; `Headers.getSetCookie` is the API that does
 * not, which is why it exists at all.
 *
 * The fallback covers a runtime whose `Headers` predates that method. It is
 * lossy for a genuinely multi-value response and correct for a single cookie,
 * which is what this deployment sends - stated plainly rather than hidden,
 * since the honest single-cookie case is the one that has to keep working.
 *
 * @param upstream - The backend's response.
 * @returns One entry per cookie; empty when the response set none.
 */
export function setCookiesFrom(upstream: Response): string[] {
  const headers: Headers & { getSetCookie?: () => string[] } = upstream.headers;
  if (typeof headers.getSetCookie === "function") return headers.getSetCookie();
  const folded = headers.get("set-cookie");
  return folded === null ? [] : [folded];
}

/**
 * Copy the backend's `Set-Cookie` headers onto an outgoing response.
 *
 * Appends rather than sets, so several cookies survive as several headers.
 * Every relay leg calls this: a login that mints a session and a logout that
 * clears one both say so in this header alone, so a leg that drops it leaves
 * the browser exactly as signed-in (or signed-out) as it was before - with no
 * error anywhere to say why.
 *
 * @param upstream - The backend's response.
 * @param headers - The headers being built for the browser.
 */
export function applySetCookies(upstream: Response, headers: Headers): void {
  for (const cookie of setCookiesFrom(upstream)) {
    headers.append("set-cookie", cookie);
  }
}

/**
 * Relay a backend response to the browser unchanged.
 *
 * Status and body pass through as-is; Content-Type is forwarded so
 * `application/problem+json` errors keep their media type, and Set-Cookie so a
 * session the backend minted or cleared actually reaches the browser (a 204
 * logout carries nothing else at all). Hop-by-hop headers from the upstream
 * response must not be copied.
 *
 * @param upstream - The backend's response.
 * @param extra - Headers this route adds on its own account, e.g. a caching
 *   rule the proxy is the right place to state rather than the backend.
 * @returns A response safe to return from a route handler.
 */
export function relay(
  upstream: Response,
  extra?: Record<string, string>,
): Response {
  const headers = new Headers({ ...extra });
  const contentType = upstream.headers.get("content-type");
  if (contentType) headers.set("content-type", contentType);
  applySetCookies(upstream, headers);
  // 204/304 must not carry a body; the Response constructor enforces this.
  const body =
    upstream.status === 204 || upstream.status === 304 ? null : upstream.body;
  return new Response(body, { status: upstream.status, headers });
}

/**
 * Relay a streaming backend response, keeping the stream's own headers.
 *
 * The three SSE routes each need a fixed set of headers the backend does not
 * send (the event-stream media type, the no-transform cache rule, the hint
 * that asks an ingress proxy not to buffer), which is why they cannot simply
 * call {@link relay}. That left each of them hand-building a response, and a
 * hand-built response is one somebody has to remember to put the session
 * cookie back onto - three separate chances to forget, and forgetting is
 * invisible: the stream works perfectly while the sliding session quietly
 * stops sliding, so a user who only ever chats is signed out at the absolute
 * TTL with nothing anywhere to say why.
 *
 * One helper instead, so remembering is not a per-route act.
 *
 * The status is 200 rather than the upstream's, which is the whole contract of
 * a stream that has already begun: every caller checks `upstream.ok` and
 * `upstream.body` first and relays a refusal through {@link relay}, so by the
 * time this runs the only thing left to report is that bytes are coming.
 *
 * @param upstream - The backend's streaming response, already known to be a
 *   success with a body.
 * @param headers - The fixed headers this stream needs.
 * @returns A response safe to return from a route handler.
 */
export function relayStream(
  upstream: Response,
  headers: Record<string, string>,
): Response {
  return relayFrame(upstream, upstream.body, headers);
}

/**
 * Relay a body this proxy WROTE, still carrying the backend's cookies.
 *
 * The sibling of {@link relayStream} for the case where the bytes are the
 * proxy's own rather than the backend's: the chat route turns a pre-stream
 * refusal into an in-band event frame, so the body is synthesized here while
 * the headers that matter still belong to the response that refused.
 *
 * Its real job is to leave route handlers with nothing to construct. A route
 * that builds even one response of its own is a place the session cookie has
 * to be remembered, and a SECOND construction in the same file gives a
 * cookie-dropping one somewhere to hide - a check that merely finds an
 * `applySetCookies` nearby cannot tell the two apart. With this helper the
 * rule becomes flat and mechanically checkable: route files construct no
 * responses at all, except the abort for a caller who has already hung up.
 *
 * @param upstream - The backend's response, when there was one to take cookies
 *   from; absent when the request never reached it.
 * @param body - The bytes to send.
 * @param headers - The fixed headers this response needs.
 * @returns A response safe to return from a route handler.
 */
export function relayFrame(
  upstream: Response | undefined,
  body: BodyInit | null,
  headers: Record<string, string>,
): Response {
  const built = new Headers({ ...headers });
  if (upstream) applySetCookies(upstream, built);
  return new Response(body, { status: 200, headers: built });
}

/**
 * Relay a backend redirect to the BROWSER rather than following it here.
 *
 * The shape every OAuth/OIDC leg on this origin needs: the caller fetches with
 * `redirect: "manual"` so the 302 arrives as data, and this mirrors its status,
 * its `Location`, and any cookie it set. The cookie is the whole point on the
 * login return leg - the redirect and the freshly minted session travel in the
 * SAME response, so dropping either one leaves a round trip that looks like it
 * worked and signs nobody in.
 *
 * A response with no `Location` is not a designed path; it falls back to the
 * ordinary relay so the browser is shown the backend's own words instead of a
 * blank page.
 *
 * @param upstream - The backend's response, fetched with `redirect: "manual"`.
 * @returns A response safe to return from a route handler.
 */
export function relayRedirect(upstream: Response): Response {
  const location = upstream.headers.get("location");
  if (location === null) return relay(upstream);
  const headers = new Headers({ location });
  applySetCookies(upstream, headers);
  // The redirect's own body is dropped, so its stream has to be released by
  // hand or the socket stays held until garbage collection. Small, but these
  // two legs now run on every single sign-in.
  void upstream.body?.cancel().catch(() => {});
  return new Response(null, { status: upstream.status, headers });
}

/**
 * The response headers a file download is not a file download without.
 *
 * `relay` forwards only the content type, which is right for a JSON API and
 * wrong for bytes: without `Content-Disposition` the browser renders the file
 * instead of saving it and invents a name from the URL (every download would be
 * called "content"), and without `Content-Length` there is no progress and no
 * way to tell a truncated transfer from a complete one. The other two are the
 * backend's own decisions about bytes it does not vouch for, and dropping them
 * in the proxy would silently undo them.
 */
const DOWNLOAD_HEADERS = [
  "content-type",
  "content-disposition",
  "content-length",
  "cache-control",
  "x-content-type-options",
] as const;

/**
 * The allowlisted download headers of one upstream response.
 *
 * Deliberately an allowlist rather than a copy of everything: hop-by-hop
 * headers must not be forwarded, and a header the proxy did not think about is
 * safer dropped than passed on.
 *
 * @param upstream - The backend's response.
 * @param keepLength - Whether `Content-Length` may travel. False for a body
 *   this proxy is dropping or re-encoding.
 * @returns Headers to build the relayed response from.
 */
function downloadHeaders(upstream: Response, keepLength: boolean): Headers {
  const headers = new Headers();
  for (const name of DOWNLOAD_HEADERS) {
    if (name === "content-length" && !keepLength) continue;
    const value = upstream.headers.get(name);
    if (value) headers.set(name, value);
  }
  return headers;
}

/**
 * Relay a backend file response, keeping the headers that make it a download.
 *
 * @param upstream - The backend's response.
 * @returns A response safe to return from a route handler.
 */
export function relayDownload(upstream: Response): Response {
  // A compressed upstream response is DECODED by the time its body is readable
  // here while its headers are not, so the declared length would describe the
  // compressed bytes and the runtime would truncate the file at that count -
  // a silently corrupt download. Nothing on the path compresses today; the
  // header simply must not outlive the body it describes.
  const encoded = upstream.headers.get("content-encoding") !== null;
  const headers = downloadHeaders(upstream, !encoded);
  // Off the allowlist deliberately: Set-Cookie is not a header about the bytes,
  // it is the session sliding forward, and a download is an ordinary
  // authenticated request like any other.
  applySetCookies(upstream, headers);
  // 204/304 must not carry a body; the Response constructor enforces this.
  const body =
    upstream.status === 204 || upstream.status === 304 ? null : upstream.body;
  return new Response(body, { status: upstream.status, headers });
}

/**
 * Relay a backend response's headers alone - the answer to a HEAD.
 *
 * The same allowlist {@link relayDownload} uses, so the two describe a file
 * identically, but the body is dropped rather than piped. It has to be
 * explicit: a HEAD upstream still hands back a (empty) readable body, and
 * forwarding it would leave the runtime relaying a stream for a method whose
 * whole point is that there is nothing to relay.
 *
 * `Content-Length` is kept, and is the one header worth keeping: it is the
 * backend's statement about the file a GET would return, which is the question
 * a HEAD was asked. It describes no body being sent here, which is exactly what
 * HTTP means by a HEAD response.
 *
 * @param upstream - The backend's response to a HEAD.
 * @returns A bodyless response safe to return from a route handler.
 */
export function relayHeaders(upstream: Response): Response {
  return new Response(null, {
    status: upstream.status,
    headers: downloadHeaders(upstream, true),
  });
}

/**
 * Problem+json 503 for when the backend itself cannot be reached.
 *
 * Mirrors the backend's RFC 7807 shape so the client has a single error
 * format whether the failure happened in the API or in front of it.
 *
 * @returns A `503 application/problem+json` response.
 */
export function backendUnreachable(): Response {
  return new Response(
    JSON.stringify({
      type: "about:blank",
      title: "Service Unavailable",
      status: 503,
      detail: "The backend is unreachable.",
      code: "service_unavailable",
    }),
    { status: 503, headers: { "content-type": "application/problem+json" } },
  );
}

/**
 * The error code Node's fetch raises when a response's HEADERS never arrive.
 *
 * Global `fetch` is undici, whose `headersTimeout` defaults to five minutes and
 * cannot be raised per request. Every streamed route is immune - SSE headers
 * arrive at once and only the body is slow - so this bites exactly one shape:
 * a proxied request whose handler does all of its work BEFORE answering.
 */
const HEADERS_TIMEOUT_CODE = "UND_ERR_HEADERS_TIMEOUT";

/** Whether a thrown fetch error is the five-minute headers timeout above. */
export function isHeadersTimeout(error: unknown): boolean {
  // undici nests the real error under `cause`; the code can sit on either.
  const cause = (error as { cause?: unknown })?.cause;
  return (
    (error as { code?: string })?.code === HEADERS_TIMEOUT_CODE ||
    (cause as { code?: string })?.code === HEADERS_TIMEOUT_CODE
  );
}

/**
 * Problem+json 504 for a backend that is working but has not answered yet.
 *
 * Distinct from {@link backendUnreachable}, and the distinction is the whole
 * point: the request WAS delivered and the work is very likely still running,
 * so a client must not tell somebody their upload failed. Nothing was
 * cancelled - the backend does not abort a non-streaming handler when its
 * caller hangs up - so the honest sentence is "not finished", not "not done".
 *
 * @param detail - What is still running, in words worth showing.
 * @returns A `504 application/problem+json` response.
 */
export function backendStillWorking(detail: string): Response {
  return new Response(
    JSON.stringify({
      type: "about:blank",
      title: "Gateway Timeout",
      status: 504,
      detail,
      code: "backend_still_working",
    }),
    { status: 504, headers: { "content-type": "application/problem+json" } },
  );
}
