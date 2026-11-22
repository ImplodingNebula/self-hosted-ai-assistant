import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Who is signed in is live per-request state; a cached answer would show one
// person's name to the next.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the signed-in probe.
 *
 * Forwards `GET /api/v1/auth/me`, returning the backend's status and body
 * untouched. All three answers matter to the client and none of them is an
 * error to be smoothed over: sign-in switched off, a valid session, or a 401
 * saying one is needed.
 *
 * The caching rules are stated here rather than left to the framework, because
 * this is the one response on the origin whose body names a specific PERSON.
 * `relay` forwards only the content type, so a `no-store` the backend sent
 * would be dropped on the way through; without it, any shared cache anywhere
 * on the path is free to keep one member's id, address and role and hand them
 * to the next caller. `vary: cookie` says the same thing in the other
 * direction - the answer depends entirely on who asked.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/auth/me`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream, { "cache-control": "no-store", vary: "cookie" });
}
