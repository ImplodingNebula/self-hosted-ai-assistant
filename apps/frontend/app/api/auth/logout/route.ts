import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Ending a session is a write; never cache it.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for ending the session.
 *
 * Forwards `POST /api/v1/auth/logout`. The backend deletes the stored session
 * and answers 204 with a cookie-clearing `Set-Cookie` - which is the entire
 * payload, since a 204 has no body. `relay` carries that header back; without
 * it the server-side session would be gone while the browser kept presenting
 * its now-dead cookie on every request.
 */
export async function POST(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/auth/logout`, {
      headers: proxyHeaders(request),
      method: "POST",
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
