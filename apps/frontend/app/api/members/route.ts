import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Who is in the workspace and what role they hold is live state that a
// concurrent change can move; a cached copy would show a role nobody holds.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the members roster.
 *
 * Forwards `GET /api/v1/members`, returning the backend's status, Content-Type
 * and JSON body untouched - including the 403 problem+json a member gets, so
 * the page can say the section is not theirs rather than render an empty list.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/members`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
