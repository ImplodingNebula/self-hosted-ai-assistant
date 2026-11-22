import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// The whole view is live state (what is installed, what a probe measured, what
// is still using a model); a cached copy would offer a Delete the API refuses.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the model management view.
 *
 * Forwards `GET /api/v1/models/manager`, returning the backend's status,
 * Content-Type and JSON body untouched - including the 503 problem+json raised
 * when the model server cannot be reached, so the page can say the view is
 * unavailable rather than render an empty inventory as an empty machine.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/manager`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
