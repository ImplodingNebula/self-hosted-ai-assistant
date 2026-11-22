import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Install flags are live provider state; a cached list would offer a model
// that is no longer pulled (or hide one that just was).
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the model listing endpoint.
 *
 * Forwards `GET /api/v1/models`, returning the backend's status, Content-Type,
 * and JSON body untouched - including the 503 problem+json the backend raises
 * when the model server cannot be reached, so the composer can say the list is
 * unavailable rather than silently showing an empty one.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
