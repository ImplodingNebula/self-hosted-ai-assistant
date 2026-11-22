import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A declaration is written on demand and changes what later turns do; never cached.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for pairing a model with its reasoning partner.
 *
 * Forwards `POST /api/v1/models/thinking-pair` with its JSON body verbatim. A
 * partner whose think mode does not already resolve to `always` or `hybrid`
 * comes back as a 422 problem+json in the backend's own wording.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/thinking-pair`, {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": "application/json" }),
      body,
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
