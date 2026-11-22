import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Written on demand and read by the very next turn; never cached.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for setting the workspace default chat model.
 *
 * Forwards `POST /api/v1/models/default` with its JSON body verbatim. A tag the
 * server does not hold, or one serving a role that cannot drive a chat turn,
 * comes back as problem+json rather than being stored for a later turn to
 * stumble over.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/default`, {
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
