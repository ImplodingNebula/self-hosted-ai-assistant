import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A declaration is written on demand and changes what later turns do; never cached.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for declaring whether a model may reason and call tools at
 * once.
 *
 * Forwards `POST /api/v1/models/thinking-with-tools` with its JSON body
 * verbatim. The answer carries `effective` beside the stored claim, because
 * granting one does not grant the capability - a deep check still has to have
 * passed the reasoning case on the installed weights.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/thinking-with-tools`, {
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
