import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A declaration is written on demand and changes what later turns do; never cached.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for declaring a model's agency ceiling.
 *
 * Forwards `POST /api/v1/models/ceiling` with its JSON body verbatim. The
 * answer carries `earned` and `effective` beside the stored `ceiling`, because
 * raising a ceiling does not raise what a model may actually do - a deep check
 * still has to have proven it on the installed weights.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/ceiling`, {
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
