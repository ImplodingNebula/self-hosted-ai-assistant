import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A declaration is written on demand and changes what later turns do; never cached.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for confirming a measured think mode.
 *
 * Forwards `POST /api/v1/models/think-mode` with its JSON body verbatim. The
 * body is a model NAME and nothing else - the value stored is the one a tool
 * check measured - so a model nothing has measured comes back as a 422
 * problem+json saying to run a check first, in the backend's own wording.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/think-mode`, {
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
