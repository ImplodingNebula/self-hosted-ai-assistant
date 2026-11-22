import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A live handshake against the target; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Proxy `POST /api/v1/integrations/servers/test-connection`: probe a server
 * with an initialize + tools/list handshake (never a tool call).
 *
 * The backend always answers 200 with an `ok`/`error.code` result; an
 * unreachable target is a diagnostic, not an HTTP failure.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/integrations/servers/test-connection`,
      {
        method: "POST",
        headers: proxyHeaders(request, { "content-type": "application/json" }),
        body,
        cache: "no-store",
        signal: request.signal,
      },
    );
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
