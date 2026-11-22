import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Deployment config, but never cache: an env change alters it on restart.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Proxy `GET /api/v1/integrations/oauth/redirect-uri`: the literal callback URL. */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/integrations/oauth/redirect-uri`,
      {
        headers: proxyHeaders(request),
        method: "GET",
        cache: "no-store",
        signal: request.signal,
      },
    );
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
