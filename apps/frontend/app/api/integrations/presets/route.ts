import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Static curated metadata, but never cache: a backend upgrade changes it.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Proxy `GET /api/v1/integrations/presets`: the curated preset catalog. */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/integrations/presets`, {
      headers: proxyHeaders(request),
      method: "GET",
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
