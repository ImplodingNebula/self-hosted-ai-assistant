import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Starts a consent (writes pending state); never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/integrations/servers/[id]/oauth/connect`. */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `POST /api/v1/integrations/servers/{id}/oauth/connect`: begin a
 * workspace OAuth consent and relay back the provider URL the browser opens.
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/integrations/servers/${encodeURIComponent(id)}/oauth/connect`,
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
