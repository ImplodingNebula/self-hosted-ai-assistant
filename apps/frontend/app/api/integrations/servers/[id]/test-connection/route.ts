import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A live handshake against a saved server; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/integrations/servers/[id]/test-connection`. */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `POST /api/v1/integrations/servers/{id}/test-connection`: probe a saved
 * server with its stored credential (or a one-shot override in the body).
 *
 * Like the pre-save probe, the backend always answers 200 with an
 * `ok`/`error.code` result; an unreachable target is a diagnostic.
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
      `${BACKEND_URL}/api/v1/integrations/servers/${encodeURIComponent(id)}/test-connection`,
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
