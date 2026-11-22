import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Revokes and deletes a grant; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/integrations/servers/[id]/oauth/disconnect`. */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `POST /api/v1/integrations/servers/{id}/oauth/disconnect`: revoke the
 * server's grant (best effort upstream) and delete the sealed bundle.
 */
export async function POST(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/integrations/servers/${encodeURIComponent(id)}/oauth/disconnect`,
      { headers: proxyHeaders(request), method: "POST", cache: "no-store", signal: request.signal },
    );
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
