import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Registry reads/writes must always hit the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SERVERS_URL = `${BACKEND_URL}/api/v1/integrations/servers`;

/**
 * Proxy `GET /api/v1/integrations/servers`: the tenant's registered MCP
 * servers (enabled and disabled), name-ordered.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(SERVERS_URL, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}

/**
 * Proxy `POST /api/v1/integrations/servers`: register a new server.
 *
 * The JSON body is forwarded verbatim; validation (URL scheme, no embedded
 * credentials) stays server-side.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(SERVERS_URL, {
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
