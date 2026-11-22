import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Registry reads/writes must always hit the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/integrations/servers/[id]` (params are async). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/** Backend URL for one server id (encoded: the segment is client input). */
function serverUrl(id: string): string {
  return `${BACKEND_URL}/api/v1/integrations/servers/${encodeURIComponent(id)}`;
}

/** Proxy `GET /api/v1/integrations/servers/{id}`: one server's metadata. */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(serverUrl(id), {
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
 * Proxy `PATCH /api/v1/integrations/servers/{id}`: enable/disable, re-point,
 * rescope, or edit config. The JSON body is forwarded verbatim.
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(serverUrl(id), {
      method: "PATCH",
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

/**
 * Proxy `DELETE /api/v1/integrations/servers/{id}`: remove the server and its
 * shared workspace credential. 204 on success.
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(serverUrl(id), {
      headers: proxyHeaders(request),
      method: "DELETE",
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
