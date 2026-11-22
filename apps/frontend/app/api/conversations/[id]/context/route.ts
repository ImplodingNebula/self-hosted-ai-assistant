import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// The estimate reads live conversation rows; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/conversations/[id]/context` (params are async in the App Router). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `GET /api/v1/conversations/{id}/context`: the pre-turn window estimate.
 *
 * The optional `model` query parameter passes through untouched, as do the
 * status, Content-Type, and body (including 404 problem+json).
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const search = request.nextUrl.search;
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/conversations/${encodeURIComponent(id)}/context${search}`,
      { headers: proxyHeaders(request), cache: "no-store", signal: request.signal },
    );
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
