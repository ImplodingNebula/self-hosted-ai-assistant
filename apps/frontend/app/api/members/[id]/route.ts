import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A role change must always reach the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/members/[id]` (params are async). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `PATCH /api/v1/members/{id}`: set one member's role.
 *
 * The JSON body is forwarded verbatim, and so is the backend's refusal - the
 * 409 problem+json for the last administrator or a deployment-listed one
 * carries the sentence the page shows.
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/members/${encodeURIComponent(id)}`,
      {
        method: "PATCH",
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
