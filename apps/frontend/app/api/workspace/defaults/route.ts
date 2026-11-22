import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Workspace defaults change what every later turn does; a cached read would
// show a stale answer to the page that just wrote it.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const UPSTREAM = `${BACKEND_URL}/api/v1/workspace/defaults`;

/**
 * Same-origin proxy for reading the workspace defaults.
 *
 * Forwards `GET /api/v1/workspace/defaults`, whose body carries the stored
 * values plus the deployment facts the page renders them against (whether a
 * reranker is wired, and the range the ceiling accepts).
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(UPSTREAM, { headers: proxyHeaders(request), cache: "no-store", signal: request.signal });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}

/**
 * Same-origin proxy for changing the workspace defaults.
 *
 * The JSON body is forwarded verbatim: the tri-state (a field absent means
 * "leave it", an explicit null means "clear it") is only expressible if this
 * layer neither fills in nor strips a key, and validation stays server-side
 * where the model-name rules and the ceiling bounds live.
 */
export async function PATCH(request: NextRequest): Promise<Response> {
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(UPSTREAM, {
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
