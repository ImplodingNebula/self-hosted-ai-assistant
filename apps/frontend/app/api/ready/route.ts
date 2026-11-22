import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// A readiness answer is the state of the deployment this instant; a cached one
// would report a backend that has since come up as still down.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the readiness probe.
 *
 * Forwards `GET /readyz` (mounted at the backend's root, not under the API
 * version: it is about the process), returning the backend's status and JSON
 * body untouched. A 503 still carries a full report naming the dependency
 * that is down, which the boot screen reads; only a backend that cannot be
 * reached at all becomes the proxy's own 503.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/readyz`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
