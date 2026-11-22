import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// What a runtime holds in memory changes minute to minute; a cached answer
// would be worse than none.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the live residency read.
 *
 * Forwards `GET /api/v1/models/residency`: what the serving runtime is holding
 * in memory right now. The backend answers `reported: false` rather than an
 * error when the runtime does not report residency at all, so this route has
 * no failure of its own to translate beyond an unreachable backend.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/residency`, {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
