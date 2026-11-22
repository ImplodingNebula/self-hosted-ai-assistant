import type { NextRequest } from "next/server";

import {
  BACKEND_URL,
  backendUnreachable,
  proxyHeaders,
  relayRedirect,
} from "@/lib/backend-proxy";

// The provider's browser return leg; never cache, never prefetch.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Proxy `GET /api/v1/integrations/oauth/callback` - the OAuth return leg.
 *
 * This path exists on the FRONTEND origin because the redirect URI registered
 * with providers is `OAUTH_REDIRECT_BASE_URL` (the frontend origin in dev) plus
 * this exact path: one origin serves the whole round trip, and the production
 * proxy keeps the same shape. The backend answers a 302 with no query string;
 * that redirect must reach the BROWSER, not be followed server-side here - so
 * the relay mirrors the status and Location verbatim.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/integrations/oauth/callback${request.nextUrl.search}`,
      {
        headers: proxyHeaders(request),
        method: "GET",
        cache: "no-store",
        redirect: "manual",
        signal: request.signal,
      },
    );
  } catch {
    return backendUnreachable();
  }
  // Shared with the OIDC login legs, which need the identical shape: mirror the
  // status and Location, carry any cookie, and fall back to an ordinary relay
  // for a response that is not the designed bodyless redirect.
  return relayRedirect(upstream);
}
