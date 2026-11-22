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
 * Proxy `GET /api/v1/auth/callback` - the identity provider's return leg.
 *
 * This path exists on the FRONTEND origin because the redirect URI configured
 * for the provider is the public origin plus this exact path - the same one
 * origin that serves the consent flow next door. The backend answers a 302
 * carrying the freshly minted session cookie, and BOTH halves have to reach
 * the browser: the cookie without the redirect strands the user on a blank
 * page, and the redirect without the cookie lands them back on the sign-in
 * page having apparently done nothing wrong.
 *
 * A refused sign-in (an unverified address, one outside the admitted domains)
 * is not a redirect at all but a plain readable response. relayRedirect falls
 * back to an ordinary relay for it, so the person reads the backend's own
 * explanation rather than an empty screen.
 *
 * As with the login leg beside it, the reverse proxy sends `/api/v1/*` to the
 * backend directly, so this handler runs only when the app is served bare
 * without that proxy. The registered redirect URI is the same either way,
 * which is the property that has to hold.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/auth/callback${request.nextUrl.search}`,
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
  return relayRedirect(upstream);
}
