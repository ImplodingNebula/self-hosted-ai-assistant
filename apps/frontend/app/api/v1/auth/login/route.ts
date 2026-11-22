import type { NextRequest } from "next/server";

import {
  BACKEND_URL,
  backendUnreachable,
  proxyHeaders,
  relayRedirect,
} from "@/lib/backend-proxy";

// The outbound leg of a browser round trip; never cache, never prefetch.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Proxy `GET /api/v1/auth/login` - the hand-off to the identity provider.
 *
 * Sits under `/api/v1/` rather than the app's usual `/api/` prefix for the
 * same reason the integrations OAuth callback does: this is a path the BROWSER
 * navigates to, and the pair of it and the callback must share one public
 * origin so the redirect URI registered with the provider stays a single
 * value. The `?next=` query rides through untouched; the backend is the one
 * that decides whether a return path is acceptable, and duplicating that
 * judgement here would only create a second opinion to disagree with.
 *
 * `redirect: "manual"` is what makes this work at all. Followed server-side,
 * the 302 would be chased to the provider from inside this container and the
 * browser would receive the provider's sign-in HTML from the wrong origin,
 * with no way to complete the flow. Mirrored instead, the browser does the
 * travelling.
 *
 * Worth knowing which deployments actually run this code: the reverse proxy
 * sends `/api/v1/*` straight to the backend, so wherever it is in front (the
 * dev stack and production both) the browser never reaches this handler and
 * the backend answers the navigation itself. That is fine - the cookie is set
 * on the one shared public origin either way - and it matches the integrations
 * callback next door. It does mean this file is exercised only when the app is
 * run bare, without the proxy, which is exactly when it is needed.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/auth/login${request.nextUrl.search}`,
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
  // With sign-in switched off the backend refuses rather than redirecting;
  // relayRedirect passes that refusal through in the backend's own words.
  return relayRedirect(upstream);
}
