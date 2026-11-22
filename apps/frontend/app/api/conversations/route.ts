import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Conversation lists must always reflect the latest writes; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for the conversation list endpoint.
 *
 * Forwards `GET /api/v1/conversations` with the `limit`/`offset` query string
 * passed through, returning the backend's status, Content-Type, and JSON body
 * untouched.
 */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/conversations${request.nextUrl.search}`,
      { headers: proxyHeaders(request), cache: "no-store", signal: request.signal },
    );
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}

/**
 * Same-origin proxy for creating an empty conversation.
 *
 * Forwards `POST /api/v1/conversations` (no body) so a client can mint a
 * conversation to attach a file to before its first chat turn exists.
 */
export async function POST(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/conversations`, {
      headers: proxyHeaders(request),
      method: "POST",
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
