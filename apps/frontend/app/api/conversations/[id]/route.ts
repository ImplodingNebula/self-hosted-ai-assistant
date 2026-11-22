import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Conversation reads/writes must always hit the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/conversations/[id]` (params are async in the App Router). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/** Backend URL for one conversation id (encoded: the segment is client input). */
function conversationUrl(id: string): string {
  return `${BACKEND_URL}/api/v1/conversations/${encodeURIComponent(id)}`;
}

/**
 * Proxy `GET /api/v1/conversations/{id}`: full detail with transcript.
 *
 * Status, Content-Type, and body (including 404 problem+json) pass through
 * untouched.
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(conversationUrl(id), {
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
 * Proxy `PATCH /api/v1/conversations/{id}`: partial update (title/pin/settings).
 *
 * The JSON request body is forwarded verbatim; validation stays server-side.
 */
export async function PATCH(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(conversationUrl(id), {
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
 * Proxy `DELETE /api/v1/conversations/{id}`: 204 on success, messages cascade.
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(conversationUrl(id), {
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
