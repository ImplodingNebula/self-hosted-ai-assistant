import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Credential state must always hit the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/integrations/servers/[id]/credential`. */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/** Backend URL for one server's credential (encoded: client input). */
function credentialUrl(id: string): string {
  return `${BACKEND_URL}/api/v1/integrations/servers/${encodeURIComponent(id)}/credential`;
}

/**
 * Proxy `GET .../credential`: connected + metadata hint (`last4`), never the
 * plaintext - the backend's write-only contract passes through untouched.
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(credentialUrl(id), {
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}

/** Proxy `PUT .../credential`: seal a credential (write-only; 503 if no key). */
export async function PUT(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  const body = await request.text();
  let upstream: Response;
  try {
    upstream = await fetch(credentialUrl(id), {
      method: "PUT",
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

/** Proxy `DELETE .../credential`: clear the credential. 204 on success. */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(credentialUrl(id), {
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
