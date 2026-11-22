import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Status polls must always hit the backend; never cache.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/attachments/[id]` (params are async in the App Router). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/** Backend URL for one attachment id (encoded: the segment is client input). */
function attachmentUrl(id: string): string {
  return `${BACKEND_URL}/api/v1/attachments/${encodeURIComponent(id)}`;
}

/**
 * Proxy `GET /api/v1/attachments/{id}`: one attachment, for status polling.
 *
 * Status, Content-Type, and body (including 404 problem+json) pass through
 * untouched, so the client sees the same `AttachmentOut` shape a poll returns.
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(attachmentUrl(id), {
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
 * Proxy `DELETE /api/v1/attachments/{id}`: remove the file for good.
 *
 * 204 on success. The backend deletes the file's indexed text, its stored bytes
 * and its row together, so a removed file stops being retrievable rather than
 * merely disappearing from the composer.
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(attachmentUrl(id), {
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
