import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Uploads must never be cached or statically prerendered.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Same-origin proxy for uploading an attachment.
 *
 * The browser POSTs a `multipart/form-data` body (the file plus its
 * `conversation_id` and `origin` fields); this forwards it to
 * `POST /api/v1/attachments` with the original Content-Type intact - the
 * multipart boundary lives in that header - and relays the backend's
 * `AttachmentOut` (or problem+json) back verbatim.
 *
 * The body is buffered here rather than streamed: the backend refuses an
 * over-large body by Content-Length before reading it, and buffering lets
 * `fetch` set that header. Uploads are already bounded by the size cap, so the
 * memory cost is small.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const contentType =
    request.headers.get("content-type") ?? "application/octet-stream";
  const body = await request.arrayBuffer();

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/attachments`, {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": contentType }),
      body,
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
