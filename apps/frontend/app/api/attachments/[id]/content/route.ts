import type { NextRequest } from "next/server";

import {
  BACKEND_URL,
  backendUnreachable,
  proxyHeaders,
  relayDownload,
  relayHeaders,
} from "@/lib/backend-proxy";

// A file's bytes are one tenant's private data behind a shared path; never
// cached by the framework, always fetched.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/attachments/[id]/content` (params are async in the App Router). */
interface RouteParams {
  params: Promise<{ id: string }>;
}

/**
 * Proxy `GET /api/v1/attachments/{id}/content`: the file itself.
 *
 * Uses {@link relayDownload} rather than the ordinary relay: that one forwards
 * the content type alone, which would strip the `Content-Disposition` naming
 * the file and the `Content-Length` a browser shows progress against - turning
 * a download into a rendered page called "content".
 *
 * Error responses (410 for an expired file, 409 while one is still being
 * processed, 404 for one that is gone) pass through with their problem+json
 * body intact, so the client can say which of those happened.
 *
 * Carries {@link proxyHeaders} like every other relayed call, and this route
 * needs it most: the backend's per-client rate limit and its audit trail both
 * key on the caller, and without the forwarded address every download in the
 * deployment would spend one shared budget attributed to this server - on the
 * one route that hands out file bytes.
 */
export async function GET(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/attachments/${encodeURIComponent(id)}/content`,
      { headers: proxyHeaders(request), cache: "no-store", signal: request.signal },
    );
  } catch {
    return backendUnreachable();
  }
  return relayDownload(upstream);
}

/**
 * Proxy `HEAD /api/v1/attachments/{id}/content`: can this file be had?
 *
 * The same statuses the GET answers with, reached without either process
 * touching the bytes.
 *
 * This app's own download path does NOT use it: {@link downloadAttachment}
 * fetches the file and reads the outcome of that one request, which is what
 * keeps a refusal from being saved to disk as the file. The method is here
 * because "is this still fetchable" is a fair question to be able to ask
 * without paying for a copy of the answer - the backend classifies it as
 * ordinary traffic for exactly that reason - and because a route that answers
 * GET should not 405 the cheap half of it.
 *
 * Forwarded as a HEAD rather than turned into a GET whose body is discarded:
 * the whole saving is upstream, in the object never being read.
 */
export async function HEAD(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { id } = await params;
  let upstream: Response;
  try {
    upstream = await fetch(
      `${BACKEND_URL}/api/v1/attachments/${encodeURIComponent(id)}/content`,
      {
        method: "HEAD",
        headers: proxyHeaders(request),
        cache: "no-store",
        signal: request.signal,
      },
    );
  } catch {
    return backendUnreachable();
  }
  return relayHeaders(upstream);
}
