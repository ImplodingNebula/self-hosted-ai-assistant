import type { NextRequest } from "next/server";

import {
  BACKEND_URL,
  backendUnreachable,
  proxyHeaders,
  relay,
  relayStream,
} from "@/lib/backend-proxy";

// SSE responses must never be cached or statically prerendered.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

const SSE_HEADERS = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache, no-transform",
  "x-accel-buffering": "no", // ask proxies (Caddy/nginx) not to buffer the stream
} as const;

/**
 * Same-origin proxy to the backend's streaming tool check.
 *
 * Forwards the body to `POST /api/v1/models/verify` and pipes the
 * `text/event-stream` response straight back. Same shape as the download proxy
 * beside it: refusals decided before the stream (unusable name, a non-chat
 * model, weights the backend does not hold, a probe already running) relay as
 * problem+json, everything after that is an in-band `verify_error` frame.
 *
 * The stream stays open for a while AFTER the verdict frame: the check evicts
 * the chat model and the embedder to measure a fresh load and re-warms both
 * afterwards. That silence is the deployment being made usable again, so
 * nothing here times it out.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/verify`, {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": "application/json" }),
      body,
      cache: "no-store",
      // Forwarded like every other stream, so a closed tab does not leave a
      // probe holding the single process-wide slot.
      signal: request.signal,
    });
  } catch {
    if (request.signal.aborted) {
      // The client is gone; nobody will read a body. 499: client closed request.
      return new Response(null, { status: 499 });
    }
    return backendUnreachable();
  }

  if (!upstream.ok || !upstream.body) return relay(upstream);
  return relayStream(upstream, SSE_HEADERS);
}
