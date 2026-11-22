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
 * Same-origin proxy to the backend's streaming model download.
 *
 * Forwards the body to `POST /api/v1/models/pull` and pipes the
 * `text/event-stream` response straight back, the same shape the chat proxy
 * uses - including the client's abort signal, so cancelling a download stops
 * the transfer rather than leaving it running headless.
 *
 * Unlike the chat proxy, a pre-stream refusal is relayed VERBATIM as
 * problem+json instead of being rewritten into an in-band `error` frame. The
 * backend decides every refusal (unusable name, a backend that cannot download,
 * a download of this tag already running, not an admin) before any work, so
 * that it arrives before a progress bar is attached to the stream; folding
 * those into the stream would throw that distinction away and make a refused
 * request look like a download that started and died.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/pull`, {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": "application/json" }),
      body,
      cache: "no-store",
      // Forward the client's abort (the Cancel button, a closed tab) so the
      // download stops deterministically. Ollama's transfer is
      // content-addressed, so a cancelled one resumes rather than restarts.
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
