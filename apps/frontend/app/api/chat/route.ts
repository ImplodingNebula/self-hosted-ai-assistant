import type { NextRequest } from "next/server";

import { proxyHeaders, relayFrame, relayStream } from "@/lib/backend-proxy";
import { TURN_ACCEPTED_HEADER } from "@/lib/sse";

// SSE responses must never be cached or statically prerendered.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

// Server-side only: the backend base URL. In Docker the frontend container
// reaches the API over the internal network (`http://backend:8000`); in bare
// local dev it's the host-published port. The browser never sees this.
const BACKEND_URL = process.env.BACKEND_INTERNAL_URL ?? "http://localhost:8000";

const SSE_HEADERS = {
  "content-type": "text/event-stream; charset=utf-8",
  "cache-control": "no-cache, no-transform",
  "x-accel-buffering": "no", // ask proxies (Caddy/nginx) not to buffer the stream
} as const;


/**
 * Same-origin proxy to the backend's streaming chat endpoint.
 *
 * The browser POSTs here (no CORS); we forward the body to
 * `POST /api/v1/chat/stream` and pipe the `text/event-stream` response straight
 * back. Transport-level failures are surfaced as an in-band SSE `error` frame so
 * the client has a single stream-parsing path for both success and failure.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const body = await request.text();

  let upstream: Response;
  try {
    upstream = await fetch(`${BACKEND_URL}/api/v1/chat/stream`, {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": "application/json" }),
      body,
      cache: "no-store",
      // Forward the client's abort (Stop button, tab close) so the backend -
      // and ultimately the GPU - stops generating deterministically.
      signal: request.signal,
    });
  } catch {
    if (request.signal.aborted) {
      // The client is gone; nobody will read a body. 499: client closed request.
      return new Response(null, { status: 499 });
    }
    return sseError("The backend is unreachable.", "backend_unreachable");
  }

  if (!upstream.ok || !upstream.body) {
    // Pre-stream rejections arrive as RFC 7807 problem+json (unknown or
    // malformed conversation, stream cap). Forward detail and code verbatim
    // so the client can react (e.g. unbind a deleted conversation on
    // not_found) instead of showing raw truncated JSON.
    const raw = await upstream.text().catch(() => "");
    let message = `Backend responded ${upstream.status}.`;
    let code = "backend_error";
    try {
      const problem = JSON.parse(raw) as { detail?: string; code?: string };
      if (typeof problem.detail === "string" && problem.detail) message = problem.detail;
      if (typeof problem.code === "string" && problem.code) code = problem.code;
    } catch {
      if (raw) message += ` ${raw.slice(0, 200)}`;
    }
    return sseError(message, code, upstream);
  }

  return relayStream(upstream, SSE_HEADERS);
}

/**
 * Build a minimal `error` + `done` SSE stream carrying an in-band failure.
 *
 * Every caller is a turn that did not stream: the backend was unreachable, or it
 * answered before opening one. So this response, and only this one, carries the
 * not-accepted marker.
 *
 * The marker means "this turn produced no stream", NOT "nothing was persisted",
 * and the difference matters for the shapes that write before they generate. A
 * REFUSAL (4xx from the prepare phase) really did change nothing - that is what
 * a prepare-phase refusal is. But the prepare phase commits partway through the
 * request, and a failure AFTER that commit and before the first byte lands here
 * too, with the same marker. A client that treats the marker as "roll back what
 * I drew" is right for a plain send (a stray user row is the worst case) and
 * wrong for an edit, which has already rewritten a question by then; that path
 * asks the server what it holds rather than guessing (see runRewind).
 *
 * @param message - Human-readable failure text for the in-band `error` frame.
 * @param code - Problem+json code the client branches on.
 * @param upstream - The backend response being rewritten, when there was one.
 *   Its cookies still travel: a refusal is exactly where the backend clears a
 *   session it has just judged invalid, and swallowing that would leave the
 *   browser retrying with a cookie the server has already disowned.
 */
function sseError(message: string, code: string, upstream?: Response): Response {
  const frame =
    `event: error\ndata: ${JSON.stringify({ message, code })}\n\n` +
    `event: done\ndata: ${JSON.stringify({ message_id: "", conversation_id: null, usage: null })}\n\n`;
  return relayFrame(upstream, frame, {
    ...SSE_HEADERS,
    [TURN_ACCEPTED_HEADER]: "0",
  });
}
