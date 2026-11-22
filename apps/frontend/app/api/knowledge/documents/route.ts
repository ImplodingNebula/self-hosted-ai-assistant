import type { NextRequest } from "next/server";

import {
  BACKEND_URL,
  backendStillWorking,
  backendUnreachable,
  isHeadersTimeout,
  proxyHeaders,
  relay,
} from "@/lib/backend-proxy";

// Knowledge-base reads and writes must never be cached or statically prerendered.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/**
 * Backend URL for the knowledge documents collection, carrying the query through.
 *
 * The query string is forwarded verbatim rather than rebuilt: the delete's
 * `document_id` is an opaque source id (it may hold a colon, a slash, anything
 * the ingest that created it chose), and re-encoding it here is one more place
 * for the two ends to disagree about what it says. The backend validates it.
 */
function documentsUrl(request: NextRequest): string {
  const query = request.nextUrl.search;
  return `${BACKEND_URL}/api/v1/knowledge/documents${query}`;
}

/** Same-origin proxy for listing what the workspace knowledge base holds. */
export async function GET(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(documentsUrl(request), {
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
 * Same-origin proxy for adding a file to the knowledge base.
 *
 * The browser POSTs a `multipart/form-data` body; this forwards it with the
 * original Content-Type intact - the multipart boundary lives in that header -
 * and relays the backend's answer (or problem+json) back verbatim.
 *
 * The body is buffered rather than streamed, exactly as the attachment upload
 * proxy does it: the backend refuses an over-large body by Content-Length before
 * reading it, and buffering lets `fetch` set that header.
 *
 * This request is long-lived: the backend parses the file and embeds every
 * passage of it BEFORE answering, which on a CPU-tier embedder is seconds to
 * minutes. There IS a ceiling on that, and it is not one this route chose -
 * Node's fetch is undici, whose `headersTimeout` is five minutes and cannot be
 * raised per request. Every other long operation in the console is streamed, so
 * its headers arrive at once and the ceiling never applies; this is the one
 * proxied request that does all its work before the first byte.
 *
 * Past five minutes the fetch rejects, and what it must NOT become is
 * {@link backendUnreachable}: the request was delivered, and the backend does
 * not abort a non-streaming handler when its caller hangs up, so the ingest
 * runs to completion and the document appears moments later. Reporting that as
 * a failure would put an error over a file that was stored. It answers 504 with
 * that fact instead.
 */
export async function POST(request: NextRequest): Promise<Response> {
  const contentType =
    request.headers.get("content-type") ?? "application/octet-stream";
  const body = await request.arrayBuffer();

  let upstream: Response;
  try {
    upstream = await fetch(documentsUrl(request), {
      method: "POST",
      headers: proxyHeaders(request, { "content-type": contentType }),
      body,
      cache: "no-store",
      signal: request.signal,
    });
  } catch (error) {
    if (isHeadersTimeout(error)) {
      return backendStillWorking(
        "This file is taking longer than five minutes to index. It is still " +
          "being processed - reload this page in a few minutes and it should " +
          "be listed.",
      );
    }
    return backendUnreachable();
  }
  return relay(upstream);
}

/** Same-origin proxy for removing one source from the knowledge base. */
export async function DELETE(request: NextRequest): Promise<Response> {
  let upstream: Response;
  try {
    upstream = await fetch(documentsUrl(request), {
      method: "DELETE",
      headers: proxyHeaders(request),
      cache: "no-store",
      signal: request.signal,
    });
  } catch {
    return backendUnreachable();
  }
  return relay(upstream);
}
