import type { NextRequest } from "next/server";

import { BACKEND_URL, backendUnreachable, proxyHeaders, relay } from "@/lib/backend-proxy";

// Removing weights is never cached, and the answer is live state either way.
export const dynamic = "force-dynamic";
export const runtime = "nodejs";

/** Route context for `/api/models/[...name]` (params are async). */
interface RouteParams {
  params: Promise<{ name: string[] }>;
}

/**
 * Proxy `DELETE /api/v1/models/{name}`: remove downloaded weights.
 *
 * A CATCH-ALL segment, not a single `[name]`: a model name can carry slashes
 * (`hf.co/org/repo`), and a single dynamic segment would simply not match those
 * rows - the ones most likely to be deleted. The segments are re-joined here
 * and re-encoded piece by piece, so a colon in a tag survives the round trip
 * while the slashes stay real path separators for the backend's `{name:path}`.
 *
 * The backend's refusals are relayed verbatim - 409 carries the sentence saying
 * what is still using the model, which is the whole answer a caller needs.
 */
export async function DELETE(
  request: NextRequest,
  { params }: RouteParams,
): Promise<Response> {
  const { name } = await params;
  const path = name.map(encodeURIComponent).join("/");
  let upstream: Response;
  try {
    // The query rides along whole: deleting the workspace default carries its
    // successor as `?replacement=`, which only the backend interprets.
    upstream = await fetch(`${BACKEND_URL}/api/v1/models/${path}${request.nextUrl.search}`, {
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
