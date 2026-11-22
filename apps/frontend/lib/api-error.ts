/**
 * Shared error type and response helper for the browser-side API clients.
 *
 * Every same-origin `/api/*` call funnels its non-2xx responses through
 * {@link raise}, so callers get one error shape (status + problem+json code)
 * whether the failure came from the backend or the proxy in front of it.
 *
 * Being that funnel is also why the signed-out reaction lives here rather than
 * at ~30 call sites: there is exactly one place every REST failure passes
 * through, so there is exactly one place that has to recognise a lapsed
 * session. The one path that does NOT come through here is the chat stream,
 * whose refusals are rewritten into in-band frames; it calls the same handler
 * from the frame it reads.
 */

import { handleAuthRequired } from "@/lib/auth";
import type { ProblemDetail } from "@/lib/types";

/** Error carrying the HTTP status and problem+json code of a failed API call. */
export class ApiError extends Error {
  /** HTTP status code of the failed response. */
  readonly status: number;
  /** Machine-readable problem+json code, when the backend provided one. */
  readonly code?: string;

  /**
   * Build an error from a failed response.
   *
   * @param message - Human-readable detail, shown to the user as-is.
   * @param status - HTTP status code of the response.
   * @param code - Problem+json `code` field, if present.
   */
  constructor(message: string, status: number, code?: string) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
  }
}

/** Throw an {@link ApiError} built from a non-2xx response's problem body. */
export async function raise(resp: Response): Promise<never> {
  let detail = `The server responded with HTTP ${resp.status}.`;
  let code: string | undefined;
  try {
    const problem = (await resp.json()) as ProblemDetail;
    detail = problem.detail ?? problem.title ?? detail;
    code = problem.code;
  } catch {
    // Non-JSON body (proxy hiccup): keep the generic status message.
  }
  // A session that has lapsed sends the visitor to sign in again. It still
  // throws afterwards: a navigation is not instant, and every caller's own
  // failure path has to keep running so nothing is left mid-flight in the
  // meantime. The banner it raises lives for the moment before the page goes.
  handleAuthRequired(resp.status, code);
  throw new ApiError(detail, resp.status, code);
}
