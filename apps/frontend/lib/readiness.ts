import type { ReadinessReport } from "@/lib/boot";

/**
 * Ask the backend whether it is ready, through the same-origin relay.
 *
 * A report arrives with a 200 (ready) or a 503 (a required service is down),
 * and both are answers: the boot screen names what is down and keeps
 * waiting. Only a response that carries no report - the relay's own 503 when
 * the backend could not be reached, or anything that is not JSON - is a
 * failure, thrown for the caller to retry.
 */
export async function getReadiness(signal?: AbortSignal): Promise<ReadinessReport> {
  const resp = await fetch("/api/ready", { cache: "no-store", signal });
  let body: unknown = null;
  try {
    body = await resp.json();
  } catch {
    body = null;
  }
  if (isReport(body)) return body;
  throw new Error(`readiness: no report (${resp.status})`);
}

function isReport(v: unknown): v is ReadinessReport {
  if (typeof v !== "object" || v === null) return false;
  const r = v as Partial<ReadinessReport>;
  return (r.status === "ready" || r.status === "unready") && Array.isArray(r.checks);
}
