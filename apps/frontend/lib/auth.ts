/**
 * Sign-in state and the one place the app reacts to being signed out.
 *
 * Deliberately free of any dependency on the other `lib/` clients, including
 * the shared error helper - that one imports FROM here, so the direction has
 * to stay one-way. Nothing in this module throws: a probe that fails is an
 * answer ("we do not know who this is"), not an incident, and treating it as
 * one would put an error banner over a deployment that simply has sign-in
 * switched off.
 */

import { draftStore, forgetAllDrafts, stopKeepingDrafts } from "@/lib/drafts";

/** Where an unauthenticated visitor is sent. */
export const LOGIN_PATH = "/login";

/**
 * The problem+json code the backend answers with when a session is required.
 *
 * Reacting to this exact code rather than to a bare 401 is what keeps a
 * sign-in-disabled deployment behaving as it always has: with OIDC off the
 * backend never emits it, so nothing here ever fires.
 */
export const AUTH_REQUIRED_CODE = "auth_required";

/** Whether the deployment has sign-in switched on. */
export type AuthMode = "none" | "oidc";

/** What to do with a request whose sign-in state has just been probed. */
export type AuthGateDecision = "redirect" | "pass";

/**
 * Whether a probe's answer means the visitor has to sign in first.
 *
 * The rule stated once, so the two places that act on it cannot drift apart:
 * {@link handleAuthRequired} is the client's reaction and the server-side gate
 * in `proxy.ts` is the one that runs before a page is rendered at all. A
 * disagreement between them is not a cosmetic bug - it is a redirect LOOP,
 * with nothing on screen to explain it.
 *
 * `null` is the honest spelling of "no answer at all": an unreachable backend
 * or a probe that timed out. It PASSES, deliberately. An API that cannot be
 * reached must not put every page behind a sign-in page that could not work
 * either; the in-app unreachable banner is where that failure belongs.
 *
 * @param status - The status the probe answered with, or null if it did not.
 * @param code - Its problem+json `code`, when it had one.
 * @returns Whether to send the visitor to sign in.
 */
export function authGateDecision(
  status: number | null,
  code: string | undefined,
): AuthGateDecision {
  return status === 401 && code === AUTH_REQUIRED_CODE ? "redirect" : "pass";
}

/** A member's permission level within the workspace. */
export type UserRole = "admin" | "member";

/** Who the backend says is making requests, as reported by the probe. */
export interface AuthSession {
  /** `"oidc"` when a real sign-in happened; `"none"` when auth is disabled. */
  auth: AuthMode;
  /** The acting user's row id. */
  user_id: string;
  /** Human-readable name, from the provider's claims when there are any. */
  display_name: string;
  /** Permission level, read fresh from the user row on every request. */
  role: UserRole;
  /** Verified address; absent on a deployment with sign-in switched off. */
  email?: string | null;
}

/**
 * Reduce an arbitrary return path to one that is safe to navigate to.
 *
 * The value survives a round trip through an identity provider, so it is not
 * trustworthy by the time it comes back. Anything that is not a plain
 * same-origin path collapses to the app root: `https://elsewhere/` is an open
 * redirect outright, and so is `//elsewhere/` - a protocol-relative URL that
 * reads like a path and is not one.
 *
 * Control characters are refused before any of that, because a string holding
 * one is not the string it appears to be. Browsers STRIP tab, newline and
 * carriage return out of a URL before parsing it, so `/<tab>/elsewhere`
 * inspects as an ordinary two-segment path here and arrives at the navigation
 * as `//elsewhere` - the very case the checks below exist to catch, wearing a
 * disguise. Backslashes go the same way, being the separator some parsers
 * normalize into `/`.
 *
 * @param raw - A candidate path, typically from a query string.
 * @returns The path if it is a safe relative one, otherwise `"/"`.
 */
export function safeNextPath(raw: string | null | undefined): string {
  if (typeof raw !== "string" || raw === "") return "/";
  if (CONTROL_OR_BACKSLASH.test(raw)) return "/";
  if (!raw.startsWith("/")) return "/";
  if (raw.startsWith("//")) return "/";
  return raw;
}

// The C0 control range, DEL, and the backslash - see safeNextPath. Written as
// escapes and hoisted out of the function so the source file carries no literal
// control character of its own.
const CONTROL_OR_BACKSLASH = /[\u0000-\u001f\u007f\\]/;

/**
 * The URL of the sign-in page, remembering where the visitor was headed.
 *
 * @param next - The path to return to after signing in.
 * @param lapsed - Whether a session ended under an open page, as opposed to a
 *   visitor arriving signed out: the page says so, and names the way back.
 * @returns A path under {@link LOGIN_PATH}.
 */
export function loginHref(next: string, lapsed = false): string {
  const target = safeNextPath(next);
  const query = [
    ...(target === "/" ? [] : [`next=${encodeURIComponent(target)}`]),
    ...(lapsed ? ["lapsed=1"] : []),
  ];
  return query.length === 0 ? LOGIN_PATH : `${LOGIN_PATH}?${query.join("&")}`;
}

/**
 * The URL that starts the provider round trip.
 *
 * A plain link, because this has to be a TOP-LEVEL navigation: the response is
 * a redirect to the identity provider's own domain, which a `fetch` could
 * neither follow nor render.
 *
 * @param next - The path to return to after signing in.
 * @returns The same-origin relay path that begins the flow.
 */
export function startLoginHref(next: string): string {
  return `/api/v1/auth/login?next=${encodeURIComponent(safeNextPath(next))}`;
}

// One navigation per page life. Several requests can fail together - a page
// load fires the conversation list, the model listing and a context estimate
// at once - and each of them would otherwise call for its own redirect,
// stacking history entries and racing each other. The latch is never cleared
// because the navigation it guards replaces the page that holds it.
let redirecting = false;

/**
 * Send the visitor to the sign-in page, once, remembering where they were.
 *
 * A no-op on the server (there is no location to navigate) and on the sign-in
 * page itself, which would otherwise be able to redirect to itself forever.
 * The query string is carried but the fragment is not: a fragment never
 * reaches a server, and the return path has to survive a round trip through
 * the backend's own relative-path validation.
 *
 * A full document navigation rather than a client-side route push, and
 * `replace` rather than `assign`, for two reasons that both outlive this
 * function. A soft push would leave the whole signed-in React tree mounted -
 * the transcript, the conversation list, the model listing - behind a page
 * whose entire premise is that the session is gone. And a history entry for a
 * page that answered 401 is an entry the Back button can only bounce off.
 */
export function redirectToLogin(): void {
  if (typeof window === "undefined") return;
  if (redirecting) return;
  if (window.location.pathname === LOGIN_PATH) return;
  redirecting = true;
  // A 401 under an open page is a session that ended mid-task (a visitor who
  // arrives signed out is bounced by the proxy before any page loads), so the
  // sign-in page is told, and says so.
  window.location.replace(loginHref(`${window.location.pathname}${window.location.search}`, true));
}

/**
 * React to a failed response that turned out to be a missing session.
 *
 * The single REACTION behind both the REST and the streaming paths, so "signed
 * out" means the same thing however the app found out. The judgement itself is
 * {@link authGateDecision}, shared with the server-side gate.
 *
 * @param status - HTTP status of the failed response.
 * @param code - Its problem+json `code`, when it had one.
 * @returns Whether this was a signed-out failure (a redirect is now under way).
 */
export function handleAuthRequired(status: number, code: string | undefined): boolean {
  if (authGateDecision(status, code) === "pass") return false;
  redirectToLogin();
  return true;
}

/**
 * React to a stream frame that turned out to be a missing session.
 *
 * The chat proxy rewrites a pre-stream refusal into an in-band `error` frame,
 * so the status is gone by the time the client sees it and only the code is
 * left to go on. Same destination, one less piece of evidence.
 *
 * @param code - The `code` field of an SSE `error` payload.
 * @returns Whether this was a signed-out failure (a redirect is now under way).
 */
export function handleAuthRequiredFrame(code: string | undefined): boolean {
  if (code !== AUTH_REQUIRED_CODE) return false;
  redirectToLogin();
  return true;
}

/**
 * Ask the backend who is signed in.
 *
 * Answers `null` for every way of not knowing, and they are all ordinary: a
 * backend too old to have the endpoint (404), no session yet (401), an
 * unreachable API, or a body that is not the shape promised. The caller
 * renders exactly what it rendered before sign-in existed, which is the point
 * - a deployment with auth switched off must be untouched by all of this.
 *
 * One of those answers is more than an absence, though. A 401 carrying the
 * code is the plainest signed-out signal the app ever gets, so the probe obeys
 * the same rule every other call does rather than being the one place that
 * quietly swallows it - "every 401 with this code sends the visitor to sign
 * in" has to be true without exceptions, or it is not a rule.
 *
 * @returns The session, or null when it could not be established.
 */
export async function fetchAuthSession(): Promise<AuthSession | null> {
  let resp: Response;
  try {
    resp = await fetch("/api/auth/me", { cache: "no-store" });
  } catch {
    return null;
  }
  if (!resp.ok) {
    if (resp.status === 401) handleAuthRequired(401, await problemCode(resp));
    return null;
  }
  try {
    return asAuthSession(await resp.json());
  } catch {
    return null;
  }
}

/** The problem+json `code` of a failed response, if it carried one. */
async function problemCode(resp: Response): Promise<string | undefined> {
  try {
    const problem = (await resp.json()) as { code?: unknown };
    return typeof problem.code === "string" ? problem.code : undefined;
  } catch {
    return undefined;
  }
}

/**
 * Narrow an untyped body to a session, or reject it.
 *
 * Checked rather than cast because this decides whether a name is shown in the
 * sidebar: a half-populated body would render an identity chip that names
 * nobody, which is worse than showing none at all.
 */
function asAuthSession(body: unknown): AuthSession | null {
  if (typeof body !== "object" || body === null) return null;
  const raw = body as Record<string, unknown>;
  if (raw.auth !== "oidc" && raw.auth !== "none") return null;
  if (typeof raw.user_id !== "string" || raw.user_id === "") return null;
  const role = raw.role === "admin" ? "admin" : "member";
  // Trimmed to nothing counts as absent. `??` alone would not catch it, and an
  // empty string used as the fallback identity renders the nameless chip this
  // whole function exists to prevent.
  const address = typeof raw.email === "string" ? raw.email.trim() : "";
  const email = address !== "" ? address : null;
  // A provider is not obliged to return a name; the address is the fallback
  // identity, and the generic word is the last resort rather than a blank chip.
  const named = typeof raw.display_name === "string" ? raw.display_name.trim() : "";
  return {
    auth: raw.auth,
    user_id: raw.user_id,
    display_name: named !== "" ? named : (email ?? "Signed in"),
    role,
    email,
  };
}

/**
 * End the session and return to the sign-in page.
 *
 * The navigation happens whether or not the request succeeded. Someone who
 * clicked "sign out" has said what they want, and leaving them on a page that
 * still shows their name because the call failed reads as the app ignoring
 * them; the sign-in page is honest about the end state either way, and any
 * session that genuinely survived is still there to be found by the probe.
 *
 * `replace`, and a full document navigation, for the reason spelled out on
 * {@link redirectToLogin} - doubly so here, where the person has explicitly
 * asked to be signed out and Back must not walk them into the app shell they
 * just left, still populated with the previous session's conversations.
 */
export async function logout(): Promise<void> {
  const canNavigate = typeof window !== "undefined";
  // Claim the navigation BEFORE the request goes out. Any other call still in
  // flight can answer 401 while this one is waiting, and that redirect carries
  // a `next` back into the app - handing someone who explicitly asked to leave
  // a return ticket to where they just were.
  if (canNavigate) redirecting = true;
  // Unsent drafts live in this browser (lib/drafts.ts). Someone who signs
  // out, on a machine that may be shared, takes them along - before the
  // request, so a hung one cannot leave them behind. A session that merely
  // lapses never comes through here, and keeps its drafts for the sign-in
  // that follows.
  const store = draftStore();
  if (store !== null) forgetAllDrafts(store);
  stopKeepingDrafts();
  try {
    await fetch("/api/auth/logout", {
      method: "POST",
      cache: "no-store",
      // A backend that HANGS is the case worth arming against: an unreachable
      // one fails fast, but nothing else on this path has a deadline, so a
      // stalled request would leave the button reading "Signing out" while the
      // person is still, as far as the screen is concerned, signed in.
      signal: AbortSignal.timeout(LOGOUT_TIMEOUT_MS),
    });
  } catch {
    // Unreachable, refused, or timed out: nothing more to try, and the
    // destination is the same. The session, if it survived, is still findable.
  }
  if (canNavigate) window.location.replace(LOGIN_PATH);
}

/**
 * How long to wait for the server to confirm before leaving anyway.
 *
 * Exported so a check can assert the deadline is a real one. `signal` alone is
 * not evidence: an `AbortController().signal` that nobody ever fires satisfies
 * every structural test of "there is a signal here" and expires never, which
 * is the exact failure the deadline exists to prevent.
 */
export const LOGOUT_TIMEOUT_MS = 5000;
