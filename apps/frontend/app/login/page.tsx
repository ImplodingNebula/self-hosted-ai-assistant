import type { Metadata } from "next";
import { headers } from "next/headers";
import { redirect } from "next/navigation";

import { Icon } from "@/components/chat/step";
import { ViewMarker } from "@/components/view-marker";
import { safeNextPath, startLoginHref } from "@/lib/auth";
import { BACKEND_URL, forwardedHeaders } from "@/lib/backend-proxy";
import { appName } from "@/lib/branding";

export const metadata: Metadata = {
  title: "Sign in",
};

// The page is a function of its query string and of who is asking, so it is
// never prerendered.
export const dynamic = "force-dynamic";

/** Query the sign-in page reads: where to return once the round trip is done. */
interface LoginPageProps {
  /** Next 15+ delivers the parsed query as a promise. */
  searchParams: Promise<Record<string, string | string[] | undefined>>;
}

/**
 * Whether this deployment has sign-in at all.
 *
 * Asked server-side, before anything renders, because a deployment with auth
 * switched off is supposed to have no sign-in page to reach: someone arriving
 * on a bookmark or a stale link should land in the app, not on a button that
 * hands them the backend's raw refusal with no way back.
 *
 * An unreachable backend answers `true` deliberately. It is the one case where
 * nothing can be known, and of the two ways to be wrong, showing a sign-in page
 * that turns out to be unnecessary is far kinder than redirecting someone who
 * genuinely needs to sign in into an app that will only bounce them back.
 */
async function signInIsAvailable(): Promise<boolean> {
  let resp: Response;
  try {
    resp = await fetch(`${BACKEND_URL}/api/v1/auth/me`, {
      headers: forwardedHeaders(await headers()),
      cache: "no-store",
      // This probe sits in front of a page render, so a backend that hangs
      // would hang the page. The catch below already holds the safe answer.
      signal: AbortSignal.timeout(PROBE_TIMEOUT_MS),
    });
  } catch {
    return true;
  }
  // A backend that predates sign-in has no such route; there is nothing here
  // for anyone who reaches this page on such a deployment.
  if (resp.status === 404) {
    await drain(resp);
    return false;
  }
  // Anything else that failed is a live auth system saying "not you" - a 401
  // is precisely the state this page exists for.
  if (!resp.ok) {
    await drain(resp);
    return true;
  }
  try {
    const body = (await resp.json()) as { auth?: unknown };
    return body.auth !== "none";
  } catch {
    return true;
  }
}

/** How long to wait for the probe before rendering the page anyway. */
const PROBE_TIMEOUT_MS = 3000;

/** Release a response whose body is not going to be read. */
async function drain(resp: Response): Promise<void> {
  try {
    await resp.body?.cancel();
  } catch {
    // Already closed or errored; there is nothing left to release.
  }
}

/** The one value of a query key, when it was given once. */
function single(value: string | string[] | undefined): string | undefined {
  return Array.isArray(value) ? value[0] : value;
}

/**
 * The sign-in page.
 *
 * A server component with no client JavaScript of its own, which is not an
 * optimisation but the shape the flow requires: the button has to be a plain
 * link, because starting the round trip means a TOP-LEVEL navigation to the
 * identity provider's own domain - something `fetch` can neither follow nor
 * render. Reading the query on the server is what lets the href be correct in
 * the first paint rather than corrected after hydration. (The one client piece
 * is the view marker, which names the page to the sky.)
 *
 * Reached two ways, and the `next` query is what makes them the same page: a
 * person opening `/login` directly (no query, so they land on the app root
 * afterwards) and one bounced here mid-task by a lapsed session, whose
 * original destination was recorded on the way out and is handed back to the
 * backend to redirect to at the end of the flow. It is re-checked here anyway
 * - the value has been through a browser and is not this app's to trust.
 *
 * The second way says so: a `lapsed` flag, set only by the signed-out
 * reaction under an open page (a visitor who arrives signed out is bounced by
 * the proxy with the path alone), turns the headline into "Your session
 * ended" and the lede into the way back. The flag is read as a word, never as
 * evidence: it changes two sentences and nothing the flow does.
 *
 * Someone already signed in who navigates here deliberately still gets the
 * page: the flow is how an account is switched, so bouncing them away would
 * remove the only route to it.
 */
export default async function LoginPage({ searchParams }: LoginPageProps) {
  const params = await searchParams;
  const raw = single(params.next);
  const next = safeNextPath(raw);
  const lapsed = single(params.lapsed) === "1";

  // The redirect is gated on arriving WITHOUT a return path, and that gate is
  // a circuit breaker rather than a nicety. Two independent things decide
  // whether sign-in is on here - this probe, and the 401 the client reacts to
  // - and if they ever disagree the disagreement is a LOOP: the app bounces
  // someone here on a 401, this page reads "auth is off" and sends them back,
  // and around again with nothing on screen. A `next` is direct evidence that
  // something just refused a real request, which outranks a probe answering a
  // more general question, so the page stands its ground and renders.
  if (raw === undefined && !(await signInIsAvailable())) redirect("/");

  return (
    <main data-part="login">
      <ViewMarker view="login" state="login" />
      <div className="login-card">
        {/* The mark the boot draws and the sidebar wears, with its halo: the
            one thing that is the same on every screen before and after. */}
        <div className="login-mark" aria-hidden="true">
          <Icon name="logo" />
        </div>
        <div className="login-name">{appName()}</div>
        <h1 className="login-head">
          {lapsed ? (
            <>
              Your session <em>ended</em>
            </>
          ) : (
            <>
              Sign in to <em>continue</em>
            </>
          )}
        </h1>
        <p className="login-lede">
          {lapsed
            ? "Sign in again to get back to where you were. Nothing you sent was lost."
            : "Use a workspace account. You will be sent to your identity provider and brought back here."}
        </p>
        {/* An anchor, not a button: see the note above - this leaves the app. */}
        <a href={startLoginHref(next)} className="btn btn-primary login-btn">
          Sign in with Google
        </a>
        <p className="login-foot">
          {lapsed
            ? "Only the session ended."
            : "Conversations live in the workspace."}
        </p>
      </div>
    </main>
  );
}
