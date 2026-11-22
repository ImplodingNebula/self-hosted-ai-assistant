"use client";

import { useEffect, useState } from "react";

import { type AuthSession, fetchAuthSession } from "@/lib/auth";

/**
 * Ask once per page load who is signed in.
 *
 * Null until the answer lands, and null forever if it never does - the two are
 * deliberately the same value, because the UI's response to both is identical:
 * render what it rendered before sign-in existed. That is the whole promise of
 * a deployment with auth switched off, and making "not yet" distinguishable
 * from "not applicable" would only invite a loading state onto a page that
 * should never flicker.
 *
 * No refresh loop and no revalidation: a session that lapses announces itself
 * on the next request the app makes, through the shared 401 handler, which is
 * both sooner and cheaper than polling for it.
 *
 * @returns The session, or null while unknown.
 */
export function useAuthSession(): AuthSession | null {
  const [session, setSession] = useState<AuthSession | null>(null);

  useEffect(() => {
    let live = true;
    // fetchAuthSession never rejects, so there is no failure branch to write.
    void fetchAuthSession().then((result) => {
      if (live) setSession(result);
    });
    return () => {
      live = false;
    };
  }, []);

  return session;
}
