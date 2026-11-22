"use client";

import { useState } from "react";

import { type AuthSession, logout } from "@/lib/auth";

/** Props for the sidebar-footer {@link UserChip}. */
export interface UserChipProps {
  /** The signed-in session; the caller renders nothing without one. */
  session: AuthSession;
}

/**
 * Who is signed in, and the way out - the sidebar footer's bottom row.
 *
 * Two lines rather than a hover tooltip: the address is what actually
 * distinguishes two people with the same display name, and it is the thing
 * someone checks when they are unsure WHICH account they are on. Hiding that
 * behind a hover would make the check impossible on a touch screen and
 * awkward everywhere else. Both lines truncate rather than wrap, so a long
 * address cannot push the panel's width around.
 *
 * The address is omitted rather than faked when the session has none - which
 * only happens on a deployment with sign-in switched off, where this component
 * is never rendered at all.
 */
export function UserChip({ session }: UserChipProps) {
  // Sign-out ends in a full page navigation, so this only has to survive until
  // the document goes; its job is to stop a second click starting a second
  // request and to say the first one landed.
  const [leaving, setLeaving] = useState(false);
  const secondary = session.email ?? null;

  return (
    <div data-part="user-chip" className="user-chip">
      <span className="user-chip-id">
        <span className="user-chip-name" title={session.display_name}>
          {session.display_name}
        </span>
        {secondary !== null && secondary !== session.display_name && (
          <span className="user-chip-mail" title={secondary}>
            {secondary}
          </span>
        )}
      </span>
      <button
        type="button"
        onClick={() => {
          setLeaving(true);
          void logout();
        }}
        disabled={leaving}
        className="user-chip-out"
      >
        {leaving ? "Signing out" : "Sign out"}
      </button>
    </div>
  );
}
