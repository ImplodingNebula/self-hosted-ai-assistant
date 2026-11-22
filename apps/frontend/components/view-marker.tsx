"use client";

import { useLayoutEffect } from "react";

import { clearView, setTurnState, setView, type TurnState, type View } from "@/lib/turn-state";

/**
 * Names the page's view on the body for as long as the page is mounted
 * (`<body data-view>`, lib/turn-state.ts), for pages that have no state of
 * their own to key it on. The chat page writes its own.
 *
 * A page with one standing sky state (the sign-in's rest) names it the same
 * way; the state returns to idle when the page leaves, so the next page starts
 * from the sky's rest rather than from this one's.
 */
export function ViewMarker({ view, state }: { view: View; state?: TurnState }) {
  useLayoutEffect(() => {
    setView(view);
    if (state !== undefined) setTurnState(state);
    return () => {
      clearView();
      if (state !== undefined) setTurnState("idle");
    };
  }, [view, state]);
  return null;
}
