"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import type { ModelWhereabouts } from "@/components/chat/model-status";
import { getResidency } from "@/lib/model-manager";

/** How often the runtime is asked what it holds, while the page is visible. */
const POLL_MS = 20_000;

/**
 * A tag as the runtime names it: `mistral` and `mistral:latest` are one model
 * (the backend's identity rules say so), and the residency report carries the
 * runtime's spelling, so both sides are compared through this.
 */
function sameTag(tag: string): string {
  return tag.trim().toLowerCase().replace(/:latest$/, "");
}

/**
 * Where a model is right now: in memory, on disk, or unknown.
 *
 * Reads the runtime's residency report on a slow poll while the page is
 * visible, and on demand (`refresh`) when a turn ends - the one moment the
 * answer is likely to have changed. A runtime that does not report residency
 * leaves the answer `unknown`, which the status shows as the name alone
 * rather than a guessed state; so does a backend that cannot be reached.
 *
 * @param tag - The model to look for, or null when the page cannot name one.
 */
export function useModelWhereabouts(tag: string | null): {
  state: ModelWhereabouts;
  refresh: () => void;
} {
  const [resident, setResident] = useState<ReadonlySet<string> | null>(null);
  const inFlight = useRef(false);
  const again = useRef(false);

  // A refresh asked for while one is in flight runs once more after it: the
  // post-turn read is the one that matters, and dropping it would leave the
  // pre-turn answer standing for a whole poll.
  const refresh = useCallback(function run() {
    if (inFlight.current) {
      again.current = true;
      return;
    }
    inFlight.current = true;
    void getResidency()
      .then((view) => {
        setResident(view.reported ? new Set(view.models.map((m) => sameTag(m.model))) : null);
      })
      .catch(() => setResident(null))
      .finally(() => {
        inFlight.current = false;
        if (again.current) {
          again.current = false;
          run();
        }
      });
  }, []);

  useEffect(() => {
    refresh();
    let timer: number | null = null;
    const start = () => {
      if (timer === null) timer = window.setInterval(refresh, POLL_MS);
    };
    const stop = () => {
      if (timer !== null) {
        window.clearInterval(timer);
        timer = null;
      }
    };
    const onVisibility = () => {
      if (document.visibilityState === "visible") {
        refresh();
        start();
      } else stop();
    };
    if (document.visibilityState === "visible") start();
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      stop();
      document.removeEventListener("visibilitychange", onVisibility);
    };
  }, [refresh]);

  let state: ModelWhereabouts = "unknown";
  if (tag !== null && resident !== null) state = resident.has(sameTag(tag)) ? "memory" : "disk";
  return { state, refresh };
}
