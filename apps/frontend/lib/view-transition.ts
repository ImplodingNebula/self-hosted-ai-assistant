import { flushSync } from "react-dom";

import { observeView, readView, type View } from "@/lib/turn-state";

/**
 * Which choreography a view change plays (DESIGN.md, "What travels between
 * views"): the composer box between the thread and a new chat; the mark, the
 * headline and the nodes from the boot screen into the new chat; and the
 * shell alone, cross-fading in, when a quick load's boot was never painted
 * (its movers would otherwise fade in on their own clock, apart from the
 * rest of the page).
 */
export type TransitionKind = "composer" | "boot" | "shell" | "console";

type TransitionDocument = Document & {
  startViewTransition?: (update: () => void | Promise<void>) => {
    ready: Promise<void>;
    finished: Promise<void>;
  };
};

/* The transition in flight, so only the latest clears the marker: a second
   one started before the first ends would otherwise lose its names mid-way. */
let inFlight = 0;

/**
 * Apply a view change inside a view transition where the browser offers one
 * and motion is allowed, and at once otherwise.
 *
 * Resolves once `apply` has run, which inside a transition is a frame or so
 * later than the call: the browser snapshots the old view first. A caller
 * that goes on to act on the new state (a send that starts its request)
 * awaits it, so a fast answer can never land before the state it answers.
 *
 * The movers are named in the stylesheet under `html[data-vt="<kind>"]`, so
 * the names exist only while the transition runs: a named element is a
 * backdrop root, and the menu inside the composer would blur nothing if it
 * were named at rest. The same attribute holds back the entrance of whatever
 * the transition carries in, which would otherwise replay as a flicker the
 * moment the transition ended.
 *
 * @param kind - Which movers the stylesheet names.
 * @param apply - The state change; it is flushed synchronously so the new
 *   view is in the DOM when the transition takes its second snapshot.
 */
export function transitionView(
  kind: TransitionKind,
  apply: () => void,
): Promise<void> {
  const doc = document as TransitionDocument;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (typeof doc.startViewTransition !== "function" || reduce) {
    apply();
    return Promise.resolve();
  }
  const root = document.documentElement;
  const id = ++inFlight;
  root.dataset.vt = kind;
  const clear = () => {
    if (inFlight === id) delete root.dataset.vt;
  };
  return new Promise<void>((resolve) => {
    let transition: { ready: Promise<void>; finished: Promise<void> };
    try {
      transition = doc.startViewTransition(() => {
        flushSync(apply);
        resolve();
      });
    } catch {
      clear();
      apply();
      resolve();
      return;
    }
    /* a skipped transition (another started, the tab hidden) rejects ready;
       that is not a fault. A failing update is: it is logged, and the
       caller's state change is applied without the choreography. */
    transition.ready.catch(() => undefined);
    transition.finished
      .catch((e: unknown) => {
        console.error("view transition: the update failed", e);
        resolve();
      })
      .finally(clear);
  });
}

/**
 * Travel between routes inside a view transition: the chat into the console,
 * where the topbar and the side rail swap for the console's in place.
 *
 * A route change lands asynchronously, so the update the transition waits on
 * is the destination page naming its view on the body (`<body data-view>`,
 * which every page writes as it mounts) rather than a state change flushed
 * here. A destination that never names it within a moment (a route refused,
 * a slow payload) releases the transition anyway, so the page is never held
 * frozen behind a snapshot. Where the browser offers no transition, or motion
 * is reduced, the navigation simply runs.
 *
 * The way back is the same travel. A chat page carried in by a route
 * transition (`arrivingByRoute`) skips its boot screen and renders its shell
 * at once, running the boot's waits behind it for the reach line, so the
 * travel never waits on the backend: the thread's content fades in behind
 * the chrome that moved.
 *
 * @param kind - Which movers the stylesheet names.
 * @param view - The view the destination writes when it has arrived.
 * @param navigate - Starts the route change.
 */
export function transitionRoute(
  kind: TransitionKind,
  view: View,
  navigate: () => void,
): void {
  const doc = document as TransitionDocument;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  if (
    typeof doc.startViewTransition !== "function" ||
    reduce ||
    readView() === view
  ) {
    navigate();
    return;
  }
  const root = document.documentElement;
  const id = ++inFlight;
  root.dataset.vt = kind;
  /* the destination, so a choreography can differ by direction */
  root.dataset.vtTo = view;
  routeInFlight = id;
  const clear = () => {
    if (inFlight === id) {
      delete root.dataset.vt;
      delete root.dataset.vtTo;
    }
    if (routeInFlight === id) routeInFlight = 0;
  };
  let transition: { ready: Promise<void>; finished: Promise<void> };
  try {
    transition = doc.startViewTransition(
      () =>
        new Promise<void>((resolve) => {
          let timer = 0;
          const stop = observeView((now) => {
            if (now === view) settle();
          });
          const settle = () => {
            stop();
            window.clearTimeout(timer);
            resolve();
          };
          timer = window.setTimeout(settle, ROUTE_SETTLE_MS);
          try {
            navigate();
          } catch (e) {
            settle();
            throw e;
          }
        }),
    );
  } catch {
    clear();
    navigate();
    return;
  }
  transition.ready.catch(() => undefined);
  transition.finished
    .catch((e: unknown) => {
      console.error("view transition: the route change failed", e);
    })
    .finally(clear);
}

/** How long a route transition waits for its destination before letting go. */
const ROUTE_SETTLE_MS = 1500;

/* The route transition in flight, if any, so the page it carries in can tell. */
let routeInFlight = 0;

/**
 * Whether a route transition is carrying the current page in: the page's own
 * arrival choreography then applies plainly, inside this transition, rather
 * than starting a second one (which would skip the first mid-way).
 */
export function arrivingByRoute(): boolean {
  return routeInFlight !== 0;
}
