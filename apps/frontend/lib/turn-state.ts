/**
 * The UI state machine's one public hook: `<body data-state>`.
 *
 * The stream handlers in the chat page know what the turn is doing; the sky,
 * the horizon on the topbar and every themed tell key on this attribute, in
 * CSS and in script, so the state is written to the DOM rather than kept in
 * React state and mirrored out. That also makes it drivable from outside the
 * app (a browser session, a review) without a global.
 */

import type { SkyState } from "@/lib/sky";

/**
 * The states the pages write: the turn's and the boot's waits (the chat page),
 * a download or a check (the console: amber while the runtime fetches, the
 * tool's teal while a model is worked), and the sign-in's rest.
 */
export type TurnState = Extract<
  SkyState,
  | "idle"
  | "waiting"
  | "loading"
  | "thinking"
  | "tool"
  | "confirm"
  | "streaming"
  | "error"
  | "done"
  | "boot"
  | "boot-model"
  | "boot-data"
  | "boot-ready"
  | "boot-slow"
  | "download"
  | "login"
>;

/**
 * Which view the window shows: the boot screen, a new chat (the greeting and
 * the composer, no thread), an open thread, the console, or the sign-in.
 * `<body data-view>` carries it for the stylesheets (the new chat's layout and
 * veil) and for the sky (the chart's place and its handoff). A page writes its
 * view on mount and clears it on unmount, so one page's view never stands over
 * another's.
 */
export type View = "boot" | "new" | "thread" | "console" | "login";

/** Set the view on the document body (there is none during server rendering). */
export function setView(view: View): void {
  if (typeof document === "undefined") return;
  if (document.body.dataset.view !== view) document.body.dataset.view = view;
}

/** Take the view off the body: the page that wrote it is leaving. */
export function clearView(): void {
  if (typeof document === "undefined") return;
  delete document.body.dataset.view;
}

/** The current view, or the thread when none has been written yet. */
export function readView(): View {
  if (typeof document === "undefined") return "thread";
  const v = document.body.dataset.view;
  return v === "boot" || v === "new" || v === "console" || v === "login" ? v : "thread";
}

/** Watch the view. The callback fires on every change; the return value stops watching. */
export function observeView(onChange: (view: View) => void): () => void {
  const mo = new MutationObserver(() => onChange(readView()));
  mo.observe(document.body, { attributes: true, attributeFilter: ["data-view"] });
  return () => mo.disconnect();
}

/** The event the boot fires as it hands off to the new chat, so the chart moves house. */
export const SKY_HANDOFF_EVENT = "sky:handoff";

/** Tell the sky the boot screen is handing off to the new chat. */
export function signalSkyHandoff(): void {
  if (typeof document === "undefined") return;
  document.dispatchEvent(new Event(SKY_HANDOFF_EVENT));
}

/** Set the turn state on the document body (there is none during server rendering). */
export function setTurnState(state: TurnState): void {
  if (typeof document === "undefined") return;
  if (document.body.dataset.state !== state) document.body.dataset.state = state;
}

/** The current turn state, or idle when none has been written yet. */
export function readTurnState(): string {
  if (typeof document === "undefined") return "idle";
  return document.body.dataset.state ?? "idle";
}

/**
 * Watch the turn state. The callback fires on every change (not on
 * subscription); the return value stops watching.
 */
export function observeTurnState(onChange: (state: string) => void): () => void {
  const mo = new MutationObserver(() => onChange(readTurnState()));
  mo.observe(document.body, { attributes: true, attributeFilter: ["data-state"] });
  return () => mo.disconnect();
}

/** The event the jump to latest fires so the sky can streak its stars. */
export const SKY_JUMP_EVENT = "sky:jump";

/** Tell the sky the reader just jumped to the end of the thread. */
export function signalSkyJump(): void {
  if (typeof document === "undefined") return;
  document.dispatchEvent(new Event(SKY_JUMP_EVENT));
}
