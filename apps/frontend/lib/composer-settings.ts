/**
 * The composer's three per-conversation pins, and how they read and write.
 *
 * A pin is what the conversation stores for one axis: a chat model tag, a
 * thinking mode, a web-search mode. `null` means UNPINNED - the axis falls
 * through to the workspace default, then the deployment's, exactly as the
 * backend's resolution chain (request > conversation > workspace > env)
 * intends. That is why the composer's "Auto" states write a `null`
 * clear rather than the string `"auto"`: a stored `"auto"` would pin the
 * heuristic for this thread and shadow a workspace default that says
 * otherwise, which is a different thing from having made no choice.
 *
 * Reading is LENIENT, mirroring the backend's stance on stored settings: a
 * blob written by another version (or by hand) must never break the composer,
 * so an unrecognized value reads as unpinned rather than throwing.
 */

import type { ConversationSettingsPatch } from "@/lib/conversations";

/** The shared request/stored vocabulary for the two enum axes. */
export type SettingMode = "auto" | "on" | "off";

/** What the composer's controls hold for one conversation. */
export interface ComposerPins {
  /** Chat model tag this conversation pins, or null for the default. */
  model: string | null;
  /** Thinking mode this conversation pins, or null for Auto. */
  thinking: SettingMode | null;
  /** Web-search mode this conversation pins, or null for Auto. */
  webSearch: SettingMode | null;
}

/** A conversation that has chosen nothing - a fresh pane, and a pane reset. */
export const NO_PINS: ComposerPins = { model: null, thinking: null, webSearch: null };

/**
 * Axes holding a stored value the composer's controls cannot show.
 *
 * Both enum axes are stored TRI-state and displayed by a TWO-state control, so
 * a value can be stored that no position of the control means: see
 * {@link readPins} for what is done about it.
 */
export interface DirtyAxes {
  /** Thinking is stored `"off"`, which the Auto/On pill cannot express. */
  thinking: boolean;
  /** Web search is stored `"on"`, which the on/off switch cannot express. */
  webSearch: boolean;
}

/** Nothing stored that the controls would misrepresent - the ordinary case. */
export const NO_DIRTY_AXES: DirtyAxes = { thinking: false, webSearch: false };

/** What a conversation's stored settings mean to the composer. */
export interface StoredPins {
  /** The pins the controls show. */
  pins: ComposerPins;
  /** Axes whose stored value was normalized away; see {@link readPins}. */
  dirty: DirtyAxes;
}

function isSettingMode(value: unknown): value is SettingMode {
  return value === "auto" || value === "on" || value === "off";
}

/**
 * Read the composer's pins out of a conversation's stored settings.
 *
 * Two stored values have no position on the control that owns them: thinking
 * `"off"` (the pill offers Auto and On) and web search `"on"` (the switch
 * offers off, and an on that means the freshness heuristic). Left as they are,
 * each would be DISPLAYED as its opposite while still being OBEYED - the pill
 * would read Auto over a conversation that pins thinking off, and every turn
 * would quietly follow the stored value instead. So they are read as UNPINNED,
 * which makes the control honest about what the next turn does, and the axis is
 * marked dirty: the next successful pin write names it explicitly and clears
 * it, rather than leaving a hidden value to resurface on the next reload.
 *
 * Reading is LENIENT otherwise, mirroring the backend's stance on stored
 * settings: a blob written by another version (or by hand) must never break the
 * composer, so an unrecognized value reads as unpinned rather than throwing.
 *
 * @param settings - The conversation detail's `settings` JSON, whatever shape
 *   it turns out to have.
 * @returns The pins to show, plus the axes whose stored value was dropped.
 */
export function readPins(settings: unknown): StoredPins {
  if (typeof settings !== "object" || settings === null) {
    return { pins: NO_PINS, dirty: NO_DIRTY_AXES };
  }
  const raw = settings as Record<string, unknown>;
  const model = raw.model;
  const thinking = isSettingMode(raw.thinking) ? raw.thinking : null;
  const webSearch = isSettingMode(raw.web_search) ? raw.web_search : null;
  return {
    pins: {
      model: typeof model === "string" && model !== "" ? model : null,
      thinking: thinking === "off" ? null : thinking,
      webSearch: webSearch === "on" ? null : webSearch,
    },
    dirty: { thinking: thinking === "off", webSearch: webSearch === "on" },
  };
}

/**
 * Whether the thinking control reads as On.
 *
 * The control is two-state (Auto / On) while the stored axis is tri-state, so
 * anything that is not an explicit `"on"` shows as Auto rather than inventing a
 * third button. The one stored value that would make that a LIE - `"off"`,
 * which Auto does not mean - never gets this far: {@link readPins} reads it as
 * unpinned and queues it to be cleared.
 *
 * @param pins - The conversation's pins.
 * @returns True when the conversation pins thinking on.
 */
export function thinkingOn(pins: ComposerPins): boolean {
  return pins.thinking === "on";
}

/**
 * Whether the web-search control reads as enabled.
 *
 * Enabled is the DEFAULT and means "auto" (the freshness heuristic decides);
 * only an explicit `"off"` - fully local, no live fetch - switches it off. A
 * stored `"on"` (search every turn, which the enabled position does not mean)
 * never gets this far; see {@link readPins}.
 *
 * @param pins - The conversation's pins.
 * @returns True unless the conversation pins web search off.
 */
export function webSearchOn(pins: ComposerPins): boolean {
  return pins.webSearch !== "off";
}

/** Whether anything is pinned at all (a fresh pane has nothing to persist). */
export function hasPins(pins: ComposerPins): boolean {
  return pins.model !== null || pins.thinking !== null || pins.webSearch !== null;
}

/**
 * Undo a refused save, one axis at a time.
 *
 * PER AXIS, never wholesale, because the axes are saved independently and can
 * be in flight at once: if a model save is refused while a thinking toggle
 * made in the meantime succeeded, only the model may go back. Comparing
 * whole objects would abandon the revert entirely in that case - leaving the
 * model control showing a pin the server rejected - and replacing the whole
 * object would undo the toggle that worked.
 *
 * An axis whose value has moved on since the attempt is left alone: the user
 * has already expressed a newer intent, and that intent has its own save.
 *
 * @param current - The pins as they stand now.
 * @param attempted - The pins the refused save tried to store.
 * @param previous - The pins as they were before that save.
 * @returns The pins with the refused axes put back.
 */
export function revertPins(
  current: ComposerPins,
  attempted: ComposerPins,
  previous: ComposerPins,
): ComposerPins {
  return {
    model: current.model === attempted.model ? previous.model : current.model,
    thinking: current.thinking === attempted.thinking ? previous.thinking : current.thinking,
    webSearch:
      current.webSearch === attempted.webSearch ? previous.webSearch : current.webSearch,
  };
}

/**
 * The settings PATCH that turns one set of pins into another.
 *
 * ONLY the axes that actually moved are named, which is what makes the
 * server's per-field merge worth having: a control writes its own axis and
 * leaves the other two exactly as stored, so two surfaces changing different
 * settings cannot overwrite each other. An axis that moved to unpinned is
 * named as an explicit `null` - a clear, which is a different instruction from
 * saying nothing about it.
 *
 * A DIRTY axis is the one exception: it is named even when it did not move,
 * because the value {@link readPins} dropped is still stored, and this write is
 * what takes it out of the conversation for good.
 *
 * @param previous - The pins as stored.
 * @param next - The pins to reach.
 * @param dirty - Axes whose stored value the controls cannot show; each is
 *   named by the patch regardless of whether it moved.
 * @returns A patch naming what changed (plus any dirty axis); empty when
 *   nothing did.
 */
export function pinsDiff(
  previous: ComposerPins,
  next: ComposerPins,
  dirty: DirtyAxes = NO_DIRTY_AXES,
): ConversationSettingsPatch {
  const patch: ConversationSettingsPatch = {};
  if (next.model !== previous.model) patch.model = next.model;
  if (next.thinking !== previous.thinking || dirty.thinking) {
    patch.thinking = next.thinking;
  }
  if (next.webSearch !== previous.webSearch || dirty.webSearch) {
    patch.web_search = next.webSearch;
  }
  return patch;
}
