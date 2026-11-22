"use client";

import { Icon } from "@/components/chat/step";

/** Props for {@link WebSearchToggle}. */
export interface WebSearchToggleProps {
  /** True when the web may be consulted (Auto); false is fully local. */
  value: boolean;
  /** Allow the web again, or switch this conversation fully local. */
  onChange: (enabled: boolean) => void;
  /** Blocked while a suspended turn waits on its confirmation. */
  disabled?: boolean;
  /**
   * Why the switch, though on, will not apply this turn - or null when it
   * will. A sentence, because a control that goes quiet without saying why
   * is the failure this replaces: thinking is pinned on, and the chosen model
   * cannot reason and use tools at once, so the turn will not search. The
   * switch is struck (`data-held`) rather than turned off: the preference
   * stands, and comes back the moment the model or the pin changes.
   */
  heldReason?: string | null;
}

/**
 * Two-state web-search control for the composer: enabled (the default) or off.
 *
 * Enabled means AUTO, not "always search": a freshness heuristic - or, on a
 * tool-driving turn, the model itself - decides whether a question is worth
 * going out for, so an ordinary turn stays local anyway. Off is the literal
 * promise: no new fetch happens on this conversation's turns at all.
 *
 * Off is what gets stored; enabled is a CLEARED pin, so a conversation left
 * alone follows the workspace default rather than freezing today's one.
 *
 * Sources already fetched into this conversation stay citable while off - the
 * switch stops new fetches, and attributing what the answer actually used is
 * honesty, not a fetch.
 */
export function WebSearchToggle({
  value,
  onChange,
  disabled,
  heldReason,
}: WebSearchToggleProps) {
  const held = value && (heldReason ?? null) !== null;
  return (
    <button
      type="button"
      aria-pressed={value}
      disabled={disabled}
      data-held={held ? "true" : undefined}
      onClick={() => onChange(!value)}
      title={
        held
          ? (heldReason as string)
          : value
            ? "Look things up online when a question needs current information."
            : "Answer from local knowledge only - nothing is fetched."
      }
      className="ctl"
    >
      <Icon name="globe" />
      <span className="lbl">Web search</span>
    </button>
  );
}
