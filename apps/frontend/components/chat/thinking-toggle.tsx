"use client";

import { Icon } from "@/components/chat/step";

/** Props for {@link ThinkingToggle}. */
export interface ThinkingToggleProps {
  /** True when this conversation pins thinking on; false is Auto. */
  value: boolean;
  /** Pin thinking on, or clear the pin back to Auto. */
  onChange: (on: boolean) => void;
  /** Blocked while a suspended turn waits on its confirmation. */
  disabled?: boolean;
  /**
   * Why this control cannot do anything for the pinned model, when that is the
   * case; null when it can.
   *
   * A SENTENCE and not a boolean, because a control that goes quiet without
   * saying why is the failure this replaces: the model answers directly, no
   * partner is paired with it, and asking for reasoning would change nothing
   * at all. It disables the button and becomes its title, so the reason is
   * reachable rather than merely true.
   */
  unavailableReason?: string | null;
}

/**
 * Two-state thinking control for the composer: Auto (the default) or On.
 *
 * There is deliberately NO Off. Auto already declines to reason on the turns
 * that do not need it, so an Off state would only differ by forbidding it on
 * the turns that do - a way to make answers worse, dressed up as a choice.
 * Auto is stored as a CLEARED pin, which is why it reads as unpressed: the
 * conversation has expressed no preference and defers to the workspace.
 *
 * What the choice really costs is worth knowing: on a model that cannot
 * reason and call tools in the same turn, On means the turn will not search or
 * look anything up. The title says so, because nothing else on screen would.
 *
 * And it goes QUIET, with a reason, when the pinned model cannot reason and has
 * no partner paired with it. A live toggle that changes nothing is worse than
 * an absent one: it is a promise the turn will not keep, and the user has no
 * way to discover that except by asking a hard question and getting a shallow
 * answer. The reason names the fix, because there is one - pairing a reasoning
 * partner with the model, in the model manager.
 */
export function ThinkingToggle({
  value,
  onChange,
  disabled,
  unavailableReason,
}: ThinkingToggleProps) {
  const unavailable = (unavailableReason ?? null) !== null;
  // A conversation can pin thinking On and then switch to a model that cannot
  // do it. Rendering the pin as-is would leave the control lit AND frozen -
  // reading as "this turn will reason" when the opposite is true, which is the
  // one thing a disabled control must never say. The pin is not cleared (the
  // conversation's preference survives a model change, and comes back with the
  // next model that can honour it); it simply stops being SHOWN as in force.
  const lit = value && !unavailable;
  return (
    <button
      type="button"
      aria-pressed={lit}
      // Not `aria-disabled`: there is nothing here for a keyboard to inspect
      // beyond the title, and unlike the model menu's rows this control has no
      // sibling whose numbering would change by skipping it.
      disabled={disabled || unavailable}
      onClick={() => onChange(!value)}
      title={
        unavailable
          ? (unavailableReason as string)
          : value
            ? "Always reason before answering. On models that cannot do both at once, this turn will not use tools."
            : "Reason first only when the question calls for it."
      }
      className="ctl"
    >
      <Icon name="thought" />
      <span className="lbl">Thinking: {lit ? "On" : "Auto"}</span>
    </button>
  );
}
