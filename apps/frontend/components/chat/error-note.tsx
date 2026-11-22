"use client";

import { Icon } from "@/components/chat/step";

/**
 * Inline error for the pane: a danger glyph and the sentence, with an
 * optional second line and an optional action. No box: in the thread the
 * danger lives on the rail's glyph and the sky's wash, and under the empty
 * pane's greeting a bordered bar read as a wall of alarm. Rendered in both the
 * empty and the populated pane states.
 */
export function ErrorNote({
  message,
  detail,
  action,
  onAction,
}: {
  message: string;
  /** A quieter second line: what did and did not happen. */
  detail?: string;
  /** The verb of the retry affordance, when there is one. */
  action?: string;
  onAction?: () => void;
}) {
  return (
    <div data-part="error" role="alert">
      <Icon name="x" className="i err-glyph" />
      <span className="err-text">{message}</span>
      {action && onAction && (
        <button type="button" className="btn btn-ghost" onClick={onAction}>
          {action}
        </button>
      )}
      {detail && <small>{detail}</small>}
    </div>
  );
}
