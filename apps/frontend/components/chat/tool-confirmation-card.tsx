"use client";

import { ArgsView } from "@/components/chat/args-view";
import type { ToolConfirmationData } from "@/lib/types";

/**
 * The approve/deny prompt for a side-effecting tool the model wants to run.
 *
 * Shows exactly what would run (tool name + its arguments) so the decision is
 * informed, then either offers the two buttons (undecided) or states the outcome
 * (once answered). A side-effecting tool never runs without a click here.
 */
export function ToolConfirmationCard({
  confirmation,
  decision,
  busy,
  onConfirm,
}: {
  confirmation: ToolConfirmationData;
  /** The user's answer, once given; undefined while still pending. */
  decision?: "approve" | "deny";
  /** A resume (or another turn) is in flight, so the buttons are disabled. */
  busy: boolean;
  onConfirm: (decision: "approve" | "deny") => void;
}) {
  const decided = decision != null;
  const heading = !decided
    ? "The assistant wants to run an action"
    : decision === "approve"
      ? "You approved this action"
      : "You declined this action";
  return (
    <div data-part="confirm">
      <p className="confirm-head">{heading}</p>
      <ul className="confirm-list">
        {confirmation.calls.map((call) => (
          <li key={call.id} className="confirm-call">
            {/* Deliberately NOT humanized, unlike the trace card: a humanized
                label is lossy, and a remote server that names itself
                "web-search" would render as "Web search" - the built-in's
                label - without ever colliding with it. The built-ins are
                read-only and never reach this card, so a familiar-looking
                label here could only be an impostor. This gate shows the
                identifier that will actually be called. */}
            <span className="tool-name">{call.name}</span>
            {Object.keys(call.arguments).length > 0 && (
              <div className="confirm-detail">
                <ArgsView args={call.arguments} />
              </div>
            )}
          </li>
        ))}
      </ul>
      {!decided && (
        <div className="confirm-actions">
          <button
            type="button"
            // Named so other surfaces can hand the keyboard here: this is the
            // decision the page is waiting on, so a control that goes dead
            // while the card is up (the composer's model menu) sends focus to
            // it rather than dropping it on the document.
            data-part="confirm-approve"
            onClick={() => onConfirm("approve")}
            disabled={busy}
            className="btn btn-primary"
          >
            Approve
          </button>
          <button
            type="button"
            onClick={() => onConfirm("deny")}
            disabled={busy}
            className="btn btn-ghost"
          >
            Deny
          </button>
        </div>
      )}
    </div>
  );
}
