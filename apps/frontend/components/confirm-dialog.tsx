"use client";

import { type ReactNode, useEffect, useId, useRef } from "react";

/** Props for {@link ConfirmDialog}. */
export interface ConfirmDialogProps {
  /** Question the dialog asks, rendered as its accessible name. */
  title: string;
  /** One line saying what the action costs, in plain words. */
  body: string;
  /** Label for the destructive action (default: "Delete"). */
  confirmLabel?: string;
  /** Label for the safe action (default: "Cancel"). */
  cancelLabel?: string;
  /**
   * The confirm button's hue (default: "danger"). "accent" is for an action
   * that is consequential but not destructive - a benign verb confirmed in
   * blood red un-says its own benignity at the moment of commitment.
   */
  tone?: "danger" | "accent";
  /**
   * Withhold the confirm button (default: false), for a question whose only
   * honest answer is Cancel - the body says why. The button stays visible so
   * the dialog's shape does not change with its answer.
   */
  confirmDisabled?: boolean;
  /** Run the confirmed action; the caller closes the dialog. */
  onConfirm: () => void;
  /** Dismiss without acting (Escape, the scrim, or the safe button). */
  onCancel: () => void;
  /**
   * An input the action needs, rendered between the body and the actions -
   * the delete-the-default dialog puts its successor picker here. Anything
   * interactive inside stays inside the focus trap.
   */
  children?: ReactNode;
}

/** Focusable controls inside the panel, in tab order. */
function focusablesIn(root: HTMLElement | null): HTMLElement[] {
  if (!root) return [];
  return Array.from(
    root.querySelectorAll<HTMLElement>("button:not([disabled]), select:not([disabled])"),
  );
}

/**
 * Centered confirmation dialog for an action that cannot be undone.
 *
 * Render it conditionally (`{pending && <ConfirmDialog ... />}`) rather than
 * toggling a `hidden` attribute: the author `display` on the scrim would beat
 * the user-agent's non-important `[hidden]` rule and the dialog would ship
 * open. Modal for real - focus starts on the safe action, Tab cycles inside
 * the panel, Escape cancels, and focus returns to whatever opened it.
 *
 * It cannot fence the page's type-anywhere listener off by itself: both live
 * on `document`, so neither can stop the other. The page suspends that
 * listener while a dialog is open instead.
 */
export function ConfirmDialog({
  title,
  body,
  confirmLabel = "Delete",
  cancelLabel = "Cancel",
  tone = "danger",
  confirmDisabled = false,
  onConfirm,
  onCancel,
  children,
}: ConfirmDialogProps) {
  const panelRef = useRef<HTMLDivElement>(null);
  const cancelRef = useRef<HTMLButtonElement>(null);
  const titleId = useId();
  const bodyId = useId();

  // The callers pass inline closures, so the keyboard effect below reads the
  // latest cancel through a ref and stays mount-only: re-running it would
  // re-capture the invoker (by then the Cancel button) on every render.
  const cancelLatest = useRef(onCancel);
  useEffect(() => {
    cancelLatest.current = onCancel;
  });

  useEffect(() => {
    const invoker = document.activeElement;
    cancelRef.current?.focus();
    // Bound to the document, not to the scrim: a click on the panel's own text
    // parks focus on the body, and a handler bound to the scrim would then
    // never see the keystroke that follows - Escape and Tab would both stop
    // working. Tab is also pulled back in from outside, so the trap holds
    // wherever focus has wandered to.
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        cancelLatest.current();
        return;
      }
      if (e.key !== "Tab") return;
      const panel = panelRef.current;
      const items = focusablesIn(panel);
      if (items.length === 0) return;
      const first = items[0];
      const last = items[items.length - 1];
      const active = document.activeElement;
      if (!(active instanceof Node) || panel?.contains(active) !== true) {
        e.preventDefault();
        (e.shiftKey ? last : first).focus();
      } else if (e.shiftKey && active === first) {
        e.preventDefault();
        last.focus();
      } else if (!e.shiftKey && active === last) {
        e.preventDefault();
        first.focus();
      }
    };
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("keydown", onKey);
      // Only if it is still on the page: the row that opened the dialog may
      // itself have been what the confirmed action removed.
      if (invoker instanceof HTMLElement && invoker.isConnected) invoker.focus();
    };
  }, []);

  return (
    <div
      className="dialog-scrim"
      // A destructive prompt defaults to "no", so dismissing on the scrim (or
      // on Escape) is the safe outcome, never the acting one.
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onCancel();
      }}
    >
      <div
        ref={panelRef}
        data-part="confirm-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={bodyId}
      >
        <h2 id={titleId} className="dialog-title">
          {title}
        </h2>
        <p id={bodyId} className="dialog-body">
          {body}
        </p>
        {children}
        <div className="dialog-actions">
          <button
            ref={cancelRef}
            type="button"
            onClick={onCancel}
            className="btn btn-ghost"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            disabled={confirmDisabled}
            onClick={onConfirm}
            className={tone === "danger" ? "btn btn-danger" : "btn btn-primary"}
          >
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
