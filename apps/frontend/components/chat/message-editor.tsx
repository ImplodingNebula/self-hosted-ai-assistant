"use client";

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from "react";

/**
 * The question bubble, swapped for a box to rewrite it in.
 *
 * It sits exactly where the bubble was rather than opening a dialog: the turn
 * being changed is the one on screen, and moving the words somewhere else to
 * change them loses that. The box grows with its content like the composer does,
 * and the two ways out are the two a text field is expected to have - Escape
 * cancels, Ctrl+Enter (or Cmd+Enter) saves.
 *
 * Save is disabled while the draft is empty or unchanged, because neither is an
 * edit: an empty question is a deletion, which this is not, and an unchanged one
 * would discard the reply below it for nothing.
 */
export function MessageEditor({
  initial,
  onSave,
  onCancel,
}: {
  /** The question's current text, which the box opens on. */
  initial: string;
  /** Commit the rewrite; the caller closes the editor. */
  onSave: (text: string) => void;
  /** Abandon it, leaving the question exactly as it was. */
  onCancel: () => void;
}) {
  const [draft, setDraft] = useState(initial);
  const ref = useRef<HTMLTextAreaElement>(null);

  const resize = useCallback(() => {
    const el = ref.current;
    if (!el) return;
    // Collapse first so a deletion is measured too, then fit the content.
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, []);

  // Before the first paint, so the box never flashes at one line and then jumps
  // to fit a long question.
  useLayoutEffect(resize, [draft, resize]);

  useEffect(() => {
    // The pencil that opened this editor, to hand focus back to on close - the
    // same courtesy the ConfirmDialog pays its invoker, and it doubles as
    // re-revealing the action row (which shows on :focus-within). Skipped when
    // the button is gone by then: a saved edit starts a stream, which hides it.
    const invoker = document.activeElement;
    const el = ref.current;
    if (!el) return;
    el.focus();
    // Caret at the end rather than a full selection: the usual edit is a tweak
    // to what is there, and selecting everything makes the first keystroke a
    // delete.
    el.setSelectionRange(el.value.length, el.value.length);
    return () => {
      if (invoker instanceof HTMLElement && invoker.isConnected) invoker.focus();
    };
  }, []);

  const trimmed = draft.trim();
  const unchanged = trimmed === initial.trim();
  // No in-flight state to guard against: the page closes the editor before it
  // starts any turn (see runRewind), so an open editor means a settled pane.
  const canSave = trimmed !== "" && !unchanged;

  return (
    <div data-part="message-editor">
      <textarea
        ref={ref}
        value={draft}
        rows={1}
        aria-label="Edit this message"
        onChange={(event) => {
          setDraft(event.target.value);
          // Synchronously, so the box never lags a line behind the text.
          resize();
        }}
        onKeyDown={(event) => {
          if (event.key === "Escape") {
            event.preventDefault();
            onCancel();
            return;
          }
          // Plain Enter is a newline here, unlike the composer: a question being
          // reworked is usually longer than one being typed, and a reflex from
          // the box below must not throw away the reply on a stray keystroke.
          if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
            event.preventDefault();
            if (canSave) onSave(trimmed);
          }
        }}
        className="message-editor-input"
      />
      <div className="message-editor-actions">
        <button type="button" className="btn btn-ghost" onClick={onCancel}>
          Cancel
        </button>
        <button
          type="button"
          className="btn btn-primary"
          disabled={!canSave}
          title={
            unchanged && trimmed !== ""
              ? "Nothing has changed yet"
              : "Ask this question again, reworded"
          }
          onClick={() => onSave(trimmed)}
        >
          Save
        </button>
      </div>
    </div>
  );
}
