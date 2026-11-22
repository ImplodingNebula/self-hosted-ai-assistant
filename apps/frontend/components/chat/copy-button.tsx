"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { Icon } from "@/components/chat/step";
import { writeToClipboard } from "@/lib/clipboard";

/** How long the button wears its tick before returning to the clipboard glyph. */
const COPIED_FEEDBACK_MS = 1500;

/**
 * Copy some text, and say so.
 *
 * The glyph cross-fades to a tick in `--s-ok` for a moment on success; a
 * failed write changes nothing, because a tick over a clipboard that never
 * took the text sends someone to paste nothing. Icon-only: the word lives in
 * the tooltip and the accessible name, and a live region announces the copy
 * to a keyboard user, since renaming an already-focused button is not
 * reliably read out.
 */
export function CopyButton({
  text,
  label,
  className,
  onAct,
}: {
  text: string;
  /** The verb as the tooltip and the accessible name ("Copy answer"). */
  label: string;
  className?: string;
  /** Told after a successful copy, for a row that holds itself open a moment. */
  onAct?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const timerRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    [],
  );
  const copy = useCallback(async () => {
    const ok = await writeToClipboard(text);
    if (!ok) return;
    setCopied(true);
    onAct?.();
    if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    timerRef.current = window.setTimeout(() => {
      setCopied(false);
      timerRef.current = null;
    }, COPIED_FEEDBACK_MS);
  }, [text, onAct]);
  return (
    <button
      type="button"
      className={`copy-btn${className ? ` ${className}` : ""}`}
      data-copied={copied ? "true" : undefined}
      title={copied ? "Copied" : label}
      onClick={() => void copy()}
    >
      <span className="ic">
        <Icon name="copy" className="i cp" />
        <Icon name="check" className="i ok" />
      </span>
      <span className="sr-only" aria-live="polite">
        {copied ? "Copied" : label}
      </span>
    </button>
  );
}
