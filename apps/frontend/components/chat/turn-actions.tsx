"use client";

import { useCallback, useEffect, useRef } from "react";

import { CopyButton } from "@/components/chat/copy-button";
import { Icon } from "@/components/chat/step";

/** How long a row stays open after one of its verbs is taken. */
const HOLD_MS = 2000;

/** The small print under a reply: which model answered, and what it wrote. */
export interface TurnMeta {
  model?: string | null;
  tokens?: number;
  /** Tokens per second over the answer's stream; null when it cannot be read. */
  tokensPerSecond?: number | null;
}

/**
 * The quiet action row that closes a turn.
 *
 * On a reply it is the footer's verbs - copy, reply, answer again - followed
 * by the figures in small mono; on a question it is edit, reply and copy
 * under the bubble's right edge. Every action is icon-only, with the word in
 * the tooltip and the accessible name; an action is ABSENT rather than
 * disabled when the pane cannot take it.
 *
 * Revealed by opacity, never by mounting, so a hover cannot reflow the turn
 * and the buttons stay in the tab order. A verb taken from the row holds it
 * open a moment (`data-hold` on the turn), long enough for the copied tick to
 * be seen before the row fades.
 */
export function TurnActions({
  text,
  align = "start",
  onRegenerate,
  onReply,
  onEdit,
  meta,
}: {
  /** The words this action row belongs to; copied verbatim. */
  text: string;
  /** Which edge of the turn the row hangs off: a question's is `end`. */
  align?: "start" | "end";
  /** Answer this turn again; absent when the pane cannot take the request. */
  onRegenerate?: () => void;
  /** Quote this message into the composer. */
  onReply?: () => void;
  /** Open this question for rewriting; absent while the pane cannot take a turn. */
  onEdit?: () => void;
  /** The reply's figures; a question has none. */
  meta?: TurnMeta;
}) {
  const rowRef = useRef<HTMLDivElement>(null);
  const holdRef = useRef<number | null>(null);
  useEffect(
    () => () => {
      if (holdRef.current !== null) window.clearTimeout(holdRef.current);
    },
    [],
  );
  /** Keep the row open a moment after a verb is taken from it. */
  const hold = useCallback(() => {
    const turn = rowRef.current?.closest(".turn");
    if (!(turn instanceof HTMLElement)) return;
    turn.dataset.hold = "true";
    if (holdRef.current !== null) window.clearTimeout(holdRef.current);
    holdRef.current = window.setTimeout(() => {
      delete turn.dataset.hold;
      holdRef.current = null;
    }, HOLD_MS);
  }, []);

  // Nothing to copy and nothing to act on - a file-only question that cannot be
  // rewritten either - so the row is not rendered at all.
  if (
    text === "" &&
    onRegenerate === undefined &&
    onReply === undefined &&
    onEdit === undefined
  ) {
    return null;
  }
  const isReply = align === "start";
  const showMeta =
    isReply &&
    meta !== undefined &&
    ((meta.model ?? null) !== null || meta.tokens !== undefined);

  // Copy leads an answer's row and closes a question's: the first verb is the
  // one most often wanted, which for a question is to rewrite it.
  const copy = text !== "" && (
    <CopyButton text={text} label={isReply ? "Copy answer" : "Copy"} onAct={hold} />
  );

  return (
    <div ref={rowRef} data-part="turn-actions" data-align={align}>
      {isReply && copy}
      {onEdit !== undefined && (
        <button type="button" className="act-btn" title="Edit" onClick={onEdit}>
          <Icon name="edit" />
          <span className="sr-only">Edit this message</span>
        </button>
      )}
      {onReply !== undefined && (
        <button
          type="button"
          className="act-btn"
          title="Reply"
          onClick={() => {
            hold();
            onReply();
          }}
        >
          <Icon name="reply" />
          <span className="sr-only">Reply to this message</span>
        </button>
      )}
      {!isReply && copy}
      {onRegenerate !== undefined && (
        // Whether it needs confirming is the caller's call, not this row's:
        // replacing the newest reply is exactly what the button says and asks
        // nothing, while further up the thread it also discards what came
        // after and the page puts a dialog in front of it.
        <button
          type="button"
          className="act-btn"
          title="Answer again"
          onClick={onRegenerate}
        >
          <Icon name="redo" />
          <span className="sr-only">Answer again</span>
        </button>
      )}
      {showMeta && (
        <span className="turn-meta">
          {(meta.model ?? null) !== null && <span className="meta-model">{meta.model}</span>}
          {meta.tokens !== undefined && (
            <span className="meta-num">{meta.tokens} tokens</span>
          )}
          {meta.tokensPerSecond != null && (
            <span className="meta-num">{meta.tokensPerSecond} tokens/s</span>
          )}
        </span>
      )}
    </div>
  );
}
