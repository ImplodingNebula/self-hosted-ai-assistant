"use client";

import {
  type RefObject,
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
} from "react";

import { writeToClipboard } from "@/lib/clipboard";

/** How long the Copy half wears its "Copied" tick before returning to normal. */
const COPIED_FEEDBACK_MS = 1500;

/** Where the group sits, and what it would act on. */
interface Pending {
  /** Viewport coordinates: the selection's START - the first character's left
   * edge, on the first line - not the bounding box's centre. The eye goes back
   * to where the selection began, so that is where the controls should be. */
  x: number;
  y: number;
  /** The first line's bottom edge, where the group lands when it must flip
   * BELOW the line (a selection starting at the top of the viewport leaves no
   * room above it). */
  bottom: number;
  text: string;
  /** The message the words were taken from, when both ends are in one turn. */
  messageId?: string;
}

/** The two places message text actually lives; nothing else in the thread quotes. */
const BUBBLE_SELECTOR = '[data-part="bubble-user"], [data-part="bubble-assist"]';

/**
 * Whether a selection endpoint sits inside a message bubble.
 *
 * Range containers are usually TEXT nodes, which have no `closest`; the walk
 * starts from the nearest element instead.
 */
function inMessageBubble(node: Node): boolean {
  const el = node instanceof Element ? node : node.parentElement;
  return el !== null && el.closest(BUBBLE_SELECTOR) !== null;
}

/** The turn a selection endpoint sits in, for the quote's record of its source. */
function turnOf(node: Node): HTMLElement | null {
  const el = node instanceof Element ? node : node.parentElement;
  return el?.closest<HTMLElement>(".turn") ?? null;
}

function ReplyGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M6 3.5L2.5 7 6 10.5" />
      <path d="M2.5 7h6a4.5 4.5 0 0 1 4.5 4.5v1" />
    </svg>
  );
}

function CopyGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.5" />
      <path d="M10.5 5.5V4a1.5 1.5 0 0 0-1.5-1.5H4A1.5 1.5 0 0 0 2.5 4v5A1.5 1.5 0 0 0 4 10.5h1.5" />
    </svg>
  );
}

function CheckGlyph() {
  return (
    <svg
      viewBox="0 0 16 16"
      className="h-3.5 w-3.5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.6"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path d="M3 8.5l3.5 3.5L13 4.5" />
    </svg>
  );
}

/**
 * A two-action group - Quote and Copy - that appears over a selection's start.
 *
 * Replying to a whole message is one button on the action row; replying to ONE
 * sentence of it needs the sentence to be pointed at, and selecting text is how
 * that is already done everywhere else. Copy rides along because the same
 * gesture is how a passage is taken OUT of the conversation, and reaching for
 * the message-level Copy would take the whole reply when the selection already
 * says which part was wanted.
 *
 * It anchors over the START of the selection - the first character, not the
 * bounding box's centre. A multi-line selection's box spans the whole thread
 * column, and controls centred on it land somewhere over the middle of a
 * paragraph, nowhere the eye is; the start is where the gesture began and where
 * the reader returns to. (Clamped to the viewport, so a selection starting at
 * the screen's edge still shows the whole group.)
 *
 * It is deliberately shy about STALENESS, not about size. It appears only for a
 * real selection inside the thread, and it goes at the first sign the selection
 * is no longer what the user is looking at - a scroll, a resize, a click
 * elsewhere, the selection collapsing. Positioned in viewport coordinates over
 * everything, so a bubble's own overflow cannot clip it.
 */
export function SelectionQuote({
  containerRef,
  onQuote,
}: {
  /** The scroll container the selection must be inside; nothing else offers a quote. */
  containerRef: RefObject<HTMLElement | null>;
  /** Take this passage as the composer's pending quote, from the message it names (one turn, with an id). */
  onQuote: (text: string, messageId?: string) => void;
}) {
  const [pending, setPending] = useState<Pending | null>(null);
  const [copied, setCopied] = useState(false);
  const groupRef = useRef<HTMLDivElement>(null);
  const timerRef = useRef<number | null>(null);

  // The copied tick belongs to ONE shown selection, so every move of the group
  // - a new selection, a dismissal - resets it at the same moment.
  const show = useCallback((next: Pending | null) => {
    setPending(next);
    setCopied(false);
  }, []);
  const dismiss = useCallback(() => show(null), [show]);

  // An unmount must not leave the copied timer firing at nothing.
  useEffect(
    () => () => {
      if (timerRef.current !== null) window.clearTimeout(timerRef.current);
    },
    [],
  );

  // The group is centred on the selection's start, which can sit at the very
  // edge of the viewport; nudge it back in AFTER it has a size to measure.
  // (The overrides survive the copied re-render because React only writes
  // style properties whose PROP VALUE changed, and pending has not.)
  useLayoutEffect(() => {
    const el = groupRef.current;
    if (!el || pending === null) return;
    const half = el.offsetWidth / 2 + 8;
    const clamped = Math.min(Math.max(pending.x, half), window.innerWidth - half);
    if (clamped !== pending.x) el.style.left = `${clamped}px`;
    // No room above the first line - the topbar counts as occupied - so flip
    // BELOW it instead. Clamping downward is not an option: it would sit the
    // group on the very words being quoted.
    const topbar = document.querySelector('[data-part="topbar"]');
    const occupied = topbar ? topbar.getBoundingClientRect().bottom : 0;
    if (pending.y - el.offsetHeight - 10 < occupied + 4) {
      el.setAttribute("data-flip", "");
      el.style.top = `${pending.bottom + 10}px`;
    } else {
      el.removeAttribute("data-flip");
      el.style.top = `${pending.y}px`;
    }
  }, [pending]);

  useEffect(() => {
    const read = () => {
      const container = containerRef.current;
      const selection = window.getSelection();
      if (!container || !selection || selection.isCollapsed || selection.rangeCount === 0) {
        show(null);
        return;
      }
      const text = selection.toString().trim();
      if (text === "") {
        show(null);
        return;
      }
      // Both ends must be inside the thread: a selection that started in the
      // transcript and ran into the composer is not a quote of anything.
      const range = selection.getRangeAt(0);
      if (
        !container.contains(range.startContainer) ||
        !container.contains(range.endContainer)
      ) {
        show(null);
        return;
      }
      // And inside MESSAGE text specifically: the empty pane's "Ask anything"
      // heading (and any other thread chrome) shares the scroll container, but
      // quoting it back to the assistant quotes nobody. Each end checks its
      // own bubble rather than both sharing one, so a drag across two messages
      // keeps offering the quote it offers today.
      if (!inMessageBubble(range.startContainer) || !inMessageBubble(range.endContainer)) {
        show(null);
        return;
      }
      // The first client rect is the first line's fragment in DOCUMENT order,
      // whichever direction the drag ran - exactly the "start of the selected
      // text" the anchor wants. The bounding box stands in for the degenerate
      // case where a range reports no rects.
      const rects = range.getClientRects();
      const first = rects.length > 0 ? rects[0] : range.getBoundingClientRect();
      // A drag across two messages quotes words from both; it names neither.
      const startTurn = turnOf(range.startContainer);
      const endTurn = turnOf(range.endContainer);
      const messageId =
        startTurn !== null && startTurn === endTurn ? startTurn.dataset.messageId : undefined;
      show({ x: first.left, y: first.top, bottom: first.bottom, text, messageId });
    };

    // Deferred a tick: on pointerup the browser has not always settled the
    // selection yet, and reading it in the same frame gives the PREVIOUS one.
    const onPointerUp = (event: PointerEvent) => {
      const target = event.target;
      // A press on the group itself is what USES the selection, not a new
      // one - reading here would race its own click.
      if (target instanceof Element && target.closest('[data-part="selection-quote"]')) {
        return;
      }
      window.setTimeout(read, 0);
    };
    // Keyboard selection (shift+arrows) never fires pointerup.
    const onKeyUp = (event: KeyboardEvent) => {
      if (event.shiftKey || event.key.startsWith("Arrow")) window.setTimeout(read, 0);
    };

    document.addEventListener("pointerup", onPointerUp);
    document.addEventListener("keyup", onKeyUp);
    // Capture phase, so a scroll INSIDE the thread reaches this too (scroll does
    // not bubble). The group is positioned in viewport coordinates, so any
    // scroll leaves it pointing at the wrong words.
    window.addEventListener("scroll", dismiss, true);
    window.addEventListener("resize", dismiss);
    return () => {
      document.removeEventListener("pointerup", onPointerUp);
      document.removeEventListener("keyup", onKeyUp);
      window.removeEventListener("scroll", dismiss, true);
      window.removeEventListener("resize", dismiss);
    };
  }, [containerRef, dismiss, show]);

  if (pending === null) return null;

  return (
    <div
      ref={groupRef}
      data-part="selection-quote"
      style={{ left: `${pending.x}px`, top: `${pending.y}px` }}
      // The press must not take the selection away before the click reads it.
      onMouseDown={(event) => event.preventDefault()}
    >
      <button
        type="button"
        className="selection-action"
        onClick={() => {
          onQuote(pending.text, pending.messageId);
          show(null);
          window.getSelection()?.removeAllRanges();
        }}
      >
        <ReplyGlyph />
        Quote
      </button>
      <button
        type="button"
        className="selection-action"
        data-copied={copied || undefined}
        onClick={() => {
          // The legacy-command fallback copies by selecting an off-screen
          // textarea, which destroys the very selection being copied; keep a
          // clone and put it back so the highlight survives on origins where
          // the async API is unavailable.
          const selection = window.getSelection();
          const kept =
            selection && selection.rangeCount > 0
              ? selection.getRangeAt(0).cloneRange()
              : null;
          void writeToClipboard(pending.text).then((ok) => {
            if (!ok) return;
            if (kept && selection && (selection.rangeCount === 0 || selection.isCollapsed)) {
              selection.removeAllRanges();
              selection.addRange(kept);
            }
            setCopied(true);
            if (timerRef.current !== null) window.clearTimeout(timerRef.current);
            timerRef.current = window.setTimeout(() => {
              setCopied(false);
            }, COPIED_FEEDBACK_MS);
          });
        }}
      >
        {copied ? <CheckGlyph /> : <CopyGlyph />}
        {copied ? "Copied" : "Copy"}
      </button>
      {/* The state change a sighted user sees on the button, said out loud -
          OUTSIDE the button, so its accessible name stays just "Copied" rather
          than the label and the announcement concatenated. */}
      <span className="sr-only" aria-live="polite">
        {copied ? "Copied to clipboard" : ""}
      </span>
    </div>
  );
}
