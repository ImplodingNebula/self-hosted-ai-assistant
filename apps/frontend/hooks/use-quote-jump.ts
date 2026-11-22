"use client";

import { useCallback, useEffect, useRef, useState, type RefObject } from "react";

import type { FollowStream } from "@/hooks/use-follow-stream";
import { canLightPassage, clearPassage, findPassage, lightPassage } from "@/lib/passage-highlight";

/**
 * How long a landed-on turn wears its cue; the passage's light runs with it,
 * and the sweep's and the glow's keyframes (app/themes/chat.css) are this long.
 */
export const LIT_MS = 1800;

/**
 * Where a lit passage comes to rest: its first line this far down the
 * thread's viewport, so there is a little of what came before it above, the
 * words are not jammed against the topbar, and the rest of the message is
 * below, for reading on.
 */
const LAND_AT = 0.28;

/**
 * A marked message (a whole reply, the question returned to) comes to rest
 * centred, the entire message in view - what the sweep says it marks -
 * except that its turn's top never rests higher than this far down: a
 * message taller than the viewport shows its beginning, not its middle, and
 * the rows above it (lit with the rail) are not left under the topbar.
 */
const KEEP_TOP = 0.08;

/**
 * Lay the rail's light over a turn as one travel. Each segment of the rail
 * runs the prose's travelling mask (app/themes/chat.css, lit-rail-light),
 * so it needs the prose's height and its own offset under the prose's top:
 * measured here, once, as the turn is lit, and left on the elements as
 * custom properties (inert while the turn wears no cue). The segment's
 * inset under its glyph is read from the segment itself (the ::after, which
 * always has a box; the light shares its geometry rule).
 */
function layRailLight(turn: HTMLElement): void {
  const prose = turn.querySelector('[data-part="content"]') ?? turn;
  const r = prose.getBoundingClientRect();
  turn.style.setProperty("--rail-h", `${Math.max(1, r.height)}px`);
  for (const glyph of turn.querySelectorAll<HTMLElement>(".step-glyph")) {
    const inset = parseFloat(getComputedStyle(glyph, "::after").top) || 0;
    const top = glyph.getBoundingClientRect().top + inset - r.top;
    glyph.style.setProperty("--seg-top", `${top}px`);
  }
}

/**
 * What a landed-on turn wears where its words are not lit: "whole" when the
 * quote was of the entire message (the sweep over its prose), "ring" when
 * it is a place (the question returned to; an answer whose quoted words
 * could not be found).
 */
export type LitKind = "whole" | "ring";

/** What the hook hands the page. */
export interface QuoteJump {
  /**
   * The question whose quote was followed, to come back down to; null when
   * there is nowhere to return (nothing followed, or the reader is back at
   * the end).
   */
  returnTo: string | null;
  /** The message wearing a cue right now, and which, if any. */
  lit: { id: string; kind: LitKind } | null;
  /**
   * Follow a question's quote up to the message it came from: glide there
   * and light the passage where its words are found, sweep the message
   * where all of it was quoted, or ring the turn.
   *
   * @param sourceId - The quoted message, which must be in the thread.
   * @param fromId - The question being left, for the way back.
   * @param passage - The quoted words, to light; null where the quote was
   *   of the whole message, which is then marked as one.
   */
  jumpToQuoted: (sourceId: string, fromId: string | undefined, passage: string | null) => void;
  /** Come back down to the question whose quote was followed, and ring it. */
  returnToQuoting: () => void;
}

/**
 * Following a quote to its source and back.
 *
 * A sent question's quote is a link when the message it came from is still
 * in the thread. Following it glides the thread up to that message (the
 * follow hook's glide, which lands "away" so the reader is not pulled back
 * to the end), and on landing lights the quoted words for a moment, sweeps
 * the message where the whole of it was quoted, or rings the turn where the
 * words cannot be found (the message was edited, or the quote is the raw
 * markdown of a formatted answer). While the reader is away after that, the
 * jump floater is the way
 * back: it returns to the question that was left and rings it in turn, and
 * forgets the way back once the reader is at the end again by any route.
 * Nothing survives a change of transcript.
 *
 * @param scrollRef - The scrolling thread element.
 * @param follow - The thread's follow hook, for the glide and the away state.
 * @param paneKey - Changes with the transcript shown.
 */
export function useQuoteJump(
  scrollRef: RefObject<HTMLElement | null>,
  follow: FollowStream,
  paneKey: number | string,
): QuoteJump {
  const [returnTo, setReturnTo] = useState<string | null>(null);
  const [lit, setLit] = useState<{ id: string; kind: LitKind } | null>(null);
  const litTimerRef = useRef<number | null>(null);

  // Back at the end by any route - the floater, a scroll, a send - and the
  // way back is no longer needed. Settled during render, as a derived
  // correction, rather than in an effect that would render twice.
  const [seenAway, setSeenAway] = useState(follow.away);
  if (seenAway !== follow.away) {
    setSeenAway(follow.away);
    if (!follow.away) setReturnTo(null);
  }

  // A different transcript: nothing in it was followed or lit. The state
  // settles during render the same way; the light and the timer, which are
  // not state, go in the effect below.
  const [seenPane, setSeenPane] = useState(paneKey);
  if (seenPane !== paneKey) {
    setSeenPane(paneKey);
    setReturnTo(null);
    setLit(null);
  }

  const turnOf = useCallback(
    (id: string) =>
      scrollRef.current?.querySelector<HTMLElement>(
        `.turn[data-message-id="${CSS.escape(id)}"]`,
      ) ?? null,
    [scrollRef],
  );

  const unlight = useCallback(() => {
    if (litTimerRef.current !== null) window.clearTimeout(litTimerRef.current);
    litTimerRef.current = null;
    setLit(null);
    clearPassage();
  }, []);

  /**
   * Light a passage in a turn, or, where there is no passage to light, mark
   * the turn: the sweep where the whole message was quoted, the ring where
   * the words were not found or it is the question being returned to. For
   * the lit span, then let go. Never two at once: a ring around a tall
   * answer next to its lit words is a second cue saying the same thing.
   */
  const light = useCallback(
    (id: string, el: HTMLElement, range: Range | null, kind: LitKind) => {
      unlight();
      if (range === null) {
        if (kind === "whole") layRailLight(el);
        setLit({ id, kind });
      } else lightPassage(range, el, LIT_MS);
      // Focus lands with the eye: the next Tab goes on from here, not from
      // the band left off-screen. The turn takes focus without a ring of its
      // own (the light or the lit ring is the cue) and without a second
      // scroll.
      el.tabIndex = -1;
      el.focus({ preventScroll: true });
      litTimerRef.current = window.setTimeout(() => {
        litTimerRef.current = null;
        setLit(null);
      }, LIT_MS);
    },
    [unlight],
  );

  /**
   * The scrollTop that rests a range's first line on the landing line, or,
   * with no range, centres the turn's message (the prose or the bubble: what
   * the cue marks, not the rows above it, which could hold a long thought
   * and push the marked words under the fold) with the turn's top kept
   * below the top margin.
   */
  const topFor = useCallback(
    (el: Element, range: Range | null) => () => {
      const scroller = scrollRef.current;
      if (!scroller) return 0;
      const sr = scroller.getBoundingClientRect();
      const rects = range?.getClientRects();
      if (rects !== undefined && rects.length > 0) {
        return scroller.scrollTop + (rects[0].top - sr.top) - scroller.clientHeight * LAND_AT;
      }
      const body = el.querySelector('[data-part="content"], [data-part="bubble-user"]') ?? el;
      const r = body.getBoundingClientRect();
      const top = scroller.scrollTop + (r.top - sr.top);
      const turnTop = scroller.scrollTop + (el.getBoundingClientRect().top - sr.top);
      const centred = top + r.height / 2 - scroller.clientHeight / 2;
      return Math.min(centred, turnTop - scroller.clientHeight * KEEP_TOP);
    },
    [scrollRef],
  );

  const jumpToQuoted = useCallback(
    (sourceId: string, fromId: string | undefined, passage: string | null) => {
      const el = turnOf(sourceId);
      if (el === null) return;
      // The words live in the message's body, never in its chrome (the clock,
      // the actions), so that is where they are looked for. A whole message
      // has no words to look for: it is marked as one.
      const body = el.querySelector('[data-part="content"], [data-part="bubble-user"]') ?? el;
      const range = passage !== null && canLightPassage() ? findPassage(body, passage) : null;
      unlight();
      follow.glideTo(topFor(el, range), () => {
        // The way back exists only where the landing left the end: a source
        // within the near-end band lands stuck, and a way back from there
        // would surface later, on an ordinary scroll up, pointing at a
        // question nobody followed a quote from.
        setReturnTo(follow.stuckRef.current ? null : (fromId ?? null));
        light(sourceId, el, range, passage === null ? "whole" : "ring");
      });
    },
    [turnOf, unlight, follow, topFor, light],
  );

  const returnToQuoting = useCallback(() => {
    if (returnTo === null) return;
    const id = returnTo;
    setReturnTo(null);
    const el = turnOf(id);
    // The question is gone (deleted under the reader): the end is the only
    // place left to go.
    if (el === null) {
      follow.jump();
      return;
    }
    unlight();
    follow.glideTo(topFor(el, null), () => light(id, el, null, "ring"));
  }, [returnTo, turnOf, follow, unlight, topFor, light]);

  // The light and its timer go with the transcript, and with the page.
  useEffect(() => {
    if (litTimerRef.current !== null) window.clearTimeout(litTimerRef.current);
    litTimerRef.current = null;
    clearPassage();
    return () => {
      if (litTimerRef.current !== null) window.clearTimeout(litTimerRef.current);
      litTimerRef.current = null;
      clearPassage();
    };
  }, [paneKey]);

  return { returnTo, lit, jumpToQuoted, returnToQuoting };
}
