"use client";

import { useEffect, useLayoutEffect, useRef } from "react";
import type { CSSProperties } from "react";

/** Cadence of the typewriter, milliseconds per tick. */
const TICK_MS = 28;
/**
 * Tick budget for a full reveal: longer titles type more characters per tick,
 * so every name lands in about a second regardless of length.
 */
const MAX_TICKS = 40;
/**
 * How long the finished name holds before the reveal hands back to plain
 * text. Must stay comfortably above the 200ms settle animations the themes
 * schedule at the end of the sweep (themes/aurora.css), or the swap to
 * static text would cut the settle short.
 */
const DONE_HOLD_MS = 260;

/** Props for {@link AnimatedTitle}. */
export interface AnimatedTitleProps {
  /** The full title being revealed. */
  text: string;
  /**
   * Fires once the reveal has played out (immediately under reduced motion).
   * Exactly ONE surface passes it - the sidebar row, which always exists while
   * a reveal plays - so one logical reveal has one owner; a mirror instance
   * (the topbar) omits it and simply follows.
   */
  onDone?: () => void;
}

/**
 * Types a just-generated conversation title into place, caret and all.
 *
 * Purely presentational: the parent decides which surface is revealing (the
 * title watch in page.tsx) and swaps back to plain text once `onDone` fires.
 * The typed fragment is written to the DOM directly rather than held as
 * state - it is aria-hidden decor beside an sr-only copy, so assistive tech
 * gets the full name from the first frame - and prefers-reduced-motion skips
 * the type-out entirely: the title simply appears.
 *
 * Two stacked REAL spans carry the paint: `.reveal-ink` types the fragment
 * as plain text, and `.reveal-glint` (hidden unless a theme styles it)
 * duplicates it for decorative treatments like aurora's gradient sweep.
 * Both are elements, never pseudo-elements: an animated ::before whose
 * content is rewritten every tick couples the animation to the pseudo box's
 * lifecycle, where element-level animations demonstrably survive text-node
 * churn. Every settle is scheduled in CSS by animation-delay off
 * --reveal-dur, so no attribute flips or transitions race the typewriter.
 */
export function AnimatedTitle({ text, onDone }: AnimatedTitleProps) {
  const inkRef = useRef<HTMLSpanElement>(null);
  const glintRef = useRef<HTMLSpanElement>(null);
  const caretRef = useRef<HTMLSpanElement>(null);
  // The parent's onDone identity can change between renders (inline lambdas);
  // reading it through a ref keeps the animation effect from restarting.
  const doneRef = useRef<(() => void) | undefined>(onDone);
  useEffect(() => {
    doneRef.current = onDone;
  });

  const step = Math.max(1, Math.ceil(text.length / MAX_TICKS));
  // Handed to the theme sheet as --reveal-dur: the TYPE-OUT only, not the
  // hold - the sweep finishes as the last character lands, and the settle
  // animations are delayed by this same value so they begin right there.
  const sweepMs = Math.ceil(text.length / step) * TICK_MS;

  // A layout effect, so a reduced-motion mount shows the whole name before
  // the first paint instead of flashing an empty row.
  useLayoutEffect(() => {
    const ink = inkRef.current;
    const glint = glintRef.current;
    const caret = caretRef.current;
    if (ink === null || glint === null || caret === null) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      ink.textContent = text;
      glint.textContent = text;
      caret.style.display = "none";
      const skip = window.setTimeout(() => doneRef.current?.(), 0);
      return () => window.clearTimeout(skip);
    }
    ink.textContent = "";
    glint.textContent = "";
    caret.style.display = "";
    let at = 0;
    let hold: number | null = null;
    const timer = window.setInterval(() => {
      at = Math.min(text.length, at + step);
      const shown = text.slice(0, at);
      ink.textContent = shown;
      glint.textContent = shown;
      if (at >= text.length) {
        window.clearInterval(timer);
        caret.style.display = "none";
        hold = window.setTimeout(() => doneRef.current?.(), DONE_HOLD_MS);
      }
    }, TICK_MS);
    return () => {
      window.clearInterval(timer);
      if (hold !== null) window.clearTimeout(hold);
    };
    // step derives from text; both are listed for the exhaustive-deps rule.
  }, [text, step]);

  return (
    <>
      <span
        // Keyed on the text: a mid-reveal title change must restart the CSS
        // animations along with the typewriter, and a running CSS animation
        // keeps its elapsed time across a duration change - a fresh node
        // does not.
        key={text}
        className="title-reveal"
        aria-hidden="true"
        style={{ "--reveal-dur": `${sweepMs}ms` } as CSSProperties}
      >
        <span ref={inkRef} className="reveal-ink" />
        <span ref={glintRef} className="reveal-glint" />
      </span>
      <span ref={caretRef} className="reveal-caret" aria-hidden="true" />
      <span className="sr-only">{text}</span>
    </>
  );
}
