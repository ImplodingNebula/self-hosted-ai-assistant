"use client";

import { type RefObject, useCallback, useEffect, useRef, useState } from "react";

/** Within this many px of the end, the reader counts as at the end. */
const NEAR_END_PX = 80;

/** How long a reader's signal (a wheel notch, a key) explains the scrolls after it. */
const WHEEL_INTENT_MS = 400;
const TOUCH_INTENT_MS = 2500;
const POINTER_INTENT_MS = 1500;

/** The jump's glide: 420ms plus a share of the distance, capped. */
const GLIDE_BASE_MS = 420;
const GLIDE_PER_PX = 0.35;
const GLIDE_MAX_MS = 1100;

/** The fraction of the remaining gap the chase closes per frame. */
const CHASE_RATE = 0.3;

/** The keys that mean "take me up", whatever the distance. */
const LEAVE_KEYS = new Set(["ArrowUp", "PageUp", "Home"]);

/** Ease in and out (a cubic), so a glide starts and lands softly. */
function easeInOut(p: number): number {
  return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
}

/** Whether a key event belongs to a field, where it edits rather than scrolls. */
function inEditable(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return (
    target.tagName === "INPUT" ||
    target.tagName === "TEXTAREA" ||
    target.tagName === "SELECT" ||
    target.isContentEditable
  );
}

/** What the hook hands the page. */
export interface FollowStream {
  /** Whether the reader has left the end (the jump floater shows). */
  away: boolean;
  /** Something arrived while the reader was away (the floater's dot). */
  unseen: boolean;
  /** Go to the end: a glide by distance, or a cut under reduced motion. */
  jump: () => void;
  /** Re-stick to the end without ceremony: on a send, on answering a confirmation. */
  pin: () => void;
  /**
   * Glide to a place in the thread and stay there: once it lands the reader
   * is away (the floater shows) unless the place is within the near-end
   * distance, where following resumes. `target` is read every frame, so
   * content growing under the glide does not strand it; `onArrive` runs on
   * a landing, never when the reader interrupts the glide.
   */
  glideTo: (target: () => number, onArrive?: () => void) => void;
  /** Whether the view is stuck to the end right now (read synchronously). */
  stuckRef: RefObject<boolean>;
}

/**
 * Following the stream.
 *
 * Following is a distance, never a direction: "at the end" means within 80px
 * of it, and only a scroll the reader made (a wheel, touch, pointer or key
 * signal just before it) is measured as them leaving. Every other scroll
 * event - anchoring, a panel collapsing, the notice closing, the thread's own
 * chase - is followed. Following eases toward the end over a few frames so
 * streamed lines land without a snap; the jump glides there and ends at the
 * end even if the content grew on the way. A wheel notch upward or a paging
 * key is intent to leave whatever the distance. Reduced motion jumps.
 *
 * A resize observer on the thread and on its content column drives the
 * follow, so anything that changes height - streamed text, a panel opening,
 * the docked notice - is followed frame by frame, and a live turn that grows
 * while the reader is away marks the floater. The column is re-observed
 * whenever the thread's children change (the empty pane and the transcript
 * are different elements).
 *
 * @param scrollRef - The scrolling thread element.
 * @param busyRef - Whether a turn is in flight (new content counts as unseen).
 */
export function useFollowStream(
  scrollRef: RefObject<HTMLElement | null>,
  busyRef: RefObject<boolean>,
): FollowStream {
  const [away, setAway] = useState(false);
  const [unseen, setUnseen] = useState(false);
  const stuckRef = useRef(true);
  const glidingRef = useRef<number | null>(null);
  const glideFrameRef = useRef<number | null>(null);
  const chaseRef = useRef<number | null>(null);
  const chaseLaterRef = useRef<number | null>(null);
  const intentUntilRef = useRef(0);
  // What a glide in flight is for: the end (it ends stuck, following) or a
  // place in the thread (it ends where it lands), and what to run on landing.
  const goalRef = useRef<"end" | "place">("end");
  const arriveRef = useRef<(() => void) | null>(null);
  // A paging key pressed: the scroll it causes, if any, is leaving. Keys are
  // heard on the document (focus rests on the body after a click on the
  // prose), where the browser may or may not route them to the thread, so
  // the key alone decides nothing - only the scroll that follows it does.
  const pendingLeaveRef = useRef(false);
  const seenHeightRef = useRef(0);

  const gap = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return 0;
    return el.scrollHeight - el.scrollTop - el.clientHeight;
  }, [scrollRef]);

  const syncAway = useCallback(() => {
    const isAway = !stuckRef.current && glidingRef.current === null;
    setAway((prev) => (prev === isAway ? prev : isAway));
    if (!isAway) setUnseen(false);
  }, []);

  const reduceMotion = () => window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /**
   * Ease toward the end over a few frames; keep chasing while it keeps
   * growing. Inside a reader's intent window the chase stands down and tries
   * again when the window closes, so a click on a button at the end of a
   * paused stream does not leave the view short of the end for good.
   */
  const chaseEnd = useCallback(function chase() {
    const el = scrollRef.current;
    if (!el) return;
    if (reduceMotion()) {
      el.scrollTop = el.scrollHeight;
      return;
    }
    if (chaseRef.current !== null) return;
    const step = () => {
      const target = el.scrollHeight - el.clientHeight;
      const d = target - el.scrollTop;
      if (!stuckRef.current || glidingRef.current !== null) {
        chaseRef.current = null;
        return;
      }
      const wait = intentUntilRef.current - performance.now();
      if (wait > 0) {
        chaseRef.current = null;
        if (chaseLaterRef.current !== null) window.clearTimeout(chaseLaterRef.current);
        chaseLaterRef.current = window.setTimeout(() => {
          chaseLaterRef.current = null;
          chase();
        }, wait + 16);
        return;
      }
      if (Math.abs(d) < 0.6) {
        chaseRef.current = null;
        el.scrollTop = target;
        return;
      }
      el.scrollTop += d * CHASE_RATE;
      chaseRef.current = requestAnimationFrame(step);
    };
    chaseRef.current = requestAnimationFrame(step);
  }, [scrollRef]);

  /** The thread's height changed: follow if stuck, else note what arrived. */
  const follow = useCallback(() => {
    const el = scrollRef.current;
    if (!el) return;
    const grew = seenHeightRef.current !== 0 && el.scrollHeight > seenHeightRef.current + 4;
    seenHeightRef.current = el.scrollHeight;
    if (glidingRef.current !== null) return;
    if (stuckRef.current) chaseEnd();
    else if (grew && busyRef.current && gap() > NEAR_END_PX) setUnseen(true);
  }, [scrollRef, busyRef, chaseEnd, gap]);

  /** Stop a glide in flight: its timer and its frame loop. */
  const cancelGlide = useCallback(() => {
    if (glidingRef.current !== null) window.clearTimeout(glidingRef.current);
    glidingRef.current = null;
    if (glideFrameRef.current !== null) cancelAnimationFrame(glideFrameRef.current);
    glideFrameRef.current = null;
    arriveRef.current = null;
  }, []);

  /**
   * A glide that ran its course: one for the end ends stuck, one for a place
   * ends where it is (stuck only if that is the end) and runs its arrival.
   */
  const endGlide = useCallback(() => {
    if (glidingRef.current === null) return;
    const arrive = arriveRef.current;
    cancelGlide();
    if (goalRef.current === "end") {
      stuckRef.current = true;
      follow();
    } else {
      stuckRef.current = gap() < NEAR_END_PX;
    }
    syncAway();
    arrive?.();
  }, [cancelGlide, follow, gap, syncAway]);

  const pinTo = useCallback(
    (smooth: boolean) => {
      const el = scrollRef.current;
      stuckRef.current = true;
      if (el && smooth && !reduceMotion() && gap() > NEAR_END_PX) {
        // One glide at a time: a second jump restarts from where the view is.
        cancelGlide();
        goalRef.current = "end";
        const from = el.scrollTop;
        const dur = Math.min(GLIDE_MAX_MS, GLIDE_BASE_MS + gap() * GLIDE_PER_PX);
        const t0 = performance.now();
        glidingRef.current = window.setTimeout(endGlide, dur + 120);
        const step = (now: number) => {
          if (glidingRef.current === null) return;
          const p = Math.min(1, (now - t0) / dur);
          el.scrollTop = from + (el.scrollHeight - el.clientHeight - from) * easeInOut(p);
          if (p < 1) glideFrameRef.current = requestAnimationFrame(step);
          else endGlide();
        };
        glideFrameRef.current = requestAnimationFrame(step);
      } else if (el) {
        el.scrollTop = el.scrollHeight;
      }
      syncAway();
    },
    [scrollRef, gap, cancelGlide, endGlide, syncAway],
  );

  const jump = useCallback(() => pinTo(true), [pinTo]);
  const pin = useCallback(() => pinTo(false), [pinTo]);

  const glideTo = useCallback(
    (target: () => number, onArrive?: () => void) => {
      const el = scrollRef.current;
      if (!el) return;
      cancelGlide();
      const clamp = () => Math.max(0, Math.min(el.scrollHeight - el.clientHeight, target()));
      const from = el.scrollTop;
      if (reduceMotion() || Math.abs(clamp() - from) < 1) {
        el.scrollTop = clamp();
        stuckRef.current = gap() < NEAR_END_PX;
        syncAway();
        onArrive?.();
        return;
      }
      goalRef.current = "place";
      arriveRef.current = onArrive ?? null;
      const dur = Math.min(GLIDE_MAX_MS, GLIDE_BASE_MS + Math.abs(clamp() - from) * GLIDE_PER_PX);
      const t0 = performance.now();
      glidingRef.current = window.setTimeout(endGlide, dur + 120);
      const step = (now: number) => {
        if (glidingRef.current === null) return;
        const p = Math.min(1, (now - t0) / dur);
        el.scrollTop = from + (clamp() - from) * easeInOut(p);
        if (p < 1) glideFrameRef.current = requestAnimationFrame(step);
        else endGlide();
      };
      glideFrameRef.current = requestAnimationFrame(step);
    },
    [scrollRef, gap, cancelGlide, endGlide, syncAway],
  );

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    seenHeightRef.current = el.scrollHeight;
    const onScroll = () => {
      if (glidingRef.current !== null) {
        if (goalRef.current === "end" && gap() < 2) endGlide();
        return;
      }
      // Only a scroll the reader made (a signal just before it) is measured;
      // every other scroll event is the page moving itself, and is followed.
      if (performance.now() < intentUntilRef.current) {
        stuckRef.current = pendingLeaveRef.current ? false : gap() < NEAR_END_PX;
        pendingLeaveRef.current = false;
        syncAway();
        return;
      }
      if (el.scrollHeight !== seenHeightRef.current) follow();
      syncAway();
    };
    // Chromium reports a scroll's end after every programmatic step, so a
    // glide's own frames would end it early; only an end AT the end counts.
    // The glide's timer is the backstop for a glide that never gets there.
    const onScrollEnd = () => {
      if (glidingRef.current !== null && goalRef.current === "end" && gap() < 2) endGlide();
    };
    /** The reader taking over cancels a glide; `leave` is intent to leave the end. */
    const takeOver = (leave: boolean, holdMs: number) => {
      intentUntilRef.current = performance.now() + holdMs;
      pendingLeaveRef.current = false;
      cancelGlide();
      stuckRef.current = leave ? false : gap() < NEAR_END_PX;
      syncAway();
    };
    const onWheel = (e: WheelEvent) => takeOver(e.deltaY < 0, WHEEL_INTENT_MS);
    // A key opens an intent window; a paging key up also marks the scroll
    // that follows as leaving. A field keeps its own keys.
    const onKeyDown = (e: KeyboardEvent) => {
      if (inEditable(e.target)) return;
      const scrolls = e.key === " " || e.key === "PageDown" || e.key === "End" || e.key === "ArrowDown";
      if (!scrolls && !LEAVE_KEYS.has(e.key)) return;
      intentUntilRef.current = performance.now() + WHEEL_INTENT_MS;
      pendingLeaveRef.current = LEAVE_KEYS.has(e.key);
      cancelGlide();
    };
    const onTouchStart = () => takeOver(false, TOUCH_INTENT_MS);
    const onTouchMove = () => {
      intentUntilRef.current = performance.now() + TOUCH_INTENT_MS;
    };
    const onPointerDown = () => takeOver(false, POINTER_INTENT_MS);
    const onPointerMove = (e: PointerEvent) => {
      if (e.buttons) intentUntilRef.current = performance.now() + POINTER_INTENT_MS;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    el.addEventListener("scrollend", onScrollEnd);
    el.addEventListener("wheel", onWheel, { passive: true });
    document.addEventListener("keydown", onKeyDown);
    el.addEventListener("touchstart", onTouchStart, { passive: true });
    el.addEventListener("touchmove", onTouchMove, { passive: true });
    el.addEventListener("pointerdown", onPointerDown, { passive: true });
    el.addEventListener("pointermove", onPointerMove, { passive: true });
    // The thread's own box never changes with its content; the column inside
    // it does, and it is a different element in the empty pane and in a
    // transcript, so it is re-observed whenever the thread's children change.
    const ro = new ResizeObserver(follow);
    const observeColumn = () => {
      ro.disconnect();
      ro.observe(el);
      for (const child of el.children) ro.observe(child);
    };
    observeColumn();
    const mo = new MutationObserver(observeColumn);
    mo.observe(el, { childList: true });
    return () => {
      el.removeEventListener("scroll", onScroll);
      el.removeEventListener("scrollend", onScrollEnd);
      el.removeEventListener("wheel", onWheel);
      document.removeEventListener("keydown", onKeyDown);
      el.removeEventListener("touchstart", onTouchStart);
      el.removeEventListener("touchmove", onTouchMove);
      el.removeEventListener("pointerdown", onPointerDown);
      el.removeEventListener("pointermove", onPointerMove);
      mo.disconnect();
      ro.disconnect();
      if (chaseRef.current !== null) cancelAnimationFrame(chaseRef.current);
      if (chaseLaterRef.current !== null) window.clearTimeout(chaseLaterRef.current);
      cancelGlide();
    };
  }, [scrollRef, gap, follow, endGlide, cancelGlide, syncAway]);

  return { away, unseen, jump, pin, glideTo, stuckRef };
}
