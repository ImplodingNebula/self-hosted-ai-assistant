"use client";

import { useEffect, useId, useRef, useState } from "react";

import { useFloaterNudge } from "@/hooks/use-floater-nudge";
import type { ContextUsageData } from "@/lib/types";

/** Props for {@link ContextMeter}. */
export interface ContextMeterProps {
  /**
   * THIS conversation's window fill - the last completed turn's reading, or
   * the eager pre-turn estimate (`estimated: true`) fetched on pane load and
   * model switch. Null when neither exists yet: a pane with no conversation
   * has nothing to measure.
   */
  usage: ContextUsageData | null;
}

/** Warn from here up: the window is filling and the oldest turns are near the edge. */
const WARN_AT = 75;
/** And from here up, say it louder - eviction is imminent rather than distant. */
const HIGH_AT = 90;

/** Radius of the arc's centreline inside the ring's 16px viewBox. */
const RING_RADIUS = 6.5;
/** Full-circle arc length; the fill offsets against this to draw a fraction. */
const RING_CIRCUMFERENCE = 2 * Math.PI * RING_RADIUS;

/**
 * The numbers, hedged because both are the backend's estimate, not a count.
 *
 * Tense follows the reading's origin: a completed turn measured what it SENT
 * ("used last turn"), while the eager pre-turn estimate describes what the
 * next prompt already opens with ("in use right now") - claiming the past
 * tense for a turn that never ran would invent a turn.
 */
function summary(usage: ContextUsageData): string {
  const used = usage.prompt_tokens.toLocaleString();
  const total = usage.num_ctx.toLocaleString();
  return usage.estimated
    ? `About ${used} of ${total} tokens in use right now.`
    : `About ${used} of ${total} tokens used last turn.`;
}

/**
 * What a full window actually does. Said everywhere the numbers are, because
 * the consequence people guess at ("it will refuse") is not the one that
 * happens.
 */
const CONSEQUENCE = "When the window fills, the oldest messages stop being sent.";

/**
 * The action that is not here yet, named so a full window is known to have
 * a remedy coming rather than read as a wall. One quiet line under the card.
 */
const LATER = "Later: compress the older turns to make room. For now a new chat starts clear.";

/**
 * How full the model's context window was on this conversation's last
 * exchange: a quiet ring in the composer's right dock that fills with the
 * window, with the numbers one click away in a drop-up card.
 *
 * It answers the one question a long thread eventually raises - "is it still
 * reading all of this?" - before the answer becomes visible as forgetting.
 * A ring rather than the label it first shipped as (owner decision
 * 2026-08-04): the percentage only matters once it is high, so the quiet
 * state earns no standing text, and the card puts the full sentence where
 * touch can reach it - a native tooltip never is. The ring's geometry never
 * changes with its level, so the escalation cannot reflow the row.
 *
 * Both numbers are the backend's own estimate (see {@link ContextUsageData}),
 * so the card hedges with "about" rather than quoting them as a count. The
 * percentage is what the ring draws, because the ratio is the part that
 * survives the estimate.
 */
export function ContextMeter({ usage }: ContextMeterProps) {
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const popId = useId();
  const popRef = useRef<HTMLDivElement>(null);
  const nudge = useFloaterNudge(popRef, open);

  /** Close and hand focus back to the trigger (the keyboard's way out). */
  const close = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };

  useEffect(() => {
    if (!open) return;
    // An outside click closes WITHOUT pulling focus back: that click has
    // already aimed the caret somewhere, and taking it would be a theft.
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  // A pane switch clears the reading while this component stays mounted; the
  // card must not survive into the next conversation and reopen over a reading
  // it never described. Adjusted during render, as the model menu does on
  // disable: an effect would paint the orphaned card for a frame first.
  const unreported = usage === null || usage.num_ctx <= 0;
  if (open && unreported) setOpen(false);

  // No reading yet, or a window the backend could not name: the meter has
  // nothing to report, and dividing by an unknown window would report noise.
  if (unreported) return null;

  const raw = (usage.prompt_tokens / usage.num_ctx) * 100;
  // A real turn always sends a system prompt, so a rounded-down "0%" would
  // claim the window is untouched. Floor at 1% whenever anything was sent -
  // a 256K window makes that the ordinary case, not an edge one. The literal
  // zero is left alone rather than floored or hidden: a turn that sent nothing
  // is not a real turn, and "0%" is the honest thing to say if one appears.
  const percent =
    usage.prompt_tokens > 0 ? Math.min(100, Math.max(1, Math.round(raw))) : 0;
  const level = percent >= HIGH_AT ? "high" : percent >= WARN_AT ? "warn" : undefined;

  return (
    <div
      ref={rootRef}
      data-part="context-meter"
      data-level={level}
      className="context-dock"
    >
      <button
        ref={triggerRef}
        type="button"
        aria-expanded={open}
        aria-controls={open ? popId : undefined}
        // The whole story, so a screen reader never needs the card open.
        aria-label={`Context ${percent}% full. ${summary(usage)} ${CONSEQUENCE}`}
        onClick={() => (open ? close() : setOpen(true))}
        className="context-ring"
        data-open={open ? "true" : undefined}
      >
        <svg viewBox="0 0 16 16" width="16" height="16" aria-hidden="true">
          <circle className="ring-track" cx="8" cy="8" r={RING_RADIUS} />
          <circle
            className="ring-fill"
            cx="8"
            cy="8"
            r={RING_RADIUS}
            strokeDasharray={RING_CIRCUMFERENCE}
            strokeDashoffset={RING_CIRCUMFERENCE * (1 - percent / 100)}
            // From 12 o'clock, the way every dial is read.
            transform="rotate(-90 8 8)"
          />
        </svg>
        {/* The figure appears only once the window is filling: the quiet
            state earns no standing text, and from warn up the number is the
            thing to read at a glance. */}
        {level !== undefined && <span className="ring-pct">{percent}%</span>}
      </button>

      {open && (
        <div
          id={popId}
          ref={popRef}
          // A named landmark, not a bare div: accessible-name computation
          // skips generic roles, and aria-controls should point at a node
          // that can introduce itself. Not a dialog - focus never moves in.
          role="region"
          aria-label="Context window"
          className="context-pop"
          // Frost the fill so a translucent theme does not let the transcript
          // behind the drop-up bleed through it; `none` in flat themes.
          style={{
            backdropFilter: "var(--c-menu-backdrop)",
            WebkitBackdropFilter: "var(--c-menu-backdrop)",
            translate: nudge !== 0 ? `${nudge}px` : undefined,
          }}
        >
          <div className="context-pop-head">
            <span>Context window</span>
            <span>{percent}%</span>
          </div>
          <div className="context-bar">
            <span style={{ width: `${percent}%` }} />
          </div>
          <p>{summary(usage)}</p>
          <p>{CONSEQUENCE}</p>
          <p className="context-later">{LATER}</p>
        </div>
      )}
    </div>
  );
}
