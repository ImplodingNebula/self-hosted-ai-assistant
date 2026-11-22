"use client";

import type { CSSProperties, ReactNode } from "react";

/** What kind of work a row is; it sets the row's colour through `--c-step-*`. */
export type StepKind = "thought" | "tool" | "confirm" | "answer" | "error" | "end" | "wait";

/**
 * The timeline turn's one row shape: time | glyph | body.
 *
 * Every row of an assistant turn is one of these, so the rail lines up down
 * the whole turn; a user turn uses the same row with its glyph cell empty (the
 * sheet hides it) and a clock time instead of an elapsed one. `end` marks the
 * closing row, which sits a little lower and ends the rail.
 */
export function Step({
  kind,
  time,
  glyph,
  end,
  part,
  state,
  className,
  children,
}: {
  kind?: StepKind;
  /** The rail's label: `+6s` on a reply's rows, a clock time on a question's. */
  time?: string | null;
  /** What the glyph cell holds: an icon, or the matrix loader while the step runs. */
  glyph?: ReactNode;
  /** The closing row of the turn. */
  end?: boolean;
  /** The `data-part` of the row, for the themes' hooks. */
  part?: string;
  /** A state the themes key on (a tool step's running / ok / danger). */
  state?: string;
  className?: string;
  children?: ReactNode;
}) {
  return (
    <div
      className={`step${end ? " step-end" : ""}${className ? ` ${className}` : ""}`}
      data-kind={kind}
      data-part={part}
      data-state={state}
    >
      <span className="step-time" aria-hidden="true">
        {time ?? ""}
      </span>
      <span className="step-glyph" aria-hidden="true">
        {glyph}
      </span>
      <div className="step-body">{children}</div>
    </div>
  );
}

/** The glyphs the rail and the rows draw; one stroke family on a 16px grid. */
export type IconName =
  | "thought"
  | "search"
  | "globe"
  | "plug"
  | "answer"
  | "check"
  | "gate"
  | "x"
  | "info"
  | "chev"
  | "copy"
  | "reply"
  | "redo"
  | "edit"
  | "file"
  | "up"
  | "cross"
  | "warn"
  | "clip"
  | "pin"
  | "panel"
  | "gear"
  | "logo";

const PATHS: Record<IconName, ReactNode> = {
  // the mark: two arcs of the aurora, drawn by the boot screen and worn by the brand
  logo: (
    <>
      <path pathLength="1" d="M2 11c2-6 4-8 6-8s4 2 6 8" />
      <path pathLength="1" d="M5 12.5c1-3.5 2-4.5 3-4.5s2 1 3 4.5" />
    </>
  ),
  thought: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <circle cx="8" cy="8" r="1.4" />
    </>
  ),
  search: (
    <>
      <circle cx="7" cy="7" r="4.2" />
      <path d="M10.2 10.2L13.5 13.5" />
    </>
  ),
  globe: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M2.5 8h11M8 2.5c1.8 1.6 2.6 3.4 2.6 5.5S9.8 11.9 8 13.5C6.2 11.9 5.4 10.1 5.4 8S6.2 4.1 8 2.5z" />
    </>
  ),
  plug: <path d="M6 2.5v3M10 2.5v3M4.5 5.5h7v2.3a3.5 3.5 0 0 1-7 0zM8 11.3v2.2" />,
  answer: <path d="M3 4.5h10M3 8h10M3 11.5h6" />,
  check: <path d="M3.5 8.5l3 3 6-7" />,
  gate: <path d="M8 2.5l5 2v3.6c0 2.6-2 4.6-5 5.4-3-.8-5-2.8-5-5.4V4.5z" />,
  x: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M6 6l4 4M10 6l-4 4" />
    </>
  ),
  info: (
    <>
      <circle cx="8" cy="8" r="5.5" />
      <path d="M8 7.4v3.2M8 5.2v.2" />
    </>
  ),
  chev: <path d="M4 6l4 4 4-4" />,
  copy: (
    <>
      <rect x="5.5" y="5.5" width="8" height="8" rx="1.8" />
      <path d="M10.5 5.5V4.3a1.8 1.8 0 0 0-1.8-1.8H4.3a1.8 1.8 0 0 0-1.8 1.8v4.4a1.8 1.8 0 0 0 1.8 1.8h1.2" />
    </>
  ),
  reply: (
    <>
      <path d="M6.5 3.8L2.8 7.5l3.7 3.7" />
      <path d="M2.8 7.5h6.4a3.8 3.8 0 0 1 3.8 3.8v1.2" />
    </>
  ),
  redo: (
    <>
      <path d="M13.2 8.6A5.3 5.3 0 1 1 12.4 5" />
      <path d="M13.3 2.2v3.2h-3.2" />
    </>
  ),
  edit: (
    <>
      <path d="M11.3 2.7l2 2L5.5 12.5 3 13l.5-2.5z" />
      <path d="M9.8 4.2l2 2" />
    </>
  ),
  file: (
    <>
      <path d="M4 2.5h5l3 3v8H4z" />
      <path d="M9 2.5v3h3" />
    </>
  ),
  up: <path d="M8 13V3.5M4 7.2l4-4 4 4" />,
  clip: (
    <path d="M13 7.5l-5 5a3.2 3.2 0 0 1-4.5-4.5l5.3-5.3a2.1 2.1 0 0 1 3 3L6.6 10.9a1 1 0 0 1-1.5-1.5L9.8 4.8" />
  ),
  pin: <path d="M6 2.5h4M7 2.7v3.6L4.8 8.9h6.4L9 6.3V2.7M8 8.9v4.1" />,
  panel: (
    <>
      <rect x="2" y="3" width="12" height="10" rx="2" />
      <path d="M6.2 3v10" />
    </>
  ),
  gear: (
    <>
      <path d="M2.5 5h6M11.5 5h2M2.5 11h2M7.5 11h6" />
      <circle cx="10" cy="5" r="1.5" />
      <circle cx="6" cy="11" r="1.5" />
    </>
  ),
  cross: <path d="M4.5 4.5l7 7M11.5 4.5l-7 7" />,
  warn: (
    <>
      <path d="M8 2.5l6 10.5H2z" />
      <path d="M8 7v3M8 12v.2" />
    </>
  ),
};

/**
 * One of the rail's glyphs. Decorative: the row's words carry the meaning,
 * and every icon-only control names itself in its own markup.
 *
 * suppressHydrationWarning on the svg: dark-mode browser extensions stamp
 * stroke attributes onto inline SVGs before React hydrates (the same cause as
 * the html suppression in the root layout).
 */
export function Icon({ name, className }: { name: IconName; className?: string }) {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className={className ?? "i"}
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {PATHS[name]}
    </svg>
  );
}

/** The dots' delay orders, per variant (in sixteenths of the cycle). */
const RING = [1, 2, 7, 11, 14, 13, 8, 4];
const TWINKLE = [7, 2, 11, 5, 14, 9, 0, 12, 3, 15, 6, 10, 13, 1, 8, 4];
const CORNERS = new Set([0, 3, 12, 15]);
const CYCLE_MS = 1200;

/** A dot's inline style: the delay rides a custom property the sheet reads. */
function dotStyle(delayMs: number | null, hidden: boolean): CSSProperties {
  const style: Record<string, string> = {};
  if (hidden) style.visibility = "hidden";
  else if (delayMs === null) style.animation = "none";
  else style["--d"] = String(delayMs);
  return style as CSSProperties;
}

/**
 * The matrix loader: sixteen 2px dots on one shared pulse, each with its own
 * delay, so the field reads as motion without anything travelling. `scan`
 * sweeps column by column, `twinkle` lights at random, `orbit` runs the ring
 * (with the corners dropped, so it reads round). The colour rides `--mx`.
 */
export function MatrixLoader({
  variant,
  color,
  rounded,
  className,
}: {
  variant: "scan" | "twinkle" | "orbit";
  /** A CSS colour the dots light in; the sheet's default is the muted ink. */
  color?: string;
  /** Drop the four corners, for a loader that sits where a round glyph did. */
  rounded?: boolean;
  className?: string;
}) {
  const dots: ReactNode[] = [];
  for (let i = 0; i < 16; i += 1) {
    let delay: number | null = 0;
    if (variant === "scan") delay = Math.round((i % 4) * (CYCLE_MS / 10));
    else if (variant === "twinkle") delay = Math.round(TWINKLE[i] * (CYCLE_MS / 16));
    else {
      const k = RING.indexOf(i);
      delay = k >= 0 ? Math.round(k * (CYCLE_MS / 8)) : null;
    }
    dots.push(<i key={i} style={dotStyle(delay, rounded === true && CORNERS.has(i))} />);
  }
  const hostStyle = color ? ({ "--mx": color } as CSSProperties) : undefined;
  return (
    <span
      className={`t-matrix${className ? ` ${className}` : ""}`}
      data-variant={variant}
      style={hostStyle}
      aria-hidden="true"
    >
      {dots}
    </span>
  );
}
