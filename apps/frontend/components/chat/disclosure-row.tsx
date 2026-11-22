"use client";

import { type ReactNode, useId } from "react";

import { Icon } from "@/components/chat/step";

/**
 * A quiet line that opens in place: the thinking row and a tool step.
 *
 * The panel is a grid row that grows from `0fr` to `1fr`, so opening never
 * mounts anything and the thread's height eases rather than jumps. With no
 * `onToggle` the summary is a plain line and there is no panel at all: a
 * disclosure that discloses nothing would still be announced as expandable.
 */
export function DisclosureRow({
  open,
  onToggle,
  summary,
  children,
  className,
}: {
  open: boolean;
  /** Absent when there is nothing to open; the row is then a plain line. */
  onToggle?: () => void;
  summary: ReactNode;
  children?: ReactNode;
  className?: string;
}) {
  const panelId = useId();
  if (onToggle === undefined) {
    return (
      <div className={`disc-btn disc-plain${className ? ` ${className}` : ""}`}>{summary}</div>
    );
  }
  return (
    <>
      <button
        type="button"
        className={`disc-btn${className ? ` ${className}` : ""}`}
        aria-expanded={open}
        aria-controls={panelId}
        onClick={onToggle}
      >
        {summary}
        <Icon name="chev" className="i chev" />
      </button>
      {/* The sheet hides a closed panel (visibility), which takes it out of
          the tab order and the accessibility tree with it. */}
      <div className="disc-panel" id={panelId}>
        <div>{children}</div>
      </div>
    </>
  );
}
