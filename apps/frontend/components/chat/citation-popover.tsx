"use client";

import Link from "next/link";
import {
  useId,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
  type RefObject,
} from "react";

import {
  useCitationHighlight,
  useDismissWhenPinned,
} from "@/components/chat/citation-context";
import {
  attachmentIdFromDocId,
  citationAction,
  citationTypeLabel,
  type CitationActionKind,
} from "@/lib/citations";
import type { CitationData } from "@/lib/types";

/**
 * One glyph per citation kind, read off `source_type`. Purely decorative
 * (`aria-hidden`) - {@link citationTypeLabel} is what a screen reader gets,
 * so a source this platform has not seen yet still announces something
 * instead of the icon silently falling back to a blank shape.
 *
 * `suppressHydrationWarning` on the svg and its shapes: dark-mode browser
 * extensions stamp stroke/fill attributes onto inline SVGs before React
 * hydrates (see the same note in `attachment-tray.tsx`).
 */
function CitationIcon({ sourceType }: { sourceType: string | null }) {
  const common = {
    suppressHydrationWarning: true,
    viewBox: "0 0 16 16",
    className: "citation-icon",
    fill: "none",
    stroke: "currentColor",
    strokeWidth: 1.5,
    strokeLinecap: "round" as const,
    strokeLinejoin: "round" as const,
    "aria-hidden": true,
  };
  switch (sourceType) {
    case "web":
      return (
        <svg {...common}>
          <circle suppressHydrationWarning cx="8" cy="8" r="6" />
          <path suppressHydrationWarning d="M2 8h12M8 2a9 9 0 0 1 0 12M8 2a9 9 0 0 0 0 12" />
        </svg>
      );
    case "records":
      return (
        <svg {...common}>
          <path suppressHydrationWarning d="M2.5 3.5c1.8-.9 3.9-.9 5.5.3 1.6-1.2 3.7-1.2 5.5-.3v9c-1.8-.9-3.9-.9-5.5.3-1.6-1.2-3.7-1.2-5.5-.3z" />
          <path suppressHydrationWarning d="M8 3.8v9" />
        </svg>
      );
    case "mcp_records":
      return (
        <svg {...common}>
          <path suppressHydrationWarning d="M10.4 2.6a3 3 0 0 0-3.9 3.9L2.7 10.3a1.4 1.4 0 0 0 2 2l3.8-3.8a3 3 0 0 0 3.9-3.9l-2 2-1.4-1.4z" />
        </svg>
      );
    case "upload":
      return (
        <svg {...common}>
          <path suppressHydrationWarning d="M11 5.2 6.6 9.6a2 2 0 0 0 2.8 2.8l4.4-4.4a3.3 3.3 0 0 0-4.7-4.7L4.7 7.7a4.6 4.6 0 0 0 6.5 6.5" />
        </svg>
      );
    default:
      return (
        <svg {...common}>
          <circle suppressHydrationWarning cx="8" cy="8" r="6" />
        </svg>
      );
  }
}

/** The small arrow marking an action that leaves the conversation in a new tab. */
function ExternalIcon() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="citation-action-icon"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.5"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M6.5 3.5h6v6M12.5 3.5 6 10M4 5.5v6.5h6.5" />
    </svg>
  );
}

/**
 * The popover's action row for one citation, derived exactly per the
 * citations-wave contract's table (`citationAction`). Absent when the
 * citation names none of the known kinds, or when the one field the action
 * needs (a parseable attachment id, off `doc_id`) is not there - a citation
 * with no usable action still shows its icon, label and passage above.
 */
function CitationActionRow({
  citation,
  action,
  onViewToolStep,
  onDownloadAttachment,
}: {
  citation: CitationData;
  action: CitationActionKind;
  onViewToolStep: () => void;
  onDownloadAttachment: (attachmentId: string) => void;
}) {
  switch (action) {
    case "open_page":
      // Guaranteed by citationAction (this arm is only chosen when url is set),
      // but the type stays nullable - re-checked rather than asserted.
      return citation.url ? (
        <a href={citation.url} target="_blank" rel="noopener noreferrer" className="citation-action">
          Open page
          <ExternalIcon />
        </a>
      ) : null;
    case "view_knowledge":
      // A new tab: clicking through must not throw away whatever the reader
      // was mid-typing in the composer behind this popover.
      return (
        <Link href="/console/knowledge" target="_blank" rel="noopener noreferrer" className="citation-action">
          View in Knowledge
          <ExternalIcon />
        </Link>
      );
    case "view_tool_step":
      return (
        <button type="button" className="citation-action" onClick={onViewToolStep}>
          View tool step
        </button>
      );
    case "download_attachment": {
      const attachmentId = attachmentIdFromDocId(citation.doc_id);
      if (!attachmentId) return null;
      return (
        <button
          type="button"
          className="citation-action"
          onClick={() => onDownloadAttachment(attachmentId)}
        >
          Download file
        </button>
      );
    }
    default:
      return null;
  }
}

/**
 * One citation's full popover content: icon, label, passage, and its action row.
 *
 * Spans, not divs/paragraphs, because a marker's popover renders INSIDE the
 * answer's `<p>` (the trigger wrapper is inline) and only phrasing content is
 * legal there - a div would be real invalid nesting that React flags as a
 * hydration hazard. The classes set their own `display`, so the tag carries no
 * layout.
 */
function CitationEntry({
  citation,
  onViewToolStep,
  onDownloadAttachment,
}: {
  citation: CitationData;
  onViewToolStep: () => void;
  onDownloadAttachment: (attachmentId: string) => void;
}) {
  const action = citationAction(citation);
  return (
    <span className="citation-entry" data-part="citation-entry">
      <span className="citation-entry-head">
        <CitationIcon sourceType={citation.source_type} />
        <span className="citation-entry-type">{citationTypeLabel(citation.source_type)}</span>
        {citation.index != null && <b className="citation-entry-index">{citation.index}</b>}
      </span>
      <span className="citation-entry-label">{citation.source}</span>
      {citation.snippet !== "" && <span className="citation-entry-snippet">{citation.snippet}</span>}
      {action && (
        <CitationActionRow
          citation={citation}
          action={action}
          onViewToolStep={onViewToolStep}
          onDownloadAttachment={onDownloadAttachment}
        />
      )}
    </span>
  );
}

/**
 * Where an open popover should sit so the reader can actually see it: whether
 * it opens downward instead of up, and how far to nudge it horizontally.
 */
interface PopoverPlacement {
  /** Open below the trigger - above would run past the pane's top edge. */
  flippedDown: boolean;
  /** Horizontal nudge in px keeping the card inside the pane's side edges. */
  shiftX: number;
}

/**
 * Measure the mounted popover against the pane that visually bounds it and
 * decide its placement. The default spot (above the trigger, left-aligned:
 * the lines above a marker have been read, the lines below are next) runs
 * off the scrollable thread pane for a trigger near its top or right edge,
 * so: flip below when above lacks room and below has more, and nudge
 * horizontally back inside the edges.
 *
 * Runs as a layout effect so the correction lands before paint - the reader
 * never sees the card jump. Deliberately NOT `position: fixed` or a portal:
 * under aurora an ancestor's `backdrop-filter` makes that ancestor the
 * containing block for fixed descendants (fixed would not escape the pane
 * anyway), and leaving the card in the DOM behind its trigger keeps it the
 * next Tab stop, which the trigger's blur handling relies on.
 */
function usePopoverPlacement(
  ref: RefObject<HTMLElement | null>,
  citationCount: number,
): PopoverPlacement {
  const [placement, setPlacement] = useState<PopoverPlacement>({ flippedDown: false, shiftX: 0 });
  useLayoutEffect(() => {
    const el = ref.current;
    const anchor = el?.parentElement;
    if (!el || !anchor) return;
    let scroller: HTMLElement | null = anchor.parentElement;
    while (
      scroller &&
      !/(auto|scroll)/.test(
        getComputedStyle(scroller).overflowY + getComputedStyle(scroller).overflowX,
      )
    ) {
      scroller = scroller.parentElement;
    }
    // The visible bounds: the scrollable pane's box intersected with the
    // viewport, so a pane taller than the window still keeps the card on
    // screen. No scrollable ancestor leaves just the viewport.
    const paneRect = scroller?.getBoundingClientRect();
    const pad = 8;
    const bounds = {
      top: Math.max(paneRect?.top ?? 0, 0),
      bottom: Math.min(paneRect?.bottom ?? window.innerHeight, window.innerHeight),
      left: Math.max(paneRect?.left ?? 0, 0),
      right: Math.min(paneRect?.right ?? window.innerWidth, window.innerWidth),
    };
    const popRect = el.getBoundingClientRect();
    const anchorRect = anchor.getBoundingClientRect();
    const spaceBelow = bounds.bottom - anchorRect.bottom;
    const spaceAbove = anchorRect.top - bounds.top;
    const flippedDown = popRect.height + pad > spaceAbove && spaceBelow > spaceAbove;
    // Measured from the anchor, not the card's current rect: the card may
    // already be wearing a previous shift, and stacking corrections drifts.
    let shiftX = 0;
    const overRight = anchorRect.left + popRect.width - (bounds.right - pad);
    if (overRight > 0) shiftX = -overRight;
    if (anchorRect.left + shiftX < bounds.left + pad) {
      shiftX = bounds.left + pad - anchorRect.left;
    }
    setPlacement((prev) =>
      prev.flippedDown === flippedDown && prev.shiftX === shiftX ? prev : { flippedDown, shiftX },
    );
  }, [ref, citationCount]);
  return placement;
}

/**
 * The one popover component every marker and chip opens - a stack of
 * {@link CitationEntry} rows (more than one only when a grouped marker like
 * `[2, 3]` names several sources at once). Informs even when no entry has an
 * action: the icon, label and passage are the popover's real payload, the
 * action row is a bonus.
 *
 * `role="region"`, not `dialog`: focus never moves in on open (matching
 * `ContextMeter`/`.model-menu`) - a hover preview must not steal focus, and a
 * pinned one is reached by Tab like anything else on the page.
 *
 * A `<span>` (block by CSS) for the same reason as {@link CitationEntry}: a
 * marker's popover lives inside the answer's `<p>`.
 */
export function CitationPopoverCard({
  id,
  citations,
  onViewToolStep,
  onDownloadAttachment,
}: {
  /** DOM id, referenced by the trigger's `aria-controls`. */
  id: string;
  citations: readonly CitationData[];
  onViewToolStep: () => void;
  onDownloadAttachment: (attachmentId: string) => void;
}) {
  const cardRef = useRef<HTMLElement>(null);
  const { flippedDown, shiftX } = usePopoverPlacement(cardRef, citations.length);
  if (citations.length === 0) return null;
  return (
    <span
      ref={cardRef}
      id={id}
      role="region"
      aria-label={citations.length > 1 ? "Citation sources" : "Citation source"}
      className="citation-popover"
      data-part="citation-popover"
      data-flip={flippedDown ? "down" : undefined}
      // Frost the fill so a translucent theme (aurora) does not let the
      // transcript behind it bleed through; `none` in flat themes - the same
      // inline pattern `ContextMeter` uses for `.context-pop`. The transform
      // is the horizontal clamp from usePopoverPlacement; layout-inert, so it
      // cannot re-trigger the measurement that set it.
      style={{
        backdropFilter: "var(--c-menu-backdrop)",
        WebkitBackdropFilter: "var(--c-menu-backdrop)",
        transform: shiftX !== 0 ? `translateX(${shiftX}px)` : undefined,
      }}
    >
      {citations.map((citation, i) => (
        <CitationEntry
          key={`${i}-${citation.id}`}
          citation={citation}
          onViewToolStep={onViewToolStep}
          onDownloadAttachment={onDownloadAttachment}
        />
      ))}
    </span>
  );
}

/**
 * The interactive wrapper shared by a marker and a chip: owns this trigger's
 * id, resolves its citations, and renders the trigger content plus (while
 * active) the shared popover, positioned relative to it.
 *
 * Not exported - {@link CitationMarker} and {@link CitationChip} are the two
 * views the rest of the app renders; this is what keeps their hover/focus/pin
 * wiring and popover placement identical without copying it twice.
 */
function CitationTrigger({
  indexes,
  className,
  onClickBehavior,
  children,
}: {
  indexes: readonly string[];
  className: string;
  /** `"pin"` toggles the popover on click; `"none"` leaves click to the caller (a link's own navigation). */
  onClickBehavior: "pin" | "none";
  /** The trigger's own visible content; the caller supplies it (label text, or a full `<a>` for chips). */
  children: (props: {
    triggerRef: RefObject<HTMLElement | null>;
    highlighted: boolean;
    active: boolean;
    popoverId: string;
    onClick: (() => void) | undefined;
  }) => ReactNode;
}) {
  const ctx = useCitationHighlight();
  const triggerId = useId();
  const popoverId = `${triggerId}-pop`;
  const containerRef = useRef<HTMLSpanElement>(null);
  const triggerRef = useRef<HTMLElement>(null);

  const citations = useMemo(
    () =>
      ctx
        ? indexes
            .map((index) => ctx.byIndex.get(index))
            .filter((c): c is CitationData => c != null)
        : [],
    [ctx, indexes],
  );

  const active = ctx?.activeTriggerId === triggerId;
  const pinned = ctx?.pinnedTriggerId === triggerId;
  const highlighted = ctx != null && indexes.some((i) => ctx.activeIndexes.has(i));

  useDismissWhenPinned(pinned, containerRef, triggerRef, () => ctx?.dismissPin());

  if (!ctx || citations.length === 0) {
    // No provider, or a marker whose indexes named nothing real (should not
    // happen - the rehype plugin only marks indexes already validated against
    // the message's citations). Either way: inert, not broken.
    return <>{children({ triggerRef, highlighted: false, active: false, popoverId, onClick: undefined })}</>;
  }

  const onClick =
    onClickBehavior === "pin" ? () => ctx.togglePin(triggerId, indexes) : undefined;

  return (
    <span
      ref={containerRef}
      className={className}
      data-part="citation-trigger"
      onMouseEnter={() => ctx.setHover(triggerId, indexes)}
      onMouseLeave={() => ctx.clearHover(triggerId)}
      onFocus={() => ctx.setHover(triggerId, indexes)}
      onBlur={(e) => {
        // Focus moving from the trigger INTO its own open popover (the next
        // stop in Tab order) must not close the very thing being tabbed into -
        // only a relatedTarget outside this whole wrapper counts as leaving.
        const next = e.relatedTarget;
        if (containerRef.current && next instanceof Node && containerRef.current.contains(next)) {
          return;
        }
        ctx.clearHover(triggerId);
      }}
    >
      {children({ triggerRef, highlighted, active, popoverId, onClick })}
      {active && (
        <CitationPopoverCard
          id={popoverId}
          citations={citations}
          onViewToolStep={ctx.revealToolTrace}
          onDownloadAttachment={ctx.downloadAttachment}
        />
      )}
    </span>
  );
}

/**
 * An inline `[n]` (or `[n, m]`) citation marker, upgraded from static text
 * into the shared popover's other trigger.
 *
 * Rendered by `markdown-content.tsx`'s `span` override wherever
 * `lib/markdown.ts` marked a real citation reference - `indexes` and `label`
 * come straight off that hast node's `data-citation-indexes` property and its
 * text, never re-derived.
 *
 * A real `<button>`, not a styled span with a click handler: it needs to be
 * in the tab order and announce its state, and a button gets both for free
 * (Enter/Space activation included) rather than needing `role="button"` plus
 * a keydown handler to reimplement them.
 */
export function CitationMarker({
  indexes,
  label,
}: {
  indexes: readonly string[];
  label: string;
}) {
  return (
    <CitationTrigger
      indexes={indexes}
      className="citation-marker-wrap"
      onClickBehavior="pin"
    >
      {({ triggerRef, highlighted, active, popoverId, onClick }) => (
        <button
          ref={triggerRef as RefObject<HTMLButtonElement>}
          type="button"
          className="cite-mark"
          data-highlighted={highlighted ? "true" : undefined}
          aria-label={`Citation ${indexes.join(", ")}`}
          aria-expanded={onClick ? active : undefined}
          aria-controls={active ? popoverId : undefined}
          onClick={onClick}
        >
          {label}
        </button>
      )}
    </CitationTrigger>
  );
}

/**
 * One source chip under an answer, upgraded with the same popover a marker
 * opens plus a reverse highlight: hovering or pinning a chip lights up every
 * marker in the answer body that names its index (`activeIndexes`, keyed by
 * `CitationHighlightProvider`).
 *
 * Per-type primary click is kept exactly as it was: a web result (`url`
 * present) is still a real link a reader can open, middle-click, or copy -
 * so it renders as `<a>` and click is left to the browser's own navigation.
 * Everything else had no primary click before this slice, so click is free to
 * pin the popover open, matching the marker and giving a chip with no `href`
 * a touch-reachable way to see its action (hover has no touch equivalent).
 */
export function CitationChip({ citation }: { citation: CitationData }) {
  const indexes = useMemo(
    () => (citation.index != null ? [String(citation.index)] : []),
    [citation.index],
  );
  const label = (
    <>
      {citation.index != null && <b>{citation.index}</b>}
      {citation.source}
    </>
  );

  if (citation.url) {
    return (
      <CitationTrigger
        indexes={indexes}
        className="citation-chip-wrap"
        onClickBehavior="none"
      >
        {({ triggerRef, highlighted }) => (
          <a
            ref={triggerRef as RefObject<HTMLAnchorElement>}
            href={citation.url ?? undefined}
            target="_blank"
            rel="noopener noreferrer"
            className="cite-chip"
            data-highlighted={highlighted ? "true" : undefined}
          >
            {label}
          </a>
        )}
      </CitationTrigger>
    );
  }

  return (
    <CitationTrigger
      indexes={indexes}
      className="citation-chip-wrap"
      onClickBehavior="pin"
    >
      {({ triggerRef, highlighted, active, popoverId, onClick }) => (
        <button
          ref={triggerRef as RefObject<HTMLButtonElement>}
          type="button"
          className="cite-chip"
          data-highlighted={highlighted ? "true" : undefined}
          aria-expanded={onClick ? active : undefined}
          aria-controls={active ? popoverId : undefined}
          onClick={onClick}
        >
          {label}
        </button>
      )}
    </CitationTrigger>
  );
}
