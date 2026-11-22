"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useState,
  type ReactNode,
  type RefObject,
} from "react";

import { citationsByIndex } from "@/lib/citations";
import type { CitationData } from "@/lib/types";

/** One popover's open state: which trigger owns it, and whether a click pinned it. */
interface HighlightState {
  /** The trigger (marker or chip) currently showing a popover; null when none is. */
  activeTriggerId: string | null;
  /** The trigger a click pinned open; null while nothing is pinned. */
  pinnedTriggerId: string | null;
  /** Citation indexes the active trigger names - what a marker or chip cross-highlights against. */
  activeIndexes: ReadonlySet<string>;
}

const CLOSED: HighlightState = {
  activeTriggerId: null,
  pinnedTriggerId: null,
  activeIndexes: new Set(),
};

/** What one marker or chip needs to drive the shared popover. See {@link CitationHighlightProvider}. */
export interface CitationHighlight {
  /** This message's citations, keyed by their string marker index. */
  byIndex: ReadonlyMap<string, CitationData>;
  /** Indexes the currently active (hovered or pinned) trigger names. */
  activeIndexes: ReadonlySet<string>;
  /** The trigger the popover is currently anchored to; null when closed. */
  activeTriggerId: string | null;
  /** The trigger holding the popover open past a mouse-out; null when nothing is pinned. */
  pinnedTriggerId: string | null;
  /** Preview on hover/focus. A pinned popover elsewhere holds the floor and ignores this. */
  setHover: (triggerId: string, indexes: readonly string[]) => void;
  /** End a hover preview. A no-op while anything is pinned, or once a different trigger took over. */
  clearHover: (triggerId: string) => void;
  /** Click/tap: pin this trigger's popover open, or unpin it if it already is. */
  togglePin: (triggerId: string, indexes: readonly string[]) => void;
  /** Click-away or Esc: close whatever is pinned. */
  dismissPin: () => void;
  /** "View tool step": reveal and scroll to this message's tool-activity trace. */
  revealToolTrace: () => void;
  /** "Download file": fetch an uploaded-file citation's bytes back, by attachment id. */
  downloadAttachment: (attachmentId: string) => void;
}

const CitationContext = createContext<CitationHighlight | null>(null);

/**
 * Read the citation popover's shared state. Null outside a
 * {@link CitationHighlightProvider} - callers degrade to inert rendering
 * rather than throwing, since a marker or chip can in principle be reused
 * somewhere that never mounts the provider.
 */
export function useCitationHighlight(): CitationHighlight | null {
  return useContext(CitationContext);
}

/**
 * Scopes one assistant message's citation markers and chips to a single
 * shared popover: only one trigger's popover is ever open, and hovering or
 * pinning either view highlights both (keyed on citation index).
 *
 * Mount once per message, wrapping both the answer body (markers) and the
 * citation chip row - the two live in different subtrees of `MessageBubble`,
 * and this is what lets them agree on what is currently active without
 * threading callbacks through `MarkdownContent`'s markdown-renderer layer.
 */
export function CitationHighlightProvider({
  citations,
  idKey,
  onDownloadFile,
  children,
}: {
  /** This message's citations (possibly empty - markers and chips then render inert). */
  citations: readonly CitationData[];
  /** Identifies this message; namespaces the tool-trace DOM id `revealToolTrace` targets. */
  idKey: string;
  /** Fetch a file this turn carried or produced, by attachment id. */
  onDownloadFile: (id: string) => void;
  children: ReactNode;
}) {
  const byIndex = useMemo(() => citationsByIndex(citations), [citations]);
  const [state, setState] = useState<HighlightState>(CLOSED);

  const setHover = useCallback((triggerId: string, indexes: readonly string[]) => {
    setState((prev) => {
      // A pinned popover holds the floor: hovering elsewhere while one is open
      // must not steal it out from under a user who deliberately clicked it.
      if (prev.pinnedTriggerId !== null && prev.pinnedTriggerId !== triggerId) return prev;
      return { ...prev, activeTriggerId: triggerId, activeIndexes: new Set(indexes) };
    });
  }, []);

  const clearHover = useCallback((triggerId: string) => {
    setState((prev) => {
      if (prev.pinnedTriggerId !== null) return prev;
      if (prev.activeTriggerId !== triggerId) return prev;
      return CLOSED;
    });
  }, []);

  const togglePin = useCallback((triggerId: string, indexes: readonly string[]) => {
    setState((prev) => (prev.pinnedTriggerId === triggerId ? CLOSED : {
      activeTriggerId: triggerId,
      pinnedTriggerId: triggerId,
      activeIndexes: new Set(indexes),
    }));
  }, []);

  const dismissPin = useCallback(() => setState(CLOSED), []);

  const revealToolTrace = useCallback(() => {
    const trace = document.getElementById(`tool-trace-${idKey}`);
    if (!trace) return;
    const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    trace.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "center" });
    // "Reveal" means readable, not just centered in view: every step in this
    // message's trace opens, since a citation names the message, not one call.
    // The rows are disclosure buttons (see DisclosureRow), so opening one is
    // pressing it while it reads closed.
    for (const button of trace.querySelectorAll('button[aria-expanded="false"]')) {
      (button as HTMLButtonElement).click();
    }
  }, [idKey]);

  const value = useMemo<CitationHighlight>(
    () => ({
      byIndex,
      activeIndexes: state.activeIndexes,
      activeTriggerId: state.activeTriggerId,
      pinnedTriggerId: state.pinnedTriggerId,
      setHover,
      clearHover,
      togglePin,
      dismissPin,
      revealToolTrace,
      downloadAttachment: onDownloadFile,
    }),
    [byIndex, state, setHover, clearHover, togglePin, dismissPin, revealToolTrace, onDownloadFile],
  );

  return <CitationContext.Provider value={value}>{children}</CitationContext.Provider>;
}

/**
 * While `pinned`, close on an outside pointerdown or Escape - the contract's
 * "click-away and Esc dismiss". Escape also returns focus to the trigger (the
 * keyboard's way out, mirroring `ContextMeter`); an outside click does not,
 * since that click already aimed at something else and taking focus would be
 * a theft.
 *
 * `containerRef` must wrap both the trigger and the popover: a click that
 * re-toggles the trigger itself is handled by the trigger's own `onClick`,
 * not by this listener, and treating the trigger as "outside" would race the
 * two.
 */
export function useDismissWhenPinned(
  pinned: boolean,
  containerRef: RefObject<HTMLElement | null>,
  triggerRef: RefObject<HTMLElement | null>,
  onDismiss: () => void,
): void {
  useEffect(() => {
    if (!pinned) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target as Node;
      if (containerRef.current && !containerRef.current.contains(target)) onDismiss();
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      onDismiss();
      triggerRef.current?.focus();
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [pinned, containerRef, triggerRef, onDismiss]);
}
