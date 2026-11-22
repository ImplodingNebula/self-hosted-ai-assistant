"use client";

import { useEffect, useId, useRef, useState } from "react";
import { createPortal } from "react-dom";

/** Props for {@link ConvoRowMenu}. */
export interface ConvoRowMenuProps {
  /** Conversation name, so the trigger says which row it acts on. */
  label: string;
  /** Whether the row is pinned; the pin item's wording follows it. */
  pinned: boolean;
  /** Toggle the row's pin (the caller owns the request and its failure). */
  onTogglePin: () => void;
  /** Start the row's inline rename. */
  onRename: () => void;
  /** Ask to delete the row (the caller owns the confirmation step). */
  onDelete: () => void;
}

/** Room the panel needs below the trigger before it has to open upward.
 *
 * Its own height (3 items at 40px, 4px of padding and a 1px border each side)
 * plus the 6px it is offset by. Under-measuring here is the failure that
 * matters: the panel opens downward and the viewport clips the Delete item.
 */
const MENU_SPACE_PX = 136;

/**
 * Per-row actions menu for a conversation in the sidebar.
 *
 * A single kebab trigger holds every row action, so the row keeps one action
 * slot however many actions it grows. Modelled on the theme picker's popover:
 * an outside mousedown or Escape closes it, and the trigger keeps the
 * `aria-expanded` / `aria-controls` pair. Opens upward when the row sits too
 * low in the scrolling list for the panel to fit below it.
 *
 * The panel's `data-part` is load-bearing, not just a test hook: the page's
 * type-anywhere listener reads it to leave an open menu's keystrokes alone.
 */
export function ConvoRowMenu({
  label,
  pinned,
  onTogglePin,
  onRename,
  onDelete,
}: ConvoRowMenuProps) {
  const [open, setOpen] = useState(false);
  const [pos, setPos] = useState<{
    left: number;
    top?: number;
    bottom?: number;
  } | null>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  /** Close and hand focus back to the trigger.
   *
   * The outside-click path closes without this: that click has already put the
   * caret wherever it was aimed, and pulling it back would be a theft.
   */
  const close = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };

  /**
   * Open at the app layer, dropping down-right from the kebab.
   *
   * The panel is a portal positioned against the viewport, so it floats above
   * the sidebar and overhangs its edge over the content instead of squeezing
   * inside the scrolling list. Rendering it INSIDE the sidebar also broke its
   * frost: the sidebar is itself a backdrop-blur layer, and a blur nested in
   * another blur cannot sample the page behind it, which left the panel with
   * a weak halo instead of frosted glass. Measured on the click rather than
   * after the paint, so the panel never renders on the wrong side first;
   * flips upward when the viewport bottom is too close.
   */
  const openMenu = () => {
    const trigger = triggerRef.current;
    if (!trigger) return;
    const r = trigger.getBoundingClientRect();
    const left = r.right - 44;
    if (r.bottom + MENU_SPACE_PX > window.innerHeight) {
      setPos({ left, bottom: window.innerHeight - r.top + 6 });
    } else {
      setPos({ left, top: r.bottom + 6 });
    }
    setOpen(true);
  };

  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      const t = e.target as Node;
      if (!triggerRef.current?.contains(t) && !menuRef.current?.contains(t)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") close();
    };
    // A viewport-anchored panel cannot follow its row, so any scroll or
    // resize dismisses it (no focus return - the scroll was deliberate).
    const onMove = () => setOpen(false);
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    window.addEventListener("scroll", onMove, true);
    window.addEventListener("resize", onMove);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
      window.removeEventListener("scroll", onMove, true);
      window.removeEventListener("resize", onMove);
    };
  }, [open]);

  // A menu opened from the keyboard must put the caret on its first item.
  useEffect(() => {
    if (!open) return;
    menuRef.current?.querySelector<HTMLElement>("[role='menuitem']")?.focus();
  }, [open]);

  /** Arrow keys walk the items; a menu is one stop in the page's tab order. */
  const onMenuKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    // The items are out of the tab sequence, so Tab has nowhere to go inside
    // the panel: treat it as leaving. Closing without preventDefault puts the
    // caret back on the trigger and lets the browser carry on from there, so
    // the menu never survives as a floating panel over an unrelated row.
    if (e.key === "Tab") {
      close();
      return;
    }
    if (e.key !== "ArrowDown" && e.key !== "ArrowUp") return;
    e.preventDefault();
    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>("[role='menuitem']") ?? [],
    );
    if (items.length === 0) return;
    const at = items.indexOf(document.activeElement as HTMLElement);
    const step = e.key === "ArrowDown" ? 1 : -1;
    items[(at + step + items.length) % items.length].focus();
  };

  return (
    <div className="convo-menu-root">
      {/* The trigger's `data-part` is a hook, not a style: pinning moves the
          row to the other group, and the sidebar finds this button again by it
          to put the caret back where the menu left it. */}
      <button
        ref={triggerRef}
        data-part="convo-menu-trigger"
        type="button"
        aria-label={`Actions for ${label}`}
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        onClick={() => (open ? setOpen(false) : openMenu())}
        className="convo-action"
        data-open={open ? "true" : undefined}
      >
        {/* suppressHydrationWarning on the svg and its shapes: dark-mode browser
            extensions stamp attributes onto inline SVGs before React hydrates
            (see the note in page.tsx); suppression is per element. */}
        <svg
          suppressHydrationWarning
          viewBox="0 0 16 16"
          className="h-3.5 w-3.5"
          fill="currentColor"
          aria-hidden="true"
        >
          <circle suppressHydrationWarning cx="8" cy="3" r="1.35" />
          <circle suppressHydrationWarning cx="8" cy="8" r="1.35" />
          <circle suppressHydrationWarning cx="8" cy="13" r="1.35" />
        </svg>
      </button>

      {open && pos !== null && createPortal(
        <div
          ref={menuRef}
          id={menuId}
          data-part="convo-menu"
          role="menu"
          aria-label={`Actions for ${label}`}
          onKeyDown={onMenuKeyDown}
          // Frost the fill so a translucent theme does not let the content
          // behind the panel bleed through; `none` in flat themes.
          style={{
            left: pos.left,
            top: pos.top,
            bottom: pos.bottom,
            backdropFilter: "var(--c-menu-backdrop)",
            WebkitBackdropFilter: "var(--c-menu-backdrop)",
          }}
        >
          {/* tabIndex -1 on every item is the ARIA menu pattern's roving focus:
              arrows move between them, and Tab leaves the menu rather than
              walking through it. */}
          {/* One item, not a pair: the row is pinned or it is not, so offering
              both would always leave one of them inert. */}
          {/* Item glyphs are decorative (aria-hidden): the labels carry the
              meaning, the icons only speed up scanning. All three share the
              topbar pencil's stroke idiom so the menu reads as one family. */}
          <button
            type="button"
            role="menuitem"
            tabIndex={-1}
            onClick={() => {
              close();
              onTogglePin();
            }}
            className="convo-menu-item"
          >
            <svg
              suppressHydrationWarning
              viewBox="0 0 16 16"
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path suppressHydrationWarning d="M6 2.5h4M7 2.7v3.6L4.8 8.9h6.4L9 6.3V2.7M8 8.9v4.1" />
            </svg>
            {pinned ? "Unpin" : "Pin"}
          </button>
          <button
            type="button"
            role="menuitem"
            tabIndex={-1}
            onClick={() => {
              close();
              onRename();
            }}
            className="convo-menu-item"
          >
            <svg
              suppressHydrationWarning
              viewBox="0 0 16 16"
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path suppressHydrationWarning d="M11.5 2.5l2 2L6 12l-2.5.5.5-2.5z" />
            </svg>
            Rename
          </button>
          <button
            type="button"
            role="menuitem"
            tabIndex={-1}
            onClick={() => {
              close();
              onDelete();
            }}
            className="convo-menu-item convo-menu-danger"
          >
            <svg
              suppressHydrationWarning
              viewBox="0 0 16 16"
              className="h-3.5 w-3.5"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path suppressHydrationWarning d="M3.5 4.5h9M6.5 4.5V3h3v1.5M5 4.5l.5 8.5h5l.5-8.5M7 7v3.5M9 7v3.5" />
            </svg>
            Delete
          </button>
        </div>,
        document.body,
      )}
    </div>
  );
}
