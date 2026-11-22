"use client";

import { useEffect, useId, useRef, useState } from "react";

import { AnimatedTitle } from "@/components/animated-title";
import { useAppName } from "@/components/branding-provider";
import { Icon } from "@/components/chat/step";
import { ConvoRowMenu } from "@/components/convo-row-menu";
import { ThemePicker } from "@/components/theme-picker";
import { TravelLink } from "@/components/travel-link";
import { UserChip } from "@/components/user-chip";
import type { AuthSession } from "@/lib/auth";
import type { ConversationSummary } from "@/lib/types";

/** Server-side cap on a conversation title. */
const TITLE_MAX_CHARS = 200;

/** Props for the conversation {@link Sidebar}. */
export interface SidebarProps {
  /** Conversations in server order (pinned first, then updated_at descending). */
  conversations: ConversationSummary[];
  /** Id of the conversation currently open in the chat pane, if any. */
  activeId: string | null;
  /** Id of a conversation whose transcript is loading (shows a row spinner). */
  pendingId: string | null;
  /** Whether the small-screen overlay is open (the panel is static on md+). */
  open: boolean;
  /** Close the small-screen overlay (backdrop click). */
  onClose: () => void;
  /** Start a fresh conversation (clears the chat pane). */
  onNewChat: () => void;
  /** Open a conversation in the chat pane. */
  onSelect: (id: string) => void;
  /** Delete a conversation; the handler owns the confirmation step. */
  onDelete: (id: string) => void;
  /**
   * Pin or unpin a conversation; the handler owns the request and its failure.
   *
   * `pinned` is the state being ASKED for, not the current one, so the row and
   * the request cannot disagree about which way the toggle went.
   */
  onTogglePin: (id: string, pinned: boolean) => Promise<boolean>;
  /** Commit a new title; the handler owns the request and its failure. */
  onRename: (id: string, title: string) => void;
  /** Row whose just-generated title is typing into place; null when none is. */
  revealId: string | null;
  /** The reveal finished (or was taken over); the row goes back to plain text. */
  onRevealDone: () => void;
  /** Non-blocking list-load failure note, shown inside the panel. */
  error: string | null;
  /**
   * Whether the FIRST list fetch is still in flight.
   *
   * Only the first: a refetch behind a create, delete or rename refreshes a
   * list already on screen, and replacing it with a loading note would read as
   * the panel losing its rows.
   */
  loading: boolean;
  /**
   * Who is signed in, or null while unknown.
   *
   * Null and a session reporting `auth: "none"` both render the footer exactly
   * as it was before sign-in existed - a deployment with auth switched off must
   * be untouched by it, and so must the moment before the probe answers.
   */
  session: AuthSession | null;
}

/**
 * Fixed-width conversation list panel.
 *
 * Always visible on md+ screens; on small screens it slides over the chat pane
 * (driven by `data-open`; see the sidebar rules in themes/chat.css) so the pane
 * keeps the full width.
 *
 * Pinned conversations are lifted into their own group above the recent list
 * and appear there ONLY - a row in two places would make "the one at the top"
 * ambiguous the moment either copy changed.
 */
export function Sidebar({
  conversations,
  activeId,
  pendingId,
  open,
  onClose,
  onNewChat,
  onSelect,
  onDelete,
  onTogglePin,
  onRename,
  revealId,
  onRevealDone,
  error,
  loading,
  session,
}: SidebarProps) {
  // Seeded by the root layout from the server environment, so the panel is
  // headed by whatever this deployment calls itself.
  const appName = useAppName();
  // Row being renamed in place, and its draft. Editing state is the sidebar's
  // own business; only the committed title leaves the component.
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  // The title the edit STARTED from. The generated title can land while the
  // input is open, so an untouched draft must be judged against the name it
  // was seeded with - comparing against the live row would read the stale
  // draft as an edit and blur-commit it over the model's title.
  const seedRef = useRef("");
  // Escape unmounts the input, and an unmounted field fires no blur, so the
  // flag has to be cleared when the next rename starts rather than on blur.
  const cancelledRef = useRef(false);
  // The two groups are separate lists, so a pin does not reorder a row - it
  // unmounts it here and mounts a fresh one there. The row menu hands focus
  // back to its trigger on the way out, and that trigger is exactly what the
  // move destroys, dropping the caret onto <body> so the next Tab restarts at
  // the top of the document. This remembers the row the menu acted on; the
  // effect below puts the caret on its new trigger once it has re-rendered.
  const refocusIdRef = useRef<string | null>(null);
  const listRef = useRef<HTMLDivElement>(null);
  // Section headings, so each list can name the group it holds. Without the
  // association a reader hears two anonymous lists in one nav and loses the
  // only thing the feature says: which group a row is in.
  const pinnedLabelId = useId();
  const recentLabelId = useId();

  const togglePin = async (c: ConversationSummary) => {
    refocusIdRef.current = c.id;
    const ok = await onTogglePin(c.id, c.pinned_at === null);
    // A refused toggle reverts the row - a SECOND move, after the one-shot
    // restore below has already spent itself - so re-arm for the revert
    // render (the failure path always renders again: the banner, and the
    // settle-refresh).
    if (!ok) refocusIdRef.current = c.id;
  };

  // No dependency list: the row lands in its new group on whichever render the
  // pin state reaches, and the ref guard makes every other render free.
  useEffect(() => {
    const id = refocusIdRef.current;
    if (id === null) return;
    const active = document.activeElement;
    // The caret is somewhere deliberate already - the move never cost it, or
    // the reader has moved on. Taking it back would be the theft this is
    // meant to prevent.
    if (active !== null && active !== document.body) {
      refocusIdRef.current = null;
      return;
    }
    const trigger = listRef.current?.querySelector<HTMLElement>(
      `[data-convo-id="${id}"] [data-part="convo-menu-trigger"]`,
    );
    // Optional because the row may have no trigger to hold: one whose
    // transcript is loading shows a spinner in that slot, and one deleted in
    // the meantime is gone altogether.
    trigger?.focus();
    // One restore per arm, deliberately: staying armed would let a much
    // later render - long after the caret had settled elsewhere - pull it
    // back to this row. A pin the server REFUSES moves the row a second time;
    // togglePin re-arms for exactly that revert, so the failure path gets one
    // restore of its own rather than an open-ended claim.
    refocusIdRef.current = null;
  });

  const startRename = (c: ConversationSummary) => {
    cancelledRef.current = false;
    // Editing takes the row over from a reveal mid-play; the input seeds from
    // the full title either way.
    if (c.id === revealId) onRevealDone();
    seedRef.current = c.title ?? "";
    setDraft(c.title ?? "");
    setRenamingId(c.id);
  };

  /** Commit the draft, unless it is empty or unchanged - then just stop editing. */
  const commitRename = (c: ConversationSummary) => {
    const next = draft.trim();
    setRenamingId(null);
    if (next === "" || next === seedRef.current || next === (c.title ?? "")) return;
    onRename(c.id, next);
  };

  // Newest pin first, which is why the server stores a time and not a flag.
  // The server already returns them in this order - the sort is what holds it
  // while a just-toggled row carries a locally guessed stamp, before the saved
  // one lands. `filter` already returns a copy, so it never reorders the
  // caller's state. The stamps are ISO-8601 UTC, so a string compare IS a time
  // compare.
  const pinned = conversations
    .filter((c) => c.pinned_at !== null)
    .sort((a, b) => (b.pinned_at ?? "").localeCompare(a.pinned_at ?? ""));
  const recent = conversations.filter((c) => c.pinned_at === null);

  /** One conversation row: the open button plus its action slot, or the rename field. */
  const renderRow = (c: ConversationSummary) =>
    c.id === renamingId ? (
      <li key={c.id} data-convo-id={c.id} className="convo-row">
        <input
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          // A blur commits, so the field must never be left holding
          // a value the list does not show; Escape flags the exit as
          // a cancel for the blur that may follow it.
          onKeyDown={(e) => {
            if (e.key === "Enter") {
              e.preventDefault();
              commitRename(c);
            } else if (e.key === "Escape") {
              e.preventDefault();
              cancelledRef.current = true;
              setRenamingId(null);
            }
          }}
          onBlur={() => {
            if (cancelledRef.current) {
              cancelledRef.current = false;
              return;
            }
            commitRename(c);
          }}
          autoFocus
          onFocus={(e) => e.currentTarget.select()}
          maxLength={TITLE_MAX_CHARS}
          aria-label="Conversation title"
          className="convo-rename"
        />
      </li>
    ) : (
      <li key={c.id} data-convo-id={c.id} className="convo-row">
        <button
          type="button"
          onClick={() => onSelect(c.id)}
          aria-current={c.id === activeId ? "true" : undefined}
          aria-busy={c.id === pendingId || undefined}
          className="convo"
          style={{ paddingRight: "2.75rem" }}
        >
          {/* Decorative pin hint (the group label announces pinned-ness to
              readers): a leading glyph rather than trailing, so it never
              fights the hover menu trigger for the row's right edge. */}
          {c.pinned_at !== null && (
            <svg
              suppressHydrationWarning
              viewBox="0 0 16 16"
              className="convo-pin h-3 w-3"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
              strokeLinejoin="round"
              aria-hidden="true"
            >
              <path suppressHydrationWarning d="M6 2.5h4M7 2.7v3.6L4.8 8.9h6.4L9 6.3V2.7M8 8.9v4.1" />
            </svg>
          )}
          {c.title === null ? (
            <span className="italic">New conversation</span>
          ) : c.id === revealId ? (
            <AnimatedTitle text={c.title} onDone={onRevealDone} />
          ) : (
            c.title
          )}
        </button>
        {/* Sibling of the row button, not a child: nesting interactive
            elements is invalid HTML and breaks keyboard focus. While
            the row's transcript loads, a spinner takes the menu
            trigger's slot - immediate feedback that the click landed. */}
        {c.id === pendingId ? (
          <span className="convo-action" aria-hidden="true">
            <svg
              suppressHydrationWarning
              viewBox="0 0 16 16"
              className="h-3.5 w-3.5 animate-spin"
              fill="none"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinecap="round"
            >
              <path suppressHydrationWarning d="M8 1.5a6.5 6.5 0 1 1-6.5 6.5" />
            </svg>
          </span>
        ) : (
          <ConvoRowMenu
            label={c.title ?? "New conversation"}
            pinned={c.pinned_at !== null}
            onTogglePin={() => void togglePin(c)}
            onRename={() => startRename(c)}
            onDelete={() => onDelete(c.id)}
          />
        )}
      </li>
    );

  return (
    <>
      {open && (
        <div
          className="fixed inset-0 z-20 bg-[var(--s-scrim)] md:hidden"
          onClick={onClose}
          aria-hidden="true"
        />
      )}
      <nav
        id="conversation-list"
        data-part="sidebar"
        data-open={open ? "true" : undefined}
        aria-label="Conversations"
      >
        <div className="brand">
          {/* the mark the boot screen draws, worn here; it is what travels between the two */}
          <Icon name="logo" className="i brand-mark" />
          {appName}
        </div>
        <button type="button" onClick={onNewChat} className="new-chat">
          {/* suppressHydrationWarning on the svg and its path: dark-mode browser
              extensions stamp stroke attributes onto inline SVGs before React
              hydrates (see the note in page.tsx); suppression is per element. */}
          <svg
            suppressHydrationWarning
            viewBox="0 0 16 16"
            className="h-3.5 w-3.5"
            fill="none"
            stroke="currentColor"
            strokeWidth="2"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path suppressHydrationWarning d="M8 3v10M3 8h10" />
          </svg>
          New chat
        </button>

        {/* Both section labels live INSIDE this scroll box, so a long pinned
            group scrolls with the list instead of pushing the recent rows off
            the panel. (The row menu sizes itself against the viewport, not
            this box.) */}
        <div ref={listRef} className="convo-scroll min-h-0 flex-1 overflow-y-auto">
          {error === null && pinned.length > 0 && (
            <>
              <div id={pinnedLabelId} className="side-label">
                Pinned
              </div>
              <ul
                data-part="convo-group"
                data-group="pinned"
                aria-labelledby={pinnedLabelId}
                className="flex flex-col gap-0.5"
              >
                {pinned.map(renderRow)}
              </ul>
            </>
          )}
          <div id={recentLabelId} className="side-label">
            Recent
          </div>
          {error ? (
            <p className="convo" style={{ color: "var(--s-text-3)" }}>
              {error}
            </p>
          ) : loading ? (
            // Deliberately the same plain line as the empty state, not a
            // skeleton: the two swap places on every cold boot, and a shape
            // that animates into a sentence draws the eye to a panel nobody
            // came here to watch.
            <p className="convo" style={{ color: "var(--s-text-3)" }}>
              Loading conversations...
            </p>
          ) : conversations.length === 0 ? (
            <p className="convo" style={{ color: "var(--s-text-3)" }}>
              No conversations yet.
            </p>
          ) : recent.length === 0 ? (
            // Said out loud rather than left as a bare heading: an empty
            // Recent group with a full Pinned one above it reads as a bug.
            <p className="convo" style={{ color: "var(--s-text-3)" }}>
              All conversations are pinned.
            </p>
          ) : (
            <ul
              data-part="convo-group"
              data-group="recent"
              aria-labelledby={recentLabelId}
              className="flex flex-col gap-0.5"
            >
              {recent.map(renderRow)}
            </ul>
          )}
        </div>

        <div className="side-foot">
          {/* One row for every management surface, not one per page: the
              console's own rail lists what is in it, and a sidebar foot that
              grew a row per section would crowd out the conversations. */}
          {/* The topbar and this panel travel into the console's (the rail
              stands where the panel stood, at its width). */}
          <TravelLink href="/console" kind="console" view="console" className="convo">
            <Icon name="gear" />
            Console
          </TravelLink>
          <ThemePicker />
          {/* Last, and only when a real sign-in happened: the account sits at
              the bottom of the panel by long convention, and a deployment with
              auth switched off has no account to name. */}
          {session?.auth === "oidc" && <UserChip session={session} />}
        </div>
      </nav>
    </>
  );
}
