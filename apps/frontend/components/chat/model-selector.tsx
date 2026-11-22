"use client";

import { useEffect, useId, useRef, useState } from "react";

import { useFloaterNudge } from "@/hooks/use-floater-nudge";

import { Icon } from "@/components/chat/step";
import { isChosen, modelChoice, sameTag } from "@/lib/model-choice";
import { getResidency } from "@/lib/model-manager";
import type { ModelEntry } from "@/lib/types";

/** Props for {@link ModelSelector}. */
export interface ModelSelectorProps {
  /**
   * Chat-role entries to offer, in the order the model server listed them.
   * Every one is a tag the server actually holds, so every one can be picked.
   * Empty is a real state and means one of three things - see `unavailable`
   * and `loaded`, which are what tell them apart.
   */
  models: ModelEntry[];
  /**
   * Whether the listing FAILED rather than simply being empty. The menu says
   * so in a line of its own: an unreachable model server and a server holding
   * no chat model look identical here, and they mean opposite things.
   */
  unavailable?: boolean;
  /**
   * Whether the listing has ANSWERED yet, either way.
   *
   * The third state, and it needs its own flag because an empty list carries
   * no evidence on its own: before the answer lands it means nothing, and
   * after it lands it means the server holds nothing that can chat. Only the
   * second earns a note saying so, and defaulting to false is what keeps a
   * caller that does not know about it from asserting the stronger one.
   */
  loaded?: boolean;
  /** The tag this conversation pins, or null when it follows the workspace default. */
  value: string | null;
  /**
   * The workspace's default tag, or null while unread or when none is set.
   * Its row wears the badge when the server lists it; the trigger names it
   * when nothing is pinned (lib/model-choice.ts has the whole rule).
   */
  defaultModel: string | null;
  /** Whether the conversation has not been created yet (a stand-in may be named for it). */
  fresh: boolean;
  /** Pin a tag, or null to clear the pin and follow the default. */
  onChange: (model: string | null) => void;
  /** Blocked while a suspended turn waits on its confirmation. */
  disabled?: boolean;
}

/** Said in the menu when the listing could not be fetched at all. */
const UNAVAILABLE_HINT = "Model list unavailable";

/** Said when the listing answered and named nothing this deployment can chat with. */
const NONE_INSTALLED_HINT =
  "Nothing installed can chat yet - pull a model in the Model Manager";

/** Where the keyboard goes when a confirmation takes the control away. */
const APPROVE_SELECTOR = '[data-part="confirm-approve"]:not(:disabled)';
/** The fallback landing spot when no confirmation is on screen. */
const COMPOSER_SELECTOR = '[data-part="composer"] textarea:not(:disabled)';

/**
 * How an entry describes itself under its tag.
 *
 * Read off the listing's own capability keys rather than hard-coded per tag:
 * what the runtime reports and what this deployment has established about a
 * model are the only facts about it, so a tag this client has never heard of
 * still gets an honest line, and a fact that changes changes the line with it.
 * Only DISTINGUISHING facts are said - tool calling is the norm for a chat
 * entry, so it is worth a word only in its absence.
 *
 * The pair sibling is named for the same reason it has no row of its own: a
 * paired entry answers directly but routes thinking turns to another tag
 * entirely, and this line is the only place that shows. Without it, picking
 * an entry that "answers directly" looks like giving up reasoning.
 */
function describe(entry: ModelEntry): string {
  const parts: string[] = [];
  const thinking = entry.capabilities.thinking;
  if (thinking === "always") parts.push("always reasons before answering");
  else if (thinking === "hybrid") parts.push("reasons on request");
  else if (thinking === false || thinking === "false") parts.push("answers directly");
  // Beside the thinking clause, whose story it finishes.
  const pair = entry.capabilities.thinking_model;
  if (typeof pair === "string" && pair !== "") parts.push(`thinking via ${pair}`);
  if (entry.capabilities.vision === true) parts.push("reads images");
  if (entry.capabilities.tools !== undefined && entry.capabilities.tools !== "native") {
    parts.push("cannot use tools");
  }
  // Every hint line is sentence-cased, whichever fact happens to lead it, so
  // the menu's descriptions read consistently against the Default entry's.
  const line = parts.join(" - ");
  return line.charAt(0).toUpperCase() + line.slice(1);
}

/**
 * Chat-model picker for the composer: a pill that opens a drop-up of the chat
 * models the deployment's model server holds, plus a Default entry that clears
 * the pin.
 *
 * Modelled on the conversation-row menu: a button trigger carrying the
 * `aria-expanded` / `aria-controls` pair, a roving-tabindex menu that arrows
 * walk, Escape closing it and handing focus back. The items are
 * `menuitemradio` because exactly one of them is in force at a time.
 *
 * Every listed tag is one the server actually holds, so every one can be
 * picked. The single exception is a pinned tag the listing does not offer: it
 * still governs the conversation, so it gets a row of its own - inert, and
 * kept focusable (`aria-disabled` rather than the `disabled` attribute) so
 * arrow keys still walk past it, since a skipped item is a menu that reads as
 * shorter than it looks. The trigger must not be the only place a
 * conversation's own model appears.
 *
 * The root's `data-part` is load-bearing, not just a styling hook: the page's
 * type-anywhere listener reads it to leave an open menu's keystrokes alone.
 */
export function ModelSelector({
  models,
  unavailable,
  loaded,
  value,
  defaultModel,
  fresh,
  onChange,
  disabled,
}: ModelSelectorProps) {
  const choice = modelChoice(
    value,
    defaultModel,
    models.map((m) => m.model),
    fresh,
  );
  const [open, setOpen] = useState(false);
  // Which tags are in memory, asked each time the menu opens: residency
  // moves with every load and unload, so a reading from the last opening is
  // no reading. Null until the runtime answers, or when it cannot say.
  const [resident, setResident] = useState<string[] | null>(null);
  useEffect(() => {
    if (!open) return;
    let live = true;
    getResidency()
      .then((r) => {
        if (live) setResident(r.reported ? r.models.map((m) => m.model) : null);
      })
      .catch(() => {});
    return () => {
      live = false;
    };
  }, [open]);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuId = useId();

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

  // Opened from the keyboard, the caret belongs on the item in force - so
  // arrows step from where the conversation actually is, not from the top.
  useEffect(() => {
    if (!open) return;
    const menu = menuRef.current;
    if (!menu) return;
    const items = menu.querySelectorAll<HTMLElement>("[role='menuitemradio']");
    const current = menu.querySelector<HTMLElement>("[aria-checked='true']");
    (current ?? items[0])?.focus();
  }, [open]);

  // A pending confirmation takes the control away, and an open menu must not
  // outlive that - it would float over a pill that can no longer answer it.
  // Adjusted DURING render rather than in an effect: an effect would paint the
  // menu over the dead trigger for a frame first, and closing is the whole
  // point. React re-renders before committing, so nothing is shown open.
  if (open && disabled) setOpen(false);

  // ...but that close cannot hand focus back the way every other close does:
  // the trigger it would go to is now disabled, so the keyboard would be left
  // on <body>, at the top of the document, away from the decision that took
  // the control away. Reachable in one move - a side-effecting tool halts the
  // stream while the menu is open - so the focus is placed deliberately: on the
  // approve button the user now has to answer, or on the composer if no card is
  // there to answer (the control can be disabled by nothing else today, but a
  // stranded keyboard is not the way to find that out).
  const wasOpenRef = useRef(false);
  const nudge = useFloaterNudge(menuRef, open);
  useEffect(() => {
    const wasOpen = wasOpenRef.current;
    wasOpenRef.current = open;
    if (!wasOpen || open || !disabled) return;
    const target =
      document.querySelector<HTMLElement>(APPROVE_SELECTOR) ??
      document.querySelector<HTMLElement>(COMPOSER_SELECTOR);
    target?.focus();
  }, [disabled, open]);

  /** Arrows walk the items; Tab leaves the menu rather than walking through it. */
  const onMenuKeyDown = (e: React.KeyboardEvent<HTMLDivElement>) => {
    if (e.key === "Tab") {
      close();
      return;
    }
    const keys = ["ArrowDown", "ArrowUp", "Home", "End"];
    if (!keys.includes(e.key)) return;
    e.preventDefault();
    const items = Array.from(
      menuRef.current?.querySelectorAll<HTMLElement>("[role='menuitemradio']") ?? [],
    );
    if (items.length === 0) return;
    if (e.key === "Home") {
      items[0].focus();
      return;
    }
    if (e.key === "End") {
      items[items.length - 1].focus();
      return;
    }
    const at = items.indexOf(document.activeElement as HTMLElement);
    const step = e.key === "ArrowDown" ? 1 : -1;
    items[(at + step + items.length) % items.length].focus();
  };

  /** Take a choice and close; picking what is already in force just closes. */
  const pick = (model: string | null) => {
    close();
    if (model !== value) onChange(model);
  };

  // A pin the listing does not offer - a tag the server has stopped holding,
  // or one it holds that no longer reports chat completion. It still governs
  // the conversation, so it gets a row of its own rather than living only on
  // the trigger: a menu where nothing is checked reads as "Default", which is
  // the opposite of what this conversation does. Inert, because the only thing
  // to do with an entry already in force is leave it or pick something else.
  const unknownPin = value !== null && !models.some((entry) => entry.model === value);

  // Whether this menu KNOWS what the server holds. The one predicate both
  // statements below are gated on, because both are claims about absence and
  // absence is only evidence once a real answer has landed: a listing still in
  // flight and one that failed are each an empty list that proves nothing.
  // Keyed on the answer rather than on `models.length`, which was the old
  // proxy and is now wrong in the case that matters - a server holding no chat
  // model at all returns a legitimately empty list, and reading that as "no
  // evidence" withholds the truest thing the menu could say.
  const listingKnown = (loaded ?? false) && !unavailable;

  // What can honestly be said about the pin. The stronger line stops at "not
  // offered" rather than "not held": this list is the CHAT-role half of the
  // inventory, so a tag the server holds and cannot chat with is absent from
  // it too, and claiming the server does not have it would overreach.
  const unknownPinHint = listingKnown
    ? "Pinned here - not among the chat models this server holds"
    : "Pinned for this conversation";

  // The listing answered and named nothing to chat with, so the menu offers
  // Default and nothing else. The note names the one move that changes that.
  const noneInstalled = listingKnown && models.length === 0;

  // Nothing to name: an empty listing that answered is "no model"; anything
  // else (the list unavailable, a stored conversation on an unlisted default)
  // runs on whatever the workspace resolves, and says so.
  const unnamed = listingKnown && models.length === 0 ? "No model" : "Workspace default";

  return (
    <div ref={rootRef} data-part="model-select" className="model-select">
      <button
        ref={triggerRef}
        type="button"
        aria-haspopup="menu"
        aria-expanded={open}
        aria-controls={open ? menuId : undefined}
        disabled={disabled}
        onClick={() => (open ? close() : setOpen(true))}
        className="ctl model-trigger"
        data-open={open ? "true" : undefined}
      >
        <span className="sr-only">Chat model: </span>
        {/* The tag lives in its own element so it can ellipsize: text-overflow
            does nothing to the anonymous flex item a bare text node becomes.
            Nothing known to answer is said plainly rather than as a tag. */}
        <span className="model-trigger-label">{choice.model ?? unnamed}</span>
        <Icon name="chev" className="i chev" />
      </button>

      {open && (
        <div
          ref={menuRef}
          id={menuId}
          role="menu"
          aria-label="Chat model"
          onKeyDown={onMenuKeyDown}
          className="model-menu"
          // Frost the fill so a translucent theme does not let the transcript
          // behind the drop-up bleed through it; `none` in flat themes. The
          // translate is the slide back inside the box, where the hinge
          // would hang the panel off the screen.
          style={{
            backdropFilter: "var(--c-menu-backdrop)",
            WebkitBackdropFilter: "var(--c-menu-backdrop)",
            translate: nudge !== 0 ? `${nudge}px` : undefined,
          }}
        >
          {/* tabIndex -1 throughout is the ARIA menu pattern's roving focus. */}
          {unknownPin && (
            <button
              type="button"
              role="menuitemradio"
              aria-checked={true}
              aria-disabled={true}
              tabIndex={-1}
              className="model-opt"
            >
              {value}
              <span>{unknownPinHint}</span>
            </button>
          )}
          {models.map((entry) => {
            // Whatever is known about the tag, and nothing where nothing is:
            // a row with no line under it is a model this deployment can say
            // little about, which is honest, where a placeholder would be a
            // sentence pretending to be a fact.
            const hint = describe(entry);
            // The workspace default's own row: choosing it clears the pin, so
            // the conversation follows the default wherever an admin moves it.
            const isDefault =
              choice.defaultListed && defaultModel !== null && sameTag(entry.model, defaultModel);
            return (
              <button
                key={entry.model}
                type="button"
                role="menuitemradio"
                aria-checked={isChosen(entry.model, choice)}
                tabIndex={-1}
                onClick={() => pick(isDefault ? null : entry.model)}
                className="model-opt"
              >
                <span className="model-tag">
                  {entry.model}
                  {isDefault && <em>default</em>}
                </span>
                {resident !== null && (
                  <em
                    className="model-where"
                    data-loaded={resident.some((t) => sameTag(t, entry.model)) ? "true" : undefined}
                  >
                    {resident.some((t) => sameTag(t, entry.model)) ? "in memory" : "on disk"}
                  </em>
                )}
                {hint !== "" && <span>{hint}</span>}
              </button>
            );
          })}
          {noneInstalled && (
            // Same shape and same reason as the unavailable line below: a
            // statement about the list, not another thing in it.
            <p className="model-note">{NONE_INSTALLED_HINT}</p>
          )}
          {unavailable && (
            // A plain line, not an item: there is nothing here to choose, and
            // a focusable entry would put a dead stop in the arrow walk. The
            // composer stays fully usable: an unpinned conversation runs on the
            // workspace default whether or not the list can be shown.
            <p className="model-note">{UNAVAILABLE_HINT}</p>
          )}
        </div>
      )}
    </div>
  );
}
