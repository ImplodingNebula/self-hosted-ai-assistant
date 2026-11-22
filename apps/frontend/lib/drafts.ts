/**
 * The composer's unsent words, kept per conversation in the browser.
 *
 * A half-written message belongs to the conversation it was typed in: opening
 * another thread shows that thread's own draft, coming back shows this one
 * again, and a reload loses neither. The pane that has no conversation yet
 * keeps its draft under a placeholder, which follows the pane to the real id
 * once a conversation exists.
 *
 * It stays in the browser on purpose. A draft is text someone is still
 * editing, and nothing about it needs the server before it is sent. That
 * makes three things true that the callers rely on:
 *
 * - Every key carries the signed-in user's id, so two people sharing one
 *   browser never read each other's drafts. Signing out clears them all.
 * - Storage can refuse (private windows, a full quota, a blocked site). Every
 *   function here then does nothing and reads as "no draft": the composer
 *   works exactly as it did before drafts were kept.
 * - It is bounded. A draft past {@link MAX_DRAFT_CHARS} is not kept, and a
 *   user holds at most {@link MAX_DRAFTS}, the oldest giving way.
 *
 * Two tabs on one conversation do not sync: the last edit wins.
 */

/** The pane with no conversation yet. */
export const NEW_CHAT_DRAFT = "new";

/** Drafts one user may hold in one browser; the oldest gives way to a new one. */
export const MAX_DRAFTS = 20;

/** Past this a draft is not kept: it would crowd the browser's small quota. */
export const MAX_DRAFT_CHARS = 50_000;

const PREFIX = "chat:draft:";

// Set on sign-out. The page that signed out stays on screen until its
// navigation lands, and a keystroke in that window must not put back what the
// sweep just removed.
let signedOut = false;

/** The slice of `Storage` this module uses, so a test can stand one in. */
export interface DraftStore {
  readonly length: number;
  key(index: number): string | null;
  getItem(key: string): string | null;
  setItem(key: string, value: string): void;
  removeItem(key: string): void;
}

/** The browser's store, or null where there is none or it refuses access. */
export function draftStore(): DraftStore | null {
  try {
    return typeof window === "undefined" ? null : window.localStorage;
  } catch {
    return null;
  }
}

function draftKey(userId: string, conversationId: string | null): string {
  return `${PREFIX}${userId}:${conversationId ?? NEW_CHAT_DRAFT}`;
}

/** Every stored key, snapshotted first: removing while walking skips entries. */
function keysUnder(store: DraftStore, prefix: string): string[] {
  const keys: string[] = [];
  for (let i = 0; i < store.length; i += 1) {
    const key = store.key(i);
    if (key !== null && key.startsWith(prefix)) keys.push(key);
  }
  return keys;
}

/** A stored entry, or null when it is not one this module wrote. */
function parse(raw: string | null): { text: string; at: number } | null {
  if (raw === null) return null;
  try {
    const entry: unknown = JSON.parse(raw);
    if (typeof entry !== "object" || entry === null) return null;
    const { text, at } = entry as { text?: unknown; at?: unknown };
    if (typeof text !== "string" || typeof at !== "number") return null;
    return { text, at };
  } catch {
    return null;
  }
}

/**
 * The draft left in a conversation's composer.
 *
 * @param store - Where drafts are kept.
 * @param userId - Whose drafts.
 * @param conversationId - The conversation, or null for the unbound pane.
 * @returns The text as it was typed, or "" when nothing usable is stored.
 */
export function readDraft(
  store: DraftStore,
  userId: string,
  conversationId: string | null,
): string {
  try {
    return parse(store.getItem(draftKey(userId, conversationId)))?.text ?? "";
  } catch {
    return "";
  }
}

/** Remove one conversation's draft. */
export function forgetDraft(
  store: DraftStore,
  userId: string,
  conversationId: string | null,
): void {
  try {
    store.removeItem(draftKey(userId, conversationId));
  } catch {
    /* storage refused: there is nothing kept to remove */
  }
}

/**
 * Keep what the composer holds for a conversation.
 *
 * Text that is empty or only whitespace is no draft, and removes whatever was
 * kept. So does text too long to keep: leaving the earlier, shorter version in
 * place would hand back stale words after a reload as though they were current.
 *
 * @param store - Where drafts are kept.
 * @param userId - Whose draft.
 * @param conversationId - The conversation, or null for the unbound pane.
 * @param text - The composer's text, exactly as it would be put back.
 * @param now - The clock, for the oldest-first limit.
 */
export function writeDraft(
  store: DraftStore,
  userId: string,
  conversationId: string | null,
  text: string,
  now: number = Date.now(),
): void {
  if (signedOut) return;
  const key = draftKey(userId, conversationId);
  try {
    if (text.trim() === "" || text.length > MAX_DRAFT_CHARS) {
      store.removeItem(key);
      return;
    }
    // Make room only when this is a draft the user did not already hold: an
    // edit to an existing one changes no count.
    if (store.getItem(key) === null) makeRoom(store, userId);
    store.setItem(key, JSON.stringify({ text, at: now }));
  } catch {
    // Refused or full. Drop this conversation's entry rather than leave an
    // older version standing in for what was just typed.
    try {
      store.removeItem(key);
    } catch {
      /* nothing more to try */
    }
  }
}

/** Drop the user's oldest drafts until one more fits under the limit. */
function makeRoom(store: DraftStore, userId: string): void {
  const held = keysUnder(store, `${PREFIX}${userId}:`)
    // An entry this module cannot read is the first to go.
    .map((key) => ({ key, at: parse(store.getItem(key))?.at ?? -1 }))
    .sort((a, b) => a.at - b.at);
  for (const { key } of held.slice(0, Math.max(0, held.length - (MAX_DRAFTS - 1)))) {
    store.removeItem(key);
  }
}

/**
 * The pane was bound to another id without its composer changing: the draft
 * follows the pane.
 *
 * This is the unbound pane gaining its conversation (an attachment or a first
 * send creates one), and a pane whose conversation turned out to be gone
 * falling back to unbound. What the composer holds is written under the new
 * id and the old entry is removed, so the words are never under both. An
 * EMPTY composer carries nothing: it must not erase a draft already waiting
 * under the id the pane lands on (the unbound pane's own, when a send into a
 * conversation deleted elsewhere drops the pane back to it).
 *
 * @param store - Where drafts are kept.
 * @param userId - Whose draft.
 * @param from - The id the pane had.
 * @param to - The id it has now.
 * @param text - What the composer holds.
 * @param now - The clock, for the oldest-first limit.
 */
export function carryDraft(
  store: DraftStore,
  userId: string,
  from: string | null,
  to: string | null,
  text: string,
  now: number = Date.now(),
): void {
  if (from === to) return;
  forgetDraft(store, userId, from);
  if (text.trim() !== "") writeDraft(store, userId, to, text, now);
}

/**
 * Stop this page keeping drafts: the other half of signing out.
 *
 * One-way on purpose. Whoever signed out is on their way to the sign-in page,
 * and the next page load starts afresh.
 */
export function stopKeepingDrafts(): void {
  signedOut = true;
}

/** Remove every draft in this browser, whoever typed it: the sign-out sweep. */
export function forgetAllDrafts(store: DraftStore): void {
  try {
    for (const key of keysUnder(store, PREFIX)) store.removeItem(key);
  } catch {
    /* storage refused: there is nothing kept to remove */
  }
}
