/**
 * The conversation the chat was on when the reader left it for another page,
 * kept for the session so the way back lands on it rather than on a new
 * chat. Session-scoped on purpose: a new tab or a later visit starts fresh,
 * as it always did.
 */

const KEY = "chat:resume";

/** Record which conversation the chat is leaving (null forgets). */
export function rememberConversation(id: string | null): void {
  try {
    if (id === null) window.sessionStorage.removeItem(KEY);
    else window.sessionStorage.setItem(KEY, id);
  } catch {
    /* storage refused: the way back opens a new chat, as before */
  }
}

/** The remembered conversation, if any, left in place. */
export function peekResumedConversation(): string | null {
  try {
    return window.sessionStorage.getItem(KEY);
  } catch {
    return null;
  }
}

/** The remembered conversation, taken: the next arrival starts fresh. */
export function takeResumedConversation(): string | null {
  const id = peekResumedConversation();
  if (id !== null) rememberConversation(null);
  return id;
}
