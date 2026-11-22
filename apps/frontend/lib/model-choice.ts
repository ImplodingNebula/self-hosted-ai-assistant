/**
 * Which model a conversation runs on, as the composer names it.
 *
 * A conversation either pins a tag or follows the workspace default. The
 * default is a suggestion: when it names a model the server lists, that row
 * wears the badge and choosing it clears the pin; when it names nothing the
 * server lists (an admin picked a tag since removed, or none was set), the
 * first listed model stands in for a fresh conversation and is pinned, so
 * the trigger always names the model that will answer. There is no separate
 * "Default" entry.
 */

/** Two tags name the same model when they differ only in case or a `:latest` suffix. */
export function sameTag(a: string, b: string): boolean {
  const norm = (t: string) => t.toLowerCase().replace(/:latest$/, "");
  return norm(a) === norm(b);
}

/** What the composer shows and checks for a conversation. */
export interface ModelChoice {
  /** The tag that will answer, or null when nothing is known to. */
  model: string | null;
  /** The workspace default is among the listed models, so its row carries the badge. */
  defaultListed: boolean;
  /** A fresh conversation should pin this tag: the default is not listed and this stands in. */
  standIn: string | null;
}

/**
 * Resolve the choice for a pin, the workspace default and the listed tags.
 *
 * Only a FRESH conversation takes a stand-in: a stored conversation that
 * follows an unlisted default is the server's to resolve, and pinning it
 * from here would rewrite its settings without anyone asking. For it the
 * choice names nothing, and the trigger says so.
 *
 * @param pinned - The conversation's pinned tag, or null when it follows the default.
 * @param workspaceDefault - The workspace's default tag, or null when unknown or unset.
 * @param listed - The chat-role tags the server lists, in its order.
 * @param fresh - Whether the conversation has not been created yet.
 */
export function modelChoice(
  pinned: string | null,
  workspaceDefault: string | null,
  listed: readonly string[],
  fresh: boolean,
): ModelChoice {
  const defaultListed =
    workspaceDefault !== null && listed.some((t) => sameTag(t, workspaceDefault));
  if (pinned !== null) return { model: pinned, defaultListed, standIn: null };
  if (defaultListed) return { model: workspaceDefault, defaultListed, standIn: null };
  if (!fresh) return { model: null, defaultListed, standIn: null };
  const first = listed[0] ?? null;
  return { model: first, defaultListed, standIn: first };
}

/** Whether a listed tag is the one the conversation runs on. */
export function isChosen(tag: string, choice: ModelChoice): boolean {
  return choice.model !== null && sameTag(tag, choice.model);
}
