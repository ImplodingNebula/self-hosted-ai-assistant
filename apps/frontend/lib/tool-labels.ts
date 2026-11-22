/**
 * Human-readable labels for tool activity, from the raw tool identifier.
 *
 * The trace card and the transient status line both receive a bare tool name
 * off the wire ("web_search", "search_records"), and a reader should not have
 * to decode a code identifier to follow what the assistant is doing. Tool names
 * carry no source marker - a remote tool is arbitrary snake_case,
 * indistinguishable in shape from a local one - so the only correct detection
 * is an exact-match allowlist for the two shipped local tools plus a generic
 * fallback that reads acceptably for anything else.
 *
 * Because the fallback is lossy (`web-search` and `Web_Search` both render as
 * "Web search", the built-in's label), these labels are for NARRATION only.
 * The confirmation card - the one surface whose job is to disclose exactly what
 * would run - keeps the raw identifier; see tool-confirmation-card.tsx.
 *
 * Deterministic and pure: same name in, same words out, no model involved.
 */

/**
 * Curated copy for the tools whose work is nameable in plain words.
 *
 * ONE entry per tool, holding all three strings, so a tool cannot be half
 * added: a name present here for the trace card but missing for the status line
 * would degrade silently, and there is no test runner to catch the omission.
 */
const KNOWN_TOOLS = new Map<string, { label: string; running: string; reading: string }>([
  [
    "web_search",
    {
      label: "Web search",
      running: "Searching the web...",
      // "what the search returned", not "search results": a search can return
      // nothing, and the reading phase happens either way.
      reading: "Reading what the search returned...",
    },
  ],
  [
    "search_knowledge",
    {
      label: "Knowledge search",
      running: "Searching the knowledge base...",
      reading: "Reading what the knowledge base returned...",
    },
  ],
]);

/**
 * The tool's name as a reader should see it.
 *
 * Known tools get their curated label; anything else is split on `_`, `-` and
 * camelCase boundaries into sentence case, so `search_records` reads "Search
 * records" and `getRecord` reads "Get record". Lossy by design (see the module
 * note), so callers keep the raw identifier reachable - on hover in the trace,
 * verbatim on the confirmation card.
 *
 * @param name - The raw tool identifier as it arrives on the wire.
 * @returns A sentence-case label, or the input unchanged if it holds no words.
 */
/** The identifier's words, lowercased: `slack.postMessage` -> ["slack", "post", "message"]. */
function toolWords(name: string): string[] {
  return (
    name
      // An acronym run ends where the next word starts: HTTPRequest -> HTTP Request.
      .replace(/([A-Z]+)([A-Z][a-z])/g, "$1 $2")
      .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
      // Dots, slashes and colons show up in namespaced remote names (slack.postMessage).
      .split(/[_\-./:\s]+/)
      .filter((word) => word !== "")
      .map((word) => word.toLowerCase())
  );
}

export function toolDisplayName(name: string): string {
  const known = KNOWN_TOOLS.get(name);
  if (known) return known.label;
  const words = toolWords(name);
  if (words.length === 0) return name;
  const phrase = words.join(" ");
  return phrase.charAt(0).toUpperCase() + phrase.slice(1);
}

/**
 * Leading tool verbs and how they read while the work is happening.
 *
 * A lookup, not an algorithm: English gerunds are irregular (get doubles its
 * consonant, create drops its e), so only listed verbs conjugate and anything
 * else falls back to "Running <name>...". Grow it as real tool vocabularies
 * appear; a missing verb degrades to the fallback, never to wrong grammar.
 */
const VERB_GERUNDS = new Map<string, string>([
  ["add", "Adding"],
  ["apply", "Applying"],
  ["check", "Checking"],
  ["create", "Creating"],
  ["delete", "Deleting"],
  ["download", "Downloading"],
  ["fetch", "Fetching"],
  ["find", "Finding"],
  ["get", "Getting"],
  ["label", "Labeling"],
  ["list", "Listing"],
  ["post", "Posting"],
  ["query", "Querying"],
  ["read", "Reading"],
  ["remove", "Removing"],
  ["run", "Running"],
  ["search", "Searching"],
  ["send", "Sending"],
  ["update", "Updating"],
  ["upload", "Uploading"],
  ["write", "Writing"],
]);

/**
 * What to say while a tool is working, or while its result is being read.
 *
 * The two local tools describe their own work ("Searching the web..."). An
 * unknown tool whose identifier leads with a recognized verb becomes a live
 * sentence of its own - `get_record` reads "Getting record..." - so the line
 * opens with the verb instead of parking a capitalized name mid-sentence.
 * Anything else stays "Running <name>...", which reads right for tools named
 * after products rather than actions. The reading phase does not repeat the
 * name at all: the trace card directly above already shows which tool just
 * finished.
 *
 * @param name - The raw tool identifier as it arrives on the wire.
 * @param phase - `running` before the tool returns, `reading` after it has.
 * @returns One line of status copy, ending in three dots.
 */
export function toolStatusLabel(name: string, phase: "running" | "reading"): string {
  const known = KNOWN_TOOLS.get(name);
  if (known) return known[phase];
  if (phase === "reading") return "Reading the result...";
  const words = toolWords(name);
  const gerund = words.length > 0 ? VERB_GERUNDS.get(words[0]) : undefined;
  if (gerund === undefined) return `Running ${toolDisplayName(name)}...`;
  const rest = words.slice(1).join(" ");
  return rest === "" ? `${gerund}...` : `${gerund} ${rest}...`;
}
