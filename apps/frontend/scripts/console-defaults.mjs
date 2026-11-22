/**
 * Battery for the console defaults page's context-ceiling and embedding fields.
 *
 * Five things here fail SILENTLY, which is why they are pinned rather than
 * left to a glance at the page:
 *
 *   - the preset select holds no state of its own. What it shows is DERIVED
 *     from the number in the field on every render, so picking a preset and
 *     typing the same number by hand are indistinguishable, and a save or a
 *     discard cannot leave the two disagreeing. A derivation that answered
 *     "Custom" for a value the list offers still renders a perfectly ordinary
 *     select, and nothing else in the app would notice.
 *   - the offered numbers have to be numbers the API accepts. A preset the
 *     save refuses is the UI handing over a value and then rejecting it, so
 *     the presets are checked against the backend's own bounds - the file
 *     that declares them, not a copy of it.
 *   - the select POPULATES, it does not save. Both controls route through the
 *     form's one edit path, which is what puts a preset on the same dirty
 *     state, Save button and Discard as a keystroke.
 *   - the embedding choice reaches the wire at all. It decides which vector
 *     space every query reads, and both halves of getting it there are
 *     invisible when broken: a listing filtered on the capability flag instead
 *     of the role would offer a chat model as an embedder, and a diff that
 *     skipped the field would leave a select that moves, dirties nothing, and
 *     saves nothing.
 *   - the embedding select stays usable once the listing has ANSWERED, empty
 *     or not. A disabled control reads as deliberate, so a gate that keyed on
 *     an empty listing would look like a considered design while it held a
 *     workspace to a stored tag the server no longer has.
 *
 * The pure ones run against `lib/context-presets.ts`, `lib/models.ts` and
 * `lib/workspace-defaults.ts` directly. The wiring facts are spread across the
 * page's markup, so they are checked statically the way `proxy-contract.mjs`
 * checks the route files - this workspace has no component runner, and a wiring
 * guarantee no test can see is one refactor from being lost.
 *
 * Like the batteries beside it, it imports the app's TypeScript directly and
 * so needs a Node that strips types on its own - 22.18 or newer.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/console-defaults.mjs)
 *
 * It reports every failing case and exits non-zero if there was one.
 */

import { readFileSync } from "node:fs";
import { register } from "node:module";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `console defaults: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

// The two modules below reach for the app's `@/` alias, which only the build
// resolves; the hook the sibling batteries use teaches plain Node the same two
// specifiers and rewrites nothing else.
register("./workspace-alias-hook.mjs", import.meta.url);

const { CEILING_DIGITS_ONLY, CONTEXT_PRESETS, contextPresetState, PRESET_CUSTOM, PRESET_UNSET } =
  await import("../lib/context-presets.ts");
const { embeddingModels } = await import("../lib/models.ts");
const { defaultsDiff } = await import("../lib/workspace-defaults.ts");

let failures = 0;
let checks = 0;

function check(label, actual, expected) {
  checks += 1;
  if (JSON.stringify(actual) !== JSON.stringify(expected)) {
    failures += 1;
    console.error(`FAIL  ${label}`);
    console.error(`  expected ${JSON.stringify(expected)}`);
    console.error(`  actual   ${JSON.stringify(actual)}`);
  }
}

/* ---------- the offered set ---------- */

// Pinned by VALUE, not merely by count. These four numbers are what a reader
// gets by picking a card size, and a silent edit to one of them changes what
// every future workspace starts from.
check(
  "presets: the four card classes and their ceilings",
  CONTEXT_PRESETS.map((preset) => [preset.card, preset.tokens]),
  [
    ["4 GB", 8192],
    ["8 GB", 16384],
    ["16 GB", 32768],
    ["24 GB", 65536],
  ],
);

// The bounds are read from the backend's own declaration rather than copied
// here, because a copy would agree with itself forever. If those constants
// ever narrow past a preset, this is what says so - otherwise the select would
// go on offering a number that populates the field, lights up Save, and then
// comes back as "Enter a ceiling between ...".
const SETTINGS_PY = new URL("../../backend/app/schemas/settings.py", import.meta.url);
const settingsSource = readFileSync(SETTINGS_PY, "utf8");

/** Read one `NAME = 1_234` module constant out of the backend schema. */
function backendConstant(name) {
  const found = new RegExp(`^${name}\\s*=\\s*([\\d_]+)`, "m").exec(settingsSource);
  // Loudly, not `null`: a rename upstream must fail this check rather than
  // quietly turn the bounds comparison below into a comparison with nothing.
  if (found === null) throw new Error(`console defaults: ${name} not found in ${SETTINGS_PY}`);
  return Number(found[1].replaceAll("_", ""));
}

const minCeiling = backendConstant("MIN_CONTEXT_CEILING");
const maxCeiling = backendConstant("MAX_CONTEXT_CEILING");

check(
  "presets: every offered ceiling is one the API would accept",
  CONTEXT_PRESETS.filter(
    (preset) => preset.tokens < minCeiling || preset.tokens > maxCeiling,
  ).map((preset) => preset.card),
  [],
);

// The field's own step is 1024, so a preset that is not a multiple of it would
// be a number the arrows beside the input can never reach again once nudged.
check(
  "presets: every ceiling sits on the field's 1024-token step",
  CONTEXT_PRESETS.filter((preset) => preset.tokens % 1024 !== 0).map((preset) => preset.card),
  [],
);

/* ---------- an empty field is "Not set", never a preset ---------- */

// The empty state is the field's documented meaning, so it has to survive
// every spelling of empty a form can hold - React re-renders with "" after a
// clear, and a value cleared through the select arrives the same way.
check("state: an empty field is Not set", contextPresetState(""), PRESET_UNSET);
check("state: a whitespace-only field is Not set", contextPresetState("   "), PRESET_UNSET);

/* ---------- an exact match shows its preset ---------- */

for (const preset of CONTEXT_PRESETS) {
  check(
    `state: ${preset.tokens} shows the ${preset.card} preset`,
    contextPresetState(String(preset.tokens)),
    String(preset.tokens),
  );
}

// Typing a preset's number by hand is indistinguishable from picking it. This
// is the reason the state is derived rather than remembered: a select holding
// its own memory would say "Custom" here while the field says 16384.
check(
  "state: a hand-typed preset number is that preset, not Custom",
  contextPresetState("16384"),
  "16384",
);

// The select can only show a value it offers, so a differently-spelled but
// equal number resolves to the OPTION's spelling. Left as the field's text,
// the select would fall back to its first option and silently read "Not set"
// over a field holding 8192.
check(
  "state: a leading-zero spelling resolves to the preset's own value",
  contextPresetState("08192"),
  "8192",
);
check("state: surrounding whitespace does not hide a preset", contextPresetState(" 8192 "), "8192");

/* ---------- anything else is Custom ---------- */

check("state: a number between presets is Custom", contextPresetState("20000"), PRESET_CUSTOM);
check("state: a number below every preset is Custom", contextPresetState("2048"), PRESET_CUSTOM);
check("state: a number above every preset is Custom", contextPresetState("131072"), PRESET_CUSTOM);

// The spellings `Number()` reads as a preset while the field does not. Only
// some of these can come out of an `<input type="number">` - it reports "" for
// anything it cannot parse - but the derivation is public and its input is a
// draft string, so the guard is checked at its own boundary rather than at the
// one control that happens to feed it today. Drop the digits-only rule and
// "0x2000" and "8.192e3" both light up the 4 GB preset over a field the save
// is about to reject.
check("state: a hex spelling of a preset is Custom", contextPresetState("0x2000"), PRESET_CUSTOM);
check(
  "state: an exponent spelling of a preset is Custom",
  contextPresetState("8.192e3"),
  PRESET_CUSTOM,
);
check("state: a decimal point is Custom", contextPresetState("8192.0"), PRESET_CUSTOM);
check("state: a negative number is Custom", contextPresetState("-8192"), PRESET_CUSTOM);
check("state: a lone minus sign is Custom", contextPresetState("-"), PRESET_CUSTOM);
check("state: letters are Custom", contextPresetState("lots"), PRESET_CUSTOM);

// Custom is a state, never a stored value: feeding the select's own Custom
// value back through the field must read as Custom again, and must never
// round-trip into a preset or into Not set.
check(
  "state: the Custom sentinel itself does not resolve to a preset",
  contextPresetState(PRESET_CUSTOM),
  PRESET_CUSTOM,
);

/* ---------- the three answers stay distinguishable ---------- */

// Not set and Custom must not collide with each other or with any preset,
// because the select tells them apart by value alone.
check(
  "state: the three answers are three distinct select values",
  new Set([PRESET_UNSET, PRESET_CUSTOM, ...CONTEXT_PRESETS.map((p) => String(p.tokens))]).size,
  CONTEXT_PRESETS.length + 2,
);

/* ---------- the spelling rule is shared, not restated ---------- */

// The save path parses the same field. Two copies of this rule would be free
// to drift, and the drift is invisible: the select would name a preset over a
// number the save refuses, or refuse one the select just offered.
check(
  "spelling: the field's rule is exported for both readers to use",
  CEILING_DIGITS_ONLY.test("8192"),
  true,
);
check(
  "spelling: it refuses what `Number()` would have accepted",
  CEILING_DIGITS_ONLY.test("0x2000"),
  false,
);

/* ---------- the select populates the field; it does not save ---------- */

const PAGE = new URL("../app/console/defaults/page.tsx", import.meta.url);
const pageSource = readFileSync(PAGE, "utf8");

// Only the one <select> that carries the preset aria-label, so a future
// control on the same page cannot make these checks pass for it.
const ARIA = 'aria-label="Preset for the context window ceiling"';
const ariaAt = pageSource.indexOf(ARIA);
const presetSelect =
  ariaAt === -1
    ? ""
    : pageSource.slice(
        pageSource.lastIndexOf("<select", ariaAt),
        pageSource.indexOf("</select>", ariaAt),
      );

check("wiring: the preset select is on the page and names itself", ariaAt !== -1, true);

// The whole design in one line. As `defaultValue`, or fed from a state of its
// own, the select would keep showing a preset that a Discard had just undone.
check(
  "wiring: the select's value is derived from the field, not held",
  presetSelect.includes("value={presetState}"),
  true,
);
check(
  "wiring: the select is controlled, never defaulted",
  presetSelect.includes("defaultValue"),
  false,
);

// POPULATES, does not save: `edit` is the same call a keystroke in the number
// field makes, so a preset lands on the same dirty state and the same Save.
// Anything else here - a direct `updateWorkspaceDefaults`, a `setDraft` that
// skips the form's bookkeeping - would make the select the one control on the
// page that commits by itself.
check(
  "wiring: choosing a preset goes through the form's one edit path",
  presetSelect.includes("edit({ context_ceiling: event.target.value })"),
  true,
);
check(
  "wiring: and both controls use that same path",
  (pageSource.match(/edit\(\{ context_ceiling: event\.target\.value \}\)/g) ?? []).length,
  2,
);

// The number field keeps an accessible name of its own. Wrapped in the same
// <label> as the select, its announced name would absorb whatever the select
// currently reads - the label's name computation substitutes an embedded
// control's VALUE - so the field would rename itself on every pick.
check(
  "wiring: the number field is labelled explicitly, not by enclosure",
  pageSource.includes('htmlFor="context-ceiling"') && pageSource.includes('id="context-ceiling"'),
  true,
);

/* ---------- the embedding default ---------- */

// A listing holding one of each shape the role field can take. The first entry
// is the trap: a tag that can embed AND complete is a CHAT model by the
// precedence the backend resolves roles with, so offering it here would let a
// workspace store a chat model as its embedder - and the stored choice is taken
// verbatim, so nothing downstream would catch it.
const LISTING = {
  models: [],
  entries: [
    {
      role: "chat",
      provider: "ollama",
      model: "talks-and-embeds",
      capabilities: { embedding: true },
    },
    {
      role: "embedding",
      provider: "ollama",
      model: "embeds-only",
      capabilities: { embedding: true },
    },
    { role: "", provider: "ollama", model: "undescribed", capabilities: {} },
  ],
};

check(
  "embedders: the offered set is the embedding ROLE, not the capability flag",
  embeddingModels(LISTING).map((entry) => entry.model),
  ["embeds-only"],
);

const STORED = {
  model: null,
  embedding_model: "embeds-only",
  thinking: null,
  web_search: null,
  system_prompt: null,
  context_ceiling: null,
  reranker: null,
  reranker_status: { configured: false, model: null, detail: "" },
  min_context_ceiling: minCeiling,
  max_context_ceiling: maxCeiling,
  max_system_prompt_chars: backendConstant("MAX_SYSTEM_PROMPT_CHARS"),
};

/** The form's fields as the page hands them to the diff, with one changed. */
function draftFrom(overrides) {
  return {
    model: STORED.model,
    embedding_model: STORED.embedding_model,
    thinking: STORED.thinking,
    web_search: STORED.web_search,
    system_prompt: STORED.system_prompt,
    context_ceiling: STORED.context_ceiling,
    reranker: STORED.reranker,
    ...overrides,
  };
}

// The three states the endpoint tells apart by PRESENCE, checked as presence:
// absent leaves the stored choice alone, so a save from a tab that never
// touched this field cannot move the corpus another tab just pointed at.
check(
  "diff: an untouched embedder stays out of the patch",
  defaultsDiff(STORED, draftFrom({})),
  {},
);
check(
  "diff: a chosen embedder rides the same save as everything else",
  defaultsDiff(STORED, draftFrom({ embedding_model: "embeds-differently" })),
  { embedding_model: "embeds-differently" },
);
// Not set is an explicit null, never an omission: omitted, the clear would be
// read as "leave it alone" and the select would spring back on the next load.
check(
  "diff: Not set clears the stored choice explicitly",
  defaultsDiff(STORED, draftFrom({ embedding_model: null })),
  { embedding_model: null },
);

// Same rule the preset select is held to, and the same failure if it is broken:
// a control that wrote straight to the API would commit by itself, outside the
// page's one Save and its Discard. The `|| null` is the other half - an empty
// option value stored as "" is a 422 from the backend's own length bound, where
// null is the clear the reader asked for.
check(
  "wiring: choosing an embedder goes through the form's one edit path",
  pageSource.includes("edit({ embedding_model: event.target.value || null })"),
  true,
);

// Only the one <select> the embedding field feeds, sliced on the same rule the
// preset select is, so a control added later cannot answer for it.
const EMBEDDER_VALUE = "value={draft.embedding_model ?? UNSET}";
const embedderAt = pageSource.indexOf(EMBEDDER_VALUE);
const embedderSelect =
  embedderAt === -1
    ? ""
    : pageSource.slice(
        pageSource.lastIndexOf("<select", embedderAt),
        pageSource.indexOf("</select>", embedderAt),
      );

// The gate keys on the UNKNOWN answer, never on an empty one - `=== null`, not
// a length. An empty listing is the state where clearing matters most, since a
// stored tag whose weights are gone is repaired by picking Not set, which is
// safe to store from any state; a gate on length locks the control precisely
// where the repair lives, and reads as deliberate while it does it.
check(
  "wiring: the embedding select is inert only while the listing is unread",
  embedderSelect.includes("disabled={saving || embedders === null}"),
  true,
);

console.log(
  failures === 0
    ? `console defaults: ${checks} checks passed`
    : `console defaults: ${failures} of ${checks} checks FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
