/**
 * Battery for the chat pane's pure message transitions.
 *
 * `lib/messages.ts` is where a stream frame becomes a rendered turn. Every
 * function in it takes the current `Message[]` and returns a new one, keyed off
 * the TRAILING message, which makes the whole file testable without React, a
 * DOM, or a component runner - none of which this workspace has. The composer's
 * `attachmentsForTurn` is checked here for the same reason: it is pure, and it
 * decides what a question carries.
 *
 * The attachment DOWNLOAD path is checked here too - the client helper and the
 * proxy's header relay - and the helper is the one thing in the file that is
 * not pure. It earns its place because its central guarantee is invisible from
 * every other angle: a refusal must never reach the browser's saver, or the
 * reader gets a JSON error document on disk under the file's own name and
 * nothing anywhere says so. A stubbed `fetch`, `URL` and link element are a
 * small price for pinning that. The static half of the proxy contract lives in
 * `proxy-contract.mjs`.
 *
 * Run it as part of the workspace's test command:
 *
 *     npm test        (or: node scripts/message-transitions.mjs)
 *
 * It reports every failing case and exits non-zero if there was one. Like the
 * composer battery beside it, it imports the app's TypeScript directly and so
 * needs a Node that strips types on its own - 22.18 or newer.
 */

import { register } from "node:module";

const MINIMUM_NODE = [22, 18];
const [nodeMajor, nodeMinor] = process.versions.node.split(".").map(Number);
if (
  nodeMajor < MINIMUM_NODE[0] ||
  (nodeMajor === MINIMUM_NODE[0] && nodeMinor < MINIMUM_NODE[1])
) {
  console.error(
    `message transitions: needs Node ${MINIMUM_NODE.join(".")} or newer for TypeScript type-stripping (running ${process.versions.node})`,
  );
  process.exit(1);
}

register("./workspace-alias-hook.mjs", import.meta.url);

const { appendFile, dropEmptyTrailingAssistant, fileLiveness, toPaneMessages } =
  await import("../lib/messages.ts");
// A pure function that happens to live in a hook module. Importing it only runs
// that module's top-level imports (React is never called), which is cheap enough
// to be worth reaching for: it decides what a turn CARRIES, so a mistake in it
// sends the backend an id naming nothing.
const { attachmentsForTurn } = await import("../hooks/use-attachments.ts");
// Same reasoning: pure, and between them they decide what a sent file's chip
// says, whether it offers the file back, and what becomes of it when the
// server refuses.
const {
  carriedFileStates,
  downloadAttachment,
  livenessAfterRefusal,
  sentFileState,
  servesBytes,
} = await import("../lib/attachments.ts");
const { ApiError } = await import("../lib/api-error.ts");
// Pure over one Response, and the only thing standing between the backend's
// answer to a HEAD and the browser's.
const { relayHeaders } = await import("../lib/backend-proxy.ts");

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

/** One `file` SSE payload, with overrides. */
function fileEvent(over = {}) {
  return {
    attachment_id: "att-1",
    filename: "q3-report.xlsx",
    content_type: "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    size_bytes: 8123,
    origin: "generated",
    ...over,
  };
}

const reply = () => ({ role: "assistant", content: "Here it is." });

// ---------------------------------------------------------------------------
// appendFile: the live half of a generated-file chip
// ---------------------------------------------------------------------------

check(
  "appendFile: folds the event onto the trailing reply's attachments",
  appendFile([{ role: "user", content: "make me a sheet" }, reply()], fileEvent()),
  [
    { role: "user", content: "make me a sheet" },
    {
      role: "assistant",
      content: "Here it is.",
      attachments: [
        {
          id: "att-1",
          filename: "q3-report.xlsx",
          content_type:
            "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
          size_bytes: 8123,
          origin: "generated",
        },
      ],
    },
  ],
);

check(
  "appendFile: a second file is appended, not replaced",
  appendFile(
    appendFile([reply()], fileEvent()),
    fileEvent({ attachment_id: "att-2", filename: "q4-report.xlsx" }),
  )[0].attachments.map((a) => a.id),
  ["att-1", "att-2"],
);

check(
  "appendFile: two files of the same name stay two files (two rows, two calls)",
  appendFile(appendFile([reply()], fileEvent()), fileEvent({ attachment_id: "att-2" }))[0]
    .attachments.length,
  2,
);

check(
  "appendFile: leaves the input array untouched",
  (() => {
    const before = [reply()];
    appendFile(before, fileEvent());
    return before[0].attachments;
  })(),
  undefined,
);

check(
  "appendFile: a file cannot land on a question",
  appendFile([{ role: "user", content: "hi" }], fileEvent()),
  [{ role: "user", content: "hi" }],
);

check("appendFile: an empty transcript is left alone", appendFile([], fileEvent()), []);

// ---------------------------------------------------------------------------
// The bubble a file arrived on is not an empty placeholder
// ---------------------------------------------------------------------------

check(
  "dropEmptyTrailingAssistant: a wordless reply that produced a file is kept",
  dropEmptyTrailingAssistant(
    appendFile([{ role: "assistant", content: "" }], fileEvent()),
  ).length,
  1,
);

check(
  "dropEmptyTrailingAssistant: a placeholder with nothing on it still goes",
  dropEmptyTrailingAssistant([{ role: "assistant", content: "" }]),
  [],
);

// ---------------------------------------------------------------------------
// Hydration carries origin, so a reloaded chip knows what it is looking at
// ---------------------------------------------------------------------------

check(
  "toPaneMessages: a reply's generated file survives the round trip with its origin",
  toPaneMessages([
    {
      id: "m1",
      role: "assistant",
      content: "Here it is.",
      thinking: null,
      model: "qwen3:4b-instruct",
      created_at: "2026-08-08T10:00:00Z",
      attachments: [
        {
          id: "att-1",
          filename: "q3-report.xlsx",
          content_type: "application/octet-stream",
          size_bytes: 12,
          origin: "generated",
        },
      ],
    },
  ])[0].attachments,
  [
    {
      id: "att-1",
      filename: "q3-report.xlsx",
      content_type: "application/octet-stream",
      size_bytes: 12,
      origin: "generated",
    },
  ],
);

check(
  "toPaneMessages: a snapshot written before origin existed reads as an upload",
  toPaneMessages([
    {
      id: "m1",
      role: "user",
      content: "read this",
      thinking: null,
      model: null,
      created_at: "2026-08-08T10:00:00Z",
      attachments: [
        {
          id: "att-9",
          filename: "notes.txt",
          content_type: "text/plain",
          size_bytes: 4,
        },
      ],
    },
  ])[0].attachments[0].origin,
  undefined,
);

// ---------------------------------------------------------------------------
// What a turn is allowed to carry out of the composer tray
// ---------------------------------------------------------------------------

/** One tray chip, with overrides. */
function trayChip(over = {}) {
  return {
    localId: "local-1",
    serverId: "srv-1",
    filename: "notes.txt",
    contentType: "text/plain",
    sizeBytes: 12,
    status: "ready",
    error: null,
    ...over,
  };
}

check(
  "attachmentsForTurn: a ready file rides the turn",
  attachmentsForTurn([trayChip()]).map((a) => a.serverId),
  ["srv-1"],
);

check(
  "attachmentsForTurn: an EXPIRED file does not - its bytes are gone",
  attachmentsForTurn([trayChip({ status: "expired" })]),
  [],
);

check(
  "attachmentsForTurn: a failed file does not",
  attachmentsForTurn([trayChip({ status: "failed" })]),
  [],
);

check(
  "attachmentsForTurn: an expired chip does not drag a healthy one out with it",
  attachmentsForTurn([
    trayChip({ localId: "local-1", serverId: "srv-1", status: "expired" }),
    trayChip({ localId: "local-2", serverId: "srv-2" }),
  ]).map((a) => a.serverId),
  ["srv-2"],
);

// ---------------------------------------------------------------------------
// Which files a chip may offer to hand back
// ---------------------------------------------------------------------------

check("servesBytes: a ready file is downloadable", servesBytes("ready", "upload"), true);

check(
  "servesBytes: a file whose INGESTION failed is still its sender's file",
  servesBytes("failed", "upload"),
  true,
);

check(
  "servesBytes: a pasted block that failed to parse is downloadable too",
  servesBytes("failed", "paste"),
  true,
);

check(
  "servesBytes: an unfinished GENERATED file has no original to hand back",
  servesBytes("failed", "generated"),
  false,
);

check(
  "servesBytes: an old snapshot with no origin reads as an upload",
  servesBytes("failed", undefined),
  true,
);

check(
  "servesBytes: an origin this build has no word for is refused, not assumed servable",
  servesBytes("failed", "imported"),
  false,
);

check(
  "servesBytes: an expired file has no bytes left at all",
  servesBytes("expired", "upload"),
  false,
);

check(
  "servesBytes: a file still being processed is not offered",
  [servesBytes("pending", "upload"), servesBytes("parsing", "upload")],
  [false, false],
);

check(
  "servesBytes: a deleted file (no live row) is not offered",
  servesBytes(undefined, "upload"),
  false,
);

check(
  "servesBytes: a status this build has no word for is refused, not assumed",
  servesBytes("quarantined", "upload"),
  false,
);

// ---------------------------------------------------------------------------
// What a sent chip says about its file
// ---------------------------------------------------------------------------

check(
  "sentFileState: a stored file shows its size and says nothing else",
  sentFileState("ready", 3400),
  { label: "3.3 KB", note: null, gone: false, problem: false },
);

check(
  "sentFileState: a deleted file is a struck-through record, not a fault",
  (({ label, gone, problem }) => ({ label, gone, problem }))(
    sentFileState(undefined, 3400),
  ),
  { label: "Removed", gone: true, problem: false },
);

check(
  "sentFileState: an EXPIRED file is the same quiet tombstone - nothing went wrong",
  (({ label, gone, problem }) => ({ label, gone, problem }))(
    sentFileState("expired", 3400),
  ),
  { label: "Expired", gone: true, problem: false },
);

check(
  "sentFileState: an expired chip says its own consequence, not the draft tray's",
  sentFileState("expired", 3400).note,
  "This file is no longer stored. It can no longer be downloaded.",
);

check(
  "sentFileState: a FAILED file is a fault, and is NOT struck through - it is still there",
  (({ label, gone, problem }) => ({ label, gone, problem }))(
    sentFileState("failed", 3400),
  ),
  { label: "Failed", gone: false, problem: true },
);

check(
  "sentFileState: a failed chip says the answer drew nothing from it, not that it is lost",
  sentFileState("failed", 3400).note,
  "This file could not be read, so the answer drew nothing from it. " +
    "The original is still stored, exactly as it was sent.",
);

check(
  "sentFileState: a file still being processed reads as ordinary, not as broken",
  sentFileState("parsing", 3400),
  { label: "3.3 KB", note: null, gone: false, problem: false },
);

// ---------------------------------------------------------------------------
// Joining the static message snapshots to the conversation's live rows
// ---------------------------------------------------------------------------

/** One live attachment row as the conversation payload reports it. */
function heldRow(over = {}) {
  return {
    id: "att-1",
    conversation_id: "c1",
    filename: "notes.txt",
    content_type: "text/plain",
    size_bytes: 12,
    status: "ready",
    origin: "upload",
    error: null,
    created_at: "2026-08-08T10:00:00Z",
    updated_at: "2026-08-08T10:00:00Z",
    ...over,
  };
}

/** One turn carrying one file snapshot. */
function turnWith(over = {}) {
  return {
    attachments: [
      { id: "att-1", filename: "notes.txt", content_type: "text/plain", size_bytes: 12, ...over },
    ],
  };
}

check(
  "fileLiveness: a live upload row carries its status through to the chip",
  [...fileLiveness([turnWith()], [heldRow()])],
  [["att-1", "ready"]],
);

check(
  "fileLiveness: an EXPIRED row is reported as expired, not as absent",
  fileLiveness([turnWith()], [heldRow({ status: "expired" })]).get("att-1"),
  "expired",
);

check(
  "fileLiveness: a FAILED row is reported as failed, so its chip can offer the original",
  fileLiveness([turnWith()], [heldRow({ status: "failed" })]).get("att-1"),
  "failed",
);

check(
  "fileLiveness: a file no live row mentions is absent - it was deleted",
  fileLiveness([turnWith()], []).has("att-1"),
  false,
);

check(
  "fileLiveness: a generated file is believed live, though `held` never lists it",
  fileLiveness([turnWith({ id: "att-9", origin: "generated" })], []).get("att-9"),
  "ready",
);

check(
  "fileLiveness: an UPLOAD is never assumed live from its snapshot alone",
  fileLiveness([turnWith({ id: "att-9", origin: "upload" })], []).has("att-9"),
  false,
);

check(
  "fileLiveness: a live row wins over the generated assumption",
  fileLiveness(
    [turnWith({ id: "att-1", origin: "generated" })],
    [heldRow({ status: "expired" })],
  ).get("att-1"),
  "expired",
);

check(
  "fileLiveness: a turn with no files at all contributes nothing",
  [...fileLiveness([{ role: "user", content: "hi" }], [])],
  [],
);

// ---------------------------------------------------------------------------
// A download hands over the file, or it hands over nothing and says why
// ---------------------------------------------------------------------------

/**
 * Run one download against a scripted response.
 *
 * `fetch`, `URL`'s object-URL pair and the three DOM calls the helper makes are
 * all stubbed - not to be clever, but because the click IS the save, and
 * whether it happens at all is the thing under test.
 *
 * Returns the requests made, what the browser was handed, and what was thrown.
 */
async function drive(responses) {
  const calls = [];
  const realFetch = globalThis.fetch;
  const realDocument = globalThis.document;
  const realCreate = URL.createObjectURL;
  const realRevoke = URL.revokeObjectURL;
  let blobReads = 0;
  globalThis.fetch = async (url, init) => {
    calls.push({ url, method: init?.method ?? "GET" });
    const next = responses[calls.length - 1];
    if (next === undefined) throw new Error(`unscripted request #${calls.length}`);
    // An Error in the script means "this request does not complete" - written
    // as a throw rather than a rejected promise built up front, which would sit
    // unhandled until the run reached it.
    if (next instanceof Error) throw next;
    // Counted, because "the refusal is never SAVED" and "the refusal is never
    // BUFFERED" are two different guarantees and only one of them is about
    // memory. Reading a body before checking the status would still pull a
    // whole object into this process on a path that is about to throw.
    const readBody = next.blob.bind(next);
    next.blob = async () => {
      blobReads += 1;
      return readBody();
    };
    return next;
  };
  let handed = null;
  let blob = null;
  let revoked = [];
  URL.createObjectURL = (b) => {
    blob = b;
    return "blob:stub-1";
  };
  URL.revokeObjectURL = (u) => revoked.push(u);
  const link = { href: "", download: "", rel: "", clicks: 0 };
  link.click = () => {
    link.clicks += 1;
    handed = { href: link.href, download: link.download };
  };
  link.remove = () => {};
  globalThis.document = {
    createElement: () => link,
    body: { append: () => {} },
  };
  // The revoke rides a timer the helper owns; capture it rather than sleep.
  const realTimeout = globalThis.setTimeout;
  const deferred = [];
  globalThis.setTimeout = (fn) => {
    deferred.push(fn);
    return 0;
  };
  let thrown = null;
  try {
    await downloadAttachment("att-1", "notes.txt");
  } catch (e) {
    thrown = e;
  } finally {
    globalThis.fetch = realFetch;
    globalThis.document = realDocument;
    globalThis.setTimeout = realTimeout;
  }
  // What had been revoked by the time the call returned - which must be
  // NOTHING, since the click only starts the save (see downloadAttachment).
  const revokedEarly = revoked.length;
  // The revoke is deferred a full second on purpose. Its callback is captured
  // rather than waited out, so the battery stays instant and the deferral is
  // free to grow. Run BEFORE the stubs come down, or the revoke it performs is
  // the real one and goes uncounted.
  for (const run of deferred.splice(0)) run();
  URL.createObjectURL = realCreate;
  URL.revokeObjectURL = realRevoke;
  const bytes = blob === null ? null : new Uint8Array(await blob.arrayBuffer());
  return {
    calls,
    clicks: link.clicks,
    handed,
    bytes,
    revoked,
    revokedEarly,
    blobReads,
    thrown,
  };
}

/** A problem+json refusal, exactly as the backend writes one. */
function problem(status, code, detail) {
  return new Response(JSON.stringify({ status, code, detail, type: "about:blank" }), {
    status,
    headers: { "content-type": "application/problem+json" },
  });
}

const FILE_BYTES = "%PDF-1.7 the actual file";
const servable = await drive([
  new Response(FILE_BYTES, {
    status: 200,
    headers: { "content-type": "application/pdf" },
  }),
]);

check(
  "downloadAttachment: one request per saved file, and it is the transfer itself",
  servable.calls,
  [{ url: "/api/attachments/att-1/content", method: "GET" }],
);

check(
  "downloadAttachment: the browser is handed the bytes that came back, under the chip's name",
  [
    servable.clicks,
    servable.handed,
    new TextDecoder().decode(servable.bytes),
  ],
  [1, { href: "blob:stub-1", download: "notes.txt" }, FILE_BYTES],
);

check(
  "downloadAttachment: the object URL is released, so a saved file is not a leak",
  servable.revoked,
  ["blob:stub-1"],
);

check(
  "downloadAttachment: but not while the save is still in flight - that writes an empty file",
  servable.revokedEarly,
  0,
);

check(
  "downloadAttachment: a servable file's body is read exactly once",
  servable.blobReads,
  1,
);

// The whole reason the transfer is a fetch rather than a navigation. Each of
// these would otherwise be saved to disk as "notes.txt" with a JSON error
// inside it, and nothing in the app would know it had happened.
for (const [label, status, code, detail] of [
  ["an expired file", 410, "attachment_expired", "This file aged out of retention."],
  ["a file still being processed", 409, "attachment_not_ready", "Still processing."],
  ["a file that is gone", 404, "not_found", "Attachment not found."],
  ["a client over its download budget", 429, "too_many_requests", "Too many requests."],
  ["a store that is down", 503, "service_unavailable", "Could not read the file."],
]) {
  const refused = await drive([problem(status, code, detail)]);
  check(
    `downloadAttachment: ${label} NEVER reaches the browser's saver`,
    [refused.clicks, refused.handed, refused.bytes],
    [0, null, null],
  );
  check(
    `downloadAttachment: ${label} is never even buffered`,
    refused.blobReads,
    0,
  );
  check(
    `downloadAttachment: ${label} is reported in the backend's own words`,
    [
      refused.thrown instanceof ApiError,
      refused.thrown?.status,
      refused.thrown?.code,
      refused.thrown?.message,
    ],
    [true, status, code, detail],
  );
}

const unreachable = await drive([new Error("the backend went away")]);

check(
  "downloadAttachment: a transfer that never completes saves nothing and still raises",
  [unreachable.clicks, unreachable.thrown instanceof Error],
  [0, true],
);

// ---------------------------------------------------------------------------
// What a refusal means for the chip that asked
// ---------------------------------------------------------------------------

check(
  "livenessAfterRefusal: an aged-out file keeps its row and becomes a tombstone",
  livenessAfterRefusal(410, "attachment_expired"),
  "expired",
);

check(
  "livenessAfterRefusal: a 410 over a file that was never finished leaves nothing to describe",
  livenessAfterRefusal(410, "attachment_unreadable"),
  "forget",
);

check(
  "livenessAfterRefusal: a gone row or gone bytes are forgotten",
  livenessAfterRefusal(404, "not_found"),
  "forget",
);

check(
  "livenessAfterRefusal: still-processing says nothing about the file, so the chip stands",
  livenessAfterRefusal(409, "attachment_not_ready"),
  "keep",
);

check(
  "livenessAfterRefusal: a rate limit or an outage must never retire a healthy chip",
  [livenessAfterRefusal(429, "too_many_requests"), livenessAfterRefusal(503)],
  ["keep", "keep"],
);

// ---------------------------------------------------------------------------
// What a send records about the files it takes, and which it must chase
// ---------------------------------------------------------------------------

check(
  "carriedFileStates: a ready file is recorded ready and needs no chasing",
  carriedFileStates([{ serverId: "srv-1", status: "ready" }]),
  { states: [["srv-1", "ready"]], unsettled: [] },
);

check(
  "carriedFileStates: a file still ingesting is recorded AS IT IS, not as ready",
  carriedFileStates([{ serverId: "srv-1", status: "parsing" }]).states,
  [["srv-1", "parsing"]],
);

check(
  "carriedFileStates: both unsettled states are collected for the post-turn read",
  carriedFileStates([
    { serverId: "srv-1", status: "pending" },
    { serverId: "srv-2", status: "parsing" },
    { serverId: "srv-3", status: "ready" },
  ]).unsettled,
  ["srv-1", "srv-2"],
);

check(
  "carriedFileStates: a chip with no server id names no row, so it is dropped",
  carriedFileStates([{ serverId: null, status: "ready" }]),
  { states: [], unsettled: [] },
);

check(
  "carriedFileStates: the CLIENT-only states are not lifecycle states and are dropped",
  carriedFileStates([
    { serverId: "srv-1", status: "uploading" },
    { serverId: "srv-2", status: "deleting" },
  ]),
  { states: [], unsettled: [] },
);

// ---------------------------------------------------------------------------
// What survives the proxy hop on the way back
// ---------------------------------------------------------------------------

/** A backend HEAD answer, with every header the download depends on. */
function headAnswer(over = {}) {
  return new Response(null, {
    status: 200,
    headers: {
      "content-type": "application/pdf",
      "content-disposition": 'attachment; filename="q3.pdf"',
      "content-length": "8123",
      "cache-control": "private, no-store",
      "x-content-type-options": "nosniff",
      "set-cookie": "session=leaked",
      ...over,
    },
  });
}

const relayed = relayHeaders(headAnswer());

check(
  "relayHeaders: the headers that make a download a download all survive the hop",
  [
    relayed.status,
    relayed.headers.get("content-type"),
    relayed.headers.get("content-disposition"),
    relayed.headers.get("content-length"),
    relayed.headers.get("cache-control"),
    relayed.headers.get("x-content-type-options"),
  ],
  [
    200,
    "application/pdf",
    'attachment; filename="q3.pdf"',
    "8123",
    "private, no-store",
    "nosniff",
  ],
);

check(
  "relayHeaders: anything outside the allowlist is dropped, not passed on",
  relayed.headers.get("set-cookie"),
  null,
);

check("relayHeaders: a HEAD answer carries no body", relayed.body, null);

check(
  "relayHeaders: a refusal keeps its status so the chip can react to it",
  relayHeaders(new Response(null, { status: 410 })).status,
  410,
);

console.log(
  failures === 0
    ? `message transitions: ${checks} checks passed`
    : `message transitions: ${failures} of ${checks} checks FAILED`,
);
process.exit(failures === 0 ? 0 : 1);
