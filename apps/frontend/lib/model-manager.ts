/**
 * Browser-side helpers for the Model Manager endpoints.
 *
 * All calls go through the same-origin `/api/models/*` proxy routes. The shapes
 * mirror the backend's `app/schemas/models.py` wire contract by hand, which is
 * this app's convention (nothing imports the generated `packages/api-types`) -
 * and unavoidable for the streamed halves anyway: a `StreamingResponse` has no
 * response model, so the SSE payloads are absent from the OpenAPI schema.
 *
 * Every fetcher throws {@link ApiError} carrying the backend's own problem+json
 * detail, and every stream reports failures as typed frames. Neither is
 * paraphrased anywhere: these messages were written to explain themselves.
 */

import { ApiError, raise } from "@/lib/api-error";
import { parseSSE } from "@/lib/sse";

/** How much a model may drive a turn, weakest first. */
export type AgencyMode = "deterministic" | "assisted" | "agentic";

/**
 * What each agency word actually permits, in plain language.
 *
 * Lives here rather than in either surface that renders it, because BOTH do -
 * the card explaining what a model runs at now, and the dialog explaining what
 * an operator is about to permit - and they have to tell one story. Two copies
 * of this wording is how a reader comes to believe the two screens are talking
 * about different things.
 *
 * Written as a consequence, never as a definition. "Agentic means the model has
 * agency" tells nobody anything; what a reader needs is what the DEPLOYMENT
 * stops doing on their behalf when it is switched on.
 */
export const AGENCY_GLOSS: Record<AgencyMode, string> = {
  deterministic:
    "Never calls a tool. Answers from what the prompt already carries, plus the grounding the deployment fetches for it.",
  assisted:
    "Tools are offered alongside the usual grounding, and the model may decline them. Nothing it does replaces work the deployment was already doing.",
  agentic:
    "The model drives: the automatic web pre-fetch and the ambient knowledge base are withdrawn, and what gets looked up is its decision.",
};

/**
 * What granting or withholding reason-and-call-tools actually does, in plain
 * language.
 *
 * Beside {@link AGENCY_GLOSS} and for the same reason: the card explaining what
 * a model does now and the dialog explaining what an operator is about to
 * permit both render it, and two copies of this wording is how a reader comes
 * to believe the two screens describe different things.
 *
 * Written as a consequence rather than a definition, and the `false` line names
 * the failure mode outright. It has to: the thing this axis guards against
 * returns an EMPTY answer rather than an error, so a reader who grants it on a
 * pair that cannot do it sees blank replies and nothing anywhere reporting a
 * fault.
 */
export const THINKING_WITH_TOOLS_GLOSS: Record<"true" | "false", string> = {
  true: "Reasoning and tool calls happen in the same turn: the model can think about a question and look something up before it answers.",
  false:
    "A turn does one or the other. When the model reasons it is offered no tools, and answers stay grounded through the deployment's usual pre-fetch instead.",
};

/**
 * Why granting reason-and-call-tools is the dangerous direction.
 *
 * Its own constant because it belongs wherever the grant is offered, and
 * because it is the sentence that must survive editing: on some model and
 * runtime pairs, asking for both at once returns nothing at all. No exception,
 * no error frame, no empty-response warning - just a blank answer, on a
 * deployment that looks healthy.
 */
export const THINKING_WITH_TOOLS_HAZARD =
  "The failure here is silent. On some model and runtime combinations, reasoning with tools offered returns an empty answer - not an error, not a warning, just nothing - so a wrong grant shows up as blank replies rather than as anything a log would catch.";

/** Whether a chat template can take a tool list at all. */
export type ToolsMode = "native" | "none";

/** How a model treats the think flag. */
export type ThinkMode = "always" | "hybrid" | "never" | "unknown";

/** The serving backend, and which management verbs it actually supports. */
export interface ProviderInfo {
  name: string;
  /** The runtime's version when it exposes one; null is "it does not", never "old". */
  runtime_version: string | null;
  can_pull: boolean;
  can_delete: boolean;
  can_list_installed: boolean;
}

/** What a tool-check probe MEASURED on the installed weights. */
export interface MeasuredFacts {
  depth: string;
  /** Null on a run that was not representative: a CPU-bound rate describes the spill. */
  tokens_per_second: number | null;
  vram_bytes: number | null;
  measured_at: string;
  runtime_version: string | null;
  /** The runtime changed since; the numbers describe a different serving stack. */
  runtime_stale: boolean;
  /** The harness cases changed since; the verdict answers a slightly different question. */
  harness_stale: boolean;
  /**
   * The probe does not describe the model server now being served from - a
   * different host, or a stored row that recorded none.
   *
   * Optional, and it is absent on this build: the field ships with the
   * endpoint binding on the backend, and the display side is here first so the
   * two can land separately without a window where a card reads "Checked" over
   * an agency line that has withdrawn the certificate. An absent value must
   * behave exactly like `false` - a build that read it as stale would put a
   * warning on every row of a healthy deployment.
   */
  endpoint_stale?: boolean;
  /** False means the model spilled off the GPU while it ran: it certifies nothing. */
  representative: boolean;
  gpu_fraction: number | null;
  /**
   * What a deep run ESTABLISHED about reasoning while calling tools, with no
   * declaration folded in.
   *
   * `true` means a certificate holds, so storing a grant takes effect at once;
   * `false` means a still-bound run measured the capability absent; `null`
   * means nothing usable measured it here.
   *
   * Never derive this from {@link ManagerRow.thinking_with_tools_effective} -
   * that is false whenever nothing DECLARES the capability, however well the
   * check went, so the derivation reports a passing deep run waiting on a
   * click as a check that failed. Only the server can tell the two apart.
   *
   * Optional so a client built against an older payload keeps working; absent
   * must read as `null` ("not established here"), never as a failure.
   */
  thinking_with_tools_passed?: boolean | null;
}

/**
 * Tool-check verdicts the installed weights have outrun.
 *
 * Derived by the backend from the probe log against the digest the tag points
 * at right now, so it appears when weights change and clears itself when a
 * check runs against the new ones. Null means it does not apply - including the
 * ordinary "nothing has ever been checked" case, which is not a warning.
 */
export interface WeightsChangeNote {
  /** Orphaned depths, weakest first; the LAST is the one the fields below describe. */
  depths: string[];
  /** What that verdict earned on the weights it measured - never a claim about these. */
  earned: AgencyMode;
  measured_at: string;
}

/** Which disagreement about the think flag a note is reporting. */
export type ThinkNoteCode = "measured_mismatch" | "nil_promotion";

/**
 * A disagreement about how this model treats the think flag.
 *
 * Three sources answer that question - what a check MEASURED on the installed
 * weights, what the resolution chain DECLARES, and what the runtime advertises
 * - and agreement is the ordinary case, which produces no note at all. Derived
 * per request like {@link WeightsChangeNote}, so it clears itself the moment a
 * check on the installed weights agrees.
 */
export interface ThinkModeNote {
  code: ThinkNoteCode | (string & {});
  /** What the check measured; null when nothing has measured these weights. */
  measured: ThinkMode | null;
  /** What the resolution says WITHOUT the measurement. */
  declared: ThinkMode;
  /** The runtime advertises think plumbing - about the serving stack, not the weights. */
  runtime_thinking: boolean;
  /** The backend's own sentence: the disagreement, and what it means. */
  message: string;
  /** A measurement bound to the installed weights exists, so confirming it would stick. */
  confirmable: boolean;
  /** Turns are ALREADY forcing think:true on the strength of the measurement. */
  routing_clamped: boolean;
}

/**
 * Why a clamp applies, when one does.
 *
 * `endpoint_mismatch` is the near-sibling of `unverified_weights` and exists
 * separately because the two send a reader to different actions: a passing
 * verdict for these weights DOES exist, it was just measured on another server
 * (or on one nothing recorded), so the fix is re-running the check here.
 *
 * `uncatalogued` outlived the catalog it was named for: the backend still emits
 * it for a tag nothing declares, and its own sentence - which is the only thing
 * rendered - no longer mentions one. Nothing here switches on it, so it is left
 * as the wire word rather than renamed on one side of a contract.
 */
export type AgencyClampCode =
  | "declared_ceiling"
  | "uncatalogued"
  | "failed_case"
  | "unverified_weights"
  | "endpoint_mismatch"
  | "tools_none"
  | "env_clamp";

/** How much this model may drive, and what is holding it back. */
export interface AgencyReport {
  ceiling: AgencyMode;
  earned: AgencyMode;
  env_clamp: AgencyMode | null;
  effective: AgencyMode;
  /** Null exactly when `clamp_detail` is - either serves as the presence test. */
  clamp_code: AgencyClampCode | null;
  clamp_detail: string | null;
}

/** One model in the management view: what is known about it, joined to live state. */
export interface ManagerRow {
  model: string;
  provider: string;
  /** Empty when the runtime's report establishes neither chat nor embedding. */
  role: string;
  installed: boolean;
  installed_size_bytes: number | null;
  digest: string | null;
  num_ctx: number | null;
  /** The window the WEIGHTS were trained for; `num_ctx` is what this deployment asks of it. */
  native_context_length: number | null;
  /** How hard the installed weights were quantized, in the runtime's own wording. */
  quantization: string | null;
  /** The parameter count as the runtime words it ("9.7B"). */
  parameter_size: string | null;
  /** The architecture family the runtime names ("qwen3"). */
  family: string | null;
  tools: ToolsMode;
  think: ThinkMode;
  /**
   * The DECLARED reason-and-call-tools claim - a stored operator declaration,
   * or nothing: no other source can grant this axis.
   *
   * A claim, never a measurement. What a turn will actually do is
   * `thinking_with_tools_effective`, and rendering this one in its place would
   * announce a capability a granted-but-unproven model is not being given.
   */
  thinking_with_tools: boolean;
  /**
   * What a TURN will actually do: the claim above, folded against the deep
   * verdict bound to these weights.
   *
   * The value any display should lead with. A stored grant reaches a turn only
   * once a deep check has passed the reasoning case at the running harness
   * version, on these weights and on the host being served from - so declared
   * and effective differ for exactly as long as it takes to run one.
   */
  thinking_with_tools_effective: boolean;
  /**
   * The claim came from a stored declaration bound to the installed weights -
   * the only source that can make it, so false here means nothing has been
   * granted, not that some other default stands in.
   *
   * The sibling of `ceiling_declared`, digest-gated for the same reason: a
   * re-pull retires the declaration, so a dialog has to know whether it is
   * about to replace one.
   */
  thinking_with_tools_declared: boolean;
  /**
   * The pair sibling that serves this tag's thinking turns, when one is set.
   *
   * The environment's own pair override where a deployment set one, else a
   * stored operator declaration - the same order a turn resolves it in, so
   * this names the tag chat actually routes to. That order is why a
   * declaration can lose: an operator who set the override in the environment
   * is not overruled by a click. The sibling has no row of its own.
   */
  thinking_model: string | null;
  /** The sibling above came from a declaration made here, not from the environment. */
  thinking_model_declared: boolean;
  vision: boolean;
  agency: AgencyReport | null;
  /** Null means unmeasured, which must render as such rather than as an estimate. */
  measured: MeasuredFacts | null;
  /**
   * Verdicts a change of weights left behind; null when none were.
   *
   * A third, distinct signal: `measured` only ever describes the INSTALLED
   * weights, its staleness flags are about the runtime moving under a still-
   * valid verdict, and this one is about the weights moving out from under it.
   */
  weights_change: WeightsChangeNote | null;
  /**
   * A disagreement about how this model treats the think flag; null when the
   * sources agree, which is the ordinary case and not a warning.
   *
   * Distinct from `think`, which is the ONE answer a turn acts on: this says
   * the sources it was resolved from do not match, and which to believe.
   */
  think_mode: ThinkModeNote | null;
  can_pull: boolean;
  can_delete: boolean;
  /**
   * A tool check can run against this row right now.
   *
   * The server's answer, never re-derived here: what may be done to a model
   * depends on facts only the backend holds, and a client guessing from the
   * role alone guesses wrongly the first time the runtime holds a chat model
   * nothing has described.
   */
  checkable: boolean;
  /** Why Delete is unavailable when something still depends on this model. */
  delete_blocked_reason: string | null;
  /** Conversations set to use this tag; a delete clears those pins. */
  pinned_conversations: number;
  /** This tag is the STORED workspace default; deleting it needs a successor. */
  workspace_default: boolean;
  /**
   * This tag is the one an unpinned turn actually runs - the stored default
   * where there is one, the deployment binding otherwise. Exactly one row in a
   * payload carries it.
   *
   * Not the same question as `workspace_default`, and the difference shows on
   * every deployment where nobody has stored a default: that flag is false on
   * all of them, while one model plainly still answers by default.
   */
  is_default: boolean;
  /**
   * The ceiling in `agency` came from a stored declaration, rather than from
   * what the model server's own capability report allows.
   *
   * The two are read in different places - a declared ceiling is changed from
   * this console, the report is not editable at all - and a re-pull retires
   * the declaration and leaves the report to answer alone, so a dialog has to
   * know which it is about to replace.
   */
  ceiling_declared: boolean;
}

/**
 * Whether this row could serve as somebody else's thinking partner.
 *
 * The candidate rule, in ONE place because two surfaces ask - the menu that
 * offers partners, and the hint that explains why a model has none - and the
 * backend refuses on the same test. Read off the row's resolved think mode,
 * which is the full chain (a stored confirmation, a measurement, this
 * deployment's own default, the runtime's template) rather than any single
 * source.
 *
 * "unknown" is excluded deliberately: it is what the resolvers answer when
 * NOTHING has established how the weights treat the flag, so offering it would
 * be a guess wearing a capability's clothes.
 */
export function canReason(row: ManagerRow): boolean {
  return row.installed && (row.think === "always" || row.think === "hybrid");
}

/**
 * Whether this row could serve as somebody else's thinking partner.
 *
 * TWO independent bars, and the second one is the one that bit. A partner has
 * to reason - and it has to REPLY. A vision model measured `always` clears the
 * first perfectly and fails the second: it reasons, and its visible content
 * comes back empty, so every thinking turn routed to it returns a blank.
 *
 * `agency` is the chat-serving tell rather than the role string, on the same
 * rule the row's own controls use: only a chat row is given an agency report,
 * and a tag the runtime vouches for gets one even where its report establishes
 * no role - which is exactly the row a remote-inventory deployment needs to
 * pair.
 *
 * The backend refuses on the same two bars. This filter exists so the menu
 * never offers what the writer would reject, not as the enforcement.
 */
export function canPartner(row: ManagerRow): boolean {
  return canReason(row) && row.agency !== null;
}

/**
 * The row's weights-change note, or null when it has nothing to say.
 *
 * The single place that decides whether a row HAS one, because four surfaces
 * ask - the collapsed pill's word, the pill's tone, the body's note, and the
 * page's standing banner - and they must never disagree. A warn pill over a
 * body holding no warning is exactly the contradiction this rules out. An
 * empty `depths` is not a note: the backend sends null instead, and only one
 * place here should have to know that the wire is unvalidated JSON.
 */
export function weightsChangeOf(row: ManagerRow): WeightsChangeNote | null {
  const note = row.weights_change;
  return note !== null && note.depths.length > 0 ? note : null;
}

/**
 * The row's think-mode note, or null when it has nothing to say.
 *
 * The single place that decides whether a row HAS one, for exactly the reason
 * {@link weightsChangeOf} exists: three surfaces ask - the collapsed pill's
 * word, the pill's tone, and the body's note - and a warn pill over a body
 * holding no warning is the contradiction that rules out. A note with no
 * message is not a note: the wire is unvalidated JSON, and only one place here
 * should have to know that.
 */
export function thinkModeOf(row: ManagerRow): ThinkModeNote | null {
  const note = row.think_mode;
  return note !== null && note.message !== "" ? note : null;
}

/**
 * The row's note ONLY when a check actually measured a disagreement.
 *
 * Separate from {@link thinkModeOf} because the two note codes are not the same
 * kind of statement and one surface must not treat them alike. A
 * `measured_mismatch` says a check watched these weights do something the
 * declaration denies - a comparison, and the strongest thing the check column
 * can say. A `nil_promotion` fires precisely when NOTHING has measured them, so
 * a column reading "Think mode differs" over it would assert a comparison
 * nobody made and bury the honest, actionable "Not checked".
 *
 * Both still warn and both still render their note in the body: the hazard is
 * real, it is just not a difference.
 */
export function thinkMismatchOf(row: ManagerRow): ThinkModeNote | null {
  const note = thinkModeOf(row);
  return note !== null && note.code === "measured_mismatch" ? note : null;
}

/** The rows serving one role, and how much this page may do to them. */
export interface RoleGroup {
  role: string;
  management: "manageable" | "locked" | "status_only";
  lock_reason: string | null;
  rows: ManagerRow[];
}

/** One collection found in the vector store. */
export interface VectorSpaceRow {
  name: string;
  /** Persistent knowledge base, or the ephemeral per-session scope. */
  kind: "kb" | "session";
  points: number;
  embed_models: string[];
  readable: boolean;
}

/** Whether stored vectors are still reachable by the active embedder. */
export interface StrandedVectorsReport {
  /** False means the report says nothing at all - never that everything is fine. */
  known: boolean;
  active_model: string | null;
  /** The alarm, and the only field a user-facing surface may act on. */
  stranded: boolean;
  stranded_points: number;
  stranded_models: string[];
  stranded_session_points: number;
  readable_points: number;
  spaces: VectorSpaceRow[];
  message: string | null;
}

/**
 * What a gap is, in the one distinction a reader can act on: what would fix it.
 *
 * `embedder_missing` is closed by a DOWNLOAD, `embedder_unselected` by a
 * configuration CHOICE (a download would not help), `tools_incapable` by a
 * different MODEL, and `tools_clamped` by a declaration or a passing check -
 * where downloading more of the same would be wasted effort.
 */
export type RoleGapCode =
  | "embedder_missing"
  | "embedder_unselected"
  | "tools_incapable"
  | "tools_clamped";

/** Something this server's INVENTORY cannot do, in feature terms. */
export interface RoleGap {
  role: string;
  /** `fatal` = a feature area is dead until somebody acts; `note` = it could work better. */
  severity: "fatal" | "note";
  /**
   * One of the known codes - or one this build has not learned yet. The wire
   * is unvalidated JSON and the backend ships first, so a fifth code must
   * degrade to a generic title, never to a card missing its scan line.
   */
  code: RoleGapCode | (string & {});
  /** The backend's own sentence: what stops working, and what would fix it. */
  message: string;
  /** A tag to DOWNLOAD, when exactly one is obviously right; null when the fix is a choice. */
  suggested_model: string | null;
}

/**
 * The whole management view: one request-time join, nothing stored.
 *
 * Every installed tag sits in exactly one group, so `groups` is the complete
 * inventory rather than the described part of it.
 */
export interface ModelManagerView {
  provider: ProviderInfo;
  groups: RoleGroup[];
  vectors: StrandedVectorsReport | null;
  /** What the inventory cannot do, worst first. Empty is the healthy answer. */
  role_gaps: RoleGap[];
}

/** One model the serving runtime holds in memory right now. */
export interface ResidentModel {
  model: string;
  total_bytes: number | null;
  vram_bytes: number | null;
  /** Near zero means the model spilled to the CPU; it still answers, far slower. */
  gpu_fraction: number | null;
}

/** What the runtime holds resident, and whether it reports that at all. */
export interface ResidencyView {
  /** False means this payload says NOTHING - not that nothing is loaded. */
  reported: boolean;
  models: ResidentModel[];
}

/** What the registry says about a tag, asked on demand and never guessed. */
export interface PreflightResult {
  name: string;
  kind: "library" | "huggingface";
  status: "available" | "not_found" | "unknown";
  /** Exact bytes the download would transfer; null unless status is available. */
  download_size_bytes: number | null;
  /**
   * How hard the weights were quantized, in the registry's own wording; null
   * where it publishes none. What makes the size above mean something.
   */
  quantization: string | null;
  /** The parameter count as the registry words it ("8.2B"); null when unpublished. */
  parameter_size: string | null;
  warning: string | null;
  detail: string | null;
}

/** One `progress` frame of a download stream. */
export interface PullProgress {
  /** The backend's own progress wording: render it, never switch on it. */
  status: string;
  completed: number | null;
  total: number | null;
  /** Null whenever the byte counts are absent - a stage with no transfer behind it. */
  percent: number | null;
  digest: string | null;
}

/** The terminal frame of a download that succeeded. */
export interface PullDone {
  model: string;
  status: string;
  /** A tool check follows on this same stream. */
  verifying: boolean;
  verify_skipped_reason: string | null;
}

/** The terminal frame of a stream that could not finish its work. */
export interface StreamError {
  /** A sentence safe to show as it stands; never exception text. */
  message: string;
  code: "model_not_found" | "provider_unavailable" | "internal_error";
}

/** One finished sample of a running check. */
export interface VerifyProgress {
  case: string;
  model: string;
  passed: boolean;
  detail: string;
  latency_s: number;
  completed: number;
  total: number;
}

/** Where the model sat while it was being measured. */
export interface VerifyResidency {
  /** Null is "the backend does not report residency", never "no". */
  gpu_resident: boolean | null;
  gpu_fraction: number | null;
  vram_bytes: number | null;
  total_bytes: number | null;
}

/** The terminal frame of a check that ran: verdicts plus what was observed. */
export interface VerifyDone {
  model: string;
  thinking_model: string | null;
  depth: "fast" | "deep";
  /** The EARNED half only - the declared ceiling still clamps beside it. */
  earned: AgencyMode;
  thinking_with_tools: boolean;
  /**
   * How the run watched the CHECKED tag treat the think flag - never the pair's
   * thinking sibling. Null when it could not tell. A measurement, not a
   * declaration: it becomes one only when an operator confirms it.
   */
  measured_think_mode: "always" | "hybrid" | "never" | null;
  /** Per-case `[passed, run]` counts. */
  tallies: Record<string, number[]>;
  tokens_per_second: number | null;
  harness_version: string;
  digest: string | null;
  runtime_version: string | null;
  residency: VerifyResidency;
  /** False marks a run the model spilled off the GPU for: it certifies nothing. */
  representative: boolean;
  warning: string | null;
  probe_id: string | null;
}

/**
 * One frame off a pull or verify stream, in the shape the page reduces over.
 *
 * A download that continues into its automatic check carries BOTH families on
 * one connection, which is why they share a type rather than a stream each.
 */
export type ManagerStreamEvent =
  | { type: "progress"; data: PullProgress }
  | { type: "done"; data: PullDone }
  | { type: "error"; data: StreamError }
  | { type: "verify_progress"; data: VerifyProgress }
  | { type: "verify_done"; data: VerifyDone }
  | { type: "verify_error"; data: StreamError };

/** The management view as the backend now holds it. */
export async function getManagerView(): Promise<ModelManagerView> {
  const resp = await fetch("/api/models/manager", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ModelManagerView;
}

/** What the serving runtime is holding in memory right now. */
export async function getResidency(): Promise<ResidencyView> {
  const resp = await fetch("/api/models/residency", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ResidencyView;
}

/** Ask the registry whether a name exists and what it would cost to download. */
export async function preflightModel(name: string): Promise<PreflightResult> {
  const resp = await fetch("/api/models/preflight", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as PreflightResult;
}

/** What a delete did beyond removing the weights. */
export interface ModelDeleteResult {
  /** The normalized tag that was removed. */
  removed: string;
  /** Conversations whose pin on the tag was cleared; they follow the default now. */
  unpinned_conversations: number;
  /** The successor written as the workspace default, when this delete replaced it. */
  new_default: string | null;
}

/**
 * Remove downloaded weights.
 *
 * The tag is split on its slashes so each piece is encoded separately: an
 * `hf.co/org/repo` name has to reach the catch-all route as real path segments,
 * while a colon inside one segment must not be.
 *
 * `replacement` names the successor for the workspace default and is only
 * meaningful when the deleted tag IS the default - the backend requires it
 * there and ignores it everywhere else.
 */
export async function deleteModel(name: string, replacement?: string): Promise<ModelDeleteResult> {
  const path = name.split("/").map(encodeURIComponent).join("/");
  const query = replacement ? `?replacement=${encodeURIComponent(replacement)}` : "";
  const resp = await fetch(`/api/models/${path}${query}`, { method: "DELETE" });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ModelDeleteResult;
}

/** What confirming a measured think mode stored. */
export interface ThinkModeResult {
  model: string;
  think_mode: "always" | "hybrid" | "never";
  /** The weights the declaration is bound to; it retires when the tag moves past them. */
  digest: string;
  probe_id: string;
}

/**
 * Store what a tool check measured about a model's think flag.
 *
 * The request carries a NAME and nothing else: the value stored is the measured
 * one, so this cannot become a way to declare a mode nothing observed. A model
 * nothing has measured is refused with the backend's own sentence.
 */
export async function confirmThinkMode(name: string): Promise<ThinkModeResult> {
  const resp = await fetch("/api/models/think-mode", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ThinkModeResult;
}

/** What declaring an agency ceiling stored, and whether it changed anything. */
export interface CeilingResult {
  model: string;
  ceiling: AgencyMode;
  /** The weights the declaration is bound to; a re-pull past them retires it. */
  digest: string;
  /** What measurement has actually established for these weights. */
  earned: AgencyMode;
  /** The minimum over every axis - what a turn on this model may now assume. */
  effective: AgencyMode;
}

/**
 * Declare how much a model may be allowed to drive.
 *
 * A CEILING and only a ceiling. Raising it withdraws the human objection and
 * nothing else: `min(CEILING, EARNED)` still holds the model at whatever a deep
 * check has proven on the weights installed now, so an `agentic` declaration on
 * an unproven model is stored and starts applying when a check passes. The
 * response reports `earned` and `effective` beside the stored value precisely
 * so a page cannot announce an escalation that did not happen.
 *
 * Lowering it needs no evidence and applies to the next turn.
 */
export async function declareCeiling(name: string, ceiling: AgencyMode): Promise<CeilingResult> {
  const resp = await fetch("/api/models/ceiling", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, ceiling }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as CeilingResult;
}

/** What declaring a reasoning-with-tools claim stored, and what a turn now does. */
export interface ThinkingWithToolsResult {
  model: string;
  thinking_with_tools: boolean;
  /** The weights the declaration is bound to; a re-pull past them retires it. */
  digest: string;
  /** What a turn will actually do - the declaration folded against the measurement. */
  effective: boolean;
}

/**
 * Declare whether a model may reason and call tools in the same turn.
 *
 * Gated exactly as an `agentic` ceiling is, and by the same mechanism: a GRANT
 * is stored whatever the log holds and reaches a turn only once a deep check
 * has passed the reasoning case on the weights installed now, at the running
 * harness version and on the host being served from. So the response reports
 * `effective` beside the stored value, and a page must not announce a
 * capability from the write alone.
 *
 * Revoking needs no evidence and applies to the next turn - see
 * {@link THINKING_WITH_TOOLS_HAZARD} for why the safe direction is the cheap
 * one here.
 */
export async function declareThinkingWithTools(
  name: string,
  thinkingWithTools: boolean,
): Promise<ThinkingWithToolsResult> {
  const resp = await fetch("/api/models/thinking-with-tools", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, thinking_with_tools: thinkingWithTools }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ThinkingWithToolsResult;
}

/** What declaring a thinking pair stored. */
export interface ThinkingPairResult {
  model: string;
  thinking_model: string;
}

/**
 * Pair a direct-answer model with the partner that serves its thinking turns.
 *
 * A PREFERENCE, not a claim about weights: it follows both tags across a
 * re-pull rather than retiring with them. The backend refuses a partner whose
 * think mode does not already resolve to `always` or `hybrid`, on the same test
 * {@link canReason} applies here - so the menu and the writer cannot disagree
 * about which models can reason.
 */
export async function declareThinkingPair(
  name: string,
  thinkingModel: string,
): Promise<ThinkingPairResult> {
  const resp = await fetch("/api/models/thinking-pair", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name, thinking_model: thinkingModel }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ThinkingPairResult;
}

/**
 * Make a model the workspace default every unpinned conversation follows.
 *
 * Validated the way a delete's successor is: installed, and a role that can
 * actually drive a chat turn. A default nothing can route to is accepted
 * silently and surfaces one turn later, on somebody else's question.
 */
export async function setDefaultModel(name: string): Promise<{ model: string }> {
  const resp = await fetch("/api/models/default", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ name }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as { model: string };
}

/** Frame names this client knows, mapped to the event type it yields. */
const FRAME_TYPES = new Set<ManagerStreamEvent["type"]>([
  "progress",
  "done",
  "error",
  "verify_progress",
  "verify_done",
  "verify_error",
]);

/**
 * POST a stream and hand each frame to `onEvent` in arrival order.
 *
 * A refusal decided BEFORE the stream (an unusable name, a probe already
 * running, a model the backend does not hold) arrives as problem+json and is
 * thrown as an {@link ApiError}, exactly as it does on the plain endpoints - it
 * is not a stream that failed, it is a request that was never started, and the
 * caller must be able to tell those apart before it attaches a progress bar.
 * Frames the client does not know are skipped rather than treated as errors, so
 * a backend that grows a frame does not break this one.
 *
 * @param url - The same-origin proxy route to POST.
 * @param body - The JSON request body.
 * @param onEvent - Called once per known frame, in order.
 * @param signal - Aborts the request; the caller owns the controller.
 */
async function runStream(
  url: string,
  body: unknown,
  onEvent: (event: ManagerStreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const resp = await fetch(url, {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(body),
    cache: "no-store",
    signal,
  });
  if (!resp.ok) await raise(resp);
  if (!resp.body) {
    throw new ApiError("The stream ended before it began.", resp.status);
  }
  for await (const msg of parseSSE(resp.body)) {
    const type = msg.event as ManagerStreamEvent["type"];
    if (!FRAME_TYPES.has(type)) continue;
    // One parse, one cast: the frame name IS the discriminant on the wire, and
    // re-deriving it from the payload's own fields would only invent a second
    // contract to keep in step with the first.
    onEvent({ type, data: JSON.parse(msg.data) } as ManagerStreamEvent);
  }
}

/**
 * Download a model, streaming its progress.
 *
 * With `verify` true (the default the backend applies) a successful download
 * continues into the fast tool check on the SAME stream, so the verify frames
 * follow the download's `done` frame on this one call.
 */
export function streamPull(
  name: string,
  options: { verify?: boolean; onEvent: (event: ManagerStreamEvent) => void; signal?: AbortSignal },
): Promise<void> {
  return runStream(
    "/api/models/pull",
    { name, verify: options.verify ?? true },
    options.onEvent,
    options.signal,
  );
}

/**
 * Measure a model's tool-calling, streaming each finished sample.
 *
 * The stream stays open after the verdict frame while the check re-warms the
 * chat model and the embedder it evicted; that silence is not a hang.
 */
export function streamVerify(
  name: string,
  options: {
    depth: "fast" | "deep";
    onEvent: (event: ManagerStreamEvent) => void;
    signal?: AbortSignal;
  },
): Promise<void> {
  return runStream(
    "/api/models/verify",
    { name, depth: options.depth },
    options.onEvent,
    options.signal,
  );
}
