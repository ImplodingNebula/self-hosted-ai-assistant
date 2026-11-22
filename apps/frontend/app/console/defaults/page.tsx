"use client";

import { useCallback, useEffect, useMemo, useState } from "react";

import { ApiError } from "@/lib/api-error";
import {
  CEILING_DIGITS_ONLY,
  CONTEXT_PRESETS,
  contextPresetState,
  PRESET_CUSTOM,
  PRESET_UNSET,
} from "@/lib/context-presets";
import { chatModels, embeddingModels, listModels } from "@/lib/models";
import type { ModelEntry } from "@/lib/types";
import {
  defaultsDiff,
  getWorkspaceDefaults,
  updateWorkspaceDefaults,
  type SettingMode,
  type WorkspaceDefaults,
  type WorkspaceDefaultsPatch,
} from "@/lib/workspace-defaults";

/** The form's own state: the same seven axes, all optional, all clearable. */
interface Draft {
  model: string | null;
  embedding_model: string | null;
  thinking: SettingMode | null;
  web_search: SettingMode | null;
  /**
   * Held as the raw text the textarea shows, so the reader's own whitespace
   * survives editing. It is trimmed on the way to the wire, where blank means
   * "clear this" - the backend normalizes it the same way.
   */
  system_prompt: string;
  /** Held as the raw text the field shows, so a half-typed number is not lost. */
  context_ceiling: string;
  reranker: boolean | null;
}

// The value an enum <select> uses for "not set". The empty string is the only
// option value a <select> can hold that is not one of the three modes, and it
// round-trips to null on the wire.
const UNSET = "";

// Each mode with the sentence that says what it DOES, not what it is called.
// A three-letter label on a control that changes every future turn is a puzzle.
const MODE_OPTIONS: { value: SettingMode | ""; label: string }[] = [
  { value: UNSET, label: "Not set - the deployment's own default decides" },
  { value: "auto", label: "Auto - decide per message" },
  { value: "on", label: "On - always" },
  { value: "off", label: "Off - never" },
];

function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Try again.";
}

function toDraft(defaults: WorkspaceDefaults): Draft {
  return {
    model: defaults.model,
    embedding_model: defaults.embedding_model,
    thinking: defaults.thinking,
    web_search: defaults.web_search,
    system_prompt: defaults.system_prompt ?? "",
    context_ceiling: defaults.context_ceiling === null ? "" : String(defaults.context_ceiling),
    reranker: defaults.reranker,
  };
}

/**
 * Parse the ceiling field into what the wire wants, or report why it cannot.
 *
 * Checked here as well as on the backend for one reason only: a number typed
 * outside the accepted range should be refused beside the field that holds it,
 * not as a banner after a round trip. The backend's own bounds remain the
 * authority - they arrive in the payload, so this can never drift from them.
 * The spelling rule is shared with the preset select rather than restated, so
 * the two cannot come to disagree about what counts as a number.
 */
function readCeiling(
  raw: string,
  defaults: WorkspaceDefaults,
): { value: number | null } | { error: string } {
  const text = raw.trim();
  if (!text) return { value: null };
  if (!CEILING_DIGITS_ONLY.test(text)) {
    return { error: "Enter a whole number of tokens, or leave empty." };
  }
  const parsed = Number(text);
  if (!Number.isSafeInteger(parsed)) {
    return { error: "Enter a whole number of tokens, or leave empty." };
  }
  if (parsed < defaults.min_context_ceiling || parsed > defaults.max_context_ceiling) {
    return {
      error:
        `Enter a ceiling between ${defaults.min_context_ceiling.toLocaleString()} and ` +
        `${defaults.max_context_ceiling.toLocaleString()} tokens, or leave it empty.`,
    };
  }
  return { value: parsed };
}

/**
 * Read the persona field into what the wire wants, or say why it cannot.
 *
 * Blank - including whitespace alone - is a CLEAR, never a way to silence the
 * assistant's instructions: turning the system prompt off entirely is
 * deployment configuration, not a text box. The length is checked beside the
 * field rather than left to a 422, against the limit the payload carries, so
 * this can never drift from what the backend accepts.
 */
function readPrompt(
  raw: string,
  defaults: WorkspaceDefaults,
): { value: string | null } | { error: string } {
  const text = raw.trim();
  if (!text) return { value: null };
  // Counted by CODE POINT, not by `.length`. A JS string measures itself in
  // UTF-16 units, where an emoji is two, and the backend's bound counts Python
  // characters, where it is one - so `.length` would refuse a persona the API
  // accepts and report a number the reader cannot reconcile with what they see.
  const characters = [...text].length;
  if (characters > defaults.max_system_prompt_chars) {
    return {
      error:
        `Shorten this to ${defaults.max_system_prompt_chars.toLocaleString()} characters or ` +
        `fewer - it is ${characters.toLocaleString()} right now.`,
    };
  }
  return { value: text };
}

/**
 * The console's Workspace defaults page.
 *
 * Every control here writes one field of the tenant-wide defaults, which sit
 * ABOVE the environment in the resolution chain and BELOW anything a
 * conversation or a single request says. So each one is a fallback, never an
 * override, and the copy says so rather than leaving a reader to discover it
 * when a pinned thread ignores what they just set.
 */
export default function WorkspaceDefaultsPage() {
  const [loaded, setLoaded] = useState<WorkspaceDefaults | null>(null);
  const [draft, setDraft] = useState<Draft | null>(null);
  const [models, setModels] = useState<ModelEntry[]>([]);
  // Null until the listing answers, which is NOT the same as an empty one: the
  // note below reports a model server that could not be read, and a read still
  // in flight would have it saying so about a healthy deployment for as long as
  // the listing takes.
  const [embedders, setEmbedders] = useState<ModelEntry[] | null>(null);
  const [loading, setLoading] = useState(true);
  const [saving, setSaving] = useState(false);
  const [pageError, setPageError] = useState<string | null>(null);
  const [formError, setFormError] = useState<string | null>(null);
  const [saved, setSaved] = useState(false);

  // Promise callbacks (not async/await) keep every setState out of the
  // synchronous call graph, which the set-state-in-effect lint rule rejects for
  // the mount-time call below - the same idiom as the other console pages.
  const refresh = useCallback(
    () =>
      getWorkspaceDefaults()
        .then((next) => {
          setLoaded(next);
          setDraft(toDraft(next));
          setPageError(null);
        })
        .catch((err: unknown) => setPageError(errorMessage(err)))
        .finally(() => setLoading(false)),
    [],
  );

  // The model list is its own read and its own failure: a model server that
  // cannot be reached must not take the other controls down with it. The two
  // selects then fall back to whatever tag is already stored.
  //
  // One read feeds both roles, so the chat select and the embedding select can
  // never describe two different inventories of the same server.
  const refreshModels = useCallback(
    () =>
      listModels()
        .then((response) => {
          setModels(chatModels(response));
          setEmbedders(embeddingModels(response));
        })
        .catch(() => {
          setModels([]);
          setEmbedders([]);
        }),
    [],
  );

  useEffect(() => {
    void refresh();
    void refreshModels();
  }, [refresh, refreshModels]);

  /**
   * The tags the model select may offer.
   *
   * The stored tag joins the list even when the server no longer reports it:
   * dropping it would silently rewrite the saved default to whatever happened
   * to be first, and a default pointing at a model that is temporarily absent
   * is a state the resolution chain already handles by falling through.
   */
  const modelOptions = useMemo(() => {
    const tags = models.map((entry) => entry.model);
    const stored = loaded?.model;
    if (stored && !tags.includes(stored)) return [stored, ...tags];
    return tags;
  }, [models, loaded?.model]);

  /**
   * The tags the embedding select may offer.
   *
   * The stored tag joins the list on the same rule the chat select uses, and it
   * matters more here: this tag names the vector space the stored passages were
   * written into, so dropping it from the list would offer to move the corpus
   * out of reach of every query the moment the server stops reporting it.
   */
  const embedderOptions = useMemo(() => {
    const tags = (embedders ?? []).map((entry) => entry.model);
    const stored = loaded?.embedding_model;
    if (stored && !tags.includes(stored)) return [stored, ...tags];
    return tags;
  }, [embedders, loaded?.embedding_model]);

  /** The first complaint a save would raise, or null when the form is usable. */
  const formComplaint = useCallback((form: Draft, base: WorkspaceDefaults): string | null => {
    const ceiling = readCeiling(form.context_ceiling, base);
    if ("error" in ceiling) return ceiling.error;
    const prompt = readPrompt(form.system_prompt, base);
    if ("error" in prompt) return prompt.error;
    return null;
  }, []);

  /** What the form holds, in the shape the wire uses. Null when a field is unusable. */
  const asPatch = useCallback(
    (form: Draft, base: WorkspaceDefaults): WorkspaceDefaultsPatch | null => {
      const ceiling = readCeiling(form.context_ceiling, base);
      if ("error" in ceiling) return null;
      const prompt = readPrompt(form.system_prompt, base);
      if ("error" in prompt) return null;
      return defaultsDiff(base, {
        model: form.model,
        embedding_model: form.embedding_model,
        thinking: form.thinking,
        web_search: form.web_search,
        system_prompt: prompt.value,
        context_ceiling: ceiling.value,
        reranker: form.reranker,
      });
    },
    [],
  );

  const dirty = useMemo(() => {
    if (loaded === null || draft === null) return false;
    const patch = asPatch(draft, loaded);
    // A ceiling the field cannot parse is still a change the reader made, so
    // Save stays live and the complaint lands beside the field on the click.
    if (patch === null) return true;
    return Object.keys(patch).length > 0;
  }, [loaded, draft, asPatch]);

  const edit = useCallback((change: Partial<Draft>) => {
    setDraft((current) => (current === null ? current : { ...current, ...change }));
    setFormError(null);
    setSaved(false);
  }, []);

  const save = useCallback(() => {
    if (loaded === null || draft === null) return;
    const patch = asPatch(draft, loaded);
    if (patch === null) {
      setFormError(formComplaint(draft, loaded));
      return;
    }
    if (Object.keys(patch).length === 0) return;
    setSaving(true);
    setFormError(null);
    updateWorkspaceDefaults(patch)
      .then((next) => {
        setLoaded(next);
        setDraft(toDraft(next));
        setSaved(true);
      })
      .catch((err: unknown) => setFormError(errorMessage(err)))
      .finally(() => setSaving(false));
  }, [loaded, draft, asPatch, formComplaint]);

  const revert = useCallback(() => {
    if (loaded === null) return;
    setDraft(toDraft(loaded));
    setFormError(null);
    setSaved(false);
  }, [loaded]);

  // Read off the field on every render, so the preset select can never drift
  // from the number beside it - including after a save, a discard, or a reload.
  const presetState = contextPresetState(draft?.context_ceiling ?? "");

  return (
    <div data-part="workspace-defaults">
      <header className="wd-headline">
        <h1 className="mm-title">Workspace defaults</h1>
        <p className="wd-lede">
          What a turn does when nothing more specific decides. A conversation&apos;s own settings
          and a single request both win over everything here; anything left unset falls through to
          this deployment&apos;s configuration.
        </p>
      </header>

      {pageError && <p className="mm-banner error">{pageError}</p>}
      {loading && <p className="mm-muted">Loading...</p>}

      {loaded !== null && draft !== null && (
        <form
          className="wd-form"
          noValidate
          onSubmit={(event) => {
            event.preventDefault();
            save();
          }}
        >
          <section className="wd-section">
            <h2 className="mm-section-title">Answering</h2>

            <label className="wd-field">
              Default chat model
              <select
                className="wd-select wd-select-mono"
                value={draft.model ?? UNSET}
                disabled={saving}
                onChange={(event) => edit({ model: event.target.value || null })}
              >
                <option value={UNSET}>Not set - the deployment&apos;s configured model</option>
                {modelOptions.map((tag) => (
                  <option key={tag} value={tag}>
                    {tag}
                  </option>
                ))}
              </select>
            </label>
            <p className="wd-hint">
              {models.length === 0
                ? "The model list could not be read, so only the stored default is offered here. The Model manager page reports what the server actually holds."
                : "A model the server stops holding is not an error: the turn substitutes an installed one and says so."}
            </p>

            <label className="wd-field">
              Reasoning
              <select
                className="wd-select"
                value={draft.thinking ?? UNSET}
                disabled={saving}
                onChange={(event) =>
                  edit({ thinking: (event.target.value || null) as SettingMode | null })
                }
              >
                {MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="wd-hint">
              Reasoning is slower and, on most local models, cannot be combined with tool calls in
              one turn - so an always-on default trades lookups for deliberation.
            </p>

            <label className="wd-field">
              Web search
              <select
                className="wd-select"
                value={draft.web_search ?? UNSET}
                disabled={saving}
                onChange={(event) =>
                  edit({ web_search: (event.target.value || null) as SettingMode | null })
                }
              >
                {MODE_OPTIONS.map((option) => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </select>
            </label>
            <p className="wd-hint">
              Off means fully local: no page is fetched and the model is not offered a search tool
              at all.
            </p>

            <label className="wd-field">
              Workspace system prompt
              {/* No `rows`: the control's height is one number, and it lives
                  in the sheet with the rest of the field's box. Setting both
                  leaves two answers to disagree the next time either moves. */}
              <textarea
                className="wd-textarea"
                placeholder="Not set - the assistant's built-in persona"
                value={draft.system_prompt}
                disabled={saving}
                onChange={(event) => edit({ system_prompt: event.target.value })}
              />
            </label>
            <p className="wd-hint">
              Replaces the assistant&apos;s built-in persona for every conversation in this
              workspace; empty keeps the default. Instructions about using tools honestly are
              always added on top, so a rewritten persona never loses them.
            </p>
          </section>

          {/* Its own card rather than a fifth control under Answering: this one
              does not change how a turn answers, it decides which stored
              passages any turn can reach at all. */}
          <section className="wd-section">
            <h2 className="mm-section-title">Embedding</h2>

            <label className="wd-field">
              Embedding model
              {/* Inert only while the answer is unknown. An empty listing IS an
                  answer, and locking the control on one takes the repair away
                  with it: clearing a stored tag the server no longer holds is
                  what that state calls for, and Not set is a safe store from
                  every state. */}
              <select
                className="wd-select wd-select-mono"
                value={draft.embedding_model ?? UNSET}
                disabled={saving || embedders === null}
                onChange={(event) => edit({ embedding_model: event.target.value || null })}
              >
                <option value={UNSET}>Not set - resolved automatically</option>
                {embedderOptions.map((tag) => (
                  <option key={tag} value={tag}>
                    {tag}
                  </option>
                ))}
              </select>
            </label>
            <p className="wd-hint">
              The embedding model defines the vector space stored passages live in: changing it
              points retrieval at a fresh, empty space until content is re-ingested. Unset resolves
              automatically; with several embedders installed and none chosen, retrieval refuses
              until this control settles it.
            </p>
            {embedders !== null && embedders.length === 0 && (
              <p className="wd-hint">
                No embedding model could be read from the model server, so this control offers only
                what is already stored. The Model manager page reports what the server actually
                holds.
              </p>
            )}
          </section>

          <section className="wd-section">
            <h2 className="mm-section-title">Machine budget</h2>

            {/* A <div> and an explicit htmlFor, not the wrapping <label> the
                other fields use. Two controls under one label puts a labelable
                element inside it that is not its own, and the accessible name
                a label builds SUBSTITUTES an embedded control's value: the
                number field would be announced as "Context window ceiling
                16 GB card - 32,768", renamed by whatever the select beside it
                happens to say. */}
            <div className="wd-field">
              <label className="wd-field-label" htmlFor="context-ceiling">
                Context window ceiling
              </label>
              <div className="wd-field-row">
                <input
                  id="context-ceiling"
                  className="wd-input"
                  type="number"
                  inputMode="numeric"
                  min={loaded.min_context_ceiling}
                  max={loaded.max_context_ceiling}
                  step={1024}
                  placeholder="Not set"
                  value={draft.context_ceiling}
                  disabled={saving}
                  onChange={(event) => edit({ context_ceiling: event.target.value })}
                />
                {/* Fills the field in; it does not save, and it stores nothing
                    of its own - which is why its value is DERIVED from the
                    input rather than held, and why a preset chosen here is
                    indistinguishable from the same number typed by hand. */}
                <select
                  className="wd-select"
                  aria-label="Preset for the context window ceiling"
                  value={presetState}
                  disabled={saving}
                  onChange={(event) => edit({ context_ceiling: event.target.value })}
                >
                  {/* Bare "Not set": the hint below already says what empty
                      means, and under base-select the closed control renders
                      the selected option, so label length is control width. */}
                  <option value={PRESET_UNSET}>Not set</option>
                  {CONTEXT_PRESETS.map((preset) => (
                    <option key={preset.tokens} value={String(preset.tokens)}>
                      {preset.card} card - {preset.tokens.toLocaleString()}
                    </option>
                  ))}
                  {/* Disabled because it names a STATE, not an action: there is
                      no such thing as choosing "custom", only typing a number
                      no preset offers. Listed all the same, so the reader can
                      see where the field currently stands. */}
                  <option value={PRESET_CUSTOM} disabled>
                    Custom
                  </option>
                </select>
              </div>
            </div>
            <p className="wd-hint">
              The window a turn asks the model server for, in tokens. A turn asks for the smaller
              of this and the window the model itself was trained for, so it only ever narrows.
              Empty defers to this deployment&apos;s own default window; accepted range{" "}
              {loaded.min_context_ceiling.toLocaleString()} to{" "}
              {loaded.max_context_ceiling.toLocaleString()}. The presets are starting points for
              common cards, not measured fits.
            </p>

            <div className="wd-field">
              <span className="wd-field-label">Reranker</span>
              {loaded.reranker_status.configured ? (
                <div className="wd-toggle-row">
                  <button
                    type="button"
                    role="switch"
                    aria-checked={draft.reranker !== false}
                    aria-label={
                      draft.reranker === false
                        ? "Turn the reranker on for this workspace"
                        : "Turn the reranker off for this workspace"
                    }
                    disabled={saving}
                    // ON stores a CLEAR, not a true. The two are behaviourally
                    // identical - neither can start a sidecar the deployment
                    // did not wire - so writing true would leave a stored key
                    // that says nothing the absent key did not already say.
                    onClick={() => edit({ reranker: draft.reranker === false ? null : false })}
                    className="switch"
                  >
                    <i />
                  </button>
                  <span className="wd-toggle-word">
                    {draft.reranker === false ? "Off - dense ranking only" : "On"}
                  </span>
                </div>
              ) : (
                <span className="mm-state">
                  <span className="pill-label">Reranker: </span>
                  Not configured
                </span>
              )}
            </div>
            <p className="wd-hint">{loaded.reranker_status.detail}</p>
            {loaded.reranker_status.configured && (
              <p className="wd-hint">
                Turning it off keeps retrieval on dense similarity alone, which is faster and
                frees the sidecar&apos;s memory at some cost in ranking quality.
              </p>
            )}
          </section>

          {formError && <p className="mm-note error">{formError}</p>}

          <div className="mm-actions">
            <button type="submit" className="btn btn-primary" disabled={saving || !dirty}>
              {saving ? "Saving..." : "Save defaults"}
            </button>
            <button
              type="button"
              className="btn btn-ghost"
              onClick={revert}
              disabled={saving || !dirty}
            >
              Discard changes
            </button>
            {saved && !dirty && (
              <span className="mm-note ok" role="status">
                Saved. New turns use these defaults.
              </span>
            )}
          </div>
        </form>
      )}
    </div>
  );
}
