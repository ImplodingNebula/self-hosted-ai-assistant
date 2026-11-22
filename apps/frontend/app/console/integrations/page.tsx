"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ApiError } from "@/lib/api-error";
import {
  type AuthType,
  type ConnectionTestResult,
  type CredentialScope,
  type CredentialStatus,
  type IntegrationPreset,
  type IntegrationServer,
  createServer,
  deleteCredential,
  deleteServer,
  disconnectOAuth,
  getCredential,
  getOAuthRedirectUri,
  listPresets,
  listServers,
  setCredential,
  startOAuthConnect,
  testConnection,
  testServerConnection,
  updateServer,
} from "@/lib/integrations";

/** A test-connection outcome ready to render (color keys on `ok`). */
interface TestNote {
  ok: boolean;
  /** Short status line, rendered inline beside the actions. */
  text: string;
  /** Advertised tool names on success, rendered as a chip grid of their own -
      a long list inlined into the status sentence buries it. */
  tools?: string[];
  /** The called-out special case (an OAuth server answering anonymously),
      never folded into the status or the list where it would get lost. */
  caveat?: string;
}

/** Human line for a failed call; ApiError carries the problem+json detail. */
function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Is the backend running?";
}

/** Why no credential can be stored, and what an operator does about it.
 *
 * Shown once at the top rather than only where a write fails: the cause is a
 * missing deployment variable, so every credential field on the page is
 * equally dead, and finding that out one failed write at a time strands a
 * registered server without its key.
 */
function CredentialStorageNotice() {
  return (
    <div className="int-info warn">
      <p>
        <strong>Credential storage is off in this deployment.</strong> No API
        key or OAuth grant can be stored, because the backend has no{" "}
        <code>SECRET_MASTER_KEY</code> set. Servers can still be registered, and
        one that needs no credential still works.
      </p>
      <p>
        To turn it on: set <code>SECRET_MASTER_KEY</code> to a base64 32-byte
        key in the deployment environment and restart the backend. Once
        credentials are stored, that key must not change - it is what decrypts
        them.
      </p>
    </div>
  );
}

/** The collapsed row's short credential state.
 *
 * Every state gets its own WORD, because the word is the primary channel: a row
 * backed by an environment variable is not in the same state as one holding a
 * sealed credential, and rendering both as "Connected" recreates one level up
 * the exact problem this column exists to fix - a list whose rows look alike
 * while differing materially. The missing case borrows the row's own vocabulary
 * too: an OAuth server is not connected, a header server has no credential.
 */
function credentialSummary(server: IntegrationServer): string {
  const state = server.credential_state;
  if (state === "connected") return "Connected";
  if (state === "deployment_key") return "Deployment key";
  // "No auth needed", not "None needed": the pill's noun must survive on its
  // own. The "Credential: " prefix that names it is visually hidden, so to a
  // sighted first-time reader "None needed" answered a question the row never
  // visibly asked.
  if (state === "not_required") return "No auth needed";
  return server.auth_type === "oauth" ? "Not connected" : "No credential";
}

/** Tone class for {@link credentialSummary} - never the only channel; the word carries it. */
function credentialTone(server: IntegrationServer): string {
  const state = server.credential_state;
  if (state === "connected" || state === "deployment_key") return "ok";
  // "not_required" is a healthy state, so it stays muted. Painting it warn
  // alongside genuine gaps is how a status column becomes noise.
  return state === "missing" ? "warn" : "";
}

/** What the credential input is for, which differs once one is stored.
 *
 * The connected wording is imperative to match the button beside it: an empty
 * box below "Credential connected" otherwise reads as a save that failed.
 */
function credentialPlaceholder(scope: CredentialScope, connected: boolean): string {
  if (!connected) {
    return scope === "user" ? "Your API key or token" : "Shared API key or token";
  }
  return scope === "user" ? "Replace your key" : "Replace the shared key";
}

/** Short badge text for a server's credential scope. */
function scopeLabel(scope: CredentialScope): string {
  return scope === "user" ? "Per-user credential" : "Shared credential";
}

/** Down-chevron glyph for a card's expand affordance. */
function Chevron() {
  return (
    <svg
      suppressHydrationWarning
      viewBox="0 0 16 16"
      className="h-4 w-4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.75"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <path suppressHydrationWarning d="M4 6l4 4 4-4" />
    </svg>
  );
}

/**
 * Compact summary of a test-connection outcome, ready to render.
 *
 * Success names the advertised tools (the handshake never runs one); failure
 * surfaces the classified code plus the backend's detail line. "Reachable",
 * not "Connected": the probe proves the server answers the protocol, and only
 * incidentally exercises a credential when the server demands one to answer.
 */
function testSummary(result: ConnectionTestResult): TestNote {
  if (result.ok) {
    const names = result.tools.map((t) => t.name);
    if (names.length === 0)
      return { ok: true, text: "Reachable. The server advertises no tools." };
    return {
      ok: true,
      text: `Reachable - ${names.length} tool${names.length === 1 ? "" : "s"}.`,
      tools: names,
    };
  }
  const code = result.error?.code ?? "unknown";
  const detail = result.error?.detail ?? "";
  return { ok: false, text: `Failed (${code}). ${detail}`.trim() };
}

/**
 * The deployment's literal OAuth redirect URI, with a copy button - providers
 * demand this exact string at app-registration time, so it is shown as a
 * copyable value, never as a template to assemble by hand. Falls back to the
 * assembly rule when the backend could not be asked.
 */
function RedirectUriLine({ uri }: { uri: string | null }) {
  const [copied, setCopied] = useState(false);
  if (!uri) {
    return (
      <span className="int-muted">
        Redirect URI: the deployment&apos;s OAUTH_REDIRECT_BASE_URL +
        /api/v1/integrations/oauth/callback.
      </span>
    );
  }
  return (
    <span className="int-uri">
      Redirect URI:
      <code className="int-code">{uri}</code>
      <button
        type="button"
        onClick={() => {
          // Optional-chained: clipboard is absent in insecure contexts (plain
          // HTTP over a LAN IP), where the value stays hand-selectable.
          void navigator.clipboard
            ?.writeText(uri)
            .then(() => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1500);
            })
            .catch(() => {});
        }}
        className="int-copy"
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </span>
  );
}

/**
 * The connect-form prefill for a server created from a preset.
 *
 * Looked up fresh from the catalog via the `preset_id` the creation stamped
 * into the row's config, so it survives remounts and reloads - unlike the
 * in-memory map it replaces, which emptied a preset card's values on a mere
 * tab switch. An unknown or absent id simply means no prefill.
 */
function presetPrefill(
  server: IntegrationServer,
  presets: IntegrationPreset[],
): IntegrationPreset["oauth"] | null {
  const presetId = server.config.preset_id;
  if (typeof presetId !== "string") return null;
  return presets.find((p) => p.id === presetId)?.oauth ?? null;
}

/** Fields of the add-server form, kept as one state object. */
interface DraftServer {
  name: string;
  base_url: string;
  credential_scope: CredentialScope;
  auth_type: AuthType;
}

const EMPTY_DRAFT: DraftServer = {
  name: "",
  base_url: "",
  credential_scope: "workspace",
  auth_type: "header",
};

/**
 * The Integrations page: register MCP tool servers and connect credentials.
 *
 * One generic form covers every server - MCP servers self-describe over the
 * protocol (initialize + tools/list), so there is no per-integration UI code.
 * Credentials are write-only: the page shows connected-or-not plus a `last4`
 * hint, never a stored value.
 *
 * A console section: the topbar, the back-to-chat action and the scroll
 * container come from the console layout, so this renders only its content
 * column - the same composition the Model Manager uses.
 */
export default function IntegrationsPage() {
  const [servers, setServers] = useState<IntegrationServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [pageError, setPageError] = useState<string | null>(null);
  // Whether this deployment can seal a credential at all. Starts true so a
  // healthy deployment never flashes the notice while the list loads; the
  // first response settles it.
  const [credentialStorage, setCredentialStorage] = useState(true);

  // Add-server form state.
  const [formOpen, setFormOpen] = useState(false);
  const [draft, setDraft] = useState<DraftServer>(EMPTY_DRAFT);
  const [draftCredential, setDraftCredential] = useState("");
  const [formBusy, setFormBusy] = useState(false);
  const [formError, setFormError] = useState<string | null>(null);
  const [formTest, setFormTest] = useState<TestNote | null>(null);

  // Presets: curated prefills for well-known servers. Loading them can fail
  // without hurting the page - the buttons simply do not render.
  const [presets, setPresets] = useState<IntegrationPreset[]>([]);
  const [activePreset, setActivePreset] = useState<IntegrationPreset | null>(null);
  // The last created server's id, so its card opens ready to connect.
  const [lastCreatedId, setLastCreatedId] = useState<string | null>(null);
  // The deployment's literal OAuth redirect URI; null until loaded (or when
  // the fetch failed, where the UI shows the assembly rule instead).
  const [redirectUri, setRedirectUri] = useState<string | null>(null);

  // Promise callbacks (not async/await) keep every setState out of the
  // synchronous call graph, which the set-state-in-effect lint rule rejects
  // for the mount-time call below (same idiom as the chat page's list load).
  const refresh = useCallback(
    () =>
      listServers()
        .then((listing) => {
          setServers(listing.servers);
          setCredentialStorage(listing.credentialStorageAvailable);
          setPageError(null);
        })
        .catch((err: unknown) => setPageError(errorMessage(err)))
        .finally(() => setLoading(false)),
    [],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  useEffect(() => {
    listPresets()
      .then(setPresets)
      .catch(() => setPresets([]));
    getOAuthRedirectUri()
      .then(setRedirectUri)
      .catch(() => setRedirectUri(null));
  }, []);

  /** Open the add form prefilled from a preset, with a free name suggested. */
  function startFromPreset(preset: IntegrationPreset) {
    const taken = new Set(servers.map((s) => s.name));
    let name = preset.server.name_suggestion;
    // First free name for a second instance of the same integration; the
    // (tenant, name) unique constraint stays the backstop.
    for (let n = 2; taken.has(name); n += 1) {
      name = `${preset.server.name_suggestion}-${n}`;
    }
    setDraft({
      name,
      base_url: preset.server.base_url,
      // Clamped like the form's own auth picker: only header auth can hold a
      // per-user credential. A preset declaring otherwise would render the
      // per-user radio checked yet disabled - selected-but-unpickable - and
      // its submit would bounce off the backend guard anyway.
      credential_scope:
        preset.server.auth_type === "header" ? preset.server.credential_scope : "workspace",
      auth_type: preset.server.auth_type,
    });
    setActivePreset(preset);
    setFormError(null);
    setFormTest(null);
    setFormOpen(true);
  }

  // ONE predicate for "the draft is offering to store a credential", used by
  // the field, the save and the pre-save probe alike. They used to test three
  // different things, so a value could stay in state after the field stopped
  // being rendered and still be sent.
  const credentialFieldShown =
    draft.auth_type === "header" &&
    draft.credential_scope === "workspace" &&
    credentialStorage;

  /** Register the drafted server, then connect its credential if one was typed. */
  async function submitDraft() {
    setFormBusy(true);
    setFormError(null);
    let created;
    try {
      created = await createServer({
        name: draft.name.trim(),
        base_url: draft.base_url.trim(),
        credential_scope: draft.credential_scope,
        auth_type: draft.auth_type,
        // The preset's id rides in the row's config, so the card can rebuild
        // its connect-form prefill from the catalog after ANY remount or
        // reload - in-memory prefill state died with a tab switch, which
        // stripped a preset card of its preset values.
        config: activePreset ? { preset_id: activePreset.id } : undefined,
      });
    } catch (err) {
      setFormError(errorMessage(err));
      setFormBusy(false);
      return;
    }
    // The server EXISTS from here on: close the form and refresh the list even
    // if the credential write below fails, or the created row would be
    // invisible and a retry would 409 on its own name.
    setLastCreatedId(created.id);
    setActivePreset(null);
    setDraft(EMPTY_DRAFT);
    setFormTest(null);
    setFormOpen(false);
    await refresh();
    // Two-call create-then-credential: the credential endpoint is the one
    // write path for secrets, so the form composes it rather than inlining a
    // secret into the registration payload. Runs AFTER the refresh (which
    // clears pageError on success) so a credential failure's note survives.
    // Gated on the FIELD being shown, not on the value being present: a key
    // typed before the scope radio moved (or before storage went away) is one
    // the form is no longer offering to store, and sealing it anyway would be
    // a write nobody could see they were asking for.
    if (credentialFieldShown && draftCredential) {
      try {
        await setCredential(created.id, draftCredential);
        // The refresh above ran BEFORE this seal (deliberately, so the row is
        // visible even when sealing fails) - which means the listing in state
        // predates the credential. Re-sync, or the new card reads "No
        // credential" until something else refetches, while calls and tests
        // happily use the stored key. refresh() never rejects, so a refetch
        // hiccup cannot land in the catch below and masquerade as a failed
        // credential write.
        await refresh();
      } catch (err) {
        // "Connect it from the server's card" is only useful advice when the
        // card would work on a retry. A 503 means the deployment cannot store
        // credentials at all, and its message already carries the remedy -
        // pointing at the card there sends an admin to a button that fails the
        // same way.
        const storageDown = err instanceof ApiError && err.status === 503;
        if (storageDown) {
          // A 503 is not proof on its own: the same-origin proxy answers 503
          // when the BACKEND is unreachable, which would blame a missing key
          // for a stopped container. Re-read the authoritative flag instead of
          // inferring it, and touch nothing else - a full refresh would clear
          // the message set just below.
          void listServers()
            .then((listing) => setCredentialStorage(listing.credentialStorageAvailable))
            .catch(() => undefined);
        }
        setPageError(
          storageDown
            ? `The server was added, but its credential was not stored. ${errorMessage(err)}`
            : `The server was added, but storing its credential failed: ${errorMessage(err)} Connect it from the server's card.`,
        );
      }
    }
    setDraftCredential("");
    setFormBusy(false);
  }

  /** Pre-save handshake for the drafted server (one-shot credential, never stored). */
  async function testDraft() {
    setFormBusy(true);
    setFormError(null);
    setFormTest(null);
    try {
      const result = await testConnection({
        base_url: draft.base_url.trim(),
        credential_scope: draft.credential_scope,
        // Same predicate as the field and the save: probing with a key the
        // form is not showing would report "auth accepted" on a credential
        // the user can neither see nor edit.
        credential: credentialFieldShown ? draftCredential || undefined : undefined,
      });
      setFormTest(testSummary(result));
    } catch (err) {
      setFormError(errorMessage(err));
    } finally {
      setFormBusy(false);
    }
  }

  return (
    <div data-part="integrations">
      {/* The console topbar names the console, not the page, so the page
          carries its own title - same idiom as the Model Manager's h1. */}
      <h1 className="int-title">Integrations</h1>
      {/* Two titled sections, split along the coming role boundary:
          per-user card actions above, workspace-level registration below.
          When role enforcement lands, the second section becomes an
          admin-only conditional render rather than a layout change. */}
      <section className="int-section">
        <h2 className="int-section-title">Servers</h2>
        <p className="panel-sub">
          Tool servers the assistant can reach. Each card manages its
          connection, credentials and whether its tools are offered.
        </p>

        {!credentialStorage && <CredentialStorageNotice />}

        {pageError && <p className="int-error">{pageError}</p>}

        {loading ? (
          <p className="int-muted">Loading...</p>
        ) : servers.length === 0 && !pageError ? (
          <p className="int-muted">No servers registered yet.</p>
        ) : (
          <ul className="server-list">
            {servers.map((server) => (
              <ServerCard
                key={server.id}
                server={server}
                onChanged={refresh}
                oauthPrefill={presetPrefill(server, presets)}
                defaultOpen={server.id === lastCreatedId}
                redirectUri={redirectUri}
                credentialStorage={credentialStorage}
              />
            ))}
          </ul>
        )}
      </section>

      <section className="int-section">
        <h2 className="int-section-title">Add servers</h2>
        <p className="panel-sub">
          Workspace setup: a server describes its own tools, so connecting
          one needs just a name and a URL. A registered server becomes
          available to the whole workspace.
        </p>

        {formOpen ? (
        <form
          className="int-form"
          onSubmit={(e) => {
            e.preventDefault();
            void submitDraft();
          }}
        >
          <h2 className="int-form-title">
            {activePreset ? `Add ${activePreset.name}` : "Add a server"}
          </h2>
          {activePreset && (
            <div className="int-info">
              <p>
                Prefilled from the {activePreset.name} preset. Register your
                own app with the provider first:{" "}
                {activePreset.registration_url.startsWith("https://") ? (
                  <a
                    href={activePreset.registration_url}
                    target="_blank"
                    rel="noreferrer"
                  >
                    {activePreset.registration_url}
                  </a>
                ) : (
                  activePreset.registration_url
                )}
              </p>
              {activePreset.server.auth_type === "oauth" && (
                <p>
                  <RedirectUriLine uri={redirectUri} />
                </p>
              )}
              <ul>
                {activePreset.setup_notes.map((note) => (
                  <li key={note}>{note}</li>
                ))}
              </ul>
            </div>
          )}
          <label className="int-field">
            Name
            <input
              value={draft.name}
              onChange={(e) => setDraft({ ...draft, name: e.target.value })}
              required
              maxLength={200}
              placeholder="crm-connector"
              className="int-input"
            />
          </label>
          <label className="int-field">
            Base URL
            <input
              value={draft.base_url}
              onChange={(e) => setDraft({ ...draft, base_url: e.target.value })}
              required
              type="url"
              placeholder="http://host:8811/mcp"
              className="int-input"
            />
          </label>
          <fieldset className="int-fieldset">
            <legend className="int-legend">Credential scope</legend>
            <label className="int-choice">
              <input
                type="radio"
                name="scope"
                checked={draft.credential_scope === "workspace"}
                onChange={() =>
                  setDraft({ ...draft, credential_scope: "workspace" })
                }
              />
              Shared: one workspace credential for every member
            </label>
            <label className="int-choice">
              <input
                type="radio"
                name="scope"
                disabled={draft.auth_type !== "header"}
                checked={draft.credential_scope === "user"}
                onChange={() => setDraft({ ...draft, credential_scope: "user" })}
              />
              Per-user: each member connects their own; calls fail closed
              without one
            </label>
            {/* A visible reason, not a title tooltip: hover-only text never
                reaches touch or keyboard, and the greyed radio alone says
                "no" without saying why. */}
            {draft.auth_type !== "header" && (
              <p className="int-hint reason">
                {draft.auth_type === "oauth"
                  ? "Shared only for now: per-user OAuth arrives with real sign-in."
                  : "Shared only: this server takes no credential, so there is nothing to scope per member."}
              </p>
            )}
          </fieldset>
          <fieldset className="int-fieldset">
            <legend className="int-legend">Authentication</legend>
            {(
              [
                ["header", "API key: paste a shared or per-user token"],
                ["oauth", "OAuth: connect through the provider's consent screen"],
                ["none", "None: the server takes no credential"],
              ] as const
            ).map(([value, label]) => (
              <label key={value} className="int-choice">
                <input
                  type="radio"
                  name="auth"
                  checked={draft.auth_type === value}
                  onChange={() =>
                    // Only header auth can hold a per-user credential, so
                    // leaving it snaps the scope to shared: OAuth grants
                    // are workspace-scoped until real login (Phase 8b),
                    // and an auth-none server has nothing to scope. The
                    // backend rejects both other combinations anyway.
                    setDraft(
                      value === "header"
                        ? { ...draft, auth_type: value }
                        : { ...draft, auth_type: value, credential_scope: "workspace" },
                    )
                  }
                />
                {label}
              </label>
            ))}
          </fieldset>
          {draft.auth_type === "oauth" && (
            <p className="int-hint pad">
              Add the server first, then set up the connection from its card
              (client id, scopes, and the Connect button).
            </p>
          )}
          {credentialFieldShown ? (
            <label className="int-field">
              Credential (optional; stored write-only)
              <input
                value={draftCredential}
                onChange={(e) => setDraftCredential(e.target.value)}
                type="password"
                autoComplete="off"
                placeholder="API key or token"
                className="int-input"
              />
            </label>
          ) : draft.auth_type === "header" &&
            draft.credential_scope === "workspace" ? (
            // Storage is off: no field at all. An enabled input here is a
            // promise the deployment cannot keep, and submitting it would
            // register the server and then fail the write, leaving a row
            // that authenticates against nothing.
            <p className="int-hint pad">
              No credential field: this deployment cannot store one yet (see
              the note above the server list). The server can be added now
              and its credential connected from its card afterwards.
            </p>
          ) : draft.auth_type === "header" ? (
            <p className="int-hint pad">
              Per-user servers take no credential here: each member connects
              their own from the server&apos;s card after it is added.
            </p>
          ) : null}
          {formError && <p className="int-note error">{formError}</p>}
          <div className="int-actions">
            <button type="submit" disabled={formBusy} className="btn btn-primary">
              {formBusy ? "Working..." : "Add server"}
            </button>
            <button
              type="button"
              onClick={() => void testDraft()}
              // An OAuth draft has no grant to test with until the server
              // exists and its consent runs from the card, so a pre-save
              // probe would report a misleading "auth rejected" against a
              // healthy server. The hint above the buttons explains the flow.
              disabled={
                formBusy || !draft.base_url.trim() || draft.auth_type === "oauth"
              }
              title={
                draft.auth_type === "oauth"
                  ? "Add the server first; test it after connecting from its card."
                  : undefined
              }
              className="btn btn-ghost"
            >
              Test connection
            </button>
            <button
              type="button"
              onClick={() => {
                setFormOpen(false);
                setFormError(null);
                setFormTest(null);
                // Drop the typed secret with the form; an abandoned token must
                // not sit in state waiting for the form to reopen.
                setDraftCredential("");
                setDraft(EMPTY_DRAFT);
                setActivePreset(null);
              }}
              className="btn btn-ghost"
            >
              Cancel
            </button>
            {formTest && (
              <span
                className={`int-note int-actions-note ${formTest.ok ? "ok" : "error"}`}
              >
                {formTest.text}
              </span>
            )}
          </div>
          {formTest?.tools && formTest.tools.length > 0 && (
            <ul className="int-tools">
              {formTest.tools.map((name) => (
                <li key={name} className="int-tool-chip" title={name}>
                  {name}
                </li>
              ))}
            </ul>
          )}
        </form>
      ) : (
        <div data-part="integrations-presets">
          {presets.map((preset) => (
            <button
              key={preset.id}
              onClick={() => startFromPreset(preset)}
              title={preset.description}
              aria-label={`Add ${preset.name}`}
              className="preset-btn"
            >
              <span className="plus" aria-hidden="true">
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
              </span>
              {preset.name}
            </button>
          ))}
          <button onClick={() => setFormOpen(true)} className="btn btn-primary">
            Add a server
          </button>
        </div>
      )}
      </section>
    </div>
  );
}

/** Props for one registered server's {@link ServerCard}. */
interface ServerCardProps {
  /** The server row as listed. */
  server: IntegrationServer;
  /** Reload the list after any mutation (the server is re-fetched, not patched locally). */
  onChanged: () => Promise<void>;
  /** Connect-form prefill from the preset that created this server, if any. */
  oauthPrefill?: IntegrationPreset["oauth"] | null;
  /** Open expanded (the just-created server, so the flow continues in place). */
  defaultOpen?: boolean;
  /** The deployment's literal OAuth redirect URI; null when not yet known. */
  redirectUri?: string | null;
  /** False when the deployment cannot seal credentials: the write paths are dead. */
  credentialStorage: boolean;
}

/**
 * One server's card: summary row plus an expandable manage panel
 * (enable/disable, credential connect/disconnect, test connection, delete).
 */
function ServerCard({
  server,
  onChanged,
  oauthPrefill,
  defaultOpen,
  redirectUri,
  credentialStorage,
}: ServerCardProps) {
  const [open, setOpen] = useState(defaultOpen ?? false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [headError, setHeadError] = useState<string | null>(null);
  const [credential, setCredentialStatus] = useState<CredentialStatus | null>(
    null,
  );
  const [credentialValue, setCredentialValue] = useState("");
  const [testResult, setTestResult] = useState<TestNote | null>(null);
  const [confirmDelete, setConfirmDelete] = useState(false);
  // Bumped on every credential WRITE so a slow expand-time GET that resolves
  // after a Connect/Disconnect cannot clobber the fresher post-write status.
  const credentialWriteSeq = useRef(0);

  // Is a credential stored, as best this card knows RIGHT NOW. The per-server
  // fetch below is authoritative once it lands (it alone carries `last4`);
  // until then the listing's answer stands, so an expanded card never disagrees
  // with the row it sits under. `??` and not `||`: once the fetch returns, a
  // deliberate false must win over the stale listing - that is the state after
  // Disconnect, before the list refresh catches up.
  const connectedNow = credential?.connected ?? server.credential_state === "connected";

  // The credential status is fetched on expand, not with the list - it is a
  // separate endpoint per server and most rows stay collapsed. What the list
  // cannot give is the `last4` hint, which is the only reason this still runs.
  // credential_state is a dependency because an OPEN card can outlive its
  // fetch's truth: the just-created card auto-expands and fetches before the
  // form's credential seal lands, and without the re-run its body kept saying
  // "No shared credential connected" while the summary pill (fed by the
  // refreshed listing) said Connected.
  useEffect(() => {
    if (!open || server.auth_type !== "header") return;
    let cancelled = false;
    const seq = credentialWriteSeq.current;
    getCredential(server.id)
      .then((status) => {
        if (!cancelled && credentialWriteSeq.current === seq)
          setCredentialStatus(status);
      })
      .catch((err) => {
        if (!cancelled) setError(errorMessage(err));
      });
    return () => {
      cancelled = true;
    };
  }, [open, server.id, server.auth_type, server.credential_state]);

  /** Run a mutation with shared busy/error handling, then refresh the list.

   * `into` picks which notice a failure lands in: the body notice by default,
   * or the header notice for the one control (the enable switch) that is
   * operable while the card is collapsed - its error must show collapsed too.
   */
  async function act(
    fn: () => Promise<void>,
    into: (message: string | null) => void = setError,
  ) {
    setBusy(true);
    setError(null);
    setHeadError(null);
    try {
      await fn();
      await onChanged();
    } catch (err) {
      into(errorMessage(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <li className="server-card">
      <div className="server-head">
        <button
          type="button"
          onClick={() => setOpen(!open)}
          aria-expanded={open}
          className="server-summary"
        >
          <span className="server-main">
            <span className="server-name">
              {server.name}
              {/* Scope answers WHOSE credential, which is a question a server
                  taking no credential does not have. Showing "shared
                  credential" beside a "None needed" pill puts two
                  contradictory claims on one line. */}
              {server.auth_type !== "none" && (
                <span className="badge">{scopeLabel(server.credential_scope)}</span>
              )}
              {/* A row the deployment itself configures. It reads as a badge
                  rather than as a footnote in the body because it changes what
                  the card's controls DO - delete is refused on it - and a
                  collapsed row would otherwise look identical to one an admin
                  added by hand. */}
              {server.seeded && <span className="badge">Bundled</span>}
            </span>
            <span className="server-url">{server.base_url}</span>
          </span>
          {/* Credential state sits in the RIGHT cluster, not beside the scope
              badge where it belongs by subject: a chip after a variable-length
              name does not line up down the list, and scanning the column for
              "which of these can actually authenticate" is the whole point. The
              URL truncates to pay for the width, which is the cheaper loss. */}
          <span className={`cred-status ${credentialTone(server)}`}>
            <span className="pill-label">Credential: </span>
            {credentialSummary(server)}
          </span>
          <span className={`server-status${server.enabled ? " on" : ""}`}>
            <span className="pill-label">Tools: </span>
            {server.enabled ? "Enabled" : "Disabled"}
          </span>
          <span className="chevron">
            <Chevron />
          </span>
        </button>
        {/* Sibling of the expand trigger, never nested inside it: the switch
            toggles enable directly (matching the prototype's inline switch),
            and nesting two buttons is invalid HTML. */}
        <button
          type="button"
          role="switch"
          aria-checked={server.enabled}
          aria-label={
            server.enabled ? `Disable ${server.name}` : `Enable ${server.name}`
          }
          disabled={busy}
          onClick={() =>
            void act(async () => {
              await updateServer(server.id, { enabled: !server.enabled });
            }, setHeadError)
          }
          className="switch"
        >
          <i />
        </button>
      </div>

      {/* Outside the expand gate: the switch is operable on a collapsed card,
          so its failure must be visible on a collapsed card too. */}
      {headError && <p className="int-note error">{headError}</p>}

      {open && (
        <div className="server-body">
          {error && <p className="int-note error">{error}</p>}

          {server.auth_type === "oauth" && (
            <OAuthPanel
              // The panel seeds its fields from the prefill ONCE, at mount; if
              // a card is expanded in the instant before the preset catalog
              // resolves, remounting on the null-to-present transition re-seeds
              // it instead of silently losing the preset values.
              key={oauthPrefill ? "prefilled" : "bare"}
              server={server}
              busy={busy}
              act={act}
              prefill={oauthPrefill ?? null}
              redirectUri={redirectUri ?? null}
              credentialStorage={credentialStorage}
            />
          )}
          {server.auth_type === "none" && (
            // Dotted and muted, matching its "None needed" pill: standing
            // credential state looks the same on every auth type, so the answer
            // to "is this connected" reads identically wherever it is found.
            <p className="int-note dotted">
              This server takes no credential.
              <span className="note-detail">
                {" "}
                Calls are sent unauthenticated by design, even if a credential
                is stored.
              </span>
            </p>
          )}
          {server.auth_type === "header" && (
            <>
              {/* Connected and not-connected used to render in the same muted
                  tone, so the two opposite outcomes looked alike. Tone plus a
                  dot plus the word, matching the enabled pill's language.
                  No "checking" state: the LIST already answered, so the card
                  opens on that answer and the fetch only adds the last4 hint.
                  Waiting for it made an open card contradict its own header
                  and flash a Connect button at an already-connected server -
                  the same "did my save fail" read this whole block exists to
                  kill. */}
              {connectedNow ? (
                <div className="int-note ok dotted">
                  Credential connected
                  {/* The mask convention, not an ellipsis: "...2026" reads as
                      a year or a truncated word; "****2026" reads as the tail
                      of a hidden key. */}
                  {credential?.last4 ? ` (key ****${credential.last4})` : ""}.
                  <span className="note-detail">
                    {" "}
                    Entering a new one below replaces it.
                  </span>
                </div>
              ) : server.credential_state === "deployment_key" ? (
                <div className="int-note ok dotted">
                  Connected by the deployment&apos;s bootstrap key.
                  <span className="note-detail">
                    {" "}
                    Nothing is stored here, so calls stop the moment that
                    environment variable changes. Storing a credential below
                    takes over from it.
                  </span>
                </div>
              ) : (
                <div className="int-note warn dotted">
                  {/* Only USER scope fails closed (mcp_sources.py:505). A
                      workspace row with no credential sends no Authorization
                      header at all, and whether that works is the upstream's
                      call - saying "fails closed" there would be wrong about
                      every server that accepts anonymous requests. */}
                  {server.credential_scope === "user" ? (
                    <>
                      No credential connected for this account.
                      <span className="note-detail">
                        {" "}
                        Calls to this server fail closed until one is.
                      </span>
                    </>
                  ) : (
                    <>
                      No shared credential connected.
                      <span className="note-detail">
                        {" "}
                        Calls are sent without one, so this server only works if
                        it accepts unauthenticated requests.
                      </span>
                    </>
                  )}
                </div>
              )}

              <div className="int-cred-row">
                <input
                  value={credentialValue}
                  onChange={(e) => setCredentialValue(e.target.value)}
                  type="password"
                  autoComplete="off"
                  // State-aware, because an empty box under "Credential
                  // connected" otherwise reads as a failed save. Only the
                  // button said "Replace"; now the field it belongs to says so
                  // too.
                  placeholder={credentialPlaceholder(server.credential_scope, connectedNow)}
                  aria-label={credentialPlaceholder(server.credential_scope, connectedNow)}
                  className="int-input"
                />
                <button
                  // The field stays usable while storage is off: a typed key
                  // still drives Test connection, which sends it without
                  // storing it. Only the write is dead, so only the write is
                  // disabled. The reason renders as a visible line below - a
                  // title on a disabled button never opens in Chrome or Safari.
                  disabled={busy || !credentialValue || !credentialStorage}
                  onClick={() =>
                    void act(async () => {
                      credentialWriteSeq.current += 1;
                      setCredentialStatus(
                        await setCredential(server.id, credentialValue),
                      );
                      setCredentialValue("");
                    })
                  }
                  className="btn btn-primary"
                >
                  {connectedNow ? "Replace" : "Connect"}
                </button>
                {connectedNow && (
                  <button
                    disabled={busy}
                    onClick={() =>
                      void act(async () => {
                        credentialWriteSeq.current += 1;
                        await deleteCredential(server.id);
                        setCredentialStatus(await getCredential(server.id));
                      })
                    }
                    className="btn btn-ghost"
                  >
                    Disconnect
                  </button>
                )}
              </div>
              {!credentialStorage && (
                // Muted, not warn: the page notice above the list is the alarm
                // for this one condition, and repeating it at full volume on
                // every card turns one problem into five.
                <p className="int-hint">
                  Connect is unavailable: this deployment cannot store a
                  credential yet (see the note above the server list). A key
                  typed here still works for Test connection, which sends it
                  without storing it.
                </p>
              )}
            </>
          )}

          <div className="int-actions">
            <button
              disabled={busy}
              onClick={() => {
                setBusy(true);
                setTestResult(null);
                // The saved-server probe uses the server's STORED credential, so
                // a connected server tests truthfully; a typed key (not yet
                // saved) rides along as a one-shot override to test before
                // replacing. Some providers answer the handshake anonymously
                // (the tool list is public), so an unconnected OAuth card
                // reaching them says so instead of implying the grant works.
                testServerConnection(server.id, credentialValue || undefined)
                  .then((result) =>
                    setTestResult({
                      ...testSummary(result),
                      caveat:
                        result.ok &&
                        server.auth_type === "oauth" &&
                        !server.oauth_connected
                          ? "The server answered without a sign-in; using its tools may still need the OAuth connection."
                          : undefined,
                    }),
                  )
                  .catch((err) =>
                    // A transport failure is still a test outcome; it renders
                    // beside this button, not in a distant notice.
                    setTestResult({ ok: false, text: errorMessage(err) }),
                  )
                  .finally(() => setBusy(false));
              }}
              className="btn btn-ghost"
            >
              Test connection
            </button>
            {/* A server the deployment configures has no delete control at
                all, rather than one that 409s. The backend refuses it because
                the next restart recreates the row, so offering the button
                would promise something no click can deliver - and the control
                that DOES last is the toggle already in the header. The
                explanation renders after the test outcome, as its own line. */}
            {server.seeded ? null : confirmDelete ? (
              <>
                <button
                  disabled={busy}
                  onClick={() => void act(() => deleteServer(server.id))}
                  className="btn btn-danger"
                >
                  Confirm delete
                </button>
                <button
                  disabled={busy}
                  onClick={() => setConfirmDelete(false)}
                  className="btn btn-ghost"
                >
                  Keep
                </button>
              </>
            ) : (
              <button
                disabled={busy}
                onClick={() => setConfirmDelete(true)}
                className="btn btn-danger-ghost"
              >
                Delete server
              </button>
            )}
            {testResult && (
              <span
                className={`int-note int-actions-note ${testResult.ok ? "ok" : "error"}`}
              >
                {testResult.text}
              </span>
            )}
            {server.seeded && (
              <span className="int-muted int-bundled-note">
                Bundled with this deployment. Turn the toggle off to withhold
                its tools; delete becomes available if the deployment stops
                configuring it.
              </span>
            )}
          </div>
          {testResult?.caveat && (
            <p className="int-note warn">{testResult.caveat}</p>
          )}
          {testResult?.tools && testResult.tools.length > 0 && (
            <ul className="int-tools">
              {testResult.tools.map((name) => (
                <li key={name} className="int-tool-chip" title={name}>
                  {name}
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </li>
  );
}

/** Props for the per-server OAuth connection panel. */
interface OAuthPanelProps {
  /** The oauth-typed server row. */
  server: IntegrationServer;
  /** Shared busy flag from the card. */
  busy: boolean;
  /** The card's mutation wrapper (busy/error handling + list refresh). */
  act: (fn: () => Promise<void>) => Promise<void>;
  /** Preset prefill for the connect form (scopes, endpoints, extra params). */
  prefill: IntegrationPreset["oauth"] | null;
  /** The deployment's literal OAuth redirect URI; null when not yet known. */
  redirectUri: string | null;
  /** False when the deployment cannot seal secrets: a grant has nowhere to land. */
  credentialStorage: boolean;
}

/** Serialize an extra-params map to the editable "key=value key=value" form. */
function formatExtraParams(params: Record<string, string>): string {
  return Object.entries(params)
    .map(([key, value]) => `${key}=${value}`)
    .join(" ");
}

/** Parse the editable form back; null when an entry is not key=value. */
function parseExtraParams(text: string): Record<string, string> | null {
  const params: Record<string, string> = {};
  for (const entry of text.trim().split(/\s+/).filter(Boolean)) {
    const eq = entry.indexOf("=");
    // Both sides must be non-empty ("a=" would only bounce off the backend's
    // validation with a vaguer message than the local one).
    if (eq < 1 || eq === entry.length - 1) return null;
    params[entry.slice(0, eq)] = entry.slice(eq + 1);
  }
  return params;
}

/**
 * The OAuth half of a server card: grant status, the connect form, disconnect.
 *
 * Connect asks the backend to start a consent and sends THIS tab to the
 * provider's approval page; the provider returns the browser to this page via
 * the callback, so no popup management is needed. Endpoints left blank mean
 * the MCP-native path (the server describes its own authorization server);
 * filling them is the generic-provider path that slice 10's presets prefill.
 */
function OAuthPanel({
  server,
  busy,
  act,
  prefill,
  redirectUri,
  credentialStorage,
}: OAuthPanelProps) {
  const [clientId, setClientId] = useState("");
  const [clientSecret, setClientSecret] = useState("");
  const [scopes, setScopes] = useState(prefill?.scopes.join(" ") ?? "");
  // Extras live inside the advanced block, so a preset carrying them must
  // open it: parameters the admin cannot see must never be sent.
  const [advanced, setAdvanced] = useState(
    prefill?.endpoints != null ||
      Object.keys(prefill?.extra_authorize_params ?? {}).length > 0,
  );
  const [issuer, setIssuer] = useState(prefill?.endpoints?.issuer ?? "");
  const [authorizeUrl, setAuthorizeUrl] = useState(
    prefill?.endpoints?.authorization_endpoint ?? "",
  );
  const [tokenUrl, setTokenUrl] = useState(
    prefill?.endpoints?.token_endpoint ?? "",
  );
  const [revokeUrl, setRevokeUrl] = useState(
    prefill?.endpoints?.revocation_endpoint ?? "",
  );
  const [extraParams, setExtraParams] = useState(
    formatExtraParams(prefill?.extra_authorize_params ?? {}),
  );
  const [paramError, setParamError] = useState<string | null>(null);

  // The account label the callback recorded (non-secret config; absent when
  // the provider had no identity lookup or it failed).
  const account =
    typeof server.config.connected_account === "string"
      ? server.config.connected_account
      : null;

  if (server.oauth_connected) {
    return (
      <div data-part="integrations-oauth" className="int-cred-row">
        <span className="int-note">
          Connected via OAuth
          {account ? ` as ${account}` : ""}
          {server.oauth_connected_at
            ? ` (last updated ${new Date(server.oauth_connected_at).toLocaleString()})`
            : ""}
          .
        </span>
        <button
          disabled={busy}
          onClick={() => void act(() => disconnectOAuth(server.id))}
          className="btn btn-ghost"
        >
          Disconnect
        </button>
      </div>
    );
  }

  return (
    <div data-part="integrations-oauth" className="int-field">
      <p className="int-note warn dotted">
        Not connected.
        <span className="note-detail">
          {" "}
          Register an OAuth app with the provider, then connect.{" "}
          <RedirectUriLine uri={redirectUri} />
        </span>
      </p>
      <div className="int-cred-row">
        <input
          value={clientId}
          onChange={(e) => setClientId(e.target.value)}
          placeholder="Client id"
          className="int-input"
        />
        <input
          value={clientSecret}
          onChange={(e) => setClientSecret(e.target.value)}
          type="password"
          autoComplete="off"
          placeholder="Client secret (optional; stored write-only)"
          className="int-input"
        />
      </div>
      <input
        value={scopes}
        onChange={(e) => setScopes(e.target.value)}
        placeholder="Scopes, space-separated (e.g. repo read:org)"
        className="int-input"
      />
      <label className="int-choice">
        <input
          type="checkbox"
          checked={advanced}
          onChange={(e) => setAdvanced(e.target.checked)}
        />
        Enter the provider&apos;s sign-in endpoints manually (only needed when
        the server cannot describe them itself)
      </label>
      {advanced && (
        <div className="int-field">
          <input
            value={issuer}
            onChange={(e) => setIssuer(e.target.value)}
            placeholder="Issuer (e.g. https://github.com)"
            className="int-input"
          />
          <input
            value={authorizeUrl}
            onChange={(e) => setAuthorizeUrl(e.target.value)}
            placeholder="Authorization endpoint URL"
            className="int-input"
          />
          <input
            value={tokenUrl}
            onChange={(e) => setTokenUrl(e.target.value)}
            placeholder="Token endpoint URL"
            className="int-input"
          />
          <input
            value={revokeUrl}
            onChange={(e) => setRevokeUrl(e.target.value)}
            placeholder="Revocation endpoint URL (optional; lets Disconnect revoke upstream)"
            className="int-input"
          />
          <input
            value={extraParams}
            onChange={(e) => setExtraParams(e.target.value)}
            placeholder="Extra sign-in parameters, key=value space-separated (optional)"
            className="int-input"
          />
          <p className="int-hint">
            Optional and provider-specific - most providers need none. Google
            needs access_type=offline for the renewal token that keeps the
            connection alive; remove an entry to not ask for it.
          </p>
        </div>
      )}
      {paramError && <p className="int-note error">{paramError}</p>}
      {/* Blocked rather than allowed-and-failed: a provider approval screen
          whose result is discarded at the end wastes a real consent. */}
      {!credentialStorage && (
        <p className="int-hint">
          Consent cannot start: the grant it returns has nowhere to be stored
          until this deployment sets a secret key (see the note above the server
          list).
        </p>
      )}
      <div className="int-actions">
        <button
          disabled={
            busy ||
            !credentialStorage ||
            !clientId.trim() ||
            !scopes.trim() ||
            (advanced && (!issuer.trim() || !authorizeUrl.trim() || !tokenUrl.trim()))
          }
          onClick={() => {
            // Everything in the advanced block rides only while it is shown:
            // unchecking it must drop the extras with the endpoints, never
            // send parameters the admin can no longer see.
            const extras = advanced ? parseExtraParams(extraParams) : {};
            if (extras === null) {
              setParamError(
                "Extra parameters must be key=value entries separated by spaces.",
              );
              return;
            }
            setParamError(null);
            void act(async () => {
              const url = await startOAuthConnect(server.id, {
                client_id: clientId.trim(),
                client_secret: clientSecret || undefined,
                scopes: scopes.trim().split(/\s+/),
                endpoints: advanced
                  ? {
                      issuer: issuer.trim(),
                      authorization_endpoint: authorizeUrl.trim(),
                      token_endpoint: tokenUrl.trim(),
                      revocation_endpoint: revokeUrl.trim() || undefined,
                    }
                  : undefined,
                // Curated by the preset (no manual input): where to ask who
                // the connected account is, for the "Connected as" line.
                identity: prefill?.identity ?? undefined,
                extra_authorize_params:
                  advanced && Object.keys(extras).length > 0 ? extras : undefined,
              });
              // Same-tab: the provider redirects back to this page when done.
              window.location.assign(url);
            });
          }}
          className="btn btn-primary"
        >
          Connect
        </button>
      </div>
    </div>
  );
}
