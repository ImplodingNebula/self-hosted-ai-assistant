/**
 * Browser-side helpers for the Integrations REST endpoints.
 *
 * All calls go through the same-origin `/api/integrations` proxy routes. The
 * shapes mirror the backend's `app/schemas/integrations.py` wire contract; a
 * credential VALUE is write-only and never appears in any response.
 */

import { raise } from "@/lib/api-error";

/** Whose credential a server's calls carry. */
export type CredentialScope = "workspace" | "user";

/** HOW a server authenticates (scope says whose credential a call carries). */
export type AuthType = "none" | "header" | "oauth";

/** One registered MCP server - metadata only, never a credential. */
export interface IntegrationServer {
  id: string;
  name: string;
  base_url: string;
  enabled: boolean;
  credential_scope: CredentialScope;
  auth_type: AuthType;
  config: Record<string, unknown>;
  created_at: string;
  /** OAuth grant status; meaningful only on auth_type "oauth" rows. */
  oauth_connected: boolean;
  oauth_connected_at: string | null;
  /**
   * Whether this server can authenticate, in one answer across all auth types,
   * so a listing can show it without a per-row credential fetch.
   *
   * `deployment_key` means no credential is stored but the env bootstrap key
   * backs this URL - working, and fragile in a way worth naming.
   */
  credential_state: "connected" | "deployment_key" | "missing" | "not_required";
  /**
   * Whether the deployment's environment currently bootstraps a row of this
   * name. Computed by the backend per response, never stored.
   *
   * Deleting such a row is refused with a 409, because the next restart would
   * recreate it; `enabled` is the off switch that lasts. It goes false on its
   * own once the deployment stops configuring that server.
   */
  seeded: boolean;
}

/** Whether a server has a stored credential, plus its metadata hint. */
export interface CredentialStatus {
  connected: boolean;
  scope: CredentialScope | null;
  /** Recognisable tail of the stored value; empty when too short to hint. */
  last4: string | null;
  created_at: string | null;
  updated_at: string | null;
}

/** One tool a server advertises, from a test-connection handshake. */
export interface ToolDescriptor {
  name: string;
  description: string;
  side_effecting: boolean;
}

/** Why a test-connection handshake failed, classified for display. */
export interface ConnectionTestError {
  code:
    | "unreachable"
    | "timeout"
    | "auth_rejected"
    | "protocol_mismatch"
    | "unknown";
  detail: string;
}

/** The handshake outcome: reachable + tools, or a classified error. */
export interface ConnectionTestResult {
  ok: boolean;
  tools: ToolDescriptor[];
  error: ConnectionTestError | null;
}

/** Fields to register a new server. */
export interface ServerCreate {
  name: string;
  base_url: string;
  credential_scope: CredentialScope;
  auth_type?: AuthType;
  config?: Record<string, unknown>;
}

/** Patchable server fields (name is immutable server-side). */
export interface ServerUpdate {
  enabled?: boolean;
  base_url?: string;
  credential_scope?: CredentialScope;
  config?: Record<string, unknown>;
}

/** Operator-supplied endpoints for a generic (non-MCP-native) provider. */
export interface OAuthEndpointsIn {
  issuer: string;
  authorization_endpoint: string;
  token_endpoint: string;
  revocation_endpoint?: string | null;
}

/** One curated preset: prefill values for the two forms, plus honest setup notes. */
export interface IntegrationPreset {
  id: string;
  name: string;
  description: string;
  server: {
    name_suggestion: string;
    base_url: string;
    auth_type: AuthType;
    credential_scope: CredentialScope;
  };
  oauth: {
    scopes: string[];
    /** null means the server is MCP-native: endpoints are discovered, not typed. */
    endpoints: OAuthEndpointsIn | null;
    /** Where to ask who the connected account is, for "Connected as ...". */
    identity: { userinfo_endpoint: string; label_field: string } | null;
    extra_authorize_params: Record<string, string>;
  };
  registration_url: string;
  setup_notes: string[];
}

/**
 * The literal redirect URI this deployment sends to OAuth providers - the
 * exact value the operator must register on the provider's app.
 */
export async function getOAuthRedirectUri(): Promise<string> {
  const resp = await fetch("/api/integrations/oauth/redirect-uri", {
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  const data = (await resp.json()) as { redirect_uri: string };
  return data.redirect_uri;
}

/** List the curated presets (static metadata; any member may read them). */
export async function listPresets(): Promise<IntegrationPreset[]> {
  const resp = await fetch("/api/integrations/presets", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  const data = (await resp.json()) as { items: IntegrationPreset[] };
  return data.items;
}

/** Start a workspace OAuth consent; the browser then opens the returned URL. */
export async function startOAuthConnect(
  id: string,
  payload: {
    client_id: string;
    client_secret?: string;
    scopes: string[];
    endpoints?: OAuthEndpointsIn;
    identity?: { userinfo_endpoint: string; label_field: string };
    extra_authorize_params?: Record<string, string>;
  },
): Promise<string> {
  const response = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/oauth/connect`,
    {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    },
  );
  if (!response.ok) await raise(response);
  const body = (await response.json()) as { authorization_url: string };
  return body.authorization_url;
}

/** Revoke a server's OAuth grant (best effort upstream) and delete it locally. */
export async function disconnectOAuth(id: string): Promise<void> {
  const response = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/oauth/disconnect`,
    { method: "POST", cache: "no-store" },
  );
  if (!response.ok) await raise(response);
}

/** The tenant's servers, plus what this deployment can do with credentials. */
export interface ServerListing {
  servers: IntegrationServer[];
  /**
   * False when the backend has no `SECRET_MASTER_KEY`: no credential can be
   * sealed anywhere, so the UI says so up front instead of letting an admin
   * type one and fail on the write.
   */
  credentialStorageAvailable: boolean;
}

/** List the tenant's registered servers (enabled and disabled). */
export async function listServers(): Promise<ServerListing> {
  const resp = await fetch("/api/integrations/servers", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  const data = (await resp.json()) as {
    items: IntegrationServer[];
    credential_storage_available: boolean;
  };
  return {
    servers: data.items,
    credentialStorageAvailable: data.credential_storage_available,
  };
}

/** Register a new server. A duplicate name is a 409 `ApiError`. */
export async function createServer(
  payload: ServerCreate,
): Promise<IntegrationServer> {
  const resp = await fetch("/api/integrations/servers", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as IntegrationServer;
}

/** Patch a server (enable/disable, re-point, rescope, edit config). */
export async function updateServer(
  id: string,
  payload: ServerUpdate,
): Promise<IntegrationServer> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}`,
    {
      method: "PATCH",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      cache: "no-store",
    },
  );
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as IntegrationServer;
}

/**
 * Delete a server (its shared workspace credential goes with it).
 *
 * A 404 is treated as success: the server is already gone, which is the end
 * state the caller asked for.
 */
export async function deleteServer(id: string): Promise<void> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}`,
    { method: "DELETE" },
  );
  if (!resp.ok && resp.status !== 404) await raise(resp);
}

/** Read a server's credential status - metadata + `last4`, never plaintext. */
export async function getCredential(id: string): Promise<CredentialStatus> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/credential`,
    { cache: "no-store" },
  );
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as CredentialStatus;
}

/**
 * Seal a server's credential (write-only).
 *
 * A 503 `ApiError` means the backend has no `SECRET_MASTER_KEY` configured; its
 * message carries the operator's remedy, so show it rather than rewording it.
 */
export async function setCredential(
  id: string,
  value: string,
): Promise<CredentialStatus> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/credential`,
    {
      method: "PUT",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ value }),
      cache: "no-store",
    },
  );
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as CredentialStatus;
}

/** Clear a server's stored credential. */
export async function deleteCredential(id: string): Promise<void> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/credential`,
    { method: "DELETE" },
  );
  if (!resp.ok && resp.status !== 404) await raise(resp);
}

/**
 * Probe a NOT-YET-SAVED server with an initialize + tools/list handshake (never
 * a tool call). Used by the add form: paste a URL and optional key, test, then
 * save. Always resolves with a result; a down target is `ok: false` plus a
 * classified `error`, not a rejection.
 */
export async function testConnection(payload: {
  base_url: string;
  credential_scope: CredentialScope;
  credential?: string;
  config?: Record<string, unknown>;
}): Promise<ConnectionTestResult> {
  const resp = await fetch("/api/integrations/servers/test-connection", {
    method: "POST",
    headers: { "content-type": "application/json" },
    body: JSON.stringify(payload),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ConnectionTestResult;
}

/**
 * Probe an ALREADY-SAVED server (by id) with the credential the chat path would
 * send - its stored credential, so a connected server tests truthfully without
 * retyping the key. An optional `credential` overrides it to test a new key
 * before replacing the stored one. Always resolves with a result.
 */
export async function testServerConnection(
  id: string,
  credential?: string,
): Promise<ConnectionTestResult> {
  const resp = await fetch(
    `/api/integrations/servers/${encodeURIComponent(id)}/test-connection`,
    {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify({ credential: credential ?? null }),
      cache: "no-store",
    },
  );
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as ConnectionTestResult;
}
