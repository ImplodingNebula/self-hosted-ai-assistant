/**
 * Browser-side helpers for the members REST endpoints.
 *
 * All calls go through the same-origin `/api/members` proxy routes. The shapes
 * mirror the backend's `app/schemas/members.py` wire contract. Both endpoints
 * are admin-gated, so a member's browser gets a 403 `ApiError` from either.
 */

import { raise } from "@/lib/api-error";

/** The two roles this deployment has. Named permissions arrive later. */
export type MemberRole = "admin" | "member";

/** One person in the workspace. */
export interface Member {
  id: string;
  email: string | null;
  display_name: string;
  role: MemberRole;
  created_at: string;
  /**
   * Whether the deployment's own configuration floors this account to admin.
   *
   * True means the role is re-applied at every sign-in, so the page renders the
   * control as fixed and says where the change actually belongs. The backend
   * refuses the demotion too - this flag is what stops the page offering it.
   */
  is_env_admin: boolean;
}

/** List the workspace's members, oldest first. */
export async function listMembers(): Promise<Member[]> {
  const resp = await fetch("/api/members", { cache: "no-store" });
  if (!resp.ok) await raise(resp);
  const data = (await resp.json()) as { items: Member[] };
  return data.items;
}

/**
 * Set one member's role.
 *
 * A 409 `ApiError` means the change was refused and its message says why - the
 * last administrator, or an account the deployment lists as one. Show that
 * message rather than rewording it.
 */
export async function setMemberRole(
  id: string,
  role: MemberRole,
): Promise<Member> {
  const resp = await fetch(`/api/members/${encodeURIComponent(id)}`, {
    method: "PATCH",
    headers: { "content-type": "application/json" },
    body: JSON.stringify({ role }),
    cache: "no-store",
  });
  if (!resp.ok) await raise(resp);
  return (await resp.json()) as Member;
}
