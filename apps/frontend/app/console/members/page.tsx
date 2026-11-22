"use client";

import { useCallback, useEffect, useRef, useState } from "react";

import { ConfirmDialog } from "@/components/confirm-dialog";
import { ApiError } from "@/lib/api-error";
import { type Member, type MemberRole, listMembers, setMemberRole } from "@/lib/members";

/** Human line for a failed call; ApiError carries the problem+json detail. */
function errorMessage(err: unknown): string {
  if (err instanceof ApiError) return err.message;
  return "Something went wrong. Is the backend running?";
}

/** What a row calls the person, falling back through what the provider sent. */
function nameOf(member: Member): string {
  return member.display_name || member.email || "Unnamed account";
}

/** The joining date, in the reader's own locale, or nothing if unparseable. */
function joined(member: Member): string {
  const at = new Date(member.created_at);
  if (Number.isNaN(at.getTime())) return "";
  return at.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

/** A change the page must ask about before it makes it. */
interface Pending {
  member: Member;
  role: MemberRole;
}

/**
 * Members: who is in this workspace, and what each of them may do.
 *
 * Two roles, and the page says what the difference buys rather than assuming
 * the words explain themselves. Nothing here removes an account: access is
 * granted by the identity provider, so revoking it belongs there - a page that
 * offered a Remove button would be promising something it cannot deliver.
 */
export default function MembersPage() {
  const [members, setMembers] = useState<Member[] | null>(null);
  const [loading, setLoading] = useState(true);
  // Two error slots, not one, and deliberately: a role change is refused and
  // then the roster is re-read, so a single slot would have the refetch clear
  // the refusal a few hundred ms after it appeared - the only explanation of a
  // refused demotion gone while the select snapped back. `loadError` is the
  // roster read's own failure; `actionError` is a refused change, and a refresh
  // never touches it.
  const [loadError, setLoadError] = useState<string | null>(null);
  const [actionError, setActionError] = useState<string | null>(null);
  // One change at a time: the guards are decided against the whole workspace
  // (the last administrator), so two in flight could be answered against a
  // roster neither of them showed.
  //
  // Held in a ref rather than enforced by disabling the controls. Disabling the
  // select mid-change takes the focus with it - the confirmation dialog returns
  // focus to whatever opened it, and a disabled element cannot receive it, so
  // the reader would be dropped onto the document body at precisely the moment
  // a refusal needs reading. The ref also cannot go stale inside a callback the
  // way the state value can.
  const inFlight = useRef(false);
  const [busy, setBusy] = useState(false);
  const [pending, setPending] = useState<Pending | null>(null);

  // Promise callbacks (not async/await) keep every setState out of the
  // synchronous call graph, which the set-state-in-effect lint rule rejects for
  // the mount-time call below (the same idiom as the model manager page).
  const refresh = useCallback(
    () =>
      listMembers()
        .then((next) => {
          setMembers(next);
          setLoadError(null);
        })
        // Only the roster's own error slot - a refused role change waiting on
        // screen is not the refetch's to clear.
        .catch((err: unknown) => setLoadError(errorMessage(err)))
        .finally(() => setLoading(false)),
    [],
  );

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const apply = useCallback(
    (member: Member, role: MemberRole) => {
      if (inFlight.current) return;
      inFlight.current = true;
      setBusy(true);
      // Clear last time's refusal only as a new attempt begins, so the message
      // stands for the whole life of the change it belongs to.
      setActionError(null);
      setMemberRole(member.id, role)
        // A refusal names itself - the last administrator, or an account the
        // deployment configures - and that sentence is the only explanation a
        // reader gets, so it is shown as it arrived, in a slot the refresh
        // below cannot wipe.
        .catch((err: unknown) => setActionError(errorMessage(err)))
        .finally(() => {
          // Released only once the roster has been re-read, not when the write
          // returns: in between, every control on screen still shows the roles
          // from before the change, and a second change decided against those
          // is exactly what the single-flight rule exists to prevent.
          void refresh().finally(() => {
            inFlight.current = false;
            setBusy(false);
          });
        });
    },
    [refresh],
  );

  /**
   * Handle a role pick.
   *
   * Promotions apply straight away - they only ever widen what somebody can do.
   * Demotions ask first, because the account being demoted loses this page, and
   * the person doing it may well be demoting themselves: the roster does not
   * say which row is the reader's own (nothing in this deployment tells the
   * browser who it is signed in as yet), so the question is asked for every
   * demotion and its wording covers both cases honestly.
   */
  const onPick = (member: Member, role: MemberRole) => {
    if (role === member.role) return;
    if (role === "member") {
      setPending({ member, role });
      return;
    }
    apply(member, role);
  };

  return (
    <div data-part="members">
      <header className="mbr-header">
        <h1 className="mbr-title">Members</h1>
        {members && (
          <p className="mm-meta">
            {members.length === 1
              ? "1 person in this workspace"
              : `${members.length} people in this workspace`}
          </p>
        )}
      </header>

      <p className="mbr-lede">
        Everybody here can chat, upload files and search the workspace knowledge
        base; administrators can additionally manage the workspace - models,
        integrations, and these roles. People appear here the first time they
        sign in.
      </p>

      {/* Announced, not merely displayed. A refused role change is this page's
          principal failure mode, and the only other thing that happens is the
          select snapping silently back to the value it already had - which is
          indistinguishable from the click having done nothing. Its own slot,
          so re-reading the roster underneath it cannot wipe it. */}
      {actionError && (
        <p className="mm-banner error" role="alert">
          {actionError}
        </p>
      )}

      {/* The roster read's own failure, kept apart from a refused change so the
          two never clear each other. */}
      {loadError && <p className="mm-banner error">{loadError}</p>}

      {loading && <p className="mm-muted">Loading...</p>}

      {members && members.length > 0 && (
        <ul className="mbr-list" aria-busy={busy}>
          {members.map((member) => (
            <MemberRow
              key={member.id}
              member={member}
              onPick={(role) => onPick(member, role)}
            />
          ))}
        </ul>
      )}

      {members && members.length === 0 && (
        // Only reachable on a deployment whose workspace was emptied out of
        // band; the seeded principal is a row like any other.
        <p className="mm-muted">Nobody has signed in to this workspace yet.</p>
      )}

      {pending && (
        <ConfirmDialog
          title="Remove administrator access?"
          body={
            `${nameOf(pending.member)} keeps full use of the assistant, but that ` +
            "account can no longer open this console - no model management, no " +
            "integrations, and no changes to these roles. If it is the account " +
            "in use right now, this page goes with it and another administrator " +
            "would have to undo it."
          }
          confirmLabel="Remove access"
          // Consequential, not destructive: nothing is deleted, and any
          // administrator can grant the role back.
          tone="accent"
          onConfirm={() => {
            apply(pending.member, pending.role);
            setPending(null);
          }}
          onCancel={() => setPending(null)}
        />
      )}
    </div>
  );
}

/** One person, and the one thing about them this page can change. */
function MemberRow({
  member,
  onPick,
}: {
  member: Member;
  onPick: (role: MemberRole) => void;
}) {
  const at = joined(member);
  return (
    <li className="mbr-row">
      <div className="mbr-identity">
        <span className="mbr-name">{nameOf(member)}</span>
        {/* The email is the account, so it is shown even when it is also the
            name - a row whose two lines agree is clearer than one that hides
            the identity the provider actually signed in. */}
        {member.email && <span className="mbr-email">{member.email}</span>}
      </div>

      <div className="mbr-control">
        {at && <span className="mm-meta">Joined {at}</span>}
        {/* Named for a screen reader, which would otherwise announce a bare
            "Administrator" with nothing saying whose role it is. An aria-label
            rather than a visually-hidden <label>: the row has one control, so
            there is no second element for the name to sit on. */}
        {/* Disabled ONLY for a role the deployment fixes. A change in flight
            is held off in the page's single-flight ref instead, because
            disabling this control mid-change is what takes focus away from the
            reader at the moment a refusal appears. */}
        <select
          className="mbr-role"
          aria-label={`Role for ${nameOf(member)}`}
          value={member.role}
          disabled={member.is_env_admin}
          onChange={(e) => onPick(e.target.value as MemberRole)}
        >
          <option value="admin">Administrator</option>
          <option value="member">Member</option>
        </select>
      </div>

      {/* Why the control is inert, said where the control is. The backend
          refuses this change too; without the sentence the disabled select
          would read as a bug. */}
      {member.is_env_admin && (
        <p className="mbr-fixed">
          Set by deployment config - this account is an administrator at every
          sign-in, and the role changes only in the deployment environment.
        </p>
      )}
    </li>
  );
}
