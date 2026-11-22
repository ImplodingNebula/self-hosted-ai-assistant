"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import type { ComponentProps } from "react";

import type { View } from "@/lib/turn-state";
import { type TransitionKind, transitionRoute } from "@/lib/view-transition";

/**
 * A link whose navigation travels: the route change runs inside a view
 * transition (lib/view-transition.ts), so what the stylesheet names for its
 * kind - the topbar and the side rail, between the chat and the console -
 * moves in place rather than cutting.
 *
 * A plain link in every other respect (prefetch, a new tab, a modified
 * click); only a client-side navigation is taken over, as a push that keeps
 * the default scroll (so `replace` and `scroll` are not offered).
 */
export function TravelLink({
  kind,
  view,
  href,
  ...rest
}: Omit<
  ComponentProps<typeof Link>,
  "href" | "onNavigate" | "replace" | "scroll"
> & {
  kind: TransitionKind;
  /** The view the destination lands on; a function is read at the click. */
  view: View | (() => View);
  href: string;
}) {
  const router = useRouter();
  return (
    <Link
      {...rest}
      href={href}
      onNavigate={(e) => {
        e.preventDefault();
        transitionRoute(kind, typeof view === "function" ? view() : view, () =>
          router.push(href),
        );
      }}
    />
  );
}
