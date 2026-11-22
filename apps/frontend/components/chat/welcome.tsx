"use client";

import type { ReactNode } from "react";

import { Icon } from "@/components/chat/step";
import type { ReachItem } from "@/lib/boot";

/**
 * A new chat is the greeting and the composer, together. With no thread to
 * protect, the serif greeting (one word in italic, in the sky's colour), the
 * reach line beneath it (what this install can reach right now, in the glyphs
 * the rail will use once a turn runs) and the composer sit a little above
 * centre, and the rest of the pane is open sky. Anything the pane has to say
 * before the first message (the first-run notice, an unreachable backend)
 * sits under the reach line.
 */
export function Welcome({ items, children }: { items: readonly ReachItem[]; children?: ReactNode }) {
  return (
    <section className="welcome" aria-label="New chat">
      <h1 className="welcome-head">
        What shall we shed <em>light</em> on?
      </h1>
      {items.length > 0 && (
        <ul className="reach">
          {items.map((item) => (
            <li key={item.label}>
              <Icon name={item.glyph} />
              <b>{item.label}</b> {item.value}
            </li>
          ))}
        </ul>
      )}
      {/* a live region, as the thread is: a note that lands here is announced */}
      <div className="welcome-note" aria-live="polite">
        {children}
      </div>
    </section>
  );
}
