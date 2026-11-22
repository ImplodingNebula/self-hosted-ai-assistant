"use client";

import { useAppName } from "@/components/branding-provider";
import { Icon } from "@/components/chat/step";
import type { BootState } from "@/hooks/use-boot";
import { BOOT_WAITS, type LitLine, litShare } from "@/lib/boot";

/**
 * The boot screen: the sky and the stars have the window while the shell
 * cannot yet mean anything. The mark draws itself in and keeps a slow halo;
 * the name sits under it; the headline names what is being waited on with
 * the waited-on word lit; a rail of nodes lights as each wait ends. A wait
 * that runs long says so, counts, and offers a retry. The chart of rings
 * behind the words is drawn by the sky onto the canvas here (components/
 * sky.tsx), which the theme without a sky never shows.
 *
 * The handoff into the new chat is the stylesheet's view transition: the
 * mark into the sidebar's brand, the headline into the greeting, each node
 * into its reach item.
 */
export function Boot({ state }: { state: BootState }) {
  const appName = useAppName();
  const done = state.nodes.filter((n) => n.phase === "done").length;
  const headKey = state.head.join("|");
  return (
    <div className="boot" aria-live="polite" aria-busy={state.phase === "booting" && state.slow === null}>
      <canvas className="boot-chart" aria-hidden="true" />
      <div className="boot-mark">
        <Icon name="logo" />
      </div>
      <div className="boot-name">{appName}</div>
      {/* keyed on its words, so a new wait's headline plays its entrance again */}
      <h1 className="boot-head swap-in" key={headKey}>
        <Lit line={state.head} />
      </h1>
      <p className="boot-sub">{state.sub}</p>
      <ol className="boot-rail" style={{ "--lit": litShare(done) } as React.CSSProperties}>
        {BOOT_WAITS.map((w, i) => (
          <li key={w.node} className="boot-node" data-step={state.nodes[i]?.phase ?? "pending"}>
            <span className="boot-dot">
              <Icon name="check" />
            </span>
            <span>
              <span className="boot-label">{w.label}</span>
              <small>{state.nodes[i]?.detail ?? ""}</small>
            </span>
          </li>
        ))}
      </ol>
      {state.slow !== null && (
        <div className="boot-slow">
          <p>{state.slow}</p>
          <div className="boot-slow-verbs">
            <button type="button" className="btn btn-ghost" onClick={state.retry}>
              Try again
            </button>
            {/* a wait the shell does not truly need (the model runtime) may be given up on */}
            {state.skip !== null && (
              <button type="button" className="btn btn-ghost" onClick={state.skip}>
                Continue without it
              </button>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/** A headline with its one lit word in italic, in the sky's colour. */
export function Lit({ line }: { line: LitLine }) {
  return (
    <>
      {line[0]}
      <em>{line[1]}</em>
      {line[2]}
    </>
  );
}
