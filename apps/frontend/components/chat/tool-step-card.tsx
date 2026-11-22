"use client";

import { useState } from "react";

import { ArgsView } from "@/components/chat/args-view";
import { DisclosureRow } from "@/components/chat/disclosure-row";
import { Icon, MatrixLoader, Step } from "@/components/chat/step";
import type { ToolStep } from "@/lib/messages";
import { elapsedLabel, sourceCountLabel, toolGlyph } from "@/lib/timeline";
import { toolDisplayName } from "@/lib/tool-labels";

/**
 * Pick the argument worth previewing inline: the first non-empty string value.
 *
 * Tool calls lead with their most salient argument in practice (a query, a
 * title, a record id), so "first string" is a deliberate heuristic, not a
 * placeholder. Non-strings are skipped rather than stringified - a stray
 * boolean or nested object previews as noise.
 */
function argPreview(args: Record<string, unknown>): string | null {
  for (const value of Object.values(args)) {
    if (typeof value === "string" && value.trim() !== "") return value;
  }
  return null;
}

/**
 * One tool the assistant used this turn, as a row of the timeline.
 *
 * The glyph is the tool's medium (a magnifier, a globe, a plug), swapped for
 * the matrix loader in the tool colour while the call runs. The line names
 * the tool, previews its leading argument in mono, badges an approved or
 * denied call, and spells out the states the glyph cannot carry alone
 * ("running", "failed", "did not finish"); it opens in place to the full
 * arguments and a short summary of what came back. The model's narration
 * before the call ("let me search instead") stays above the row, so an
 * agentic detour is preserved rather than vanishing when the answer lands.
 */
export function ToolStepRow({ step, questionAt }: { step: ToolStep; questionAt?: string }) {
  const [open, setOpen] = useState(false);
  // A step with no result and no error was left mid-run (Stop/disconnect); it
  // is finished-inconclusive, not still running. A denied call ran nothing.
  const state = step.running
    ? "running"
    : step.denied || step.isError
      ? "danger"
      : step.result != null
        ? "ok"
        : "idle";
  const verb = step.running
    ? "Running"
    : step.denied
      ? "Blocked"
      : step.isError
        ? "Failed to run"
        : step.result != null
          ? "Used"
          : "Did not finish";
  // The trailing meta: the states the glyph cannot carry alone, or on a clean
  // finish the "N sources" the tool reported. A denied call already wears its
  // "denied" badge, so it gets no second word here.
  const meta = step.running
    ? "running"
    : step.isError
      ? "failed"
      : state === "idle"
        ? "did not finish"
        : state === "ok"
          ? sourceCountLabel(step.count)
          : null;
  const preview = argPreview(step.arguments);
  const hasDetail =
    Object.keys(step.arguments).length > 0 || (step.result != null && step.result !== "");
  const summary = (
    <>
      <span className="sr-only">{verb}</span>
      {/* Humanized for reading; the raw identifier stays on hover for debugging.
          Safe to soften here because this row NARRATES what already happened -
          the authorization gate (tool-confirmation-card) keeps the literal
          name, since the humanized form cannot distinguish a built-in from a
          remote tool that dresses like one. */}
      <span className="tool-label" title={step.name}>
        {toolDisplayName(step.name)}
      </span>
      {preview && <span className="tool-query">{preview}</span>}
      {step.approved && <span className="tool-badge approved">approved</span>}
      {step.denied && <span className="tool-badge denied">denied</span>}
      {meta && <span className="tool-meta">{meta}</span>}
    </>
  );
  return (
    <Step
      kind="tool"
      part="tool-trace"
      state={state}
      time={elapsedLabel(questionAt, step.at)}
      glyph={
        step.running ? (
          <MatrixLoader variant="scan" color="var(--c-step-tool)" />
        ) : (
          <Icon name={state === "danger" ? "x" : toolGlyph(step.name)} />
        )
      }
    >
      {step.narration && <div className="tool-narration">{step.narration}</div>}
      <DisclosureRow
        open={open}
        onToggle={hasDetail ? () => setOpen((o) => !o) : undefined}
        summary={summary}
      >
        <div className="tool-detail">
          <ArgsView args={step.arguments} />
          {step.result != null && step.result !== "" && (
            <div className={`tool-result${step.isError ? " error" : ""}`}>{step.result}</div>
          )}
        </div>
      </DisclosureRow>
    </Step>
  );
}
