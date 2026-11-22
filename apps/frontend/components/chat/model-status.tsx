"use client";

import { MatrixLoader } from "@/components/chat/step";

/** Where the chat model is right now, as far as the page can tell. */
export type ModelWhereabouts = "memory" | "disk" | "loading" | "unknown";

const STATE_WORD: Record<ModelWhereabouts, string> = {
  memory: "in memory",
  disk: "on disk",
  loading: "loading",
  unknown: "",
};

/**
 * The topbar's model status: the model's name in mono, a state word and a
 * dot - filled while the model is in memory, a ring while it is on disk, the
 * matrix loader in the warn colour while it loads. On narrow screens the
 * sheet keeps only the dot.
 *
 * `unknown` is "nobody has said": the runtime does not report residency, or
 * the page has not heard yet. It shows the name alone, with no dot, rather
 * than guessing a state.
 */
export function ModelStatus({
  model,
  state,
}: {
  /** The model a turn would run on, or null when the page cannot name it. */
  model: string | null;
  state: ModelWhereabouts;
}) {
  const word = STATE_WORD[state];
  return (
    <span data-part="model-status" data-model-state={state} role="status">
      {state === "loading" && <MatrixLoader variant="scan" color="var(--s-warn)" />}
      <span className="ms-dot" aria-hidden="true" />
      <span className="ms-name">{model ?? "Default model"}</span>
      {word !== "" && <span className="ms-state">{word}</span>}
    </span>
  );
}
