"use client";

import Link from "next/link";

/** Props for {@link NoChatModelNote}. */
export interface NoChatModelNoteProps {
  /**
   * The server's own explanation, shown verbatim: it is the one place that
   * knows WHICH way the deployment came up short, and it names the fix.
   */
  message: string;
}

/**
 * What the thread says when the model server holds nothing that can chat.
 *
 * A first-run state, not a fault - a fresh deployment pointed at an empty
 * server, or at one whose whole inventory is embedders - so it is deliberately
 * NOT the red error note every other failed turn gets: nothing is broken, and
 * alarming a setup step reads as a bug report for a step the user has simply
 * not taken yet. It carries the one action that resolves it instead, because a
 * notice that names a fix and then makes the reader go find it is only half a
 * notice.
 */
export function NoChatModelNote({ message }: NoChatModelNoteProps) {
  return (
    <div data-part="no-chat-model" role="alert">
      <p>{message}</p>
      <Link href="/console/models" className="btn btn-primary no-chat-model-action">
        Open the Model Manager
      </Link>
    </div>
  );
}
