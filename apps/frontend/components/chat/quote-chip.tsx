"use client";

/**
 * The passage a turn is replying to, as one quiet line.
 *
 * Two surfaces, one shape. In the composer it is the pending quote and carries a
 * dismiss; on a sent question it is the record of what was replied to and
 * carries nothing - the passage is stored with the turn, so it is still here
 * after a reload even when the message it came from has been edited away.
 *
 * On a sent question the band is the way back to its source while that
 * message is in the thread: a button that glides the thread to it (see
 * useQuoteJump). The passage is authoritative and the source may be gone -
 * edited away, replaced, or quoted before it had an id - and then the band is
 * greyed, with the reason as its tooltip: a dead link that looked live would
 * be worse than none, and a link that is only ever offered where it leads
 * somewhere is not that. Not a rich card; a richer treatment belongs with the
 * transcript work, not here.
 */
export function QuoteChip({
  text,
  onDismiss,
  onJump,
  gone,
}: {
  /** The quoted passage; rendered on one line and clipped by CSS, never cut here. */
  text: string;
  /** Drop the pending quote. Omitted on a sent turn, where there is nothing to drop. */
  onDismiss?: () => void;
  /** Go to the message the passage came from; given only while that message is in the thread. */
  onJump?: () => void;
  /** Why the band leads nowhere, when it does not: the greyed band's tooltip. */
  gone?: string;
}) {
  // What a reader without the pointer hears before the words: the button's
  // verb, or the dead band's reason, neither of which the title alone carries
  // (a title is pointer-only). Out of flow, so the band's grid never sees it.
  const spoken =
    onJump !== undefined ? "Go to the quoted message: " : gone !== undefined ? `${gone} ` : null;
  const inner = (
    <>
      <span className="quote-mark" aria-hidden="true" />
      {/* The passage is untrusted text from anywhere in the transcript, so it is
          rendered as inert text - React escapes it - and clipped to one line. */}
      <span className="quote-text">
        {spoken !== null && <span className="sr-only">{spoken}</span>}
        {text}
      </span>
      {onDismiss !== undefined && (
        <button
          type="button"
          className="quote-dismiss"
          aria-label="Remove the quoted passage"
          title="Remove quote"
          onClick={onDismiss}
        >
          <svg
            viewBox="0 0 16 16"
            className="h-3 w-3"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.6"
            strokeLinecap="round"
            aria-hidden="true"
          >
            <path d="M4 4l8 8M12 4l-8 8" />
          </svg>
        </button>
      )}
    </>
  );
  // The passage rides along as an attribute too: on a sent question the
  // band's hidden spacer (CSS, `attr(data-text)`) measures the width the
  // passage asks for, so the bubble sizes to a short quote and is capped
  // on a long one. Inert text either way; nothing reads it as markup.
  if (onJump !== undefined) {
    return (
      <button
        type="button"
        data-part="quote-chip"
        data-text={text}
        title="Go to the quoted message"
        onClick={onJump}
      >
        {inner}
      </button>
    );
  }
  return (
    <div
      data-part="quote-chip"
      data-text={text}
      data-dead={gone !== undefined ? "true" : undefined}
      title={gone}
    >
      {inner}
    </div>
  );
}
