"use client";

import { Icon, MatrixLoader } from "@/components/chat/step";

/**
 * A notice that belongs to the composer, docked ABOVE it in the layout.
 *
 * It opens as a row (a grid track from 0fr to 1fr), never as a toast over the
 * thread, so it can never overlap the status row; closed, it keeps its last
 * words while the track collapses, so the text does not vanish mid-fade. The
 * loader marks the one subject it has today: a model loading from disk.
 */
export function ComposerNotice({
  open,
  title,
  body,
  loading,
  onDismiss,
}: {
  open: boolean;
  title: string;
  body: string;
  /** The notice is about a load in progress: show the loader in the warn colour. */
  loading?: boolean;
  onDismiss: () => void;
}) {
  return (
    <div data-part="composer-notice" data-open={open ? "true" : undefined} role="status">
      <div>
        <div className="cn-box">
          {loading && <MatrixLoader variant="scan" color="var(--s-warn)" />}
          <div className="cn-text">
            <b>{title}</b>
            <span>{body}</span>
          </div>
          <button
            type="button"
            className="cn-close"
            aria-label="Dismiss"
            onClick={onDismiss}
            tabIndex={open ? 0 : -1}
          >
            <Icon name="cross" />
          </button>
        </div>
      </div>
    </div>
  );
}
