"use client";

import { Icon } from "@/components/chat/step";

/**
 * Standing honesty note for the current turn (a substitute model answered, a
 * search found nothing, a file could not be used): a quiet line in the reply's
 * closing row, never a toast.
 */
export function InfoNote({ message }: { message: string }) {
  return (
    <div data-part="notice" role="status">
      <Icon name="info" />
      <span>{message}</span>
    </div>
  );
}
