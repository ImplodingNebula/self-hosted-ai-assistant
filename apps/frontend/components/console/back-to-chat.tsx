"use client";

import type { ReactNode } from "react";

import { TravelLink } from "@/components/travel-link";
import { peekResumedConversation } from "@/lib/resume";

/**
 * The console's way back: the bar and the rail travel into the chat's, and
 * the chat lands on the conversation the reader left (lib/resume.ts) or on
 * a new chat. Which of the two the travel waits for is read at the click,
 * since the console's layout is rendered on the server and cannot know.
 */
export function BackToChat({ children }: { children: ReactNode }) {
  return (
    <TravelLink
      href="/"
      kind="console"
      view={() => (peekResumedConversation() === null ? "new" : "thread")}
      className="console-back"
    >
      {children}
    </TravelLink>
  );
}
