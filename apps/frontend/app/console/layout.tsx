import { BackToChat } from "@/components/console/back-to-chat";
import { ConsoleRail } from "@/components/console/console-rail";
import { ViewMarker } from "@/components/view-marker";

/**
 * The admin console shell: the chat-consistent topbar, plus a rail beside the
 * page it frames.
 *
 * A server component holding one client rail (the rail alone needs the current
 * path), so every page under it is free to be either.
 */
export default function ConsoleLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <div className="console-page">
      <ViewMarker view="console" />
      <header data-part="topbar">
        {/* The bar and the rail travel back into the chat's. */}
        <BackToChat>
          <svg
            suppressHydrationWarning
            viewBox="0 0 16 16"
            className="h-4 w-4"
            fill="none"
            stroke="currentColor"
            strokeWidth="1.75"
            strokeLinecap="round"
            strokeLinejoin="round"
            aria-hidden="true"
          >
            <path suppressHydrationWarning d="M10 3L5 8l5 5" />
          </svg>
          Back to chat
        </BackToChat>
        <span className="title">Console</span>
      </header>

      <div className="console-body">
        <ConsoleRail />
        {/* The work column, framed: a theme paints its veil on this box (the
            sky held back from the page), so it must not scroll - the scroll
            container sits inside it. */}
        <div className="console-main">
          <main className="console-scroll">{children}</main>
        </div>
      </div>
    </div>
  );
}
