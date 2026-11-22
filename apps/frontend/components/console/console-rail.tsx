"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

/** One rail entry: where it goes, and what it is called. */
interface RailEntry {
  href: string;
  label: string;
}

// Every entry goes somewhere that exists. A disabled or dead entry would be a
// promise the console cannot keep; a page arriving later arrives here with it.
const ENTRIES: RailEntry[] = [
  { href: "/console/models", label: "Model manager" },
  { href: "/console/knowledge", label: "Knowledge" },
  { href: "/console/integrations", label: "Integrations" },
  { href: "/console/defaults", label: "Workspace defaults" },
  { href: "/console/members", label: "Members" },
];

/**
 * The console's left rail.
 *
 * Under 700px it becomes a horizontal tab strip (see themes/console.css), so a
 * narrow screen keeps the full width for the page beside it rather than
 * squeezing both.
 */
export function ConsoleRail() {
  const pathname = usePathname();
  return (
    <nav data-part="console-rail" aria-label="Console sections">
      {ENTRIES.map((entry) => {
        // Prefix match, so a deeper page under a section keeps its entry lit.
        const active = pathname === entry.href || pathname.startsWith(`${entry.href}/`);
        return (
          <Link
            key={entry.href}
            href={entry.href}
            aria-current={active ? "page" : undefined}
            className="console-rail-item"
          >
            {entry.label}
          </Link>
        );
      })}
    </nav>
  );
}
