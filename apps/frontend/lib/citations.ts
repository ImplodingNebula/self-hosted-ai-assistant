/**
 * Pure derivations over `CitationData`, shared by the marker and chip views of
 * the citation popover.
 *
 * One object, two views (`components/chat/citation-popover.tsx`), and both read
 * a citation the same way - so the derivation lives here once rather than being
 * reimplemented per view and drifting.
 */
import type { CitationData } from "@/lib/types";

/** The one action a popover offers for a citation, chosen by {@link citationAction}. */
export type CitationActionKind =
  | "open_page"
  | "view_knowledge"
  | "view_tool_step"
  | "download_attachment";

/**
 * The popover's action for one citation, derived strictly from fields already
 * on the wire (the citations-wave contract locks the wire shape - no new field
 * exists to carry this, so it is read off what is already there).
 *
 * Order matters and is exclusive: a web result carries both a `url` and
 * `source_type: "web"`, and `url` wins, so it is checked first. Falls through
 * to `null` (no action row) for anything that names none of the known kinds -
 * a citation still shows its icon, label and passage with nothing to click.
 */
export function citationAction(citation: CitationData): CitationActionKind | null {
  if (citation.url) return "open_page";
  if (citation.source_type === "records") return "view_knowledge";
  if (citation.source_type === "mcp_records") return "view_tool_step";
  // The contract calls this source "attachment"; the wire's actual SourceType
  // literal (app/providers/rag/base.py) is "upload" - there is no "attachment"
  // value anywhere in the backend, so this reads the real one.
  if (citation.source_type === "upload") return "download_attachment";
  return null;
}

/** Stable id format `attachment_ingest.py`'s `_doc_id` stamps on an uploaded file's chunks. */
const ATTACHMENT_DOC_ID = /^attachment:([0-9a-fA-F-]{36}):\d+:\d+$/;

/**
 * Recover the attachment id backing an uploaded-file citation, or null when
 * `doc_id` does not name one.
 *
 * `doc_id` for an upload is `attachment:<uuid>:<ordinal>:<part>` (one part of
 * one parsed unit, `_doc_id` in `attachment_ingest.py`) - stable by design, so
 * reading the uuid back out is safe. Null covers both an absent `doc_id` and a
 * shape that does not match (a citation of some other kind), so a caller never
 * needs to guess which.
 */
export function attachmentIdFromDocId(docId: string | null): string | null {
  if (!docId) return null;
  const match = ATTACHMENT_DOC_ID.exec(docId);
  return match ? match[1] : null;
}

/** Short, human label for a citation's kind - the popover's icon and header text. */
export function citationTypeLabel(sourceType: string | null): string {
  switch (sourceType) {
    case "web":
      return "Web result";
    case "records":
      return "Knowledge base";
    case "mcp_records":
      return "Tool result";
    case "upload":
      return "Uploaded file";
    default:
      return "Source";
  }
}

/**
 * Index a message's citations by their (stringified) marker number.
 *
 * Strings, not numbers, matching `answerMarkPlugin`'s `citationIndexes`: a
 * marker's digits are compared as text there so a padded `"01"` cannot borrow
 * source `1`'s entry, and a lookup built from numbers would reopen that gap.
 * Citations without an `index` (unnumbered - chip-only) are left out, which is
 * exactly the set of degenerate lookups a marker could ever perform.
 */
export function citationsByIndex(
  citations: readonly CitationData[],
): ReadonlyMap<string, CitationData> {
  const byIndex = new Map<string, CitationData>();
  for (const citation of citations) {
    if (citation.index != null) byIndex.set(String(citation.index), citation);
  }
  return byIndex;
}
