/**
 * Teach plain Node the two specifiers the app's build resolves for it.
 *
 * Next.js resolves `@/lib/x` against the workspace root; a script run straight
 * from Node does not, so a module that uses the alias cannot be loaded and
 * checked without a bundler. `next/server` is the same shape of gap from the
 * other direction - the package publishes it as a plain `next/server.js` file
 * with no export map entry, so the bare specifier every Next file writes is
 * unresolvable outside the build. Both are pure resolution; nothing here
 * transpiles or rewrites anything else.
 */

import { existsSync } from "node:fs";
import path from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";

const ROOT = fileURLToPath(new URL("../", import.meta.url));

/** Resolve `@/...` and `next/server` the way the bundler would. */
export function resolve(specifier, context, nextResolve) {
  if (specifier === "next/server") return nextResolve("next/server.js", context);
  if (!specifier.startsWith("@/")) return nextResolve(specifier, context);
  const base = path.join(ROOT, specifier.slice(2));
  for (const candidate of [`${base}.ts`, `${base}.tsx`, base, path.join(base, "index.ts")]) {
    if (existsSync(candidate)) {
      return nextResolve(pathToFileURL(candidate).href, context);
    }
  }
  return nextResolve(specifier, context);
}
