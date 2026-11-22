/**
 * The theme registry: the single source the picker, the ThemeProvider, and
 * the no-flash script agree on. Adding a catalog theme later is one entry
 * here plus one `app/themes/<id>.css` file - the rest is data-driven.
 *
 * See docs/design/DESIGN.md for the catalog and the token architecture.
 */

/** A selectable UI theme, applied via `data-theme` on `<html>`. */
export interface ThemeDef {
  /** Stable id: written to `data-theme` and persisted as `app.theme`. */
  id: string;
  /** Human label shown in the picker (e.g. "Aurora"). */
  label: string;
  /**
   * Mode behavior. `paired` follows `data-mode` (a user pin, else the OS via
   * `prefers-color-scheme`); `light` / `dark` are single-mode and ignore the
   * OS. The picker shows a mode toggle only for `paired` themes.
   */
  modes: "paired" | "light" | "dark";
  /** Accent variants, when a theme offers them (e.g. Phosphor); omitted otherwise. */
  accents?: string[];
  /** One-line description shown under the theme name in the picker. */
  description: string;
}

/**
 * Shipped themes, in DESIGN.md catalog order. F1 ships Aurora (the default,
 * dark glass) and Minimal (the OS-following light/dark pair); the remaining
 * catalog themes register here as their token files land in Phase 7.
 */
export const THEMES: ThemeDef[] = [
  {
    id: "aurora",
    label: "Aurora",
    modes: "dark",
    description: "Ambient instrument - deep indigo glass under a slow aurora.",
  },
  {
    id: "minimal",
    label: "Minimal",
    modes: "paired",
    description: "Nothing but the conversation; follows the system light or dark.",
  },
];

/** The theme applied before any user choice - dark glass on first visit. */
export const DEFAULT_THEME = "aurora";
