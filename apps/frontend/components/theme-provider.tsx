"use client";

import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useSyncExternalStore,
  type ReactNode,
} from "react";

import { DEFAULT_THEME, THEMES, type ThemeDef } from "@/lib/themes";

/** A resolved, applied color mode. */
export type Mode = "light" | "dark";

/**
 * The user's mode preference. `system` means "follow the OS" and is the
 * default for paired themes; a pinned `light` / `dark` overrides the OS.
 * Single-mode themes ignore this entirely.
 */
export type ModePref = Mode | "system";

/** The theme state exposed by {@link useTheme}. */
export interface ThemeState {
  /** Active theme id (also on `<html data-theme>`). */
  theme: string;
  /** The full registry entry for the active theme. */
  activeTheme: ThemeDef;
  /** The mode currently applied (on `<html data-mode>`). */
  mode: Mode;
  /** The user's mode preference; `system` follows the OS. */
  modePref: ModePref;
  /** The active accent variant, or `null` when the theme has none. */
  accent: string | null;
  /** The full theme registry, for the picker. */
  themes: ThemeDef[];
  /** Switch the active theme. */
  setTheme: (id: string) => void;
  /** Set the mode preference (`system` clears any pin). */
  setModePref: (pref: ModePref) => void;
  /** Set the accent variant (ignored by themes without accents). */
  setAccent: (accent: string | null) => void;
}

const KEY_THEME = "app.theme";
const KEY_MODE = "app.mode";
const KEY_ACCENT = "app.accent";

/** The persisted choice plus the current OS color scheme. */
interface Prefs {
  theme: string;
  modePref: ModePref;
  accent: string | null;
  os: Mode;
}

// Stable server/first-paint snapshot: the default theme is dark-only, so the
// OS value is inert here and the object can be a constant reference (required
// by useSyncExternalStore to avoid render loops during hydration).
const SERVER_PREFS: Prefs = {
  theme: DEFAULT_THEME,
  modePref: "system",
  accent: null,
  os: "dark",
};

function themeById(id: string): ThemeDef {
  return THEMES.find((t) => t.id === id) ?? THEMES[0];
}

/**
 * Resolve the mode to apply. Single-mode themes force their own mode (so
 * Aurora is always dark, even on a light-OS machine - which the dark-variant
 * bridge in globals.css depends on); paired themes honor a pin, else the OS.
 */
function resolveMode(theme: ThemeDef, pref: ModePref, os: Mode): Mode {
  if (theme.modes === "dark") return "dark";
  if (theme.modes === "light") return "light";
  return pref === "system" ? os : pref;
}

/* --------------------------------------------------------------------------
 * External store: persisted prefs (localStorage) + the OS color scheme.
 * Modeled as an external store so the provider reads it through
 * useSyncExternalStore - no setState-in-effect, and hydration-safe (the
 * server snapshot renders first, then the client snapshot settles in).
 * ------------------------------------------------------------------------ */

const listeners = new Set<() => void>();
let snapshot: Prefs = SERVER_PREFS;
let mql: MediaQueryList | null = null;

function osQuery(): MediaQueryList {
  return (mql ??= window.matchMedia("(prefers-color-scheme: dark)"));
}

function readPrefs(): Prefs {
  // Storage access can THROW (Firefox with dom.storage.enabled=false, a
  // sandboxed iframe without allow-same-origin), not just return null. Fall
  // back to the default snapshot rather than crash the render commit - the
  // no-flash script degrades the same way via its own try/catch.
  try {
    const storedTheme = localStorage.getItem(KEY_THEME);
    const storedMode = localStorage.getItem(KEY_MODE);
    const theme =
      storedTheme && THEMES.some((t) => t.id === storedTheme)
        ? storedTheme
        : DEFAULT_THEME;
    const modePref: ModePref =
      storedMode === "light" || storedMode === "dark" ? storedMode : "system";
    return {
      theme,
      modePref,
      accent: localStorage.getItem(KEY_ACCENT),
      os: osQuery().matches ? "dark" : "light",
    };
  } catch {
    return SERVER_PREFS;
  }
}

function samePrefs(a: Prefs, b: Prefs): boolean {
  return (
    a.theme === b.theme &&
    a.modePref === b.modePref &&
    a.accent === b.accent &&
    a.os === b.os
  );
}

// Recompute from the platform and notify subscribers only on a real change,
// so getSnapshot keeps returning a stable reference between renders.
function refresh(): void {
  const next = readPrefs();
  if (!samePrefs(snapshot, next)) {
    snapshot = next;
    listeners.forEach((l) => l());
  }
}

function subscribe(onChange: () => void): () => void {
  if (listeners.size === 0) {
    window.addEventListener("storage", refresh);
    osQuery().addEventListener("change", refresh);
  }
  listeners.add(onChange);
  // Settle the snapshot to real values now that we are on the client.
  refresh();
  return () => {
    listeners.delete(onChange);
    if (listeners.size === 0) {
      window.removeEventListener("storage", refresh);
      osQuery().removeEventListener("change", refresh);
    }
  };
}

function getSnapshot(): Prefs {
  return snapshot;
}

function getServerSnapshot(): Prefs {
  return SERVER_PREFS;
}

// Persist a change (a null value clears the key) and re-read, so this tab's
// store updates immediately (the `storage` event only fires in other tabs).
function persist(key: string, value: string | null): void {
  try {
    if (value === null) {
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, value);
    }
  } catch {
    // Storage unavailable (see readPrefs): silently no-op. The environment is
    // already degraded and readPrefs falls back to defaults, so there is
    // nothing useful to reflect.
    return;
  }
  refresh();
}

const ThemeContext = createContext<ThemeState | null>(null);

/**
 * Owns the theme selection attributes on `<html>` (`data-theme` /
 * `data-mode` / `data-accent`), persists the choice to `localStorage`, and
 * keeps paired themes in sync with the OS when no mode is pinned.
 *
 * The first paint is themed by the inline no-flash script in the document
 * head (see app/layout.tsx); this provider settles to the same values after
 * hydration and owns every change afterward, so there is no flash.
 */
export function ThemeProvider({ children }: { children: ReactNode }) {
  const prefs = useSyncExternalStore(subscribe, getSnapshot, getServerSnapshot);
  const activeTheme = themeById(prefs.theme);
  const mode = resolveMode(activeTheme, prefs.modePref, prefs.os);

  // Apply the resolved selection to <html>. DOM writes only (no setState),
  // reaffirming what the no-flash script already set.
  useEffect(() => {
    const root = document.documentElement;
    root.dataset.theme = prefs.theme;
    root.dataset.mode = mode;
    if (activeTheme.accents && prefs.accent) {
      root.dataset.accent = prefs.accent;
    } else {
      delete root.dataset.accent;
    }
  }, [prefs.theme, prefs.accent, mode, activeTheme]);

  const setTheme = useCallback((id: string) => persist(KEY_THEME, id), []);
  const setModePref = useCallback(
    (pref: ModePref) => persist(KEY_MODE, pref === "system" ? null : pref),
    [],
  );
  const setAccent = useCallback((a: string | null) => persist(KEY_ACCENT, a), []);

  const value: ThemeState = {
    theme: prefs.theme,
    activeTheme,
    mode,
    modePref: prefs.modePref,
    accent: prefs.accent,
    themes: THEMES,
    setTheme,
    setModePref,
    setAccent,
  };

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
}

/** Read and control the active theme. Must be used under a {@link ThemeProvider}. */
export function useTheme(): ThemeState {
  const ctx = useContext(ThemeContext);
  if (!ctx) {
    throw new Error("useTheme must be used within a ThemeProvider");
  }
  return ctx;
}
