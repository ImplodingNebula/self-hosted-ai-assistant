"use client";

import { useEffect, useId, useRef, useState } from "react";

import { type ModePref, useTheme } from "@/components/theme-provider";

/** Mode-preference options offered for paired themes. */
const MODE_OPTIONS: { value: ModePref; label: string }[] = [
  { value: "system", label: "System" },
  { value: "light", label: "Light" },
  { value: "dark", label: "Dark" },
];

/**
 * Two-swatch preview for a theme. Scopes a nested `data-theme` so the real
 * `--s-bg-app` / `--s-accent` tokens resolve for the previewed theme rather
 * than the active one; paired themes preview in light.
 */
function Swatch({ themeId }: { themeId: string }) {
  return (
    <span
      data-theme={themeId}
      data-mode="light"
      aria-hidden="true"
      className="flex h-5 w-5 shrink-0 items-center justify-center overflow-hidden rounded-full border border-border-line"
      style={{ backgroundColor: "var(--s-bg-app)" }}
    >
      <span
        className="h-2.5 w-2.5 rounded-full"
        style={{ backgroundColor: "var(--s-accent)" }}
      />
    </span>
  );
}

/**
 * Compact theme picker for the sidebar footer: a "Theme: <name>" trigger that
 * opens a popover above it, listing the registered themes with descriptions
 * and swatches. A mode control (System / Light / Dark) shows only for paired
 * themes; accent dots show only for themes that declare accents.
 */
export function ThemePicker() {
  const {
    themes,
    theme,
    activeTheme,
    modePref,
    setTheme,
    setModePref,
    accent,
    setAccent,
  } = useTheme();
  const [open, setOpen] = useState(false);
  const rootRef = useRef<HTMLDivElement>(null);
  const menuId = useId();

  // Close on outside click or Escape while open.
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (rootRef.current && !rootRef.current.contains(e.target as Node)) {
        setOpen(false);
      }
    };
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") setOpen(false);
    };
    document.addEventListener("mousedown", onDown);
    document.addEventListener("keydown", onKey);
    return () => {
      document.removeEventListener("mousedown", onDown);
      document.removeEventListener("keydown", onKey);
    };
  }, [open]);

  return (
    <div ref={rootRef} data-part="theme-picker" className="relative">
      {/* Styled as a `.convo` footer row so its hover matches the Integrations
          link exactly (the owner-reported "no hover feedback" was this trigger
          reading differently from its neighbours). */}
      <button
        type="button"
        aria-expanded={open}
        aria-controls={menuId}
        onClick={() => setOpen((o) => !o)}
        className="convo"
      >
        <Swatch themeId={theme} />
        <span className="min-w-0 flex-1 truncate">Theme: {activeTheme.label}</span>
      </button>

      {open && (
        <div
          id={menuId}
          role="group"
          aria-label="Theme"
          className="absolute bottom-full left-0 z-40 mb-2 w-64 rounded-[var(--radius-m)] border border-border-line p-1.5 shadow-[0_12px_32px_var(--s-shadow)]"
          // The shared floating-panel recipe: the menu fill (themes re-point
          // it - aurora's raised surface is not the panel glass) and the
          // frost that keeps content behind the popover from bleeding
          // through; `none` in flat themes.
          style={{
            background: "var(--c-menu-bg, var(--s-bg-raised))",
            backdropFilter: "var(--c-menu-backdrop)",
            WebkitBackdropFilter: "var(--c-menu-backdrop)",
          }}
        >
          {themes.map((t) => {
            const selected = t.id === theme;
            return (
              <div key={t.id}>
                <button
                  type="button"
                  aria-pressed={selected}
                  onClick={() => {
                    setTheme(t.id);
                    if (!t.accents) setAccent(null);
                  }}
                  className={`flex w-full items-start gap-2.5 rounded-[var(--radius-s)] px-2.5 py-2 text-left transition-colors hover:bg-inset ${
                    selected ? "bg-inset" : ""
                  }`}
                >
                  <span className="mt-0.5">
                    <Swatch themeId={t.id} />
                  </span>
                  <span className="min-w-0 flex-1">
                    <span className="flex items-center gap-1.5 text-sm font-medium text-fg">
                      {t.label}
                      {selected && (
                        <svg
                          aria-hidden="true"
                          viewBox="0 0 16 16"
                          className="h-3.5 w-3.5 text-accent"
                          fill="none"
                          stroke="currentColor"
                          strokeWidth="2"
                          strokeLinecap="round"
                          strokeLinejoin="round"
                        >
                          <path d="M3.5 8.5l3 3 6-6.5" />
                        </svg>
                      )}
                    </span>
                    <span className="block text-xs text-fg-3">{t.description}</span>
                  </span>
                </button>

                {/* Accent dots: only for themes that declare accents. */}
                {selected && t.accents && (
                  <div
                    role="group"
                    aria-label="Accent"
                    className="flex gap-1.5 px-2.5 pb-1.5 pl-10"
                  >
                    {t.accents.map((a) => (
                      <button
                        key={a}
                        type="button"
                        aria-label={a}
                        aria-pressed={accent === a}
                        onClick={() => setAccent(a)}
                        className={`h-4 w-4 rounded-full border transition-transform ${
                          accent === a
                            ? "border-accent scale-110"
                            : "border-border-line"
                        }`}
                        style={{ backgroundColor: "var(--s-accent)" }}
                      />
                    ))}
                  </div>
                )}
              </div>
            );
          })}

          {/* Mode control: paired themes only. */}
          {activeTheme.modes === "paired" && (
            <div className="mt-1.5 border-t border-border-line px-1 pt-2">
              <div
                role="group"
                aria-label="Color mode"
                className="flex items-center gap-0.5 rounded-[var(--radius-s)] bg-inset p-0.5"
              >
                {MODE_OPTIONS.map((opt) => {
                  const active = modePref === opt.value;
                  return (
                    <button
                      key={opt.value}
                      type="button"
                      aria-pressed={active}
                      onClick={() => setModePref(opt.value)}
                      className={`min-h-10 flex-1 rounded-[calc(var(--radius-s)_-_2px)] px-2 py-1 text-xs transition-colors ${
                        active
                          ? "bg-raised font-medium text-fg"
                          : "text-fg-2 hover:text-fg"
                      }`}
                    >
                      {opt.label}
                    </button>
                  );
                })}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
