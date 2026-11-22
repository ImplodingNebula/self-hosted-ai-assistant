import { GeistMono } from "geist/font/mono";
import { GeistSans } from "geist/font/sans";
import type { Metadata } from "next";
import localFont from "next/font/local";
import "katex/dist/katex.min.css";
import "./globals.css";

import { BrandingProvider } from "@/components/branding-provider";
import { Sky } from "@/components/sky";
import { ThemeProvider } from "@/components/theme-provider";
import { appName } from "@/lib/branding";

// The three faces ship inside the app (the `geist` package carries Vercel's
// two, each registering its own `--font-*` variable; Source Serif 4 comes
// from its fontsource package), so a build never reaches out for them: a
// build box without Google Fonts used to get the metric fallback instead,
// silently, and every serif surface rendered in Times New Roman.

// Aurora's one added face, for the answer column and the display titles
// (DESIGN.md, Typography). The files are read straight from the package:
// the path is relative to this file, and a layout that moved the package
// (a hoisted workspace) would fail the build loudly rather than fall back.
// Not preloaded: a theme's prose token points at it, and the browser fetches
// it only once that token is read.
const sourceSerif = localFont({
  variable: "--font-source-serif",
  src: [
    {
      path: "../node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-normal.woff2",
      style: "normal",
    },
    {
      path: "../node_modules/@fontsource-variable/source-serif-4/files/source-serif-4-latin-opsz-italic.woff2",
      style: "italic",
    },
  ],
  weight: "200 900",
  preload: false,
});

// Every route under this layout renders per request rather than being built
// ahead of time. That is what makes the product name CONFIGURATION: a
// prerendered page bakes whatever APP_NAME held when the image was built, so
// the name would be fixed at build time exactly as a NEXT_PUBLIC_ variable
// would - and it would still look right in development, where every request
// renders anyway, so the failure would only ever appear in a deployment. The
// pages this covers are client components that fetch their own data, so their
// prerendered output was an empty shell; rendering it per request costs the
// server a small React pass and no round trips - measured indistinguishable
// from serving the prebuilt shell, because the sign-in gate in proxy.ts
// already asks the backend a question on every page request either way.
export const dynamic = "force-dynamic";

/** Title and description, named after whatever this deployment calls itself. */
export function generateMetadata(): Metadata {
  return {
    title: appName(),
    description: "Local-first, model-agnostic AI chat with RAG + MCP.",
  };
}

// No-flash theme script: runs synchronously in <head>, before first paint, so
// the initial render is already themed and reload never flashes the default.
// It sets data-theme / data-mode / data-accent on <html> from localStorage,
// forcing the mode for single-mode themes (Aurora is dark-only) and otherwise
// following a pin or the OS.
//
// Deliberately thin - it does NOT import the registry, so two known drifts are
// reconciled by the ThemeProvider on hydration rather than handled here:
//   - The `forced` map is a hand-copy of the single-mode entries of THEMES in
//     lib/themes.ts and must be kept in sync with them.
//   - The stored theme is NOT validated against the registry, so a stale
//     app.theme (e.g. a theme removed by a rolled-back deploy) stamps an
//     unknown data-theme and the first paint is unthemed until the provider
//     falls it back to the default.
//   - data-accent is stamped unconditionally from storage here, whereas the
//     provider gates it on activeTheme.accents (a stray accent is inert to the
//     shipped themes, which declare none).
//   - data-sky is aurora's renderer choice (DESIGN.md, "The sky"). The first
//     paint needs only the off/on decision - reduced transparency means the
//     flat night, otherwise the veil and the shade are already right - and the
//     Sky component settles shader vs css once it knows whether WebGL works.
const NO_FLASH_SCRIPT = `(function(){try{var d=document.documentElement;var t=localStorage.getItem("app.theme")||"aurora";var forced={aurora:"dark"};var pin=localStorage.getItem("app.mode");var os=matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light";var m=forced[t]||(pin==="light"||pin==="dark"?pin:os);d.dataset.theme=t;d.dataset.mode=m;var a=localStorage.getItem("app.accent");if(a)d.dataset.accent=a;if(t==="aurora")d.dataset.sky=matchMedia("(prefers-reduced-transparency: reduce)").matches?"off":"shader";}catch(e){}})()`;

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  // suppressHydrationWarning on <html>: the no-flash script (and browser
  // extensions like Dark Reader) set attributes on the html element before
  // React hydrates, causing a benign server/client mismatch. This silences
  // that one-level-deep warning only; it does not disable hydration checks
  // for the rest of the tree.
  return (
    <html
      lang="en"
      suppressHydrationWarning
      className={`${GeistSans.variable} ${GeistMono.variable} ${sourceSerif.variable} h-full antialiased`}
    >
      <head>
        <script dangerouslySetInnerHTML={{ __html: NO_FLASH_SCRIPT }} />
        {/* Dark Reader rewrites inline styles/attributes on the server-rendered
            DOM before React hydrates, which surfaces as recurring hydration
            mismatches on whatever element it touches next - unfixable piecemeal
            (per-element suppressHydrationWarning is whack-a-mole). This lock
            tag asks the extension to leave the page alone entirely; the app
            ships its own dark themes, so there is nothing for it to add. */}
        <meta name="darkreader-lock" />
      </head>
      <body className="min-h-full flex flex-col">
        <BrandingProvider name={appName()}>
          <ThemeProvider>
            {/* Behind every page, under the shell: the aurora theme's sky.
                Renders nothing on any other theme. */}
            <Sky />
            {children}
          </ThemeProvider>
        </BrandingProvider>
      </body>
    </html>
  );
}
