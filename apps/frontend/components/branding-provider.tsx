"use client";

import { createContext, useContext, type ReactNode } from "react";

const AppNameContext = createContext<string | null>(null);

/** Props for {@link BrandingProvider}. */
export interface BrandingProviderProps {
  /** The product name, read from the environment by a server parent. */
  name: string;
  children: ReactNode;
}

/**
 * Carries the deployment's product name into the client tree.
 *
 * A context rather than a prop because the one client surface that shows the
 * name (the sidebar's brand line) sits several levels inside a page that is
 * itself a client component - there is no server parent close enough to hand
 * it down. The root layout reads the environment and seeds this; nothing
 * below it touches `process.env`, which in a client bundle would read
 * `undefined` and silently answer with the default name.
 */
export function BrandingProvider({ name, children }: BrandingProviderProps) {
  return <AppNameContext.Provider value={name}>{children}</AppNameContext.Provider>;
}

/**
 * The deployment's product name. Must be used under a {@link BrandingProvider}.
 *
 * Throws rather than falling back to the default, because a missing provider
 * is a wiring mistake that would otherwise show every deployment the working
 * name and look, on screen, exactly like a deployment that set nothing.
 */
export function useAppName(): string {
  const name = useContext(AppNameContext);
  if (name === null) {
    throw new Error("useAppName must be used within a BrandingProvider");
  }
  return name;
}
