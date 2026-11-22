// The working name, used until a deployment chooses its own. This is the one
// place it is written down; the compose file deliberately does not repeat it.
const DEFAULT_APP_NAME = "AI Chatbot Platform";

/**
 * The name this deployment goes by, read from the server environment.
 *
 * SERVER ONLY. `APP_NAME` is deliberately not a `NEXT_PUBLIC_` variable:
 * those are substituted into the bundle when it is BUILT, which would make
 * the name part of the shipped artifact rather than part of its
 * configuration - one image per customer. Read here, one image serves every
 * deployment and the name is a line in an env file. The cost is that a client
 * component calling this reads `undefined` and quietly gets the default, so
 * client code takes the value as a prop or through the branding provider
 * instead.
 *
 * Read per call rather than captured at module load, so nothing depends on
 * when this module was first imported.
 *
 * Blank counts as unset: an `APP_NAME=` line left empty in an env file is a
 * deployment that never chose a name, not one that chose to have none.
 *
 * The backend reads the same variable for its API title, so one setting names
 * the whole deployment.
 *
 * @returns The configured product name, or the working name if none is set.
 */
export function appName(): string {
  return process.env.APP_NAME?.trim() || DEFAULT_APP_NAME;
}
