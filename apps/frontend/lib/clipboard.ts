/**
 * Clipboard writing, shared by every surface that offers a Copy.
 *
 * One implementation rather than one per component, so the fallback story stays
 * identical wherever the button lives: the action row under a message and the
 * selection pill over one must not succeed on different origins.
 */

/**
 * Put `text` on the clipboard, falling back to the legacy command.
 *
 * The async Clipboard API is the right call and the only one that needs no DOM
 * at all - but it is unavailable outside a secure context, which includes the
 * plain-http origin this app is reached on in local development and on any
 * deployment that has not fronted itself with TLS yet. Copy is too small a
 * feature to be broken there, so the old `execCommand` route (an off-screen
 * textarea, selected and copied) stands behind it. Deprecated, still universally
 * implemented, and only ever reached once the modern path has actually failed.
 *
 * @returns Whether the text reached the clipboard, so the caller only claims
 *   success when there was some.
 */
export async function writeToClipboard(text: string): Promise<boolean> {
  try {
    if (navigator.clipboard?.writeText) {
      await navigator.clipboard.writeText(text);
      return true;
    }
  } catch {
    // Denied permission, or an insecure origin: fall through and try the
    // command below rather than reporting a failure the fallback can fix.
  }
  const area = document.createElement("textarea");
  area.value = text;
  area.setAttribute("readonly", "");
  // Off-screen but focusable: `display: none` cannot be selected, and a visible
  // one would flash a text box over the transcript for a frame.
  area.style.position = "fixed";
  area.style.top = "0";
  area.style.left = "-9999px";
  document.body.appendChild(area);
  try {
    area.select();
    return document.execCommand("copy");
  } catch {
    return false;
  } finally {
    area.remove();
  }
}
