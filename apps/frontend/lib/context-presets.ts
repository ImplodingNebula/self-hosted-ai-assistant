/**
 * The context-ceiling presets, and the select state a field value derives.
 *
 * One small module so the model layer can lift it whole later: the page knows
 * how to render these, never what the numbers are.
 */

/** One offered ceiling: the card class it was chosen for, and the tokens. */
export interface ContextPreset {
  /** The card class the number suits, e.g. "8 GB". */
  card: string;
  /** The ceiling in tokens. */
  tokens: number;
}

/**
 * Conservative starting points for common card sizes - NOT fit predictions.
 *
 * This platform measures what fits on the hardware in front of it and never
 * predicts it, so these are numbers worth trying first and nothing more. The
 * ceiling only ever lowers a window, so one that turns out too generous costs
 * a turn rather than a deployment.
 */
export const CONTEXT_PRESETS: readonly ContextPreset[] = [
  { card: "4 GB", tokens: 8192 },
  { card: "8 GB", tokens: 16384 },
  { card: "16 GB", tokens: 32768 },
  { card: "24 GB", tokens: 65536 },
];

/** The select's value for an empty field, which is the field's "no ceiling". */
export const PRESET_UNSET = "";

/** The select's value when nothing matches. A state label, never a choice. */
export const PRESET_CUSTOM = "custom";

/**
 * The one spelling the ceiling field accepts: plain decimal digits.
 *
 * Exported because the field has to agree with itself. `Number()` alone reads
 * "0x2000" as 8192 and "8.192e3" as 8192, so a looser rule on either side ends
 * with the select naming a preset over a number the save is about to refuse -
 * or refusing a number the select just offered. One regex, both readers.
 */
export const CEILING_DIGITS_ONLY = /^\d+$/;

/**
 * Which preset the ceiling field currently holds.
 *
 * Derived from the field rather than remembered beside it: a number typed by
 * hand that happens to equal a preset IS that preset, and a select carrying
 * its own memory would end up disagreeing with the input it describes.
 *
 * @param raw - The ceiling field's text, exactly as the form holds it.
 * @returns The matched preset's token count as a string, `PRESET_UNSET` for an
 *   empty field, or `PRESET_CUSTOM` for anything else.
 */
export function contextPresetState(raw: string): string {
  const text = raw.trim();
  if (!text) return PRESET_UNSET;
  if (!CEILING_DIGITS_ONLY.test(text)) return PRESET_CUSTOM;
  const tokens = Number(text);
  const hit = CONTEXT_PRESETS.find((preset) => preset.tokens === tokens);
  // The preset's own spelling, not the field's: "08192" is the 8,192 preset,
  // and the select can only show an option value it actually offers.
  return hit === undefined ? PRESET_CUSTOM : String(hit.tokens);
}
