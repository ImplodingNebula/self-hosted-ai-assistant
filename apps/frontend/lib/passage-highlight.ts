/**
 * Lighting a quoted passage inside the message it came from.
 *
 * The light is painted by the CSS Custom Highlight API on the words
 * themselves, so the rendered answer is untouched: no wrapping, no re-render,
 * nothing for the markdown to re-layout. Highlight pseudo-elements take no
 * transition, so the light is eased by hand: its colour is rewritten frame by
 * frame on one stylesheet rule, in over a short rise, held, and out over a
 * longer fall, which keeps it under the same law as every other change of
 * brightness here. A browser without the API lights nothing (the turn's ring
 * still says where the reader landed).
 */

/** The highlight's registered name; the rule is `::highlight(quote-source)`. */
const NAME = "quote-source";

/**
 * The light's rise and fall, in ms. The rise is the ring's `--dur-slow`; the
 * fall is longer, as a fade out reads slower than a fade in of the same
 * length. Its colour and strength at the hold are the theme's
 * (`--c-passage-light`). Under reduced motion the glide and the ring are
 * cuts, and so is the light.
 */
const RISE_MS = 320;
const FALL_MS = 460;

/** One rendered character: the text node it lives in and its offset there. */
interface Slot {
  node: Text;
  offset: number;
}

/** Whether this browser can paint a highlight on a range. */
export function canLightPassage(): boolean {
  return (
    typeof CSS !== "undefined" && "highlights" in CSS && typeof Highlight !== "undefined"
  );
}

/** Ease in and out (a cubic), the follow glide's own curve. */
function easeInOut(p: number): number {
  return p < 0.5 ? 4 * p * p * p : 1 - Math.pow(-2 * p + 2, 3) / 2;
}

/**
 * Find a passage's words inside a container's rendered text, as a Range.
 *
 * The passage is matched against the container's text with whitespace
 * squashed the way the renderer squashes it, across element boundaries (a
 * quote can start inside one paragraph's bold run and end in the next), and
 * the match is mapped back to the text nodes it spans. Text inside a nested
 * quote band is skipped: a question's own quote is not the question. A
 * clamped quote ends in "..." - the marker, not the words - so a trailing one
 * is dropped before the search. Null when the words are not there: the
 * message was edited, or the quote was taken from the raw markdown of a
 * formatted answer, and then the turn's ring stands alone.
 */
export function findPassage(container: Element, passage: string): Range | null {
  const want = passage.replace(/\.\.\.$/, "").replace(/\s+/g, " ").trim();
  if (want === "") return null;
  const walker = document.createTreeWalker(container, NodeFilter.SHOW_TEXT, {
    acceptNode: (node) =>
      node.parentElement?.closest('[data-part="quote-chip"]')
        ? NodeFilter.FILTER_REJECT
        : NodeFilter.FILTER_ACCEPT,
  });
  let text = "";
  const slots: Slot[] = [];
  let afterSpace = true;
  for (let node = walker.nextNode(); node !== null; node = walker.nextNode()) {
    const data = (node as Text).data;
    for (let i = 0; i < data.length; i++) {
      const ch = data[i];
      if (ch === " " || ch === "\n" || ch === "\t" || ch === "\r") {
        if (afterSpace) continue;
        text += " ";
        afterSpace = true;
      } else {
        text += ch;
        afterSpace = false;
      }
      slots.push({ node: node as Text, offset: i });
    }
  }
  const at = text.indexOf(want);
  if (at < 0) return null;
  const first = slots[at];
  const last = slots[at + want.length - 1];
  const range = document.createRange();
  range.setStart(first.node, first.offset);
  range.setEnd(last.node, last.offset + 1);
  return range;
}

let rule: CSSStyleRule | null = null;
let frame: number | null = null;
let timer: number | null = null;

/**
 * The one rule the light is painted through, made on first use. Null where
 * a stylesheet cannot be made (a content security policy that refuses an
 * inline sheet), and then the range is registered but never coloured.
 */
function lightRule(): CSSStyleRule | null {
  if (rule !== null) return rule;
  try {
    const style = document.createElement("style");
    style.setAttribute("data-part", "passage-light");
    document.head.appendChild(style);
    const sheet = style.sheet;
    if (sheet === null) return null;
    const i = sheet.insertRule(
      `::highlight(${NAME}) { background-color: transparent; text-decoration: underline; text-decoration-color: transparent; text-decoration-thickness: 2px; text-underline-offset: 3px; }`,
      0,
    );
    rule = sheet.cssRules[i] as CSSStyleRule;
    return rule;
  } catch {
    return null;
  }
}

/**
 * The light's colour as the browser resolves it where `anchor` sits (the
 * theme's token, at its hold strength), for the mix. The probe goes into
 * React-owned DOM and comes straight back out in the same task, before
 * anything can observe it; nothing watches this subtree's children anyway.
 */
function lightColorAt(anchor: Element): string {
  const probe = document.createElement("span");
  probe.style.color = "var(--c-passage-light)";
  anchor.appendChild(probe);
  const color = getComputedStyle(probe).color;
  probe.remove();
  return color;
}

/** Take the light off the words, mid-fade or not. */
export function clearPassage(): void {
  if (frame !== null) cancelAnimationFrame(frame);
  frame = null;
  if (timer !== null) window.clearTimeout(timer);
  timer = null;
  if (canLightPassage()) CSS.highlights.delete(NAME);
  if (rule !== null) {
    rule.style.backgroundColor = "transparent";
    rule.style.textDecorationColor = "transparent";
  }
}

/**
 * Light a range for `ms`: rise, hold, fall, then nothing. One light at a
 * time; a new one takes the previous off first.
 *
 * @param range - The words to light.
 * @param anchor - An element in the lit message, for the accent's resolved value.
 * @param ms - The whole light's length, rise and fall included.
 */
export function lightPassage(range: Range, anchor: Element, ms: number): void {
  if (!canLightPassage()) return;
  clearPassage();
  CSS.highlights.set(NAME, new Highlight(range));
  const paint = lightRule();
  if (paint === null) return;
  const light = lightColorAt(anchor);
  /** Paint the light at a fraction of its hold strength. */
  const at = (a: number) => {
    const color = `color-mix(in srgb, ${light} ${Math.round(a * 100)}%, transparent)`;
    paint.style.backgroundColor = color;
    paint.style.textDecorationColor = color;
  };
  if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
    at(1);
    timer = window.setTimeout(() => {
      timer = null;
      clearPassage();
    }, ms);
    return;
  }
  const t0 = performance.now();
  const fallAt = Math.max(RISE_MS, ms - FALL_MS);
  const step = (now: number) => {
    const t = now - t0;
    let a: number;
    if (t < RISE_MS) a = easeInOut(t / RISE_MS);
    else if (t < fallAt) a = 1;
    else a = 1 - easeInOut(Math.min(1, (t - fallAt) / FALL_MS));
    at(a);
    if (t < ms) frame = requestAnimationFrame(step);
    else {
      frame = null;
      clearPassage();
    }
  };
  frame = requestAnimationFrame(step);
}
