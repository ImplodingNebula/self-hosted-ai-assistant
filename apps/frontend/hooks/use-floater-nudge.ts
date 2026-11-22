"use client";

import { useLayoutEffect, useState, type RefObject } from "react";

/** The composer box's inner inset a floater keeps clear of. */
const PAD = 12;

/**
 * How far a floater hinged on its button must slide sideways to stay inside
 * the composer's box. Zero where it fits as it hangs (the usual case); on a
 * narrow screen a button near an edge would otherwise hang its panel off the
 * screen. Measured in a layout effect on each opening, so the correction
 * lands before paint and never stacks on an earlier one; applied through the
 * `translate` property, which the entrance animation (on `transform`) does
 * not touch.
 *
 * @param ref - The floater, mounted while open.
 * @param open - Whether it is open.
 * @returns The sideways shift in px, positive to the right.
 */
export function useFloaterNudge(ref: RefObject<HTMLElement | null>, open: boolean): number {
  const [shift, setShift] = useState(0);
  useLayoutEffect(() => {
    const el = open ? ref.current : null;
    let next = 0;
    if (el) {
      const box = el.closest<HTMLElement>(".composer-box")?.getBoundingClientRect();
      const bounds = box ?? { left: 0, right: window.innerWidth };
      const r = el.getBoundingClientRect();
      if (r.right > bounds.right - PAD) next = bounds.right - PAD - r.right;
      if (r.left + next < bounds.left + PAD) next = bounds.left + PAD - r.left;
    }
    setShift((prev) => (prev === next ? prev : next));
  }, [ref, open]);
  return shift;
}
