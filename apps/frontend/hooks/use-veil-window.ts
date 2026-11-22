"use client";

import { type RefObject, useEffect } from "react";

/**
 * Tell the veil where the composer sits.
 *
 * Aurora's veil holds the sky back from the reading column, and the composer
 * is a pane straight onto the sky, so the veil carries a feathered window cut
 * exactly where the box is (themes/aurora.css reads the `--pane-*` variables
 * this writes). Measured from the pane's bottom edge, so nothing opening above
 * the composer moves the window. Other themes read none of it.
 *
 * What stands above the box moves it without resizing it: the greeting
 * arriving or growing, the thread taking the pane (on a phone the box is the
 * pane's width in a new chat and in a thread alike). A resize observer on the
 * box and the pane reports none of that, so the pane's children are observed
 * too, as they come and go - by their border box, since the composer's own
 * padding is one of the things that places it.
 */
export function useVeilWindow(
  pane: RefObject<HTMLElement | null>,
  box: RefObject<HTMLElement | null>,
): void {
  useEffect(() => {
    const paneEl = pane.current;
    const boxEl = box.current;
    if (!paneEl || !boxEl) return;
    const measure = () => {
      const m = paneEl.getBoundingClientRect();
      const b = boxEl.getBoundingClientRect();
      paneEl.style.setProperty("--pane-x", `${b.left - m.left}px`);
      paneEl.style.setProperty("--pane-b", `${m.bottom - b.bottom}px`);
      paneEl.style.setProperty("--pane-w", `${b.width}px`);
      paneEl.style.setProperty("--pane-h", `${b.height}px`);
    };
    measure();
    const ro = new ResizeObserver(measure);
    const watch = (el: Element) => ro.observe(el, { box: "border-box" });
    watch(paneEl);
    watch(boxEl);
    for (const el of paneEl.children) watch(el);
    const mo = new MutationObserver((records) => {
      for (const record of records) {
        for (const node of record.removedNodes) if (node instanceof Element) ro.unobserve(node);
        for (const node of record.addedNodes) if (node instanceof Element) watch(node);
      }
      measure();
    });
    mo.observe(paneEl, { childList: true });
    return () => {
      ro.disconnect();
      mo.disconnect();
    };
  }, [pane, box]);
}
