/**
 * The math half of the answer renderer's plugin chain, kept in one place so the
 * component and the tests that drive the real parser cannot drift apart.
 */
import type { Options } from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkBreaks from "remark-breaks";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import { answerMarkPlugin, type AnswerMarkOptions } from "@/lib/markdown";

/**
 * Single-dollar math stays OFF so currency ("$5 and $10") can never become a
 * formula; what remains is `$$...$$`, which `normalizeMathDelimiters` targets.
 */
export const REMARK_MATH = [remarkMath, { singleDollarTextMath: false }] as const;

/**
 * KaTeX never throws into the message: a bad formula renders as its source in
 * the surrounding text colour. `trust` stays at its default (off).
 */
export const REHYPE_KATEX = [
  rehypeKatex,
  {
    throwOnError: false,
    strict: "ignore",
    errorColor: "inherit",
    // A huge rule or runaway macro from a model must not size the page.
    maxSize: 20,
    maxExpand: 200,
  },
] as const;

/**
 * GitHub-flavoured markdown, plus single newlines as real line breaks, plus math.
 *
 * `remark-breaks` is what keeps loosely formatted output readable: a small model
 * writes lists and stanzas separated by ONE newline, which strict markdown joins
 * into a paragraph. Shared by the component and the tests so the chain that is
 * tested is the chain that ships.
 */
export const answerRemarkPlugins: NonNullable<Options["remarkPlugins"]> = [
  remarkGfm,
  remarkBreaks,
  [...REMARK_MATH],
];

/**
 * KaTeX runs first: the citation walk must see the finished formula markup
 * (which it skips), not the LaTeX source, whose `[1]` it would turn into a chip.
 */
export function answerRehypePlugins(
  options: AnswerMarkOptions,
): NonNullable<Options["rehypePlugins"]> {
  return [[...REHYPE_KATEX], answerMarkPlugin(options)];
}
