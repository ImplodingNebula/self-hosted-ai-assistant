"use client";

import { type ReactNode, useMemo } from "react";

import { CopyButton } from "@/components/chat/copy-button";
import { tokenizeCode } from "@/lib/code-tokens";

const TOKEN_CLASS = {
  comment: "tk-c",
  string: "tk-s",
  number: "tk-n",
  keyword: "tk-k",
  function: "tk-f",
} as const;

/**
 * A fenced code block as a component: a head with the language label and a
 * copy button, then the code with syntax colour on the shared `--c-code-*`
 * tokens.
 *
 * The colour is applied only when the whole block is plain text. While an
 * answer streams, the markdown pipeline can place the caret inside the block;
 * then the children are rendered as they came, uncoloured, and colour arrives
 * when the block settles.
 */
export function CodeBlock({
  language,
  code,
  children,
}: {
  /** The fence's language tag, or null when it had none. */
  language: string | null;
  /** The block's text, when the children are nothing but text. */
  code: string | null;
  children?: ReactNode;
}) {
  const tokens = useMemo(() => (code === null ? null : tokenizeCode(code)), [code]);
  return (
    <div data-part="code-block">
      <div className="code-head">
        <span className="code-lang">{language ?? "text"}</span>
        <CopyButton text={code ?? ""} label="Copy code" />
      </div>
      <pre>
        <code>
          {tokens === null
            ? children
            : tokens.map((token, i) =>
                token.kind === null ? (
                  token.text
                ) : (
                  <span key={i} className={TOKEN_CLASS[token.kind]}>
                    {token.text}
                  </span>
                ),
              )}
        </code>
      </pre>
    </div>
  );
}
