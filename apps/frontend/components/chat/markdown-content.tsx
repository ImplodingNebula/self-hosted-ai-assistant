"use client";

import { Component, isValidElement, memo, useDeferredValue, useMemo, type ReactNode } from "react";
import Markdown, { type Components, type ExtraProps } from "react-markdown";
import { CitationMarker } from "@/components/chat/citation-popover";
import { CodeBlock } from "@/components/chat/code-block";
import { normalizeMathDelimiters } from "@/lib/math-delimiters";
import { answerRehypePlugins, answerRemarkPlugins } from "@/lib/math-markdown";

/** Schemes a model-written address may keep; see `sanitizeUrl`. */
const ALLOWED_SCHEME = /^(?:https?|mailto)$/i;

/**
 * Narrow the renderer's default URL surface to the schemes an answer needs.
 *
 * Scheme detection is the default transform's: a colon BEFORE the first `/`,
 * `?` or `#` starts a protocol, and a colon after any of them is just a colon
 * in a path or query. No colon at all means the address is relative or a bare
 * fragment, which is kept as written.
 *
 * The default admits `irc`, `ircs` and `xmpp` alongside http/https/mailto.
 * Those hand the address to an OS protocol handler - an app launch driven by
 * text a model copied off a web page - and nothing in an answer needs them, so
 * they are dropped here. Anything rejected comes back empty, which the anchor
 * below renders as inert text.
 */
function sanitizeUrl(value: string): string {
  const colon = value.indexOf(":");
  const slash = value.indexOf("/");
  const questionMark = value.indexOf("?");
  const numberSign = value.indexOf("#");
  if (
    colon === -1 ||
    (slash !== -1 && colon > slash) ||
    (questionMark !== -1 && colon > questionMark) ||
    (numberSign !== -1 && colon > numberSign)
  ) {
    return value;
  }
  return ALLOWED_SCHEME.test(value.slice(0, colon)) ? value : "";
}

/**
 * An anchor's leftover props with the hast `node` taken out.
 *
 * The renderer hands each override the parsed node alongside the real
 * attributes, and spreading that onto a DOM element would put a whole subtree
 * in an attribute. Everything else is forwarded untouched, which is the point:
 * a footnote link carries `id`, `data-footnote-ref` and `aria-describedby`, and
 * an anchor rebuilt from `href` alone is a footnote that cannot be jumped back
 * from or announced.
 */
function anchorAttributes(
  rest: React.ComponentPropsWithoutRef<"a"> & ExtraProps,
): React.ComponentPropsWithoutRef<"a"> {
  const attributes = { ...rest };
  delete attributes.node;
  return attributes;
}

/**
 * How the tags a model can emit are rendered.
 *
 * The entries here are safety, not styling. Raw HTML never reaches this map as
 * markup: the renderer hands remark-rehype `allowDangerousHtml` and carries the
 * raw nodes through the tree, then turns every one of them into a plain TEXT
 * node on the way to JSX - so `<img onerror=...>` in an answer renders as those
 * characters, escaped and inert. That guarantee is what `rehype-raw` (which
 * parses those nodes into real elements) and `skipHtml` (which deletes them,
 * hiding what the model wrote) would each undo: NEVER add either here.
 * Addresses arrive already narrowed by `sanitizeUrl`.
 *
 * Typed as `Components` so a mistyped key is a build error rather than an
 * override that silently never runs - `img` in particular is a security
 * control, and `image` would drop it without a word.
 */
const COMPONENTS: Components = {
  /**
   * The only `span` elements this renderer ever produces are the two
   * `lib/markdown.ts` injects directly into the parsed tree (raw HTML never
   * survives parsing - see the note above): the streaming caret, and a
   * citation marker. Markdown syntax itself has no way to write a `<span>`, so
   * this override never intercepts anything the model actually typed.
   *
   * A citation marker carries its source indexes on the hast node itself
   * (`data-citation-indexes`, set by `citationElement`); `node` is read
   * directly rather than re-parsed from the rendered text, which is what
   * `passNode` on the underlying renderer is for.
   *
   * The caret falls through to a plain span with its properties forwarded
   * untouched (`node` excluded by the destructure below, the same trick
   * `anchorAttributes` uses) - it already carries everything it needs
   * (`aria-hidden`) off the hast tree, and this override must not quietly
   * drop it now that it sits in the path.
   */
  span({ node, className, children, ...rest }) {
    const classes = typeof className === "string" ? className.split(" ") : [];
    if (classes.includes("cite-mark")) {
      const raw = node?.properties?.["data-citation-indexes"];
      const indexes = Array.isArray(raw) ? raw.map(String) : [];
      // The chip shows the number alone; the brackets are the source's
      // spelling, kept in the stored text and in the marker's own name.
      const label = (typeof children === "string" ? children : String(children ?? ""))
        .replace(/^\[|\]$/g, "");
      return <CitationMarker indexes={indexes} label={label} />;
    }
    return (
      <span className={className} {...rest}>
        {children}
      </span>
    );
  },
  /**
   * Where a model-written link opens, decided by where it points.
   *
   * An address the sanitizer rejected arrives empty and renders as plain text:
   * an empty `href` points at the page the reader is on, so clicking it would
   * throw the conversation away to reload it. A fragment or same-app path is
   * navigation WITHIN this app - footnote markers (which this renderer emits)
   * and in-page anchors - and a new tab there would boot a second copy of the
   * app rather than move the reader down the page. A `//` address is NOT one of
   * those: scheme-relative means another HOST, so it takes the off-site branch
   * like any absolute address. Everything else is off-site and leaves the
   * conversation intact behind it.
   *
   * The rest of the props ride along untouched - see `anchorAttributes`.
   */
  a({ href, children, ...rest }) {
    if (href === undefined || href === "") return <span>{children}</span>;
    const attributes = anchorAttributes(rest);
    if (href.startsWith("#") || (href.startsWith("/") && !href.startsWith("//"))) {
      return (
        <a href={href} {...attributes}>
          {children}
        </a>
      );
    }
    // Spread first: target and rel are the safety half of this branch, and
    // written last they win over anything that arrives in the attributes.
    return (
      <a href={href} {...attributes} target="_blank" rel="noopener noreferrer">
        {children}
      </a>
    );
  },
  /**
   * An image is NEVER fetched, and never clickable either.
   *
   * The address in a model's answer can come from a web page it just read, and
   * loading it would have the reader's browser call out to whatever host that
   * page named, the moment the answer appears - a tracking pixel, or worse,
   * that nobody asked for. Rendering it as a LINK had its own problem:
   * `[![alt](img)](link)` nests an anchor inside an anchor, which is invalid
   * HTML the browser untangles by letting the inner (attacker-supplied) address
   * capture the click. So it is an inert span, labelled with its alt text, with
   * the sanitized source readable in the tooltip.
   */
  img({ src, alt }) {
    // React's own img typing admits a Blob (for a `srcObject`-style source);
    // a parsed document only ever carries the sanitized string - which is EMPTY
    // for a rejected address, and an empty tooltip is no tooltip.
    const source = typeof src === "string" && src !== "" ? src : undefined;
    return (
      <span className="md-image" title={source}>
        {alt || "image"}
      </span>
    );
  },
  /**
   * A fenced block is a component: a head with its language and a copy
   * button, then the code with syntax colour (see CodeBlock). The parser hands
   * over a single <code> child whose class names the fence's language; its
   * text is lifted out for the tokenizer when it is nothing but text, which a
   * streaming block (the caret sits inside it) is not.
   */
  pre({ children }) {
    const code = isValidElement<{ className?: string; children?: ReactNode }>(children)
      ? children
      : null;
    const className = code?.props.className ?? "";
    const match = /language-([\w+#.-]+)/.exec(className);
    const inner = code?.props.children;
    const text =
      typeof inner === "string"
        ? inner
        : Array.isArray(inner) && inner.every((part) => typeof part === "string")
          ? inner.join("")
          : null;
    return (
      <CodeBlock language={match ? match[1] : null} code={text === null ? null : text.replace(/\n$/, "")}>
        {inner}
      </CodeBlock>
    );
  },
  /** Tables scroll on their own rather than widening the bubble past the pane. */
  table({ children }) {
    return (
      <div className="md-table-wrap">
        <table>{children}</table>
      </div>
    );
  },
  /**
   * Task-list checkboxes are drawn by hand, not by the browser.
   *
   * The native control would arrive disabled (it reports what the answer said,
   * it is not a control), and every browser renders a disabled checkbox washed
   * out - unreadable on the dark themes. A span the theme's tokens style keeps
   * the mark legible everywhere, and the image role tells a screen reader what
   * the glyph says instead of announcing a form field nobody can operate.
   */
  input({ type, checked }) {
    // GFM emits no input other than the task-list checkbox.
    if (type !== "checkbox") return null;
    return (
      <span
        className="md-check"
        data-checked={checked ? "true" : undefined}
        role="img"
        aria-label={checked ? "done" : "not done"}
      />
    );
  },
};

/** Longest answer worth parsing; see `isPathological`. */
const MAX_PARSED_LENGTH = 65536;

/**
 * Any single character repeated 500 or more times in a row.
 *
 * Spelled `[\s\S]` rather than `.` with the `s` flag so it compiles under this
 * app's ES2017 target; the two match the same thing.
 */
const CHARACTER_RUN = /([\s\S])\1{499,}/;

/**
 * The blockquote bomb in its spaced spelling: `> > > ...` nests exactly like a
 * solid run of `>` but never repeats one character, so the run rule misses it.
 * Two hundred consecutive quote markers is depth no written answer reaches.
 */
const BLOCKQUOTE_RUN = /(?:>[ \t]*){200,}/;

/**
 * True for text the markdown parser should not be handed at all.
 *
 * A model can echo a page it just read, and a page can carry a markdown bomb:
 * a few thousand `>` (nested blockquotes, solid or spaced) or twenty thousand
 * `*` around one character (nested emphasis) blow the stack inside the
 * mdast-to-hast walk. These rules cover the measured shapes cheaply - not to
 * start is better than to catch - and the error boundary below backstops
 * whatever shape has not been measured yet.
 */
function isPathological(content: string): boolean {
  return (
    content.length > MAX_PARSED_LENGTH ||
    CHARACTER_RUN.test(content) ||
    BLOCKQUOTE_RUN.test(content)
  );
}

/**
 * The answer exactly as it was shown before markdown: honest, and legible.
 *
 * Used whenever the document cannot be rendered - it says nothing the model did
 * not write, and the caret keeps sitting where the next token will land.
 */
function PlainAnswer({ content, live }: { content: string; live: boolean }) {
  return (
    <div className="md-fallback">
      {content}
      {live && <span className="caret" aria-hidden="true" />}
    </div>
  );
}

interface BoundaryProps {
  /** The text being rendered, so a throw can be pinned to the string that caused it. */
  content: string;
  live: boolean;
  children: ReactNode;
}

interface BoundaryState {
  /** A render below has thrown and has not been recovered from. */
  caught: boolean;
  /** The content that threw, once it has been read off the props. */
  failedOn: string | null;
}

/**
 * Keeps a document the parser cannot handle from taking the page down.
 *
 * There is no other error boundary above this: a throw inside the markdown walk
 * unmounts the whole chat page, so one unlucky answer costs the reader the
 * conversation they were having. The guard above catches the shapes that have
 * actually been measured; this catches the ones that have not been.
 *
 * Recovery is deliberately asymmetric, because the throwing text usually keeps
 * ARRIVING. While the content still starts with the string that failed, the
 * stream is the same answer growing past its bomb, and re-parsing it would
 * throw again on every delta - so the boundary stays down without re-trying.
 * When the content no longer starts with it, this instance is showing a
 * different message (a regenerate reusing the same slot), which has earned a
 * fresh attempt.
 */
class MarkdownErrorBoundary extends Component<BoundaryProps, BoundaryState> {
  state: BoundaryState = { caught: false, failedOn: null };

  static getDerivedStateFromError(): Partial<BoundaryState> {
    // The error carries no props, so the text that threw is read on the render
    // this schedules - see `getDerivedStateFromProps`.
    return { caught: true };
  }

  static getDerivedStateFromProps(
    props: BoundaryProps,
    state: BoundaryState,
  ): Partial<BoundaryState> | null {
    if (!state.caught) return null;
    if (state.failedOn === null) return { failedOn: props.content };
    if (!props.content.startsWith(state.failedOn)) {
      return { caught: false, failedOn: null };
    }
    return null;
  }

  render(): ReactNode {
    if (this.state.caught) {
      return <PlainAnswer content={this.props.content} live={this.props.live} />;
    }
    return this.props.children;
  }
}

/**
 * One assistant answer, rendered as markdown.
 *
 * Assistant turns only: a question is shown as the user typed it, and the
 * reasoning pane and tool narration stay plain text - markdown there would style
 * the model's working as if it were the answer.
 *
 * Every prop is a primitive so the `memo` below can do its job: one delta
 * re-renders the whole transcript, and without it every settled bubble on
 * screen re-parses its markdown on every token of the one being written.
 */
function AnswerMarkdown({
  content,
  live,
  markers,
  idKey,
}: {
  /** The answer text, as streamed or as persisted. */
  content: string;
  /** True while this message is still receiving deltas; draws the caret. */
  live: boolean;
  /**
   * The `[n]` markers this message has a source for, space separated. Any other
   * number in the text stays prose - see `answerMarkPlugin`.
   */
  markers: string;
  /** Identifies this message among those on screen; namespaces its footnotes. */
  idKey: string;
}) {
  const citationIndexes = useMemo(
    () => new Set(markers === "" ? [] : markers.split(" ")),
    [markers],
  );
  // Memoized for a stable identity, not for a saving: the renderer rebuilds its
  // processor on every render regardless. The real guard against re-parsing is
  // the `memo` around this component.
  const rehypePlugins = useMemo(
    () => answerRehypePlugins({ citationIndexes, live }),
    [citationIndexes, live],
  );
  // Footnote ids are global to the document, and every message that has one
  // would otherwise mint the same `user-content-fn-1`: the second footnote's
  // link jumps to the first message's note. Namespacing by message keeps them
  // apart. The section's "Footnotes" LABEL id is hardcoded upstream
  // ('footnote-label', outside the prefix) and still repeats across messages;
  // it names the heading and is never a link target, so that one is cosmetic.
  const remarkRehypeOptions = useMemo(
    () => ({ clobberPrefix: `md-${idKey}-` }),
    [idKey],
  );
  // The parse is the expensive half of a delta, and it is not what the reader is
  // waiting on. Deferring it lets React paint the rest of the turn first under a
  // fast stream; the caret can lag a token behind at full speed, deliberately.
  const deferred = useDeferredValue(content);
  // Rewritten only when the text will actually be parsed: the pass is linear,
  // but a document the guard below refuses should cost nothing.
  const parsed = useMemo(
    () => (isPathological(deferred) ? deferred : normalizeMathDelimiters(deferred)),
    [deferred],
  );

  if (isPathological(deferred)) {
    return <PlainAnswer content={deferred} live={live} />;
  }

  return (
    <MarkdownErrorBoundary content={deferred} live={live}>
      <Markdown
        remarkPlugins={answerRemarkPlugins}
        rehypePlugins={rehypePlugins}
        remarkRehypeOptions={remarkRehypeOptions}
        components={COMPONENTS}
        urlTransform={sanitizeUrl}
      >
        {parsed}
      </Markdown>
    </MarkdownErrorBoundary>
  );
}

export const MarkdownContent = memo(AnswerMarkdown);
