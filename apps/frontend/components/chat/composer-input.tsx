"use client";

import { CheckListPlugin } from "@lexical/react/LexicalCheckListPlugin";
import { LexicalComposer } from "@lexical/react/LexicalComposer";
import { useLexicalComposerContext } from "@lexical/react/LexicalComposerContext";
import { ContentEditable } from "@lexical/react/LexicalContentEditable";
import { EditorRefPlugin } from "@lexical/react/LexicalEditorRefPlugin";
import { LexicalErrorBoundary } from "@lexical/react/LexicalErrorBoundary";
import { HistoryPlugin } from "@lexical/react/LexicalHistoryPlugin";
import { ListPlugin } from "@lexical/react/LexicalListPlugin";
import { RichTextPlugin } from "@lexical/react/LexicalRichTextPlugin";
import { mergeRegister } from "@lexical/utils";
import {
  $getRoot,
  $getSelection,
  $isRangeSelection,
  $setSelection,
  type LexicalEditor,
} from "lexical";
import {
  forwardRef,
  useCallback,
  useEffect,
  useImperativeHandle,
  useRef,
} from "react";

import {
  type ComposerBehaviourHandlers,
  COMPOSER_NODES,
  COMPOSER_THEME,
  registerCheckGutter,
  registerComposerBehaviour,
} from "@/lib/composer-editor";
import { $buildComposerNodes } from "@/lib/composer-markdown";

/**
 * What the page can ask of the composer.
 *
 * The page still owns the draft as a STRING - send enablement, the failure
 * restore and the quote paths all read and write text - so everything here
 * either moves the caret or swaps the whole draft. There is deliberately no
 * "is it empty" question: the page already answers it from the draft it holds,
 * and a second implementation of that rule is a second rule.
 */
export interface ComposerInputHandle {
  /** Put the cursor in the box. Ignored while disabled. */
  focus(): void;
  /** Replace the whole draft, reading its markers back as glyphs. */
  setMarkdown(text: string): void;
  /** Empty the draft. */
  clear(): void;
  /** Type `text` at the cursor, as though the user had. */
  insertText(text: string): void;
}

/** How the page dresses and wires the composer. */
export interface ComposerInputProps extends ComposerBehaviourHandlers {
  /** Blocks typing; the box dims and refuses focus. */
  disabled?: boolean;
  /** Shown while the box is empty, and announced to assistive tech. */
  placeholder: string;
  /** The box's accessible name. */
  ariaLabel: string;
}

/** Hooks the composer's own keystrokes, paste and draft reporting onto the editor. */
function ComposerBehaviourPlugin({
  onChangeMarkdown,
  onPasteIntercept,
  onSend,
}: ComposerBehaviourHandlers): null {
  const [editor] = useLexicalComposerContext();
  // Read through a ref so re-registering never depends on a caller's identity:
  // an Enter handler dropped and rebuilt between renders is a message the user
  // watched vanish. Refreshed in its own effect, declared FIRST so the newest
  // callbacks are in place before anything below can reach for them.
  const handlers = useRef<ComposerBehaviourHandlers>({
    onChangeMarkdown,
    onPasteIntercept,
    onSend,
  });
  useEffect(() => {
    handlers.current = { onChangeMarkdown, onPasteIntercept, onSend };
  });

  useEffect(
    () =>
      mergeRegister(
        registerComposerBehaviour(editor, () => handlers.current),
        // The half of the composer's behaviour that needs a real root element,
        // which is why it is its own registration rather than part of the one
        // above - see its own comment for what it widens and why.
        registerCheckGutter(editor),
      ),
    [editor],
  );

  return null;
}

/**
 * The chat composer's input: a rendered editor that sends plain markdown.
 *
 * List bullets, ordinals, task boxes and code blocks appear as glyphs while
 * they are typed; everything else stays literal. What leaves the box is the
 * string a plain text area would have held, so nothing downstream - send
 * enablement, the failure restore, the wire - can tell the difference.
 */
export const ComposerInput = forwardRef<ComposerInputHandle, ComposerInputProps>(
  function ComposerInput(
    { ariaLabel, disabled = false, onChangeMarkdown, onPasteIntercept, onSend, placeholder },
    ref,
  ) {
    const editorRef = useRef<LexicalEditor | null>(null);

    /**
     * Swap the whole draft for `text`, cursor at the end.
     *
     * Only when the box ALREADY had the cursor. Every send and every failed-send
     * restore comes through here, and moving the caret in an unfocused editor
     * makes the browser focus it - which would pop the on-screen keyboard on a
     * touch device the page deliberately refuses to focus, and steal the cursor
     * out of whatever the user had moved on to.
     */
    const setMarkdown = useCallback((text: string) => {
      const editor = editorRef.current;
      if (editor === null) return;
      const hadFocus = editor.getRootElement() === document.activeElement;
      editor.update(() => {
        const root = $getRoot();
        root.clear();
        root.append(...$buildComposerNodes(text));
        if (hadFocus) root.selectEnd();
        // No selection at all, so the reconciler has none to write to the DOM.
        else $setSelection(null);
      });
    }, []);

    useImperativeHandle(
      ref,
      () => ({
        clear: () => setMarkdown(""),
        focus: () => {
          const editor = editorRef.current;
          if (editor === null || !editor.isEditable()) return;
          // A caret already parked mid-draft is restored; a first focus lands at
          // the end, which is where a draft is continued from.
          editor.focus(undefined, { defaultSelection: "rootEnd" });
        },
        insertText: (text: string) => {
          const editor = editorRef.current;
          if (editor === null || !editor.isEditable()) return;
          editor.update(() => {
            const selection = $getSelection();
            if ($isRangeSelection(selection)) selection.insertText(text);
          });
        },
        setMarkdown,
      }),
      [setMarkdown],
    );

    useEffect(() => {
      editorRef.current?.setEditable(!disabled);
    }, [disabled]);

    return (
      <LexicalComposer
        initialConfig={{
          editable: !disabled,
          namespace: "composer",
          nodes: COMPOSER_NODES,
          // A broken editor must not take the chat page with it: the transcript
          // above it is the thing the reader came for.
          onError: (error: Error) => {
            console.error("Composer editor error", error);
          },
          theme: COMPOSER_THEME,
        }}
      >
        <EditorRefPlugin editorRef={editorRef} />
        <RichTextPlugin
          contentEditable={
            <ContentEditable
              aria-label={ariaLabel}
              aria-multiline
              aria-placeholder={placeholder}
              className="composer-input"
              data-part="composer-input"
              placeholder={<div className="composer-placeholder">{placeholder}</div>}
            />
          }
          ErrorBoundary={LexicalErrorBoundary}
        />
        <ListPlugin />
        <CheckListPlugin />
        <HistoryPlugin />
        <ComposerBehaviourPlugin
          onChangeMarkdown={onChangeMarkdown}
          onPasteIntercept={onPasteIntercept}
          onSend={onSend}
        />
      </LexicalComposer>
    );
  },
);
