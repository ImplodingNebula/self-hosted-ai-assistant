/**
 * A small, dependency-free tokenizer for syntax colour in answer code blocks.
 *
 * Five classes - comment, string, number, keyword, function - on the shared
 * `--c-code-*` tokens, which is what the dossier asks for and all a reader
 * scanning a snippet needs. It is not a parser: it walks the text once with a
 * handful of rules that hold across the C family, Python, shell, SQL, JSON and
 * YAML, and leaves anything it is unsure of as plain text. Wrong colour is
 * worse than no colour, so every rule errs toward plain: a `--flag` is not a
 * comment, `#fff` is not a comment, `http://` is not a comment, and a word
 * that is a keyword in one language but an ordinary name in most is not in
 * the list.
 *
 * Pure and deterministic, so the battery can pin it.
 */

/** One run of the source, with the class it wears or null for plain text. */
export interface CodeToken {
  kind: "comment" | "string" | "number" | "keyword" | "function" | null;
  text: string;
}

/**
 * Words that read as keywords wherever they appear. Deliberately without the
 * words that are also everyday identifiers (`type`, `key`, `set`, `source`,
 * `match`, `select`...), which would colour a variable or a call.
 */
const KEYWORDS = new Set([
  // the C family, JavaScript and TypeScript
  "if", "else", "for", "while", "do", "switch", "case", "default", "break", "continue",
  "return", "function", "const", "let", "var", "new", "typeof", "instanceof",
  "class", "extends", "implements", "interface", "enum", "import", "export",
  "from", "as", "async", "await", "try", "catch", "finally", "throw", "this", "super",
  "static", "public", "private", "protected", "readonly", "void", "null", "undefined",
  "true", "false", "yield", "in", "of",
  // Python
  "def", "lambda", "pass", "with", "elif", "except", "raise", "global", "nonlocal",
  "and", "or", "not", "is", "None", "True", "False", "self",
  // Go, Rust, Java, C#, Swift, Kotlin
  "func", "defer", "chan", "package", "struct", "fn", "mut", "impl", "pub", "mod",
  "loop", "trait", "where", "final", "abstract", "override", "namespace", "using",
  // shell
  "then", "fi", "done", "esac",
]);

/** SQL's words, matched only in upper case, where they are unmistakable. */
const SQL_WORDS = new Set([
  "select", "insert", "update", "delete", "where", "join", "left", "right", "inner",
  "outer", "on", "group", "by", "order", "limit", "create", "table", "drop", "alter",
  "into", "values", "set", "begin", "commit", "rollback", "primary", "key", "references",
  "from", "as", "and", "or", "not", "in", "is", "null", "default",
]);

/**
 * Keywords that are followed by `(` in ordinary code (`if (`, `while (`,
 * `return (`). Any other keyword directly before `(` is a call after all.
 */
const CONTROL_CALLERS = new Set([
  "if", "for", "while", "switch", "catch", "return", "typeof", "await", "function",
  "elif", "except", "with", "yield", "throw", "new", "in", "not", "and", "or",
]);

const IDENT_START = /[A-Za-z_$]/;
const IDENT = /[A-Za-z0-9_$]/;
const DIGIT = /[0-9]/;
const SPACE = /[ \t]/;

/**
 * Tokenize one code block.
 *
 * Comments: `//` not preceded by `:` (a URL), `#` at a line's start or after
 * whitespace and followed by whitespace or the line's end (not `#fff`, not
 * `#!`), `--` under the same rule (not `--flag`), and `/* ... *\/` blocks.
 * Strings: single, double and backtick quoted, with backslash escapes, ending
 * at the line's end if unterminated. Numbers: a digit run with an optional
 * fraction, exponent or hex prefix, standing alone. Keywords: the shared
 * list, matched whole; the SQL words in upper case only. Functions: an
 * identifier directly followed by `(`, unless it is a control keyword.
 *
 * Never throws, and concatenating the tokens' `text` restores the source.
 *
 * @param source - The block's text, as written.
 * @returns The text split into runs.
 */
export function tokenizeCode(source: string): CodeToken[] {
  const out: CodeToken[] = [];
  let plain = "";
  const flush = () => {
    if (plain !== "") {
      out.push({ kind: null, text: plain });
      plain = "";
    }
  };
  const push = (kind: CodeToken["kind"], text: string) => {
    flush();
    out.push({ kind, text });
  };
  const n = source.length;
  let i = 0;
  while (i < n) {
    const ch = source[i];
    const next = i + 1 < n ? source[i + 1] : "";
    const prev = i > 0 ? source[i - 1] : "\n";
    const atWordStart = !IDENT.test(prev);
    const afterSpace = prev === "\n" || SPACE.test(prev);
    // block comment
    if (ch === "/" && next === "*") {
      const end = source.indexOf("*/", i + 2);
      const stop = end === -1 ? n : end + 2;
      push("comment", source.slice(i, stop));
      i = stop;
      continue;
    }
    // line comments: // (not inside a URL); # and -- only as a word of
    // their own, where a flag or a colour could not be
    const after2 = i + 2 < n ? source[i + 2] : "\n";
    const standsAlone = after2 === "\n" || SPACE.test(after2);
    const lineComment =
      (ch === "/" && next === "/" && prev !== ":") ||
      (ch === "#" && afterSpace && (next === "" || next === "\n" || SPACE.test(next))) ||
      (ch === "-" && next === "-" && afterSpace && standsAlone);
    if (lineComment) {
      let end = source.indexOf("\n", i);
      if (end === -1) end = n;
      push("comment", source.slice(i, end));
      i = end;
      continue;
    }
    // strings
    if (ch === '"' || ch === "'" || ch === "`") {
      let j = i + 1;
      while (j < n && source[j] !== ch && source[j] !== "\n") {
        if (source[j] === "\\") j += 1;
        j += 1;
      }
      const stop = j < n && source[j] === ch ? j + 1 : Math.min(j, n);
      push("string", source.slice(i, stop));
      i = stop;
      continue;
    }
    // numbers: a run that starts where a word could
    if (DIGIT.test(ch) && atWordStart) {
      let j = i + 1;
      if (ch === "0" && (next === "x" || next === "X")) {
        j = i + 2;
        while (j < n && /[0-9a-fA-F_]/.test(source[j])) j += 1;
      } else {
        while (j < n && /[0-9_]/.test(source[j])) j += 1;
        if (source[j] === "." && DIGIT.test(source[j + 1] ?? "")) {
          j += 1;
          while (j < n && DIGIT.test(source[j])) j += 1;
        }
        if ((source[j] === "e" || source[j] === "E") && /[0-9+-]/.test(source[j + 1] ?? "")) {
          j += 2;
          while (j < n && DIGIT.test(source[j])) j += 1;
        }
      }
      // a digit run glued to letters (a hash, an id) is not a number
      if (j < n && IDENT_START.test(source[j])) {
        while (j < n && IDENT.test(source[j])) j += 1;
        plain += source.slice(i, j);
        i = j;
        continue;
      }
      push("number", source.slice(i, j));
      i = j;
      continue;
    }
    // identifiers: keyword, function call, or plain
    if (IDENT_START.test(ch) && atWordStart) {
      let j = i + 1;
      while (j < n && IDENT.test(source[j])) j += 1;
      const word = source.slice(i, j);
      const lower = word.toLowerCase();
      const isKeyword =
        KEYWORDS.has(word) ||
        (SQL_WORDS.has(lower) && word === word.toUpperCase() && word.length > 1);
      const called = source[j] === "(";
      if (called && !(isKeyword && CONTROL_CALLERS.has(lower))) push("function", word);
      else if (isKeyword) push("keyword", word);
      else plain += word;
      i = j;
      continue;
    }
    plain += ch;
    i += 1;
  }
  flush();
  return out;
}
