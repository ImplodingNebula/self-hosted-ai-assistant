/**
 * Minimal Server-Sent Events parser for `fetch` response streams.
 *
 * The browser's native `EventSource` only does GET; our chat endpoint is a POST,
 * so we read the streamed body ourselves. This turns a `text/event-stream`
 * {@link ReadableStream} into an async iterator of `{ event, data }` messages,
 * decoding incrementally so tokens surface the moment they arrive.
 */

/**
 * Response header marking a stream that carries a failure INSTEAD of a turn.
 *
 * Part of the contract between the chat proxy and the composer, which is why it
 * lives here rather than in either of them. Surfacing transport failures in-band
 * (one parsing path for success and failure alike) costs the client the one
 * distinction it cannot afford to lose: a backend that never took the turn -
 * nothing persisted, so the composer should hand the question and its files back
 * for another go - produces the same 200 and the same error frame as one that
 * took the turn and died mid-answer, where the user message IS on record and
 * rolling back would erase a turn the transcript actually has. The proxy sets
 * this to "0" on the former, and only on the former.
 */
export const TURN_ACCEPTED_HEADER = "x-turn-accepted";

/** A single parsed SSE message (one blank-line-separated event block). */
export interface SSEMessage {
  /** The `event:` field, or `"message"` when omitted (per the SSE spec). */
  event: string;
  /** The concatenated `data:` field(s), joined with newlines. */
  data: string;
}

// A blank line ends an event block; the SSE spec allows CRLF, LF, or CR line
// endings, so the separator is any of "\r\n\r\n", "\n\n", or "\r\r".
const BLOCK_SEPARATOR = /\r\n\r\n|\n\n|\r\r/;

/**
 * Parse an SSE stream into discrete messages.
 *
 * Event blocks end at a blank line. Because the SSE spec permits CRLF, LF, or
 * CR line endings, blocks are split on the earliest occurrence of
 * `\r\n\r\n`, `\n\n`, or `\r\r` in the buffer.
 *
 * @param stream - The `text/event-stream` body from a `fetch` response.
 * @yields One {@link SSEMessage} per event block, in arrival order.
 */
export async function* parseSSE(
  stream: ReadableStream<Uint8Array>,
): AsyncGenerator<SSEMessage> {
  const reader = stream.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      // Emit every complete block; a partial separator at the end of the
      // buffer simply waits for the next chunk.
      for (
        let sep = BLOCK_SEPARATOR.exec(buffer);
        sep !== null;
        sep = BLOCK_SEPARATOR.exec(buffer)
      ) {
        const block = buffer.slice(0, sep.index);
        buffer = buffer.slice(sep.index + sep[0].length);
        const msg = parseBlock(block);
        if (msg) yield msg;
      }
    }
    // Flush a trailing block that lacked its terminating blank line.
    const tail = parseBlock(buffer.trim());
    if (tail) yield tail;
  } finally {
    reader.releaseLock();
  }
}

/** Parse one event block into an {@link SSEMessage}, or `null` if empty. */
function parseBlock(block: string): SSEMessage | null {
  if (!block) return null;

  let event = "message";
  const data: string[] = [];

  // Field lines inside a block may use any spec line ending (CRLF, LF, or CR),
  // matching the block-separator handling above.
  for (const line of block.split(/\r\n|\n|\r/)) {
    if (line === "" || line.startsWith(":")) continue; // blank / comment (heartbeat)

    const colon = line.indexOf(":");
    const field = colon === -1 ? line : line.slice(0, colon);
    let value = colon === -1 ? "" : line.slice(colon + 1);
    if (value.startsWith(" ")) value = value.slice(1); // spec: strip one leading space

    if (field === "event") event = value;
    else if (field === "data") data.push(value);
  }

  if (data.length === 0 && event === "message") return null;
  return { event, data: data.join("\n") };
}
