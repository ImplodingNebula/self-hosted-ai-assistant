# doc-gen

The document-generation sidecar - a Streamable-HTTP MCP server that turns a
structured specification into a **real file**: a spreadsheet, a formatted
document, a PDF, a CSV, a Markdown file, or a chart image. The assistant fills a
schema, and pinned library code renders it.

That split is the whole design. A model that could write and run code to produce
a file would be a code interpreter, with everything that implies; here the model
only ever fills in a JSON shape, and the rendering is ordinary reviewed Python
that ships with the image. There is no place for model-authored code to run,
because none is ever executed.

Both tools are **read-only** in the MCP sense: they reach no upstream system and
change nothing anywhere - they turn their arguments into bytes and hand them
back. Implemented in
[ROADMAP Phase 9a](../../docs/ROADMAP.md#phase-9a---document-generation-tool-service);
the tool seam it plugs into is described in
[../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) section 7.

## Endpoints

| Path | What it serves |
|---|---|
| `/docgen/mcp` | The MCP endpoint (Streamable HTTP: `POST` for requests, `GET` for the event stream, `DELETE` to end a session) |
| `GET /healthz` | Liveness probe; always open, returns `200 ok` |

A client connects to the full base-plus-path URL - `http://localhost:8765/docgen/mcp`
on the same host, or `http://doc-gen:8765/docgen/mcp` from a compose network.
The path prefix exists for the same reason the MCP chassis uses one: each mounted
server is its own registration in the platform's server registry, with its own
credential and its own tool toggles.

## The tools

### `create_document(format, spec)`

Renders one file in one of five formats. The spec is deliberately **flat** -
one optional field per format rather than a variant per format - because a small
model picks fields far more reliably than it picks a variant.

| Format | Spec fields | Rendered by |
|---|---|---|
| `xlsx` | `sheets: [{name?, rows, header_row?}]` | openpyxl |
| `docx` | `title?`, `blocks` | python-docx |
| `pdf` | `title?`, `blocks` | reportlab (platypus) |
| `csv` | `rows` | the standard library `csv` writer |
| `md` | `text` (or `blocks`) | passthrough |

A **block** is `{type, text?, items?, rows?, level?}`, where `type` is one of
`heading`, `paragraph`, `bullets`, `numbered` or `table`.

Rendering is forgiving throughout, because a near-miss should still produce the
document that was asked for:

- An unrecognized block type renders as a paragraph rather than failing the
  document.
- A ragged table is padded to its widest row.
- A heading level outside 1-3 is clamped.
- A near-miss field is accepted where the meaning is unambiguous: bare `rows`
  sent for an `xlsx` becomes a single sheet, `blocks` sent for `md` are rendered
  down to Markdown.
- Worksheet names are repaired rather than rejected. Excel forbids several
  characters and caps names at 31 of them; the writer refuses an illegal
  character outright and merely warns about an over-long name while writing a
  file some applications then cannot open.

### `render_chart(spec)`

Renders one chart image from numeric series: `{type, title?, x_label?, y_label?,
labels?, series: [{name?, values}], format}`, where `type` is `line`, `bar`,
`pie` or `scatter` and `format` is `png` or `svg`. Common aliases are mapped
(`column` to bar, `donut` to pie), and an unrecognized type renders as a bar
chart with the summary line naming what was actually drawn.

Charts are drawn through matplotlib's object-oriented API on the Agg backend.
Neither detail is incidental. Agg is the headless renderer, chosen before
anything can reach for a display that a container does not have. And pyplot
keeps a global registry of figures plus a notion of the "current" one, which two
requests on two worker threads would be reaching for at the same time; the
object-oriented API removes that shared state. That is a narrower claim than
thread safety, which matplotlib does not promise.

## What a call returns

Every file-producing call answers with exactly two content blocks, in this
order:

1. A **text summary** - one line, the only part of the answer the model reads
   (`Created quarterly_report.xlsx (2 sheets, 12.4 KB).`).
2. One **embedded resource** carrying the bytes: `uri` is
   `artifact://<filename>`, `blob` is base64, and `mimeType` is the format's
   real media type.

The order and the count are a contract the platform host reads positionally, so
they are built in one place rather than at each tool. One call produces one file.

Filenames are **sanitized here and never trusted**: the name is taken from
`spec.filename`, else the title, else a default describing the file; path
segments are dropped, anything outside `[A-Za-z0-9._-]` becomes a hyphen, the
stem is capped, and the extension the format dictates is always applied. A
download name is not a place to discover that a model wrote `../../etc/passwd`.

One consequence worth knowing: the allowed set is ASCII, so a title written
entirely in another script leaves nothing behind and the file takes its default
name instead. Only the NAME is restricted - the content is UTF-8 throughout.

## Limits

Hard caps, each a module constant with a test pinning it. They exist so that one
malformed specification cannot turn into a gigabyte of base64, and each refusal
comes back as plain readable text - naming the limit and suggesting a smaller
shape - rather than as an error, so the caller can correct it and try again.

| Limit | Value | Why |
|---|---|---|
| Characters of text per request | 1,000,000 | The only limit on how much WORK a render is handed - see below |
| Cells per workbook | 50,000 | The writer holds every cell in memory |
| Blocks per document | 200 | A document is a report, not a data dump |
| Rows per table block | 500 | The one block that can be arbitrarily tall alone |
| Rows per CSV | 50,000 | CSV has no structural limit of its own |
| Points per chart | 10,000 | Render cost is linear in points |
| Wedges per pie chart | 50 | Pie cost is not linear - a 10,000-wedge pie measured at roughly 53 seconds |
| Bytes per file | 1,200,000 | Set by what the calling host can receive - see below |
| Files per call | 1 | The host reads the single artifact positionally |

The **text limit** is the only one that bounds size rather than quantity. Every
other row above counts things, and counting things does not bound a request: a
few dozen bullet items, each well within every per-item rule, can still be
megabytes of prose. That cost is paid before any output check can refuse the
result, because a layout engine has to lay text out to find out how large it
becomes. So the total is summed across every string in the request - titles,
block text, list items, table and sheet cells, chart labels and series names -
and checked once, before a renderer starts. It is deliberately a single total
rather than a length rule per field, since a per-field rule stops covering the
newest field the day one is added.

It is the second line of defence rather than the first: the MCP server refuses a
request body over 4 MiB outright, before parsing. What that leaves uncovered is
the band underneath - a request that is accepted, breaks no per-item count, and
is still far more text than a document should carry.

The byte limit is the one that is not a matter of taste. The platform's MCP
client reads a whole response body under a hard 2,000,000-byte ceiling, and
breaking it **raises** - the entire tool call fails, with no path that keeps the
summary and drops the file. Base64 costs a third on top of the raw bytes, so
1,200,000 bytes becomes 1,600,000 characters and leaves 400,000 for the summary
and the JSON envelope. Refusing here costs the caller one correctable message;
being refused there costs them the whole call.

A practical consequence for charts: an SVG holds one element per data point, so
a large scatter or bar chart can reach that limit while the same chart as a
`png` stays a fixed-size raster of a few tens of kilobytes. The refusal says so
rather than suggesting less data.

## Notes on the generated files

- **Spreadsheet cells are always text, never formulas.** A string beginning with
  `=` would otherwise become a live formula that runs when the recipient opens
  the download, and the content of a generated document can be steered by
  whatever the assistant read on the way to writing it. The value reads back
  unchanged, so honest data loses nothing.
- **CSV is a verbatim data dump.** It has no cell types to mark, so a
  spreadsheet application opening a CSV will interpret a leading `=` itself -
  which is a property of the CSV format rather than of this service. Prefer
  `xlsx` when the file is meant to be opened in a spreadsheet.
- Files are written as UTF-8 (CSV and Markdown) or as their own binary format.
- A very wide table in a PDF stays on the page by squeezing its columns, which
  wraps long cell text mid-word. Nothing is dropped, but a table more than about
  eight columns across reads better as a spreadsheet.

## Configuration

All configuration is via environment variables.

| Variable | Default | Effect |
|---|---|---|
| `DOCGEN_ALLOWED_HOSTS` | `localhost,localhost:*,127.0.0.1,127.0.0.1:*,doc-gen,doc-gen:*` | Comma-separated Host allow list |
| `DOCGEN_API_KEY` | unset | When set, every MCP request must carry `Authorization: Bearer <key>` |

Detail on each:

- **Allowed hosts.** The MCP SDK ships DNS-rebinding protection with an EMPTY
  allow list, so every request - localhost included - is rejected with `421`
  until the accepted Host values are listed explicitly. The default covers local
  runs and the `doc-gen` compose service name; a deployment reaching this
  service under another hostname must list it here. A set-but-empty value falls
  back to the default with a startup warning, since an empty allow list would
  silently reject everything.
- **API key.** Unset means open, which is suitable only for a private compose
  network, and the startup log says so. A set-but-blank value is treated as
  unset and warned about, since that is what a failed variable substitution
  produces. When set, requests without the exact bearer key get `401`;
  `GET /healthz` stays open so orchestrators can probe liveness without holding
  the key.

The backend side of the same wiring is `DOCGEN_MCP_URL` (which seeds the server
into the registry), `DOCGEN_API_KEY` (the same value as above, sent by the
backend) and `DOCGEN_CREDENTIAL_NAME` (the name the seeded row is filed under) -
all documented in `infra/.env.example`.

## Running standalone

From `services/doc-gen/`:

```bash
uv sync
uv run uvicorn app.main:app --port 8765
```

Or containerized:

```bash
docker build -t doc-gen .
docker run --rm -p 8765:8765 doc-gen
```

The server is stateless (`stateless_http=True`), so any Streamable-HTTP MCP
client can initialize against `/docgen/mcp` without session affinity.

## Tests

```bash
uv run pytest
```

The tests are hermetic - the real MCP protocol over an in-process ASGI
transport, no sockets. Every format is written through a real tool call and then
**reopened with a library that reads it** (openpyxl, python-docx, pypdf, the
`csv` module), because "bytes came back" is equally true of a truncated file.
The rest cover the artifact contract itself (block order, the `artifact://` URI,
media types), each limit at its exact boundary, the read-only annotations, and
the auth and Host gates.

Lint runs with `uv run ruff check .` and formatting with `uv run ruff format .`.
