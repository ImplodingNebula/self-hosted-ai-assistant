# mcp-chassis

The MCP chassis - a generic Streamable-HTTP container that mounts one or more MCP
wrapper servers at distinct URL paths. Each wrapper fronts an external
**API-backed datastore** - a JS/HTTP API that fronts a database (for example a
support-ticket system, a CRM, or an orders service). The wrapper calls that
upstream API and never touches the database directly; the wrapper exists because
upstream APIs do not speak MCP, so it translates.

One container, several servers: each mounted path is its own registration in the
platform's `mcp_servers` registry, with its own credential scope and tool
toggles. The bundled `/example` mount is registered by default - `MCP_CHASSIS_URL`
seeds its row on a tenant's first use. A single merged mega-server is
deliberately avoided - it would force one credential scope across unrelated
upstreams.

This service ships one **example wrapper**, mounted at `/example`, which
stands its upstream in with an in-process mock store so the container runs with
no credentials and no network. Deployment-specific wrappers live on other
branches. Implemented in
[ROADMAP Phase 5](../../docs/ROADMAP.md#phase-5---mcp-subsystem); see
[../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md) section 7.

## Endpoints

| Path | What it serves |
|---|---|
| `POST /example/mcp` | The example wrapper's MCP endpoint (Streamable HTTP) |
| `GET /healthz` | Liveness probe; always open, returns `200 ok` |

A client connects to the full base-plus-path URL, for example
`http://localhost:8765/example/mcp` on the same host or
`http://mcp-chassis:8765/example/mcp` from a compose network. Adding a wrapper
means one more mount in `app/main.py` - `/tickets/mcp`, `/orders/mcp`, and so
on - each a separate MCP server behind the same port.

## The example wrapper

`app/example_wrapper.py` builds a FastMCP server named `example-datastore` over
a module-level dict of records (`id -> {title, body, status}`):

- `search_records(query)` - substring search over titles and bodies; returns the
  matches as **structured content**: a `records` list of `{id, title, text}`,
  where `text` is the full body rather than a display snippet. That shape is the
  platform's citable-records contract - it lets the backend ingest each record
  and cite it by `[n]` marker instead of parsing a formatted string. Annotated
  `readOnlyHint=True`.
- `get_record(record_id)` - one record's full text; an unknown id returns a
  plain "no record with id ..." message rather than an error. Annotated
  `readOnlyHint=True`.
- `create_record(title, body)` - inserts a record and returns its new id.
  **Deliberately unannotated:** it is side-effecting, and the platform fails
  closed on a missing `readOnlyHint` - a missing annotation counts as
  side-effecting and requires confirmation.

## Configuration

All configuration is via environment variables.

| Variable | Default | Effect |
|---|---|---|
| `MCP_CHASSIS_ALLOWED_HOSTS` | `localhost,localhost:*,127.0.0.1,127.0.0.1:*,mcp-chassis,mcp-chassis:*` | Comma-separated Host allow list |
| `MCP_CHASSIS_API_KEY` | unset | When set, every MCP request must carry `Authorization: Bearer <key>` |
| `EXAMPLE_API_BASE_URL` | unset | Reserved for a real upstream; see below |

Detail on each:

- **Allowed hosts.** The MCP SDK ships DNS-rebinding protection with an EMPTY
  allow list, so every request - localhost included - is rejected with `421`
  until the accepted Host values are listed explicitly. The default covers
  local runs and the `mcp-chassis` compose service name; a deployment that
  reaches the chassis under another hostname must list it here. A set-but-empty
  value falls back to the default, with a startup warning - an empty allow list
  would silently reject everything.
- **API key.** Unset means open (suitable only for a private compose network),
  and the startup log says so. A set-but-blank value is treated as unset and
  warned about, since that is what a failed variable substitution produces.
  When set, requests without the exact bearer key get `401`; `GET /healthz`
  stays open so orchestrators can probe liveness without holding the key.
- **Upstream base URL.** On this branch the example wrapper always serves the
  built-in mock store. Setting `EXAMPLE_API_BASE_URL` logs a warning that a
  real upstream is configured but not implemented here; a deployment branch
  replaces the mock with an HTTP client against that URL.

## Running standalone

From `services/mcp-chassis/`:

```bash
uv sync
uv run uvicorn app.main:app --port 8765
```

Or containerized:

```bash
docker build -t mcp-chassis .
docker run --rm -p 8765:8765 mcp-chassis
```

The server is stateless (`stateless_http=True`), so any Streamable-HTTP MCP
client can initialize against `/example/mcp` without session affinity.

## Tests

```bash
uv run pytest
```

The tests are hermetic - the real MCP protocol over an in-process ASGI
transport, no sockets, mirroring the transport harness in
[`apps/backend/tests/unit/test_mcp_transport.py`](../../apps/backend/tests/unit/test_mcp_transport.py).
They cover the tool round trips, the annotations, the auth gate (rejected
without the bearer key, accepted with it), and the open health probe.

Lint runs with `uv run ruff check .`.
