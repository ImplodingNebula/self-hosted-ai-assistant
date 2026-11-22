# backend

FastAPI MCP host (Python 3.13). Layered: `api/` (HTTP/SSE) → `services/` (orchestration) →
`providers/` (model/RAG/MCP seams) → `repositories/` (tenant-scoped DB) → `core/` (config/DI/logging).

The subsystems below are merged. They do not correspond to a clean phase boundary - several later phases are
partly landed - so [../../docs/ROADMAP.md](../../docs/ROADMAP.md) is the per-phase and per-slice status of
record.

- **Chat.** The model seam and the streaming chat slice, persisted conversations with server-side
  history (PostgreSQL + Alembic migrations), and RFC 7807 problem+json errors.
- **Retrieval.** The RAG subsystem - Qdrant-backed retrieval behind the
  VectorStore/Embedder/Chunker/Reranker seam, `/rag` ingest and query endpoints, and automatic
  grounding of chat turns with streamed and persisted citations - plus local-first web search behind
  a `SearchProvider` seam, and the arq worker that sweeps expired session vectors and parses
  uploaded files into that same pipeline.
- **Attachments.** Upload, parse and index files of every supported family, answerable on the turn
  that sent them or any later turn that points at them, with a picture passed through to a model
  that can actually see it - served back on request, and aged out by a two-tier retention policy
  that expires tool-generated files by default and leaves person-sent ones alone unless a
  deployment says otherwise.
- **Tools.** The MCP host: a re-entrant tool loop under a clamped agency policy, local and remote
  tools behind one registry, side-effecting calls that suspend for confirmation, and tool servers
  registered per workspace with encrypted credentials or an OAuth grant.
- **Model management.** One `plan_turn()` per turn over the request > conversation > workspace > env
  resolution chain, the manager view's request-time join of the server's own inventory, the
  code-shipped model-quirk seed, stored operator declarations and measured probe facts, streamed
  pull, guarded delete, the tool-check harness with its `model_probes` measurement log, and the
  stranded-vectors alarm.
- **Document generation.** The `create_document` and `render_chart` tools, served by the `doc-gen`
  MCP sidecar and kept as stored attachments on the reply that produced them.
- **Identity and access.** OpenID Connect sign-in with server-side sessions, a real principal
  resolved per request, and a `require_role` dependency gating the routes that only an administrator
  may reach - with per-member isolation beneath it: each member's own stream budget and their own
  ephemeral vector scope.
- **Deployment hardening.** A readiness endpoint that separates hard dependencies from
  informational ones, and Redis-backed per-client rate limiting that fails open.

See [../../docs/ARCHITECTURE.md](../../docs/ARCHITECTURE.md).
