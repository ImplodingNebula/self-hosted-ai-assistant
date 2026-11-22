# api-types

TypeScript types generated from the backend's OpenAPI schema, meant to keep the
frontend and backend from drifting on request/response shapes once they are
consumed - see [Consuming the types](#consuming-the-types) for where that
stands. Both artifacts are committed: `openapi.json` (the exported schema) and
`src/schema.ts` (the types derived from it). Nothing here is hand-written; edits
belong in the backend's Pydantic models and routers, from which everything below
regenerates.

## Regenerating

Whenever the backend API changes (new endpoint, changed request/response
model), regenerate both files and commit them alongside the backend change:

```bash
# 1. Export the schema (offline - no database or model server needed)
cd apps/backend
uv run python scripts/export_openapi.py --out ../../packages/api-types/openapi.json

# 2. Regenerate the TypeScript types
cd ../../packages/api-types
npm ci
npm run generate
```

Output is deterministic: the exporter serializes with sorted keys and LF line
endings, and `openapi-typescript` (pinned in `package.json`) emits stable
output, so an unchanged API always reproduces byte-identical files.

## Drift check in CI

The `api-types` job in the CI workflow re-runs both steps above on every pull
request and on pushes to `main`, then fails if the regenerated files differ
from the committed ones. A red check means the backend API changed without the steps
in [Regenerating](#regenerating) being run - regenerate, commit, and push.

## Consuming the types

**Nothing in the monorepo imports this package yet.** The frontend deliberately
hand-mirrors the backend's request and response shapes in `apps/frontend/lib/`
instead, and says so in those files - partly because the streaming chat
responses carry no response model and so never appear in the OpenAPI schema at
all. The drift protection above is therefore the CI check on this package's own
artifacts, not a type error in the app.

The package is private to the monorepo. A consumer would import the `paths` /
`components` types from `src/schema.ts` (for example via a relative import or a
workspace alias) to type API clients and route handlers.
