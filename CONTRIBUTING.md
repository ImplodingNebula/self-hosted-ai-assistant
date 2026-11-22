# Contributing

Conventions for working in this repo. They're light but enforced, so the project stays clean as it grows.

## Getting set up

```bash
cp infra/.env.example .env
make up                 # start the stack (see docs/SETUP.md for GPU notes)

# Install the pre-commit hooks (formats + lints on every commit):
pipx install pre-commit   # or: pip install pre-commit
pre-commit install
```

On Windows, `make` needs Git Bash (or WSL2) and GNU Make - see the
**[Windows setup notes](docs/SETUP.md#windows-install-make-and-a-unix-shell)**.

## Make targets

`make` is the single command surface. `make help` prints this list at any time;
each target just wraps the `docker compose` (or tool) command shown in the
`Makefile`.

| Target | What it does |
|---|---|
| `make init` | Write a first `.env` from the template, with a fresh `SECRET_MASTER_KEY` |
| `make up` | Start the whole stack, in the shape `.env` describes |
| `make down` | Stop and remove the containers |
| `make restart` | `down` then `up` |
| `make gpu-check` | Verify the model server registered a GPU; restart it once if discovery fell back to CPU - see [below](#the-targets-worth-explaining) |
| `make rebuild` | Rebuild the images (one service at a time) and refresh dependency volumes (after changing dependencies) |
| `make ps` | List the running services |
| `make logs` | Tail the logs from every service |
| `make models` | List models installed in Ollama (`ollama list`) and those currently loaded (`ollama ps`) |
| `make status` | Stack-health snapshot: services with health, a backend probe, and the loaded models |
| `make config` | Validate and print the merged compose configuration |
| `make db-upgrade` | Apply database migrations in the backend container (they also run at backend start) |
| `make dev-session` | Mint a signed-in session for local testing without a browser (refused in the `prod` layout) |
| `make env-check` | Warn when `.env` lacks variables the template defines |
| `make sync-env` | Add variables the template defines and `.env` lacks - see [below](#the-targets-worth-explaining) |
| `make warm` | Prime the model server so the first request is fast - see [below](#the-targets-worth-explaining) |
| `make warm-search` | Prime the web-search backend, so the first search does not pay a cold start |
| `make cool` | Unload every resident model, freeing the GPU for other work (`make warm` re-primes) |
| `make deploy` | Bring a running stack up to date, rebuilding images only when they are actually stale - see [below](#the-targets-worth-explaining) |
| `make redeploy` | The same refresh with the rebuild forced - see [below](#the-targets-worth-explaining) |
| `make perf` | Time a chat turn, so a change can be shown not to have made the assistant slower - see [below](#the-targets-worth-explaining) |
| `make tool-check` | Check the served model's tool-calling (opt-in diagnostic; run after `make warm`) |
| `make embedder-check` | Check the embedder's relevance bands against the retrieval floors |
| `make backup` | Back up the database, vector store and uploaded files into `backups/<timestamp>` |
| `make restore` | Restore a backup: `make restore DIR=backups/<timestamp>` |
| `make prune` | Reclaim Docker disk left behind by dev churn; leaves the named data volumes alone - see [below](#the-targets-worth-explaining) |
| `make disk` | Report how much of Docker's disk use is reclaimable (Windows; read-only) - see [below](#the-targets-worth-explaining) |
| `make host-disk` | Report host disk headroom and the size of each data volume (read-only, all platforms) |
| `make compact` | Return that reclaimable space to the operating system (Windows; needs elevated rights) - see [below](#the-targets-worth-explaining) |
| `make backend-test` | Run the backend tests (fakes; no Ollama needed) |
| `make frontend-test` | Run the frontend check batteries (composer fidelity, message transitions, proxy contract, auth contract, console defaults) |
| `make lint` | Lint both apps (ruff + eslint) |
| `make fmt` | Format both apps (ruff + prettier) |
| `make audit` | Run the JS dependency audit manually |
| `make help` | Print the target list |

There is one `make up`, and `.env` decides what it starts: `DEPLOY_ENV`
(`dev` | `prod`) picks the compose layout and `PROFILE` (`gpu` | `cpu` |
`remote`) picks where models are served. The two axes are independent - a server
whose GPU is a remote pod runs `DEPLOY_ENV=prod` with `PROFILE=remote`.

*Which* models run is not a make variable at all. It is chosen against the
inventory the model server actually reports: a conversation's own pinned model,
above the workspace defaults set on the console, above `DEFAULT_CHAT_MODEL`.
Downloading weights belongs to the Model Manager page for the same reason - it
is the surface that can see what is already there. The
[setup guide](docs/SETUP.md#getting-models-onto-the-machine) carries the headless
equivalent, for a machine with no browser to hand.

To change one of the two axes for a single command, write it **after** the
target: `make up PROFILE=cpu`, `make up DEPLOY_ENV=prod`. The env-prefix form
(`PROFILE=cpu make up`) is silently ignored whenever `.env` defines that key,
because `make` reads `.env` as part of its own makefile and a makefile
assignment outranks the environment.

Handy overrides: `make warm CHAT_MODEL=qwen3:8b` to warm a model other than the
one the deployment resolves (a partial name is enough - the reply says which
model was warmed), and `make warm OLLAMA_KEEP_ALIVE=15m` (it applies to
`warm-chat`/`warm-embed` too) for a time-capped warm on a testing or shared box,
where the default keeps a warmed model resident until `make cool` or a stack
stop. The reranker sidecar's compose profile follows `RERANKER_URL`;
`RERANKER=0` / `RERANKER=1` force it off or on regardless.

### The targets worth explaining

**`make sync-env`** appends the variables `infra/.env.example` defines but `.env` lacks, using the template's
defaults, and flags keys the template no longer mentions.

- It is append-only: existing lines are never edited or removed.
- So a drifted VALUE (the key is present, its value is stale) is not something it can fix. Drifted values and
  flagged keys need a manual review.
- `make env-check` is the read-only half, and it runs automatically before `up`, `rebuild` and `warm`.

**`make warm`** primes the model server so the first request does not pay a cold load.

- It works the same in either layout, by a different route. In `dev` it calls the backend's published port
  directly. A deployment publishes no such port, so the same request is made from inside the backend container
  instead; `make status` probes the same way. Nothing to pass or remember - `DEPLOY_ENV` already says which.

- Bare `make warm` warms the chat model and then the embedder, leaving both resident. The order is deliberate:
  on a tight-VRAM box, loading both at once can transiently evict the CPU-pinned embedder.
- `make warm-chat` warms only the chat model and primes its prompt cache. A cold run takes a minute or two, and
  the first non-thinking reply is fast afterwards. A first turn that needs the *thinking* model still pays that
  model's own cold load.
- `make warm-embed` warms only the embedding model, removing the cold stall on the first ingest or search.
- `make cool` is warm's opposite: it unloads every resident model (chat, thinking, vision, embedder alike) so
  the GPU - and the RAM the CPU-pinned embedder holds - is free for other work on the machine. Nothing breaks;
  the next request simply pays the cold load warm exists to avoid, and `make warm` re-primes deliberately.
- Every warm step retries once before giving up. A warm is the heaviest thing the stack does on a constrained
  host, so one refusal usually means "busy loading" rather than "broken" and the retry succeeds. A second
  failure is real: it is reported and fails the target rather than being printed and walked past. The one
  exception is the search warm, where "web search is not configured" is a legitimate answer and is skipped
  rather than retried. `WARM_RETRY_SLEEP=<seconds>` tunes the pause.

**`make deploy`** brings a running stack up to date and picks the cheapest way that actually works.

- **In the `prod` layout it always rebuilds**, and the rest of this explanation does not apply there. Nothing is
  bind-mounted in a deployment: the application code is copied into the images, so a rebuild is the only thing
  that can deploy a change to it.
- Most changes need no image build at all. The development overlay bind-mounts `apps/backend` and
  `apps/frontend` into their containers, so application code is already whatever the working tree says. Only a
  change to something baked INTO an image makes the images stale: `pyproject.toml`, `uv.lock`, `package.json`,
  `package-lock.json`, either `Dockerfile` or `.dockerignore`, or anything under `services/`.
- So `deploy` compares exactly those files against the commit the running images were built from, which each
  deploy records in a git-ignored `.deploy-stamp`. Nothing changed: apply `.env`, recreate only the containers
  whose configuration moved, apply pending migrations, restart the backend, worker and web UI, check the GPU.
  That takes seconds, and any model already loaded stays loaded as long as the model server is not one of the
  containers being recreated. Something changed: the full rebuild, exactly as `make redeploy` does it.
- The migrations and the restart are in that list because a bind mount delivers files, not behaviour. Migrations
  run from the backend container's start command, so an update carrying one would otherwise leave the new code
  on the old schema until something restarted that container - and the restart would then re-run them silently,
  since a failed migration exits the container while `compose restart` still reports success. Running them
  explicitly makes the step loud and stops the deploy on a failure. And file-change notifications do not cross
  every bind mount - not the Windows one at all - so the reloading backend and the web UI's dev watcher can hold
  an edited file and go on serving what they started with. Both failures are silent, which is why neither is
  left to be noticed later.
- One residual race is left in knowingly: when `up` has just recreated the backend, that container runs its own
  `alembic upgrade head` while the explicit one execs in beside it. The database serialises them and the loser
  errors out, so the deploy fails loudly with the schema already correct and a re-run passes. Closing it means
  waiting on the backend's healthcheck first, and a wait loop that can hang a deploy was the worse trade against
  a rare, self-clearing failure.
- Anything that leaves "what is deployed" uncertain - no stamp yet, no repository to compare against, a stamped
  commit the repository no longer knows, an uncommitted edit to one of those files - falls back to the full
  rebuild and prints which case it was. The cheap path is only ever taken on positive evidence that no image
  input moved.
- The compose files are the one gap in that list, on purpose. They carry each service's `build:` stanza, so
  editing a build context or target there does make an image stale - but they change far more often for reasons
  a build does not care about (ports, environment, healthchecks), which `up` applies by itself in seconds.
  Listing them would send the most common cheap deploy down the expensive path to cover a rare edit, so after
  changing a `build:` stanza, reach for `make redeploy`.

**`make redeploy`** is the same refresh with the rebuild forced: `sync-env`, `stop`, `rebuild`, `gpu-check`,
`warm`. Reach for it when the images need rebuilding for a reason the file list cannot see, or simply when in
doubt.

- It stops the containers for the duration of the build and starts them again afterwards. On a
  memory-constrained host the build needs the memory the running stack is holding, and a resident model is the
  largest part of that. Nothing is removed: containers, networks and every data volume survive.
- Images are built one service at a time. Building them concurrently starts several builder processes at once,
  and on a host already running the stack one of them gets killed, which fails the whole build.
- It refuses to warm a model server that has fallen back to the CPU.

**`make gpu-check`** answers one question: did the model server register a GPU, or is it about to run everything
on the CPU?

- It matters because the failure is silent. The server probes for GPUs at startup, and on a busy host that probe
  hits its watchdog; the server then registers the CPU as its only inference device for the rest of its life.
  Nothing errors. Generation simply becomes roughly ten times slower, which reads as a performance regression
  rather than a broken deployment.
- That is why `make redeploy` runs it before warming, and why its non-zero exit deliberately stops the warm:
  warming a CPU fallback wastes minutes and hides the fault, and any timing taken from that stack is measuring
  the CPU.
- The check reads only the current process's log, scoped by the container's start time. A container keeps its
  log across restarts, so searching the whole log finds a healthy line from an earlier boot and passes a server
  that is CPU-only right now. It also reads the `library=` field of the `inference compute` line specifically,
  because the discovery failure messages themselves name library paths containing "cuda".
- A discovery timeout in the log is not by itself a failure: the server routinely times out probing one library
  directory and then succeeds with another. Timeouts are quoted as context once the verdict is already
  CPU-only, never used to reach it.
- On failure it says what it found first, then restarts the model server once and re-checks, since a quiet retry
  usually finds the GPU. It only restarts for the states a retry can change: a backend name it does not
  recognise, or a container it could not inspect, are reported and left alone. `GPU_CHECK_TIMEOUT=<seconds>`
  (default 120) sets how long it waits for the model server to register a device before drawing a conclusion.
- The matching logic lives in [`scripts/gpu_check.sh`](scripts/gpu_check.sh), which can be run against a
  captured log and exits non-zero for anything that is not a GPU. Scope the log to one process, or the check
  reproduces the very trap described above - it judges whatever text it is given:
  ```bash
  c=ollama-gpu
  docker logs --since "$(docker inspect -f '{{.State.StartedAt}}' $c)" $c 2>&1 \
    | sh scripts/gpu_check.sh classify
  ```

**`make prune`** reclaims what development churn leaves behind - orphaned anonymous volumes, dangling images
and build cache - and leaves the named data volumes (`postgres_data`, `qdrant_data`, `ollama_models`, ...)
alone.

- That last part is a property of the **Docker engine**, not of a flag. Limiting the sweep to anonymous
  volumes arrived in Docker Engine 23.0 (API 1.42). On an older daemon the same command also removes named
  volumes that no running container holds - and with the stack stopped, that is the database, the vector
  store, the uploaded files and the model weights.
- No option restores the newer behaviour, so the target checks the running daemon first and declines rather
  than assuming, via [`scripts/prune_guard.sh`](scripts/prune_guard.sh). It is not a hypothetical gap: Ubuntu
  22.04 still packages Docker 20.10 (API 1.41) next to a current Compose plugin, which satisfies the stated
  prerequisites exactly.
- `make rebuild` runs the same guarded sweep, in its lenient form: the rebuild has already succeeded by then,
  so an unreclaimed scratch volume is reported rather than treated as a failure.

**`make disk` and `make compact`** deal with a Windows-only quirk that quietly costs tens of gigabytes.

- On Windows, Docker keeps every image, build-cache entry and volume inside a single virtual-disk file. Docker
  treats it as a hard drive; the operating system sees one large file. The catch is that the file never shrinks
  on its own. Deleting an image frees space *inside* the virtual drive, so Docker reports the space as
  available, while the file on disk stays as large as it ever grew. Docker Desktop used to collapse that gap
  automatically, but the feature was disabled in version 4.59.0 and has not returned.
- `make disk` names the two numbers that get confused: what the operating system is giving up, and what
  Docker's content actually needs. The difference is dead space, and it is the only thing compaction can
  return. It reads nothing and changes nothing, so it is safe at any time.
- It also reports **who owns the disk file**, which sounds unrelated and is not. Relocating the file to
  another drive is a supported setting, but the usual ways of doing it leave the new folder owned by the
  Administrators group, or copy the file without its access rules. Windows gives every virtual machine its
  own account access to the disk files it opens, and that account changes identity when the machine
  restarts, so the grant must be renewed - which requires *owning* the file. Administrator rights do not
  substitute, because an ordinary terminal runs with that group membership disabled. The result is a
  relocation that works perfectly until the first reboot and then refuses to start, reporting
  `Wsl/Service/CreateInstance/MountDisk/HCS/E_ACCESSDENIED`. The report names the two commands that fix it.
- `make compact` collapses that gap. Order matters: compaction recovers only what has already been deleted, so
  the target prunes first, trims the filesystem, stops the container runtime, verifies the file is genuinely
  released, and only then rewrites it. On one measured run this took a 65.6 GB file down to 40.6 GB.
- It is deliberately not part of `make prune`. Pruning is instant, needs no special rights, and is safe while
  the stack is running; compaction is the opposite on every count. Folding them together would stop the stack
  on every routine cleanup, usually to reclaim nothing - ordinary use *grows* the file, which compaction cannot
  help with.
- Both targets print an explanation and exit successfully on Linux and macOS, which have no such file: Linux
  stores layers directly on the filesystem, and macOS manages its own disk image. The logic lives in
  [`scripts/wsl_disk.sh`](scripts/wsl_disk.sh).

**`make perf`** times a chat turn against the running stack - time to first token, and tokens per second.

- Only the plain turn runs by default, because that is the one `make warm` leaves warm. `PERF_THINK=1` adds a
  thinking turn and `PERF_WEB=1` a web-search turn.
- `PERF_MAX_TTFT` and `PERF_MAX_TOTAL` (in seconds) turn it into a gate.
- Warm the stack first, or a cold model load dominates the numbers and the reading is meaningless.

## Branching (trunk-based)

- `main` is always deployable. Do work on **short-lived branches** named `type/scope-short-desc`
  (e.g. `feat/rag-hybrid-search`), and open a PR within a day or three.
- **Squash-merge** PRs. No long-lived `develop`/`release` branches.
- `main` takes PRs with passing checks (CI, commitlint, audit), and no force-pushes. That is the rule, but note
  it is a rule people keep rather than one the host enforces: branch protection is unavailable on this
  repository's plan, so a red run does not block the merge button. Check the run, then merge.

## Commit messages (Conventional Commits)

Format: `type(scope): subject` - imperative, lowercase, no trailing period.

- **types:** `feat, fix, chore, refactor, docs, test, ci, build, perf, style`
- **scopes:** `backend, rag, mcp, models, model-manager, chat, auth, attachments, frontend, ui, infra, ci, repo, docs, deps, deps-dev`

Examples: `feat(mcp): add search_records tool` · `fix(chat): flush partial message on disconnect`.
The body explains **why**; footers carry `BREAKING CHANGE:` and `Closes #NN`. Enforced by commitlint in CI.

## Security - dependency audits (required)

**Whenever a JS dependency is added or updated (`npm/pnpm/yarn install|add|update`), run an audit immediately and
resolve any moderate-or-higher findings before proceeding** - via `npm audit fix`, pinning/replacing the package,
or a documented, signed-off risk acceptance. This is enforced by:

- **CI:** `npm audit` runs on every PR/push that touches a JS workspace, and nightly - across every workspace
  with a lockfile (`apps/frontend` and `packages/api-types`), not just the app. It is advisory rather than a
  required check (see the branching note above) - treat a red audit as blocking anyway. OSV-Scanner joins it in
  the Phase 8 hardening pass.
- **Dependabot:** opens security-update PRs automatically.

Also: **commit lockfiles**, use `npm ci` (never `npm install`) in CI/Docker, and keep **Next.js pinned at or above
the latest patched release**.

## Tests

Mock-by-default, real-by-opt-in - the model/RAG/MCP seams are faked so tests are fast and deterministic. New to
testing? Read **[docs/TESTING.md](docs/TESTING.md)**.

```bash
make backend-test      # pytest (fakes; no Ollama needed)
make frontend-test     # composer fidelity, message transitions, proxy contract, auth contract, console defaults
```

The frontend has no general-purpose test runner. What `make frontend-test` runs is five plain Node scripts under
`apps/frontend/scripts/`, each pinning one contract that lint and type-check cannot see; they are described in
[docs/TESTING.md](docs/TESTING.md).

A `@pytest.mark.ollama` marker exists for real-model tests and is deselected on every CI run
(`-m "not ollama and not ocr"`). No test carries it yet, and there is no scheduled job that would run one - see
[docs/TESTING.md](docs/TESTING.md).

## Before opening a PR

Run the app (`make up`), run tests, `make lint`, and fill in the PR template. If a change touches the tenant/auth
or MCP/secrets boundary, do a security review. Update the docs if behavior or a decision changed. If a change
alters the backend API (routes or request/response models), regenerate the committed types in `packages/api-types`
and include them in the PR - CI fails on drift (see [packages/api-types/README.md](packages/api-types/README.md)).

## Where things live / build order

See **[docs/ROADMAP.md](docs/ROADMAP.md)** (phases) and **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)** (design).
