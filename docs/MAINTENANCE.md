# Maintenance

Getting the platform running is one command. Keeping a deployment healthy over the months afterwards is what this
document covers: applying an update, keeping the environment file honest, backing the data up, replacing a model,
reading the health endpoints, knowing what ages out on its own, and connecting an external tool server.

First-time setup is in [SETUP.md](SETUP.md) and [DEPLOYMENT.md](DEPLOYMENT.md); the design behind any of it is in
[ARCHITECTURE.md](ARCHITECTURE.md), and every table in the database is described in
[DATA-MODEL.md](DATA-MODEL.md). Every command below runs from the repository root, on the machine hosting the
stack, in a shell where `make` resolves.

---

## Contents

1. [Applying an update](#1-applying-an-update)
2. [Keeping the environment file current](#2-keeping-the-environment-file-current)
3. [Backups and restores](#3-backups-and-restores)
4. [Models over time](#4-models-over-time)
5. [Health, and the failure that does not announce itself](#5-health-and-the-failure-that-does-not-announce-itself)
6. [What expires, and what does not](#6-what-expires-and-what-does-not)
7. [Connecting an integration](#7-connecting-an-integration)

---

## 1. Applying an update

Pull the new source, then run one command:

```bash
git pull
make deploy
```

`make deploy` works out for itself whether the container images need rebuilding, because a rebuild costs minutes
and a recreate costs seconds. What it compares is a short list of **image inputs** - the dependency manifests
(`pyproject.toml`, `uv.lock`, `package.json`, `package-lock.json`), both `Dockerfile`s, both `.dockerignore`
files, and everything under `services/` - against the commit the running images were built from. That commit is
recorded in a git-ignored `.deploy-stamp` each time a deploy finishes, so the comparison describes *this
machine's containers* rather than the source tree.

### When it rebuilds

- **Any image input moved** since the stamped commit, committed or not.
- **The answer is uncertain**: no stamp yet, a stamped commit the repository no longer contains, or no git
  checkout at all.
- **The layout is `prod`.** A deployment bind-mounts nothing - the application source is copied into the images -
  so a rebuild is the only thing that can deploy a change to it, and `deploy` does not consult the list there.

Whichever case it is, the reason is printed. The cheap path is only ever taken on positive evidence that nothing
relevant changed.

### What the cheap path does

Reachable only in the development layout, where `apps/backend` and `apps/frontend` are bind-mounted into their
containers. In order:

| Step | What it delivers |
|---|---|
| `sync-env` | Variables the template has added since arrive in `.env` with their defaults |
| `up` | Containers whose configuration changed are recreated; the rest keep running |
| `db-upgrade` | Pending database migrations are applied |
| restart of backend, worker and web UI | Those three processes begin serving this working tree's code |
| deploy stamp | Records the commit that is now deployed |
| `gpu-check` | Confirms the model server registered a GPU rather than falling back to the CPU |

The two middle steps are there because **a bind mount delivers files, not behaviour**, and two gaps sit exactly
in that difference:

- **The database can be a migration behind.** Migrations run from the backend container's start command, so an
  update that adds one leaves the new code querying the old schema for as long as nothing restarts that
  container. Where a reloader picks host changes up by itself, the new code begins serving against the old schema
  straight away. The restart further down the list would re-run them, but *silently*: a migration that fails
  exits the container, and restarting a container reports success either way. Applying them explicitly, inside
  the container that is already running, makes the step synchronous and loud instead, and a failure stops the
  deploy where it can be read.
- **The processes can still be serving yesterday's code.** File-change notifications do not cross every bind
  mount, and they do not cross the one used on Windows at all, so the backend's reloader and the web UI's
  development watcher can hold the edited files and go on running what they started with. Restarting those three
  services is what actually puts the working tree on the air, and it costs seconds: none of the three holds a
  model.

Neither failure announces itself. The deploy reports success, the stack is up, and the wrong thing is running -
which is why both steps are part of the command rather than something to remember afterwards.

Two consequences worth knowing before they are met in the wild:

- **The restart is unconditional**, so a cheap deploy briefly interrupts whatever those three services are
  doing: a streaming reply, a document being parsed. On a machine somebody is using, it is worth running
  between turns rather than during one.
- **A migration step that fails immediately after containers were recreated is usually a collision, not a
  broken migration.** A recreated backend applies migrations at startup too, so it can be part-way through its
  own run when the explicit one starts beside it; the database serialises them and one of the two reports an
  error with the schema already correct. Re-running `make deploy` finds nothing left to apply and passes.

### The unconditional form

**`make redeploy`** forces the rebuild: sync the environment file, stop the containers, rebuild the images one
service at a time, stamp, check the GPU, warm the models. Reach for it when the images need rebuilding for a
reason the file list cannot see, or simply when in doubt.

The compose files are that list's one known blind spot, and deliberately so. They carry each service's `build:`
stanza, so editing a build context or target there really does make an image stale - but they change far more
often for reasons a build does not care about (ports, environment variables, healthchecks, volumes), which `up`
applies by itself in seconds. Listing them would send the most common cheap deploy down the expensive path to
cover a rare edit. **After editing a `build:` stanza, use `make redeploy`.**

The stop in the middle of a redeploy is not incidental: on a memory-constrained host the image build needs the
memory the running stack is holding, and a resident model is the largest part of that. Nothing is removed -
containers, networks and every data volume survive - and the rebuild starts everything again afterwards.

### What the cheap path deliberately does not do

It does not warm a model. With the compose configuration unchanged, `up` recreates nothing and every model stays
exactly as resident as it was; the restart names three application services rather than the whole stack, so the
model server is untouched either way. When `up` *does* recreate the model server, a cold load is minutes of work
that belongs to whoever asked for it, so the closing line reports what happened and names
[`make warm`](#priming-and-freeing-the-gpu) instead of spending that time unasked.

---

## 2. Keeping the environment file current

`.env` is one machine's configuration and is never committed. [`infra/.env.example`](../infra/.env.example) is
the template that documents every variable the platform reads. The two drift apart the moment an update
introduces a variable, and the drift is silent: the code default wins and the configured setting simply has no
effect.

- **`make env-check`** is the read-only half. It warns about variables the template defines and `.env` lacks, and
  it runs automatically before `up`, `rebuild` and `warm` - so a stale file is usually reported without anyone
  going looking. It also refuses outright on the one mismatch it can prove: a profile that starts a local model
  container while `OLLAMA_URL` points at another machine, or a remote profile with nothing to dial.
- **`make sync-env`** appends the missing variables with the template's defaults. It is **append-only**: existing
  lines - secrets, tuned values - are never edited or removed, so the worst it can do is add a default worth
  deleting by hand. It also lists keys the template no longer mentions and leaves them in place, since a
  deployment branch or local tooling may still read them.

**A drifted VALUE is the case neither command can fix.** When an update changes a default rather than adding a
key, the key is already present in `.env` and nothing flags it - no script can know which local values are
deliberate. Those changes are announced in the template itself, near the top, as dated migration notes:

```
# MIGRATION NOTE (2026-08-10) - the model tier is gone, and with it the
# committed catalog that decided which models a deployment ran.
```

Reading the notes added since the last update, and checking the values they name by hand, is the whole upkeep
routine. Notes are pruned once deployments have caught up.

Two details that save an afternoon each:

- **Both commands read only uncommented lines in the template.** A commented-out key there is deliberate - its
  code default is the intended value - so it is neither reported as missing nor appended.
- **Overriding a variable for a single command takes a specific form.** `make` reads `.env` as part of its own
  makefile, and a makefile assignment outranks the environment, so the env-prefix form `PROFILE=cpu make up` is
  **silently ignored** whenever `.env` defines that key. The form that works is the make-variable one, written
  after the target: `make up PROFILE=cpu`.

---

## 3. Backups and restores

Three of the stack's volumes hold something nothing can reproduce: the database (`postgres_data`), the uploaded
file bytes (`attachment_data`) and the retrieval vectors (`qdrant_data`). Those three are exactly what
`make backup` copies. The rest - model weights, the reranker's cache, the background job queue, the proxy's
certificates - are downloaded or derived again and are left out on purpose. The full volume-by-volume mapping,
and the reasoning behind each inclusion and exclusion, is in
[docs/DATA-MODEL.md](DATA-MODEL.md#what-lives-in-each-volume).

```bash
make host-disk           # free space, plus each volume's size and whether a backup covers it
make backup              # -> backups/<UTC timestamp>/ ; DEST=/mnt/backups writes elsewhere
make backup PLAN=1       # print what a backup would copy; write nothing
```

A backup runs against the **live** stack and stops nothing. PostgreSQL is dumped through the running server
rather than copied file by file (a data directory is written continuously, so a plain copy would capture pages
from different moments), the vectors come out through Qdrant's snapshot API for the same reason, and the
uploaded files are archived from a read-only mount that no running container notices. Each run writes a manifest
recording the size and checksum of every artifact plus the database migration the dump ends at, then opens what
it just wrote: the dump is listed back through `pg_restore`, the archive through `tar`, and each vector snapshot
is checked against the checksum Qdrant reported for it. A backup nobody can open is indistinguishable from no
backup, and stays that way until the worst possible moment. A local copy is only half of one, so `DEST=` pointing
at another disk or another machine is worth setting up as a scheduled job.

Both commands talk to Qdrant from **inside** the stack's Docker network rather than through a published host
port: each request runs in a throwaway `curlimages/curl` container (pinned, about 35 MB, pulled once) joined to
that network and addressing Qdrant by service name. That is what makes them work unchanged on a hardened
deployment - the production layout publishes no store ports at all - and it means the host itself needs no HTTP
client, only Docker and `tar`.

**Under `STORAGE_BACKEND=s3`, the backup skips the attachments and says so.** Archiving a bucket's contents would
produce something that looks like the uploaded files and holds none of them, so a deployment on an object store
backs the bucket up with that store's own tooling instead: mirroring it to a second location with the MinIO
client (`mc mirror`) against the live service, or, for the bundled MinIO profile, stopping the container and
copying the `minio_data` volume cold. Copying that volume while MinIO is running is not guaranteed consistent -
the same reason PostgreSQL is dumped through the server rather than copied.

Restoring replaces all three, and has one prerequisite: the backend, worker and web UI must be stopped first,
while PostgreSQL and Qdrant stay up - the restore works *through* them rather than around them. `make restore`
checks, refuses while the application is running, and prints the exact stop command for whichever layout `.env`
names - the development layout's second compose file is `infra/docker-compose.override.yml`, a deployment's is
`infra/docker-compose.prod.yml`, and using the wrong one recreates the containers in the wrong layout.

```bash
make restore DIR=backups/20260101-030000   # PLAN=1 verifies and prints the plan, restoring nothing
make up
```

`DIR` is required rather than defaulting to the newest run: a restore replaces the database wholesale, and
"whichever one it picked" is not a thing to find out afterwards.

**One direction of a restore is free and the other is not.** The backend applies pending migrations before it
serves, so a backup taken from *older* code heals itself on the next boot - the missing migrations simply run. A
backup from *newer* code cannot: the restored database sits at a migration the deployed checkout has never heard
of, and the backend exits at startup with `Can't locate revision identified by <id>`. Nothing is damaged, and the
fix is to deploy code at or after the commit the manifest records - but the failure lands minutes later wearing a
message that says nothing about a restore, so `make restore` compares the two before writing anything and states
which case it is holding.

**Re-check the copies that are being kept.** `sh scripts/restore.sh verify backups/<timestamp>` re-hashes every
artifact against its manifest and needs neither Docker nor a running stack, so it can be pointed at an archive
sitting anywhere. A backup only fails once, and by then it is the only thing left.

**Where the volumes physically live is a deployment choice, not a code one.** By default they sit under
Docker's own data root - `make host-disk` prints the path, typically `/var/lib/docker` on the root filesystem -
which means the database, the uploads, the vectors and every model weight share whatever space that filesystem
has. Giving them a dedicated data disk needs no change here: either mount the disk at the data root, or point
the Docker daemon's `data-root` setting at a directory on it and restart the daemon. Both move the whole engine,
images included, which is usually what is wanted - the model weights are the largest single consumer and are not
worth protecting, only worth housing.

---

## 4. Models over time

Models are downloaded, replaced and retired from the **Model Manager** in the console, at `/console/models` on
the deployment's own address. None of it is a deploy-time decision: the page lists what the model server actually
holds, and a download streams its progress there.

### Re-pulling a tag withdraws what was measured about it

Before the platform lets a model drive a turn with tools, it measures whether that model can. The resulting
verdict is bound to the **digest** of the weights it was measured against, not to the tag name - and tags are
mutable, so pulling a name that is already installed can hand back different weights. When the digest changes the
verdict no longer matches, the model reads as unverified again, and the platform clamps how much it is trusted to
do. The fix is to re-run the check from the Model Manager: a fast check takes about 45 seconds; a deep one takes
longer and is the only thing that can earn the strongest modes.

Two related rules are worth knowing before they surprise anyone:

- **Re-pointing the deployment at a different model server withdraws a passing verdict too.** Tool-calling
  behaviour belongs to the runtime as a particular machine runs it, so a check earned against a rented GPU
  certifies nothing for a laptop. Evidence of **failure** is never withdrawn by either move - relocating a
  deployment is not a way to rehabilitate a model that failed a guard.
- **A runtime upgrade marks existing results stale and re-probes nothing.** Upgrade day should not turn into a
  probe storm; the checks are re-run deliberately, on the models that matter.

### The reranker's weights are named twice

The reranker sidecar re-scores retrieval hits so only the best reach the model, and it belongs in the standard
retrieval path: it reserves no GPU and runs on the CPU in roughly 1-2 seconds per retrieval at realistic
candidate counts. Its weights are named in two places that must agree: `RERANKER_MODEL` in `.env` tells the
backend which model to ask for, and the sidecar's own compose command tells the sidecar which one to serve.
Changing one without the other makes every rerank call fail. The sidecar downloads roughly 1.1 GB of weights into
a named cache volume on its **first** start, so the first boot after enabling it is slow and every later one is
not.

### Priming and freeing the GPU

- **`make warm`** loads the chat model and primes its prompt cache, then the embedding model, then the web-search
  backend - so the first real request pays none of that. The order matters on a tight-VRAM machine: loading both
  models at once can transiently evict the one that got there first.
- **`make cool`** is the opposite. It unloads every resident model so the GPU, and the memory a CPU-pinned
  embedder holds, are free for other work. Nothing breaks; the next request simply pays the cold start again.
- **`make models`** lists what is installed and what is resident right now.

---

## 5. Health, and the failure that does not announce itself

Two unauthenticated endpoints report on the backend, and they answer different questions.

- **`GET /healthz`** - *liveness*. Always `200` with `{"status": "ok"}` whenever the process is answering at all.
  This is the one a container healthcheck or an uptime monitor should poll.
- **`GET /readyz`** - *readiness*. `200` when the deployment can serve and `503` when it cannot, with a body
  listing every dependency it probed: its name, whether it is required, its state (`ok`, `unreachable` or
  `unconfigured`) and how long the probe took.

**Only three dependencies can produce a `503`:** PostgreSQL, Redis and Qdrant. Without them there is no
conversation to load, no background job to queue and no passage to retrieve, so the deployment genuinely cannot
serve.

**The model server and the two sidecars are reported and never flip the verdict.** That is a decision,
not leniency. The model server may be a separate machine whose first load of a large model takes well over a
minute, and a deployment that marked itself unready for that would be pulled out of rotation at exactly the
moment it was about to become useful. Web search and the reranker are configured per deployment rather than
required to serve, so `unconfigured` is a state rather than a fault. The report still names all three, which
means the cause of a slow or ungrounded first answer is one request away.

Both endpoints are exempt from rate limiting: a limiter able to answer `429` to a readiness probe would take a
deployment down by itself.

From the host, **`make status`** gives the same picture in one place - the services and their health, a backend
probe, and the models resident right now - while `make ps` and `make logs` cover the rest.

### The GPU fallback

The model server probes for GPUs at startup, behind a watchdog. On a busy boot that probe can time out, and the
server then registers the **CPU as its only inference device for the rest of that process's life**. Nothing
errors. Every model afterwards loads onto the CPU, generation becomes roughly ten times slower, and it reads as a
performance regression rather than as a broken deployment.

- **Detect it** with `make gpu-check`, which reads the verdict off the current process's log rather than the
  whole log a container keeps across restarts, and restarts the model server once if discovery genuinely landed
  on the CPU. It is also why `make redeploy` runs the check *before* warming: warming a CPU fallback wastes
  minutes and hides the fault.
- **Confirm it by hand** with `make models` - a resident model reported as `100% CPU` is the reliable tell.
- **Fix it** by waiting for the machine to go idle, restarting the model server alone, and then re-running
  `make gpu-check` and `make warm`. A quiet retry usually finds the GPU.

---

## 6. What expires, and what does not

Nothing here needs attention day to day; it is listed so that "where did that file go" and "why is this volume
growing" both have an answer. Every name below is a variable in `.env`, and the sweeps that enforce them run in
the background worker.

**Conversations and messages are kept indefinitely, deliberately.** Chat text is tiny next to documents and
vectors, so nothing prunes it and no retention window has to be explained to the people using the system. Disk
monitoring is the guard instead, and `make host-disk` is how it is watched - its per-volume report names whichever
volume is actually growing, and counts the local backup copies too, since nothing prunes those either.

**Stored files age out on a two-tier policy, split by who chose to keep them:**

- **A file the assistant's tools produced** - a rendered document, an exported spreadsheet - expires after
  `GENERATED_FILE_RETENTION_DAYS`, **30 days by default**. Nobody asked for it; a tool made one as a side effect
  of answering, and the next question makes another. `0` keeps them forever.
- **A file a person sent** is kept indefinitely unless a deployment sets `ATTACHMENT_RETENTION_DAYS`. Somebody
  chose to send it, so nothing ages it out until a real retention policy names a number of days. Blank and `0`
  both read as unset.

Past the window an hourly sweep removes the bytes and the indexed text and marks the row expired, so a transcript
can still say the file existed rather than losing the record of it. A separate cleanup runs independently of
that window and always has: files whose conversation was deleted, orphans left by a queue outage, and blobs
from a crash between writing the bytes and writing the row.

**Two shorter-lived things expire without any policy being set:**

- **Session-scoped retrieval vectors** - the passages a web search or a per-user tool result contributes to one
  conversation - live for `RAG_SESSION_TTL_MINUTES`, **120 by default**, and are swept every
  `RAG_SWEEP_INTERVAL_MINUTES` (10). They are deliberately not part of the shared knowledge base.
- **Sign-in sessions** last `SESSION_TTL_SECONDS`, **seven days by default**, and the window *slides*: it is
  seven days without use, not seven days from signing in.

---

## 7. Connecting an integration

An integration is an external **MCP server** - a small HTTP service that describes the tools it offers and runs
them on request. Registering one is done from **Console -> Integrations** (`/console/integrations`), and there is
one generic form for every server, because a conformant server describes itself: the platform performs a
handshake, asks for the tool list, and needs no per-integration code.

**Before anything can be connected, the deployment needs a `SECRET_MASTER_KEY`.** It is what encrypts stored
credentials at rest, and it is never generated implicitly - a key minted on the fly looks fine in development and
destroys every stored credential the next time the container starts with a different one. Without it, servers can
still be registered and one that takes no credential still works, but every attempt to store a credential is
refused, and the page says so at the top rather than failing silently.

The form asks for four things:

| Field | What it means |
|---|---|
| **Name** | A label for the server, shown wherever its tools appear |
| **Base URL** | Where the server answers, for example `http://host:8811/mcp` |
| **Authentication** | `API key` (a token this platform sends), `OAuth` (a consent screen at the provider), or `None` |
| **Credential scope** | `Shared` - one workspace credential for everyone - or `Per-user`, where each member connects their own and calls fail closed without one |

Per-user scope is available for API-key servers; OAuth and credential-free servers are workspace-scoped. A
credential can be pasted on the form or added afterwards from the server's card, and is **write-only** in both
places: it can be replaced or removed, never read back.

**Presets prefill the same form.** GitHub, Gmail, Google Drive, Slack and Asana each ship with their base URL,
authentication shape and setup notes, so choosing one leaves only the provider-side registration to do. They are
data rather than integration code - the form and the tool path are identical whether a preset was used or not.

**Test connection performs the handshake and asks for the tool list, and stops there.** It never runs a tool, so
it is safe against a server with side effects. A success reports how many tools the server advertises; a failure
reports which kind it was - unreachable, timed out, credentials rejected, or a protocol mismatch - which is
usually enough to tell a wrong URL from a wrong token.

Once a server is registered its card carries a switch that takes all of its tools out of circulation without
deleting the registration, which is the fastest way to isolate a misbehaving upstream. Individual tools can also
be suppressed through the server's stored configuration; a newly advertised tool is offered by default.

**Registering, re-pointing and removing a server are administrator actions**, as is storing a shared workspace
credential; connecting a personal credential to a per-user server is not. Every one of those actions is logged
with the actor and, for a registration, the URL that was registered - which matters because pointing the platform
at a URL is a request the deployment will then make on its own behalf.
