# Setup

Getting a development checkout running from nothing: the prerequisites, the first boot, downloading models,
the local addresses, and the Windows and GPU specifics. The short version lives in the
[README](../README.md#quickstart); this is the long one.

Running the platform for other people is a different shape of the same stack - see
[DEPLOYMENT.md](DEPLOYMENT.md). Keeping either one healthy over time is [MAINTENANCE.md](MAINTENANCE.md).

---

## Quickstart

This is the **development shape**: a checkout running on the machine in front of
whoever is editing it, with the app hot-reloading and the backing services
published on loopback so they can be inspected. It is one `DEPLOY_ENV` line away
from the shape a real installation runs in, and
[Deploying to a server](DEPLOYMENT.md#deploying-to-a-server) is that story - the place to
start for a machine that will serve other people.

Prereqs: **Docker Desktop** (WSL2 backend on Windows), running **Docker Engine
23.0 or newer** (API 1.42 - any current Docker Desktop is well past it). An
**NVIDIA GPU is recommended** - the default profile uses it, and local LLMs are
impractically slow on CPU. No GPU (or passthrough not set up)? Use
`make up PROFILE=cpu` - everything still runs, just slower.

The engine floor is not cosmetic. Reclaiming unused volumes is limited to
anonymous ones from 23.0 onward; on an older daemon the same command also
removes **named** volumes that nothing is currently using, which with the stack
stopped means the database, the vector store and the uploaded files. The
maintenance targets check for this and refuse rather than assume, so an older
engine costs a cleanup rather than the data - but it is a floor worth meeting.

```bash
make init                        # writes .env from the template, with a fresh SECRET_MASTER_KEY
make up                          # GPU profile (default)  ->  make up PROFILE=cpu  without a GPU
make ps                          # see what's running
make down                        # stop
```

`make init` is the first-run step. Any `make` target creates a plain `.env` from
`infra/.env.example` when one is missing, but only `init` also generates the
`SECRET_MASTER_KEY` that encrypts stored integration credentials - and it
refuses to touch a `.env` that has already been edited, so it cannot replace a
key that is already protecting something. Passwords and ports are then filled in
by hand; [Configuring the stack](DEPLOYMENT.md#configuring-the-stack) walks through what a
given deployment actually needs to change.

There is one start command for every machine: `.env` describes what this box is
(`DEPLOY_ENV`, `PROFILE`) and `make up` reads it. To change a setting for a
single command, write it **after** the target, as above - the env-prefix form
(`PROFILE=cpu make up`) is ignored whenever `.env` defines that key.

> **After pulling updates into an existing checkout, run `make deploy` once.**
> `make up` reuses the container images already on the machine, so changes
> baked into those images (dependencies, a Dockerfile, a sidecar) only take
> effect after a rebuild. `make deploy` works out whether the update touched
> any of them: if it did, it rebuilds and refreshes the cached dependency
> volumes to match; if it did not, it recreates only what moved, applies any
> new database migrations and restarts the application services, which takes
> seconds. `make redeploy` forces the rebuild unconditionally.
> [MAINTENANCE.md](MAINTENANCE.md) covers this and the rest of
> running a deployment over time.

Today `make up` brings up Postgres (where conversations persist; database migrations apply automatically at
startup), Redis, Qdrant (where retrieval vectors live), Ollama, SearXNG (the self-hosted metasearch engine that
backs web search), the background worker (sweeping expired session vectors; parsing uploaded files), and the app itself -
the FastAPI backend (streaming chat + conversations + RAG APIs) and the Next.js web UI with its conversation
sidebar and citation chips. Two MCP sidecars come up with it: the MCP chassis - a reference tool server that
wraps an external API-backed datastore over Streamable HTTP - and doc-gen, which renders documents and charts
on request. The reranker sidecar starts when `RERANKER_URL` names one, which is the same signal retrieval
reads, so a deployment that leaves it unset retrieves dense-only (`RERANKER=0` / `RERANKER=1` force the compose
profile either way).

### Getting models onto the machine

A fresh stack holds no models, and nothing guesses which ones it should. Two are wanted: a **chat** model to
answer with, and an **embedding** model so uploaded files and web results can be searched. Both are downloaded
from the **Model Manager** at **http://localhost:3000/console/models**, which checks a name against the model
registry before it starts, shows real download progress that survives a page reload, and runs a quick
tool-calling check on what arrives.

The same thing headless, when a browser is not to hand. Run it from the repo root - the compose file pair follows
`DEPLOY_ENV`, and `make config` prints the one in use, so a deployment substitutes
`infra/docker-compose.prod.yml` for the second file:

```bash
docker compose --env-file .env -f infra/docker-compose.yml -f infra/docker-compose.override.yml \
  --profile gpu exec ollama-gpu ollama pull qwen3:4b-instruct
```

A stack serving models from another machine (`PROFILE=remote`) has no local Ollama container to reach, so that
route does not apply - see [`infra/remote-ollama/README.md`](../infra/remote-ollama/README.md).

A machine with a 4GB card is comfortable with a 4B-class chat model and a small embedder - for example
`qwen3:4b-instruct` (about 2.5GB) and `qwen3-embedding:0.6b` (about 0.6GB). A larger card affords a larger chat
model; nothing in the app has to be told which was chosen, because the console lists whatever the model server
actually holds.

Two of those models can be **paired**, and it is worth knowing why the console offers it. Some tags always reason
before answering and cannot be told not to; others never do. `qwen3:4b` is the first kind and
`qwen3:4b-instruct` the second, so pairing them gives one deployment both behaviours: everyday turns answer fast,
and turns where step-by-step reasoning is worth the latency route to the reasoning half. A single model whose
reasoning genuinely toggles per request needs no pair at all.

Once a chat model is downloaded, pick it as the workspace default from **Console -> Workspace defaults** (a
deployment holding exactly one chat model adopts it automatically on the first turn), then `make warm`.

### Local URLs

Once `make up` is running, the stack is reachable at:

| Service | URL | What's there |
|---|---|---|
| Web UI | http://localhost:3000 | The chat app |
| Backend API docs | http://localhost:8000/docs | Interactive OpenAPI (Swagger) docs |
| Backend health | http://localhost:8000/healthz | Liveness check: is the process up |
| Backend readiness | http://localhost:8000/readyz | Whether the database, job queue and vector store are reachable (503 if not), plus the model server and the optional sidecars for information |
| Qdrant dashboard | http://localhost:6333/dashboard | Vector collections browser |
| Ollama API | http://localhost:11434 | Local model-serving API (no UI) |

Ports follow `infra/.env.example` (`FRONTEND_PORT`, `BACKEND_PORT`, `QDRANT_HTTP_PORT`, `OLLAMA_PORT`) - override there if one is already taken.

Every entry above the web UI is published to the loopback address by default, so it answers on the machine
running the stack and nowhere else. `BIND_HOST` in `.env` widens that when the stack needs to be reached from
another machine.

These are the development layout's addresses. A deployment (`DEPLOY_ENV=prod`)
publishes none of them and serves everything through one HTTPS origin - see
[Deploying to a server](DEPLOYMENT.md#deploying-to-a-server).

### Windows: install `make` (and a Unix shell)

The `make` targets are the intended command surface, but Windows doesn't ship
with `make`, and the target recipes call Unix tools (`grep`, `awk`, `sed`,
`cp`, `cmp`, `curl`, `openssl`). So on Windows they need two things: **GNU
Make**, and a **Unix-style shell** to run it from - Git Bash bundles all of
those tools. Three quick installs cover it:

- **Git for Windows** - the standard Git installer, which also bundles **Git
  Bash**: a terminal that understands Unix commands (unlike `cmd` or
  PowerShell). This is where the `make` targets run.
- **Chocolatey** - a package manager for Windows (like `apt` on Linux): one
  command installs software from the terminal.
- **`make`** - installed through Chocolatey.

Steps (once Docker Desktop is installed):

1. Install **[Git for Windows](https://git-scm.com/download/win)** (accept the
   defaults). This adds a "Git Bash" entry to the Start menu.
2. Install **Chocolatey** by pasting its one-line command into an **elevated**
   PowerShell (right-click PowerShell -> "Run as administrator"); the exact line
   is on the [install page](https://chocolatey.org/install).
3. In that same admin PowerShell, install Make:
   ```powershell
   choco install make -y
   ```
4. Open **Git Bash** (Start menu -> "Git Bash"), `cd` into the cloned repo, and
   run the `make` targets there so `grep`/`awk`/`curl` resolve:

**First boot, start to finish** (in Git Bash, from the repo root):

```bash
make up            # 1. downloads container images + starts the stack (slow the first time)
                   # 2. open http://localhost:3000/console/models and download a chat model
                   #    and an embedding model (a few GB, one time - see above)
make warm          # 3. loads the chat model (+ primes its prompt cache) then the embedder, so the first reply and first search are fast
```

Then open **http://localhost:3000** and start chatting. `make warm` primes both
the chat model and the embedding model, so neither the first reply nor the first
ingest or search stalls on a cold load (`make warm-chat` or `make warm-embed`
warm just one); `make cool` is its opposite, unloading every resident model so
the GPU is free for other work - the next reply simply pays the cold start
again. Steps 1 and 2
are one-time downloads; later boots are just `make up` (and `make warm` if the
model has been idle), plus one `make deploy` after pulling updates so the
containers match the code. `make status` shows the stack's health at a glance, `make down`
stops everything, and `make help` lists every target -
[CONTRIBUTING.md](../CONTRIBUTING.md#make-targets) documents what each one does, and
[MAINTENANCE.md](MAINTENANCE.md) covers the ones that matter once a
deployment is live: updates, `.env` upkeep, backups, model changes and health.

Two alternatives if installing Make is not an option: run everything under
**WSL2** (a full Linux environment inside Windows, with its own shell and
package manager), or skip `make` and run the raw `docker compose` commands the
targets wrap (the recipes in the `Makefile` show the exact command each one
runs).

### GPU on Windows (WSL2) - checklist
`make up` uses the GPU profile, so verify passthrough **before the first boot**
(without it, the `gpu` profile fails with a cryptic `could not select device
driver "nvidia"`):
1. Latest **NVIDIA driver on the Windows host** (not inside WSL).
2. Docker Desktop -> Settings -> enable **WSL2 integration** for the distro.
3. Verify passthrough:
   ```bash
   docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
   ```
4. If that prints the GPU, `make up` works as-is. If not, use
   `make up PROFILE=cpu` - everything still runs, just slower. (CI and
   fake-provider tests need no GPU.)
5. **After boot, confirm Ollama actually grabbed the GPU.** Ollama probes for
   GPUs at startup with a watchdog; on a busy cold boot (especially under
   WSL2, where the first CUDA init can be slow) the probe can time out. Ollama
   then registers the CPU as its only inference device for the rest of that
   process's life - everything still works, so nothing errors, but generation
   crawls. Check and fix in one step:
   ```bash
   make gpu-check   # reports the discovery verdict; restarts Ollama once if it landed on CPU
   ```
   Or manually - note the `--since`, which matters: a container keeps its log
   across restarts, so without it a healthy line from an earlier boot shows up
   and hides the current state.
   ```bash
   c=ai-chatbot-platform-ollama-gpu-1
   docker logs --since "$(docker inspect -f '{{.State.StartedAt}}' $c)" $c 2>&1 \
     | grep "inference compute"
   # library=CUDA plus the GPU name is healthy; library=cpu alone is the
   # fallback. If it says cpu, wait for the machine to go idle and restart
   # just Ollama:
   docker restart $c
   ```
