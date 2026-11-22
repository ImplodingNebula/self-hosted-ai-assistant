# Entrypoints for the platform. See docs/ROADMAP.md for the build order.
# `make help` lists targets. Windows: run under Git Bash / WSL, or use the raw
# `docker compose` commands these wrap.

# Which compose files every target reads. The LIST is chosen by DEPLOY_ENV,
# defined below the .env include (make expands a `=` variable at use time, so it
# is fine to name it here and decide it there).
COMPOSE = docker compose --env-file .env $(COMPOSE_FILES)
# WHERE MODELS ARE SERVED. GPU is the default: every target machine (dev
# laptops, the deploy VM) has an NVIDIA GPU, and LLM inference on CPU is
# impractically slow. `cpu` is the fallback for a host without a GPU or working
# passthrough; `remote` means the models live on another machine entirely, so
# no Ollama container starts here. Normally a line in .env; for one command,
# `make up PROFILE=cpu`.
PROFILE ?= gpu
# Read the configured stack so make targets follow .env (ignored if absent).
# NOTE: every key in .env becomes a plain make variable here, which is why the
# override form matters below - a makefile assignment outranks the environment.
-include .env
# WHICH LAYOUT this machine runs: `dev` = the base compose file plus the
# development override (hot-reload bind mounts, published ports for host
# tooling); `prod` = the base file plus the hardening overlay (Caddy in front,
# nothing else published, restart: always). Declared once in .env, so `make up`
# is the same command everywhere and the machine's own configuration decides
# what it builds - a routine deploy on a server can no longer quietly recreate
# its containers from the development layout.
#
# It is a SEPARATE AXIS from PROFILE. PROFILE says where models are served (a
# local GPU container, CPU, or a remote server); DEPLOY_ENV says how the stack
# itself is assembled. A server whose GPU lives elsewhere is DEPLOY_ENV=prod
# with PROFILE=remote, and either can change without touching the other.
#
# THE OVERRIDE FORM MATTERS, and the wrong one fails silently. `-include .env`
# above assigns this as an ordinary make variable, and a makefile assignment
# outranks the environment - so the env-prefix form `DEPLOY_ENV=prod make up` is
# IGNORED whenever .env defines the key, and the stack comes up in whatever .env
# says with no warning. The form that works for a one-off is the make-variable
# one, written after the target: `make up DEPLOY_ENV=prod`. PROFILE behaves
# identically, for the same reason.
DEPLOY_ENV ?= dev
ifeq ($(DEPLOY_ENV),dev)
COMPOSE_FILES = -f infra/docker-compose.yml -f infra/docker-compose.override.yml
else ifeq ($(DEPLOY_ENV),prod)
COMPOSE_FILES = -f infra/docker-compose.yml -f infra/docker-compose.prod.yml
else
# Refused at parse time, so EVERY target stops rather than one of them picking a
# layout nobody asked for. A typo here would otherwise pick the dev layout and
# look like a working deploy.
$(error DEPLOY_ENV is "$(DEPLOY_ENV)", which names no compose layout - use dev or prod. It is normally a line in .env; for one command write `make <target> DEPLOY_ENV=prod` (the env-prefix form is ignored when .env sets the key))
endif
# Ollama host port (published by compose) + how long a warmed model stays
# resident: forever by default (`make cool` unloads on demand). Override on
# the command line, e.g. `make warm OLLAMA_KEEP_ALIVE=15m` for a time-capped
# warm on a testing/shared box.
# CHAT_MODEL exists for one thing: an explicit `make warm CHAT_MODEL=x`
# override (see WARM_MODEL_FIELD below); a partial name is enough, since the
# backend resolves it against the models the server holds and the reply names
# the one it warmed. It is deliberately NOT given a default here - no tag this
# file could name is more likely to be installed than the one the deployment
# itself resolves, and a default would only be a guess at somebody's inventory.
OLLAMA_PORT ?= 11434
OLLAMA_KEEP_ALIVE ?= -1
BACKEND_PORT ?= 8000
# Reranker sidecar: an opt-in compose profile, started exactly when RERANKER_URL
# names one. That key is also the whole gate the backend reads, so the container
# and the code now key on ONE fact and cannot disagree - a URL set with the
# sidecar down was the old failure, and a sidecar started with the backend
# ignoring it was the other. This used to mirror a committed model file's
# `enabled:` flag by hardware-class name; nothing needs mirroring now.
# RERANKER=0 / RERANKER=1 still force it off / on and still outrank the key.
ifeq ($(RERANKER),0)
RERANKER_FLAG =
else ifeq ($(RERANKER),1)
RERANKER_FLAG = --profile reranker
else ifneq ($(RERANKER_URL),)
RERANKER_FLAG = --profile reranker
else
RERANKER_FLAG =
endif
PROFILE_FLAGS = --profile $(PROFILE) $(RERANKER_FLAG)
# warm sends an explicit model ONLY for a command-line override ('make warm
# CHAT_MODEL=x'). By default the backend warms its own configured default, so
# a host .env edited after the containers were created cannot prime a model
# the backend will not route to.
ifeq ($(origin CHAT_MODEL),command line)
WARM_MODEL_FIELD = "model":"$(CHAT_MODEL)",
else
WARM_MODEL_FIELD =
endif
# The files that actually go INTO an image: the dependency manifests, the
# Dockerfiles, the .dockerignore files that define each build context, and the
# sidecars under services/. Everything else the dev stack runs is bind-mounted
# from the working tree, so the containers already see it and rebuilding for it
# would cost minutes and change nothing. `make deploy` reads exactly this list
# to decide whether a build is needed, so it is kept in one place rather than
# restated per target.
#
# The compose files are deliberately NOT here, and that is the list's one blind
# spot. They do carry each service's `build:` stanza, so editing a build context
# or target there really does make an image stale - but they change far more
# often for reasons a build does not care about (ports, env, healthchecks,
# volumes), which `up` applies on its own in seconds. Listing them would send
# the most common cheap deploy down the expensive path to cover a rare edit.
# After changing a `build:` stanza, run `make redeploy` rather than `make
# deploy`.
IMAGE_INPUTS = apps/backend/pyproject.toml apps/backend/uv.lock \
               apps/frontend/package.json apps/frontend/package-lock.json \
               apps/backend/Dockerfile apps/frontend/Dockerfile \
               apps/backend/.dockerignore apps/frontend/.dockerignore \
               services
# Records the commit the running images were built from. Git-ignored: it
# describes THIS machine's containers, not the source tree.
DEPLOY_STAMP = .deploy-stamp
# How long a failed warm waits before its single retry (see `make warm`).
WARM_RETRY_SLEEP ?= 15

# HOW HOST TOOLING REACHES THE BACKEND'S API. The warm and status targets ask
# the backend over HTTP, and two things decide which address they can use:
#
#   dev   the backend publishes a host port, so curl on the host is the
#         cheapest possible probe - no container, no interpreter, and it works
#         even when the stack is too broken to exec into.
#   prod  nothing is published but the proxy, so there is no port to curl. The
#         proxy is not a substitute either: its site address is the
#         deployment's real hostname, so a request to `localhost` matches no
#         site and never reaches a route. The probe is therefore made from
#         INSIDE the backend container, against its own port - the same move
#         the model targets already make for a remote model server.
#
# ... and LOGIN forces the in-container form whatever the layout. With
# OIDC_ISSUER set, every /api/v1 route requires a signed-in session, and these
# probes have no browser and nobody to sign in as. What they do have is the one
# thing the backend accepts instead: a request made from inside the backend
# container reaches its own loopback interface, and the app grants that the
# operator context (reaching it already requires `docker exec` on the host,
# which is root-equivalent there). Curl from the host arrives from the bridge
# gateway instead, which is not loopback and is answered with 401 - so without
# this line `make warm` would break the moment somebody tried login on a dev
# layout, which is exactly where login gets tried first.
#
# The four commands are defined here rather than branched inside the recipes so
# that each recipe still runs ONE command, and the dev definitions stay exactly
# the text they have always been. scripts/backend_http.py explains its three
# output modes, which are shaped to match the curl invocations they replace so
# the recipes need no second parsing path. It arrives on the container's stdin,
# so nothing is mounted or baked in; `uv run` can execute it as the image's
# unprivileged user because the prod stage sets UV_CACHE_DIR and UV_NO_SYNC.
#
# The request paths below are written WITHOUT a leading slash on purpose. Git
# Bash rewrites an argument that looks like an absolute Unix path into a Windows
# path when it is handed to a native executable, and `docker` is one - so
# `/healthz` reaches the script as `C:/Program Files/Git/healthz` and the probe
# fails on a filesystem path nobody typed. The script puts the slash back.
BACKEND_PROBE_MODE = curl
ifeq ($(DEPLOY_ENV),prod)
BACKEND_PROBE_MODE = exec
endif
ifneq ($(strip $(OIDC_ISSUER)),)
BACKEND_PROBE_MODE = exec
endif

ifeq ($(BACKEND_PROBE_MODE),exec)
BACKEND_EXEC = $(COMPOSE) --profile $(PROFILE) exec -T backend uv run python -
WARM_CHAT_PROBE = $(BACKEND_EXEC) POST api/v1/models/warm '{$(WARM_MODEL_FIELD)"keep_alive":"$(OLLAMA_KEEP_ALIVE)"}' < scripts/backend_http.py
WARM_EMBED_PROBE = $(BACKEND_EXEC) POST api/v1/models/warm-embed '{"keep_alive":"$(OLLAMA_KEEP_ALIVE)"}' < scripts/backend_http.py
WARM_SEARCH_PROBE = $(BACKEND_EXEC) --coded POST api/v1/models/warm-search < scripts/backend_http.py
# stderr is discarded to match `curl -fs`, which the status probe uses so that a
# stopped stack prints only this target's own friendly line. Here the noise
# would be compose's "service backend is not running" rather than curl's.
HEALTHZ_PROBE = $(BACKEND_EXEC) --quiet GET healthz < scripts/backend_http.py 2>/dev/null
else
WARM_CHAT_PROBE = curl -fsS -X POST "http://localhost:$(BACKEND_PORT)/api/v1/models/warm" -H 'content-type: application/json' -d '{$(WARM_MODEL_FIELD)"keep_alive":"$(OLLAMA_KEEP_ALIVE)"}'
WARM_EMBED_PROBE = curl -fsS -X POST "http://localhost:$(BACKEND_PORT)/api/v1/models/warm-embed" -H 'content-type: application/json' -d '{"keep_alive":"$(OLLAMA_KEEP_ALIVE)"}'
WARM_SEARCH_PROBE = curl -sS -w '\n__HTTP__%{http_code}' -X POST "http://localhost:$(BACKEND_PORT)/api/v1/models/warm-search"
HEALTHZ_PROBE = curl -fs -o /dev/null "http://localhost:$(BACKEND_PORT)/healthz"
endif

.PHONY: help env-check sync-env init up down restart gpu-check rebuild deploy deploy-stamp redeploy perf tool-check embedder-check prune disk host-disk compact backup restore logs ps models status config db-upgrade dev-session warm warm-chat warm-embed warm-search cool fmt lint backend-test frontend-test audit

help: ## Show this help
	@# firstword: `-include .env` adds .env to MAKEFILE_LIST, and grep prefixes
	@# matches with the filename when given multiple files - breaking the output.
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(firstword $(MAKEFILE_LIST)) | awk 'BEGIN{FS=":.*?## "}{printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "  Standing settings live in .env and every target reads them:"
	@echo "    DEPLOY_ENV   dev (hot-reload layout, the default) | prod (Caddy in front, nothing else published)"
	@echo "    PROFILE      gpu (default) | cpu | remote (models served elsewhere)"
	@echo "  This run: DEPLOY_ENV=$(DEPLOY_ENV) PROFILE=$(PROFILE)"
	@echo ""
	@echo "  To override one for a single command, write it AFTER the target:"
	@echo "      make up DEPLOY_ENV=prod"
	@echo "  The env-PREFIX form (DEPLOY_ENV=prod make up) is silently ignored whenever"
	@echo "  .env defines the key - make gives a file's assignment precedence over the"
	@echo "  environment, and .env is read as part of this makefile."

.env: ## Create .env from the template on first run
	@test -f .env || cp infra/.env.example .env

# A .env created from an older template silently misconfigures the stack (e.g.
# a pre-split DEFAULT_CHAT_MODEL routes every turn to the always-thinking
# variant). This cannot fix stale VALUES, but missing KEYS are a reliable tell
# that the file needs a re-sync.
env-check: .env ## Warn when .env lacks variables that infra/.env.example defines
	@missing=""; \
	for key in $$(grep -E '^[A-Z0-9_]+=' infra/.env.example | cut -d= -f1); do \
		grep -qE "^$$key=" .env || missing="$$missing $$key"; \
	done; \
	if [ -n "$$missing" ]; then \
		echo "WARNING: .env is missing variables the template now defines:$$missing"; \
		echo "         Run 'make sync-env' to append them with template defaults, and"; \
		echo "         check existing values against infra/.env.example (MIGRATION NOTE):"; \
		echo "         stale values route chat to the wrong models or size turns wrongly."; \
	fi
	@# Where the model server actually IS has to agree with which profile starts
	@# one. A mismatch is not a warning-shaped problem: PROFILE=gpu against a
	@# remote OLLAMA_URL starts a container nothing uses AND points every model
	@# target at it, so the targets report a machine that is not serving anything.
	@# This runs here because env-check is a prerequisite of up/rebuild/warm, so
	@# one refusal stops all three before they act.
	@sh scripts/remote_check.sh check "$(PROFILE)"

# Append-only by design: existing lines (secrets, tuned values) are NEVER
# edited or removed, so the worst case is an unwanted default to delete by
# hand. Values that exist but drifted from the template still need a manual
# review - a script cannot know which local values are intentional. Keys the
# template no longer mentions AT ALL (not even commented out) are reported as
# likely obsolete but left in place: they may belong to a deployment branch
# or local tooling, so removal stays a human call.
sync-env: .env ## Append variables infra/.env.example defines but .env lacks (never edits existing lines); flag likely-obsolete keys
	@if [ -n "$$(tail -c 1 .env)" ]; then echo "" >> .env; fi; \
	added=""; \
	for key in $$(grep -E '^[A-Z0-9_]+=' infra/.env.example | cut -d= -f1); do \
		if ! grep -qE "^$$key=" .env; then \
			grep -E "^$$key=" infra/.env.example | head -n 1 >> .env; \
			added="$$added $$key"; \
		fi; \
	done; \
	if [ -n "$$added" ]; then \
		echo "Added to .env with template defaults - review the values:$$added"; \
	else \
		echo ".env already defines every variable the template does."; \
	fi; \
	obsolete=""; \
	for key in $$(grep -E '^[A-Z0-9_]+=' .env | cut -d= -f1); do \
		grep -qE "^[# ]*$$key=" infra/.env.example || obsolete="$$obsolete $$key"; \
	done; \
	if [ -n "$$obsolete" ]; then \
		echo "NOTE: .env defines keys the template no longer mentions:$$obsolete"; \
		echo "      Likely obsolete (renamed or removed) - verify and delete by hand; left"; \
		echo "      untouched in case a branch or local tooling still reads them."; \
	fi

# One-command first configuration: writes .env from the template and generates
# a fresh SECRET_MASTER_KEY - deterministic and promptless (the locked no-wizard
# rule; it composes with env-check/sync-env instead of replacing them). It takes
# no arguments, because there is nothing left to select: which models a
# deployment runs is chosen on the console against what the model server
# actually holds, not from a hardware class named at setup time.
#
# It REFUSES to touch an existing .env that differs from the template: a
# generated key must never silently replace one that already encrypted
# credentials. A byte-identical copy is fine to replace - `-include .env` makes
# make auto-create one from the template before ANY target runs, so on a fresh
# clone that pristine copy always exists by the time this recipe executes.
init: ## Write a first .env from the template, with a fresh SECRET_MASTER_KEY
	@if [ -f .env ] && ! cmp -s .env infra/.env.example; then \
		echo ".env already exists and has been edited - refusing to overwrite it."; \
		echo "Edit it by hand, or remove the file and rerun."; \
		exit 1; \
	fi; \
	cp infra/.env.example .env; \
	key=$$(openssl rand -base64 32); \
	if [ -z "$$key" ]; then \
		echo "openssl produced no key - is openssl on PATH? .env is written but has"; \
		echo "no SECRET_MASTER_KEY; generate one by hand before using Integrations."; \
		exit 1; \
	fi; \
	sed -i "s|^SECRET_MASTER_KEY=.*|SECRET_MASTER_KEY=$$key|" .env; \
	grep -q "^SECRET_MASTER_KEY=." .env || { \
		echo "Failed to write SECRET_MASTER_KEY into .env - the template no longer"; \
		echo "declares it in the form this recipe substitutes. Fix the recipe."; \
		exit 1; \
	}; \
	echo "Wrote .env with a fresh SECRET_MASTER_KEY."; \
	echo "Review passwords and ports in .env, then: make up && make gpu-check"; \
	echo "Then download a chat model and an embedding model from the Model Manager"; \
	echo "at http://localhost:3000/console/models, and run 'make warm'."

# ONE start command, whatever the machine is. DEPLOY_ENV and PROFILE in .env say
# what this box is, and `up` reads them - so there are no per-shape variants to
# pick between, and no variant to forget on the box where it mattered.
#
# For a one-off, name the setting AFTER the target (`make up PROFILE=cpu`,
# `make up DEPLOY_ENV=prod`): a command-line variable outranks everything,
# including the .env line. The env-PREFIX form does NOT - see the note beside
# DEPLOY_ENV at the top of this file.
#
# PROFILE=remote deserves a word, because it names no service in the compose
# file and that IS the mechanism: every unprofiled service starts as usual while
# ollama-gpu and ollama-cpu, which are profile-gated, do not. Nothing had to be
# added to compose for it.
up: .env env-check ## Start the stack; one-off overrides go after the target (make up PROFILE=cpu|remote, make up DEPLOY_ENV=prod)
	$(COMPOSE) $(PROFILE_FLAGS) up -d

down: ## Stop and remove containers (plus any orphans from removed/renamed services)
	$(COMPOSE) --profile cpu --profile gpu --profile minio --profile reranker down --remove-orphans

restart: down up ## Restart the stack

# Ollama probes for GPUs at startup with a watchdog; on a busy boot the probe
# times out and Ollama registers the CPU as its ONLY inference device for the
# rest of the process's life (generation crawls, no error). This target's
# non-zero exit is what stops `redeploy` from warming - and what stops a
# CPU-fallback perf reading from being mistaken for a regression - so it has
# to be right about which state it is in.
#
# The detection lives in scripts/gpu_check.sh rather than here, because it is a
# parser and a parser that cannot be run against a captured log is a parser
# nobody can check. That script carries the full reasoning; the short version
# is that evidence must come from the CURRENT process (a container keeps its
# log across restarts, so a whole-log search finds a healthy line from hours
# ago and passes a server that is CPU-only right now) and must be read off the
# `library=` field of an `inference compute` line (the discovery FAILURE
# messages name paths like `cuda_v13`, so a search for "CUDA" matches the very
# messages reporting the failure). Both of those were live bugs here.
#
# All that stays in the Makefile is which container to look at.
gpu-check: ## Verify the model server registered a GPU; auto-restart it once if discovery fell back to CPU
	@# A remote server's GPU discovery happens on a machine this stack has no log
	@# from, so there is nothing here to parse. What CAN be checked is the thing
	@# the rest of the refresh depends on: that the server answers at all, and
	@# what it currently has resident. The non-zero exit on failure is what stops
	@# `redeploy` from warming, exactly as the local GPU failure does - warming an
	@# unreachable server just fails twice, slowly.
	@# ONE shell for all three branches, because `exit` only ends the recipe LINE
	@# it runs in. Split across lines, the skip branch printed "only applies to
	@# the gpu profile" and make then ran the next line anyway, so a non-gpu
	@# profile went looking for an ollama-gpu container and failed on not finding
	@# one - which, since a non-zero gpu-check stops `redeploy` before it warms,
	@# broke the refresh for every profile but gpu.
	@if [ "$(PROFILE)" = "remote" ]; then \
		echo "PROFILE=remote: checking the remote model server rather than a local GPU."; \
		bcid=$$($(COMPOSE) --profile $(PROFILE) ps -q backend 2>/dev/null); \
		if [ -z "$$bcid" ]; then \
			echo "The backend container is not running, so the remote model server cannot be"; \
			echo "reached from here - start the stack first ('make up PROFILE=remote')."; \
			exit 1; \
		fi; \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - version < scripts/ollama_http.py || exit 1; \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - ps < scripts/ollama_http.py || exit 1; \
		echo "The remote model server is reachable. Anything resident is listed above;"; \
		echo "a '100% CPU' row there means that model is not on the remote GPU."; \
		exit 0; \
	fi; \
	if [ "$(PROFILE)" != "gpu" ]; then \
		echo "PROFILE=$(PROFILE): gpu-check only applies to the gpu profile."; \
		exit 0; \
	fi; \
	cid=$$($(COMPOSE) --profile gpu ps -q ollama-gpu); \
	if [ -z "$$cid" ]; then echo "Ollama is not running - start the stack first ('make up')."; exit 1; fi; \
	GPU_CHECK_TIMEOUT="$(GPU_CHECK_TIMEOUT)" sh scripts/gpu_check.sh "$$cid"

# Images are built ONE AT A TIME, on purpose. Recent Compose builds through
# `buildx bake` by default, which fans every image build out concurrently: four
# Dockerfile frontend processes start at once, and on a memory-constrained host
# (an 8GB WSL2 VM already hosting the whole stack plus a resident model) one of
# them gets killed. The build then dies with "frontend grpc server closed
# unexpectedly" and takes the rebuild with it. COMPOSE_BAKE=false plus one
# `build` call per service keeps a single builder alive at a time: slower, but
# it finishes. COMPOSE_PARALLEL_LIMIT=1 covers the fallback branch below, which
# hands the whole build to Compose and would otherwise fan out again on the
# classic (non-bake) path. The buildable services are read off the merged config
# rather than listed here, so adding or removing one needs no edit to this
# recipe.
rebuild: .env env-check ## Rebuild images (one service at a time) and refresh dependency volumes (after changing deps)
	@export COMPOSE_BAKE=false COMPOSE_PARALLEL_LIMIT=1; \
	services=$$($(COMPOSE) $(PROFILE_FLAGS) config 2>/dev/null \
		| awk '/^[^ ]/{s=0} /^services:/{s=1} s&&/^  [A-Za-z0-9._-]+:$$/{n=$$1; sub(/:$$/,"",n)} s&&/^    build:/{print n}'); \
	if [ -z "$$services" ]; then \
		echo "Could not read the buildable services off the compose config - building them together instead."; \
		$(COMPOSE) $(PROFILE_FLAGS) build || { echo "BUILD FAILED - stopping here."; exit 1; }; \
	else \
		for svc in $$services; do \
			echo "==> building $$svc"; \
			$(COMPOSE) $(PROFILE_FLAGS) build "$$svc" \
				|| { echo "BUILD FAILED for service '$$svc' - stopping here; nothing was recreated."; exit 1; }; \
		done; \
	fi
	COMPOSE_BAKE=false $(COMPOSE) $(PROFILE_FLAGS) up -d --renew-anon-volumes
	@# --renew-anon-volumes recreates the dev dependency volumes (.venv,
	@# node_modules, .next) and ORPHANS the previous set; reclaim those now so
	@# they cannot pile up (tens of GB over many rebuilds). The containers just
	@# recreated hold the fresh anon volumes, and named data volumes are never
	@# anonymous, so an anon-only prune leaves both untouched. Anon-only is a
	@# property of Engine 23.0+ though, so the same guard applies here - `soft`,
	@# because the rebuild has already succeeded and an unreclaimed scratch
	@# volume is not worth failing it over.
	@sh scripts/prune_guard.sh volumes soft

# Most changes need no build at all IN THE DEV LAYOUT. The dev overlay
# bind-mounts apps/backend and apps/frontend into their containers, so
# application code is already whatever the working tree says - only a change to
# something baked INTO an image (dependencies, a Dockerfile, a sidecar under
# services/) makes the images stale. `make deploy` tells those two cases apart
# and takes the cheap path when it can: minutes of rebuild against seconds of
# recreate, and the recreate keeps the warmed models resident because it does
# not touch the model server at all.
#
# A BIND MOUNT DELIVERS FILES, NOT BEHAVIOUR, so the cheap path does two more
# things before it calls itself deployed. Neither is something the mount can do,
# and both used to be left to whoever noticed:
#
#   MIGRATIONS  Alembic runs from the backend container's start command, so a
#               pull that adds one leaves the new code on the old schema for as
#               long as nothing restarts that container - and where a reloader
#               picks host edits up by itself, the new code begins serving
#               against the old schema right away. The restart below would
#               re-run them, but SILENTLY: a failed migration exits the
#               container and `compose restart` still reports success.
#               `db-upgrade` applies them in the container that is already
#               running, synchronously, and its failure stops this chain - and
#               running it first means the schema is at head before any
#               restarted process serves a request.
#               ONE RESIDUAL RACE, left in knowingly. When `up` above has just
#               RECREATED the backend, that container is running its own
#               `alembic upgrade head` while this step execs a second one in
#               beside it (nothing here waits on its health, and env.py takes
#               no advisory lock). Postgres serialises the two and the loser
#               errors, which fails the deploy LOUDLY with the schema already
#               correct - a re-run then finds head and passes. The fix is a
#               wait on the backend's healthcheck, and a wait loop that can
#               itself hang a deploy was judged the worse trade against a rare,
#               self-clearing, non-destructive failure.
#   RESTART     File-change notifications do not cross every bind mount - they
#               do not cross the WSL2 one at all - so a reloading uvicorn and
#               the web UI's dev watcher can hold a host edit in the container
#               and still be serving the code they started with. Restarting the
#               three application services is what actually puts this working
#               tree on the air, and it is seconds: none of them holds a model.
#
# THAT ENTIRE PREMISE IS FALSE IN A DEPLOYMENT, so `deploy` always rebuilds
# there. Nothing is bind-mounted in the prod layout: the application source is
# COPIED into the images, and IMAGE_INPUTS deliberately does not list it (in
# dev, listing it would rebuild for every edit and change nothing). Left to the
# same test, a server that pulled a pure application change would find no image
# input moved, take the cheap path, recreate its containers from the OLD image,
# and report a successful deploy - the worst shape a deploy bug can take, since
# nothing errors and nothing looks wrong.
#
# "What is deployed" is read from $(DEPLOY_STAMP), written whenever a deploy
# finishes. Anything that makes that answer uncertain - no stamp, no git, a
# stamped commit git no longer knows - falls back to the FULL REBUILD and says
# which it was: the cheap path is only ever taken on positive evidence that no
# image input moved. Uncommitted edits to an image input count as a change too,
# so a Dockerfile being edited right now cannot slip through on a stale stamp.
#
# Both paths propagate failure - the cheap one by &&-chaining its steps, the
# rebuild one because the sub-make is the last command in its branch. A deploy
# that could not bring the stack up must not report success, and must not leave
# a stamp claiming it did. The stamp goes down as soon as the stack is up and
# BEFORE the GPU check, for the same reason it does in `redeploy` - a CPU
# fallback is fixable in place and is not evidence that the images are stale.
#
# The cheap path deliberately does NOT warm. Usually there is nothing to warm:
# with the compose config unchanged, `up` recreates nothing and every model
# stays exactly as resident as it was, and the restart names the three
# application services rather than the whole stack, so it leaves the model
# server alone too. When `up` does recreate the model server, a cold load is
# minutes of work that belongs to whoever asked for it, so the closing line says
# what actually happened and names `make warm` rather than spending that time
# unasked.
deploy: ## Deploy the working tree, rebuilding images only when an image input actually changed
	@reason=""; \
	if [ "$(DEPLOY_ENV)" != "dev" ]; then \
		reason="DEPLOY_ENV=$(DEPLOY_ENV) bakes the application code into the images, so only a rebuild can deploy a change"; \
	elif ! git rev-parse --git-dir >/dev/null 2>&1; then \
		reason="this is not a git checkout, so nothing can say what is already deployed"; \
	elif [ ! -f $(DEPLOY_STAMP) ]; then \
		reason="no deploy stamp yet, so nothing records what the running images were built from"; \
	else \
		stamp=$$(cat $(DEPLOY_STAMP)); \
		if ! git cat-file -e "$$stamp^{commit}" 2>/dev/null; then \
			reason="the stamped commit ($$stamp) is unknown to git - the branch was rebased or pruned"; \
		elif ! diffed=$$(git diff --name-only "$$stamp" HEAD -- $(IMAGE_INPUTS)) \
			|| ! dirty=$$(git status --porcelain --no-renames -- $(IMAGE_INPUTS)); then \
			reason="git could not compare the working tree against the stamped commit"; \
		else \
			changed=$$(printf '%s\n%s\n' "$$diffed" "$$(echo "$$dirty" | cut -c4-)" \
				| sed '/^$$/d' | sort -u | tr '\n' ' '); \
			if [ -n "$$changed" ]; then \
				reason="image inputs changed since $$(echo "$$stamp" | cut -c1-7): $$changed"; \
			fi; \
		fi; \
	fi; \
	if [ -n "$$reason" ]; then \
		echo "Rebuilding images - $$reason."; \
		$(MAKE) --no-print-directory redeploy; \
	else \
		echo "Skipping the build: no image input changed since the deployed commit."; \
		echo "  The dev stack bind-mounts apps/backend and apps/frontend, so the containers already"; \
		echo "  HOLD this working tree's code - but the processes inside them need not be running it"; \
		echo "  yet, and the database can be a migration behind. Applying .env and compose config,"; \
		echo "  then the migrations, then restarting the application services."; \
		$(MAKE) --no-print-directory sync-env \
			&& $(MAKE) --no-print-directory up \
			&& $(MAKE) --no-print-directory db-upgrade \
			&& $(COMPOSE) $(PROFILE_FLAGS) restart backend worker frontend \
			&& $(MAKE) --no-print-directory deploy-stamp \
			&& $(MAKE) --no-print-directory gpu-check \
			&& echo "Deployed without a rebuild: config applied, migrations at head, the application services restarted onto this tree's code. The model server was not restarted, so whatever was resident still is - unless 'up' recreated it, in which case 'make warm' re-primes it."; \
	fi

# Records the commit the running images were built from, for `make deploy` to
# compare against next time. Whenever there is no commit that honestly
# describes what was built, the stamp is REMOVED rather than left stale, so the
# next deploy falls back to a full rebuild.
#
# That includes the case where an image input is uncommitted. The images then
# hold content no commit contains, and stamping HEAD would claim otherwise -
# discard the edit afterwards and the next deploy would compare clean against
# clean, take the cheap path, and leave the containers running the abandoned
# dependency set. Refusing to stamp costs a rebuild per deploy while a
# Dockerfile or a lockfile is being iterated on, which is the correct answer.
deploy-stamp:
	@if [ -n "$$(git status --porcelain --no-renames -- $(IMAGE_INPUTS) 2>/dev/null)" ]; then \
		rm -f $(DEPLOY_STAMP); \
		echo "No stamp written: an image input is uncommitted, so no commit describes what was just built."; \
		echo "  The next 'make deploy' will rebuild until those edits are committed."; \
	elif git rev-parse HEAD > $(DEPLOY_STAMP) 2>/dev/null; then \
		echo "Deploy stamp: $$(cut -c1-7 $(DEPLOY_STAMP))"; \
	else \
		rm -f $(DEPLOY_STAMP); \
		echo "No commit to stamp - the next 'make deploy' will rebuild to be safe."; \
	fi

# The whole after-a-change refresh in one command. Ordered deliberately:
# sync-env first (a rebuild bakes the env into the recreated containers, so a
# stale .env would be baked in); then free memory for the build (see below);
# then rebuild, then confirm Ollama actually found the GPU (a busy boot can
# silently fall back to CPU), then prime the models so the first real turn is
# not a cold load. Each step is invoked as a sub-make rather than a prerequisite
# so they run in this order even under -j. gpu-check exits non-zero when the GPU
# is genuinely unreachable, which stops the warm on purpose: warming a CPU
# fallback wastes minutes and hides the fault.
#
# The `stop` is the fix for a recurring first-rebuild crash. BuildKit's grpc
# frontend dies ("grpc server closed unexpectedly") when the rebuild builds
# images while the whole stack - Ollama's resident model above all - still holds
# memory on a constrained (8GB WSL2) host. Stopping first gives the build that
# memory back; the rebuild's `up` starts everything again and `warm` re-primes
# the model, so the stop costs nothing a redeploy was not already going to do.
# It is `stop`, not `down`: containers, networks and every data volume survive,
# so nothing is removed or re-initialised - this is the automatic, gentler form
# of the down-then-redeploy that clears the crash by hand.
# The stamp is written the moment `rebuild` succeeds, BEFORE the GPU check and
# the warm: at that point the images genuinely match this commit, and neither
# of the later steps can un-build them. Stamping at the very end would make a
# CPU fallback or a failed warm - both fixable in place - masquerade as "the
# images are stale" and cost a needless rebuild next time.
redeploy: ## Refresh a running stack after a change: sync .env, free memory, rebuild, verify the GPU, warm
	@$(MAKE) sync-env
	-$(COMPOSE) $(PROFILE_FLAGS) stop
	@$(MAKE) rebuild
	@$(MAKE) deploy-stamp
	@$(MAKE) gpu-check
	@$(MAKE) warm

# A change can be correct and still make the assistant unusably slow, and that
# regression is invisible to the test suite. This drives the real SSE endpoint
# and reports time-to-first-token and tokens/second, which is the pair a person
# actually feels. Run it before and after a phase against the same machine.
# Only the plain turn runs by default - it is the one `make warm` leaves warm.
# PERF_THINK=1 adds a thinking turn (a second model load where a reasoning
# partner is paired with the chat model),
# PERF_WEB=1 adds a web turn, PERF_TOOL=1 adds an agentic tool-calling turn
# (model emits a web_search call, the tool runs live, a second generation
# answers); PERF_MAX_TTFT / PERF_MAX_TOTAL turn it into a gate.
# PERF_AGENCY=<auto|agentic|assisted|deterministic> clamps every scenario to one
# agency mode, so the three-way control (base / change / change clamped to
# `deterministic`) costs three runs rather than three .env edits and restarts.
perf: ## Time a chat turn (time-to-first-token, tokens/sec); PERF_THINK=1 / PERF_WEB=1 / PERF_TOOL=1 add turns; PERF_AGENCY=<mode> clamps them
	@$(COMPOSE) --profile $(PROFILE) exec -T \
		-e PERF_WEB="$(PERF_WEB)" -e PERF_THINK="$(PERF_THINK)" \
		-e PERF_TOOL="$(PERF_TOOL)" -e PERF_AGENCY="$(PERF_AGENCY)" \
		-e PERF_MAX_TTFT="$(PERF_MAX_TTFT)" -e PERF_MAX_TOTAL="$(PERF_MAX_TOTAL)" \
		backend uv run python - < scripts/perf_smoke.py

# An agency ceiling and a thinking_with_tools clearance are claims until
# something measures them; this is that measurement. It drives the served model
# through the cases that break small tool-callers (must-call, must-NOT-call,
# selection, the invented-tool guard, continuation, a loop guard,
# reasoning-with-tools) with canned tools and canned results - no real tool ever
# runs. Opt-in on purpose: it is never run at boot or on a chat turn (on a box
# whose card fits one model, a probe would evict the very model it is about to
# serve), so run it by hand AFTER `make warm` so the reading reflects the warm,
# serve-time state. FAST by default (2 cases, ~45s, earns at most 'assisted');
# TOOL_CHECK_DEEP=1 runs every case three times, and both tags where a reasoning
# partner is paired with the chat model (earns 'agentic' and
# 'thinking_with_tools'); TOOL_CHECK_JSON=1 emits the summary as JSON.
tool-check: ## Check the served model's tool-calling (opt-in diagnostic; run after 'make warm'); TOOL_CHECK_DEEP=1 / TOOL_CHECK_JSON=1 / TOOL_CHECK_NO_DB=1
	@$(COMPOSE) --profile $(PROFILE) exec -T \
		-e TOOL_CHECK_DEEP="$(TOOL_CHECK_DEEP)" -e TOOL_CHECK_JSON="$(TOOL_CHECK_JSON)" \
		-e TOOL_CHECK_NO_DB="$(TOOL_CHECK_NO_DB)" \
		backend uv run python - < scripts/tool_check.py

# The retrieval-side sibling of tool-check. RAG_MIN_SCORE and RAG_REL_MARGIN
# are calibrated for ONE embedder on ONE query wrapper; swap either and the
# floors say nothing, silently - a wrong floor is a citation that should not be
# there, never an error. This embeds a fixed corpus of query/answer/distractor
# triplets (wrapped and raw), reports the score bands, and either validates the
# declared floors or proposes a pair from the bands. Seconds rather than
# minutes, and it never applies anything: a proposal is a number to put in the
# configuration by hand. EMBEDDER_CHECK_MODEL=<tag> measures an installed
# embedder other than the active one; EMBEDDER_CHECK_JSON=1 emits JSON.
embedder-check: ## Check the embedder's relevance bands against the retrieval floors; EMBEDDER_CHECK_MODEL=<tag> / EMBEDDER_CHECK_JSON=1 / EMBEDDER_CHECK_NO_DB=1
	@$(COMPOSE) --profile $(PROFILE) exec -T \
		-e EMBEDDER_CHECK_MODEL="$(EMBEDDER_CHECK_MODEL)" \
		-e EMBEDDER_CHECK_JSON="$(EMBEDDER_CHECK_JSON)" \
		-e EMBEDDER_CHECK_NO_DB="$(EMBEDDER_CHECK_NO_DB)" \
		backend uv run python - < scripts/embedder_check.py

# Periodic deep reclaim of Docker disk that dev churn leaves behind: orphaned
# anonymous volumes (mostly from `make rebuild` above), dangling images, and
# build cache. It removes ANONYMOUS volumes only, so named data volumes
# (postgres_data, qdrant_data, ollama_models, ...) are left alone, and it
# deliberately omits `-a`/`--volumes`, which WOULD remove named volumes and
# force a full re-pull. `docker volume prune` is machine-wide, so it also
# clears anon volumes from any other local Docker projects.
#
# ANONYMOUS-ONLY IS A PROPERTY OF THE ENGINE, NOT A FLAG. It arrived in Docker
# Engine 23.0 (API 1.42); on anything older the very same command also deletes
# NAMED volumes that no running container holds - with the stack stopped, that
# is the database, the vector store, the uploaded files and the model weights.
# No option turns the newer behaviour back on, so scripts/prune_guard.sh asks
# the running daemon and declines instead of assuming.
prune: ## Reclaim Docker disk (orphaned anon volumes + dangling images + build cache); guarded on pre-23.0 engines
	@# Images and build cache hold no data and are safe to reclaim on any engine,
	@# so they go FIRST: a refusal below then costs only the volume sweep rather
	@# than the whole target, on a host whose disk pressure is why it was run.
	docker image prune -f
	docker builder prune -f
	@sh scripts/prune_guard.sh volumes

# On Windows, everything Docker stores lives inside ONE virtual-disk file, and
# that file never shrinks on its own: `make prune` frees space INSIDE it, while
# the file Windows sees stays as large as it ever grew. The difference is dead
# space that only a compaction pass returns. `make disk` reports that gap in
# plain numbers; `make compact` collapses it. Both print an explanation and exit
# cleanly on Linux and macOS, which have no such file. The full reasoning, and
# why the automatic path stopped working, is in scripts/wsl_disk.sh.
disk: ## Report Docker's disk-image size against what is actually used (Windows; read-only)
	@sh scripts/wsl_disk.sh report

# The deployment sibling of `disk`, and a different question despite the
# similar name: `disk` asks how much dead space a Windows virtual-disk file is
# holding, this asks how much room the machine has left and which of the
# stack's volumes is using it. It works on every host, and it labels each
# volume with whether `make backup` covers it - "qdrant_data is 12 GB" and
# "qdrant_data is backed up" are the two facts wanted at the same moment.
host-disk: ## Report host disk headroom and the size of each of the stack's data volumes (read-only, all platforms)
	@sh scripts/host_disk.sh

# Deliberately NOT folded into `make prune`. Pruning is instant, needs no
# privileges and is safe while the stack runs; compacting needs Administrator,
# needs Docker Desktop fully stopped, and takes minutes. Joining them would stop
# the stack on every routine cleanup, usually to reclaim nothing - normal use
# GROWS the file, which compaction cannot help with. This target prunes first,
# because compaction can only reclaim space that has already been freed.
compact: ## Prune, then reclaim dead space in Docker's disk image (Windows; needs Administrator; stops Docker)
	@sh scripts/wsl_disk.sh compact

# Copies the three volumes a reinstall cannot reproduce: the database (dumped
# through the running server, because a file copy of a live data directory is
# torn), the vector store (via Qdrant's snapshot API, same reason) and the
# uploaded files (a tar of a read-only mount). Runs against a LIVE stack and
# stops nothing. DEST=<dir> sends the run somewhere other than ./backups -
# a backup on the same disk as the thing it protects is half a backup.
# PLAN=1 prints what it would do and writes nothing.
backup: ## Back up the database, vector store and uploaded files into backups/<timestamp>; DEST=<dir> / PLAN=1
	@sh scripts/backup.sh $(if $(PLAN),plan) $(if $(DEST),"$(DEST)")

# DIR is required rather than defaulting to the newest backup: a restore
# replaces the database wholesale, and "whichever one it picked" is not a thing
# to find out afterwards. The script refuses to run while the application
# services are up and never starts or stops anything itself. PLAN=1 verifies
# the backup and prints the plan without touching a thing.
restore: ## Restore a backup: make restore DIR=backups/<timestamp> (stop backend/worker/frontend first); PLAN=1
	@if [ -z "$(DIR)" ]; then \
		echo "Usage: make restore DIR=backups/<timestamp>"; \
		echo "  DIR is required - a restore replaces the database, so which backup is"; \
		echo "  not something to leave to a default. List them with: ls backups"; \
		echo "  Add PLAN=1 to verify a backup and print the plan without restoring."; \
		exit 1; \
	fi
	@sh scripts/restore.sh $(if $(PLAN),plan) "$(DIR)"

logs: ## Tail all logs
	$(COMPOSE) --profile cpu --profile gpu --profile minio --profile reranker logs -f

ps: ## List running services
	$(COMPOSE) --profile cpu --profile gpu --profile minio --profile reranker ps

# Every model target below asks scripts/remote_check.sh where the server is
# before deciding how to reach it. A local one is a container to `docker exec`
# into; a remote one is an HTTP API, spoken from inside the backend container
# (see scripts/ollama_http.py for why that side and not the host). The local
# path is unchanged - a stack with a local Ollama runs exactly the commands it
# always did.
models: ## List models: installed + loaded now (local container, or a remote model server)
	@if [ "$$(sh scripts/remote_check.sh classify-env)" = "remote" ]; then \
		bcid=$$($(COMPOSE) --profile $(PROFILE) ps -q backend 2>/dev/null); \
		if [ -z "$$bcid" ]; then \
			echo "The backend container is not running, so the remote model server cannot be"; \
			echo "reached from here - start the stack first ('make up PROFILE=remote')."; \
			exit 1; \
		fi; \
		echo "== Installed models (remote model server) =="; \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - tags < scripts/ollama_http.py || exit 1; \
		echo ""; \
		echo "== Loaded now =="; \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - ps < scripts/ollama_http.py || exit 1; \
		exit 0; \
	fi; \
	cid=$$($(COMPOSE) --profile $(PROFILE) ps -q ollama-$(PROFILE) 2>/dev/null); \
	if [ -z "$$cid" ]; then \
		echo "Ollama is not running under profile $(PROFILE) - start the stack ('make up')."; exit 1; \
	fi; \
	echo "== Installed models (ollama list) =="; \
	$(COMPOSE) --profile $(PROFILE) exec -T ollama-$(PROFILE) ollama list; \
	echo ""; \
	echo "== Loaded now (ollama ps) =="; \
	$(COMPOSE) --profile $(PROFILE) exec -T ollama-$(PROFILE) ollama ps

status: ## Show stack status: services (with health) + backend probe + loaded models
	@echo "== Services =="
	@$(COMPOSE) --profile cpu --profile gpu --profile minio --profile reranker ps
	@echo ""
	@echo "== Backend =="
	@# Quiet on a down stack, so only the friendly line below shows: `curl -fs`
	@# in dev, and the script's --quiet mode in a deployment.
	@#
	@# The reported address differs because the reachable address differs. In
	@# dev the published port is genuinely where the API answers; in a
	@# deployment it answers only behind the proxy, and naming a host port there
	@# would be an address nobody can open.
	@if $(HEALTHZ_PROBE); then \
		if [ "$(DEPLOY_ENV)" = "prod" ]; then \
			echo "  healthy - behind the proxy (no published port in the prod layout)"; \
		else \
			echo "  healthy - http://localhost:$(BACKEND_PORT)"; \
		fi; \
	elif [ "$(DEPLOY_ENV)" = "prod" ]; then \
		echo "  not responding inside the backend container (stack down, or still starting)"; \
	else \
		echo "  not responding on port $(BACKEND_PORT) (stack down, or still starting)"; \
	fi
	@echo ""
	@echo "== Loaded models (ollama ps) =="
	@if [ "$$(sh scripts/remote_check.sh classify-env)" = "remote" ]; then \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - ps < scripts/ollama_http.py \
			|| echo "  could not reach the remote model server named by OLLAMA_URL"; \
	else \
		$(COMPOSE) --profile $(PROFILE) exec -T ollama-$(PROFILE) ollama ps 2>/dev/null \
			|| echo "  ollama not running under profile $(PROFILE)"; \
	fi

config: .env ## Validate & render the merged compose config
	$(COMPOSE) $(PROFILE_FLAGS) config

db-upgrade: ## Apply database migrations inside the backend container
	$(COMPOSE) --profile $(PROFILE) exec -T backend uv run alembic upgrade head

# Hands back the cookie a completed sign-in would have left, so a signed-in
# surface can be exercised without a browser completing a consent screen at the
# identity provider. Sessions are server-side records, so this writes one
# through the app's own create_session and prints its id - see
# scripts/dev_session.py for the mechanism and docs/TESTING.md for the posture.
#
# DEPLOY_ENV is forwarded because the backend's own settings do not read it, and
# the script refuses to run under anything but the development layout. The
# arguments ride the same way the diagnostics' flags do: EMAIL=<address> mints
# for a member who has signed in at least once, TTL=<minutes> sets the window
# (default 120). An empty value means "unset", so a plain `make dev-session`
# mints for the seeded default user.
dev-session: ## Mint a signed-in session for development testing (refused in the prod layout); EMAIL=<addr> / TTL=<minutes>
	@$(COMPOSE) --profile $(PROFILE) exec -T \
		-e DEPLOY_ENV="$(DEPLOY_ENV)" \
		-e DEV_SESSION_EMAIL="$(EMAIL)" -e DEV_SESSION_TTL="$(TTL)" \
		backend uv run python - < scripts/dev_session.py

# There is deliberately no model-pulling target. Downloading weights is a
# decision about what THIS deployment should hold, made against the inventory
# the model server actually reports - which is the Model Manager page's whole
# job (/console/models): it checks a tag against the registry before starting,
# streams resumable progress, and runs a fast tool check on what arrives. The
# target it replaces could only ever pull the set a committed file predicted.
# The headless equivalent, for a box with no browser to hand, is written out in
# docs/SETUP.md under "Getting models onto the machine" - it needs the same
# --env-file and -f flags every compose command here carries, so it lives in one
# place rather than being half-remembered in three.

# `make warm` primes the stack so the first request is fast. With no arguments
# it warms EVERYTHING - the chat model, then the embedder, then the web-search
# backend; warm-chat / warm-embed / warm-search warm just one:
#   make warm         chat model, then the embedder, then SearXNG
#   make warm-chat    chat model only - load + prime its prompt cache
#   make warm-embed   embedding model only - fixes the first-ingest/query stall
#   make warm-search  SearXNG only - fixes the first-web-search cold-start miss
# Warming the two models is deliberately SEQUENTIAL, not parallel: on a tight
# 4GB card the chat model spills to CPU RAM at 8192 ctx, and a simultaneous
# cold load transiently evicts the CPU-pinned embedder that loaded first. Warming
# chat first, then the embedder (~6s), guarantees co-residency; they sit together
# fine once settled (OLLAMA_MAX_LOADED_MODELS=2, embed on CPU at 0 VRAM). A
# roomier card has headroom either way. warm-search is last and touches no GPU:
# it just fires one throwaway query so SearXNG's DNS/pool/workers are warm.
#
# Every warm step RETRIES ONCE before giving up. A warm is the single heaviest
# thing the stack does on a constrained host - a cold load can take minutes -
# so the model server answering 503 once usually means "busy right now", not
# "broken": the retry succeeds. One failure is therefore not evidence of
# anything; two are, and the second is reported and fails the target rather
# than being walked past. WARM_RETRY_SLEEP=<seconds> tunes the pause.
warm: env-check ## Prime the stack: chat model, then the embedder, then the search backend
	@echo "Warming chat, then the embedder, then SearXNG (chat first so its heavy cold load cannot evict the embedder)..."
	@$(MAKE) --no-print-directory warm-chat \
		&& $(MAKE) --no-print-directory warm-embed \
		&& $(MAKE) --no-print-directory warm-search

# Warm goes through the BACKEND (not Ollama directly) on purpose: a bare model
# load leaves the standing system prompt unevaluated, and on a small GPU with
# CPU offload that evaluation alone can stall the first real reply for 15-30s.
# The backend endpoint runs a one-token turn through the real chat prompt, so
# the prefix lands in the model server's prompt cache and the first real reply
# only evaluates its own user message.
warm-chat: ## Load the default chat model AND prime its prompt cache (first non-thinking reply is then fast; a cold run can take a minute or two)
	@echo "Warming the chat model via the backend (load + prompt-cache prime)..."
	@for attempt in 1 2; do \
		if $(WARM_CHAT_PROBE); then \
			echo ""; \
			if [ "$$attempt" = 2 ]; then echo "(that was the retry - the first attempt did not answer)"; fi; \
			echo "Warmed and primed (keep_alive=$(OLLAMA_KEEP_ALIVE)); the response above names the model."; \
			exit 0; \
		fi; \
		if [ "$$attempt" = 1 ]; then \
			echo ""; \
			echo "Chat warm did not answer - often just memory pressure while the model loads. Retrying once in $(WARM_RETRY_SLEEP)s..."; \
			sleep $(WARM_RETRY_SLEEP); \
		fi; \
	done; \
	echo ""; \
	echo "Chat warm FAILED TWICE - is the stack up and healthy ('make up', 'make ps') and a chat model downloaded (Model Manager, /console/models)?"; \
	exit 1

# The embedding model defines the vector space, so choosing one is a workspace
# decision rather than a command-line argument: warm-embed sends NO model field
# and the backend resolves it the way a retrieval does - the workspace's stored
# choice, else the model this deployment's own stored passages were written by,
# else the single installed embedder. Its placement comes from EMBEDDING_DEVICE.
# One trivial embed call pulls the weights in; there is no prompt cache to
# prime, so the first ingest or retrieval no longer stalls on a cold load.
warm-embed: ## Load the deployment's embedding model (removes the first-ingest/query cold-load stall)
	@echo "Warming the embedding model via the backend..."
	@for attempt in 1 2; do \
		if $(WARM_EMBED_PROBE); then \
			echo ""; \
			if [ "$$attempt" = 2 ]; then echo "(that was the retry - the first attempt did not answer)"; fi; \
			echo "Embedder warmed (keep_alive=$(OLLAMA_KEEP_ALIVE)); the response above names the model."; \
			exit 0; \
		fi; \
		if [ "$$attempt" = 1 ]; then \
			echo ""; \
			echo "Embedder warm did not answer - often just memory pressure while the chat model settles. Retrying once in $(WARM_RETRY_SLEEP)s..."; \
			sleep $(WARM_RETRY_SLEEP); \
		fi; \
	done; \
	echo ""; \
	echo "Embedder warm FAILED TWICE - is the stack up ('make up', 'make ps') and an embedding model downloaded (Model Manager, /console/models)?"; \
	exit 1

# The search backend (SearXNG) pays DNS + connection-pool + worker cold-start on
# its FIRST outbound call, so the first web search of a fresh stack can time out
# ("first search fails, second works"). One throwaway query primes it. No body is
# sent - the backend resolves the configured provider.
#
# This one reads the STATUS CODE rather than just curl's exit code, because the
# endpoint's two failures mean opposite things. 404 is "web search is not
# configured (SEARXNG_URL unset)": permanent, expected on a deliberately local
# stack, and no reason to retry or to fail - so it is reported and skipped.
# Anything else (503, or no answer at all) is a search backend that is supposed
# to be there and is not, so it gets the same one retry as the model warms and
# then fails rather than being reported and walked past. curl's own stderr is
# deliberately NOT captured - it goes straight to the terminal, where it reads
# as the explanation it is, instead of being glued onto the status code. The
# case guard covers a reply so mangled that no status can be read out of it.
warm-search: ## Prime the web-search backend (fixes the first-search cold-start miss)
	@echo "Warming the web-search backend (SearXNG) via the backend..."
	@for attempt in 1 2; do \
		out=$$($(WARM_SEARCH_PROBE)); \
		code=$${out##*__HTTP__}; \
		case "$$code" in [0-9][0-9][0-9]) ;; *) code="000";; esac; \
		if [ "$$code" = "200" ]; then \
			echo "$${out%__HTTP__*}"; \
			if [ "$$attempt" = 2 ]; then echo "(that was the retry - the first attempt did not answer)"; fi; \
			echo "Search backend warmed; the response above reports the priming query timing."; \
			exit 0; \
		fi; \
		if [ "$$code" = "404" ]; then \
			echo "Search warm skipped: web search is not configured (SEARXNG_URL unset). A fully local stack is fine."; \
			exit 0; \
		fi; \
		if [ "$$attempt" = 1 ]; then \
			echo "Search warm did not answer (HTTP $$code) - retrying once in $(WARM_RETRY_SLEEP)s..."; \
			sleep $(WARM_RETRY_SLEEP); \
		fi; \
	done; \
	echo "Search warm FAILED TWICE (HTTP $$code) - is the stack up ('make up', 'make ps') and the search backend reachable at SEARXNG_URL?"; \
	exit 1

# The opposite of warm: evict every resident model so the GPU - and the RAM the
# CPU-pinned embedder holds - is free for other work on the machine. Nothing
# breaks: Ollama reloads a model on the next turn (slower, exactly the cold
# start warm exists to avoid), and `make warm` re-primes deliberately.
# `ollama stop` asks the server to unload NOW instead of waiting out
# keep_alive; the loop reads `ollama ps` so it clears whatever is actually
# resident (chat, thinking, vision, embedder) rather than guessing from env.
#
# The remote branch is not a convenience. Looking for a container and finding
# none reads as "nothing to unload" - a cheerful all-clear reported while the
# remote GPU stays pinned, which is worse than an error because nobody goes
# looking. Against a remote server the unload goes over HTTP, model by model,
# and says which ones would not let go.
cool: ## Unload every resident model (frees VRAM/RAM for other work; `make warm` re-primes)
	@if [ "$$(sh scripts/remote_check.sh classify-env)" = "remote" ]; then \
		bcid=$$($(COMPOSE) --profile $(PROFILE) ps -q backend 2>/dev/null); \
		if [ -z "$$bcid" ]; then \
			echo "The backend container is not running, so the remote model server cannot be"; \
			echo "reached from here - NOTHING WAS UNLOADED and whatever is resident on the"; \
			echo "remote GPU stays resident. Start the stack ('make up PROFILE=remote') and"; \
			echo "rerun, or unload from the pod's own console."; \
			exit 1; \
		fi; \
		$(COMPOSE) --profile $(PROFILE) exec -T backend uv run python - unload-all < scripts/ollama_http.py || exit 1; \
		echo "'make warm' re-primes when the remote GPU is wanted again."; \
		exit 0; \
	fi; \
	cid=$$($(COMPOSE) --profile gpu ps -q ollama-gpu 2>/dev/null); \
	if [ -z "$$cid" ]; then cid=$$($(COMPOSE) --profile cpu ps -q ollama-cpu 2>/dev/null); fi; \
	if [ -z "$$cid" ]; then echo "No Ollama container is running - nothing to unload."; exit 0; fi; \
	models=$$(docker exec "$$cid" ollama ps | tail -n +2 | awk '{print $$1}'); \
	if [ -z "$$models" ]; then echo "No models are resident - already cool."; exit 0; fi; \
	for m in $$models; do \
		echo "Unloading $$m..."; \
		docker exec "$$cid" ollama stop "$$m" >/dev/null 2>&1 || echo "  could not unload $$m (it may already be gone)"; \
	done; \
	echo "All resident models unloaded. 'make warm' re-primes when the GPU is yours again."

fmt: ## Format code (ruff + prettier)
	-cd apps/backend && ruff format .
	-cd apps/frontend && npm run format --if-present

lint: ## Lint code (ruff + eslint)
	cd apps/backend && ruff check .
	cd apps/frontend && npm run lint

backend-test: ## Run backend tests (fakes; no Ollama needed)
	cd apps/backend && uv run pytest

frontend-test: ## Run frontend tests (today: the composer wire-fidelity battery)
	cd apps/frontend && npm run test --if-present

audit: ## Run the JS dependency audit manually (every workspace with a lockfile)
	@# Every workspace, matching the CI gate. Auditing only the app leaves the
	@# others exempt, which is how two high advisories once sat unnoticed in the
	@# type-generator package. `|| exit 1` after the loop so one clean workspace
	@# cannot mask a dirty one.
	@failed=0; \
	for ws in apps/frontend packages/api-types; do \
		if [ -f "$$ws/package-lock.json" ]; then \
			echo "==> $$ws"; \
			(cd "$$ws" && npm audit --audit-level=moderate) || failed=1; \
		fi; \
	done; \
	exit $$failed
