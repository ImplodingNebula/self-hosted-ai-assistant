# Deployment and configuration

How the stack is configured, and how it runs on a server that other people reach: the `.env` keys that matter,
HTTPS, ports, sign-in and roles, and where the data lives.

A first run on a development machine is in [SETUP.md](SETUP.md). Looking after a deployment once it is live -
updates, backups, model changes, health - is [MAINTENANCE.md](MAINTENANCE.md).

---

## Configuring the stack

Everything the platform reads is one variable in `.env`, and every variable is
documented in [`infra/.env.example`](../infra/.env.example). A development checkout
needs almost none of it changed; a deployment changes a handful, and those - the
compose layout, the hostname, sign-in, the proxy range - are gathered in one
table in [Deploying to a server](#deploying-to-a-server). The sections below are
the pieces either shape genuinely chooses between, each with the key that turns
it on. A one-off override for a single command uses the make-variable form
described in the [Quickstart](SETUP.md#quickstart), never the env-prefix form.

`make env-check` warns when `.env` is missing a variable the template has since
added, and `make sync-env` appends the missing ones with their defaults - see
[MAINTENANCE.md](MAINTENANCE.md#2-keeping-the-environment-file-current)
for the upkeep routine.

**Naming the deployment.** `APP_NAME` (default `AI Chatbot Platform`) is read by
both halves of the app: it names the browser tab, the sign-in card, the heading
above the conversation list, and the title of the API documentation.

**Encrypting stored credentials.** `SECRET_MASTER_KEY` is a base64-encoded
32-byte key, written by `make init`. It seals every integration credential at
rest. Left blank, the integration endpoints answer `503` and store nothing -
they never fall back to plaintext, and no key is ever generated implicitly,
because a key minted on the fly looks perfect in development and destroys every
stored credential the next time a different one is minted. **Never regenerate a
key that is already in use:** the credentials it sealed become undecryptable and
have to be reconnected.

**Web search.** `SEARCH_PROVIDER` selects it explicitly: `searxng` (the default,
backed by the bundled metasearch sidecar at `SEARXNG_URL`) or `none`, which
refuses regardless of the URL. A blank URL leaves the question of whether that
was deliberate unanswerable, which is why the selector exists.

**The reranker.** A sidecar that re-scores retrieval hits so only the best reach
the model - part of the standard retrieval path, and cheap enough to stay there.
It reserves no GPU and runs on the CPU: roughly 1-2 seconds per retrieval at
realistic candidate counts (measured on a laptop-class CPU; server CPUs match or
beat it), holding about 1.6 GB once loaded. Setting `RERANKER_URL` is what turns
it on - the same key starts the sidecar's compose profile and tells the backend
to use it, so the container and the code cannot disagree about whether reranking
exists. `RERANKER=0` and `RERANKER=1` force the profile off or on and outrank the
URL. `RERANKER_MODEL` is the model id the backend asks for, and has to name the
same weights the sidecar was started with in
[`infra/docker-compose.yml`](../infra/docker-compose.yml): swapping one without the
other leaves the backend asking for a model the sidecar is not serving, and
retrieval quietly falls back to the dense ranking. Its first start downloads
about 1.1 GB of weights into a named cache volume, so that boot is slow and
later ones are not.

The template ships the key commented out because it ships the development
layout, which is the one shape with a real reason to skip the sidecar: a
memory-constrained development host - an 8GB virtual machine, say - cannot hold
it beside a warmed chat model and a warmed embedder without exhausting itself.
Retrieval then runs dense-only, which works and is simply less sharp.

**Where uploaded files are stored.** `STORAGE_BACKEND` is `local` or `s3`.
`local` (the default) keeps the bytes in the `attachment_data` Docker volume,
and on a single server that is a settled answer rather than a stopgap: one
filesystem holding one copy, the retention windows below bounding how far it
grows, and `make backup` archiving it natively beside the database and the
vectors.

`s3` is the path onward, and a first-class one. Point `S3_ENDPOINT_URL`,
`S3_BUCKET`, `S3_ACCESS_KEY_ID`, `S3_SECRET_ACCESS_KEY` and `S3_REGION` at a real
object store - a cloud provider's, or a MinIO deployment of its own on its own
host and disks - copy the existing files across, and restart. The bucket is
created on the first upload if it does not exist, and none of the five is read
while the backend is `local`, so they can be filled in before the switch. Having
the objects somewhere the application server is not is the whole point: an object
store beside the app on the same disk would add the S3 API without adding any
durability.

The bundled MinIO service is there for the two jobs that follow from that -
exercising the `s3` path locally against the same adapter a deployment uses, and
serving as the building block for a dedicated store later. It sits on its own
`minio` compose profile, so it starts only when that profile is passed (`make up`
does not start it), with `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` as its
credentials.

Which backend is in use also changes what `make backup` covers: under `s3` it
skips the attachments deliberately and records that it did, since a real object
store carries its own durability - see
[Data, backups and disk](#data-backups-and-disk).

**How large a turn may be.** `DEFAULT_CONTEXT_WINDOW` (8192) caps how much of
the card one turn may spend. It is a boot-anywhere floor rather than a
recommendation: a machine with room should raise it, and a workspace context
ceiling set on the console replaces it.

**Which models answer.** Nothing needs to be named here. A deployment holding
exactly one chat model adopts it on the first turn, and the workspace default is
otherwise picked on the console. `DEFAULT_CHAT_MODEL` and `THINKING_CHAT_MODEL`
exist as a deployment-level fallback beneath that choice - see
[Getting models onto the machine](SETUP.md#getting-models-onto-the-machine).

## Deploying to a server

**This is the shape the platform is built to run in**, and the one the development
layout in [SETUP.md](SETUP.md) is the exception to: a single HTTPS origin with nothing else
published, sign-in against a real identity provider, rate limits that count each
browser separately, and data on a disk that gets backed up.

The same containers run unchanged here. What grows with the GPU is what gets
downloaded onto it and how large a turn may be: a 16GB-class card runs a
14B-class chat model comfortably where a laptop runs a 4B one, and
`DEFAULT_CONTEXT_WINDOW` (or a workspace context ceiling set on the console)
says how much of that card one turn may spend. Nothing detects the hardware and
nothing decides on its behalf - the models are chosen in the Model Manager,
against what the server actually holds.

On a fresh machine (Linux is the typical target):

```bash
# prereqs: Docker Engine 23.0+ (API 1.42) + Compose plugin, make
# with a local GPU, also: NVIDIA driver + Container Toolkit, verified first with
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi

git clone <repository> && cd <repository>
make init          # writes .env from the template with a fresh SECRET_MASTER_KEY
# now edit .env (see the table below), then:
make up            # starts everything; the reranker sidecar joins when RERANKER_URL names one
make gpu-check     # asserts the model server is on a GPU, not a CPU fallback
# then download a chat model and an embedding model from the Model Manager
# (the console's /console/models page) and pick the chat one as the workspace default
make warm          # primes chat, embedder, and the search backend
```

Docker Engine 23.0 (API 1.42) is a real minimum on a server, not a
nice-to-have. Long-term distributions still ship 20.10 as `docker.io`, which
satisfies "Docker Engine + Compose plugin" and yet has the volume-pruning
behaviour described in the [Quickstart](SETUP.md#quickstart) - so check `docker version` before
treating a cleanup command as safe.

The edit in the middle is the whole difference between a development checkout
and a deployment. `make init` writes a development `.env`; these are the keys
that move:

| Key | Set it to | Why |
|---|---|---|
| `DEPLOY_ENV` | `prod` | Swaps the development compose layout for the hardened one |
| `DOMAIN` | the public hostname | Turns on publicly-trusted HTTPS; blank is still HTTPS, self-signed |
| `ACME_EMAIL` | an address that is read | The only warning before a certificate-renewal problem |
| `POSTGRES_PASSWORD`, `SEARXNG_SECRET` | fresh random values | The template ships placeholders |
| `SECRET_MASTER_KEY` | leave what `make init` generated | It encrypts stored integration credentials |
| `OAUTH_REDIRECT_BASE_URL` | `https://` + the domain | Where a provider sends the browser back - after a sign-in and after an integration consent alike; it defaults to `localhost` |
| `OIDC_ISSUER` | the provider's issuer URL | Turns sign-in on. Left unset there is no login, and every visitor is the same default user |
| `OIDC_CLIENT_ID`, `OIDC_CLIENT_SECRET` | the credentials the provider issued | Registered against the redirect URI above |
| `OIDC_ADMIN_EMAILS` | the first administrator's address | Bootstrap and standing floor: listed addresses are administrators at every sign-in |
| `OIDC_ALLOWED_DOMAINS`, `OIDC_ALLOWED_EMAILS` | who else may sign in | Admission is deny-by-default; without these, only the administrators get in |
| `TRUSTED_PROXY_IPS` | `172.16.0.0/12` | Lets rate limits count each browser separately; left empty, every request looks like the frontend relay and the whole deployment shares one budget |
| `DEFAULT_CONTEXT_WINDOW` | what the card affords | How much of the GPU one turn may spend; the 8192 default is a boot-anywhere floor, not a recommendation |
| `RERANKER_URL` | `http://reranker:7997` | Sharper retrieval, for a second or so of CPU per search; the one key that both starts the sidecar and switches the backend to it |
| `PROFILE` + `OLLAMA_URL` | `remote` + the model server's address | Only when models are served from another machine |

`DEPLOY_ENV=prod` is what makes `make up` build the deployment. It is read by
every target, so there is nothing extra to remember later: a routine `make up`
or `make deploy` on that machine can no longer rebuild it in the development
shape. Trying it for a single command follows the same rule as any other
setting - the make-variable form after the target (`make up DEPLOY_ENV=prod`),
never the env-prefix one; see the [Quickstart](SETUP.md#quickstart) for why.

`PROFILE` is a separate axis and stays free: a server that runs the stack itself
but serves models from a rented GPU elsewhere is `DEPLOY_ENV=prod` together with
`PROFILE=remote` plus that server's `OLLAMA_URL` (and, for a protected one,
`OLLAMA_AUTH_HEADERS`).

First-boot expectations: database migrations apply automatically; the model
downloads run from the console and are the slow part (a server-class chat model
plus an embedder is roughly 10-20GB, once); the reranker sidecar fetches its own
model on first start; and the first HTTPS request against a real `DOMAIN` waits
a few seconds while the certificate is issued.

### HTTPS and the single public entrance

In the `prod` layout, one service faces the outside: a
[Caddy](https://caddyserver.com) reverse proxy on ports 80 and 443. Everything
else - the backend, the web UI, PostgreSQL, Redis, Qdrant, the model server -
publishes no host port at all and is reachable only on the internal Docker
network.

Caddy serves the web UI and the API from the **same origin**, which is why the
browser never makes a cross-origin request and CORS never enters the picture. It
also terminates TLS and obtains the certificates itself:

- **`DOMAIN` set to a real hostname** whose DNS already points at the machine:
  Caddy requests a publicly-trusted certificate on first start and renews it for
  as long as the stack runs. Nothing to install or rotate by hand. Ports 80 and
  443 must be reachable from the internet: that is how the authority verifies
  the machine really answers for the hostname.
- **`DOMAIN` left blank:** Caddy serves `localhost` with a certificate from its
  own internal authority. HTTPS works immediately on a box that has no DNS
  record yet; browsers warn that the certificate is untrusted, which is
  accurate. Moving to a real hostname later is a one-line `.env` edit.

Certificates and the certificate-authority account live in the `caddy_data`
volume, so recreating the container does not re-request them.

### Ports

**In the `prod` layout** exactly two ports are published, both by Caddy, on
every interface: **80** (redirects to HTTPS, and answers the certificate
challenge) and **443** (the site). Nothing else is reachable from off the
machine, and none of the variables below apply.

**An unpublished port cannot clash.** Publishing is what binds a container's
port to the host, and the prod layout binds only Caddy's two - so the services
in the table below can never conflict with anything else the machine runs. An
existing PostgreSQL on 5432 or Redis on 6379 coexists untouched, and the
stack's own copies stay reachable only over the internal compose network. The
only collisions worth checking for before a prod boot are **80** and **443**
themselves.

**In the `dev` layout** every service below publishes a host port. Each one is
set in `.env` - if another service on the host already uses one, change the
variable before `make up`:

| Host port (default) | Service | `.env` variable |
|---|---|---|
| 3000 | Web UI | `FRONTEND_PORT` |
| 8000 | Backend API | `BACKEND_PORT` |
| 11434 | Ollama | `OLLAMA_PORT` |
| 5432 | PostgreSQL | `POSTGRES_PORT` |
| 6379 | Redis | `REDIS_PORT` |
| 6333 / 6334 | Qdrant (HTTP / gRPC) | `QDRANT_HTTP_PORT` / `QDRANT_GRPC_PORT` |
| 9000 / 9001 | MinIO (API / console) | `MINIO_PORT` / `MINIO_CONSOLE_PORT` |

MinIO only starts under the `minio` profile. The reranker, SearXNG, the MCP
chassis and doc-gen publish no host ports in either layout - only the backend
reaches them, over the internal network.

### Access

**Signing in.** The app authenticates people against an OpenID Connect
identity provider - any conformant one, Google included. Nobody's password is
ever handled here: the browser is redirected to the provider, and what comes
back is a set of signed claims about who signed in. A session then lives on the
server, keyed by an httpOnly cookie.

**Who is let in.** Deny by default, from three keys in `.env`. An address is
admitted only if the provider says its email is verified *and* it appears in
`OIDC_ALLOWED_EMAILS`, or its domain appears in `OIDC_ALLOWED_DOMAINS`, or it
is one of the `OIDC_ADMIN_EMAILS`. Anybody else is turned away with a plain
message, whether or not they have an account with the provider.

**Who administers it.** Two roles, and the line between them is drawn per
action rather than per page. A member can chat, and can view the console's
model and integration pages - those are reads. What is administrators-only is
the actions that change the workspace: pulling or deleting a model, registering
or re-pointing an integration, storing a workspace credential, and the member
list in full (its roster names every colleague's address, so both reading and
changing it are admin-only). `OIDC_ADMIN_EMAILS` is both the bootstrap and a
standing floor - a listed address is an administrator at every sign-in and
cannot be demoted in-app, which is what keeps a workspace from locking itself
out. Everyone else is a member on first sign-in, and an administrator can
promote them from **Console -> Members**. The last remaining administrator
cannot be demoted either. There is no way to delete an account from the app:
access comes from the identity provider, so removing it belongs there.

**With `OIDC_ISSUER` unset, there is no login at all** - and then anyone who
can reach the app acts as the workspace's default user, including any
integration credentials that user has connected. That is the right shape for a
laptop and the wrong shape for anything reachable, so a deployment sets the
OIDC keys (see [the deployment table](#deploying-to-a-server) above).

**Network hardening still matters, sign-in or not.** Authentication decides who
may act; it does nothing about who may knock. Keep the install on an internal
network, behind a VPN, or restricted at the firewall, and treat a broadly
reachable deployment holding live third-party credentials with the caution it
deserves. The `prod` layout narrows the surface as far as the topology can: the
backing stores and the API are not published at all, so the only thing an
outside request can reach is Caddy, over HTTPS.

**Run the backend in its container, or behind a reverse proxy, whenever sign-in
is enabled.** A request that arrives from a loopback peer without an
`X-Forwarded-For` header is treated as the operator, so that the maintenance
commands (model warm, status) work over `docker exec` inside the container. That
is sound in the shipped layout - `prod` publishes no backend port, and the
frontend relay dials from a container address rather than loopback. It is a hole
only in one configuration to avoid: the backend run as a bare host process,
reachable on `localhost`, with login on - there any co-located process that can
reach the port and omits the header is admitted as an administrator. The
container layout and any reverse proxy that sets `X-Forwarded-For` both close
it; a bare-host backend with sign-in enabled is the shape not to deploy.

One caveat about *where* that firewall sits, because it is a common and
expensive surprise on Linux. Publishing a container port inserts a rule into
Docker's own firewall chain, and that chain is consulted before the one a
host firewall such as `ufw` manages - so a host-level `deny` on a published
port looks applied and does not take effect. What does close a port is not
publishing it beyond loopback in the first place (`BIND_HOST`, the default
here), or a firewall upstream of the host such as a cloud provider's security
group.

### Data, backups and disk

Three of the stack's Docker volumes hold something nothing can reproduce: the
database (`postgres_data`), the uploaded file bytes (`attachment_data`) and the
retrieval vectors (`qdrant_data`). `make backup` copies exactly those three
into a timestamped, self-verifying archive, and `make restore` puts one back;
both work against the live stack over the Docker network it already runs on,
with nothing extra to publish. Stored files age out on a retention policy and
conversations do not.

The commands, what a backup and a restore actually do, and what changes under
`STORAGE_BACKEND=s3` are covered in
[MAINTENANCE.md](MAINTENANCE.md#3-backups-and-restores); what
expires and what is kept forever is in
[MAINTENANCE.md](MAINTENANCE.md#6-what-expires-and-what-does-not);
and the full volume-by-volume mapping, with the reasoning behind each
inclusion and exclusion, is in
[DATA-MODEL.md](DATA-MODEL.md#what-lives-in-each-volume).
